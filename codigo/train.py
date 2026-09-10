"""
Script principal de entrenamiento de TResNet-LA (UAMTR) sobre el CSV
multi-etiqueta entregado.

Uso:
    python train.py

Antes de correr:
    1. Editar config.py -> AUDIO_DIR apuntando a la carpeta con los .wav
       referenciados en la columna "filename" del CSV.
    2. (Opcional) ajustar EPOCHS/BATCH_SIZE según tu GPU.

El script:
    1. Carga el CSV y hace split 80% train / 20% validación (requisito del
       proyecto), preservando el balance multi-label. La semilla usada es
       config.RANDOM_SEED (fija, para que el split sea 100% reproducible).
    2. Extrae Mel-spectrogramas (cacheados en disco como .npy la 1ra vez que
       se leen, para no recalcular en cada época).
    3. Entrena TResNet-LA con Asymmetric Loss y AdamW.
    4. Evalúa en validación cada época, aplica early stopping y guarda el
       mejor checkpoint según val_global_acc.
    5. Al final, guarda las métricas de validación del mejor checkpoint
       (final_val_metrics.json) y un reporte por clase (per_class_report.csv).

Nota importante: como el proyecto pide 80/20 train/validación (no train/val/
test), NO existe un tercer conjunto etiquetado separado para "test" -- la
única evaluación con etiquetas reales posible es sobre el 20% de validación.
La carpeta test/ (sin etiquetas) se usa aparte, solo para generar
predicciones con predict.py, sin poder calcular métricas allí.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from model import build_model
from losses import AsymmetricLoss
from dataset import (
    MultiLabelAudioDataset, MelConfig, load_dataframe, stratified_train_val_split,
)
from metrics import evaluate_all, per_class_report


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismo de cuDNN: para que la ablacion con_SE vs sin_SE sea
    # reproducible y la unica fuente de diferencia sea la arquitectura.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    """Siembra numpy y random por worker del DataLoader. PyTorch solo siembra
    el RNG de torch por worker; sin esto, los N workers generan el mismo
    SpecAugment (np.random) en paralelo (footgun conocido de PyTorch)."""
    import random
    base_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def build_dataloaders(cfg, error_log_name: str = "bad_audio_files.log",
                      resample_method: str = "none"):
    """Construye train_loader/val_loader con el split 80/20 (semilla fija
    en cfg.RANDOM_SEED). Se reutiliza tal cual desde ablation.py para que
    todas las variantes comparadas usen EXACTAMENTE el mismo split.

    resample_method: 'none' | 'msw' | 'uabos' -- se aplica SOLO al training
    set (igual que el paper), despues del split. La validacion nunca se
    resamplea."""
    df, label_cols = load_dataframe(cfg)
    train_df, val_df = stratified_train_val_split(df, label_cols, cfg)

    print(f"Clases ({len(label_cols)}): {label_cols}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    cache_dir = os.path.join(cfg.OUTPUT_DIR, "melspec_cache")
    mel_cfg = MelConfig.from_config_module(cfg)   # objeto liviano y picklable (ver dataset.py)
    error_log_path = os.path.join(cfg.OUTPUT_DIR, error_log_name)
    if os.path.exists(error_log_path):
        os.remove(error_log_path)  # limpiar log de una corrida anterior

    if resample_method not in (None, "none"):
        from resampling import resample_training_df
        train_df, rs_stats = resample_training_df(
            train_df, label_cols, resample_method, mel_cfg=mel_cfg,
            cache_dir=cache_dir, audio_dir=cfg.AUDIO_DIR, seed=cfg.RANDOM_SEED)
        print(f"Resampling '{resample_method}': {rs_stats}")

    train_ds = MultiLabelAudioDataset(train_df, label_cols, cfg.AUDIO_DIR, mel_cfg,
                                       cache_dir=cache_dir, augment=True, error_log_path=error_log_path)
    val_ds = MultiLabelAudioDataset(val_df, label_cols, cfg.AUDIO_DIR, mel_cfg,
                                     cache_dir=cache_dir, augment=False, error_log_path=error_log_path)

    g = torch.Generator()
    g.manual_seed(cfg.RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
                               worker_init_fn=_seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True,
                             worker_init_fn=_seed_worker)

    return train_loader, val_loader, label_cols


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    use_amp = (scaler is not None) and (device.type == "cuda")

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(x, return_logits=True)
                    loss = criterion(logits, y)

                # Guardia anti-NaN: si la loss diverge (NaN/inf) en un batch,
                # se descarta ese paso en vez de dejar que corrompa los pesos
                # del modelo y arruine el resto del entrenamiento.
                if not torch.isfinite(loss):
                    print("⚠️  Loss no finita (NaN/inf) en un batch — se omite "
                          "este paso de optimización.")
                    optimizer.zero_grad()
                    continue

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            else:
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(x, return_logits=True)
                    loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    metrics = evaluate_all(y_true, y_prob, cfg.SIGMOID_THRESHOLD)
    metrics["loss"] = avg_loss
    return metrics, y_true, y_prob


def train_model(model, train_loader, val_loader, device, cfg, ckpt_path,
                 label_cols, run_name: str = "modelo", verbose: bool = True):
    """
    Bucle de entrenamiento reutilizable: entrena `model` con early stopping,
    guarda el mejor checkpoint según val_global_acc, y devuelve el historial
    completo + las métricas de validación del mejor checkpoint.

    Se usa tanto desde main() (entrenamiento normal) como desde ablation.py
    (para entrenar cada variante del estudio de ablación bajo exactamente
    las mismas condiciones).
    """
    criterion = AsymmetricLoss(cfg.ASL_GAMMA_NEG, cfg.ASL_GAMMA_POS,
                                cfg.ASL_CLIP, cfg.ASL_EPS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                                   weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Metrica de seleccion de modelo: mAP (usa las probabilidades, es
    # independiente del umbral 0.8). global_acc/macro_acc estan inflados por
    # los verdaderos negativos en un problema multi-label de 42 clases muy
    # desbalanceado, y elegir el mejor checkpoint por ellos puede quedarse
    # con un modelo que "predice casi todo 0".
    ckpt_metric = getattr(cfg, "CHECKPOINT_METRIC", "mAP")
    best_ckpt_metric = -1.0
    best_metric_value = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=scaler)
        scheduler.step()
        dt = time.time() - t0

        if verbose:
            print(f"[{run_name}] [Epoch {epoch:03d}/{cfg.EPOCHS}] "
                  f"train_loss={train_metrics['loss']:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"val_global_acc={val_metrics['global_acc']:.2f} "
                  f"val_macro_acc={val_metrics['macro_acc']:.2f} "
                  f"val_mAP={val_metrics['mAP']:.2f} "
                  f"({dt:.1f}s)")

        history.append({"epoch": epoch,
                         "train_loss": train_metrics["loss"],
                         "val_loss": val_metrics["loss"],
                         "val_global_acc": val_metrics["global_acc"],
                         "val_macro_acc": val_metrics["macro_acc"],
                         "val_f1": val_metrics["f1"],
                         "val_mAP": val_metrics["mAP"]})

        if val_metrics[ckpt_metric] > best_ckpt_metric:
            best_ckpt_metric = val_metrics[ckpt_metric]
            torch.save({"model_state": model.state_dict(),
                        "label_cols": label_cols,
                        "epoch": epoch,
                        "val_metrics": val_metrics}, ckpt_path)
            if verbose:
                print(f"  -> Nuevo mejor modelo guardado (val_{ckpt_metric}={best_ckpt_metric:.2f})")

        # Guardado incremental: history en cada epoca (sobrevive a un corte),
        # estado de la ultima epoca, y snapshot cada 10 epocas.
        _dir = os.path.dirname(ckpt_path)
        _tag = run_name.replace(" ", "_").replace("/", "_")
        pd.DataFrame(history).to_csv(
            os.path.join(_dir, f"{_tag}_history_live.csv"), index=False)
        torch.save({"model_state": model.state_dict(), "label_cols": label_cols,
                    "epoch": epoch, "val_metrics": val_metrics},
                   os.path.join(_dir, f"{_tag}_last.pt"))
        # Snapshot completo de CADA epoca (~61 MB c/u): permite retomar o
        # evaluar cualquier epoca despues, no solo la mejor.
        _snap_dir = os.path.join(_dir, f"{_tag}_snapshots")
        os.makedirs(_snap_dir, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "label_cols": label_cols,
                    "epoch": epoch, "val_metrics": val_metrics},
                   os.path.join(_snap_dir, f"ep{epoch:03d}.pt"))

        # --------------------------------------------------------------
        # Early stopping sobre EARLY_STOPPING_METRIC. El checkpoint guardado
        # siempre es el de mejor val_<CHECKPOINT_METRIC>, asi que cortar antes
        # no pierde el mejor resultado visto. Desactivado en ablation.py.
        # --------------------------------------------------------------
        if cfg.EARLY_STOPPING:
            current_metric = val_metrics[cfg.EARLY_STOPPING_METRIC]
            if current_metric > best_metric_value + cfg.EARLY_STOPPING_MIN_DELTA:
                best_metric_value = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if verbose:
                    print(f"  (sin mejora significativa en '{cfg.EARLY_STOPPING_METRIC}': "
                          f"{epochs_without_improvement}/{cfg.EARLY_STOPPING_PATIENCE} épocas)")

            if epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
                if verbose:
                    print(f"\n⏹️  Early stopping activado en la época {epoch} para '{run_name}'.")
                break

    # Evaluación final con el mejor checkpoint guardado
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_val_metrics, y_true, y_prob = run_epoch(model, val_loader, criterion, optimizer,
                                                    device, train=False, scaler=scaler)

    history_df = pd.DataFrame(history)
    return history_df, final_val_metrics, y_true, y_prob, checkpoint


def main():
    set_seed(cfg.RANDOM_SEED)
    device = torch.device(cfg.DEVICE)
    print(f"Device: {device}")
    print(f"Semilla utilizada para el split train/validación: {cfg.RANDOM_SEED}")

    train_loader, val_loader, label_cols = build_dataloaders(cfg)
    num_classes = len(label_cols)

    model = build_model(num_classes, cfg, use_se=True).to(device)
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")

    history_df, final_val_metrics, y_true, y_prob, checkpoint = train_model(
        model, train_loader, val_loader, device, cfg, ckpt_path, label_cols,
        run_name="TResNet-LA",
    )

    history_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "training_history.csv"), index=False)

    print("\n=== Métricas finales de VALIDACIÓN (mejor checkpoint, época "
          f"{checkpoint['epoch']}) ===")
    for k, v in final_val_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(os.path.join(cfg.OUTPUT_DIR, "final_val_metrics.json"), "w") as f:
        json.dump(final_val_metrics, f, indent=2)

    # Predicciones por muestra del 20% de validacion: las usa analyze_results.py
    # para el analisis de errores y el desglose de metricas por nº de especies
    # por clip (analogo al analisis "mixed target quantity" del paper).
    np.savez(os.path.join(cfg.OUTPUT_DIR, "val_predictions.npz"),
             y_true=y_true, y_prob=y_prob, label_cols=np.array(label_cols))

    report = per_class_report(y_true, y_prob, cfg.SIGMOID_THRESHOLD, label_cols)
    report.to_csv(os.path.join(cfg.OUTPUT_DIR, "per_class_report.csv"), index=False)
    print("\nReporte por clase guardado en per_class_report.csv")
    print(report.sort_values("support", ascending=False).head(10).to_string(index=False))

    bad_log = os.path.join(cfg.OUTPUT_DIR, "bad_audio_files.log")
    if os.path.exists(bad_log):
        n_bad = sum(1 for _ in open(bad_log, encoding="utf-8"))
        print(f"\n⚠️  {n_bad} archivo(s) de audio no se pudieron leer durante el "
              f"entrenamiento (se sustituyeron por silencio). Detalle en: {bad_log}")


if __name__ == "__main__":
    main()
