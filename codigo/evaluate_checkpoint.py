"""
Evalúa un checkpoint ya entrenado (best_model.pt) sobre el 20% de
VALIDACIÓN, SIN volver a entrenar. Útil cuando una corrida se cae a mitad
de camino (ej. por NaN) pero ya tenés un buen checkpoint guardado de una
época anterior.

Uso:
    python evaluate_checkpoint.py
    python evaluate_checkpoint.py --checkpoint ruta/a/otro_modelo.pt

Importante: usa el MISMO split 80/20 (train/validación) que generó
train.py (mismo CSV + misma RANDOM_SEED en config.py = mismo split
reproducible), así que no hace falta guardar los índices por separado.

Nota: como el proyecto usa 80% train / 20% validación (sin un tercer split
de "test" etiquetado), esto evalúa sobre el mismo 20% de validación que ya
usó train.py durante el entrenamiento -- no es un conjunto "nuevo". Sirve
para recalcular métricas de un checkpoint puntual sin re-entrenar, no para
una evaluación en datos jamás vistos por el proceso de selección de modelo.
Para eso último (datos realmente nuevos), usá predict.py sobre tu carpeta
test/ (sin etiquetas, solo genera predicciones, no métricas).
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from model import build_model
from dataset import MelConfig, load_dataframe, stratified_train_val_split, MultiLabelAudioDataset
from metrics import evaluate_all, per_class_report
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))
    parser.add_argument("--threshold", type=float, default=cfg.SIGMOID_THRESHOLD)
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE)
    print(f"Device: {device}")
    print(f"Cargando checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    label_cols = checkpoint["label_cols"]
    print(f"Checkpoint de la época {checkpoint.get('epoch', '?')}, "
          f"val_metrics guardadas: {checkpoint.get('val_metrics', {})}")

    # Reconstruir EXACTAMENTE el mismo split 80/20 que usó train.py (mismo
    # CSV + mismo RANDOM_SEED en config.py => mismo split reproducible).
    df, csv_label_cols = load_dataframe(cfg)
    assert list(csv_label_cols) == list(label_cols), (
        "Las clases del CSV actual no coinciden con las del checkpoint. "
        "¿Cambiaste el CSV desde que entrenaste este modelo?"
    )
    _, val_df = stratified_train_val_split(df, csv_label_cols, cfg)
    print(f"Validación: {len(val_df)} muestras (semilla: {cfg.RANDOM_SEED})")

    mel_cfg = MelConfig.from_config_module(cfg)
    cache_dir = os.path.join(cfg.OUTPUT_DIR, "melspec_cache")
    error_log_path = os.path.join(cfg.OUTPUT_DIR, "bad_audio_files_eval.log")
    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    val_ds = MultiLabelAudioDataset(val_df, label_cols, cfg.AUDIO_DIR, mel_cfg,
                                     cache_dir=cache_dir, augment=False,
                                     error_log_path=error_log_path)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = build_model(len(label_cols), cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluando en validación"):
            x = x.to(device)
            logits = model(x, return_logits=True)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    # Guardia por si quedara algún NaN en las predicciones (ej. de un
    # checkpoint guardado justo antes de que el entrenamiento divergiera).
    n_nan = np.isnan(y_prob).sum()
    if n_nan > 0:
        print(f"\n⚠️  ADVERTENCIA: {n_nan} valores NaN en las predicciones. "
              f"Este checkpoint podría estar dañado. Reemplazando por 0.5 "
              f"para poder calcular métricas de todas formas.")
        y_prob = np.nan_to_num(y_prob, nan=0.5)

    metrics = evaluate_all(y_true, y_prob, args.threshold)
    print("\n=== Resultados en VALIDACIÓN ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out_metrics_path = os.path.join(cfg.OUTPUT_DIR, "val_metrics_from_checkpoint.json")
    with open(out_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétricas guardadas en: {out_metrics_path}")

    report = per_class_report(y_true, y_prob, args.threshold, label_cols)
    report_path = os.path.join(cfg.OUTPUT_DIR, "per_class_report_from_checkpoint.csv")
    report.to_csv(report_path, index=False)
    print(f"Reporte por clase guardado en: {report_path}")
    print(report.sort_values("support", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
