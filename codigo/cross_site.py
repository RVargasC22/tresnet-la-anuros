"""
Analogo al Apendice D del paper (validacion cross-dataset con DeepShip): mide
si el modelo generaliza a un dominio no visto durante el entrenamiento.

El paper usa OTRO dataset (DeepShip) para esto. No lo tenemos, pero el dataset
propio tiene 4 sitios de grabacion bien distintos (prefijo del filename:
INCT20955, INCT4, INCT41, INCT17), con densidades de especies muy distintas
(algunos sitios ~2.5 especies/clip, otros ~0.5). Leave-one-site-out (entrenar
con 3 sitios, testear en el 4to, nunca visto) es el analogo directo: mide
generalizacion a un dominio nuevo, igual que el paper mide generalizacion a
otro dataset.

4 corridas independientes (una por sitio dejado afuera), TResNet-LA completo,
mismos hiperparametros que el arm principal de experiments.py.

Uso:
    python cross_site.py --epochs 80
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as cfg
from model import build_model
from dataset import MelConfig, MultiLabelAudioDataset, load_dataframe, stratified_train_val_split
from train import set_seed, _seed_worker, train_model
from metrics import evaluate_all, per_class_report

SITE_RE = re.compile(r"^([A-Za-z]+\d*)_")


def site_of(filename):
    m = SITE_RE.match(filename)
    return m.group(1) if m else "unknown"


def build_site_loaders(cfg, df, label_cols, held_out_site, error_log_name):
    sites = df[cfg.FILENAME_COL].map(site_of)
    train_pool = df[sites != held_out_site].reset_index(drop=True)
    test_df = df[sites == held_out_site].reset_index(drop=True)

    # Split 80/20 dentro de los sitios de entrenamiento, solo para
    # seleccionar el mejor checkpoint (igual criterio que el resto del proyecto).
    train_df, val_df = stratified_train_val_split(train_pool, label_cols, cfg)

    cache_dir = os.path.join(cfg.OUTPUT_DIR, "melspec_cache")
    mel_cfg = MelConfig.from_config_module(cfg)
    error_log_path = os.path.join(cfg.OUTPUT_DIR, error_log_name)
    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    train_ds = MultiLabelAudioDataset(train_df, label_cols, cfg.AUDIO_DIR, mel_cfg,
                                      cache_dir=cache_dir, augment=True, error_log_path=error_log_path)
    val_ds = MultiLabelAudioDataset(val_df, label_cols, cfg.AUDIO_DIR, mel_cfg,
                                    cache_dir=cache_dir, augment=False, error_log_path=error_log_path)
    test_ds = MultiLabelAudioDataset(test_df, label_cols, cfg.AUDIO_DIR, mel_cfg,
                                     cache_dir=cache_dir, augment=False, error_log_path=error_log_path)

    g = torch.Generator(); g.manual_seed(cfg.RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
                              worker_init_fn=_seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True,
                            worker_init_fn=_seed_worker)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True,
                             worker_init_fn=_seed_worker)

    print(f"[{held_out_site}] train={len(train_df)} val={len(val_df)} "
          f"test(held-out site)={len(test_df)}")
    return train_loader, val_loader, test_loader


def evaluate_on_test(model, test_loader, device, threshold):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x, return_logits=True)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(y.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets)
    return evaluate_all(y_true, y_prob, threshold), y_true, y_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--sites", type=str, nargs="+",
                    default=["INCT20955", "INCT4", "INCT41", "INCT17"])
    args = ap.parse_args()

    cfg.EPOCHS = args.epochs
    cfg.EARLY_STOPPING = False
    cfg.OUTPUT_DIR = "outputs/cross_site/"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    _cache = os.path.join(cfg.OUTPUT_DIR, "melspec_cache")
    if not os.path.exists(_cache):
        os.symlink(os.path.abspath("outputs/run_v2/melspec_cache"), _cache)

    device = torch.device(cfg.DEVICE)
    df, label_cols = load_dataframe(cfg)
    sites_present = sorted(df[cfg.FILENAME_COL].map(site_of).unique())
    print(f"Device: {device} | sitios detectados: {sites_present} | epocas/sitio: {cfg.EPOCHS}")

    results = []
    for held_out in args.sites:
        if held_out not in sites_present:
            print(f"(sitio {held_out} no encontrado, se omite)")
            continue
        print(f"\n{'='*72}\nLEAVE-ONE-SITE-OUT: test = {held_out}\n{'='*72}")
        set_seed(cfg.RANDOM_SEED)
        train_loader, val_loader, test_loader = build_site_loaders(
            cfg, df, label_cols, held_out, f"bad_audio_{held_out}.log")

        set_seed(cfg.RANDOM_SEED)
        model = build_model(len(label_cols), cfg, use_se=True).to(device)
        ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"held_out_{held_out}_best.pt")
        history_df, val_metrics, _, _, checkpoint = train_model(
            model, train_loader, val_loader, device, cfg, ckpt_path, label_cols,
            run_name=f"held_out_{held_out}")
        history_df.to_csv(os.path.join(cfg.OUTPUT_DIR, f"{held_out}_history.csv"), index=False)

        # Recargar el mejor checkpoint y evaluar sobre el sitio nunca visto.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        test_metrics, y_true, y_prob = evaluate_on_test(model, test_loader, device,
                                                         cfg.SIGMOID_THRESHOLD)
        per_class_report(y_true, y_prob, cfg.SIGMOID_THRESHOLD, label_cols).to_csv(
            os.path.join(cfg.OUTPUT_DIR, f"{held_out}_test_per_class.csv"), index=False)
        np.savez(os.path.join(cfg.OUTPUT_DIR, f"{held_out}_test_predictions.npz"),
                 y_true=y_true, y_prob=y_prob, label_cols=np.array(label_cols))

        row = {"held_out_site": held_out, "best_epoch": int(checkpoint["epoch"]),
              "n_test": len(y_true)}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        results.append(row)
        pd.DataFrame(results).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "cross_site_results.csv"), index=False)
        print(f">>> {held_out}: val_mAP={val_metrics['mAP']:.2f} | "
              f"TEST(unseen site)_mAP={test_metrics['mAP']:.2f} "
              f"global_acc={test_metrics['global_acc']:.2f} f1={test_metrics['f1']:.2f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(cfg.OUTPUT_DIR, "cross_site_results.csv"), index=False)
    with open(os.path.join(cfg.OUTPUT_DIR, "cross_site_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*72}\n=== RESULTADOS CROSS-SITE (analogo Apendice D) ===\n{'='*72}")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
