"""
Experimento unificado: mismo split (80/20, semilla 42), mismas epocas, sin
early stopping. Un solo script cubre las tres tablas del paper:

    Ablacion del SE (Tabla 8, "-A" vs sin)   -> arms 1 vs 2, 5 vs 6, 7 vs 8
    Efecto de los resampling (Tabla 6)        -> arm 1 vs 3 vs 4
    Comparacion de backbones (Tabla 8)        -> arms 1,2,5,6,7,8

Arms:
  1. TResNet-LA          use_se=T   resample=none    (modelo del paper)
  2. TResNet-L           use_se=F   resample=none    (ablacion del SE)
  3. TResNet-LA + MSW    use_se=T   resample=msw
  4. TResNet-LA + UABOS  use_se=T   resample=uabos
  5. ResNet50            baseline   resample=none
  6. ResNet50-A          baseline   resample=none    (ResNet50 + SE)
  7. TResNet-M           baseline   resample=none
  8. TResNet-MA          baseline   resample=none    (TResNet-M + SE)

Uso:
    python experiments.py                 # 8 arms, 40 epocas
    python experiments.py --epochs 30 --arms 1 2 3 4
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import config as cfg
from model import build_model
from baselines import build_baseline
from train import set_seed, build_dataloaders, train_model
from metrics import per_class_report

# (name, kind, kw)   kind: 'tresnet' -> build_model(use_se); 'baseline' -> build_baseline(name)
ARMS = {
    1: ("TResNet-LA",       "tresnet",  dict(use_se=True),  "none"),
    2: ("TResNet-L",        "tresnet",  dict(use_se=False), "none"),
    3: ("TResNet-LA+MSW",   "tresnet",  dict(use_se=True),  "msw"),
    4: ("TResNet-LA+UABOS", "tresnet",  dict(use_se=True),  "uabos"),
    5: ("ResNet50",         "baseline", dict(bl="resnet50"),   "none"),
    6: ("ResNet50-A",       "baseline", dict(bl="resnet50-a"), "none"),
    7: ("TResNet-M",        "baseline", dict(bl="tresnet-m"),  "none"),
    8: ("TResNet-MA",       "baseline", dict(bl="tresnet-ma"), "none"),
    9: ("TResNet-L+MSW",    "tresnet",  dict(use_se=False), "msw"),   # Tabla 7 del paper
    10: ("TResNet-L+UABOS", "tresnet",  dict(use_se=False), "uabos"),
}


def _tag(name):
    return name.replace("/", "_").replace("+", "_")


def run_arm(arm_id, device):
    name, kind, kw, resample = ARMS[arm_id]
    print(f"\n{'='*72}\nARM {arm_id}: {name}  ({kind}, {kw}, resample={resample})\n{'='*72}")

    set_seed(cfg.RANDOM_SEED)
    train_loader, val_loader, label_cols = build_dataloaders(
        cfg, error_log_name=f"bad_audio_{_tag(name)}.log", resample_method=resample)

    set_seed(cfg.RANDOM_SEED)   # misma init de pesos
    if kind == "tresnet":
        model = build_model(len(label_cols), cfg, use_se=kw["use_se"])
    else:
        model = build_baseline(kw["bl"], cfg, len(label_cols))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"{_tag(name)}_best.pt")
    history_df, final_val_metrics, y_true, y_prob, checkpoint = train_model(
        model, train_loader, val_loader, device, cfg, ckpt_path, label_cols,
        run_name=name)

    tag = _tag(name)
    history_df.to_csv(os.path.join(cfg.OUTPUT_DIR, f"{tag}_history.csv"), index=False)
    per_class_report(y_true, y_prob, cfg.SIGMOID_THRESHOLD, label_cols).to_csv(
        os.path.join(cfg.OUTPUT_DIR, f"{tag}_per_class_report.csv"), index=False)
    np.savez(os.path.join(cfg.OUTPUT_DIR, f"{tag}_val_predictions.npz"),
             y_true=y_true, y_prob=y_prob, label_cols=np.array(label_cols))

    result = dict(final_val_metrics)
    result.update(arm=arm_id, variant=name, kind=kind, resample=resample,
                  n_params=int(n_params), best_epoch=int(checkpoint["epoch"]),
                  n_train=len(train_loader.dataset))
    return result, history_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--arms", type=int, nargs="+", default=list(ARMS))
    args = ap.parse_args()

    cfg.EPOCHS = args.epochs
    cfg.EARLY_STOPPING = False
    cfg.OUTPUT_DIR = "outputs/experiments/"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    _cache = os.path.join(cfg.OUTPUT_DIR, "melspec_cache")
    if not os.path.exists(_cache):
        os.symlink(os.path.abspath("outputs/run_v2/melspec_cache"), _cache)

    device = torch.device(cfg.DEVICE)
    print(f"Device: {device} | epocas/arm: {cfg.EPOCHS} | semilla: {cfg.RANDOM_SEED} | arms: {args.arms}")

    # Metricas de balance del dataset antes/despues del resampling (Tabla 3
    # del paper). Solo depende del CSV, no del entrenamiento.
    bm_path = os.path.join(cfg.OUTPUT_DIR, "balance_metrics.json")
    if not os.path.exists(bm_path):
        try:
            import balance_metrics as _bm
            _bm.main()
        except Exception as e:
            print(f"(balance_metrics omitido: {e})")

    results, histories = [], {}
    for a in args.arms:
        res, hist = run_arm(a, device)
        results.append(res)
        histories[ARMS[a][0]] = hist
        pd.DataFrame(results).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "experiments_results.csv"), index=False)
        print(f"\n>>> ARM {a} listo: global_acc={res['global_acc']:.2f} "
              f"f1={res['f1']:.2f} mAP={res['mAP']:.2f}")

    df = pd.DataFrame(results)
    cols = ["arm", "variant", "kind", "resample", "n_params", "n_train",
            "best_epoch", "global_acc", "macro_acc", "precision", "recall", "f1", "mAP"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(cfg.OUTPUT_DIR, "experiments_results.csv"), index=False)
    print(f"\n{'='*72}\n=== RESULTADOS ===\n{'='*72}\n{df.to_string(index=False)}")

    with open(os.path.join(cfg.OUTPUT_DIR, "experiments_summary.json"), "w") as f:
        json.dump({"results": results, "epochs": cfg.EPOCHS,
                   "random_seed": cfg.RANDOM_SEED}, f, indent=2)

    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = ["global_acc", "f1", "mAP"]
    x = np.arange(len(metrics)); w = 0.8 / max(len(df), 1)
    for i, row in df.reset_index(drop=True).iterrows():
        off = (i - (len(df) - 1) / 2) * w
        vals = [row[m] for m in metrics]
        b = ax.bar(x + off, vals, w, label=row["variant"])
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(["global_acc", "macro-F1", "mAP"])
    ax.set_ylabel("%"); ax.set_title("Comparacion de arms (20% val)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "experiments_comparison.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, h in histories.items():
        ax.plot(h["epoch"], h["val_mAP"], label=name)
    ax.set_xlabel("Epoca"); ax.set_ylabel("val_mAP (%)")
    ax.set_title("Convergencia por arm"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "experiments_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"\nGuardado en {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
