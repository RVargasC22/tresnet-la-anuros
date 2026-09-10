"""
Estudio de ablación: aísla la contribución del mecanismo de atención de
canal (Squeeze-and-Excitation) comparando dos variantes IDÉNTICAS en
profundidad/arquitectura, entrenadas con el MISMO split (80/20, misma
semilla) y los MISMOS hiperparámetros -- la única diferencia es la
presencia o ausencia de los bloques SE. Esto replica la lógica del paper
(Sección 5.5, Tabla 8): comparar TResNet-LA vs TResNet-L para mostrar
cuánto aporta el canal de atención.

Variantes comparadas:
    - "con_SE"  (TResNet-LA completo): SE en las 3 primeras etapas + SE final
    - "sin_SE"  (ablación): mismo backbone, sin ningún bloque SE

Uso:
    python ablation.py
    python ablation.py --epochs 15   # sobrescribe cfg.EPOCHS solo para la ablación
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
from train import set_seed, build_dataloaders, train_model
from metrics import per_class_report


def run_variant(use_se: bool, variant_name: str, train_loader, val_loader,
                 label_cols, device):
    print(f"\n{'='*70}\nEntrenando variante: {variant_name} (use_se={use_se})\n{'='*70}")

    # Reinicio de semilla ANTES de crear cada modelo: garantiza que la única
    # diferencia entre variantes sea la arquitectura (SE sí/no), no el
    # azar de la inicialización de pesos ni el orden de shuffling.
    set_seed(cfg.RANDOM_SEED)

    model = build_model(len(label_cols), cfg, use_se=use_se).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"ablation_{variant_name}_best.pt")
    history_df, final_val_metrics, y_true, y_prob, checkpoint = train_model(
        model, train_loader, val_loader, device, cfg, ckpt_path, label_cols,
        run_name=variant_name,
    )
    history_df.to_csv(
        os.path.join(cfg.OUTPUT_DIR, f"ablation_{variant_name}_history.csv"), index=False)

    report = per_class_report(y_true, y_prob, cfg.SIGMOID_THRESHOLD, label_cols)
    report.to_csv(
        os.path.join(cfg.OUTPUT_DIR, f"ablation_{variant_name}_per_class_report.csv"), index=False)

    result = dict(final_val_metrics)
    result["variant"] = variant_name
    result["use_se"] = use_se
    result["n_params"] = n_params
    result["best_epoch"] = checkpoint["epoch"]
    return result, history_df


def plot_comparison(results: list, out_path: str):
    df = pd.DataFrame(results)
    metrics_to_plot = ["global_acc", "macro_acc", "precision", "recall", "f1", "mAP"]

    x = np.arange(len(metrics_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, row in df.iterrows():
        offset = (i - (len(df) - 1) / 2) * width
        values = [row[m] for m in metrics_to_plot]
        bars = ax.bar(x + offset, values, width, label=row["variant"])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylabel("%")
    ax.set_title("Ablación: contribución del mecanismo de atención de canal (SE)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nGráfico de ablación guardado en: {out_path}")


def plot_training_curves_comparison(histories: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, hist in histories.items():
        ax.plot(hist["epoch"], hist["val_global_acc"], label=f"{name} (val_global_acc)")
    ax.set_xlabel("Época")
    ax.set_ylabel("val_global_acc (%)")
    ax.set_title("Ablación: convergencia con vs. sin atención de canal")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico de convergencia guardado en: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                         help="Sobrescribe cfg.EPOCHS solo para este estudio de ablación "
                              "(útil para correr una ablación más corta que el entrenamiento final)")
    args = parser.parse_args()

    if args.epochs is not None:
        cfg.EPOCHS = args.epochs

    # Ablacion justa: ambas variantes (con_SE / sin_SE) corren EXACTAMENTE las
    # mismas epocas. Con early stopping activo, con_SE se estancaba y cortaba
    # ~epoca 24 mientras sin_SE seguia hasta ~49, confundiendo la comparacion.
    cfg.EARLY_STOPPING = False

    set_seed(cfg.RANDOM_SEED)
    device = torch.device(cfg.DEVICE)
    print(f"Device: {device}")
    print(f"Semilla utilizada: {cfg.RANDOM_SEED} | Épocas por variante: {cfg.EPOCHS}")

    # Mismo split (80/20, misma semilla) para AMBAS variantes -> comparación justa.
    train_loader, val_loader, label_cols = build_dataloaders(
        cfg, error_log_name="bad_audio_files_ablation.log")

    results = []
    histories = {}

    for use_se, name in [(True, "con_SE"), (False, "sin_SE")]:
        result, history_df = run_variant(use_se, name, train_loader, val_loader, label_cols, device)
        results.append(result)
        histories[name] = history_df

    results_df = pd.DataFrame(results)
    cols_order = ["variant", "use_se", "n_params", "best_epoch", "global_acc",
                  "macro_acc", "precision", "recall", "f1", "mAP", "loss"]
    results_df = results_df[[c for c in cols_order if c in results_df.columns]]

    out_csv = os.path.join(cfg.OUTPUT_DIR, "ablation_results.csv")
    results_df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}\n=== RESULTADOS DE LA ABLACIÓN ===\n{'='*70}")
    print(results_df.to_string(index=False))
    print(f"\nGuardado en: {out_csv}")

    diff = results_df.loc[results_df["variant"] == "con_SE", "global_acc"].values[0] - \
        results_df.loc[results_df["variant"] == "sin_SE", "global_acc"].values[0]
    print(f"\n➡️  El mecanismo de atención de canal (SE) cambió global_acc en "
          f"{diff:+.2f} puntos porcentuales.")

    plot_comparison(results, os.path.join(cfg.OUTPUT_DIR, "ablation_comparison.png"))
    plot_training_curves_comparison(histories, os.path.join(cfg.OUTPUT_DIR, "ablation_convergence.png"))

    with open(os.path.join(cfg.OUTPUT_DIR, "ablation_summary.json"), "w") as f:
        json.dump({"results": results, "global_acc_diff_se_vs_no_se": diff,
                    "random_seed": cfg.RANDOM_SEED, "epochs": cfg.EPOCHS}, f, indent=2)


if __name__ == "__main__":
    main()
