"""
Copia/renombra los outputs del arm 'TResNet-LA' de experiments.py a los
nombres de archivo que espera analyze_results.py (training_history.csv,
final_val_metrics.json, per_class_report.csv, val_predictions.npz), sin
tocar los archivos originales por-arm (quedan para comparar los 10 arms).
"""

import json
import os
import shutil

import pandas as pd

OUT = "outputs/experiments"
ARM = "TResNet-LA"


def main():
    shutil.copy(os.path.join(OUT, f"{ARM}_history.csv"),
               os.path.join(OUT, "training_history.csv"))
    shutil.copy(os.path.join(OUT, f"{ARM}_per_class_report.csv"),
               os.path.join(OUT, "per_class_report.csv"))
    shutil.copy(os.path.join(OUT, f"{ARM}_val_predictions.npz"),
               os.path.join(OUT, "val_predictions.npz"))

    results = pd.read_csv(os.path.join(OUT, "experiments_results.csv"))
    row = results[results["variant"] == ARM].iloc[0]
    metrics_cols = ["global_acc", "mAP", "macro_acc", "precision", "recall", "f1", "loss"]
    final_metrics = {c: float(row[c]) for c in metrics_cols if c in row and pd.notna(row[c])}
    with open(os.path.join(OUT, "final_val_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Consolidado: training_history.csv, per_class_report.csv, "
         f"val_predictions.npz, final_val_metrics.json (arm={ARM})")


if __name__ == "__main__":
    main()
