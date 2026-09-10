"""
Métricas de evaluación multi-etiqueta equivalentes a las usadas en el paper
(Sección 5.1, Ecuaciones 17-23):

    - global_acc: exact match ratio (todo el vector de etiquetas debe coincidir)
    - macro_acc / precision / recall / F1: promedio macro por clase
    - mAP: mean Average Precision (usa las probabilidades, no las binarias)
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score,
)
import pandas as pd

def global_accuracy(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """Ecuación (17): exige coincidencia exacta de todo el vector de etiquetas."""
    exact_match = np.all(y_true == y_pred_bin, axis=1)
    return exact_match.mean() * 100


def per_class_accuracy(y_true: np.ndarray, y_pred_bin: np.ndarray) -> np.ndarray:
    return (y_true == y_pred_bin).mean(axis=0) * 100


def macro_metrics(y_true: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    return {
        "macro_acc": per_class_accuracy(y_true, y_pred_bin).mean(),
        "precision": precision_score(y_true, y_pred_bin, average="macro", zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred_bin, average="macro", zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred_bin, average="macro", zero_division=0) * 100,
    }


def mean_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue  # clase ausente en el split -> no definida, se omite
        ap = average_precision_score(y_true[:, c], y_prob[:, c])
        aps.append(ap)
    return (np.mean(aps) * 100) if aps else 0.0


def evaluate_all(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    # Guardia defensiva: si por algún motivo llegan NaN/inf en las
    # probabilidades (ej. divergencia numérica puntual), no debe crashear
    # todo el cálculo de métricas -- se sustituyen por 0.5 (máxima incerteza).
    n_bad = (~np.isfinite(y_prob)).sum()
    if n_bad > 0:
        print(f"⚠️  {n_bad} valores no finitos (NaN/inf) en las predicciones "
              f"al calcular métricas; se reemplazan por 0.5.")
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    y_pred_bin = (y_prob >= threshold).astype(int)
    metrics = {
        "global_acc": global_accuracy(y_true, y_pred_bin),
        "mAP": mean_average_precision(y_true, y_prob),
    }
    metrics.update(macro_metrics(y_true, y_pred_bin))
    return metrics


def per_class_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                      class_names) -> pd.DataFrame:
    import pandas as pd
    y_pred_bin = (y_prob >= threshold).astype(int)
    rows = []
    for i, name in enumerate(class_names):
        support = int(y_true[:, i].sum())
        acc = (y_true[:, i] == y_pred_bin[:, i]).mean() * 100
        prec = precision_score(y_true[:, i], y_pred_bin[:, i], zero_division=0) * 100
        rec = recall_score(y_true[:, i], y_pred_bin[:, i], zero_division=0) * 100
        f1 = f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0) * 100
        ap = average_precision_score(y_true[:, i], y_prob[:, i]) * 100 if support > 0 else float("nan")
        rows.append({"class": name, "support": support, "accuracy": acc,
                      "precision": prec, "recall": rec, "f1": f1, "AP": ap})
    return pd.DataFrame(rows)