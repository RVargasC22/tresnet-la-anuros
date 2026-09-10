"""
Consolida los resultados del proyecto para el informe/exposición del curso.

Combina DOS fuentes de información distintas -- es importante no
confundirlas al escribir el informe:

    A) 20% de VALIDACIÓN (con etiquetas reales, generado automáticamente por
       train.py a partir del split 80/20 de train.csv). Acá SÍ hay ground
       truth, así que estas son las métricas "de verdad" (global_acc,
       macro_acc, precision/recall/F1, mAP) -- son las comparables con las
       Tablas 4/8 del paper.

    B) Carpeta test/ SIN etiquetas (predictions_binarized.csv /
       predictions_probabilities.csv generados por predict.py). Acá NO hay
       ground truth, por lo tanto NO se puede calcular accuracy/F1 real.
       Lo único que se puede hacer es un chequeo de sanidad: ¿el modelo
       predice una cantidad de etiquetas por muestra similar a la del
       training set? ¿predice las clases frecuentes con una tasa parecida
       a la del training set, o colapsó a predecir siempre lo mismo?

Genera en OUTPUT_DIR:
    - training_curves.png          (loss / accuracy por época, fuente: training_history.csv)
    - per_class_metrics.png        (accuracy/precision/recall/F1/AP por clase, fuente: A)
    - label_cardinality_check.png  (LCard train vs LCard predicciones test/, fuente A+B)
    - class_frequency_check.png    (tasa de positivos por clase: train vs predicciones test/, fuente A+B)
    - analysis_summary.md          (resumen en texto con los números clave, para pegar en el informe)

Uso:
    python analyze_results.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg


def load_available(path, loader):
    if os.path.exists(path):
        return loader(path)
    print(f"⚠️  No encontrado: {path} (se omite esa parte del análisis)")
    return None


def _binarize(y_prob, threshold):
    return (y_prob >= threshold).astype(int)


def plot_metrics_by_label_count(y_true, y_prob, threshold, out_path):
    """Desglosa las metricas segun cuantas especies hay realmente en el clip
    (0, 1, 2, 3+). Analogo al analisis de "cantidad de blancos mezclados" del
    paper: cuantos mas blancos simultaneos, mas dificil la tarea."""
    from sklearn.metrics import f1_score
    y_bin = _binarize(y_prob, threshold)
    n_active = y_true.sum(axis=1).astype(int)
    buckets = [("0", n_active == 0), ("1", n_active == 1),
               ("2", n_active == 2), ("3+", n_active >= 3)]
    rows = []
    for name, mask in buckets:
        if mask.sum() == 0:
            continue
        exact = float(np.all(y_true[mask] == y_bin[mask], axis=1).mean() * 100)
        f1 = float(f1_score(y_true[mask], y_bin[mask], average="micro", zero_division=0) * 100)
        rows.append({"n_especies": name, "n_muestras": int(mask.sum()),
                     "exact_match": exact, "micro_f1": f1})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["exact_match"], w, label="Exact match (%)")
    ax.bar(x + w / 2, df["micro_f1"], w, label="micro-F1 (%)")
    for i, r in enumerate(df.itertuples()):
        ax.text(i - w / 2, r.exact_match, f"{r.exact_match:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, r.micro_f1, f"{r.micro_f1:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.n_especies}\n(n={r.n_muestras})" for r in df.itertuples()])
    ax.set_xlabel("Nº de especies presentes en el clip (ground truth)")
    ax.set_ylabel("%")
    ax.set_title("Desempeño vs. cantidad de especies simultáneas")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")
    return df


def error_analysis(y_true, y_prob, threshold, label_cols, top_n=8):
    """Peores clases por F1 y clases con mas falsos positivos / negativos."""
    from sklearn.metrics import f1_score
    y_bin = _binarize(y_prob, threshold)
    rows = []
    for i, name in enumerate(label_cols):
        support = int(y_true[:, i].sum())
        fp = int(((y_bin[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_bin[:, i] == 0) & (y_true[:, i] == 1)).sum())
        f1 = float(f1_score(y_true[:, i], y_bin[:, i], zero_division=0) * 100)
        rows.append({"clase": name, "support": support, "f1": f1, "FP": fp, "FN": fn})
    df = pd.DataFrame(rows)
    worst = df[df["support"] > 0].sort_values("f1").head(top_n)
    most_fp = df.sort_values("FP", ascending=False).head(top_n)
    most_fn = df.sort_values("FN", ascending=False).head(top_n)
    return df, worst, most_fp, most_fn


def threshold_sweep(y_true, y_prob, out_path, grid=None):
    """Barre el umbral global y grafica exact-match / macro-F1 / micro-F1.
    El paper no da el valor del umbral sigma; esto lo elige sobre validacion."""
    from sklearn.metrics import f1_score
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in grid:
        yb = (y_prob >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "exact_match": float(np.all(y_true == yb, axis=1).mean() * 100),
            "macro_f1": float(f1_score(y_true, yb, average="macro", zero_division=0) * 100),
            "micro_f1": float(f1_score(y_true, yb, average="micro", zero_division=0) * 100),
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in ["exact_match", "macro_f1", "micro_f1"]:
        ax.plot(df["threshold"], df[col], marker="o", ms=3, label=col)
    best = df.loc[df["macro_f1"].idxmax()]
    ax.axvline(best["threshold"], ls="--", c="gray",
               label=f"mejor macro-F1 @ {best['threshold']:.2f}")
    ax.set_xlabel("Umbral σ"); ax.set_ylabel("%")
    ax.set_title("Barrido de umbral (20% val)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Guardado: {out_path}")
    return df, best


def per_class_thresholds(y_true, y_prob):
    """Umbral por clase que maximiza F1 en validacion. Devuelve el vector y
    la macro-F1 resultante, comparada con umbral fijo 0.5 y 0.8."""
    from sklearn.metrics import f1_score
    L = y_true.shape[1]
    thr = np.full(L, 0.5)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for c in range(L):
        if y_true[:, c].sum() == 0:
            continue
        f1s = [f1_score(y_true[:, c], (y_prob[:, c] >= t).astype(int), zero_division=0)
               for t in grid]
        thr[c] = float(grid[int(np.argmax(f1s))])
    yb_tuned = (y_prob >= thr).astype(int)
    out = {
        "macro_f1_thr_0.5": float(f1_score(y_true, (y_prob >= 0.5).astype(int),
                                           average="macro", zero_division=0) * 100),
        "macro_f1_thr_0.8": float(f1_score(y_true, (y_prob >= 0.8).astype(int),
                                           average="macro", zero_division=0) * 100),
        "macro_f1_per_class_tuned": float(f1_score(y_true, yb_tuned,
                                                   average="macro", zero_division=0) * 100),
    }
    return thr, out


def cooccurrence_and_confusion(y_true, y_prob, threshold, label_cols, out_path):
    """Dos matrices L x L:
       (a) co-ocurrencia real: P(clase j presente | clase i presente)
       (b) confusion multi-label: P(clase j predicha | clase i presente real)
    Ayuda a ver que especies se confunden / co-predicen."""
    yb = (y_prob >= threshold).astype(int)
    L = len(label_cols)
    co = np.zeros((L, L)); conf = np.zeros((L, L))
    for i in range(L):
        mask = y_true[:, i] == 1
        s = mask.sum()
        if s == 0:
            continue
        co[i] = y_true[mask].mean(axis=0)
        conf[i] = yb[mask].mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(2 * max(8, L * 0.32), max(7, L * 0.3)))
    for ax, M, title in [(axes[0], co, "Co-ocurrencia real  P(j | i real)"),
                         (axes[1], conf, "Predicción  P(j pred | i real)")]:
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(L)); ax.set_xticklabels(label_cols, rotation=90, fontsize=6)
        ax.set_yticks(range(L)); ax.set_yticklabels(label_cols, fontsize=6)
        ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Guardado: {out_path}")
    return co, conf


def plot_training_curves(history_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss (Asymmetric Loss)")
    axes[0].set_title("Curva de pérdida")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["val_global_acc"], label="val_global_acc")
    axes[1].plot(history_df["epoch"], history_df["val_macro_acc"], label="val_macro_acc")
    axes[1].plot(history_df["epoch"], history_df["val_mAP"], label="val_mAP")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("%")
    axes[1].set_title("Métricas de validación por época")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Evolución del entrenamiento (TResNet-LA)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def plot_per_class_metrics(report_df, out_path, top_n=20):
    df = report_df.dropna(subset=["AP"]).sort_values("support", ascending=False).head(top_n)
    if len(df) == 0:
        print("⚠️  per_class_report.csv no tiene clases con support>0; se omite el gráfico.")
        return

    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.5), 5))
    ax.bar(x - 1.5 * width, df["accuracy"], width, label="Accuracy")
    ax.bar(x - 0.5 * width, df["precision"], width, label="Precision")
    ax.bar(x + 0.5 * width, df["recall"], width, label="Recall")
    ax.bar(x + 1.5 * width, df["f1"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=90)
    ax.set_ylabel("%")
    ax.set_title(f"Métricas por clase en el test set interno (top {top_n} por support)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def label_cardinality(binary_matrix: np.ndarray) -> float:
    """Ecuación (1) del paper: promedio de etiquetas activas por muestra."""
    return binary_matrix.sum(axis=1).mean()


def plot_label_cardinality_check(train_df, label_cols, pred_bin_df, out_path):
    lcard_train = label_cardinality(train_df[label_cols].values)
    lcard_pred = label_cardinality(pred_bin_df[label_cols].values)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Train (train.csv)", "Predicciones (test/)"],
                   [lcard_train, lcard_pred], color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("Label Cardinality (promedio de clases activas por muestra)")
    ax.set_title("Chequeo de sanidad: LCard train vs. predicciones en test/")
    for b, v in zip(bars, [lcard_train, lcard_pred]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")
    return lcard_train, lcard_pred


def plot_class_frequency_check(train_df, label_cols, pred_bin_df, out_path, top_n=20):
    freq_train = train_df[label_cols].mean().sort_values(ascending=False)
    top_classes = freq_train.head(top_n).index
    freq_pred = pred_bin_df[label_cols].mean()

    x = np.arange(len(top_classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(top_classes) * 0.5), 5))
    ax.bar(x - width / 2, freq_train[top_classes] * 100, width, label="% positivos en train.csv")
    ax.bar(x + width / 2, freq_pred[top_classes] * 100, width, label="% positivos predichos en test/")
    ax.set_xticks(x)
    ax.set_xticklabels(top_classes, rotation=90)
    ax.set_ylabel("% de muestras con la clase activa")
    ax.set_title(f"Chequeo de sanidad: frecuencia de clase, train vs. predicciones (top {top_n})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default=cfg.OUTPUT_DIR)
    args = ap.parse_args()
    out_dir = args.output_dir

    history_df = load_available(os.path.join(out_dir, "training_history.csv"), pd.read_csv)
    val_metrics_dict = load_available(os.path.join(out_dir, "final_val_metrics.json"),
                                   lambda p: json.load(open(p)))
    report_df = load_available(os.path.join(out_dir, "per_class_report.csv"), pd.read_csv)
    pred_bin_df = load_available(os.path.join(out_dir, "predictions_binarized.csv"), pd.read_csv)
    train_df, label_cols = None, None
    if os.path.exists(cfg.CSV_PATH):
        train_df = pd.read_csv(cfg.CSV_PATH)
        label_cols = [c for c in train_df.columns if c != cfg.FILENAME_COL]

    summary_lines = ["# Resumen de resultados — TResNet-LA (UAMTR)\n"]

    # ------------------------------------------------------------------
    # A) Curvas de entrenamiento
    # ------------------------------------------------------------------
    if history_df is not None:
        plot_training_curves(history_df, os.path.join(out_dir, "training_curves.png"))
        _sel = "val_mAP" if "val_mAP" in history_df.columns else "val_global_acc"
        best_epoch = history_df.loc[history_df[_sel].idxmax()]
        _f1 = f", val_f1={best_epoch['val_f1']:.2f}%" if "val_f1" in history_df.columns else ""
        summary_lines.append(
            f"## Entrenamiento\n"
            f"- Épocas corridas: {len(history_df)}\n"
            f"- Mejor época (por {_sel}): {int(best_epoch['epoch'])} "
            f"con val_global_acc={best_epoch['val_global_acc']:.2f}%, "
            f"val_macro_acc={best_epoch['val_macro_acc']:.2f}%, "
            f"val_mAP={best_epoch['val_mAP']:.2f}%{_f1}\n"
        )

    # ------------------------------------------------------------------
    # B) Métricas finales en el test set INTERNO (con ground truth real)
    # ------------------------------------------------------------------
    if val_metrics_dict is not None:
        summary_lines.append(
            "## Métricas en el 20% de validación (con etiquetas reales)\n"
            "Split reproducible: 80% train / 20% validación, semilla="
            f"{cfg.RANDOM_SEED}. Estas son las métricas comparables con las "
            "Tablas 4/8 del paper (TResNet-LA sobre SE_raw/SE_norm/SE_en0):\n\n"
            "| Métrica | Valor |\n|---|---|\n" +
            "\n".join(f"| {k} | {v:.2f} |" for k, v in val_metrics_dict.items()) + "\n"
        )
    if report_df is not None:
        plot_per_class_metrics(report_df, os.path.join(out_dir, "per_class_metrics.png"))

    # ------------------------------------------------------------------
    # B2) Analisis por muestra del 20% de validacion (val_predictions.npz):
    #     - desempeño vs. nº de especies simultaneas (analogo "mixed target
    #       quantity" del paper)
    #     - analisis de errores: peores clases, falsos positivos/negativos
    # ------------------------------------------------------------------
    vp = load_available(os.path.join(out_dir, "val_predictions.npz"),
                        lambda p: np.load(p, allow_pickle=True))
    if vp is not None:
        y_true, y_prob = vp["y_true"], vp["y_prob"]
        vp_labels = list(vp["label_cols"]) if "label_cols" in vp else (label_cols or [])
        thr = cfg.SIGMOID_THRESHOLD

        cnt_df = plot_metrics_by_label_count(
            y_true, y_prob, thr, os.path.join(out_dir, "metrics_by_species_count.png"))
        cnt_df.to_csv(os.path.join(out_dir, "metrics_by_species_count.csv"), index=False)
        summary_lines.append(
            "## Desempeño vs. cantidad de especies simultáneas en el clip\n"
            "Análogo al análisis de \"cantidad de blancos mezclados\" del paper: "
            "más especies a la vez => tarea más difícil.\n\n"
            "| Nº especies | Nº muestras | Exact match | micro-F1 |\n|---|---|---|---|\n" +
            "\n".join(f"| {r.n_especies} | {r.n_muestras} | {r.exact_match:.1f} | {r.micro_f1:.1f} |"
                      for r in cnt_df.itertuples()) + "\n"
        )

        err_df, worst, most_fp, most_fn = error_analysis(y_true, y_prob, thr, vp_labels)
        err_df.to_csv(os.path.join(out_dir, "error_analysis_per_class.csv"), index=False)
        summary_lines.append(
            "## Análisis de errores (20% de validación)\n"
            "**Peores clases por F1 (con support > 0):**\n\n"
            "| Clase | support | F1 | FP | FN |\n|---|---|---|---|---|\n" +
            "\n".join(f"| {r.clase} | {r.support} | {r.f1:.1f} | {r.FP} | {r.FN} |"
                      for r in worst.itertuples()) + "\n\n"
            "**Clases con más falsos positivos:** " +
            ", ".join(f"{r.clase} ({r.FP})" for r in most_fp.itertuples()) + "\n\n"
            "**Clases con más falsos negativos:** " +
            ", ".join(f"{r.clase} ({r.FN})" for r in most_fn.itertuples()) + "\n"
        )

        # Barrido de umbral (el paper no da el valor de sigma)
        sweep_df, best = threshold_sweep(
            y_true, y_prob, os.path.join(out_dir, "threshold_sweep.png"))
        sweep_df.to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)
        thr_vec, thr_cmp = per_class_thresholds(y_true, y_prob)
        np.save(os.path.join(out_dir, "per_class_thresholds.npy"), thr_vec)
        summary_lines.append(
            "## Selección de umbral (el paper no especifica σ)\n"
            f"- Mejor umbral global por macro-F1: **{best['threshold']:.2f}** "
            f"(macro-F1 {best['macro_f1']:.1f}, exact-match {best['exact_match']:.1f})\n"
            f"- macro-F1 con σ=0.5: {thr_cmp['macro_f1_thr_0.5']:.1f} | "
            f"σ=0.8: {thr_cmp['macro_f1_thr_0.8']:.1f} | "
            f"**umbral por clase tuneado en val: {thr_cmp['macro_f1_per_class_tuned']:.1f}**\n"
        )

        # Co-ocurrencia y confusion entre especies
        cooccurrence_and_confusion(
            y_true, y_prob, thr, vp_labels,
            os.path.join(out_dir, "cooccurrence_confusion.png"))

    # ------------------------------------------------------------------
    # B3) Metricas de balance del dataset (Ec. 1-10 del paper, Tabla 3)
    # ------------------------------------------------------------------
    bm = load_available(os.path.join(out_dir, "balance_metrics.json"),
                        lambda p: json.load(open(p)))
    if bm is not None:
        summary_lines.append(
            "## Balance del dataset (Ec. 1-10 del paper) — antes/después del resampling\n\n"
            "| Conjunto | N | LCard | MeanIR | CVIR | MeanImR | LP_MeanIR | LP_CVIR |\n"
            "|---|---|---|---|---|---|---|---|\n" +
            "\n".join(
                f"| {k} | {v['N']} | {v['LCard']:.2f} | {v['MeanIR']:.1f} | {v['CVIR']:.2f} | "
                f"{v['MeanImR']:.1f} | {v['LP_MeanIR']:.1f} | {v['LP_CVIR']:.2f} |"
                for k, v in bm.items()) + "\n\n"
            "Menor = más balanceado. Equivalente a la Tabla 3 del paper "
            "(UABOS/MSW reducen fuerte el desbalance label-powerset).\n"
        )

    # ------------------------------------------------------------------
    # C) Chequeo de sanidad sobre la carpeta test/ SIN etiquetas
    # ------------------------------------------------------------------
    if pred_bin_df is not None and train_df is not None:
        lcard_train, lcard_pred = plot_label_cardinality_check(
            train_df, label_cols, pred_bin_df,
            os.path.join(out_dir, "label_cardinality_check.png"))
        plot_class_frequency_check(
            train_df, label_cols, pred_bin_df,
            os.path.join(out_dir, "class_frequency_check.png"))

        n_zero_labels = (pred_bin_df[label_cols].sum(axis=1) == 0).sum()
        pct_zero = 100 * n_zero_labels / len(pred_bin_df)

        summary_lines.append(
            "## Chequeo de sanidad sobre test/ (SIN ground truth)\n"
            "⚠️ No hay etiquetas reales para esta carpeta, así que NO se puede "
            "calcular accuracy/F1 -- esto es solo un chequeo de coherencia:\n\n"
            f"- Label Cardinality en train.csv: **{lcard_train:.2f}** clases activas/muestra en promedio\n"
            f"- Label Cardinality en predicciones de test/: **{lcard_pred:.2f}** clases activas/muestra en promedio\n"
            f"- Muestras de test/ sin ninguna clase predicha: {n_zero_labels} ({pct_zero:.1f}%)\n\n"
            "Si estos dos valores de Label Cardinality son muy distintos entre sí "
            "(ej. predicciones muy por debajo de train), es señal de que el modelo "
            "está siendo demasiado conservador (colapsando a predecir 'nada') -- "
            "algo consistente con lo que describe el paper (Sección 5.2) sobre el "
            "sesgo del modelo frente al desbalance de clases.\n"
        )
    elif pred_bin_df is None:
        summary_lines.append(
            "## Chequeo de sanidad sobre test/\n"
            "No se encontró predictions_binarized.csv -- corré predict.py primero.\n"
        )

    summary_path = os.path.join(out_dir, "analysis_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n✅ Resumen guardado en: {summary_path}")
    print("   (pegalo o adaptalo directamente en el informe/exposición)")


if __name__ == "__main__":
    main()
