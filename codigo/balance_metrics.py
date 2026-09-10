"""
Metricas de balance de dataset multi-etiqueta del paper (Chen et al. 2026,
Ecuaciones 1-10). Se usan para cuantificar el desbalance antes y despues del
resampling (equivalente a la Tabla 3 del paper).

    LCard      (Ec. 1)  : promedio de etiquetas activas por muestra
    IRLbl_i    (Ec. 2)  : conteo de la clase mayor / conteo de la clase i
    MeanIR     (Ec. 3)  : promedio de IRLbl
    CVIR       (Ec. 4)  : coef. de variacion de IRLbl (std muestral / MeanIR)
    ImR_i      (Ec. 5)  : max(|D+_i|,|D-_i|) / min(|D+_i|,|D-_i|)
    MeanImR    (Ec. 6)  : promedio de ImR
    CVImR      (Ec. 7)  : coef. de variacion de ImR
    LP_MeanIR  (Ec. 9)  : MeanIR sobre las clases label-powerset
    LP_CVIR    (Ec. 10) : CVIR sobre las clases label-powerset

Uso CLI:
    python balance_metrics.py            # train.csv completo + split 80/20 + tras MSW/UABOS
"""

import numpy as np


def _cv(values):
    """Coef. de variacion con std muestral (denominador n-1), como Ec. 4/7."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or v.mean() == 0:
        return 0.0
    return float(v.std(ddof=1) / v.mean())


def balance_metrics(Y):
    """Y: matriz binaria (N, L). Devuelve dict con todas las metricas."""
    Y = np.asarray(Y, dtype=int)
    N, L = Y.shape
    pos = Y.sum(axis=0).astype(float)                     # |D+_i| por clase
    neg = N - pos                                         # |D-_i|

    lcard = float(Y.sum(axis=1).mean())

    max_pos = pos.max() if pos.max() > 0 else 1.0
    with np.errstate(divide="ignore"):
        ir = np.where(pos > 0, max_pos / np.where(pos == 0, 1, pos), np.nan)
    ir_valid = ir[np.isfinite(ir)]
    mean_ir = float(ir_valid.mean()) if len(ir_valid) else 0.0
    cv_ir = _cv(ir_valid)

    both = (pos > 0) & (neg > 0)
    imr = np.maximum(pos[both], neg[both]) / np.minimum(pos[both], neg[both])
    mean_imr = float(imr.mean()) if len(imr) else 0.0
    cv_imr = _cv(imr)

    # label powerset
    lp = np.array(["".join(map(str, row)) for row in Y])
    _, counts = np.unique(lp, return_counts=True)
    counts = counts.astype(float)
    lp_ir = counts.max() / counts
    lp_mean_ir = float(lp_ir.mean())
    lp_cv_ir = _cv(lp_ir)

    return {
        "N": int(N), "L": int(L), "n_powerset_classes": int(len(counts)),
        "LCard": round(lcard, 3),
        "MeanIR": round(mean_ir, 3), "CVIR": round(cv_ir, 3),
        "MeanImR": round(mean_imr, 3), "CVImR": round(cv_imr, 3),
        "LP_MeanIR": round(lp_mean_ir, 3), "LP_CVIR": round(lp_cv_ir, 3),
        "n_zero_pos_classes": int((pos == 0).sum()),
    }


def _print_row(name, m):
    print(f"{name:22s} N={m['N']:>6d}  LCard={m['LCard']:.2f}  "
          f"MeanIR={m['MeanIR']:.2f}  CVIR={m['CVIR']:.2f}  "
          f"MeanImR={m['MeanImR']:.1f}  CVImR={m['CVImR']:.2f}  "
          f"LP_MeanIR={m['LP_MeanIR']:.2f}  LP_CVIR={m['LP_CVIR']:.2f}")


def main():
    import json
    import os
    import config as cfg
    from dataset import load_dataframe, stratified_train_val_split, MelConfig
    from resampling import msw_resample, uabos_resample

    df, label_cols = load_dataframe(cfg)
    train_df, val_df = stratified_train_val_split(df, label_cols, cfg)

    rows = {}
    rows["train.csv (completo)"] = balance_metrics(df[label_cols].values)
    rows["train split (80%)"] = balance_metrics(train_df[label_cols].values)
    rows["val split (20%)"] = balance_metrics(val_df[label_cols].values)

    mel_cfg = MelConfig.from_config_module(cfg)
    cache_dir = "outputs/run_v2/melspec_cache"

    msw_df, _ = msw_resample(train_df, label_cols, verbose=False)
    rows["train + MSW"] = balance_metrics(msw_df[label_cols].values)

    try:
        uabos_df, _ = uabos_resample(train_df, label_cols, mel_cfg, cache_dir,
                                     cfg.AUDIO_DIR, verbose=False)
        rows["train + UABOS"] = balance_metrics(uabos_df[label_cols].values)
    except Exception as e:
        print(f"(UABOS omitido: {e})")

    print()
    for name, m in rows.items():
        _print_row(name, m)

    out = os.path.join("outputs/experiments", "balance_metrics.json")
    os.makedirs("outputs/experiments", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
