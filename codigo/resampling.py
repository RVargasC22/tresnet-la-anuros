"""
Metodos de resampling del paper (Chen et al. 2026, Seccion 3.2), adaptados al
dataset multi-etiqueta de 42 clases de anuros.

    - MSW  (Minority Samples Weighted): peso entero de duplicacion por clase con
      conteo por debajo del promedio; se duplican las muestras de esas clases.
      Paper: PF=3, MB=5, RV=5 (tuneados a mano). Aca: w_i = clip(round(N_bar/N_i), 2, w_max).

    - UABOS (Underwater Acoustic Borderline Oversampling, Algoritmo 2): version
      de MLBOS (X.-Y. Zhang et al. 2023). Label powerset -> t-SNE 2D de los
      mel-spectrogramas -> para cada muestra de clase minoritaria, u_i = fraccion
      de k vecinos con distinta etiqueta powerset (Ec. 11) -> tipo 'ds' si
      u_i >= 0.5 (Ec. 12) -> clonar n*p*gamma_i copias, gamma_i por Ec. 13.

Adaptaciones documentadas (dataset != ShipsEar):
    - k: el paper usa k = 1/2 * min(N_j); con 42 bits hay muchas combinaciones
      con 1 sola muestra -> min(N_j)=1 -> k=0. Se usa k fijo (default 10).
    - t-SNE sobre ~50k mel-specs de ~39k dims: PCA a 50 dims primero, luego
      openTSNE (FFT, ~2-3 min).
    - Resampling SOLO sobre el training set (igual que el paper).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MSW
# ---------------------------------------------------------------------------
def msw_weights(train_df, label_cols, w_max=8):
    """Peso entero de duplicacion por clase sub-representada (N_i < N_bar)."""
    counts = {c: int(train_df[c].sum()) for c in label_cols}
    present = [n for n in counts.values() if n > 0]
    n_bar = float(np.mean(present))
    weights = {}
    for c in label_cols:
        n = counts[c]
        if 0 < n < n_bar:
            weights[c] = int(np.clip(round(n_bar / n), 2, w_max))
        else:
            weights[c] = 1
    return weights, n_bar


def msw_resample(train_df, label_cols, w_max=8, verbose=True):
    """Duplica cada muestra segun el maximo peso MSW de sus clases positivas."""
    weights, n_bar = msw_weights(train_df, label_cols, w_max)
    lab = train_df[label_cols].values.astype(int)
    reps = np.ones(len(train_df), dtype=int)
    wvec = np.array([weights[c] for c in label_cols])
    for i in range(len(train_df)):
        pos = np.where(lab[i] == 1)[0]
        if len(pos):
            reps[i] = int(wvec[pos].max())
    out = train_df.loc[train_df.index.repeat(reps)].reset_index(drop=True)
    if verbose:
        dup = {c: w for c, w in weights.items() if w > 1}
        print(f"[MSW] N_bar={n_bar:.0f} | clases duplicadas ({len(dup)}): {dup}")
        print(f"[MSW] train {len(train_df)} -> {len(out)} muestras (x{len(out)/len(train_df):.2f})")
    return out, {"weights": weights, "n_bar": n_bar,
                 "n_before": len(train_df), "n_after": len(out)}


# ---------------------------------------------------------------------------
# UABOS
# ---------------------------------------------------------------------------
def _powerset_labels(train_df, label_cols):
    y = train_df[label_cols].values.astype(int)
    lp = np.array(["".join(map(str, row)) for row in y])
    classes, lp_idx = np.unique(lp, return_inverse=True)
    return lp_idx, np.bincount(lp_idx)


def _load_mel_matrix(train_df, mel_cfg, cache_dir, audio_dir):
    """Matriz (n, T*F) con los mel-specs (usa el cache .npy si existe)."""
    import os
    from dataset import extract_melspec
    n = len(train_df)
    fn_col = mel_cfg.filename_col
    dim = mel_cfg.target_time_frames * mel_cfg.n_mels
    X = np.zeros((n, dim), dtype=np.float32)
    for i, filename in enumerate(train_df[fn_col].values):
        cache_path = os.path.join(cache_dir, filename + ".npy") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            mel = np.load(cache_path)
        else:
            mel = extract_melspec(os.path.join(audio_dir, filename), mel_cfg)
        X[i] = mel.reshape(-1)
    return X


def uabos_resample(train_df, label_cols, mel_cfg, cache_dir, audio_dir,
                   p=1.0, k=10, pca_dim=50, seed=42, verbose=True):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from openTSNE import TSNE

    lp_idx, Nj = _powerset_labels(train_df, label_cols)
    n_classes = len(Nj)
    N_bar = Nj.mean()

    if verbose:
        print(f"[UABOS] {n_classes} clases label-powerset | N_bar={N_bar:.1f} | "
              f"min={Nj.min()} max={Nj.max()}")
        print(f"[UABOS] cargando {len(train_df)} mel-specs...")
    X = _load_mel_matrix(train_df, mel_cfg, cache_dir, audio_dir)

    if verbose:
        print(f"[UABOS] PCA {X.shape[1]} -> {pca_dim} + t-SNE -> 2D...")
    Xp = PCA(n_components=min(pca_dim, X.shape[1]), random_state=seed).fit_transform(X)
    del X
    emb = np.asarray(TSNE(n_components=2, random_state=seed, n_jobs=-1,
                          verbose=False).fit(Xp))

    minority = Nj[lp_idx] < N_bar
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]

    u = np.zeros(len(train_df))
    for i in np.where(minority)[0]:
        u[i] = np.mean(lp_idx[idx[i]] != lp_idx[i])          # Ec. 11
    is_ds = minority & (u >= 0.5)                            # Ec. 12

    deficit = np.clip(N_bar - Nj, 0.0, None)
    sum_deficit = deficit.sum()
    ds_per_class = np.array([np.sum(is_ds & (lp_idx == j)) for j in range(n_classes)])

    gamma = np.zeros(len(train_df))
    for i in np.where(is_ds)[0]:
        j = lp_idx[i]
        if ds_per_class[j] > 0 and sum_deficit > 0:
            gamma[i] = deficit[j] / (ds_per_class[j] * sum_deficit)   # Ec. 13

    n_clones = np.floor(sum_deficit * p * gamma).astype(int)
    clone_rows = np.repeat(np.arange(len(train_df)), n_clones)
    out = pd.concat([train_df, train_df.iloc[clone_rows]]).reset_index(drop=True)

    if verbose:
        print(f"[UABOS] borderline (ds): {is_ds.sum()} | clones: {len(clone_rows)}")
        print(f"[UABOS] train {len(train_df)} -> {len(out)} muestras (x{len(out)/len(train_df):.2f})")
    return out, {"n_powerset_classes": int(n_classes), "n_ds": int(is_ds.sum()),
                 "n_clones": int(len(clone_rows)), "k": k,
                 "n_before": len(train_df), "n_after": len(out)}


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def resample_training_df(train_df, label_cols, method, mel_cfg=None,
                         cache_dir=None, audio_dir=None, seed=42):
    """method: 'none' | 'msw' | 'uabos'. Devuelve (df_resampleado, stats)."""
    if method in (None, "none"):
        return train_df.reset_index(drop=True), {"method": "none",
                                                 "n_before": len(train_df),
                                                 "n_after": len(train_df)}
    if method == "msw":
        out, stats = msw_resample(train_df, label_cols)
    elif method == "uabos":
        out, stats = uabos_resample(train_df, label_cols, mel_cfg, cache_dir,
                                    audio_dir, seed=seed)
    else:
        raise ValueError(f"metodo de resampling desconocido: {method}")
    stats["method"] = method
    return out, stats
