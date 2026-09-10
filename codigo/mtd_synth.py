"""
Analogo a "Algoritmo 1: Construct interfered MTDs" + "Seccion 5.2: Performance
across different TIRs" del paper, adaptado a nuestro dominio.

El paper mezcla senales AISLADAS de un solo barco (ShipsEar) con interferencia
de barcos no-objetivo a un TIR (Target-to-Interference Ratio) controlado, y
mide como degrada el reconocimiento al ensanchar el rango de TIR. Nuestro
dataset son grabaciones de campo ya mezcladas por la naturaleza -- no hay TIR
que medir directamente. Pero SI tenemos:

    - ~12.9k clips de UNA sola especie (label sum == 1)   -> "target" aislado
    - ~22.5k clips SIN ninguna especie (label sum == 0)   -> ruido de fondo
      real del sitio, hace el papel de "interferencia no-objetivo" (NTInf)

Este script construye un MTD sintetico (MTD_synth) siguiendo la misma logica
del Algoritmo 1 (Mix = suma de senales, sin normalizar por energia individual)
y evalua un checkpoint YA ENTRENADO en el, midiendo:

    (a) accuracy/F1 vs. TIR (target vs. ruido de fondo), analogo Fig. 9 / Tabla 4
    (b) accuracy/F1 vs. numero de especies mezcladas (k=1,2,3), analogo Fig. 12/13

Es inferencia pura (no reentrena) -- corre en minutos sobre un checkpoint de
TResNet-LA ya entrenado.

Uso:
    python mtd_synth.py --checkpoint outputs/experiments/TResNet-LA_best.pt
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib.pyplot as plt

import config as cfg
from model import build_model
from dataset import MelConfig, load_dataframe

RNG_SEED = 42
TIR_DB_GRID = [-20, -10, -5, 0, 5, 10, 20]     # rango del paper es mucho mas ancho
                                                # (hasta +-45dB); aca lo acotamos porque
                                                # nuestras clases raras tienen pocos clips
N_PER_CELL = 60          # muestras sintetizadas por (k, TIR) -- rapido, ~5-10 min total
MIN_CLIPS_PER_CLASS = 3  # clases con < 3 clips de una sola especie se excluyen del pool


def _load_wave(path, sr, target_len):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def _rms(y):
    return float(np.sqrt(np.mean(y ** 2)) + 1e-12)


def build_pools(cfg):
    """Pool de clips single-especie por clase + pool de clips vacios (ruido)."""
    df, label_cols = load_dataframe(cfg)
    lc = df[label_cols].sum(axis=1)
    single = df[lc == 1]
    empty = df[lc == 0]

    pools = {}
    for c in label_cols:
        files = single.loc[single[c] == 1, cfg.FILENAME_COL].tolist()
        if len(files) >= MIN_CLIPS_PER_CLASS:
            pools[c] = files
    noise_pool = empty[cfg.FILENAME_COL].tolist()
    print(f"[pools] {len(pools)}/{len(label_cols)} clases con >= {MIN_CLIPS_PER_CLASS} "
          f"clips single-especie | ruido de fondo disponible: {len(noise_pool)}")
    return pools, noise_pool, label_cols


def synthesize(pools, noise_pool, label_cols, sr, target_len, seed=RNG_SEED):
    """Genera MTD_synth: para cada k in {1,2,3} y cada TIR de la grilla,
    mezcla k clips single-especie (targets) con un clip de ruido a ese TIR."""
    rng = np.random.default_rng(seed)
    classes = list(pools.keys())
    rows = []
    waves = []

    for k in (1, 2, 3):
        if len(classes) < k:
            continue
        for tir_db in TIR_DB_GRID:
            for _ in range(N_PER_CELL):
                chosen = rng.choice(classes, size=k, replace=False)
                target_files = [rng.choice(pools[c]) for c in chosen]
                targets = [_load_wave(os.path.join(cfg.AUDIO_DIR, f), sr, target_len)
                          for f in target_files]
                target_mix = np.mean(targets, axis=0)  # Mix = sum/len, Algoritmo 1

                noise_file = rng.choice(noise_pool)
                noise = _load_wave(os.path.join(cfg.AUDIO_DIR, noise_file), sr, target_len)

                # escalar el ruido para lograr el TIR pedido: TIR_dB = 20*log10(rms_t/rms_n)
                rms_t, rms_n0 = _rms(target_mix), _rms(noise)
                desired_rms_n = rms_t / (10 ** (tir_db / 20))
                noise_scaled = noise * (desired_rms_n / rms_n0)

                mixed = target_mix + noise_scaled
                peak = np.abs(mixed).max()
                if peak > 1.0:
                    mixed = mixed / peak   # evitar clipping (no altera el TIR relativo)

                waves.append(mixed)
                label = np.zeros(len(label_cols), dtype=int)
                for c in chosen:
                    label[label_cols.index(c)] = 1
                rows.append({"k": k, "tir_db": tir_db, "classes": ",".join(chosen),
                             **{lc_: int(v) for lc_, v in zip(label_cols, label)}})

    meta = pd.DataFrame(rows)
    print(f"[synth] MTD_synth generado: {len(meta)} muestras "
          f"({len(TIR_DB_GRID)} TIR x {N_PER_CELL} x hasta 3 valores de k)")
    return np.stack(waves), meta


def waves_to_melspecs(waves, mel_cfg):
    out = np.zeros((len(waves), mel_cfg.target_time_frames, mel_cfg.n_mels), dtype=np.float32)
    for i, y in enumerate(waves):
        mel = librosa.feature.melspectrogram(
            y=y, sr=mel_cfg.sample_rate, n_fft=mel_cfg.n_fft, hop_length=mel_cfg.hop_length,
            n_mels=mel_cfg.n_mels, fmin=mel_cfg.fmin, fmax=mel_cfg.fmax, window="hann")
        mel_db = librosa.power_to_db(mel, ref=np.max).T
        t = mel_db.shape[0]
        if t < mel_cfg.target_time_frames:
            mel_db = np.pad(mel_db, ((0, mel_cfg.target_time_frames - t), (0, 0)), mode="edge")
        elif t > mel_cfg.target_time_frames:
            mel_db = mel_db[:mel_cfg.target_time_frames, :]
        mn, mx = mel_db.min(), mel_db.max()
        out[i] = (mel_db - mn) / (mx - mn) if mx - mn > 1e-6 else 0.0
    return out


def evaluate(model, X, label_cols, device, batch_size=64):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).unsqueeze(1).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(xb, return_logits=True)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs)


def plot_by(meta, y_true, y_prob, groupcol, xlabel, out_path, threshold):
    from sklearn.metrics import f1_score
    yb = (y_prob >= threshold).astype(int)
    rows = []
    for val, idx in meta.groupby(groupcol).groups.items():
        idx = np.array(idx)
        exact = float(np.all(y_true[idx] == yb[idx], axis=1).mean() * 100)
        f1 = float(f1_score(y_true[idx], yb[idx], average="micro", zero_division=0) * 100)
        rows.append({groupcol: val, "n": len(idx), "exact_match": exact, "micro_f1": f1})
    df = pd.DataFrame(rows).sort_values(groupcol)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df[groupcol], df["exact_match"], marker="o", label="Exact match (%)")
    ax.plot(df[groupcol], df["micro_f1"], marker="s", label="micro-F1 (%)")
    ax.set_xlabel(xlabel); ax.set_ylabel("%")
    ax.set_title(f"MTD sintético: desempeño vs. {xlabel}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Guardado: {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                    default="outputs/experiments/TResNet-LA_best.pt")
    ap.add_argument("--out_dir", type=str, default="outputs/mtd_synth")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(cfg.DEVICE)
    mel_cfg = MelConfig.from_config_module(cfg)

    pools, noise_pool, label_cols = build_pools(cfg)
    waves, meta = synthesize(pools, noise_pool, label_cols, mel_cfg.sample_rate,
                             int(mel_cfg.sample_rate * mel_cfg.clip_duration))

    print("[synth] extrayendo mel-spectrogramas...")
    X = waves_to_melspecs(waves, mel_cfg)
    y_true = meta[label_cols].values.astype(int)

    print(f"[eval] cargando checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ck_labels = ckpt["label_cols"]
    assert list(ck_labels) == list(label_cols), "orden de clases no coincide con el checkpoint"
    model = build_model(len(label_cols), cfg, use_se=True).to(device)
    model.load_state_dict(ckpt["model_state"])

    y_prob = evaluate(model, X, label_cols, device)
    thr = cfg.SIGMOID_THRESHOLD

    meta.to_csv(os.path.join(args.out_dir, "mtd_synth_meta.csv"), index=False)
    np.savez(os.path.join(args.out_dir, "mtd_synth_predictions.npz"),
             y_true=y_true, y_prob=y_prob, label_cols=np.array(label_cols))

    df_tir = plot_by(meta, y_true, y_prob, "tir_db", "TIR (dB, target vs. ruido de fondo)",
                     os.path.join(args.out_dir, "mtd_synth_by_tir.png"), thr)
    df_tir.to_csv(os.path.join(args.out_dir, "mtd_synth_by_tir.csv"), index=False)

    df_k = plot_by(meta, y_true, y_prob, "k", "Nº de especies mezcladas (k)",
                   os.path.join(args.out_dir, "mtd_synth_by_k.png"), thr)
    df_k.to_csv(os.path.join(args.out_dir, "mtd_synth_by_k.csv"), index=False)

    print("\n=== Por TIR ===\n", df_tir.to_string(index=False))
    print("\n=== Por k (especies mezcladas) ===\n", df_k.to_string(index=False))


if __name__ == "__main__":
    main()
