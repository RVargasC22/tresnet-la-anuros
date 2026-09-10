"""
Genera predicciones del modelo entrenado sobre una carpeta de audios SIN
etiquetas (ej. tu carpeta test/), usando el mejor checkpoint guardado.

Como no hay ground truth, este script NO calcula métricas (accuracy, F1,
etc.) -- eso solo se puede hacer si tenés las etiquetas reales. Lo que
hace es correr el modelo sobre cada .wav y guardar:

    1. predictions_probabilities.csv -> filename + probabilidad (0-1) por
       clase (útil si necesitás un archivo de "submission" tipo Kaggle).
    2. predictions_binarized.csv     -> filename + 0/1 por clase, aplicando
       el umbral configurado (más fácil de leer / interpretar).

Uso:
    python predict.py --test_dir "C:\\ruta\\a\\test"
    python predict.py --test_dir "C:\\ruta\\a\\test" --checkpoint otro_modelo.pt --threshold 0.5
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config as cfg
from model import build_model
from dataset import MelConfig, extract_melspec, AudioLoadError


class UnlabeledAudioDataset(Dataset):
    """Igual que MultiLabelAudioDataset pero sin columnas de etiquetas --
    solo carga el audio y devuelve el tensor + el nombre de archivo."""

    def __init__(self, filenames, audio_dir, mel_cfg: MelConfig, error_log_path=None):
        self.filenames = filenames
        self.audio_dir = audio_dir
        self.mel_cfg = mel_cfg
        self.error_log_path = error_log_path

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        wav_path = os.path.join(self.audio_dir, filename)
        try:
            mel = extract_melspec(wav_path, self.mel_cfg)
        except (AudioLoadError, FileNotFoundError, OSError, RuntimeError) as e:
            if self.error_log_path:
                with open(self.error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{filename}\t{type(e).__name__}\t{e}\n")
            mel = np.zeros((self.mel_cfg.target_time_frames, self.mel_cfg.n_mels),
                            dtype=np.float32)
        x = torch.from_numpy(mel).unsqueeze(0)
        return x, filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                         help="Carpeta con los .wav a predecir (ej. tu carpeta test/)")
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))
    parser.add_argument("--threshold", type=float, default=cfg.SIGMOID_THRESHOLD)
    parser.add_argument("--recursive", action="store_true",
                         help="Buscar .wav también en subcarpetas de test_dir")
    parser.add_argument("--output_dir", type=str, default=cfg.OUTPUT_DIR,
                         help="Carpeta donde guardar las predicciones (default: cfg.OUTPUT_DIR)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(cfg.DEVICE)
    print(f"Device: {device}")
    print(f"Cargando checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    label_cols = checkpoint["label_cols"]
    print(f"Checkpoint de la época {checkpoint.get('epoch', '?')} | "
          f"{len(label_cols)} clases: {label_cols}")

    # Listar archivos .wav en test_dir
    pattern = "**/*.wav" if args.recursive else "*.wav"
    wav_paths = sorted(glob.glob(os.path.join(args.test_dir, pattern), recursive=args.recursive))
    filenames = [os.path.relpath(p, args.test_dir) for p in wav_paths]
    print(f"Encontrados {len(filenames)} archivos .wav en: {args.test_dir}")

    if len(filenames) == 0:
        print("⚠️  No se encontraron .wav. ¿Probaste con --recursive si están en subcarpetas?")
        return

    mel_cfg = MelConfig.from_config_module(cfg)
    error_log_path = os.path.join(args.output_dir, "bad_audio_files_predict.log")
    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    ds = UnlabeledAudioDataset(filenames, args.test_dir, mel_cfg, error_log_path)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                         num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = build_model(len(label_cols), cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_probs, all_filenames = [], []
    with torch.no_grad():
        for x, fnames in tqdm(loader, desc="Prediciendo"):
            x = x.to(device)
            logits = model(x, return_logits=True)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_filenames.extend(fnames)

    y_prob = np.concatenate(all_probs)
    y_prob = np.nan_to_num(y_prob, nan=0.5)  # guardia por si algún audio dio NaN

    # ------------------------------------------------------------------
    # 1) CSV de probabilidades (formato tipo submission)
    # ------------------------------------------------------------------
    df_prob = pd.DataFrame(y_prob, columns=label_cols)
    df_prob.insert(0, cfg.FILENAME_COL, all_filenames)
    prob_path = os.path.join(args.output_dir, "predictions_probabilities.csv")
    df_prob.to_csv(prob_path, index=False)

    # ------------------------------------------------------------------
    # 2) CSV binarizado (0/1 según threshold) -- más fácil de leer
    # ------------------------------------------------------------------
    y_bin = (y_prob >= args.threshold).astype(int)
    df_bin = pd.DataFrame(y_bin, columns=label_cols)
    df_bin.insert(0, cfg.FILENAME_COL, all_filenames)
    bin_path = os.path.join(args.output_dir, "predictions_binarized.csv")
    df_bin.to_csv(bin_path, index=False)

    print(f"\n✅ Predicciones guardadas en:")
    print(f"   {prob_path}")
    print(f"   {bin_path}")

    n_no_class = (y_bin.sum(axis=1) == 0).sum()
    print(f"\nResumen: {len(filenames)} audios procesados, "
          f"{n_no_class} sin ninguna clase por encima del umbral ({args.threshold}).")

    if os.path.exists(error_log_path):
        n_bad = sum(1 for _ in open(error_log_path, encoding="utf-8"))
        print(f"⚠️  {n_bad} archivo(s) no se pudieron leer (ver {error_log_path}).")


if __name__ == "__main__":
    main()