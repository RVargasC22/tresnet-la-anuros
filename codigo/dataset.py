"""
Dataset multi-etiqueta para audio: carga .wav, extrae Mel-spectrogram con los
parámetros del paper (Sección 5.1) y entrega (tensor, etiquetas).

Uso esperado:
    - config.CSV_PATH  -> csv con columna "filename" + una columna por clase (0/1)
    - config.AUDIO_DIR -> carpeta donde están físicamente los .wav referenciados

Si tus audios no están en formato "clips de 5s ya recortados" (como en el
paper), extract_melspec recorta/rellena automáticamente para que todos los
espectrogramas tengan el mismo shape (TARGET_TIME_FRAMES, N_MELS).
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import librosa


class AudioLoadError(Exception):
    """Error propio para audios ilegibles, con la ruta del archivo incluida
    en el mensaje (a diferencia de LibsndfileError, que en Windows a veces
    no se puede convertir a texto)."""
    pass



@dataclass
class MelConfig:
    """Config liviana y 100% picklable con solo los valores necesarios para
    extraer el Mel-spectrogram. Reemplaza pasar el módulo `config` completo
    dentro del Dataset: los módulos de Python NO se pueden picklear, y en
    Windows el DataLoader con num_workers>0 usa 'spawn' (necesita picklear
    el Dataset para mandarlo a cada proceso worker). En Linux/Mac (fork) esto
    no pasaba, por eso el bug no aparecía en pruebas fuera de Windows."""
    sample_rate: int
    clip_duration: float
    n_fft: int
    hop_length: int
    n_mels: int
    fmin: int
    fmax: int
    target_time_frames: int
    filename_col: str

    @classmethod
    def from_config_module(cls, cfg):
        return cls(
            sample_rate=cfg.SAMPLE_RATE,
            clip_duration=cfg.CLIP_DURATION,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS,
            fmin=cfg.FMIN,
            fmax=cfg.FMAX,
            target_time_frames=cfg.TARGET_TIME_FRAMES,
            filename_col=cfg.FILENAME_COL,
        )


def extract_melspec(wav_path: str, mel_cfg: MelConfig) -> np.ndarray:
    """Carga un audio y devuelve su Mel-spectrogram en dB, shape
    (TARGET_TIME_FRAMES, N_MELS), recortado/rellenado a duración fija.

    Si el archivo no se puede leer (corrupto, vacío, formato no soportado,
    etc.), lanza AudioLoadError con la ruta exacta incluida en el mensaje
    -- en Windows, soundfile a veces no puede convertir su propio error a
    texto (`<exception str() failed>`), así que acá NO se usa str(e) del
    error original, solo su tipo, para evitar ese problema."""
    try:
        y, sr = librosa.load(wav_path, sr=mel_cfg.sample_rate, mono=True)
    except Exception as e:
        raise AudioLoadError(
            f"No se pudo leer el audio: '{wav_path}' "
            f"(error interno: {type(e).__name__})"
        ) from None

    target_len = int(mel_cfg.sample_rate * mel_cfg.clip_duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=mel_cfg.sample_rate, n_fft=mel_cfg.n_fft, hop_length=mel_cfg.hop_length,
        n_mels=mel_cfg.n_mels, fmin=mel_cfg.fmin, fmax=mel_cfg.fmax, window="hann",
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (N_MELS, T)
    mel_db = mel_db.T                               # (T, N_MELS)

    # Ajustar al número de frames objetivo (recorte/padding por si el largo
    # calculado difiere en 1-2 frames por redondeo de librosa).
    t = mel_db.shape[0]
    target_t = mel_cfg.target_time_frames
    if t < target_t:
        mel_db = np.pad(mel_db, ((0, target_t - t), (0, 0)), mode="edge")
    elif t > target_t:
        mel_db = mel_db[:target_t, :]

    # Normalización min-max simple por muestra a [0, 1] (estabiliza entrenamiento)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > 1e-6:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_db = np.zeros_like(mel_db)

    return mel_db.astype(np.float32)


class MultiLabelAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols, audio_dir: str, mel_cfg: MelConfig,
                 cache_dir: str = None, augment: bool = False, error_log_path: str = None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.audio_dir = audio_dir
        self.mel_cfg = mel_cfg
        self.cache_dir = cache_dir
        self.augment = augment
        self.error_log_path = error_log_path
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _log_error(self, filename: str, err: Exception):
        msg = f"{filename}\t{type(err).__name__}\t{err}\n"
        if self.error_log_path:
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(msg)

    def _load_features(self, filename: str) -> np.ndarray:
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, filename + ".npy")
            if os.path.exists(cache_path):
                return np.load(cache_path)

        wav_path = os.path.join(self.audio_dir, filename)
        try:
            mel = extract_melspec(wav_path, self.mel_cfg)
        except (AudioLoadError, FileNotFoundError, OSError, RuntimeError) as e:
            # Un archivo corrupto/faltante NO debe tirar abajo todo el
            # entrenamiento: se registra en el log y se sustituye por un
            # espectrograma de silencio (todo ceros) del shape esperado.
            self._log_error(filename, e)
            mel = np.zeros((self.mel_cfg.target_time_frames, self.mel_cfg.n_mels),
                            dtype=np.float32)

        if cache_path:
            np.save(cache_path, mel)
        return mel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row[self.mel_cfg.filename_col]
        mel = self._load_features(filename)

        if self.augment:
            mel = self._augment(mel)

        labels = row[self.label_cols].values.astype(np.float32)

        x = torch.from_numpy(mel).unsqueeze(0)   # (1, T, F)
        y = torch.from_numpy(labels)              # (num_classes,)
        return x, y

    @staticmethod
    def _augment(mel: np.ndarray) -> np.ndarray:
        """SpecAugment ligero: enmascarado aleatorio de bandas de tiempo/
        frecuencia, útil dado el desbalance de clases descrito en el paper."""
        mel = mel.copy()
        t, f = mel.shape
        if np.random.rand() < 0.5:
            t0 = np.random.randint(0, max(t - t // 10, 1))
            mel[t0: t0 + t // 10, :] = 0.0
        if np.random.rand() < 0.5:
            f0 = np.random.randint(0, max(f - f // 10, 1))
            mel[:, f0: f0 + f // 10] = 0.0
        return mel


def load_dataframe(cfg):
    df = pd.read_csv(cfg.CSV_PATH)
    label_cols = [c for c in df.columns if c != cfg.FILENAME_COL]
    return df, label_cols


def stratified_train_val_split(df: pd.DataFrame, label_cols, cfg):
    """
    Split TRAIN/VALIDATION estratificado de forma aproximada para multi-label
    (requisito del proyecto: 80% train / 20% validación, semilla fija en
    cfg.RANDOM_SEED para reproducibilidad).

    Usa el patrón de etiquetas (label powerset simplificado) como estrato
    cuando hay suficientes muestras por combinación, y hace fallback a split
    aleatorio simple para combinaciones raras (evita que sklearn falle por
    clases con muy pocos miembros).
    """


    rng = cfg.RANDOM_SEED
    y = df[label_cols].values
    pattern = ["".join(row.astype(int).astype(str)) for row in y]
    df = df.copy()
    df["_pattern"] = pattern

    # Una combinación de etiquetas necesita al menos 2 muestras para poder
    # estratificarse (train_test_split exige >=2 por clase); si tiene menos,
    # se resuelve con split aleatorio simple.
    counts = df["_pattern"].value_counts()
    rare_patterns = counts[counts < 2].index
    df_rare = df[df["_pattern"].isin(rare_patterns)]
    df_ok = df[~df["_pattern"].isin(rare_patterns)]

    if len(df_ok) > 0:
        train_ok, val_ok = train_test_split(
            df_ok, test_size=(1 - cfg.TRAIN_RATIO), random_state=rng,
            stratify=df_ok["_pattern"],
        )
    else:
        train_ok = val_ok = df_ok

    if len(df_rare) > 0:
        train_rare, val_rare = train_test_split(
            df_rare, test_size=(1 - cfg.TRAIN_RATIO), random_state=rng,
        )
    else:
        train_rare = val_rare = df_rare

    train_df = pd.concat([train_ok, train_rare]).drop(columns=["_pattern"]).reset_index(drop=True)
    val_df = pd.concat([val_ok, val_rare]).drop(columns=["_pattern"]).reset_index(drop=True)

    print(f"Split reproducible con RANDOM_SEED={cfg.RANDOM_SEED}: "
          f"{len(train_df)} train ({cfg.TRAIN_RATIO*100:.0f}%) / "
          f"{len(val_df)} val ({cfg.VAL_RATIO*100:.0f}%)")

    return train_df, val_df
