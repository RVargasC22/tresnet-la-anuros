"""
Configuración central del pipeline TResNet-LA (UAMTR).

IMPORTANTE sobre el dataset:
El CSV entregado (train.csv) NO es el dataset ShipsEar/MTD descrito en el paper
(barcos PF/MB/RV + interferencia). Es un dataset de clasificación multi-etiqueta
de audio (42 clases, nombres tipo SPHSUR, BOABIS, etc. -> especies de anuros,
dataset INCT), donde cada .wav puede contener 0, 1 o varias especies vocalizando
simultáneamente. Estructuralmente el problema es IDÉNTICO al del paper:

    - Multi-label binary classification sobre audio.
    - Cada muestra puede tener 0..N etiquetas activas simultáneamente.
    - Se necesita suprimir "interferencia" (otras especies / ruido de fondo)
      para reconocer correctamente qué clases están presentes.

Por eso este código implementa la arquitectura TResNet-LA (ResNet + SE / channel
attention) y la metodología del paper (Mel-spectrogram -> SpaceToDepth -> Basic/
Bottleneck blocks con SE -> Sigmoid multi-label), generalizada a N clases en vez
de fijarla a 3, para que puedas entrenar directamente con tu CSV.

Solo tenés que ajustar AUDIO_DIR para que apunte a la carpeta donde están
físicamente los .wav listados en la columna "filename" del CSV.
"""

import os

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
CSV_PATH = "train.csv"   # csv con filename + labels
AUDIO_DIR = "train/"          # carpeta con TODOS los .wav (AJUSTAR a tu carpeta "train" completa)
OUTPUT_DIR = "outputs/uamtr_run/"     # checkpoints, logs, resultados
FILENAME_COL = "filename"

# ---------------------------------------------------------------------------
# Audio / Mel-spectrogram
# ---------------------------------------------------------------------------
# El paper usa sr=32000/fmax=14000 (Tabla C2) porque ShipsEar es ~52.7 kHz
# nativo -> esos valores tienen contenido real. Nuestro dataset es 22.05 kHz
# nativo (Nyquist = 11025 Hz): usar 32000/14000 sobre-muestrearia (interpola,
# no agrega informacion) y pediria bandas Mel por encima del Nyquist real
# (puro artefacto de interpolacion, ~15% de las bandas sin senal). Se adaptan
# los valores al Nyquist real del dataset -- desviacion deliberada y
# documentada del paper, justificada por la consigna ("si algun componente no
# puede aplicarse razonablemente, se puede simplificar/reemplazar").
SAMPLE_RATE = 22050          # Hz = sample rate nativo real del dataset (sin resamplear)
CLIP_DURATION = 3.0          # segundos por segmento -> tus .wav duran 3.0s exactos
N_FFT = 1024                 # ventana Hanning de 1024 puntos (igual al paper)
HOP_LENGTH = 320              # overlap de 320 frames (igual al paper)
N_MELS = 128                  # 128 bancos de filtros Mel (igual al paper)
FMIN = 0
FMAX = 11025                  # = Nyquist real (22050/2); el paper usa 14000 sobre 32000
# Nº de frames temporales: con 3.0s @ 22050Hz/hop=320 salen 207 frames
# (verificado con librosa), redondeado al múltiplo de 4 más cercano hacia
# arriba (exigido por SpaceToDepth block_size=4) -> 208.
TARGET_TIME_FRAMES = 208

# ---------------------------------------------------------------------------
# Split de datos
# ---------------------------------------------------------------------------
# Requisito del proyecto: 80% entrenamiento / 20% validación (sobre los datos
# ETIQUETADOS, es decir train.csv + AUDIO_DIR). La carpeta test/ NO tiene
# etiquetas, así que no forma parte de este split -- se usa aparte, solo para
# inferencia, con predict.py.
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20
RANDOM_SEED = 42   # semilla fija para que el split sea reproducible

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
# repeticiones de bloques igual que TResNet-L / TResNet-LA (Tabla C1 del paper)
STAGE_DEPTHS = [4, 5, 18, 3]          # BasicBlock, BasicBlock, Bottleneck, Bottleneck
STAGE_USE_SE = [True, True, True, False]  # las 3 primeras etapas con SE, la última no
FINAL_SE_REDUCTION = 16               # r=16 en la capa SE final del paper
SIGMOID_THRESHOLD = 0.8               # umbral de binarización (paper usa 0.8)
SPACE_TO_DEPTH_BLOCK = 4              # debe dividir exacto a TARGET_TIME_FRAMES y N_MELS

if TARGET_TIME_FRAMES % SPACE_TO_DEPTH_BLOCK != 0:
    raise ValueError(
        f"TARGET_TIME_FRAMES ({TARGET_TIME_FRAMES}) debe ser divisible por "
        f"SPACE_TO_DEPTH_BLOCK ({SPACE_TO_DEPTH_BLOCK}). Redondealo al múltiplo "
        f"de {SPACE_TO_DEPTH_BLOCK} más cercano hacia arriba."
    )
if N_MELS % SPACE_TO_DEPTH_BLOCK != 0:
    raise ValueError(
        f"N_MELS ({N_MELS}) debe ser divisible por SPACE_TO_DEPTH_BLOCK "
        f"({SPACE_TO_DEPTH_BLOCK})."
    )

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Asymmetric Loss (ASL) - Sección 5.1 del paper
ASL_GAMMA_NEG = 6
ASL_GAMMA_POS = 0
ASL_CLIP = 0.05   # probability margin estándar de ASL (Ridnik et al.)
ASL_EPS = 1e-8

# Early stopping: si val_global_acc no mejora al menos MIN_DELTA durante
# PATIENCE épocas consecutivas, se corta el entrenamiento (evita seguir
# entrenando horas cuando el modelo ya dejó de aprender algo significativo).
# Metrica de seleccion del mejor checkpoint. mAP porque usa las probabilidades
# (independiente del umbral) y no se infla con los verdaderos negativos como
# global_acc/macro_acc en este problema multi-label muy desbalanceado.
CHECKPOINT_METRIC = "mAP"

EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 0.1   # puntos porcentuales mínimos para contar como "mejora"
EARLY_STOPPING_METRIC = "mAP"    # una de las keys que devuelve evaluate_all()

def _resolve_device() -> str:
    """Detecta el device real disponible, en vez de asumir 'cuda' a ciegas.
    Evita el error confuso "Torch not compiled with CUDA enabled" que
    aparece recién al mover el modelo a la GPU -- acá se detecta antes y
    se avisa con un mensaje claro sobre cómo arreglarlo."""
    import torch
    if os.environ.get("FORCE_CPU") == "1":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    print("⚠️  CUDA no está disponible en esta instalación de PyTorch — "
          "se va a entrenar en CPU (mucho más lento). Si tenés GPU NVIDIA, "
          "probablemente instalaste la versión CPU-only de torch por error. "
          "Reinstalá con: pip install torch --index-url "
          "https://download.pytorch.org/whl/cu124")
    return "cpu"


DEVICE = _resolve_device()

os.makedirs(OUTPUT_DIR, exist_ok=True)