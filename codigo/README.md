# TResNet-LA — implementación

## El dataset

`train.csv` es un dataset de **audio bioacústico multi-etiqueta** (42
columnas tipo `SPHSUR`, `BOABIS`, `DENNAH`... — códigos de especies de
anuros, dataset INCT, sitios INCT4/INCT17/INCT41/INCT20955), donde cada
`.wav` puede contener 0, 1 o varias especies vocalizando a la vez.

**Estructuralmente el problema es el mismo que el del paper**: clasificación
multi-etiqueta de audio donde hay que reconocer qué clases están presentes
suprimiendo "interferencia" (otras especies, ruido de fondo). Por eso este
código implementa **fielmente la arquitectura TResNet-LA y la metodología**
del paper (Mel-spectrogram → SpaceToDepth → bloques Basic/Bottleneck con SE →
Sigmoid multi-label → Asymmetric Loss), generalizada a las 42 clases del
dataset en vez de las 3 del paper.

El CSV solo trae `filename` + etiquetas, no los audios — los `.wav` reales
deben estar en una carpeta aparte para poder entrenar.

## Cómo correr esto

1. Conseguir la carpeta con los `.wav` que aparecen en la columna `filename`
   del CSV (ej. `INCT20955_20190909_050000_0_3.wav`).
2. Editar `config.py`:
   ```python
   AUDIO_DIR = "/ruta/a/la/carpeta/de/wavs"
   ```
3. Instalar dependencias (idealmente en un entorno con GPU):
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecutar:
   ```bash
   python train.py                 # entrenamiento único
   python experiments.py           # los 10 arms (ablación + resampling + backbones)
   python cross_site.py            # leave-one-site-out
   python mtd_synth.py --checkpoint outputs/experiments/TResNet-LA_best.pt
   python predict.py --test_dir <carpeta test/>
   python analyze_results.py
   ```

## Estructura del proyecto

| Archivo | Contenido |
|---|---|
| `config.py` | Todos los hiperparámetros y rutas (editar `AUDIO_DIR` acá) |
| `model.py` | Arquitectura TResNet-LA: `SpaceToDepth`, `SEBlock`, `BasicBlockSE`, `BottleneckBlockSE`, red completa. Soporta `use_se=False` para el estudio de ablación |
| `losses.py` | Asymmetric Loss (γ_neg=6, γ_pos=0, como en el paper) |
| `dataset.py` | Carga de audio, extracción de Mel-spectrograma, split 80/20 train/validación estratificado |
| `metrics.py` | global_acc, macro_acc, precision/recall/F1, mAP (igual al paper) |
| `resampling.py` | UABOS y MSW (los dos métodos de resampling del paper) |
| `baselines.py` | ResNet50, ResNet50-A, TResNet-M, TResNet-MA — para la comparación de backbones |
| `balance_metrics.py` | Métricas de balance del dataset (LCard, MeanIR, CVIR, etc. — Ec. 1-10 del paper) |
| `train.py` | Entrena, valida cada época, guarda el mejor checkpoint (por mAP) y snapshots por época |
| `ablation.py` | Ablación directa: TResNet-LA (con SE) vs. la misma arquitectura sin atención de canal, mismo split/semilla |
| `experiments.py` | Orquestador de los 10 arms: ablación SE + resampling + comparación de backbones |
| `cross_site.py` | Leave-one-site-out sobre los 4 sitios de grabación — análisis de generalización |
| `mtd_synth.py` | Análogo sintético al análisis TIR/MTD del paper (mezcla clips a razón de interferencia controlada) |
| `predict.py` | Inferencia pura sobre una carpeta sin etiquetas (`test/`) |
| `validate_audio.py` | Chequea integridad de todos los `.wav` referenciados en el CSV antes de entrenar |
| `analyze_results.py` | Gráficos, análisis de errores, sweep de umbral y resumen en Markdown |

## Split de datos: 80% train / 20% validación

Como pide la consigna del proyecto, el split es **80/20** (no hay un tercer
conjunto de test etiquetado). La semilla es fija y queda documentada en
`config.py`:

```python
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20
RANDOM_SEED = 42
```

`train.py` imprime explícitamente la semilla usada al arrancar, y el split
es 100% reproducible: correr `train.py`, `evaluate_checkpoint.py` o
`ablation.py` con el mismo CSV y el mismo `RANDOM_SEED` siempre da exactamente
las mismas filas en train y en validación.

La carpeta `test/` (sin etiquetas) queda fuera de este split — se usa aparte
solo para generar predicciones con `predict.py`, ya que sin etiquetas no se
pueden calcular métricas ahí.

## Estudio de ablación

```bash
python ablation.py
```

Entrena dos variantes **idénticas en arquitectura y con el mismo split/
semilla**, y solo difieren en si tienen o no los bloques de atención de canal
(Squeeze-and-Excitation):

- **con_SE**: TResNet-LA completo (el modelo del paper).
- **sin_SE**: mismo backbone (misma profundidad, mismos bloques Basic/
  Bottleneck), pero sin ningún módulo SE — equivalente a un TResNet-L "puro".

Esto aísla exactamente la contribución del mecanismo de atención de canal,
replicando la lógica de la Tabla 8 del paper (donde comparan TResNet-LA vs.
TResNet-L y reportan hasta +4 puntos de accuracy por el SE).

Genera:
- `ablation_results.csv` — tabla comparativa (global_acc, macro_acc, precision, recall, F1, mAP, nº de parámetros)
- `ablation_comparison.png` — gráfico de barras comparando ambas variantes
- `ablation_convergence.png` — curvas de val_global_acc por época, ambas variantes
- `ablation_summary.json` — resumen en JSON, incluye la diferencia de global_acc atribuible al SE

## Qué hace cada etapa

- **Preprocesamiento** (`dataset.py::extract_melspec`): ventana Hanning 1024
  puntos, hop=320, 128 bandas Mel, banda 0-11 025 Hz (Nyquist real del
  dataset), clips de 3 s a 22.05 kHz → espectrograma de 208×128 (adaptación
  de la Sección 5.1 del paper — ver `docs/GAP_ANALISIS.md`).
- **Split**: 80% train / 20% validación (requisito del proyecto), preservando
  balance multi-label, semilla fija en `config.RANDOM_SEED`, con fallback a
  split aleatorio para combinaciones de etiquetas muy raras.
- **Modelo** (`model.py`): sigue exactamente Fig. 7 y Tabla C1 — SpaceToDepth
  (block=4) → conv 1×1 → 2 etapas BasicBlock con SE (r=4) → 1 etapa
  Bottleneck con SE (r=8) → 1 etapa Bottleneck sin SE → SE final (r=16) →
  GlobalAvgPool → FC → Sigmoid. ~53.1 M parámetros.
- **Loss**: Asymmetric Loss asimétrica para mitigar el desbalance
  positivo/negativo característico de multi-label (Sección 3.2 y 5.1).
- **Métricas** (`metrics.py`): exactamente las Ecuaciones (17)-(23) del
  paper: global_acc (exact match), macro_acc/precision/recall/F1 (macro
  promedio por clase), mAP.

## Notas de implementación

- Los espectrogramas se cachean como `.npy` en `OUTPUT_DIR/melspec_cache/`
  la primera vez que se leen, para no recalcularlos en cada época.
- Se aplica un SpecAugment ligero (enmascarado de tiempo/frecuencia) solo en
  train, sembrado por worker (`_seed_worker`) para que no se duplique entre
  procesos del DataLoader — dado el fuerte desbalance de clases del dataset
  (algunas clases como `SCIFUS`/`SCINAS` tienen 0 muestras, y varias tienen
  <50). Se puede desactivar pasando `augment=False` en `train.py`.
- El umbral de binarización (`SIGMOID_THRESHOLD = 0.8`) es el que usa el
  paper (confirmado en el Apéndice B) — `analyze_results.py::threshold_sweep`
  chequea si sigue siendo óptimo para 42 clases con desbalance más extremo.
