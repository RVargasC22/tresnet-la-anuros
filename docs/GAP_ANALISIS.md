# Fidelidad al paper y decisiones de implementación

Este documento explica **qué se corrigió durante el desarrollo, qué se
adaptó al dataset y por qué** (bugs, decisiones, supuestos). Los **números
resultantes** de las corridas están en `docs/RESULTADOS.md`, no acá.

---

## 1. Bugs identificados y corregidos durante el desarrollo

### Correcciones de bugs

| Archivo | Bug | Fix |
|---|---|---|
| `losses.py` | `AsymmetricLoss.forward` no casteaba a fp32 → `log()` con probs≈0 en fp16 (autocast) daba **NaN**. Era la causa del `loss=NaN` en las primeras corridas. | `logits = logits.float()`. Verificado: val_loss ahora finito. |
| `train.py` (validación) | El camino de validación no tenía la guardia anti-NaN que sí tiene el de train. | El fix de `losses.py` lo resuelve de raíz. |
| `dataset.py` / `train.py` | `DataLoader(num_workers=N)` sin `worker_init_fn`: PyTorch siembra el RNG de `torch` por worker pero **no el de numpy** → los N workers generan el mismo SpecAugment. | `_seed_worker` + `generator` fijo. |
| `ablation.py` | Con early stopping activo, `con_SE` cortaba ~época 24 y `sin_SE` seguía a ~49 → **las dos variantes NO entrenaban las mismas épocas** → la ablación era inválida (y por eso daba que "el SE empeora", al revés del paper). | `cfg.EARLY_STOPPING = False` en los experimentos. |

### Mejoras de metodología

| Cambio | Por qué |
|---|---|
| Mejor checkpoint y early-stopping por **`mAP`** (no `global_acc`) | `global_acc` y `macro_acc` están inflados por los verdaderos negativos (42 clases, mayoría de etiquetas 0). Elegir por ellos puede quedarse con un modelo que "predice casi todo 0". `mAP` usa las probabilidades, es independiente del umbral. |
| `cudnn.deterministic = True` | Reproducibilidad de la ablación. Costo medido: ~0 (69 vs 64 ms/step). |
| **Anchos de la Tabla C1** (`stage_width_mult=(1,2,4,8)` → 76/152/304/608) | La primera versión del modelo tenía los Bottleneck a **media anchura** → ~15 M params. La Tabla C1 del paper da ~53 M (= TResNet-L). Ahora coincide. |
| Guardado por época: `<run>_history_live.csv`, `<run>_last.pt`, `<run>_snapshots/epXXX.pt` | Un corte no pierde horas; se puede evaluar cualquier época. |
| `val_predictions.npz` | Habilita análisis de errores por muestra. |

### Análisis nuevos (`analyze_results.py`)

- `plot_metrics_by_label_count`: desempeño (exact-match, micro-F1) **vs. nº de
  especies simultáneas por clip** — análogo directo al análisis de "cantidad de
  blancos mezclados" del paper (Sección 5.3).
- `error_analysis`: peores clases por F1, clases con más falsos positivos /
  falsos negativos.

### Módulos implementados para cubrir el paper completo

| Archivo | Qué implementa |
|---|---|
| `resampling.py` | **UABOS y MSW** (Sección 3.2 del paper). |
| `baselines.py` | **ResNet50, ResNet50-A, TResNet-M, TResNet-MA** (Tabla 8). |
| `experiments.py` | 1 script, 10 brazos, mismo split/semilla, sin early stopping: cubre ablación del SE + comparación de resampling (Tabla 6) + comparación de backbones (Tabla 8). |

---

## 2. Fidelidad respecto al paper

### ✅ Implementado y fiel

| Componente del paper | Referencia |
|---|---|
| Arquitectura TResNet-LA: SpaceToDepth(4), Basic×[4,5], Bottleneck×[18,3], SE en etapas 1-3 + **SE final** (el "A"), anchos 76/152/304/608, ~53 M params | Fig. 7, Tabla C1 |
| `IBN` = Inplace-BatchNorm (`nn.BatchNorm2d`) | texto Fig. 7 |
| Downsampling = `avgpool 2×2 stride 2` | Tabla C1 |
| SE block: squeeze (GAP) → excitation (2 FC + σ) → scale (mult. canal a canal) | Ec. 14-16 |
| Mel: 1024 Hann, hop 320, FFT 1024, 128 bins, banda [0, 14000] Hz, sr 32000 | Tabla C2 |
| Asymmetric Loss γ_neg=6, γ_pos=0; AdamW; LR 1e-4; batch 32; **80 épocas**; init aleatoria, sin transfer learning | Tabla C2, Sección 5.1 |
| Reformulación multi-etiqueta binaria (Sigmoid por clase) | Sección 3.2 |
| Métricas: global_acc (exact match), macro accuracy/precision/recall/F1, mAP | Ec. 17-23 |
| **UABOS** (label powerset → t-SNE 2D → kNN borderline `u_i` → tipo `ds` si `u_i≥0.5` → clonar `n·p·γ_i`) | Algoritmo 2, Ec. 11-13 |
| **MSW** (peso de duplicación por clase sub-representada) | Sección 3.2 |
| Resampling **solo sobre el training set** | Sección 3.2 |
| Comparación de backbones: ResNet50, ResNet50-A, TResNet-M, TResNet-MA, TResNet-L, TResNet-LA | Tabla 8, Tabla C3 |
| Análisis por cantidad de blancos/especies | Sección 5.3 (análogo) |

### ⚠️ Adaptado (con justificación — la consigna lo permite)

| Componente | Paper | Nuestra adaptación | Motivo |
|---|---|---|---|
| Dataset | ShipsEar, 3 clases barco, TIR controlado, 5 s @ 32 kHz | 42 especies anuros, 3 s @ 22.05 kHz, ya mezclado/etiquetado | Es el dataset de la consigna. Problema estructuralmente idéntico. |
| Entrada Mel | 500 × 128 (5 s) | 304 × 128 (3 s) | Duración de los clips del dataset. |
| Sample rate real | 52.7 kHz nativo → sr 32 kHz, fmax 14 kHz (Tabla C2) OK | 22.05 kHz nativo → **sr=22050, fmax=11025** (Nyquist real) | Usar los valores del paper sobre-muestrearía (interpola, no agrega info) y pediría ~15 % de bandas Mel por encima del Nyquist real (artefacto puro). Se adaptó al Nyquist real; `TARGET_TIME_FRAMES` 304→208. **Decisión activa, no "seguimos el paper a ciegas".** |
| Stem `Conv1` | Tabla C1 (texto) dice "3×3, stride 2" — inconsistente con su propia columna de Output Size (125×32, sin cambio). **Fig. 1 y Fig. 7 (diagramas reales, vistos como imagen) dicen explícitamente "Conv 1×1"**, dos veces, de forma independiente. | kernel **1×1**, stride 1 (sin padding) | Se probó cambiar a 3×3 primero (mala lectura de la tabla, antes de ver las figuras); revertido a 1×1 tras verificar Fig. 1/Fig. 7 con `poppler` + Read de imagen. Ver nota abajo. |
| Split | 70 / 15 / 15 (train / val / test etiquetado) | **80 / 20** (train / val); `test/` sin etiquetas | Requisito de la consigna. Semilla 42. |
| UABOS `k` | `k = ½·min(N_j)` | `k` fijo (10) | Con 42 bits hay muchas combinaciones powerset con 1 sola muestra → `min(N_j)=1` → `k=0`. |
| UABOS t-SNE | t-SNE sobre los datos | PCA→50 + openTSNE 2D sobre ~50 k mel-specs de ~39 k dims | Escala (el paper tenía ~27 k muestras). |
| MSW pesos | tuneados a mano (PF=3, MB=5, RV=5) | `w_i = clip(round(N̄/N_i), 2, 8)` automático | 42 clases, no se pueden tunear todas a mano. |

### ✅ Corrección: el umbral σ SÍ está en el paper

Una versión anterior de este análisis afirmaba que el paper no especificaba
σ. **Falso** — Apéndice B (pág. 19, verificado leyendo la página como imagen,
no solo el texto plano extraído):
*"Predicted probabilities greater than 0.8 are treated as 1, while the rest are treated as 0"*.
**σ=0.8 es del paper, no un valor heredado sin fuente.** El `threshold_sweep`
que se agregó sigue teniendo valor — no porque el paper no lo diga, sino para
chequear si 0.8 (tuneado por el paper para 3 clases de barco) sigue siendo
óptimo para 42 especies con desbalance mucho más extremo.

### ❌ No aplica (fuera de alcance justificado)

| Componente | Por qué no |
|---|---|
| Construcción de MTD (Algoritmo 1: mezclar target + interferencia a TIR) | El dataset ya viene mezclado y etiquetado. |
| Análisis por rango de TIR (Sección 5.2, Tabla 4) | No hay etiquetas de TIR en audio de ranas. |
| Validación cross-dataset con DeepShip (Apéndice D) | No se dispone de DeepShip. |

### ✅ Brechas cerradas

| Brecha | Cómo se cerró |
|---|---|
| Métricas de balance (Tabla 3 del paper) | `balance_metrics.py` — Ec. 1-10 completas, antes/después de MSW/UABOS |
| Sweep de umbral + umbral por clase tuneado | `analyze_results.py::threshold_sweep` + `per_class_thresholds` |
| TResNet-L + UABOS / + MSW (Tabla 7) | `experiments.py` arms 9-10 |
| Heatmap de co-ocurrencia / confusión | `analyze_results.py::cooccurrence_and_confusion` |
| Stem: verificado con Fig. 1/Fig. 7 (imagen) | `model.py` — kernel **1×1**, stride 1 (revertido; ver nota abajo) |
| Construcción de MTD + análisis por TIR (análogo) | `mtd_synth.py` — mezcla clips single-especie + ruido de fondo a TIR controlado |
| Cross-dataset / DeepShip (análogo) | `cross_site.py` — leave-one-site-out sobre los 4 sitios reales |
| `sr`/`fmax` Nyquist-correctos | `config.py` — 22050/11025 en vez de 32000/14000 (ver arriba) |
| `TResNet-M` depths | Confirmado `(3,4,11,3)` viendo Tabla C3 como imagen (antes era una suposición sin verificar) |

### 📖 Revisión completa del PDF como imagen (no solo texto)

`pypdf` extrae texto plano, que **rompe las matrices con corchetes** de las Tablas C1/C3
(las filas de cada bloque residual se aplanan en una sola línea, perdiendo la estructura).
Eso causó una lectura incorrecta del stem (ver arriba). Se instaló `poppler-utils` y se
releyeron las **25 páginas del PDF como imágenes**, incluyendo Fig. 1, 3, 5,
6, 7, 8, Tablas 1-8, Apéndices A-E completos. Confirmado: arquitectura (Fig. 1/7), reducción
SE `r=4/8/16` (Fig. 7, no las anotaciones "C64/C152" de la tabla — esas parecen otro
artefacto de esa tabla específica), posición del SE en el Bottleneck (después de conv3×3,
antes del conv1×1 de expansión — coincide con el código sin cambios), UABOS/MSW (Fig. 3 +
Algoritmo 2, sin PCA — el PCA-previo es adaptación nuestra por escala, documentada),
`TResNet-M` depths `(3,4,11,3)` (Tabla C3), y **σ=0.8** (Apéndice B, ver arriba).

### 🔧 Pendiente (opcional, bajo valor)

| Brecha | Costo |
|---|---|
| Visualización de predicción (espectrograma + probs) de muestras de `test/` (Apéndice B) | ~30 líneas, lindo para el video |
