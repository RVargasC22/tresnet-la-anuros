# Notas del paper — Chen et al. 2026, TResNet-LA (UAMTR)

Datos extraídos del PDF (`Chen2026_TResNet-LA_UAMTR.txt`). Referencia rápida
para la presentación y para implementar UABOS/MSW.

## Problema

Reconocer **tipo y número** de blancos (buques) desde ruido radiado mezclado,
bajo **interferencia no-objetivo** (buques que NO interesan). Formulado como
clasificación **multi-etiqueta** a nivel de muestra.

- **TIR** (Target-to-Interference Ratio): energía de cada blanco / energía de la
  interferencia. Rango de TIR más angosto ⇒ tarea más fácil.

## Datasets (MTD) — construidos desde ShipsEar

3 clases de blanco: PF, MB, RV + interferencia (NTInf). Etiqueta = vector de 3
bits `[PF, MB, RV]`; `[0,0,0]` = solo interferencia/ruido.

| MTD | Cómo se mezcla | Rango TIR |
|---|---|---|
| SE_raw | señales crudas | ancho (p. ej. [-44, 38] dB) |
| SE_norm | normalización por máximo | medio |
| SE_en0 | normalización por energía | ≈ 0 dB (blanco = interferencia) |

Split del paper: **70 % / 15 % / 15 %** (train 26 831 / val 5 748 / test 5 756).
(Nuestra consigna pide 80/20, semilla 42.)

Métricas de balance propias: MeanIR, CVIR, MeanImR, CVImR (por clase, Ec. 2-7) +
versión *label-powerset* LP_MeanIR, LP_CVIR (Ec. 8-10). Antes de resampling
LP_MeanIR ≈ 14.

## Resampling (Sección 3.2) — las dos técnicas propuestas

### UABOS — Underwater Acoustic Borderline Oversampling (Algoritmo 2)

Versión mejorada de **MLBOS** (X.-Y. Zhang et al. 2023), adaptada a señales
acústicas submarinas con interferencia. Solo se aplica al **training set**.

1. Conversión **label powerset**: cada conjunto de etiquetas → una clase única.
   `N_j = |D_j|`, `d_l` = nº de clases powerset.
2. `D_emb = t-SNE(D)` a **2D** (reduce el costo del resampling; MLBOS no lo hace).
   `k = ½ · min(N_j)`.
3. Para cada muestra de **clase minoritaria** (`N_j` < promedio), calcular con
   sus `k` vecinos más cercanos en el embedding:
   `u_i = (1/K) · Σ_{j∈kNN} I(y_j ≠ y_i)` — fracción de vecinos con distinta
   etiqueta powerset. (Ec. 11)
4. Tipo de muestra (Ec. 12):
   - `ss` (safe) si `0 ≤ u_i < 0.5`
   - `ds` (dangerous / borderline) si `0.5 ≤ u_i ≤ 1`
   UABOS **incluye** el caso `u_i = 1` en `ds` (MLBOS lo excluía).
5. Para las muestras `ds`, nivel de peligro (Ec. 13):
   `γ_i = (N̄ − N_j) / ( |D_j^ds| · Σ_j (N̄ − N_j) )`
   donde `N̄` = media de muestras por clase, `N_j` = muestras de la clase j,
   `|D_j^ds|` = nº de muestras ds en la clase j.
6. Clonar `n · p · γ_i` copias de cada muestra `ds` (con `p` = tasa de
   resampling = **1**).
7. Resultado: LP_MeanIR 14.09 → 4.05.

### MSW — Minority Samples Weighted

Asigna un **peso entero de duplicación** a cada clase con nº de muestras por
debajo del promedio; se **duplican** las muestras de esas clases
sub-representadas. Los pesos se **ajustan experimentalmente** (no salen de las
métricas de balance).

- Paper: pesos finales PF = **3**, MB = **5**, RV = **5**.
- Resultado: LP_MeanIR → 5.18.

### Hallazgo clave del paper sobre resampling (Sección 5.4)

- Mejora `global_acc` en general **poco**: +1.25, +0.82, −0.17, +1.65, +1.79,
  +0.37 pp en los 6 grupos (TResNet-L y TResNet-LA × 3 MTD).
- En **SE_en0** (TIR 0 dB) el resampling **empeora** levemente a TResNet-LA:
  el modelo ya rinde bien, y duplicar muestras mete información redundante.
- Ayuda **más a TResNet-L que a TResNet-LA** (el modelo más débil se beneficia
  más del rebalanceo).
- 3 grupos mejoran con MSW, 2 con UABOS, 1 sin resampling.
- **"El balance de datos por sí solo no determina el desempeño final."**
  Tunear los pesos de MSW no necesariamente mejora la accuracy.
- El resampling ajusta **cantidad**, no **diversidad** → mejora acotada.
- Todas las tablas de la Sección 5.5 usan el dataset **MSW-resampleado**.

## Arquitectura TResNet-LA (Fig. 7, Tabla C1)

Entrada: Mel-espectrograma **500 × 128**.

| Capa | Output | Config |
|---|---|---|
| Space to depth | 125 × 32 | block size 4 |
| Conv1 | 125 × 32 | 3×3, 76 (ajuste de canales) |
| SE_conv1 | 125 × 32 | `[3×3,76; 3×3,76; SE(C64)] × 4` (Basic + SE) |
| SE_conv2 | 63 × 16 | `[3×3,152; 3×3,152; avgpool2×2 s2; 1×1,152; SE(C64)] × 5` |
| SE_conv3 | 32 × 8 | `[1×1,304; 3×3,304; 1×1,1216; avgpool2×2 s2; 1×1,1216; SE(C152)] × 18` (Bottleneck + SE) |
| Conv2 | 16 × 4 | `[1×1,608; 3×3,608; 1×1,2432; avgpool2×2 s2; 1×1,2432] × 3` (Bottleneck, **sin SE**) |
| SE block | 16 × 4 | C152 — **esta capa SE final es el "A" de TResNet-LA** |
| Global avg pool → FC → Sigmoid | 1 × 3 | |

- Repeticiones `[4, 5, 18, 3]` = TResNet-L. SE en los 3 primeros bloques, no en
  el 4°. **TResNet-LA = TResNet-L + capa SE final.**
- `IBN` = **Inplace-BatchNorm** (BatchNorm in-place, sin copias extra) — NO es
  Instance-Batch Norm. `nn.BatchNorm2d` es la implementación correcta.
- Downsampling = `average pool 2×2 stride 2` (lo dice la Tabla C1 explícita).
- SE block: squeeze = global avg pool (Ec. 14); excitation = 2 FC + σ (Ec. 15);
  scale = multiplicación canal a canal (Ec. 16). `r = 4 u 8`.
- Umbral σ para binarizar la salida Sigmoid (el paper no da el valor exacto en
  el texto principal / Tabla C2).

### Parámetros de entrenamiento (Tabla C2)

| | valor |
|---|---|
| sample rate Mel | **32000 Hz** |
| frame length / hop / FFT | 1024 / 320 / 1024 |
| Mel bins | 128 |
| ancho de banda | [0, 14000] Hz |
| ASL γ_neg / γ_pos | 6 / 0 |
| batch size | 32 |
| epochs | **80** |
| learning rate | 1e-4 |
| optimizador | AdamW |
| init | aleatoria, sin transfer learning |

Nota: `sample rate 32000` y banda `14 kHz` son los valores **del paper**
(ShipsEar es ~52.7 kHz nativo). Nuestro dataset es 22.05 kHz → usarlos implica
sobre-muestrear; las bandas Mel > 11 kHz quedan como artefactos. Decisión:
mantener los del paper y documentarlo.

## Métricas (Ec. 17-23)

- **global_acc** (Ec. 17): exact match — la predicción cuenta solo si acierta
  tipo Y cantidad de blancos. Métrica más estricta y la más sensible al modelo.
- **macro** (Ec. 18) y micro (Ec. 19) de: accuracy, precision, recall, F1, mAP.
- El paper usa **macro** como métrica principal (trata todas las clases igual).

## Resultados del paper

### Tabla 4 — TResNet-LA por rango de TIR (sin resampling)

| MTD | global_acc | macro_acc | precision | recall | F1 | mAP |
|---|---|---|---|---|---|---|
| SE_raw | 49.69 | 78.77 | 84.49 | 86.77 | 85.60 | 93.65 |
| SE_norm | 68.16 | 87.30 | 91.14 | 91.52 | 91.32 | 97.57 |
| SE_en0 | 91.38 | 96.77 | 97.39 | 98.21 | 97.80 | 99.71 |

### Tabla 6 — TResNet-LA con resampling (SE_raw / norm / en0)

| MTD | método | global_acc | F1 | mAP |
|---|---|---|---|---|
| SE_raw | none / UABOS / MSW | 49.69 / 50.66 / **50.94** | 85.60 / 85.92 / 85.96 | 93.65 / 93.75 / 93.62 |
| SE_norm | none / UABOS / MSW | 68.16 / 68.79 / **68.98** | 91.32 / 91.49 / 91.61 | 97.57 / 97.51 / 97.42 |
| SE_en0 | none / UABOS / MSW | **91.38** / 90.83 / 91.21 | 97.80 / 97.68 / 97.75 | 99.71 / 99.71 / 99.72 |

### Tabla 7 — TResNet-L con resampling (el resampling ayuda más acá)

SE_norm: none 63.19 → UABOS 64.37 → MSW 64.98 global_acc.

### Tabla 8 — Comparación de backbones (todo sobre dataset MSW-resampleado)

| MTD | ResNet50 | ResNet50-A | TResNet-M | TResNet-MA | TResNet-L | **TResNet-LA** |
|---|---|---|---|---|---|---|
| SE_raw global_acc | 43.12 | 44.26 | 47.64 | 49.32 | 47.41 | **50.94** |
| SE_norm global_acc | 63.00 | 66.93 | 63.63 | 66.41 | 64.98 | **68.98** |
| SE_en0 global_acc | 82.19 | 82.19 | 88.57 | 90.48 | 89.09 | **91.21** |

- El bloque de atención de canal (`-A`) sube global_acc ~2.8-4.0 pp (SE_norm) en
  las 3 arquitecturas. Excepción: ResNet50 vs ResNet50-A en SE_en0 (empatan).
- **TResNet-LA gana en las 3 MTD.** El "-A" (SE final) da hasta +4 pp.
- ResNet50/-A: recall alto pero precision baja (sobre-clasifican positivos).

## Conclusión / limitaciones del paper (Sección 6-7)

- MSW mejora `global_acc` de TResNet-LA en SE_raw en > 1 pp.
- TResNet-LA supera a todos los baselines (SOTA en UAMTR interferido).
- Limitaciones que reconocen: señales sintéticas (sesgo de distribución), falta
  de interpretabilidad, el resampling mejora balance pero **no diversidad** →
  mejora acotada. Futuro: generación de muestras, uncertainty, physics-informed.
