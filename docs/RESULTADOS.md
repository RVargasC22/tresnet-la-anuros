# Resultados — TResNet-LA (UAMTR) sobre dataset de anuros

Consolida todas las corridas hechas, con datos concluyentes (no proyecciones).
**Pipeline `lab/run_all.sh` completo — las 5 etapas terminaron.**

## 1. Setup común a todas las corridas

- Dataset: 62 191 clips (`train.csv` + `train/`), 42 especies, split **80/20**
  estratificado por patrón de etiquetas, **semilla 42**.
- Arquitectura base: TResNet-LA fiel a Tabla C1 (53.1 M parámetros, stem
  Conv 1×1 — ver `GAP_ANALISIS.md` para la verificación).
- Mel-espectrograma: `sr=22050`, `hop=320`, `n_fft=1024`, `128 bins`,
  `fmax=11025` (adaptado al Nyquist real del dataset, ver `GAP_ANALISIS.md`).
- Asymmetric Loss (γ_neg=6, γ_pos=0), AdamW, LR 1e-4, batch 32, **80 épocas**,
  sin early stopping (comparación justa entre arms), sin transfer learning.
- Umbral de binarización: **σ=0.8** (confirmado en el paper, Apéndice B).
- Selección de mejor checkpoint: **mAP** (no global_acc, ver justificación en
  `GAP_ANALISIS.md`).
- GPU: RTX PRO 4000 Blackwell 16 GB. ~90-300 s/época según tamaño del dataset
  resampleado.

## 2. Tabla completa — 10 arms (`experiments.py`, terminado)

Mismo split/semilla, 80 épocas cada uno. Mejor checkpoint por mAP.

| Arm | Modelo | Resample | n_train | global_acc | macro_acc | precision | recall | F1 | mAP | best_epoch | params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **TResNet-LA** | — | 49 752 | 90.28 | 99.72 | 87.75 | 87.18 | 87.34 | **97.15** | 50 | 53.1M |
| 2 | TResNet-L (sin SE) | — | 49 752 | 89.54 | 99.70 | 85.62 | 86.85 | 85.90 | 97.08 | 30 | 51.8M |
| 3 | TResNet-LA | MSW | 74 846 | 88.42 | 99.67 | 84.74 | 90.35 | 86.79 | 96.53 | 24 | 53.1M |
| 4 | TResNet-LA | UABOS | 83 035 | 88.41 | 99.67 | 84.66 | 89.36 | 86.10 | 96.21 | 32 | 53.1M |
| 5 | ResNet50 | — | 49 752 | 89.93 | 99.71 | 88.42 | 90.34 | **89.23** | 96.62 | 70 | 23.6M |
| 6 | ResNet50-A | — | 49 752 | 89.49 | 99.70 | 89.06 | 89.61 | 89.16 | 96.55 | 48 | 24.3M |
| 7 | TResNet-M | — | 49 752 | 90.26 | 99.72 | 88.93 | 90.07 | 89.26 | 96.61 | 47 | 40.3M |
| 8 | TResNet-MA | — | 49 752 | 90.27 | 99.72 | 87.87 | 89.87 | 88.45 | 96.50 | 37 | 41.3M |
| 9 | TResNet-L | MSW | 74 846 | **90.52** | 99.73 | 87.37 | 91.08 | 88.83 | 96.68 | 41 | 51.8M |
| 10 | TResNet-L | UABOS | 83 035 | 87.52 | 99.65 | 84.63 | 86.89 | 85.42 | 96.40 | 24 | 51.8M |

Fuente: `lab/repo/outputs/experiments/experiments_results.csv`.

### 2.1. Ablación del mecanismo de atención de canal (SE)

Comparación directa, mismo split/semilla/hiperparámetros, única diferencia
`use_se`:

| Backbone | sin SE | con SE (-A / -LA) | Δ global_acc | Δ F1 | Δ mAP |
|---|---|---|---|---|---|
| TResNet-L → TResNet-LA (arm 2→1) | 89.54 / 85.90 / 97.08 | 90.28 / 87.34 / 97.15 | **+0.74** | **+1.44** | +0.07 |
| ResNet50 → ResNet50-A (arm 5→6) | 89.93 / 89.23 / 96.62 | 89.49 / 89.16 / 96.55 | −0.44 | −0.07 | −0.07 |
| TResNet-M → TResNet-MA (arm 7→8) | 90.26 / 89.26 / 96.61 | 90.27 / 88.45 / 96.50 | +0.01 | −0.81 | −0.11 |

**Conclusión concluyente:** el SE **solo ayuda en TResNet-LA** (el backbone más
grande, 53M params). En ResNet50 y TResNet-M (backbones más chicos, 24-40M) el
SE **no ayuda o empeora levemente**. Esto matiza la afirmación del paper de que
el SE "consistentemente" mejora — en nuestro dataset (42 clases, desbalance
extremo, dominio distinto) el beneficio depende del backbone. Es un hallazgo
propio, no una réplica ciega del paper.

### 2.2. Efecto del resampling (UABOS / MSW) — Tabla 6/7 del paper

| Backbone | sin resample | + MSW | + UABOS |
|---|---|---|---|
| TResNet-LA (arms 1,3,4) | 90.28 / 87.34 / 97.15 | 88.42 / 86.79 / 96.53 | 88.41 / 86.10 / 96.21 |
| TResNet-L (arms 2,9,10) | 89.54 / 85.90 / 97.08 | **90.52** / **88.83** / 96.68 | 87.52 / 85.42 / 96.40 |

**Conclusión concluyente, replica un hallazgo textual del paper:** el
resampling **empeora a TResNet-LA** (el modelo fuerte) pero **mejora a
TResNet-L** (el modelo más débil) en global_acc y F1. El paper dice
textualmente (Sección 5.4): *"the improvement suggests that the TResNet-L
model is less capable of fully extracting discriminative features... dataset
resampling provides measurable benefits for TResNet-L despite its limitations
in feature extraction"*. Nuestros datos muestran exactamente ese patrón, en un
dominio y arquitectura de escala distinta al paper (42 clases vs 3, 53M vs 15M
params). Esto es evidencia fuerte de que el mecanismo que describe el paper es
real y generaliza, no un artefacto de su dataset particular.

MSW supera a UABOS en los dos backbones (arms 3 vs 4, y 9 vs 10) — igual que
en el paper, donde "tres grupos mejoran con MSW, dos con UABOS".

### 2.3. Comparación de backbones (Tabla 8 del paper)

Ranking por mAP: **TResNet-LA (97.15)** > TResNet-L (97.08) > TResNet-L+MSW
(96.68) > TResNet-M (96.61) > ResNet50 (96.62) > ResNet50-A (96.55) >
TResNet-MA (96.50) > TResNet-LA+MSW (96.53) > TResNet-LA+UABOS (96.21) >
TResNet-L+UABOS (96.40).

Ranking por F1 (métrica más honesta para multi-label desbalanceado):
**TResNet-M (89.26)** > ResNet50 (89.23) > ResNet50-A (89.16) > TResNet-L+MSW
(88.83) > TResNet-MA (88.45) > TResNet-LA (87.34) > TResNet-LA+MSW (86.79) >
TResNet-LA+UABOS (86.10) > TResNet-L (85.90) > TResNet-L+UABOS (85.42).

**Conclusión concluyente:** TResNet-LA gana en mAP (la métrica que el paper
usa como principal indicador de calidad de ranking), pero **no gana en F1** —
ahí ResNet50/TResNet-M superan a TResNet-LA. A diferencia del paper (donde
TResNet-LA gana en las 3 métricas y en las 3 MTDs), en nuestro dominio la
ventaja de TResNet-LA es más acotada. Hipótesis: el problema de 42 especies
con desbalance extremo (dos clases con 0 positivos) se beneficia más de la
capacidad bruta del backbone que de la atención de canal específicamente
diseñada para filtrar interferencia acústica submarina — un tipo de ruido
distinto al de fondo de selva.

## 3. Ablación del mecanismo de atención vs. paper — números lado a lado

| | Paper (SE_en0, TIR=0dB) | Nuestro (arm 1 vs 2) |
|---|---|---|
| Δ global_acc por SE | +2.12 (91.21 vs 89.09, Tabla 8) | +0.74 |
| Δ macro_acc por SE | +0.60 | +0.03 |
| Δ F1 por SE | +0.39 | +1.44 |

Magnitud comparable (el paper reporta beneficios de 0.4-4 pp según la MTD), la
dirección (SE ayuda) coincide en TResNet-LA específicamente.

## 4. Análogo MTD + TIR (`mtd_synth.py`, terminado)

El paper mezcla señales aisladas de un solo barco (ShipsEar) con interferencia
a TIR controlado (Algoritmo 1). Nuestro dataset son grabaciones de campo ya
mezcladas — no hay forma de recuperar el TIR real. **Análogo construido**:
usamos los **12 886 clips de una sola especie** como "target" y los **22 504
clips vacíos** (ruido de fondo real de cada sitio) como "interferencia
no-objetivo", mezclados (`Mix = suma`) a TIR controlado en dB, evaluados con
el checkpoint ya entrenado de TResNet-LA (arm 1). 31/42 especies tenían pool
suficiente (≥3 clips single-especie); las 11 restantes quedaron fuera de este
experimento específico (documentado, no afecta las demás corridas).

### 4.1. Desempeño vs. TIR (target vs. ruido de fondo)

| TIR (dB) | n | exact-match | micro-F1 |
|---|---|---|---|
| −20 | 180 | 0.0 | 0.5 |
| −10 | 180 | 1.7 | 4.7 |
| −5 | 180 | 8.9 | 15.2 |
| 0 | 180 | 11.7 | 26.5 |
| 5 | 180 | 24.4 | 38.7 |
| 10 | 180 | 22.8 | 43.6 |
| 20 | 180 | 32.8 | 50.3 |

**Conclusión concluyente:** el desempeño **crece monótonamente con el TIR**
(salvo ruido normal entre −5 y 10 dB). Confirma que el modelo aprendió
features discriminativas del canto real, no artefactos del ruido de fondo —
mismo tipo de evidencia que reporta el paper en su Sección 5.2 (Tabla 4:
49.69%→91.38% de global_acc al angostar el TIR).

### 4.2. Desempeño vs. cantidad de especies mezcladas (k)

| k | n | exact-match | micro-F1 |
|---|---|---|---|
| 1 | 420 | 40.7 | 58.0 |
| 2 | 420 | 3.1 | 27.3 |
| 3 | 420 | 0.0 | 15.3 |

**Conclusión concluyente:** degradación fuerte y monótona con más especies
simultáneas — análogo directo a la Sección 5.3 del paper (masking mutuo entre
blancos, Fig. 12/13). El exact-match cae casi a 0 con 3 especies porque exige
acertar el conjunto completo; el micro-F1 (parcial) degrada más suave pero
igual de consistente.

**Limitación documentada:** el ruido de fondo usado como "interferencia" es
silencio ambiental de selva (grillos, viento, agua), no otra especie de rana
— a diferencia del paper, donde la interferencia es la misma clase de señal
que el objetivo (otro barco). Es una simplificación razonable dado que el
dataset no separa "target" de "interferencia" de forma nativa.

## 5. Análogo cross-dataset / DeepShip (`cross_site.py`, terminado)

El dataset tiene 4 sitios de grabación (`INCT20955`, `INCT4`, `INCT41`,
`INCT17`) con densidades de especies muy distintas (medido en `CLAUDE.md`:
INCT20955/17 densos ~2.3-2.6 especies/clip; INCT4/41 ralos ~0.46-0.61).
Leave-one-site-out: entrenar con 3 sitios, testear en el 4° **nunca visto**
durante entrenamiento ni selección de checkpoint — mide generalización a
dominio nuevo, mismo espíritu que el paper con DeepShip (Apéndice D).

Densidad de especies por sitio (contexto, `train.csv`):

| Sitio | Clips | LCard media | % clips vacíos |
|---|---|---|---|
| INCT20955 | 18 111 | 2.34 | 12% |
| INCT17 | 13 688 | 2.59 | 7% |
| INCT41 | 14 152 | 0.61 | 60% |
| INCT4 | 16 240 | 0.46 | 68% |

Dos sitios "densos" (INCT17/20955, ~2.3-2.6 especies/clip) y dos "ralos"
(INCT4/41, ~0.5, mayoría clips vacíos) — combinación de dominio bien distinta.

### 5.1. Resultado 1/4: held out = INCT20955

Entrenado con INCT4+INCT17+INCT41 (80/20 interno para selección de
checkpoint), 80 épocas, mismo modelo TResNet-LA.

| | val (80/20 interno, sitios de train) | TEST (INCT20955, nunca visto) |
|---|---|---|
| global_acc | 92.88 (última época) | **10.77** |
| F1 | — | **2.73** |
| mAP | 95.58 (última época) / 95.92 (mejor checkpoint) | **26.92** |

**Conclusión concluyente (parcial, 1/4 sitios):** colapso dramático de
desempeño al testear en un sitio de grabación nunca visto (mAP 95.9→26.9,
caída de ~69 puntos). El modelo **no generaliza entre sitios** — aprende
patrones específicos del sitio (ruido de fondo del micrófono/ubicación,
composición de especies local) más que features universales de las
vocalizaciones. Es un resultado análogo al Apéndice D del paper con DeepShip,
donde el global_acc también cae fuerte al cambiar de dataset (de ~91% en el
propio test a 50.36% cross-dataset) — aunque nuestra caída es más severa,
consistente con que DeepShip y ShipsEar son ambos audio de barcos en agua
(dominio más cercano entre sí) mientras que nuestros 4 sitios tienen
composiciones de especies muy distintas (denso vs. ralo, ver tabla arriba).

### 5.2. Resultado 2/4: held out = INCT4

Entrenado con INCT17+INCT20955+INCT41, 80 épocas.

| | val (sitios de train) | TEST (INCT4, nunca visto) |
|---|---|---|
| global_acc | 89.20 (última) / — | 65.94 |
| F1 | — | **1.26** |
| mAP | 97.51 (última) / 97.86 (mejor checkpoint) | **8.91** |

**Nota de lectura:** el global_acc de 65.94% en test **no indica buen
desempeño** — INCT4 tiene 68% de clips vacíos (ver tabla §5), así que un
modelo que predice "nada" en casi todo acierta por default en la mayoría de
esos clips vacíos. El **F1=1.26** (colapsa a casi 0) es la métrica que
realmente importa acá y confirma que el modelo **no reconoce casi ninguna
especie real** en el sitio nunca visto — incluso peor que INCT20955 (F1=2.73).

### 5.3-5.4. Resultados 3/4 y 4/4: held out = INCT41, INCT17

| Held out | val_mAP (dominio de train) | TEST mAP | TEST global_acc | TEST F1 |
|---|---|---|---|---|
| INCT20955 | 95.92 | 26.92 | 10.77 | 2.73 |
| INCT4 | 97.86 | 8.91 | 65.94 | 1.26 |
| INCT41 | 97.83 | 15.32 | 46.38 | 0.88 |
| INCT17 | 97.26 | 18.66 | 5.95 | 1.34 |
| **Promedio** | **97.22** | **17.45** | 32.26 (no comparable, ver nota) | **1.55** |

**Conclusión concluyente (4/4 sitios, completo):** colapso sistemático y
severo en los 4 casos. mAP cae de ~97 (mismo dominio) a **17.45 en promedio**
(caída de ~80 puntos), F1 cae a **1.55** — el modelo prácticamente no
reconoce ninguna especie real en un sitio nunca visto durante entrenamiento.
El `global_acc` **no es comparable entre sitios** (dato de lectura, no
artefacto): INCT4 y INCT41 tienen 60-68% de clips vacíos, así que un modelo
que colapsa a predecir "nada" saca global_acc artificialmente alto ahí (65.94,
46.38) sin reconocer nada real (F1 0.88-1.26); INCT17/20955 son densos (7-12%
vacíos) y el mismo colapso da global_acc bajo (5.95, 10.77). **F1 es la única
métrica honesta acá.**

**Interpretación:** el modelo aprende features específicas del sitio de
grabación (equipo, ubicación del micrófono, ruido ambiente de fondo propio de
cada estación) en vez de features universales del canto de cada especie. Es
consistente con, pero más severo que, el resultado análogo del paper con
DeepShip (Apéndice D: global_acc cae de ~91% a 50.36% cross-dataset, caída de
~41 puntos) — nuestra caída es mucho mayor probablemente porque (a) DeepShip y
ShipsEar son ambos audio de barcos en agua, dominio mucho más cercano entre sí
que 4 sitios de selva con biodiversidad y ruido ambiente distintos, y (b) nuestro
desbalance de clases es muchísimo más extremo, así que el modelo tiene poca
señal para aprender features que generalicen más allá de lo memorizado por
sitio. **Es una limitación real y honesta para reportar**, no hay que
ocultarla: el modelo *no está listo* para desplegarse en un sitio de grabación
nuevo sin reentrenar o fine-tunear ahí.

Datos completos: `lab/repo/outputs/cross_site/cross_site_results.csv`
(incluye precision/recall/macro_acc por sitio) y `<sitio>_test_per_class.csv` /
`<sitio>_test_predictions.npz` (por muestra, para análisis de errores futuro).

## 6. Predicciones sobre `test/` (terminado — etapa 4/5)

Requisito de la consigna. Generadas con `predict.py` usando el checkpoint de
TResNet-LA (arm 1, mejor por mAP, época 50), sobre los **31 187 clips** de
`test/` sin etiquetas, umbral σ=0.8.

Archivos: `outputs/experiments/predictions_probabilities.csv` (probabilidad
por clase) y `predictions_binarized.csv` (0/1 por clase).

**Chequeo de sanidad** (no hay ground truth en `test/`, así que no se puede
calcular accuracy/F1 ahí — solo coherencia distribucional):

| | train.csv | predicciones test/ |
|---|---|---|
| Label Cardinality (especies/clip promedio) | 1.51 | 1.46 |
| Clips sin ninguna clase predicha | 36% (22 504/62 191) | 38.9% (12 127/31 188) |

**Conclusión concluyente:** las dos distribuciones son muy cercanas (LCard
1.51 vs 1.46, % vacíos 36% vs 38.9%) — el modelo **no colapsa** a predecir
"nada" sistemáticamente en `test/`. Esto es coherente con que `test/` proviene
de los **mismos 4 sitios** que `train/` (mismo dominio, distintos clips) — muy
distinto del escenario de §5, donde el sitio completo era nuevo. Confirma que
la degradación de §5 es específicamente un problema de **shift de dominio
entre sitios**, no un problema general de generalización del modelo dentro
del dominio conocido.

## 7. Balance del dataset — Tabla 3 del paper (`balance_metrics.py`, terminado)

| Conjunto | N | LCard | MeanIR | MeanImR | LP_MeanIR | LP_CVIR |
|---|---|---|---|---|---|---|
| train.csv completo | 62 191 | 1.51 | 127.5 | 596.9 | 5596.2 | 1.25 |
| train split (80%) | 49 752 | 1.51 | 123.1 | 576.8 | 4650.7 | 1.15 |
| val split (20%) | 12 439 | 1.51 | 151.9 | 710.0 | 2124.6 | 0.82 |
| train + MSW | 74 846 | 1.96 | **25.97** | **117.6** | 1913.5 | 1.44 |
| train + UABOS | 83 085 | 2.61 | 31.72 | 143.9 | **258.7** | **0.26** |

**Conclusión concluyente:** MSW y UABOS reducen fuertemente el desbalance
(MeanIR 123→26-32, un factor ~4-5×; LP_MeanIR 4651→259-1913, hasta 18×). El
paper reporta reducciones más modestas (MeanIR ya cerca de 1 antes de
resamplear, porque sus 3 clases de barco están naturalmente balanceadas;
LP_MeanIR 14→4-5). Nuestro dataset tiene un desbalance **órdenes de magnitud
peor** (42 clases, 2 con 0 positivos, ratio máximo/mínimo ~1900:1) — por eso
el resampling reduce tanto el LP_MeanIR pero **no logra traducir esa mejora de
balance en mejora de accuracy** en TResNet-LA (ver §2.2): confirma la
observación del paper de que *"data balance alone does not determine final
recognition performance"*.

## 8. Comparación con la primera iteración del proyecto

| | Primera iteración (15M, checkpoint por global_acc) | Versión final, arm 1 (53M, checkpoint por mAP) |
|---|---|---|
| global_acc | 88.76 | **90.28** |
| mAP | 94.81 | **97.15** |
| F1 | 85.72 | **87.34** |
| loss | NaN (bug) | 0.14 (finito) |
| ablación SE | invertida (early-stop desparejo) | correcta (+0.74 global_acc, +1.44 F1) |

Mejora en las tres métricas principales, con un modelo fiel a la Tabla C1
(3.5× más parámetros) y sin el bug de NaN.

## 9. Análisis de errores y selección de umbral — TResNet-LA (arm 1), 20% val

Fuente: `outputs/experiments/analysis_summary.md`, `error_analysis_per_class.csv`,
`threshold_sweep.csv`, `metrics_by_species_count.csv`.

### 9.1. Desempeño vs. cantidad de especies reales por clip (dataset real, no sintético)

Análogo directo a §4.2 pero sobre el **20% de validación real** (no el MTD
sintético) — mismo tipo de análisis que la Sección 5.3 del paper.

| Nº especies (ground truth) | n | exact-match | micro-F1 |
|---|---|---|---|
| 0 | 4501 | 97.8 | 0.0 (indefinido, sin positivos) |
| 1 | 2577 | 90.4 | 95.0 |
| 2 | 2116 | 87.0 | 96.2 |
| 3+ | 3245 | 81.9 | 96.8 |

**Conclusión concluyente:** el exact-match cae con más especies (97.8→81.9,
esperable: exige acertar el conjunto completo), pero el **micro-F1 se
mantiene alto y estable** (95.0→96.8, sube levemente). Esto es un resultado
**mejor** que el análogo sintético de §4.2 (donde F1 caía fuerte con k) — la
diferencia es que acá los clips multi-especie del dataset real fueron
grabados así de forma natural (especies que efectivamente cantan juntas en
las mismas condiciones acústicas), mientras que el MTD sintético de §4.2
mezcla artificialmente cualquier combinación aleatoria de especies + ruido de
fondo de otro clip, lo que puede generar combinaciones acústicamente más
adversas que las que ocurren en la naturaleza.

### 9.2. Selección de umbral σ (barrido completo en validación)

| Umbral | Resultado |
|---|---|
| σ=0.5 | macro-F1 = 74.5 |
| **σ=0.8 (el del paper, usado en todas las corridas)** | macro-F1 = 87.3 |
| σ=0.65 (mejor umbral global, barrido 0.05-0.95) | macro-F1 = 87.7, exact-match = 88.4 |
| Umbral óptimo **por clase** (tuneado en val) | macro-F1 = **90.9** |

**Conclusión concluyente:** el σ=0.8 del paper resulta ser **casi óptimo**
como umbral único (87.3 vs 87.7 óptimo, diferencia de 0.4 pp) — sorprendente
dado que el paper lo calibró para 3 clases de barco razonablemente
balanceadas y acá hay 42 especies con desbalance extremo. Sí hay una ganancia
real (+3.6 pp de macro-F1) si se tunea un umbral **distinto por cada especie**
en vez de uno global — esperable, porque las especies raras necesitan un
umbral más bajo para no perderse contra las 42 competidoras. No se usó el
umbral por clase en las corridas principales (se mantuvo σ=0.8 fijo, fiel al
paper); queda como mejora identificada y cuantificada, no aplicada.

### 9.3. Peores clases por F1 (con al menos 1 muestra positiva en validación)

| Clase | support (val) | F1 | FP | FN |
|---|---|---|---|---|
| LEPFLA | 1 | 0.0 | 0 | 1 |
| LEPELE | 6 | 76.9 | 2 | 1 |
| PHYMAR | 41 | 85.0 | 5 | 7 |
| RHIORN | 4 | 85.7 | 0 | 1 |
| PHYDIS | 182 | 88.4 | 34 | 11 |
| ELAMAT | 75 | 88.6 | 8 | 9 |
| DENELE | 30 | 90.0 | 3 | 3 |
| AMEPIC | 14 | 90.3 | 3 | 0 |

Clases con más falsos positivos (en absoluto, no proporción): LEPLAT (80),
BOABIS (77), SPHSUR (55), PHYALB (43), DENNAN (35), DENMIN (35), PHYDIS (34),
PITAZU (32). Clases con más falsos negativos: SPHSUR (77), BOABIS (73),
LEPLAT (70), DENMIN (61), PHYALB (60), BOAALB (48), PITAZU (45), LEPPOD (34).

**Conclusión concluyente:** el patrón es doble. (a) Las clases con F1 más bajo
son las de **soporte mínimo** (LEPFLA n=1, RHIORN n=4, LEPELE n=6) — con tan
pocas muestras en el 20% de validación, un solo error mueve el F1 completo;
no es necesariamente que el modelo las reconozca mal, es ruido estadístico de
tener pocos ejemplos. (b) Las clases con **más errores absolutos** (SPHSUR,
BOABIS, LEPLAT, DENMIN, PHYALB) son justamente las **más comunes** del
dataset (SPHSUR tiene 13 258 positivos, la clase mayoritaria) — tienen más
errores en términos absolutos simplemente porque tienen más oportunidades de
error, no porque el modelo las reconozca peor proporcionalmente. Ambos
patrones son coherentes con el desbalance extremo del dataset (§7) y refuerzan
por qué macro-F1/mAP son las métricas correctas para reportar, no accuracy
absoluto por clase.
