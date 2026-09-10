# TResNet-LA para anuros amazónicos

Adaptación e implementación de **TResNet-LA** (Chen, Luo, Zhou, Shen, Chen, Huan —
*Expert Systems with Applications* 296, 2026, DOI [10.1016/j.eswa.2025.129122](https://doi.org/10.1016/j.eswa.2025.129122))
al reconocimiento multi-especie de **42 anuros amazónicos** a partir de grabaciones
de campo (dataset INCT). El paper original resuelve **UAMTR** (Underwater Acoustic
Multi-Target Recognition): identificar tipo y cantidad de buques bajo interferencia
acústica submarina. El problema es estructuralmente idéntico —clasificación
multi-etiqueta de audio con interferencia— cambiando el dominio de 3 clases de
barco a 42 especies.

Trabajo para el Laboratorio 1 de la Maestría en Deep Learning.

## Estructura

```
├── codigo/          # Implementación (PyTorch): arquitectura, entrenamiento,
│                     ablaciones, baselines, resampling, cross-site
├── docs/            # Paper (PDF + notas), resultados, análisis de fidelidad
│                     al paper, informe Word
└── presentacion/     # Deck de 20 slides: HTML autocontenido (16:9) + versión
                       # PowerPoint 100% editable
```

- [`docs/Informe_Laboratorio1_TResNet-LA.docx`](docs/Informe_Laboratorio1_TResNet-LA.docx) — informe completo (paper, arquitectura, depuración, fidelidad arquitectónica, todos los resultados, conclusiones).
- [`presentacion/TResNet-LA_Presentacion.pptx`](presentacion/TResNet-LA_Presentacion.pptx) — la presentación en PowerPoint editable (tablas y gráficos nativos, no imágenes). Ver [`presentacion/README.md`](presentacion/README.md) para cuándo usar cada formato.

## Resultados principales

| | Primera iteración (15M) | Versión final (53.1M) |
|---|---|---|
| global_acc | 88.76 | **90.28** |
| mAP | 94.81 | **97.15** |
| F1 | 85.72 | **87.34** |
| loss | NaN (bug) | 0.14 (finito) |

10 arms (ablación SE + resampling UABOS/MSW + 4 backbones), leave-one-site-out
en 4 sitios de grabación, y un análogo sintético del análisis TIR del paper.
Detalle completo, con todos los números y las conclusiones honestas
(incluida la limitación real que encontramos: el modelo no generaliza a un
sitio de grabación nunca visto), en [`docs/RESULTADOS.md`](docs/RESULTADOS.md).

## Cómo se construyó

La implementación replica fielmente la arquitectura del paper aplicada a este
dataset y pasó por un proceso de depuración e iteración real antes de llegar
a la versión final:

- **4 bugs identificados y corregidos** durante el desarrollo: pérdida en
  NaN (precisión mixta + log de probabilidad ≈0), una ablación que comparaba
  cantidades de épocas distintas, SpecAugment duplicado entre workers del
  DataLoader, y selección de checkpoint sesgada por una métrica que se infla
  con el desbalance de clases.
- **1 desviación arquitectónica real, corregida**: la primera versión del
  modelo tenía ~15M parámetros en vez de los ~53M que especifica la Tabla C1
  del paper (bloques Bottleneck a media anchura). Se verificó releyendo el
  PDF **como imágenes** (no solo texto, que rompe las tablas con matrices)
  para confirmar la arquitectura exacta contra las Figuras 1 y 7.
- **Componentes del paper implementados desde cero**: los dos métodos de
  resampling (UABOS, MSW), la comparación con baselines (ResNet50,
  TResNet-M/MA), y los análisis de generalización (cross-site, análogo
  TIR/MTD).

El detalle completo de cada decisión de fidelidad al paper y su justificación
está en [`docs/GAP_ANALISIS.md`](docs/GAP_ANALISIS.md).

## Reproducir

```bash
cd codigo
pip install -r requirements.txt
# editar config.py: AUDIO_DIR -> carpeta con los .wav de train/
python train.py                 # entrenamiento único
python experiments.py           # los 10 arms (ablación + resampling + backbones)
python cross_site.py            # leave-one-site-out
python mtd_synth.py --checkpoint outputs/experiments/TResNet-LA_best.pt
python predict.py --test_dir <carpeta test/>
python analyze_results.py
```

Dataset: grabaciones INCT (Instituto Nacional de Ciência e Tecnologia),
22.05 kHz, 3 s por clip, 42 especies, multi-etiqueta. No incluido en este
repo por tamaño.

## Créditos

Paper: Chen et al. 2026 (ver cita arriba). Implementación, fidelidad
arquitectónica, módulos de resampling/baselines/cross-site/MTD, experimentos
y resultados: este repositorio.
