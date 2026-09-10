# Referencias

## 1. Paper base

**A robust deep learning model for underwater acoustic multi-target recognition
under non-target interference conditions**

- Autores: Lu Chen, Xinwei Luo, Hanlu Zhou, Qifan Shen, Long Chen, Chuanming Huan
- Afiliación: Key Laboratory of Underwater Acoustic Signal Processing (Southeast
  University), Ministry of Education, Nanjing, China
- Revista: *Expert Systems with Applications* 296 (2026) 129122
- DOI: `10.1016/j.eswa.2025.129122` — https://doi.org/10.1016/j.eswa.2025.129122
- ScienceDirect (PII): `S0957417425027393`
  https://www.sciencedirect.com/science/article/abs/pii/S0957417425027393
- PDF completo: `docs/Chen2026_TResNet-LA_UAMTR.pdf`
- Texto extraído (para búsqueda): `docs/Chen2026_TResNet-LA_UAMTR.txt`
- Notas y datos clave: `docs/NOTAS_PAPER.md`

## 2. Dataset

Grabaciones amazónicas de anuros. WAV 22.05 kHz, mono, 3.0 s, 42 especies,
multi-etiqueta a nivel de clip. Proyecto INCT (Instituto Nacional de Ciência
e Tecnologia). No incluido en este repositorio por tamaño.

## 3. Referencias internas del paper que importan para la implementación

- Ridnik et al. (2021) — **TResNet** (arquitectura base). WACV 2021.
- Ridnik, Ben-Baruch et al. (2021) — **Asymmetric Loss** (ASL).
- Hu et al. (2018) — **Squeeze-and-Excitation** (bloque de atención de canal).
- X.-Y. Zhang et al. (2023) — **MLBOS** (multi-label borderline oversampling),
  base del método UABOS del paper.
- Sandler et al. (2019) — SpaceToDepth.
