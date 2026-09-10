# Presentación — 20 slides, 16:9

Dos formatos, mismo contenido (metodología / implementación / resultados),
generados independientemente el uno del otro (no son una conversión
automática exacta slide-a-slide en cada detalle visual, pero cubren la misma
información y el mismo orden).

## `presentacion.html`

Autocontenida (sin dependencias externas), con gráficos SVG a medida
(auto-medidos para no desbordar el slide) y navegación por teclado/botones.
Usarla para:

- **Verla en el navegador** y grabar el video directamente desde ahí (abrir el
  archivo, pantalla completa, navegar con flechas).
- Está también publicada como Artifact (ver el link en el historial de la
  conversación con Claude) para compartirla sin enviar el archivo.

## `TResNet-LA_Presentacion.pptx`

Generada con `python-pptx`: tablas y gráficos **nativos de Office** (no son
imágenes — clic derecho sobre un gráfico → "Editar datos" abre una hoja de
cálculo embebida), formas conectadas para los diagramas de flujo. Usarla para:

- **Editar el contenido** (texto, tablas, colores, agregar/quitar slides) en
  PowerPoint, LibreOffice Impress o Google Slides — la opción a usar si hay
  que ajustar algo antes de grabar.

## Si hay que resincronizar contenido

Ambos archivos se generaron a mano a partir de los mismos datos fuente
(`../docs/RESULTADOS.md`, `../docs/GAP_ANALISIS.md`, `../docs/NOTAS_PAPER.md`).
Si esos documentos cambian, hay que regenerar o editar los dos por separado —
no hay un pipeline que los mantenga sincronizados automáticamente.
