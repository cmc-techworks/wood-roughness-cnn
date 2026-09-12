# 1.14 — interpretabilidad por Grad-CAM

Grad-CAM adaptado a regresion sobre la arquitectura publicada. NO se usa CAM clasico
porque hay tres capas densas entre el GlobalAveragePooling2D y la salida.

Contrasta la atencion de la red contra dos hipotesis alternativas: que mire los bordes
del recorte (artefacto) o el tono de la veta (baja frecuencia) en vez de las marcas de
lijado (alta frecuencia).

Genera: `python CODE/4_validacion/interpretabilidad_gradcam.py`
