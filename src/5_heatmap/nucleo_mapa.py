# -*- coding: utf-8 -*-
"""Nucleo sin GUI del mapa espacial de rugosidad.

`interfaz_heatmap.py` hace este mismo trabajo, pero importa tkinter y fija el
backend de matplotlib en TkAgg en la linea 5, asi que no se puede importar desde
un script headless sin arrastrar esos dos efectos. Este modulo tiene la cadena
pura -- cargar, teselar, predecir, acumular -- sin importar matplotlib ni tkinter.

Diferencias de fondo con la GUI, no solo de forma:

1. **Rota, no redimensiona.** La GUI, cuando la tesela no coincide con la entrada
   del modelo, la pasa por `cv2.resize` (interfaz_heatmap.py:84-91). Si la tesela
   se corto como 130 alto x 390 ancho, eso estira la textura 3x en un eje y la
   aplasta 3x en el otro. Aqui la orientacion horizontal se resuelve con
   `np.rot90`, que es la operacion que corresponde: el recorte de entrenamiento
   de una traza horizontal es la misma region fisica girada, no deformada.

2. **Tesela con solape.** La GUI corta en rejilla disjunta y descarta el resto de
   la imagen (interfaz_heatmap.py:48-52). Aqui el paso es libre y las
   estimaciones solapadas se promedian.

3. **Dos orientaciones.** El protocolo de muestreo tiene 6 trazas por zona, las
   1-3 horizontales y las 4-6 verticales, todas almacenadas como recortes de
   390x130. El modelo vio las dos orientaciones fisicas. Recorrer la panoramica
   en las dos y promediar reproduce ese muestreo; recorrerla en una sola no.

El contrato de inferencia (normalizacion /255.0, one-hot de grano, mapeo de
entradas por rango) es identico al de `interfaz_heatmap.py:98-119` y al de
`4_validacion/validar_headless.py:144-154`. Si alguno cambia, cambian los tres.
"""
import numpy as np
import cv2

# Mismo orden que interfaz_heatmap.py:65 y validar_headless.py. El indice del
# one-hot sale de aqui, asi que reordenarlo invalida cualquier modelo entrenado.
GRAIN_CATEGORIES = [40, 80, 120, 180]

# Entrada del modelo: 390 alto x 130 ancho x 1 canal = 15 x 5 mm de superficie.
ALTO_MODELO = 390
ANCHO_MODELO = 130

# Escala de la panoramica cosida. La probeta mide 307 x 277 mm (manuscrito) y la
# panoramica de P11/C1/G40 sale de 7899 x 7070 px: 7899/307 = 25.7 px/mm y
# 7070/277 = 25.5 px/mm. Se toma el promedio. Verificado ademas por template
# matching: la panoramica esta a escala 1:1 con las capturas crudas, de las que
# se recortaron los parches de entrenamiento.
PX_POR_MM = 25.6


def cargar_panoramica(ruta):
    """Carga la panoramica en escala de grises, sin redimensionar."""
    img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    return img


def teselar(img, alto, ancho, paso_y, paso_x):
    """Corta la imagen en teselas de `alto` x `ancho` avanzando `paso_y`/`paso_x`.

    Devuelve (teselas, posiciones) donde posiciones[i] = (y0, x0) de la esquina
    superior izquierda de la tesela i.

    A diferencia de la rejilla disjunta de la GUI, el ultimo paso se ancla al
    borde en vez de descartarse, de modo que el mapa cubre la imagen completa.
    """
    h, w = img.shape[:2]
    if alto > h or ancho > w:
        raise ValueError(f"Tesela {alto}x{ancho} mayor que la imagen {h}x{w}")

    ys = list(range(0, h - alto + 1, paso_y))
    xs = list(range(0, w - ancho + 1, paso_x))
    if ys[-1] != h - alto:
        ys.append(h - alto)
    if xs[-1] != w - ancho:
        xs.append(w - ancho)

    teselas = np.empty((len(ys) * len(xs), alto, ancho), dtype=img.dtype)
    posiciones = np.empty((len(ys) * len(xs), 2), dtype=np.int32)
    k = 0
    for y0 in ys:
        for x0 in xs:
            teselas[k] = img[y0:y0 + alto, x0:x0 + ancho]
            posiciones[k] = (y0, x0)
            k += 1
    return teselas, posiciones


def predecir(modelo, teselas, grano, lote=256, rotar=False, verbose=0):
    """Estima el parametro de rugosidad de cada tesela.

    `rotar=True` gira cada tesela 90 grados antes de entrar al modelo. Es lo que
    corresponde a las teselas cortadas como 130 alto x 390 ancho: la region
    fisica es la misma que la de una traza horizontal del protocolo de medicion,
    y el recorte de entrenamiento equivalente esta guardado girado, no estirado.

    Normalizacion y one-hot identicos a validar_headless.py:144-154.
    """
    if rotar:
        # k=1 gira antihorario: (n, 130, 390) -> (n, 390, 130).
        teselas = np.rot90(teselas, k=1, axes=(1, 2))

    if teselas.shape[1] != ALTO_MODELO or teselas.shape[2] != ANCHO_MODELO:
        raise ValueError(
            f"Tesela {teselas.shape[1]}x{teselas.shape[2]} no coincide con la entrada "
            f"del modelo {ALTO_MODELO}x{ANCHO_MODELO}. No se redimensiona a proposito: "
            f"deformaria la textura. Corregir el corte o usar rotar=True.")

    x = np.expand_dims(teselas, axis=-1).astype('float32') / 255.0

    if len(modelo.inputs) > 1:
        idx = GRAIN_CATEGORIES.index(int(grano))
        oh = np.zeros((x.shape[0], len(GRAIN_CATEGORIES)), dtype='float32')
        oh[:, idx] = 1.0
        # El modelo se construyo con entradas nombradas: hay que pasar un dict.
        # Se mapea por rango (4-D = imagen, resto = grano), como interfaz_heatmap.py:113-118.
        entradas = {}
        for inp in modelo.inputs:
            nombre = inp.name.split(':')[0]
            entradas[nombre] = x if len(inp.shape) == 4 else oh
        preds = modelo.predict(entradas, batch_size=lote, verbose=verbose)
    else:
        preds = modelo.predict(x, batch_size=lote, verbose=verbose)

    return preds.flatten().astype('float64')


def acumular(preds, posiciones, alto, ancho, forma_img, celda, suma=None, conteo=None):
    """Reparte cada prediccion sobre las celdas que cubre su tesela.

    El mapa vive en una reticula de `celda` px de lado, no en la rejilla de
    teselas: asi las teselas de las dos orientaciones y de todos los pasos caen
    en el mismo sistema de coordenadas y se pueden promediar entre si.

    `suma` y `conteo` permiten encadenar varias llamadas (una por orientacion)
    sobre los mismos acumuladores.
    """
    h, w = forma_img[:2]
    n_f = int(np.ceil(h / celda))
    n_c = int(np.ceil(w / celda))
    if suma is None:
        suma = np.zeros((n_f, n_c), dtype='float64')
        conteo = np.zeros((n_f, n_c), dtype='int32')

    for p, (y0, x0) in zip(preds, posiciones):
        f0, f1 = y0 // celda, int(np.ceil((y0 + alto) / celda))
        c0, c1 = x0 // celda, int(np.ceil((x0 + ancho) / celda))
        suma[f0:f1, c0:c1] += p
        conteo[f0:f1, c0:c1] += 1

    return suma, conteo


def mapa_desde_acumuladores(suma, conteo):
    """Promedio por celda. Las celdas sin cobertura quedan en NaN."""
    with np.errstate(invalid='ignore', divide='ignore'):
        mapa = np.where(conteo > 0, suma / np.maximum(conteo, 1), np.nan)
    return mapa


def mapa_dos_orientaciones(modelo, img, grano, fraccion_paso=0.5, celda=65,
                           lote=256, verbose=0):
    """Mapa completo: recorre la panoramica en las dos orientaciones y promedia.

    Devuelve (mapa, detalle, preds) con `detalle` describiendo cada pasada -- para
    poder declarar en el pie cuantas estimaciones respaldan el mapa -- y `preds`
    las estimaciones crudas de las dos pasadas concatenadas, que es sobre lo que
    corresponde correr los criterios de plausibilidad.
    """
    suma = conteo = None
    detalle = []
    todas = []

    pasadas = [
        # (nombre, alto, ancho, rotar)
        ('vertical',   ALTO_MODELO, ANCHO_MODELO, False),
        ('horizontal', ANCHO_MODELO, ALTO_MODELO, True),
    ]

    for nombre, alto, ancho, rotar in pasadas:
        paso_y = max(1, int(round(alto * fraccion_paso)))
        paso_x = max(1, int(round(ancho * fraccion_paso)))
        teselas, posiciones = teselar(img, alto, ancho, paso_y, paso_x)
        preds = predecir(modelo, teselas, grano, lote=lote, rotar=rotar, verbose=verbose)
        suma, conteo = acumular(preds, posiciones, alto, ancho, img.shape, celda,
                                suma, conteo)
        todas.append(preds)
        detalle.append({
            'orientacion': nombre,
            'tesela_alto': alto,
            'tesela_ancho': ancho,
            'paso_y': paso_y,
            'paso_x': paso_x,
            'n_teselas': int(len(preds)),
            'pred_min': float(preds.min()),
            'pred_max': float(preds.max()),
            'pred_media': float(preds.mean()),
            'pred_sd': float(preds.std(ddof=1)),
        })

    return mapa_desde_acumuladores(suma, conteo), detalle, np.concatenate(todas)


def mapa_rejilla_disjunta(modelo, img, grano, alto, ancho, rotar=False,
                          redimensionar=False, lote=256, verbose=0):
    """Rejilla disjunta, sin solape. Reproduce el comportamiento de la GUI.

    Solo se usa para el chequeo forense de que geometria produjo la figura
    publicada. `redimensionar=True` replica el `cv2.resize` de
    interfaz_heatmap.py:84-91, que es justamente lo que este modulo evita.
    """
    h, w = img.shape[:2]
    n_f, n_c = h // alto, w // ancho
    teselas = np.empty((n_f * n_c, alto, ancho), dtype=img.dtype)
    k = 0
    for i in range(n_f):
        for j in range(n_c):
            teselas[k] = img[i * alto:(i + 1) * alto, j * ancho:(j + 1) * ancho]
            k += 1

    if redimensionar:
        red = np.empty((teselas.shape[0], ALTO_MODELO, ANCHO_MODELO), dtype=teselas.dtype)
        for i in range(teselas.shape[0]):
            red[i] = cv2.resize(teselas[i], (ANCHO_MODELO, ALTO_MODELO),
                                interpolation=cv2.INTER_AREA)
        teselas, rotar = red, False

    preds = predecir(modelo, teselas, grano, lote=lote, rotar=rotar, verbose=verbose)
    return preds.reshape((n_f, n_c)), preds


def plausible(preds, rango=(1.750, 9.833)):
    """Criterios de plausibilidad de test_pipeline_heatmap.py:122-133.

    El rango por defecto es el del conjunto limpio de entrenamiento en Ra
    (Ra 1.750-9.833 um). Devuelve (ok, lista de mensajes).
    """
    msgs = []
    ok = True
    if not np.all(np.isfinite(preds)):
        msgs.append(f"FALLA: {int((~np.isfinite(preds)).sum())} predicciones no finitas")
        ok = False
    finitas = preds[np.isfinite(preds)]
    fuera = int(((finitas < rango[0]) | (finitas > rango[1])).sum())
    if fuera:
        msgs.append(f"AVISO: {fuera} de {finitas.size} predicciones fuera del rango de "
                    f"entrenamiento {rango[0]}-{rango[1]} um "
                    f"(min {finitas.min():.3f}, max {finitas.max():.3f})")
    if finitas.std() <= 1e-3:
        msgs.append(f"FALLA: desviacion {finitas.std():.2e} — el modelo no usa la imagen")
        ok = False
    if ok and not msgs:
        msgs.append("OK: finitas, dentro de rango y con variacion real")
    return ok, msgs
