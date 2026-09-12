# -*- coding: utf-8 -*-
"""Verificaciones del mapa espacial de la Fig. 20/21. No dibuja nada.

Dos comprobaciones independientes:

**--anclaje** — contrasta el mapa contra el perfilometro. La panoramica esta a
escala 1:1 con las capturas crudas, asi que las zonas efectivamente medidas con
el SJ-310 se pueden localizar dentro de ella y comparar celda a celda contra su
Ra de referencia. Es la unica forma de saber si el mapa es correcto en valor
absoluto y no solo en forma.

Ojo con dos cosas al leer el resultado:

* La panoramica es de la cara **C1**. Las 48 mediciones de P11 en P40 se reparten
  24 en C1 y 24 en C2, y las dos caras difieren (6.483 vs 7.197 de media). La
  comparacion valida es contra las 24 de C1, no contra las 48.
* `Panorama_A1_Recorte` y `Panorama_A2_Recorte` son dos recortes de la **misma
  zona**: comparten las 6 trazas del SJ-310 y tienen identico Ra de referencia.
  Son 3 zonas de medicion por cara, no 4, que es lo que declara el protocolo.

**--forense** — que geometria de teselado produjo la figura publicada. Su
recuadro dice *Mean value: 5.23 um*. Se corre el modelo con el que se genero
sobre las dos geometrias posibles, en rejilla disjunta como hacia la GUI, y se
ve cual reproduce esa cifra. Contesta con un numero lo que de otro modo es una
conjetura sobre una figura ya enviada.

    # activar el entorno: conda activate roughness-vision
    $TESIS verificar_mapa_espacial.py
    $TESIS verificar_mapa_espacial.py --anclaje
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import comun as C

sys.path.insert(0, str(C.CODE / '5_heatmap'))
import nucleo_mapa as NM

PANORAMICA = C.DATOS / 'P11' / 'C1' / 'G40' / 'PREPROCESADAS' / 'madera_FinalX.png'
ZONAS_DIR = C.DATOS / 'P11' / 'C1' / 'G40'
DATASET = C.DATOS / 'DATASET.xlsx'
MODELO_D = C.RAIZ / 'Entrenar modelo' / 'MULTI_VC_RA' / 'modelo_final_multimodal_VC.keras'

GRANO = 40
CARA = 'C1'
CELDA = 65
MEDIA_PUBLICADA = 5.23          # el recuadro de la Fig. 20b actual

# A1 y A2 son dos recortes de la misma zona de medicion: comparten trazas.
ZONA_DE_RECORTE = {'A1': 'A', 'A2': 'A', 'B': 'B', 'C': 'C'}


def parsear():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--anclaje', action='store_true')
    p.add_argument('--forense', action='store_true')
    p.add_argument('--panoramica', default=str(PANORAMICA))
    p.add_argument('--celda', type=int, default=CELDA)
    p.add_argument('--seed', type=int, default=None,
                   help='semilla del modelo B-Ra; por defecto la de referencia')
    a = p.parse_args()
    if a.seed is None:
        a.seed = C.SEED_REF['Ra']
    return a


def cargar(ruta):
    from tensorflow.keras.models import load_model
    return load_model(str(ruta), compile=False)


# --------------------------------------------------------------- anclaje ---

def localizar_zona(pan, zona, margen=250):
    """Ubica un recorte de zona dentro de la panoramica. Devuelve (x0, y0, ok).

    Dos etapas. La gruesa es correlacion sobre la imagen reducida 8x, que da la
    vecindad. La fina es SIFT + RANSAC dentro de esa vecindad, que es lo que
    resuelve de verdad: el cosido no hace mezcla fotometrica, asi que la misma
    region puede venir de una captura vecina con otra exposicion y la
    correlacion cruda se degrada aunque la posicion sea exacta.
    """
    ds = 8
    ps = cv2.resize(pan, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    ts = cv2.resize(zona, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    _, _, _, loc = cv2.minMaxLoc(cv2.matchTemplate(ps, ts, cv2.TM_CCOEFF_NORMED))
    gx, gy = loc[0] * ds, loc[1] * ds

    xa, ya = max(0, gx - margen), max(0, gy - margen)
    win = pan[ya:gy + zona.shape[0] + margen, xa:gx + zona.shape[1] + margen]

    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(zona, None)
    k2, d2 = sift.detectAndCompute(win, None)
    if d1 is None or d2 is None:
        return gx, gy, {'metodo': 'correlacion', 'inliers': 0}

    bf = cv2.BFMatcher()
    good = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return gx, gy, {'metodo': 'correlacion', 'inliers': 0}

    src = np.float32([k1[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=3)
    if M is None:
        return gx, gy, {'metodo': 'correlacion', 'inliers': 0}

    x0, y0 = M @ np.array([0.0, 0.0, 1.0])
    return (int(round(xa + x0)), int(round(ya + y0)),
            {'metodo': 'SIFT', 'n_matches': len(good), 'inliers': int(mask.sum()),
             'escala': round(float(np.hypot(M[0, 0], M[0, 1])), 4),
             'rotacion_deg': round(float(np.degrees(np.arctan2(M[0, 1], M[0, 0]))), 3)})


def media_mapa_en(mapa, celda, x0, y0, ancho, alto):
    """Media de las celdas del mapa que caen dentro de la ventana en px."""
    f0, f1 = y0 // celda, int(np.ceil((y0 + alto) / celda))
    c0, c1 = x0 // celda, int(np.ceil((x0 + ancho) / celda))
    sub = mapa[f0:f1, c0:c1]
    return float(np.nanmean(sub)), int(np.isfinite(sub).sum())


def anclaje(args):
    print("=" * 74)
    print("ANCLAJE METROLOGICO — mapa contra SJ-310")
    print("=" * 74)

    pan = NM.cargar_panoramica(args.panoramica)
    # Verificacion anclada a Ra: contrasta el mapa contra las mediciones del
    # SJ-310, que son de Ra. Si algun dia se verifica el mapa de Rz, hay que
    # parametrizarlo aqui y en las referencias del dataset.
    modelo = cargar(C.oficial('Ra') / 'Ra' / 'modelos' /
                    f"modelo_{C.ARQ_PUBLICADA['Ra']}_Ra_final_seed{args.seed}.keras")
    mapa, _, _ = NM.mapa_dos_orientaciones(modelo, pan, GRANO, celda=args.celda)

    ref = pd.read_excel(DATASET, sheet_name='Validacion')
    ref = ref[ref.Grano == GRANO].copy()
    ref['cara'] = ref.nombre_imagen.str.extract(r'_(C\d)_')
    ref['recorte'] = ref.nombre_imagen.str.extract(r'Recorte_Recorte_([A-C]\d?)_')
    ref_c1 = ref[ref.cara == CARA]

    filas = []
    for rec in ['A1', 'A2', 'B', 'C']:
        f = ZONAS_DIR / f'Panorama_{rec}_Recorte.tiff'
        if not f.is_file():
            print(f"  falta {f.name}, se omite")
            continue
        zona = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        x0, y0, info = localizar_zona(pan, zona)
        est, n_celdas = media_mapa_en(mapa, args.celda, x0, y0,
                                      zona.shape[1], zona.shape[0])
        med = ref_c1[ref_c1.recorte == rec]
        filas.append({
            'recorte': rec, 'zona_medicion': ZONA_DE_RECORTE[rec],
            # ancho/alto: fig20_21_mapa_espacial.py dibuja con esto los recuadros
            # de las zonas medidas sobre el mapa
            'x0': x0, 'y0': y0, 'ancho_px': int(zona.shape[1]), 'alto_px': int(zona.shape[0]),
            'localizacion': info.get('metodo'),
            'inliers': info.get('inliers'), 'escala': info.get('escala'),
            'rotacion_deg': info.get('rotacion_deg'),
            'n_trazas': int(len(med)),
            'Ra_medido': round(float(med.Ra.mean()), 3),
            'Ra_medido_sd': round(float(med.Ra.std(ddof=1)), 3),
            'Ra_mapa': round(est, 3), 'n_celdas': n_celdas,
            'error': round(est - float(med.Ra.mean()), 3),
        })

    df = pd.DataFrame(filas)
    print()
    print(df[['recorte', 'zona_medicion', 'localizacion', 'inliers', 'escala',
              'rotacion_deg', 'n_trazas', 'Ra_medido', 'Ra_medido_sd',
              'Ra_mapa', 'error']].to_string(index=False))

    # Zonas distintas: A cuenta una sola vez, aunque tenga dos recortes.
    dist = df.groupby('zona_medicion').agg(Ra_medido=('Ra_medido', 'mean'),
                                           Ra_mapa=('Ra_mapa', 'mean')).reset_index()
    mae = float((dist.Ra_mapa - dist.Ra_medido).abs().mean())

    cara = float(np.nanmean(mapa))
    med_c1 = float(ref_c1.Ra.mean())
    sd_c1 = float(ref_c1.Ra.std(ddof=1))
    med_todas = float(ref.Ra.mean())

    print()
    print(f"  Zonas de medicion distintas : {len(dist)} (A1 y A2 comparten trazas)")
    print(f"  MAE zona a zona             : {mae:.3f} µm")
    print()
    print(f"  Media del mapa, cara {CARA} entera : {cara:.3f} µm")
    print(f"  Media SJ-310, cara {CARA} (n={len(ref_c1)})   : {med_c1:.3f} ± {sd_c1:.3f} µm")
    print(f"  Diferencia                        : {cara - med_c1:+.3f} µm "
          f"({abs(cara - med_c1) / sd_c1:.2f} desviaciones)")
    print()
    print(f"  Referencia: la figura publicada declaraba {MEDIA_PUBLICADA:.2f} µm, "
          f"a {MEDIA_PUBLICADA - med_c1:+.3f} µm de la medicion.")
    print(f"  Nota: las 48 mediciones de P40 en P11 promedian {med_todas:.3f} µm, pero "
          f"incluyen la cara C2, que no es la de esta panoramica.")

    return df, {'semilla': args.seed,
                'media_mapa_cara': round(cara, 4),
                'media_sj310_C1': round(med_c1, 4), 'sd_sj310_C1': round(sd_c1, 4),
                'diferencia': round(cara - med_c1, 4),
                'mae_zona_a_zona': round(mae, 4),
                'media_publicada': MEDIA_PUBLICADA,
                'n_zonas_distintas': int(len(dist))}


# --------------------------------------------------------------- forense ---

def forense(args):
    print()
    print("=" * 74)
    print("FORENSE — que geometria produjo la figura publicada")
    print("=" * 74)

    if not MODELO_D.is_file():
        print(f"  Falta {MODELO_D}; se omite.")
        return None

    pan = NM.cargar_panoramica(args.panoramica)
    modelo = cargar(MODELO_D)
    print(f"  Modelo D: {modelo.count_params():,} parametros")
    print(f"  Objetivo: reproducir el 'Mean value: {MEDIA_PUBLICADA:.2f} µm' de la Fig. 20b actual")
    print()

    # Las cuatro primeras son las opciones del combobox de la GUI
    # (interfaz_heatmap.py:407-419): el alto lo elige el usuario y el ancho sale
    # de round(alto * 130/390), asi que solo la de 390 entra sin redimensionar.
    # Se agrega la transpuesta del predecesor en historico/, por descarte.
    casos = [
        ('130x43   GUI "Muy alta"',  130,  43, False, True),
        ('260x87   GUI "Alta"',      260,  87, False, True),
        ('390x130  GUI "Media"',     390, 130, False, False),
        ('780x260  GUI "Baja"',      780, 260, False, True),
        ('130x390  transpuesta',     130, 390, False, True),
        ('130x390  + rotacion',      130, 390, True,  False),
    ]
    filas = []
    for nombre, alto, ancho, rotar, redim in casos:
        rejilla, preds = NM.mapa_rejilla_disjunta(
            modelo, pan, GRANO, alto, ancho, rotar=rotar, redimensionar=redim)
        media = float(preds.mean())
        filas.append({'geometria': nombre,
                      'tesela_mm': f'{alto / NM.PX_POR_MM:.1f} x {ancho / NM.PX_POR_MM:.1f}',
                      'rejilla': f'{rejilla.shape[0]} x {rejilla.shape[1]}',
                      'n_teselas': int(preds.size),
                      'media': round(media, 3),
                      'min': round(float(preds.min()), 3),
                      'max': round(float(preds.max()), 3),
                      'dif_vs_publicada': round(media - MEDIA_PUBLICADA, 3)})
        print(f"  {nombre:26s} {rejilla.shape[0]:3d} x {rejilla.shape[1]:4d} "
              f"= {preds.size:5d} teselas de {alto / NM.PX_POR_MM:4.1f} x "
              f"{ancho / NM.PX_POR_MM:4.1f} mm   media {media:6.3f} µm "
              f"({media - MEDIA_PUBLICADA:+.3f})")

    df = pd.DataFrame(filas)
    gana = df.loc[df.dif_vs_publicada.abs().idxmin()]
    print()
    print(f"  Coincide: '{gana.geometria}' -> {gana.media:.3f} µm, a "
          f"{abs(gana.dif_vs_publicada):.3f} µm de la publicada.")
    print(f"  Su tesela mide {gana.tesela_mm} mm, contra los 15.2 x 5.1 mm del recorte")
    print("  de entrenamiento: el modelo recibio regiones mucho mas chicas que su")
    print("  campo de vision nominal, reescaladas para calzar en la entrada.")
    return df


def main():
    args = parsear()
    if not args.anclaje and not args.forense:
        args.anclaje = args.forense = True

    # La semilla de referencia conserva el nombre ya citado; las demas llevan sufijo,
    # para poder anclar las cuatro sin pisar la verificacion de la figura publicada.
    salida = C.SALIDAS / ('Fig20_21_verificacion.xlsx' if args.seed == C.SEED_REF['Ra']
                          else f'Fig20_21_verificacion_seed{args.seed}.xlsx')
    hojas = {}

    if args.anclaje:
        df_a, resumen = anclaje(args)
        hojas['Anclaje_por_zona'] = df_a
        hojas['Anclaje_resumen'] = pd.DataFrame(list(resumen.items()),
                                                columns=['clave', 'valor'])
    if args.forense:
        df_f = forense(args)
        if df_f is not None:
            hojas['Forense_geometria'] = df_f

    if hojas:
        with pd.ExcelWriter(salida) as w:
            for nombre, df in hojas.items():
                df.to_excel(w, sheet_name=nombre, index=False)
        print(f"\n  -> {salida}")


if __name__ == '__main__':
    main()
