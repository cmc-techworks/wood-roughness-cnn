#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 18 (nueva): interpretabilidad, solo para los modelos publicados (B-Ra, C-Rz).

La pregunta es donde pone atencion la red. La via habitual seria Grad-CAM.
El Grad-CAM no es fiel tras un Flatten (ver interpretabilidad_gradcam.py y
interpretabilidad_oclusion.py: su correlacion con la oclusion cae a -0.04..+0.01
en B, C y D, contra 0.56-0.59 en A). Esta figura responde la misma pregunta con
oclusion, que es fiel por construccion:

    (a)/(b)  DONDE mira la red: mapa de oclusion (perturbacion 'media') sobre una
             tesela recortada de la MISMA panoramica que usa la Fig. 19/20 (mapa
             espacial), para B-Ra y C-Rz.
    (c)/(d)  QUE informacion usa: textura borrada contra tono borrado, para esos
             mismos dos modelos (subconjunto de la Fig. A5/A6, que trae las 4
             arquitecturas).

La tesela de (a)/(b) no sale del set de validacion: se recorta directamente de
`madera_FinalX.png` (la panoramica de P11/C1/G40), en la esquina de la zona 'A'
que la Fig. 19b ya marca con un recuadro tras el anclaje contra el SJ-310. Asi
la Fig. 18 y la Fig. 19/20 muestran, literalmente, la misma fotografia.

No vuelve a correr interpretabilidad_oclusion.py (con solo 2 combinaciones
sobrescribiria oclusion_vs_gradcam.xlsx, que trae las 8). Reutiliza su funcion
oclusion(), y lee las cajas de textura/tono de la hoja Por_imagen ya calculada
ahi (esas si vienen del set de validacion completo, no de esta sola tesela).

Uso, con el entorno tesis (el unico que tiene TensorFlow):
    # activar el entorno: conda activate roughness-vision
    $TESIS fig_interpretabilidad.py
"""
import io
import os
import sys
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import tensorflow as tf

import comun as C                      # fija matplotlib en Agg
import matplotlib.pyplot as plt

sys.path.insert(0, str(C.CODE / '4_validacion'))
sys.path.insert(0, str(C.CODE / '5_heatmap'))
from interpretabilidad_oclusion import oclusion, GRANOS
import nucleo_mapa as NM

SEED = 13
GRANO_EJEMPLO = 40
VENTANA, PASO = 26, 13
RESUMEN = C.RESULTADOS / 'interpretabilidad_oclusion' / 'oclusion_vs_gradcam.xlsx'

# La misma panoramica que fig20_21_mapa_espacial.py (Fig. 19/20): P11, cara C1,
# grano 40, la unica cara con anclaje contra el SJ-310.
PANORAMICA = C.DATOS / 'P11' / 'C1' / 'G40' / 'PREPROCESADAS' / 'madera_FinalX.png'
# Zonas ya localizadas dentro de esa panoramica por verificar_mapa_espacial.py y
# marcadas en la Fig. 19b. Se toma la esquina de la zona A (recorte A1).
ANCLAJE = C.SALIDAS / 'Fig20_21_verificacion.xlsx'
ZONA_EJEMPLO = 'A1'

# Los mismos dos modelos que titulan el articulo.
COMBOS = [('B', 'Ra'), ('C', 'Rz')]


def cargar_modelo(arch, param):
    ruta = (C.RESULTADOS / f'{arch}_oficial' / param / 'modelos' /
            f'modelo_{arch}_{param}_final_seed{SEED}.keras')
    if not ruta.is_file():
        raise SystemExit(f"Falta {ruta}")
    return tf.keras.models.load_model(str(ruta), compile=False)


def imagen_ejemplo():
    """Tesela de tamano de entrada (390x130) recortada de la misma panoramica que
    la Fig. 19/20, en la esquina de la zona A ya marcada ahi tras el anclaje.

    No usa una imagen suelta del set de validacion: usa la fotografia que el
    lector ya vio en el mapa espacial, para que las dos figuras sean
    consistentes entre si.
    """
    if not PANORAMICA.is_file():
        raise SystemExit(f"Falta la panoramica: {PANORAMICA}")
    if not ANCLAJE.is_file():
        raise SystemExit(
            f"Falta {ANCLAJE}. Correr primero, con el entorno tesis:\n"
            f"  verificar_mapa_espacial.py --anclaje --seed {SEED}")
    zonas = pd.read_excel(ANCLAJE, sheet_name='Anclaje_por_zona')
    fila = zonas[zonas.recorte == ZONA_EJEMPLO].iloc[0]
    x0, y0 = int(fila.x0), int(fila.y0)

    pan = NM.cargar_panoramica(PANORAMICA)
    recorte = pan[y0:y0 + NM.ALTO_MODELO, x0:x0 + NM.ANCHO_MODELO]
    img = (recorte.astype(np.float32) / 255.0)[..., None]
    oh = np.eye(len(GRANOS), dtype=np.float32)[GRANOS.index(GRANO_EJEMPLO)]
    etiqueta = f'panorama P11-C1-G40, zona {ZONA_EJEMPLO} (x0={x0}, y0={y0})'
    return img, oh, etiqueta


def panel_mapa(ax, img, occ_abs, arch, param, letra):
    ax.imshow(img[..., 0], cmap='gray', aspect='auto')
    ax.imshow(occ_abs, cmap='inferno', alpha=0.55, aspect='auto')
    ax.set_title(f'{arch}-{param}', fontsize=12)
    ax.axis('off')
    C.etiqueta_panel(ax, letra, dx=-0.04, dy=1.06)


def panel_cajas(ax, resumen, param, letra):
    """Textura contra tono, solo el modelo publicado de este parametro."""
    arch = C.ARQ_PUBLICADA[param]
    d = resumen[(resumen.arch == arch) & (resumen.param == param)]
    caja_t = ax.boxplot([d.ef_textura.values], positions=[0], widths=0.55,
                        patch_artist=True)
    caja_o = ax.boxplot([d.ef_tono.values], positions=[1], widths=0.55,
                        patch_artist=True)
    caja_t['boxes'][0].set_facecolor('#bcd4e6')
    caja_o['boxes'][0].set_facecolor('#f7d9b0')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['texture\nremoved', 'tone\nremoved'])
    ax.set_ylabel(f'|Δ {param}| per 1 mm window (µm)')
    ax.set_title(f'{arch}-{param}: texture vs. tone', fontsize=12)
    razon = d.ef_textura.mean() / d.ef_tono.mean()
    ax.text(0.97, 0.95, f'texture/tone = {razon:.1f}×', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#333333', lw=0.6, alpha=0.9))
    C.etiqueta_panel(ax, letra, dx=-0.18, dy=1.08)
    return razon, d.ef_textura.mean(), d.ef_tono.mean()


def figura_appendix_textura_tono(resumen_full, param, nombre):
    """FigA5/A6: las 4 arquitecturas, en Arial. Mismo contenido que
    interpretabilidad_oclusion.figura_textura_tono(), pero con C.estilo() en vez
    del estilo por defecto de matplotlib, para que el anexo tambien vaya en Arial.
    """
    d = resumen_full[resumen_full.param == param]
    archs = [a for a in 'ABCD' if a in set(d.arch)]
    fig, ax = plt.subplots(figsize=(1.9 * len(archs) + 1.5, 4.6))
    pos = np.arange(len(archs)) * 3.0
    datos_t = [d[d.arch == a].ef_textura.values for a in archs]
    datos_o = [d[d.arch == a].ef_tono.values for a in archs]
    b1 = ax.boxplot(datos_t, positions=pos - 0.55, widths=0.9, patch_artist=True)
    b2 = ax.boxplot(datos_o, positions=pos + 0.55, widths=0.9, patch_artist=True)
    for caja in b1['boxes']:
        caja.set_facecolor('#bcd4e6')
    for caja in b2['boxes']:
        caja.set_facecolor('#f7d9b0')
    ax.set_xticks(pos)
    ax.set_xticklabels(archs)
    ax.set_ylabel(f'|Δ {param}| per 1 mm window (µm)')
    ax.legend([b1['boxes'][0], b2['boxes'][0]],
              ['texture removed (tone kept)', 'tone removed (texture kept)'],
              loc='upper left', fontsize=9)
    ax.set_title(f'{param}: which information does the network use?', fontsize=12)
    plt.tight_layout()
    C.guardar(fig, nombre)


def main():
    C.estilo()
    if not RESUMEN.is_file():
        raise SystemExit(
            f"Falta {RESUMEN}. Correr primero, con el entorno tesis:\n"
            f"  interpretabilidad_oclusion.py")
    resumen_full = pd.read_excel(RESUMEN, sheet_name='Por_imagen')
    # Una sola tesela para las dos combinaciones: es la misma fotografia (la
    # panoramica de la Fig. 19/20), y el punto es comparar como cada modelo
    # publicado la lee, no comparar sobre imagenes distintas.
    img, oh, etiqueta = imagen_ejemplo()

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.6))
    filas_resumen = []
    for j, (arch, param) in enumerate(COMBOS):
        modelo = cargar_modelo(arch, param)
        occ, pred, ef_media = oclusion(modelo, img, oh, VENTANA, PASO, 'media')
        panel_mapa(axes[0, j], img, np.abs(occ), arch, param, f'({chr(97 + j)})')
        razon, ef_t, ef_o = panel_cajas(axes[1, j], resumen_full, param, f'({chr(99 + j)})')
        filas_resumen.append(dict(arch=arch, param=param, tesela=etiqueta,
                                  ef_media_ejemplo=ef_media,
                                  ef_textura_media=ef_t, ef_tono_media=ef_o,
                                  razon_textura_tono=razon))
    plt.tight_layout(h_pad=3.0, w_pad=2.5)
    C.guardar(fig, 'Fig18_interpretabilidad')

    # Appendix A5/A6: las 4 arquitecturas, mismo dato que oclusion_vs_gradcam.xlsx
    # pero redibujado en Arial (interpretabilidad_oclusion.py usa el estilo por
    # defecto de matplotlib porque no importa comun.py).
    figura_appendix_textura_tono(resumen_full, 'Ra', 'FigA5_textura_tono_Ra')
    figura_appendix_textura_tono(resumen_full, 'Rz', 'FigA6_textura_tono_Rz')

    tabla = pd.DataFrame(filas_resumen)
    tabla.to_excel(C.SALIDAS / 'Fig18_interpretabilidad_cifras.xlsx', index=False)
    print(tabla.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f"\n(a)/(b) mapa de oclusion (perturbacion 'media') sobre la tesela de {etiqueta},")
    print("        la misma panoramica y zona que marca la Fig. 19/20, semilla 13.")
    print("(c)/(d) razon texture/tone de la hoja Por_imagen de oclusion_vs_gradcam.xlsx "
          "(8 combinaciones ya calculadas; aqui solo se filtran las 2 publicadas).")


if __name__ == '__main__':
    main()
