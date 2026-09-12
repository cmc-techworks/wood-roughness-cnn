#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpretabilidad del modelo: donde mira la red al estimar la rugosidad.

Busca una interpretacion fisica de lo que aprende la CNN: comprobar, con tecnicas de
IA explicable, que el modelo se apoya en la topografia de la superficie y no en
artefactos irrelevantes.

METODO. Grad-CAM adaptado a regresion. Se escribio para la arquitectura A, que tras la ultima
convolucion tiene un GlobalAveragePooling2D y tres capas densas antes de la salida; por
esas densas no aplica el CAM clasico (que exige que el pooling alimente directamente la
salida) y se usa Grad-CAM. Se usa
el gradiente de la prediccion escalar respecto del ultimo mapa de caracteristicas
convolucional: el peso de cada canal es la media espacial de ese gradiente, y el mapa es la
suma ponderada de canales, rectificada.

ADVERTENCIA (2026-09-10). Desde esa fecha se publican B para Ra y C para Rz, que tras
la ultima convolucion tienen MaxPooling y Flatten, no GlobalAveragePooling. Con Flatten
cada posicion espacial tiene su propio peso en la capa densa, y el promedio espacial del
gradiente que usa Grad-CAM mezcla posiciones de signo opuesto: el mapa puede dejar de ser
fiel a lo que usa la red. Para B, C y D usar interpretabilidad_oclusion.py, que mide el
efecto real de perturbar cada zona y compara ambos metodos (columna fidelidad_gc_oc en
resultados/interpretabilidad_oclusion/oclusion_vs_gradcam.xlsx). Este script queda como
referencia para A y como insumo de esa comparacion.

Como un mapa de calor es facil de sobreinterpretar, el script ademas lo CONTRASTA contra
dos hipotesis alternativas que podrian explicarlo igual de bien:

  (a) que la red mire los bordes del recorte (artefacto de encuadre, no topografia);
  (b) que la red mire la veta -variacion de tono de baja frecuencia- en vez de las marcas
      de lijado, que son textura de alta frecuencia.

Para (a) se compara la activacion media del borde contra la del centro. Para (b) se mide la
correlacion entre el mapa de activacion y dos descriptores locales calculados sobre la
propia imagen: el contraste local de alta frecuencia (desviacion tipica tras filtro
paso-alto) y el tono local de baja frecuencia (media tras desenfoque). Si la red lee
topografia, la activacion debe correlacionar con el primero y no con el segundo.

Uso:
    python interpretabilidad_gradcam.py
    python interpretabilidad_gradcam.py --n-por-grano 12
"""
import argparse
import io
import os
import sys
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage, stats
from tensorflow.keras.utils import img_to_array, load_img

IMG_H, IMG_W = 390, 130
GRANOS = [40, 80, 120, 180]
_DATA = [os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6'), 'fotos_recortes', os.path.join('data', 'fotos_recortes')]


def encontrar_data_dir():
    inicio = Path(__file__).resolve().parent
    for c in [inicio, *inicio.parents]:
        for s in _DATA:
            if (c / s / 'DATASET.xlsx').is_file():
                return c / s
    raise SystemExit('ERROR: no se encontro DATASET.xlsx')


def cargar(ruta):
    return img_to_array(load_img(str(ruta), color_mode='grayscale',
                                 target_size=(IMG_H, IMG_W))).astype(np.float32) / 255.0


def gradcam(modelo, capa_conv, img, grano_oh):
    """Mapa de activacion por gradiente para una salida escalar de regresion."""
    sub = tf.keras.Model(modelo.inputs, [modelo.get_layer(capa_conv).output, modelo.output])
    x = {'image_input': img[None, ...], 'grain_input': grano_oh[None, ...]}
    with tf.GradientTape() as cinta:
        mapas, pred = sub(x)
        cinta.watch(mapas)
        objetivo = pred[:, 0]
    grad = cinta.gradient(objetivo, mapas)[0].numpy()
    mapas = mapas[0].numpy()
    pesos = grad.mean(axis=(0, 1))                    # peso por canal
    cam = np.maximum((mapas * pesos).sum(axis=-1), 0)  # suma ponderada, rectificada
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, float(pred[0, 0])


def descriptores(img):
    """Contraste local de alta frecuencia y tono local de baja frecuencia."""
    g = img[..., 0]
    bajo = ndimage.gaussian_filter(g, sigma=8)         # tono / veta
    alto = g - bajo                                     # marcas de lijado
    contraste = ndimage.uniform_filter(alto ** 2, size=17) ** 0.5
    return contraste, bajo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['A', 'B', 'C', 'D'], default='B',
                    help='arquitectura publicada del parametro analizado')
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    ap.add_argument('--seed', type=int, default=10,
                    help='semilla del modelo final; la de referencia del parametro')
    ap.add_argument('--modelo', default=None)
    ap.add_argument('--capa', default=None, help='nombre de la ultima capa convolucional')
    ap.add_argument('--n-por-grano', type=int, default=10)
    ap.add_argument('--salida', default=None)
    args = ap.parse_args()

    raiz = Path(__file__).resolve().parent.parent
    base = raiz / 'resultados' / f'{args.arch}_oficial' / args.param
    if args.modelo is None:
        args.modelo = str(base / 'modelos' /
                          f'modelo_{args.arch}_{args.param}_final_seed{args.seed}.keras')
    out = Path(args.salida) if args.salida else (base / 'interpretabilidad')
    out.mkdir(parents=True, exist_ok=True)

    SUB = 'R$_a$' if args.param == 'Ra' else 'R$_z$'
    data_dir = encontrar_data_dir()
    modelo = tf.keras.models.load_model(args.modelo, compile=False)
    if args.capa is None:
        convs = [l.name for l in modelo.layers if isinstance(l, tf.keras.layers.Conv2D)]
        args.capa = convs[-1]
    print(f'  modelo: {Path(args.modelo).name}')
    print(f'  capa convolucional analizada: {args.capa}')

    df = pd.read_excel(data_dir / 'DATASET.xlsx', sheet_name='Validacion')
    df = df[df['nombre_imagen'].apply(
        lambda n: (data_dir / 'Validacion' / str(n)).is_file())]

    filas, ejemplos = [], {}
    for g in GRANOS:
        sub = df[df['Grano'] == g].head(args.n_por_grano)
        oh = np.eye(len(GRANOS), dtype=np.float32)[GRANOS.index(g)]
        for _, r in sub.iterrows():
            img = cargar(data_dir / 'Validacion' / str(r['nombre_imagen']))
            cam, pred = gradcam(modelo, args.capa, img, oh)
            cam_g = np.array(tf.image.resize(cam[..., None], (IMG_H, IMG_W)))[..., 0]

            contraste, tono = descriptores(img)
            r_alta = stats.pearsonr(cam_g.ravel(), contraste.ravel())[0]
            r_baja = stats.pearsonr(cam_g.ravel(), tono.ravel())[0]

            m = 30  # margen considerado "borde" en pixeles
            mask = np.zeros_like(cam_g, dtype=bool)
            mask[:m, :] = mask[-m:, :] = True
            mask[:, :m] = mask[:, -m:] = True
            filas.append({'archivo': r['nombre_imagen'], 'grano': g,
                          f'{args.param}_real': r[args.param],
                          f'{args.param}_pred': pred,
                          'r_alta_frec': r_alta, 'r_baja_frec': r_baja,
                          'act_borde': cam_g[mask].mean(), 'act_centro': cam_g[~mask].mean()})
            if g not in ejemplos:
                ejemplos[g] = (img[..., 0], cam_g, r[args.param], pred)

    t = pd.DataFrame(filas)
    t['razon_borde_centro'] = t['act_borde'] / t['act_centro']

    print('\n' + '=' * 74)
    print('  DONDE MIRA LA RED  (Grad-CAM sobre el conjunto de prueba)')
    print('=' * 74)
    print(f'  {"grano":>6} {"n":>3} {"r con alta frec.":>18} {"r con baja frec.":>18}'
          f' {"borde/centro":>13}')
    for g in GRANOS:
        s = t[t.grano == g]
        print(f'  P{g:<5} {len(s):>3} {s.r_alta_frec.mean():>18.3f}'
              f' {s.r_baja_frec.mean():>18.3f} {s.razon_borde_centro.mean():>13.3f}')
    print(f'  {"TODOS":>6} {len(t):>3} {t.r_alta_frec.mean():>18.3f}'
          f' {t.r_baja_frec.mean():>18.3f} {t.razon_borde_centro.mean():>13.3f}')

    w = stats.wilcoxon(t.r_alta_frec, t.r_baja_frec)
    print(f'\n  alta frecuencia vs baja frecuencia: Wilcoxon p = {w.pvalue:.2e}')
    print(f'  activacion en el borde vs el centro: razon media '
          f'{t.razon_borde_centro.mean():.3f} '
          f'({"sin sesgo al borde" if t.razon_borde_centro.mean() < 1.1 else "ATENCION: sesgo al borde"})')

    t.to_excel(out / 'R114_gradcam_metricas.xlsx', index=False)

    # --- figura: un ejemplo por grano ---
    fig, axes = plt.subplots(2, len(GRANOS), figsize=(11, 7.5))
    for j, g in enumerate(GRANOS):
        img, cam_g, real, pred = ejemplos[g]
        axes[0, j].imshow(img, cmap='gray', aspect='auto')
        axes[0, j].set_title(f'P{g}\n{SUB} medida {real:.2f} $\\mu$m', fontsize=9)
        axes[0, j].axis('off')
        axes[1, j].imshow(img, cmap='gray', aspect='auto')
        axes[1, j].imshow(cam_g, cmap='inferno', alpha=0.5, aspect='auto')
        axes[1, j].set_title(f'estimada {pred:.2f} $\\mu$m', fontsize=9)
        axes[1, j].axis('off')
    axes[0, 0].set_ylabel('recorte')
    axes[1, 0].set_ylabel('Grad-CAM')
    fig.suptitle('Zonas del recorte que sostienen la estimacion de rugosidad', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'R114_gradcam_ejemplos.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / 'R114_gradcam_ejemplos.pdf', bbox_inches='tight')

    # --- figura: contraste de hipotesis ---
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.boxplot([t.r_alta_frec, t.r_baja_frec], tick_labels=['Textura de\nalta frecuencia\n(marcas de lijado)',
                                                            'Tono de\nbaja frecuencia\n(veta)'])
    a1.axhline(0, color='gray', ls='--', lw=1)
    a1.set_ylabel('Correlacion con el mapa de activacion')
    a1.set_title('Con que correlaciona la atencion de la red')
    a1.grid(axis='y', alpha=0.3)
    a2.boxplot([t[t.grano == g].razon_borde_centro for g in GRANOS],
               tick_labels=[f'P{g}' for g in GRANOS])
    a2.axhline(1, color='crimson', ls='--', lw=1.2, label='sin sesgo al borde')
    a2.set_ylabel('Activacion en el borde / en el centro')
    a2.set_title('Control de artefacto de encuadre')
    a2.legend(fontsize=8)
    a2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out / 'R114_gradcam_hipotesis.png', dpi=300, bbox_inches='tight')
    fig2.savefig(out / 'R114_gradcam_hipotesis.pdf', bbox_inches='tight')

    (out / 'README.md').write_text(
        '# 1.14 — interpretabilidad por Grad-CAM\n\n'
        'Grad-CAM adaptado a regresion sobre la arquitectura publicada. NO se usa CAM clasico\n'
        'porque hay tres capas densas entre el GlobalAveragePooling2D y la salida.\n\n'
        'Contrasta la atencion de la red contra dos hipotesis alternativas: que mire los bordes\n'
        'del recorte (artefacto) o el tono de la veta (baja frecuencia) en vez de las marcas de\n'
        'lijado (alta frecuencia).\n\n'
        'Genera: `python CODE/4_validacion/interpretabilidad_gradcam.py`\n', encoding='utf8')
    print(f'\n  Resultados en: {out}')


if __name__ == '__main__':
    main()
