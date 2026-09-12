#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Oclusion contra Grad-CAM, en las cuatro arquitecturas y los dos parametros.

Por que existe (2026-09-10). El Grad-CAM de los modelos publicados (B-Ra y C-Rz)
invirtio la conclusion que se habia obtenido con la arquitectura A en agosto: con A
la atencion correlacionaba con la textura fina (marcas de lijado); con B y C
correlaciona con el tono (veta). Grad-CAM reduce el gradiente de cada canal a su
media espacial. Eso es exacto cuando tras la capa analizada hay un
GlobalAveragePooling (A), pero puede no ser fiel cuando hay Flatten (B, C, D): ahi
cada posicion espacial tiene su propio peso en la capa densa, y promediar sus
gradientes mezcla posiciones de signo opuesto.

La oclusion no depende de la arquitectura: modifica una ventana de la tesela, mide
cuanto cambia la estimacion, y desliza la ventana por toda la tesela. Es fiel por
construccion. Se aplican TRES perturbaciones sobre la misma ventana:

  media    la ventana se reemplaza por el gris medio de la imagen. Es la oclusion
           clasica. Da el mapa que se compara con el Grad-CAM. Advertencia: cuando
           el efecto es chico, la forma del mapa la domina el propio relleno, porque
           el cambio depende de cuanto difiere el tono de la ventana del tono medio;
           eso correlaciona con el tono por construccion.
  textura  la ventana se reemplaza por su version desenfocada (sigma 8 px, el mismo
           filtro que define el tono en los descriptores). Borra las marcas de
           lijado y CONSERVA el tono.
  tono     a la ventana se le quita su tono local y se le pone el tono medio de la
           imagen, conservando el residuo de alta frecuencia. Borra la veta y
           CONSERVA la textura.

La comparacion textura contra tono responde la pregunta de fondo sin pasar por
correlaciones con descriptores: si borrar la textura mueve la estimacion mas que
borrar el tono, la red usa la textura.

Sobre las MISMAS imagenes del test (probeta 11) y para las ocho combinaciones
arquitectura x parametro, con la semilla 13 en todas, se reportan:

  1. Magnitud del efecto de cada perturbacion, en um, y la razon textura/tono.
  2. Correlacion del mapa de oclusion (media) con textura y con tono, como se hizo
     con el Grad-CAM.
  3. Fidelidad del Grad-CAM: correlacion entre su mapa y el de oclusion.
  4. Sesgo de borde.

La muestra, el one-hot, los descriptores y el Grad-CAM se importan de
interpretabilidad_gradcam.py para que la comparacion sea exacta.

Uso (entorno tesis):
    python interpretabilidad_oclusion.py
    python interpretabilidad_oclusion.py --combos B:Ra C:Rz --n-por-grano 2
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage, stats

# Importar este modulo ya envuelve stdout en UTF-8; no se vuelve a envolver aqui.
from interpretabilidad_gradcam import (GRANOS, IMG_H, IMG_W, cargar, descriptores,
                                       encontrar_data_dir, gradcam)

RAIZ = Path(__file__).resolve().parent.parent
# results/ en el repositorio publicado, resultados/ en el arbol de trabajo original
_RES = RAIZ.parent / 'results'
if not _RES.is_dir():
    _RES = _RES
COMBOS = ['A:Ra', 'B:Ra', 'C:Ra', 'D:Ra', 'A:Rz', 'B:Rz', 'C:Rz', 'D:Rz']
MODOS = ('media', 'textura', 'tono')
MARGEN_BORDE = 30   # px, el mismo del script de Grad-CAM
SIGMA_TONO = 8      # px, el mismo filtro que define el tono en descriptores()


def r(a, b):
    """Pearson que devuelve NaN en vez de fallar si un mapa es constante."""
    a, b = np.ravel(a), np.ravel(b)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def ventanas(H, W, ventana, paso):
    ys = list(range(0, H - ventana + 1, paso))
    xs = list(range(0, W - ventana + 1, paso))
    if ys[-1] != H - ventana:
        ys.append(H - ventana)
    if xs[-1] != W - ventana:
        xs.append(W - ventana)
    return [(y, x) for y in ys for x in xs]


def oclusion(modelo, img, oh, ventana, paso, modo):
    """Mapa de sensibilidad con signo (estimacion original menos perturbada,
    promediada sobre las ventanas que cubren cada pixel) y el efecto medio por
    ventana en valor absoluto, en unidades del parametro."""
    H, W = img.shape[:2]
    g2 = img[..., 0]
    bajo = ndimage.gaussian_filter(g2, sigma=SIGMA_TONO)
    alto = g2 - bajo
    media = float(g2.mean())

    x0 = {'image_input': img[None], 'grain_input': oh[None]}
    base = float(modelo(x0, training=False).numpy()[0, 0])

    pos = ventanas(H, W, ventana, paso)
    lote = np.repeat(img[None], len(pos), axis=0).astype(np.float32)
    for k, (y, x) in enumerate(pos):
        sy, sx = slice(y, y + ventana), slice(x, x + ventana)
        if modo == 'media':
            lote[k, sy, sx, 0] = media
        elif modo == 'textura':
            lote[k, sy, sx, 0] = bajo[sy, sx]
        elif modo == 'tono':
            lote[k, sy, sx, 0] = alto[sy, sx] + media
        else:
            raise ValueError(modo)
    ohs = np.repeat(oh[None], len(lote), axis=0)
    pred = modelo.predict({'image_input': lote, 'grain_input': ohs},
                          batch_size=128, verbose=0)[:, 0]
    delta = base - pred

    acum = np.zeros((H, W))
    cuenta = np.zeros((H, W))
    for d, (y, x) in zip(delta, pos):
        acum[y:y + ventana, x:x + ventana] += d
        cuenta[y:y + ventana, x:x + ventana] += 1
    return acum / np.maximum(cuenta, 1), base, float(np.abs(delta).mean())


def mascara_borde(forma):
    m = np.zeros(forma, dtype=bool)
    m[:MARGEN_BORDE, :] = m[-MARGEN_BORDE:, :] = True
    m[:, :MARGEN_BORDE] = m[:, -MARGEN_BORDE:] = True
    return m


def analizar(arch, param, seed, muestra, data_dir, ventana, paso):
    ruta = (_RES / f'{arch}_oficial' / param / 'modelos' /
            f'modelo_{arch}_{param}_final_seed{seed}.keras')
    if not ruta.is_file():
        print(f'  [{arch}-{param}] FALTA el modelo {ruta.name}; se omite')
        return None, None
    modelo = tf.keras.models.load_model(str(ruta), compile=False)
    capa = [l.name for l in modelo.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
    borde = mascara_borde((IMG_H, IMG_W))

    filas, ejemplos = [], {}
    for _, fila in muestra.iterrows():
        g = int(fila['Grano'])
        oh = np.eye(len(GRANOS), dtype=np.float32)[GRANOS.index(g)]
        img = cargar(data_dir / 'Validacion' / str(fila['nombre_imagen']))

        occ, pred, ef_media = oclusion(modelo, img, oh, ventana, paso, 'media')
        _, _, ef_textura = oclusion(modelo, img, oh, ventana, paso, 'textura')
        _, _, ef_tono = oclusion(modelo, img, oh, ventana, paso, 'tono')
        occ_abs, occ_pos = np.abs(occ), np.maximum(occ, 0)

        cam, _ = gradcam(modelo, capa, img, oh)
        cam = np.array(tf.image.resize(cam[..., None], (IMG_H, IMG_W)))[..., 0]

        contraste, tono = descriptores(img)
        filas.append({
            'arch': arch, 'param': param, 'seed': seed,
            'archivo': fila['nombre_imagen'], 'grano': g,
            'real': fila[param], 'pred': pred,
            'ef_media': ef_media, 'ef_textura': ef_textura, 'ef_tono': ef_tono,
            'oc_r_textura': r(occ_abs, contraste), 'oc_r_tono': r(occ_abs, tono),
            'gc_r_textura': r(cam, contraste), 'gc_r_tono': r(cam, tono),
            # like con like: Grad-CAM (con ReLU) contra la parte positiva de la oclusion
            'fidelidad_gc_oc': r(cam, occ_pos),
            'fidelidad_gc_oc_abs': r(cam, occ_abs),
            'oc_borde_centro': occ_abs[borde].mean() / max(occ_abs[~borde].mean(), 1e-12),
        })
        if g not in ejemplos:
            ejemplos[g] = (img[..., 0], occ_abs, cam)
    print(f'  [{arch}-{param}] listo, {len(filas)} imagenes, capa {capa}')
    return pd.DataFrame(filas), ejemplos


def wilcoxon(d, a, b):
    ok = d[[a, b]].dropna()
    if len(ok) < 6 or (ok[a] - ok[b]).abs().sum() == 0:
        return np.nan
    return float(stats.wilcoxon(ok[a], ok[b]).pvalue)


def resumir(t):
    out = []
    for (arch, param), d in t.groupby(['arch', 'param'], sort=False):
        out.append({
            'arch': arch, 'param': param, 'n': len(d),
            'mae': (d.real - d.pred).abs().mean(),
            # Magnitud: cuanto mueve la estimacion cada perturbacion de 1 mm
            'ef_media': d.ef_media.mean(),
            'ef_textura': d.ef_textura.mean(),
            'ef_tono': d.ef_tono.mean(),
            'razon_textura_tono': d.ef_textura.mean() / max(d.ef_tono.mean(), 1e-12),
            'textura_gana': int((d.ef_textura > d.ef_tono).sum()),
            'p_textura_vs_tono': wilcoxon(d, 'ef_textura', 'ef_tono'),
            # Correlaciones con descriptores, para comparar con el Grad-CAM
            'oc_r_textura': d.oc_r_textura.mean(), 'oc_r_tono': d.oc_r_tono.mean(),
            'gc_r_textura': d.gc_r_textura.mean(), 'gc_r_tono': d.gc_r_tono.mean(),
            'fidelidad_gc_oc': d.fidelidad_gc_oc.median(),
            'oc_borde_centro': d.oc_borde_centro.mean(),
        })
    return pd.DataFrame(out)


def figura_ejemplos(ejemplos_por_arch, param, destino):
    archs = [a for a in 'ABCD' if a in ejemplos_por_arch]
    if not archs:
        return
    fig, axes = plt.subplots(len(archs), len(GRANOS),
                             figsize=(2.3 * len(GRANOS), 4.6 * len(archs)))
    axes = np.atleast_2d(axes)
    for i, a in enumerate(archs):
        for j, g in enumerate(GRANOS):
            ax = axes[i, j]
            ax.axis('off')
            if g not in ejemplos_por_arch[a]:
                continue
            img, occ_abs, _ = ejemplos_por_arch[a][g]
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.imshow(occ_abs, cmap='inferno', alpha=0.5, aspect='auto')
            ax.set_title(f'{a} - P{g}', fontsize=10)
    fig.suptitle(f'Occlusion sensitivity, {param}', fontsize=12)
    plt.tight_layout()
    fig.savefig(destino / f'oclusion_ejemplos_{param}.png', dpi=200, bbox_inches='tight')
    fig.savefig(destino / f'oclusion_ejemplos_{param}.pdf', bbox_inches='tight')
    plt.close(fig)


def figura_textura_tono(t, param, destino):
    """Efecto de borrar la textura contra efecto de borrar el tono, por arquitectura."""
    d = t[t.param == param]
    archs = [a for a in 'ABCD' if a in set(d.arch)]
    if not archs:
        return
    fig, ax = plt.subplots(figsize=(1.9 * len(archs) + 1.5, 4.2))
    pos_t = np.arange(len(archs)) * 3.0
    datos_t = [d[d.arch == a].ef_textura.values for a in archs]
    datos_o = [d[d.arch == a].ef_tono.values for a in archs]
    b1 = ax.boxplot(datos_t, positions=pos_t - 0.55, widths=0.9, patch_artist=True)
    b2 = ax.boxplot(datos_o, positions=pos_t + 0.55, widths=0.9, patch_artist=True)
    for caja in b1['boxes']:
        caja.set_facecolor('#bcd4e6')
    for caja in b2['boxes']:
        caja.set_facecolor('#f7d9b0')
    ax.set_xticks(pos_t)
    ax.set_xticklabels(archs)
    ax.set_ylabel(f'Mean |change| in estimated {param} per 1 mm window')
    ax.legend([b1['boxes'][0], b2['boxes'][0]],
              ['texture removed (tone kept)', 'tone removed (texture kept)'],
              loc='upper left', fontsize=9)
    ax.set_title(f'{param}: which information does the network use?', fontsize=11)
    plt.tight_layout()
    fig.savefig(destino / f'perturbacion_textura_tono_{param}.png', dpi=200, bbox_inches='tight')
    fig.savefig(destino / f'perturbacion_textura_tono_{param}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--combos', nargs='+', default=COMBOS,
                    help='pares ARCH:PARAM, por defecto las ocho combinaciones')
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--n-por-grano', type=int, default=10)
    ap.add_argument('--ventana', type=int, default=26, help='lado en px (26 px = 1 mm)')
    ap.add_argument('--paso', type=int, default=13)
    ap.add_argument('--salida', default=None)
    args = ap.parse_args()

    destino = (Path(args.salida) if args.salida
               else _RES / 'interpretabilidad_oclusion')
    destino.mkdir(parents=True, exist_ok=True)

    data_dir = encontrar_data_dir()
    df = pd.read_excel(data_dir / 'DATASET.xlsx', sheet_name='Validacion')
    df = df[df['nombre_imagen'].apply(lambda n: (data_dir / 'Validacion' / str(n)).is_file())]
    muestra = pd.concat([df[df['Grano'] == g].head(args.n_por_grano) for g in GRANOS])
    print(f'  muestra: {len(muestra)} imagenes del test, {args.n_por_grano} por grano')
    print(f'  ventana {args.ventana} px, paso {args.paso} px, semilla {args.seed}')
    print(f'  perturbaciones: {", ".join(MODOS)}')

    tablas, ejemplos = [], {'Ra': {}, 'Rz': {}}
    for combo in args.combos:
        arch, param = combo.split(':')
        t, ej = analizar(arch, param, args.seed, muestra, data_dir, args.ventana, args.paso)
        if t is not None:
            tablas.append(t)
            ejemplos[param][arch] = ej

    if not tablas:
        raise SystemExit('ningun modelo disponible')
    t = pd.concat(tablas, ignore_index=True)
    res = resumir(t)

    with pd.ExcelWriter(destino / 'oclusion_vs_gradcam.xlsx') as w:
        res.to_excel(w, sheet_name='Resumen', index=False)
        t.to_excel(w, sheet_name='Por_imagen', index=False)
    for param in ('Ra', 'Rz'):
        figura_ejemplos(ejemplos[param], param, destino)
        figura_textura_tono(t, param, destino)

    pd.set_option('display.width', 220)
    print('\n' + '=' * 100)
    print(res.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\n-> {destino}')


if __name__ == '__main__':
    main()
