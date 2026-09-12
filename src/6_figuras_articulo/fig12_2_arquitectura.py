#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 12.2: variante del diagrama de arquitectura, con las variantes dentro del flujo.

Alternativa a `fig12_arquitectura.py`, que **no se toca**. La diferencia esta en
como se representa la etapa de reduccion del mapa de caracteristicas:

  Fig. 12    la ruta principal muestra la columna verde de GlobalAveragePooling2D,
             y las cuatro variantes cuelgan aparte, en una tabla punteada abajo.
  Fig. 12.2  esa columna verde desaparece y en su lugar, dentro del flujo, va un
             recuadro que ES la etapa: enumera las cuatro variantes evaluadas con
             su conteo de parametros, sin marcar ninguna.

Se lee como un conmutador dentro de la red: por ahi pasa el mapa de
caracteristicas y habia cuatro maneras de reducirlo. Ocupa el ancho que antes
gastaba la tabla, asi que la figura no crece de alto y el bloque queda a la altura
de la ruta convolucional.

Reutiliza los ayudantes de dibujo de `fig12_arquitectura.py` por importacion, sin
modificarlo. Los conteos de parametros salen de `comun.ARQUITECTURAS`, que a su
vez replica `3_entrenamiento/entrenar_final_limpio.py`.

Uso:  python fig12_2_arquitectura.py
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import comun as C
from fig12_arquitectura import (AZUL_CONV, NARANJA_POOL, VERDE_DENSA, LILA_CONCAT,
                                AMARILLO_SALIDA, ROJO_FLECHA, BORDE, RECORTE,
                                slabs, columna, circulos)

NOMBRE = 'Fig12.2_arquitectura'

GRIS_CAJA = '#f4f6f8'          # relleno del recuadro de variantes
GRIS_TEXTO = '#4a4a4a'         # variantes no seleccionadas
BANDA_ELEGIDA = '#dbe8f5'      # realce de la fila seleccionada
GRIS_FLECHA = '#6b7785'


def compacto(n):
    """Conteo abreviado, para que las cuatro filas quepan en una linea cada una."""
    return f'{n / 1e6:.2f} M' if n >= 1e6 else f'{n / 1e3:.1f} k'


def rotulo(ax, x, texto, y=99.5):
    ax.text(x, y, texto, ha='center', va='top', fontsize=10.5, linespacing=1.25)


def recuadro_variantes(ax, x0, y0, ancho, alto):
    """La etapa de reduccion, dibujada como un recuadro con las cuatro variantes."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), ancho, alto, boxstyle='round,pad=0.5,rounding_size=1.4',
        facecolor=GRIS_CAJA, edgecolor=BORDE, linewidth=1.4, zorder=4))

    ax.text(x0 + ancho / 2, y0 + alto - 4.5, 'Four variants evaluated',
            ha='center', va='center', fontsize=10, style='italic',
            color='#333333', zorder=6)
    ax.plot([x0 + 3, x0 + ancho - 3], [y0 + alto - 8.5] * 2,
            color='#b8c4d0', linewidth=0.9, zorder=5)

    alto_fila = (alto - 17.0) / len(C.ARQUITECTURAS)
    y = y0 + alto - 11.0 - alto_fila / 2
    for etiqueta, descripcion, params, elegida in C.ARQUITECTURAS:
        if elegida:
            ax.add_patch(mpatches.Rectangle(
                (x0 + 1.6, y - alto_fila / 2 + 0.4), ancho - 3.2, alto_fila - 0.8,
                facecolor=BANDA_ELEGIDA, edgecolor='none', zorder=5))
        color = '#000000' if elegida else GRIS_TEXTO
        peso = 'bold' if elegida else 'normal'

        ax.text(x0 + 3.4, y, etiqueta, ha='left', va='center', fontsize=10.5,
                fontweight='bold', color=color, zorder=6)
        ax.text(x0 + 7.8, y, descripcion, ha='left', va='center', fontsize=9.2,
                fontweight=peso, color=color, zorder=6)
        ax.text(x0 + ancho - 3.4, y, compacto(params), ha='right', va='center',
                fontsize=9.2, fontweight=peso, color=color, zorder=6)
        y -= alto_fila

    ax.text(x0 + ancho / 2, y0 + 3.6, 'one of four; the selection is argued in the text',
            ha='center', va='center', fontsize=9.2, color='#666666', zorder=6)


def flecha_gris(ax, x_desde, x_hasta, y):
    ax.annotate('', xy=(x_hasta, y), xytext=(x_desde, y),
                arrowprops=dict(arrowstyle='-|>', color=GRIS_FLECHA, linewidth=2.0,
                                shrinkA=0, shrinkB=0), zorder=3)


def main():
    C.estilo()
    fig, ax = plt.subplots(figsize=(18.0, 8.8))
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 100)
    ax.axis('off')

    Y = 70                       # eje de la ruta convolucional

    # ------------------------------------------------ entrada: recorte real --
    try:
        from PIL import Image
        img = np.asarray(Image.open(RECORTE).convert('L'))
        ax.imshow(img, cmap='gray', extent=(5, 15, Y - 17, Y + 17),
                  aspect='auto', zorder=5)
        ax.add_patch(mpatches.Rectangle((5, Y - 17), 10, 34, fill=False,
                                        edgecolor=BORDE, linewidth=1.0, zorder=6))
    except Exception as e:
        print(f"  aviso: no se pudo cargar la miniatura ({e})")
        ax.add_patch(mpatches.Rectangle((5, Y - 17), 10, 34, facecolor='#dddddd',
                                        edgecolor=BORDE, linewidth=1.0, zorder=5))
    rotulo(ax, 10, 'Input image\n390 × 130 × 1')

    # ---------------------------------------------------- base convolucional --
    slabs(ax, 21, Y, 32, 6, 3, AZUL_CONV, dy=0.9)
    rotulo(ax, 26, 'Conv2D\n(3×3, 32 filters)')

    slabs(ax, 39, Y, 21, 4.5, 3, NARANJA_POOL, dy=0.9)
    rotulo(ax, 43, 'MaxPooling\n(2×2)')

    slabs(ax, 53, Y, 23, 4.5, 6, AZUL_CONV, dy=0.9)
    rotulo(ax, 60, 'Conv2D\n(3×3, 64 filters)')

    slabs(ax, 70, Y, 15, 3.5, 6, NARANJA_POOL, dy=0.9)
    rotulo(ax, 76, 'MaxPooling\n(2×2)')

    # ------------- la etapa de reduccion, ahora es el recuadro de variantes --
    BX0, BW = 88, 40
    BY0, BH = Y - 28, 54
    flecha_gris(ax, 84.5, BX0 - 0.8, Y)
    recuadro_variantes(ax, BX0, BY0, BW, BH)
    rotulo(ax, BX0 + BW / 2, 'Feature-map reduction')

    # ------------------------------------------------------- cabeza densa ---
    flecha_gris(ax, BX0 + BW + 0.8, BX0 + BW + 6, Y)
    columna(ax, BX0 + BW + 8, Y, 7, VERDE_DENSA, elipsis=True)
    rotulo(ax, BX0 + BW + 9.4, 'Dense\n(64, ReLU)')

    CX0 = BX0 + BW + 18
    ax.add_patch(mpatches.Rectangle((CX0, 42), 15, 38, fill=False,
                                    edgecolor=BORDE, linewidth=1.2, zorder=4))
    columna(ax, CX0 + 3, 62, 6, LILA_CONCAT, lado=2.4, hueco=1.6, elipsis=True)
    circulos(ax, CX0 + 11, 62, 6, LILA_CONCAT, radio=1.3, hueco=1.9, elipsis=True)
    ax.text(CX0 + 7.5, 83, 'Concatenation\n(80)', ha='center', va='bottom',
            fontsize=10.5, linespacing=1.25)

    columna(ax, CX0 + 19, 62, 5, VERDE_DENSA, lado=3.4, hueco=1.6)
    ax.text(CX0 + 20.7, 78, 'Dense\n(32, ReLU)', ha='center', va='bottom',
            fontsize=10.5, linespacing=1.25)

    ax.add_patch(mpatches.FancyBboxPatch(
        (CX0 + 29, 58), 8, 8, boxstyle='round,pad=0.4,rounding_size=1.2',
        facecolor=AMARILLO_SALIDA, edgecolor=BORDE, linewidth=1.0, zorder=5))
    ax.text(CX0 + 33, 72, 'Output\n(Ra or Rz)', ha='center', va='bottom',
            fontsize=10.5, linespacing=1.25)

    # ------------------------------------------ segunda entrada: el grano ----
    circulos(ax, 8, 21, 4, AZUL_CONV, radio=1.7, hueco=1.6)
    ax.text(8, 33, 'Input grit\n(one-hot, 4)', ha='center', va='bottom',
            fontsize=10.5, linespacing=1.25)

    circulos(ax, 24, 21, 7, AZUL_CONV, radio=1.5, hueco=1.5, elipsis=True)
    ax.text(24, 37, 'Dense\n(16, ReLU)', ha='center', va='bottom',
            fontsize=10.5, linespacing=1.25)

    # ------------------------------------------------- flechas de fusion ----
    ax.annotate('', xy=(CX0 - 0.5, Y), xytext=(CX0 - 7.5, Y),
                arrowprops=dict(arrowstyle='-|>', color=ROJO_FLECHA, linewidth=3.2,
                                shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(CX0 + 7.5, 41.5), xytext=(CX0 + 7.5, 21),
                arrowprops=dict(arrowstyle='-|>', color=ROJO_FLECHA, linewidth=3.2,
                                shrinkA=0, shrinkB=0))
    ax.plot([27, CX0 + 7.5], [21, 21], color=ROJO_FLECHA, linewidth=3.2,
            solid_capstyle='butt')

    plt.tight_layout()
    C.guardar(fig, NOMBRE)
    print("  la etapa de reduccion es el recuadro; no hay tabla aparte")
    print("  variantes: " + '   '.join(
        f"{e} {compacto(p)}"
        for e, _, p, s in C.ARQUITECTURAS))


if __name__ == '__main__':
    main()
