#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 12: arquitectura de la red multimodal, con las cuatro variantes evaluadas.

Reemplaza el diagrama actual del manuscrito, que muestra una capa `Flatten` y
declara 12.2 millones de parametros. Dos cambios:

1. La ruta principal pasa a mostrar la arquitectura seleccionada, con
   `GlobalAveragePooling2D` en lugar de `Flatten`.
2. En esa misma posicion cuelga un recuadro con las cuatro variantes que se
   sometieron a entrenamiento, cada una con su conteo de parametros, marcando la
   seleccionada: la eleccion se apoya en una comparacion contra arquitecturas
   mas livianas, no en justificar 12.2 M de parametros por asercion.

La base convolucional es identica en las cuatro; lo unico que cambia es como se
reduce el mapa de caracteristicas antes de la primera capa densa. Fuente de la
arquitectura y de los conteos: entrenar_final_limpio.py:107-143 (no los README,
que tienen B y C invertidas).

Uso:  python fig12_arquitectura.py
"""
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import comun as C

NOMBRE = 'Fig12_arquitectura'

# Miniatura de entrada: un recorte real del conjunto, como en la figura original.
RECORTE = C.DATOS / 'P10_C1_G40_Recorte_Recorte_A1_1.1.tiff'

AZUL_CONV = '#5b9bd5'
NARANJA_POOL = '#f4b183'
VERDE_DENSA = '#a9d18e'
LILA_CONCAT = '#b4a7d6'
AMARILLO_SALIDA = '#ffc000'
ROJO_FLECHA = '#c00000'
BORDE = '#1f3864'


def slabs(ax, x, y_centro, alto, ancho, n, color, dx=1.4, dy=1.1):
    """Pila de laminas con desplazamiento diagonal, el volumen 3D de la figura original."""
    for i in range(n - 1, -1, -1):
        ax.add_patch(mpatches.Rectangle(
            (x + i * dx, y_centro - alto / 2 + i * dy), ancho, alto,
            facecolor=color, edgecolor=BORDE, linewidth=0.9, zorder=10 - i))


def columna(ax, x, y_centro, n, color, lado=2.6, hueco=1.0, elipsis=False):
    """Columna de celdas, la representacion de un vector de activaciones."""
    total = n * lado + (n - 1) * hueco
    y0 = y_centro + total / 2 - lado
    for i in range(n):
        y = y0 - i * (lado + hueco)
        if elipsis and i == n - 2:
            for k in range(3):
                ax.plot(x + lado / 2, y + lado / 2 - 1.0 + k * 1.0, '.',
                        color='#333333', markersize=3)
            continue
        ax.add_patch(mpatches.Rectangle((x, y), lado, lado, facecolor=color,
                                        edgecolor=BORDE, linewidth=0.8, zorder=5))
    return total


def circulos(ax, x, y_centro, n, color, radio=1.5, hueco=1.4, elipsis=False):
    total = n * 2 * radio + (n - 1) * hueco
    y0 = y_centro + total / 2 - radio
    for i in range(n):
        y = y0 - i * (2 * radio + hueco)
        if elipsis and i == n - 2:
            for k in range(3):
                ax.plot(x, y - 1.2 + k * 1.2, '.', color='#333333', markersize=3)
            continue
        ax.add_patch(mpatches.Circle((x, y), radio, facecolor=color,
                                     edgecolor=BORDE, linewidth=0.8, zorder=5))
    return total


# Escala de los tamanos de letra del diagrama, ajustada para que el texto sea
# legible al ancho de columna de la revista. Los tamanos base se
# conservan como estaban y se multiplican aqui, de modo que la proporcion entre
# rotulos, titulo y notas al pie no cambia y basta tocar un numero para reajustar.
ESCALA_LETRA = 1.40


def rotulo(ax, x, texto, y=99.5):
    ax.text(x, y, texto, ha='center', va='top', fontsize=10.5 * ESCALA_LETRA, linespacing=1.25)


def main():
    C.estilo()
    fig, ax = plt.subplots(figsize=(16.0, 9.6))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_facecolor('white')

    Y = 70          # eje de la ruta convolucional

    # ------------------------------------------------ entrada: recorte real --
    try:
        from PIL import Image
        img = np.asarray(Image.open(RECORTE).convert('L'))
        ax.imshow(img, cmap='gray', extent=(5, 15, Y - 17, Y + 17),
                  aspect='auto', zorder=5)
        ax.add_patch(mpatches.Rectangle((5, Y - 17), 10, 34, fill=False,
                                        edgecolor=BORDE, linewidth=1.0, zorder=6))
    except Exception as e:                                   # sin PIL o sin el archivo
        print(f"  aviso: no se pudo cargar la miniatura ({e}); se dibuja un marco vacio")
        ax.add_patch(mpatches.Rectangle((5, Y - 17), 10, 34, facecolor='#dddddd',
                                        edgecolor=BORDE, linewidth=1.0, zorder=5))
    rotulo(ax, 10, 'Input image\n390 × 130 × 1')

    # ----------------------------------------------------- base convolucional --
    slabs(ax, 21, Y, 32, 6, 3, AZUL_CONV, dy=0.9)
    rotulo(ax, 26, 'Conv2D\n(3×3, 32 filters)')

    slabs(ax, 39, Y, 21, 4.5, 3, NARANJA_POOL, dy=0.9)
    rotulo(ax, 43, 'MaxPooling\n(2×2)')

    slabs(ax, 53, Y, 23, 4.5, 6, AZUL_CONV, dy=0.9)
    rotulo(ax, 60, 'Conv2D\n(3×3, 64 filters)')

    slabs(ax, 70, Y, 15, 3.5, 6, NARANJA_POOL, dy=0.9)
    rotulo(ax, 76, 'MaxPooling\n(2×2)')

    # ----------------------------------- reduccion: la etapa que se compara --
    # Neutra a proposito. La figura abre una seccion que compara cuatro maneras
    # de reducir el mapa de caracteristicas; dibujar aqui la de una de ellas
    # adelantaria la conclusion antes de mostrar la evidencia. El recuadro
    # punteado de abajo enumera las cuatro, ninguna senalada.
    ax.add_patch(mpatches.FancyBboxPatch(
        (85.0, Y - 11.0), 9.0, 22.0,
        boxstyle='round,pad=0.5,rounding_size=1.2',
        facecolor='#eceff3', edgecolor='#555555', linewidth=1.1,
        linestyle=(0, (5, 3)), zorder=5))
    ax.text(89.5, Y, '?', ha='center', va='center', fontsize=17 * ESCALA_LETRA,
            color='#555555', zorder=6)
    rotulo(ax, 89.5, 'Feature-map\nreduction')

    columna(ax, 101, Y, 7, VERDE_DENSA, elipsis=True)
    rotulo(ax, 102.5, 'Dense\n(64, ReLU)')

    # -------------------------------------------------------- concatenacion --
    ax.add_patch(mpatches.Rectangle((113, 44), 15, 36, fill=False,
                                    edgecolor=BORDE, linewidth=1.2, zorder=4))
    columna(ax, 116, 62, 6, LILA_CONCAT, lado=2.4, hueco=1.6, elipsis=True)
    circulos(ax, 124, 62, 6, LILA_CONCAT, radio=1.3, hueco=1.9, elipsis=True)
    ax.text(120.5, 83, 'Concatenation\n(80)', ha='center', va='bottom', fontsize=10.5 * ESCALA_LETRA,
            linespacing=1.25)

    columna(ax, 134, 62, 5, VERDE_DENSA, lado=3.4, hueco=1.6)
    ax.text(135.7, 78, 'Dense\n(32, ReLU)', ha='center', va='bottom', fontsize=10.5 * ESCALA_LETRA,
            linespacing=1.25)

    ax.add_patch(mpatches.FancyBboxPatch(
        (147, 58), 8, 8, boxstyle='round,pad=0.4,rounding_size=1.2',
        facecolor=AMARILLO_SALIDA, edgecolor=BORDE, linewidth=1.0, zorder=5))
    ax.text(151, 72, 'Output\n(Ra or Rz)', ha='center', va='bottom', fontsize=10.5 * ESCALA_LETRA,
            linespacing=1.25)

    # ------------------------------------------ segunda entrada: el grano ----
    circulos(ax, 8, 15, 4, AZUL_CONV, radio=1.7, hueco=1.6)
    ax.text(8, 27, 'Input grit\n(one-hot, 4)', ha='center', va='bottom', fontsize=10.5 * ESCALA_LETRA,
            linespacing=1.25)

    circulos(ax, 24, 15, 7, AZUL_CONV, radio=1.5, hueco=1.5, elipsis=True)
    ax.text(24, 31, 'Dense\n(16, ReLU)', ha='center', va='bottom', fontsize=10.5 * ESCALA_LETRA,
            linespacing=1.25)

    # ------------------------------------------------- flechas de fusion ----
    ax.annotate('', xy=(112.5, Y), xytext=(105.5, Y),
                arrowprops=dict(arrowstyle='-|>', color=ROJO_FLECHA, linewidth=3.2,
                                shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(120.5, 43.5), xytext=(120.5, 15),
                arrowprops=dict(arrowstyle='-|>', color=ROJO_FLECHA, linewidth=3.2,
                                shrinkA=0, shrinkB=0))
    ax.plot([27, 120.5], [15, 15], color=ROJO_FLECHA, linewidth=3.2,
            solid_capstyle='butt')

    # --------------------------------- recuadro de las variantes evaluadas ---
    bx0, by0, bw, bh = 44, 25, 62, 23
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx0, by0), bw, bh, boxstyle='round,pad=0.6,rounding_size=1.0',
        facecolor='#f7f7f7', edgecolor='#555555', linewidth=1.1,
        linestyle=(0, (5, 3)), zorder=3))
    ax.plot([89.5, 89.5], [by0 + bh, Y - 11.5], color='#555555', linewidth=1.0,
            linestyle=(0, (3, 3)), zorder=2)

    ax.text(bx0 + bw / 2, by0 + bh - 2.6,
            'Feature-map reduction — four variants trained and compared',
            ha='center', va='top', fontsize=10.5 * ESCALA_LETRA, style='italic', color='#333333')

    y_fila = by0 + bh - 8.0
    for etiqueta, descripcion, params, elegida in C.ARQUITECTURAS:
        peso = 'bold' if elegida else 'normal'
        color = '#000000' if elegida else '#444444'
        ax.text(bx0 + 3.5, y_fila, etiqueta, ha='left', va='center',
                fontsize=10.5 * ESCALA_LETRA, fontweight='bold', color=color)
        ax.text(bx0 + 8.0, y_fila, descripcion, ha='left', va='center',
                fontsize=10 * ESCALA_LETRA, fontweight=peso, color=color)
        ax.text(bx0 + bw - 12.0, y_fila, f'{params:,}'.replace(',', ' '),
                ha='right', va='center', fontsize=10 * ESCALA_LETRA, fontweight=peso, color=color)
        y_fila -= 4.4

    ax.text(bx0 + bw - 12.0, by0 + bh - 5.2, 'parameters', ha='right', va='center',
            fontsize=9 * ESCALA_LETRA, style='italic', color='#666666')

    plt.tight_layout()
    C.guardar(fig, NOMBRE)
    print("  etapa de reduccion neutra; las cuatro variantes van sin senalar")


if __name__ == '__main__':
    main()
