#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 15 (Ra) y Fig. 16 (Rz): distribucion de valores medidos y estimados por grano.

Reemplaza las figuras actuales del manuscrito, generadas en diciembre de 2025 con
el modelo D. Misma estructura: por cada grano, dos cajas contiguas —medido y
estimado— con los puntos individuales jitterados encima; azul para lo medido,
naranja para lo estimado en Ra y verde en Rz.

Vuelca ademas MAE, sesgo y dispersion por grano a un .xlsx. Esa tabla es la que
hace falta para reverificar las afirmaciones cualitativas del texto de Resultados
—que el mayor error esta en P40 y el menor en P120— que hoy describen a D y no
tienen por que seguir siendo ciertas con A.

Datos: resultados/<X>_oficial/<param>/test/seed<REF>/resultados_validacion_<param>.xlsx
       (190 muestras validas de la probeta 11, generadas por 00_predicciones_test.py)

Uso:
    python fig15_16_medido_vs_estimado.py
    python fig15_16_medido_vs_estimado.py --param Ra
"""
import argparse
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import comun as C

FIGURA = {'Ra': 'Fig17_medido_vs_estimado_Ra', 'Rz': 'Fig18_medido_vs_estimado_Rz'}
ANCHO_CAJA = 0.30
SEPARACION = 0.19          # media distancia entre la caja medida y la estimada


def caja(ax, x, valores, color_borde, color_relleno):
    bp = ax.boxplot(valores, positions=[x], widths=ANCHO_CAJA, showfliers=False,
                    patch_artist=True, manage_ticks=False)
    for pieza in bp['boxes']:
        pieza.set(facecolor=color_relleno, edgecolor='#555555', linewidth=1.0)
    for clave in ('whiskers', 'caps'):
        for pieza in bp[clave]:
            pieza.set(color='#555555', linewidth=1.0)
    for pieza in bp['medians']:
        pieza.set(color='#333333', linewidth=1.4)


def construir(param):
    df = C.leer_predicciones_test(param)
    col_real, col_est = f'{param}_real', f'{param}_estimado'
    color_est = C.COLOR_ESTIMADO[param]
    relleno_est = C.RELLENO_ESTIMADO[param]

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    rng = np.random.default_rng(C.SEED_REF[param])

    for i, grano in enumerate(C.GRANOS):
        sub = df[df.grano == grano]
        if sub.empty:
            continue
        for desplaz, valores, color, relleno in (
                (-SEPARACION, sub[col_real].values, C.AZUL, C.RELLENO_MEDIDO),
                (+SEPARACION, sub[col_est].values, color_est, relleno_est)):
            x = i + desplaz
            caja(ax, x, valores, color, relleno)
            ax.plot(x + rng.normal(0, 0.045, len(valores)), valores, 'o',
                    markersize=4.2, color=color, alpha=0.85, markeredgewidth=0,
                    zorder=3)

    ax.set_xticks(range(len(C.GRANOS)))
    ax.set_xticklabels([f'P{g}' for g in C.GRANOS])
    ax.set_xlim(-0.6, len(C.GRANOS) - 0.4)
    ax.set_xlabel('Grit Size')
    ax.set_ylabel(f'{param} (µm)')
    ax.legend(handles=[mpatches.Patch(color=C.AZUL, label='Measured'),
                       mpatches.Patch(color=color_est, label='Estimated')],
              loc='upper right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    C.guardar(fig, FIGURA[param])

    # ------------------------------------------------- cifras por grano ------
    filas = []
    for grano in C.GRANOS:
        sub = df[df.grano == grano]
        real, est = sub[col_real].values, sub[col_est].values
        filas.append({
            'Grit': f'P{grano}', 'n': len(sub),
            'Measured_mean': real.mean(), 'Measured_sd': real.std(ddof=1),
            'Estimated_mean': est.mean(), 'Estimated_sd': est.std(ddof=1),
            'MAE': np.abs(est - real).mean(),
            'Bias': (est - real).mean(),
            'MAE_over_sd': np.abs(est - real).mean() / real.std(ddof=1),
        })
    tabla = pd.DataFrame(filas)
    tabla.to_excel(C.SALIDAS / f'{FIGURA[param]}_por_grano.xlsx', index=False)

    print(f"\n{param} · corrida de referencia seed {C.SEED_REF[param]} · n = {len(df)}")
    print(tabla.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    peor = tabla.loc[tabla.MAE.idxmax(), 'Grit']
    mejor = tabla.loc[tabla.MAE.idxmin(), 'Grit']
    print(f"  MAE mayor en {peor}, menor en {mejor}")
    print(f"  MAE/sd por grano: {tabla.MAE_over_sd.min():.0%} – {tabla.MAE_over_sd.max():.0%}")
    return tabla


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param', nargs='+', choices=['Ra', 'Rz'], default=['Ra', 'Rz'])
    args = ap.parse_args()

    C.estilo()
    for p in args.param:
        construir(p)


if __name__ == '__main__':
    main()
