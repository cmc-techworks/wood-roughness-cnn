#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 17: coeficiente de determinacion de los modelos de estimacion, (a) Ra y (b) Rz.

Reemplaza la figura actual del manuscrito, que corresponde al modelo D. Misma
estructura: dos paneles apilados, dispersion de medido contra estimado, recta de
regresion por minimos cuadrados con su banda de confianza al 95 % y la diagonal
1:1 punteada en rojo como referencia.

La banda de confianza se calcula aqui en vez de delegarla a seaborn.regplot, que
es lo que hacia validar_headless.py, para que el intervalo quede documentado:
es el de la media condicional de la recta, no un intervalo de prediccion.

Datos: resultados/<X>_oficial/<param>/test/seed<REF>/resultados_validacion_<param>.xlsx

Uso:  python fig17_r2.py
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import comun as C

NOMBRE = 'Fig19_r2_Ra_Rz'


def r2_identidad(y, y_est):
    """Coeficiente de determinacion contra la identidad, 1 - SSE/SST.

    Es el que reportan la Tabla 6, el texto y el abstract, y el que calcula
    sklearn.metrics.r2_score en validar_headless.py. NO es el r^2 de la recta
    ajustada, que da 0.887 en vez de 0.873 para Ra porque premia a la recta por
    absorber la compresion de escala del modelo en su pendiente. Mezclarlos
    haria que la figura contradijera a la tabla.
    """
    return 1 - np.sum((y - y_est) ** 2) / np.sum((y - y.mean()) ** 2)


def banda_confianza(x, y, xs, alfa=0.05):
    """Banda de confianza de la media condicional de la recta ajustada."""
    n = len(x)
    pend, inter, _, _, _ = stats.linregress(x, y)
    ajuste = inter + pend * x
    s_err = np.sqrt(np.sum((y - ajuste) ** 2) / (n - 2))
    t = stats.t.ppf(1 - alfa / 2, n - 2)
    medio = inter + pend * xs
    semi = t * s_err * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    return medio, medio - semi, medio + semi, pend, inter


def panel(ax, df, param):
    real = df[f'{param}_real'].values
    est = df[f'{param}_estimado'].values

    lo = min(real.min(), est.min())
    hi = max(real.max(), est.max())
    margen = 0.04 * (hi - lo)
    xs = np.linspace(real.min(), real.max(), 200)

    medio, inf, sup, pend, inter = banda_confianza(real, est, xs)

    ax.plot([lo - margen, hi + margen], [lo - margen, hi + margen],
            '--', color=C.ROJO, linewidth=1.4, zorder=1, label='1:1')
    ax.fill_between(xs, inf, sup, color=C.AZUL, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(xs, medio, color=C.AZUL, linewidth=1.8, zorder=3)
    ax.plot(real, est, 'o', markersize=4.5, color=C.AZUL, alpha=0.55,
            markeredgewidth=0, zorder=4)

    # Recuadro con las cifras del panel: la figura se titula por el R2, asi que el
    # R2 tiene que estar visible dentro de ella.
    r2 = r2_identidad(real, est)
    mae = np.abs(real - est).mean()
    dec = 3 if param == 'Ra' else 2
    ax.text(0.04, 0.96,
            f'R² = {r2:.3f}\nSlope = {pend:.2f}\nMAE = {mae:.{dec}f} µm\nn = {len(df)}',
            transform=ax.transAxes, ha='left', va='top', fontsize=11, zorder=6,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#333333', lw=0.6, alpha=0.9))

    ax.set_xlabel(f'{param} measured values (µm)')
    ax.set_ylabel(f'{param} estimated values (µm)')
    ax.set_xlim(lo - margen, hi + margen)
    ax.set_ylim(lo - margen, hi + margen)
    ax.set_aspect('equal', adjustable='box')
    return dict(param=param, n=len(df),
                R2=r2, MAE=mae,
                r2_de_la_recta=stats.linregress(real, est).rvalue ** 2,
                pendiente=pend, intercepto=inter)


def main():
    C.estilo()
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 12.2))
    resumen = []
    for ax, param in zip(axes, ('Ra', 'Rz')):
        resumen.append(panel(ax, C.leer_predicciones_test(param), param))
    C.etiqueta_panel(axes[0], '(a)', dx=-0.16, dy=1.05)
    C.etiqueta_panel(axes[1], '(b)', dx=-0.16, dy=1.05)
    plt.tight_layout(h_pad=2.5)
    C.guardar(fig, NOMBRE)

    tabla = pd.DataFrame(resumen)
    tabla.to_excel(C.SALIDAS / f'{NOMBRE}_cifras.xlsx', index=False)
    print(f"corrida de referencia --- "
          f"Ra: arquitectura {C.ARQ_PUBLICADA['Ra']}, seed {C.SEED_REF['Ra']}   "
          f"Rz: arquitectura {C.ARQ_PUBLICADA['Rz']}, seed {C.SEED_REF['Rz']}")
    print(tabla.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print("\nR2 es el que va al pie de figura y coincide con la Tabla 6. "
          "r2_de_la_recta se muestra solo para dejar constancia de que son "
          "distintos y de cual se reporta.")
    print("La pendiente por debajo de 1 es la compresion hacia la media por grano: "
          "el modelo subestima la cola rugosa y sobreestima la lisa.")


if __name__ == '__main__':
    main()
