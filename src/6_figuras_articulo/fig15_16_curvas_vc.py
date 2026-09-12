#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 15 (Ra) y Fig. 16 (Rz): curvas de entrenamiento y validacion del modelo adoptado.

Reemplaza las figuras actuales del manuscrito, que corresponden al modelo D de
12 211 281 parametros. Misma estructura de dos paneles: (a) perdida MSE y (b) MAE,
azul entrenamiento y naranja validacion, con la banda sombreada representando la
dispersion entre los seis pliegues de la validacion cruzada.

Lo que se ve cambia, y ese es el punto. Con D la curva de entrenamiento se hundia
hasta un MAE de ~0.05 um mientras la de validacion se estancaba en ~0.6 um: la
firma clasica de memorizacion. Con las variantes que reducen el mapa de
caracteristicas las dos curvas convergen juntas.

Datos: 3_entrenamiento/resultados_barrido/<X>_seed<REF>_limpio{,_Rz}/metricas/
       historial_por_epoca.xlsx  (6 hojas, una por pliegue, 40 epocas)

Uso:
    python fig13_14_curvas_vc.py               # las dos
    python fig13_14_curvas_vc.py --param Ra
"""
import argparse
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import comun as C

FIGURA = {'Ra': 'Fig15_curvas_Ra', 'Rz': 'Fig16_curvas_Rz'}


def apilar(historiales, columna):
    """Matriz (pliegues x epocas) de una metrica."""
    return np.vstack([h[columna].values for h in historiales.values()])


def panel(ax, epocas, tr, va, ylabel):
    """Media entre pliegues con banda de +-1 SD, entrenamiento contra validacion."""
    for datos, color, etiqueta in ((tr, C.AZUL, 'Training'),
                                   (va, C.NARANJA, 'Validation')):
        media, sd = datos.mean(axis=0), datos.std(axis=0, ddof=1)
        ax.plot(epocas, media, color=color, linewidth=1.8, label=etiqueta)
        ax.fill_between(epocas, media - sd, media + sd, color=color, alpha=0.22,
                        linewidth=0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, len(epocas) + 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)


def construir(param):
    hist = C.leer_historial(C.VC_REF[param])
    if len(hist) != C.N_PLIEGUES:
        raise SystemExit(f"Se esperaban {C.N_PLIEGUES} pliegues y hay {len(hist)}")

    n_epocas = len(next(iter(hist.values())))
    epocas = np.arange(1, n_epocas + 1)

    loss_tr, loss_va = apilar(hist, 'loss'), apilar(hist, 'val_loss')
    mae_tr, mae_va = apilar(hist, 'mae'), apilar(hist, 'val_mae')

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    panel(axes[0], epocas, loss_tr, loss_va, 'Loss (MSE)')
    panel(axes[1], epocas, mae_tr, mae_va, 'MAE [µm]')
    C.etiqueta_panel(axes[0], '(a)')
    C.etiqueta_panel(axes[1], '(b)')
    plt.tight_layout()
    C.guardar(fig, FIGURA[param])

    # Cifras que hacen falta para redactar el parrafo que comenta la figura
    final = pd.DataFrame({
        'metrica': ['MSE train', 'MSE val', 'MAE train [µm]', 'MAE val [µm]'],
        'epoca_40_media': [loss_tr[:, -1].mean(), loss_va[:, -1].mean(),
                           mae_tr[:, -1].mean(), mae_va[:, -1].mean()],
        'epoca_40_sd': [loss_tr[:, -1].std(ddof=1), loss_va[:, -1].std(ddof=1),
                        mae_tr[:, -1].std(ddof=1), mae_va[:, -1].std(ddof=1)],
    })
    print(f"\n{param} · corrida {C.VC_REF[param]} · {C.N_PLIEGUES} pliegues, {n_epocas} epocas")
    print(final.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    brecha = mae_va[:, -1].mean() - mae_tr[:, -1].mean()
    razon = mae_va[:, -1].mean() / mae_tr[:, -1].mean()
    print(f"  brecha final MAE val-train: {brecha:+.4f} µm  (razon val/train = {razon:.2f})")
    final.to_excel(C.SALIDAS / f'{FIGURA[param]}_cifras.xlsx', index=False)
    return final


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
