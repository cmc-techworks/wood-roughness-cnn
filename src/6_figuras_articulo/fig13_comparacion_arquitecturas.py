#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura nueva: error de entrenamiento contra error de validacion en las cuatro arquitecturas.

Muestra graficamente por que 12.2 millones de parametros no estaban justificados.
En un rango de 476x en numero de parametros:

    el error de VALIDACION es plano   (~0.57 - 0.67 um, bandas que se solapan)
    el error de ENTRENAMIENTO colapsa (0.548 -> 0.064 um, monotono)

Es decir, la capacidad adicional no compra generalizacion: compra ajuste al ruido
de la medicion de referencia. La razon val/train impresa en cada panel resume el
panel y crece de forma monotona con el tamano.

DOS VERSIONES, para elegir cual entra al articulo. Cambia solo que representa la
banda sombreada, y eso cambia el pie de figura:

  --modo pliegues       Una sola corrida por arquitectura (semilla 10) y banda =
                        dispersion entre los SEIS PLIEGUES. Es la misma
                        construccion de las Fig. 13 y 14, asi que el lector ya
                        sabe leerla, y es comparable panel a panel porque son la
                        misma semilla y las mismas probetas. La banda mide
                        variabilidad ENTRE PROBETAS.

  --modo repeticiones   Las cuatro repeticiones del protocolo en las cuatro
                        variantes y banda = dispersion sobre el conjunto de
                        curvas repeticion x pliegue. La banda mide la
                        variabilidad TOTAL del procedimiento, inicializacion
                        incluida. Es la mas honesta sobre la incertidumbre, y
                        desde el 2026-09-04 el numero de curvas es el mismo en
                        los cuatro paneles (4 x 6 = 24).

CUIDADO AL REDACTAR EL PIE DE FIGURA. En ninguno de los dos modos la banda es la
dispersion entre repeticiones sola, que es la cantidad (0.073 en R2) sobre la que
se apoya el argumento del texto de que "las cuatro empatan dentro del ruido".
Esta figura demuestra otra cosa, complementaria: que la validacion es plana
mientras el entrenamiento colapsa.

Datos: 3_entrenamiento/resultados_barrido/*_limpio[_Rz]/metricas/historial_por_epoca.xlsx,
       seis hojas de pliegue por corrida, 40 epocas.

Uso:
    python fig_comparacion_arquitecturas.py            # genera las dos, Ra
    python fig_comparacion_arquitecturas.py --modo pliegues
    python fig_comparacion_arquitecturas.py --param Rz  # comparacion de arquitecturas en Rz

--param Rz agregado el 2026-09-09, una vez que C y D tuvieron su barrido en Rz (hasta entonces
solo A y B lo tenian). El eje Y se recalcula para Rz en vez de reusar el 1.5 um de Ra: la escala
de Rz es un orden de magnitud mayor.
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

SEED_PAREJA = 10          # la semilla comun a las cuatro, para el modo "pliegues"


def repeticiones(param):
    """Repeticiones con historial por epoca: las cuatro inicializaciones del
    protocolo en las cuatro variantes, todas posteriores al arreglo de
    determinismo del 2026-08-17.

    Hasta el 2026-09-04 solo la variante publicada llevaba las cuatro y las otras
    tres entraban con una sola corrida. Y hasta el 2026-09-09, C y D no tenian
    ninguna corrida en Rz (solo A y B) — completado ese dia con el mismo
    protocolo, 4 semillas x 6 pliegues cada una.
    """
    sufijo = '' if param == 'Ra' else f'_{param}'
    return {
        a: [f'{C.ETIQUETA_BARRIDO[a]}_seed{s}_limpio{sufijo}' for s in C.SEEDS_VC]
        for a, _, _, _ in C.ARQUITECTURAS
    }


def nombre_modo(modo, param):
    base = {'pliegues': 'Fig13_comparacion_arquitecturas_pliegues',
            'repeticiones': 'Fig13_comparacion_arquitecturas_repeticiones'}[modo]
    return base if param == 'Ra' else f'{base}_{param}'


MODOS = {
    'pliegues': dict(banda='dispersión entre los 6 pliegues, semilla 10'),
    'repeticiones': dict(banda='dispersión sobre repeticiones × pliegues'),
}

Y_MAX_RA = 1.5             # la epoca 1 de A llega a 3.23 um y aplastaria el resto


def curvas(arq, modo, param='Ra'):
    """Devuelve (train, val, n_repeticiones), cada matriz de (curvas x epocas)."""
    carpetas = repeticiones(param)[arq]
    if modo == 'pliegues':
        carpetas = [c for c in carpetas if f'seed{SEED_PAREJA}_' in c]
        if not carpetas:
            raise SystemExit(f"{arq}: no hay corrida con semilla {SEED_PAREJA}")

    tr, va = [], []
    for carpeta in carpetas:
        hist = C.leer_historial(carpeta)
        if len(hist) != C.N_PLIEGUES:
            raise SystemExit(f"{carpeta}: {len(hist)} pliegues, se esperaban {C.N_PLIEGUES}")
        tr += [h['mae'].values for h in hist.values()]
        va += [h['val_mae'].values for h in hist.values()]
    return np.vstack(tr), np.vstack(va), len(carpetas)


def panel(ax, epocas, tr, va, etiqueta, params, elegida, n_rep, modo, y_max):
    for datos, color, nombre in ((tr, C.AZUL, 'Training'),
                                 (va, C.NARANJA, 'Validation')):
        media, sd = datos.mean(axis=0), datos.std(axis=0, ddof=1)
        ax.plot(epocas, media, color=color, linewidth=1.8, label=nombre, zorder=4)
        ax.fill_between(epocas, media - sd, media + sd, color=color, alpha=0.22,
                        linewidth=0, zorder=3)

    razon = va[:, -1].mean() / tr[:, -1].mean()
    ax.text(0.965, 0.93, f'{razon:.2f}×', transform=ax.transAxes,
            ha='right', va='top', fontsize=14, fontweight='bold', color='#333333')
    ax.text(0.965, 0.845, 'val / train', transform=ax.transAxes,
            ha='right', va='top', fontsize=9.5, style='italic', color='#666666')

    if modo == 'repeticiones':
        ax.text(0.035, 0.055, f'{n_rep} rep. × {C.N_PLIEGUES} folds',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=9, style='italic', color='#666666')

    titulo = f"{etiqueta} · {f'{params:,}'.replace(',', ' ')} parameters"
    if elegida:
        titulo += '  (selected)'
    ax.set_title(titulo, fontsize=12,
                 fontweight='bold' if elegida else 'normal',
                 color='#000000' if elegida else '#333333', pad=8)

    ax.set_xlim(0, len(epocas) + 1)
    ax.set_ylim(0, y_max)
    return razon


def construir(modo, param='Ra'):
    nombre = nombre_modo(modo, param)
    banda = MODOS[modo]['banda']

    # Primero se cargan las cuatro arquitecturas para poder fijar un eje Y comun.
    # Para Ra se reusa el 1.5 um ya citado en el pie de figura publicado; para Rz
    # no hay una cifra previa que reusar, asi que se calcula del propio dato.
    datos_arq = {}
    for etiqueta, descripcion, params, elegida in C.ARQUITECTURAS:
        tr, va, n_rep = curvas(etiqueta, modo, param)
        datos_arq[etiqueta] = (tr, va, n_rep, descripcion, params, elegida)

    if param == 'Ra':
        y_max = Y_MAX_RA
    else:
        pico = max(tr.max() for tr, _, _, _, _, _ in datos_arq.values())
        y_max = float(np.ceil(pico * 1.05 * 2) / 2)  # margen 5%, redondeado a 0.5

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=True, sharey=True)
    planos = axes.ravel()

    filas = []
    for ax, (etiqueta, descripcion, params, elegida) in zip(planos, C.ARQUITECTURAS):
        tr, va, n_rep, descripcion, params, elegida = datos_arq[etiqueta]
        epocas = np.arange(1, tr.shape[1] + 1)
        razon = panel(ax, epocas, tr, va, etiqueta, params, elegida, n_rep, modo, y_max)
        filas.append(dict(
            arquitectura=etiqueta, reduccion=descripcion, parametros=params,
            repeticiones=n_rep, curvas=tr.shape[0],
            MAE_train=tr[:, -1].mean(), MAE_train_sd=tr[:, -1].std(ddof=1),
            MAE_val=va[:, -1].mean(), MAE_val_sd=va[:, -1].std(ddof=1),
            razon_val_train=razon,
        ))

    unidad = 'µm'
    for ax, letra in zip(planos, '(a) (b) (c) (d)'.split()):
        C.etiqueta_panel(ax, letra, dx=-0.10, dy=1.14)
    for ax in axes[1, :]:
        ax.set_xlabel('Epoch')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'{param} MAE [{unidad}]')

    # Leyenda unica al pie, para no tapar curvas en ningun panel.
    manejadores, nombres = planos[0].get_legend_handles_labels()
    fig.legend(manejadores, nombres, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.015), frameon=True, framealpha=0.9, fontsize=11)

    plt.tight_layout(h_pad=2.6, w_pad=1.8, rect=(0, 0.035, 1, 1))
    C.guardar(fig, nombre)

    tabla = pd.DataFrame(filas)
    tabla.to_excel(C.SALIDAS / f"{nombre}_cifras.xlsx", index=False)

    print(f"\n=== {param} · modo '{modo}' · banda = {banda} ===")
    print(tabla[['arquitectura', 'parametros', 'repeticiones', 'curvas',
                 'MAE_train', 'MAE_val', 'razon_val_train']].to_string(
        index=False, float_format=lambda v: f'{v:.3f}'))

    razones = tabla.razon_val_train.values
    vmin, vmax = tabla.MAE_val.min(), tabla.MAE_val.max()
    print(f"  [{'OK ' if np.all(np.diff(razones) > 0) else 'REVISAR'}] la razón "
          f"val/train crece con el tamaño: {', '.join(f'{r:.2f}×' for r in razones)}")
    print(f"  [{'OK ' if vmax - vmin < 0.15 * (Y_MAX_RA if param == 'Ra' else y_max) else 'REVISAR'}] "
          f"el error de validación es plano: {vmin:.3f} – {vmax:.3f} µm (rango {vmax - vmin:.3f})")
    print(f"        el de entrenamiento va de {tabla.MAE_train.max():.3f} a "
          f"{tabla.MAE_train.min():.3f} µm, un factor de "
          f"{tabla.MAE_train.max() / tabla.MAE_train.min():.1f}")
    print(f"  eje Y recortado en {y_max:g} µm.")
    return tabla


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modo', nargs='+', choices=list(MODOS), default=list(MODOS))
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    args = ap.parse_args()

    C.estilo()
    for modo in args.modo:
        construir(modo, args.param)


if __name__ == '__main__':
    main()
