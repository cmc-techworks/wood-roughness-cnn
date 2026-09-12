# -*- coding: utf-8 -*-
"""Diagnostico: R2 y pendiente DENTRO de cada grano, B contra C, en validacion cruzada.

Respalda lo que se dice de la discretizacion de la Fig. 19b (regresion de Rz
sobre el test): C dispersa mas que B dentro de cada grano, pero el R2 dentro de
cada grano es cercano a cero o negativo en las dos. La nube se ordena en bandas
por grano y la recta global se apoya en el salto entre granos, no en la
variacion dentro de cada uno.

Se guardo desde el scratchpad de la sesion del 2026-09-10 (`analisis_grano.py`)
para que la cifra citada tenga script. Corre con el Python del sistema.
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

import comun as C

ARQS = ['B', 'C']


def r2_identidad(y, yest):
    return 1 - np.sum((y - yest) ** 2) / np.sum((y - y.mean()) ** 2)


def agrupar_pliegues(carpeta):
    trozos = []
    for i in range(1, C.N_PLIEGUES + 1):
        d = pd.read_excel(C.BARRIDO / carpeta / 'predicciones' / f'predicciones_fold{i}.xlsx')
        d.columns = [c.strip() for c in d.columns]
        trozos.append(d)
    d = pd.concat(trozos, ignore_index=True)
    if 'Grano' not in d.columns and 'grano' in d.columns:
        d = d.rename(columns={'grano': 'Grano'})
    return d


def por_grano(d, param):
    filas = []
    for g in C.GRANOS:
        sub = d[d['Grano'] == g]
        real = sub[f'{param}_real'].values
        est = sub[f'{param}_predicho'].values
        if len(real) < 3:
            continue
        pend, _, rval, _, _ = stats.linregress(real, est)
        filas.append(dict(Grano=g, n=len(real), R2=r2_identidad(real, est), pendiente=pend,
                          r2_recta=rval ** 2, MAE=np.abs(real - est).mean(),
                          sd_real=real.std(ddof=1), sd_est=est.std(ddof=1)))
    return pd.DataFrame(filas)


def carpetas(arch, param):
    sufijo = '' if param == 'Ra' else f'_{param}'
    return [f'{C.ETIQUETA_BARRIDO[arch]}_seed{s}_limpio{sufijo}' for s in C.SEEDS_VC]


def main():
    for param in ['Ra', 'Rz']:
        print(f"\n{'=' * 80}\nPARAM = {param}\n{'=' * 80}")
        for arch in ARQS:
            todas = []
            for c in carpetas(arch, param):
                t = por_grano(agrupar_pliegues(c), param)
                t['corrida'] = c
                todas.append(t)
            todas = pd.concat(todas, ignore_index=True)
            todas['sd_est_sobre_sd_real'] = todas.sd_est / todas.sd_real
            resumen = todas.groupby('Grano').agg(
                R2_media=('R2', 'mean'), R2_sd=('R2', 'std'),
                pendiente_media=('pendiente', 'mean'), pendiente_sd=('pendiente', 'std'),
                MAE_media=('MAE', 'mean'),
                sd_est_sobre_sd_real=('sd_est_sobre_sd_real', 'mean')).reset_index()
            print(f"\n--- {arch}, promedio sobre las 4 semillas, por grano ---")
            print(resumen.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
            print(f"  razon sd_est/sd_real media sobre granos: "
                  f"{resumen.sd_est_sobre_sd_real.mean():.3f}")


if __name__ == '__main__':
    main()
