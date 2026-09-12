# -*- coding: utf-8 -*-
"""Diagnostico: R2, pendiente y razon de dispersion sobre la nube AGRUPADA de validacion
cruzada (los 6 pliegues juntos, todos los granos), por semilla y en media, B contra C.

Es la cifra de pendiente con que se discutio la eleccion de C para Rz (Fig. 14 del
anexo). Se guardo desde el scratchpad de la sesion del 2026-09-10
(`agregado_4semillas.py`) para que tenga script. Corre con el Python del sistema.
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
    return pd.concat(trozos, ignore_index=True)


def main():
    for param in ['Ra', 'Rz']:
        print(f"\n=== {param}: nube agrupada de validacion cruzada, 4 semillas ===")
        sufijo = '' if param == 'Ra' else f'_{param}'
        for arch in ARQS:
            filas = []
            for s in C.SEEDS_VC:
                d = agrupar_pliegues(f'{C.ETIQUETA_BARRIDO[arch]}_seed{s}_limpio{sufijo}')
                real, est = d[f'{param}_real'].values, d[f'{param}_predicho'].values
                pend = stats.linregress(real, est).slope
                filas.append(dict(seed=s, R2=r2_identidad(real, est), pendiente=pend,
                                  sd_ratio=est.std(ddof=1) / real.std(ddof=1)))
            t = pd.DataFrame(filas)
            print(f"\n{arch}:")
            print(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
            print(f"  media: R2={t.R2.mean():.4f}±{t.R2.std():.4f}  "
                  f"pendiente={t.pendiente.mean():.4f}±{t.pendiente.std():.4f}  "
                  f"sd_ratio={t.sd_ratio.mean():.4f}")


if __name__ == '__main__':
    main()
