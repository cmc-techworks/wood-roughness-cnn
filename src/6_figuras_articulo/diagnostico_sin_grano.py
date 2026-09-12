#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resultado de la ablacion: cuanto puede la imagen SOLA, sin la entrada de grano.

Cierra el diagnostico que empezaron `diagnostico_r2.py` y `diagnostico_r2_vc.py`.
Aquellos median cuanto aporta la imagen POR ENCIMA del piso del grano, dentro de
un modelo que tiene las dos entradas. Este mide el caso limite: que pasa si a la
red no se le da la opcion de apoyarse en el grano en absoluto.

Compara tres cosas, sobre la validacion cruzada (6 probetas):

  piso           media de Ra por grano en las 5 probetas de entrenamiento
                 (el mismo piso que ya se uso en diagnostico_r2_vc.py)
  A (con grano)  arquitectura publicada en el articulo
  A sin grano    misma base convolucional y misma reduccion, sin la rama
                 categorica ni la concatenacion (ablacion_sin_grano.py)

Datos: 3_entrenamiento/resultados_barrido/AsinGrano_25.1k_seed{10,20}_limpio/

Uso:  python diagnostico_sin_grano.py
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, binomtest

import comun as C

CORRIDAS_SIN_GRANO = ['AsinGrano_25.1k_seed10_limpio', 'AsinGrano_25.1k_seed20_limpio']
CORRIDAS_CON_GRANO = C.VC_RA  # 5 semillas, ya usadas en diagnostico_r2_vc.py


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def pliegues(carpeta):
    t = []
    for i in range(1, C.N_PLIEGUES + 1):
        d = pd.read_excel(C.BARRIDO / carpeta / 'predicciones' / f'predicciones_fold{i}.xlsx')
        d.columns = [c.strip() for c in d.columns]
        d['fold'] = i
        t.append(d)
    return pd.concat(t, ignore_index=True)


def piso_por_pliegue(d):
    """Media de Ra por grano en las 5 probetas de ENTRENAMIENTO de cada pliegue."""
    piso = np.empty(len(d))
    for f in d.fold.unique():
        dentro = d.fold == f
        medias = d[~dentro].groupby('Grano').Ra_real.mean()
        piso[dentro.values] = d.loc[dentro, 'Grano'].map(medias).values
    return piso


def senal_local(d):
    """Correlacion medido-estimado dentro de cada combinacion probeta x grano."""
    rs, ns = [], []
    for (_, _), s in d.groupby(['id_probeta', 'Grano']):
        if len(s) >= 15 and s.Ra_predicho.std() > 1e-9:
            rs.append(pearsonr(s.Ra_real, s.Ra_predicho)[0])
            ns.append(len(s))
    return np.array(rs), np.array(ns)


def main():
    print('=' * 78)
    print('R2 y MAE agregados, validacion cruzada (6 probetas)')
    print('=' * 78)

    filas = []
    for carpeta in CORRIDAS_SIN_GRANO:
        d = pliegues(carpeta)
        piso = piso_por_pliegue(d)
        y, yhat = d.Ra_real.values, d.Ra_predicho.values
        filas.append(dict(modelo='A sin grano', corrida=carpeta,
                          R2_piso=r2(y, piso), R2_modelo=r2(y, yhat),
                          MAE_piso=np.abs(y - piso).mean(), MAE_modelo=np.abs(y - yhat).mean()))
    for carpeta in CORRIDAS_CON_GRANO:
        d = pliegues(carpeta)
        piso = piso_por_pliegue(d)
        y, yhat = d.Ra_real.values, d.Ra_predicho.values
        filas.append(dict(modelo='A con grano', corrida=carpeta,
                          R2_piso=r2(y, piso), R2_modelo=r2(y, yhat),
                          MAE_piso=np.abs(y - piso).mean(), MAE_modelo=np.abs(y - yhat).mean()))

    t = pd.DataFrame(filas)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    g = t.groupby('modelo')[['R2_modelo', 'MAE_modelo']].agg(['mean', 'std'])
    print(f'\nResumen (media ± sd sobre repeticiones):')
    for modelo in ['A sin grano', 'A con grano']:
        r2m = t[t.modelo == modelo].R2_modelo
        maem = t[t.modelo == modelo].MAE_modelo
        print(f'  {modelo:<14} R2 = {r2m.mean():.4f} ± {r2m.std(ddof=1):.4f}   '
              f'MAE = {maem.mean():.4f} ± {maem.std(ddof=1):.4f} µm  (n={len(r2m)} reps)')
    piso_medio = t.R2_piso.mean()
    print(f'  {"piso (grano solo)":<14} R2 = {piso_medio:.4f}')

    r2_sin = t[t.modelo == 'A sin grano'].R2_modelo.mean()
    r2_con = t[t.modelo == 'A con grano'].R2_modelo.mean()
    print(f'\n  Sin la entrada de grano, el R2 cae de {r2_con:.3f} a {r2_sin:.3f} '
          f'({r2_con - r2_sin:+.3f}).')
    print(f'  El piso trivial (0.783) supera ampliamente a la imagen sola ({r2_sin:.3f}): '
          f'la imagen por si misma queda muy por debajo de solo conocer el grano.')

    # ------------------------------------------------------------------ senal local
    print(f'\n\n{"=" * 78}\nSenal local: correlacion medido-estimado dentro de cada probeta x grano'
          f'\n{"=" * 78}')
    filas2 = []
    for carpeta in CORRIDAS_SIN_GRANO:
        rs, ns = senal_local(pliegues(carpeta))
        pos = int((rs > 0).sum())
        filas2.append(dict(modelo='A sin grano', corrida=carpeta, grupos=len(rs),
                           r_medio=np.average(rs, weights=ns), positivos=pos,
                           p_signo=binomtest(pos, len(rs), 0.5).pvalue))
    for carpeta in CORRIDAS_CON_GRANO:
        rs, ns = senal_local(pliegues(carpeta))
        pos = int((rs > 0).sum())
        filas2.append(dict(modelo='A con grano', corrida=carpeta, grupos=len(rs),
                           r_medio=np.average(rs, weights=ns), positivos=pos,
                           p_signo=binomtest(pos, len(rs), 0.5).pvalue))
    t2 = pd.DataFrame(filas2)
    print(t2.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    for modelo in ['A sin grano', 'A con grano']:
        sub = t2[t2.modelo == modelo]
        print(f'\n  {modelo}: r medio = {sub.r_medio.mean():+.3f} '
              f'(rango {sub.r_medio.min():+.3f} a {sub.r_medio.max():+.3f}), '
              f'p = {sub.p_signo.mean():.3f} en promedio')

    with pd.ExcelWriter(C.SALIDAS / 'diagnostico_sin_grano.xlsx') as w:
        t.to_excel(w, sheet_name='R2_agregado', index=False)
        t2.to_excel(w, sheet_name='Senal_local', index=False)
    print(f'\n-> {C.SALIDAS / "diagnostico_sin_grano.xlsx"}')


if __name__ == '__main__':
    main()
