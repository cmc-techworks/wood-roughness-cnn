#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""El mismo diagnostico del piso, pero sobre la validacion cruzada.

`diagnostico_r2.py` mide sobre el conjunto de prueba, que es una sola probeta.
Este mide sobre los seis pliegues, es decir seis probetas distintas, para saber
si el hallazgo —que casi todo el R2 lo aporta la entrada de grano y no la
imagen— es una peculiaridad de la probeta 11 o una propiedad del modelo.

El piso de cada pliegue se calcula con las medias por grano de las CINCO probetas
de entrenamiento de ese pliegue, nunca con las de la probeta evaluada.

Uso:  python diagnostico_r2_vc.py
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, binomtest

import comun as C

CORRIDAS = {
    'A · 25.7k': [f'A_25.7k_seed{s}_limpio' for s in (10, 11, 12, 13, 20)],
    'D · 12.21M': ['D_12.21M_seed10_limpio', 'D_12.21M_seed20_limpio'],
}


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def pliegues(carpeta):
    """Predicciones de los 6 pliegues de una corrida, concatenadas."""
    trozos = []
    for i in range(1, C.N_PLIEGUES + 1):
        f = C.BARRIDO / carpeta / 'predicciones' / f'predicciones_fold{i}.xlsx'
        d = pd.read_excel(f)
        d.columns = [c.strip() for c in d.columns]
        d['fold'] = i
        trozos.append(d)
    return pd.concat(trozos, ignore_index=True)


def main():
    filas = []
    for nombre, carpetas in CORRIDAS.items():
        for carpeta in carpetas:
            d = pliegues(carpeta)
            # el piso de cada pliegue: media por grano de las OTRAS cinco probetas
            piso = np.empty(len(d))
            for f in d.fold.unique():
                dentro = d.fold == f
                fuera = d[~dentro]
                medias = fuera.groupby('Grano').Ra_real.mean()
                piso[dentro.values] = d.loc[dentro, 'Grano'].map(medias).values

            y, yhat = d.Ra_real.values, d.Ra_predicho.values
            r2_piso, r2_mod = r2(y, piso), r2(y, yhat)
            filas.append(dict(
                modelo=nombre, corrida=carpeta, n=len(d),
                R2_piso=r2_piso, R2_modelo=r2_mod, aporte=r2_mod - r2_piso,
                frac_restante=(r2_mod - r2_piso) / (1 - r2_piso),
                MAE_piso=np.abs(y - piso).mean(), MAE_modelo=np.abs(y - yhat).mean(),
                sd_est_sobre_sd_med=yhat.std(ddof=1) / y.std(ddof=1),
            ))

    t = pd.DataFrame(filas)
    print('Validacion cruzada, Ra — piso = media por grano de las otras 5 probetas\n')
    print(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print('\n\nResumen por arquitectura (media sobre repeticiones):')
    g = t.groupby('modelo')[['R2_piso', 'R2_modelo', 'aporte', 'frac_restante',
                             'MAE_piso', 'MAE_modelo']].mean()
    print(g.to_string(float_format=lambda v: f'{v:.4f}'))

    print('\n  aporte        = cuanto sube el R2 respecto de ignorar la imagen')
    print('  frac_restante = que fraccion de la varianza no explicada por el grano '
          'recupera la imagen')
    for m, r in g.iterrows():
        print(f'    {m:<12} la imagen aporta {r.aporte:+.4f} en R2 '
              f'({r.frac_restante:.1%} de lo que quedaba) y '
              f'{r.MAE_piso - r.MAE_modelo:+.4f} µm en MAE')

    # ---------------------------------------------------------------------
    # El agregado de arriba dice que la imagen no aporta. Pero agrupa las seis
    # probetas, y el modelo no puede saber cual esta mirando: el desplazamiento
    # de nivel entre probetas dentro de un mismo grano es varianza que no puede
    # explicar y que arrastra la correlacion a cero. La pregunta correcta es si
    # aporta DENTRO de una probeta y un grano, que es donde vive la senal local.
    print(f'\n\n{"=" * 78}\nSENAL LOCAL: correlacion dentro de cada probeta x grano'
          f'\n{"=" * 78}')
    filas2 = []
    for nombre, carpetas in CORRIDAS.items():
        for carpeta in carpetas:
            d = pliegues(carpeta)
            rs, ns = [], []
            for (_, _), s in d.groupby(['id_probeta', 'Grano']):
                if len(s) >= 15 and s.Ra_predicho.std() > 1e-9:
                    rs.append(pearsonr(s.Ra_real, s.Ra_predicho)[0])
                    ns.append(len(s))
            rs, ns = np.array(rs), np.array(ns)
            pos = int((rs > 0).sum())
            filas2.append(dict(
                modelo=nombre, corrida=carpeta, grupos=len(rs),
                r_medio=np.average(rs, weights=ns), r_mediana=np.median(rs),
                positivos=pos,
                p_signo=binomtest(pos, len(rs), 0.5).pvalue))
    t2 = pd.DataFrame(filas2)
    print(t2.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n  r_medio es la correlacion entre lo medido y lo estimado dentro de cada')
    print('  una de las 24 combinaciones probeta x grano. p_signo contrasta que la')
    print('  mitad de los grupos daria correlacion positiva por azar.')
    g2 = t2.groupby('modelo')[['r_medio', 'p_signo']].mean()
    for m, r in g2.iterrows():
        veredicto = 'significativa' if r.p_signo < 0.05 else 'NO significativa'
        print(f'    {m:<12} r = {r.r_medio:+.3f}   ({veredicto})')

    with pd.ExcelWriter(C.SALIDAS / 'diagnostico_r2_vc.xlsx') as w:
        t.to_excel(w, sheet_name='Piso_agregado', index=False)
        t2.to_excel(w, sheet_name='Senal_local', index=False)
    print(f'\n-> {C.SALIDAS / "diagnostico_r2_vc.xlsx"}')


if __name__ == '__main__':
    main()
