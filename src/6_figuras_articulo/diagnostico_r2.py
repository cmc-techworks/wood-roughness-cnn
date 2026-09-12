#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""De donde sale el R2 del conjunto de prueba, y por que la Fig. 17 se ve peor de lo que dice.

Responde a una objecion concreta: el diagrama de dispersion muestra bandas
horizontales y una pendiente de 0.81, es decir peor capacidad aparente de seguir
la rugosidad medida, y sin embargo las metricas agregadas no empeoran.

La clave es que el modelo tiene DOS entradas, la imagen y el grano abrasivo en
codificacion one-hot. Como los cuatro granos ocupan rangos de rugosidad casi
disjuntos, un estimador que solo mirara el grano y devolviera la media de su
grupo ya alcanza un R2 alto. Este script cuantifica ese piso y mide cuanto
aporta la imagen por encima de el, para las dos arquitecturas.

Tres niveles de comparacion:

  piso           predecir la media de entrenamiento del grano, ignorando la imagen
  modelo         la prediccion real
  intra-grano    correlacion entre lo medido y lo estimado dentro de cada grano,
                 que es la unica parte que la imagen puede explicar

Uso:  python diagnostico_r2.py
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

import comun as C

# Predicciones del modelo publicado D sobre el mismo conjunto de prueba.
# Elegidas por coincidencia exacta con la Tabla 6 del manuscrito enviado:
#   Ra  MAE 0.428  RMSE 0.594  R2 0.885
#   Rz  MAE 3.788  RMSE 5.193  R2 0.758
# Ojo: existe otra corrida de D para Rz en resultados_validacion/final/, con
# MAE 3.536 y R2 0.786, que NO es la publicada. Usarla inflaria a D en la
# comparacion.
D_RA = C.RESULTADOS / 'resultados_validacion' / 'ra' / 'resultados_validacion_Ra.xlsx'
D_RZ = C.RESULTADOS / 'resultados_validacion' / 'resultados_validacion_Rz.xlsx'

# Conjunto de entrenamiento limpio, 1030 muestras
DATASET = C.DATOS / 'DATASET.xlsx'


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def cargar_D(param):
    ruta = D_RA if param == 'Ra' else D_RZ
    df = pd.read_excel(ruta, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={'grano': 'grano'})


def medias_entrenamiento(param):
    """Media de cada grano en el conjunto de entrenamiento, sin mirar el test."""
    tr = pd.read_excel(DATASET, sheet_name='sin_outliers')
    tr.columns = [c.strip() for c in tr.columns]
    col_g = next(c for c in tr.columns if c.lower().startswith('gran'))
    return tr.groupby(col_g)[param].mean(), len(tr)


def analizar(param):
    print(f'\n{"=" * 78}\n{param}\n{"=" * 78}')

    medias, n_tr = medias_entrenamiento(param)
    print(f'  medias de entrenamiento por grano (n={n_tr}): ' +
          '  '.join(f'P{int(g)}={v:.3f}' for g, v in medias.items()))

    modelos = {}
    a = C.leer_predicciones_test(param)
    modelos[f'A · 25.7k (seed {C.SEED_REF[param]})'] = (
        a[f'{param}_real'].values, a[f'{param}_estimado'].values, a['grano'].values)
    d = cargar_D(param)
    modelos['D · 12.21M (publicada)'] = (
        d[f'{param}_real'].values, d[f'{param}_estimado'].values, d['grano'].values)

    real = modelos[f'A · 25.7k (seed {C.SEED_REF[param]})'][0]
    granos = modelos[f'A · 25.7k (seed {C.SEED_REF[param]})'][2]
    piso = np.array([medias[g] for g in granos])

    filas = []
    for nombre, (y, yhat, g) in [('piso: solo el grano', (real, piso, granos))] + \
                                [(k, v) for k, v in modelos.items()]:
        sl, ic, _, _, _ = stats.linregress(y, yhat)
        # R2 dentro de cada grano, agrupado: cuanto explica por encima de la media del grano
        resid_modelo = np.sum((y - yhat) ** 2)
        medias_test = pd.Series(y).groupby(g).transform('mean').values
        resid_piso_oraculo = np.sum((y - medias_test) ** 2)
        filas.append(dict(
            modelo=nombre,
            R2=r2(y, yhat),
            MAE=np.abs(y - yhat).mean(),
            RMSE=np.sqrt(((y - yhat) ** 2).mean()),
            pendiente=sl,
            sd_est_sobre_sd_med=yhat.std(ddof=1) / y.std(ddof=1),
            R2_intra=1 - resid_modelo / resid_piso_oraculo,
        ))

    t = pd.DataFrame(filas)
    print()
    print(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    base = t.loc[0, 'R2']
    print(f'\n  El piso —predecir la media del grano, sin mirar la imagen— ya da '
          f'R2 = {base:.4f}.')
    for _, r in t.iloc[1:].iterrows():
        gana = r.R2 - base
        print(f'    {r.modelo:<28} R2 = {r.R2:.4f}   aporte de la imagen: '
              f'{gana:+.4f}  ({gana / (1 - base):.1%} de lo que quedaba por explicar)')

    print('\n  R2_intra es la fraccion de la varianza DENTRO de cada grano que el '
          'modelo explica.')
    print('  Negativo significa que, dentro de un grano, predecir la media del grupo '
          'seria mejor.')
    return t


def main():
    tablas = {p: analizar(p) for p in ('Ra', 'Rz')}
    with pd.ExcelWriter(C.SALIDAS / 'diagnostico_r2.xlsx') as w:
        for p, t in tablas.items():
            t.to_excel(w, sheet_name=p, index=False)
    print(f'\n-> {C.SALIDAS / "diagnostico_r2.xlsx"}')


if __name__ == '__main__':
    main()
