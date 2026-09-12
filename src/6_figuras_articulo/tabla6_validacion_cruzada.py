#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabla 6: metricas de validacion cruzada por pliegue, arquitectura publicada.

Reemplaza la Tabla 5 del manuscrito, que corresponde al modelo D. Cada celda es
media +- desviacion sobre las repeticiones completas de la validacion cruzada
(cuatro por parametro), no una corrida unica: asi la tabla muestra tambien la
variabilidad entre pliegues y entre inicializaciones, que es lo que domina la
dispersion cuando solo hay siete probetas.

Se agrega la columna de probeta de validacion, que el manuscrito no traia: el
pliegue 1 no es un numero arbitrario sino la probeta 8, y esa correspondencia es
la que permite leer la dispersion como variabilidad entre piezas.

Datos: 3_entrenamiento/resultados_barrido/<X>_seed*_limpio{,_Rz}/metricas/
       metricas_validacion_cruzada.xlsx

Uso:  python tabla5_validacion_cruzada.py
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

import comun as C

DECIMALES = {'Ra': dict(MAE=3, RMSE=3, R2=3), 'Rz': dict(MAE=2, RMSE=2, R2=3)}


def recopilar(carpetas):
    """Concatena las metricas por pliegue de todas las repeticiones."""
    trozos = []
    for carpeta in carpetas:
        df = C.leer_vc(carpeta)
        df['corrida'] = carpeta
        trozos.append(df)
    return pd.concat(trozos, ignore_index=True)


def resumir(param, carpetas):
    todo = recopilar(carpetas)
    dec = DECIMALES[param]

    filas = []
    for fold in range(1, C.N_PLIEGUES + 1):
        sub = todo[todo.Fold == fold]
        probetas = sub.Probeta.unique()
        assert len(probetas) == 1, f"pliegue {fold} con probetas distintas: {probetas}"
        filas.append({
            'Fold': fold,
            'Specimen': int(probetas[0]),
            'n_val': int(sub.n_val.iloc[0]),
            'MAE': C.ms(sub.MAE, dec['MAE']),
            'RMSE': C.ms(sub.RMSE, dec['RMSE']),
            'R2': C.ms(sub.R2, dec['R2']),
            '_MAE': sub.MAE.mean(), '_RMSE': sub.RMSE.mean(), '_R2': sub.R2.mean(),
        })

    # La fila Mean promedia primero dentro de cada repeticion y luego entre ellas:
    # asi su desviacion es la que hay entre repeticiones completas, que es la que
    # cita la carta, y no la que hay entre pliegues.
    por_corrida = todo.groupby('corrida')[['MAE', 'RMSE', 'R2']].mean()
    filas.append({
        'Fold': 'Mean', 'Specimen': '', 'n_val': int(todo.groupby('Fold').n_val.first().sum()),
        'MAE': C.ms(por_corrida.MAE, dec['MAE']),
        'RMSE': C.ms(por_corrida.RMSE, dec['RMSE']),
        'R2': C.ms(por_corrida.R2, dec['R2']),
        '_MAE': por_corrida.MAE.mean(), '_RMSE': por_corrida.RMSE.mean(),
        '_R2': por_corrida.R2.mean(),
    })
    return pd.DataFrame(filas), todo, por_corrida


def main():
    tablas, crudos, porcorrida = {}, {}, {}
    for param, carpetas in (('Ra', C.VC_RA), ('Rz', C.VC_RZ)):
        t, todo, pc = resumir(param, carpetas)
        tablas[param], crudos[param], porcorrida[param] = t, todo, pc

        print(f"\n{'=' * 78}\n{param} · arquitectura {C.ARQ_PUBLICADA[param]} · {len(carpetas)} repeticiones "
              f"de la validacion cruzada\n{'=' * 78}")
        print(t[['Fold', 'Specimen', 'n_val', 'MAE', 'RMSE', 'R2']].to_string(index=False))
        print(f"\n  dispersion ENTRE PLIEGUES (dentro de una repeticion, media): "
              f"R2 sd = {todo.groupby('corrida').R2.std(ddof=1).mean():.3f}")
        print(f"  dispersion ENTRE REPETICIONES (de la media): "
              f"R2 sd = {pc.R2.std(ddof=1):.3f}")
        print(f"  rango de R2 por pliegue: {todo.R2.min():.3f} – {todo.R2.max():.3f}")

    # ------------------------------------------------------- tabla combinada --
    ra, rz = tablas['Ra'], tablas['Rz']
    comb = pd.DataFrame({
        'Fold': ra.Fold, 'Specimen': ra.Specimen,
        'Ra_MAE': ra.MAE, 'Ra_RMSE': ra.RMSE, 'Ra_R2': ra.R2,
        'Rz_MAE': rz.MAE, 'Rz_RMSE': rz.RMSE, 'Rz_R2': rz.R2,
    })

    destino = C.SALIDAS / 'Tabla6_validacion_cruzada.xlsx'
    with pd.ExcelWriter(destino) as w:
        comb.to_excel(w, sheet_name='Tabla6', index=False)
        for p in ('Ra', 'Rz'):
            crudos[p][['corrida', 'Fold', 'Probeta', 'n_val', 'MAE', 'RMSE', 'R2']].to_excel(
                w, sheet_name=f'Crudo_{p}', index=False)
            porcorrida[p].to_excel(w, sheet_name=f'Por_repeticion_{p}')
    print(f"\n-> {destino}")

    # ------------------------------------------------- version para el .docx --
    md = ['| Fold | Specimen | Ra MAE [µm] | Ra RMSE [µm] | Ra R² | '
          'Rz MAE [µm] | Rz RMSE [µm] | Rz R² |',
          '|---|---|---|---|---|---|---|---|']
    for _, r in comb.iterrows():
        md.append(f"| {r.Fold} | {r.Specimen} | {r.Ra_MAE} | {r.Ra_RMSE} | {r.Ra_R2} | "
                  f"{r.Rz_MAE} | {r.Rz_RMSE} | {r.Rz_R2} |")
    texto = '\n'.join(md)
    (C.SALIDAS / 'Tabla6_validacion_cruzada.md').write_text(texto, encoding='utf-8')
    print(f"\n{texto}")


if __name__ == '__main__':
    main()
