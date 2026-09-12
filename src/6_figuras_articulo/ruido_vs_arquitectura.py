#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""El argumento de que las cuatro variantes empatan, medido en MAE y en R2.

El parrafo que hoy esta propuesto para la seccion de arquitectura razona en R2:
la diferencia entre la mejor y la peor variante es equivalente al rango que
produce repetir la misma validacion cruzada con otra inicializacion. Pero R2 es
la metrica secundaria del articulo; la principal es el MAE, porque es la que la
red optimiza. Este script calcula las dos versiones del argumento para poder
elegir con el numero delante.

La comparacion honesta es entre dos rangos:

  entre arquitecturas   max - min de las medias de las cuatro variantes
  entre repeticiones    max - min de las medias de una misma variante,
                        repitiendo la validacion cruzada completa con otra
                        semilla de inicializacion

Si el segundo es comparable o mayor que el primero, la comparacion entre
arquitecturas no tiene resolucion para ordenarlas.

Desde el 2026-09-04 el ruido de repeticion se mide en las CUATRO variantes, no
solo en una: las cuatro tienen las cuatro inicializaciones del protocolo. Eso
arregla dos cosas a la vez. La primera, que antes se extrapolaba a las cuatro un
ruido medido en una sola. La segunda, un sesgo aritmetico: max - min crece con el
numero de muestras, y se estaba comparando el rango de cinco repeticiones contra
el de cuatro medias por arquitectura. Ahora los dos rangos son max - min de
cuatro valores y se comparan sin correccion.

Uso:  python ruido_vs_arquitectura.py
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

import comun as C
from fig13_comparacion_arquitecturas import REPETICIONES

METRICAS = [('MAE', 'µm', 3), ('R2', '', 3)]


def medias_por_repeticion():
    """Para cada arquitectura, la media sobre los 6 pliegues de cada repeticion."""
    filas = []
    for etiqueta, descripcion, params, elegida in C.ARQUITECTURAS:
        for carpeta in REPETICIONES[etiqueta]:
            folds = C.leer_vc(carpeta)
            filas.append(dict(arch=etiqueta, params=params, corrida=carpeta,
                              MAE=folds.MAE.mean(), R2=folds.R2.mean(),
                              RMSE=folds.RMSE.mean()))
    return pd.DataFrame(filas)


def main():
    df = medias_por_repeticion()

    print('Media sobre los seis pliegues, una fila por repeticion\n')
    print(df[['arch', 'corrida', 'MAE', 'R2']].to_string(
        index=False, float_format=lambda v: f'{v:.4f}'))

    resumen = []
    for metrica, unidad, dec in METRICAS:
        # menor es mejor en MAE, mayor es mejor en R2
        menor_mejor = metrica == 'MAE'
        por_arq = df.groupby('arch')[metrica].mean()
        entre_arq = por_arq.max() - por_arq.min()
        mejor = por_arq.idxmin() if menor_mejor else por_arq.idxmax()
        peor = por_arq.idxmax() if menor_mejor else por_arq.idxmin()

        rangos_rep = df.groupby('arch')[metrica].agg(lambda s: s.max() - s.min())
        n_rep = df.groupby('arch')[metrica].size()
        # El ruido de repeticion tipico: la media de los rangos de las cuatro
        # variantes. Comparable termino a termino con el rango entre arquitecturas
        # porque los dos son max - min de cuatro valores.
        rango_rep = rangos_rep.mean()

        print(f'\n{"=" * 74}\n{metrica}{(" [" + unidad + "]") if unidad else ""}\n{"=" * 74}')
        print('  media por arquitectura:')
        for a in ['A', 'B', 'C', 'D']:
            marca = '  <- mejor' if a == mejor else ('  <- peor' if a == peor else '')
            print(f'    {a}  {por_arq[a]:.{dec}f}   (rango entre sus {n_rep[a]} '
                  f'repeticiones: {rangos_rep[a]:.{dec}f}){marca}')

        print(f'\n  rango ENTRE ARQUITECTURAS       : {entre_arq:.{dec}f}   '
              f'({mejor} mejor, {peor} peor)')
        print(f'  rango ENTRE REPETICIONES, medio : {rango_rep:.{dec}f}   '
              f'(media de los rangos de las cuatro variantes)')
        razon = rango_rep / entre_arq
        print(f'  razon repeticiones / arquitecturas: {razon:.2f}×')
        if razon >= 1:
            print('    -> el ruido de repetir el experimento SUPERA a lo que separa '
                  'a las cuatro arquitecturas')
        elif razon >= 0.8:
            print('    -> el ruido de repetir el experimento es COMPARABLE a lo que '
                  'separa a las cuatro arquitecturas')
        else:
            print('    -> el ruido de repetir queda por DEBAJO: el argumento es mas debil '
                  'en esta metrica')

        # La desviacion estandar es lo que se cita en el texto, y no depende del
        # numero de muestras como el rango. La de repeticion se agrupa sobre las
        # cuatro variantes: raiz de la media de las varianzas dentro de cada una,
        # que es la sd tipica de repetir el experimento con 4 x 3 = 12 grados de
        # libertad, en vez de los 3 que daria una sola variante.
        sd_arq = por_arq.std(ddof=1)
        sd_rep = float(np.sqrt(df.groupby('arch')[metrica].var(ddof=1).mean()))
        sd_rep_por_arq = df.groupby('arch')[metrica].std(ddof=1)
        print('\n  sin el sesgo del rango, comparando desviaciones estandar:')
        print(f'    entre las 4 medias por arquitectura : {sd_arq:.{dec}f}')
        print(f'    entre repeticiones, agrupada        : {sd_rep:.{dec}f}'
              f'   ({sd_rep / sd_arq:.2f}×)')
        print('      por variante: ' + '  '.join(
            f'{a} {sd_rep_por_arq[a]:.{dec}f}' for a in ['A', 'B', 'C', 'D']))

        resumen.append(dict(metrica=metrica, entre_arquitecturas=entre_arq,
                            entre_repeticiones=rango_rep, razon=razon,
                            sd_arquitecturas=sd_arq, sd_repeticiones=sd_rep,
                            razon_sd=sd_rep / sd_arq,
                            mejor=mejor, peor=peor))

    # ------------------------------------------------------------------------
    print(f'\n{"=" * 74}\nCUAL DE LAS DOS VERSIONES CONVIENE\n{"=" * 74}')
    r = pd.DataFrame(resumen).set_index('metrica')
    print(r[['entre_arquitecturas', 'entre_repeticiones', 'razon',
         'sd_arquitecturas', 'sd_repeticiones', 'razon_sd']].to_string(
        float_format=lambda v: f'{v:.4f}'))

    mae_rel = r.loc['MAE', 'entre_arquitecturas'] / df.MAE.mean()
    print(f'\n  El rango entre arquitecturas en MAE es {mae_rel:.1%} del MAE tipico '
          f'({df.MAE.mean():.3f} µm).')
    print(f'  Y la dispersion intra-zona del perfilometro, la vara fisica, es '
          f'0.350 µm: el rango completo entre las cuatro arquitecturas '
          f'({r.loc["MAE", "entre_arquitecturas"]:.3f} µm) cabe '
          f'{0.350 / r.loc["MAE", "entre_arquitecturas"]:.1f} veces dentro de ella.')

    destino = C.SALIDAS / 'ruido_vs_arquitectura.xlsx'
    with pd.ExcelWriter(destino) as w:
        df.to_excel(w, sheet_name='Por_repeticion', index=False)
        r.to_excel(w, sheet_name='Resumen')
    print(f'\n-> {destino}')


if __name__ == '__main__':
    main()
