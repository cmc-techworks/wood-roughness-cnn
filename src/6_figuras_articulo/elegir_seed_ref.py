#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Que semilla muestran las figuras que ensenan una sola corrida.

Las Tablas 6 y 7 informan media y desviacion sobre las cuatro repeticiones del
entrenamiento. Las Fig. 15 a 19, en cambio, dibujan una sola: no se puede graficar
una media de curvas de perdida ni una nube de dispersion promediada. Esa corrida
tiene que ser representativa, o la figura contradice a la tabla que tiene al lado.

Criterio, el mismo con que se eligio la semilla de referencia de la arquitectura
anterior: para cada una de las cuatro cantidades que el articulo reporta -R2 de
validacion cruzada y R2 de test, en Ra y en Rz- se mide cuanto se aparta esa
semilla de la media de las cuatro, en unidades de la desviacion entre ellas. Gana
la de menor suma.

No basta con mirar una sola cantidad. La semilla 10 de la arquitectura A era
aceptable en Ra y la peor de las cuatro en Rz sobre el test (0.745 contra una
media de 0.769): elegida por Ra, habria hecho que la Fig. 17b contradijera a la
Tabla 6.

Escribe nada: imprime el resultado para copiarlo a `comun.SEED_REF`.

Uso:  python elegir_seed_ref.py
"""
import argparse
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

import comun as C


def r2_test(param, seed):
    f = (C.oficial(param) / param / 'test' / f'seed{seed}' /
         f'resultados_validacion_{param}.xlsx')
    if not f.is_file():
        return np.nan
    m = pd.read_excel(f, sheet_name='Métricas').set_index('Métrica')['Valor']
    return float(m['R²'])


def r2_vc(param, seed):
    sufijo = '' if param == 'Ra' else '_Rz'
    carpeta = f'{C.ETIQUETA_BARRIDO[C.ARQ_PUBLICADA[param]]}_seed{seed}_limpio{sufijo}'
    if not (C.BARRIDO / carpeta / 'metricas' /
            'metricas_validacion_cruzada.xlsx').is_file():
        return np.nan
    return float(C.leer_vc(carpeta).R2.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    args = ap.parse_args()
    param = args.param

    semillas = sorted(set(C.SEEDS_VC) | set(C.SEEDS_TEST))
    tabla = pd.DataFrame(
        {f'{origen}_{param}': [f(param, s) for s in semillas]
         for origen, f in (('VC', r2_vc), ('test', r2_test))},
        index=semillas)
    tabla.index.name = 'seed'

    print(f"{param} con arquitectura {C.ARQ_PUBLICADA[param]} — R² por semilla\n")
    print(tabla.to_string(float_format=lambda v: f'{v:.4f}'))

    faltan = tabla.columns[tabla.isna().any()].tolist()
    if faltan:
        print(f"\nFALTAN corridas en: {', '.join(faltan)}")
        print("El criterio necesita las cuatro cantidades. Completar el computo "
              "antes de fijar SEED_REF.")
        return

    # Distancia a la media de cada columna, en unidades de su propia desviacion.
    # Estandarizar es lo que permite sumar R2 de test y de validacion cruzada, que
    # se mueven en rangos distintos: sin eso, la columna mas dispersa decidiria sola.
    z = (tabla - tabla.mean()).abs() / tabla.std(ddof=1)
    z['suma'] = z.sum(axis=1)

    print("\nDesvio respecto de la media, en unidades de desviacion estandar\n")
    print(z.to_string(float_format=lambda v: f'{v:.2f}'))

    elegida = int(z.suma.idxmin())
    print(f"\n-> SEED_REF = {elegida}   (suma {z.suma.min():.2f}; "
          f"le siguen {', '.join(f'{int(s)} con {z.suma[s]:.2f}' for s in z.suma.drop(elegida).sort_values().index)})")
    if elegida != C.SEED_REF[param]:
        print(f"\n   ATENCION: comun.SEED_REF[{param!r}] vale {C.SEED_REF[param]}. Actualizarlo a "
              f"{elegida} y regenerar las Fig. 15 a 19.")
    else:
        print(f"\n   comun.SEED_REF ya vale {elegida}. Nada que cambiar.")


if __name__ == '__main__':
    main()
