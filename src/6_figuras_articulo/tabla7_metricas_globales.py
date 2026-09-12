#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabla 7: metricas globales sobre el conjunto de prueba independiente.

Reemplaza la Tabla 6 del manuscrito, que corresponde al modelo D. Cada celda es
media +- desviacion sobre las cuatro repeticiones completas del entrenamiento
final, no una corrida unica.

Agrega la comparacion contra la configuracion D publicada como hoja aparte. No va
al articulo, pero es lo que hace falta para redactar sin elegir la metrica que
conviene: si la arquitectura adoptada gana en MAE y pierde en RMSE, o al reves,
la asimetria se declara en la Discusion en vez de esconderse.

Datos: resultados/<X>_oficial/<param>/test/metricas_test_por_semilla.xlsx
       (lo produce 00_predicciones_test.py)

Uso:  python tabla6_metricas_globales.py
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd

import comun as C

# Cifras del modelo D (12 211 281 parametros), corrida de dic-2025, que sirven
# de referencia historica. Fuente: resultados_validacion/.
D_PUBLICADO = {
    'Ra': dict(MAE=0.428, MSE=0.353, RMSE=0.594, R2=0.885),
    'Rz': dict(MAE=3.788, MSE=26.969, RMSE=5.193, R2=0.758),
}
DECIMALES = {'Ra': 3, 'Rz': 3}


def main():
    filas, comparacion, crudos = [], [], []
    for param in ('Ra', 'Rz'):
        por_semilla, _ = C.leer_metricas_test(param)
        crudos.append(por_semilla)
        d = DECIMALES[param]

        filas.append({
            'Roughness parameter': param,
            'MAE [µm]': C.ms(por_semilla.MAE, d),
            'MSE [µm²]': C.ms(por_semilla.MSE, d),
            'RMSE [µm]': C.ms(por_semilla.RMSE, d),
            'R²': C.ms(por_semilla.R2, 3),
        })

        for metrica, mejor_es_menor in (('MAE', True), ('MSE', True),
                                        ('RMSE', True), ('R2', False)):
            a = por_semilla[metrica].mean()
            dd = D_PUBLICADO[param][metrica]
            delta = a - dd
            gana_a = (delta < 0) if mejor_es_menor else (delta > 0)
            comparacion.append({
                'param': param, 'metrica': metrica,
                'D_12.2M': dd, 'publicada': round(a, 4),
                'arquitectura': C.ARQ_PUBLICADA[param],
                'sd': round(por_semilla[metrica].std(ddof=1), 4),
                'delta': round(delta, 4),
                'gana': C.ARQ_PUBLICADA[param] if gana_a else 'D',
                'dentro_de_1sd': abs(delta) < por_semilla[metrica].std(ddof=1),
            })

    tabla = pd.DataFrame(filas)
    comp = pd.DataFrame(comparacion)

    destino = C.SALIDAS / 'Tabla7_metricas_globales.xlsx'
    with pd.ExcelWriter(destino) as w:
        tabla.to_excel(w, sheet_name='Tabla7', index=False)
        comp.to_excel(w, sheet_name='Comparacion_pub_vs_D', index=False)
        pd.concat(crudos, ignore_index=True).to_excel(
            w, sheet_name='Crudo_por_semilla', index=False)

    print(f"Tabla 7 --- Ra con {C.ARQ_PUBLICADA['Ra']}, Rz con {C.ARQ_PUBLICADA['Rz']}, "
          f"media +- desviacion sobre "
          f"{len(crudos[0])} repeticiones")
    print(tabla.to_string(index=False))
    print(f"\n-> {destino}\n")
    print("Comparacion contra el modelo D publicado:")
    print(comp.to_string(index=False))

    md = ['| Roughness parameter | MAE [µm] | MSE [µm²] | RMSE [µm] | R² |',
          '|---|---|---|---|---|']
    for _, r in tabla.iterrows():
        md.append(f"| {r['Roughness parameter']} | {r['MAE [µm]']} | {r['MSE [µm²]']} | "
                  f"{r['RMSE [µm]']} | {r['R²']} |")
    texto = '\n'.join(md)
    (C.SALIDAS / 'Tabla7_metricas_globales.md').write_text(texto, encoding='utf-8')
    print(f"\n{texto}")


if __name__ == '__main__':
    main()
