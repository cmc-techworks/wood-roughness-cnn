# -*- coding: utf-8 -*-
"""Anclaje del mapa de Ra contra el SJ-310, agregado sobre las 4 semillas de B-Ra.

La Fig. 20 muestra una sola corrida (la semilla de referencia). Las tablas del
articulo informan media ± SD sobre las 4 repeticiones, asi que la cifra de anclaje
que va al texto tambien: se lee la hoja `Anclaje_resumen` que escribe
`verificar_mapa_espacial.py --anclaje --seed N` para cada semilla.

La semilla de referencia escribe `Fig20_21_verificacion.xlsx`; las otras,
`Fig20_21_verificacion_seed{N}.xlsx`. Si falta alguna:

    # activar el entorno: conda activate roughness-vision
    $TESIS verificar_mapa_espacial.py --anclaje --seed N

Corre con el Python del sistema. Salida: salidas/Fig20_21_anclaje_4_semillas.xlsx
"""
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd

import comun as C


def archivo(seed):
    return C.SALIDAS / ('Fig20_21_verificacion.xlsx' if seed == C.SEED_REF['Ra']
                        else f'Fig20_21_verificacion_seed{seed}.xlsx')


def main():
    filas = []
    for s in C.SEEDS_TEST:
        f = archivo(s)
        if not f.is_file():
            raise SystemExit(f"Falta {f.name}. Correr verificar_mapa_espacial.py --anclaje --seed {s}")
        r = dict(pd.read_excel(f, sheet_name='Anclaje_resumen').values)
        filas.append({'semilla': s,
                      'mae_zona_a_zona': float(r['mae_zona_a_zona']),
                      'media_mapa_cara': float(r['media_mapa_cara']),
                      'media_sj310_C1': float(r['media_sj310_C1']),
                      'sd_sj310_C1': float(r['sd_sj310_C1']),
                      'diferencia': float(r['diferencia'])})
    t = pd.DataFrame(filas)
    t['diferencia_en_sd'] = t.diferencia / t.sd_sj310_C1
    print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    agg = {
        'mae_zona_a_zona_media': t.mae_zona_a_zona.mean(),
        'mae_zona_a_zona_sd': t.mae_zona_a_zona.std(ddof=1),
        'diferencia_media_um': t.diferencia.mean(),
        'diferencia_sd_um': t.diferencia.std(ddof=1),
        'diferencia_media_en_sd': t.diferencia_en_sd.mean(),
        'todas_dentro_de_1_sd': bool((t.diferencia_en_sd.abs() < 1).all()),
        'semilla_de_la_figura': C.SEED_REF['Ra'],
    }
    print(f"\nMAE zona a zona, 4 semillas : {agg['mae_zona_a_zona_media']:.3f} ± "
          f"{agg['mae_zona_a_zona_sd']:.3f} µm")
    print(f"Diferencia de la media      : {agg['diferencia_media_um']:+.3f} ± "
          f"{agg['diferencia_sd_um']:.3f} µm ({agg['diferencia_media_en_sd']:+.2f} SD del SJ-310)")
    print(f"Todas dentro de 1 SD        : {agg['todas_dentro_de_1_sd']}")

    destino = C.SALIDAS / 'Fig20_21_anclaje_4_semillas.xlsx'
    with pd.ExcelWriter(destino) as w:
        t.to_excel(w, sheet_name='Por_semilla', index=False)
        pd.DataFrame(list(agg.items()), columns=['clave', 'valor']).to_excel(
            w, sheet_name='Resumen', index=False)
    print(f"\n-> {destino}")


if __name__ == '__main__':
    main()
