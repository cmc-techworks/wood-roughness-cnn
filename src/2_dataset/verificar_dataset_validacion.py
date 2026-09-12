#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chequeo rápido de consistencia: ¿todas las imágenes de la hoja 'Validacion'
del DATASET.xlsx existen en disco? Rutas auto-resueltas respecto a la raíz del repo.

Uso:
    python verificar_dataset_validacion.py            # hoja Validacion
    python verificar_dataset_validacion.py --hoja sin_outliers
"""
import argparse
import os
from pathlib import Path

import pandas as pd

# Dos layouts aceptados: el original de la tesis y el de esta carpeta de revisión.
_DATA_SUBPATHS = [
    os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6'),
    'fotos_recortes',
    os.path.join('data', 'fotos_recortes'),
]


def _encontrar_data_dir():
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in _DATA_SUBPATHS:
            candidato = carpeta / sub
            if candidato.is_dir() and (candidato / 'DATASET.xlsx').is_file():
                return candidato
    raise SystemExit(f"ERROR: no se encontró ninguna carpeta de datos con DATASET.xlsx "
                     f"({' | '.join(_DATA_SUBPATHS)}) subiendo desde {inicio}")


def main():
    p = argparse.ArgumentParser(description='Verifica que las imágenes del DATASET.xlsx existan en disco.')
    p.add_argument('--hoja', default='Validacion', help="Hoja del Excel a verificar (p. ej. 'sin_outliers').")
    args = p.parse_args()

    data_dir = _encontrar_data_dir()
    excel = data_dir / 'DATASET.xlsx'
    df = pd.read_excel(excel, sheet_name=args.hoja)

    # Las imágenes de la hoja Validacion viven en la subcarpeta Validacion/ si existe
    candidatos = [data_dir / 'Validacion', data_dir]
    archivos = set()
    for c in candidatos:
        if c.is_dir():
            archivos.update(os.listdir(c))

    encontrados = df['nombre_imagen'].isin(archivos).sum()
    print(f"Excel: {excel}")
    print(f"Hoja '{args.hoja}': {len(df)} filas")
    print(f"Imágenes encontradas en disco: {encontrados}")

    no_encontradas = df[~df['nombre_imagen'].isin(archivos)]
    if len(no_encontradas) > 0:
        print(f"Faltantes: {len(no_encontradas)}. Ejemplos:")
        for nombre in no_encontradas['nombre_imagen'].head(5):
            print(f"  - {nombre}")
    else:
        print("OK: todas las imágenes de la hoja existen en disco.")


if __name__ == '__main__':
    main()
