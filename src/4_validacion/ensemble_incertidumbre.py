#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incertidumbre por desacuerdo de ensemble — R1.4.

No reentrena nada. Los 6 modelos de pliegue publicados (`Entrenar modelo/MULTI_VC_RA/`) ya
existen; cada uno vio 5 de las 6 probetas de entrenamiento/validación cruzada. Este script los
corre a los seis sobre las 190 imágenes de test (probeta 7, nunca vista por ninguno de los 6) y
usa la dispersión entre sus 6 predicciones como proxy de incertidumbre por punto — el mismo
principio que un ensemble deep, sin entrenar nada nuevo.

Solo Ra: los 6 .keras de MULTI_VC_RA se entrenaron con Ra como target (ver
`Red_multimodal_VC.py` linea 141, `batch_y.append(float(row['Ra']))`). No existe el
equivalente para Rz en este repositorio.

Uso:
    python ensemble_incertidumbre.py                # rutas por defecto
    python ensemble_incertidumbre.py --salida <dir>
"""
import argparse
import io
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

GRAIN_CATEGORIES = [40, 80, 120, 180]
NUM_GRAIN_CATEGORIES = len(GRAIN_CATEGORIES)
N_FOLDS = 6

_DATA_SUBPATHS = [
    os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6'),
    'fotos_recortes',
    os.path.join('data', 'fotos_recortes'),
]
_MODELOS_DIR_SUBPATHS = [
    os.path.join('resultados_entrenamiento', 'modelos'),
    os.path.join('Entrenar modelo', 'MULTI_VC_RA'),
]


def _encontrar(subpaths, valido):
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in subpaths:
            candidato = carpeta / sub
            if candidato.exists() and valido(candidato):
                return carpeta, candidato
    return inicio, None


def parsear_argumentos():
    raiz, data_dir = _encontrar(
        _DATA_SUBPATHS, lambda c: c.is_dir() and (c / 'DATASET.xlsx').is_file())
    if data_dir is None:
        data_dir = raiz / _DATA_SUBPATHS[0]
    _, modelos_dir = _encontrar(
        _MODELOS_DIR_SUBPATHS,
        lambda c: c.is_dir() and (c / 'modelo_multimodal_VC_fold1.keras').is_file())
    if modelos_dir is None:
        modelos_dir = raiz / _MODELOS_DIR_SUBPATHS[1]
    salida = Path(__file__).resolve().parent.parent / 'resultados' / 'r14_incertidumbre'

    p = argparse.ArgumentParser(description='Incertidumbre por ensemble de los 6 folds (Ra).')
    p.add_argument('--modelos-dir', default=str(modelos_dir),
                    help='Carpeta con los 6 modelos de pliegue')
    p.add_argument('--patron', default='modelo_multimodal_VC_fold{i}.keras',
                    help="Patron del nombre de archivo, con {i} por el numero de pliegue. "
                         "Para la arquitectura A: 'modelo_A_fold{i}.keras'")
    p.add_argument('--imagenes', default=str(data_dir / 'Validacion'))
    p.add_argument('--excel', default=str(data_dir / 'DATASET.xlsx'))
    p.add_argument('--salida', default=str(salida))
    return p.parse_args()


def extract_grain(filename):
    import re
    for g in GRAIN_CATEGORIES:
        if f'G{g}' in str(filename).upper():
            return g
    for num in re.findall(r'\d+', str(filename)):
        if int(num) in GRAIN_CATEGORIES:
            return int(num)
    print(f"Advertencia: grano no encontrado en '{filename}', usando 80 por defecto")
    return 80


def cargar_imagen(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (130, 390), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img


def main():
    args = parsear_argumentos()
    os.makedirs(args.salida, exist_ok=True)

    rutas_modelos = [
        os.path.join(args.modelos_dir, args.patron.format(i=i))
        for i in range(1, N_FOLDS + 1)
    ]
    for etiqueta, ruta in [('Imágenes', args.imagenes), ('Excel', args.excel)]:
        if not os.path.exists(ruta):
            sys.exit(f"ERROR: no existe la ruta de {etiqueta}: {ruta}")
    faltantes = [r for r in rutas_modelos if not os.path.exists(r)]
    if faltantes:
        sys.exit("ERROR: faltan modelos de pliegue:\n  " + "\n  ".join(faltantes))

    print(f"Cargando {N_FOLDS} modelos de pliegue desde: {args.modelos_dir}")
    modelos = []
    for ruta in rutas_modelos:
        m = load_model(ruta, compile=False)
        modelos.append(m)
        print(f"  {os.path.basename(ruta)} cargado.")

    df_gt = pd.read_excel(args.excel, sheet_name='Validacion')
    df_gt.columns = df_gt.columns.str.strip()
    df_gt['nombre_lower'] = df_gt['nombre_imagen'].str.lower()
    print(f"Ground truth cargado: {len(df_gt)} filas")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in os.listdir(args.imagenes)
                   if os.path.splitext(f)[1].lower() in image_extensions]
    print(f"Imágenes encontradas: {len(image_files)}")

    filas = []
    for img_file in image_files:
        base = os.path.splitext(img_file)[0].lower()
        matches = df_gt[df_gt['nombre_lower'] == base]
        if matches.empty:
            matches = df_gt[df_gt['nombre_lower'].str.contains(base, na=False)]
        if matches.empty:
            print(f"Sin coincidencia: {img_file}")
            continue

        ra_real = float(matches.iloc[0]['Ra'])
        img = cargar_imagen(os.path.join(args.imagenes, img_file))
        if img is None:
            print(f"No se pudo leer: {img_file}")
            continue

        grain = extract_grain(img_file)
        grain_oh = np.zeros((1, NUM_GRAIN_CATEGORIES), dtype=np.float32)
        grain_oh[0, GRAIN_CATEGORIES.index(grain)] = 1.0

        preds = [
            float(m.predict({'image_input': img, 'grain_input': grain_oh}, verbose=0)[0][0])
            for m in modelos
        ]
        preds = np.array(preds)
        media = float(preds.mean())
        std = float(preds.std(ddof=1))

        fila = {
            'archivo': img_file,
            'grano': grain,
            'Ra_real': ra_real,
            'Ra_media_ensemble': media,
            'Ra_std_ensemble': std,
            'error_abs_media': abs(ra_real - media),
            'dentro_1sigma': abs(ra_real - media) <= std,
            'dentro_2sigma': abs(ra_real - media) <= 2 * std,
        }
        for i, p in enumerate(preds, 1):
            fila[f'fold{i}_pred'] = p
        filas.append(fila)

    if not filas:
        sys.exit("ERROR: no se procesó ninguna imagen.")

    df = pd.DataFrame(filas)
    print(f"\nImágenes procesadas: {len(df)}")

    y_true = df['Ra_real'].values
    y_pred_media = df['Ra_media_ensemble'].values
    r2 = r2_score(y_true, y_pred_media)
    mae = mean_absolute_error(y_true, y_pred_media)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_media))
    std_medio = df['Ra_std_ensemble'].mean()
    cobertura_1s = df['dentro_1sigma'].mean() * 100
    cobertura_2s = df['dentro_2sigma'].mean() * 100

    print(f"\n=== ENSEMBLE DE 6 PLIEGUES SOBRE TEST (Ra) ===")
    print(f"R² (media del ensemble) : {r2:.4f}")
    print(f"MAE                     : {mae:.4f} µm")
    print(f"RMSE                    : {rmse:.4f} µm")
    print(f"SD entre folds, media   : {std_medio:.4f} µm")
    print(f"Cobertura dentro ±1 SD  : {cobertura_1s:.1f}%  (referencia normal: ~68%)")
    print(f"Cobertura dentro ±2 SD  : {cobertura_2s:.1f}%  (referencia normal: ~95%)")

    print("\nSD medio del ensemble por grano:")
    for g in sorted(df['grano'].unique()):
        sub = df[df['grano'] == g]
        print(f"  P{g}: SD medio={sub['Ra_std_ensemble'].mean():.4f} µm  (n={len(sub)})")

    plt.figure(figsize=(8, 6.5))
    orden = np.argsort(y_true)
    plt.errorbar(y_true[orden], y_pred_media[orden], yerr=df['Ra_std_ensemble'].values[orden],
                 fmt='o', alpha=0.55, ecolor='gray', elinewidth=1, capsize=2, markersize=5,
                 color='steelblue', label='Media ± SD del ensemble (6 folds)')
    mn, mx = min(y_true.min(), y_pred_media.min()), max(y_true.max(), y_pred_media.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='1:1')
    plt.xlabel('Ra Medido (µm)', fontsize=13)
    plt.ylabel('Ra Estimado (µm)', fontsize=13)
    plt.title(f'Incertidumbre por ensemble de pliegues — Ra\n'
              f'R² = {r2:.4f}  |  SD medio = {std_medio:.3f} µm  |  '
              f'cobertura ±1SD = {cobertura_1s:.0f}%, ±2SD = {cobertura_2s:.0f}%',
              fontsize=12)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.salida, 'R14_incertidumbre_ensemble_Ra.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nFigura guardada en: {fig_path}")

    metrics_df = pd.DataFrame({
        'Métrica': ['R² (media ensemble)', 'MAE (µm)', 'RMSE (µm)', 'SD medio entre folds (µm)',
                    'Cobertura ±1 SD (%)', 'Cobertura ±2 SD (%)'],
        'Valor': [r2, mae, rmse, std_medio, cobertura_1s, cobertura_2s],
    })
    out_file = os.path.join(args.salida, 'R14_ensemble_incertidumbre_Ra.xlsx')
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicciones')
        metrics_df.to_excel(writer, index=False, sheet_name='Métricas')
    print(f"Datos guardados en: {out_file}")
    print("Listo.")


if __name__ == '__main__':
    main()
