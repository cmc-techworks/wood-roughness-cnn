#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validación NO interactiva del modelo multimodal (sin diálogos tkinter).

Equivalente headless de `Validar_modelo_multimodal_vc.py` (raíz): carga el modelo
final, evalúa sobre la carpeta de imágenes de validación y guarda métricas + gráfico R².
Todas las rutas se resuelven automáticamente respecto a la raíz del repositorio;
cualquier ruta puede sobreescribirse por línea de comandos.

Uso:
    python validar_headless.py                     # Ra, rutas por defecto
    python validar_headless.py --param Rz
    python validar_headless.py --modelo "..\\MODELO PROBADO TESIS\\modelo_final_multimodal_VC.keras"
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
import seaborn as sns
import cv2
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from tensorflow.keras.models import load_model

GRAIN_CATEGORIES = [40, 80, 120, 180]
NUM_GRAIN_CATEGORIES = len(GRAIN_CATEGORIES)
# Se aceptan dos layouts: el original de la tesis (Fotos/Recortes15x15mm/recortes_x6)
# y el de esta carpeta de revisión (fotos_recortes/), que trae su propia copia de
# DATASET.xlsx junto a las imágenes. Mismo criterio que Red_multimodal_VC_con_outliers.py.
_DATA_SUBPATHS = [
    os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6'),
    'fotos_recortes',
    os.path.join('data', 'fotos_recortes'),
]
_MODELO_SUBPATHS = [
    os.path.join('resultados_entrenamiento', 'modelos', 'modelo_final_multimodal_VC.keras'),
    os.path.join('Entrenar modelo', 'MULTI_VC_RA', 'modelo_final_multimodal_VC.keras'),
]


def _encontrar(subpaths, valido):
    """Sube por los padres hasta dar con el primer subpath que exista y pase `valido`.

    Devuelve (raiz, ruta) o (carpeta_del_script, None) si no encuentra nada.
    """
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in subpaths:
            candidato = carpeta / sub
            if candidato.exists() and valido(candidato):
                return carpeta, candidato
    return inicio, None


# Arquitectura publicada por parametro (B para Ra, C para Rz) y semilla de referencia.
_ARQ_PUBLICADA = {'Ra': 'B', 'Rz': 'C'}
_SEED_REF = 13


def _modelo_publicado(raiz, param):
    """Pesos publicados del parametro pedido, con respaldo a los layouts antiguos."""
    arch = _ARQ_PUBLICADA[param]
    nombre = 'modelo_' + arch + '_' + param + '_final_seed' + str(_SEED_REF) + '.keras'
    for c in [raiz / 'models' / (param + '_arch_' + arch) / nombre,
              raiz / 'results' / (arch + '_oficial') / param / 'modelos' / nombre]:
        if c.is_file():
            return c
    _, m = _encontrar(_MODELO_SUBPATHS, lambda c: c.is_file())
    return m if m is not None else raiz / 'models' / (param + '_arch_' + arch) / nombre


def parsear_argumentos():
    raiz, data_dir = _encontrar(
        _DATA_SUBPATHS, lambda c: c.is_dir() and (c / 'DATASET.xlsx').is_file())
    if data_dir is None:
        data_dir = raiz / _DATA_SUBPATHS[0]   # ruta esperada, para que el error sea claro
    # Salidas junto al resto de resultados: results/ en el repositorio publicado,
    # resultados/ en el arbol de trabajo original.
    _src = Path(__file__).resolve().parent.parent
    _res = _src.parent / 'results'
    if not _res.is_dir():
        _res = _src / 'resultados'
    salida = _res / 'validacion_revision'

    p = argparse.ArgumentParser(description='Validación headless del modelo multimodal.')
    p.add_argument('--param', choices=['Ra', 'Rz'], default='Ra',
                   help='Parámetro de rugosidad a evaluar (columna del Excel).')
    p.add_argument('--modelo', default=None,
                   help='Ruta al modelo .keras. Por defecto, los pesos publicados '
                        'del parametro elegido.')
    p.add_argument('--imagenes', default=str(data_dir / 'Validacion'),
                   help='Carpeta con las imágenes del conjunto de prueba.')
    p.add_argument('--excel', default=str(data_dir / 'DATASET.xlsx'),
                   help='Excel con la hoja "Validacion" (verdad de campo).')
    p.add_argument('--salida', default=str(salida),
                   help='Carpeta donde guardar métricas y gráficos.')
    args = p.parse_args()
    if args.modelo is None:
        args.modelo = str(_modelo_publicado(raiz, args.param))
    return args


def extract_grain(filename):
    """Extrae el grano del nombre de archivo (convención del proyecto: G40, G80, ...)."""
    import re
    for g in GRAIN_CATEGORIES:
        if f'G{g}' in str(filename).upper():
            return g
    for num in re.findall(r'\d+', str(filename)):
        if int(num) in GRAIN_CATEGORIES:
            return int(num)
    print(f"Advertencia: grano no encontrado en '{filename}', usando 80 por defecto")
    return 80


def main():
    args = parsear_argumentos()
    param = args.param
    os.makedirs(args.salida, exist_ok=True)

    for etiqueta, ruta in [('Modelo', args.modelo), ('Imágenes', args.imagenes), ('Excel', args.excel)]:
        if not os.path.exists(ruta):
            sys.exit(f"ERROR: no existe la ruta de {etiqueta}: {ruta}")

    print(f"Cargando modelo: {args.modelo}")
    model = load_model(args.modelo, compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Modelo cargado.")

    df_gt = pd.read_excel(args.excel, sheet_name='Validacion')
    df_gt.columns = df_gt.columns.str.strip()
    print(f"Ground truth cargado: {len(df_gt)} filas")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in os.listdir(args.imagenes)
                   if os.path.splitext(f)[1].lower() in image_extensions]
    print(f"Imágenes encontradas: {len(image_files)}")

    df_gt['nombre_lower'] = df_gt['nombre_imagen'].str.lower()

    results = []
    for img_file in image_files:
        base = os.path.splitext(img_file)[0].lower()
        matches = df_gt[df_gt['nombre_lower'] == base]
        if matches.empty:
            matches = df_gt[df_gt['nombre_lower'].str.contains(base, na=False)]
        if matches.empty:
            print(f"Sin coincidencia: {img_file}")
            continue

        gt_row = matches.iloc[0]
        rugosidad_real = float(gt_row[param])

        img = cv2.imread(os.path.join(args.imagenes, img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"No se pudo leer: {img_file}")
            continue
        # cv2.resize usa (ancho, alto); el modelo espera (alto=390, ancho=130)
        img = cv2.resize(img, (130, 390), interpolation=cv2.INTER_AREA)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)

        grain = extract_grain(img_file)
        grain_oh = np.zeros((1, NUM_GRAIN_CATEGORIES), dtype=np.float32)
        grain_oh[0, GRAIN_CATEGORIES.index(grain)] = 1.0

        pred = float(model.predict({'image_input': img, 'grain_input': grain_oh}, verbose=0)[0][0])
        results.append({
            'archivo': img_file,
            'grano': grain,
            f'{param}_real': rugosidad_real,
            f'{param}_estimado': pred,
            'diferencia_absoluta': abs(rugosidad_real - pred),
        })

    if not results:
        sys.exit("ERROR: no se procesó ninguna imagen (revisar nombres vs. Excel).")

    print(f"\nImágenes procesadas: {len(results)}")
    df_res = pd.DataFrame(results)
    y_true = df_res[f'{param}_real'].values
    y_pred = df_res[f'{param}_estimado'].values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    print(f"\n=== MÉTRICAS FINALES ({param}) ===")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.4f} µm")
    print(f"RMSE : {rmse:.4f} µm")
    print(f"MSE  : {mse:.4f} µm²")
    print(f"MAPE : {mape:.2f}%")

    print("\nMétricas por grano:")
    for g in sorted(df_res['grano'].unique()):
        sub = df_res[df_res['grano'] == g]
        g_mae = mean_absolute_error(sub[f'{param}_real'], sub[f'{param}_estimado'])
        print(f"  P{g}: MAE={g_mae:.4f} µm  (n={len(sub)})")

    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha': 0.6})
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='1:1')
    plt.xlabel(f'{param} Medido (µm)', fontsize=13)
    plt.ylabel(f'{param} Estimado (µm)', fontsize=13)
    plt.title(f'Validación — {param}  |  R² = {r2:.4f}', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.salida, f'r2_{param}.png'), dpi=300)
    plt.close()

    metrics_df = pd.DataFrame({
        'Métrica': ['R²', 'MAE (µm)', 'RMSE (µm)', 'MSE (µm²)', 'MAPE (%)'],
        'Valor': [r2, mae, rmse, mse, mape]
    })
    out_file = os.path.join(args.salida, f'resultados_validacion_{param}.xlsx')
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Predicciones')
        metrics_df.to_excel(writer, index=False, sheet_name='Métricas')

    print(f"\nResultados guardados en: {out_file}")
    print("Listo.")


if __name__ == '__main__':
    main()
