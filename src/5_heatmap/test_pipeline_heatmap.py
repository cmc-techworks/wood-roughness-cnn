"""Prueba headless del pipeline de interfaz_heatmap con un modelo y una panorámica.

Ejecuta la misma cadena que la GUI (cargar imagen -> tiles -> inferencia
multimodal -> mapa) sin abrir ventanas, e imprime diagnósticos para verificar
que el modelo multimodal recibe correctamente {image_input, grain_input} y que
los valores de Ra estimados son razonables.

La imagen debe ser una PANORÁMICA, no un recorte de 390x130: con un solo tile
la comprobación de variación espacial no significa nada.

Uso:
    python test_pipeline_heatmap.py --imagen ruta/a/panoramica.png --grano 40
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Importar las funciones del pipeline real
from interfaz_heatmap import (
    load_and_prepare_image,
    split_into_tiles,
    predict_ra_values,
    create_heatmap,
    GRAIN_CATEGORIES,
)
from tensorflow.keras.models import load_model

_MODELO_SUBPATHS = [
    os.path.join('resultados_entrenamiento', 'modelos', 'modelo_final_multimodal_VC.keras'),
    os.path.join('Entrenar modelo', 'MULTI_VC_RA', 'modelo_final_multimodal_VC.keras'),
]


def _encontrar_modelo():
    """Sube por los padres buscando el modelo final en cualquiera de los layouts conocidos."""
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in _MODELO_SUBPATHS:
            candidato = carpeta / sub
            if candidato.is_file():
                return str(candidato)
    return None


def parsear_argumentos():
    p = argparse.ArgumentParser(description='Prueba headless del pipeline de heatmap.')
    p.add_argument('--imagen', required=True,
                   help='Panorámica a analizar (no un recorte de 390x130).')
    p.add_argument('--grano', type=int, default=40, choices=GRAIN_CATEGORIES,
                   help='Grano de lija de la imagen.')
    p.add_argument('--modelo', default=None,
                   help='Modelo .keras. Por defecto se busca subiendo por los padres.')
    return p.parse_args()


def main():
    args = parsear_argumentos()
    print("=" * 70)
    print("PRUEBA DEL PIPELINE MULTIMODAL")
    print("=" * 70)

    MODEL_PATH = args.modelo or _encontrar_modelo()
    if not MODEL_PATH:
        sys.exit("ERROR: no se encontró 'modelo_final_multimodal_VC.keras' subiendo por los "
                 "padres. Indicarlo con --modelo.")
    IMAGE_PATH = args.imagen
    GRAIN = args.grano

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"ERROR: no existe el modelo: {MODEL_PATH}")
    if not os.path.exists(IMAGE_PATH):
        sys.exit(f"ERROR: no existe la imagen: {IMAGE_PATH}")

    # 1. Cargar modelo
    print("\n[1] Cargando modelo...")
    print(f"    {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"    Entradas del modelo: {len(model.inputs)}")
    for inp in model.inputs:
        print(f"      - {inp.name.split(':')[0]}  shape={tuple(inp.shape)}")
    print(f"    Salida: shape={tuple(model.outputs[0].shape)}")

    # 2. Cargar imagen
    print("\n[2] Cargando imagen...")
    img = load_and_prepare_image(IMAGE_PATH)
    print(f"    Imagen (alto x ancho): {img.shape[0]} x {img.shape[1]}")

    # 3. Dividir en tiles (390 alto x 130 ancho, geometría del modelo)
    print("\n[3] Dividiendo en tiles...")
    tiles, n_h, n_w = split_into_tiles(img)
    print(f"    Tiles: {len(tiles)}  (cuadrícula {n_h} x {n_w})")
    print(f"    Forma de un tile (alto x ancho): {tiles.shape[1]} x {tiles.shape[2]}")

    # 4. Inferencia por lotes
    print(f"\n[4] Infiriendo Ra (grano G{GRAIN})...")
    batch_size = 32
    ra_values = []
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size]
        ra_values.extend(predict_ra_values(model, batch, grain=GRAIN))
        pct = min(100, int((i + len(batch)) / len(tiles) * 100))
        print(f"\r    Progreso: {pct}%", end="", flush=True)
    print()
    ra_values = np.array(ra_values, dtype=float)

    # 5. Mapa y estadísticas
    heatmap = create_heatmap(ra_values, n_h, n_w)
    print("\n[5] RESULTADOS")
    print("-" * 70)
    print(f"    Tiles evaluados : {ra_values.size}")
    print(f"    Ra mínimo       : {ra_values.min():.4f} µm")
    print(f"    Ra máximo       : {ra_values.max():.4f} µm")
    print(f"    Ra promedio     : {ra_values.mean():.4f} µm")
    print(f"    Ra mediana      : {np.median(ra_values):.4f} µm")
    print(f"    Ra desv. est.   : {ra_values.std():.4f} µm")
    print(f"    Forma del mapa  : {heatmap.shape}")
    print(f"    Esquina sup-izq del mapa:\n{np.round(heatmap[:3, :5], 2)}")

    # 6. Chequeo de plausibilidad: Ra de madera lijada ~ [1, 30] µm aprox.
    plausible = np.isfinite(ra_values).all() and ra_values.min() > 0 and ra_values.max() < 100
    varied = ra_values.std() > 1e-3  # que no sea una constante (modelo realmente usa la imagen)
    print("-" * 70)
    print(f"    Valores finitos y en rango plausible (0-100 um)? {plausible}")
    print(f"    Hay variacion espacial (std > 0.001)?            {varied}")

    ok = plausible and varied
    print("\n" + ("=" * 70))
    print("RESULTADO: " + ("RELEVANTE [OK]" if ok else "REVISAR [X]"))
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
