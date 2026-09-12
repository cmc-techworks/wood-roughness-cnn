#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reentrenamiento limpio del modelo final, UNA sola etapa.

Corrige el defecto de Red_multimodal_VC.py seccion 8+9: alli 'final_model' se crea una
vez, se entrena 40 epocas, y despues el MISMO objeto se recompila (Adam nuevo, momentos
en cero) y se entrena hasta 40 epocas mas -- no es reentrenamiento desde cero, es una
segunda etapa sobre un modelo ya convergido cuyo optimizador se reinicia. Bug de scope,
no decision de diseno: 'create_model()' esta definida y nunca se llama (linea 546).

Este script entrena el modelo final UNA sola vez, misma arquitectura (D, Flatten, ~12.2M
parametros), mismos datos (hoja 'sin_outliers', 1030 filas, las 6 probetas completas),
mismo protocolo (SEED=10, Adam, MSE, MAE, batch 32, 40 epocas, EarlyStopping+ReduceLROnPlateau
monitoreando 'loss' porque no hay particion de validacion al usar el 100% de los datos --
la ausencia de senal de validacion es una limitacion real del disegno, no algo que este
script pueda resolver, se declara en R1.11).

Uso:
    python entrenar_final_limpio.py                 # 40 epocas, seed 10
    python entrenar_final_limpio.py --epochs 40 --seed 10

Sale a 3_entrenamiento/resultados_final_limpio/ -- nunca escribe en
'Entrenar modelo/MULTI_VC_RA/' ni en 'resultados_entrenamiento/' (esos son el registro de
lo publicado). Evaluar el resultado con:
    python ../4_validacion/validar_headless.py --param Ra --modelo resultados_final_limpio/modelo_final_limpio.keras
"""
import argparse
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.utils import img_to_array, load_img, to_categorical

IMG_HEIGHT, IMG_WIDTH = 390, 130
BATCH_SIZE = 32
GRAIN_CATEGORIES = [40, 80, 120, 180]
NUM_GRAIN_CATEGORIES = len(GRAIN_CATEGORIES)

_DATA_SUBPATHS = [os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6'), 'fotos_recortes', os.path.join('data', 'fotos_recortes')]


def encontrar_data_dir():
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in _DATA_SUBPATHS:
            c = carpeta / sub
            if c.is_dir() and (c / 'DATASET.xlsx').is_file():
                return str(c)
    raise SystemExit("ERROR: no se encontro ninguna carpeta de datos con DATASET.xlsx")


_CACHE_IMAGENES = {}


class MultimodalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, img_dir, batch_size=32, img_size=(390, 130), shuffle=True, param='Ra'):
        super().__init__()
        self.df = df.copy()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.param = param
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = self.df.iloc[idx]
        x1, x2, y = [], [], []
        for _, row in batch.iterrows():
            ruta = os.path.join(self.img_dir, row['nombre_imagen'])
            clave = (ruta, self.img_size)
            img_u8 = _CACHE_IMAGENES.get(clave)
            if img_u8 is None:
                img_u8 = img_to_array(load_img(ruta, color_mode='grayscale',
                                               target_size=self.img_size)).astype(np.uint8)
                _CACHE_IMAGENES[clave] = img_u8
            x1.append(img_u8.astype(np.float32) / 255.0)
            x2.append(to_categorical(GRAIN_CATEGORIES.index(int(row['Grano'])),
                                     num_classes=NUM_GRAIN_CATEGORIES))
            y.append(float(row[self.param]))
        return ({'image_input': np.array(x1, dtype=np.float32),
                 'grain_input': np.array(x2, dtype=np.float32)},
                np.array(y, dtype=np.float32))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)


def construir_modelo(arch='D'):
    """Misma fabrica que barrido_arquitecturas.py: solo cambia la reduccion previa a la densa.

        A  GlobalAveragePooling2D            ~25.7 k parametros  (la que se publica)
        B  Flatten + 2 MaxPooling2D extra    ~0.71 M
        C  Flatten + 1 MaxPooling2D extra    ~2.97 M
        D  Flatten directo                   ~12.21 M  (la original)
    """
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    if arch == 'A':
        x = layers.GlobalAveragePooling2D()(x)
    elif arch == 'B':
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
    elif arch == 'C':
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
    elif arch == 'D':
        x = layers.Flatten()(x)
    else:
        raise ValueError(f"arquitectura desconocida: {arch}")

    x = layers.Dense(64, activation='relu')(x)

    input_grain = Input(shape=(NUM_GRAIN_CATEGORIES,), name='grain_input')
    g = layers.Dense(16, activation='relu')(input_grain)

    combined = layers.Concatenate()([x, g])
    z = layers.Dense(32, activation='relu')(combined)
    output = layers.Dense(1, activation='linear', name='output')(z)
    return Model(inputs={'image_input': input_img, 'grain_input': input_grain}, outputs=output)


def main():
    ap = argparse.ArgumentParser(description='Reentrenamiento limpio, una sola etapa, del modelo final.')
    ap.add_argument('--arch', choices=['A', 'B', 'C', 'D'], default='A',
                    help='Arquitectura. A es la que se publica desde 2026-08-17.')
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()

    # keras.utils.set_random_seed siembra Python, NumPy y TensorFlow a la vez. Verificado el
    # 2026-08-17 con diagnostico_determinismo.py: np.random.seed + tf.random.set_seed NO fija
    # la inicializacion de pesos en Keras 3, y esa era la fuente de toda la variabilidad
    # entre corridas. Con esto, dos corridas con la misma semilla son bit a bit identicas.
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = encontrar_data_dir()
    excel = os.path.join(data_dir, 'DATASET.xlsx')
    df = pd.read_excel(excel, sheet_name='sin_outliers')

    # Guardado ordenado: CODE/resultados/<arch>_oficial/<param>/ -- ver PLAN_ACCION_2026-08-17.md
    out = Path(__file__).resolve().parent.parent / 'resultados' / f'{args.arch}_oficial' / args.param
    for sub in ['modelos', 'metricas']:
        (out / sub).mkdir(parents=True, exist_ok=True)

    config = {
        'arquitectura': args.arch,
        'parametro': args.param,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': BATCH_SIZE,
        'muestras': int(len(df)),
        'probetas': sorted(int(p) for p in df['id_probeta'].unique()),
        'etapas': 1,
        'hoja': 'sin_outliers',
        'comando': ' '.join(sys.argv),
        'nota': "correccion de Red_multimodal_VC.py secciones 8-9: una sola etapa de entrenamiento, sin recompilar el mismo objeto",
        'tensorflow': tf.__version__,
        'keras': tf.keras.__version__,
        'python': platform.python_version(),
        'plataforma': platform.platform(),
        'gpus': [d.name for d in tf.config.list_physical_devices('GPU')],
        'data_dir': data_dir,
    }
    print("=" * 78)
    for k, v in config.items():
        print(f"  {k:>16}: {v}")
    print("=" * 78)

    gen_train = MultimodalDataGenerator(df, data_dir, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH),
                                        shuffle=True, param=args.param)

    model = construir_modelo(args.arch)
    config['parametros'] = int(model.count_params())
    (out / f'config_corrida_seed{args.seed}.json').write_text(
        json.dumps(config, indent=2), encoding='utf8')
    print(f"Parametros: {model.count_params():,}")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ]

    print("\nEntrenando modelo final (una sola etapa)...")
    t0 = time.time()
    hist = model.fit(gen_train, epochs=args.epochs, callbacks=callbacks, verbose=2)
    dt = time.time() - t0
    print(f"\nTiempo de entrenamiento: {dt/60:.1f} min ({dt/args.epochs:.1f} s/epoca)")
    print(f"Epocas efectivas: {len(hist.history['loss'])} (EarlyStopping patience=10 sobre 'loss')")
    print(f"Perdida final (MSE, train): {hist.history['loss'][-1]:.6f}")
    print(f"MAE final (train): {hist.history['mae'][-1]:.6f}")

    # El nombre SIEMPRE lleva la semilla: dos corridas con semillas distintas no pueden
    # pisarse entre si. Se perdio un modelo por esto el 2026-08-17.
    nombre = f'modelo_{args.arch}_{args.param}_final_seed{args.seed}.keras'
    destino = out / 'modelos' / nombre
    if destino.exists():
        raise SystemExit(f"ERROR: ya existe {destino}. Borrar a mano si se quiere reemplazar.")
    model.save(destino)
    pd.DataFrame(hist.history).rename_axis('epoca').to_excel(
        out / 'metricas' / f'historial_por_epoca_seed{args.seed}.xlsx')
    (out / 'README.md').write_text(
        f"# Modelo final oficial — arquitectura {args.arch}, {args.param}\n\n"
        f"- Corrida: `{' '.join(sys.argv)}`\n"
        f"- {config['parametros']:,} parametros · seed {args.seed} · {args.epochs} epocas · "
        f"{len(df)} muestras (hoja `sin_outliers`, 6 probetas)\n"
        f"- Una sola etapa, sin el doble compile de Red_multimodal_VC.py\n"
        f"- Alimenta: R1.3 (arquitectura), Tabla 6, Fig. 15 y 17, abstract\n",
        encoding='utf8')

    print(f"\nModelo guardado en: {destino}")
    print("Evaluar sobre el test independiente con:")
    print(f"  python ../4_validacion/validar_headless.py --param {args.param} "
          f"--modelo \"{destino}\"")


if __name__ == '__main__':
    main()
