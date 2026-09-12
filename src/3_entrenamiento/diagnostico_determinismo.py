#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Que exactamente varia entre dos corridas con la misma semilla, y que no.

Fijar la semilla en NumPy y TensorFlow no basta para reproducir un entrenamiento en esta
plataforma. Este script aisla la causa en lugar de suponerla, comprobando por separado:

  1. INICIALIZACION DE PESOS  -- construir el modelo dos veces con la misma semilla,
     comparar los pesos bit a bit. Si coinciden, la semilla SI controla la inicializacion.
  2. ORDEN DE BARAJADO        -- generar dos veces la permutacion del generador de datos.
     Si coincide, la semilla SI controla el orden en que se ven las muestras.
  3. INFERENCIA               -- misma imagen, mismo modelo, dos pasadas hacia adelante.
     Si coinciden, la prediccion es reproducible.
  4. ENTRENAMIENTO            -- dos entrenamientos cortos identicos. Si divergen mientras
     1, 2 y 3 coinciden, la causa esta en la acumulacion en punto flotante del paso de
     entrenamiento, no en la semilla.
  5. CON DETERMINISMO FORZADO -- lo mismo activando enable_op_determinism(), que obliga a
     TensorFlow a usar reducciones de orden fijo.

Uso:
    python diagnostico_determinismo.py
    python diagnostico_determinismo.py --determinista
"""
import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

ap = argparse.ArgumentParser()
ap.add_argument('--determinista', action='store_true',
                help='Activa enable_op_determinism() y desactiva oneDNN')
ap.add_argument('--seed', type=int, default=10)
ap.add_argument('--pasos', type=int, default=30, help='lotes de entrenamiento por prueba')
args = ap.parse_args()

if args.determinista:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers

if args.determinista:
    tf.config.experimental.enable_op_determinism()

IMG_H, IMG_W, NG = 390, 130, 4


def construir():
    inp = Input(shape=(IMG_H, IMG_W, 1), name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    ing = Input(shape=(NG,), name='grain_input')
    g = layers.Dense(16, activation='relu')(ing)
    z = layers.Dense(32, activation='relu')(layers.Concatenate()([x, g]))
    return Model(inputs={'image_input': inp, 'grain_input': ing},
                 outputs=layers.Dense(1, activation='linear')(z))


def sembrar(s):
    """El patron que usan HOY todos los scripts del expediente."""
    np.random.seed(s)
    tf.random.set_seed(s)


def sembrar_keras(s):
    """El patron correcto en Keras 3: siembra Python, NumPy y TensorFlow a la vez."""
    tf.keras.utils.set_random_seed(s)


def pesos(m):
    return np.concatenate([w.ravel() for w in m.get_weights()])


print('=' * 74)
print(f"  DIAGNOSTICO DE DETERMINISMO   modo: "
      f"{'FORZADO' if args.determinista else 'POR DEFECTO'}")
print(f"  TensorFlow {tf.__version__} | oneDNN={os.environ.get('TF_ENABLE_ONEDNN_OPTS', '1')} | "
      f"GPUs={len(tf.config.list_physical_devices('GPU'))}")
print(f"  hilos intra-op: {tf.config.threading.get_intra_op_parallelism_threads()} | "
      f"inter-op: {tf.config.threading.get_inter_op_parallelism_threads()}")
print('=' * 74)

# --- 1. inicializacion de pesos, con el patron actual y con el correcto ---
sembrar(args.seed); w1 = pesos(construir())
sembrar(args.seed); w2 = pesos(construir())
d_actual = np.abs(w1 - w2).max()
print(f'  1a. Init con np.random.seed + tf.random.set_seed   dif: {d_actual:.3e}   '
      f'{"IDENTICA" if d_actual == 0 else "DISTINTA  <-- causa raiz"}')

sembrar_keras(args.seed); k1 = pesos(construir())
sembrar_keras(args.seed); k2 = pesos(construir())
d_keras = np.abs(k1 - k2).max()
print(f'  1b. Init con keras.utils.set_random_seed          dif: {d_keras:.3e}   '
      f'{"IDENTICA  <-- la solucion" if d_keras == 0 else "DISTINTA"}')

# --- 2. orden de barajado ---
sembrar(args.seed); p1 = np.random.permutation(1030)
sembrar(args.seed); p2 = np.random.permutation(1030)
print(f'  2. Orden de barajado              iguales: {np.array_equal(p1, p2)}   '
      f'{"IDENTICO" if np.array_equal(p1, p2) else "DISTINTO"}')

# --- datos sinteticos deterministas para las pruebas 3 y 4 ---
rng = np.random.default_rng(0)
X = rng.random((args.pasos * 32, IMG_H, IMG_W, 1)).astype(np.float32)
G = np.eye(NG, dtype=np.float32)[rng.integers(0, NG, args.pasos * 32)]
Y = rng.random(args.pasos * 32).astype(np.float32) * 10

# --- 3. inferencia ---
sembrar(args.seed); m = construir()
a = m.predict({'image_input': X[:32], 'grain_input': G[:32]}, verbose=0)
b = m.predict({'image_input': X[:32], 'grain_input': G[:32]}, verbose=0)
d = np.abs(a - b).max()
print(f'  3. Inferencia (misma entrada)     diferencia maxima: {d:.3e}   '
      f'{"IDENTICA" if d == 0 else "DISTINTA"}')

# --- 4. entrenamiento partiendo de pesos IDENTICOS, para aislar el paso de optimizacion ---
sembrar_keras(args.seed)
w0 = construir().get_weights()
res = []
for _ in range(2):
    sembrar_keras(args.seed)
    m = construir()
    m.set_weights(w0)          # misma inicializacion exacta en ambas corridas
    m.compile(optimizer='adam', loss='mse')
    m.fit({'image_input': X, 'grain_input': G}, Y, epochs=1, batch_size=32,
          shuffle=False, verbose=0)
    res.append(pesos(m))
d_train = np.abs(res[0] - res[1]).max()
print(f'  4. Entrenamiento desde pesos identicos ({args.pasos} lotes)  dif: {d_train:.3e}   '
      f'{"IDENTICO" if d_train == 0 else "DIVERGE"}')

print('=' * 74)
print('  DIAGNOSTICO')
if d_actual > 0 and d_keras == 0:
    print('  La causa raiz es el sembrado. np.random.seed() + tf.random.set_seed() NO fija')
    print('  la inicializacion de pesos en Keras 3: los inicializadores de las capas usan su')
    print('  propio generador. keras.utils.set_random_seed() si lo fija.')
    print('  ES CORREGIBLE: basta cambiar la funcion de sembrado en los scripts.')
if d_train > 0:
    print('  Ademas, partiendo de pesos identicos el entrenamiento aun diverge: eso si es')
    print('  acumulacion en punto flotante de reducciones multihilo (la suma no es asociativa).')
    print('  Se corrige con enable_op_determinism(), a costa de velocidad. Correr con')
    print('  --determinista para medir cuanto queda.')
elif d_actual > 0:
    print('  Partiendo de pesos identicos el entrenamiento SI es reproducible: toda la')
    print('  variabilidad observada venia del sembrado, no del punto flotante.')
