#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incertidumbre por Monte Carlo sobre replicas bootstrap del conjunto de entrenamiento.

Amplia la cuantificacion de incertidumbre mas alla de MAE y RMSE, hasta intervalos
de confianza aplicables a los mapas espaciales.

Metodo: replicas con DATOS distintos, no con semillas distintas. Se entrenan N modelos sobre remuestreos bootstrap del conjunto de
entrenamiento; los percentiles de sus predicciones dan la banda.

NIVEL DE REMUESTREO -- decision metodologica, no cosmetica
---------------------------------------------------------
La descomposicion de la brecha CV-test (CODE/resultados/A_oficial/Ra/brecha_cv_test/)
muestra que la identidad de la probeta explica el 94% de la variabilidad, y que la SD
entre probetas es 5.6 veces la SD entre modelos. Un bootstrap que remuestree solo
subzonas DENTRO de cada probeta ignora la fuente dominante y produce una banda
optimista, que ademas contradice el propio analisis del articulo.

Por eso el valor por defecto es 'jerarquico': se remuestrean primero las probetas con
reemplazo y despues las filas dentro de cada probeta elegida. Es el bootstrap estandar
para datos agrupados. Las otras dos opciones quedan disponibles para comparar:

    --nivel jerarquico  probetas con reemplazo, luego filas dentro de cada una  (defecto)
    --nivel probeta     solo probetas con reemplazo, tomando todas sus filas
    --nivel subzona     filas con reemplazo ignorando la probeta  (subestima la banda)

Uso:
    python montecarlo_incertidumbre.py --arch A --param Ra --n-replicas 20
    python montecarlo_incertidumbre.py --arch A --param Ra --n-replicas 20 --nivel subzona

Guarda SIEMPRE los indices de cada replica: sin ellos el resultado no es reproducible
ni auditable, y habria que reentrenar las 20 desde cero.
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
NUM_GRAIN = len(GRAIN_CATEGORIES)
_DATA_SUBPATHS = [os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6'), 'fotos_recortes', os.path.join('data', 'fotos_recortes')]
_CACHE = {}


def encontrar_data_dir():
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in _DATA_SUBPATHS:
            c = carpeta / sub
            if c.is_dir() and (c / 'DATASET.xlsx').is_file():
                return str(c)
    raise SystemExit('ERROR: no se encontro DATASET.xlsx')


class Gen(tf.keras.utils.Sequence):
    def __init__(self, df, img_dir, batch_size=32, shuffle=True, param='Ra'):
        super().__init__()
        self.df, self.img_dir = df.reset_index(drop=True), img_dir
        self.batch_size, self.shuffle, self.param = batch_size, shuffle, param
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, i):
        idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        b = self.df.iloc[idx]
        x1, x2, y = [], [], []
        for _, row in b.iterrows():
            ruta = os.path.join(self.img_dir, row['nombre_imagen'])
            img = _CACHE.get(ruta)
            if img is None:
                img = img_to_array(load_img(ruta, color_mode='grayscale',
                                            target_size=(IMG_HEIGHT, IMG_WIDTH))).astype(np.uint8)
                _CACHE[ruta] = img
            x1.append(img.astype(np.float32) / 255.0)
            x2.append(to_categorical(GRAIN_CATEGORIES.index(int(row['Grano'])), num_classes=NUM_GRAIN))
            y.append(float(row[self.param]))
        return ({'image_input': np.array(x1, np.float32),
                 'grain_input': np.array(x2, np.float32)}, np.array(y, np.float32))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)


def construir_modelo(arch='A'):
    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    if arch == 'A':
        x = layers.GlobalAveragePooling2D()(x)
    elif arch == 'B':
        x = layers.MaxPooling2D((2, 2))(x); x = layers.MaxPooling2D((2, 2))(x); x = layers.Flatten()(x)
    elif arch == 'C':
        x = layers.MaxPooling2D((2, 2))(x); x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    ing = Input(shape=(NUM_GRAIN,), name='grain_input')
    g = layers.Dense(16, activation='relu')(ing)
    z = layers.Dense(32, activation='relu')(layers.Concatenate()([x, g]))
    return Model(inputs={'image_input': inp, 'grain_input': ing},
                 outputs=layers.Dense(1, activation='linear', name='output')(z))


def remuestrear(df, nivel, rng):
    """Devuelve los indices posicionales de una replica bootstrap."""
    if nivel == 'subzona':
        return rng.integers(0, len(df), len(df))
    probetas = df['id_probeta'].unique()
    elegidas = rng.choice(probetas, size=len(probetas), replace=True)
    idx = []
    for p in elegidas:
        pos = np.flatnonzero((df['id_probeta'] == p).to_numpy())
        idx.extend(pos if nivel == 'probeta' else rng.choice(pos, size=len(pos), replace=True))
    return np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['A', 'B', 'C', 'D'], default='A')
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    ap.add_argument('--n-replicas', type=int, default=20)
    ap.add_argument('--nivel', choices=['jerarquico', 'probeta', 'subzona'], default='jerarquico')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--salida', default=None)
    args = ap.parse_args()

    data_dir = encontrar_data_dir()
    df = pd.read_excel(os.path.join(data_dir, 'DATASET.xlsx'), sheet_name='sin_outliers')
    test = pd.read_excel(os.path.join(data_dir, 'DATASET.xlsx'), sheet_name='Validacion')
    test = test[test['nombre_imagen'].apply(
        lambda n: os.path.isfile(os.path.join(data_dir, 'Validacion', str(n))))].reset_index(drop=True)

    out = Path(args.salida) if args.salida else (
        Path(__file__).resolve().parent.parent / 'resultados' /
        f'{args.arch}_oficial' / args.param / 'montecarlo')
    for sub in ['modelos', 'indices_bootstrap']:
        (out / sub).mkdir(parents=True, exist_ok=True)

    cfg = {'arquitectura': args.arch, 'parametro': args.param, 'n_replicas': args.n_replicas,
           'nivel_remuestreo': args.nivel, 'epochs': args.epochs, 'seed': args.seed,
           'muestras_entrenamiento': int(len(df)), 'muestras_test': int(len(test)),
           'comando': ' '.join(sys.argv), 'tensorflow': tf.__version__,
           'keras': tf.keras.__version__, 'python': platform.python_version(),
           'justificacion_nivel': ('la identidad de la probeta explica el 94% de la brecha CV-test '
                                   'y la SD entre probetas es 5.6x la SD entre modelos, de modo que '
                                   'un remuestreo que ignore el agrupamiento subestima la banda')}
    (out / 'config_corrida.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf8')
    print('=' * 74)
    for k, v in cfg.items():
        print(f'  {k:>24}: {v}')
    print('=' * 74)

    gen_test = Gen(test, os.path.join(data_dir, 'Validacion'), BATCH_SIZE, shuffle=False, param=args.param)
    preds = np.zeros((args.n_replicas, len(test)), dtype=np.float32)
    t_ini = time.time()

    for r in range(args.n_replicas):
        rng = np.random.default_rng(args.seed * 1000 + r)
        # Ver diagnostico_determinismo.py (2026-08-17): keras.utils.set_random_seed es la que
        # fija tambien la inicializacion de pesos; el patron anterior no lo hacia.
        tf.keras.utils.set_random_seed(args.seed * 1000 + r)

        idx = remuestrear(df, args.nivel, rng)
        pd.DataFrame({'pos': idx, 'nombre_imagen': df.iloc[idx]['nombre_imagen'].values,
                      'id_probeta': df.iloc[idx]['id_probeta'].values}).to_csv(
            out / 'indices_bootstrap' / f'replica_{r+1:02d}.csv', index=False)

        rep = df.iloc[idx]
        n_prob = rep['id_probeta'].nunique()
        print(f"\n{'='*60}\nReplica {r+1}/{args.n_replicas} | {len(rep)} filas | "
              f"{n_prob} probetas distintas: {sorted(rep['id_probeta'].unique())}")

        modelo = construir_modelo(args.arch)
        modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
        t0 = time.time()
        modelo.fit(Gen(rep, data_dir, BATCH_SIZE, shuffle=True, param=args.param),
                   epochs=args.epochs, verbose=0,
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       monitor='loss', patience=10, restore_best_weights=True)])
        preds[r] = modelo.predict(gen_test, verbose=0).flatten()
        modelo.save(out / 'modelos' / f'modelo_{args.arch}_{args.param}_bootstrap{r+1:02d}.keras')
        print(f'  entrenada en {(time.time()-t0)/60:.1f} min | '
              f'prediccion media sobre test {preds[r].mean():.3f}')

    real = test[args.param].to_numpy(dtype=np.float32)
    res = pd.DataFrame({'archivo': test['nombre_imagen'], 'grano': test['Grano'], f'{args.param}_real': real})
    res[f'{args.param}_media'] = preds.mean(axis=0)
    res[f'{args.param}_sd'] = preds.std(axis=0, ddof=1)
    for q, nom in [(2.5, 'p02_5'), (5, 'p05'), (50, 'p50'), (95, 'p95'), (97.5, 'p97_5')]:
        res[nom] = np.percentile(preds, q, axis=0)
    for i in range(args.n_replicas):
        res[f'rep{i+1:02d}'] = preds[i]

    cob90 = float(((real >= res['p05']) & (real <= res['p95'])).mean())
    cob95 = float(((real >= res['p02_5']) & (real <= res['p97_5'])).mean())
    ancho90 = float((res['p95'] - res['p05']).mean())
    mae = float(np.abs(real - res[f'{args.param}_media']).mean())
    ss = 1 - ((real - res[f'{args.param}_media']) ** 2).sum() / ((real - real.mean()) ** 2).sum()

    print('\n' + '=' * 74)
    print(f'  MONTE CARLO ({args.n_replicas} replicas, nivel {args.nivel})')
    print('=' * 74)
    print(f'  R2 de la media          : {ss:.4f}')
    print(f'  MAE de la media         : {mae:.4f}')
    print(f'  Cobertura del 90% (p05-p95)   : {cob90:.1%}   ancho medio {ancho90:.3f}')
    print(f'  Cobertura del 95% (p2.5-p97.5): {cob95:.1%}')
    print(f'  Tiempo total: {(time.time()-t_ini)/60:.1f} min')

    resumen = pd.DataFrame([{'n_replicas': args.n_replicas, 'nivel': args.nivel, 'R2_media': ss,
                             'MAE_media': mae, 'cobertura_90': cob90, 'cobertura_95': cob95,
                             'ancho_medio_90': ancho90}])
    with pd.ExcelWriter(out / f'R14_montecarlo_{args.param}.xlsx') as w:
        res.to_excel(w, sheet_name='Predicciones', index=False)
        resumen.to_excel(w, sheet_name='Resumen', index=False)
        res.groupby('grano').agg(
            n=('grano', 'size'), MAE=(f'{args.param}_media', lambda s: np.nan),
            ancho90=('p95', 'mean')).to_excel(w, sheet_name='Por_grano')

    (out / 'README.md').write_text(
        f'# Monte Carlo — {args.n_replicas} replicas bootstrap, arquitectura {args.arch}, {args.param}\n\n'
        f'- Nivel de remuestreo: **{args.nivel}**. {cfg["justificacion_nivel"]}\n'
        f'- Corrida: `{" ".join(sys.argv)}`\n'
        f'- Cobertura empirica del intervalo del 90%: {cob90:.1%} (ancho medio {ancho90:.3f})\n'
        f'- Los indices de cada replica estan en `indices_bootstrap/`: sin ellos no es reproducible\n'
        f'- Alimenta: R1.4 (intervalos de confianza para los mapas espaciales)\n', encoding='utf8')
    print(f'\n  Resultados en: {out}')


if __name__ == '__main__':
    main()
