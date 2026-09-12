#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablacion: arquitectura A, pero sin la rama categorica de grano abrasivo.

Comprueba si el R2 lo explica la imagen o la entrada categorica de grano. El
diagnostico de `CODE/6_figuras_articulo/diagnostico_r2_vc.py` mostro que una
tabla de consulta -- la media de Ra por grano en el conjunto de entrenamiento,
SIN mirar la imagen -- ya alcanza R2 = 0.783 en validacion cruzada, mientras que
el modelo completo (imagen + grano) da 0.761. Casi toda la R2 publicada la
explica la entrada categorica.

Este script aisla lo que falta saber: cuanto puede la imagen SOLA, sin la
muleta del grano. Se construye la arquitectura A completa (misma base
convolucional, mismo GlobalAveragePooling2D, mismas Dense(64) y Dense(32)) y se
elimina unicamente la rama de grano y la concatenacion: la salida de Dense(64)
alimenta directamente a Dense(32). Es el experimento minimo: cambia una sola
cosa respecto de `barrido_arquitecturas.py --arch A`.

Mismo protocolo que el resto del barrido: GroupKFold leave-one-probeta-out,
6 pliegues, 40 epocas, batch 32, Adam/MSE/MAE, una sola etapa.

El Excel de predicciones conserva la columna 'Grano' -- el modelo nunca la ve,
pero hace falta para poder comparar contra el piso y contra la senal
intra-grano que ya se midio para la version completa.

Uso:
    python ablacion_sin_grano.py --seed 10 --folds todas
    python ablacion_sin_grano.py --seed 20 --folds 1        # una sola, rapido

Sale a resultados_barrido/AsinGrano_<params>_seed<N>_limpio/, junto a las
carpetas A_/B_/C_/D_ del barrido original, para que las herramientas de
CODE/6_figuras_articulo/ (comun.leer_vc, etc.) la lean sin cambios.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.utils import img_to_array, load_img

IMG_HEIGHT, IMG_WIDTH = 390, 130
BATCH_SIZE = 32
GRAIN_CATEGORIES = [40, 80, 120, 180]

_DATA_SUBPATHS = [os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6'), 'fotos_recortes', os.path.join('data', 'fotos_recortes')]


def encontrar_data_dir():
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in _DATA_SUBPATHS:
            c = carpeta / sub
            if c.is_dir() and (c / 'DATASET.xlsx').is_file():
                return str(c)
    raise SystemExit("ERROR: no se encontro ninguna carpeta de datos con DATASET.xlsx")


# -----------------------------------------------------------------------------
# Generador: identico al de barrido_arquitecturas.py, salvo que no arma la
# entrada categorica de grano. 'Grano' viaja igual en el DataFrame para poder
# etiquetar las predicciones despues.
# -----------------------------------------------------------------------------
_CACHE_IMAGENES = {}


class ImagenSoloGenerator(tf.keras.utils.Sequence):
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
        x1, y = [], []
        for _, row in batch.iterrows():
            ruta = os.path.join(self.img_dir, row['nombre_imagen'])
            clave = (ruta, self.img_size)
            img_u8 = _CACHE_IMAGENES.get(clave)
            if img_u8 is None:
                img_u8 = img_to_array(load_img(ruta, color_mode='grayscale',
                                               target_size=self.img_size)).astype(np.uint8)
                _CACHE_IMAGENES[clave] = img_u8
            x1.append(img_u8.astype(np.float32) / 255.0)
            y.append(float(row[self.param]))
        return ({'image_input': np.array(x1, dtype=np.float32)}, np.array(y, dtype=np.float32))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)


def construir_modelo_sin_grano():
    """Arquitectura A sin la rama de grano: Dense(64) alimenta directo a Dense(32).

    Todo lo demas -- base convolucional, GlobalAveragePooling2D, anchos de las
    capas densas -- es identico a construir_modelo('A') en barrido_arquitecturas.py.
    """
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    z = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='linear', name='output')(z)
    return Model(inputs={'image_input': input_img}, outputs=output)


def main():
    ap = argparse.ArgumentParser(description='Ablacion sin grano, arquitectura A.')
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--folds', default='todas',
                    help="'todas' o una lista separada por comas, p.ej. '1' o '1,3'")
    ap.add_argument('--dataset', choices=['limpio', 'outliers'], default='limpio')
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--guardar-modelos', action='store_true')
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    data_dir = encontrar_data_dir()
    excel = os.path.join(data_dir, 'DATASET.xlsx')

    if args.dataset == 'limpio':
        df = pd.read_excel(excel, sheet_name='sin_outliers')
    else:
        todo = pd.read_excel(excel, sheet_name='Todo')
        val = pd.read_excel(excel, sheet_name='Validacion')
        df = todo[~todo['nombre_imagen'].isin(set(val['nombre_imagen']))].reset_index(drop=True)

    probe_ids = df['id_probeta'].to_numpy()
    n_folds = len(np.unique(probe_ids))

    modelo_ref = construir_modelo_sin_grano()
    n_params = modelo_ref.count_params()
    etiqueta_params = f"{n_params/1e6:.2f}M" if n_params >= 1e6 else f"{n_params/1e3:.1f}k"

    base = Path(__file__).resolve().parent / 'resultados_barrido'
    sufijo_param = '' if args.param == 'Ra' else f'_{args.param}'
    out = base / f"AsinGrano_{etiqueta_params}_seed{args.seed}_{args.dataset}{sufijo_param}"
    for sub in ['metricas', 'graficos', 'predicciones', 'modelos']:
        (out / sub).mkdir(parents=True, exist_ok=True)

    config = {
        'experimento': 'ablacion_sin_grano',
        'arquitectura_base': 'A (GlobalAveragePooling2D), sin rama de grano ni concatenacion',
        'parametros': int(n_params),
        'parametro_objetivo': args.param,
        'seed': args.seed,
        'dataset': args.dataset,
        'comando': ' '.join(sys.argv),
        'muestras': int(len(df)),
        'epochs': args.epochs,
        'batch_size': BATCH_SIZE,
        'folds_totales': int(n_folds),
        'folds_corridos': args.folds,
        'motivo': ('comprobar si el R2 de la arquitectura A lo explica la imagen '
                  'o la entrada categorica de grano'),
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
    (out / 'config_corrida.json').write_text(json.dumps(config, indent=2), encoding='utf8')

    quiere = (list(range(1, n_folds + 1)) if args.folds == 'todas'
              else [int(s) for s in args.folds.split(',')])

    gkf = GroupKFold(n_splits=n_folds)
    filas, historiales, t_inicio = [], {}, time.time()

    for fold, (tr, va) in enumerate(gkf.split(df, groups=probe_ids), 1):
        if fold not in quiere:
            continue
        tr_fold, va_fold = df.iloc[tr], df.iloc[va]
        print(f"\n{'='*60}\nFold {fold}/{n_folds}  "
              f"| entrena {len(tr_fold)} | valida {len(va_fold)} "
              f"(probeta {','.join(map(str, va_fold['id_probeta'].unique()))})")

        gen_tr = ImagenSoloGenerator(tr_fold, data_dir, BATCH_SIZE,
                                     (IMG_HEIGHT, IMG_WIDTH), shuffle=True, param=args.param)
        gen_va = ImagenSoloGenerator(va_fold, data_dir, BATCH_SIZE,
                                     (IMG_HEIGHT, IMG_WIDTH), shuffle=False, param=args.param)

        modelo = construir_modelo_sin_grano()
        modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

        t0 = time.time()
        hist = modelo.fit(gen_tr, validation_data=gen_va, epochs=args.epochs, verbose=2)
        dt = time.time() - t0

        pred = modelo.predict(gen_va, verbose=0).flatten()
        real = va_fold[args.param].values
        r2 = r2_score(real, pred)
        mae = mean_absolute_error(real, pred)
        rmse = float(np.sqrt(mean_squared_error(real, pred)))

        print(f"\n  Fold {fold}: R2={r2:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}  "
              f"| {dt:.1f} s ({dt/args.epochs:.2f} s/epoca)")

        filas.append({'Fold': fold, 'Probeta': int(va_fold['id_probeta'].iloc[0]),
                      'n_val': len(va_fold), 'R2': r2, 'MAE': mae, 'RMSE': rmse,
                      'segundos': round(dt, 1), 's_por_epoca': round(dt / args.epochs, 2)})
        historiales[fold] = hist.history

        # 'Grano' viaja en las predicciones para el diagnostico posterior, aunque
        # el modelo jamas lo recibio como entrada.
        pd.DataFrame({'id_probeta': va_fold['id_probeta'].values, 'Ra_real': real,
                      'Ra_predicho': pred, 'Grano': va_fold['Grano'].values}
                     ).to_excel(out / 'predicciones' / f'predicciones_fold{fold}.xlsx', index=False)
        if args.guardar_modelos:
            modelo.save(out / 'modelos' / f'modelo_AsinGrano_fold{fold}.keras')

    with pd.ExcelWriter(out / 'metricas' / 'historial_por_epoca.xlsx') as w:
        for fold, h in historiales.items():
            pd.DataFrame(h).rename_axis('epoca').to_excel(w, sheet_name=f'fold{fold}')

    res = pd.DataFrame(filas)
    if len(res) > 1:
        res = pd.concat([res, pd.DataFrame([{
            'Fold': 'Media', 'R2': res['R2'].mean(), 'MAE': res['MAE'].mean(),
            'RMSE': res['RMSE'].mean(), 'segundos': res['segundos'].sum()}, {
            'Fold': 'Desv. Est.', 'R2': res['R2'].std(), 'MAE': res['MAE'].std(),
            'RMSE': res['RMSE'].std()}])], ignore_index=True)
    res.to_excel(out / 'metricas' / 'metricas_validacion_cruzada.xlsx', index=False)

    if historiales:
        plt.figure(figsize=(14, 5))
        for i, (clave, titulo) in enumerate([('loss', 'Perdida (MSE)'), ('mae', 'MAE')], 1):
            plt.subplot(1, 2, i)
            for fold, h in historiales.items():
                ep = range(1, len(h[clave]) + 1)
                plt.plot(ep, h[clave], alpha=.7, label=f'fold{fold} entren.')
                plt.plot(ep, h['val_' + clave], '--', alpha=.7, label=f'fold{fold} valid.')
            plt.xlabel('Epoca'); plt.ylabel(titulo); plt.title(titulo)
            plt.legend(fontsize=7); plt.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(out / 'graficos' / 'curvas_por_epoca.png', dpi=200, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 78)
    print(res.to_string(index=False))
    print(f"\nTiempo total: {(time.time()-t_inicio)/60:.1f} min")
    print(f"Salidas en: {out}")


if __name__ == '__main__':
    main()
