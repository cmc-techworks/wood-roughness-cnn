#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Barrido de arquitecturas y diagnostico de semilla, sobre la misma base convolucional.

Replica exactamente el bloque de validacion cruzada de Red_multimodal_VC.py (mismo
generador, mismo preprocesamiento, GroupKFold leave-one-probeta-out, 40 epocas, batch 32,
Adam/MSE/MAE) y deja UNA sola cosa como variable: como se reduce el mapa de
caracteristicas antes de la capa densa.

    A  GlobalAveragePooling2D en vez de Flatten      ~25.7 k parametros
    B  Flatten + 2 MaxPooling2D extra                ~0.71 M
    C  Flatten + 1 MaxPooling2D extra                ~2.97 M
    D  Flatten directo (la arquitectura publicada)   ~12.21 M

Uso:
    python barrido_arquitecturas.py --arch D --seed 20 --folds 1
    python barrido_arquitecturas.py --arch A --folds todas
    python barrido_arquitecturas.py --arch D --dataset outliers --folds todas

Nunca escribe en 'Entrenar modelo/MULTI_VC_RA/'. Cada corrida deja su propia carpeta con
la configuracion completa, para que se explique sola.
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
from tensorflow.keras.utils import img_to_array, load_img, to_categorical

IMG_HEIGHT, IMG_WIDTH = 390, 130
BATCH_SIZE = 32
GRAIN_CATEGORIES = [40, 80, 120, 180]
NUM_GRAIN_CATEGORIES = len(GRAIN_CATEGORIES)

# Dos layouts aceptados: el original de la tesis y el de esta carpeta de revision.
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
# Generador: identico al de Red_multimodal_VC.py, con cache en memoria
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Fabrica de arquitecturas: solo cambia la reduccion previa a la capa densa
# -----------------------------------------------------------------------------
def construir_modelo(arch):
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
    ap = argparse.ArgumentParser(description='Barrido de arquitecturas / diagnostico de semilla.')
    ap.add_argument('--arch', choices=['A', 'B', 'C', 'D'], default='D')
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--folds', default='todas',
                    help="'todas' o una lista separada por comas, p.ej. '1' o '1,3'")
    ap.add_argument('--dataset', choices=['limpio', 'outliers'], default='limpio',
                    help="limpio = hoja 'sin_outliers' (1030); outliers = hoja 'Todo' menos test (1056)")
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra',
                    help='Parametro de rugosidad a predecir')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--guardar-modelos', action='store_true',
                    help='guarda los .keras (pesan ~146 MB en la arquitectura D)')
    args = ap.parse_args()

    # Semillas fijadas ANTES de cualquier consumo de aleatoriedad, igual que el original
    # Ver diagnostico_determinismo.py (2026-08-17): np.random.seed + tf.random.set_seed no fija
    # la inicializacion de pesos en Keras 3. keras.utils.set_random_seed si lo hace.
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

    modelo_ref = construir_modelo(args.arch)
    n_params = modelo_ref.count_params()
    etiqueta_params = f"{n_params/1e6:.2f}M" if n_params >= 1e6 else f"{n_params/1e3:.1f}k"

    base = Path(__file__).resolve().parent / 'resultados_barrido'
    sufijo_param = '' if args.param == 'Ra' else f'_{args.param}'
    out = base / f"{args.arch}_{etiqueta_params}_seed{args.seed}_{args.dataset}{sufijo_param}"
    for sub in ['metricas', 'graficos', 'predicciones', 'modelos']:
        (out / sub).mkdir(parents=True, exist_ok=True)

    config = {
        'arquitectura': args.arch,
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

        gen_tr = MultimodalDataGenerator(tr_fold, data_dir, BATCH_SIZE,
                                         (IMG_HEIGHT, IMG_WIDTH), shuffle=True, param=args.param)
        gen_va = MultimodalDataGenerator(va_fold, data_dir, BATCH_SIZE,
                                         (IMG_HEIGHT, IMG_WIDTH), shuffle=False, param=args.param)

        modelo = construir_modelo(args.arch)
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

        pd.DataFrame({'id_probeta': va_fold['id_probeta'].values,
                      f'{args.param}_real': real, f'{args.param}_predicho': pred,
                      'Grano': va_fold['Grano'].values}
                     ).to_excel(out / 'predicciones' / f'predicciones_fold{fold}.xlsx', index=False)
        if args.guardar_modelos:
            modelo.save(out / 'modelos' / f'modelo_{args.arch}_fold{fold}.keras')

    # --- historial por epoca a disco: es lo que el script original nunca guardo ---
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
