#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# VARIANTE de Red_multimodal_VC.py para R1.3 / T1.2 (2026-08-04).
# Único cambio de fondo: entrena con las 1056 candidatas de entrenamiento/validación
# (las 1030 de 'sin_outliers' + las 26 excluidas por el filtro IQR), en vez de solo 1030.
# Todo lo demás -- arquitectura, GroupKFold leave-one-probe-out, 40 épocas, batch 32,
# semilla 10 -- queda idéntico al script original para que la comparación de métricas de
# test contra el modelo publicado sea válida (mismo protocolo, único factor que cambia
# es la inclusión de los 26 outliers).
#
# Motivo: comprobar que excluir esas 26 muestras (identificadas por
# 1.5×IQR por grano, Ra/Rz por separado -- criterio confirmado exacto contra DATASET.xlsx)
# no sesga el modelo hacia condiciones "ideales": las 26 caen casi todas en la cola
# rugosa (15/15 en Ra, 23/24 en Rz), que es justamente el riesgo. Este script entrena la
# variante "con los 26 incluidos" para poder comparar el delta de métricas de test contra
# el modelo publicado. Un delta chico es la evidencia más directa posible.
#
# Salida en 'resultados_entrenamiento_con_outliers/' (carpeta separada) -- no toca ni
# sobrescribe nada de 'Entrenar modelo/MULTI_VC_RA/' (el modelo publicado).

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -----------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# -----------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import load_img, img_to_array, to_categorical
import pandas as pd
import os
import gc
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Verificar y configurar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Configurar crecimiento de memoria para cada GPU disponible
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # type: ignore[attr-defined]
        print(f"GPUs disponibles: {gpus}")
    except RuntimeError as e:
        print(f"Error al configurar la GPU: {e}")
else:
    print("No se detectaron GPUs. El entrenamiento se ejecutará en CPU.")

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# -----------------------------------------------------------------------------

# Fijar semillas para reproducibilidad -- idéntico al original, no se toca
SEED = 10

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Directorio principal donde se encuentran las imágenes y el archivo de etiquetas.
# Se prueban dos layouts conocidos: el de trabajo original (Fotos/Recortes15x15mm/
# recortes_x6) y el de esta carpeta de revisión (fotos_recortes/), que trae su propia
# copia de DATASET.xlsx junto a las imágenes.
_DATA_SUBPATHS = [
    os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6'),
    'fotos_recortes',
    os.path.join('data', 'fotos_recortes'),
]

def _encontrar_data_dir():
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for subpath in _DATA_SUBPATHS:
            candidato = carpeta / subpath
            if candidato.is_dir() and (candidato / 'DATASET.xlsx').is_file():
                return str(candidato)
    # Si no se encuentra, devolver la ruta esperada junto al script para un error claro
    return str(inicio / _DATA_SUBPATHS[0])

DATA_DIR = _encontrar_data_dir()
EXCEL_PATH = os.path.join(DATA_DIR, 'DATASET.xlsx')
print(f"DATA_DIR resuelto: {DATA_DIR}")

# Carpetas de salida -- separadas del script original para no pisar sus resultados
OUT_DIR       = os.path.join(os.path.dirname(__file__), 'resultados_entrenamiento_con_outliers')
OUT_GRAFICOS  = os.path.join(OUT_DIR, 'graficos')
OUT_PRED      = os.path.join(OUT_DIR, 'predicciones')
OUT_METRICAS  = os.path.join(OUT_DIR, 'metricas')
OUT_MODELOS   = os.path.join(OUT_DIR, 'modelos')
for _d in [OUT_GRAFICOS, OUT_PRED, OUT_METRICAS, OUT_MODELOS]:
    os.makedirs(_d, exist_ok=True)

# Parámetros para el entrenamiento -- idénticos al original
IMG_HEIGHT = 390
IMG_WIDTH = 130
BATCH_SIZE = 32
EPOCHS = 40

# Categorías de grano
GRAIN_CATEGORIES = [40, 80, 120, 180]
NUM_GRAIN_CATEGORIES = len(GRAIN_CATEGORIES)

# -----------------------------------------------------------------------------
# 3. CARGA Y PREPARACIÓN DE DATOS
# -----------------------------------------------------------------------------
_CACHE_IMAGENES = {}

class MultimodalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, img_dir, batch_size=32, img_size=(390, 130), shuffle=True):
        self.df = df.copy()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        batch_x1 = []
        batch_x2 = []
        batch_y = []

        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.img_dir, row['nombre_imagen'])
            clave_cache = (img_path, self.img_size)
            img_uint8 = _CACHE_IMAGENES.get(clave_cache)
            if img_uint8 is None:
                img_pil = load_img(
                    img_path,
                    color_mode='grayscale',
                    target_size=self.img_size
                )
                img_uint8 = img_to_array(img_pil).astype(np.uint8)
                _CACHE_IMAGENES[clave_cache] = img_uint8
            img = img_uint8.astype(np.float32) / 255.0

            grain = int(row['Grano'])
            grain_idx = GRAIN_CATEGORIES.index(grain)
            grain_one_hot = to_categorical(grain_idx, num_classes=NUM_GRAIN_CATEGORIES)

            batch_x1.append(img)
            batch_x2.append(grain_one_hot)
            batch_y.append(float(row['Ra']))

        batch_x1 = np.array(batch_x1, dtype=np.float32)
        batch_x2 = np.array(batch_x2, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return {'image_input': batch_x1, 'grain_input': batch_x2}, batch_y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

# --- ÚNICO CAMBIO DE FONDO respecto al script original ---
# El original carga solo la hoja 'sin_outliers' (1030 filas). Acá se reconstruyen las
# 1056 candidatas de entrenamiento/validación uniendo 'sin_outliers' (1030) con las 26
# filas de 'Outliers' (mismas columnas nombre_imagen/Grano/Ra/Rz/id_probeta, más los dos
# flags Es_Outlier_Ra/Es_Outlier_Rz que no hacen falta para entrenar y se descartan).
try:
    df_sin_outliers = pd.read_excel(EXCEL_PATH, sheet_name='sin_outliers')
    df_outliers = pd.read_excel(EXCEL_PATH, sheet_name='Outliers')
    columnas_comunes = ['nombre_imagen', 'Grano', 'Ra', 'Rz', 'id_probeta']
    df = pd.concat([df_sin_outliers[columnas_comunes], df_outliers[columnas_comunes]],
                    ignore_index=True)
    print(f"Hoja 'sin_outliers': {len(df_sin_outliers)} filas")
    print(f"Hoja 'Outliers': {len(df_outliers)} filas")
    print(f"Total combinado (candidatas completas, con los 26 outliers incluidos): {len(df)}")
    assert len(df) == len(df_sin_outliers) + len(df_outliers) == 1056, \
        f"Se esperaban 1056 filas combinadas, se obtuvieron {len(df)} -- revisar DATASET.xlsx antes de entrenar"
except Exception as e:
    print(f"Error al cargar el archivo Excel: {e}")
    xls = pd.ExcelFile(EXCEL_PATH)
    print("Hojas disponibles en el archivo:", xls.sheet_names)
    raise

# Obtener los IDs de las probetas únicas
probe_ids = df['id_probeta'].to_numpy()
unique_probes = np.unique(probe_ids)
num_folds = len(unique_probes)  # Leave-one-probe-out cross-validation

print(f"\nConfiguración de la validación cruzada:")
print(f"- Número de pliegues (probetas únicas): {num_folds}")
print(f"- Número total de muestras: {len(df)}")
print(f"- Tamaño aproximado de cada conjunto de prueba: {len(df) // num_folds} muestras")

# Inicializar listas para almacenar métricas
all_r2 = []
all_mae = []
all_rmse = []
all_histories = []

# Crear el objeto GroupKFold
group_kfold = GroupKFold(n_splits=num_folds)

# Realizar validación cruzada K-fold
for fold, (train_idx, val_idx) in enumerate(group_kfold.split(df, groups=probe_ids), 1): # type: ignore
    print(f"\n{'='*50}")
    print(f"Fold {fold}/{num_folds}")
    print(f"Probetas en conjunto de entrenamiento: {len(np.unique(probe_ids[train_idx]))}")
    print(f"Probetas en conjunto de validación: {len(np.unique(probe_ids[val_idx]))}")

    print(f"\nDistribución de granos en Fold {fold}:")
    print(f"  - Entrenamiento: {dict(sorted(df.iloc[train_idx]['Grano'].value_counts().items()))}")
    print(f"  - Validación:    {dict(sorted(df.iloc[val_idx]['Grano'].value_counts().items()))}")

    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]

    train_generator = MultimodalDataGenerator(
        train_fold,
        img_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True
    )

    validation_generator = MultimodalDataGenerator(
        val_fold,
        img_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False
    )

    # Arquitectura idéntica al modelo publicado
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')

    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)

    input_grain = Input(shape=(NUM_GRAIN_CATEGORIES,), name='grain_input')
    g = layers.Dense(16, activation='relu')(input_grain)

    combined = layers.Concatenate()([x, g])

    z = layers.Dense(32, activation='relu')(combined)
    output = layers.Dense(1, activation='linear', name='output')(z)

    model = Model(inputs={'image_input': input_img, 'grain_input': input_grain}, outputs=output)

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    print(f"\nEntrenando fold {fold}...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        verbose=2
    )
    all_histories.append(history.history)

    val_pred = model.predict(validation_generator)
    val_true = val_fold['Ra'].values

    residuos = val_true - val_pred.flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(val_pred.flatten(), residuos, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title(f'Gráfico de Residuos - Fold {fold} (con outliers incluidos)')
    plt.savefig(os.path.join(OUT_GRAFICOS, f'residuos_fold{fold}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Gráfico de residuos guardado en '{OUT_GRAFICOS}/residuos_fold{fold}.png'")

    predictions_df = pd.DataFrame({
        'id_probeta': val_fold['id_probeta'].values,
        'Ra_real': val_true,
        'Ra_predicho': val_pred.flatten(),
        'Grano': val_fold['Grano'].values
    })
    predictions_df.to_excel(os.path.join(OUT_PRED, f'predicciones_fold{fold}.xlsx'), index=False)
    print(f"Predicciones del fold {fold} guardadas en '{OUT_PRED}/predicciones_fold{fold}.xlsx'")

    fold_r2 = r2_score(val_true, val_pred)
    fold_mae = mean_absolute_error(val_true, val_pred)
    fold_rmse = np.sqrt(mean_squared_error(val_true, val_pred))

    print(f"\nMétricas del fold {fold}:")
    print(f"R²: {fold_r2:.4f}")
    print(f"MAE: {fold_mae:.4f}")
    print(f"RMSE: {fold_rmse:.4f}")

    tf.keras.backend.clear_session()
    gc.collect()

    all_r2.append(fold_r2)
    all_mae.append(fold_mae)
    all_rmse.append(fold_rmse)

    model.save(os.path.join(OUT_MODELOS, f'modelo_multimodal_VC_fold{fold}.keras'))
    print(f"\nModelo del fold {fold} guardado en '{OUT_MODELOS}/modelo_multimodal_VC_fold{fold}.keras'")

# Calcular y mostrar métricas finales
print("\n" + "="*50)
print("RESULTADOS FINALES DE LA VALIDACIÓN CRUZADA (CON OUTLIERS INCLUIDOS)")
print("="*50)
print(f"Número de pliegues: {num_folds}")
print(f"R² promedio: {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
print(f"MAE promedio: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
print(f"RMSE promedio: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")

metrics_data = []
for i in range(num_folds):
    metrics_data.append({
        'Fold': i + 1,
        'R2': all_r2[i],
        'MAE': all_mae[i],
        'RMSE': all_rmse[i]
    })

metrics_data.extend([
    {'Fold': 'Media', 'R2': np.mean(all_r2), 'MAE': np.mean(all_mae), 'RMSE': np.mean(all_rmse)},
    {'Fold': 'Desv. Est.', 'R2': np.std(all_r2), 'MAE': np.std(all_mae), 'RMSE': np.std(all_rmse)}
])

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_excel(os.path.join(OUT_METRICAS, 'metricas_validacion_cruzada.xlsx'), index=False)
print(f"\nMétricas detalladas guardadas en '{OUT_METRICAS}/metricas_validacion_cruzada.xlsx'")

# -----------------------------------------------------------------------------
# 4. ENTRENAMIENTO DEL MODELO FINAL CON TODOS LOS DATOS (1056, con outliers)
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("ENTRENANDO MODELO FINAL CON TODOS LOS DATOS (CON OUTLIERS INCLUIDOS)")
print("="*50)

train_generator_all = MultimodalDataGenerator(
    df,
    img_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True
)

input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')
x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

input_grain = Input(shape=(NUM_GRAIN_CATEGORIES,), name='grain_input')
g = layers.Dense(16, activation='relu')(input_grain)

combined = layers.Concatenate()([x, g])
z = layers.Dense(32, activation='relu')(combined)
output = layers.Dense(1, activation='linear', name='output')(z)

final_model = Model(inputs={'image_input': input_img, 'grain_input': input_grain}, outputs=output)

final_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUT_MODELOS, 'mejor_modelo_final.keras'),
        save_best_only=True,
        monitor='loss',
        mode='min',
        verbose=1
    )
]

print("\nIniciando entrenamiento final...")
final_history = final_model.fit(
    train_generator_all,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

final_model.save(os.path.join(OUT_MODELOS, 'modelo_final_multimodal_VC_con_outliers.keras'))
print(f"\nModelo final guardado en '{OUT_MODELOS}/modelo_final_multimodal_VC_con_outliers.keras'")

hist_df = pd.DataFrame(final_history.history)
hist_df.to_excel(os.path.join(OUT_METRICAS, 'historial_entrenamiento_final.xlsx'), index=False)
print(f"Historial de entrenamiento guardado en '{OUT_METRICAS}/historial_entrenamiento_final.xlsx'")

print("\n" + "="*80)
print("ENTRENAMIENTO FINALIZADO -- MODELO 'CON OUTLIERS INCLUIDOS' (1056 muestras)")
print("Siguiente paso manual: correr Validar_modelo_multimodal_vc.py con")
print(f"  {os.path.join(OUT_MODELOS, 'modelo_final_multimodal_VC_con_outliers.keras')}")
print("sobre el mismo set de test independiente (hoja 'Validacion' de DATASET.xlsx) y")
print("comparar contra 'Entrenar modelo/MULTI_VC_RA/resultados_validacion_Ra.xlsx'")
print("(el modelo publicado, entrenado solo con las 1030 sin outliers).")
print("="*80)
