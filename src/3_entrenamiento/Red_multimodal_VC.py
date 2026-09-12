#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# SCRIPT PARA ENTRENAR UNA RED NEURONAL CONVOLUCIONAL MULTIMODAL (CNN)
# PARA LA PREDICCIÓN DE RUGOSIDAD A PARTIR DE IMÁGENES Y CATEGORÍA DE GRANO
# =============================================================================

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

# Fijar semillas para reproducibilidad
SEED = 10

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Directorio principal donde se encuentran las imágenes y el archivo de etiquetas.
# Se ancla a la raíz del repositorio buscando la carpeta de datos hacia arriba, de modo
# que el script funcione sin importar desde dónde se ejecute (raíz o CODIGO/) y en
# cualquier equipo, sin rutas absolutas que se rompan al mover la carpeta.
_DATA_SUBPATH = os.path.join('Fotos', 'Recortes15x15mm', 'recortes_x6')

def _encontrar_data_dir():
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        candidato = carpeta / _DATA_SUBPATH
        if candidato.is_dir():
            return str(candidato)
    # Si no se encuentra, devolver la ruta esperada junto al script para un error claro
    return str(inicio / _DATA_SUBPATH)

DATA_DIR = _encontrar_data_dir()
EXCEL_PATH = os.path.join(DATA_DIR, 'DATASET.xlsx')
print(f"DATA_DIR resuelto: {DATA_DIR}")

# Carpetas de salida
OUT_DIR       = os.path.join(os.path.dirname(__file__), 'resultados_entrenamiento')
OUT_GRAFICOS  = os.path.join(OUT_DIR, 'graficos')
OUT_PRED      = os.path.join(OUT_DIR, 'predicciones')
OUT_METRICAS  = os.path.join(OUT_DIR, 'metricas')
OUT_MODELOS   = os.path.join(OUT_DIR, 'modelos')
for _d in [OUT_GRAFICOS, OUT_PRED, OUT_METRICAS, OUT_MODELOS]:
    os.makedirs(_d, exist_ok=True)

# Parámetros para el entrenamiento
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
# Caché de imágenes en memoria (compartido entre todos los generadores y folds).
# Evita releer las ~1030 imágenes desde disco en cada época. Se guardan en uint8
# (0-255, sin pérdida) y la normalización /255 se aplica al servir cada batch, de
# modo que los valores entregados al modelo son idénticos a los de la versión original.
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
            # Cargar y preprocesar imagen (con caché en memoria; clave = ruta + tamaño)
            img_path = os.path.join(self.img_dir, row['nombre_imagen'])
            clave_cache = (img_path, self.img_size)
            img_uint8 = _CACHE_IMAGENES.get(clave_cache)
            if img_uint8 is None:
                img_pil = load_img(
                    img_path,
                    color_mode='grayscale',
                    target_size=self.img_size
                )
                # Se almacena en uint8 (0-255, sin pérdida) para usar ~4x menos RAM
                img_uint8 = img_to_array(img_pil).astype(np.uint8)
                _CACHE_IMAGENES[clave_cache] = img_uint8
            img = img_uint8.astype(np.float32) / 255.0  # Normalización idéntica a la original
            
            # Obtener categoría de grano
            grain = int(row['Grano'])  # Asegurarse de que sea entero
            grain_idx = GRAIN_CATEGORIES.index(grain)
            grain_one_hot = to_categorical(grain_idx, num_classes=NUM_GRAIN_CATEGORIES)
            
            batch_x1.append(img)
            batch_x2.append(grain_one_hot)
            batch_y.append(float(row['Ra']))  # Asegurar que sea float
            
        # Convertir a arrays de numpy con tipos de datos específicos
        batch_x1 = np.array(batch_x1, dtype=np.float32)
        batch_x2 = np.array(batch_x2, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)
        
        # Devolver como un diccionario con los nombres de las entradas
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

# Cargar datos de la hoja 'sin_outliers'
try:
    df = pd.read_excel(EXCEL_PATH, sheet_name='sin_outliers')
    print("Columnas disponibles en el archivo Excel:", df.columns.tolist())
    print(f"Se encontraron {len(df)} registros en la hoja 'sin_outliers' del archivo Excel.")
except Exception as e:
    print(f"Error al cargar el archivo Excel: {e}")
    # Mostrar las hojas disponibles en caso de error
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
    
    # Verificar distribución de granos en cada fold
    print(f"\nDistribución de granos en Fold {fold}:")
    print(f"  - Entrenamiento: {dict(sorted(df.iloc[train_idx]['Grano'].value_counts().items()))}")
    print(f"  - Validación:    {dict(sorted(df.iloc[val_idx]['Grano'].value_counts().items()))}")
    
    # Crear generadores para este fold
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
    
    # Crear un nuevo modelo para este fold
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')
    
    # Rama de la CNN para procesar imágenes
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Entrada para la categoría de grano
    input_grain = Input(shape=(NUM_GRAIN_CATEGORIES,), name='grain_input')
    g = layers.Dense(16, activation='relu')(input_grain)  # Capa de embedding
    
    # Combinar ambas ramas
    combined = layers.Concatenate()([x, g])
    
    # Capas densas finales
    z = layers.Dense(32, activation='relu')(combined)
    output = layers.Dense(1, activation='linear', name='output')(z)
    
    # Crear modelo
    model = Model(inputs={'image_input': input_img, 'grain_input': input_grain}, outputs=output)
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Entrenar modelo
    print(f"\nEntrenando fold {fold}...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        verbose=2
    )
    all_histories.append(history.history)
    
    # Evaluar en el conjunto de validación
    val_pred = model.predict(validation_generator)
    val_true = val_fold['Ra'].values
    
    # Calcular y graficar residuos
    residuos = val_true - val_pred.flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(val_pred.flatten(), residuos, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title(f'Gráfico de Residuos - Fold {fold}')
    plt.savefig(os.path.join(OUT_GRAFICOS, f'residuos_fold{fold}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Gráfico de residuos guardado en '{OUT_GRAFICOS}/residuos_fold{fold}.png'")
    
    # Guardar predicciones para este fold
    predictions_df = pd.DataFrame({
        'id_probeta': val_fold['id_probeta'].values,
        'Ra_real': val_true,
        'Ra_predicho': val_pred.flatten(),
        'Grano': val_fold['Grano'].values
    })
    predictions_df.to_excel(os.path.join(OUT_PRED, f'predicciones_fold{fold}.xlsx'), index=False)
    print(f"Predicciones del fold {fold} guardadas en '{OUT_PRED}/predicciones_fold{fold}.xlsx'")
    
    # Calcular métricas
    fold_r2 = r2_score(val_true, val_pred)
    fold_mae = mean_absolute_error(val_true, val_pred)
    fold_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
    
    print(f"\nMétricas del fold {fold}:")
    print(f"R²: {fold_r2:.4f}")
    print(f"MAE: {fold_mae:.4f}")
    print(f"RMSE: {fold_rmse:.4f}")
    
    # Liberar memoria al final de cada fold (gc ya está importado arriba)
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Almacenar métricas
    all_r2.append(fold_r2)
    all_mae.append(fold_mae)
    all_rmse.append(fold_rmse)
    
    # Guardar modelo de este fold en formato .keras
    model.save(os.path.join(OUT_MODELOS, f'modelo_multimodal_VC_fold{fold}.keras'))
    print(f"\nModelo del fold {fold} guardado en '{OUT_MODELOS}/modelo_multimodal_VC_fold{fold}.keras'")
    
    # Eliminar archivos antiguos .h5 si existen
    old_h5_file = f'modelo_multimoda_VC_fold{fold}.h5'
    if os.path.exists(old_h5_file):
        try:
            os.remove(old_h5_file)
            print(f"Archivo antiguo {old_h5_file} eliminado")
        except Exception as e:
            print(f"No se pudo eliminar {old_h5_file}: {e}")

# Calcular y mostrar métricas finales
print("\n" + "="*50)
print("RESULTADOS FINALES DE LA VALIDACIÓN CRUZADA")
print("="*50)
print(f"Número de pliegues: {num_folds}")
print(f"R² promedio: {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
print(f"MAE promedio: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
print(f"RMSE promedio: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")

# Guardar métricas
metrics_data = []
for i in range(num_folds):
    metrics_data.append({
        'Fold': i + 1,
        'R2': all_r2[i],
        'MAE': all_mae[i],
        'RMSE': all_rmse[i]
    })

# Agregar estadísticas
metrics_data.extend([
    {'Fold': 'Media', 'R2': np.mean(all_r2), 'MAE': np.mean(all_mae), 'RMSE': np.mean(all_rmse)},
    {'Fold': 'Desv. Est.', 'R2': np.std(all_r2), 'MAE': np.std(all_mae), 'RMSE': np.std(all_rmse)}
])

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_excel(os.path.join(OUT_METRICAS, 'metricas_validacion_cruzada.xlsx'), index=False)
print(f"\nMétricas detalladas guardadas en '{OUT_METRICAS}/metricas_validacion_cruzada.xlsx'")

# -----------------------------------------------------------------------------
# 8. ENTRENAMIENTO DEL MODELO FINAL CON TODOS LOS DATOS
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("ENTRENANDO MODELO FINAL CON TODOS LOS DATOS")
print("="*50)

# Crear generador con todos los datos
train_generator_all = MultimodalDataGenerator(
    df, 
    img_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True
)

# Crear modelo final con la misma arquitectura
input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')

# Rama de la CNN para procesar imágenes
x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

# Entrada para la categoría de grano
input_grain = Input(shape=(NUM_GRAIN_CATEGORIES,), name='grain_input')
g = layers.Dense(16, activation='relu')(input_grain)  # Capa de embedding

# Combinar ambas ramas
combined = layers.Concatenate()([x, g])

# Capas densas finales
z = layers.Dense(32, activation='relu')(combined)
output = layers.Dense(1, activation='linear', name='output')(z)

# Crear modelo
final_model = Model(inputs={'image_input': input_img, 'grain_input': input_grain}, outputs=output)

# Compilar modelo
final_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Callbacks para el entrenamiento
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
]

# Entrenar el modelo final con todos los datos
print("Entrenando modelo final con todos los datos...")
final_history = final_model.fit(
    train_generator_all,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# -----------------------------------------------------------------------------
# 9. VISUALIZACIÓN DE RESULTADOS DE LA VALIDACIÓN CRUZADA
# -----------------------------------------------------------------------------
def plot_cross_validation_metrics(histories, num_epochs):
    """
    Grafica las métricas promedio y su desviación estándar a lo largo de las épocas
    para todos los pliegues de la validación cruzada.
    """
    # Inicializar arrays para almacenar métricas de todos los pliegues
    train_loss = np.zeros((len(histories), num_epochs))
    val_loss = np.zeros((len(histories), num_epochs))
    train_mae = np.zeros((len(histories), num_epochs))
    val_mae = np.zeros((len(histories), num_epochs))
    
    # Recopilar métricas de todos los pliegues
    for i, history in enumerate(histories):
        train_loss[i] = history['loss']
        val_loss[i] = history['val_loss']
        train_mae[i] = history['mae']
        val_mae[i] = history['val_mae']
    
    # Calcular media y desviación estándar
    train_loss_mean = np.mean(train_loss, axis=0)
    train_loss_std = np.std(train_loss, axis=0)
    val_loss_mean = np.mean(val_loss, axis=0)
    val_loss_std = np.std(val_loss, axis=0)
    
    train_mae_mean = np.mean(train_mae, axis=0)
    train_mae_std = np.std(train_mae, axis=0)
    val_mae_mean = np.mean(val_mae, axis=0)
    val_mae_std = np.std(val_mae, axis=0)
    
    epochs_range = range(1, num_epochs + 1)
    
    # Crear figura con dos subplots
    plt.figure(figsize=(16, 6))
    
    # Gráfico de la Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_mean, label='Entrenamiento', color='blue')
    plt.fill_between(epochs_range, 
                    train_loss_mean - train_loss_std, 
                    train_loss_mean + train_loss_std, 
                    color='blue', alpha=0.2)
    
    plt.plot(epochs_range, val_loss_mean, label='Validación', color='orange')
    plt.fill_between(epochs_range, 
                    val_loss_mean - val_loss_std, 
                    val_loss_mean + val_loss_std, 
                    color='orange', alpha=0.2)
    
    plt.title('Pérdida Promedio (MSE) en Validación Cruzada')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico del Error Absoluto Medio (MAE)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_mae_mean, label='Entrenamiento', color='blue')
    plt.fill_between(epochs_range, 
                    train_mae_mean - train_mae_std, 
                    train_mae_mean + train_mae_std, 
                    color='blue', alpha=0.2)
    
    plt.plot(epochs_range, val_mae_mean, label='Validación', color='orange')
    plt.fill_between(epochs_range, 
                    val_mae_mean - val_mae_std, 
                    val_mae_mean + val_mae_std, 
                    color='orange', alpha=0.2)
    
    plt.title('Error Absoluto Medio (MAE) en Validación Cruzada')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_GRAFICOS, 'historial_validacion_cruzada.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Graficar métricas de validación cruzada
if all_histories:
    print("\nGenerando gráficos de validación cruzada...")
    plot_cross_validation_metrics(all_histories, EPOCHS)
    print(f"Gráficos de validación cruzada guardados en '{OUT_GRAFICOS}/historial_validacion_cruzada.png'")

# Graficar comparación de métricas entre pliegues
plt.figure(figsize=(14, 5))

# Gráfico de R² por pliegue
plt.subplot(1, 3, 1)
plt.bar(range(1, num_folds + 1), all_r2, color='skyblue')
plt.axhline(y=np.mean(all_r2), color='r', linestyle='--', label=f'Media: {np.mean(all_r2):.3f}') # type: ignore
plt.title('R² por Pliegue')
plt.xlabel('Pliegue')
plt.ylim(0, 1.0)
plt.legend()

# Gráfico de MAE por pliegue
plt.subplot(1, 3, 2)
plt.bar(range(1, num_folds + 1), all_mae, color='lightgreen')
plt.axhline(y=np.mean(all_mae), color='r', linestyle='--', label=f'Media: {np.mean(all_mae):.3f}') # type: ignore
plt.title('MAE por Pliegue')
plt.xlabel('Pliegue')
plt.legend()

# Gráfico de RMSE por pliegue
plt.subplot(1, 3, 3)
plt.bar(range(1, num_folds + 1), all_rmse, color='salmon')
plt.axhline(y=np.mean(all_rmse), color='r', linestyle='--', label=f'Media: {np.mean(all_rmse):.3f}') # type: ignore
plt.title('RMSE por Pliegue')
plt.xlabel('Pliegue')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_GRAFICOS, 'metricas_por_pliegue.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráficos de métricas por pliegue guardados en '{OUT_GRAFICOS}/metricas_por_pliegue.png'")

def create_model():
    """Función auxiliar para crear la arquitectura del modelo"""
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')
    
    # Rama de la CNN para procesar imágenes
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Entrada para la categoría de grano
    input_grain = Input(shape=(NUM_GRAIN_CATEGORIES,), name='grain_input')
    g = layers.Dense(16, activation='relu')(input_grain)  # Capa de embedding
    
    # Combinar ambas ramas
    combined = layers.Concatenate()([x, g])
    
    # Capas densas finales
    z = layers.Dense(32, activation='relu')(combined)
    output = layers.Dense(1, activation='linear', name='output')(z)
    
    return Model(inputs={'image_input': input_img, 'grain_input': input_grain}, outputs=output)

# =============================================================================
# 9. ENTRENAMIENTO FINAL CON TODOS LOS DATOS
# =============================================================================
print("\n" + "="*50)
print("ENTRENAMIENTO FINAL CON TODOS LOS DATOS")
print("="*50)

# Crear generador con todos los datos
train_generator_all = MultimodalDataGenerator(
    df, 
    img_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True
)

# Compilar el modelo final
final_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Callbacks para el entrenamiento
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

# Entrenar el modelo final con todos los datos
print("\nIniciando entrenamiento final...")
final_history = final_model.fit(
    train_generator_all,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# Guardar el modelo final
final_model.save(os.path.join(OUT_MODELOS, 'modelo_final_multimodal_VC.keras'))
print(f"\nModelo final guardado en '{OUT_MODELOS}/modelo_final_multimodal_VC.keras'")

# Guardar el historial de entrenamiento
hist_df = pd.DataFrame(final_history.history)
hist_df.to_excel(os.path.join(OUT_METRICAS, 'historial_entrenamiento_final.xlsx'), index=False)
print(f"Historial de entrenamiento guardado en '{OUT_METRICAS}/historial_entrenamiento_final.xlsx'")

# =============================================================================
# 10. RESUMEN FINAL DE MÉTRICAS
# =============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL DEL MODELO MULTIMODAL")
print("="*80)

# Mostrar resumen de datos por fold
print("\n" + "="*50)
print("DISTRIBUCIÓN DE DATOS")
print("="*50)
print(f"Total de muestras en el dataset: {len(df)}")
print(f"Total de probetas únicas: {len(unique_probes)}")
print(f"Número de pliegues (folds): {num_folds}")

# Mostrar distribución de datos por fold
print("\nMuestras por fold (entrenamiento/validación):")
for fold, (train_idx, val_idx) in enumerate(group_kfold.split(df, groups=probe_ids), 1):
    print(f"Fold {fold}: {len(train_idx)} entrenamiento, {len(val_idx)} validación")

# Calcular y mostrar métricas finales del modelo
print("\n" + "="*50)
print("MÉTRICAS FINALES (Validación Cruzada)")
print("="*50)
print(f"R² promedio: {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
print(f"MAE promedio: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
print(f"RMSE promedio: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")

# Calcular MAPE promedio
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100

# Calcular MAPE para cada fold
all_mape = []
for fold in range(1, num_folds + 1):
    pred_file = os.path.join(OUT_PRED, f'predicciones_fold{fold}.xlsx')
    if os.path.exists(pred_file):
        df_pred = pd.read_excel(pred_file)
        mape = mean_absolute_percentage_error(df_pred['Ra_real'], df_pred['Ra_predicho'])
        all_mape.append(mape)

if all_mape:
    print(f"MAPE promedio: {np.mean(all_mape):.2f}% ± {np.std(all_mape):.2f}%")

# Mostrar métricas por fold
print("\n" + "="*50)
print("MÉTRICAS POR FOLD")
print("="*50)
print(f"{'Fold':<8} {'Muestras':<12} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'MAPE (%)':<10}")
print("-" * 60)

for fold in range(1, num_folds + 1):
    pred_file = os.path.join(OUT_PRED, f'predicciones_fold{fold}.xlsx')
    if os.path.exists(pred_file):
        df_pred = pd.read_excel(pred_file)
        mape = mean_absolute_percentage_error(df_pred['Ra_real'], df_pred['Ra_predicho'])
        print(f"{fold:<8} {len(df_pred):<12} {all_r2[fold-1]:.4f}   {all_mae[fold-1]:.4f}    {all_rmse[fold-1]:.4f}    {mape:.2f}%")

# Mostrar métricas del modelo final
print("\n" + "="*50)
print("MÉTRICAS DEL MODELO FINAL (entrenado con todos los datos)")
print("="*50)
print(f"Épocas de entrenamiento: {len(final_history.history['loss'])}")
print(f"Pérdida final (MSE): {final_history.history['loss'][-1]:.6f}")
print(f"MAE final: {final_history.history['mae'][-1]:.6f}")

# Hacer predicciones con el modelo final
print("\nRealizando predicciones con el modelo final...")
final_predictions = final_model.predict(train_generator_all)

# Obtener valores reales
y_true = []
for i in range(len(train_generator_all)):
    _, y_batch = train_generator_all[i]
    y_true.extend(y_batch.flatten())
y_true = np.array(y_true)

# Asegurarse de que las predicciones tengan la forma correcta
final_predictions = final_predictions.flatten()

# Calcular RMSE (mean_squared_error ya está importado arriba)
rmse_final = np.sqrt(mean_squared_error(y_true, final_predictions))
print(f"RMSE final: {rmse_final:.6f}")

# Calcular MAPE
mape_final = mean_absolute_percentage_error(y_true, final_predictions)
print(f"MAPE final: {mape_final:.2f}%")

print("\n" + "="*80)
print("ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
print("="*80)