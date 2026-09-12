import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tkinter import Tk, filedialog, messagebox
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
import cv2

def load_trained_model():
    """Abre un diálogo tkinter para seleccionar el archivo del modelo y devuelve su ruta."""
    print("\n[DEBUG] Iniciando selección del modelo...")

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    print("[DEBUG] Ventana de Tkinter creada")

    try:
        print("[DEBUG] Mostrando diálogo de selección de archivo...")
        model_path = filedialog.askopenfilename(
            title="Seleccione el modelo pre-entrenado (.h5 o .keras)",
            filetypes=[("Model files", "*.h5;*.keras"), ("H5 files", "*.h5"), ("Keras files", "*.keras")],
            initialdir=os.getcwd()
        )
        print(f"[DEBUG] Ruta del modelo seleccionada: {model_path}")
        return model_path
    except Exception as e:
        print(f"[DEBUG] Error al seleccionar el archivo: {str(e)}")
        raise
    finally:
        root.destroy()
        

def select_images_directory():
    """Prompt user to select a directory containing images."""
    root = Tk()
    root.withdraw()
    
    directory = filedialog.askdirectory(
        title="Seleccione la carpeta con las imágenes a evaluar"
    )
    
    if not directory:
        raise ValueError("No se seleccionó ninguna carpeta.")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(directory) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        raise ValueError("No se encontraron imágenes en el directorio seleccionado.")
    
    return directory, image_files

def load_ground_truth(roughness_param='Ra'):
    """Load ground truth values from 'Validacion' sheet in Excel file.
    
    Args:
        roughness_param (str): The roughness parameter to load ('Ra' or 'Rz')
    """
    root = Tk()
    root.withdraw()
    
    excel_path = filedialog.askopenfilename(
        title="Seleccione el archivo Excel con las mediciones reales",
        filetypes=[("Excel files", "*.xlsx *.xls")],
        initialdir=os.getcwd()
    )
    
    if not excel_path:
        raise ValueError("No se seleccionó ningún archivo Excel.")
    
    try:
        # Try to get sheet names to check if 'Validacion' exists
        excel = pd.ExcelFile(excel_path)
        if 'Validacion' not in excel.sheet_names:
            available_sheets = ", ".join(str(s) for s in excel.sheet_names)
            raise ValueError(f"No se encontró la hoja 'Validacion' en el archivo Excel. Hojas disponibles: {available_sheets}")
        
        # Try reading the validation sheet. Use explicit engine when needed.
        try:
            df = pd.read_excel(excel_path, sheet_name='Validacion')
        except Exception:
            if excel_path.lower().endswith('.xls'):
                df = pd.read_excel(excel_path, sheet_name='Validacion', engine='xlrd')
            else:
                df = pd.read_excel(excel_path, sheet_name='Validacion', engine='openpyxl')
        
        # Clean column names (remove extra spaces, convert to lowercase)
        df.columns = df.columns.str.strip().str.lower()
        
        # Check required columns (case insensitive)
        param_lower = roughness_param.lower()
        required_columns = ['nombre_imagen', param_lower]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Provide more helpful error message with available columns
            available_cols = ", ".join([f'"{c}"' for c in df.columns])
            raise ValueError(
                f"No se encontró la columna '{param_lower}' en el archivo Excel.\n"
                f"Columnas disponibles: {available_cols}\n"
                f"Asegúrese de que el archivo Excel contenga una columna con el nombre '{param_lower}' (no sensible a mayúsculas)"
            )
        
        # Convert parameter column to numeric, handling both comma and period as decimal separator
        try:
            # First try direct conversion
            df[param_lower] = pd.to_numeric(df[param_lower], errors='raise')
        except ValueError:
            try:
                # If that fails, try replacing comma with period
                df[param_lower] = df[param_lower].astype(str).str.replace(',', '.').astype(float)
            except Exception as e:
                raise ValueError(f"No se pudo convertir la columna '{roughness_param}' a valores numéricos. Asegúrese de que los valores usan coma o punto como separador decimal. Error: {str(e)}")
        
        # Rename the parameter column to 'rugosidad' for consistency
        df = df.rename(columns={param_lower: 'rugosidad'})
        
        # Print first few rows for debugging
        print(f"\nPrimeras filas de datos cargados para {roughness_param}:")
        print(df[['nombre_imagen', 'rugosidad']].head())
                
        return df
    except Exception as e:
        raise Exception(f"Error al leer el archivo Excel: {str(e)}")

def load_and_prepare_image(image_path, target_size=(130, 390)):
    """Carga y preprocesa una imagen en escala de grises para el modelo.

    target_size sigue la convención de OpenCV: (ancho, alto) = (130, 390).
    El array resultante tiene shape (1, 390, 130, 1) — lo que espera Keras.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Load image in grayscale using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}. The file might be corrupted or in an unsupported format.")
        
        # Get image dimensions
        height, width = img.shape
        print(f"Processing: {os.path.basename(image_path)} - Original size: {width}x{height} pixels")
        
        # Resize if needed
        if (width, height) != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            print(f"  Resized to: {target_size[0]}x{target_size[1]} pixels")
        
        # Convert to float32 and normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch and channel dimensions (model expects 4D: batch, height, width, channels)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=-1)  # Add channel dimension (1 for grayscale)
        
        return img
        
    except Exception as e:
        raise Exception(f"Error processing image {os.path.basename(image_path)}: {str(e)}")


def preprocess_image(image_path, target_size=(130, 390)):
    """Alias de load_and_prepare_image."""
    return load_and_prepare_image(image_path, target_size)

# Categorías de grano compatibles
GRAIN_CATEGORIES = [40, 80, 120, 180]
NUM_GRAIN_CATEGORIES = len(GRAIN_CATEGORIES)

def extract_grain(filename):
    """Extract and return grain information from filename as a one-hot encoded vector."""
    # Try to find grain in format G40, G80, etc.
    grain_str = None
    for g in [40, 80, 120, 180]:
        if f'G{g}' in str(filename).upper():
            grain_str = g
            break
    
    # If not found, try to find just the number
    if grain_str is None:
        for g in [40, 80, 120, 180]:
            if str(g) in str(filename):
                grain_str = g
                break
    
    # If still not found, try to extract any number that matches grain sizes
    if grain_str is None:
        import re
        numbers = re.findall(r'\d+', str(filename))
        for num in numbers:
            num_int = int(num)
            if num_int in GRAIN_CATEGORIES:
                grain_str = num_int
                break
    
    # Default to 80 if grain not found
    if grain_str is None:
        print(f"Warning: No se pudo determinar el grano para {filename}, usando 80 por defecto")
        grain_str = 80
    
    # Convert to one-hot encoding
    grain_idx = GRAIN_CATEGORIES.index(int(grain_str))
    grain_one_hot = np.zeros(NUM_GRAIN_CATEGORIES, dtype=np.float32)
    grain_one_hot[grain_idx] = 1.0
    
    return grain_one_hot, int(grain_str)

def plot_comparison_point_to_point(y_true, y_pred, output_dir, roughness_param='Ra', title_suffix=''):
    """Create a point-to-point comparison plot between true and predicted values.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        output_dir: Directory to save the plot
        roughness_param: The roughness parameter ('Ra' or 'Rz')
        title_suffix: Optional suffix for the output filename
    """
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot with points
    indices = np.arange(len(y_true))
    plt.scatter(indices, y_true, color='red', alpha=0.6,
                label=f'{roughness_param} Medido', s=50)
    plt.scatter(indices, y_pred, color='blue', alpha=0.6,
                label=f'{roughness_param} Estimado', s=50)

    # Connect corresponding points with lines
    for i in range(len(y_true)):
        plt.plot([i, i], [y_true[i], y_pred[i]], 'gray', alpha=0.3, linestyle='--')

    # Calculate error metrics
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Add labels and title with increased font sizes
    plt.title(f'Comparación Punto a Punto - {roughness_param}\nMAE: {mae:.3f} µm, MAPE: {mape:.1f}%',
              fontsize=16, pad=15)
    plt.xlabel('Índice de la Muestra', fontsize=14, labelpad=10)
    plt.ylabel(f'{roughness_param} (µm)', fontsize=14, labelpad=10)
    plt.legend(fontsize=12, frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Increase tick label size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'comparacion_punto_a_punto_{roughness_param.lower()}{title_suffix}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_sorted_comparison(y_true, y_pred, output_dir, roughness_param='Ra', title_suffix=''):
    """Create a comparison plot with measurements sorted from smallest to largest.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        output_dir: Directory to save the plot
        roughness_param: The roughness parameter ('Ra' or 'Rz')
        title_suffix: Optional suffix for the output filename
    """
    plt.figure(figsize=(12, 6))
    
    # Sort both true and predicted values based on true values
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    # Create scatter plot with points
    indices = np.arange(len(y_true))
    plt.scatter(indices, y_true_sorted, color='red', alpha=0.6,
                label=f'{roughness_param} Medido', s=50)
    plt.scatter(indices, y_pred_sorted, color='blue', alpha=0.6,
                label=f'{roughness_param} Estimado', s=50)
    
    # Connect corresponding points with lines
    for i in range(len(y_true)):
        plt.plot([i, i], [y_true_sorted[i], y_pred_sorted[i]], 'gray', alpha=0.3, linestyle='--')
    
    # Calculate error metrics
    mae = mean_absolute_error(y_true_sorted, y_pred_sorted)
    mape = mean_absolute_percentage_error(y_true_sorted, y_pred_sorted) * 100
    
    # Add labels and title
    plt.title(f'Comparación Ordenada (Menor a Mayor) - {roughness_param}\nMAE: {mae:.3f} µm, MAPE: {mape:.1f}%')
    plt.xlabel(f'Muestras Ordenadas por Valor de {roughness_param}')
    plt.ylabel(f'{roughness_param} (µm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'comparacion_ordenada_{roughness_param.lower()}{title_suffix}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_ra_by_grain(df, output_dir, roughness_param='Ra', title_suffix=''):
    """Create a boxplot comparing roughness values by grain type.
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the plot
        roughness_param: Roughness parameter to plot ('Ra' or 'Rz')
        title_suffix: Optional suffix for the output filename
    """
    plt.figure(figsize=(12, 6))
    
    # Ensure grain order and format
    grain_order = ['G40', 'G80', 'G120', 'G180']
    
    # Convert grain values to the expected format (add 'G' prefix if missing)
    df = df.copy()
    df['Grano'] = df['Grano'].apply(
        lambda x: f"G{int(x)}" if str(x).isdigit() else (
            f"G{x}" if not str(x).startswith('G') and str(x)[1:].isdigit() else x
        )
    )
    
    # Create mapping for x-tick labels (G -> P)
    grain_labels = [f"P{g[1:]}" for g in grain_order]
    
    # Convert to categorical with proper ordering
    df['Grano'] = pd.Categorical(df['Grano'], categories=grain_order, ordered=True)
    df = df.sort_values('Grano')
    
    # Create column names based on selected parameter
    real_col = f'{roughness_param}_real'
    pred_col = f'{roughness_param}_estimado'
    
    # Check if required columns exist
    if real_col not in df.columns or pred_col not in df.columns:
        print(f"[WARNING] No se encontraron las columnas {real_col} o {pred_col} en los datos")
        return None
    
    # Prepare data for plotting
    plot_data = df[['Grano', real_col, pred_col]].copy()
    plot_data = plot_data.rename(columns={
        real_col: f'{roughness_param}_Medido',
        pred_col: f'{roughness_param}_Estimado'
    })
    
    # Melt the dataframe for easier plotting
    df_melted = pd.melt(plot_data, id_vars=['Grano'], 
                       value_vars=[f'{roughness_param}_Medido', f'{roughness_param}_Estimado'],
                       var_name='Tipo', value_name=roughness_param)
    
    # Create boxplot
    sns.boxplot(x='Grano', y=roughness_param, hue='Tipo', data=df_melted, 
               palette={f'{roughness_param}_Medido': 'lightblue', 
                       f'{roughness_param}_Estimado': 'peachpuff'},
               width=0.7, dodge=True, legend=False, showfliers=False)
    
    # Add individual points with some jitter
    sns.stripplot(x='Grano', y=roughness_param, hue='Tipo', data=df_melted,
                  dodge=True, jitter=True, alpha=0.7, size=6,
                  palette={f'{roughness_param}_Medido': 'blue', 
                          f'{roughness_param}_Estimado': 'orange'})
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Medido',
               markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Estimado',
               markerfacecolor='peachpuff', markersize=10)
    ]
    # Legend will be updated later with larger font
    
    # Set x-tick labels to use P instead of G
    plt.xticks(ticks=range(len(grain_order)), labels=grain_labels)
    
    # Add labels and title with increased font sizes
    plt.title(f'Comparación de {roughness_param} por Grano de Lija', fontsize=16, pad=15)
    plt.xlabel('Tamaño de grano de lija', fontsize=14, labelpad=10)
    plt.ylabel(f'{roughness_param} (µm)', fontsize=14, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Increase tick label size
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    # Increase legend size
    plt.legend(handles=legend_elements, title='Tipo de Medición', 
              title_fontsize=13, fontsize=13, frameon=True, framealpha=0.9)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{roughness_param.lower()}_por_grano{title_suffix}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[DEBUG] Gráfico de {roughness_param} por grano guardado en: {plot_path}")
    return plot_path

def create_grain_analysis_table(df, roughness_param='Ra'):
    """
    Create a table with grain analysis including count, average measurements, 
    average predictions, absolute differences, MAE, and MSE.
    
    Args:
        df: DataFrame containing the results with grain information
        roughness_param: The roughness parameter being analyzed ('Ra' or 'Rz')
        
    Returns:
        DataFrame: Table with grain analysis
    """
    # Asegurarse de que las columnas necesarias existen
    required_columns = [f'{roughness_param}_real', f'{roughness_param}_estimado', 'grano']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {', '.join(required_columns)}")
    
    # Calcular MAE y MSE para cada grupo de grano
    def calculate_metrics(group):
        y_true = group[f'{roughness_param}_real']
        y_pred = group[f'{roughness_param}_estimado']
        return pd.Series({
            'Cantidad': len(group),
            'Promedio_mediciones': y_true.mean(),
            'Promedio_estimaciones': y_pred.mean(),
            'MAE': np.mean(np.abs(y_true - y_pred)),
            'MSE': np.mean((y_true - y_pred) ** 2)
        })
    
    # Agrupar por grano y calcular métricas
    grain_analysis = df.groupby('grano').apply(calculate_metrics).reset_index()
    
    # Calcular diferencia absoluta
    grain_analysis['Diferencia_absoluta'] = abs(
        grain_analysis['Promedio_mediciones'] - grain_analysis['Promedio_estimaciones']
    )
    
    # Reordenar columnas
    column_order = [
        'grano', 'Cantidad', 'Promedio_mediciones', 'Promedio_estimaciones',
        'Diferencia_absoluta', 'MAE', 'MSE'
    ]
    grain_analysis = grain_analysis[column_order]
    
    # Renombrar columnas para mejor visualización
    grain_analysis = grain_analysis.rename(columns={
        'grano': 'Grano de lijado',
        'Cantidad': 'Cantidad de datos',
        'Promedio_mediciones': f'Promedio de mediciones ({roughness_param})',
        'Promedio_estimaciones': f'Promedio de estimaciones ({roughness_param})',
        'Diferencia_absoluta': 'Diferencia absoluta',
        'MAE': f'MAE ({roughness_param}, µm)',
        'MSE': f'MSE ({roughness_param}, µm²)'
    })
    
    # Ordenar por tamaño de grano
    grain_analysis = grain_analysis.sort_values('Grano de lijado')
    
    # Redondear columnas numéricas a 4 decimales
    numeric_cols = grain_analysis.select_dtypes(include=[np.number]).columns
    grain_analysis[numeric_cols] = grain_analysis[numeric_cols].round(4)
    
    return grain_analysis


def plot_ra_evolution(df, output_dir, roughness_param='Ra', title_suffix=''):
    """Create a plot showing the evolution of average roughness by grain type.
    
    Args:
        df (DataFrame): DataFrame containing the data
        output_dir (str): Directory to save the plot
        roughness_param (str): The roughness parameter ('Ra' or 'Rz')
        title_suffix (str): Optional suffix for the output filename
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure grain values are in the correct format
    df['Grano'] = df['Grano'].apply(
        lambda x: f"G{int(x)}" if str(x).isdigit() else (
            f"G{x}" if not str(x).startswith('G') and str(x)[1:].isdigit() else x
        )
    )
    
    # Calculate averages by grain
    promedios = df.groupby('Grano').agg({
        f'{roughness_param}_Medido': 'mean',
        f'{roughness_param}_Estimado': 'mean'
    }).reset_index()
    
    # Order grains
    granos_ordenados = ['G40', 'G80', 'G120', 'G180']
    promedios['Grano'] = pd.Categorical(promedios['Grano'], 
                                      categories=granos_ordenados, 
                                      ordered=True)
    promedios = promedios.sort_values('Grano')
    
    # Create figure with larger size
    plt.figure(figsize=(12, 8))
    
    # Plot measured and estimated values with enhanced visibility
    plt.plot(promedios['Grano'].astype(str), promedios[f'{roughness_param}_Medido'], 
             'o-', color='red', label=f'{roughness_param} Medido', 
             markersize=10, linewidth=2.5, markerfacecolor='white', markeredgewidth=2)
             
    plt.plot(promedios['Grano'].astype(str), promedios[f'{roughness_param}_Estimado'], 
             's-', color='blue', label=f'{roughness_param} Estimado', 
             markersize=10, linewidth=2.5, markerfacecolor='white', markeredgewidth=2)
    
    # Add value labels on top of the points
    for i, (medido, estimado) in enumerate(zip(promedios[f'{roughness_param}_Medido'], promedios[f'{roughness_param}_Estimado'])):
        plt.text(i, medido + 0.1, f'{medido:.2f}', ha='center', va='bottom', color='red', fontweight='bold')
        plt.text(i, estimado + 0.1, f'{estimado:.2f}', ha='center', va='bottom', color='blue', fontweight='bold')
    
    # Add labels and title with enhanced styling
    plt.title(f'Evolución del Promedio de {roughness_param} por Grano de Lija', fontsize=14, pad=20)
    plt.xlabel('Grano de Lija', fontsize=12, labelpad=10)
    plt.ylabel(f'{roughness_param} Promedio (µm)', fontsize=12, labelpad=10)
    
    # Customize grid and legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'evolucion_{roughness_param.lower()}{title_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_r2(y_true, y_pred, output_dir, roughness_param='Ra', title_suffix=''):
    """Generate and save R² plot.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        output_dir: Directory to save the plot
        roughness_param: The roughness parameter ('Ra' or 'Rz')
        title_suffix: Optional suffix for the output filename
        
    Returns:
        tuple: (plot_path, r2_score)
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    # Create scatter plot with regression line
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.6})
    
    # Add labels and title with increased font sizes
    plt.xlabel(f'Valores Medidos de {roughness_param} (µm)', fontsize=14, labelpad=10)
    plt.ylabel(f'Valores Estimados de {roughness_param} (µm)', fontsize=14, labelpad=10)
    plt.title(f'Validación del Modelo - {roughness_param}\nR² = {r2:.4f}', fontsize=16, pad=15)
    
    # Add 1:1 line for reference
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add legend and adjust tick sizes
    plt.legend(fontsize=12, loc='upper left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'r2_plot_{roughness_param.lower()}{title_suffix}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path, r2

def select_roughness_parameter():
    """Prompt user to select Ra or Rz roughness parameter."""
    import sys
    
    while True:
        try:
            print("\nSeleccione el parámetro de rugosidad a evaluar:")
            print("1. Ra (Rugosidad media aritmética)")
            print("2. Rz (Altura máxima de rugosidad)")
            print("3. Salir")
            
            choice = input("\nIngrese el número de su elección (1-3): ").strip()
            
            if choice == '1':
                return 'Ra'
            elif choice == '2':
                return 'Rz'
            elif choice.lower() in ['3', 'salir', 'exit', 'q', 'quit']:
                print("Saliendo del programa...")
                sys.exit(0)
            else:
                print("\n❌ Opción no válida. Por favor, ingrese 1 para Ra, 2 para Rz o 3 para salir.")
                continue
                
        except KeyboardInterrupt:
            print("\n\nOperación cancelada por el usuario.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error inesperado: {str(e)}")
            continue

def main():
    try:
        print("=== Validación de Modelo de Rugosidad ===")
        
        # Step 0: Select roughness parameter
        print("[DEBUG] Solicitando parámetro de rugosidad...")
        roughness_param = select_roughness_parameter()
        print(f"\n[DEBUG] Parámetro seleccionado: {roughness_param}")
        
        # Step 1: Load the trained model
        print("[DEBUG] Cargando modelo entrenado...")
        try:
            model_path = load_trained_model()
            print("[DEBUG] Ruta del modelo obtenida, cargando el modelo...")
            
            # Cargar el modelo con manejo de errores mejorado
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
                print("[DEBUG] Modelo cargado exitosamente")
            except Exception as e:
                print(f"[DEBUG] Error al cargar el modelo: {str(e)}")
                print("[DEBUG] Intentando cargar con métricas personalizadas...")
                custom_objects = {'r2_score': r2_score, 'mean_squared_error': mean_squared_error}
                model = load_model(model_path, custom_objects=custom_objects, compile=True)
                print("[DEBUG] Modelo cargado con métricas personalizadas")
        except Exception as e:
            print(f"[DEBUG] Error en la carga del modelo: {str(e)}")
            raise
        
        # Step 2: Select images directory
        print("[DEBUG] Solicitando directorio de imágenes...")
        images_dir, image_files = select_images_directory()
        print(f"[DEBUG] Se encontraron {len(image_files)} imágenes en el directorio.")
        print(f"[DEBUG] Ruta del directorio: {images_dir}")
        print(f"[DEBUG] Archivos encontrados (primeros 3): {image_files[:3]}")
        
        # Step 3: Load ground truth data with the selected parameter
        df_ground_truth = load_ground_truth(roughness_param)
        
        # Create results directory
        results_dir = os.path.join(os.getcwd(), 'resultados_validacion')
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results list
        results = []
        
        # Convert ground truth to lowercase for case-insensitive matching
        df_ground_truth['nombre_archivo'] = df_ground_truth['nombre_imagen'].str.lower()
        
        # Process each image
        processed_count = 0
        for img_file in image_files:
            try:
                # Get base name without extension and convert to lowercase
                base_name = os.path.splitext(img_file)[0].lower()
                
                # Try different matching strategies
                # First try exact match
                matches = df_ground_truth[df_ground_truth['nombre_archivo'] == base_name]
                
                # If no exact match, try partial matches
                if matches.empty:
                    matches = df_ground_truth[
                        df_ground_truth['nombre_archivo'].str.contains(base_name, na=False) |
                        df_ground_truth['nombre_archivo'].apply(lambda x: base_name in str(x))
                    ]
                
                if matches.empty:
                    print(f"Advertencia: No se encontró coincidencia para la imagen: {img_file}")
                    print(f"Nombres disponibles en Excel (primeros 5): {df_ground_truth['nombre_imagen'].head().tolist()}")
                    continue
                    
                # Take the first match if multiple
                gt_row = matches.iloc[0]
                try:
                    # Get the roughness value (column is now always 'rugosidad' after loading)
                    rugosidad_real = float(gt_row['rugosidad'])
                except (KeyError, ValueError) as e:
                    print(f"\nError al obtener el valor de rugosidad para {img_file}:")
                    print(f"Contenido de la fila encontrada: {gt_row.to_dict()}")
                    print(f"Columnas disponibles: {', '.join(gt_row.index.tolist())}")
                    raise ValueError(f"No se pudo obtener el valor de rugosidad ({roughness_param}) para la imagen {img_file}")
                
                # Preprocess image and get grain information
                img_path = os.path.join(images_dir, img_file)
                try:
                    # Preprocess image
                    img_array = preprocess_image(img_path)
                    
                    # Get grain one-hot encoded and as integer
                    grain_one_hot, grain_value = extract_grain(img_file)
                    
                    # Reshape grain for prediction
                    grain_input = np.expand_dims(grain_one_hot, axis=0)
                    
                    # Make prediction with multimodal input
                    rugosidad_predicha = float(model.predict(
                        {'image_input': img_array, 'grain_input': grain_input}, 
                        verbose=0
                    )[0][0])
                    
                    # Store results with the selected roughness parameter
                    results.append({
                        'archivo': img_file,
                        'grano': grain_value,
                        f'{roughness_param}_real': rugosidad_real,
                        f'{roughness_param}_estimado': rugosidad_predicha,
                        'diferencia_absoluta': abs(rugosidad_real - rugosidad_predicha),
                    })
                    
                    print(f"Procesado: {img_file} | Real: {rugosidad_real:.4f} µm | Estimado: {rugosidad_predicha:.4f} µm | Diferencia: {abs(rugosidad_real - rugosidad_predicha):.4f} µm")
                    processed_count += 1
                    
                except Exception as img_error:
                    print(f"Error al procesar la imagen {img_file}: {str(img_error)}")
                    
            except Exception as e:
                print(f"Error procesando {img_file}: {str(e)}")
        
        if processed_count == 0:
            print("\nNo se pudo procesar ninguna imagen. Posibles causas:")
            print("1. Los nombres de las imágenes no coinciden con los del Excel")
            print("2. Las extensiones de las imágenes no son compatibles (.jpg, .jpeg, .png, .bmp, .tif, .tiff)")
            print("3. El archivo Excel no contiene las columnas 'nombre_imagen' y 'Ra'")
            print("4. Las imágenes no están en el formato esperado por el modelo")
            print("\nNombres de archivo en el Excel (primeros 5):")
            print(df_ground_truth['nombre_imagen'].head().to_string(index=False))
            print("\nNombres de archivo en la carpeta (primeros 5):")
            print("\n".join(image_files[:5]))
            return
        
        if not results:
            raise ValueError("No se pudo procesar ninguna imagen con los datos proporcionados.")
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Calculate metrics
        y_true = df_results[f'{roughness_param}_real'].to_numpy(dtype=float)
        y_pred = df_results[f'{roughness_param}_estimado'].to_numpy(dtype=float)
        
        # Generate and save all plots with the selected parameter
        plot_path, r2 = plot_r2(y_true, y_pred, results_dir, roughness_param=roughness_param)
        print(f"\nGráfico R² guardado en: {plot_path}")
        
        # Create point-to-point comparison plot with proper labels
        point_plot_path = plot_comparison_point_to_point(
            y_true, y_pred, results_dir, roughness_param=roughness_param
        )
        print(f"Gráfico de comparación punto a punto guardado en: {point_plot_path}")
        
        # Create sorted comparison plot with proper labels
        sorted_plot_path = plot_sorted_comparison(
            y_true, y_pred, results_dir, roughness_param=roughness_param
        )
        print(f"Gráfico de comparación ordenada guardado en: {sorted_plot_path}")
        
        # Create roughness by grain plot for the selected parameter
        # Ensure consistent column names for the plot function
        plot_df = df_results.copy()
        if 'Grano' not in plot_df.columns and 'grano' in plot_df.columns:
            plot_df['Grano'] = plot_df['grano']
        
        grain_plot_path = plot_ra_by_grain(
            plot_df, 
            output_dir=results_dir,
            roughness_param=roughness_param,
            title_suffix=f'_{len(image_files)}imgs'
        )
        
        if grain_plot_path:
            print(f"Gráfico de {roughness_param} por grano guardado en: {grain_plot_path}")
        
        # Create evolution plot for the selected parameter
        # Ensure consistent column names for the evolution plot function
        evolution_plot_path = plot_ra_evolution(
            plot_df.rename(columns={
                f'{roughness_param}_real': f'{roughness_param}_Medido',
                f'{roughness_param}_estimado': f'{roughness_param}_Estimado'
            }), 
            output_dir=results_dir,
            roughness_param=roughness_param,
            title_suffix=f'_{len(image_files)}imgs'
        )
        if evolution_plot_path:
            print(f"Gráfico de evolución de {roughness_param} guardado en: {evolution_plot_path}")
        
        # Save results to Excel with appropriate filename based on roughness parameter
        results_file = os.path.join(results_dir, f'resultados_validacion_{roughness_param}.xlsx')
        
        # Ensure the output directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the main results
        df_results.to_excel(results_file, index=False)
        print(f"\nResultados guardados en: {results_file}")
        
        # Create grain analysis table
        grain_analysis = create_grain_analysis_table(df_results, roughness_param)
        
        # Create metrics table
        metrics = pd.DataFrame({
            'Métrica': [f'R² ({roughness_param})', 
                       f'RMSE ({roughness_param}, µm)', 
                       f'MAE ({roughness_param}, µm)',
                       f'MSE ({roughness_param}, µm²)'],  # Agregar MSE
            'Valor': [
                r2,
                np.sqrt(mean_squared_error(y_true, y_pred)),
                np.mean(np.abs(y_true - y_pred)),
                mean_squared_error(y_true, y_pred)  # Agregar cálculo de MSE
            ]
        })
        
        # Save all results to Excel
        with pd.ExcelWriter(results_file, engine='openpyxl', mode='a') as writer:
            metrics.to_excel(writer, index=False, sheet_name='Métricas')
            grain_analysis.to_excel(writer, index=False, sheet_name='Análisis por grano')
            
            # Auto-adjust column widths for the grain analysis sheet
            worksheet = writer.sheets['Análisis por grano']
            for idx, col in enumerate(grain_analysis.columns):
                max_length = max(
                    grain_analysis[col].astype(str).apply(len).max(),
                    len(str(col))
                ) + 2  # Add a little extra space
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 30)
        
        print(f"\nProceso completado exitosamente!")
        print(f"- Resultados guardados en: {results_file}")
        print(f"- Gráfico R² guardado en: {plot_path}")
        print(f"- R² del modelo: {r2:.4f}")
        
        # Show completion message
        messagebox.showinfo(
            "Validación Completada",
            f"Proceso de validación completado.\n\n"
            f"Parámetro evaluado: {roughness_param}\n"
            f"R² del modelo: {r2:.4f}\n"
            f"MAE: {metrics.loc[metrics['Métrica'].str.startswith('MAE'), 'Valor'].values[0]:.4f} µm\n"
            f"RMSE: {metrics.loc[metrics['Métrica'].str.startswith('RMSE'), 'Valor'].values[0]:.4f} µm\n"
            f"MSE: {metrics.loc[metrics['Métrica'].str.startswith('MSE'), 'Valor'].values[0]:.4f} µm²\n\n"
            f"Resultados guardados en:\n{results_file}"
        )
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)

if __name__ == "__main__":
    main()
