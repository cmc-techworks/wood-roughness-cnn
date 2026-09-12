import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import numpy as np

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel file",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    return file_path

def extract_p_value(filename):
    match = re.search(r'P(\d+)', str(filename))  # Convert to string in case of NaN
    return f"P{match.group(1)}" if match else "Unknown"

def create_plot(df, parameter, title):
    plt.figure(figsize=(14, 7))
    
    # Create a boxplot using seaborn with hue for P_Group
    sns.boxplot(
        x='Grano',
        y=parameter,
        hue='P_Group',
        data=df,
        palette='Set2'
    )
    
    # Add a title with the total number of samples
    total_samples = len(df)
    plt.title(f'Distribución de {parameter} por Grano y Grupo P\n(Total de muestras: {total_samples})', fontsize=14)
    
    plt.xlabel('Grano', fontsize=12)
    plt.ylabel(f'{parameter} (μm)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Improve legend
    plt.legend(
        title='Grupo P', 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        borderaxespad=0
    )
    
    plt.tight_layout()
    plt.show()

def create_summary_table(df, writer):
    # Calculate statistics for Ra and Rz
    stats_ra = df.groupby('Grano')['Ra'].agg(['count', 'min', 'max', 'mean', 'std']).round(3)
    stats_rz = df.groupby('Grano')['Rz'].agg(['min', 'max', 'mean', 'std']).round(3)
    
    # Rename columns for clarity
    stats_ra = stats_ra.rename(columns={
        'count': 'Muestras',
        'min': 'Ra Mínimo',
        'max': 'Ra Máximo',
        'mean': 'Ra Promedio',
        'std': 'Ra Desv. Estándar'
    })
    
    stats_rz = stats_rz.rename(columns={
        'min': 'Rz Mínimo',
        'max': 'Rz Máximo',
        'mean': 'Rz Promedio',
        'std': 'Rz Desv. Estándar'
    })
    
    # Combine the statistics
    summary = pd.concat([stats_ra, stats_rz], axis=1)
    
    # Reorder columns
    summary = summary[['Muestras', 'Ra Mínimo', 'Ra Máximo', 'Ra Promedio', 'Ra Desv. Estándar',
                      'Rz Mínimo', 'Rz Máximo', 'Rz Promedio', 'Rz Desv. Estándar']]
    
    # Add total row
    total_row = pd.DataFrame({
        'Muestras': [df.shape[0]],
        'Ra Mínimo': [df['Ra'].min()],
        'Ra Máximo': [df['Ra'].max()],
        'Ra Promedio': [df['Ra'].mean()],
        'Ra Desv. Estándar': [df['Ra'].std()],
        'Rz Mínimo': [df['Rz'].min()],
        'Rz Máximo': [df['Rz'].max()],
        'Rz Promedio': [df['Rz'].mean()],
        'Rz Desv. Estándar': [df['Rz'].std()]
    }, index=['TOTAL']).round(3)
    
    summary = pd.concat([summary, total_row])
    
    # Write to Excel with basic formatting
    summary.to_excel(writer, sheet_name='Resumen_Estadistico')
    
    # Auto-adjust column widths (basic version)
    worksheet = writer.sheets['Resumen_Estadistico']
    for idx, col in enumerate(summary.columns):
        max_length = max(
            len(str(col)),
            summary[col].astype(str).str.len().max()
        )
        worksheet.column_dimensions[chr(65 + idx + 1)].width = max_length + 2
    
    # Set width for index column
    max_index_length = max([len(str(i)) for i in summary.index] + [len('Grano')])
    worksheet.column_dimensions['A'].width = max_index_length + 2
    
    return summary

def plot_grain_distribution():
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting...")
        return
    
    try:
        # Read both sheets
        df_sin_outliers = pd.read_excel(file_path, sheet_name="Sin Outliers")
        df_validacion = pd.read_excel(file_path, sheet_name="Validacion")
        
        # Add a column to identify the source of each row
        df_sin_outliers['Source'] = 'Sin Outliers'
        df_validacion['Source'] = 'Validacion'
        
        # Combine both dataframes
        df_combined = pd.concat([df_sin_outliers, df_validacion], ignore_index=True)
        
        # Extract P value from nombre_imagen for both dataframes
        df_combined['P_Group'] = df_combined['nombre_imagen'].apply(extract_p_value)
        df_sin_outliers['P_Group'] = df_sin_outliers['nombre_imagen'].apply(extract_p_value)
        
        # Create plots for Ra and Rz
        create_plot(df_combined, 'Ra', 'Distribución de Ra')
        create_plot(df_combined, 'Rz', 'Distribución de Rz')
        
        # Create a Pandas Excel writer using the existing file
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # Create summary table using only 'Sin Outliers' data
            summary = create_summary_table(df_sin_outliers, writer)
            
        print("\nResumen estadístico (solo datos 'Sin Outliers') guardado en la hoja 'Resumen_Estadistico'")
        print("\nEstadísticas por grupo (Ra) - Todos los datos:")
        print(df_combined.groupby(['Grano', 'P_Group'])['Ra'].describe())
        print("\nEstadísticas por grupo (Rz) - Todos los datos:")
        print(df_combined.groupby(['Grano', 'P_Group'])['Rz'].describe())
        print("\nEstadísticas por grupo (Ra) - Solo 'Sin Outliers':")
        print(df_sin_outliers.groupby(['Grano', 'P_Group'])['Ra'].describe())
        print("\nEstadísticas por grupo (Rz) - Solo 'Sin Outliers':")
        print(df_sin_outliers.groupby(['Grano', 'P_Group'])['Rz'].describe())
        
        messagebox.showinfo("Proceso completado", 
                          "Gráficos mostrados y resumen estadístico guardado exitosamente.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    plot_grain_distribution()