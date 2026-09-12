import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def analizar_estadisticas():
    # Configurar la ventana para seleccionar archivo
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    
    try:
        # Abrir diálogo para seleccionar archivo Excel
        file_path = filedialog.askopenfilename(
            title="Seleccione el archivo Excel",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        
        if not file_path:
            messagebox.showinfo("Información", "No se seleccionó ningún archivo.")
            return
        
        # Leer el archivo Excel
        with pd.ExcelFile(file_path) as xls:
            if 'Todo' not in xls.sheet_names:
                messagebox.showerror("Error", 'No se encontró la hoja "Todo" en el archivo seleccionado.')
                return
            
            df = pd.read_excel(xls, sheet_name='Todo')
            
            # Verificar que existan las columnas necesarias
            required_columns = ['Ra', 'Rz', 'Grano']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                messagebox.showerror("Error", f'No se encontraron las columnas: {", ".join(missing_columns)}')
                return
            
            # Filtrar solo los granos de interés (40, 80, 120, 180)
            granos_interes = [40, 80, 120, 180]
            df = df[df['Grano'].isin(granos_interes)].copy()
            
            # Crear un nuevo ExcelWriter para guardar los resultados
            output_path = os.path.join(os.path.dirname(file_path), 'todos_los_datos.xlsx')
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Lista para almacenar los DataFrames de resultados
                resultados = []
                
                # Función para detectar outliers usando IQR
                def detect_outliers(series):
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    return ~series.between(lower_bound, upper_bound)
                
                # Para cada grano, calcular estadísticas
                for grano in sorted(df['Grano'].unique()):
                    df_grano = df[df['Grano'] == grano].copy()
                    
                    # Detectar outliers
                    outliers_ra = detect_outliers(df_grano['Ra'])
                    outliers_rz = detect_outliers(df_grano['Rz'])
                    
                    # Calcular estadísticas para Ra
                    ra_stats = df_grano['Ra'].describe()
                    ra_iqr = ra_stats['75%'] - ra_stats['25%']
                    
                    # Calcular estadísticas para Rz
                    rz_stats = df_grano['Rz'].describe()
                    rz_iqr = rz_stats['75%'] - rz_stats['25%']
                    
                    # Contar outliers
                    num_outliers_ra = outliers_ra.sum()
                    num_outliers_rz = outliers_rz.sum()
                    num_total_outliers = (outliers_ra | outliers_rz).sum()
                    num_datos_normales = len(df_grano) - num_total_outliers
                    
                    # Crear DataFrame con los resultados
                    resultados_grano = pd.DataFrame({
                        'Métrica': ['Ra', 'Rz'],
                        'Grano': [grano, grano],
                        'Mínimo': [ra_stats['min'], rz_stats['min']],
                        'Máximo': [ra_stats['max'], rz_stats['max']],
                        'Promedio': [ra_stats['mean'], rz_stats['mean']],
                        'Desviación Estándar': [ra_stats['std'], rz_stats['std']],
                        'Q1': [ra_stats['25%'], rz_stats['25%']],
                        'Mediana': [ra_stats['50%'], rz_stats['50%']],
                        'Q3': [ra_stats['75%'], rz_stats['75%']],
                        'IQR': [ra_iqr, rz_iqr],
                        'Límite Inferior': [ra_stats['25%'] - 1.5*ra_iqr, rz_stats['25%'] - 1.5*rz_iqr],
                        'Límite Superior': [ra_stats['75%'] + 1.5*ra_iqr, rz_stats['75%'] + 1.5*rz_iqr],
                        'Número de Outliers': [num_outliers_ra, num_outliers_rz],
                        'Número de Datos Normales': [len(df_grano) - num_outliers_ra, len(df_grano) - num_outliers_rz],
                        'Total de Datos': [len(df_grano), len(df_grano)]
                    })
                    
                    resultados.append(resultados_grano)
                    
                    # Guardar los datos del grano en una hoja separada
                    df_grano.to_excel(writer, sheet_name=f'Grano_{grano}', index=False)
                
                # Combinar todos los resultados
                if resultados:
                    df_resultados = pd.concat(resultados, ignore_index=True)
                    
                    # Calcular estadísticas generales
                    stats_general = []
                    for metrica in ['Ra', 'Rz']:
                        df_metrica = df_resultados[df_resultados['Métrica'] == metrica]
                        
                        stats_general.append({
                            'Métrica': metrica,
                            'Mínimo Global': df_metrica['Mínimo'].min(),
                            'Máximo Global': df_metrica['Máximo'].max(),
                            'Promedio Global': df_metrica['Promedio'].mean(),
                            'Desviación Estándar Global': df_metrica['Desviación Estándar'].mean(),
                            'Total Outliers': int(df_metrica['Número de Outliers'].sum()),
                            'Total Datos': int(df_metrica['Total de Datos'].sum()),
                            'Porcentaje Outliers': (df_metrica['Número de Outliers'].sum() / 
                                                 df_metrica['Total de Datos'].sum()) * 100
                        })
                    
                    df_stats_general = pd.DataFrame(stats_general)
                    
                    # Guardar resultados en hojas separadas para Ra y Rz
                    df_ra = df_resultados[df_resultados['Métrica'] == 'Ra'].drop('Métrica', axis=1)
                    df_rz = df_resultados[df_resultados['Métrica'] == 'Rz'].drop('Métrica', axis=1)
                    
                    # Renombrar columnas para mayor claridad
                    ra_columns = {col: col.replace('Ra - ', '') for col in df_ra.columns}
                    rz_columns = {col: col.replace('Rz - ', '') for col in df_rz.columns}
                    df_ra = df_ra.rename(columns=ra_columns)
                    df_rz = df_rz.rename(columns=rz_columns)
                    
                    # Guardar en hojas separadas
                    df_ra.to_excel(writer, sheet_name='Estadisticas_Ra', index=False)
                    df_rz.to_excel(writer, sheet_name='Estadisticas_Rz', index=False)
                    df_stats_general.to_excel(writer, sheet_name='Estadisticas_Generales', index=False)
                    
                    # Crear resumen por grano
                    resumen_granos = []
                    for grano in sorted(df['Grano'].unique()):
                        df_grano = df_resultados[df_resultados['Grano'] == grano].copy()
                        
                        resumen_granos.append({
                            'Grano': grano,
                            'Muestras Totales': int(df_grano['Total de Datos'].iloc[0]),
                            'Ra - Promedio': df_grano[df_grano['Métrica'] == 'Ra']['Promedio'].values[0],
                            'Ra - Mediana': df_grano[df_grano['Métrica'] == 'Ra']['Mediana'].values[0],
                            'Ra - Desviación Estándar': df_grano[df_grano['Métrica'] == 'Ra']['Desviación Estándar'].values[0],
                            'Ra - Outliers': int(df_grano[df_grano['Métrica'] == 'Ra']['Número de Outliers'].values[0]),
                            'Rz - Promedio': df_grano[df_grano['Métrica'] == 'Rz']['Promedio'].values[0],
                            'Rz - Mediana': df_grano[df_grano['Métrica'] == 'Rz']['Mediana'].values[0],
                            'Rz - Desviación Estándar': df_grano[df_grano['Métrica'] == 'Rz']['Desviación Estándar'].values[0],
                            'Rz - Outliers': int(df_grano[df_grano['Métrica'] == 'Rz']['Número de Outliers'].values[0])
                        })
                    
                    df_resumen_granos = pd.DataFrame(resumen_granos)
                    df_resumen_granos.to_excel(writer, sheet_name='Resumen_Por_Grano', index=False)
                    
                    messagebox.showinfo("Éxito", f"Análisis completado. Resultados guardados en:\n{output_path}")
                else:
                    messagebox.showinfo("Información", "No se encontraron datos para analizar.")
                    
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al procesar el archivo:\n{str(e)}")

if __name__ == "__main__":
    analizar_estadisticas()
