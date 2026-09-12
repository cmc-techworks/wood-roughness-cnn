import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Configurar el estilo de los gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('pastel')

# Configurar fuente Arial para todo el gráfico
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

def detect_outliers(series):
    """
    Detecta outliers usando el método del rango intercuartílico (IQR).
    
    Args:
        series: Serie de pandas a analizar
        
    Returns:
        Series de booleanos indicando si cada valor es un outlier
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

def analizar_datos():
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
            
            # Inicializar listas para almacenar resultados
            outliers_por_grano = []
            estadisticas_por_grano = []
            
            # Generar gráficos de distribución
            graficar_distribucion_rugosidad(file_path)
            
            # Obtener lista única de granos
            granos = sorted(df['Grano'].unique())
            
            for grano in granos:
                # Filtrar datos por grano
                mask_grano = df['Grano'] == grano
                df_grano = df[mask_grano].copy()
                
                if len(df_grano) < 5:  # Mínimo 5 muestras para análisis estadístico
                    continue
                    
                # Detectar outliers para este grano
                outliers_ra = detect_outliers(df_grano['Ra'])
                outliers_rz = detect_outliers(df_grano['Rz'])
                
                # Identificar outliers
                mask_outliers = outliers_ra | outliers_rz
                if not mask_outliers.any():
                    continue
                    
                # Crear DataFrame de outliers para este grano
                outliers_grano = df_grano[mask_outliers].copy()
                outliers_grano['Es_Outlier_Ra'] = outliers_ra[mask_outliers]
                outliers_grano['Es_Outlier_Rz'] = outliers_rz[mask_outliers]
                outliers_por_grano.append(outliers_grano)
                
                # Calcular estadísticas para este grano
                total = len(df_grano)
                total_outliers = mask_outliers.sum()
                solo_ra = (outliers_ra & ~outliers_rz).sum()
                solo_rz = (~outliers_ra & outliers_rz).sum()
                ambos = (outliers_ra & outliers_rz).sum()
                
                estadisticas_por_grano.append({
                    'Grano': grano,
                    'Muestras_Totales': total,
                    'Outliers_Totales': total_outliers,
                    'Porcentaje_Outliers': (total_outliers / total) * 100 if total > 0 else 0,
                    'Solo_Ra': solo_ra,
                    'Solo_Rz': solo_rz,
                    'Ambas_Metricas': ambos
                })
            
            # Si no se encontraron outliers en ningún grano
            if not outliers_por_grano:
                messagebox.showinfo("Información", "No se encontraron outliers en ningún grupo de grano.")
                # Crear hoja sin_outliers idéntica a Todo
                with pd.ExcelWriter(
                    file_path, 
                    engine='openpyxl', 
                    mode='a', 
                    if_sheet_exists='replace'
                ) as writer:
                    df.to_excel(writer, sheet_name='sin_outliers', index=False)
                return
                
            # Combinar todos los outliers
            outliers_df = pd.concat(outliers_por_grano, ignore_index=True)
            
            # Crear una columna de identificación única para cada registro
            # Usamos las columnas que identifican de forma única cada medición
            columnas_identificacion = ['Grano', 'Ra', 'Rz']  # Ajustar según sea necesario
            
            # Crear identificador único para cada fila en ambos DataFrames
            df['id_unico'] = df[columnas_identificacion].astype(str).agg('|'.join, axis=1)
            outliers_df['id_unico'] = outliers_df[columnas_identificacion].astype(str).agg('|'.join, axis=1)
            
            # Filtrar los datos originales para excluir los outliers
            df_sin_outliers = df[~df['id_unico'].isin(outliers_df['id_unico'])].copy()
            
            # Eliminar la columna temporal de identificación
            df_sin_outliers = df_sin_outliers.drop(columns=['id_unico'])
            outliers_df = outliers_df.drop(columns=['id_unico'])
            df = df.drop(columns=['id_unico'])
            
            # Ordenar los DataFrames
            outliers_df = outliers_df.sort_values(by=['Grano', 'Ra', 'Rz'], 
                                               ascending=[True, False, False])
            df_sin_outliers = df_sin_outliers.sort_values(by=['Grano', 'Ra', 'Rz'], 
                                                       ascending=[True, False, False])
            
            # Crear DataFrame de estadísticas
            stats_df = pd.DataFrame(estadisticas_por_grano)
            
            # Calcular totales
            totales = {
                'Grano': 'TOTAL',
                'Muestras_Totales': stats_df['Muestras_Totales'].sum(),
                'Outliers_Totales': stats_df['Outliers_Totales'].sum(),
                'Porcentaje_Outliers': (stats_df['Outliers_Totales'].sum() / 
                                     stats_df['Muestras_Totales'].sum() * 100) if stats_df['Muestras_Totales'].sum() > 0 else 0,
                'Solo_Ra': stats_df['Solo_Ra'].sum(),
                'Solo_Rz': stats_df['Solo_Rz'].sum(),
                'Ambas_Metricas': stats_df['Ambas_Metricas'].sum()
            }
            
            # Agregar fila de totales
            stats_df = pd.concat([stats_df, pd.DataFrame([totales])], ignore_index=True)
            
            # Guardar resultados en el archivo Excel
            with pd.ExcelWriter(
                file_path, 
                engine='openpyxl', 
                mode='a', 
                if_sheet_exists='replace'
            ) as writer:
                # Guardar outliers
                outliers_df.to_excel(writer, sheet_name='Outliers', index=False)
                
                # Guardar datos sin outliers
                df_sin_outliers.to_excel(writer, sheet_name='sin_outliers', index=False)
                
                # Guardar estadísticas detalladas
                stats_df.to_excel(writer, sheet_name='Estadisticas_Outliers', index=False)
                
                # Crear hoja de resumen
                resumen = pd.DataFrame({
                    'Estadístico': [
                        'Total de datos', 
                        'Outliers detectados', 
                        'Porcentaje de outliers',
                        'Solo outliers en Ra',
                        'Solo outliers en Rz',
                        'Outliers en ambas métricas'
                    ],
                    'Valor': [
                        totales['Muestras_Totales'],
                        totales['Outliers_Totales'],
                        f"{totales['Porcentaje_Outliers']:.2f}%",
                        f"{totales['Solo_Ra']} ({(totales['Solo_Ra']/totales['Muestras_Totales']*100):.2f}%)",
                        f"{totales['Solo_Rz']} ({(totales['Solo_Rz']/totales['Muestras_Totales']*100):.2f}%)",
                        f"{totales['Ambas_Metricas']} ({(totales['Ambas_Metricas']/totales['Muestras_Totales']*100):.2f}%)"
                    ]
                })
                resumen.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Crear hoja de composición del dataset sin outliers
                if not df_sin_outliers.empty:
                    # Verificar que no haya duplicados entre outliers y sin_outliers
                    if any(df_sin_outliers.duplicated()):
                        messagebox.showwarning("Advertencia", "Se detectaron duplicados en los datos sin outliers.")
                    
                    # Verificar que no haya intersección entre outliers y sin_outliers
                    cols_check = ['Grano', 'Ra', 'Rz']
                    set_outliers = set(outliers_df[cols_check].astype(str).agg('|'.join, axis=1))
                    set_sin_outliers = set(df_sin_outliers[cols_check].astype(str).agg('|'.join, axis=1))
                    
                    if set_outliers.intersection(set_sin_outliers):
                        messagebox.showwarning("Advertencia", "Se detectó superposición entre outliers y datos normales.")
                    
                    # Calcular estadísticas por grano
                    composicion = df_sin_outliers.groupby('Grano').agg(
                        cantidad_muestras=('Ra', 'count'),
                        Ra_min=('Ra', 'min'),
                        Ra_max=('Ra', 'max'),
                        Ra_promedio=('Ra', 'mean'),
                        Ra_desv_std=('Ra', 'std'),
                        Rz_min=('Rz', 'min'),
                        Rz_max=('Rz', 'max'),
                        Rz_promedio=('Rz', 'mean'),
                        Rz_desv_std=('Rz', 'std')
                    ).reset_index()
                    
                    # Calcular totales
                    totales_composicion = {
                        'Grano': 'TOTAL',
                        'cantidad_muestras': composicion['cantidad_muestras'].sum(),
                        'Ra_min': composicion['Ra_min'].min(),
                        'Ra_max': composicion['Ra_max'].max(),
                        'Ra_promedio': (composicion['Ra_promedio'] * composicion['cantidad_muestras']).sum() / composicion['cantidad_muestras'].sum(),
                        'Ra_desv_std': df_sin_outliers['Ra'].std(),
                        'Rz_min': composicion['Rz_min'].min(),
                        'Rz_max': composicion['Rz_max'].max(),
                        'Rz_promedio': (composicion['Rz_promedio'] * composicion['cantidad_muestras']).sum() / composicion['cantidad_muestras'].sum(),
                        'Rz_desv_std': df_sin_outliers['Rz'].std()
                    }
                    
                    # Agregar fila de totales
                    composicion = pd.concat([
                        composicion,
                        pd.DataFrame([totales_composicion])
                    ], ignore_index=True)
                    
                    # Formatear TODOS los números a 4 decimales
                    for col in composicion.columns[1:]:  # Excluir columna Grano
                        if pd.api.types.is_numeric_dtype(composicion[col]):
                            composicion[col] = composicion[col].round(4)
                    
                    # Asegurar que la columna de conteo sea entera
                    if 'cantidad_muestras' in composicion.columns:
                        composicion['cantidad_muestras'] = composicion['cantidad_muestras'].astype(int)
                    
                    # Guardar hoja de composición
                    composicion.to_excel(writer, sheet_name='Composicion_Dataset', index=False)
            
            # Mostrar mensaje de éxito
            messagebox.showinfo(
                "Análisis Completado",
                f"Se analizaron {len(granos)} tipos de granos.\n"
                f"Se identificaron {totales['Outliers_Totales']} outliers en total.\n"
                "Se han creado cinco hojas en el archivo:\n"
                "1. 'Outliers': Contiene los registros identificados como atípicos\n"
                "2. 'sin_outliers': Contiene todos los registros que NO son atípicos\n"
                "3. 'Estadisticas_Outliers': Muestra estadísticas detalladas por grano\n"
                "4. 'Resumen': Muestra un resumen general del análisis\n"
                "5. 'Composicion_Dataset': Muestra estadísticas detalladas de los datos sin outliers"
            )
            
    
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al procesar el archivo:\n{str(e)}")

def graficar_distribucion_rugosidad(file_path):
    """
    Crea gráficos de caja para mostrar la distribución de Ra y Rz por grano de lijado.
    
    Args:
        file_path: Ruta al archivo Excel que contiene la hoja 'Todo' con los datos
    """
    try:
        # Leer la hoja 'Todo' del archivo Excel
        df = pd.read_excel(file_path, sheet_name='Todo')
        
        # Verificar que existan las columnas necesarias
        required_columns = ['Ra', 'Rz', 'Grano']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            messagebox.showerror("Error", f'No se encontraron las columnas: {", ".join(missing_columns)}')
            return
            
        # Definir los granos de interés y su orden
        granos_numericos = [40, 80, 120, 180]
        orden_granos = [f'P{grano}' for grano in granos_numericos]
        
        # Convertir la columna Grano a numérico, manejando posibles errores
        df['Grano'] = pd.to_numeric(df['Grano'], errors='coerce')
        
        # Filtrar solo los granos de interés y eliminar filas con valores faltantes
        df_filtrado = df[df['Grano'].isin(granos_numericos)].copy()
        
        if df_filtrado.empty:
            messagebox.showwarning("Advertencia", "No se encontraron datos para los granos 40, 80, 120 o 180 en la hoja 'Todo'.")
            return
            
        # Agregar columna con prefijo 'P' para mostrar en el gráfico
        df_filtrado['Grano_mostrar'] = 'P' + df_filtrado['Grano'].astype(str)
    
            # Crear figura con dos subgráficos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Función para crear cada gráfico
        def crear_grafico(ax, parametro, titulo, color):
            # Crear boxplot usando la columna numérica pero con etiquetas formateadas
            box = sns.boxplot(
                x='Grano', 
                y=parametro, 
                data=df_filtrado, 
                order=[str(g) for g in granos_numericos],  # Ordenar numéricamente
                ax=ax,
                color=color,
                width=0.6,
                showfliers=False  # No mostrar outliers ya que los mostraremos todos
            )
            
            # Agregar todos los puntos con jitter para mejor visualización
            for i, grano_num in enumerate(granos_numericos):
                grano_str = str(grano_num)
                if grano_str not in df_filtrado['Grano'].astype(str).values:
                    continue
                    
                # Obtener datos para este grano
                datos = df_filtrado[df_filtrado['Grano'].astype(str) == grano_str][parametro].dropna()
                
                # Calcular cuartiles y rango intercuartil para resaltar outliers
                Q1 = datos.quantile(0.25)
                Q3 = datos.quantile(0.75)
                IQR = Q3 - Q1
                
                # Identificar outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Separar datos en normales y outliers
                normales = datos[(datos >= lower_bound) & (datos <= upper_bound)]
                outliers = datos[(datos < lower_bound) | (datos > upper_bound)]
                
                # Graficar puntos normales en negro
                if not normales.empty:
                    jitter = np.random.normal(0, 0.1, size=len(normales))  # Pequeño jitter
                    ax.plot(i + jitter, normales, 'o', alpha=0.5, markersize=4, 
                           markerfacecolor='black', markeredgecolor='none', label='Normal data' if i == 0 and 'Normal data' not in ax.get_legend_handles_labels()[1] else '')
                
                # Graficar outliers en rojo
                if not outliers.empty:
                    jitter = np.random.normal(0, 0.1, size=len(outliers))
                    ax.plot(i + jitter, outliers, 'o', alpha=0.7, markersize=4, 
                           markerfacecolor='red', markeredgecolor='black', label='Outliers' if i == 0 and 'Outliers' not in ax.get_legend_handles_labels()[1] else '')
            
            # Actualizar etiquetas del eje x con el formato 'P' + número
            ax.set_xticklabels([f'P{g}' for g in granos_numericos])
            
            # Personalizar el gráfico
            ax.set_xlabel('Grain Size', fontsize=16, labelpad=10)
            ax.set_ylabel(f'{parametro} (μm)', fontsize=16, labelpad=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Aumentar tamaño de fuente de los ticks
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # Ajustar los límites del eje y para mejor visualización
            y_min = max(0, df_filtrado[parametro].min() * 0.9)
            y_max = df_filtrado[parametro].max() * 1.1
            ax.set_ylim(y_min, y_max)
            
            # Añadir líneas de referencia horizontales
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            
            # Rotar etiquetas del eje x
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Añadir leyenda
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Eliminar duplicados
            if by_label:  # Solo agregar leyenda si hay elementos para mostrar
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=14)
        
        # Crear gráfico para Ra
        crear_grafico(ax1, 'Ra', '', 'peachpuff')
        
        # Crear gráfico para Rz
        crear_grafico(ax2, 'Rz', '', 'lightgreen')
        
        # Ajustar el layout
        plt.tight_layout()
        
        # Guardar el gráfico en la misma carpeta que el archivo Excel
        import os
        from datetime import datetime
        
        # Obtener el directorio del archivo Excel
        excel_dir = os.path.dirname(file_path)
        
        # Crear nombre de archivo con marca de tiempo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(excel_dir, f"distribucion_rugosidad_{timestamp}.png")
        
        # Guardar la figura
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado como: {output_filename}")
        
        # Mostrar el gráfico
        plt.show()
        
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al generar los gráficos:\n{str(e)}")

if __name__ == "__main__":
    analizar_datos()
