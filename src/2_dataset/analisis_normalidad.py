import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from scipy.stats import shapiro, normaltest, anderson, ttest_ind, f_oneway

class AnalisisNormalidad:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        self.file_path = None
        self.df = None
        self.numeric_columns = []
        self.selected_columns = []

    def select_file(self):
        """Permite al usuario seleccionar un archivo Excel"""
        self.file_path = filedialog.askopenfilename(
            title="Seleccione el archivo Excel",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not self.file_path:
            print("No se seleccionó ningún archivo.")
            return False
        return True

    def select_variables(self):
        """Permite al usuario seleccionar entre las variables disponibles"""
        # Filtrar solo columnas que contienen 'ra' o 'rz' (case insensitive)
        ra_rz_columns = [col for col in self.numeric_columns 
                        if isinstance(col, str) and ('ra' in col.lower() or 'rz' in col.lower())]
        
        if not ra_rz_columns:
            print("No se encontraron columnas 'Ra' o 'Rz' en el archivo.")
            return False
            
        print("\nVariables disponibles para análisis:")
        for i, col in enumerate(ra_rz_columns, 1):
            print(f"{i}. {col}")
            
        while True:
            try:
                selection = input("\nSeleccione las variables a analizar (ej. 1, 1 2, 2): ").strip()
                if not selection:
                    print("Por favor ingrese al menos un número.")
                    continue
                    
                selected_indices = [int(x) - 1 for x in selection.split()]
                self.selected_columns = [ra_rz_columns[i] for i in selected_indices 
                                       if 0 <= i < len(ra_rz_columns)]
                
                if not self.selected_columns:
                    print("Selección inválida. Intente nuevamente.")
                    continue
                    
                print(f"\nVariables seleccionadas: {', '.join(self.selected_columns)}")
                return True
                
            except (ValueError, IndexError):
                print("Entrada inválida. Por favor ingrese números separados por espacios.")
                continue

    def load_data(self):
        """Carga los datos del archivo Excel"""
        try:
            self.df = pd.read_excel(self.file_path)
            
            # Convertir la columna 'Grano' a numérico si existe
            if 'Grano' in self.df.columns:
                self.df['Grano'] = pd.to_numeric(self.df['Grano'], errors='coerce')
            
            # Seleccionar solo columnas numéricas
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not self.numeric_columns:
                print("No se encontraron columnas numéricas en el archivo.")
                return False
                
            print("\n" + "="*80)
            print(f"Archivo cargado: {os.path.basename(self.file_path)}")
            print("="*80)
            
            # Mostrar información sobre las columnas
            print("\nVariables numéricas encontradas:", 
                 ", ".join(self.numeric_columns) if len(self.numeric_columns) < 10 
                 else f"{len(self.numeric_columns)} columnas numéricas")
                 
            if 'Grano' in self.df.columns:
                grano_values = self.df['Grano'].dropna().unique()
                print(f"\nValores únicos en columna 'Grano': {', '.join(map(str, sorted(grano_values)))}")
                
            return True
            
        except Exception as e:
            print(f"Error al cargar el archivo: {str(e)}")
            return False

    def plot_histogram(self, column, ax):
        """Genera un histograma con curva normal"""
        sns.histplot(data=self.df, x=column, kde=True, ax=ax)
        ax.set_title(f'Histograma de {column}')
        ax.set_xlabel('Valores')
        ax.set_ylabel('Frecuencia')

    def plot_qq(self, column, ax):
        """Genera un gráfico Q-Q"""
        stats.probplot(self.df[column].dropna(), dist="norm", plot=ax)
        ax.set_title(f'Gráfico Q-Q de {column}')

    def perform_normality_tests_for_series(self, data):
        """Realiza pruebas de normalidad para una serie de datos"""
        results = {}
        
        # Test de Shapiro-Wilk
        stat, p = shapiro(data)
        results['Shapiro-Wilk'] = {
            'estadistico': stat,
            'p_valor': p,
            'normal': p > 0.05
        }
        
        # Test de D'Agostino
        stat, p = normaltest(data)
        results['D\'Agostino'] = {
            'estadistico': stat,
            'p_valor': p,
            'normal': p > 0.05
        }
        
        # Test de Anderson-Darling
        result = anderson(data)
        results['Anderson-Darling'] = {
            'estadistico': result.statistic,
            'valores_criticos': result.critical_values,
            'niveles_significancia': result.significance_level,
            'normal': result.statistic < result.critical_values[2]  # Para alpha = 0.05
        }
        
        return results
        
    def perform_normality_tests(self, column):
        """Realiza pruebas de normalidad"""
        data = self.df[column].dropna()
        return self.perform_normality_tests_for_series(data)

    def perform_t_test(self, column1, column2):
        """Realiza una prueba t de Student para muestras relacionadas usando Grano"""
        try:
            # Verificar si existe la columna Grano
            if 'Grano' not in self.df.columns:
                print("\n¡ADVERTENCIA: No se encontró la columna 'Grano'. Se realizará una prueba t para muestras independientes.")
                data1 = self.df[column1].dropna()
                data2 = self.df[column2].dropna()
                _, p_levene = stats.levene(data1, data2)
                equal_var = p_levene > 0.05
                t_stat, p_val = ttest_ind(data1, data2, equal_var=equal_var)
                test_type = "independientes"
            else:
                # Asegurarse de que Grano sea numérico
                self.df['Grano'] = pd.to_numeric(self.df['Grano'], errors='coerce')
                
                # Filtrar solo filas donde ambas columnas tengan valores y Grano no sea NaN
                paired_data = self.df[[column1, column2, 'Grano']].dropna()
                
                # Verificar que hay suficientes datos apareados
                if len(paired_data) < 2:
                    print(f"\nNo hay suficientes datos apareados para realizar la prueba t. Se necesitan al menos 2 pares completos.")
                    return
                
                # Ordenar por Grano para asegurar el emparejamiento correcto
                paired_data = paired_data.sort_values('Grano')
                
                # Obtener los datos apareados
                data1 = paired_data[column1].values
                data2 = paired_data[column2].values
                
                # Realizar prueba t para muestras relacionadas
                t_stat, p_val = stats.ttest_rel(data1, data2)
                test_type = "relacionadas (emparejadas por Grano)"
                
                # Mostrar información sobre los pares
                print(f"\nSe encontraron {len(paired_data)} pares de datos para el análisis.")
                print("Primeros 5 pares:")
                print(paired_data[[column1, column2, 'Grano']].head().to_string(index=False))
            
            print(f"\n{'='*80}")
            print(f"Prueba t de Student para muestras {test_type}:")
            print(f"Comparación: {column1} vs {column2}")
            print(f"Hipótesis nula (H0): No hay diferencia entre las medias de {column1} y {column2}")
            print(f"Estadístico t: {t_stat:.4f}")
            print(f"Valor p: {p_val:.4f}")
            
            if p_val > 0.05:
                print("\nConclusión: No se rechaza H0 (p > 0.05)")
                print(f"No hay evidencia significativa de diferencia entre las medias de {column1} y {column2}")
            else:
                print("\nConclusión: Se rechaza H0 (p < 0.05)")
                print(f"Existe evidencia significativa de diferencia entre las medias de {column1} y {column2}")
            print("="*80)
            
            # Gráfico de cajas
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df[[column1, column2]].melt(), 
                       x='variable', y='value')
            plt.title(f'Distribución de {column1} vs {column2}')
            plt.xlabel('Variable')
            plt.ylabel('Valor')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error al realizar la prueba t: {str(e)}")

    def perform_anova(self, columns):
        """Realiza un análisis de varianza (ANOVA) para múltiples grupos"""
        try:
            # Preparar datos para ANOVA
            data = [self.df[col].dropna() for col in columns]
            
            # Realizar prueba de Levene para homocedasticidad
            _, p_levene = stats.levene(*data)
            print(f"\nPrueba de Levene para homocedasticidad: p = {p_levene:.4f}")
            
            # Realizar ANOVA
            f_stat, p_val = f_oneway(*data)
            
            print(f"\n{'='*80}")
            print("Análisis de Varianza (ANOVA)")
            print(f"Variables: {', '.join(columns)}")
            print(f"Hipótesis nula (H0): Las medias de todos los grupos son iguales")
            print(f"Estadístico F: {f_stat:.4f}")
            print(f"Valor p: {p_val:.4f}")
            
            if p_val > 0.05:
                print("Conclusión: No se rechaza H0 (p > 0.05)")
                print("No hay evidencia significativa de diferencias entre las medias de los grupos")
            else:
                print("Conclusión: Se rechaza H0 (p < 0.05)")
                print("Existe evidencia significativa de que al menos dos medias son diferentes")
            print("="*80)
            
            # Gráfico de cajas múltiple
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.df[columns].melt(), 
                       x='variable', y='value')
            plt.title('Comparación de distribuciones (ANOVA)')
            plt.xlabel('Variables')
            plt.ylabel('Valores')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error al realizar el ANOVA: {str(e)}")

    def analyze_normality(self):
        """Analiza la normalidad de las columnas seleccionadas"""
        if not self.selected_columns:
            print("No hay columnas seleccionadas para analizar.")
            return

        for column in self.selected_columns:
            print(f"\n{'='*80}")
            print(f"ANÁLISIS DE NORMALIDAD PARA: {column}")
            print("="*80)
            
            # Mostrar estadísticos descriptivos
            print("\nEstadísticos descriptivos:")
            print(self.df[column].describe().round(2))
            
            # Realizar pruebas de normalidad
            test_results = self.perform_normality_tests(column)
            
            # Mostrar resultados de las pruebas
            print("\nPruebas de normalidad:")
            for test_name, result in test_results.items():
                print(f"\n{test_name}:")
                if test_name == 'Anderson-Darling':
                    print(f"  Estadístico: {result['estadistico']:.4f}")
                    print(f"  Valor crítico (α=0.05): {result['valores_criticos'][2]:.4f}")
                    print(f"  Conclusión: {'Normal' if result['normal'] else 'No normal'}")
                else:
                    print(f"  Estadístico: {result['estadistico']:.4f}")
                    print(f"  Valor p: {result['p_valor']:.4f}")
                    print(f"  Conclusión: {'Normal (p > 0.05)' if result['normal'] else 'No normal (p < 0.05)'}")
            
            # Crear visualizaciones
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            self.plot_histogram(column, ax1)
            self.plot_qq(column, ax2)
            plt.suptitle(f'Análisis de normalidad: {column}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def create_category_plots(self, data, column, category):
        """
        Crea gráficos para una categoría específica
        
        Args:
            data: DataFrame con los datos
            column: Nombre de la columna a analizar
            category: Valor de la categoría de Grano
        """
        # Configuración de estilo
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Formatear el título para mostrar números enteros sin decimales cuando corresponda
        category_title = str(int(category)) if float(category).is_integer() else str(category)
        
        # --- Histograma ---
        sns.histplot(
            data=data, 
            x=column, 
            kde=True, 
            ax=ax1,
            color='#4C72B0',
            edgecolor='white',
            linewidth=0.5,
            bins='auto'
        )
        
        # Calcular estadísticos
        mean_val = data[column].mean()
        median_val = data[column].median()
        std_val = data[column].std()
        
        # Añadir líneas de media y mediana
        ax1.axvline(mean_val, color='#DD8452', linestyle='--', linewidth=2, 
                   label=f'Media: {mean_val:.2f}')
        ax1.axvline(median_val, color='#55A868', linestyle='-', linewidth=2,
                   label=f'Mediana: {median_val:.2f}')
        
        ax1.set_title(f'Distribución de {column}\n(Grano {category_title} μm)', fontsize=14, pad=15)
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel('Frecuencia', fontsize=12)
        ax1.legend()
        
        # --- Gráfico Q-Q mejorado ---
        # Calcular los cuantiles
        qq = stats.probplot(data[column].dropna(), dist="norm")
        x = np.array([qq[0][0][0], qq[0][0][-1]])
        
        # Graficar la línea de referencia
        ax2.plot(x, qq[1][1] + qq[1][0] * x, 'r-', lw=2, alpha=0.8, label='Distribución normal')
        
        # Graficar los puntos Q-Q
        ax2.scatter(qq[0][0], qq[0][1], alpha=0.8, color='#4C72B0', 
                   edgecolor='white', linewidth=0.7, s=80, 
                   label='Datos observados')
        
        # Añadir intervalo de confianza (aproximado)
        n = len(qq[0][0])
        se = (qq[1][0] / np.sqrt(n)) * 1.96  # Intervalo de confianza del 95%
        ax2.fill_between(x, 
                        qq[1][1] + (qq[1][0] - se) * x,
                        qq[1][1] + (qq[1][0] + se) * x,
                        color='red', alpha=0.15, 
                        label='IC 95%')
        
        # Configuraciones del gráfico Q-Q
        ax2.set_title(f'Gráfico Q-Q de {column}\n(Grano {category_title} μm)', fontsize=14, pad=15)
        ax2.set_xlabel('Cuantiles teóricos', fontsize=12)
        ax2.set_ylabel('Cuantiles observados', fontsize=12)
        ax2.legend()
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Añadir título general
        fig.suptitle(
            f'Análisis de {column} - Grano {category_title} μm\n' +
            f'n = {len(data[column].dropna())} muestras | ' +
            f'Media = {mean_val:.2f} ± {std_val:.2f} (1σ)',
            y=1.05, fontsize=14, fontweight='bold'
        )
        
        # Ajustar márgenes
        plt.subplots_adjust(top=0.8, wspace=0.3)
        
        # Mostrar la figura
        plt.show()
        
        # Cerrar la figura para liberar memoria
        plt.close(fig)

    def analyze_normality_by_category(self):
        """Analiza la normalidad de las variables seleccionadas por categoría de Grano"""
        if not self.selected_columns:
            print("No hay columnas seleccionadas para analizar.")
            return
            
        # Verificar si existe la columna Grano
        if 'Grano' not in self.df.columns:
            print("\n¡ADVERTENCIA: No se encontró la columna 'Grano'. No se puede analizar por categorías.")
            return
            
        # Asegurarse de que Grano sea numérico
        self.df['Grano'] = pd.to_numeric(self.df['Grano'], errors='coerce')
        
        # Obtener categorías únicas, eliminar NaN y ordenar
        categories = sorted([c for c in self.df['Grano'].dropna().unique() if not pd.isna(c)])
        
        print(f"\n{'='*80}")
        print(f"ANÁLISIS POR CATEGORÍAS (Grano)")
        print("="*80)
        print(f"\nSe encontraron {len(categories)} categorías únicas de Grano:")
        print(", ".join(map(str, categories)))
        
        # Analizar cada categoría
        for category in categories:
            # Usar isclose para comparación numérica segura
            cat_data = self.df[np.isclose(self.df['Grano'], category)]
            
            print(f"\n\n{'='*80}")
            print(f"CATEGORÍA: Grano = {category}")
            print(f"Número de muestras: {len(cat_data)}")
            print("="*80)
            
            for column in self.selected_columns:
                # Verificar si la columna tiene datos para esta categoría
                if column not in cat_data.columns or cat_data[column].isna().all():
                    print(f"\nNo hay datos para la columna '{column}' en esta categoría.")
                    continue
                    
                # Mostrar estadísticos descriptivos
                print(f"\nEstadísticos para {column}:")
                print(cat_data[column].describe().round(2))
                
                # Realizar pruebas de normalidad
                test_results = self.perform_normality_tests_for_series(cat_data[column].dropna())
                
                # Mostrar resultados de las pruebas
                print("\nPruebas de normalidad:")
                for test_name, result in test_results.items():
                    print(f"\n{test_name}:")
                    if test_name == 'Anderson-Darling':
                        print(f"  Estadístico: {result['estadistico']:.4f}")
                        print(f"  Valor crítico (α=0.05): {result['valores_criticos'][2]:.4f}")
                        print(f"  Conclusión: {'Normal' if result['normal'] else 'No normal'}")
                    else:
                        print(f"  Estadístico: {result['estadistico']:.4f}")
                        print(f"  Valor p: {result['p_valor']:.4f}")
                        print(f"  Conclusión: {'Normal (p > 0.05)' if result['normal'] else 'No normal (p < 0.05)'}")
                
                # Crear visualizaciones
                self.create_category_plots(cat_data, column, category)

    def run(self):
        """Ejecuta el análisis completo"""
        if not self.select_file():
            return
            
        if not self.load_data():
            return
            
        # Permitir al usuario seleccionar variables Ra/Rz
        if not self.select_variables():
            return
            
        # Análisis general
        self.analyze_normality()
        
        # Análisis por categoría de id_probeta
        self.analyze_normality_by_category()
        
        # Realizar pruebas t si hay al menos dos columnas seleccionadas
        if len(self.selected_columns) >= 2:
            # Realizar prueba t entre las dos primeras columnas seleccionadas
            self.perform_t_test(self.selected_columns[0], self.selected_columns[1])
            
            # Realizar ANOVA si hay al menos tres columnas seleccionadas
            if len(self.selected_columns) >= 3:
                self.perform_anova(self.selected_columns[:3])
        
        print("\nAnálisis completado. Puede cerrar las ventanas de gráficos para finalizar.")

if __name__ == "__main__":
    print("ANÁLISIS DE NORMALIDAD ESTADÍSTICA")
    print("="*50)
    print("Este programa analiza la normalidad de variables Ra y Rz en un archivo Excel.")
    print("Solo se mostrarán las columnas que contengan 'Ra' o 'Rz' en su nombre.")
    print("Por favor, seleccione un archivo Excel para analizar...\n")
    
    analizador = AnalisisNormalidad()
    analizador.run()
    
    # Mantener la ventana abierta hasta que el usuario cierre los gráficos
    plt.show(block=True)
