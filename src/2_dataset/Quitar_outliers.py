import pandas as pd
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

def extract_grit(image_name):
    """Extract grit number from image name (e.g., G120 -> 120)"""
    try:
        parts = image_name.split('_')
        for part in parts:
            if part.startswith('G') and part[1:].isdigit():
                return int(part[1:])
        return None
    except:
        return None

def identify_outliers(series):
    """Identify outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ~series.between(lower_bound, upper_bound)

def process_file():
    # Open file dialog to select Excel file
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Seleccione el archivo Excel",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    
    if not file_path:
        print("No se seleccionó ningún archivo.")
        return
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=0)
        
        # Extract grit from image names
        df['Grano'] = df['nombre_imagen'].apply(extract_grit)
        
        # Filter out rows where grit couldn't be determined
        df = df.dropna(subset=['Grano'])
        df['Grano'] = df['Grano'].astype(int)
        
        # Initialize DataFrames for storing results
        all_outliers = pd.DataFrame()
        all_non_outliers = pd.DataFrame()
        summary_data = []
        
        # Process each grit size
        for grit in [40, 80, 120, 180]:
            grit_data = df[df['Grano'] == grit].copy()
            if grit_data.empty:
                continue
                
            # Identify outliers
            is_outlier = identify_outliers(grit_data['Ra'])
            outliers = grit_data[is_outlier]
            non_outliers = grit_data[~is_outlier]
            
            # Add to summary
            summary_data.append({
                'Grano': f'G{grit}',
                'Total Datos': len(grit_data),
                'Outliers': len(outliers),
                'No Outliers': len(non_outliers)
            })
            
            # Add to combined DataFrames
            all_outliers = pd.concat([all_outliers, outliers])
            all_non_outliers = pd.concat([all_non_outliers, non_outliers])
        
        # Print summary
        summary_df = pd.DataFrame(summary_data)
        print("\nResumen de outliers por grano de lija:")
        print(summary_df.to_string(index=False))
        
        # Create output file path
        output_path = os.path.join(
            os.path.dirname(file_path),
            'sin_outliers.xlsx'
        )
        
        # Save to Excel with two sheets
        with pd.ExcelWriter(output_path) as writer:
            all_non_outliers[['nombre_imagen', 'Grano', 'Ra']].to_excel(
                writer, 
                sheet_name='Sin Outliers', 
                index=False
            )
            all_outliers[['nombre_imagen', 'Grano', 'Ra']].to_excel(
                writer, 
                sheet_name='Outliers', 
                index=False
            )
        
        print(f"\nArchivo guardado en: {output_path}")
        print("\nCriterio para determinar outliers:")
        print("Se utilizó el método del rango intercuartílico (IQR):")
        print("- Se calcula el primer cuartil (Q1) y tercer cuartil (Q3)")
        print("- Se calcula el IQR = Q3 - Q1")
        print("- Los límites para considerar un valor como outlier son:")
        print("  * Límite inferior = Q1 - 1.5 * IQR")
        print("  * Límite superior = Q3 + 1.5 * IQR")
        print("  * Cualquier valor fuera de estos límites se considera un outlier")
        
    except Exception as e:
        print(f"Ocurrió un error: {str(e)}")

if __name__ == "__main__":
    process_file()