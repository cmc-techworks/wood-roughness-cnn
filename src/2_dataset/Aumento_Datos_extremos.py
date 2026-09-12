import pandas as pd
from tkinter import filedialog, messagebox, Tk
import os
import shutil
from pathlib import Path

def copiar_y_renombrar_imagenes(imagenes, carpeta_origen, carpeta_destino):
    """Copia y renombra las imágenes agregando '_AD' antes de la extensión."""
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    imagenes_renombradas = {}
    
    for img_nombre in imagenes:
        # Construir rutas de origen y destino
        # Primero verificar si el nombre ya tiene _AD
        nombre_original = img_nombre.replace('_AD', '')  # Quitar _AD si existe
        origen = os.path.join(carpeta_origen, nombre_original)
        
        # Verificar si la extensión es .tiff (case insensitive)
        nombre_base, extension = os.path.splitext(nombre_original)
        if extension.lower() in ['.tiff', '.tif']:
            # Asegurarse de que el nuevo nombre siempre tenga _AD
            if not nombre_base.endswith('_AD'):
                nuevo_nombre = f"{nombre_base}_AD{extension}"
            else:
                nuevo_nombre = f"{nombre_base}{extension}"
                
            destino = os.path.join(carpeta_destino, nuevo_nombre)
            
            # Verificar si el archivo de origen existe
            if not os.path.exists(origen):
                print(f"Advertencia: No se encontró el archivo de origen: {origen}")
                continue
                
            # Copiar y renombrar la imagen
            try:
                shutil.copy2(origen, destino)
                imagenes_renombradas[img_nombre] = os.path.basename(destino)
                print(f"Copiado: {nombre_original} -> {nuevo_nombre}")
            except Exception as e:
                print(f"Error al copiar {nombre_original} a {destino}: {str(e)}")
    
    return imagenes_renombradas

def modificar_nombres_imagenes():
    # Crear ventana Tkinter (oculta)
    root = Tk()
    root.withdraw()
    
    while True:
        # Pedir al usuario que seleccione el archivo Excel
        archivo_excel = filedialog.askopenfilename(
            title="Seleccione el archivo Excel",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        
        if not archivo_excel:
            print("No se seleccionó ningún archivo.")
            return
        
        try:
            # Verificar si el archivo está abierto en otro programa
            try:
                with open(archivo_excel, 'a', encoding='utf-8'):
                    pass
            except PermissionError:
                respuesta = messagebox.askretrycancel(
                    "Error de permiso",
                    "El archivo está abierto en otro programa o no tiene permisos.\n"
                    f"Por favor cierre el archivo: {os.path.basename(archivo_excel)}\n\n"
                    "¿Desea seleccionar otro archivo?"
                )
                if respuesta:
                    continue
                else:
                    return
            
            # Leer el archivo Excel
            with pd.ExcelFile(archivo_excel) as xls:
                # Verificar si la hoja "Aumento_Datos" existe
                if 'Aumento_Datos' not in xls.sheet_names:
                    print('Error: No se encontró la hoja "Aumento_Datos" en el archivo.')
                    return
                
                # Leer la hoja "Aumento_Datos"
                df = pd.read_excel(xls, sheet_name='Aumento_Datos')
                
                # Verificar si la columna "nombre_imagen" existe
                if 'nombre_imagen' not in df.columns:
                    print('Error: No se encontró la columna "nombre_imagen" en la hoja.')
                    return
                
                # Obtener la lista de nombres de imágenes únicos
                imagenes_unicas = df['nombre_imagen'].dropna().unique()
                
                # Pedir al usuario que seleccione la carpeta de origen de las imágenes
                messagebox.showinfo("Selección de carpeta", "Por favor seleccione la carpeta que contiene las imágenes originales.")
                carpeta_origen = filedialog.askdirectory(title="Seleccione la carpeta con las imágenes originales")
                
                if not carpeta_origen:
                    print("No se seleccionó ninguna carpeta de origen.")
                    return
                
                # Crear la ruta para la carpeta de destino
                carpeta_destino = os.path.join(os.path.dirname(carpeta_origen), "Imagenes_AD")
                
                # Copiar y renombrar las imágenes
                imagenes_renombradas = copiar_y_renombrar_imagenes(imagenes_unicas, carpeta_origen, carpeta_destino)
                
                # Actualizar los nombres en el DataFrame
                df['nombre_imagen'] = df['nombre_imagen'].apply(
                    lambda x: imagenes_renombradas.get(x, x) if pd.notna(x) else x
                )
                
                # Crear un escritor de Excel
                with pd.ExcelWriter(archivo_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    # Escribir la hoja modificada
                    df.to_excel(writer, sheet_name='Aumento_Datos', index=False)
                
                print(f"Se han copiado y renombrado {len(imagenes_renombradas)} imágenes en la carpeta: {carpeta_destino}")
                print(f"Se han actualizado los nombres en el archivo: {archivo_excel}")
                messagebox.showinfo("Éxito", 
                    f"Se han procesado {len(imagenes_renombradas)} imágenes.\n"
                    f"Las imágenes se han guardado en: {carpeta_destino}"
                )
                break
                
        except Exception as e:
            print(f"Ocurrió un error: {str(e)}")
            respuesta = messagebox.askretrycancel(
                "Error", 
                f"Error al procesar el archivo:\n{str(e)}\n\n¿Desea seleccionar otro archivo?"
            )
            if not respuesta:
                break

if __name__ == "__main__":
    modificar_nombres_imagenes()