import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from tkinter import messagebox

def seleccionar_archivo():
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal de tkinter
    
    # Abrir el diálogo para seleccionar archivo
    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo Excel",
        filetypes=[("Archivos Excel", "*.xlsx *.xls"), ("Todos los archivos", "*.*")]
    )
    
    if not archivo:
        print("No se seleccionó ningún archivo.")
        return None
    
    return archivo

def graficar_datos():
    archivo = seleccionar_archivo()
    if not archivo:
        return
    
    try:
        # Leer el archivo Excel, específicamente la hoja 'DATA'
        df = pd.read_excel(archivo, sheet_name='DATA')
        
        # Verificar que existan las columnas E y F (índices 4 y 5 en base 0)
        if len(df.columns) < 6:
            messagebox.showerror("Error", "El archivo no tiene suficientes columnas (se necesitan al menos 6 columnas).")
            return
        
        # Obtener los datos de las columnas E y F
        x = df.iloc[:, 4]  # Columna E (índice 4)
        y = df.iloc[:, 5]  # Columna F (índice 5)
        
        # Crear el gráfico con mayor altura para mejor visualización vertical
        plt.figure(figsize=(10, 8))  # Aumentado de 6 a 8 la altura
        plt.plot(x, y, '-', linewidth=2)  # Línea continua sin puntos
        
        # Personalizar el gráfico
        plt.title('Perfil primario de rugosidad')
        plt.xlabel('Distancia [mm]')
        plt.ylabel('Pa [µm]')
        
        # Ajustar ejes
        plt.xlim(0, 15)  # Rango del eje X de 0 a 15 mm
        plt.ylim(-55, 30)  # Rango del eje Y de -50 a 50
        
        # Asegurar que aparezca el 15 en el eje X
        plt.xticks(range(0, 16, 1))  # Marcas cada 1 mm de 0 a 15
        
        # Centrar el eje Y en 0
        ax = plt.gca()
        ax.axhline(y=0, color='black', linewidth=0.5)  # Línea horizontal en y=0
        
        # Mostrar el gráfico con más espacio vertical
        plt.tight_layout(pad=2.0)  # Aumentado el padding para mejor espaciado
        plt.show()
        
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al procesar el archivo:\n{str(e)}")

if __name__ == "__main__":
    graficar_datos()
