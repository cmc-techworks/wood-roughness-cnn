#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def calcular_parametros_rugosidad(z, x=None):
    """
    Calcula los parámetros de rugosidad a partir del perfil z
    
    Parámetros:
    z -- array con los valores de altura del perfil
    x -- array con las posiciones horizontales (opcional, si no se proporciona se usan índices)
    """
    if x is None:
        x = np.arange(len(z))  # Si no se proporciona x, usar índices
    # Ra - Rugosidad media aritmética
    Ra = np.mean(np.abs(z - np.mean(z)))
    
    # Rq - Raíz cuadrática media de la rugosidad
    Rq = np.sqrt(np.mean((z - np.mean(z))**2))
    
    # Rsk - Asimetría del perfil
    Rsk = (1 / (len(z) * Rq**3)) * np.sum((z - np.mean(z))**3)
    
    # Rku - Curtosis del perfil
    Rku = (1 / (len(z) * Rq**4)) * np.sum((z - np.mean(z))**4)
    
    # Rt - Altura total del perfil
    Rt = np.max(z) - np.min(z)
    
    # RSm - Espaciado medio de los perfiles
    # Identificar cruces por cero
    z_mean = np.mean(z)
    cruces_cero = np.where(np.diff(np.sign(z - z_mean)))[0]
    if len(cruces_cero) > 1:
        # Calcular el espaciado medio entre cruces por cero
        if len(x) > 1:
            RSm = np.mean(np.diff(x[cruces_cero]))
        else:
            RSm = np.mean(np.diff(cruces_cero))  # Usar diferencia de índices si no hay x
    else:
        RSm = np.nan
    
    # Rp - Altura máxima del perfil sobre la línea media
    Rp = np.max(z) - np.mean(z)
    
    # Rv - Profundidad máxima del perfil bajo la línea media
    Rv = np.mean(z) - np.min(z)
    
    # Rz - Altura media de los 5 picos más altos y 5 valles más profundos
    if len(z) >= 10:
        picos = -np.sort(-z)[:5]  # 5 picos más altos
        valles = np.sort(z)[:5]    # 5 valles más profundos
        Rz = (np.mean(picos) - np.mean(valles))
    else:
        Rz = Rt
    
    return {
        'Ra': Ra,
        'Rq': Rq,
        'Rsk': Rsk,
        'Rku': Rku,
        'Rt': Rt,
        'RSm': RSm,
        'Rp': Rp,
        'Rv': Rv,
        'Rz': Rz
    }

"""
Script para graficar perfiles de rugosidad antes y después del procesamiento.
Muestra una comparación visual entre el perfil original y el procesado.
"""

def lectura_archivo_excel(ruta_archivo):
    # Función robusta para leer datos de posición (E) y altura (F)
    df = pd.read_excel(ruta_archivo, sheet_name='DATA', header=None)
    # Filtra solo filas donde ambas columnas sean numéricas
    df = df[pd.to_numeric(df[4], errors='coerce').notnull() & pd.to_numeric(df[5], errors='coerce').notnull()]
    xdata = pd.to_numeric(df[4], errors='coerce')
    zdata = pd.to_numeric(df[5], errors='coerce')
    return xdata.values, zdata.values

def filtro_gaussiano_iso16610(xg, zg, lambdac = 0.008):
    #función para aplicar el filtro gausiano segun norma ISO 16610-21
    lc = lambdac * 1000
    dx = (xg[1] - xg[0]) * 1000
    n = len(zg)
    nl = np.floor(lc // dx) + 1
    nhalf = np.floor((n - 1) // 2)
    x = np.arange(0, nhalf + nl) * dx
    nn = len(x)
    x2 = np.concatenate((x, -x[nn-1:0:-1]))
    n2 = len(x2)
    alpha = np.sqrt(np.log(2) / np.pi)
    s = np.exp(-np.pi * x2**2 / (alpha * lc)**2) / (alpha * lc)
    HP = 1 - np.fft.fft(s) * dx
    LP = np.fft.fft(s) * dx
    z2 = np.concatenate((zg, np.zeros(n2 - n)))
    Fz = np.fft.fft(z2)
    z3 = np.real(np.fft.ifft(Fz * HP))
    z_HP = z3[:n]
    z4 = np.real(np.fft.ifft(Fz * LP))
    z_LP = z4[:n]
    return z_HP, z_LP

def operacion_f (xF, zF):
    #función para aplicar filtro de ajuste polinomial
    ajuste = np.polyfit(xF, zF, 6)
    f = np.polyval(ajuste, xF)
    z_P = zF-f
    return z_P

def recorte_extremos(x_R, z_R):
    #función para recortar los extremos de la señal
    a = np.where(x_R > 0.15)[0]
    b = np.where(x_R < 12.65)[0]
    li = a[0]
    lf = b[-1]
    xrec = x_R[li:lf]
    zrec = z_R[li:lf]
    zrec = zrec - np.mean(zrec)
    return xrec, zrec

def graficar_perfil(ax, x, z, titulo, color='b', label='Perfil', mostrar_leyenda=True):
    """
    Función auxiliar para graficar un perfil en un eje dado
    """
    ax.plot(x, z, color=color, linewidth=0.8, label=label if mostrar_leyenda else "_nolegend_")
    ax.set_title(titulo, fontsize=10)
    ax.set_ylim(-60, 60)
    ax.set_xlabel('Posición (mm)', fontsize=9)
    ax.set_ylabel('Altura (µm)', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    if mostrar_leyenda:
        ax.legend(fontsize=8)
    ax.set_aspect('auto')
    ax.grid(False)

def guardar_grafico(x, z, titulo, nombre_archivo, color='b'):
    """
    Guarda un gráfico en un archivo sin mostrar la leyenda
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, z, color=color, linewidth=0.8)
    ax.set_title(titulo, fontsize=10)
    ax.set_ylim(-60, 60)
    ax.set_xlabel('Posición (mm)', fontsize=9)
    ax.set_ylabel('Altura (µm)', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(False)
    
    # Crear directorio de salida si no existe
    os.makedirs('graficos_guardados', exist_ok=True)
    
    # Guardar la figura
    ruta_completa = os.path.join('graficos_guardados', f"{nombre_archivo}.png")
    plt.tight_layout()
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_guardado = os.path.join(ruta_script, f"{nombre_archivo}.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return ruta_completa

def procesar_archivo(archivo):
    """
    Procesa el archivo y devuelve los datos originales y procesados
    """
    try:
        # 1. Leer datos originales
        x, z = lectura_archivo_excel(archivo)
        
        # 2. Aplicar filtro S (filtro de paso bajo)
        z_hp1, z_lp1 = filtro_gaussiano_iso16610(x, z)
        zs = z_lp1
        
        # 3. Aplicar operación f (ajuste polinomial)
        zP = operacion_f(x, zs)
        
        # 4. Aplicar filtro L (filtro de paso alto)
        z_hp2, _ = filtro_gaussiano_iso16610(x, zP, 2.5)
        zR = z_hp2
        
        # 5. Recortar extremos
        xr, zr = recorte_extremos(x, zR)
        
        return {
            'original': {'x': x, 'z': z},
            'procesado': {'x': xr, 'z': zr}
        }
        
    except Exception as e:
        raise Exception(f"Error al procesar el archivo: {str(e)}")

def mostrar_graficos(archivo, ax1, ax2, canvas):
    try:
        # Limpiar los ejes
        ax1.clear()
        ax2.clear()
        
        # Procesar el archivo
        datos = procesar_archivo(archivo)
        
        # Obtener el nombre base del archivo sin extensión
        nombre_base = os.path.splitext(os.path.basename(archivo))[0]
        
        # Graficar perfil original en la interfaz
        graficar_perfil(
            ax1, 
            datos['original']['x'], 
            datos['original']['z'], 
            'Perfil Original', 
            'r',
            'Sin procesar'
        )
        
        # Graficar perfil procesado en la interfaz
        graficar_perfil(
            ax2, 
            datos['procesado']['x'], 
            datos['procesado']['z'], 
            'Perfil Procesado', 
            'b',
            'Filtrado y recortado'
        )
        
        # Asegurar que ambos gráficos tengan la misma escala en el eje Y
        ax1.set_ylim(-60, 60)
        ax2.set_ylim(-60, 60)
        
        # Ajustar el layout
        plt.tight_layout()
        canvas.draw()
        
        # Calcular y mostrar parámetros de rugosidad
        try:
            # Calcular parámetros para el perfil original y procesado
            parametros_original = calcular_parametros_rugosidad(datos['original']['z'])
            parametros_procesado = calcular_parametros_rugosidad(datos['procesado']['z'])
            
            # Mostrar parámetros en la terminal
            print("\n" + "="*80)
            print("PARÁMETROS DE RUGOSIDAD")
            print("="*80)
            print(f"{'PARÁMETRO':<10} | {'ORIGINAL':^15} | {'PROCESADO':^15} | {'UNIDAD':^10}")
            print("-"*60)
            
            for param in ['Ra', 'Rq', 'Rsk', 'Rku', 'Rt', 'RSm', 'Rp', 'Rv', 'Rz']:
                print(f"{param:<10} | {parametros_original[param]:<15.4f} | {parametros_procesado[param]:<15.4f} | {'µm' if param != 'RSm' else 'mm'}")
            
            print("="*80 + "\n")
            
            # Guardar perfil original
            ruta_original = guardar_grafico(
                datos['original']['x'],
                datos['original']['z'],
                'Perfil Original',
                f"{nombre_base}_original"
            )
            
            # Guardar perfil procesado
            ruta_procesado = guardar_grafico(
                datos['procesado']['x'],
                datos['procesado']['z'],
                'Perfil Procesado',
                f"{nombre_base}_procesado"
            )
            
            print(f"Gráficos guardados en:\n{ruta_original}\n{ruta_procesado}")
            
        except Exception as e:
            print(f"Error al procesar los datos: {e}")
        
        return True
        
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return False

class AplicacionGraficador:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparador de Perfiles de Rugosidad")
        self.root.geometry("1000x800")
        
        # Configurar el estilo
        self.estilo = ttk.Style()
        self.estilo.configure('TButton', font=('Arial', 10))
        self.estilo.configure('TLabel', font=('Arial', 10))
        
        # Crear marco principal con desplazamiento
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Marco para el título
        self.frame_titulo = ttk.Frame(self.main_frame)
        self.frame_titulo.pack(fill=tk.X, pady=(0, 10))
        
        # Título de la aplicación
        self.lbl_titulo = ttk.Label(
            self.frame_titulo, 
            text="Comparación de Perfiles de Rugosidad",
            font=('Arial', 12, 'bold')
        )
        self.lbl_titulo.pack()
        
        # Marco para los controles
        self.frame_controles = ttk.LabelFrame(self.main_frame, text="Controles", padding=10)
        self.frame_controles.pack(fill=tk.X, pady=(0, 10))
        
        # Botón para seleccionar archivo
        self.btn_seleccionar = ttk.Button(
            self.frame_controles, 
            text="Seleccionar Archivo Excel", 
            command=self.seleccionar_archivo,
            width=25
        )
        self.btn_seleccionar.pack(side=tk.LEFT, padx=5)
        
        # Etiqueta para mostrar el archivo actual
        self.lbl_archivo = ttk.Label(
            self.frame_controles, 
            text="Ningún archivo seleccionado",
            foreground='gray'
        )
        self.lbl_archivo.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Marco para los gráficos
        self.frame_graficos = ttk.Frame(self.main_frame)
        self.frame_graficos.pack(fill=tk.BOTH, expand=True)
        
        # Crear figura y ejes para los gráficos (1 fila, 2 columnas)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('Comparación de Perfiles', fontsize=12)
        
        # Ajustar el espaciado
        plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)
        
        # Canvas para mostrar la figura
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_graficos)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mostrar mensaje inicial
        for ax in [self.ax1, self.ax2]:
            ax.text(0.5, 0.5, 'Seleccione un archivo para comenzar', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   color='gray',
                   fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        self.canvas.draw()
        
        # Barra de estado
        self.status_bar = ttk.Label(
            self.main_frame, 
            text="Listo", 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
    
    def seleccionar_archivo(self):
        tipos_archivo = (
            ('Archivos Excel', '*.xlsx *.xls'),
            ('Todos los archivos', '*.*')
        )
        
        archivo = filedialog.askopenfilename(
            title="Seleccione un archivo Excel con datos de perfil",
            filetypes=tipos_archivo
        )
        
        if archivo:
            self.lbl_archivo.config(text=os.path.basename(archivo))
            self.status_bar.config(text="Procesando archivo...")
            self.root.update()
            
            if mostrar_graficos(archivo, self.ax1, self.ax2, self.canvas):
                self.status_bar.config(text=f"Archivo procesado: {os.path.basename(archivo)}")
            else:
                self.status_bar.config(text="Error al procesar el archivo")

def main():
    root = tk.Tk()
    app = AplicacionGraficador(root)
    root.mainloop()

if __name__ == "__main__":
    main()
