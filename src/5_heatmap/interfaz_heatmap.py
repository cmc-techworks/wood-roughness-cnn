import numpy as np
from tensorflow.keras.models import load_model
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Usar el backend de TkAgg que es más estable
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
import threading
import pandas as pd
from datetime import datetime

def load_and_prepare_image(image_path, target_size=(390, 130)):
    """Load and preprocess the image in grayscale."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    
    # Get image dimensions
    height, width = img.shape
    print(f"Original image size: {width}x{height} pixels")
    
    return img

def split_into_tiles(img, tile_height=390, tile_width=130):
    """Divide la imagen en tiles no solapados con la geometría del modelo.

    Los recortes de entrenamiento son rectángulos de 130 px de ancho x 390 px
    de alto (= entrada del modelo (390, 130, 1)), correspondientes a una región
    física de 15x5 mm. Por eso cada tile se corta como 390 alto x 130 ancho:
    coincide exactamente con lo que el modelo espera, sin deformar la imagen.

    Args:
        img: imagen (panorama) en escala de grises.
        tile_height: alto del tile en píxeles (390 por defecto).
        tile_width: ancho del tile en píxeles (130 por defecto).
    """
    height, width = img.shape[:2]

    # Calcular número de tiles en cada dimensión
    n_tiles_h = height // tile_height
    n_tiles_w = width // tile_width

    # Recortar la imagen a un múltiplo exacto del tamaño de tile
    cropped = img[:n_tiles_h * tile_height, :n_tiles_w * tile_width]

    # Dividir en tiles
    tiles = []
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            tile = cropped[i*tile_height:(i+1)*tile_height,
                           j*tile_width:(j+1)*tile_width]
            tiles.append(tile)

    print(f"Split image into {len(tiles)} tiles of {tile_width}x{tile_height} pixels (ancho x alto)")
    return np.array(tiles), n_tiles_h, n_tiles_w

GRAIN_CATEGORIES = [40, 80, 120, 180]

def predict_ra_values(model, tiles, grain=None):
    """Predict Ra values for each tile.

    Soporta tanto modelos de una sola entrada (solo imagen) como modelos
    multimodales que esperan imagen + grano (vector one-hot).
    """
    # Determinar la forma de imagen que espera el modelo (None, alto, ancho, canal)
    image_input_shape = None
    for inp in model.inputs:
        if len(inp.shape) == 4:
            image_input_shape = inp.shape
            break

    # Red de seguridad: si por alguna razón el tile no coincide con la forma que
    # el modelo espera, se redimensiona. Con la geometría por defecto los tiles
    # ya salen en (390, 130), por lo que normalmente esto no hace nada.
    # OpenCV usa (ancho, alto) en cv2.resize, de ahí el orden invertido.
    if image_input_shape is not None:
        exp_h = int(image_input_shape[1])
        exp_w = int(image_input_shape[2])
        if tiles.shape[1] != exp_h or tiles.shape[2] != exp_w:
            resized = np.empty((tiles.shape[0], exp_h, exp_w), dtype=tiles.dtype)
            for k in range(tiles.shape[0]):
                resized[k] = cv2.resize(tiles[k], (exp_w, exp_h), interpolation=cv2.INTER_AREA)
            tiles = resized

    # Add channel dimension for grayscale (batch_size, height, width, 1)
    if len(tiles.shape) == 3:  # If input is (n_tiles, height, width)
        tiles = np.expand_dims(tiles, axis=-1)  # Add channel dimension

    # Normalize pixel values to [0,1]
    tiles_float = tiles.astype('float32') / 255.0

    # Make predictions
    if len(model.inputs) > 1:
        # Modelo multimodal: construir el one-hot del grano seleccionado
        if grain is None:
            raise ValueError("El modelo es multimodal y requiere seleccionar un grano.")
        grain_idx = GRAIN_CATEGORIES.index(int(grain))
        grain_onehot = np.zeros((tiles_float.shape[0], len(GRAIN_CATEGORIES)), dtype='float32')
        grain_onehot[:, grain_idx] = 1.0

        # El modelo se construyó con entradas nombradas (dict), por lo que se
        # debe pasar un diccionario {nombre: arreglo}. Se mapea cada entrada por
        # su forma (4D = imagen, resto = grano) usando su nombre real.
        model_inputs = {}
        for inp in model.inputs:
            name = inp.name.split(':')[0]
            if len(inp.shape) == 4:
                model_inputs[name] = tiles_float
            else:
                model_inputs[name] = grain_onehot
        predictions = model.predict(model_inputs)
    else:
        predictions = model.predict(tiles_float)

    # Flatten predictions to 1D array
    ra_values = predictions.flatten()

    return ra_values

def create_heatmap(ra_values, n_tiles_h, n_tiles_w):
    """Create heatmap from predicted Ra values."""
    # Reshape to 2D array
    heatmap = ra_values.reshape((n_tiles_h, n_tiles_w))
    return heatmap

def plot_heatmap(heatmap, original_img, alpha=0.4, colormap='jet', show_contour=False, show_minmax=False, rugosity_app=None):
    """
    Create a heatmap visualization with configurable parameters.
    Optimized for better performance and exact size matching.
    """
    
    # Usar la imagen original para visualización
    display_img = original_img
    display_height, display_width = original_img.shape[:2]
    
    # Crear figura con tamaño adecuado para la visualización
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # Mostrar la imagen redimensionada en escala de grises
    ax.imshow(display_img, cmap='gray', aspect='auto', 
              extent=(0.0, float(original_img.shape[1]), float(original_img.shape[0]), 0.0))
    if rugosity_app is not None and hasattr(rugosity_app, 'smooth_var'):
        smooth_enabled = rugosity_app.smooth_var.get()
        interpolation = 'bilinear' if smooth_enabled else 'nearest'
        cv_interpolation = cv2.INTER_LINEAR if smooth_enabled else cv2.INTER_NEAREST
    else:
        interpolation = 'bilinear'  # Valor por defecto
        cv_interpolation = cv2.INTER_LINEAR
    
    # Redimensionar el heatmap usando la interpolación seleccionada
    resized_heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]), 
                               interpolation=cv_interpolation)
    
    # Mostrar el heatmap redimensionado con límite inferior fijo de 3.5 y superior dinámico
    im = ax.imshow(resized_heatmap, 
                   cmap=colormap, 
                   alpha=alpha,
                   interpolation=interpolation,
                   origin='upper',
                   extent=(0.0, float(original_img.shape[1]), float(original_img.shape[0]), 0.0),
                   vmin=3.5)  # Límite inferior fijo en 3.5
                   # vmin=1.75,  # Límite inferior fijo (comentado para referencia)
                   # vmax=8.0)  # Límite superior fijo (comentado para referencia)
    
    # Añadir líneas de contorno si están habilitadas
    if show_contour:
        # Reducir la resolución para el cálculo de contornos
        contour_scale = 0.8  # Aumentado para mayor suavizado
        contour_height = int(original_img.shape[0] * contour_scale)
        contour_width = int(original_img.shape[1] * contour_scale)
        
        # Redimensionar para contornos con interpolación cúbica para mayor suavizado
        contour_heatmap = cv2.resize(resized_heatmap, (contour_width, contour_height), 
                                   interpolation=cv2.INTER_CUBIC)  # Interpolación cúbica para contornos más suaves
        
        # Generar niveles para las líneas de contorno (menos niveles = líneas menos densas)
        levels = np.linspace(heatmap.min(), heatmap.max(), 5)  # Reducido a 5 niveles para líneas menos densas
        
        # Crear líneas de contorno con color blanco y semi-transparencia
        contour = ax.contour(contour_heatmap, 
                            levels=levels,
                            colors='white',
                            linewidths=0.5,
                            alpha=1,
                            extent=(0.0, float(original_img.shape[1]), float(original_img.shape[0]), 0.0),
                            origin='upper')
        # Add contour labels con fuente más pequeña
        ax.clabel(contour, inline=True, fontsize=6, fmt='%.2f', colors='white')  # Reducido de 8 a 6
    
    # Add min and max values if enabled
    if show_minmax or (rugosity_app is not None and hasattr(rugosity_app, 'show_minmax') and rugosity_app.show_minmax):
        min_val = heatmap.min()
        max_val = heatmap.max()
        min_pos = np.unravel_index(np.argmin(resized_heatmap), resized_heatmap.shape)
        max_pos = np.unravel_index(np.argmax(resized_heatmap), resized_heatmap.shape)
        
        # Convert positions to image coordinates
        min_x = (min_pos[1] / resized_heatmap.shape[1]) * original_img.shape[1]
        min_y = (min_pos[0] / resized_heatmap.shape[0]) * original_img.shape[0]
        max_x = (max_pos[1] / resized_heatmap.shape[1]) * original_img.shape[1]
        max_y = (max_pos[0] / resized_heatmap.shape[0]) * original_img.shape[0]
        
        # Add markers and text for min and max
        ax.plot(min_x, min_y, 'bo', markersize=8, markeredgecolor='white', markeredgewidth=1)
        ax.plot(max_x, max_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1)
        
        # Add text annotations with background for better visibility
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8)
        ax.text(min_x, min_y + 15, f'Min: {min_val:.2f} µm', 
                color='blue', ha='center', va='bottom', fontsize=8, bbox=bbox_props)
        ax.text(max_x, max_y - 15, f'Max: {max_val:.2f} µm', 
                color='red', ha='center', va='top', fontsize=8, bbox=bbox_props)
    
    # Add colorbar with larger font and actual min/max values
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Ra [µm]', rotation=270, labelpad=20, fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Se han comentado las líneas que muestran los valores de Max, Min y Prom en la visualización
    # pero se mantiene el cálculo por si se necesita en la terminal
    min_val = heatmap.min()
    max_val = heatmap.max()
    # cbar.ax.text(0.5, -0.06, f'Max: {max_val:.2f} µm', 
    #             transform=cbar.ax.transAxes, 
    #             ha='center', va='top', fontsize=9, weight='normal')
    # cbar.ax.text(0.5, -0.11, f'Min: {min_val:.2f} µm', 
    #             transform=cbar.ax.transAxes, 
    #             ha='center', va='top', fontsize=9, weight='normal')
    
    # El promedio se calcula e imprime en process_image cuando se necesita
    # cbar.ax.text(0.5, -0.16, f'Prom: {avg_rugosity:.2f} µm',
    #             transform=cbar.ax.transAxes, 
    #             ha='center', va='top', fontsize=9, weight='normal')
    
    ax.axis('off')
    plt.tight_layout()
    
    # Add click event handler if rugosity_app is provided
    if rugosity_app is not None:
        def onclick(event):
            if event.inaxes == ax:
                # Convert click position to heatmap coordinates
                x = int(event.xdata / original_img.shape[1] * resized_heatmap.shape[1])
                y = int(event.ydata / original_img.shape[0] * resized_heatmap.shape[0])
                
                # Ensure coordinates are within bounds
                x = max(0, min(x, resized_heatmap.shape[1] - 1))
                y = max(0, min(y, resized_heatmap.shape[0] - 1))
                
                # Get roughness value at clicked position
                value = resized_heatmap[y, x]
                
                # Remove previous point if it exists
                if hasattr(rugosity_app, 'click_point') and rugosity_app.click_point is not None:
                    try:
                        rugosity_app.click_point[0].remove()
                    except (ValueError, AttributeError):
                        pass
                
                # Add new point and text
                rugosity_app.click_point = ax.plot(event.xdata, event.ydata, 'go', 
                                                markersize=8, markeredgecolor='white', 
                                                markeredgewidth=1, alpha=0.7)
                
                # Add text with roughness value
                if hasattr(rugosity_app, 'click_text') and rugosity_app.click_text is not None:
                    try:
                        rugosity_app.click_text.remove()
                    except (ValueError, AttributeError):
                        pass
                
                bbox_props = dict(boxstyle="round,pad=0.3", fc="white", 
                                ec="green", lw=1, alpha=0.8)
                rugosity_app.click_text = ax.text(
                    event.xdata, event.ydata - 15, 
                    f'Ra: {value:.2f} µm', 
                    color='green', ha='center', va='top', 
                    fontsize=8, bbox=bbox_props)
                
                # Redraw the figure
                fig.canvas.draw()
        
        # Connect the click event
        fig.canvas.mpl_connect('button_press_event', onclick)
    
    return fig

class RugosityAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Rugosidad Superficial")
        self.root.geometry("1000x700")
        
        # Variables de control
        self.last_update_time = 0
        self.update_delay = 0.5  # segundos entre actualizaciones
        self.show_minmax = True  # Mostrar valores mínimos y máximos por defecto
        self.click_point = None  # Para almacenar el punto de clic
        self.click_text = None   # Para almacenar el texto del valor de rugosidad
        
        # Almacenamiento de datos de rugosidad
        self.ra_data = []  # Lista para almacenar los datos de rugosidad
        self.current_grain = tk.StringVar(value='40')  # Grano seleccionado (40, 80, 120, 180)
        self.current_subtile = [1, 1]  # Subtile actual [1-10, 1-3]
        self.subtile_counter = 1  # Contador para el segundo número (1-3)
        self.tile_counter = 1  # Contador para el primer número (1-10)
        
        # Variables de la aplicación
        self.model = None
        self.original_img = None
        self.heatmap_data = None
        self.current_figure = None
        self.processing = False  # Evita reentradas mientras se infiere
        self.show_minmax = False  # Controla si se muestran los valores máx/mín
        
        # Configuración de estilos
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', padding=5)
        style.configure('TLabel', background='#f0f0f0')
        
        # Crear la interfaz
        self.create_widgets()

        # Cargar el modelo por defecto al iniciar solo si el archivo existe;
        # de lo contrario, el usuario debe elegirlo con "Seleccionar Modelo".
        default_model = 'modelo_rugosidad2.h5'
        if os.path.exists(default_model):
            self.load_model(default_model)
        else:
            self.status_var.set("Seleccione un modelo con 'Seleccionar Modelo'")
    
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo para controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Botón para seleccionar el modelo
        ttk.Button(control_frame, text="Seleccionar Modelo", command=self.select_model).pack(fill=tk.X, pady=5)

        # Etiqueta con el modelo cargado actualmente
        self.model_name_var = tk.StringVar(value="(ningún modelo)")
        ttk.Label(control_frame, textvariable=self.model_name_var,
                  font=('Arial', 8), foreground='#555555', wraplength=160).pack(fill=tk.X, pady=(0, 5))

        # Botón para cargar imagen
        ttk.Button(control_frame, text="Cargar Imagen", command=self.load_image).pack(fill=tk.X, pady=5)

        # Controles de visualización
        ttk.Label(control_frame, text="Opciones de Visualización", font=('Arial', 9, 'bold')).pack(pady=(10,3))
        
        # Checkbox para líneas de contorno
        self.contour_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Mostrar líneas de contorno", 
                       variable=self.contour_var, command=self.toggle_contour).pack(anchor=tk.W, pady=2)
        
        # Checkbox para mostrar máximos y mínimos
        self.minmax_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Mostrar máx/mín",
                       variable=self.minmax_var, command=self.toggle_minmax).pack(anchor=tk.W, pady=2)
        
        # Checkbox para suavizado
        self.smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Suavizado", 
                       variable=self.smooth_var, command=self.toggle_smooth).pack(anchor=tk.W, pady=2)
        
        # Control de transparencia
        ttk.Label(control_frame, text="Transparencia:", font=('Arial', 8)).pack(anchor=tk.W, pady=(10,0))
        self.alpha_var = tk.DoubleVar(value=0.4)  # Valor predeterminado: moderado
        
        # Frame para los botones de opción
        alpha_frame = ttk.Frame(control_frame)
        alpha_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Botones de opción para la transparencia
        ttk.Radiobutton(alpha_frame, text="Sutil", variable=self.alpha_var, 
                       value=0.2, command=self.schedule_update).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(alpha_frame, text="Moderada", variable=self.alpha_var, 
                       value=0.4, command=self.schedule_update).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(alpha_frame, text="Fuerte", variable=self.alpha_var, 
                       value=0.7, command=self.schedule_update).pack(side=tk.LEFT, padx=2)
        
        # Selector de mapa de colores
        ttk.Label(control_frame, text="Mapa de colores:", font=('Arial', 8)).pack(anchor=tk.W, pady=(15,0))
        self.cmap_var = tk.StringVar(value='jet')
        cmaps = ['jet', 'viridis', 'plasma', 'coolwarm']  # Menos opciones para simplificar
        cmap_menu = ttk.Combobox(control_frame, textvariable=self.cmap_var, 
                                values=cmaps, state='readonly', height=4)
        cmap_menu.pack(fill=tk.X, pady=2)
        cmap_menu.bind('<<ComboboxSelected>>', lambda e: self.schedule_update())

        # Resolución del mapa (tamaño de tile). El tile nativo del modelo es
        # 390x130 (alto x ancho); se mantiene la relación 3:1 para no deformar.
        ttk.Label(control_frame, text="Resolución del mapa:", font=('Arial', 8)).pack(anchor=tk.W, pady=(15,0))
        self.tile_size_var = tk.StringVar(value='Media (390x130)')
        # etiqueta -> alto del tile en píxeles (ancho = alto / 3)
        self.tile_size_options = {
            'Muy alta (130x43)': 130,
            'Alta (260x87)': 260,
            'Media (390x130)': 390,
            'Baja (780x260)': 780,
        }
        tile_menu = ttk.Combobox(control_frame, textvariable=self.tile_size_var,
                                 values=list(self.tile_size_options.keys()),
                                 state='readonly', height=4)
        tile_menu.pack(fill=tk.X, pady=2)
        tile_menu.bind('<<ComboboxSelected>>', lambda e: self.reprocess_image())

        # Botón para guardar
        ttk.Button(control_frame, text="Guardar Visualización",
                  command=self.save_visualization).pack(fill=tk.X, pady=(10,5))
        
        # Panel derecho para la visualización
        self.viz_frame = ttk.Frame(main_frame)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Etiqueta de estado
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X)

        # Barra de progreso (justo encima de la barra de estado)
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL,
                                         mode='determinate', maximum=100)
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)

        # Frame para los botones
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Frame para la selección de grano
        grain_frame = ttk.LabelFrame(button_frame, text="Selección de Grano", padding=5)
        grain_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Botones de selección de grano
        ttk.Radiobutton(grain_frame, text="G40", variable=self.current_grain, value="40").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(grain_frame, text="G80", variable=self.current_grain, value="80").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(grain_frame, text="G120", variable=self.current_grain, value="120").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(grain_frame, text="G180", variable=self.current_grain, value="180").pack(side=tk.LEFT, padx=5)
        
        # Frame para los botones de acción
        action_frame = ttk.LabelFrame(button_frame, text="Acciones", padding=5)
        action_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Botón para guardar los datos
        ttk.Button(action_frame, text="Guardar Datos", command=self.save_ra_data).pack(side=tk.LEFT, padx=5)
        
        # Botón para limpiar los datos
        ttk.Button(action_frame, text="Limpiar Datos", command=self.clear_ra_data).pack(side=tk.LEFT, padx=5)
    
    def select_model(self):
        """Abre un diálogo para elegir un archivo de modelo y lo carga."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar modelo",
            filetypes=(("Modelos Keras", "*.keras;*.h5"), ("All files", "*.*"))
        )
        if not file_path:
            return
        self.load_model(file_path)

    def load_model(self, model_path='modelo_rugosidad2.h5'):
        try:
            self.status_var.set(f"Cargando modelo desde {model_path}...")
            self.root.update()

            try:
                self.model = load_model(model_path)
            except (ValueError, ImportError):
                def mse(y_true, y_pred):
                    from tensorflow.keras import backend as K
                    return K.mean(K.square(y_pred - y_true))

                custom_objects = {'mse': mse, 'mean_squared_error': mse}
                self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            self.model_name_var.set(os.path.basename(model_path))
            self.status_var.set("Modelo cargado exitosamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {str(e)}")
            self.status_var.set("Error al cargar el modelo")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(("Image files", "*.tiff;*.tif;*.png;*.jpg;*.jpeg"), ("All files", "*.*"))
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set(f"Procesando imagen: {os.path.basename(file_path)}")
            self.root.update()
            
            # Cargar y preparar la imagen
            self.original_img = load_and_prepare_image(file_path)
            
            # Procesar la imagen
            self.process_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")
            self.status_var.set("Error al procesar la imagen")
    
    def reprocess_image(self):
        """Re-procesa la imagen ya cargada (p. ej. al cambiar la resolución)."""
        if self.original_img is not None and not self.processing:
            self.process_image()

    def _current_tile_dims(self):
        """Devuelve (tile_height, tile_width) según la resolución elegida,
        manteniendo la relación 3:1 del tile nativo del modelo (390x130)."""
        tile_h = self.tile_size_options.get(self.tile_size_var.get(), 390)
        tile_w = max(1, round(tile_h * 130 / 390))
        return tile_h, tile_w

    def process_image(self):
        if self.original_img is None:
            return
        if self.model is None:
            messagebox.showwarning("Advertencia",
                                   "Primero seleccione un modelo con 'Seleccionar Modelo'.")
            return
        if self.processing:
            return

        # Lanzar la inferencia en un hilo aparte para no congelar la interfaz
        self.processing = True
        self.progress['value'] = 0
        self.status_var.set("Procesando imagen...")
        tile_h, tile_w = self._current_tile_dims()
        grain = self.current_grain.get()
        worker = threading.Thread(
            target=self._process_image_worker,
            args=(self.original_img, tile_h, tile_w, grain),
            daemon=True)
        worker.start()

    def _process_image_worker(self, img, tile_h, tile_w, grain):
        """Se ejecuta en un hilo de fondo: divide en tiles e infiere.
        Las actualizaciones de la interfaz se reenvían al hilo principal
        mediante self.root.after()."""
        try:
            tiles, n_tiles_h, n_tiles_w = split_into_tiles(img, tile_h, tile_w)
            total = len(tiles)
            if total == 0:
                raise ValueError("La imagen es más pequeña que un tile; no hay nada que analizar.")

            batch_size = 8  # Reducir si es necesario
            ra_values = []
            for i in range(0, total, batch_size):
                batch = tiles[i:i + batch_size]
                ra_batch = predict_ra_values(self.model, batch, grain=grain)
                ra_values.extend(ra_batch)
                progress = min(100, int((i + len(batch)) / total * 100))
                self.root.after(0, self._set_progress, progress)

            heatmap = create_heatmap(np.array(ra_values), n_tiles_h, n_tiles_w)

            # Estadísticas en la terminal
            avg_rugosity = float(np.mean(ra_values))
            print("\n" + "="*60)
            print("ANÁLISIS DE RUGOSIDAD COMPLETADO")
            print("-"*60)
            print(f"Cantidad total de mosaicos procesados: {n_tiles_h * n_tiles_w}")
            print(f"Tamaño de tile: {tile_w}x{tile_h} px (ancho x alto)")
            print(f"Dimensión de la cuadrícula: {n_tiles_h} filas x {n_tiles_w} columnas")
            print(f"Valor promedio de rugosidad (Ra): {avg_rugosity:.4f} µm")
            print(f"Rango de valores: Min = {np.min(ra_values):.4f} µm, "
                  f"Max = {np.max(ra_values):.4f} µm")
            print("="*60 + "\n")

            self.root.after(0, self._on_processing_done, heatmap)
        except Exception as e:
            self.root.after(0, self._on_processing_error, str(e))

    def _set_progress(self, value):
        """Actualiza la barra de progreso (hilo principal)."""
        self.progress['value'] = value
        self.status_var.set(f"Procesando... {value}%")

    def _on_processing_done(self, heatmap):
        """Finaliza el análisis en el hilo principal: dibuja y libera el flag."""
        self.heatmap_data = heatmap
        self.processing = False
        self.progress['value'] = 100
        self.update_plot()
        self.status_var.set("Análisis completado.")

    def _on_processing_error(self, msg):
        """Maneja errores del hilo de fondo en el hilo principal."""
        self.processing = False
        self.progress['value'] = 0
        messagebox.showerror("Error", f"Error en el análisis: {msg}")
        self.status_var.set("Error en el análisis")

    def schedule_update(self, *args):
        """Programa una actualización con retraso para evitar actualizaciones frecuentes"""
        current_time = time.time()
        if current_time - self.last_update_time > self.update_delay:
            self.update_plot()
            self.last_update_time = current_time
        else:
            # Si ya hay una actualización pendiente, no hacer nada
            if not hasattr(self, '_update_pending'):
                self._update_pending = True
                self.root.after(int(self.update_delay * 1000), self.delayed_update)
    
    def delayed_update(self):
        """Actualización retrasada para mejor rendimiento"""
        if hasattr(self, '_update_pending'):
            del self._update_pending
            self.update_plot()
    
    def toggle_contour(self):
        """Activa/desactiva las líneas de contorno"""
        self.update_plot()
    
    def toggle_minmax(self):
        """Activa/desactiva la visualización de máximos y mínimos"""
        self.show_minmax = self.minmax_var.get()
        self.update_plot()
    
    def toggle_smooth(self):
        """Activa/desactiva el suavizado del mapa de calor"""
        # Forzar una actualización completa del plot
        current_fig = getattr(self, 'current_figure', None)
        if current_fig is not None:
            fig, canvas = current_fig
            plt.close(fig)
        self.update_plot()
    
    def on_scale_change(self, value):
        """Método obsoleto - Se mantiene por compatibilidad"""
        pass
    
    def update_plot(self, *args):
        if self.original_img is None or self.heatmap_data is None:
            return
        
        try:
            # Obtener parámetros actuales
            alpha = self.alpha_var.get()
            cmap = self.cmap_var.get()
            
            # Actualizar el estado de la barra
            self.status_var.set("Actualizando visualización...")
            self.root.update()  # Forzar actualización inmediata
            
            # Crear la figura con los parámetros actuales
            fig = plot_heatmap(
                self.heatmap_data, 
                self.original_img, 
                alpha=alpha,
                colormap=cmap,
                show_contour=self.contour_var.get(),
                show_minmax=self.show_minmax,
                rugosity_app=self
            )
            
            # Mostrar la figura en la interfaz
            self.show_figure(fig)
            self.status_var.set("Listo")
            
        except Exception as e:
            self.status_var.set("Error al actualizar")
            messagebox.showerror("Error", f"Error al actualizar la visualización: {str(e)}")
            self.status_var.set("Listo")
    
    def show_figure(self, fig):
        try:
            # Limpiar el frame de visualización
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
            
            # Convertir la figura de matplotlib a un widget de Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Mantener una referencia para evitar que sea recolectado por el GC
            self.current_figure = (fig, canvas)
            
            # Forzar actualización de la interfaz
            self.viz_frame.update_idletasks()
            
        except Exception as e:
            raise RuntimeError(f"Error al mostrar la figura: {str(e)}")
    
    def save_visualization(self):
        if not hasattr(self, 'current_figure') or not self.current_figure:
            messagebox.showwarning("Advertencia", "No hay visualización para guardar")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")),
            title="Guardar visualización como"
        )
        
        if file_path:
            try:
                self.current_figure[0].savefig(file_path, bbox_inches='tight', dpi=300, transparent=True) 
                messagebox.showinfo("Éxito", f"Visualización guardada en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el archivo: {str(e)}")

    def on_click(self, event):
        if event.inaxes is None:
            return
            
        # Obtener las coordenadas del clic en los ejes de datos
        x, y = int(event.xdata), int(event.ydata)
        self.click_point = (x, y)
        
        # Obtener el valor de rugosidad en la posición del clic
        if self.heatmap_data is not None:
            # Asegurarse de que las coordenadas estén dentro de los límites
            # Comprobar que heatmap_data tiene la estructura esperada (array con .shape)
            try:
                h_shape = self.heatmap_data.shape
            except Exception:
                # heatmap_data no tiene .shape o es inválido
                return

            if (isinstance(h_shape, tuple) and len(h_shape) >= 2 and
                0 <= y < h_shape[0] and 0 <= x < h_shape[1]):
                ra_value = self.heatmap_data[y, x]
                
                # Crear el nombre del archivo con el formato G[grano]_[tile].[subtile]
                file_name = f"G{self.current_grain.get()}_{self.tile_counter}.{self.subtile_counter}"
                
                # Agregar los datos a la lista
                self.ra_data.append({
                    'ARCHIVO': file_name,
                    'Ra_estimado': ra_value,
                    'Grano': f"G{self.current_grain.get()}",
                    'Tile': self.tile_counter,
                    'SubTile': self.subtile_counter,
                    'X': x,
                    'Y': y
                })
                
                # Actualizar contadores
                self.subtile_counter += 1
                if self.subtile_counter > 3:
                    self.subtile_counter = 1
                    self.tile_counter += 1
                    if self.tile_counter > 10:
                        self.tile_counter = 1
                
                print(f"Datos guardados: {file_name} - Ra: {ra_value:.4f}")
        
        # Actualizar el gráfico para mostrar el punto de clic
        self.update_plot()

    def save_ra_data(self):
        """Guarda los datos de rugosidad en un archivo Excel"""
        if not self.ra_data:
            messagebox.showwarning("Advertencia", "No hay datos para guardar")
            return
        
        # Crear un DataFrame con los datos
        df = pd.DataFrame(self.ra_data)
        
        # Ordenar por archivo para mejor visualización
        df = df.sort_values(['Grano', 'Tile', 'SubTile'])
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"datos_rugosidad_{timestamp}.xlsx"
        
        # Pedir al usuario dónde guardar el archivo
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Guardar datos de rugosidad",
            initialfile=default_filename
        )
        
        if file_path:
            try:
                # Guardar a Excel
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Éxito", f"Datos guardados en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar los datos:\n{str(e)}")
    
    def clear_ra_data(self):
        """Limpia todos los datos de rugosidad guardados"""
        if self.ra_data:
            if messagebox.askyesno("Confirmar", "¿Está seguro de que desea borrar todos los datos guardados?"):
                self.ra_data = []
                self.tile_counter = 1
                self.subtile_counter = 1
                messagebox.showinfo("Éxito", "Datos borrados correctamente")
        else:
            messagebox.showinfo("Información", "No hay datos para borrar")

def main():
    root = tk.Tk()
    RugosityAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


