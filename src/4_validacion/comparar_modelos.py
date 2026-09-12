#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para comparar modelos guardados en formato .keras y .h5

Este programa permite comparar la arquitectura, pesos y salidas de dos modelos
de TensorFlow guardados en diferentes formatos (.keras y .h5).
"""

import tensorflow as tf
import numpy as np
import h5py
import os
import sys

try:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox
    GUI_DISPONIBLE = True
except ImportError:
    GUI_DISPONIBLE = False
    import argparse

def cargar_modelo(ruta):
    """Carga un modelo desde un archivo .keras o .h5 con manejo de errores mejorado"""
    try:
        print(f"\nCargando modelo: {ruta}")
        # Intentar cargar el modelo con compile=False primero
        modelo = tf.keras.models.load_model(ruta, compile=False)
        
        # Si llegamos aquí, la carga fue exitosa
        print(f"  - Tipo: {type(modelo).__name__}")
        print(f"  - Número de capas: {len(modelo.layers)}")
        
        # Compilar el modelo con una configuración estándar
        modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("  - Modelo compilado con configuración estándar")
        
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo {ruta}: {e}")
        print("  - Intentando cargar con custom_objects...")
        try:
            # Intentar con custom_objects para manejar métricas personalizadas
            modelo = tf.keras.models.load_model(
                ruta,
                custom_objects={
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mae': tf.keras.metrics.MeanAbsoluteError(),
                    'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError(),
                    'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                },
                compile=False
            )
            print("  - Modelo cargado exitosamente con custom_objects")
            return modelo
        except Exception as e2:
            print(f"  - Error al cargar con custom_objects: {e2}")
            return None

def comparar_arquitectura(modelo_keras, modelo_h5):
    """Compara la arquitectura de dos modelos de manera detallada"""
    print("\n=== COMPARACIÓN DE ARQUITECTURA ===")
    
    # Obtener configuraciones
    config_keras = modelo_keras.get_config()
    config_h5 = modelo_h5.get_config()
    
    # Comparar configuraciones generales
    print("\n1. Configuración general del modelo:")
    print(f"  - Nombre Keras: {config_keras.get('name', 'No especificado')}")
    print(f"  - Nombre H5:    {config_h5.get('name', 'No especificado')}")
    
    # Comparar número de capas
    capas_keras = modelo_keras.layers
    capas_h5 = modelo_h5.layers
    print(f"\n2. Número de capas:")
    print(f"  - Keras: {len(capas_keras)} capas")
    print(f"  - H5:    {len(capas_h5)} capas")
    
    # Comparar tipos de capas
    print("\n3. Tipos de capas (Keras | H5):")
    tipos_keras = [type(capa).__name__ for capa in capas_keras]
    tipos_h5 = [type(capa).__name__ for capa in capas_h5]
    
    max_len = max(len(tipos_keras), len(tipos_h5))
    tipos_keras += ['-'] * (max_len - len(tipos_keras))
    tipos_h5 += ['-'] * (max_len - len(tipos_h5))
    
    print("  " + "-" * 50)
    print(f"  {'N°':<5} {'Keras':<20} | {'H5':<20}")
    print("  " + "-" * 50)
    for i, (tk, th) in enumerate(zip(tipos_keras, tipos_h5)):
        igual = "✓" if tk == th else "✗"
        print(f"  {i+1:<5} {tk:<20} | {th:<20} {igual}")
    print("  " + "-" * 50)
    
    # Comparar formas de entrada/salida
    print("\n4. Formas de entrada/salida:")
    print("  Entradas Keras:", [inp.shape for inp in modelo_keras.inputs])
    print("  Entradas H5:   ", [inp.shape for inp in modelo_h5.inputs])
    print("  Salidas Keras: ", [out.shape for out in modelo_keras.outputs])
    print("  Salidas H5:    ", [out.shape for out in modelo_h5.outputs])
    
    # Verificar compatibilidad
    print("\n5. Compatibilidad:")
    compatible = True
    
    # Verificar número de capas
    if len(capas_keras) != len(capas_h5):
        print("  ✗ Los modelos tienen distinto número de capas")
        compatible = False
    else:
        print("  ✓ Mismo número de capas")
    
    # Verificar tipos de capas
    if tipos_keras != tipos_h5:
        print("  ✗ Los modelos tienen diferentes tipos de capas")
        compatible = False
    else:
        print("  ✓ Mismos tipos de capas")
    
    # Verificar formas de entrada/salida
    if ([inp.shape for inp in modelo_keras.inputs] != 
        [inp.shape for inp in modelo_h5.inputs]):
        print("  ✗ Las formas de entrada no coinciden")
        compatible = False
    else:
        print("  ✓ Formas de entrada coincidentes")
    
    if ([out.shape for out in modelo_keras.outputs] != 
        [out.shape for out in modelo_h5.outputs]):
        print("  ✗ Las formas de salida no coinciden")
        compatible = False
    else:
        print("  ✓ Formas de salida coincidentes")
    
    print("\n" + "="*50)
    if compatible:
        print("✅ Los modelos son estructuralmente compatibles")
    else:
        print("⚠️  Los modelos NO son completamente compatientes")
    print("="*50)
    
    return compatible

def comparar_pesos(modelo_keras, modelo_h5):
    """Verifica si los modelos tienen la misma estructura de pesos"""
    print("\n=== VERIFICACIÓN DE PESOS ===")
    
    capas_keras = modelo_keras.layers
    capas_h5 = modelo_h5.layers
    
    print("\nVerificando estructura de pesos...")
    
    # Verificar que tengan el mismo número de capas con pesos
    capas_con_pesos_k = [c for c in capas_keras if c.weights]
    capas_con_pesos_h5 = [c for c in capas_h5 if c.weights]
    
    print(f"  - Capas con pesos (Keras): {len(capas_con_pesos_k)}")
    print(f"  - Capas con pesos (H5):    {len(capas_con_pesos_h5)}")
    
    if len(capas_con_pesos_k) != len(capas_con_pesos_h5):
        print("  ✗ Los modelos tienen distinto número de capas con pesos")
        return False
    
    # Verificar que las capas con pesos tengan la misma estructura
    for i, (ck, ch) in enumerate(zip(capas_con_pesos_k, capas_con_pesos_h5)):
        if len(ck.weights) != len(ch.weights):
            print(f"  ✗ La capa {i} tiene distinto número de tensores de peso")
            print(f"     - {ck.name}: {len(ck.weights)} pesos")
            print(f"     - {ch.name}: {len(ch.weights)} pesos")
            return False
            
        for wk, wh in zip(ck.weights, ch.weights):
            if wk.shape != wh.shape:
                print(f"  ✗ Formas de peso no coinciden en capa {i} ({ck.name})")
                print(f"     - {wk.name}: {wk.shape} vs {wh.name}: {wh.shape}")
                return False
    
    print("  ✓ La estructura de pesos es compatible")
    return True

def comparar_salidas(modelo_keras, modelo_h5):
    """Compara los tipos de salida de los modelos"""
    print("\n=== COMPARACIÓN DE TIPOS DE SALIDA ===")
    
    try:
        # Crear una entrada de prueba simple
        if hasattr(modelo_keras.input, 'shape'):
            entrada_prueba = np.random.rand(1, *modelo_keras.input.shape[1:]).astype(np.float32)
        else:
            # Para modelos con múltiples entradas
            entrada_prueba = [np.random.rand(1, *inp.shape[1:]).astype(np.float32) 
                             for inp in modelo_keras.inputs]
        
        # Obtener salidas
        salida_keras = modelo_keras.predict(entrada_prueba)
        salida_h5 = modelo_h5.predict(entrada_prueba)
        
        # Función para describir la salida
        def describir_salida(salida, nombre):
            print(f"\n  {nombre}:")
            if isinstance(salida, (list, tuple)):
                print(f"    Tipo: Lista de {len(salida)} elementos")
                for i, s in enumerate(salida):
                    print(f"    - Elemento {i}: {type(s).__name__}, forma: {s.shape}")
            elif hasattr(salida, 'shape'):
                print(f"    Tipo: {type(salida).__name__}")
                print(f"    Forma: {salida.shape}")
            else:
                print(f"    Tipo: {type(salida).__name__}")
        
        print("\nEstructura de salidas:")
        describir_salida(salida_keras, "Salida Keras")
        describir_salida(salida_h5, "Salida H5")
        
        # Verificar compatibilidad de tipos
        def comparar_estructura(s1, s2, nivel=0):
            indent = "  " * nivel
            if isinstance(s1, (list, tuple)) and isinstance(s2, (list, tuple)):
                if len(s1) != len(s2):
                    print(f"{indent}✗ Número diferente de salidas: {len(s1)} vs {len(s2)}")
                    return False
                print(f"{indent}✓ Mismo número de salidas: {len(s1)}")
                return all(comparar_estructura(a, b, nivel+1) for a, b in zip(s1, s2))
            elif hasattr(s1, 'shape') and hasattr(s2, 'shape'):
                if s1.shape == s2.shape:
                    print(f"{indent}✓ Formas coincidentes: {s1.shape}")
                    return True
                else:
                    print(f"{indent}✗ Formas diferentes: {s1.shape} vs {s2.shape}")
                    return False
            else:
                tipo1 = type(s1).__name__
                tipo2 = type(s2).__name__
                if tipo1 == tipo2:
                    print(f"{indent}✓ Mismo tipo: {tipo1}")
                    return True
                else:
                    print(f"{indent}✗ Tipos diferentes: {tipo1} vs {tipo2}")
                    return False
        
        print("\nCompatibilidad de salidas:")
        compatible = comparar_estructura(salida_keras, salida_h5)
        
        print("\n" + "="*50)
        if compatible:
            print("✅ Las salidas son estructuralmente compatibles")
        else:
            print("⚠️  Las salidas NO son estructuralmente compatibles")
        print("="*50)
        
        return compatible
        
    except Exception as e:
        print(f"\nError al comparar salidas: {e}")
        import traceback
        traceback.print_exc()
        return False

def seleccionar_archivos():
    """Muestra un diálogo para seleccionar los archivos .keras y .h5"""
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    
    print("Selecciona el archivo .keras")
    archivo_keras = filedialog.askopenfilename(
        title="Seleccionar archivo .keras",
        filetypes=[("Keras model", "*.keras"), ("Todos los archivos", "*.*")]
    )
    
    if not archivo_keras:
        print("No se seleccionó ningún archivo .keras")
        return None, None
    
    print("Selecciona el archivo .h5")
    archivo_h5 = filedialog.askopenfilename(
        title="Seleccionar archivo .h5",
        filetypes=[("H5 model", "*.h5"), ("Todos los archivos", "*.*")]
    )
    
    if not archivo_h5:
        print("No se seleccionó ningún archivo .h5")
        return None, None
    
    return archivo_keras, archivo_h5

def main():
    # Intentar usar la interfaz gráfica si está disponible
    if GUI_DISPONIBLE and len(sys.argv) == 1:
        print("=== Comparador de Modelos Keras (.keras) y H5 (.h5) ===\n")
        archivo_keras, archivo_h5 = seleccionar_archivos()
        if not archivo_keras or not archivo_h5:
            print("Operación cancelada por el usuario.")
            return
    else:
        # Usar argumentos de línea de comandos
        if not GUI_DISPONIBLE or len(sys.argv) < 3:
            print("Uso: python comparar_modelos.py ruta/al/modelo.keras ruta/al/modelo.h5")
            return
            
        parser = argparse.ArgumentParser(description='Compara modelos .keras y .h5')
        parser.add_argument('modelo_keras', help='Ruta al archivo .keras')
        parser.add_argument('modelo_h5', help='Ruta al archivo .h5')
        args = parser.parse_args()
        archivo_keras = args.modelo_keras
        archivo_h5 = args.modelo_h5
    
    # Verificar que los archivos existen
    for ruta in [archivo_keras, archivo_h5]:
        if not os.path.exists(ruta):
            print(f"Error: El archivo no existe: {ruta}")
            return
    
    print(f"\nArchivos seleccionados:")
    print(f"- Modelo Keras: {archivo_keras}")
    print(f"- Modelo H5:    {archivo_h5}")
    
    # Cargar modelos
    print("\nCargando modelos...")
    modelo_keras = cargar_modelo(archivo_keras)
    if modelo_keras is None:
        return
        
    modelo_h5 = cargar_modelo(archivo_h5)
    if modelo_h5 is None:
        return
    
    # Realizar comparaciones
    arquitectura_compatible = comparar_arquitectura(modelo_keras, modelo_h5)
    pesos_compatibles = comparar_pesos(modelo_keras, modelo_h5)
    salidas_compatibles = comparar_salidas(modelo_keras, modelo_h5)
    
    # Mostrar resumen final
    print("\n" + "="*60)
    print("RESUMEN DE COMPATIBILIDAD".center(60))
    print("="*60)
    print(f"✅ Arquitectura: {'Compatible' if arquitectura_compatible else 'No compatible'}")
    print(f"✅ Pesos:        {'Compatible' if pesos_compatibles else 'No compatible'}")
    print(f"✅ Salidas:      {'Compatible' if salidas_compatibles else 'No compatible'}")
    print("="*60)
    
    if arquitectura_compatible and pesos_compatibles and salidas_compatibles:
        print("🎉 ¡Los modelos son completamente compatibles!".center(60))
    else:
        print("⚠️  Los modelos NO son completamente compatibles".center(60))
    print("="*60 + "\n")
    
    # Mostrar mensaje de finalización si se está usando la interfaz gráfica
    if GUI_DISPONIBLE and len(sys.argv) == 1:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Comparación completada", 
            "La comparación de modelos ha finalizado.\nRevisa la consola para ver los resultados detallados."
        )

if __name__ == "__main__":
    main()
