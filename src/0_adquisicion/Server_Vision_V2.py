#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Importación de bibliotecas necesarias
import socket      # Para la comunicación por red
import time        # Para manejo de tiempos y generación de timestamps
import matplotlib.pyplot as plt  # Para visualización de imágenes
import os          # Para operaciones del sistema de archivos

# Set the environment variable for PySpin before importing it
os.environ['FLIR_GENTL64_CTI_VS140'] = r'C:\FLIR_GenTL\FLIR_GenTL_v140.cti'

# Intenta importar PySpin (para cámaras FLIR/PointGrey)
try:
    import PySpin
    HAS_PYSPIN = True  # Bandera para verificar si PySpin está disponible
except ImportError:
    HAS_PYSPIN = False
    print("Advertencia: PySpin no está instalado. La captura de imágenes no estará disponible.")

# Configuración del servidor
HOST = "0.0.0.0"  # Escucha en todas las interfaces de red
PORT = 30002      # Puerto para la comunicación

# Configuración de directorios
CARPETA_FOTOS = "FOTOS_V2"
# Crear la carpeta FOTOS si no existe
if not os.path.exists(CARPETA_FOTOS):
    os.makedirs(CARPETA_FOTOS)
    print(f"Carpeta '{CARPETA_FOTOS}' creada exitosamente")

# Inicialización de la cámara
sistema = PySpin.System.GetInstance()        # Obtiene la instancia del sistema de cámaras
lista_camaras = sistema.GetCameras()         # Obtiene la lista de cámaras conectadas
camara = lista_camaras[0]                    # Selecciona la primera cámara disponible

# Contadores independientes para cada tipo de imagen
contadores_fotos = {"PIEZA": 0, "LIJA": 0, "PANORAMA": 0}

def tomar_foto(tipo):
    """
    Captura una imagen de la cámara y la guarda en disco.
    Args:
        tipo (str): Prefijo para el nombre del archivo (PIEZA, LIJA, PANORAMA)
    Retorna:
        bool: True si la captura fue exitosa, False en caso contrario.
    """
    global contadores_fotos
    
    # Verificar si la cámara está inicializada
    if not camara.IsInitialized():
        try:
            camara.Init()  # Inicializar la cámara si no lo está
        except Exception as e:
            print(f"Error al inicializar la cámara: {e}")
            return False
    
    # Incrementar el contador para el tipo correspondiente
    if tipo not in contadores_fotos:
        contadores_fotos[tipo] = 0
    contadores_fotos[tipo] += 1
    
    # Generar nombre del archivo y ruta completa
    timestamp = str(int(time.time()))
    nombre_archivo = f"{tipo}_{contadores_fotos[tipo]:02d}_{timestamp}.tiff"
    ruta_archivo = os.path.join(CARPETA_FOTOS, nombre_archivo)
    
    try:
        # Iniciar el proceso de adquisición de imágenes
        camara.BeginAcquisition()
        
        try:
            # Pequeña pausa para asegurar que la cámara esté lista
            time.sleep(0.1)
            
            # Capturar una imagen con un timeout de 2000ms (2 segundos) 
            captura = camara.GetNextImage(2000)
            
            # Verificar si la imagen se capturó correctamente
            if captura.IsIncomplete(): 
                print(f'Error: Imagen incompleta - {captura.GetImageStatus()}')
                return False
            
            # Crear un procesador de imágenes
            processor = PySpin.ImageProcessor()
            # Configurar el procesamiento de color a alta calidad
            processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
            
            # Convertir la imagen a formato Mono8 (escala de grises de 8 bits)
            imagen = processor.Convert(captura, PySpin.PixelFormat_Mono8)
            
            # Guardar la imagen en disco
            imagen.Save(ruta_archivo)
            print(f"Imagen guardada exitosamente: {ruta_archivo}")
            
            # Intenta mostrar una vista previa de la imagen (opcional)
            #try:
                # Obtener la imagen como un array de NumPy
                #arreglo = captura.GetNDArray()
                # Configurar matplotlib para modo interactivo
                #plt.ion()
                # Crear una nueva figura de 8x8 pulgadas
                #plt.figure(figsize=(13, 13))
                # Mostrar la imagen en escala de grises
                #plt.imshow(arreglo, cmap='gray')
                # Establecer el título de la figura
                #plt.title(nombre_archivo)
                # Dibujar la figura
                #plt.draw()
                # Pausa breve para permitir la actualización
                #plt.pause(0.1)
                # Mostrar la figura sin bloquear la ejecución
                #plt.show(block=False)
            #except Exception as e:
                # Si hay un error al mostrar la imagen, continuar de todos modos
                #print(f"Advertencia: No se pudo mostrar la imagen - {e}")
            
            return True  # Indicar que la captura fue exitosa
            
        except PySpin.SpinnakerException as ex:
            # Manejar errores específicos de Spinnaker
            print(f'Error al capturar la imagen: {ex}')
            return False
            
        finally:
            # Este bloque siempre se ejecuta, incluso si hay una excepción
            # Asegurarse de liberar los recursos de la captura
            if 'captura' in locals() and captura is not None:
                captura.Release()
            
            # Finalizar la adquisición de imágenes
            camara.EndAcquisition()
            
    except Exception as e:
        # Manejar cualquier otro error durante la adquisición
        print(f'Error durante la adquisición: {e}')
        # Intentar finalizar la adquisición en caso de error
        try:
            camara.EndAcquisition()
        except:
            pass  # Ignorar errores al finalizar la adquisición
        return False  # Indicar que hubo un error

def manejar_cliente(conn, addr):
    """
    Maneja la comunicación con un cliente conectado.
    Args:
        conn: Objeto de conexión del socket
        addr: Tupla con la dirección (IP, puerto) del cliente
    """
    print(f"🤖 Conexión establecida desde {addr}")
    try:
        # Establecer un timeout de 10 segundos para la conexión
        conn.settimeout(60.0)
        
        # Recibir datos del cliente (máximo 1024 bytes) y decodificarlos
        data = conn.recv(1024).decode().strip()
        
        # Verificar si se recibieron datos
        if data:
            print(f"📨 Mensaje recibido: {data}")
            
            # Procesar el comando recibido
            if data in ("PIEZA", "LIJA", "PANORAMA"):
                print(f"🚀 Comando {data} recibido, iniciando captura...")
                try:
                    if tomar_foto(data):
                        conn.sendall(b"OK\n")
                        print("✅ Captura completada exitosamente")
                    else:
                        conn.sendall(b"ERROR\n")
                        print("❌ Error durante la captura")
                except Exception as e:
                    print(f"⚠️ Error inesperado en tomar_foto(): {e}")
                    conn.sendall(b"ERROR\n")
            else:
                print(f"⚠️ Comando no reconocido: {data}")
                conn.sendall(b"UNKNOWN_COMMAND\n")
        else:
            # Datos vacíos recibidos
            print("⚠️ Datos vacíos recibidos")
    except Exception as e:
        # Manejar errores de comunicación
        print(f"⚠️ Error durante la comunicación: {e}")
    finally:
        # Cerrar la conexión con el cliente
        conn.close()
        print("🔌 Conexión cerrada.")

def main():
    """
    Función principal que inicia el servidor y maneja las conexiones entrantes.
    """
    # Crear un socket TCP/IP
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Permitir reutilizar la dirección para evitar errores de "Address already in use"
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Vincular el socket a la dirección y puerto especificados
    server_socket.bind((HOST, PORT))
    
    # Escuchar conexiones entrantes (máximo 1 en la cola)
    server_socket.listen(1)

    print(f"📡 Servidor escuchando en {HOST}:{PORT}...")

    try:
        # Bucle principal del servidor
        while True:
            print("\n⏳ Esperando conexión del robot...")
            # Aceptar una nueva conexión
            conn, addr = server_socket.accept()
            # Manejar la conexión del cliente
            manejar_cliente(conn, addr)
            
    except KeyboardInterrupt:
        # Manejar interrupción por teclado (Ctrl+C)
        print("\n🛑 Recibida señal de interrupción, cerrando servidor...")
    except Exception as e:
        # Manejar otros errores inesperados
        print(f"\n❌ Error en el servidor: {e}")
    finally:
        # Asegurarse de que el socket se cierre correctamente
        server_socket.close()
        print("✅ Servidor detenido correctamente")

# Punto de entrada principal
if __name__ == "__main__":
    main()