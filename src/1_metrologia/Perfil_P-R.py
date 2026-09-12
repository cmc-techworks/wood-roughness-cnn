#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

"""
Crea un archivo CSV y un EXCEL con los parametros de rugosidad obtenidos a partir de los perfiles primarios medidos.
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

def calculo_de_parametros(xr, zr):
    #Ra: Desviación media aritmética del perfil evaluado
    Ra = np.mean(np.abs(zr))
    #Rq: Desviación media cuadrática del perfil evaluado
    Rq =np.sqrt(np.mean(zr**2))
    #Rsk: Factor de asimetría del perfil evaluado (skewness)
    Rsk = (1/Rq**3)*np.mean(zr**3)
    #Rku: Factor de aplastamiento del perfil evaluado (kurtosis)
    Rku = (1/Rq**4)*np.mean(zr**4)
    #Rt: Altura total del perfil
    valor_menor = zr.min()
    valor_mayor = zr.max()
    Rt = valor_mayor - valor_menor
    #Rp: Altura de pico media
    #Rv: Altura de valle media
    #Rz: Máxima altura
    def parametrosR_z_p_v (zr):
        nsc = 5
        le = len(zr)
        lsc = le // nsc
        Rzx_i = np.ones(nsc)
        Rp_i = np.zeros(nsc)
        Rv_i = np.zeros(nsc)
        for i in range(nsc):
            section = zr[(lsc*i):(lsc*(i+1))]
            Rzx_i[i] = np.max(section) - np.min(section)
            Rp_i[i] = np.max(section)
            Rv_i[i] = np.min(section)
        Rz = np.mean(Rzx_i)
        Rp = np.mean(Rp_i)
        Rv = np.mean(Rv_i)
        return Rp, Rv, Rz
    Rp, Rv, Rz = parametrosR_z_p_v (zr)
    def parametroRsm(xr, zr, Rp, Rv):
        Hu = 0.1 * Rp
        Hl = 0.1 * Rv
        OH = 0.0001 * Rp
        OD = 0.0001 * Rv
        def arg_max(zr):
            kh = np.argmax(zr)
            return kh
        def sgm(zr, l, u):
            if zr >= u:
                v = 1
            elif zr <= -l:
                v = -1
            else:
                v = 0
            return v
        def root(xa, za, xb, zb):
            if zb == za:
                    x0 = (xa + xb) / 2
            else:
                x0 = min(max((xa * zb - xb * za) /
                (zb - za), xa), xb)
            return x0
        def crossing_the_line_segmentation(xr, zr, OD, OH, Hl, Hu):
            profile_data = []
            n = len(zr)
            i = 0
            j = 1
            nPF = 0
            while j < n:
                if (zr[j - 1] <= OH and zr[j] >= -OD
                    ) or (
                    zr[j - 1] >= -OD and zr[j] <= OH):
                    nPF += 1
                    kh = arg_max(abs(zr[i:j])) + i
                    profile_data.append({
                      't': sgm(zr[kh], min(OD, Hl), min(OH, Hu)),
                      'h': abs(zr[kh]),
                      'xl': root(
                         xr[max(i - 1, 0)],
                         zr[max(i - 1, 0)],
                         xr[i], zr[i]),
                      'xr': root(
                         xr[j - 1],
                         zr[j - 1],
                         xr[j],
                         zr[j])
                    })
                    i = j
                j += 1
            nPF += 1
            kh = arg_max(abs(zr[i:n])) + i
            profile_data.append({
                't': sgm(zr[kh], min(OD, Hl), min(OH, Hu)),
                'h': abs(zr[kh]),
                'xl': root(
                    xr[max(i - 1, 0)],
                    zr[max(i - 1, 0)],
                    xr[i], zr[i]),
                'xr': xr[n - 1]
            })
            PF = pd.DataFrame(profile_data)
            k = nPF - 1
            while k >= 0:
                if (PF.loc[k, 't'] == -1 and PF.loc[k, 'h'] <
                    max(OD, Hl)):
                    PF.drop(k, inplace=True)
                    nPF -= 1
                elif PF.loc[k, 't'] == 0:
                    PF.drop(k, inplace=True)
                    nPF -= 1
                elif PF.loc[k, 't'] == 1 and PF.loc[k, 'h'] < max(OH, Hu):
                    PF.drop(k, inplace=True)
                    nPF -= 1
                k -= 1
            PF.reset_index(drop=True, inplace=True)
            k = nPF - 1
            while k >= 1:
                if PF.loc[k - 1, 't'] == PF.loc[k, 't']:
                    PF.loc[k - 1, 'xr'] = PF.loc[k, 'xr']
                    PF.loc[k - 1, 'h'] = max(PF.loc[k - 1, 'h'],
                    PF.loc[k, 'h'])
                    PF.drop(k, inplace=True)
                    nPF -= 1
                k -= 1
            PF.reset_index(drop=True, inplace=True)
            return PF, nPF
        PF, nPF = crossing_the_line_segmentation(xr, zr, OD, OH, Hl, Hu)
        m = 2 * (nPF // 2 - 1)
        Xs = []
        Zt = []
        for k in range(1, m // 2 + 1):
            Xs.append(PF.loc[2 * (k + 1) - 1, 'xl'] -
                    PF.loc[2 * k - 1, 'xl'])
            Xs.append(PF.loc[nPF - 2 * k, 'xr'] -
                    PF.loc[nPF - 2 * (k + 1), 'xr'])
            Zt.append(PF.loc[2 * k - 1, 'h'] +
                    PF.loc[2 * k, 'h'])
            Zt.append(PF.loc[nPF - 2 * k, 'h'] +
                    PF.loc[nPF - 2 * k - 1, 'h'])
        RSm = np.sum(Xs) / m
        return RSm
    RSm = parametroRsm(xr, zr, Rp, Rv)
    parametrosR=pd.DataFrame()
    parametrosR['Ra'] = [Ra]
    parametrosR['Rq'] = [Rq]
    parametrosR['Rsk'] = [Rsk]
    parametrosR['Rku'] = [Rku]
    parametrosR['Rt'] = [Rt]
    parametrosR['RSm'] = [RSm]
    parametrosR['Rp'] = [Rp]
    parametrosR['Rv'] = [Rv]
    parametrosR['Rz'] = [Rz]
    return parametrosR

def parametros_rugosidad(archivo):
    try:
        x, z = lectura_archivo_excel(archivo)

        #aplicación filtro S
        z_hp1, z_lp1 = filtro_gaussiano_iso16610(x, z)
        zs = z_lp1

        #aplicación operación f, se obtiene perfil primario P
        zP = operacion_f(x, zs)

        #aplicación filtro L, se obtiene perfil secundario R
        z_hp2, z_lp2 = filtro_gaussiano_iso16610(x, zP, 2.5)
        zR = z_hp2

        #aplicación recorte de extremos
        xr, zr = recorte_extremos(x, zR)
        parametrosR = calculo_de_parametros(xr, zr)

        #reconsideración de Setting Class
        if parametrosR['Ra'].iloc[0] > 2:
            parametrosR['Setting Class'] = ['SC4']

        elif 0.1 < parametrosR['Ra'].iloc[0] <= 2:
            z_hp2, z_lp2 = filtro_gaussiano_iso16610(x, zP, 0.8)
            zR = z_hp2
            xr, zr = recorte_extremos(x, zR)
            parametrosR = calculo_de_parametros(xr, zr)
            parametrosR['Setting Class'] = ['SC3']
        return parametrosR
            
    except Exception as e:
        print(f"Error procesando {archivo}: {e}")
        # Devuelve un DataFrame con información del error
        error_df = pd.DataFrame({'Error': [str(e)]})
        error_df['Archivo'] = os.path.basename(archivo)
        return error_df

def procesar_carpeta_excel(ruta_carpeta, archivo_salida='resultados_rugosidad.xlsx'):
    """
    Procesa todos los archivos Excel en la carpeta especificada y genera un archivo Excel consolidado
    con los parámetros de rugosidad de cada archivo.
    
    Parámetros:
    - ruta_carpeta: Ruta de la carpeta que contiene los archivos Excel a procesar
    - archivo_salida: Nombre del archivo Excel de salida (por defecto: 'resultados_rugosidad.xlsx')
    """
    # Verificar si la carpeta existe
    if not os.path.isdir(ruta_carpeta):
        print(f"Error: La carpeta {ruta_carpeta} no existe.")
        return
    
    # Obtener lista de archivos Excel en la carpeta
    archivos_excel = [f for f in os.listdir(ruta_carpeta) 
                     if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
    
    if not archivos_excel:
        print("No se encontraron archivos Excel en la carpeta especificada.")
        return
    
    print(f"Procesando {len(archivos_excel)} archivos Excel...")
    
    # Lista para almacenar los DataFrames de resultados
    resultados = []
    
    # Procesar cada archivo Excel
    for i, archivo in enumerate(archivos_excel, 1):
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        print(f"Procesando archivo {i}/{len(archivos_excel)}: {archivo}")
        
        # Obtener parámetros de rugosidad
        try:
            parametros = parametros_rugosidad(ruta_completa)
            if not parametros.empty:
                # Agregar columna con el nombre del archivo
                parametros.insert(0, 'Archivo', os.path.splitext(archivo)[0])
                resultados.append(parametros)
        except Exception as e:
            print(f"Error al procesar {archivo}: {e}")
            continue
    
    if not resultados:
        print("No se pudo procesar ningún archivo correctamente.")
        return
    
    # Combinar todos los resultados en un solo DataFrame
    try:
        df_resultados = pd.concat(resultados, ignore_index=True)
        
        # Reordenar columnas para que 'Archivo' sea la primera
        columnas = ['Archivo'] + [col for col in df_resultados.columns if col != 'Archivo']
        df_resultados = df_resultados[columnas]
        
        # Guardar en archivo Excel
        df_resultados.to_excel(archivo_salida, index=False)
        print(f"\nProceso completado. Resultados guardados en: {os.path.abspath(archivo_salida)}")
        return df_resultados
    except Exception as e:
        print(f"Error al guardar los resultados: {e}")
        return None

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox
    
    # Configurar la ventana de selección de carpeta
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    
    # Solicitar al usuario que seleccione una carpeta
    print("Seleccione la carpeta que contiene los archivos Excel a procesar...")
    ruta_carpeta = filedialog.askdirectory(title="Seleccione la carpeta con los archivos Excel")
    
    if not ruta_carpeta:
        print("No se seleccionó ninguna carpeta. Saliendo...")
    else:
        # Procesar la carpeta seleccionada
        procesar_carpeta_excel(ruta_carpeta)
        
        # Mostrar mensaje de finalización
        messagebox.showinfo("Proceso completado", 
                          "El procesamiento de archivos ha finalizado. \n"
                          "Los resultados se han guardado en 'resultados_rugosidad.xlsx'")

