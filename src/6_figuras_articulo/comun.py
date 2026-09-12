# -*- coding: utf-8 -*-
"""Estilo compartido de las figuras del articulo.

Reproduce la gramatica visual de las figuras del articulo que no se regeneran
para que las regeneradas con la arquitectura publicada no desentonen junto a las que no
cambian (Fig. 1-11): fondo blanco con rejilla suave, azul para lo medido y lo de
entrenamiento, naranja para lo estimado y lo de validacion en Ra, verde para lo
estimado en Rz, etiquetas de panel (a)/(b) arriba a la izquierda fuera del eje,
ejes en ingles y exportacion a 300 dpi en PNG y PDF.

Todo lo de aqui corre con el Python del sistema: no importa TensorFlow.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------ rutas ---
AQUI = Path(__file__).resolve().parent
CODE = AQUI.parent
RAIZ = CODE.parent


def _primero(*candidatos):
    """Primera ruta que exista; si ninguna existe, la primera de la lista."""
    for c in candidatos:
        if c.exists():
            return c
    return candidatos[0]


# Se aceptan dos disposiciones: la del repositorio publicado (data/, results/)
# y la del arbol de trabajo original (fotos_recortes/, CODE/resultados/).
DATOS = _primero(RAIZ / 'data' / 'fotos_recortes', RAIZ / 'fotos_recortes')
RESULTADOS = _primero(RAIZ / 'results', CODE / 'resultados')
BARRIDO = _primero(RESULTADOS / 'architecture_sweep',
                   CODE / '3_entrenamiento' / 'resultados_barrido')
SALIDAS = AQUI / 'salidas'
SALIDAS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------ arquitectura publicada ---
# UNA ARQUITECTURA POR PARAMETRO desde el 2026-09-10: B para Ra, C para Rz.
# Antes era una sola letra para los dos (B desde el 2026-09-02, A antes de eso).
#
# Por que B en Ra: es la que menos colapsa hacia la media por grano. Sobre la
# validacion cruzada tiene la pendiente mas cercana a la identidad, la mayor
# razon entre la dispersion estimada y la medida, el menor MAE y el mayor R2.
#
# Por que C en Rz: se eligio por la pendiente de la regresion medido-estimado.
# Queda registrado que sobre las metricas agregadas de Rz, B da mejor MAE (4.051
# contra 4.164), mejor R2 (0.662 contra 0.635) y es mas estable entre semillas
# (SD 0.033 contra 0.087); C gana en pendiente (0.696 contra 0.680).
ARQ_PUBLICADA = {'Ra': 'B', 'Rz': 'C'}

# Etiqueta con que barrido_arquitecturas.py nombra la carpeta de cada variante.
ETIQUETA_BARRIDO = {'A': 'A_25.7k', 'B': 'B_709.7k', 'C': 'C_2.97M', 'D': 'D_12.21M'}


def oficial(param):
    """Carpeta de resultados oficiales de la arquitectura publicada para `param`.

    Es funcion y no constante a proposito. La version anterior exponia un unico
    `OFICIAL`, y al pasar a una arquitectura por parametro ese nombre habria
    seguido resolviendo, apuntando en silencio a la carpeta equivocada para uno
    de los dos parametros. Quitarlo obliga a que cada consumidor declare de que
    parametro habla y convierte un fallo silencioso en un AttributeError.
    """
    return RESULTADOS / f'{ARQ_PUBLICADA[param]}_oficial'

# --------------------------------------------------------------- corridas ---
# Cuatro inicializaciones independientes, en validacion cruzada y en el modelo
# final, para Ra y para Rz. Las semillas son reproducibles desde el arreglo del
# 2026-08-17: tf.keras.utils.set_random_seed si fija la inicializacion de pesos,
# que np.random.seed + tf.random.set_seed no fijaban en Keras 3.
SEEDS_VC = [10, 11, 12, 13]
SEEDS_TEST = [10, 11, 12, 13]

_ETQ = {p: ETIQUETA_BARRIDO[a] for p, a in ARQ_PUBLICADA.items()}
VC_RA = [f'{_ETQ["Ra"]}_seed{s}_limpio' for s in SEEDS_VC]
VC_RZ = [f'{_ETQ["Rz"]}_seed{s}_limpio_Rz' for s in SEEDS_VC]

# Corrida de referencia de las figuras. Las tablas promedian sobre todas las
# repeticiones; las figuras muestran una sola y el pie de figura lo declara.
#
# Se elige la mas representativa: la semilla cuya suma de desvios respecto de la
# media, en unidades de la propia desviacion, es menor sobre las dos cantidades
# que el articulo reporta para ese parametro (R2 de validacion cruzada y de
# test). Lo calcula `elegir_seed_ref.py --param Ra|Rz`; no fijar este numero a ojo.
#
# Resultado del 2026-09-10, criterio por parametro:
#   Ra con B: 13 (suma 0.21), luego 10 (0.77), 11 (1.95), 12 (2.66)
#   Rz con C: 13 (suma 0.33), luego 10 (0.68), 11 (2.12), 12 (2.47)
# Hasta ese dia Ra usaba la 10, elegida con un criterio CONJUNTO que sumaba Ra y
# Rz bajo B. Con Rz publicado en C, la mitad Rz de ese criterio dejo de aplicar a
# B y la semilla representativa de B-Ra pasa a ser la 13. Las tablas no cambian
# (promedian las cuatro); cambian las figuras de una sola corrida (15, 17, 19a).
# El criterio depende de la arquitectura Y del parametro: recalcular al cambiar
# cualquiera de las dos.
SEED_REF = {'Ra': 13, 'Rz': 13}

VC_REF = {'Ra': f'{_ETQ["Ra"]}_seed{SEED_REF["Ra"]}_limpio',
          'Rz': f'{_ETQ["Rz"]}_seed{SEED_REF["Rz"]}_limpio_Rz'}

GRANOS = [40, 80, 120, 180]
N_PLIEGUES = 6

# Arquitecturas del barrido. Fuente: entrenar_final_limpio.py:107-143, no los
# README, que tienen B y C invertidas.
#
# El cuarto campo era la marca de variante adoptada. Va en False en las cuatro a
# proposito: la Fig. 12 y las que la acompanan abren una seccion que COMPARA las
# variantes, y senalar ahi a la ganadora adelanta la conclusion. La eleccion se
# argumenta en el texto y en la tabla que cierra la seccion.
#
# Ninguna descripcion califica a D como 'original' ni como configuracion previa.
# Las cuatro son variantes evaluadas en pie de igualdad, y D es simplemente la
# que aplana sin reducir.
ARQUITECTURAS = [
    ('A', 'GlobalAveragePooling2D',        25_681,     False),
    ('B', '2 × MaxPooling2D + Flatten',    709_713,    False),
    ('C', '1 × MaxPooling2D + Flatten',    2_970_705,  False),
    ('D', 'Flatten',                       12_211_281, False),
]

# ---------------------------------------------------------------- colores ---
AZUL = '#1f4ed8'        # entrenamiento, medido
NARANJA = '#f5a623'     # validacion, estimado (Ra)
VERDE = '#2ca02c'       # estimado (Rz)
ROJO = '#d62728'        # linea 1:1
GRIS = '#7f7f7f'

# Relleno de las cajas de los boxplot, en el tono palido de las figuras actuales
RELLENO_MEDIDO = '#bcd4e6'
RELLENO_ESTIMADO = {'Ra': '#f7d9b0', 'Rz': '#bfe3bf'}
COLOR_ESTIMADO = {'Ra': NARANJA, 'Rz': VERDE}


def estilo():
    """Aplica el estilo base. Llamar una vez al comienzo de cada script."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.9,
        'grid.color': '#cccccc',
        'grid.linewidth': 0.6,
        'grid.linestyle': '--',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def etiqueta_panel(ax, texto, dx=-0.09, dy=1.06):
    """Etiqueta (a)/(b) fuera del eje, arriba a la izquierda."""
    ax.text(dx, dy, texto, transform=ax.transAxes,
            fontsize=15, fontweight='normal', va='top', ha='left')


def guardar(fig, nombre):
    """Guarda PNG y PDF a 300 dpi en salidas/ y devuelve la ruta del PNG."""
    png = SALIDAS / f'{nombre}.png'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(SALIDAS / f'{nombre}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {png}")
    return png


def _exigir_corrida(f, carpeta):
    """Falla con el comando que regenera la corrida, en vez de con un traceback.

    Las tablas y figuras de arquitectura consumen las cuatro repeticiones de las
    cuatro variantes. Si falta una, conviene que se note y se sepa cual: degradar
    en silencio a las corridas que haya es como se llego a publicar una tabla con
    desviacion estandar en una sola fila.
    """
    if f.is_file():
        return
    partes = carpeta.split('_')
    arch = partes[0]
    seed = next((p[4:] for p in partes if p.startswith('seed')), '?')
    param = 'Rz' if carpeta.endswith('_Rz') else 'Ra'
    dataset = 'outliers' if 'outliers' in partes else 'limpio'
    raise SystemExit(
        f"Falta {f}\nCorrer primero, con el entorno tesis:\n"
        f"  python "
        f"../3_entrenamiento/barrido_arquitecturas.py --arch {arch} --param {param} "
        f"--seed {seed} --folds todas --dataset {dataset}")


def leer_vc(carpeta):
    """Metricas por pliegue de una corrida de validacion cruzada (solo los 6 pliegues)."""
    f = BARRIDO / carpeta / 'metricas' / 'metricas_validacion_cruzada.xlsx'
    _exigir_corrida(f, carpeta)
    df = pd.read_excel(f)
    return df[df['Fold'].isin(range(1, N_PLIEGUES + 1))].copy()


def leer_historial(carpeta):
    """Historial por epoca de las 6 hojas de pliegue de una corrida."""
    f = BARRIDO / carpeta / 'metricas' / 'historial_por_epoca.xlsx'
    _exigir_corrida(f, carpeta)
    xl = pd.ExcelFile(f)
    return {h: xl.parse(h) for h in xl.sheet_names if h.lower().startswith('fold')}


def leer_predicciones_test(param, seed=None):
    """Predicciones punto a punto sobre la probeta 11. Las genera 00_predicciones_test.py."""
    if seed is None:
        seed = SEED_REF[param]
    f = oficial(param) / param / 'test' / f'seed{seed}' / f'resultados_validacion_{param}.xlsx'
    if not f.is_file():
        raise SystemExit(
            f"Falta {f}\nCorrer primero, con el entorno tesis:\n"
            f"  python 00_predicciones_test.py")
    return pd.read_excel(f, sheet_name='Predicciones')


def leer_metricas_test(param):
    """Metricas de test por semilla + resumen (media y sd sobre repeticiones)."""
    f = oficial(param) / param / 'test' / 'metricas_test_por_semilla.xlsx'
    if not f.is_file():
        raise SystemExit(f"Falta {f}. Correr 00_predicciones_test.py primero.")
    return (pd.read_excel(f, sheet_name='Por_semilla'),
            pd.read_excel(f, sheet_name='Resumen'))


def ms(serie, dec=3):
    """Formatea una serie como 'media ± sd' con la precision pedida."""
    return f"{serie.mean():.{dec}f} ± {serie.std(ddof=1):.{dec}f}"
