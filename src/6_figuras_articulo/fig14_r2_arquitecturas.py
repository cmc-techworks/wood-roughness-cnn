#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 17.2: la regresion medido - estimado de la Fig. 17, para las cuatro arquitecturas.

Misma gramatica visual que `fig17_r2.py` —dispersion, recta por minimos cuadrados
con banda de confianza al 95 %, diagonal 1:1 punteada en rojo— pero cuatro paneles,
uno por variante de reduccion del mapa de caracteristicas, con ejes compartidos
para que las nubes se comparen a simple vista.

DOS BASES DE DATOS POSIBLES, y no dan lo mismo:

  --base vc     validacion cruzada, 1030 puntos, seis probetas. Es lo que dejo
                el barrido de arquitecturas y esta disponible para las cuatro
                variantes sin reentrenar nada. Cada punto lo predice el modelo
                del pliegue que NO vio esa probeta. Cada panel muestra la
                repeticion mas representativa de su arquitectura, ver
                `corrida_representativa`.
  --base test   probeta 11, 190 puntos, modelo final entrenado sobre las seis
                probetas, con la semilla de referencia. Exige un modelo final por
                arquitectura, entrenado con `entrenar_final_limpio.py --arch <X>
                --param Ra --seed <SEED_REF>`.

La base `vc` es la comparacion mas solida de las dos: seis probetas contra una, y
1030 puntos contra 190. Es la que va al articulo, en la seccion de arquitectura,
porque ahi lo que se compara son las variantes entre si y no el desempeno final.

--param Rz agregado el 2026-09-09: C y D no tenian corridas de Rz en el barrido hasta esa fecha
(A y B si las tenian desde antes), asi que la comparacion de las cuatro en Rz exigia entrenar
cuarenta y ocho modelos de pliegue mas -dos variantes, cuatro repeticiones, seis pliegues- que se
corrieron ese dia. `--base test` en Rz exige ademas un modelo final por arquitectura entrenado con
`entrenar_final_limpio.py --arch <X> --param Rz --seed <N>`, que no se genero: el alcance acordado
para Rz es solo `--base vc`, igual que se uso para decidir la arquitectura en Ra.

El chequeo de ruido de inicializacion puro (PAR_MISMA_SEMILLA) solo existe para A en Ra -es el
unico par de corridas con semilla identica y configuracion constante que sobrevivio del barrido
anterior al arreglo de determinismo- y no tiene equivalente en Rz, asi que ese chequeo se omite
cuando --param Rz.

Uso:  python fig17_2_r2_arquitecturas.py --base vc
      python fig17_2_r2_arquitecturas.py --base test
      python fig17_2_r2_arquitecturas.py --base vc --param Rz
"""
import argparse
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import comun as C
from fig19_r2 import banda_confianza, r2_identidad

PARAM = 'Ra'
SEED_VC = 10        # semilla comun, solo para el criterio alternativo --criterio seed10
SEED_TEST = None    # se resuelve en main() segun el parametro elegido

# Repeticiones de cada arquitectura: las cuatro del protocolo en las cuatro
# variantes, todas posteriores al arreglo de determinismo del 2026-08-17, de
# modo que la semilla que declaran es la que de verdad fija su inicializacion.
#
# Hasta el 2026-09-04 solo la publicada tenia las cuatro (en Ra). Importaba mas
# aqui que en ninguna otra figura, porque de estos paneles sale la pendiente con
# la que se argumenta la eleccion de variante: compararla entre una arquitectura
# promediada sobre cuatro corridas y tres que van con una sola no sostiene el
# argumento. Se recalcula segun PARAM porque las rutas en Rz llevan sufijo `_Rz`.
def _repeticiones_vc():
    sufijo = '' if PARAM == 'Ra' else f'_{PARAM}'
    return {
        a: [f'{C.ETIQUETA_BARRIDO[a]}_seed{s}_limpio{sufijo}' for s in C.SEEDS_VC]
        for a, _, _, _ in C.ARQUITECTURAS
    }


REPETICIONES_VC = _repeticiones_vc()

# Dos corridas historicas de A con configuracion y semilla declarada identicas.
# Sobreviven del barrido del 2026-08-05, anterior al arreglo de determinismo:
# entonces np.random.seed + tf.random.set_seed no fijaban la inicializacion de
# pesos en Keras 3, asi que la misma semilla arrancaba de pesos distintos. Esa
# coincidencia accidental es hoy el unico par disponible con todo constante salvo
# la inicializacion, y por eso se conserva: mide ruido de inicializacion puro y es
# la vara con la que hay que juzgar la distancia entre paneles. Las corridas que
# alimentan la figura son en cambio posteriores al arreglo y si son reproducibles.
PAR_MISMA_SEMILLA = ['A_25.7k_seed10_limpio_hist20260805',
                     'A_25.7k_seed10_limpio_PRE_modelos_backup']


def agrupar_pliegues(carpeta):
    """Predicciones de los seis pliegues de una corrida, concatenadas: a cada
    probeta la predice el modelo que no la vio."""
    trozos = []
    for i in range(1, C.N_PLIEGUES + 1):
        f = C.BARRIDO / carpeta / 'predicciones' / f'predicciones_fold{i}.xlsx'
        d = pd.read_excel(f)
        d.columns = [c.strip() for c in d.columns]
        d['fold'] = i
        trozos.append(d)
    d = pd.concat(trozos, ignore_index=True)
    return d[f'{PARAM}_real'].values, d[f'{PARAM}_predicho'].values


def corrida_representativa(arch):
    """La repeticion cuyo R2 agrupado queda mas cerca de la media de esa
    arquitectura.

    Un panel muestra una sola corrida, y elegir la semilla 10 en las cuatro seria
    arbitrario: para A es la peor de sus repeticiones (0.723 contra una media
    de 0.761), y asi la variante aparecia con su peor cara. Se elige la
    corrida mas representativa de cada arquitectura, el mismo criterio con el que
    `comun.SEED_REF` fija la corrida de referencia del resto de las figuras. Los
    empates se resuelven por la semilla mas baja.
    """
    reps = [(c, r2_identidad(*agrupar_pliegues(c))) for c in REPETICIONES_VC[arch]]
    media = np.mean([r for _, r in reps])
    # Se redondea la distancia antes de comparar. Es una guarda heredada de cuando
    # alguna arquitectura entraba con dos repeticiones y ambas quedaban a la misma
    # distancia de su media, de modo que la comparacion exacta la decidia el ultimo
    # bit del punto flotante. Con cuatro repeticiones el empate es improbable, pero
    # la guarda se conserva: si ocurre, lo gana la primera de la lista, que es la
    # semilla mas baja.
    return min(reps, key=lambda cr: round(abs(cr[1] - media), 9))[0]


def datos_vc(arch, criterio='representativa'):
    carpeta = (corrida_representativa(arch) if criterio == 'representativa'
               else f'{C.ETIQUETA_BARRIDO[arch]}_seed{SEED_VC}_limpio')
    CORRIDA_USADA[arch] = carpeta
    return agrupar_pliegues(carpeta)


CORRIDA_USADA = {}


def datos_test(arch):
    """Predicciones del modelo final sobre la probeta 11.

    Prefiere la semilla de referencia, pero acepta cualquier otra si esa no se
    entreno: las variantes que no se publican solo tienen una corrida final, y no
    tiene por que ser la misma semilla que eligio el criterio de representatividad
    de la publicada. Se declara cual se uso.
    """
    base = C.RESULTADOS / f'{arch}_oficial' / PARAM / 'test'
    for seed in [SEED_TEST] + [s for s in C.SEEDS_TEST if s != SEED_TEST]:
        f = base / f'seed{seed}' / f'resultados_validacion_{PARAM}.xlsx'
        if f.is_file():
            SEMILLA_USADA[arch] = seed
            d = pd.read_excel(f, sheet_name='Predicciones')
            return d[f'{PARAM}_real'].values, d[f'{PARAM}_estimado'].values
    raise SystemExit(
        f"No hay ningun modelo final de la arquitectura {arch} evaluado sobre el "
        f"test.\nBuscado en {base}/seed*/\nGenerarlo con el entorno tesis:\n"
        f"  entrenar_final_limpio.py --arch {arch} --param {PARAM} --seed {SEED_TEST}\n"
        f"  validar_headless.py --param {PARAM} --modelo <ruta> --salida <carpeta>")


SEMILLA_USADA = {}


def descripcion_base(base):
    # Funcion y no diccionario de modulo: SEED_TEST depende del parametro y no se
    # conoce hasta que main() parsea los argumentos.
    return {'vc': 'cross-validation, six specimens',
            'test': f'independent test specimen, seed {SEED_TEST}'}[base]


def panel(ax, real, est, etiqueta, descripcion, params, elegida, lim):
    xs = np.linspace(real.min(), real.max(), 200)
    medio, inf, sup, pend, inter = banda_confianza(real, est, xs)

    ax.plot(lim, lim, '--', color=C.ROJO, linewidth=1.4, zorder=1)
    ax.fill_between(xs, inf, sup, color=C.AZUL, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(xs, medio, color=C.AZUL, linewidth=1.8, zorder=3)
    ax.plot(real, est, 'o', markersize=3.6, color=C.AZUL, alpha=0.45,
            markeredgewidth=0, zorder=4)

    r2 = r2_identidad(real, est)
    mae = np.abs(real - est).mean()

    # El resumen del panel va dentro del eje: evita que el lector tenga que medir
    # alturas para comparar los cuatro.
    ax.text(0.035, 0.965,
            f'$R^2$ = {r2:.3f}\nMAE = {mae:.3f} µm\nslope = {pend:.2f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=10.5,
            linespacing=1.45,
            bbox=dict(boxstyle='round,pad=0.42', facecolor='white',
                      edgecolor='#c9c9c9', linewidth=0.8, alpha=0.92), zorder=6)

    titulo = f'{etiqueta} · {descripcion} · {params:,} parameters'.replace(',', ' ')
    if elegida:
        titulo += '  (adopted)'
    ax.set_title(titulo, fontsize=11.5, pad=8,
                 fontweight='bold' if elegida else 'normal')

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal', adjustable='box')
    return dict(arquitectura=etiqueta, parametros=params, n=len(real),
                R2=r2, MAE=mae, RMSE=np.sqrt(((real - est) ** 2).mean()),
                pendiente=pend, intercepto=inter,
                sd_est_sobre_sd_med=est.std(ddof=1) / real.std(ddof=1),
                r2_de_la_recta=stats.linregress(real, est).rvalue ** 2)


def main():
    global PARAM, REPETICIONES_VC, SEED_TEST

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', choices=['vc', 'test'], default='vc')
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    ap.add_argument('--criterio', choices=['representativa', 'seed10'],
                    default='representativa',
                    help='Solo con --base vc: que repeticion muestra cada panel.')
    args = ap.parse_args()

    PARAM = args.param
    SEED_TEST = C.SEED_REF[PARAM]
    REPETICIONES_VC = _repeticiones_vc()

    if args.base == 'vc':
        series = {e: datos_vc(e, args.criterio) for e, _, _, _ in C.ARQUITECTURAS}
    else:
        series = {e: datos_test(e) for e, _, _, _ in C.ARQUITECTURAS}

    # Ejes comunes a los cuatro paneles: si cada uno se autoescala, una nube
    # comprimida se ve igual de ancha que una que sigue a la medicion.
    todo = np.concatenate([np.concatenate(v) for v in series.values()])
    margen = 0.04 * (todo.max() - todo.min())
    lim = (todo.min() - margen, todo.max() + margen)

    C.estilo()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 11.4))
    resumen = []
    for ax, letra, (etiqueta, descripcion, params, elegida) in zip(
            axes.ravel(), 'abcd', C.ARQUITECTURAS):
        resumen.append(panel(ax, *series[etiqueta], etiqueta, descripcion,
                             params, elegida, lim))
        C.etiqueta_panel(ax, f'({letra})', dx=-0.19, dy=1.14)

    for ax in axes[1, :]:
        ax.set_xlabel(f'{PARAM} measured values (µm)')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'{PARAM} estimated values (µm)')

    plt.tight_layout(h_pad=3.0, w_pad=2.2)
    sufijo = '' if PARAM == 'Ra' else f'_{PARAM}'
    nombre = f'Fig14_r2_arquitecturas_{args.base}{sufijo}'
    C.guardar(fig, nombre)

    tabla = pd.DataFrame(resumen)
    tabla.to_excel(C.SALIDAS / f'{nombre}_cifras.xlsx', index=False)

    print(f"\nbase: {descripcion_base(args.base)}  ·  {PARAM}  ·  "
          f"n = {tabla.n.iloc[0]} por panel\n")
    print(tabla.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    if args.base == 'test':
        # Las variantes que no se publican tienen una sola corrida final, y no
        # tiene por que ser la semilla de referencia de la publicada.
        print("\nsemilla del modelo final de cada panel:")
        for e, _, _, _ in C.ARQUITECTURAS:
            print(f"    {e}  seed {SEMILLA_USADA[e]}")

    if args.base == 'vc':
        print("\ncorrida mostrada en cada panel:")
        for e, _, _, _ in C.ARQUITECTURAS:
            print(f"    {e}  {CORRIDA_USADA[e]}")

        # El R2 de la nube agrupada NO es el R2 medio de la Tabla 6, y la
        # diferencia no es un error: agrupar los seis pliegues mide tambien el
        # desplazamiento de nivel entre probetas, que el promedio de R2 por
        # pliegue no ve porque cada pliegue se centra en su propia media.
        por_pliegue = pd.DataFrame([
            dict(arquitectura=e,
                 R2_medio_por_pliegue=C.leer_vc(CORRIDA_USADA[e]).R2.mean(),
                 MAE_medio_por_pliegue=C.leer_vc(CORRIDA_USADA[e]).MAE.mean())
            for e, _, _, _ in C.ARQUITECTURAS])
        cotejo = tabla[['arquitectura', 'R2', 'MAE']].merge(por_pliegue, on='arquitectura')
        print("\n  El R2 del panel es el de la nube agrupada; la Tabla 6 informa la "
              "media\n  de los seis pliegues. Son cantidades distintas:\n")
        print('  ' + cotejo.to_string(index=False,
                                      float_format=lambda v: f'{v:.4f}').replace('\n', '\n  '))
        print("\n  La brecha es el desplazamiento de nivel entre probetas: al agrupar, "
              "el\n  error de calibracion de cada probeta cuenta; al promediar R2 por "
              "pliegue,\n  no. Declararlo en el pie de figura para que no parezca "
              "contradecir la tabla.")

        # --------------------------------------------------------------------
        # Cada panel es UNA corrida. Antes de leer diferencias entre paneles hay
        # que saber cuanto se mueve una misma arquitectura al repetirla.
        print(f"\n{'=' * 74}\nCUANTO DE LA DIFERENCIA ENTRE PANELES ES RUIDO\n{'=' * 74}")
        rep = pd.DataFrame([
            dict(arquitectura=a, corrida=c,
                 R2_agrupado=r2_identidad(*agrupar_pliegues(c)),
                 MAE_agrupado=np.abs(np.subtract(*agrupar_pliegues(c))).mean())
            for a, cs in REPETICIONES_VC.items() for c in cs])
        print(rep.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

        entre_arq = rep.groupby('arquitectura').R2_agrupado.mean()
        print(f"\n  entre arquitecturas, promediando repeticiones : "
              f"{entre_arq.max() - entre_arq.min():.4f} en R2 "
              f"({entre_arq.idxmax()} mejor, {entre_arq.idxmin()} peor)")

        if PARAM == 'Ra':
            a, b = (r2_identidad(*agrupar_pliegues(c)) for c in PAR_MISMA_SEMILLA)
            print(f"  entre dos corridas de A con la MISMA semilla   : {abs(a - b):.4f} "
                  f"en R2   ({a:.4f} contra {b:.4f})")
            print("\n  Las dos corridas de A comparten arquitectura, datos, pliegues, "
                  "epocas y\n  numero de semilla: la unica diferencia es la "
                  "inicializacion de pesos, que\n  la semilla no fijaba antes del arreglo "
                  "de determinismo del 2026-08-17. Es\n  ruido de inicializacion puro, "
                  "medido a configuracion constante, y es la vara\n  con la que hay que "
                  "comparar la distancia entre paneles: son del mismo orden.\n  Las "
                  "corridas que alimentan la figura si son posteriores al arreglo.")
        else:
            print("  (el par de corridas con semilla identica que mide ruido de "
                  "inicializacion puro\n  solo existe para A en Ra; no hay equivalente "
                  "en Rz, se omite este chequeo)")

    print(f"\n  rango de R2 entre las cuatro : {tabla.R2.max() - tabla.R2.min():.4f}")
    print(f"  rango de MAE entre las cuatro: "
          f"{tabla.MAE.max() - tabla.MAE.min():.4f} µm   "
          f"(la repetibilidad del SJ-310 dentro de una zona es 0.350 µm)")
    print("\n  sd_est_sobre_sd_med por debajo de 1 es compresion hacia la media: la "
          "nube\n  se aplana contra la horizontal y la pendiente cae por debajo de "
          "la diagonal.")


if __name__ == '__main__':
    main()
