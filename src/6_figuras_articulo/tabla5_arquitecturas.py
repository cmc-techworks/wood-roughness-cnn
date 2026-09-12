#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabla 5: comparacion de las cuatro variantes de reduccion del mapa de caracteristicas.

Es la evidencia que sostiene la eleccion de variante frente a la alternativa de
12.2 millones de parametros para 1030 muestras. Cierra la seccion de arquitectura
de la CNN, despues de la
Fig. 12, la Fig. 13 de curvas y la Fig. 14 de regresion por variante.

Trae dos bloques de columnas:

  Validacion cruzada   MAE, RMSE y R2 medios sobre los seis pliegues, promediados
                       a su vez sobre las repeticiones disponibles.
  Epoca 40             MAE de entrenamiento y de validacion, y su razon. Es la
                       columna que decide: crece de forma monotona con el tamano
                       del modelo mientras el error de validacion se mantiene
                       plano, que es la firma de la memorizacion.

Dos advertencias sobre la procedencia de los numeros:

1. NO reutiliza resultados/r12_arquitecturas/R12_tabla_comparativa.xlsx, calculada
   antes del arreglo de determinismo del 2026-08-17. Aquella corrida no se
   reproduce: hasta esa fecha la semilla declarada no fijaba la inicializacion de
   pesos en Keras 3, asi que citarla seria citar un numero irrepetible.

2. El numero de repeticiones es desigual: cuatro para la variante publicada, una
   para las otras tres. La tabla lo declara en una columna en vez de esconderlo,
   porque de ahi depende cuanto pesa cada desviacion.

El argumento que sostiene la tabla no es que una gane por desempeno agregado -las
cuatro caen dentro del ruido de repetir el mismo experimento, y el conteo de
parametros no compra R2- sino que la variante publicada es la que menos comprime
la escala y la que no memoriza: su error sobre el propio conjunto de entrenamiento
se detiene por encima de la repetibilidad del perfilometro, y el de D queda 64
veces por debajo.

Uso:  python tabla5_arquitecturas.py
      python tabla5_arquitecturas.py --param Rz

--param Rz agregado el 2026-09-09, una vez completado el barrido de C y D en Rz. La columna de
comparacion contra el MAE de entrenamiento de D publicado (MAE_D_PUBLICADO) es especifica de Ra
-sale del historial del modelo final original, que solo se guardo para Ra ("MULTI_VC_RA")- y no
tiene equivalente documentado para Rz, asi que para Rz esa celda queda en blanco en vez de
inventarse un numero.
"""
import argparse
import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

import comun as C
from fig13_comparacion_arquitecturas import curvas, repeticiones

# MAE final sobre el propio conjunto de entrenamiento, modelo final de 6 probetas.
# Es la cifra que desarma la objecion de sobreajuste: D bajaba a 0.0055 um, muy por
# debajo de la repetibilidad del perfilometro, o sea que memorizaba ruido de
# medicion. Las variantes con reduccion espacial se detienen en el orden de su
# propio error sobre datos no vistos.
#
# D sale de la corrida archivada de dic-2025, ultima epoca de
# "Entrenar modelo/MULTI_VC_RA/historial_entrenamiento_final.xlsx". Es la unica que
# se deja escrita porque su historial no vive en resultados/.
MAE_D_PUBLICADO = 0.0055


def mae_entrenamiento_publicada(param):
    """Media, sobre las repeticiones, del MAE de entrenamiento de la ultima epoca.

    Se lee de resultados/<X>_oficial/<param>/metricas/historial_por_epoca_seed*.xlsx
    en vez de escribirse a mano: cambiar de arquitectura no debe obligar a recordar
    actualizar una constante.
    """
    carpeta = C.oficial(param) / param / 'metricas'
    archivos = sorted(carpeta.glob('historial_por_epoca_seed*.xlsx'))
    if not archivos:
        raise SystemExit(
            f"Falta el historial del modelo final en {carpeta}. Correr primero, "
            f"con el entorno tesis:\n"
            f"  entrenar_final_limpio.py --arch {C.ARQ_PUBLICADA[param]} --param {param} --seed <N>")
    return float(np.mean([pd.read_excel(f)['mae'].iloc[-1] for f in archivos])), len(archivos)


# Dispersion entre las seis trazas del SJ-310 dentro de una zona de 15 x 15 mm.
# Es la vara fisica: por debajo de esto, el modelo ajusta ruido de medicion.
REPETIBILIDAD_REFERENCIA = 0.350


def ms(v, dec=3):
    """Formatea un vector como 'media ± sd', o solo la media si hay un valor."""
    if len(v) > 1:
        return f"{v.mean():.{dec}f} ± {v.std(ddof=1):.{dec}f}"
    return f"{v.mean():.{dec}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    args = ap.parse_args()
    param = args.param

    mae_final_pub, n_rep_final = mae_entrenamiento_publicada(param)
    reps = repeticiones(param)
    filas, detalle = [], []
    for etiqueta, descripcion, params, elegida in C.ARQUITECTURAS:
        # --- validacion cruzada, una media por repeticion --------------------
        r2, mae, rmse = [], [], []
        for carpeta in reps[etiqueta]:
            folds = C.leer_vc(carpeta)
            r2.append(folds.R2.mean())
            mae.append(folds.MAE.mean())
            rmse.append(folds.RMSE.mean())
            detalle.append(dict(arch=etiqueta, corrida=carpeta,
                                R2=folds.R2.mean(), MAE=folds.MAE.mean(),
                                RMSE=folds.RMSE.mean(),
                                R2_sd_entre_pliegues=folds.R2.std(ddof=1)))
        r2, mae, rmse = np.array(r2), np.array(mae), np.array(rmse)

        # --- epoca 40, sobre todas las repeticiones x pliegues ---------------
        tr, va, _ = curvas(etiqueta, 'repeticiones', param)
        mae_tr, mae_va = tr[:, -1].mean(), va[:, -1].mean()

        # La comparacion contra el D publicado originalmente solo existe para Ra
        # (MULTI_VC_RA es el unico historial de modelo final que se guardo del
        # envio original). No inventar el numero para Rz.
        if etiqueta == C.ARQ_PUBLICADA[param]:
            final_train = f'{mae_final_pub:.3f}'
        elif etiqueta == 'D' and param == 'Ra':
            final_train = f'{MAE_D_PUBLICADO:g}'
        else:
            final_train = '—'

        filas.append({
            'Variant': etiqueta,
            'Reduction stage': descripcion,
            'Parameters': params,
            'Repetitions': len(r2),
            'MAE [µm]': ms(mae),
            'RMSE [µm]': ms(rmse),
            'R²': ms(r2),
            'Train MAE [µm]': f'{mae_tr:.3f}',
            'Val / Train': f'{mae_va / mae_tr:.2f}',
            'Final-model train MAE [µm]': final_train,
            '_R2': r2.mean(), '_razon': mae_va / mae_tr,
            '_tr': mae_tr, '_va': mae_va, '_cv_mae': mae.mean(),
        })

    tabla = pd.DataFrame(filas)
    det = pd.DataFrame(detalle)
    visible = [c for c in tabla.columns if not c.startswith('_')]
    # Uniformidad con la Tabla 6 (validacion cruzada), que no repite "CV" en cada
    # columna. Las columnas quedan sin el prefijo; el grupo se declara una sola
    # vez, como encabezado de dos niveles en el xlsx y como nota en el md.
    grupo_cv = ['MAE [µm]', 'RMSE [µm]', 'R²']
    grupo_epoca = ['Train MAE [µm]', 'Val / Train', 'Final-model train MAE [µm]']
    grupo = {c: 'Cross-validation (6 folds)' if c in grupo_cv
             else 'Epoch 40' if c in grupo_epoca else '' for c in visible}

    print(f"Comparación de las cuatro variantes — validación cruzada sobre {param}\n")
    print(tabla[visible].to_string(index=False))

    # ------------------------------------------------------- el argumento ----
    entre_arq = tabla._R2.max() - tabla._R2.min()
    # La vara es la variante que tiene repeticiones: repetir el MISMO experimento
    # con otra inicializacion y ver cuanto se mueve. Si eso es comparable a lo que
    # separa a las cuatro, la comparacion no tiene resolucion para ordenarlas.
    rep_pub = det[det.arch == C.ARQ_PUBLICADA[param]].R2
    entre_rep = rep_pub.max() - rep_pub.min()
    print(f"\nRango de R² ENTRE arquitecturas               : {entre_arq:.3f}  "
          f"(mejor {tabla.loc[tabla._R2.idxmax(), 'Variant']}, "
          f"peor {tabla.loc[tabla._R2.idxmin(), 'Variant']})")
    print(f"Rango de R² ENTRE repeticiones de {C.ARQ_PUBLICADA[param]} sola      : "
          f"{entre_rep:.3f}  ({len(rep_pub)} repeticiones)")
    print(f"  -> la variabilidad de repetir el mismo experimento es "
          f"{'comparable o mayor' if entre_rep >= entre_arq * 0.8 else 'menor'} "
          f"que la que separa a las cuatro arquitecturas")

    # Chequeo cruzado: el MAE de validacion de la epoca 40 tiene que coincidir con
    # el MAE medio de la validacion cruzada. Son dos archivos distintos
    # (historial_por_epoca.xlsx y metricas_validacion_cruzada.xlsx) y si no cuadran,
    # alguno de los dos esta mal.
    desvio = (tabla._va - tabla._cv_mae).abs().max()
    print(f"\n[{'OK ' if desvio < 5e-3 else 'REVISAR'}] el MAE de validación de la "
          f"época 40 coincide con el de la validación cruzada (máx. "
          f"{desvio:.4f} µm de diferencia)")

    razones = tabla._razon.values
    print(f"[{'OK ' if np.all(np.diff(razones) > 0) else 'REVISAR'}] la razón "
          f"val/train crece con el tamaño: {', '.join(f'{r:.2f}×' for r in razones)}")
    print(f"      el error de validación se mantiene plano entre "
          f"{tabla._va.min():.3f} y {tabla._va.max():.3f} µm mientras el de "
          f"entrenamiento cae de {tabla._tr.max():.3f} a {tabla._tr.min():.3f} µm")

    print("\nMAE del modelo final sobre su propio entrenamiento (6 probetas):")
    print(f"  {C.ARQ_PUBLICADA[param]} : {mae_final_pub:.3f} µm  "
          f"(media de {n_rep_final} repeticiones)")
    if param == 'Ra':
        print(f"  D : {MAE_D_PUBLICADO:g} µm")
        print("  repetibilidad de la referencia (6 trazas por zona): "
              f"{REPETIBILIDAD_REFERENCIA:.3f} µm")
        print(f"  -> D ajusta {REPETIBILIDAD_REFERENCIA / MAE_D_PUBLICADO:.0f} veces por "
              f"debajo de lo que el perfilómetro puede repetir; {C.ARQ_PUBLICADA[param]} se "
              f"detiene {mae_final_pub / REPETIBILIDAD_REFERENCIA:.1f} veces por encima")
    else:
        print("  D : — (no hay historial del modelo final original de D para Rz, "
              "solo se guardo para Ra en MULTI_VC_RA; no se compara)")

    sufijo = '' if param == 'Ra' else f'_{param}'
    destino = C.SALIDAS / f'Tabla5_arquitecturas{sufijo}.xlsx'
    with pd.ExcelWriter(destino) as w:
        # Encabezado de dos niveles: el grupo (arriba) declara "Cross-validation"
        # o "Epoch 40" una sola vez; las columnas ya no llevan el prefijo "CV ".
        con_grupo = tabla[visible].copy()
        con_grupo.columns = pd.MultiIndex.from_tuples(
            [(grupo[c], c) for c in visible])
        # xlsxwriter no soporta MultiIndex de columnas con index=False; se deja
        # un indice posicional sin nombre y no molesta (no es dato del articulo).
        con_grupo.to_excel(w, sheet_name='Comparacion', index=True, index_label='')
        det.to_excel(w, sheet_name='Por_corrida', index=False)
    print(f"\n-> {destino}")

    # El md es tabla simple (sin fusionar celdas); el grupo va como nota, no en
    # el encabezado, para no repetir "CV" y para no depender de que el visor de
    # Word sepa fusionar celdas al pegarlo.
    md = ['| ' + ' | '.join(visible) + ' |', '|' + '---|' * len(visible)]
    for _, r in tabla.iterrows():
        celdas = [f"{r['Parameters']:,}".replace(',', ' ') if col == 'Parameters'
                  else str(r[col]) for col in visible]
        md.append('| ' + ' | '.join(celdas) + ' |')
    nota = (f"\n*{', '.join(grupo_cv)} are 6-fold cross-validation metrics; "
            f"{', '.join(grupo_epoca)} refer to epoch 40 of training.*")
    texto = '\n'.join(md) + nota
    (C.SALIDAS / f'Tabla5_arquitecturas{sufijo}.md').write_text(texto, encoding='utf-8')
    print(f"\n{texto}")


if __name__ == '__main__':
    main()
