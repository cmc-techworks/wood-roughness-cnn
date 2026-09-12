#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predicciones del modelo final A sobre el test independiente, semilla por semilla.

Es el unico paso de esta carpeta que necesita TensorFlow: corre con el entorno
`tesis` (python), no con el Python del
sistema.

Por que hace falta. El consolidado que dejan los entrenamientos trae
solo R2 y MAE agregados; la Tabla 6 del manuscrito pide ademas MSE y RMSE, y las
Fig. 15-17 necesitan las predicciones punto a punto. La unica corrida con
predicciones guardadas (`resultados/validacion_revision/`) es de semilla no
identificada, asi que se regeneran las cuatro.

No reentrena nada: carga los cuatro `.keras` ya guardados y hace inferencia sobre
las 190 muestras validas de la probeta 11. Delega en `4_validacion/validar_headless.py`
para que el preprocesado sea identico al del resto del expediente.

Salidas:
    resultados/<X>_oficial/<param>/test/seed<N>/resultados_validacion_<param>.xlsx
    resultados/<X>_oficial/<param>/test/metricas_test_por_semilla.xlsx   <- consolidado

Uso:
    python 00_predicciones_test.py
    ... 00_predicciones_test.py --seeds 10          # solo una, para probar
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun as C

AQUI = Path(__file__).resolve().parent
CODE = AQUI.parent
RESULTADOS = CODE / 'resultados'
VALIDAR = CODE / '4_validacion' / 'validar_headless.py'

SEEDS = [10, 11, 12, 13]
PARAMS = ['Ra', 'Rz']

# Cifras de control de la arquitectura A, que fue la primera que se consolido con
# este script. Solo se cotejan si ARQ_PUBLICADA vuelve a ser A; con cualquier otra
# no hay contra que comparar todavia y el cotejo se omite en vez de dar falsa alarma.
ESPERADO = {'A': {'Ra': dict(R2=0.87640, MAE=0.418425),
                  'Rz': dict(R2=0.76865, MAE=3.549575)}}


def metricas(xlsx, param, seed):
    met = pd.read_excel(xlsx, sheet_name='Métricas').set_index('Métrica')['Valor']
    return dict(param=param, seed=seed,
                n=len(pd.read_excel(xlsx, sheet_name='Predicciones')),
                R2=float(met['R²']), MAE=float(met['MAE (µm)']),
                RMSE=float(met['RMSE (µm)']), MSE=float(met['MSE (µm²)']),
                MAPE=float(met['MAPE (%)']))


def correr(param, seed, reusar=False):
    arq = C.ARQ_PUBLICADA[param]
    modelo = (RESULTADOS / f'{arq}_oficial' / param / 'modelos' /
              f'modelo_{arq}_{param}_final_seed{seed}.keras')
    salida = RESULTADOS / f'{arq}_oficial' / param / 'test' / f'seed{seed}'
    if not modelo.is_file():
        sys.exit(f"ERROR: no existe el modelo {modelo}")
    salida.mkdir(parents=True, exist_ok=True)
    xlsx = salida / f'resultados_validacion_{param}.xlsx'

    # La campana de entrenamiento llama a validar_headless.py despues de cada
    # modelo, asi que las predicciones suelen existir ya. Releerlas en vez de
    # rehacer la inferencia ahorra un cuarto de hora y da exactamente lo mismo:
    # mismo modelo, mismo preprocesado, misma probeta.
    if reusar and xlsx.is_file():
        print(f"{param} · seed {seed}: se reusa {xlsx.name}")
        return metricas(xlsx, param, seed)

    print(f"\n{'=' * 70}\n{param} · seed {seed}\n{'=' * 70}")
    cmd = [sys.executable, str(VALIDAR),
           '--param', param,
           '--modelo', str(modelo),
           '--salida', str(salida)]
    res = subprocess.run(cmd, cwd=str(VALIDAR.parent))
    if res.returncode != 0:
        sys.exit(f"ERROR: validar_headless.py fallo para {param} seed {seed}")

    return metricas(xlsx, param, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--params', nargs='+', choices=PARAMS, default=PARAMS)
    ap.add_argument('--reusar', action='store_true',
                    help='no rehacer la inferencia si el .xlsx ya existe')
    args = ap.parse_args()

    filas = [correr(p, s, args.reusar) for p in args.params for s in args.seeds]
    df = pd.DataFrame(filas)

    print(f"\n\n{'=' * 70}\nCONSOLIDADO\n{'=' * 70}")
    for param in args.params:
        sub = df[df.param == param]
        destino = C.oficial(param) / param / 'test'
        destino.mkdir(parents=True, exist_ok=True)

        resumen = pd.DataFrame([dict(
            param=param, n_repeticiones=len(sub),
            R2_medio=sub.R2.mean(), R2_sd=sub.R2.std(ddof=1),
            MAE_medio=sub.MAE.mean(), MAE_sd=sub.MAE.std(ddof=1),
            RMSE_medio=sub.RMSE.mean(), RMSE_sd=sub.RMSE.std(ddof=1),
            MSE_medio=sub.MSE.mean(), MSE_sd=sub.MSE.std(ddof=1),
        )])
        with pd.ExcelWriter(destino / 'metricas_test_por_semilla.xlsx') as w:
            sub.to_excel(w, sheet_name='Por_semilla', index=False)
            resumen.to_excel(w, sheet_name='Resumen', index=False)

        print(f"\n{param}")
        print(sub[['seed', 'n', 'R2', 'MAE', 'RMSE', 'MSE']].to_string(
            index=False, float_format=lambda v: f'{v:.4f}'))
        print(f"  media  R2 = {sub.R2.mean():.4f} ± {sub.R2.std(ddof=1):.4f}"
              f"   MAE = {sub.MAE.mean():.4f} ± {sub.MAE.std(ddof=1):.4f}"
              f"   RMSE = {sub.RMSE.mean():.4f} ± {sub.RMSE.std(ddof=1):.4f}")

        control = ESPERADO.get(C.ARQ_PUBLICADA[param], {})
        if len(sub) == len(SEEDS) and param in control:
            e = control[param]
            d_r2 = abs(sub.R2.mean() - e['R2'])
            d_mae = abs(sub.MAE.mean() - e['MAE'])
            ok = "OK" if (d_r2 < 5e-4 and d_mae < 5e-4) else "REVISAR"
            print(f"  [{ok}] contra metricas_test_por_repeticion.xlsx: "
                  f"dR2 = {d_r2:.5f}, dMAE = {d_mae:.5f}")

    print("\nListo.")


if __name__ == '__main__':
    main()
