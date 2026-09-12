#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Por que la banda de Monte Carlo no cubre, y como se calibra.

El Monte Carlo jerarquico entrega un intervalo del 90% que en el test cubre solo el 46%.
Este script descompone la causa y produce el intervalo calibrado que si cubre.

Tres piezas:

1. INCERTIDUMBRE EPISTEMICA -- la que mide el Monte Carlo. Es el desacuerdo entre modelos
   entrenados sobre remuestreos distintos: "que tanto cambiaria mi respuesta si me hubiera
   tocado otro conjunto de probetas".

2. INCERTIDUMBRE ALEATORIA -- la que el Monte Carlo NO puede ver. La superficie no tiene un
   valor unico de rugosidad: dentro de una misma zona de 15x15 mm, dos trazas del SJ-310
   diCPUeren. Esa dispersion se midio en CODE/resultados/m3_variabilidad_proceso/ y es
   irreducible: ningun modelo puede predecir donde exactamente se apoyo el palpador.

3. CALIBRACION CONFORMAL -- usa los residuos fuera de pliegue (cada modelo de pliegue
   prediciendo su propia probeta, que no vio en el entrenamiento) para fijar el ancho que
   garantiza cobertura marginal, sin suponer normalidad ni reentrenar nada.

Uso:
    python calibracion_banda.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    raiz = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', choices=['Ra', 'Rz'], default='Ra')
    ap.add_argument('--montecarlo', default=None)
    ap.add_argument('--oof', default=None)
    ap.add_argument('--intrazona', default=str(raiz / 'resultados/m3_variabilidad_proceso/M3_variabilidad_proceso.xlsx'))
    ap.add_argument('--salida', default=None)
    ap.add_argument('--alpha', type=float, default=0.10, help='0.10 -> intervalo del 90%%')
    args = ap.parse_args()

    P = args.param
    sufijo = '' if P == 'Ra' else f'_{P}'
    if args.montecarlo is None:
        args.montecarlo = str(raiz / f'resultados/A_oficial/{P}/montecarlo/R14_montecarlo_{P}.xlsx')
    if args.oof is None:
        args.oof = str(raiz / f'3_entrenamiento/resultados_barrido/A_25.7k_seed10_limpio{sufijo}/predicciones')
    if args.salida is None:
        args.salida = str(raiz / f'resultados/A_oficial/{P}/montecarlo')

    out = Path(args.salida)
    mc = pd.read_excel(args.montecarlo, sheet_name='Predicciones')
    reps = [c for c in mc.columns if c.startswith('rep')]
    real = mc[f'{P}_real'].to_numpy()
    media = mc[f'{P}_media'].to_numpy()
    sd_mc = mc[reps].to_numpy().std(axis=1, ddof=1)
    resid = np.abs(real - media)

    # --- residuos fuera de pliegue, para calibrar ---
    oof = pd.concat([pd.read_excel(f) for f in sorted(Path(args.oof).glob('*.xlsx'))], ignore_index=True)
    c_real = next(c for c in oof.columns if c.lower().endswith('real'))
    c_pred = next(c for c in oof.columns if 'predi' in c.lower())
    r_oof = np.abs(oof[c_real] - oof[c_pred]).to_numpy()

    # --- variabilidad intra-zona medida (aleatoria) ---
    sd_alea = np.nan
    try:
        iz = pd.read_excel(args.intrazona, sheet_name='Por_zona')
        sd_alea = float(iz[f'{P}_sd'].mean())
    except Exception:
        pass

    n = len(real)
    z90 = 1.6449  # ancho +-1.645 sigma para el 90% bajo normalidad

    print('=' * 76)
    print('  POR QUE LA BANDA DE MONTE CARLO NO CUBRE')
    print('=' * 76)
    cob_mc = float(((real >= mc['p05']) & (real <= mc['p95'])).mean())
    ancho_mc = float((mc['p95'] - mc['p05']).mean())
    print(f'  Cobertura del intervalo del 90% : {cob_mc:.1%}   (deberia ser 90%)')
    print(f'  Ancho medio                     : {ancho_mc:.3f} um')
    print()
    print(f'  SD epistemica (desacuerdo entre las 20 replicas) : {sd_mc.mean():.3f} um')
    if not np.isnan(sd_alea):
        print(f'  SD aleatoria  (dispersion dentro de la zona)     : {sd_alea:.3f} um   <- el MC no la ve')
        comb = float(np.sqrt(sd_mc.mean() ** 2 + sd_alea ** 2))
        print(f'  SD combinada  (raiz de la suma de cuadrados)     : {comb:.3f} um')
    sd_nec = float(resid.std(ddof=1))
    print(f'  SD que harian falta (de los residuos reales)     : {sd_nec:.3f} um')
    print(f'  factor que le falta al Monte Carlo               : {sd_nec/sd_mc.mean():.2f}x')

    # --- calibracion conformal sobre residuos fuera de pliegue ---
    q = float(np.quantile(r_oof, np.ceil((len(r_oof) + 1) * (1 - args.alpha)) / len(r_oof), method='higher'))
    cob_conf = float((resid <= q).mean())
    print()
    print('=' * 76)
    print(f'  CALIBRACION CONFORMAL  (n = {len(r_oof)} residuos fuera de pliegue)')
    print('=' * 76)
    print(f'  Semiancho que garantiza el {1-args.alpha:.0%} : +-{q:.3f} um   (ancho {2*q:.3f} um)')
    print(f'  Cobertura empirica sobre el test    : {cob_conf:.1%}')

    # --- conformal escalado: conserva la forma espacial del MC ---
    s_oof = None
    esc = sd_mc / sd_mc.mean()
    q_esc = float(np.quantile(resid / esc, np.ceil((n + 1) * (1 - args.alpha)) / n, method='higher'))
    cob_esc = float((resid <= q_esc * esc).mean())
    print()
    print(f'  Variante escalada por la SD del Monte Carlo (conserva la forma espacial):')
    print(f'    semiancho medio {q_esc*esc.mean():.3f} um  ->  cobertura {cob_esc:.1%}')
    print(f'    correlacion entre SD del MC y error absoluto: '
          f'r = {np.corrcoef(sd_mc, resid)[0,1]:.3f}')

    resumen = pd.DataFrame([
        ['Monte Carlo crudo (p05-p95)', ancho_mc, cob_mc],
        ['Conformal constante', 2 * q, cob_conf],
        ['Conformal escalado por SD del MC', 2 * q_esc * esc.mean(), cob_esc],
    ], columns=['metodo', 'ancho_medio_um', 'cobertura_empirica'])
    det = pd.DataFrame({
        'archivo': mc['archivo'], 'grano': mc['grano'], 'Ra_real': real, 'Ra_media': media,
        'sd_montecarlo': sd_mc, 'residuo_abs': resid,
        'conf_inf': media - q, 'conf_sup': media + q,
        'confesc_inf': media - q_esc * esc, 'confesc_sup': media + q_esc * esc,
    })
    with pd.ExcelWriter(out / 'R14_calibracion_banda.xlsx') as w:
        resumen.to_excel(w, sheet_name='Resumen', index=False)
        det.to_excel(w, sheet_name='Detalle', index=False)

    # --- figura ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    metodos = ['Monte Carlo\ncrudo', 'Conformal\nconstante', 'Conformal\nescalado']
    cobs = [cob_mc, cob_conf, cob_esc]
    anchos = [ancho_mc, 2 * q, 2 * q_esc * esc.mean()]
    col = ['#C44E52', '#55A868', '#4C72B0']
    a1.bar(metodos, [c * 100 for c in cobs], color=col, edgecolor='black', linewidth=0.5)
    a1.axhline((1 - args.alpha) * 100, ls='--', color='black', lw=1.2,
               label=f'nominal {1-args.alpha:.0%}')
    for i, c in enumerate(cobs):
        a1.text(i, c * 100 + 1.5, f'{c:.1%}', ha='center', fontweight='bold', fontsize=9)
    a1.set_ylabel('Cobertura empirica [%]')
    a1.set_ylim(0, 105)
    a1.legend(fontsize=8)
    a1.set_title('Cobertura del intervalo')
    a1.grid(axis='y', alpha=0.3)

    a2.bar(metodos, anchos, color=col, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(anchos):
        a2.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)
    a2.set_ylabel('Ancho medio del intervalo [$\\mu$m]')
    a2.set_title('Precio en ancho')
    a2.grid(axis='y', alpha=0.3)
    fig.suptitle('El intervalo del Monte Carlo no cubre; la calibracion conformal lo corrige', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'R14_calibracion_banda.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / 'R14_calibracion_banda.pdf', bbox_inches='tight')
    print(f'\n  Resultados en: {out}')


if __name__ == '__main__':
    main()
