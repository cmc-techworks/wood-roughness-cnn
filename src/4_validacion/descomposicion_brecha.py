#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descomposicion de la brecha entre validacion cruzada y test independiente.

Responde a R3.3 (viñeta 3): separa cuanto de la brecha se explica por el tamano del
conjunto de entrenamiento (los modelos de pliegue ven 5 probetas, el final ve 6) y
cuanto por la identidad de la probeta evaluada.

No entrena nada. Usa los 6 modelos de pliegue ya guardados, evaluados sobre la probeta
de test por ensemble_incertidumbre.py, mas las metricas de validacion cruzada de la
misma corrida.

Uso:
    python descomposicion_brecha.py \
        --ensemble ../resultados/A_oficial/Ra/incertidumbre_ensemble/R14_ensemble_incertidumbre_Ra.xlsx \
        --cv ../3_entrenamiento/resultados_barrido/A_25.7k_seed10_limpio/metricas/metricas_validacion_cruzada.xlsx \
        --r2-final 0.8778 --salida ../resultados/A_oficial/Ra/brecha_cv_test
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', required=True)
    ap.add_argument('--cv', required=True)
    ap.add_argument('--r2-final', type=float, required=True,
                    help='R2 del modelo final (6 probetas) sobre el test')
    ap.add_argument('--mae-final', type=float, default=None)
    ap.add_argument('--salida', required=True)
    ap.add_argument('--etiqueta', default='Arquitectura A (25 681 parametros)')
    args = ap.parse_args()

    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_excel(args.ensemble, sheet_name='Predicciones')
    cv = pd.read_excel(args.cv)
    cvf = cv[pd.to_numeric(cv['Fold'], errors='coerce').notna()].copy()
    cvf['Fold'] = cvf['Fold'].astype(int)

    filas = []
    for _, r in cvf.iterrows():
        i = int(r['Fold'])
        col = f'fold{i}_pred'
        filas.append({
            'fold': i,
            'probeta_holdout': int(r['Probeta']),
            'n_val': int(r['n_val']),
            'R2_cv': float(r['R2']),
            'MAE_cv': float(r['MAE']),
            'R2_test': r2_score(pred['Ra_real'], pred[col]),
            'MAE_test': mean_absolute_error(pred['Ra_real'], pred[col]),
        })
    t = pd.DataFrame(filas)

    cv_m, cv_s = t['R2_cv'].mean(), t['R2_cv'].std(ddof=1)
    te_m, te_s = t['R2_test'].mean(), t['R2_test'].std(ddof=1)
    fin = args.r2_final
    d_probeta = te_m - cv_m
    d_datos = fin - te_m
    total = fin - cv_m

    print('=' * 74)
    print(f'  DESCOMPOSICION DE LA BRECHA CV-TEST  |  {args.etiqueta}')
    print('=' * 74)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print()
    print(f'  Media CV   (cada modelo en su holdout) : {cv_m:.4f}  SD {cv_s:.4f}')
    print(f'  Media TEST (los mismos 6 modelos)      : {te_m:.4f}  SD {te_s:.4f}')
    print(f'  Modelo final de 6 probetas sobre test  : {fin:.4f}')
    print()
    print(f'  Identidad de la probeta evaluada : {d_probeta:+.4f}  ({d_probeta/total:.0%} de la brecha)')
    print(f'  Tamano del conjunto de entrenam. : {d_datos:+.4f}  ({d_datos/total:.0%} de la brecha)')
    print(f'  Brecha total                     : {total:+.4f}')
    print()
    print(f'  SD entre probetas evaluadas : {cv_s:.4f}')
    print(f'  SD entre modelos            : {te_s:.4f}')
    print(f'  razon                       : {cv_s/te_s:.2f}x')

    resumen = pd.DataFrame([
        ['Media CV (cada modelo en su holdout)', cv_m, cv_s],
        ['Media test (los mismos 6 modelos)', te_m, te_s],
        ['Modelo final de 6 probetas sobre test', fin, np.nan],
        ['Aporte de la identidad de probeta', d_probeta, np.nan],
        ['Aporte del tamano de entrenamiento', d_datos, np.nan],
        ['Brecha total', total, np.nan],
    ], columns=['concepto', 'valor', 'SD'])
    with pd.ExcelWriter(out / 'R33_descomposicion_brecha.xlsx') as w:
        t.to_excel(w, sheet_name='Por_pliegue', index=False)
        resumen.to_excel(w, sheet_name='Resumen', index=False)

    # ---- figura ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1.35, 1]})
    x = np.arange(len(t))
    ax1.bar(x - 0.2, t['R2_cv'], 0.4, label='En su propia probeta de holdout (CV)',
            color='#4C72B0', edgecolor='black', linewidth=0.5)
    ax1.bar(x + 0.2, t['R2_test'], 0.4, label='Sobre la probeta de test',
            color='#DD8452', edgecolor='black', linewidth=0.5)
    ax1.axhline(cv_m, color='#4C72B0', ls='--', lw=1.2)
    ax1.axhline(te_m, color='#DD8452', ls='--', lw=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'F{r.fold}\nP{r.probeta_holdout}' for r in t.itertuples()])
    ax1.set_ylabel('R$^2$')
    ax1.set_xlabel('Pliegue / probeta dejada fuera')
    ax1.set_title('El mismo modelo, evaluado sobre dos probetas distintas')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)

    niveles = [cv_m, te_m, fin]
    etiquetas = ['CV\n(5 probetas,\nholdout propio)', 'Test\n(5 probetas,\nprobeta 11)',
                 'Test\n(6 probetas,\nprobeta 11)']
    ax2.bar(range(3), niveles, color=['#4C72B0', '#DD8452', '#55A868'],
            edgecolor='black', linewidth=0.5)
    ax2.errorbar([0, 1], [cv_m, te_m], yerr=[cv_s, te_s], fmt='none', ecolor='black', capsize=4)
    for i, v in enumerate(niveles):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax2.annotate('', xy=(1, te_m), xytext=(0, cv_m),
                 arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5))
    ax2.text(0.5, (cv_m + te_m) / 2 - 0.09, f'probeta\n{d_probeta:+.3f}',
             ha='center', color='crimson', fontsize=8, fontweight='bold')
    ax2.annotate('', xy=(2, fin), xytext=(1, te_m),
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    ax2.text(1.5, (te_m + fin) / 2 - 0.09, f'datos\n{d_datos:+.3f}',
             ha='center', color='darkgreen', fontsize=8, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(etiquetas, fontsize=8)
    ax2.set_ylabel('R$^2$')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Descomposicion de la brecha')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(args.etiqueta, fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(out / 'R33_descomposicion_brecha.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / 'R33_descomposicion_brecha.pdf', bbox_inches='tight')
    print(f'\n  Figura y tabla en: {out}')


if __name__ == '__main__':
    main()
