#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Efecto de la orientacion de la traza sobre la rugosidad medida y sobre el error del modelo.

Responde la parte del comentario 1.13 que el proxy de luz ambiente NO cubre, textual:
"the shadowing effect is critical for texture enhancement but may introduce biases
depending on fibre orientation."

El diseno de muestreo lo permite sin ningun experimento nuevo. Cada zona de 15x15 mm tiene
seis trazas del SJ-310: las numeradas 1-3 son HORIZONTALES y las 4-6 son VERTICALES
(confirmado por el autor, 2026-08-17). La iluminacion rasante bilateral es fija, de modo
que las trazas horizontales quedan alineadas con la direccion de la sombra y las verticales
perpendiculares a ella.

Dos preguntas distintas, y hay que separarlas:

  (1) LA SUPERFICIE: la rugosidad medida depende de la direccion en que se mide?
      Es una pregunta de metrologia sobre la madera de testa, ajena al modelo.

  (2) EL MODELO: estima peor las trazas perpendiculares a la iluminacion que las alineadas?
      Si la respuesta es no, no hay sesgo direccional atribuible al modelo.

Uso:
    python efecto_orientacion.py
    python efecto_orientacion.py --validacion ../resultados/validacion_revision/resultados_validacion_Ra.xlsx
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PATRON = re.compile(
    r'P(?P<probeta>\d+)_C(?P<cara>\d+)_G(?P<grano>\d+)_Recorte_Recorte_'
    r'(?P<zona>[ABC])(?P<zrep>\d*)_(?P<znum>\d+)\.(?P<traza>\d+)\.tiff', re.I)


def encontrar_dataset():
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in [Path('Fotos') / 'Recortes15x15mm' / 'recortes_x6', Path('fotos_recortes'), Path('data') / 'fotos_recortes']:
            if (carpeta / sub / 'DATASET.xlsx').is_file():
                return carpeta / sub / 'DATASET.xlsx'
    raise SystemExit('ERROR: no se encontro DATASET.xlsx')


def parsear(nombres):
    out = []
    for n in nombres:
        m = PATRON.match(str(n))
        out.append(m.groupdict() if m else {k: None for k in
                   ['probeta', 'cara', 'grano', 'zona', 'zrep', 'znum', 'traza']})
    return pd.DataFrame(out)


def orienta(t):
    return 'horizontal' if int(t) <= 3 else 'vertical'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hoja', default='sin_outliers')
    ap.add_argument('--validacion', default=None)
    ap.add_argument('--salida', default=None)
    args = ap.parse_args()

    excel = encontrar_dataset()
    out = Path(args.salida) if args.salida else (
        Path(__file__).resolve().parent.parent / 'resultados' / 'r113_orientacion')
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_excel(excel, sheet_name=args.hoja)
    d = pd.concat([d.reset_index(drop=True), parsear(d['nombre_imagen'])], axis=1)
    d = d.dropna(subset=['traza']).copy()
    d['orientacion'] = d['traza'].apply(orienta)

    print('=' * 76)
    print('  (1) LA SUPERFICIE: depende la rugosidad de la direccion de medicion?')
    print('=' * 76)
    print(f'  {"grano":>6} {"n H / V":>12} {"Ra H":>8} {"Ra V":>8} {"p":>9}'
          f' {"Rz H":>8} {"Rz V":>8} {"p":>9}')
    filas = []
    for g in sorted(d['grano'].unique(), key=int):
        s = d[d['grano'] == g]
        h, v = s[s.orientacion == 'horizontal'], s[s.orientacion == 'vertical']
        pra = stats.mannwhitneyu(h['Ra'], v['Ra']).pvalue
        prz = stats.mannwhitneyu(h['Rz'], v['Rz']).pvalue
        print(f'  P{g:<5} {len(h):>5} / {len(v):<4} {h.Ra.mean():>8.3f} {v.Ra.mean():>8.3f}'
              f' {pra:>9.4f} {h.Rz.mean():>8.2f} {v.Rz.mean():>8.2f} {prz:>9.4f}')
        filas.append({'grano': f'P{g}', 'n_h': len(h), 'n_v': len(v),
                      'Ra_h': h.Ra.mean(), 'Ra_v': v.Ra.mean(), 'p_Ra': pra,
                      'Rz_h': h.Rz.mean(), 'Rz_v': v.Rz.mean(), 'p_Rz': prz})
    h, v = d[d.orientacion == 'horizontal'], d[d.orientacion == 'vertical']
    p_ra = stats.mannwhitneyu(h['Ra'], v['Ra']).pvalue
    p_rz = stats.mannwhitneyu(h['Rz'], v['Rz']).pvalue
    print(f'  {"GLOBAL":>6} {len(h):>5} / {len(v):<4} {h.Ra.mean():>8.3f} {v.Ra.mean():>8.3f}'
          f' {p_ra:>9.4f} {h.Rz.mean():>8.2f} {v.Rz.mean():>8.2f} {p_rz:>9.4f}')

    # comparacion pareada dentro de cada zona: controla por posicion
    z = d.groupby(['probeta', 'cara', 'grano', 'zona', 'orientacion'])['Ra'].mean().unstack()
    z = z.dropna()
    if len(z):
        dif = z['horizontal'] - z['vertical']
        w = stats.wilcoxon(z['horizontal'], z['vertical'])
        print()
        print(f'  Pareado dentro de cada zona (n = {len(z)} zonas con ambas orientaciones):')
        print(f'    diferencia media H - V = {dif.mean():+.4f} um   Wilcoxon p = {w.pvalue:.4f}')

    res_modelo = None
    if args.validacion and Path(args.validacion).is_file():
        print()
        print('=' * 76)
        print('  (2) EL MODELO: estima peor las trazas perpendiculares a la iluminacion?')
        print('=' * 76)
        vdf = pd.read_excel(args.validacion)
        cols = {c.lower(): c for c in vdf.columns}
        carch = next((cols[c] for c in cols if 'arch' in c or 'imagen' in c or 'nombre' in c), None)
        creal = next((cols[c] for c in cols if 'real' in c or 'medid' in c), None)
        cpred = next((cols[c] for c in cols if 'pred' in c or 'estim' in c), None)
        if carch and creal and cpred:
            campos = parsear(vdf[carch])
            vdf = vdf.drop(columns=[c for c in campos.columns if c in vdf.columns])
            vdf = pd.concat([vdf.reset_index(drop=True), campos], axis=1)
            vdf = vdf.dropna(subset=['traza']).copy()
            vdf['orientacion'] = vdf['traza'].apply(orienta)
            vdf['err'] = (vdf[creal] - vdf[cpred]).abs()
            hh, vv = vdf[vdf.orientacion == 'horizontal'], vdf[vdf.orientacion == 'vertical']
            pe = stats.mannwhitneyu(hh['err'], vv['err']).pvalue
            print(f'  n horizontales {len(hh)} | n verticales {len(vv)}')
            print(f'  MAE trazas horizontales (alineadas con la iluminacion) : {hh.err.mean():.4f} um')
            print(f'  MAE trazas verticales   (perpendiculares)              : {vv.err.mean():.4f} um')
            print(f'  diferencia {vv.err.mean()-hh.err.mean():+.4f} um   Mann-Whitney p = {pe:.4f}')
            print()
            print(f'  {"grano":>6} {"MAE H":>9} {"MAE V":>9} {"dif":>9} {"p":>9}')
            for g in sorted(vdf['grano'].dropna().unique(), key=int):
                s = vdf[vdf['grano'] == g]
                a, b = s[s.orientacion == 'horizontal'], s[s.orientacion == 'vertical']
                if len(a) > 2 and len(b) > 2:
                    pp = stats.mannwhitneyu(a['err'], b['err']).pvalue
                    print(f'  P{g:<5} {a.err.mean():>9.4f} {b.err.mean():>9.4f}'
                          f' {b.err.mean()-a.err.mean():>+9.4f} {pp:>9.4f}')
            res_modelo = pd.DataFrame([{'MAE_horizontal': hh.err.mean(), 'MAE_vertical': vv.err.mean(),
                                        'diferencia': vv.err.mean() - hh.err.mean(), 'p': pe,
                                        'n_h': len(hh), 'n_v': len(vv)}])

    t = pd.DataFrame(filas)
    with pd.ExcelWriter(out / 'R113_efecto_orientacion.xlsx') as w:
        t.to_excel(w, sheet_name='Superficie_por_grano', index=False)
        if res_modelo is not None:
            res_modelo.to_excel(w, sheet_name='Modelo', index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(t))
    a1.bar(x - 0.2, t['Ra_h'], 0.4, label='Trazas horizontales (alineadas con la luz)',
           color='#4C72B0', edgecolor='black', linewidth=0.5)
    a1.bar(x + 0.2, t['Ra_v'], 0.4, label='Trazas verticales (perpendiculares)',
           color='#DD8452', edgecolor='black', linewidth=0.5)
    for i, r in t.iterrows():
        a1.text(i, max(r['Ra_h'], r['Ra_v']) + 0.15, 'n.s.' if r['p_Ra'] > 0.05 else f"p={r['p_Ra']:.3f}",
                ha='center', fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(t['grano'])
    a1.set_ylabel('R$_a$ medida [$\\mu$m]'); a1.set_xlabel('Grano abrasivo')
    a1.set_title('La superficie: rugosidad segun direccion de medicion')
    a1.legend(fontsize=7.5); a1.grid(axis='y', alpha=0.3)

    if res_modelo is not None:
        a2.bar(['Horizontales\n(alineadas)', 'Verticales\n(perpendiculares)'],
               [res_modelo['MAE_horizontal'][0], res_modelo['MAE_vertical'][0]],
               color=['#4C72B0', '#DD8452'], edgecolor='black', linewidth=0.5)
        a2.set_ylabel('MAE del modelo [$\\mu$m]')
        a2.set_title(f"El modelo: error segun orientacion (p = {res_modelo['p'][0]:.3f})")
        a2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'R113_efecto_orientacion.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / 'R113_efecto_orientacion.pdf', bbox_inches='tight')

    (out / 'README.md').write_text(
        '# 1.13 — sesgo por orientacion de la traza\n\n'
        'Las trazas 1-3 son horizontales y las 4-6 verticales (confirmado por el autor,\n'
        '2026-08-17). Como la iluminacion rasante es fija, las horizontales quedan alineadas\n'
        'con la direccion de la sombra y las verticales perpendiculares. Separa dos preguntas:\n'
        'si la superficie es anisotropa, y si el modelo tiene sesgo direccional.\n\n'
        'Genera: `python CODE/2_dataset/efecto_orientacion.py --validacion <xlsx de test>`\n',
        encoding='utf8')
    print(f'\n  Resultados en: {out}')


if __name__ == '__main__':
    main()
