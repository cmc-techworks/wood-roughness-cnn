#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Variabilidad natural del proceso dentro de una misma zona de medicion.

Pone el error del modelo en escala: cuanto significa un MAE dado frente a la
variabilidad propia del proceso de lijado.

La clave es el diseno de muestreo: cada zona de 15x15 mm recibe SEIS trazas del SJ-310
(3 horizontales + 3 verticales, separadas entre si). Esas seis mediciones son la misma
superficie nominal medida seis veces en posiciones distintas, asi que su dispersion ES
la variabilidad espacial natural del proceso de lijado a la escala del parche de imagen.

Comparar el MAE del modelo contra esa dispersion da la respuesta directa: cuanto del
error es irreducible porque la propia superficie no tiene un unico valor de rugosidad.

No entrena nada. Solo lee DATASET.xlsx y, opcionalmente, las predicciones del test.

Uso:
    python variabilidad_proceso.py
    python variabilidad_proceso.py --validacion ../resultados/validacion_revision/resultados_validacion_Ra.xlsx
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATRON = re.compile(
    r'P(?P<probeta>\d+)_C(?P<cara>\d+)_G(?P<grano>\d+)_Recorte_Recorte_'
    r'(?P<zona>[ABC])(?P<zrep>\d*)_(?P<znum>\d+)\.(?P<mnum>\d+)\.tiff', re.I)


def encontrar_dataset():
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        for sub in [Path('Fotos') / 'Recortes15x15mm' / 'recortes_x6', Path('fotos_recortes'), Path('data') / 'fotos_recortes']:
            c = carpeta / sub
            if (c / 'DATASET.xlsx').is_file():
                return c / 'DATASET.xlsx'
    raise SystemExit('ERROR: no se encontro DATASET.xlsx')


def parsear(df):
    campos = []
    for n in df['nombre_imagen']:
        m = PATRON.match(str(n))
        campos.append(m.groupdict() if m else {k: None for k in
                      ['probeta', 'cara', 'grano', 'zona', 'zrep', 'znum', 'mnum']})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(campos)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hoja', default='sin_outliers')
    ap.add_argument('--validacion', default=None,
                    help='Excel de validar_headless.py, para comparar MAE contra la dispersion')
    ap.add_argument('--salida', default=None)
    args = ap.parse_args()

    excel = encontrar_dataset()
    out = Path(args.salida) if args.salida else (
        Path(__file__).resolve().parent.parent / 'resultados' / 'm3_variabilidad_proceso')
    out.mkdir(parents=True, exist_ok=True)

    d = parsear(pd.read_excel(excel, sheet_name=args.hoja))
    sin_parsear = d['probeta'].isna().sum()
    if sin_parsear:
        print(f'Advertencia: {sin_parsear} nombres no siguen el patron esperado')
    d = d.dropna(subset=['probeta']).copy()
    d['mnum'] = d['mnum'].astype(int)

    # --- etiquetas duplicadas: dos recortes que comparten la MISMA traza del SJ-310 ---
    clave = ['probeta', 'cara', 'grano', 'zona', 'mnum']
    n_medidas = d.groupby(clave).ngroups
    print('=' * 74)
    print('  INDEPENDENCIA DE LAS MUESTRAS')
    print('=' * 74)
    print(f'  Filas en la hoja "{args.hoja}"          : {len(d)}')
    print(f'  Trazas independientes del SJ-310       : {n_medidas}')
    print(f'  Recortes que comparten etiqueta        : {len(d) - n_medidas} '
          f'({(len(d)-n_medidas)/len(d):.1%})')

    # --- dispersion DENTRO de cada zona de 15x15 mm ---
    zonas = d.groupby(['probeta', 'cara', 'grano', 'zona']).agg(
        n=('Ra', 'size'), Ra_sd=('Ra', 'std'), Ra_media=('Ra', 'mean'),
        Rz_sd=('Rz', 'std'), Rz_media=('Rz', 'mean')).reset_index()
    zonas = zonas[zonas['n'] >= 4]

    print()
    print('=' * 74)
    print('  VARIABILIDAD NATURAL DENTRO DE UNA ZONA DE 15 x 15 mm')
    print('=' * 74)
    print(f'  Zonas con al menos 4 trazas: {len(zonas)}')
    print()
    filas = []
    for g in sorted(zonas['grano'].unique(), key=int):
        z = zonas[zonas['grano'] == g]
        sub = d[d['grano'] == g]
        filas.append({
            'grano': f'P{g}',
            'n_zonas': len(z),
            'Ra_sd_intrazona': z['Ra_sd'].mean(),
            'Ra_sd_global': sub['Ra'].std(),
            'Rz_sd_intrazona': z['Rz_sd'].mean(),
            'Rz_sd_global': sub['Rz'].std(),
        })
    t = pd.DataFrame(filas)
    t['Ra_frac'] = t['Ra_sd_intrazona'] / t['Ra_sd_global']
    t['Rz_frac'] = t['Rz_sd_intrazona'] / t['Rz_sd_global']

    print(f'  {"grano":>6} {"zonas":>6} {"SD Ra intra":>12} {"SD Ra global":>13} {"%":>6}'
          f' {"SD Rz intra":>12} {"SD Rz global":>13} {"%":>6}')
    for _, r in t.iterrows():
        print(f'  {r["grano"]:>6} {r["n_zonas"]:>6.0f} {r["Ra_sd_intrazona"]:>12.3f}'
              f' {r["Ra_sd_global"]:>13.3f} {r["Ra_frac"]:>5.0%}'
              f' {r["Rz_sd_intrazona"]:>12.2f} {r["Rz_sd_global"]:>13.2f} {r["Rz_frac"]:>5.0%}')
    glob_ra = zonas['Ra_sd'].mean()
    glob_rz = zonas['Rz_sd'].mean()
    print(f'  {"TODOS":>6} {len(zonas):>6} {glob_ra:>12.3f} {d["Ra"].std():>13.3f}'
          f' {glob_ra/d["Ra"].std():>5.0%}'
          f' {glob_rz:>12.2f} {d["Rz"].std():>13.2f} {glob_rz/d["Rz"].std():>5.0%}')

    # --- comparacion contra el error del modelo ---
    if args.validacion and Path(args.validacion).is_file():
        v = pd.read_excel(args.validacion)
        cols = {c.lower(): c for c in v.columns}
        creal = next((cols[c] for c in cols if 'real' in c or 'medid' in c), None)
        cpred = next((cols[c] for c in cols if 'pred' in c or 'estim' in c), None)
        cgra = next((cols[c] for c in cols if 'grano' in c), None)
        if creal and cpred:
            v['err'] = (v[creal] - v[cpred]).abs()
            print()
            print('=' * 74)
            print('  EL ERROR DEL MODELO FRENTE A LA VARIABILIDAD DEL PROCESO')
            print('=' * 74)
            print(f'  {"grano":>6} {"MAE modelo":>12} {"SD intrazona":>13} {"MAE/SD":>8}')
            comp = []
            for _, r in t.iterrows():
                g = int(r['grano'][1:])
                mae = v[v[cgra] == g]['err'].mean() if cgra else np.nan
                comp.append({'grano': r['grano'], 'MAE': mae,
                             'SD_intrazona': r['Ra_sd_intrazona'],
                             'ratio': mae / r['Ra_sd_intrazona']})
                print(f'  {r["grano"]:>6} {mae:>12.3f} {r["Ra_sd_intrazona"]:>13.3f}'
                      f' {mae/r["Ra_sd_intrazona"]:>8.2f}')
            mae_tot = v['err'].mean()
            print(f'  {"TODOS":>6} {mae_tot:>12.3f} {glob_ra:>13.3f} {mae_tot/glob_ra:>8.2f}')
            t = t.merge(pd.DataFrame(comp), on='grano', how='left')

    with pd.ExcelWriter(out / 'M3_variabilidad_proceso.xlsx') as w:
        t.to_excel(w, sheet_name='Por_grano', index=False)
        zonas.to_excel(w, sheet_name='Por_zona', index=False)
        pd.DataFrame([{'filas': len(d), 'trazas_independientes': n_medidas,
                       'etiquetas_compartidas': len(d) - n_medidas}]).to_excel(
            w, sheet_name='Independencia', index=False)

    # --- figura ---
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t['Ra_sd_intrazona'], 0.4, label='Dispersion natural dentro de la zona (15 x 15 mm)',
           color='#8C8C8C', edgecolor='black', linewidth=0.5)
    if 'MAE' in t.columns and t['MAE'].notna().any():
        ax.bar(x + 0.2, t['MAE'], 0.4, label='Error absoluto medio del modelo',
               color='#C44E52', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(t['grano'])
    ax.set_ylabel('R$_a$  [$\\mu$m]')
    ax.set_xlabel('Grano abrasivo')
    ax.set_title('El error del modelo frente a la variabilidad propia de la superficie')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'M3_variabilidad_proceso.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / 'M3_variabilidad_proceso.pdf', bbox_inches='tight')

    (out / 'README.md').write_text(
        '# Variabilidad del proceso\n\n'
        'Dispersion de las 6 trazas del SJ-310 dentro de cada zona de 15x15 mm,\n'
        'comparada con el error del modelo: da el significado practico del error\n'
        'frente a la variabilidad natural del proceso de lijado.\n\n'
        'Genera: `python src/2_dataset/variabilidad_proceso.py --validacion <xlsx de test>`\n',
        encoding='utf8')
    print(f'\n  Resultados en: {out}')


if __name__ == '__main__':
    main()
