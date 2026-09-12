# -*- coding: utf-8 -*-
"""Fig. 20 y 21 — mapa espacial de rugosidad con la arquitectura publicada.

Eran las dos ultimas figuras del articulo generadas con una variante distinta de
la que el texto describe. Se regeneran aqui con `comun.ARQ_PUBLICADA`.

    Fig. 20  (a) panoramica reconstruida  (b) mapa de rugosidad estimada
    Fig. 21  el mismo mapa suavizado, con isolineas y el promedio de la superficie

Que cambia respecto de como se generaron las publicadas, con
`5_heatmap/interfaz_heatmap.py`:

* **Teselado en las dos orientaciones fisicas, con solape.** El protocolo tiene 6
  trazas por zona, las 1-3 horizontales y las 4-6 verticales; el modelo vio las
  dos. La GUI recorre la panoramica en una sola orientacion y, cuando la tesela
  no coincide con la entrada, la redimensiona -- lo que estira la textura 3x en un
  eje. Aqui la orientacion horizontal se resuelve rotando, y el paso es de media
  tesela, de modo que las estimaciones solapadas se promedian.
* **Geometria honesta.** `aspect='equal'` y ejes en milimetros, contra el
  `aspect='auto'` de la GUI, que deforma el mapa respecto de la superficie real.
* **Limites de color explicitos e iguales en las dos figuras**, contra el
  `vmin=3.5` cableado con `vmax` libre de la GUI.
* **PNG y PDF a 300 dpi con fondo blanco** via `comun.guardar()`, contra el
  `transparent=True` de la GUI.

La contrapartida del solape hay que declararla: promediar estimaciones solapadas
es suavizado, asi que el rango dinamico del mapa es mas estrecho que el de una
rejilla disjunta. El script imprime los dos rangos y los deja en el .xlsx.

Uso, con el entorno tesis (el unico que tiene TensorFlow):

    # activar el entorno: conda activate roughness-vision
    $TESIS fig20_21_mapa_espacial.py
    $TESIS fig20_21_mapa_espacial.py --vmin 3.5 --vmax 8.5   # escala de la publicada
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import comun as C                      # fija matplotlib en Agg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(C.CODE / '5_heatmap'))
import nucleo_mapa as NM                # sin tkinter, no toca el backend

# Panoramica por defecto: cara C1 de la probeta 11 tras el ultimo pasado de P40.
# P11 es la probeta de test -- no participo de ninguno de los seis pliegues de
# validacion cruzada ni del entrenamiento del modelo final, asi que el mapa cae
# integramente sobre superficie que el modelo nunca vio.
PANORAMICA = C.DATOS / 'P11' / 'C1' / 'G40' / 'PREPROCESADAS' / 'madera_FinalX.png'
GRANO = 40

# Rango del conjunto limpio de entrenamiento, para el criterio de plausibilidad.
RANGO_ENTRENAMIENTO = {'Ra': (1.750, 9.833), 'Rz': (14.511, 62.780)}


def parsear():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--panoramica', default=str(PANORAMICA))
    p.add_argument('--grano', type=int, default=GRANO, choices=NM.GRAIN_CATEGORIES)
    p.add_argument('--param', default='Ra', choices=['Ra', 'Rz'])
    p.add_argument('--seed', type=int, default=None,
                   help='por defecto, la semilla de referencia del parametro')
    p.add_argument('--modelo', default=None, help='por defecto, el final de comun.ARQ_PUBLICADA')
    p.add_argument('--celda', type=int, default=65,
                   help='lado de la celda del mapa, en px (65 px = 2.5 mm)')
    p.add_argument('--fraccion-paso', type=float, default=0.5,
                   help='paso del teselado como fraccion de la tesela')
    p.add_argument('--alpha', type=float, default=0.45,
                   help='opacidad del mapa sobre la panoramica')
    p.add_argument('--sigma', type=float, default=1.6,
                   help='suavizado gaussiano de la Fig. 21, en celdas (1.6 = 4 mm)')
    p.add_argument('--vmin', type=float, default=None)
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--cmap', default='jet')
    p.add_argument('--sin-zonas', action='store_true',
                   help='no marcar en la Fig. 20b las zonas medidas con el SJ-310 (solo Ra)')
    a = p.parse_args()
    if a.seed is None:
        a.seed = C.SEED_REF[a.param]
    return a


def ruta_modelo(args):
    if args.modelo:
        return Path(args.modelo)
    return (C.oficial(args.param) / args.param / 'modelos' /
            f'modelo_{C.ARQ_PUBLICADA[args.param]}_{args.param}_final_seed{args.seed}.keras')


# ------------------------------------------------------------------ dibujo ---

def _extent_mm(forma_px):
    """(izq, der, abajo, arriba) en mm para imshow con origin='upper'."""
    h, w = forma_px[:2]
    return (0.0, w / NM.PX_POR_MM, h / NM.PX_POR_MM, 0.0)


def _barra_escala(ax, ancho_mm, largo=50.0):
    """Barra de escala abajo a la izquierda. Un mapa espacial la necesita."""
    x0 = ancho_mm * 0.04
    y0 = ax.get_ylim()[0] * 0.955          # ylim esta invertido (origin upper)
    ax.plot([x0, x0 + largo], [y0, y0], color='white', lw=3.4,
            solid_capstyle='butt', zorder=5)
    ax.plot([x0, x0 + largo], [y0, y0], color='black', lw=1.4,
            solid_capstyle='butt', zorder=6)
    ax.text(x0 + largo / 2, y0 - ancho_mm * 0.020, f'{largo:.0f} mm',
            color='white', ha='center', va='bottom', fontsize=10, zorder=7,
            bbox=dict(boxstyle='round,pad=0.18', fc='black', ec='none', alpha=0.55))


def _pintar(ax, img, mapa, ext_img, ext_mapa, vmin, vmax, cmap, alpha, suave):
    ax.imshow(img, cmap='gray', aspect='equal', extent=ext_img, origin='upper',
              interpolation='antialiased')
    im = ax.imshow(np.ma.masked_invalid(mapa), cmap=cmap, alpha=alpha,
                   aspect='equal', extent=ext_mapa, origin='upper',
                   interpolation='bilinear' if suave else 'nearest',
                   vmin=vmin, vmax=vmax, zorder=2)
    ax.set_xlim(ext_img[0], ext_img[1])
    ax.set_ylim(ext_img[2], ext_img[3])
    ax.axis('off')
    return im


def _reservar_cax(ax, size='4%', pad=0.10):
    """Reserva a la derecha del eje el hueco de una barra de color.

    `fig.colorbar(im, ax=...)` le quita ancho al eje que la aloja. Con
    `aspect='equal'` ese eje encoge la imagen para conservar la geometria, de
    modo que el panel con barra sale mas chico que el que no la lleva. Por eso
    se reserva el mismo hueco en los dos paneles de la Fig. 20 y se apaga el del
    panel (a): asi los dos rinden exactamente al mismo tamano.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    return make_axes_locatable(ax).append_axes('right', size=size, pad=pad)


def _colorbar(fig, im, cax, etiqueta):
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(etiqueta, rotation=270, labelpad=20, fontsize=12)
    cb.ax.tick_params(labelsize=11)
    cb.solids.set_alpha(1.0)               # la barra, opaca; el overlay, no
    return cb


def _verificar_paneles_iguales(fig, ax_a, ax_b, tol=0.5):
    """Comprueba que los dos paneles de la Fig. 20 rindan al mismo tamano.

    Es el invariante de la figura: (a) y (b) muestran la misma superficie y
    tienen que ocupar lo mismo. Se mide sobre la geometria ya resuelta, despues
    de que matplotlib aplique el aspecto.
    """
    fig.canvas.draw()
    ba = ax_a.get_window_extent()
    bb = ax_b.get_window_extent()
    dw, dh = abs(ba.width - bb.width), abs(ba.height - bb.height)
    ok = dw <= tol and dh <= tol
    print(f"  paneles (a) {ba.width:.1f} x {ba.height:.1f} px · "
          f"(b) {bb.width:.1f} x {bb.height:.1f} px · "
          f"{'iguales' if ok else f'DIFIEREN en {dw:.1f} x {dh:.1f} px'}")
    return ok


def _suavizar(mapa, sigma):
    """Suavizado gaussiano del mapa, tolerante a celdas sin cobertura.

    Es lo que la Fig. 21 declara en su pie. Hace falta ademas por una razon
    practica: con celdas de 2.5 mm el campo tiene mucho mas detalle que la
    rejilla gruesa de la version publicada, y trazar isolineas sobre el campo
    crudo produce cientos de contornos cerrados diminutos que no se leen.
    """
    from scipy.ndimage import gaussian_filter
    if sigma <= 0:
        return mapa
    valido = np.isfinite(mapa)
    relleno = np.where(valido, mapa, np.nanmean(mapa))
    suave = gaussian_filter(relleno, sigma=sigma, mode='nearest')
    return np.where(valido, suave, np.nan)


def _caja_promedio(ax, texto):
    ax.text(0.975, 0.03, texto, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, zorder=8,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='black',
                      lw=0.6, alpha=0.85))


def _leer_zonas(args):
    """Zonas medidas con el SJ-310, ya localizadas por verificar_mapa_espacial.py.

    El anclaje del mapa contra el perfilometro se describe en el texto, y estos
    recuadros son lo que lo muestra dentro de la figura.
    Solo existe para Ra, porque las mediciones de referencia del mapa son de Ra, y
    solo para la semilla con que se corrio la verificacion. Si algo no calza, la
    figura sale sin recuadros y lo avisa; no se inventan posiciones.
    """
    if args.param != 'Ra' or args.sin_zonas:
        return None
    f = C.SALIDAS / ('Fig20_21_verificacion.xlsx' if args.seed == C.SEED_REF['Ra']
                     else f'Fig20_21_verificacion_seed{args.seed}.xlsx')
    if not f.is_file():
        print(f"  [aviso] falta {f.name}: la Fig. 20 sale sin las zonas del SJ-310")
        return None
    zonas = pd.read_excel(f, sheet_name='Anclaje_por_zona')
    resumen = dict(pd.read_excel(f, sheet_name='Anclaje_resumen').values)
    if not {'ancho_px', 'alto_px'} <= set(zonas.columns) or 'semilla' not in resumen:
        print(f"  [aviso] {f.name} es de antes del 2026-09-10: correr "
              "verificar_mapa_espacial.py --anclaje; la Fig. 20 sale sin zonas")
        return None
    if int(resumen['semilla']) != args.seed:
        print(f"  [aviso] {f.name} es de la semilla {resumen['semilla']}, no de la "
              f"{args.seed}; la Fig. 20 sale sin zonas")
        return None
    return zonas


def _marcar_zonas(ax, zonas):
    """Recuadro de cada recorte medido, y por zona: Ra del SJ-310 (media ± SD) | Ra del mapa.

    A1 y A2 son dos recortes de la misma zona (comparten trazas): se dibujan los
    dos recuadros y una sola etiqueta, con el Ra del mapa promediado entre ambos,
    igual que el MAE zona a zona de verificar_mapa_espacial.py.
    """
    for zona, d in zonas.groupby('zona_medicion'):
        for _, r in d.iterrows():
            xy = (r.x0 / NM.PX_POR_MM, r.y0 / NM.PX_POR_MM)
            w, h = r.ancho_px / NM.PX_POR_MM, r.alto_px / NM.PX_POR_MM
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec='white', lw=2.8, zorder=6))
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec='black', lw=1.1, zorder=7))
        ax.text(d.x0.min() / NM.PX_POR_MM, d.y0.min() / NM.PX_POR_MM - 1.5,
                f'{zona}: {d.Ra_medido.iloc[0]:.2f} ± {d.Ra_medido_sd.iloc[0]:.2f} | '
                f'{d.Ra_mapa.mean():.2f}',
                ha='left', va='bottom', fontsize=9.5, zorder=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))


def figura_20(img, mapa, vmin, vmax, args, media, zonas=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.4))
    ext_img = _extent_mm(img.shape)
    ext_mapa = _extent_mm((mapa.shape[0] * args.celda, mapa.shape[1] * args.celda))

    # El hueco de la barra de color se reserva en los DOS paneles y se apaga en
    # el (a). Sin esto, (b) cede ancho a la barra y su imagen encoge.
    cax_a = _reservar_cax(axes[0])
    cax_a.set_axis_off()
    cax_b = _reservar_cax(axes[1])

    axes[0].imshow(img, cmap='gray', aspect='equal', extent=ext_img,
                   origin='upper', interpolation='antialiased')
    axes[0].set_xlim(ext_img[0], ext_img[1])
    axes[0].set_ylim(ext_img[2], ext_img[3])
    axes[0].axis('off')
    _barra_escala(axes[0], ext_img[1])
    C.etiqueta_panel(axes[0], '(a)', dx=-0.02, dy=1.04)

    im = _pintar(axes[1], img, mapa, ext_img, ext_mapa, vmin, vmax,
                 args.cmap, args.alpha, suave=False)
    _colorbar(fig, im, cax_b, f'{args.param} [µm]')
    _caja_promedio(axes[1], f'Mean value: {media:.2f} µm')
    if zonas is not None:
        _marcar_zonas(axes[1], zonas)
    C.etiqueta_panel(axes[1], '(b)', dx=-0.02, dy=1.04)

    fig.tight_layout()
    _verificar_paneles_iguales(fig, axes[0], axes[1])
    return C.guardar(fig, 'Fig20_mapa_espacial' + ('' if args.param == 'Ra' else f'_{args.param}'))


def figura_21(img, mapa, vmin, vmax, args, media):
    fig, ax = plt.subplots(figsize=(7.6, 8.0))
    ext_img = _extent_mm(img.shape)
    ext_mapa = _extent_mm((mapa.shape[0] * args.celda, mapa.shape[1] * args.celda))

    cax = _reservar_cax(ax)
    suave = _suavizar(mapa, args.sigma)
    im = _pintar(ax, img, suave, ext_img, ext_mapa, vmin, vmax,
                 args.cmap, args.alpha, suave=True)

    # Isolineas sobre los centros de celda, en mm. Cinco niveles interiores: los
    # extremos dan contornos degenerados pegados al borde del rango.
    paso_mm = args.celda / NM.PX_POR_MM
    xs = (np.arange(suave.shape[1]) + 0.5) * paso_mm
    ys = (np.arange(suave.shape[0]) + 0.5) * paso_mm
    niveles = np.linspace(np.nanmin(suave), np.nanmax(suave), 7)[1:-1]
    cs = ax.contour(xs, ys, suave, levels=niveles, colors='white',
                    linewidths=0.8, zorder=4)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f', colors='white')

    _colorbar(fig, im, cax, f'{args.param} [µm]')
    _caja_promedio(ax, f'Mean value: {media:.2f} µm')

    fig.tight_layout()
    return C.guardar(fig, 'Fig21_mapa_suavizado' + ('' if args.param == 'Ra' else f'_{args.param}'))


# ------------------------------------------------------------------- main ---

def main():
    args = parsear()
    C.estilo()

    pan = Path(args.panoramica)
    mod = ruta_modelo(args)
    for etiqueta, r in [('panoramica', pan), ('modelo', mod)]:
        if not r.is_file():
            raise SystemExit(f"Falta la {etiqueta}: {r}")

    print(f"Arquitectura {C.ARQ_PUBLICADA[args.param]} · {args.param} · semilla {args.seed}")
    print(f"Panoramica : {pan}")
    print(f"Modelo     : {mod}")

    from tensorflow.keras.models import load_model
    modelo = load_model(str(mod), compile=False)
    print(f"Parametros : {modelo.count_params():,}")

    img = NM.cargar_panoramica(pan)
    h_mm, w_mm = img.shape[0] / NM.PX_POR_MM, img.shape[1] / NM.PX_POR_MM
    print(f"Imagen     : {img.shape[1]} x {img.shape[0]} px = {w_mm:.0f} x {h_mm:.0f} mm")

    mapa, detalle, preds = NM.mapa_dos_orientaciones(
        modelo, img, args.grano, fraccion_paso=args.fraccion_paso, celda=args.celda)

    print("\nPasadas:")
    for d in detalle:
        print(f"  {d['orientacion']:11s} tesela {d['tesela_alto']}x{d['tesela_ancho']}"
              f"  paso {d['paso_y']}x{d['paso_x']}  n={d['n_teselas']:5d}"
              f"  {d['pred_media']:.3f} ± {d['pred_sd']:.3f} µm"
              f"  [{d['pred_min']:.3f}, {d['pred_max']:.3f}]")

    # --- plausibilidad --------------------------------------------------------
    ok, msgs = NM.plausible(preds, RANGO_ENTRENAMIENTO[args.param])
    print("\nPlausibilidad:")
    for m in msgs:
        print(f"  {m}")

    media = float(np.nanmean(mapa))
    celdas = mapa[np.isfinite(mapa)]
    print(f"\nMapa       : {mapa.shape[0]} x {mapa.shape[1]} celdas de "
          f"{args.celda} px ({args.celda / NM.PX_POR_MM:.1f} mm)")
    print(f"  promedio de la superficie   : {media:.3f} µm")
    print(f"  rango del mapa (promediado) : [{celdas.min():.3f}, {celdas.max():.3f}]")
    print(f"  rango de tesela individual  : [{preds.min():.3f}, {preds.max():.3f}]"
          "   <- el solape estrecha el rango, declararlo en el pie")

    # --- limites de color -----------------------------------------------------
    # Rango completo redondeado a medio micrometro: sin saturacion, para que la
    # barra de color permita leer el valor absoluto de cualquier punto del mapa,
    # que es lo que el pie de figura promete.
    p1, p99 = np.nanpercentile(mapa, [1, 99])
    vmin = args.vmin if args.vmin is not None else float(np.floor(celdas.min() * 2) / 2)
    vmax = args.vmax if args.vmax is not None else float(np.ceil(celdas.max() * 2) / 2)
    print(f"\nLimites de color : {vmin:.2f} - {vmax:.2f} µm"
          f"  (rango completo; p1={p1:.2f}, p99={p99:.2f})")
    print("  la publicada usaba 3.5 con vmax dinamico; para replicarla: --vmin 3.5 --vmax 8.5")

    # --- figuras --------------------------------------------------------------
    print("\nFiguras:")
    zonas = _leer_zonas(args)
    if zonas is not None:
        print(f"  zonas del SJ-310 marcadas en la Fig. 20b: {', '.join(zonas.recorte)}")
    figura_20(img, mapa, vmin, vmax, args, media, zonas)
    figura_21(img, mapa, vmin, vmax, args, media)

    # --- cifras ---------------------------------------------------------------
    # Sin sufijo para Ra (los nombres ya citados), con sufijo para Rz: antes del
    # 2026-09-10 generar el mapa de Rz sobrescribia en silencio el de Ra.
    xlsx = C.SALIDAS / ('Fig20_21_mapa_espacial_cifras'
                        + ('' if args.param == 'Ra' else f'_{args.param}') + '.xlsx')
    resumen = pd.DataFrame([
        ('arquitectura', C.ARQ_PUBLICADA[args.param]), ('parametro', args.param),
        ('semilla', args.seed), ('modelo', str(mod)), ('panoramica', str(pan)),
        ('grano', args.grano), ('probeta', 'P11 (test, no vista en entrenamiento)'),
        ('px_por_mm', NM.PX_POR_MM),
        ('superficie_mm', f'{w_mm:.1f} x {h_mm:.1f}'),
        ('celda_px', args.celda), ('celda_mm', round(args.celda / NM.PX_POR_MM, 2)),
        ('mapa_celdas', f'{mapa.shape[0]} x {mapa.shape[1]}'),
        ('n_estimaciones', int(preds.size)),
        ('media_superficie_um', round(media, 4)),
        ('mapa_min_um', round(float(celdas.min()), 4)),
        ('mapa_max_um', round(float(celdas.max()), 4)),
        ('tesela_min_um', round(float(preds.min()), 4)),
        ('tesela_max_um', round(float(preds.max()), 4)),
        ('vmin', vmin), ('vmax', vmax),
        ('sigma_suavizado_fig21_celdas', args.sigma),
        ('sigma_suavizado_fig21_mm', round(args.sigma * args.celda / NM.PX_POR_MM, 2)),
        ('plausibilidad', '; '.join(msgs)),
    ], columns=['clave', 'valor'])

    with pd.ExcelWriter(xlsx) as w:
        resumen.to_excel(w, sheet_name='Resumen', index=False)
        pd.DataFrame(detalle).to_excel(w, sheet_name='Pasadas', index=False)
        pd.DataFrame(np.round(mapa, 4)).to_excel(w, sheet_name='Mapa', index=True)
    print(f"  -> {xlsx}")

    if not ok:
        raise SystemExit("\nLa plausibilidad fallo: revisar antes de usar las figuras.")


if __name__ == '__main__':
    main()
