# Reproducing the results

Every number, figure and table in the paper can be regenerated from this repository. Nothing
requires access to the robotic cell or to the profilometer.

```bash
conda env create -f environment.yml
conda activate roughness-vision
```

All commands below are run from the directory of the script, which is how the scripts resolve
their relative outputs. Figures are written to `src/6_figuras_articulo/salidas/`.

## Three levels of effort

| Level | What it does | Cost |
|---|---|---|
| **1. Regenerate figures and tables** | reads the committed metrics in `results/` | seconds to minutes |
| **2. Re-evaluate the published weights** | runs inference with `models/` on the test specimen | about 1 minute per parameter |
| **3. Retrain from scratch** | reruns training | 4.5–5 min per fold on CPU; the full sweep is ~40 h |

Level 1 and 2 need nothing but this repository. Level 3 reproduces the contents of `results/`.

## Level 2 — check the headline numbers first

This is the fastest meaningful check that the repository is intact and the environment correct.

```bash
cd src/4_validacion
python validar_headless.py --param Ra    # expect R2 0.8654, MAE 0.4985 um
python validar_headless.py --param Rz    # expect R2 0.7662, MAE 3.6805 um
```

Those are the seed-13 models. The values published in the paper are means over seeds 10–13:
Ra MAE 0.495 ± 0.061 µm and Rz MAE 3.898 ± 0.326 µm.

## Level 1 — figures and tables

**A note on file names.** The scripts write internal file names that do not always match the
figure numbers used in the paper. `figures/` holds the files under the numbering used in the paper.
The mapping is below; the left column is what the paper calls it, the right is the file the script
writes into `salidas/`.

### Body

| Paper | Command (from `src/6_figuras_articulo/`) | Output in `salidas/` |
|---|---|---|
| Fig. 12 | `python fig12_arquitectura.py` | `Fig12_arquitectura` |
| Fig. 13 | `python fig15_16_curvas_vc.py --param Ra` | `Fig15_curvas_Ra` |
| Fig. 14 | `python fig15_16_curvas_vc.py --param Rz` | `Fig16_curvas_Rz` |
| Fig. 15 | `python fig17_18_medido_vs_estimado.py --param Ra` | `Fig17_medido_vs_estimado_Ra` |
| Fig. 16 | `python fig17_18_medido_vs_estimado.py --param Rz` | `Fig18_medido_vs_estimado_Rz` |
| Fig. 17 | `python fig19_r2.py` | `Fig19_r2_Ra_Rz` |
| Fig. 18 | `python fig_interpretabilidad.py` | `Fig18_interpretabilidad` |
| Fig. 19 | `python fig20_21_mapa_espacial.py` | `Fig20_mapa_espacial` |
| Fig. 20 | `python fig20_21_mapa_espacial.py` | `Fig21_mapa_suavizado` |
| Table 5 | `python tabla5_arquitecturas.py --param Ra` (and `--param Rz`) | `Tabla5_arquitecturas` |
| Table 6 | `python tabla6_validacion_cruzada.py` | `Tabla6_validacion_cruzada` |
| Table 7 | `python tabla7_metricas_globales.py` | `Tabla7_metricas_globales` |

### Appendix A

| Paper | Command | Output in `salidas/` |
|---|---|---|
| Fig. A1 | `python fig13_comparacion_arquitecturas.py --param Ra` | `Fig13_comparacion_arquitecturas_repeticiones` |
| Fig. A2 | `python fig13_comparacion_arquitecturas.py --param Rz` | `..._repeticiones_Rz` |
| Fig. A3 | `python fig14_r2_arquitecturas.py --param Ra` | `Fig14_r2_arquitecturas_vc` |
| Fig. A4 | `python fig14_r2_arquitecturas.py --param Rz` | `Fig14_r2_arquitecturas_vc_Rz` |
| Fig. A5, A6 | `python fig_interpretabilidad.py` | `FigA5_textura_tono_Ra`, `FigA6_textura_tono_Rz` |
| Fig. A7, A8 | `python ../4_validacion/interpretabilidad_oclusion.py` | `oclusion_ejemplos_Ra`, `..._Rz` |
| Fig. A9, A10 | `python fig20_21_mapa_espacial.py --param Rz` | `Fig20_mapa_espacial_Rz`, `Fig21_mapa_suavizado_Rz` |

Each figure script also writes a `*_cifras.xlsx` next to its output, holding the exact values
plotted. Those spreadsheets are the ones to compare against if a regenerated figure looks
different.

## Level 3 — retraining

Final models, one per parameter and seed:

```bash
cd src/3_entrenamiento
python entrenar_final_limpio.py --arch B --param Ra --seed 13
python entrenar_final_limpio.py --arch C --param Rz --seed 13
```

Architecture sweep — the four variants under leave-one-specimen-out cross-validation. This is
what fills `results/architecture_sweep/`:

```bash
python barrido_arquitecturas.py --arch B --param Ra --seed 13 --folds todas
python barrido_arquitecturas.py --arch A --param Ra --seed 13 --folds 1   # one fold, quick
```

`--arch` is `A` (GlobalAveragePooling2D, 25 681 parameters), `B` (+1 MaxPooling2D, 709 713),
`C` (+2 MaxPooling2D, 2 970 705) or `D` (Flatten, 12 211 281). `--dataset outliers` trains on
the 1056 unfiltered patches instead of the 1030 clean ones.

The published results average seeds 10, 11, 12 and 13 for each of the 4 variants, 2 parameters
and 6 folds. Running the whole thing is roughly 40 hours on CPU, which is why the computed
metrics are committed.

Supporting analyses:

```bash
python montecarlo_incertidumbre.py           # bootstrap ensemble, uncertainty bands
python ablacion_sin_grano.py                 # ablation: drop the grit input
python diagnostico_determinismo.py           # verifies what is and is not reproducible
cd ../4_validacion
python ensemble_incertidumbre.py             # fold ensemble on the test specimen
python interpretabilidad_oclusion.py         # occlusion study, texture vs tone
python descomposicion_brecha.py              # decomposes the cross-validation / test gap
```

## Determinism, and what it does not cover

Seeds are fixed with `tf.keras.utils.set_random_seed()`, which does fix weight initialisation in
Keras 3. The earlier combination of `np.random.seed()` and `tf.random.set_seed()` does **not**,
and runs with a nominally fixed seed diverged because of it; `diagnostico_determinismo.py`
isolates exactly which stages are reproducible.

Even with seeds fixed, results are only bitwise reproducible on the same platform and library
versions. TensorFlow's oneDNN kernels reorder floating-point operations, so a different CPU, a
GPU build or another TensorFlow version will shift the last digits. Differences of a few
thousandths in R² are expected and are far smaller than the spread across seeds, which is the
quantity the paper reports.

Reference platform: Windows 11, Python 3.11.15, TensorFlow 2.21.0, Keras 3.14.1, CPU only. Each
run records its own environment in a `config_corrida*.json` beside its outputs.

## Data checks

```bash
cd src/2_dataset
python verificar_dataset_validacion.py   # every referenced image is present
python criterio_outliers_iqr.py          # reproduces the 26 exclusions
python analisis_normalidad.py            # distribution statistics per grit
```

Note that `criterio_outliers_iqr.py` writes back into `DATASET.xlsx`, adding its own sheets and
preserving the rest. Copy the file first if you want the original byte-for-byte.
