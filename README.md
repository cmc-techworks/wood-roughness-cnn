# Robotic vision-based inspection for surface roughness estimation in wood

Code, data and trained models for the paper:

> **A robotic vision-based inspection system for spatially distributed surface roughness
> estimation in high-value wood products**
> Cristóbal Mora-Concha, Fabián Iglesias, Arturo Padilla, Walter Gómez, Eduardo Diez
> Universidad de La Frontera, Temuco, Chile — *Journal of Intelligent Manufacturing* (under review)

A multimodal convolutional neural network estimates the roughness parameters Ra and Rz from
grazing-light images of sanded Chilean oak (*Nothofagus obliqua*), combining a 390 × 130 px
grayscale patch (15 × 5 mm of surface) with a one-hot encoding of the abrasive grit. Images are
acquired by a UR5e inspection robot in a dual-robot sanding cell and referenced against contact
profilometry (Mitutoyo SJ-310, ISO 21920-2).

This repository is self-contained: it holds the full image dataset, the reference measurements,
the training and evaluation code, the published model weights, and the scripts that regenerate
every figure and table in the paper.

## Headline results

Mean ± standard deviation over 4 weight initialisations (seeds 10–13). The two roughness
parameters use different network variants, selected independently.

| | Variant | Parameters | Cross-validation (6 folds) | Independent test (specimen 11, n = 190) |
|---|---|---|---|---|
| **Ra** | B | 709 713 | MAE 0.551 ± 0.029 µm · R² 0.806 ± 0.021 | MAE 0.495 ± 0.061 µm · R² 0.864 ± 0.019 |
| **Rz** | C | 2 970 705 | MAE 4.164 ± 0.349 µm · R² 0.635 ± 0.087 | MAE 3.898 ± 0.326 µm · R² 0.762 ± 0.012 |

Cross-validation is leave-one-specimen-out over the six training specimens; the test specimen is
never seen during training or model selection. MAE is the primary metric and R² the secondary one:
the network optimises a squared-error loss with MAE as its reported metric, and R² is sensitive to
the roughness spread of whichever specimen is held out.

## Repository layout

```
data/
  fotos_recortes/          1056 training patches + Validacion/ with 192 test patches (TIFF)
    DATASET.xlsx           reference measurements, original working format
  processed/               the same measurements as UTF-8 CSV (open format)
  splits/                  the six cross-validation folds, materialised
  README.md                data dictionary and provenance
src/
  0_adquisicion/           TCP acquisition server that drives the camera from the robot
  1_metrologia/            profilometer traces to Ra/Rz, ISO 16610-21 / 21920-2
  2_dataset/               outlier criterion, dataset statistics, process variability
  3_entrenamiento/         training: published model, architecture sweep, bootstrap ensemble
  4_validacion/            evaluation on the test specimen, uncertainty, interpretability
  5_heatmap/               spatial roughness map from a stitched panorama
  6_figuras_articulo/      one script per figure and table of the paper
models/
  Ra_arch_B/               4 final Ra models (one per seed)
  Rz_arch_C/               4 final Rz models (one per seed)
results/                   metrics, per-fold predictions and run logs
figures/body/              Fig. 12–20 as published (PNG 300 dpi + vector PDF)
figures/appendix/          Fig. A1–A10
```

## Quickstart

```bash
git clone https://github.com/cmc-techworks/wood-roughness-cnn.git
cd wood-roughness-cnn
conda env create -f environment.yml
conda activate roughness-vision

# check that every referenced image is on disk
python src/2_dataset/verificar_dataset_validacion.py

# evaluate a published model on the independent test specimen (inference only, ~1 min)
python src/4_validacion/validar_headless.py --param Ra
```

[REPRODUCING.md](REPRODUCING.md) maps every figure and table of the paper to the exact command
that regenerates it, with expected runtimes.

## Data

1248 image patches from 7 oak specimens, each paired with a contact-profilometry measurement of
the same 15 × 5 mm area. 1056 patches from 6 specimens form the training set (1030 after outlier
removal) and 192 patches from the seventh specimen form the independent test set (190 with a valid
reference measurement). Full description, including the sampling protocol and the outlier
criterion, is in [data/README.md](data/README.md).

## Reproducibility

Randomness is fixed with `tf.keras.utils.set_random_seed()`. Note that `np.random.seed()` plus
`tf.random.set_seed()` does **not** fix weight initialisation in Keras 3 — an earlier version of
this code used them and produced varying results with a nominally fixed seed. All reported figures
are therefore means over four seeds rather than single runs, and the spread across seeds is
reported alongside every number.

Training runs on CPU. TensorFlow ≥ 2.11 has no native GPU support on Windows, so the reference
environment is CPU-only; one cross-validation fold takes roughly 4.5–5 minutes. Exact library
versions, platform string and the command line of each run are recorded in the
`config_corrida*.json` files under `results/`.

## What is not included

Large intermediate artefacts are left out because they are regenerable from the code in this
repository:

- **Model checkpoints of the architecture sweep** (~4 GB). Regenerate with
  `src/3_entrenamiento/barrido_arquitecturas.py`. Their computed metrics, per-epoch histories and
  per-fold predictions *are* included, under `results/architecture_sweep/`, so every architecture
  figure regenerates without retraining.
- **Raw panorama captures of specimen 11** (~2.3 GB). The spatial map in Fig. 19–20 is built from
  the already-stitched panorama, which is included. The raw 12-pose captures are available from
  the corresponding author on reasonable request.

## License

Code in `src/` is released under the MIT License ([LICENSE](LICENSE)). Data in `data/`, the figures
in `figures/` and the model weights in `models/` are released under Creative Commons Attribution
4.0 International ([LICENSE-DATA](LICENSE-DATA)). If you use the images or the measurements, please
cite the paper.

## How to cite

See [CITATION.cff](CITATION.cff), or use the "Cite this repository" button on GitHub. Please cite
both the paper and the archived release of this repository.

## Funding

ANID Chile, FONDEF IT21i0069, and project DIE21-0010 of Universidad de La Frontera.

## Contact

Eduardo Diez — eduardo.diez@ufrontera.cl
