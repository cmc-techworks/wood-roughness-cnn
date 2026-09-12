# What this repository leaves out, and why

The working tree of the study is about 15 GB. This repository is 0.36 GB and reproduces every
published result. What was left out, and how to get it if you need it:

## Regenerable by running the code

| Left out | Size | How to regenerate |
|---|---|---|
| Model checkpoints of the architecture sweep | ~4.1 GB | `barrido_arquitecturas.py --guardar-modelos` |
| Per-fold models of the final runs | ~0.5 GB | `entrenar_final_limpio.py` |
| Bootstrap ensemble weights (20 replicas per parameter) | ~0.2 GB | `montecarlo_incertidumbre.py` |

The **metrics, per-epoch histories and per-fold predictions** of all 42 sweep runs *are*
committed, under `results/architecture_sweep/`, at 25 MB. Only the weights were dropped. Every
architecture figure and table therefore regenerates without retraining anything.

## Available from the authors on request

| Left out | Size | Note |
|---|---|---|
| Raw 12-pose captures of specimen 11 | 2.3 GB | the already-stitched panorama needed for the spatial map *is* included |
| `Mosaico_Superficial.tif` and the 18 raw SJ-310 spreadsheets of zone G40 | ~100 MB | the derived measurements are in `DATASET.xlsx` |
| Raw captures of the other specimens | several GB | the 15 × 5 mm patches derived from them are included |

## Deliberately not published

**The ambient-light robustness campaign.** Roughly 6.5 GB of captures of one additional specimen
under three ambient-light conditions. Two problems in that campaign are unresolved: the specimen
falls below the Ra range of the training set, and the tile grid is not registered between
conditions, with shifts of up to 150 px. Its numbers can only be read as relative degradation,
not absolute performance, so no published claim rests on them and they are not distributed as if
they did.

**Data-augmentation variants.** Three augmented image sets exist (`Imagenes_AD`, `Imagenes_AD2`,
`Imagenes_AD400`). They were built on an earlier cleaning criterion — Ra only, 1037 rows — rather
than the published Ra ∪ Rz criterion of 1030 rows, so they are not directly comparable to the
published training set. The final model uses no augmentation.

**Superseded scripts.** A `historico/` directory of earlier versions, kept during development and
excluded here. It includes `analisis_rz.py`, which destroys the spreadsheet it is pointed at by
writing an empty workbook over its own input, and an early validation script that transposes the
image tile. Publishing broken code next to working code invites someone to run the wrong one.

**Manuscript tooling.** Scripts for editing and checking the manuscript `.docx`. They automate the
writing process, not the science, and depend on files that are not part of this repository.

**The revision record.** Reviewer correspondence, response letters, meeting transcripts, planning
documents and manuscript drafts are private working material and are excluded by `.gitignore`.

## One script kept deliberately unmodified

`src/3_entrenamiento/Red_multimodal_VC.py` produced the originally submitted model. It only knows
the original directory layout and it contains the double-training defect described in
[known_limitations.md](known_limitations.md). It is kept as the record of what was executed, not
as code to run. Use `barrido_arquitecturas.py` and `entrenar_final_limpio.py` instead.
