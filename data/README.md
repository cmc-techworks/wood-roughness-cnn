# Dataset

Image patches of robotically sanded Chilean oak (*Nothofagus obliqua*), each paired with a
contact-profilometry measurement of the same surface area.

## Contents

| Path | What it is |
|---|---|
| `fotos_recortes/*.tiff` | 1056 training patches, 6 specimens (5–10) |
| `fotos_recortes/Validacion/*.tiff` | 192 test patches, specimen 11 only |
| `fotos_recortes/DATASET.xlsx` | reference measurements, original working format |
| `fotos_recortes/P11/C1/G40/` | stitched panorama and zone crops used by the spatial map |
| `processed/train_measurements.csv` | the same measurements as UTF-8 CSV, with outlier flags and fold assignment |
| `processed/test_measurements.csv` | test measurements as CSV |
| `splits/cv_folds.csv`, `splits/cv_folds.json` | the six cross-validation folds |

`processed/` is generated from `DATASET.xlsx` and is the recommended entry point: it is an open,
diffable format and it carries the outlier flags and fold assignment that the spreadsheet keeps
in separate sheets. The training code reads the spreadsheet, not the CSV.

## Column dictionary

`train_measurements.csv`

| Column | Type | Description |
|---|---|---|
| `image` | string | file name, relative to `fotos_recortes/` |
| `specimen_id` | int | specimen, 5–10 for training |
| `grit` | int | abrasive grit of the last sanding pass: 40, 80, 120 or 180 |
| `Ra_um` | float | arithmetic mean deviation of the profile, µm |
| `Rz_um` | float | maximum height of the profile, µm |
| `is_outlier` | bool | excluded from the published training set by the 1.5 × IQR criterion |
| `outlier_Ra` | bool | flagged as an outlier on Ra |
| `outlier_Rz` | bool | flagged as an outlier on Rz |
| `cv_fold` | int | 1–6, the fold in which this row is the validation set; `0` for excluded rows |

`test_measurements.csv` has the first five columns only. Specimen 11 is never used for training
or model selection.

## Counts

| Set | Images | With measurement | After outlier removal |
|---|---|---|---|
| Training (specimens 5–10) | 1056 | 1056 | **1030** |
| Test (specimen 11) | 192 | **190** | 190 |

Per grit after outlier removal: P40 263 · P80 257 · P120 249 · P180 261.
Ranges of the clean training set: Ra 1.750–9.834 µm (mean 4.271) · Rz 14.511–62.780 µm
(mean 30.056).

Two test images have no reference measurement — `P11_C2_G80_Recorte_Recorte_C_3.5.tiff` and
`...C_3.6.tiff`, both P80 — and are skipped at evaluation time. This is why the test set is
described as "192 defined, 190 valid".

## File naming

```
P11_C1_G120_Recorte_Recorte_A1_1.2.tiff
 │   │    │                  │   └── zone index . trace number within the zone (1–6)
 │   │    │                  └────── measurement zone: A (subdivided A1/A2), B, C
 │   │    └───────────────────────── abrasive grit of the last pass
 │   └────────────────────────────── face of the specimen: C1 or C2
 └────────────────────────────────── specimen id
```

## Sampling protocol

Three measurement zones per face and specimen. Each zone covers 15 × 15 mm and receives **six
SJ-310 traces**: traces 1–3 run horizontally and 4–6 vertically. Each trace yields one image
patch of 5 × 15 mm, with the 12.5 mm evaluation length centred on it. Horizontal and vertical
patches overlap, and the study does not distinguish them.

## Caveat: repeated labels in zone A

Zone A is cropped twice, as `A1` and `A2`. **Both crops are different images of the same measured
trace and therefore carry an identical Ra and Rz value.** Concretely:

| Set | Images | Independent traces | Images sharing a trace |
|---|---|---|---|
| Training, clean | 1030 | **843** | 187 |
| Test | 190 | **142** | 48 |

Anyone computing significance, effective sample size or independence assumptions should use the
trace count, not the image count. The published models were trained on all images; the splits are
grouped by specimen, so a shared trace never straddles the train/validation boundary.

## Image format

TIFF, 130 px wide × 390 px tall, 8-bit. Physically 5 mm × 15 mm, so 26 px/mm on both axes. The
network input is `(390, 130, 1)` — height first — and pixel values are scaled to [0, 1] by
dividing by 255.

12 of the 1056 training files are stored as RGBA rather than 8-bit grayscale. The loader opens
every file with `color_mode='grayscale'`, so they are converted on read and no special handling
is needed; the count is noted here only so the discrepancy is not mistaken for corruption.

## Outlier criterion

Points beyond 1.5 × IQR are removed, with two details that matter and are easy to get wrong:

1. The IQR is computed **within each grit group**, never over the pooled set. Grits differ by a
   factor of two in mean roughness, so a pooled IQR would flag whole grit levels.
2. The criterion is the **union of Ra and Rz**: a point is removed if it is an outlier on either
   parameter.

This removes 26 of 1056 training points (2.46 %) and 2 of 192 test points. `outlier_Ra` and
`outlier_Rz` let you reconstruct either criterion separately. Reproduce with
`python src/2_dataset/criterio_outliers_iqr.py`.

Excluding those 26 points is not cosmetic: retraining with them included degrades every one of
the six specimens (mean ΔR² = −0.066). Before removal, the maximum Rz of P80 (76.618 µm) exceeded
that of P40 (72.980 µm), which is physically inconsistent because P80 is the finer abrasive. The
excluded points contradict the known ordering of the process rather than merely being difficult.

## Reference measurement

Mitutoyo SJ-310 contact profilometer, ISO 21920-2. Sampling length λc = 2.5 mm, evaluation length
12.5 mm, gaussian S-filter with an 8 µm cutoff, form removed with a 6th-degree least-squares
polynomial. λc drops to 0.8 mm when the preliminary Ra falls in 0.1 < Ra ≤ 2 µm. About one minute
per measurement.

## Acquisition

Monochrome Genie Nano M4020 camera with a V2528-MPY lens at roughly 300 mm working distance, and
two diffuse LED modules in a bilateral grazing configuration 250 mm apart. Images are captured by
a UR5e inspection robot in a 3 × 4 = 12-pose grid over each face; see
`src/0_adquisicion/Server_Vision_V2.py` for the capture protocol and the file naming it produces.

## Specimens and sanding

Seven end-grain specimens of 307 × 277 mm, glued blocks CNC-milled. Sanding with 3M Cubitron II
775L: 6 passes at P40, 4 at P80, 2 at P120 and 2 at P180, 14 per face. Constant parameters:
20 N normal force, 0.05 m/s feed, 1 m/s² acceleration, 1.5 mm blend radius, 2000 rpm.

## License

CC BY 4.0 — see [../LICENSE-DATA](../LICENSE-DATA).
