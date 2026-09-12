| Variant | Reduction stage | Parameters | Repetitions | MAE [µm] | RMSE [µm] | R² | Train MAE [µm] | Val / Train | Final-model train MAE [µm] |
|---|---|---|---|---|---|---|---|---|---|
| A | GlobalAveragePooling2D | 25 681 | 4 | 4.168 ± 0.096 | 5.476 ± 0.105 | 0.636 ± 0.019 | 3.806 | 1.09 | — |
| B | 2 × MaxPooling2D + Flatten | 709 713 | 4 | 4.051 ± 0.188 | 5.298 ± 0.191 | 0.662 ± 0.033 | 3.806 | 1.06 | — |
| C | 1 × MaxPooling2D + Flatten | 2 970 705 | 4 | 4.164 ± 0.349 | 5.342 ± 0.416 | 0.635 ± 0.087 | 3.272 | 1.27 | 2.875 |
| D | Flatten | 12 211 281 | 4 | 4.322 ± 0.113 | 5.755 ± 0.167 | 0.610 ± 0.023 | 1.286 | 3.36 | — |
*MAE [µm], RMSE [µm], R² are 6-fold cross-validation metrics; Train MAE [µm], Val / Train, Final-model train MAE [µm] refer to epoch 40 of training.*