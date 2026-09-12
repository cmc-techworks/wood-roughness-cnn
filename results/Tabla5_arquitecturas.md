| Variant | Reduction stage | Parameters | Repetitions | MAE [µm] | RMSE [µm] | R² | Train MAE [µm] | Val / Train | Final-model train MAE [µm] |
|---|---|---|---|---|---|---|---|---|---|
| A | GlobalAveragePooling2D | 25 681 | 4 | 0.595 ± 0.012 | 0.811 ± 0.018 | 0.762 ± 0.012 | 0.546 | 1.09 | — |
| B | 2 × MaxPooling2D + Flatten | 709 713 | 4 | 0.551 ± 0.029 | 0.724 ± 0.032 | 0.806 ± 0.021 | 0.416 | 1.33 | 0.387 |
| C | 1 × MaxPooling2D + Flatten | 2 970 705 | 4 | 0.587 ± 0.024 | 0.793 ± 0.032 | 0.773 ± 0.022 | 0.199 | 2.94 | — |
| D | Flatten | 12 211 281 | 4 | 0.633 ± 0.020 | 0.869 ± 0.022 | 0.728 ± 0.013 | 0.083 | 7.60 | 0.0055 |
*MAE [µm], RMSE [µm], R² are 6-fold cross-validation metrics; Train MAE [µm], Val / Train, Final-model train MAE [µm] refer to epoch 40 of training.*