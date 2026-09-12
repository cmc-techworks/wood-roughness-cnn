# Model selection

## The four variants

All four share an identical convolutional base — two 3×3 convolution blocks, 32 and 64 filters.
They differ only in how the feature map is reduced before the first dense layer, which is where
essentially all the parameters live.

| Variant | Reduction stage | Parameters |
|---|---|---|
| A | `GlobalAveragePooling2D` | 25 681 |
| B | 2 × `MaxPooling2D` + `Flatten` | 709 713 |
| C | 1 × `MaxPooling2D` + `Flatten` | 2 970 705 |
| D | `Flatten` | 12 211 281 |

In variant D, **99.8 % of the parameters sit in a single layer**, the `Flatten` → `Dense(64)`
transition: 190 464 flattened features × 64 units = 12 189 760 weights. The convolutions together
account for fewer than 19 000. The parameter count therefore measures how little the feature map
was reduced, not depth or representational capacity.

## Cross-validation, 6 folds, mean ± SD over 4 seeds

Ra:

| Variant | MAE [µm] | RMSE [µm] | R² | Train MAE [µm] | Val / Train |
|---|---|---|---|---|---|
| A | 0.595 ± 0.012 | 0.811 ± 0.018 | 0.762 ± 0.012 | 0.546 | 1.09 |
| **B** | **0.551 ± 0.029** | **0.724 ± 0.032** | **0.806 ± 0.021** | 0.416 | 1.33 |
| C | 0.587 ± 0.024 | 0.793 ± 0.032 | 0.773 ± 0.022 | 0.199 | 2.94 |
| D | 0.633 ± 0.020 | 0.869 ± 0.022 | 0.728 ± 0.013 | 0.083 | 7.60 |

Rz:

| Variant | MAE [µm] | RMSE [µm] | R² | Train MAE [µm] | Val / Train |
|---|---|---|---|---|---|
| A | 4.168 ± 0.096 | 5.476 ± 0.105 | 0.636 ± 0.019 | 3.806 | 1.09 |
| B | 4.051 ± 0.188 | 5.298 ± 0.191 | 0.662 ± 0.033 | 3.806 | 1.06 |
| **C** | 4.164 ± 0.349 | 5.342 ± 0.416 | 0.635 ± 0.087 | 3.272 | 1.27 |
| D | 4.322 ± 0.113 | 5.755 ± 0.167 | 0.610 ± 0.023 | 1.286 | 3.36 |

The pattern that matters is the last two columns. Validation error is essentially flat across a
476× range in parameter count, while training error collapses monotonically. Extra capacity does
not buy generalisation; it buys fit to the noise of the reference measurement.

## Why B for Ra and C for Rz

**Ra → B.** B wins on every aggregate criterion: lowest MAE, lowest RMSE, highest R², and the
regression of estimated against measured has the slope closest to unity, meaning it collapses
least towards the per-grit mean.

**Rz → C.** This one is a trade-off and is worth stating plainly. On aggregate metrics **B is
better than C on Rz**: MAE 4.051 vs 4.164 µm, R² 0.662 vs 0.635, and it is markedly more stable
across seeds (SD 0.033 vs 0.087). C was selected because it has the better regression slope
(0.696 vs 0.680) and wins on 3 of the 6 specimens taken individually — that is, it compresses the
rough tail less, which matters for a measurement instrument.

Both choices are fixed in code, in `src/6_figuras_articulo/comun.py`:

```python
ARQ_PUBLICADA = {'Ra': 'B', 'Rz': 'C'}
```

The high standard deviation of C on Rz comes almost entirely from one cell: specimen 5 with
seed 12, where R² is −0.385. Specimen 5 is hard for B as well, but C amplifies the failure. See
[known_limitations.md](known_limitations.md).

## Why the grit input is not doing the work

A lookup table of mean Ra per grit, which never sees the image at all, already reaches R² = 0.783
in cross-validation. That invites the objection that the network is an expensive lookup table.
The ablation answers it: removing the grit branch drops variant A from R² 0.761 to 0.426. The
grit input does not replace the image — it is what lets the network exploit it, by removing the
between-grit offset so the convolutions can model within-grit variation.

Reproduce with `python src/3_entrenamiento/ablacion_sin_grano.py`.
