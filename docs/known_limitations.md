# Known limitations

Collected here so that anyone reusing this work knows where it is weak before they find out the
hard way. Each item names the models it was measured on, because several analyses predate the
final choice of published variant.

## 1. Seven specimens is few, and between-specimen variance dominates

Every uncertainty in this study is bounded by the fact that there are seven specimens: six in
cross-validation and one held out. The standard deviation of R² across specimens is roughly three
to five times the standard deviation across model seeds. With this design, no analysis can
separate "this model is better" from "this model got an easier specimen" beyond that margin.

The practical consequence: **the cross-validation mean with its spread is the honest summary, and
the single-specimen test result is a confirmation, not the headline.**

## 2. The cross-validation / test gap is mostly about which specimen you evaluate

The test R² exceeds the cross-validation mean. Decomposing it (variant A, Ra, in
`results/A_oficial/Ra/brecha_cv_test/`) attributes about **99 % of the gap to the identity of the
evaluated specimen and about 1 % to the larger training set** of the final model; the standard
deviation across specimens is 4.85× that across models.

An earlier decomposition on the six published fold models of variant D put it at roughly 80 % /
20 %. Both analyses agree on the direction and on the conclusion: cross-validation R² does not
predict test R². Fold 1 was the worst in cross-validation and also the worst on the test set, but
fold 6 was nearly the worst in cross-validation and the **best** on the test set.

## 3. Variant C on Rz is unstable, and one cell drives the spread

The standard deviation of C on Rz (R² 0.635 ± 0.087) comes almost entirely from **specimen 5 with
seed 12, where R² = −0.385**. Specimen 5 is difficult for variant B too, but C amplifies the
failure. Variant B is better on every aggregate Rz metric; C was chosen on regression slope. See
[model_selection.md](model_selection.md).

## 4. The Rz comparison against prior work has little margin

The published Rz RMSE of 5.152 ± 0.125 µm improves on the 5.31 µm reported by Iglesias et al.
(2024), but **on the mean, and not by much**. One of the four seeds gives 5.317 µm, which is above
that reference. Any claim of improvement should carry the spread, not just the mean.

## 5. Grad-CAM is not a valid probe for the published models

Grad-CAM averages each channel gradient over space, which is exact after a
`GlobalAveragePooling2D` (variant A) and **not faithful after a `Flatten`** (variants B, C, D —
including both published models). Its correlation with a direct occlusion test is 0.59 for A-Ra
and 0.56 for A-Rz, but between −0.04 and +0.01 for the other six combinations.

Use `src/4_validacion/interpretabilidad_oclusion.py` instead, which erases either the texture or
the tone of a 1 mm window and measures how much the estimate moves. On that test, texture
dominates tone in all 8 combinations, in 40 of 40 images, p = 1.8e-12, with a texture/tone ratio
of 3.8× for B-Ra and 3.9× for C-Rz. The physical reading — the network looks at the sanding marks
rather than the grain — holds; the method that establishes it is occlusion, not Grad-CAM.

## 6. Repeated labels in zone A

Zone A is cropped twice. Both crops carry the same measured Ra and Rz, so the 1030 clean training
images correspond to **843 independent traces**, and the 190 test images to 142. Use the trace
count for anything that assumes independent samples. Details in
[../data/README.md](../data/README.md).

## 7. Errors are not uniform across grit

Measured on the fold ensemble of the originally published variant D: P40 carries a systematic
bias of −0.586 µm, the model underestimating the rough tail by half a micron; P180 is the worst
calibrated, with the largest ratio of error to predicted uncertainty. Both extremes of grit fail,
for different reasons. The regression slope below unity in every variant is the same phenomenon:
compression towards the per-grit mean.

## 8. The uncertainty band is not calibrated out of the box

The spread across the six fold models (variant D) covers only 27.9 % of points at ±1 SD against a
nominal 68 %, and 50.5 % at ±2 SD against 95 % — roughly a factor of three too narrow. The reason
is conceptual rather than a defect: disagreement between models captures epistemic uncertainty
only, not the aleatoric part contributed by profilometer noise and by genuine surface variation
within the patch. A conformal rescaling reaches 91.1 % coverage for both parameters. Report the
raw ensemble spread as a relative indicator of where the model hesitates, not as a confidence
interval.

## 9. The spatial map is anchored on three zones of one face

The map is validated against SJ-310 measurements in three zones. Over four seeds the zone-by-zone
MAE is 0.470 ± 0.071 µm, and the map **overestimates the mean of the face by +0.24 ± 0.26 µm**.
All four seeds fall within one standard deviation of the profilometer. Three zones on one face of
one specimen is a weak anchor; it establishes that the map is not drifting, not that it is
accurate everywhere.

The Rz map (Fig. A9, A10) has no anchoring of its own, because the reference measurements taken
for the map are of Ra.

## 10. The camera trigger is open loop

The robot sends a capture command and waits a fixed time; it never reads the acknowledgement from
the server. If a capture took longer than the one-second guard — a camera stall, a hiccup on the
wireless link — the robot would move during exposure and nothing would flag it. Acquisition is
estimated at about 0.4 s against a 1 s guard, a margin of 2.5×, and a failure is detectable after
the fact because the file counter skips and the blur is visible. The measured jitter is below the
1 s resolution of the timestamps.

## 11. Defects in the code, left in place on purpose

`Red_multimodal_VC.py`, the script that produced the originally submitted model, trains the final
model twice: it recompiles the same object rather than creating a new one, and `create_model()`
is defined but never called. Recompiling resets the Adam moments, so the first stage is lost in
practice and the published model is equivalent to a single 40-epoch run. This was confirmed by
retraining, not only by reading the code: a clean single-stage run gives R² 0.878 and MAE
0.4347 µm against 0.885 and 0.428 published, a difference within the sampling noise of a single
specimen. The script is kept unmodified as the record of what was executed;
`entrenar_final_limpio.py` is the corrected single-stage version.

## 12. A stale comment

The module docstring of `src/6_figuras_articulo/fig12_arquitectura.py` still describes the
selected variant as the one using `GlobalAveragePooling2D`, which was true when only variant A
was published. The code itself is correct and follows `ARQ_PUBLICADA = {'Ra': 'B', 'Rz': 'C'}`;
only the prose is out of date.
