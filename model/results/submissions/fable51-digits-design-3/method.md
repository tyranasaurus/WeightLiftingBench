# Method

**Architecture.** `image (1,28,28)` → `feat`: conv 1→7, 11×11, pad 5 (fixed filter bank) → relu →
maxpool 3×3 stride 1 (±1 px local tolerance) → avgpool 2×2 → `(7,13,13)` → `tmpl`: conv 7→8000,
13×13 kernel, pad 1 (one template per output channel, tried at ±1 cell shifts) → maxpool 3×3 (best
shift) → reshape `(1,10,800)` → maxpool `[1,800]` (best template of each class) → reshape `(10,)`.
The class score is the best normalised correlation between the image's feature map and any template
of that class: a nearest-template rule computed inside the network.

**Filter bank (formulas only).** Channel 0 is a Gaussian blur (σ = 1.5) of the ink. Channels 1–6 are
ridge detectors at 0°, 30°, …, 150°: the negative second derivative across the stroke of an
elongated Gaussian (σ_across = 1.3, σ_along = 2.2), zero-mean, scaled so an ideal 3-px-wide bar
gives response 1. Gains 1.5 (intensity) and 0.8 (ridges) were chosen by hand so the two groups carry
comparable energy on a stroke.

**Templates (knowledge, not data).** Each digit class is drawn as pen strokes along paths typed from
general knowledge of how the digit is written, with several styles each (open/closed 4, plain,
looped, S-shaped or Z-like 2, 7 with crossbar / tick / hooked bar, straight, hooked or open-loop 9,
figure-eight, two-loop or S-plus-diagonal 8, flat-topped or round-bowl 5, curved or straight 6
descent, round or flat-topped 3). Style parameters, slant, rotation, width-to-height ratio, pen
width and a smooth low-frequency warp are drawn from hand-chosen ranges with a fixed seed. Each
drawing is rendered at 3× resolution, then normalised the way the task describes MNIST: cropped,
scaled to fit 20×20 keeping proportions, centre of mass placed at the centre of 28×28. It is passed
through the same filter bank, and the resulting 7×13×13 map, with its own mean subtracted and
divided by its own norm, is the template (one conv output channel). 800 templates per class
(8000 in all; an earlier 3000-per-class build of the same code was interrupted by the time limit,
so the submitted answer uses the smaller count).

**Checks.** Two of the five allowed training checks were used (10,000 training images each):
check 1 (first stroke models, 2000 templates/class) 87.0%; check 2 (broadened stroke models,
3000/class) 88.7%. The confusion tables showed the remaining errors were in classes whose
handwriting varies most (5→6, 2→3, 8→9), so the 2, 5 and 8 stroke models were widened further from
general knowledge of those digits (flat-topped 5 bowls with slanted verticals, slanted S-plus-diagonal
8s, small high heads on 2s). The submitted build was not re-checked on training data.

**What was not done.** No training image or label influenced any weight; no statistic over the
generated drawings was used (each template is normalised by itself alone); no threshold, gain or
setting was chosen by scoring generated images; no variants were scored to pick the best.
