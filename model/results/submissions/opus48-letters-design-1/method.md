# Method

## Architecture
A translation- and polarity-invariant synthetic-template matcher (6 layers):

1. `norm` — per-image contrast normalization (zero mean, unit variance), so absolute brightness/contrast don't matter.
2. `conv` `tmatch` — 48 full-frame 28×28 filters = 24 handshape templates, each stored twice (once negated), padding 6, stride 1. The output at each position is the correlation of the normalized image with a template.
3. `relu` — keep positive correlations.
4. `maxpool` (13×13) — take the best-matching position for every filter, giving translation invariance (the hand is not centered by any rule).
5. `reshape` → 48.
6. `dense` `cls` — for each class, sum its two polarity channels (fixed 0/1 weights), so the score is high if the handshape matches the image whether the hand is lighter *or* darker than the background. Argmax over the 24 scores is the prediction.

## How the weights were produced
Every number is derived from geometry, not from data. For each of the 24 letters I drew a synthetic 28×28 greyscale handshape from general knowledge of the ASL manual alphabet — a palm disk plus finger/thumb "capsules" placed at the angles and positions that each sign uses (fingers up for B/D/U/V/W, pointing left for G/H, down for P/Q, a ring with a hole for O, an open crescent for C, a spread thumb+pinky for Y, and so on). Each drawing is zero-meaned and L2-normalized to form a correlation template; the template and its negation become the conv filters. The `dense` layer uses only fixed 0/1 weights.

`build.py` constructs all of this from code and reads no files; `check_rebuild.py` reproduces `weights.npz` exactly (max relative difference 0.0).

## Rules compliance
No gradient descent, no fitting, no class averages, no statistics of the training images, and no pixels/coordinates copied from any image are used as weights. The training set was used only, within the allowed budget, to view images and simple summaries (per-class means, gradient-magnitude maps) to decide each sign's canonical orientation and handedness, and for the 5 logged accuracy checks. The templates are built purely from the general shape of each ASL letter.

## Notes
Distinct shapes (C, B, O, the pointing-up and pointing-sideways groups) match well; the several fist-like signs (A, E, M, N, S, T) are nearly indistinguishable at 28×28 and are the main source of error. On a training-set check the network scored ~0.13 accuracy, well above the 1/24 ≈ 0.042 chance level.
