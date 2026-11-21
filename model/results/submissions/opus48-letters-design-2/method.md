# Method

## Architecture
A fixed, formula-based front end followed by synthetic matched filters. No training,
fitting, or statistic of the training data is used; every weight comes from closed-form
filter formulas and from hand-shape geometry typed from general knowledge of the ASL
alphabet. `build.py` writes `network.json` and `weights.npz` with no file input.

Layers (`network.json`):
1. `norm` — standardise the input image (removes brightness/contrast).
2. `conv` 5×5, 10 channels — a Gaussian blur ("mass"/silhouette) plus four oriented
   Gaussian second-derivative line detectors, each kept in a `+` and a `-` channel.
3. `norm` — per-channel standardisation (equalises channel scales per image).
4. `relu` — the `+`/`-` channel pair of each feature rectifies to its magnitude |f|.
5. `avgpool` 2×2 → a 10×14×14 contrast- and sign-invariant magnitude descriptor.
6. `dense` — matched filters: one row per (letter, scale, shift) template.
7. `reshape` + `maxpool` + `reshape` — for each of the 24 letters, keep the best-matching
   template variant, giving the 24 class scores.

## How the weights were made
The conv kernels are evaluated from the Gaussian / Gaussian-derivative formulas.
For every letter I render an idealised 28×28 hand silhouette from a typed geometric
model (palm ellipse, wrist, finger capsules at chosen angles/lengths, thumb, and rings
for C/O) — the shapes come from knowledge of each ASL sign, not from any image. Each
render is pushed through the same front end to get its descriptor; that descriptor
(mean-removed, made orthogonal to a single typed "generic hand" direction so matching
keys on each letter's distinctive protrusions/holes rather than the shared hand body,
then unit-normalised) becomes one dense matched-filter row. Each letter is rendered at
5 scales × 9 small shifts for translation/scale tolerance, and the max over its variants
is taken. Rule compliance: each generated template normalises only itself; no average,
prototype or other statistic across generated images or across training images sets any
number, and the network was run on training images only for the 5 permitted checks.

## Honest note on performance
Hand-setting weights to separate 24 visually similar ASL handshapes at 28×28 is hard.
On the training set the design reaches ~7.6% (chance is 4.17%). The main limitation is
that the idealised binary templates do not perfectly reproduce the feature statistics of
real shaded photographs, so several letters (especially the many fist-like signs) remain
confused; the most distinctive silhouettes (C, O) are recognised best.
