# Method

Prototype-matching network with hand-written weights; no training data is used anywhere in build.py.

**Preprocessing.** A 1x1 conv forms five maps from RGB: intensity I, and three linear "colour cone" tests
(red = R - 1.45G + 0.3B, blue = B - 0.65G - 0.5R, yellow = G - 1.2B - 0.15R, each minus a small threshold,
then relu) plus an all-zero reference map. I alone goes through `norm` (per-image contrast normalisation of the
grey image). The five maps are reshaped into one channel and jointly normalised, which makes the colour maps
brightness-invariant with one shared scale; the zero map carries the shared offset so that templates with
non-zero sums can be used. A 7x7 Laplacian-of-Gaussian conv, rectified in both signs, gives an edge-energy map.

**Templates.** Every one of the 43 signs is drawn procedurally from its known geometry (red ring of width 0.1 D,
warning triangles with a red border, yield, octagon with STOP letters, yellow diamond, blue discs with arrows,
end-of-limit discs with the black diagonal band, digits as DIN-like strokes, simple pictograms). Each sign is
rendered at 7 sizes (56 to 104 % of the frame) and 3 aspect ratios, blurred, and turned into: a zero-mean
unit-norm grey pattern (interior weighted higher), an expected-edge-layout pattern, and colour patterns that
reward a colour where it is expected and penalise it elsewhere on the sign (small coloured pictogram parts are
area-capped and must be surrounded by non-colour). These are 32x32 conv kernels applied with padding 3, so the
network searches +-3 px of shift; max-pooling then takes the best shift and best size/aspect variant per class.
A second, rectified pictogram-only branch (tight zero-mean box around digits or pictograms) adds class-specific
evidence for circular and triangular signs. The final dense layer is the identity over the 43 class evidences.

**Checks.** Two of the five allowed checks were used (66 % and 49 % on the training set); the second showed that
large-scale templates became generic attractors, which led to the unit-norm merged grey pattern, stronger colour
evidence and the pictogram branch in the submitted version. The third check run was killed before finishing, so
the submitted version is validated only on synthetic renderings (about 91 % on perturbed synthetic signs).
