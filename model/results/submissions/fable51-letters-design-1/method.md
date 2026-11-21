# Method

## Architecture (network.json)
Two correlation branches over hand-designed features, combined by a max over hand-drawn templates.

* `n0 = norm(image)` standardises the picture.
* **Pixel branch**: `n0 -> 5x5 Gaussian blur -> norm -> conv "pix"` (1728 kernels of 28x28, stride 2, padding 2).
  Each kernel is a template, so the 3x3 output is the template's correlation at shifts of -2/0/+2 px.
* **Structure branch**: `n0 -> conv "filt"` (80 fixed 13x13 filters: 8 orientations of even-Gabor bar detectors at two
  widths, one finger and two fingers wide, 4 Gaussian-derivative edge detectors, each with its negative and a copy
  biased by -0.3) `-> relu -> 1x1 conv "mix"` (forms min(relu(x), 0.3): 16 bright-bar, 16 dark-bar channels for
  fingers and creases, 4 unsigned edge channels) `-> maxpool 3x3/1 -> avgpool 4x4/2` (36 x 12 x 12 cells)
  `-> joint norm -> conv "str"` (1728 kernels of 12x12, padding 1, giving the same 3x3 shift grid).
* **Polarity gate**: a 28x28 conv on the blurred image measures centre-minus-border brightness; relu of -2p / +2p is
  subtracted (1x1 conv "gate") from light-hand / dark-hand templates respectively, so the dark-hand templates are
  only used when the hand is darker than its surroundings.
* `score = pix + str + gate -> maxpool over shifts -> reshape (24, 72, 1) -> maxpool over the 72 templates of each
  class -> reshape (24,)`.

## Weights (build.py, no data read)
All filters come from formulas (Gaussian, Gaussian derivative, Gabor).  The templates come from a parametric hand
renderer typed from general knowledge of the ASL alphabet: a rounded palm, forearm, finger capsules with lengths and
joint angles per letter, curled finger stubs with dark creases for fists, thumb positions (alongside, across,
out, between), side views for C, O, G, H, P, Q, T, with a light-from-the-left shading.  For each of the 24 letters:
2 typed pose variants x 3 rotations (-12, 0, 12 deg) x 3 scales x 2 vertical placements x 2 lighting cases
(hand lighter / darker than the background) = 72 renders.  Each render is passed through the same front end as the
network and becomes one template, centred and normalised by itself alone, multiplied by an emphasis map built from
that render's own geometry (extended fingers, thumbs and hooks 1.0, rest of hand 0.6, forearm and background 0.25).
No statistic of any kind across renders or training images is used.

## Checks
Five runs on the training set (see checks.log): 18.0 % (raw pixel + signed-gradient correlation),
18.9 % (bar/edge features), 22.4 % (low-contrast shaded renders with explicit dark-hand variants), 23.3 % (emphasis
maps, two placements), 18.9 % (a broader pose rewrite, which hurt).  The submitted network is the 23.3 % design
(rebuilt from source; the rebuild matches the recorded self-test fingerprints exactly).  Distinctive silhouettes
(B, D, F, I, L, O, Y, A) are mostly right; fist letters (E, M, N, S, T) and one-finger-up letters (R, U, K) are
largely confused with D and X, the generic fist-plus-finger templates.
