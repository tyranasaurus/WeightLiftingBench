# Method

## Architecture (5 layers, ~10,000 weights)

```
image (1,28,28)
 └ norm                          -> contrast-normalised image        (1,28,28)
 └ conv 9x9, 8 filters           -> oriented ridge (bar) filters:     (8,28,28)
                                    4 orientations, each with its negative
 └ relu                          -> polarity-invariant bar energy
 └ maxpool 4x4                    -> peak energy per 7x7 cell/orient    (8,7,7)
 └ dense -> 24                    -> hand-written orientation templates (24,)
```

## How the weights were produced

**Filters.** The 8 convolution filters are oriented second-derivative-of-Gaussian *ridge* detectors written
from a formula (four orientations: vertical, horizontal, and the two diagonals), each stored together with
its negative. Passed through `relu`, a filter and its negative together give the magnitude of oriented bar
energy, so the response does not depend on whether the hand is lighter or darker than the background (the
dataset contains both). `maxpool` then records the peak energy of each orientation in each cell of a 7x7
grid.

**Readout.** The dense layer is a hand-drawn template over (orientation, grid-cell), one row per letter.
It is written from the ASL alphabet using the idea that different handshapes put edges of different
*orientation* in different *places*:

* a fist/blob (A, E, S, M, N, T) has a rounded **horizontal** top and bottom edge and no vertical reach up;
* fingers-up letters have **vertical** finger edges in the top rows — B wide, D one in the centre, U two
  together, V two with a gap, W three, I one to the side, K/R a pair with a thumb/cross, X a short hook,
  F three on the right with a loop, L a left post plus a horizontal thumb;
* sideways letters (G, H) have a **horizontal** bar reaching the far left;
* C and O are rings (vertical left/right sides plus horizontal top & bottom), O closed and C open on the right;
* P and Q point down; Y reaches out to both sides.

The two bottom grid rows — the wrist/forearm that enters from the bottom edge of every image — are given
zero weight so the arm cannot drive the decision, and every letter's template is unit-normalised so letters
compete fairly.

No gradient descent, regression, class averages, clustering, PCA or stored image data are used. `build.py`
contains only the filter formula and the typed templates and writes `network.json` and `weights.npz` with no
input files; viewing the training images only guided which rows/columns each handshape occupies.
