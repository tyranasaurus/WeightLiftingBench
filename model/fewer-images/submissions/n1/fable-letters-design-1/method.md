# Method

## Architecture (16 layers, 17.3 M weights)

1. **Oriented edges.** `g` (3x3 conv, no padding) holds Sobel derivatives steered to 8 orientations, each with both
   signs; after a relu, the 1x1 conv `ori` forms `|d_k| - |d_(k+90deg)|` and a relu keeps the positive part. This gives
   8 maps of *unsigned* edge orientation, so it does not matter whether the hand is lighter or darker than the
   background.
2. **Pooling.** `pool` is a Gaussian (sigma 1.5 px, stride 2) giving 13 x 13 cells, for tolerance to small deformations.
3. **Local contrast normalisation.** The dense layer `gather` lays out, for every cell, 32 numbers: `+F, -F` for its 8
   orientation energies and `+G, -G` for a neighbourhood average of them. Reshaped to (169 cells, 32, 1), the `norm` op
   then divides each cell by the root-mean-square of its own block (the +/- pairs make the mean zero, and an input gain
   turns the op's 1e-5 into a hand-chosen noise floor). A max-pool with stride 4 picks the `+F` entries, and the dense
   layer `back` is a permutation back to (8, 13, 13). Weak and strong outlines end up with similar strength.
4. **Matched filters.** `tm` is a 13x13 conv with padding 2 holding 6,000 templates (250 per letter), so each template is
   tried at 5 x 5 shifts (up to +-4 px). Max-pooling over shifts, then over each letter's 250 templates, gives the 24
   class scores.

## Where the weights come from

Layers 1-3 are formulas (Sobel, cos/sin steering, Gaussians, a permutation) with constants chosen by hand.

Each template is the feature map of **one image drawn by code in `build.py`**: a small 3-D hand model (palm slab,
tapered finger and thumb capsules, forearm; proportions from general anatomy) posed as one ASL letter using joint angles
and thumb placements I typed from my knowledge of the manual alphabet, rendered with a z-buffer and simple shading.
Viewpoint, size, lighting and small joint jitter are drawn from hand-chosen ranges with a fixed seed; a fifth of the
drawings are mirrored. Each template is normalised by itself alone (own mean removed, own norm divided out). No
statistic across drawings, and nothing from the training images, sets any number. The training images were only
looked at (as PNGs) to understand how the letters are framed and viewed, e.g. that C, O, D and X are seen side-on and
that G/H point sideways.

## Checks, and how the final design was chosen

All 5 checks were used, on the 24 training samples:

| check | design | accuracy |
|---|---|---|
| 1 | the design described above, every number fixed before any check | 9/24 = 37.5% |
| 2 | + crease shading, wider pose/light ranges, 400 drawings per letter | 7/24 = 29.2% |
| 3 | + same random draws for every letter, an edge threshold, wider size range | 5/24 = 20.8% |
| 4 | threshold removed from the network, size range wider still | 4/24 = 16.7% |
| 5 | check-1 design restored exactly (confirmation of the submitted files) | 9/24 = 37.5% |

The submitted network is the check-1 design. To be plain about it: this is also the design with the best check score,
so the 37.5% is flattered by that choice and by the tiny sample (one image per class, roughly +-10 points of noise).
My reason for reverting is that none of the revisions I made after seeing check results showed any benefit, and the
original is the only version whose settings were all fixed before any check. I expect held-out accuracy to be lower
than 37.5%, plausibly in the 20-35% range.

Known weaknesses: G and P were missed in every check (G read as Q, P as K), and the fist-like letters
(A, E, M, N, S, T) are largely confused with each other, since they differ only in thumb placement at 28 x 28.

`build.py` reads no files, runs in about 40 s, and `check_rebuild.py` reproduces the submitted weights exactly.
