# Method

**Architecture.** A fixed edge-orientation front end followed by a template bank.

1. `grad` (5x5 conv, no padding): Gaussian-derivative filters (sigma = 1) at 8 orientations, both signs; `relu`.
2. `ori` (1x1 conv) + `relu`: unsigned orientation energy, sharpened by subtracting the response at the perpendicular
   orientation (so the result does not depend on whether the hand is lighter or darker than the background).
3. `cell` (4x4 conv, stride 2, binomial weights): pooling into an 11 x 11 grid of overlapping cells.
4. Local contrast normalisation built from `dense` (a permutation), a 1x1 `conv` that gathers, for every cell, +/- its own
   8 values, +/- the values of its 3 x 3 neighbour cells and a +/- pedestal, and `norm` over each such group (mean 0 by
   construction, so `norm` divides by the local root mean square); a 1x1 `conv` keeps the cell's own normalised values.
5. `match` (dense): one row per template; `maxpool` over the templates of each class gives the 24 class scores.

**Where the weights come from.** Nothing is trained or fitted and no training image is used in any weight.
The front-end filters are formulas. Each template is the front-end response to one synthetic picture rendered by
`build.py` from a 3D hand model (capsule bones with ordinary human proportions, z-buffer, Lambert shading with crease
darkening) posed in one manual-alphabet handshape. The 24 handshapes, the way each is held towards the viewer, and the
ranges of the seeded random variation (joint angles, viewpoint, size, placement, light) are typed in by hand from general
knowledge of the ASL alphabet and from looking at what the classes look like. Each synthetic picture becomes its own
template, lightly smoothed over neighbouring cells and normalised by itself alone (zero mean, unit length). No statistic
across generated images is used, and no setting was chosen by scoring on generated images.

**Checks used.** Three of the five allowed runs on the 240-image training sample. Two were on earlier revisions
(40.8 %, then 44.6 % after correcting hand size and several poses). The submitted revision (4000 templates per class,
further pose/size corrections made from side-by-side viewing, plain best-match scoring) scored 39.6 % on its one check:
it stopped the flat-hand "B" templates from absorbing other upright letters, but did not improve overall accuracy, and
U, D and the fist letters (A, E, M, N, S) remain poorly separated.
