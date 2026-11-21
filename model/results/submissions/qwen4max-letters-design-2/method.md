# Method

## Architecture (20 layers, ~2.0M weight entries, 9.2k nonzero)

1. **`n0` — norm** on the input image: removes global brightness/contrast (the photos
   vary in background brightness and the hand is not always the lighter region).
2. **`f1` — 11x11 conv, 24 channels.** A formula-only filter bank: first-derivative-of-
   Gaussian edge filters at 0/45/90/135 degrees; elongated difference-of-Gaussian *ridge*
   (line) detectors at the same four angles, tuned to a 3 px finger width and smoothed
   along the finger with sigma 4; two wider DoG *gap* detectors (6 px); an isotropic DoG
   blob detector (hole/ring scale); and a Gaussian smoothing pair. Every filter is made
   exactly zero-sum and normalised to unit response on an **analytic ideal target**
   (a unit-contrast step, an infinite 3 px stripe, a 6 px stripe, a radius-3 disc).
   Each filter appears with both signs.
3. **`r1` relu + `a1` 1x1 conv** sum the +/- pairs, giving 11 polarity-invariant
   `|response|` channels plus bright/dark half-maps.
4. **Four pooling views**: `bv` = max over each of four row bands (rows 0-6, 7-13,
   14-20, 21-27) keeping every column; `bh` = the same over four column bands keeping
   every row; `g7` = 4x4 average grid; `g4` = 7x7 max grid.
5. **`d_bv/d_bh/d_g7/d_g4` + `F` add** — 519 hand-specified linear functionals:
   per-band column/row profiles, their first and second spatial moments, left/centre/
   right and far-left/far-right splits, and regional energies for 21 rectangles.
6. **`dh`/`H` — 82 detectors.** Each is `relu(energy_A - lambda * energy_B)`: where the
   prongs sit vertically, ridge-vs-boundary ratio (narrow finger vs wide blob), wide
   gaps between prongs, spread and lateral position, far-left/far-right horizontal
   prongs (thumb, G/H), enclosed-hole fractions, C-crescent vs O-ring asymmetry,
   diagonal ridges, and top/bottom strip occupancy. **All are positive-homogeneous**,
   so the whole score vector scales with image contrast and the argmax is contrast
   invariant — no absolute thresholds had to be calibrated.
7. **`dp`/`P` — 12 pose units** (up / down / right / fist / ring / crescent / spread /
   narrow / left-thumb / right-thumb / left-position / centre-position).
8. **`s1` + `s2` + `scores`** — a hand-written 24 x (82+12) rule table.

## How the weights were produced

Everything comes from closed-form filter formulas and from a hand-written geometric
rule table based on general knowledge of the static ASL letters: thumb side on the
image right, pinky on the left; B = four together, U = two together, V = two apart,
W = three apart, I = single far-left prong, D = single centre prong plus a ring,
L/Y = single prong plus a horizontal thumb to the right, G/H = horizontal prongs to
the upper right, P/Q = prongs pointing down, C = crescent open to the right,
O = closed ring with a central hole, and the six fists (A/E/M/N/S/T) separated by
thumb placement (vertical right / hidden / low left / across the front / nub on top).
No training image, patch, descriptor, statistic, label average or fitted readout is
used anywhere; `build.py` reads no files and regenerates both answer files exactly.

## Honest status

`check_network.py` passes and `check_rebuild.py` reports REBUILD OK (max relative
difference 0.0). Three of the five allowed checks were used. The first two exposed a
real indexing bug (per-view dense rows were written at local instead of global feature
indices), which was fixed. **The current answer nevertheless only reaches 0.0385
accuracy on the training set** — the hand-written rule table is not correctly
calibrated against the actual filter-response magnitudes, and I ran out of time to
diagnose it. The architecture and the weight-generation code are complete, valid and
reproducible, but the classifier's accuracy is poor and I am reporting that plainly
rather than claiming success.
