# Method

## Architecture

Six-layer network, all weights hand-designed:

```
image (3,32,32)
  -> conv 1x1, 8 channels      (col)   — colour-opponent features
  -> relu                      (col_r)
  -> avgpool 4x4 stride 4       (pool) — 8x8 spatial grid of colour maps
  -> norm                      (pool_n) — per-colour-channel standardisation
  -> reshape to (512,)         (flat)
  -> dense 512 -> 43           (out)   — class-wise shape/colour templates
```

Total: 22,091 weights (limit 120,000,000).

## How the weights were produced

**`col` (1x1 conv, 8 colour channels)**: a hand-chosen colour-opponent basis
computed per pixel from raw RGB —
  0 red = 2R − G − B,
  1 blue = 2B − R − G,
  2 yellow = R + G − 2B,
  3 bright = R + G + B,
  4 dark = −(R + G + B),
  5 red_pure = 2R − 2G (red but not yellow),
  6 yellow_pure = 0.3R + 1.3G − 2B (yellow, not red),
  7 blue_dim = 1.5B − 0.3R − 0.7G (works on weakly lit blue signs).
Biases threshold each so that typical colour pixels trigger after ReLU.

**`pool` → `pool_n`**: 4×4 average-pool gives an 8×8 spatial map per colour. The
subsequent `norm` layer zero-means and unit-variances each colour channel over
its 8×8 cells, so the dense layer looks at *where* each colour is concentrated
rather than the absolute brightness of the sign (which varies with lighting).

**`out` (dense 512 → 43)**: each class's weight vector is a hand-written template
over (colour × 8×8 location) that encodes what the sign looks like:
circles get red along a ring mask (and specifically at the circle-only
mid-side cells), upward warning triangles get red at triangle-edge and
base-corner cells (and explicitly suppress the circle-side cells), the yield
sign uses the inverted-triangle mask, the priority road uses a diamond mask
in yellow, the blue mandatory signs ask for blue over the whole disc and a
white arrow pattern in the centre. Masks are generated analytically (ring =
distance-to-centre band, triangle edges from line equations, diamonds from
|y|+|x|, arrow silhouettes typed in as small 2D patterns). After all
contributions are added, each class vector is L2-normalised so classes are
directly comparable, and a small per-class bias (e.g. +0.25 for the
standard 50 km/h sign, +0.2 for the common 30 km/h, −0.1 for all-vehicles-
prohibited) tilts the final argmax toward the kind of signs that are
common on German roads. These biases come from general knowledge of road
signs, not from training-set frequencies.

No gradient descent or fitting is used anywhere; `build.py` writes every
number from constants in its own source.

## Notes from the five accuracy checks

- Check 1 (0.136): Initial design confirmed that blue mandatory signs and the
  shape-specific classes (priority road, yield, stop, roundabout) work;
  the eight red-ring classes collapsed to one dominant class because their
  templates were too similar.
- Check 2 (0.145): Added negative weights at triangle-corner cells and at
  the inverted-triangle top row for circle classes, boosting group
  separation. Yellow colour threshold lowered to help dim signs.
- Check 3 (0.140): Replaced a mis-placed side-mid mask with correct
  circle-ring middle cells; strengthened per-class positive/negative
  cell-level discriminators. Slight regression; needed stronger
  within-group discriminators.
- Check 4 (0.148): Added per-class interior patterns (anti-dark for empty
  circles, strong dark for vehicle silhouettes, pictogram position hints
  for warnings) and small class priors from general road-sign knowledge.
- Check 5 (0.155): Strengthened class 15 anti-dark and class 32
  very-low-colour signatures, and sharpened the stop vs no-entry contrast
  (stop has a broad white block in the centre, no-entry a thin horizontal
  bar).
