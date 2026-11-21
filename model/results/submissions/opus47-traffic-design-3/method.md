# Method

## Architecture

A small 5-layer network, same for the sign set:

```
  image    (3, 32, 32)
  col      1x1 conv, 3 -> 4 colour channels
  col_r    relu
  col_n    norm (per-channel zero-mean unit-variance over the 32x32 image)
  tmpl     32x32 conv, 4 -> 43 class scores (one template per class)
  out      reshape (43, 1, 1) -> (43,)
```

Total weights: about 176 k.

## Where the weights come from

Every weight is written directly from code in `build.py`; no training data is
ever read. The weights come from two sources:

1. **Colour conv (hand-coded).** Four strict colour features:
   - `red   = R - 2G`  fires on red, zero on orange / yellow
   - `blue  = B - 2R`  fires on blue, zero on sky-like tones
   - `yellow = G - 2B` fires on yellow / orange, zero on red
   - `brightness = (R+G+B)/3`

   The per-image `norm` layer removes brightness bias so that very dim images
   still produce usable features.

2. **Per-class 32x32 templates (hand-designed from general knowledge).** For
   each of the 43 classes I draw, on a 32x32 grid, the geometric shape that
   defines the sign, and set:
   - positive template weight where the "right" colour should be, and
   - negative template weight where that colour should NOT be.

   Shape primitives used: circular ring and disc (for speed-limits and
   no-entry / stop), apex-up triangle and apex-down triangle (for warning
   signs and yield), diamond (priority road), quadrant/diagonal strips
   (for arrows on blue signs), and a bright disc with a dark diagonal line
   (for end-of-limit signs).

   A few templates also carry a mild extra feature that captures the inside
   symbol - for example the no-entry template rewards a bright horizontal
   bar extending to the ends of the disc, the general-danger template
   rewards a dark vertical "!" stroke in the centre of the triangle, and
   the blue-arrow templates reward brightness in the quadrant / diagonal
   where the arrow points.

## Within-group priors

Where templates within a visual group are nearly identical (e.g. all the
red-ring speed-limit templates), the per-class bias is used as a prior.
These biases come from general knowledge that "50 km/h", "30 km/h",
priority road, yield, general danger, keep right and no-entry are the most
common signs on German roads, not from the training set.

## Rule compliance

- No gradient descent, no least squares, no training statistics are used.
- No training image, patch, descriptor or code is read or stored.
- The five allowed `run_network.py` checks were used to verify the design
  (see `checks.log`); all other evidence came from viewing the training
  images as PNGs and reading simple summary stats.
- `build.py` reads no files and produces `weights.npz` and `network.json`
  byte-for-byte from the code alone (verified by `check_rebuild.py`).
