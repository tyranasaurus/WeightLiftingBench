# Method

A small hand-designed network that classifies 96×96 colour photos into "person"
vs "no person" using a skin-colour signal with a shape-aware readout.

## Architecture

1. `conv_a` (1×1, 3 → 14): seven half-space tests for skin, each duplicated so
   that a later subtraction turns `relu(f_i) - relu(f_i - 1)` into a clip of the
   signed distance into [0, 1]. The seven tests use standard YCrCb skin colour
   bounds (Cr ∈ [0.522, 0.678], Cb ∈ [0.302, 0.498]), a minimum luminance
   (Y > 0.07), and two colour-ordering constraints (R > G, G > B) all scaled by
   s = 20 so that a comfortable margin gives a clip value of 1.
2. `relu_a`, `conv_b` (1×1, 14 → 1), `relu_b`: sums the seven clipped distances
   and subtracts a threshold of 5.5, so each pixel gets a "skin indicator"
   in [0, 1.5] that fires only when nearly all skin conditions are well
   satisfied.
3. `pool_cell` (avgpool 12×12, stride 12) reduces to an 8×8 grid of per-cell
   skin density.
4. `pool_max` and `pool_avg` extract the max and the average cell density into
   two scalars.
5. `dense_max`, `dense_avg`, `scores = add[...]` combine them linearly:
   * class-0 score = `beta` (threshold bias)
   * class-1 score = `a * max_cell - b_avg * avg_cell`
   
   The network predicts "person" whenever the peak cell density is sufficiently
   larger than the overall average — faces/bodies are a concentrated blob of
   skin colour, whereas brown walls/sand/wood surfaces produce high peak *and*
   high average and are rejected. Values chosen: a = 4, b_avg = 16, beta = 1.1.

## How the weights were produced

`build.py` writes the weights from these fixed numbers alone — standard YCrCb
skin colour bounds, the scaling `s = 20`, the clip threshold `5.5`, and the
`(a, b_avg, beta) = (4, 16, 1.1)` readout constants. No training, no fitting,
no statistic of the training set appears as a weight; training images were
only viewed and reasoned about qualitatively (dominant object colours, typical
compositions).
