# Method

## Architecture

```
image (3, 128, 128)
  -> avgpool 2x2 stride 2           -> (3, 64, 64)
  -> conv 1x1  (RGB -> lum/blue/red)-> (3, 64, 64)
  -> avgpool 2x2 stride 2           -> (3, 32, 32)
  -> norm (per-channel)             -> (3, 32, 32)
  -> relu                           -> (3, 32, 32)

Branch A (non-oriented templates)
  -> conv 32x32  (16 filters)       -> (16, 1, 1)
  -> reshape                        -> (16,)
  -> dense -> (10,)                 (dense_a)

Branch B (oriented templates, 8 angles)
  -> conv 32x32  (8 filters)        -> (8, 1, 1)
  -> reshape                        -> (1, 1, 8)
  -> maxpool [1,8] stride [1,8]     -> (1, 1, 1)   <- max over orientations
  -> avgpool [1,8] stride [1,8]     -> (1, 1, 1)   <- mean over orientations
  -> two parallel dense layers (dense_max, dense_avg) -> (10,) each

out = dense_a + dense_max + dense_avg          (add)
```

## How the weights were produced

All weights were written out by `build.py` as closed-form formulas; no training, fitting, or image-derived statistics were used.

1. **Color 1x1 conv**: constant mixing matrix producing a luminance channel
   `(R+G+B)/3` and two colour-contrast channels `B-R` (blueness) and `R-B`
   (redness).

2. **`norm` + `relu`**: standardises each channel so sky background is near
   zero; ReLU keeps only above-background pixels, giving a rough galaxy mask
   in each channel.

3. **`feats` (16 filters, 32x32)**: closed-form templates built from
   formulas on a 32x32 centred grid. They include isotropic Gaussians of
   several sizes (compact vs extended), annular rings, differences of
   Gaussians (concentration contrast), and colour-channel templates (blue
   centre, blue ring, red centre, red extended, blue-dominant indicator).
   Each kernel is normalised to unit sum (or unit sum per positive lobe for
   difference-of-Gaussian kernels).

4. **`oriented` (8 filters, 32x32)**: narrow elongated Gaussians
   (sigma_major=8, sigma_minor=1.5) at 8 evenly spaced angles 0..7pi/8,
   applied to the luminance channel. The max-pool across orientations
   gives a rotation-invariant "strongest axis" response; the avg-pool
   gives the mean. Using them through separate dense layers with opposite
   signs implements X*(max - avg), a scale-invariant elongation measure
   that fires for edge-on and cigar-shaped galaxies but not for round
   galaxies.

5. **Dense weights** were set by hand, class by class, from general knowledge
   of what each galaxy class looks like. Examples:
   * round smooth (2): large red extent, strongly disfavours blue, modal-class bias.
   * in-between round (3): similar to class 2 but with mild compactness preference.
   * cigar (4): small reddish + elongation from path B.
   * barred (5), unbarred tight (6), unbarred loose (7) spirals: various blue
     channel templates (centre, ring, extended).
   * edge-on no bulge (8): preferred edge-on default; dislikes a bright central spot.
   * edge-on with bulge (9): elongation plus a red bulge detector with a
     bias threshold so it only beats class 8 when a real bulge is present.
   * classes 0 (disturbed) and 1 (merging) are hard to detect with
     hand-crafted filters and receive mostly default weights.

## Lessons from the 5 training-set checks

* **Check 1** (0.1020): The classifier predicted almost exclusively class 2
  with a few class 9 predictions. The dense weights overpowered any
  feature-specific signal; features had much higher raw magnitudes than
  the ~1 I had assumed.
* **Check 2** (0.2280): After rebalancing weights with better scale
  calibration, classes 2 (~75% recall), 9 (~70%) and 7 (~35%) started
  working; classes 0, 1, 3, 4, 5 were still never predicted.
* **Check 3** (0.2150): Boosting class 3's bias too far made it steal
  essentially all of class 2's predictions -- the "modal red smooth"
  cluster is hard to split without a true size discriminator.
* **Check 4** (0.2280): Restoring class 2 as the red-smooth default and
  slightly lowering class 9's elongation weight was the best balanced
  state; this is the weight set that is submitted.
* **Check 5** (0.1835): Further tweaks to boost class 3/5 again caused
  class 3 to eat class 2/9 and class 5 to eat classes 6/7 (loose spiral
  confusion). Reverted to the check-4 weights.

The final submission uses the check-4 weight set, which achieves roughly
0.228 accuracy on a 2000-sample stratified training sample (random is 0.10).
