# Architecture and how the weights were produced

## Architecture (all weights from formulas + general astronomical knowledge)

A hand-designed, fully interpretable pipeline, ~130 layers, 4.2 M weights:

1. **Photometric pre-processing.** 1x1 conv splits the image into luminance
   `L=(r+g+b)/3` and raw colour `r-b`; a `norm` layer standardises each channel over
   the frame (removes sky level and exposure/contrast differences); `relu(L-0.15)`
   yields a sky-free, contrast-normalised light map. Two avgpools give 64 px and
   32 px scales.
2. **Oriented ridge banks.** At 8 orientations and two scales, conv filters of the
   form `G(along,across) - G(across,along) - c*surround`. Cross-orientation
   suppression makes them blind to any radially symmetric profile (cores, profile
   shoulders), and the surround cancels broad bumps, so they respond only to
   genuinely 1-D light: spiral arms, dust lanes, bars, edge-on discs. A running-max
   chain (1x1 conv selectors + add + relu) takes the max over orientations, giving
   orientation-invariant ridge-strength maps; regional means of these are the
   "texture / thinness / bar" measurements.
3. **Analytic photometry.** Dense layers whose weights are closed-form masks
   (Gaussian rings, ring x von-Mises sectors, colour apertures, centroid weights)
   measure: 9 radial ring means, 4 radius bands x 8 sector means, inner/outer
   colour, light centroid, and (via an 8x8 maxpool grid) central and off-centre
   peak brightness.
4. **Small statistic chains.** reshape+maxpool and dense/relu/add chains turn the
   sector means into per-band max, mean, opposite-sector asymmetry and
   opposite-pair minimum (multiplicity cues); everything is fused into one
   80-vector of interpretable measurements.
5. **Indicator layer.** A dense+relu layer computes 43 hand-chosen, scale-free
   thresholded comparisons: sector-elongation ladders (max/mean sector light at
   ratios 1.2/1.55/1.9 etc., which separate round / intermediate / cigar / edge-on
   shapes), roundness ladders, profile steepness, ridge-texture ratios
   (arms vs smooth), thin-ridge vs thick-ridge contrasts (edge-on disc, bulge),
   colour comparisons, lopsidedness and pair-imbalance (mergers/disturbed),
   centroid offset.
6. **Readout.** A final zero-bias dense layer is a hand-written rule table over
   indicators + raw measurements (zero bias keeps the decision invariant to a
   global rescaling of all indicators).

## How the weights were produced

Every number comes from code in `build.py`: Gaussian/derivative filter formulas,
ring/sector/aperture mask formulas, and hand-set thresholds and rule weights.
No gradient descent, no fitting, no statistics of the training images, and no
training pixels are stored anywhere in the weights. Training images were viewed
(as PNG montages) only to understand the classes. The indicator thresholds and
rule-table weights were calibrated by running the network on **synthetic phantom
galaxies painted from analytic morphology models** (Sersic ellipsoids of several
axis ratios, exponential discs with logarithmic-spiral arms, bars, edge-on discs
with/without a bulge, merger pairs, tidal plumes - i.e. the "shape models typed
from general knowledge" the rules allow), at several sizes, angles and
brightnesses; no survey data enter that calibration. Two logged checks on the
training set were used to confirm the network runs and to sanity-check the
confusion pattern.
