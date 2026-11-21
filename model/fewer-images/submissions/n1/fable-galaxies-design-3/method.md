# Method

## Architecture (82 layers, 100.9 M weight entries, about 1.2 M non-zero)

A hand-built measurement pipeline, written entirely in `build.py` from formulas and hand-typed tables. Nothing is trained or fitted.

1. **Polar resampling.** Luminance minus the sky level (the sky value follows from the viewer's rendering formula, `arcsinh(Q*m)/sqrt(Q)` with Q = 20, m = 0.03) is resampled by one dense layer of Gaussian blobs onto 56 radii x 72 directions about the frame centre. Red minus blue and a 2x2-binned luminance are resampled the same way for colour.
2. **Isophote masks.** Soft thresholds at 6 absolute surface-brightness levels and 3 levels relative to the central peak. Along each ray only the first unbroken run of bright samples is kept (a 1x1 conv with triangular weights across the radius channels), so field stars and galaxies not connected to the centre are ignored.
3. **Isophote radius per direction** for each level. Opposite directions are combined by min and max: the min ignores a neighbour on one side, and max minus min measures lopsidedness.
4. **Scalar measurements:**
   - semi-major and semi-minor axis per level (axis ratios, size, core length against total length);
   - mirror asymmetry of the stack of isophotes (twist from arms or bars);
   - an inner isophote flatter than the outer one (bar);
   - colour index of outskirts and core;
   - one-sided bright secondary blobs (companion nucleus);
   - central brightness;
   - radial bumps in linear flux present on both sides of the centre (arms, rings, bar ends), using a piecewise-linear sinh to undo the arcsinh stretch.
5. **Read-out.** Each measurement is a ratio, read as thermometer bits `[numerator - t * denominator > 0]`. Class scores are sums of hand-typed log-probability tables over those bits plus class frequencies. "Merging" and "disturbed" are scored as the best ordinary host class (max-pool over host types) plus evidence from the interaction measurements.

## Where the numbers come from

- Geometry: the polar grid, ring areas and smoothing kernels are formulas.
- Photometry: the sky level and the flux linearisation come from the viewer's stretch formula. Colour bins come from typical g-r and r-z colours of red-sequence and star-forming galaxies pushed through the viewer's band scales (g 6.0, r 3.4, z 2.2, recalled from memory).
- Thresholds, bins, table probabilities and feature weights are typed by hand from general knowledge of galaxy morphology and Galaxy Zoo class definitions.

## What I looked at and ran

- **Training images:** viewed the 10 images as PNGs. Printed three pooled summaries over all 10: channel percentiles, a sky scatter, and the mean centroid offset. These confirmed the rendering formula's sky level and that galaxies sit at pixel 63.5. The pooled sky scatter (about 0.02) informed my sense of the noise when I chose the faintest threshold and the bump noise guard by hand; no measured value is copied into a weight.
- **Synthetic scenes** (Sersic profiles, spirals, pairs, rendered with the same stretch): used as unit tests that each measurement computes what it is specified to. Two design changes followed from those tests:
  - the secondary-blob measure was made one-sided after it fired on symmetric arms;
  - the host-mixture scoring for merging/disturbed was introduced after a synthetic pair was scored as a round galaxy.

  I regard both as fixes to make components match their specification, but they were prompted by synthetic runs, so I am stating it plainly. No threshold or table value was tuned on synthetic images.
- **Checks on training images:** 1 of the 5 was used, giving 4/10 (merging, round, in-between and edge-on without bulge correct). It showed that small regular face-on spirals were called round smooth because several weak "looks regular" tables outvoted colour.

## State of the submitted network

After that check I revised the tables by reasoning and added the radial-bump detector. **This final version has not been run on the training images**, so its accuracy is unmeasured; the only real-data figure is the 4/10 of the earlier version.

One known weakness: in the last synthetic unit test the bump detector responded only weakly (about 0.002 of the flux) to a strong synthetic two-armed spiral, below its first table bin, so it may contribute little as built. I ran out of time before diagnosing whether that is a stencil or threshold problem.
