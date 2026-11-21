# Method

Hand-designed, untrained network for the 10 Galaxy10 DECaLS classes. Every weight is written by `build.py` from
formulas and hand-chosen constants; no training image, label statistic or fitted quantity enters the weights.

## Architecture (63 layers, 73.5M weights, 86k non-zero)

1. **Smoothing and colour maps** (conv 11x11 from the image): 3x3 binomial-smoothed luminance, red (z band) and
   blue (g band) with the sky background 0.13 subtracted; a Gaussian band-pass (sigma 1 minus sigma 3) for
   clumpiness; steep relu ramps that make a pixel mask at a mid isophote; 1x1 convs gate the band-pass and
   luminance maps by that mask (relu(x + m - 1)).
2. **Polar resampling** (dense layers whose weights are pixel/cell overlap areas, computed geometrically):
   32 sectors x 24 radial bins of 2 px for luminance, red and blue. A neighbouring star only disturbs one or
   two rays, which makes the shape measures far more robust than image moments.
3. **Isophotal ray extents** at four surface-brightness levels (0.06, 0.16, 0.30, 0.50 above sky) via relu ramps
   summed along each ray; major and minor axes as max and min over sectors (maxpool on a reshaped vector).
4. **Ratios without division**: the `norm` layer applied to (a, b, 0) triplets returns a monotonic function of
   a/b. This gives isophotal axis ratios at each level, outer and inner blue/red colour, 180-degree asymmetry
   (dense layer subtracting the point-reflected 64x64 map), clumpiness and concentration.
5. **Decision tree in relus**: soft threshold indicators (clipped ramps) combined with Lukasiewicz AND
   (relu(a+b-1)) into class scores: very elongated -> edge-on, split by a round bright inner isophote (bulge) and
   colour; strongly asymmetric -> merging; yellow and symmetric -> round / in-between / cigar by axis ratio;
   blue and symmetric -> barred (inner isophote more elongated than outer), tight (concentrated, yellow centre)
   or loose (clumpy, blue centre); mildly asymmetric blue -> disturbed. Small prior-based biases break ties.

Thresholds come from general galaxy knowledge (e.g. edge-on below axis ratio about 0.3, round above about 0.8,
elliptical galaxies redder than spiral disks) and from the rendering facts read off the data as whole-sample
summaries (sky level 0.13, pixel noise about 0.025). Time ran out before iterating on the training checks, so the
rule table is the first-pass design.

## Check result

One check on 3,000 random training images (logged in checks.log): accuracy 0.284 (score about 0.20 above
chance). Learned from it: the edge-on test works (bulgeless edge-ons 94% recalled), but the bulge rule rarely
fires so edge-on-with-bulge galaxies are mostly called bulgeless; small round galaxies measure as elliptical with
2 px polar cells, so round smooth is mostly called in-between; the asymmetry threshold is too high for
mergers; the concentration rule over-selects tight spirals. These thresholds were not retuned for lack of time.
