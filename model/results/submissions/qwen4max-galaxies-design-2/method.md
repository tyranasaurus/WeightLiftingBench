# Method

## Architecture

A hand-built morphological measuring instrument, not a learned feature hierarchy.
135 layers, 38.0M weight entries (270k nonzero), all generated from formulas.

1. **Photometry front end.** A 1x1 conv forms a luminance map and a `warm = R - B`
   colour map (the JPEG sky is grey, so `warm` is nearly sky-free). A ladder of
   Gaussian convs (sigma 1.0, 1.41, 2.06) gives a low-noise shape map; `norm`
   removes the sky offset and the per-image brightness scale. `|lum - blur|`
   gives a texture map.

2. **Cartesian measurements.** Dense layers with fixed radial masks and moment
   weights read out ring means, second moments over r<36 / r<10 / 10-36, a
   flux-weighted radius, colour, texture and an outer-annulus noise level.
   Sky is subtracted *in the measurement domain* (a linear combination of masked
   means), which keeps everything linear and cheap.

3. **Log-polar branch.** A dense layer resamples the smoothed image onto a fixed
   (rho = ln r, theta) grid, r = 3.5..50 px, 26 x 48 cells with 8 wrap columns,
   by bilinear interpolation averaged over 3x3 sub-cells. In these coordinates a
   change of galaxy size is a translation along rho and a rotation is a
   translation along theta, so the measurements below are scale- and
   orientation-free. From this map: a sky-subtracted radial profile; angular
   harmonics m=1..4 per radius (amplitude = max over 8 phase projections, done
   with `maxpool`); complex m=1,2,3 coefficients; and a 180-degree asymmetry
   |P(theta) - P(theta+pi)| built with a fixed circular-shift dense.

4. **ReLU arithmetic.** The op set has no multiply or divide, so both are built
   as piecewise-linear ReLU ladders on geometric knot grids: `1/x` (42 knots,
   ~0.5% relative error) and `x*y = (f(x+y) - f(x-y))/4` with `f ~ x^2`
   (36 knots, ~1% relative error). These turn the raw measurements into ~35
   dimensionless diagnostics: axis ratio from the moment quadrupole, inner/outer
   axis-ratio ratio (bar vs bulge), concentration, central dip (double nucleus),
   size, colour, clumpiness relative to noise, harmonic amplitude ratios,
   asymmetry, and the **spiral pitch angle** from the drift of the m=2 / m=3
   phase per unit log radius (a logarithmic spiral is a straight line in
   log-polar coordinates, so its pitch is a phase slope).

5. **Readout.** ~86 bounded trapezoid indicators (soft bins and soft steps, all
   in [0,1]) plus significance gates (is there outer disk light? is the harmonic
   phase signal coherent?), ~40 ReLU products forming soft ANDs, and a final
   dense layer whose weights are a rule table typed by hand.

## How the weights were produced

Every number comes from `build.py` itself: Gaussian kernels, the bilinear
log-polar grid, cos/sin harmonic weights, radial mask areas, the knot grids for
the arithmetic ladders, and the rule table. The rules encode general knowledge
of the classes: roundness separates the three smooth classes; extreme thinness
plus a4/a2 harmonic ratio selects edge-on; inner-vs-outer ellipticity separates
a bar (inner region *more* elongated than the outer disk) from an edge-on bulge
(inner region *rounder* than the outer disk); m=2 phase drift separates tight from loose spirals; lopsidedness
and outer asymmetry select disturbed; a central dip plus inner asymmetry selects
mergers. Priors are mild hand-set values (cigar-smooth is intrinsically rare).

No training image, patch, descriptor or label statistic was used to set any
weight, and no fitting of any kind was performed. Training images were only
viewed as PNG montages to understand what the classes look like, plus two
printed global summaries (centering and apparent size) used solely to choose the
polar grid's radial range. The readout was checked on synthetic archetype
feature vectors typed from general knowledge, not on images.

## Verification status

`check_network.py` reports the network usable (correct shapes, all weights
finite, within the size limit). The ReLU reciprocal/product ladders were verified
numerically as standalone formulas, and the indicator/rule arithmetic was
verified on hand-written synthetic archetypes (9 of 10 archetypes score their
intended class highest; the merging archetype lands on disturbed, its nearest
neighbour). **No `run_network.py` check was spent** - 0 of the 5 allowed image
runs were used, so the network has not been evaluated on the training images.
