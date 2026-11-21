# Method

**Architecture.** A hand-wired measurement network (about 230 layers, 28.6 M weight entries, almost all zero),
built only from conv / dense / relu / pool / add / reshape / norm. No weight was trained or fitted; `build.py`
generates every number from formulas and a table of hand-chosen constants.

1. *Luminance and noise.* Mean of the three channels minus the sky pedestal of the survey viewer's arcsinh
   mapping (0.125), binomial smoothing, and a per-image noise estimate (mean absolute high-pass) that can only
   raise thresholds.
2. *Which pixels belong to the central galaxy.* Grayscale morphological reconstruction at half resolution:
   28 rounds of `m <- min(X, maxpool3x3(m))` seeded in the central block. The result R(x) is the highest
   brightness level at which x is still connected to the image centre, so thresholding R gives the
   centre-connected isophote at any level and cuts away field stars and neighbouring galaxies.
3. *Isophotes.* Soft-threshold masks of the full-resolution image (space-to-depth layout, gated by R) at one
   absolute outer level and four levels relative to the central peak. A dense layer of wedge-integration
   weights (1/(2 pi r) with a triangular angular window) turns each mask into its radial extent along 64 rays,
   plus its area. From the rays: chord lengths, their max and min (axis ratio, by two estimators), lopsidedness,
   the m=2 direction, and a "fat middle" shape index for edge-on bulges.
4. *Texture, second nucleus, colour.* A difference-of-Gaussians ridge map counts arm/bar/thin-disc pixels inside
   the galaxy (nuclei and star-like peaks blanked out). The h-dome X - R measures a second nucleus by its height
   above the level where it joins the central galaxy, and its area tells a companion galaxy from a point source.
   Gated colour sums give a (blue - red)/(sum) index.
5. *Ratios.* A `norm` over a channel laid out as [x, -x, K*D, -K*D] returns x/D, which makes every feature
   independent of image brightness and galaxy size.
6. *Rules.* Three small relu stages form soft indicators (thin disc, round / in-between / cigar body, spiral
   evidence from colour + ridges + size, loose vs tight, bar = inner isophotes more elongated than the outer one,
   bulge, second nucleus, lopsidedness). Each class score is minus the sum of penalties for the conditions it
   fails, plus a small bias.

**Status.** This is the first complete version. It was run once on the training set (check 1 of the 5 allowed):
42.1 % accuracy. No constant was changed after that check; the submitted weights are exactly the checked ones,
and `check_rebuild.py` confirms `build.py` reproduces them without any data. The check showed too many
"merging" and "tight spiral" predictions and almost no bars recognised; a reworked version was begun but not
finished or checked, so it is not submitted.

**Where the numbers come from.** Shape and threshold constants come from general knowledge: the viewer's colour
mapping, axis ratios that separate round / in-between / cigar / edge-on, typical colours of early- and late-type
galaxies, and Sersic / exponential-disc / point-source models that I rendered synthetically (`sim.py`, never
part of the weights) to verify that each measurement returns what it should (e.g. measured axis ratio equals
the true one; a star's dome area is 20-31 x its height, a companion's is 44 x or more). Training images were
only viewed, plus the five permitted checks of the complete network.
