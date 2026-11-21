# Method

**Architecture (211 layers, 22.7M weights, almost all of them fixed geometric templates).** The input is 2×2-averaged to 64×64. 1×1 convs form luminance L and colour D = R−B. The sky level is the minimum of 16×16 block means (avgpool, then a negated maxpool), and a constant-weight dense layer broadcasts it back so it can be subtracted. Relu ramps cut isophote masks at four display levels above the sky (0.05, 0.18, 0.38, 0.62). Each mask is restricted to the part connected to the image centre by a flood fill: repeated dilation (pad conv + maxpool) followed by a fuzzy AND with the mask (add + relu). Levels 0–2 are flooded on a 32×32 grid and upsampled with a nearest-neighbour dense matrix; the bright core (level 3) is flooded on the 64×64 grid.

From these masks, dense rows compute:
- the area;
- the centroid offset;
- directional second moments (x cos a + y sin a)² over 16 directions, where a maxpool gives λmax and λmin, so the axis ratio at each level is q = √(λmin/λmax).

A log-polar resampling (24 rings × 32 angles, Gaussian anti-aliasing) of the masked luminance feeds 2-D Fourier templates cos(mθ + p ln r − α), for m = 1–3, 13 radial wavenumbers p and 16 phases. A maxpool over phase gives the amplitudes |A(m,p)|. The chirality asymmetry |A(m,p)| − |A(m,−p)| is exactly zero for any mirror-symmetric object (ellipse, edge-on disk, bar, star), so it measures spiral arms. Where in p the asymmetric power sits gives the pitch angle, which separates tight from loose arms.

Further features are:
- colour inside the unsaturated level-1 region;
- companion cores (level-3 regions not connected to the central core);
- m=1 lopsidedness and odd-harmonic irregularity;
- area ratios between levels (concentration, bulge, disk-like versus de Vaucouleurs profile).

Ratios are computed with the `norm` layer: a channel holding [a, −a, b, −b] normalises to √2·t/√(1+t²) with t = a/b. A hand-written fuzzy rule head (relu ramps, AND units, an additive class-evidence table) turns the features into the 10 class scores.

**How the weights were produced.** Every weight is written by build.py from formulas and hand-typed constants; nothing was fitted.
- **Geometry:** the centre is (63.5, 63.5) from the cropping geometry, checked once against the mean-image peak.
- **Display stretch:** I identified the DESI viewer stretch as the legacy `sdss_rgb` asinh mapping, which reproduces the observed sky level of about 0.127. In this stretch, isophote levels are surface-brightness levels.
- **Colour thresholds:** computed from typical red-sequence and blue-cloud g−r / r−z colours pushed through that stretch.
- **Axis-ratio boundaries:** taken from the Galaxy Zoo definitions of round, in-between, cigar and edge-on.
- **Profile cue:** derived from exponential versus r^¼ profiles under the asinh stretch.

**Testing.** I unit-tested the feature extractors only on synthetic galaxies generated from formulas: Sérsic bulges, exponential disks, log-spiral arms, bars, PSF, noise and the asinh stretch. From the training data I only viewed image montages and printed simple summaries: pixel percentiles, the sky level and noise in the image corners, and the location of the mean-image centre. I used the 5 allowed labelled checks on the full training set to read the confusion tables, and adjusted rule thresholds and weights by hand between checks. Accuracy across the checks was 39.1%, 48.5%, 49.6%, 49.9% and 51.4%; the submitted network is the one from the last check.
