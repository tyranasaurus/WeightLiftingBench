# Method

**Status:** the network builds, passes `check_network.py` and `check_rebuild.py`, and gives finite scores on two
generated test images. It has **not** been run on any training image (none of the 5 checks were used), so its
accuracy on real galaxies is unknown and every threshold below is an untested first guess.

## Architecture (16 layers, 5.6M weights)

1. `pre` (5x5 conv + relu): Gaussian blur (sigma 1 px), then four sky-subtracted maps: luminance above a faint
   isophote, luminance above a bright isophote, red channel and blue channel above the faint isophote.
   The sky level is not measured from data: it is the value the Legacy Survey viewer stretch gives to empty sky,
   asinh(20 x 0.03) / sqrt(20), about 0.127.
2. `featp`, `feats` (128x128 convs, one output position): moments about the image centre under a Gaussian window.
   `featp` projects the maps onto cos(m*phi - psi) for m = 1, 2, 4 at 16 phases psi; a maxpool over the phases
   gives the amplitude of each azimuthal Fourier mode. `feats` gives windowed fluxes (total, central, bright,
   red, blue).
3. `va` + `vs` -> `norm`: each group of features is laid out as [20 x flux, f1, f2, ...] followed by its negative,
   so the `norm` op divides every feature by the flux. This yields brightness-independent ratios:
   - m=2 amplitude / flux, which for any elliptically symmetric galaxy equals (1 - q) / (1 + q), q the axis ratio;
   - m=4 amplitude / flux (needle-like shapes), m=1 amplitude / flux (lopsidedness);
   - the same m=2 and m=1 ratios for the bright isophote only (bar, bulge, second nucleus);
   - central flux fraction (apparent size), bright flux fraction (core), red minus blue (colour).
4. `hid` (dense + relu) -> `out` (dense): eleven clipped ramps (soft thresholds) on those ratios, combined by a
   hand-written evidence table, one column per class.

## Where the numbers come from

All weights are written by formulas in `build.py`: Gaussian windows, cosines, and constants typed by hand.
Ellipticity thresholds come from the identity above with axis ratios of about 0.85, 0.5 and 0.3 for
round / in-between / cigar / edge-on. The other thresholds and the evidence table are hand-chosen from general
knowledge of the classes (ellipticals and bulges are red and concentrated, bulgeless edge-on discs and loose
spirals are blue and low surface brightness, mergers show a second bright nucleus off centre, bars make the
bright inner part more elongated than the outer disc). No training image, label or statistic of them was used
to set any weight; the training set was only viewed as PNG montages to see what the classes look like.
