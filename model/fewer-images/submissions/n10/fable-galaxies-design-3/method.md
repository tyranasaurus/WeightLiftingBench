# Method

**Architecture.** A small feature network with no learned parts:

1. `c1` (conv 5x5, stride 2) blurs and halves the image to 64x64 and makes three maps after a relu: luminance above
   a fixed sky level, red excess (R-B) and blue excess (B-R).
2. Three dense layers read those maps with analytic, centred weight patterns:
   quadrupole moments `w(r) r^2 cos 2(phi - theta)` at 24 orientations (three Gaussian windows, five rings and the
   ring total), dipole moments at 12 directions (two windows), and scalar moments (aperture fluxes, second radial
   moments, colour sum). A max-pool over orientations turns the steered moments into amplitudes |Q| and |D|.
3. Each feature is a ratio a/b (ellipticity |Q|/R at three scales, lopsidedness |D|/flux, core concentration,
   compactness, colour, and position-angle twist = sum of ring |Q| minus |Q| of the ring total). The pair is written
   as `[a, b, -a, -b]` in one channel so that the `norm` layer returns `sqrt(2) a / sqrt(a^2 + b^2)`, a brightness-
   independent monotone function of the ratio.
4. A dense + relu layer forms soft steps (two hinges each) on those normalised ratios, and the output layer is a
   hand-written additive rule table (class weights per step) plus log class priors of the published Galaxy10 DECaLS
   catalogue.

**Where the numbers come from.** Filters are formulas (Gaussians, rings, angular harmonics). Thresholds are chosen by
hand from general knowledge: ellipticity `(1-q^2)/(1+q^2)` for the axis ratios that separate round / in-between /
cigar / edge-on, red old-star versus blue star-forming colours, bulges being concentrated, mergers being lopsided,
spiral arms and bars twisting the position angle with radius. No training, fitting, statistics of training images,
or tuning on generated images was used; the training images were only viewed as contact sheets.

**Status.** Time ran out before any refinement: the constants are first-principles guesses and were not iterated.
One labelled check was run (1 of the 5 allowed): 29% accuracy on the 100 training images. What it showed: the network
runs and is well above chance, but it over-predicts "in-between smooth" (41 of 100 predictions), never predicts
barred or tight spirals, and sends most edge-on-with-bulge galaxies to in-between, so the measured ellipticities are
lower and the spiral cues (colour, twist) weaker than the hand-set thresholds assume. Nothing was changed after the
check; the submitted weights are exactly what `build.py` wrote before it.
