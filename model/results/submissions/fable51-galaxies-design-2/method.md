# Method

## Architecture (407 layers, 23.7M weights, 1.4M nonzero)

The network is a hand-built image-analysis pipeline expressed with conv / dense / relu / pooling / add / reshape:

1. **Sky subtraction and smoothing.** A 1x1 conv forms the luminance (R+G+B)/3 minus the sky level 0.125 (the JPEG cutouts
   have a near-black but nonzero background), followed by a 3x3 box smoothing and a ReLU.
2. **Geodesic reconstruction from the centre.** Fifty iterations of "dilate by a 3x3 max (1x1 conv with padding + maxpool),
   then clip by the image (min implemented as a - relu(a - b))", starting from a seed disk at the image centre. This
   grayscale morphological reconstruction keeps only light connected to the central galaxy and flattens every secondary
   peak. The difference image - smoothed - reconstructed - is the "dome" map of secondary peaks (companion nuclei, stars,
   HII clumps).
3. **Measurements as formula templates.** Soft isophote masks are cut at four levels (0.08, 0.25, 0.50, 0.75 above sky).
   Dense layers whose weights are closed-form coordinate functions compute, per level, the area, the centroid sums and
   the second moment along 16 directions; a reshape + maxpool over the direction axis gives the largest and smallest
   moments, i.e. the axis ratio. Other dense templates give total and central flux, red and blue sums over two annuli
   (colour), the highest dome within 28 px of the centre and its extent (second nucleus vs point-like star), a capped dome
   sum (clumpiness), the outer-isophote centroid offset, and a 180-degree rotational asymmetry (a permutation dense layer
   at 64x64, minimised over 3x3 centre shifts).
4. **Decision tree in ReLUs.** Each test "a > k*b" becomes a clipped ramp (two ReLUs); ANDs are relu(sum - n + 1), ORs are
   clipped sums, ladders of thresholds turn colour, size, bulge fraction and "inner isophote more elongated / rounder than
   outer" into graded evidence. The logic follows the Galaxy Zoo tree: very flat outer isophote -> edge-on, split into
   with/without bulge by a round inner isophote, a large round saturated core and red colour; an extended second nucleus
   inside the envelope -> merging; otherwise a colour/size/bulge-fraction "featuredness" axis separates smooth (round vs
   in-between by axis ratio) from featured (barred if an inner isophote is much flatter than the outer disk; tight vs loose
   by bulge fraction, colour and clumpiness).

## How the weights were produced

All weights are written by `build.py` from formulas and hand-chosen constants: coordinate polynomials, Gaussian kernels,
permutation matrices, identity/shift kernels, and threshold constants chosen from general knowledge of galaxy morphology
(e.g. edge-on disks have axis ratio below ~0.35, ellipticals are red with nested similar isophotes, bars make the inner
isophote flatter than the disk, merger companions sit within ~25 px). No training image, statistic of the training set or
fitted quantity enters any weight. The training set was used only to look at example images of each class and to run the
five allowed checks; after each check the confusion table guided hand adjustments of thresholds and rule weights
(e.g. raising the envelope level for dome detection above the noise, requiring a large inner region before calling a
round core a bulge).

Checks on 3,000 training images: 40.9% -> 45.8% -> 48.9% -> 47.6% -> 49.3% accuracy. The final change after the last
check (down-weighting non-specific bulge cues for edge-on galaxies) was made on reasoning alone and is unverified.
