# Method

**Architecture (283 layers, 46.8M weight entries, ~0.77M non-zero).** The network does a classical
morphology pipeline using only the allowed ops:

1. **Space-to-depth input.** A stride-2 conv turns the 128x128 image into 4 full-resolution luminance
   phases plus 2x2-averaged R, G, B (all at 64x64). The sky is estimated per image as the minimum
   16x16-pixel block mean (avgpool, then negate, then maxpool) and subtracted. Smoothing uses binomial
   kernels written in polyphase form.
2. **Isophote segmentation.** Six surface-brightness levels (0.05–0.62 above sky) give threshold maps.
   For each level, the component connected to the image centre is found by geodesic dilation:
   `X <- maxpool(pad(X)) + (T - 1)`, repeated 11–18 times. Full-resolution level masks are
   `relu(CC + T128 - 1)`.
3. **Measurements, all fixed linear maps of the masks and images.**
   - Moments of each level (area, x²+y², x²−y², 2xy) give the axis ratio, an ellipticity vector, and an
     ellipse-normalised second moment J (1 for a centred ellipse; larger for pairs and irregular shapes).
   - Outer and inner colour, as log₂(g/z).
   - Rotational asymmetry, from a fixed 180° pair-difference dense layer, taking the minimum over 5
     rotation centres.
   - A three-scale ridge field (oriented second derivatives). Its sums against cos/sin of 2(α−θ) give
     the radial/tangential balance and the net spiral handedness.
   - Bright light inside the segment that is not connected to the central core (companions), with an
     allowance for field stars.
   - Central brightness.
4. **Scalar maths.**
   - Ratios use `norm` on a reshaped (G, 1, 2m) layout holding `[a, −a, b, −b, …]`, which gives
     a/√(a²+b²+…) exactly.
   - log₂, axis-ratio conversion and squares are piecewise-linear relu functions.
   - Min, max and median use relu identities or reshape + maxpool.
   - Bar score: a bright isophote that is misaligned with, or more elongated than, the disk.
   - Edge-on bulge: bright isophotes rounder than the thinnest aligned outer isophote.
5. **Classifier.** There are 16 fuzzy predicates (hand-set ramps) and a hand-typed prototype table (the
   typical predicate values for each Galaxy Zoo class). The score is −Σ trust·(p − prototype)², which
   is linear in p because the trust weights are shared across classes. There is a rarity offset for
   cigar-shaped galaxies.

**How the weights were produced.** No fitting of any kind was used. Every number comes from a formula
or a hand choice in `build.py`:
- stencils and coordinate maps;
- permutation matrices for the 180° rotation;
- piecewise-linear function tables;
- thresholds and prototypes chosen from astrophysical knowledge. For example, I used the Legacy Survey
  viewer's asinh RGB mapping to predict that red-sequence galaxies give log₂(g/z) ≈ −0.25 and blue disks
  give ≈ +0.5. The same mapping gives a sky level of 0.127, which matches the data.

I studied the training images only by viewing PNG grids and by printing a few global pixel summaries.
To debug each feature I generated synthetic galaxies from formulas (Sérsic profiles, logarithmic-spiral
arms, bars, pairs, clumps, PSF, noise, asinh stretch) and ran the network on those, never on training
images.

**Checks.** I used the 5 labelled checks to see the confusion matrix and then revised thresholds and
prototypes by reasoning about the failure modes:
1. Stars and dust lanes were faking mergers, so I switched to per-level connected components.
2. Pairs were mimicking edge-ons with bulges, so I added alignment and asymmetry guards.
3. Per-class "don't care" entries created sink classes, so I went back to shared trust weights.
4. Measurement changes shifted the spiral/smooth balance, so I recentred the prototypes.

Checks 1–4 used one random half of the training set (accuracy 43.7%, 48.4%, 46.9%, 48.6%). Check 5
used the untouched other half and gave **50.1%**.
