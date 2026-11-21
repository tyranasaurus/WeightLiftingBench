# Method

Hand-designed template-matching network for the 24 static ASL letters; no training, no fitting, no
statistics of the training images are used anywhere. `build.py` regenerates every weight from code alone.

## Architecture (network.json)

1. `norm` on the image (removes brightness / contrast differences).
2. `c1` 7x7 conv: derivative-of-Gaussian edge filters at 4 orientations, each as +/- pairs with two biases so
   that, after ReLU, the next layer can form the unsigned, thresholded and capped edge magnitude
   `clip(|e| - T, 0, C)` (polarity-invariant: the hand may be lighter or darker than the wall), plus a
   Gaussian-blurred copy of the image (+/-).
3. `c2` 13x13 conv: unsigned edges, a coarse edge-energy ("textured region") map, and bar (ridge) and
   centre-surround (hole) detectors at 4 orientations / 2 widths applied to the blurred image, again as
   +/- pairs with threshold and cap biases.
4. `c3` 1x1 conv combines the pairs into 15 feature channels; `avgpool` 2x2 to 14x14; `norm` standardises every
   channel per image, so matching compares patterns rather than energy levels.
5. `tm` 12x12 conv = a bank of 648 templates (24 letters x 3 pose variants x 3 scales x 3 rotations), evaluated at
   5x5 shifts; `maxpool` over shifts, `reshape`, `maxpool` over each letter's variants, `reshape` to 24 scores.

## Where the weights come from

* Filters: closed-form formulas (Gaussian, Gaussian derivatives, bar and ring profiles).
* Templates: a parametric hand model typed from general knowledge of the manual alphabet (shared fist / palm
  and forearm for all upright letters, fingers as capsules at given positions and angles, thumb, curled-finger
  bumps, crease lines; C as a thick arc; O as an ellipse with a hole; G/H sideways; P/Q hanging). Each pose is
  scaled so the hand spans most of the frame (the photos are cropped around the hand), rendered with
  anti-aliasing, passed through the same fixed filters, and the standardised feature maps (per feature group
  normalised to unit norm times a hand-chosen group weight) become the template conv weights. The edge-energy
  template is a hedge between the outline energy of the clean render and the blurred filled hand region
  (real skin is textured all over).

## Checks

Five runs on the training set were used (logged in checks.log): 17.3% (raw-intensity silhouettes),
14.8% (polarity-invariant edges without per-channel standardisation), 11.6% (standardised features but
templates too small), 19.9% (this design, after scaling the hand model to fill the frame), and a fifth check
was not run. The submitted network is the 19.9% design; its main remaining weakness is that the "fist + one
finger" (D) template wins too often for other upright letters.
