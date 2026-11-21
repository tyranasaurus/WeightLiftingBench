# Hand-designed galaxy classifier

## Architecture

The network computes rotation-invariant cosine similarities between a normalised,
downsampled image and 80 hand-designed templates (10 classes × 8 orientations):

```
image (3, 128, 128)
  --conv 1x1 2 out--> (luma, redness) channels
  --avgpool 4x4--> (2, 32, 32)
  --norm--> each channel standardised to mean 0, variance 1
  --dense out=80--> template cosine-similarity scores
  --reshape (10, 1, 8)--maxpool 1x8--> max over 8 rotations
  --reshape (10,)--> class scores
```

- The 1×1 colour conv has fixed weights: channel 0 is the standard
  Rec. 601 luma (0.299, 0.587, 0.114), channel 1 is `R − B` to capture
  "redness" (old-star populations are red, young star-forming regions are blue).
- Avgpool reduces to a 32×32 grid, which is coarse enough for hand-designed
  shapes to be drawn at integer-pixel scales and fine enough to preserve the
  galaxy shape information.
- `norm` removes image-to-image brightness differences so template inner
  products are directly comparable.
- Each template is a 2-channel 32×32 pattern (concatenated into 2048 weights)
  with zero mean per channel and L2 unit norm, so the output is a cosine
  similarity.
- Reshape + maxpool makes the per-class score the maximum response across the
  8 orientation templates, giving rotation invariance.

## How the weights are produced

All weights come from explicit formulas inside `build.py`; no training image is
ever read, and no statistic is taken from labels. Templates are drawn in a
rotated (xp, yp) coordinate system using Gaussian ellipses, offset Gaussians,
and sums/differences of these. For each class I used:

- Round smooth (2): compact circular Gaussian + red colour.
- In-between round smooth (3): mildly elliptical Gaussian + red.
- Cigar (4): elongated smooth Gaussian, mildly red.
- Barred spiral (5): short central bar + two offset "arm knot" Gaussians.
- Unbarred tight spiral (6): compact bulge + nearby disk ring, red bulge.
- Unbarred loose spiral (7): extended disk + strong blue preference.
- Edge-on without bulge (8): long thin Gaussian ridge with a central dip.
- Edge-on with bulge (9): long thin ridge + modest red central bulge.
- Merging (1): two offset Gaussian peaks with a negative central gap.
- Disturbed (0): one core + one off-centre lobe + blue.

Each template is zero-meaned per channel so it ignores the DC level that `norm`
would have removed anyway, then L2 normalised so all 80 templates produce
responses on the same scale and the per-class max pools pick the best-fitting
orientation. The classifier is purely the argmax of these scores.
