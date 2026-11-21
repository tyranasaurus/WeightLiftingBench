# Method

**Architecture** (5.9M weights): `conv 1x1 -> relu -> dense(120) -> relu -> dense(40) -> relu -> dense(10)`.

- `c0` + relu: sky-subtracted luminance, red and blue channels (`relu(channel - 0.16)`, a hand-chosen level above the near-black sky).
- `d1`: centre-windowed (Gaussian) moment functionals, each written as a comparison `a - t*b`, so a ratio-above-threshold test is a single linear unit:
  - elongation `(1-q^2)/(1+q^2)` from second moments, 12 orientations x 6 thresholds;
  - centroid offset from the frame centre, 12 directions x 2 thresholds (double nuclei / mergers);
  - colour `(red - blue) / luminance`, 4 thresholds;
  - flux-weighted mean radius, 5 thresholds;
  - central surface brightness, 3 thresholds.
- `d2`: sums the orientation units (an OR over angles) and forms pairs `(s, s-1)` so `relu(s) - relu(s-1)` is a 0-to-1 step.
- `d3`: an additive rule table typed by hand (one score per class per feature bin) plus the log class frequencies of the public Galaxy10 DECaLS catalogue.

**How the weights were produced**: all from formulas (Gaussian windows, moment polynomials, angle banks) and thresholds / rule tables chosen by hand from general knowledge of galaxy morphology (axis ratios of round / in-between / cigar / edge-on, red smooth galaxies vs blue disks, compact vs extended, bright bulge vs none). No training, fitting or statistics of the training images; `build.py` reads no files.

**Checks used: 1 of 5.** Check 1 (1,000 training images, 100 per class): accuracy 0.311. What it showed: the pipeline runs and the elongation and offset features work (merging 53%, edge-on with bulge 66%, barred 65%), but the barred-spiral class is over-predicted and absorbs many smooth galaxies, edge-on without bulge is mostly called edge-on with bulge or loose spiral, and disturbed and cigar are almost never predicted. Weighting the per-class recalls by the catalogue class frequencies gives roughly 0.37 expected accuracy on a naturally distributed test set.

**Status**: saved as it stood when time ran out; the rule table was not revised after check 1.
