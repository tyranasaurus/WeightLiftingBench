# Method

## Architecture
`image -> e1 (conv 18x5x5, pad 2) -> relu -> 3-level maxpool pyramid (13x13, 6x6, 3x3)
-> per level: dense (10*M prototype scores) -> reshape (10,M,1) -> maxpool over M -> reshape (10,)
-> add the three level scores -> (10,) class scores.`

* `e1` filters are pure formulas: 8 orientations x 2 phases of Gabor filters
  (cos/sin, lambda=4.6, sigma=1.3, DC removed, unit norm) plus two Gaussian blur
  channels (sigma=1,2) — an oriented-edge + ink-density front end.
* Overlapping 3x3-stride-2 pooling gives +-1 cell translation tolerance; the three
  pool levels form a spatial pyramid (fine detail / layout / global ink).
* Each dense row is one *prototype*: max over M prototypes per class implements
  nearest-prototype matching, and the three levels are summed with hand-chosen
  weights (0.75, 1.0, 0.75).

## How the weights were produced (no training data)
1. **Typed pen-path models.** Every digit class is described by 4-5 handwriting
   forms typed as polylines of control points (strokes: oval, flag+stem, loop+tail,
   bar+stem, ...), i.e. from general knowledge of how digits are written.
2. **Synthetic rendering.** Each form is rendered by Catmull-Rom resampling of the
   strokes and laying amplitude-normalised Gaussian blobs along them (peak 1,
   sigma ~0.9 px), with fixed-seed sinusoidal pen wobble and mid-stroke taper, then
   normalised exactly like the task description: scaled to a 20x20 box, centre of
   mass at the frame centre. Style variation: 3 slants x 3 stroke widths x
   2 aspects x 5 sub-pixel shifts = 450 prototypes per class.
3. **Readout from prototypes.** The same fixed front end (conv+relu+pools, computed
   in `build.py`) is applied to the synthetic prototypes. Per pyramid level, with
   mu = mean prototype feature, each row is the centred prototype vector
   Q = F_prot - mu (unit-normalised) and the bias is -<mu, Q>, so a row's score is
   the centred cosine similarity between the input's features and that prototype;
   max over prototypes + sum over levels gives the class score.
   No statistic of the training images or labels enters any weight; the training
   set was used only for viewing images/summaries and the 5 allowed accuracy checks.

## Checks (what each taught)
1. 47.5% - scoring algebra bug (per-prototype 1/||Q|| factor) made sparse classes win.
2. 49.1% - Euclidean version still poor: renders' strokes were ~2x too thick (blob amplitude accumulated along the path).
3. 58.7% - amplitude-normalised strokes + rightward slant + centred-cosine scoring closed most of the sim-to-real gap.
4. 63.9% - overlapping pooling (shift tolerance), aspect variation and repaired 7/9/4 forms helped most.
5. 62.9% - added linear hole/loop detector channels did not help (synthetic closed loops fire stronger than real open ones), so they were removed in the final build; final build instead widens style coverage (more forms, 3 widths, pen wobble/taper) and rebalances the pyramid.
