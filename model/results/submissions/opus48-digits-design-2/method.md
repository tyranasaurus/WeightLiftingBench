# Method

## Architecture
A translation-tolerant template matcher, built from the allowed ops:

1. `blur` (conv, 1→1, 5×5): a fixed normalised Gaussian that smooths the input so
   thin strokes overlap the soft templates.
2. `nrm` (norm): per-image zero-mean / unit-variance. This makes every image have
   the same L2 norm, so the template dot products below are directly comparable
   cosine similarities across the ten classes.
3. `tmpl` (conv, 1→280, 28×28 kernel, padding 2): a bank of 280 digit templates
   (28 per class) used as matched filters. The padding of 2 lets each template slide
   ±2 px, so a digit that is a little off-centre can still find its best alignment.
   Output is (280, 5, 5): 280 templates × 25 shift positions.
4. `resh` → (10, 28, 25) groups the 25 shifts and 28 templates under each class.
5. `pool` (maxpool 28×25): for each class, the single best score over all its
   templates and all shifts.
6. `out` (reshape): the 10 class scores; the argmax is the prediction.

## How the weights were produced (no training, no data)
Every number comes from typed stroke paths and formulas encoding general knowledge
of how the ten digits are drawn — no training image, statistic, average or fit is
used anywhere.

- Each digit is defined as a few pen strokes (lines and elliptical arcs) in a
  canonical 28×28 frame, in two style variants (e.g. open/closed 4, plain/crossbar
  7, plain/flagged 1, curved/straight-tail 9). Strokes are rendered with a soft
  round pen (Gaussian cross-section).
- Each ideal glyph is put through the *same* preprocessing the task describes for
  the real digits: its ink bounding box is scaled to fit a 20×20 region, then it is
  shifted so its intensity centre of mass sits at the frame centre. (Discovering
  that the data is centre-of-mass centred, not bounding-box centred, was the single
  biggest accuracy gain.)
- Each template is an "ink-expected" map minus a penalty on the region inside the
  digit's box but away from its own strokes — this vetoes ink where a given digit
  should have none (e.g. the hollow centre of a 0, the empty sides of a 1), adding
  discrimination beyond plain correlation.
- The 28 templates per class are the combinations of 2 styles × 7 slants (shear) ×
  2 stroke widths, each flattened, zero-meaned and L2-normalised so the matches are
  comparable cosines.
- The Gaussian blur kernel is a closed-form formula.

`build.py` regenerates `network.json` and `weights.npz` deterministically from code
alone, reading no files.

## Checks
Five logged checks were used only to verify the design and catch systematic
alignment bugs (not to grid-search variants). Final measured accuracy: 0.695 on a
held-out 10,000-image sample of the training set.
