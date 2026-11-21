# Method

**Architecture (686 weights, all typed by hand in `build.py`; no training, no data read).**

1. `v`, `m` — two 1x1 convs with relu turn RGB into two soft colour maps in [0, 1]. Each is
   `relu(1 - 10 * sum of rule violations)`:
   - *skin*: R > G >= B, R bright enough, hue between red and orange, saturation between 0.15 and 0.70
     (the usual published RGB skin rules written as linear inequalities);
   - *sky*: clearly blue and bright.
2. `pA`, `pB`, `pC` — an average-pool pyramid of those maps with cells of 2, 4 and 8 pixels.
3. `bA`, `bB`, `bC` — at each level, two 7x7 centre-surround "face blob" templates applied to the skin map:
   a Gaussian centre (slightly taller than wide, like a face) minus a ring over the top and both sides
   (hair and background should not be skin; the row underneath is left free for neck and arms).
   The response lies in [-1, 1]; `relu(response - 0.50)` followed by a global max gives the strongest isolated
   skin blob for face widths from about 4 to 36 pixels.
4. `g`, `gc` — global fractions of skin-coloured and sky-coloured pixels, thresholded at 0.25 and 0.40.
5. `out` — person score = blob evidence (finest level at half weight) − 3·(excess skin fraction)
   − 1.5·(excess sky fraction) − 0.20; no-person score = 0. A photo that is mostly skin-coloured is more
   likely wood, food or sand than a person, and a photo that is mostly sky rarely has a large person.

**How the numbers were chosen.** Colour rules come from general knowledge of skin and sky colour, template
shapes from face proportions, and thresholds were picked by hand. Nothing is computed from the training images.

**Checks used (2 of 5).**
- Check 1 (4,000 images, which turned out to be all "no person" because the file is sorted by label):
  86% were called "person" — the blob threshold was far too permissive.
- Check 2 (2,000 per class, after raising the blob threshold from 0.35 to 0.50, the decision margin from 0.15 to
  0.20 and halving the finest level): 56.9% accuracy, 63% of "no person" and 51% of "person" correct.

**Limitations.** This is a single-cue detector (isolated skin-coloured blobs) with two colour-context penalties.
I ran out of time before adding the body-shape and scene cues I had planned, so it misses people with no visible
skin, black-and-white photos, and very small figures, and it is fooled by warm-coloured food, wood and fur.
