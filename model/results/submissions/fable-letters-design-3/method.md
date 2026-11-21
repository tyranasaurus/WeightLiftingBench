# Method

## Architecture
A fixed, hand-designed template matcher (about 18.7 M weights, nothing trained or fitted):

1. `norm` of the image (per-image z-score).
2. **Edge map**: 12 Gaussian-derivative filters (30 degrees apart, sigma 1 px) -> relu -> a 1x1 conv that
   sharpens orientation tuning (each direction minus sqrt(3) times its two perpendicular directions, minus a
   small noise floor) -> relu -> a three-piece compressive curve (two more relu knees) -> a conv that mixes in
   40 % of the opposite polarity, blurs (two Gaussians, 1.3 and 2.6 px) and subsamples to 12x12 -> joint
   z-score of the whole 12x12x12 tensor (reshape + `norm`).
3. **Figure map**: a dense layer subtracts the plane through the picture's upper border (closed form in pixel
   coordinates, i.e. "the wall") -> |residual| with a dead zone and a ceiling (relu pairs) -> blur and
   subsample to 12x12 -> `norm`.
4. The two maps are stacked as 13 channels and matched by one conv layer holding 9,600 templates
   (24 letters x 400 drawings), kernel 12x12 with padding 2, so each template is tried at 5x5 shifts
   (+-4 px). Max-pool over shifts, reshape to (1, 24, 400), max-pool over the 400 drawings of each letter
   gives the 24 class scores.

## Where the weights come from
- Front-end weights are formulas (Gaussian derivatives, orientation sharpening rule, hand-chosen knees,
  Gaussian blurs, the border-plane projection).
- Templates are drawings made by `build.py` of a schematic right hand seen from the front (palm box, ball
  of the thumb, finger capsules, curled-finger lumps, thumb, forearm), posed by hand for each letter from
  what the manual alphabet looks like (e.g. G/H/T/X side views, P/Q pointing down), with fixed-seed jitter
  of finger angles, lengths, widths, tilt, size and shading (light hand on dark wall or shaded hand on
  light wall, dark creases where parts touch). Each drawing is passed through the same front-end and
  becomes one template, zero-meaned and normalised by itself alone, with a weight map from that drawing's
  own outline (full weight from 2 px inside the outline to 5 px outside, 0.3 deep inside and far outside,
  lower rows down-weighted). No statistic over several drawings or over any training image sets a number.

## Checks on the training images (all 5 used; one sentence each)
1. 38.4 % - edge-only matching works at a basic level, but fists were absorbed by the O drawing and
   finger-up letters by K/B/R.
2. 37.3 % - adding crease shading and reworking drawings only moved the errors: the busiest drawing (Q)
   became the attractor.
3. 37.3 % - adding the figure/ground map and orientation opponency helped blob-like letters and hurt
   thin-finger letters, no net gain.
4. 36.3 % - outline-weighted templates and a corrected Q removed the Q attractor and kept fists inside
   the fist group, but letters with holes or long fingers lost ground.
5. 32.0 % - longer fingers, a tighter framing rule, G/H redrawn lying across and a harder figure-map
   ceiling made things worse (many letters fell into the A fist), so I did not keep these changes.

## What is submitted
The submitted `build.py`, `network.json` and `weights.npz` are exactly the design of check 4
(36.3 % on the training images, about 0.34 on the task's score scale). It is the best design I can
reproduce exactly; check 1 scored 2 points higher but I had not kept that version of the builder.
The fist letters (A, E, M, N, S) remain largely confused with each other, and I have no evidence that this
network beats either comparison answer; a trained network of the same shape would very likely do better.
