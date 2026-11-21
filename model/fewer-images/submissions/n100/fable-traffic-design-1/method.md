# Method

## Architecture
A hand-built template matcher, written entirely by `build.py` (no training, no data read).

- **Colour path.** A 5x5 conv turns the image into blurred luminance, red-ness `R-(G+B)/2` and blue-ness
  `B-(R+G)/2`. Central crops (28x28, 22x22, and a 2x2-averaged 14x14) are normalised: luminance loses its
  mean, then all three channels are divided by their joint r.m.s. (reshape + `norm`, with the map stacked
  against its negative so the colour channels keep grey = 0).
- **Edge path.** Derivative-of-Gaussian filters in 8 directions on each of R, G, B, ReLU, a 1x1 conv that
  subtracts the two perpendicular directions (sharper direction tuning), ReLU, sum over colour, then Gaussian
  pooling at a fine (stride 1) and a coarse (stride 2) cell size, each jointly normalised with `norm`.
- **Templates.** Six conv layers hold the templates (817 each = 43 classes x 19 size/aspect variants):
  outer-part cues `w1` (coarse colour), `w2` (fine colour), `h1` (coarse edges) and inner-part cues `w3`
  (fine colour), `h2` (coarse edges), `h3` (fine edges). The convs slide every template over +-3 pixels.
- **Decision.** Cue scores are added per shift, max-pooled over shifts, then max-pooled over the 19 size
  variants of each class, giving the 43 class scores. 5.94 million weights in total.

## Where the weights come from
- Every filter is a formula (Gaussians, Gaussian derivatives, colour-opponent mixes, crop/selection deltas).
- The 43 signs are drawn in code from general knowledge of the German sign catalogue (rings, triangles,
  DIN-like digits as strokes, arrows, pictograms as simple shapes) with typed paint colours.
- Each drawing is resampled to one size variant, passed through the same colour/edge formulas, and becomes
  its own template, normalised by itself alone (own mean, own spread, own support). No statistic is taken
  across drawings, and nothing was fitted to or copied from the training images. Training images were only
  looked at to understand what the classes look like.

## Checks used (2 of the 5 allowed)
- **Check 1: 0.7777** (first version). Learned: the matcher works on real photos, but the empty-ring class
  swallowed faint speed limits, blue signs could fall into red classes, and several pictogram drawings
  (classes 11, 21, 23, 24, 27, 40) were poor likenesses.
- **Check 2: 0.7230** (this submitted version: absolute colour channels, separate outer/inner cues, redrawn
  pictograms and lighter digits). Learned: the revision was a net step backwards. Treating colour as absolute
  over-rewards the blue roundabout template (predicted 285 times out of 4300, pulling in end-of signs, speed
  limits and others), the slimmer pedestrian drawing is now almost always read as the exclamation mark, and
  the outer/inner split did not stop the empty ring taking faint speed limits. Blue classes and classes 11,
  21, 23, 40 did improve.

## State of the answer
The submitted files are the check-2 version (0.7230 on the 4300 training images; about 0.73 when classes are
weighted by the benchmark's usual class mix). The check-1 version scored higher, but I ran out of time before
I could restore or repair it, so the saved answer is not my best-measured one. Known weak spots: speed
limits 50 / 80 / 120, pedestrians (27), double bend (21), road narrows (24), and over-prediction of the
roundabout class (40).
