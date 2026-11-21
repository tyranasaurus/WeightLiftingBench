# Method

## Architecture
A small, fully hand-specified network (5,998 weights). All convolutions are fixed image
filters; the only "decision" weights are three dense readouts built from a hand-written rule
table. Branches:

- `opp` (1x1 conv): four colour-opponent channels from RGB — luminance `(R+G+B)/3`, and
  red/green/blue opponents `X-(sum of other two)/2`.
- `cpool` = avgpool of `opp` to a 6x6 grid → coarse "what colour is where".
- `oppr` (1x1 conv) → relu → `spool`: the six signed opponents rectified then pooled, so that
  summing pairs gives |red|+|grn|+|blu|, i.e. a colourfulness (saturation) map.
- `edge` (3x3 conv) → relu → `tpool`: Sobel-x, Sobel-y and Laplacian stencils applied to
  luminance, as signed pairs, so rectified sums give edge/texture energy.
- `scoreC`, `scoreS`, `scoreT`: dense layers turning the three pooled maps into (10,) class
  scores; `add` sums them.

## How the weights were produced (no training, no fitting)
The convolution weights are textbook constants (opponent-colour combinations, Sobel and
Laplacian kernels). The classifier is a hand-written rule table `R` (10 classes x 9
interpretable cues: brightness top/bottom, greenness overall and at the bottom, blueness
top/bottom, warm colour in the centre, colourfulness, texture energy). Each entry is my own
small integer in {-2..2} expressing general knowledge of these everyday objects — e.g. planes
and ships sit in bright blue sky (bluT=+2), frogs are dark, warm, colourful and bumpy, deer are
very green with no blue below, cars and trucks are grey and edgy with little green or blue.

The table is turned into weights in a self-balancing way so no class or cue can dominate by
construction:
1. each cue is centred across the ten classes (zero-sum), so a cue cannot bias the vote;
2. cues are standardised by a general-knowledge neutral point and spread (gray-world prior:
   colour opponents 0, luminance 0.5; round typical levels for colourfulness and texture) —
   these are priors about natural photos, **not** measured from the data;
3. each class template is normalised to unit length, so a class cannot win merely by having
   larger numbers.
The score is then `score_c = unit_template_c · standardised_cues`, a balanced nearest-direction
(matched-filter) classifier. `build.py` contains every number and reads no files.

## Use of the training data (within the rules)
I viewed montages of each class and printed per-class qualitative summaries only to understand
the data (allowed by rule 4). These confirmed general-knowledge cue directions and corrected a
couple of my priors (e.g. that car/truck/urban scenes carry *more* edge texture than
sky-background objects, and that CIFAR frogs read more warm/colourful than vividly green). No
measured number was copied into any weight; magnitudes are my own hand-chosen integers and
generic priors. Of the five allowed checks I used three to confirm the network runs and is
balanced (14% collapsed → 22.7% balanced), and kept the final design on reasoning rather than
tuning to the check scores.
