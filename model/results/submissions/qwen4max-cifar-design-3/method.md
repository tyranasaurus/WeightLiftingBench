# Method

## Architecture (49 layers, 69,028 weights)

A hand-designed, fully feed-forward feature hierarchy followed by a hand-written rule table.
No weight was learned, fitted, or copied from data.

1. **`c1` — 5x5 filter bank on the photo (26 channels, 32x32).** Raw R/G/B, Gaussian-blurred
   colour at two scales (sigma 1 and 2), luminance, first Gaussian derivatives of luminance at
   four orientations and two scales, two Laplacians, and red-green / blue-yellow opponent edges.
   Every kernel is a closed-form formula (`gauss`, `dgauss`, `log` in `build.py`).
2. **`c2`/`r2`/`c2b`/`r2b`/`c2c`/`hc` — soft semantic categories at 16x16.** Each category
   (sky-blue, overcast/pale, teal water, deep blue, green / bright green / dark green, tan, brown,
   red, yellow, saturated blue, grey, white, dark, pink, smooth, rough, very rough, horizontal /
   vertical edge) is a conjunction of linear conditions on colour, luminance and local edge
   energy. Each condition becomes a hinge `clip(k*(form - threshold))` in [0,1]; the conjunction
   is `relu(sum hinges - (n-1))` (a Lukasiewicz t-norm). Clipping is done with an identity conv,
   bias -1, relu, negate, add. Oriented edge energies are `|d| = relu(d) + relu(-d)`.
3. **`cblob` — multi-scale blob detectors.** A 5x5 Laplacian of the broadly blurred luminance,
   sampled at 16x16, giving effective scales ~3.1 px (wheel-sized) and ~4.5 px (body/head-sized),
   with both signs (dark and bright blobs). Added into the semantic stack.
4. **`cshape` — coarse shape/layout detectors at 8x8**, written from general object structure:
   a pair of dark round blobs with a gap and a body above them (`wheels`, `wheels_w`), a horizon
   with smooth bright sky above and water / ground below (`horiz_w`, `horiz_g`), thin dark
   verticals hanging below a body (`legs`), and a textured centre against a smooth surround
   (`popout`).
5. **Pooling.** Semantic maps -> 4x4 layout grid (region fractions), 1x1 global mean, 1x1 global
   max; shape maps -> 4x4 and 1x1 max (shape evidence is sparse, so max not mean); `c1` -> global
   mean colour. 794 named features in total.
6. **Readout.** A linear rule table (`RULES`) plus 16 thresholded, clipped-to-[0,1] scene/object
   concept gates (`CONCEPTS` -> `gate`) feeding a second table (`CRULES`). The two are summed.
   Patterns like `teal@bot`, `white@up`, `brown@cen` name image regions; each pattern's weight is
   divided by the number of cells it covers so every feature contributes `w * mean response`.
   Each class's positive weights are scaled to a common total (14) and its negative weights to a
   common total (14), so no class wins merely by having more terms; concept gates likewise (6/6).

## How the weights were produced

Only from (a) closed-form formulas — Gaussians, Gaussian derivatives, Laplacians, annuli — and
(b) constants I chose by hand from general knowledge of what sky, sea, grass, fur, paint, metal
and road look like, and of where these ten objects appear in a photo. I looked at training images
as PNG montages to ground that knowledge. `build.py` reads no files; every number comes from its
own source. The only "normalisation" is the fixed positive/negative budget scaling described
above, which is a formula applied to my hand-written table, not a fit to data.

## Checks (5 runs, all through `/app/run_network.py`)

The log records four labelled runs; the fifth was an unlabelled run of a diagnostic variant that
emits ten internal feature values instead of class scores (saved to `last_scores.npy`). I count it
as one of the five. I used it only to find *structurally dead or saturated units*, not to read
thresholds off the data.

1. **17.07%** — airplane was never predicted and ship swallowed it: my "water" category fired on
   blue sky, because both are blue. Fixed by splitting sky from water on brightness.
2. **16.40%** — cat dominated everything. Two bugs found by inspection: the Laplacian blob signs
   were inverted (so "wheel" detectors looked for *bright* blobs), and the 4-hinge smoothness AND
   was too strict to ever fire, which deleted every class's inhibition.
3. **16.74%** — still cat-dominated, truck never predicted. Added proper multi-scale blob
   detectors (a 5x5 kernel truncates a sigma=2 Laplacian badly) and made the rule table
   position-aware.
4. **diagnostic** — showed `pale` and `white` were 99% zero, `gray` 78% zero, `rough` never zero,
   and the `fur` gate saturated on 43% of all images (handing cat its whole concept budget
   everywhere). Fixed the hinge plateaus, raised the texture thresholds, and made `fur` require
   fur colour plus the absence of sky/water/vegetation.
5. **18.66%** — airplane and ship now work, but truck was still never predicted.

**After the fifth run** I found the root cause by inspection, without running anything: the
symmetric "absolute value" conditions were written `sm(RG, -0.07)`, which means *RG < -0.07*, not
*RG > -0.07*. For any neutral pixel one hinge of each pair was therefore always 0, so the AND could
never reach its threshold — `white`, `pale` and `gray` were structurally dead, which is exactly what
the diagnostic showed and why truck (which depends on `white`) never fired. I corrected the
direction of those six hinges (`sm` -> `bg` with a negative threshold). This is a sign correction,
not tuning. **It is unverified**: I had no checks left, so the final accuracy is unknown; 18.66% is
the last measured value, from before this fix.

## Honest assessment

This is a knowledge-based classifier, not a learned one, and it shows: the last measured accuracy
(18.7%) is far below what gradient descent on the same architecture would reach. The hand-written
readout is the bottleneck — the categories are reasonable, but choosing ~700 readout weights by
reasoning alone, with only five looks at the results, left several classes poorly separated
(notably cat/dog and deer/horse). The final sign fix should recover the truck, airplane and ship
evidence that was silently missing, but I cannot claim a number for it.
