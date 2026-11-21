# Method

## Architecture

A hand-designed, fully interpretable "person-parts" detector, 44 layers, 169,246 weights.

1. **Stage 1** (`s1`, conv 3x3 stride 2 + relu) turns the RGB photo into 11 rectified
   cue channels at 48x48, all from formulas: a skin-excess channel
   `relu(blur(R-2G+B)-0.03)` (skin is the region where R-G > G-B), warm/cool/green
   colour channels, bright/dark luminance channels, four half-wave luminance
   gradients, and an over-saturated-red channel `relu(blur(R-G)-0.35)`.
2. **Two branches per pyramid level** (24x24, 12x12, 6x6 by avgpool): the colour
   channels stay in physical units (so "absent" stays exactly zero, and the skin
   channel is cleaned by `-3 * oversat` in a 1x1 conv, which removes tomatoes,
   red flowers and red shirts whose R-G is far larger than any skin tone); the
   structure channels (bright, dark, vertical-edge energy, horizontal-edge energy)
   are contrast-normalised per image with `norm` so the detector is exposure
   independent. The branches are padded and added into one 8-channel tensor.
3. **Template bank** (`L*t`, one conv per level, stride 2): 24 hand-drawn part
   templates - frontal face (skin ellipse + dark hair cap + dark eye band +
   bright cheeks), head-and-shoulders, narrow upright body, compact skin blob
   with non-skin surround, dark upright figure, and a pair of legs - each at four
   head diameters per level, covering person heights from ~8 px to full frame.
   Templates are drawn in per-pixel SNR units (expected activation divided by
   channel noise: skin 6, contour/dark 3, bright 1.5) and converted to
   likelihood-ratio matched filters `w = x_exp/sigma^2`, normalised so a perfect
   match scores 1.0; a match missing one cue scores only that cue's share of the
   template energy, so skin-less objects cannot fire the skin-carrying templates.
4. **Evidence accumulation**: each template score is soft-thresholded
   (`relu(score - t)`, t = 0.72/0.65/0.58 for small/medium/large heads, hand-set),
   max-pooled over position, and summed over scales and levels (`add`).
5. **Readout**: a hand-set linear rule table (`ro`, dense) weights the six
   evidence totals (face 3.0, head-shoulders 2.6, narrow body 1.2, skin blob 1.2,
   dark figure 0.5, legs 0.4, bias -0.55), plus a small global-colour context term
   (`g`: frame-filling warm colour as in food/wood close-ups and frame-filling
   green argue against a person subject). The two partial score vectors are
   added into the final (2,) logits.

## How the weights were produced

Every number comes from code in `build.py`: Gaussian/Prewitt filter formulas,
geometric shape models of people typed from general knowledge (ellipse faces,
hair caps, eye bands, shoulder lines, upright torsos, leg pairs), the hand-set
SNR/threshold/importance tables quoted above, and the matched-filter
normalisation. No training image, label, statistic of them, or generated-image
measurement enters any weight. The training set was used only to look at example
photos of each class and for the five allowed accuracy checks; those checks
verified the pipeline works and exposed two design bugs (colour cues carrying no
energy in a naive matched filter, and over-generic structure templates), which
were corrected by re-deriving the filter math, not by fitting. Final verified
training accuracy: 0.566 (check 3); the submitted variant adds only the
over-saturation-suppressed skin channel to that configuration.
