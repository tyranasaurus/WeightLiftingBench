# Method

## Architecture
A hand-designed template-matching network, 12 layers:

1. `c1` conv 1→13: a filter bank typed from formulas — six oriented first-derivatives of a
   Gaussian (edge normals 0°,30°,…,150°, σ=1.0) taken with both signs (12 channels), plus one
   Gaussian blur channel (σ=1.1) that carries ink presence.
2. `r1` relu, `p1` avgpool 2×2 (features are matched at half resolution for shift/style
   tolerance), `n1` norm (per-channel standardisation makes the feature vector's norm
   constant across images, so every match score is a cosine).
3. `m` conv 13→536 with 14×14 kernels, stride 1, padding 3: correlation of the feature map
   with 536 typed templates over a 7×7 grid of shifts; `mp` maxpool 7×7 keeps the best shift
   per template.
4. Readout: `mx`/`mr`/`mpv`/`mv` lay each class's template scores out as (10, N, 1) and
   max-pool along the variant axis = best template per class; `mn` adds 0.15× the
   mean-over-templates as a tie-breaker; `out` sums the two.

## How the weights were produced
No training data, fitting or image-derived numbers are used anywhere; `build.py` reads no
files. Every weight comes from typed knowledge of how digits are written:

* 67 handwriting styles (6–7 per class) are drawn as pen paths by parametric generators with
  hand-typed style parameters — oval axes, flag/serif presence, flat vs curved base of 2,
  round vs flat top of 3, crossbar height and closed counter of 4, bowl type of 5, hook and
  loop size of 6, crossbar and descender slope of 7, waist height and loop ratio of 8, tail
  type and loop size of 9 — plus typed slants.
* Each style is rendered at two stroke widths (3.0, 2.4), with an extra slant (0, 0.12) and a
  jittered copy (fixed seed 5) = 536 templates; each render is aspect-scaled per class and
  re-centred by its centre of mass, mirroring the dataset's stated normalisation.
* Each rendered glyph is passed through the same filter bank the network applies to the
  image (half-resolution averaging, per-channel standardisation, Gaussian softening of the
  weights for near-miss tolerance). A 13th template channel encodes ink expectation:
  positive on the modelled strokes, negative inside the glyph's box and outside it, so ink
  in the wrong place lowers the match. Templates are L2-normalised (equal norms make
  correlation ranking equal nearest-neighbour ranking) and scaled so scores are cosines.
* All readout weights are exact 0/1/± tables (class membership, variant layout); biases are
  hand-chosen constants.

## Checks
Pipeline behaviour was debugged on synthetic digits drawn by an independent typed "simulated
writer" (jitter, shear, width, per-stroke displacement — no dataset images), which reaches
~99% on its own drawings. The five allowed runs on training images gave 47.4% (v1), 47.3%
(size fix alone), 55.1% (soft weights), 59.9% (half-res + style grid) and 60.8% (ink-dominant
channel + jitter copies); the submitted network is that last configuration with the style
set expanded from 37 to 67 typed styles, which strictly increases coverage of the shapes
each class can take.
