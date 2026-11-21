# Method

## Architecture
A three-stage convolutional feature pyramid plus two linear read-out heads, built only from
`conv / relu / maxpool / avgpool / reshape / dense / add`:

* **s1** (conv 5x5, 61 channels): hand-written Gabor / Gaussian-derivative filters (8 orientations x
  {bright bar, dark bar, two edge polarities} on luminance, 4 orientations x 2 phases on the R-G and
  B-Y opponent axes), centre-surround DoG, and Gaussian-blurred colour projections
  (L, R-G, G-R, B-Y, Y-B, G-M, M-G, R-B, B-R) plus thresholded highlight/shadow channels.
  Followed by relu and 2x2 max-pool.
* **s2** (conv 5x5, 26 channels): base detectors composed from s1 by formula - dark/bright rings at two
  radii (tangentially-oriented bar detectors placed on a circle, for wheels/eyes/spots), long
  horizontal/vertical/diagonal lines, mean edge energy (`texture`), and its thresholded complements
  `smooth` = relu(0.45 - texture) and `rough` = relu(texture - 0.90), plus local colour states
  (blue, green, red, yellow, warm, chroma, achromatic-bright/mid, bright, dark).
* **s3** (conv 1x1, 25 channels): semantic combinations of s2 - `foliage` (green + rough),
  `fur` (rough + warm, minus green/blue), `sky` (blue + bright + smooth), `water` (blue, not bright,
  smooth), `road` (achromatic mid + smooth, not highlight), `metal`, `wheel` (dark ring),
  `white_p` (achromatic bright + smooth), `warm_bg` (indoor beige), `dark_blob` (small dark ring),
  plus pass-throughs of the base channels the rules need.
* **Heads**: `h2d` reads a 4x4 regional average-pool of s3 (layout-sensitive rules), `h3d` reads the
  global average of s3 (whole-image priors); their 10-dim outputs are summed by `add`.

## How the weights were produced
No training, fitting or data-derived statistics were used. `build.py` generates every number from
closed-form formulas (Gabor/Gaussian/ring/line masks) and from a hand-written rule table that encodes
general knowledge of the classes: e.g. *ship* = water in the lower/middle band + sky above + bright
hull; *airplane* = smooth bright sky above + slim bright body, no water; *automobile/truck* = dark
wheel rings + grey road below, truck additionally a tall smooth white box; *horse/deer* = vegetation
around + warm body; *cat/dog* = close-up fur (cat with warm indoor backdrop, dog often white or on
grass); *frog* = green centred blob with dark wet surround and dark eyes; *bird* = uniform sky/field
backdrop with a small textured body. Region masks (top/bottom/middle/centre/all) place each cue where
it belongs in the frame. The five allowed `run_network.py` executions were used only to catch
implementation pathologies (a normalisation head that emitted constants, and texture/line channels
that fired everywhere); each fix was a redesign of the detector semantics, not a fit to labels.
