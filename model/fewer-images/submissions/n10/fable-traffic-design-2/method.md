# Method

**Architecture** (6 layers, about 660k weights): a hand-set template matcher.

1. `feat` — 3x3 conv on the photo: a small binomial blur combined with a fixed colour transform into three maps:
   luminance (R+G+B)/3, red-ness R-(G+B)/2 and blue-ness B-(R+G)/2.
2. `match` — conv with 32x32 kernels and padding 3: 43 classes x 5 sign sizes = 215 templates, each evaluated at
   7x7 shifts (up to 3 pixels each way).
3. `shiftmax` — maxpool over the 7x7 shifts; `grid`/`scalemax` — reshape to (1, 43, 5) and maxpool over the 5 sizes;
   `scores` — reshape to the 43 class scores.

The class score is therefore the best correlation between the photo and that class's drawing over shift and size.
Because all scores are linear in the photo, overall brightness and contrast do not change which class wins.

**Where the weights come from.** `build.py` draws each of the 43 signs from general knowledge of the German sign
catalogue using typed geometry only: red ring with white centre and stroke-drawn digits (20–120), cars / truck
pictograms, red-bordered triangles with stroke pictograms, the yield triangle, the STOP octagon, no-entry disc with a
white bar, the yellow priority diamond, white "end of" discs with a diagonal stripe, and blue discs with white arrows
(left/right variants are mirror images). Each drawing is rendered at 4x supersampling at five diameters (17–29 px),
converted to the same three colour features, and turned into its own template normalised by itself alone: its mean
inside the sign outline is subtracted, everything outside the outline is set to zero (background is ignored), and the
template is scaled to unit length. No statistic across drawings is used, and no number was fitted to, measured on or
copied from the training images; the training images were only viewed as contact sheets to see what the classes look
like. All sizes, colours, stroke widths and scales are hand-chosen constants.

**Status.** This is a first, untuned version saved when time ran out; shapes and constants were not iterated.
One of the 5 allowed checks was used: 38.1% accuracy on the 430 training images (score about 0.37). It showed that
the network runs and the templates carry real signal, but nothing was changed afterwards.
