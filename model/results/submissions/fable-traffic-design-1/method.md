# Method

**Architecture** (5 layers, about 5.9M weights): `conv 3x3` (binomial blur combined with a fixed colour transform:
luminance, red-green, blue-yellow) -> `dense` template bank (43 classes x 45 templates) -> `reshape` to (1, 43, 45)
-> `maxpool` over the 45 templates of each class -> `reshape` to 43 class scores.

**Weights**: nothing is trained or fitted and no training image or statistic of the training set is used.
`build.py` draws every sign itself from general knowledge of the German sign designs (red ring with stroke-drawn
digits, cars and trucks; red warning triangles with simple pictograms; inverted yield triangle; stop octagon with
STOP; no-entry disc with bar; priority diamond; blue discs with white arrows; white "end of" discs with a diagonal band).
Each class is drawn at 5 sign sizes (19 to 27 pixels wide, derived from the stated border rule) and 9 one-pixel
shifts. Each drawing is passed through the same blur + colour transform and becomes its own template: made zero-mean
inside its own sign mask, zeroed outside the sign (so background is ignored), and scaled to unit length by itself alone.
The class score is the best correlation over that class's templates. Because the features are linear and templates are
zero-mean, the prediction does not depend on overall brightness or contrast of the photo.

**Status**: this is a first, simple version saved when time ran out; the pictograms are coarse and no local
contrast normalisation or edge-orientation features were added.
