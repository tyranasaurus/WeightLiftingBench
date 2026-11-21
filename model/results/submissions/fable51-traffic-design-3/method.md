# Method

Hand-designed template-matching network for the 43 German traffic sign classes; no weight is learned or
derived from the training images (they were only viewed as PNG sheets to understand the classes).

**Architecture** (all weights written by `build.py` from formulas and drawings):
1. Joint brightness/contrast normalisation of the RGB image (reshape -> norm -> reshape).
2. A fixed 5x5 filter bank: 8 signed Gaussian-derivative orientation channels on luminance and 5 colour
   channels (red, blue, yellow, bright, dark) from hand-chosen colour formulas. A second norm layer fed with
   tiny values only subtracts each channel's image mean (so colour casts cancel); every channel is then
   thresholded and clipped with two relus combined by a 7x7 Gaussian smoothing conv, which also adds a
   coarse "dark mass" channel.
3. Two 32x32 template convs: for every class, 5 sign sizes x 2 palettes (normal and dim) are drawn from general
   knowledge of the sign designs (ring widths, DIN-like digit strokes, arrows, pictograms, vertically stretched
   triangles because the crops are square). One conv holds the whole sign (separately normalised colour
   layout and edge structure, zero-mean orientations), the other the pictogram alone rendered on a plain canvas
   of the interior colour (zero-mean, masked to the interior away from the border). Each template is
   normalised by its own norm alone.
4. The interior score gets one extra pixel of shift freedom (3x3 max), is clipped to [-1, 2] via relus and added
   to the whole-sign score; classes with empty interiors carry a penalty on dark mass instead; max over
   positions (+-2 px) and over the 10 variants gives the 43 scores. A single hand-chosen bias (+0.3) is
   given to "no vehicles".

**Checks**: 4 of the 5 allowed runs on training images were used (47.7%, 45.3%, 32.4% on a random 6000-image
subset, and one diagnostic run on modified subsets). The diagnostic showed that interior templates were matching
the sign's own border edge and that triangles are stretched in the crops; the final build fixes these but
could not be re-checked on training data before the time ran out, so its accuracy is unverified.
