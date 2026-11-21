# Method

**Architecture.** A fixed front end feeds one large template layer and a soft nearest-template vote.

- *Colour path:* the image is 2x2-averaged, then 1x1 convolutions and relus turn every pixel into 51 soft colour
  categories (3 lightness levels x [grey + 8 hues at a muted and a vivid chroma ring] in lightness / red-green /
  yellow-blue space, L1 cones built from rectified halves). These are average-pooled on 4x4, 2x2 and 1x1 grids,
  passed through a concave piecewise-linear squashing, and each grid is normalised with the `norm` op.
- *Shape path:* brightness, normalised over the picture, is filtered with oriented derivative-of-Gaussian edge
  filters (6 orientations, both polarities, rectified and summed) at three scales, pooled to 4x4 cells, squashed
  and normalised; plus the normalised brightness picture shrunk to 8x8.
- The groups are placed side by side in one 1423-number descriptor (each scaled by sqrt(weight / size)), so a
  dot product of two descriptors is a weighted sum of per-group correlations.
- *Read-out:* dense layer `sim` (one row per template) -> `norm` over all templates -> relu(z - theta) at
  theta = 1, 2, 3 (weights 1, 10, 200) -> average per class -> 10 scores.

**Where the weights come from.** No training, fitting or statistics of any images.

- Front-end weights are formulas (colour lattice, Gaussian-derivative filters, pooling, fixed thresholds chosen by hand).
- The template rows are *drawings*: `build.py` contains shape models typed from general knowledge of the ten classes
  (fuselage + swept wings + fin; hull + superstructure + sails; car body + dark glass + two wheels; tall cab + box +
  many wheels; animals from ellipses, tapered legs, necks, ears, antlers, beaks ...), typed colour ranges, and typed
  scene layouts (sky / water / road / grass / woodland / indoor, horizon at camera height), drawn with a fixed
  random seed. Each drawing becomes its own template: its row is the network's own descriptor of that one drawing,
  normalised by itself alone by the same `norm` layers. Nothing is averaged or measured across drawings.
- Training images were only viewed as contact sheets (also side by side with my drawings) to judge by eye what the
  classes and photos look like in general; no pixel values, measurements or summaries were turned into numbers.

**Checks on the training images** (logged in checks.log; each was one full run with labels):

1. 34.5% - first complete version; busy-looking template classes (car, frog, truck, cat) attracted too many photos.
2. 34.1% - more natural textures / clutter in the drawings and an 8x8 brightness group; fixed the frog
   over-prediction but no overall gain; car vs truck stayed at chance.
3. 33.4% - vote split into joint / colour / shape / centre experts; no gain, so it was dropped again.

4. 32.8% - the submitted version. After check 3, a side-by-side look showed general mismatches (drawings darker,
   duller, speckled, objects too small); these were corrected throughout the generator and the single joint vote
   of checks 1-2 was restored. The check showed no gain: all four versions sit at 33-34.5%, within the noise of
   1,000 images, and car vs truck, dog vs cat and deer/horse vs frog/bird remain the main confusions.

The fifth check was not used. `check_rebuild.py` confirms that `build.py` reproduces the submitted files exactly.
