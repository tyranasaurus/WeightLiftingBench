# Method

## Architecture (a matched-filter bank)

```
image (1,28,28)
  -> norm             per-image zero-mean / unit-variance over the 784 pixels
  -> dense (20)       correlation with 20 idealized silhouette templates (2 per class)
  -> reshape (10,1,2)
  -> maxpool [1,2]    best-matching of the 2 pose-variants for each class
  -> reshape (10)     class scores; argmax = prediction
```

Total weights: 15,700 (`scores.W` is 20x784, `scores.b` is 20).

## How the weights were produced (no training, no data statistics)

Every number is a **geometric constant describing a class in general**, drawn with a handful of
primitives (soft rectangles, trapezoids, thick line segments, disks). `build.py` renders 2
templates per class, then normalises each to **zero-mean / unit-variance over its 784 pixels** —
exactly the transform the `norm` op applies to each image. Template and image therefore live in
the same space, so `template · image` is an unbiased nearest-template match. `maxpool` picks the
better of each class's two variants (mirror pairs for the symmetric garments and the
left/right-facing shoes; A-line vs straight dress; heeled vs flat sandal; tote vs clutch bag).
No gradient descent, least squares, class averages, or any statistic of the training images is
used; biases are all zero.

## Design reasoning

The task states each item is centred and scaled so its longest side is 28 px. Garments (tops,
trouser, dress, bag) are taller than wide, so they **fill the full height**; footwear is wider
than tall, so it sits **centred in the middle rows** (empty top *and* bottom) — the clean cue
separating shoes from everything else. Within those groups the templates encode general shape
knowledge and are made discriminative by giving each class **one distinctive region the others
lack**, rather than by strong negative lobes (which over-penalise when a real item's mass falls
slightly outside the drawn region): a T-shirt's short sleeves (empty lower sides); the trouser's
central leg-gap; the dress's narrow solid skirt (vs the trouser's gap) and narrow sides (vs a
bulky coat); the coat's bulk out to the frame edges; the pullover's solid centre; the shirt's
fitted body with a central placket; the sneaker's low solid block; the sandal's open interior and
slim heel; the ankle boot's tall L-shape; the bag's rectangle and handle loop.

Training images were only ever *looked at* (saved as PNGs) to confirm these general proportions,
and the network was run on them through the 5 permitted accuracy checks; no pixel values,
coordinates, or statistics were copied into the weights.
