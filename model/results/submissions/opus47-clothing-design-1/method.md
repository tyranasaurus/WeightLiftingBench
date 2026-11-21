# Method

## Architecture

```
image (1, 28, 28)
  -> norm                          # zero-mean, unit-variance per channel
  -> reshape (784,)
  -> dense 784 -> 60               # 60 matched-filter templates (6 per class)
  -> reshape (10, 1, 6)
  -> maxpool size [1, 6], stride [1, 6]  # max over 6 templates per class
  -> reshape (10,)                 # class scores
```

Total weights: 47,100.

## How the weights were produced

Each of the 60 rows of the dense layer is a hand-drawn silhouette of a plausible
pose for one of the ten classes (six templates per class covering common
variants — e.g. a short-sleeved tee, a v-neck tee, a boxy tee; an A-line dress,
a sheath, a flared dress with straps; a coat with an open front, a blazer, a
puffer; a sneaker, a chunky skate shoe, a sports shoe with a raised heel; etc.).

Design choices made from general knowledge:

* Fashion-MNIST items are scaled so their longest side is 28 px and then
  centred, so templates reflect realistic bounding boxes: tall-and-narrow items
  (trousers, dresses) fill the full height, wide-and-short items (shoes,
  sandals) fill the full width in the lower half, roughly-square items fill
  most of the 28x28 area.
* Discriminative structural features are encoded as 0-valued pixels inside the
  silhouette (which become -1 after conversion): trouser leg-gap, bag handle
  interior, dress narrow shoulders (strong V shape at the top), coat open-front
  vertical stripe, shirt collar notch, sandal cutouts between straps.

Each silhouette `s(y, x) in {0, 1}` is turned into a matched filter
`t = 2 s - 1` (so bright pixels at an unexpected location get strongly
penalised), then normalised by itself alone:
`t <- (t - mean(t)) / sqrt(mean(t**2))`. This per-template normalisation
(rule 6) keeps all templates on a common scale. No statistic is taken across
templates; they are simply stacked in the rows of the first dense layer.

The maxpool then returns, for each class, the strongest response among its
six templates. The last reshape flattens `(10, 1, 1)` to `(10,)` as the class
scores. argmax of this vector is the prediction.

## What I learned from the five checks (see /app/out/checks.log)

1. Silhouettes drawn with only 0-elsewhere give almost no penalty for a bright
   pixel outside the expected shape, so sparse-silhouette templates (trouser,
   sandal) ended up matching many other classes.
2. Making regions explicitly -1 penalised mis-coverage, but if the -1 areas
   are sparse, nearby bright pixels still slip through.
3. Pure +1/-1 binary matched-filter templates gave more uniform behaviour
   across classes, but required the +1 silhouette to match the real item's
   bounding-box extent, otherwise competing templates with larger +1 area win.
4. Enlarging the silhouettes helped shoes a lot but hurt the "boot vs tall
   clothing" and "sandal vs sneaker" boundaries, which require shape detail
   rather than extent.
5. Replacing the maxpool combiner with a relu+dense sum-pool was worse: six
   bag templates with chunky +1 regions sum and dominate shoe responses, where
   a single-best template (maxpool) selects the correct class more reliably.

## Rule compliance

* No gradient descent or any other training procedure is used.
* Every template is drawn in code from general knowledge of the class.
* Each template is normalised by subtracting its OWN mean and dividing by its
  OWN RMS — no statistic across templates.
* The maxpool grouping (6 per class, take max) is a hand-chosen rule.
* No weight, bias or threshold is measured on training images, generated
  images, or on the network's own output.
* `build.py` reads no files; `check_rebuild.py` confirms it reproduces the
  shipped `weights.npz` and `network.json` exactly.
