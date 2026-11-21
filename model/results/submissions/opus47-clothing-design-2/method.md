# Method

## Architecture

A small 2-layer feature + subtraction classifier:

1. **norm** — spatial whitening of the input image
2. **dense "feat"** → 20 activations (2 per class). Row `2k` is a positive
   silhouette of class `k` (where class `k` typically has ink), row `2k+1` is
   an anti silhouette (where class `k`'s common confusers have ink but `k` does
   not).
3. **relu** — zero out negative activations (so only clear matches contribute)
4. **dense "out"** → 10 class scores. `out[k] = feat[2k] − feat[2k+1]`.

Total parameters: 15 910 (well under the 120 M budget).

## How the weights were produced

All 20 templates are drawn from simple geometric primitives (axis-aligned
rectangles) encoding my general knowledge of what each class looks like:

- Clothing / bag classes (0, 2, 3, 4, 6, 8): positive shoulder/sleeve/body
  rectangles; anti rectangles at regions where their main confusers have ink
  (e.g. full-width sleeves penalise T-shirt, long bottom penalises pullover).
- Trouser (1): two leg rectangles with an explicit dark-gap anti at the center
  columns.
- Shoe classes (5, 7, 9): bottom-only sole plus structural details (sandal
  straps, sneaker solid upper, ankle-boot back shaft); anti rectangles for the
  top half (STRONG, since shoes never have ink there).

After drawing, each template is spatially centered and L2-normalised so that
scores are comparable across classes. The second dense layer simply implements
`pos − anti` per class.

`build.py` is deterministic and fully self-contained — no training data, no
statistics of the training set, no copying from specific images.
