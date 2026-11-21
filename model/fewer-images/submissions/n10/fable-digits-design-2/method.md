# Method

## Architecture (all weights written by `build.py`, nothing trained or fitted)

```
image (1,28,28)
 c0 + relu        1x1 conv: the two relu halves of a soft binarisation  s = clip(2x - 0.2, 0, 1)
 rays (dense)     5 maps: s itself and the amount of ink strictly to the left / right / above / below each pixel
 f1 + relu        5x5 conv: 8 Gaussian-derivative filters (directional derivatives of s, sigma 1), s, and the
                  relu halves of clip(ray, 0, 1) ("is there ink in sight in that direction?")
 f2 + relu        1x1 conv: (a) each gradient is split between its two neighbouring directions
                  (B_d = relu(A_d - A_d+2 - A_d-2), the classical 8-direction decomposition), (b) the ink map,
                  (c) 11 "concavity codes": a background pixel that sees ink in exactly a given set of >= 2
                  of the 4 directions (hole, bay open to the left / right / top / bottom, corners, channels)
 pg               12x12 Gaussian pooling (sigma 2, stride 4) of the 8 direction maps + ink -> 9 x 6 x 6
 mp, pc           3x3 max-pool, then 12x12 Gaussian pooling (sigma 2.5, stride 4) of the codes -> 11 x 6 x 6
 dg, dc, add      template layer: 150,000 rows (15,000 per class), split over the two feature groups
 reshape, maxpool, reshape   best template match of each class -> 10 class scores
```

About 114.5 million weights; nearly all of them are the template rows.

## Where the weights come from

* **Front end**: formulas only (Gaussian derivatives, Gaussians, triangular 0/1 matrices for the ray sums,
  +-1 rule tables for the direction split and the concavity codes). The few gains and thresholds (binarisation
  2x - 0.2, pooling widths, group gains 0.2 / 0.32 / 0.22) were chosen by hand from pencil-and-paper estimates of
  the group energies and were never changed afterwards.
* **Templates**: a stroke model of handwritten digits typed from general knowledge of how digits are written:
  for each digit a few pen paths (splines / straight strokes) with the usual style variants (open or closed 4,
  7 with or without crossbar, 2 with a corner, loop or rounded base, flat-topped 3, 9 with straight or hooked
  stem, 1 with flag / foot, ...), random proportions, slant, tilt, smooth bending and pen width from a
  fixed-seed generator. Each drawn digit is rendered like the data set describes (20-pixel box, centre of mass
  at the frame centre), passed through the fixed front end, and stored as **its own template, divided by its own
  length**. No average, statistic, scale or threshold is taken over several drawn images, and no training image,
  label or measurement of them enters any weight. `build.py` reads no file.
* The class score is the largest dot product with that class's templates (nearest template by cosine similarity).

## Use of the training images

* Looked at them as a picture (contact sheet) to remind myself of the writing styles; no values, coordinates or
  measurements were taken from them.
* 4 of the 5 allowed checks were used (`checks.log`), each on all 100 images:
  1. 94% - the design works; 3s were weakest (read as 5 or 9). Response: bolder pens, more bending, more 3 / 6 / 9
     styles in the stroke model, more templates.
  2. 96% - the 3 -> 5 errors went away; only single slips left. Response: pooling grid restricted to the digit box
     (6x6 instead of 7x7, which frees weights for more templates), independent tilt of horizontal strokes,
     a few ambiguous drawn styles tightened (e.g. a long flag on a 1 that reads as a 7).
  3. 97% - holds up; remaining errors 2 -> 3, 3 -> 5, 3 -> 9. Response: wider 3s with long tails, larger loops on 2s.
  4. 97% - same three images still wrong, so they are individually ambiguous digits and not a coverage gap;
     stopped there.
  Each version was kept after its check (no variants were scored against each other and none was reverted).
* The network was run on drawn (generated) digits only to see that it runs, that my interpreter agrees with the
  task's, and to time it at batch size 500 (about 5 s per 1000 images, about 1.9 GB peak memory). No number or
  design choice was taken from those runs.

## Caveats

* The 97% is measured on only 100 images (roughly +-2 points), so the held-out accuracy is probably somewhat
  lower - I would expect the mid 90s.
* The same architecture trained by gradient descent on 60,000 images would very likely do better than this
  hand-written version; I did not try to make the architecture hard to train.
* `build.py` takes about 7.5 minutes of single-thread CPU here (limit 30 minutes).
