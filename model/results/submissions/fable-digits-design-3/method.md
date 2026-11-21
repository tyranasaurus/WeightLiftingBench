# Method

**Architecture.** A fixed, formula-built feature extractor followed by template matching:

1. `conv` 5x5: eight directional derivative-of-Gaussian filters (directions 0, 45, ..., 315 degrees), then `relu`.
2. `conv` 1x1 + `relu`: `r_k - r_(k+90) - r_(k-90)`, which splits the gradient vector onto its two nearest of the
   eight directions (the classical parallelogram decomposition of gradient-direction features).
3. `conv` 13x13, stride 3: Gaussian blur (sigma 2 px) of each direction plane, sampled on an 8 x 8 grid
   -> 8 x 8 x 8 = 512 numbers saying "how much edge of each direction is near each place".
4. `conv` 1x1 + `relu` + `conv` 1x1: a piecewise-linear square root (variance-stabilising compression).
5. `dense`: 230,000 rows, one per template, each row a unit-length feature vector, so the output is a cosine
   similarity with every template.
6. `reshape` + `maxpool` over the 23,000 templates of each class -> 10 class scores (nearest template wins).

**Weights.** No training, fitting or statistics of the training images. Layers 1-4 are closed-form filters. The
templates come from pen-path models typed from general knowledge of how digits are written: for each digit several
ways of writing it (e.g. 2 with a corner, a loop, a hook, a wavy base, or Z-like; 4 open, u-shaped or closed;
7 plain, with serif, with cross-bar, rounded; 9 with straight tail, hook, or ring + stem), each given as strokes
through control points with randomly drawn (fixed seed) style parameters, control-point jitter, gentle bends,
proportions, slant, rotation and pen width. Each drawing is splined, rendered with a round pen, scaled to a
20 x 20 box and shifted so its centre of mass is at the frame centre (as the task describes the data), passed
through the same fixed front-end, and divided by its own length. Each drawing is one template by itself; nothing is
averaged or measured across drawings. `build.py` regenerates everything from its own source.

**Checks on the training images (3 of the 5 allowed were used).**

- Check 1, 94.06%: first version (narrower, more "textbook" pen paths, 9 x 9 grid, 180k templates). Learned that the
  design works but 2 was badly covered and several classes were too regular.
- Check 2, 95.78%: broader, more handwriting-like pen paths and 230k templates on an 8 x 8 grid. This is the
  submitted network. Learned that 2 is still the weakest class and that many remaining errors are between shapes
  differing by a closed vs open region.
- Check 3, 95.45%: added a second "concavity" branch (holes and bays found by ray casting) plus more ways of writing
  a 2. It fixed some confusions (2->8, 3->8) but hurt digits with loops (6, 8, 9), so it was not kept; the
  submission is the check-2 network, rebuilt from the saved check-2 script.

Looking at training images was used only to judge by eye which writing styles and proportions occur; no pixel
values, coordinates or measured statistics from them enter the weights.
