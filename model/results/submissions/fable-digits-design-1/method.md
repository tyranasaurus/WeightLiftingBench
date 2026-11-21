# Method

## Architecture (all weights written by `build.py`, which reads no files)

1. **Normalising front-end** (dense + ReLU + add + 1x1 conv). Six stages remove, in turn, the digit's residual
   offset (x, y), its slant (coarse, then fine), and bring its width and height to fixed ink spreads. Each stage
   holds a handful of transformed copies of its input (linear-interpolation matrices from a formula: shifts,
   row shears, x- or y-scalings) and keeps exactly one. The choice is made from image moments: "moment ratio
   above/below a bin edge" is linear in the image, so two ReLUs per copy produce a veto that pushes every
   out-of-bin copy below zero; a ReLU and a 1x1 conv then sum the surviving copy.
2. **Features** (conv, ReLU, conv, ReLU, conv). Derivative-of-Gaussian filters in 8 signed directions, a 1x1
   layer that sharpens direction tuning (each direction minus its two perpendicular neighbours), and a Gaussian
   pooling filter (mean of a fine and a coarse Gaussian) with stride 2, giving 8 x 14 x 14 maps. A reshape +
   `norm` z-scores the whole map.
3. **Template matching in shiftable parts** (dense crop, conv, maxpool, add). 26,400 templates. Each template's
   central 12 x 12 cells are cut into 3 x 3 parts; each part is correlated with the test map at its home position
   and one cell (2 px) left/right/up/down, the best of the five is kept, and the nine parts are summed.
4. **Read-out** (maxpool, dense, maxpool). The best template of each class gives that class's score. Classes
   have different numbers of templates (1,200 to 4,800, see `PER_CLASS`), handled by pooling groups of 100.

72.0M weights in total.

## Where the weights come from

* Front-end matrices and filters are formulas. Constants (bin ranges, target spreads, filter scales, part size)
  were chosen by hand from geometry, not measured from data.
* Each template is **one** digit image drawn by `build.py`: pen paths for the ten digits typed from general
  knowledge of how they are written (46 style variants: open/closed 4, looped/sharp/hooked 2, crossed 7, ...),
  with a seeded random wobble, proportions, slant, width and pen thickness, rendered and placed like the dataset
  (20 px box, centre of mass at the frame centre). Each image goes through the same normalisation and filters and
  is z-scored **by itself alone**. No statistic is taken across generated images; no training image, label or
  statistic enters any weight.

## Use of the training set

* Looked at contact sheets of training digits to learn which writing styles occur; printed the class counts.
* Five logged checks on one fixed random subset of 20,000 training images:
  1. 92.3% - rigid whole-template matching works; 2s and 8s weakest.
  2. 93.4% - added deslanting and richer pen-path models; same confusions persist, so rigid matching is the limit.
  3. 94.2% - added width/height normalisation and shiftable parts (with a third as many templates); 2s got worse.
  4. 95.4% - reworked the model of 2, more templates for complex digits, fewer 2-like forms of 1.
  5. 95.6% - added exact re-centring and a few more forms of 2, 8, 9. **This is the submitted network.**
  The checks guided which hand-typed digit models to enrich and how many templates each class gets; nothing was
  fitted, and no variants were scored against each other beyond these five sequential runs.

## Things a reviewer should know

* Runs on generated digits were used only to confirm the network computes what was designed (scorer output vs.
  my own arithmetic, one copy kept per stage). One such run showed the width stage saturating because my first
  target spread (4.8) came from a wrong geometric estimate; I re-derived it from the ring/filled-shape spread
  formula (4.2). The new value comes from the formula, but the run is what prompted the correction.
* Peak memory when run in batches of 500 is about 3.8 GB (the per-part match maps); build takes under a minute.
* Remaining errors are concentrated in 2 (to 3/8/7), 8 (to 6/2/9), 6 (to 4) and 9 (to 7), i.e. where my pen-path models
  still cover real handwriting least well. I do not expect this to match the same architecture trained by
  gradient descent.
