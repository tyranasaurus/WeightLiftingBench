# Method

**Architecture.** A fixed feature extractor followed by region-wise template matching.

1. `g`, `dir`: eight 5 x 5 Gaussian-derivative filters give the image gradient projected on 8 directions; a relu, a
   1 x 1 conv and a second relu turn that into `|g| * relu(cos - |sin|)`, which splits each gradient between its two
   nearest directions (the classic 8-direction gradient feature).
2. `blur`: a Gaussian blur (sigma 1.6 px) with stride 3 reduces each direction map to 9 x 9 cells.
3. `sq1`, `sq2`: a piecewise-linear square root built from 12 relus per channel (knots at (0.05 k)^2) compresses
   strong responses.
4. `nrm`: the whole 648-number feature vector is normalised (zero mean, unit variance) per image.
5. Four overlapping 6 x 6-cell regions (one-hot "crop" convs). Each region has a dense layer holding 103,600 templates
   (10,360 per class). A unit computes `<x, t> - |t|^2 / 2`, i.e. the negative squared distance to its template up to
   a term shared by all classes.
6. Per region and class: max over the templates in each of 4 groups, average of the 4 maxima; the four regions are
   summed to give the 10 class scores. Matching regions separately lets a digit take its top from one template and
   its bottom from another of the same class.

**Where the weights come from.** Nothing is trained or fitted, and no weight uses the training images.

- Filters, blur, square-root knots and crops are closed-form formulas with constants chosen by hand.
- Templates are drawings made by `build.py` itself. For each digit I typed pen paths from general knowledge of how
  people write it (several styles per digit: e.g. open, closed and rounded 4; looped, cornered and Z-like 2; 7 with
  or without tick and crossbar; 9 with straight or hooked tail), with random variation of proportions, slant,
  rotation, hand wobble and pen width from a fixed seed. Each path is rendered as a stroke, scaled to a 20 x 20 box
  and centred on its centre of mass, as the task describes the data. Each drawing is passed through the feature
  extractor and becomes one template, normalised by itself alone. No average or other statistic across drawings sets
  any number, and no setting was chosen by scoring drawings.

**Checks.** Three of the five allowed runs on the 10 training images were used, each only to confirm the network
works (10 of 10 each time). Accuracy on unseen digits is therefore not measured.
