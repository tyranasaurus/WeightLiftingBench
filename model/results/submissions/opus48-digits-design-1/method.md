# Method

## Architecture
A single matched-filter bank, built entirely from formulas and hand-drawn shapes
(no training data, no gradient descent, no statistics of the training set):

```
image (1,28,28)
 -> conv "feat"  : 10 closed-form filters, 7x7
                   - 2 isotropic Gaussian blurs (sigma 0.8 and 1.6) -> ink/shape
                   - 8 oriented ridge detectors (2nd-derivative-of-Gaussian bars
                     at 0..157.5 deg) -> stroke orientation energy
 -> relu         : non-negative ink / stroke energy
 -> avgpool 4x4 stride 2 -> (10,13,13) feature map
 -> dense "match": 1920 rows, each the unit-norm feature vector of one hand-drawn
                   digit template (192 templates per class)
 -> reshape (10,192,1) -> maxpool over the 192 templates -> reshape (10,)
```

The score of class *k* is the largest dot product between the input feature map
and any of class *k*'s templates; the highest score wins.

## Why it works
Features are non-negative (relu energies) and every template is unit-norm, so the
match is a cosine on non-negative vectors: a busy digit matches a busy template
(more overlap) and a thin digit a thin template, with no bias toward sparse
prototypes. Because the filters are linear and relu is positively homogeneous,
the feature map scales with the image's contrast, so the match is automatically
contrast-invariant — no normalisation layer is needed.

## How the weights were produced
* **Filters** (`feat.W`): closed-form Gaussian blurs and oriented
  second-derivative-of-Gaussian ridge detectors, each written from a formula and
  normalised analytically. No data.
* **Templates** (`match.W`): each digit is drawn from typed pen-path strokes
  (general knowledge of digit shape: ellipses for loops, slanted bars, arcs),
  rendered anti-aliased, scaled to a 20px box and centred by centre of mass
  (matching the MNIST preprocessing). For coverage of handwriting variation each
  digit is drawn in 2-3 styles, 4 slants, 2 pen widths and 8 smooth elastic warps
  (fixed random seeds) = 192 templates per class. Each rendered image is passed
  through the same fixed filter bank and the resulting feature vector, normalised
  to unit length **by itself alone**, becomes one row of `match.W`. No average,
  prototype or other statistic is ever taken across templates, and no training
  image, label or measurement is used anywhere.

`build.py` reproduces `network.json` and `weights.npz` deterministically from the
code alone (verified with `check_rebuild.py`).
