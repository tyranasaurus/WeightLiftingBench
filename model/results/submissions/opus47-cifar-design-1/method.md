# Method

## Architecture

```
image (3, 32, 32)
  c1  conv 3 -> 6, 1x1        six colour-opponent features (hand-picked)
  r1  relu
  p1  avgpool 4x4 stride 4    -> (6, 8, 8)   8x8 regional feature map
  rs  reshape                 -> [384]
  fc  dense 384 -> 10         class templates + calibrated biases
```

Total: 3,874 weights.

## How the weights were produced

Every weight comes from my own understanding of the ten CIFAR-10 classes —
no training data was read, and no statistic of the training images or
labels was turned into a weight.  I did view a grid of randomly selected
samples and print simple colour summaries (as explicitly allowed) to
confirm my intuitions about the classes, but the numbers below are
general-knowledge priors, not measurements.

### c1 — colour-opponent features

Six per-pixel colour features chosen so each class has a plausible signal:

* **BLU** `B − 0.5R − 0.5G` — saturated blue (sky, water).
* **GRN** `G − 0.5R − 0.5B` — saturated green (foliage).
* **BRN** `0.4R + 0.3G − 0.7B` — warm brown.  The G coefficient is
  deliberately smaller than R, so pure green pixels do **not** fire BRN.
* **RED** `R − 0.4G − 0.6B` — red / orange, and crucially *not* orange-brown.
* **DRK** `0.3 − mean(R,G,B)` — a pixel below ~0.30 brightness.
* **VBR** `mean(R,G,B) − 0.75` — a highlight above ~0.75 brightness.

### p1 — 4×4 average pool

Collapses the 32×32 feature maps to an 8×8 grid of regional means — enough
spatial resolution to express "blue at top", "dark at bottom", "something
warm in the centre", etc.

### fc — class templates

For each class I encode an 8×8 spatial map of how much each feature should
be present, based on common-sense priors:

| class       | what it looks like                                          |
|-------------|--------------------------------------------------------------|
| airplane    | blue sky dominating top half, plane body bright in centre   |
| automobile  | varied centre body, dark road strip at bottom, no grass     |
| bird        | small warm/dark body in centre, natural fringes             |
| cat         | warm/dark furry face in middle, no outdoor cues             |
| deer        | tan body centre, green grass/forest around it               |
| dog         | warm furry body in middle, varied background                |
| frog        | green and muted overall, not bright, not red, not blue      |
| horse       | brown or dark body, grass at the bottom                     |
| ship        | blue at **both** top (sky) **and** bottom (water)           |
| truck       | large body, dark strip at the very bottom (wheels + shadow) |

### Bias calibration

For each class I compute the expected score when every feature sits at a
general-knowledge prior magnitude `MU[f]`, and set the per-class bias to
cancel it.  This keeps the ten classes comparable, so no class wins just
by giving a large weight to a feature (like `DRK`) that is common in most
images; the argmax is driven by the image's actual deviation from typical.

## Rules compliance

* No gradient descent, no fitting, no class means, no clustering,
  nothing derived from image contents of specific training images.
* No pixel values or coordinates typed from any training image.  Template
  colours and feature weights are general-knowledge magnitudes.
* `build.py` reads no files (checked with `check_rebuild.py`).
