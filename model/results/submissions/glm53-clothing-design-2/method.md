# Method

## Architecture (26 layers, ~63k weights)

1. **`imgn = norm(image)`** — per-image mean/std normalisation. Training images vary in
   contrast by >3x, so every later feature is built to be scale-invariant: all conv
   filters and templates are exactly zero-sum, so `conv(imgn)` = `conv(image)/s`.
2. **`imgp = relu(imgn)`** — positive-deviation occupancy. A cell of `avgpool(imgp)` is
   exactly 0 for empty background and ~0.5–2.0 for garment pixels, giving a stable
   "how much item is here" measure that, unlike signed contrast, does not compress for
   dense garments.
3. **Branches** (all pooled from these two):
   - `econv`: 12 formula filters (5x5 Gaussian first derivatives at 0/45/90/135 degrees
     plus second-derivative thin-line detectors, both polarities, L2-scaled) → relu →
     4x4 avgpool = edge/line energy grid.
   - `sp`/`sp2`: 7x7 and 14x14 occupancy grids; `rm`/`cm`: row/column max profiles.
   - `tc`: 13 hand-drawn 28x28 **shape templates** (t-shirt with neck notch and short
     sleeves, trousers with waistband and leg gap, pullover, flaring dress, coat with
     dark front opening, sandal with X-straps, shirt with collar and button placket,
     sneaker with thick sole, bag with handle straps and dark hole, boots with
     left/right shaft, plus a sleeved-dress and a second sandal). Each is zero-sum with
     a full-canvas dark-expectation ring, self-normalised by its mass, and matched over
     a +-8 pixel translation window (`maxpool`), so a perfect fit scores ~1 regardless
     of image contrast, and any extra mass anywhere is penalised.
   - `spars`: sqrt(fill*(1-fill)) fill-fraction scalar.
4. **First hidden layer (55 units)**: 21 hand-written concept rules over the interpretable
   features (sole bars per row band, trouser leg gap, tall sleeve columns at the image
   edge, skirt flare, bag handle straps + hole, button placket line, boot shaft
   asymmetry, hem brightness, sparsity, ...), 13 template pass-throughs, and a "capped
   clone" of every rule whose gate is lowered by 0.8, so `relu(X) - relu(Xc) =
   min(activation, 0.8)`: no rule can exceed a bounded contribution.
5. **Second hidden layer (11 units)**: one conjunction unit per class (shirt = placket
   line AND long sleeves; bag = handle straps AND hole AND no high body; boot = shaft
   AND sole). This prevents generic garment features from carrying a class alone.
6. **Output**: each class score reads its own evidence unit (boot sums its left/right
   variants); argmax decides.

## How the weights were produced

Everything comes from general knowledge of garment geometry (a top's sleeve span is its
widest dimension, so sleeves hug columns 0-3 and 24-27 after scaling to 28px; a shoe's
sole is a bright bar in the lower rows; a bag has thin handle straps around a dark hole;
a dress flares from narrow shoulders), from formulas (Gaussian derivatives), and from
hand-chosen rule tables. No weights were fitted, and no training images or statistics
of them were stored: build.py reads no files and reproduces the answer bit-exactly.

Numeric debugging used only self-drawn synthetic shapes (the same kind of geometric
sketches as the templates) to verify layer wiring and feature scales; real-image
feedback was limited to the five logged confusion tables of run_network.py, which were
used to find and fix structural mistakes (a template translation loophole, a
sign error in the capping clones, occupancy-scale runaways, and generic-vote collapse
into single classes), not to fit weights.

## Honest assessment

The five checks scored 0.26, 0.37, 0.29, 0.29, 0.32 on 4000 training images. The final
network classifies all ten classes correctly on the synthetic geometry probes with clear
margins, but its absolute calibration on real photographs remains uncertain, and the
hard pairs (pullover/coat/shirt, sandal/sneaker) are separated only weakly. Expected
held-out accuracy is modest (likely 0.3-0.6), well below what gradient descent would
reach on the same architecture.
