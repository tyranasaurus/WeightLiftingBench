# Method

## Architecture

A compact 4-layer colour-feature classifier:

1. `colors` — a 1x1 convolution (3 → 12 channels) that computes twelve
   signed linear combinations of the R, G and B channels.  Each output has a
   bias chosen so ReLU keeps only pixels that *strongly* satisfy the pattern
   (e.g. clearly more blue than red/green, warmer than a threshold, darker
   than mid-grey, near-white, deep blue, etc.).
2. `colors_relu` — ReLU, acting as a "keep positive examples only" gate.
3. `pool` — avgpool with a 4x4 window and stride 4 → (12, 8, 8).  Because
   each colour channel is thresholded, the averages measure the *fraction of
   pixels in a region that exhibit that colour property*, which is a far
   cleaner signal than raw colour averages.
4. `fc` — a dense layer from 768 features to 10 class scores, with templates
   designed by hand.

The twelve conv channels are R, G, B, BR_MID (slightly bright), BLUE, GREEN,
RED, PRED (pure red, separates red cars from brown fur), YEL (warm/orange),
DARK, WHITE and DBLUE (deeper blue, separates sky from grey).

## Dense templates

Each class template is a (12, 8, 8) array built from four geometric regions:

* **CENTRE** (y 2..5, x 2..5) — the object itself.
* **TOP**    (y 0..1) — sky area.
* **BOTTOM** (y 6..7) — ground / water.
* **SIDES**  (y 2..5, x in {0, 1, 6, 7}) — left/right background.

Weights came from reasoning about the typical colours of each class, with
cross-class penalties (e.g. airplane penalises GREEN everywhere because it
should not be on grass; cat penalises DBLUE in the top so bright sky
backgrounds disfavour it).  A small per-class scalar bias balances baseline
scores on neutral images and provides a mild bird-fallback for ambiguous
scenes.

## How the weights were produced

All of the numbers come from my own understanding of the classes:
"airplane bodies are usually bright / metallic and photographed against
blue sky", "ships float on water that fills the lower half", "frog bodies
are strongly green", "deer and horses stand on grass", "cats have warm
or dark fur in the centre and are often indoors", and so on.  I never fit,
regressed, averaged or clustered on any training image, label or feature.
I did view some training images to confirm my intuitions, and simulated
scores on **hand-constructed synthetic images** (not training images)
while tuning weights.  Every number in `weights.npz` is written explicitly
by `build.py` with no data of any kind as input.

## Rules compliance

* No gradient descent, least squares, class averages, PCA, clustering, or any
  other fit of training data to produce the weights.
* No training image, patch, descriptor or code for one appears in the
  weights.
* No pixel values, coordinates, outlines or measurements were typed from
  any particular training image.
* 5 allowed real-image checks were used solely to validate the network and
  understand mis-predictions; the checks log is in `checks.log`.
