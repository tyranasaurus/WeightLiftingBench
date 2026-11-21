# Method

**No training, no fitting.** Every weight comes from a formula or from a digit that `build.py` draws itself
from typed pen paths. `build.py` reads no files.

## Architecture

1. `c1` – eight 3x3 directional derivative filters (Sobel, `cos(a)·Sx + sin(a)·Sy` for 8 directions), then relu.
2. `c2` – 1x1 conv `g_k − g_{k+2} − g_{k−2}` then relu: each gradient is shared between its two nearest
   directions (the classical 8-direction gradient feature for handwritten digits).
3. `c3` – Gaussian pooling (sigma 1.8, stride 4) of each direction map onto a 7x7 grid.
4. `c4`/`c5` – a piecewise-linear square root (knots at 0.01, 0.04, 0.09, 0.16, 0.25) built from relus, so that
   long straight strokes do not dominate curved ones; `c5` outputs `[f, −f]` so that the following `norm`
   layer divides the feature vector by its own length (mean is exactly zero).
5. `sel0..3` – one-hot convs that cut the grid into four overlapping quadrants (the middle row and column are
   shared, with weights that add up to 1).
6. `m0..3` – dense template layers, one per quadrant: 150,000 templates (15,000 per class), score
   `x·w − |w|²/2`, i.e. minus half the squared distance to the template, up to a constant.
7. `add` layers sum the quadrant scores per template into top/bottom halves, left/right halves and the whole image.
8. For each of the nine regions (whole, 4 halves, 4 quadrants) a `maxpool` takes the best template of each
   class; the nine results are added to give the class scores. This is nearest-template matching, made
   tolerant by letting each part of the image find its own best template of a class.

## Where the templates come from

Each template is one synthetic drawing. For each digit I typed a pen-path model from general knowledge of how
the digit is written (e.g. 4 = down-and-across stroke plus a long vertical, open or closed top; 7 = bar plus
diagonal, sometimes a tick or a middle bar; 2 with a corner or a loop at the bottom left; 9 with a straight
or hooked tail). Control points are drawn at random within typed ranges (fixed seed), joined by a spline,
slanted / widened / slightly bent, drawn with a pen 1.5–5 pixels wide, scaled into a 20-pixel box and placed
with its centre of mass at the centre of the frame, as the task statement describes the data.
Each drawing goes through the same feature formulas and becomes one row of the dense layers, normalised by
its own length only. Nothing is averaged, selected, thresholded or measured across drawings, and no
setting was chosen by scoring the network on drawings or on training images.

## Checks used

The network was run on the 10 training images with `run_network.py` only to confirm that it works
(see `checks.log`); no number was changed in response to those results.
