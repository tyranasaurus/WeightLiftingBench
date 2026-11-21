# Method

## Architecture

A template-matching convolutional network:
1. `norm` — normalize the image to zero mean, unit variance so templates can treat
   the input uniformly regardless of overall brightness/contrast.
2. `conv` with 100 kernels of size 28×28 (10 classes × 10 style variants per class),
   padding 2. Each kernel is a hand-drawn template for one digit and style.
3. `maxpool` 5×5 over the conv output — picks the best spatial alignment for each
   template, so templates do not need to match at exactly one pixel offset.
4. `reshape` to (10, 1, 10): group the 100 templates back into 10 classes of 10 variants.
5. `maxpool` across the 10 variants — pick the best-matching style variant per class.
6. `reshape` to (10,) — final class scores.

Argmax of the output is the predicted digit.

## How the weights were produced

Each template was rendered from a short list of stroke paths (lines and arcs)
in a normalized [-1, 1] coordinate frame, using only general knowledge of how
handwritten digits are drawn (0 is an oval ring; 1 is a near-vertical stroke;
2 has a top curve and a bottom horizontal; etc.). No training data was used in
choosing any coordinate, radius, or weight.

For each class a Gaussian-blurred positive stroke is drawn into a 28×28 canvas.
A uniform negative "forbidden" value fills the 20×20 interior box minus a
tolerance zone around the stroke, penalizing ink that appears where a digit of
that class would not have ink. Additional class-specific "anti-feature" disks
are subtracted at places where confusable classes typically have ink (e.g. the
upper-left of a 6 is where a 0 would have ink; the lower-left of a 9 is where
a 0 or 4 would have ink; the center column of a 7 is where a 1 has ink).

Ten variants per class differ in slant (strong left-italic, mild left-italic,
upright, right-italic), horizontal/vertical scale, and stroke thickness. The
spatial max-pool and per-class variant max-pool together give tolerance to
small translations and style variation.

Each template is L2-normalized so that score magnitudes are comparable across
classes. All biases are zero.
