# Method

**Architecture** (7 layers, about 282,000 weights):

1. `blur` – 7 x 7 Gaussian conv (sigma 1) to make matching tolerant of small stroke differences.
2. `nrm` – the `norm` op standardises the blurred image, so the next layer computes a correlation coefficient
   that does not depend on stroke brightness or amount of ink.
3. `tm` – a 28 x 28 conv with padding 2: 360 digit templates, each evaluated at 5 x 5 small shifts.
4. `shift` – maxpool over the 5 x 5 shifts (best alignment per template).
5. `byclass` / `best` / `scores` – reshape to (1, 10, 36) and maxpool over the 36 templates of each class, giving
   one score per class.

**Where the weights come from.** No training, fitting or data of any kind. `build.py` holds pen paths that I typed
from general knowledge of how each digit is written (three ways of writing each digit: e.g. open and closed 4,
7 with and without a crossbar, 1 with and without flag and base, 2 with and without a loop). Each path is drawn
with 3 slants x 2 width-to-height ratios x 2 pen widths, scaled so the ink fits a 20 x 20 box and shifted so its
centre of mass is at the frame centre (the format the task describes). Each drawn image becomes exactly one
template, blurred with the same Gaussian and normalised by itself alone (own mean subtracted, own length divided
out). All constants (slants, aspect ratios, pen widths, blur sigma, shift range) were chosen by hand in advance;
none was tuned on training or generated images.

**Status.** I ran out of time before any refinement: this is the first version as written, with no tuning.
`check_network.py` and `check_rebuild.py` both pass. The network was never run on any image (none of the 5
allowed checks on training images was used), so its accuracy is unmeasured.
