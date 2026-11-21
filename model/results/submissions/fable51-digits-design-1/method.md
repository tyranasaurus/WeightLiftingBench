# Method

## Architecture

A fixed, formula-based feature front-end followed by nearest-template matching, all expressed with the allowed ops:

1. `c1` conv 11x11, 9 channels: 8 oriented "ridge" filters (negative second derivative of a Gaussian across the
   stroke, sigma 1.6 px, times a Gaussian along it, sigma 2.5 px, zero-mean, positive lobe summing to 1) at
   22.5 degree steps, plus one Gaussian ink-blur channel (sigma 1 px, gain 1.5). The across-stroke sigma was chosen
   from the filter formula so that strokes 2 to 5 px wide give a similar response. Then `relu`.
2. `p1` max-pool 2x2, stride 2 (28 -> 14).
3. Two branches: `c2` a depthwise Gaussian (sigma 1.0) with stride 2 giving a coarse 9x7x7 map, and `c3` a light
   depthwise Gaussian (sigma 0.7) keeping the fine 9x14x14 map. Each is reshaped to (1, 1, D) and passed through
   `norm`, which standardises the whole feature vector of the image (zero mean, unit variance).
4. `dA`, `dB` dense layers, one row per template: the template's own standardised feature vector divided by D, so
   each output is the Pearson correlation between the image and that template. `add` sums the two branches.
5. `reshape` to (1, 10, 2500), `maxpool` over the 2500 templates of each class, `reshape` to (10,).

So the score of class k is the best (coarse + fine) correlation between the image and any synthetic drawing of
digit k: nearest-neighbour matching against a bank of drawn digits.

## Where the weights come from

No training data is used anywhere; `build.py` reads no files. The filters are closed-form formulas. The templates
are 2,500 drawings per digit: pen paths typed from general knowledge of how each digit is written (ellipse arcs,
lines and Catmull-Rom splines through hand-typed control points, with several style families per digit such as
open / closed 4, 7 with or without a crossbar, 2 with or without a bottom loop, 5 in three stroke orders,
round-loop and "P-like" 9), with hand-chosen ranges of proportions, slant (shear), rotation, width scale, stroke
thickness and a smooth random warp drawn with a fixed seed. Each drawing is rasterised with a round brush, scaled to
a 20x20 box keeping its aspect ratio and placed with its centre of mass at the centre of the 28x28 frame, like
MNIST. Each drawing is passed through the fixed front-end and standardised by its own mean and standard deviation
to become one dense row. No statistic across drawings (no means, scales or thresholds measured on them) is used.

## Checks

Five runs on 10,000 training images were allowed (`checks.log`). They were used to verify the pipeline and to see
which confusions dominated; the changes between checks were made by reasoning about those tables (missing digit
styles, the matching metric, pooling), not by fitting any number to the data:

- check 1 (83.4%): first version, 1,000 templates per class, coarse 7x7 features only;
- check 2 (86.0%): added the fine 14x14 branch and broader digit styles;
- checks 3 and 4 (84.1%, 84.7%): tried squared-Euclidean nearest-neighbour matching (with blur, then with 4x4
  max-pool tolerance); both lost accuracy on the ink-heavy classes, so correlation matching was kept;
- the submitted network (correlation matching as in check 2, plus thickness-uniform ridge filters and extended
  digit styles) was built last; time ran out before the fifth check could be run on it, so its measured reference
  point is check 2 (86.0%), whose metric and pooling it shares.

Note on the comparison baselines: this architecture trained by gradient descent on the 60,000 images would be a
large max-out network and would very likely reach a higher accuracy than hand-set weights can; the aim here was to
get as far as possible from general knowledge alone.
