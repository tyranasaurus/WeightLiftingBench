# Method

## Architecture

A fixed feature front end followed by a large bank of drawn templates; the class score is the best match among
that class's templates.

| layer | op | what it does |
|---|---|---|
| `c1`, `r1` | conv 7x7 (8 filters) + relu | Gaussian-derivative filters (sigma 1): brightness slope along 8 directions, 45 degrees apart |
| `c2`, `r2` | conv 1x1 + relu | each direction minus its two neighbours at 90 degrees: sharpens the tuning so an edge is shared only between its two nearest directions |
| `c3` | conv 12x12, stride 3 | Gaussian pooling (sigma 2) of each direction map onto a 9 x 9 grid: 8 x 9 x 9 = 648 numbers |
| `c4`, `r4`, `c5` | conv 1x1 + relu + conv 1x1 | a square-root-like curve made of five straight pieces, so strong and weak edges count more evenly |
| `vec`, `unit` | reshape + norm | the 648 numbers as one vector with zero mean and unit variance |
| `match` | dense, 180,000 outputs | one row per template (18,000 per class); the output is the correlation between image and template |
| `byclass`, `best`, `scores` | reshape + maxpool + reshape | maximum over each class's 18,000 templates gives the 10 class scores |

116.8 million weights in total (limit 120 million).

## Where the weights come from

Nothing was trained or fitted, and `build.py` reads no files.

* **Front end**: formulas only. Gaussian derivatives, the direction-sharpening rule, a Gaussian pooling window, and
  the knots of the square-root curve placed at fixed fractions of one analytically derived value (the pooled
  response of a long full-contrast edge, `1 / (sqrt(2 pi) * 2)`).
* **Templates**: `build.py` draws every digit itself as pen strokes along paths typed from general knowledge of how
  digits are written (an oval for 0; a bar with optional flag and foot for 1; hook, diagonal and base with a corner,
  bend or loop for 2; two bowls for 3; open or closed 4; bar, down stroke and bowl for 5; stem into a loop for 6;
  bar and diagonal with optional tick, cross bar or foot for 7; a figure-eight stroke for 8; loop and tail for 9).
  Proportions, slant, rotation, pen width, pen wobble and a smooth bend of the whole drawing are drawn at random
  from hand-chosen ranges with a fixed seed. Each drawing is scaled to a 20-pixel box and centred on its centre of
  mass, as the task describes the images, passed through the same front end, and becomes **one** template row,
  normalised by its own mean and standard deviation only. No average, statistic or score over several drawings sets
  any number.

## Use of the training images

I looked at contact sheets of the training digits to understand which general shapes occur (for example that 2s
often have a loop and a tail sweeping down, that 9s often have a leaf-shaped loop, that 8s often begin with a
long tail). No pixel values, coordinates or measurements from particular images were typed into the program, and no
statistic of the images was computed for the weights.

The network was run on the 1,000 training images five times (all logged in `checks.log`):

| check | accuracy | what it showed |
|---|---|---|
| 1 | 92.3% | the design works; drawings of 2, 8 and 9 were too narrow in variety |
| 2 | 95.0% | broader 2 / 8 / 9 drawings helped; 2 -> 3 and 9 -> 4 remained |
| 3 | 95.9% | tilted-loop 2s fixed 2 -> 3; 9 -> 4 and class 8 still weak |
| 4 | 96.1% | start-tail 8s and footed 7s helped; they also began to capture some 5s; 9 -> 4 unchanged |
| 5 | 95.7% | keeping the 4's top open and its bar crossing the stem cut 9 -> 4 from 8 to 2; bolder pen widths and a tighter tail 8, changed in the same round, did not help |

## The submitted version was not itself checked

The submitted network is not the one scored in check 5. It keeps the 4 fix that check 5 confirmed, puts the pen
widths and the start-tail 8 back to their check-4 settings, and adds one change that no check has seen: half of the
5 drawings now have a long top bar rising to the right, which is my explanation for 5s matching tail 8s. All five
checks were used before this, so its accuracy on real digits is an expectation (about 96%), not a measurement. It
was only run on digits drawn by the program itself, to confirm that it executes and reproduces the build
arithmetic. `check_rebuild.py` confirms that `build.py` reproduces the submitted weights exactly.
