# Method

## Architecture

`image -> edge (conv 7x7, 8 ch) -> relu -> sharp (conv 1x1, 16 ch) -> relu -> pool (conv 12x12, stride 4, 8 ch, 7x7 grid)
-> split (conv 1x1, 32 ch) -> relu -> root (conv 1x1, 8 ch) -> match (dense, 300,000 outputs)
-> reshape (1, 40, 7500) -> maxpool over each group of 7,500 -> reshape (1, 10, 4) -> avgpool over the 4 groups -> reshape (10)`

117.9M weights in total (limit 120M).

## Where the weights come from

Nothing is trained or fitted, and `build.py` reads no file. Two sources only:

1. **Formulas (the front end).**
   - `edge`: Gaussian-derivative filters (sigma 1) at 8 directions, scaled so a full-contrast step edge gives about 1.
   - `sharp`: each direction minus the two directions at +-90 degrees, which narrows the direction tuning; a second
     copy shifted by a constant lets the next layer clip the response (`min(s, 0.35)`), so thin and thick,
     faint and dark strokes give similar edge strength.
   - `pool`: a Gaussian window (sigma 1.8) sampled on a 7 x 7 grid, giving tolerance to small displacements.
   - `split` / `root`: a piecewise-linear square root (exact at 0, 1/64, 1/16, 1/4, 1), the usual compression for
     direction histograms.
   All constants were chosen by hand from how these stages behave on pen strokes 2-4 pixels wide.

2. **Typed pen paths (the dense layer).** Each digit is written in `build.py` as strokes through control points and
   ellipse arcs that I typed from general knowledge of how the digit is written, with several styles each (for example
   2 with a sharp or a looped foot, 4 open or closed, 7 with or without a crossbar or a tick, 8 as an S closed by a
   straight or curved return stroke, 9 with a straight, hooked or kicked tail). A seeded generator (uniform draws only)
   varies the shape parameters, slant, width, rotation and pen width. Each drawing is rendered like the data (long side
   20 pixels, centre of mass at the centre of the frame), passed through the same formula front end, divided by its own
   length, and stored as one row of `match`: one drawing is one template, normalised by itself alone. No statistic
   across drawings is used anywhere. There are 30,000 templates per class.

A class's score is the mean, over 4 groups of its templates, of the best match in each group (a nearest-template rule
that needs more than one lone template to agree).

## How the training images were used

I looked at them (as PNG sheets) to study how each digit is written, and compared them by eye with sheets of my own
drawings; no pixel values, coordinates or measurements were taken from them and no weight derives from them. The
network was run on them four times, each with `run_network.py`:

| check | accuracy | what I learned |
|---|---|---|
| 1 | 0.943 | The design works; errors concentrated in 2 (read as 8) and 8 (read as 9). |
| 2 | 0.950 | Redrawn 2, 3, 5, 6, 8 helped 5 and removed 2-as-8, but the hard 2s moved to 3; 2, 8, 9 still weak. |
| 3 | 0.958 | A richer 9 model lifted 9 from 90 to 95; 2-as-3 and 8-as-9 barely moved, so the cause was systematic. |
| 4 | 0.958 | A large round foot loop lifted 2 from 89 to 94, but 9 fell back to 92 (5 read as 4) and 8 stayed at 89: no net gain. |

The fifth check was not used. The submitted files are the build measured by check 4.

## Known weaknesses

On the 1,000 training images the build scores 95.8%: 8 (89), 9 (92) and 6 (94) are the weak classes; the other seven
are at 94-99. The remaining errors are 8 read as 9, 9 and 6 read as 4, and 2 read as 3. I ran out of time before
finding the cause of the 6-as-4 and 9-as-4 confusions, and the last round of changes (check 4) traded gains on 2 for
losses on 9 rather than improving the total. Held-out accuracy should be expected near 95-96%, which is well short of
what the same architecture would reach if trained on the full training set.
