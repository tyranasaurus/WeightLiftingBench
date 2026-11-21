# Method

**Architecture (one-stage template matching, all weights hand-derived).**
1. `col`: 1x1 conv to three channels: intensity, and two colour-opponent channels (R-G, B-(R+G)/2) scaled by 0.1 so that the following `norm` layer's epsilon acts as a soft noise gate (weak colour is suppressed, real sign colour is standardised).
2. `cn`: per-channel `norm` (removes brightness/contrast differences; about one image in five is very dark).
3. `feat`: 7x7 conv with filters from formulas: Gaussian first derivatives at 4 orientations (both polarities) on intensity, second-derivative "bar" detectors at 4 orientations (dark and bright bars) for thin strokes, and identity (+/-) taps of the three normalised channels. `relu` then 4x4 average pooling with stride 2 gives HOG-like 22x15x15 features.
4. `tm`: dense layer whose rows are unit-normalised feature vectors of procedurally drawn signs: 43 classes x 162 variants (3 sizes x 9 shifts x 2 backgrounds x 3 horizontal squashes, mild blur). The drawings (rings, triangles, octagon, diamond, DIN-like digits drawn as strokes and arcs, arrows, car/truck/pedestrian/deer/snowflake pictograms, diagonal stripes) come from general knowledge of German sign designs. Warning triangles are rendered stretched to square proportions because the crops were resized without preserving aspect ratio.
5. `reshape` to (43, 162, 1), `maxpool` over the variants, `reshape` to (43): the class score is the best cosine-type match of any of its variants.

**How the weights were produced.** `build.py` renders every template, runs it through the same feature layers (numpy re-implementation of the scorer's ops) and writes the normalised feature vectors as dense weights. No training images, labels, statistics or fitting were used; the training set was only viewed (PNG grids) to understand the classes, and the network was run on training images twice via `run_network.py` (checks logged). Design choices between iterations were tuned on perturbed renderings of my own drawings, never on the data.

**Status.** Check 2 (an earlier version of this design, before the colour gating, geometry and pictogram fixes) scored 47.2% on 13,070 training images. The saved version adds those fixes but was not re-checked on real images before time ran out.
