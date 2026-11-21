# Method

## Architecture

`image → c1 → relu → c2 → relu → c3 → c4 → relu → c5 → fc → reshape → maxpool → reshape(10)`

| layer | what it does | where its numbers come from |
|---|---|---|
| `c1` conv 3×3, 8 channels | directional derivative of the image along 8 directions (45° apart) | Sobel kernels ÷ 8, `cos θ · Sx + sin θ · Sy` |
| `c2` conv 1×1 | splits each gradient between its two neighbouring directions: `r[d] − r[d+90°] − r[d−90°]`, which equals `|g|(cos δ − |sin δ|)` after the relu | parallelogram rule, entries 1 and −1 |
| `c3` conv 11×11, stride 3 | blurs each direction map and samples it on an 8×8 grid (centres at pixels 3, 6, …, 24) | Gaussian, σ = 1.6 |
| `c4`, `c5` conv 1×1 around a relu | piecewise-linear square root of each feature (8 knots at `(0.07k)²`) | formula for the chords of √x |
| `fc` dense, 512 → 230,000 | cosine match of the 512 features against 23,000 templates per class | drawings made by `build.py` (below) |
| reshape `[1, 10, 23000]`, maxpool `[1, 23000]` | class score = best match among that class's templates | — |

Total 117,999,000 weights. Nothing in the network is trained or fitted.

## How the templates are produced

`build.py` contains a small pen-path model of each digit, typed from general knowledge of how
digits are written: an oval for 0; a stem with optional flag and foot for 1; hook, sweep and foot
(with or without a loop) for 2; two bowls for 3; open or closed 4; bar, neck and bowl for 5; stem
into a loop for 6; bar and stem with optional serif and cross stroke for 7; S-stroke, teardrop or
two-ring 8; loop and straight or hooked stem for 9. Control points are joined by Catmull-Rom splines.

A fixed-seed random number generator varies the writing style of each drawing: the model's own
proportions (bar length, loop size, …), a small wobble of the control points, a smooth second-order
warp, width-to-height ratio, slant, a few degrees of rotation, pen width and edge softness. The pen
is never allowed to be so wide that it fills a loop or opening that defines the digit.

Each drawing is rendered the way the task describes the data: ink scaled to fit a 20×20 box,
soft-edged strokes, then shifted by whole pixels so that its centre of mass is nearest the centre of
the 28×28 frame. Because I do not know whether "centre" means between the two middle pixels or on
the 15th pixel, half the drawings use each.

Every drawing is passed through the front end above and becomes one row of `fc.W`, divided by its
own length and nothing else. No average, variance, cluster, selection or any other statistic is
taken across drawings, and no number in the network was chosen by scoring it on generated images.

## Use of the training images

I looked at the 100 training images (as contact sheets, alone and beside my own drawings) to see
what the classes look like in general — for example that fives often have a long bar flying off to
the right, that threes and twos have long sweeping tails, and that loops stay open even with a thick
pen. No pixel values, coordinates or measurements of particular images were typed into the code, and
no statistic of the training images was computed.

The network was run on the training images four times (one check is unused):

| check | accuracy | what I learned |
|---|---|---|
| 1 | 96/100 | the pipeline works on real digits; misses were 2→3, 3→5, 3→9, 9→7 |
| 2 | 97/100 | dropping the near-empty border ring of the feature grid (more templates fit) and wider style ranges did no harm; 9→7 gone |
| 3 | 98/100 | more varied proportions and keeping loops open helped; a new 2→1 miss showed that a slanted 1 drawn with flag and foot is the same shape as an angular 2 |
| 4 | 99/100 | drawing the flag-and-foot 1 upright removed the 2→1 miss; one 3 is still read as 9 |

The submitted network is the one run in check 4. With only 100 images these figures are rough;
accuracy on unseen digits is likely a few points lower than 99%.
