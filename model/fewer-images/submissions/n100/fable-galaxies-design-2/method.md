# Method

**Architecture.** `image -> 1x1 conv (luminance) -> 2x2 average pool (64x64) -> norm -> dense template bank
(3,240 templates) -> reshape (10 classes x 324) -> max over each class's templates -> dense (gain + class biases)`.
13.3 M weights in total.

**How the weights were produced.** No training, fitting or use of the training images. `build.py` renders analytic
galaxy models typed from general knowledge and uses each render as its own template (mean removed and scaled to unit
length by itself alone):

- round / in-between / cigar smooth: Sersic ellipses with axis ratios 1.0-0.9 / 0.75-0.6 / 0.45-0.35, several sizes
  and angles;
- edge-on without bulge: long thin ellipses (axis ratio 0.22 and 0.14) with a flat profile; edge-on with bulge: the
  same thin disk plus a round central bulge;
- barred spiral: bulge + bar + disk with two arms starting at the bar ends; tight and loose spirals: exponential disk
  with two logarithmic arms of 12 and 30 degree pitch, both chiralities, face-on and inclined;
- merging: two blobs of comparable brightness a few pixels apart; disturbed: lopsided one-armed models and a core in
  an offset envelope.

Because the frame is normed and every template is zero-mean and unit-length, a template's response is the cosine
between frame and model, so the largest response is the best least-squares fit with free brightness and sky level.
The class score is the best response of that class's models times a hand-chosen gain (40), plus the log of the class
frequency in this catalogue (known class sizes of Galaxy10 DECaLS) and a hand-set handicap of 0.5 for the merging and
disturbed models, which have an extra free component.

**Check 1 (the only check used).** 28.6 % on the 1,000 balanced training images. What it showed: the models for
round smooth (73/100), in-between smooth (69/100) and edge-on without bulge (77/100) work, but cigar, barred and
tight-spiral models never win, spirals fall to round / in-between / loose, edge-on with bulge mostly goes to
edge-on without bulge, and merging and disturbed are rarely recognised. Nothing was changed after this check.

**Status.** This is a first, simple version saved when time ran out; a more elaborate measurement-based design
(log-polar isophote shapes, colour, texture) was planned but not built. Gain, handicaps and model parameters were set
by hand and were not tuned on anything.
