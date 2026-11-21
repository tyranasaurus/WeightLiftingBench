# Method

**Architecture.** A matched-filter network, written by hand.

1. `avgpool 2x2` halves the image; a 1x1-style conv makes two maps, brightness `L = (R+G+B)/3` and colour
   `D = B - 0.9 R` (positive = blue); three identity 5x5 convs without padding crop to the central 48 x 48;
   `norm` standardises each map.
2. Seven dense layers hold analytic shape templates, drawn from formulas at 7 sizes and 18 orientations:
   ellipses with Gaussian and exponential profiles at axis ratios for round (1.0, 0.88), in-between (0.76-0.54),
   cigar (0.45, 0.37) and thin edge-on (0.28-0.14) shapes; thin exponential disk + central bulge; two separated blobs
   (merger); round disk + narrow central bar. Each template is made zero-mean and unit-norm by itself alone, so each
   dense output is the Pearson correlation between the image and that shape.
3. `reshape` + `maxpool` over each family gives the best fit per family. Small dense layers and an `add` place these
   seven numbers plus a centre-weighted colour cue into one feature vector.
4. A 10-unit ReLU rule layer (clipped colour, max of round/in-between fit) and a final dense layer of hand-chosen
   rules give the class scores: best-fitting shape family wins; blue pushes towards spirals / bulgeless edge-on and
   yellow-red towards smooth classes and edge-on with bulge; a constant "nothing fits" score stands for disturbed;
   small priors reflect general knowledge that cigar-shaped galaxies are rare.

**Weights.** Every number is typed or computed from a formula in `build.py` (ellipse / disk / bar / double-blob shape
models, a Gaussian window, hand-chosen gains and thresholds). No training, fitting or statistics of the training
images were used; the training images were only viewed as montages to understand the classes.

**Status.** Time ran out before the design could be refined: gains, thresholds and priors are first guesses from
general reasoning and were not tuned.

**Checks used: 1 of 5.** Check 1 (100 training images, accuracy 0.30) showed that the network runs end to end and
that round smooth, loose spiral and bulgeless edge-on galaxies are mostly right, while the disturbed, barred and
edge-on-with-bulge rules never fire (edge-on with bulge mostly lands in in-between smooth) and cigar galaxies go to
in-between or loose spiral. Nothing was changed after this check.
