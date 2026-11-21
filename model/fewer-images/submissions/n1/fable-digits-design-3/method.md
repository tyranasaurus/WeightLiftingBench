# Method

No training, fitting or data statistics are used anywhere: `build.py` reads no files, and every weight comes
from formulas or from pen paths typed by hand from general knowledge of how digits are written.

## Architecture (21 layers, 119.2M weights)

Two fixed feature branches read the image, each ends in a template layer, and the class score is the best
template response of that class.

1. **Oriented edge energy.** `g` is a 7x7 conv with 16 first-derivative-of-Gaussian filters (sigma 1, directions
   22.5 degrees apart); after a relu, the 1x1 conv `mix` forms `|g.u| - |g.u_perp|` for 8 unsigned orientations
   and a second relu keeps the positive part (tuning that vanishes 45 degrees away). `pool_o` is a Gaussian
   (sigma 2) conv with stride 3 that reduces each channel to 8x8 cells, giving 512 numbers that tolerate
   stroke shifts of about two pixels.
2. **Concavities.** The dense layer `rays` integrates the ink along the 8 compass rays leaving each pixel; a 1x1
   conv + relu pair clips each sum to [0, 1] ("there is ink in that direction"). `cc_pre` + `ink` + relu mark
   background pixels that are closed in all 8 directions (a hole) or in the 5 directions facing away from
   up / down / left / right (a cup). `pool_c` pools hole and (cup - hole) the same way: 320 numbers.
3. **Templates.** Dense layers `t_or` (512 -> 137,000) and `t_cc` (320 -> 137,000) are added; row k of the two
   together is one template, so the sum is the inner product of the image's features with template k.
   Templates are stored class by class (13,700 per digit); a reshape and two maxpools take the largest response
   within each class, which is the class score (nearest template by cosine similarity, since the norm of the
   image's own features is the same for all templates).

## Where the weights come from

* Front end: formulas only (Gaussian derivatives, Gaussian pooling, ray sums, thresholds chosen by hand).
  The relative weight of the concavity branch (0.5) was set from a calculation on an ideal stroke, written in
  the comments of `build.py`. (A debugging printout of feature norms for a handful of drawings showed that my
  first version of that calculation had left out the attenuation of thin strokes by the derivative filter; I
  redid the calculation by hand rather than taking any value from the printout.)
* Templates: each digit is drawn as pen strokes along Catmull-Rom curves through knots I typed for 2 to 4
  writing styles per digit (e.g. 1 with or without flag and foot, 2 with or without a loop, open or closed 4,
  7 with or without a crossbar, 9 with a straight or hooked tail). A fixed-seed random generator varies
  proportions, slant, rotation, smooth bending, knot sloppiness and pen width. Each drawing is laid out as the
  task describes its images (ink box scaled to 20 pixels, centre of mass at the centre of the frame), passed
  through the fixed front end, and divided by its own norm; that vector is one row of the template layers.
  No statistic over several drawings is used: every drawing is its own template, normalised by itself alone.

## Checks

Four of the five allowed runs on the ten training images were used, each only to confirm that the network
works on real images (10/10 each time): after the first full build, after widening the range of drawing
styles, after fixing the final front-end constants, and on the final build. Runs on generated drawings were
used only to confirm that the network computes what was intended and to time it; no number was chosen from
them. `check_rebuild.py` confirms that `build.py` alone reproduces the submitted weights (about 2 minutes of
CPU time).
