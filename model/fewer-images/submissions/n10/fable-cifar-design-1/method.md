# Method

**Architecture** (1,890 weights): `conv 3x3 (3 -> 10) -> relu -> avgpool 8x8 -> dense (160 -> 10)`.

**Layer 1, typed from formulas.** Six channels use only the centre tap and are soft colour indicators:
blue (`B - (R+G)/2`, sky and water), green (`G - (R+B)/2`, vegetation), warm (`R - B`, fur, earth, tan),
red (`R - G`, paint), bright (`luminance - 0.7`) and dark (`0.3 - luminance`). Four channels are the two
signs of the horizontal and vertical Sobel filters on luminance, giving vertical-edge and horizontal-edge
energy after the relu. Gains and small negative offsets were chosen by hand so that an active feature is of order 1.

**Pooling.** An 8x8 average gives each feature on a 4x4 grid (rows top to bottom, side versus centre columns).

**Readout, a hand-typed rule table.** For each class and feature I typed four row weights plus a side/centre
factor, from general knowledge of the classes: airplane = blue or bright smooth surround with few edges;
ship = blue increasing towards the bottom with a horizontal hull/horizon line in the middle; horse = sky
above, green below, warm body in the centre, vertical leg edges low down; deer = green everywhere with a warm
centre; frog = olive/yellow-brown (green + warm, no blue); cat and dog = warm, non-green, non-blue scenes;
automobile = red centre, dark underside, horizontal edges in the lower half; truck = red, bright top,
strong vertical and horizontal edges throughout.

**How the weights were produced.** `build.py` contains every number; it reads no files and uses no
randomness. Nothing was trained, fitted or measured on the training images or on generated images. The
training images were only viewed as contact sheets to get a qualitative feel for the classes.

**Status.** This is a minimal version saved when time ran out; a richer design was planned but not built.
The rule table was not tuned in any way.

**Check 1 of 5 (the only one used):** 15% on the 100 training images, with 75 of 100 predicted as truck.
What it showed: the truck row is the only one with positive weights on every edge feature, so edge energy
acts as a bias towards truck on any textured photo. This was left as it is; no weight was changed after the check.
