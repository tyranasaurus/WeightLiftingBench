# Method

**Architecture.** Two small branches whose 10 outputs are added.
- Colour branch: a 1x1 conv + relu computes soft per-pixel indicators (bluer-than-red, bright, dark, greener-than-blue,
  R>G>B "warm", saturated red, and absolute channel differences for saturation); a second 1x1 conv + relu ANDs them into
  eight "materials" (bright blue sky, dark blue water, green vegetation, brown/tan, red paint, white, mid grey, dark).
  These are average-pooled on a 4x4 grid and read out by a dense layer.
- Edge branch: a 3x3 conv with Sobel derivatives of brightness (both signs, relu), average-pooled on a 3x3 grid, read out
  by a dense layer using surround texture, top texture and horizontal-minus-vertical edge strength.

**Weights.** All typed by hand in `build.py` from general knowledge: colour thresholds for sky/grass/earth/road, and a rule
table giving, per material and per image band (top / middle / bottom / centre), how much it speaks for each class
(e.g. blue everywhere -> airplane, blue at the bottom -> ship, green at the bottom -> horse/deer, grey at the bottom or
red paint -> automobile/truck, brown/tan -> cat/dog/frog/deer/horse, smooth surround -> airplane/bird/ship).
No gradient descent, fitting or statistics of the training images were used; the training set was only viewed as
contact sheets (two sheets per class) to see what the classes look like.

**Check 1** (every fifth training image, 10,000 images): accuracy 0.2011. What it showed: the network runs and is
about twice chance, but "truck" is predicted for more than half of all images, so the edge-texture evidence is scaled
too strongly against the colour evidence; bird and deer are almost never predicted. This was not corrected.

**Status.** Time ran out before the planned refinements (object/shape cues, conjunction units) and before the rule
table could be revised after a check; see `checks.log` for the checks actually run.
