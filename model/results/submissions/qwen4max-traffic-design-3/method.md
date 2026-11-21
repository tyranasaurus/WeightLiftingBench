# Method

**Idea.** The network is a hand-written *render-and-match* recogniser. I drew procedural
models of the 43 German signs from general knowledge of the signs (red ring of a given
width, rounded warning triangle, yellow diamond, red octagon, blue disc, DIN-style digits
as stroke models, and each warning pictogram as strokes/discs/polygons). Each drawing is
rendered, warped into a few scale/rotation/aspect variants, pushed through exactly the
same feature filters the network uses, and turned into one conv kernel. No training image,
label, statistic, fit or measurement enters any weight.

**Front end (`feat` + `fn`).** One 11x11 conv computes seven analytic channels: the three
colour-opponency combinations R-(G+B)/2, G-(R+B)/2, B-(R+G)/2 (invariant to additive grey
light shifts), luminance, and three Gaussian high-passes (sigma 0.85 / 1.7 / 3.2). `norm`
then standardises each channel over the image, which removes the multiplicative lighting
scale, so very dark and washed-out photos land in the same range.

**Two banks.**
* *Shape* (`sconv`, 144 kernels of 30x30): eight outer-shape/colour family models
  (red ring, no-entry disc, stop octagon, yield triangle, warning triangle, priority
  diamond, blue disc, white end-of disc), 18 variants each. Colour-heavy channel gains.
* *Content* (`cconv`, 774 kernels of 28x28): the inner symbol of every class, masked to
  the sign's inner field so the border does not leak in. Its front end (`cA`/`cBp`/`cBr`/
  `cB`/`cin`/`cinn`) turns the high-passes into *rectified, blurred ink-density maps*
  (darkness at two scales, brightness for white arrows on blue) — these are ~0 on a blank
  interior, which is what makes the "no vehicles" rule work, and they tolerate small
  misalignment.

Each kernel is one generated drawing normalised **by itself alone**: per channel, subtract
its own masked mean, divide by its own L2 norm, apply a hand-set channel gain, renormalise
by the sum of the active gains. Nothing is averaged or measured across drawings.

**Readout.** `maxpool` over the 3x3 conv positions gives +-2 px translation tolerance;
`reshape` + `maxpool` takes the max over each class's 18 variants. Score(k) =
2 * shape(family(k)) + content(k), a hand-set rule table. Yield and priority-road are
unique by shape/colour, so they get a fixed offset instead of a content term. Class 15
("no vehicles") is decided by an explicit rule: it wins when the best symbol score in the
red-ring family stays below a hand-chosen threshold (0.30), i.e. when the interior is blank.

**Checks used.** Three of the five allowed runs: 0.4383 -> 0.5076 -> 0.4896. The submitted
answer is the second one (0.5076); the third experiment (finer *signed* detail features)
regressed, so it was discarded. What the checks taught me is recorded in the conversation:
the first showed the shape/colour stage was sound but faint pictograms and digits were not
matching; the second confirmed rectified ink maps fix most of that; the third showed that
signed fine detail is too sensitive to the real photos' blur and aliasing.

`build.py` is self-contained, reads no files, and regenerates both answer files from its
own source in about 8 seconds.
