# Method

Everything in `weights.npz` is produced by `build.py` from formulas and from general
knowledge about what these ten kinds of photo look like. No training image, label,
statistic, crop or descriptor is read or used; `build.py` reads no files at all.

## Architecture (37 layers, 426,742 weights, ~10k non-zero)

Three hand-designed vision branches run on the 32x32x3 image and are concatenated
into a 66-channel trunk (concatenation is done by expanding each branch to the common
width with a 1x1 conv and then `add`, since the block set has no concat op).

1. **Colour / material branch.** A 1x1 conv emits 34 signed photometric quantities and
   their negations (R, G, B, Rec.601 luminance, the three colour opponencies, several
   luminance hinges, and gate hinges of the form `(B-R)+L-c`). After ReLU, every
   quantity and every absolute value is available as a linear combination, and
   `min(a,b) = a - relu(a-b)` lets a colour difference be *conjunctively* limited by a
   luminance range. A second 1x1 conv builds 18 soft indicators, each clipped to [0,1]
   by the `relu(z) - relu(z-1)` idiom: blue-and-bright sky, bright achromatic sky,
   blue-and-not-bright water, neutral grey water, vegetation green, dark foliage green,
   brown, tan, mid grey, dark, bright, red, yellow, yellow-green, saturated, achromatic,
   blue-dominant, white. Gains and offsets were chosen by evaluating the formulas on
   canonical colours typed from general knowledge (clear sky, sea, grass, foliage, fur,
   asphalt, car paint) and checking each indicator fires only for its intended material.

2. **Texture branch.** 16 filters from formulas: Gaussian derivatives at three scales,
   Laplacian-of-Gaussian at two scales, and real Gabor filters at four orientations and
   two scales. Each is zero-meaned and normalised so a unit-contrast aligned pattern
   gives a response of about 1, then scaled by 2.5. Both signs are emitted and summed
   after ReLU to give oriented energy |response|.

3. **Shape branch.** 14 detectors typed from geometry: a wheel as a dark annulus with a
   bright hub and bright surround, dark/bright centre-surround blobs, a coarse
   difference-of-Gaussian blob, horizontal and vertical step edges, thin dark vertical /
   horizontal / two diagonal lines (legs, masts, branches, hulls, wings), wide dark bars,
   a bright ring, and a corner. Both half-wave signs are kept separately so edge
   *direction* (bright-above-dark vs the reverse) survives.

The trunk is pooled four ways: 4x4 grid means (8px cells), global mean, global max, and
2x2 quadrant max. Four dense layers map these into 253 **named region aggregates**
(e.g. `sky_ring`, `water_layout`, `gdark_ring`, `brown_center`, `wheel_max`,
`tex_center_ring`); region weights are hand-defined averages over the grid.

## Readout

* A dense layer gives each class a hand-written linear evidence sum over the aggregates
  (scene layout + object colour + object shape/texture), with hand-set biases.
* A **ReLU rule engine** adds 132 hand-written rules. Each condition is a capped hinge
  `clip((a - thr)/width, 0, 1)`; a rule is `relu(sum(conditions) - (n-1))`, which is a
  soft AND that reaches 1 only when every condition holds. Rules encode knowledge such
  as "blue sky around a smooth rigid achromatic object -> aeroplane", "water below but
  not above, with a horizon -> ship", "brown body in dark foliage -> deer", "big warm
  body on bright grass -> horse", "yellow-green bumpy body, no sky -> frog", "wheels on
  grey -> car", "wheels plus a big flat bright box -> truck", "fur texture with no sky,
  green or water -> cat/dog". Class scores are the sum of the two paths.

Texture comparisons that gate the rules are **scale-invariant** (border vs whole image,
centre vs whole image) rather than absolute, so they do not depend on the overall
contrast level of a photo. Ship uses a water *layout* aggregate (water below minus water
above) because deep blue sky is chromatically identical to sea.

## Verification

`check_network.py` reports the network usable; `check_rebuild.py` reproduces both answer
files exactly from `build.py` alone (max relative difference 0.0). Four labelled checks
were used. The first three exposed a real bug — scalar offsets written inside the
indicator coefficient vectors were broadcast across every coefficient by numpy instead
of acting as biases, which made all 18 colour indicators read as achromatic grey. After
fixing that (constants now live only in the bias vector) and verifying the actual weight
matrices against the intended formulas on typed canonical colours, accuracy on a
stratified 20,000-image sample rose from 0.166 to 0.2195; the final round rebalanced
truck (which was winning on generic brightness), automobile, aeroplane, bird, deer,
horse and frog evidence. The fifth check, on all 50,000 images, was killed by the
environment before it printed a result, so the last measured accuracy is 0.2195 on the
20,000-image sample.
