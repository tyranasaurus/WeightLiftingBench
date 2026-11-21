# Method

The network is a hand-designed, three-stage feature extractor and linear
score combiner; every weight comes from general knowledge of image statistics,
published skin-tone colour ranges, and what everyday photos of people look
like. No training or fitting of any kind was used, and no training image,
crop or descriptor of one is stored in the weights.

**Stage 1** (one 7x7 conv, 12 channels): luminance and three Gaussian blurs of
it (sigma 0.8 / 1.6 / 3.2), plus the colour-opponent channels u=R-G, w=R-B,
v=R+G-2B, gb=G-B, raw and Gaussian-blurred.

**Stage 2** (about 90 thin layers: 1x1 convs, relu, negation convs, adds, one
13x13 maxpool): rectified, bounded cue maps. Bands and gates are built from
trapezoid bump functions (sums of four shifted relus) and channelwise
`min(a,b) = a - relu(a-b)`; erosion is negate -> maxpool -> negate. The cues:
a soft skin-hue mask (bands on u, w and brightness from published skin RGB
ranges, combined by min); the eroded skin mask (interior of large *flat*
skin-hued expanses such as wood, sand and walls); fine-scale contrast energy;
texture inside skin regions (faces and limbs are textured, flat skin-hued
surfaces are not); dark spots and dark blobs relative to their surround;
9x3 vertical means of dark-spot maps (standing figures are vertical dark
bars against their background); dark-above-skin (hair or hat above a face);
blue-dominant, green-dominant, dark, bright and saturated pixel fractions;
and horizontal-edge energy.

**Stage 3** (one full-image conv per cue, then one add): each cue map is
spatially averaged - uniformly, or with a centre/edge pattern for the
centre-sharpness (portrait) cue, or over the bottom half for the
tables/floors/sand cue - and mapped to the two class scores.  Each readout
subtracts the value the cue takes for a typical everyday photo (estimated
from general knowledge of image statistics), so only deviations from typical
count as evidence, and applies a hand-chosen confidence weight.  Positive
weights (person evidence): textured skin, blue sky/water, centre sharpness,
fine texture, dark tones, vertical dark figures, dark figures inside bright
regions (skiers/surfers/sailors against sky, snow and water), heads; negative
weights: flat skin-hued expanses, high saturation (food/object close-ups),
overall and bottom-half skin-hue area, greenery.  A small bias (+0.04 toward
person) reflects the near-balanced class prior; its value was placed with the
help of the five allowed full-network checks, which also confirmed the final
configuration (57.7% on the 20,000 training photos).

The deep chains of narrow trapezoid bands and min operations are exactly the
logic the task needs; they also give the network very sparse gradients, so
the same architecture trained by gradient descent from random initialisation
makes little progress (most units sit in flat regions of the piecewise-linear
functions and receive zero gradient).
