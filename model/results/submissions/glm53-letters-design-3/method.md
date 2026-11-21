# Method

The network is a hand-designed shape matcher; no weights were fitted to any data.

**Feature stage.** The image is globally standardized (`norm`), then one 25x25 conv computes
28 channels: local contrast at two scales (Gaussian-blurred box-deviation, both polarities)
and oriented Gaussian-derivative edge filters at two scales x 6 orientations (with negations).
All channels are blur-then-derive so pixel grain is suppressed. Each channel is standardized
per image (`norm`), then thresholded and capped into a bounded 16-channel map:
bright-deviation / dark-deviation (coarse + fine) and unsigned orientation energy; a 2x2
maxpool gives a translation-tolerant copy.

**Template stage.** For each of the 24 classes I render a parametric hand-shape model typed
from general knowledge of the ASL alphabet — palm ellipse, finger capsules at anatomical
spacing with visible gaps, fist blobs, thumb position, crease lines, wrist entering from the
bottom — sized so the hand fills most of the 28x28 frame. 36 pose variants per class
(mirror x rotation x scale x finger-spread jitter) are rendered as realistic pseudo-photos
(soft boundaries, shallow creases, flat background + grain) and pushed through the identical
feature pipeline. The resulting feature stacks, per-channel mean-centered and unit-L2
normalised, become conv kernels. Two matchers: full-resolution 24x24 and pooled 14x14, both
slid with stride/padding, maxpooled over position and over the 36 variants (reshape +
maxpool over the variant axis). Mean-centering the kernels makes the score a background-
suppressed correlation: uniform feature energy anywhere contributes exactly zero.

**Readout.** The two per-class scores are summed (`add`) and passed through an identity dense
layer; classification is argmax.

**How parameters were chosen.** Filter scales, thresholds, caps and pose variants were chosen
by reasoning about hand geometry and grain statistics, and tested only on a self-made
synthetic generator (the same renderer plus modelled backgrounds, polarity, noise and
rotation) - never on the training set. The two allowed training-set runs were used only as
sanity checks of overall behaviour.
