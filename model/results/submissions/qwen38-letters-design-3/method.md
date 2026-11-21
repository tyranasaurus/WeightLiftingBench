# Method

## Architecture

The network is a hand-designed HOG-like front end followed by a synthetic-template bank
readout.

1. **Illumination high-pass** (`hp`): one 15x15 conv whose kernel is `delta - Gaussian(sigma=3.5)`,
   a formula. Photos carry letter-irrelevant low-frequency luminance (exposure vignette,
   shading ramps, smooth background); a second-derivative-style high-pass removes it while
   keeping edges, bars and creases.
2. **Oriented bar filters** at three scales (`e1,e2,e3`): 6 orientations x 2 polarities of the
   oriented second derivative of a Gaussian (sigma = 1.2 / 2.2 / 4.5; the coarsest at stride 2).
   Bar (line) detectors, not edge detectors, on purpose: they respond to the medial lines of
   fingers, creases and finger gaps - the letter-specific structure - and are blind to linear
   shading gradients. `relu` + a 1x1 conv (`m1,m2,m3`) sums each polarity pair, giving
   polarity-insensitive oriented-bar-magnitude maps (the hand is not always lighter than the
   background).
3. **Spatial pyramid** (`a2,b2,c2`): overlapping avgpool blur + subsample to 14x14, 8x8 and 5x5
   per orientation - a coarse HOG-like descriptor at three tolerances to misalignment.
4. **Template bank readout** (`dA,dB,dC` -> `add` -> `reshape` -> `maxpool` -> `reshape`):
   each of the 24 letters is drawn by a parametric hand model (palm ellipse, finger capsules at
   typed angles/lengths, folded-finger block with creases, thumb, wrist; right hand, palm to the
   viewer, thumb on the viewer's left, exactly as the ASL alphabet is held) in 360 poses
   (4 scales x 9 translations x 5 rotations x 2 finger-thickness styles). Every drawing is run
   through the same front end and becomes **one template row**, normalised by itself alone
   (hand-typed spatial mask that ignores the wrist strip and frame border, then per-channel
   mean removal, then unit L2 norm, per pyramid level). A template's score is the sum of its
   three per-level cosines; a class score is the **max** over its 360 template scores
   (template-bank matching implemented as reshape + maxpool over the variant axis).

Total 14,806,324 weights, all produced by `build.py` from formulas and typed shape knowledge.

## How the weights were produced

`build.py` reads no files. Filter kernels come from closed-form Gaussian derivatives; the
mixer/high-pass weights are 0/1 or formula constants. The dense readout rows are the
self-normalised front-end responses of the synthetic hand drawings described above - each
generated image contributes exactly one template, normalised by itself alone; no statistic
(mean, prototype, variance, ...) is ever taken across generated images, and the class
aggregation is a hand-chosen max over each class's own templates. Hand-shape parameters
(finger angles, lengths, thumb placement per letter, C opening to the viewer's right, G/H
pointing to the viewer's right, Q pointing down, etc.) were typed from general knowledge of
hands and the ASL alphabet, after looking at training montages only to confirm which
viewpoint/handedness the photos use. No gradient descent, no fitting, no training-set
statistic, and no number read off any particular image enters the weights.

The five allowed training runs were used only to verify the network executes and to diagnose
failure modes (degenerate template dominance), which motivated the design changes above
(high-pass illumination removal, bar filters, mask, pose coverage); no weight value was
chosen by scoring variants.
