# Method

**Architecture (weights designed by hand, no training of any kind).**
The network is a multi-scale correlation matcher between a photo and a library of
hand-drawn shape templates for the 24 static ASL letters.

1. **Front end** (`f3` + `norm`): two zero-mean Difference-of-Gaussians "center-surround"
   filters (scales 1.8/4.8 and 3.2/8.5 px) are applied to the raw image. DoG filters respond
   to a compact central object (the hand) regardless of its absolute brightness or polarity,
   and respond exactly zero to smooth background gradients, which the data summary showed to
   be large (corner-to-corner background spread median 0.22, while the hand-background contrast
   is only ~0.15). Each channel is standardised per image by the `norm` op.

2. **Template library**: each letter is drawn in code from general hand knowledge - a palm
   ellipse, finger capsules at typed angles, thumb capsules, annular arcs for C/O, holes for
   O/D/F, thin slots for creases between fingers - using per-class parameters I chose by
   studying printed renderings of the training images (appearance notes: which letters point
   sideways/down, where the thumb sits, the stair-step of F, the hole of D, the crossed tips
   of R, etc.) and coarse radial/axial extent summaries (fists span x~5-24, y~5-20). Each
   class gets 16 pose variants: two hand-posed drawings, their mirrors (chirality coverage:
   e.g. L thumb-right, Y thumb-left, C opening right), rotations, and seeded elastic-jitter
   clones, to cover the natural pose variation of real signers.

3. **Matching**: the photo's feature maps (7x7 and 14x14 pooled) are correlated against every
   template's own feature maps (computed in build.py through the identical filter pipeline) by
   strided `conv` layers whose kernels are the unit-normalised template features, each stored
   twice with opposite sign. `maxpool` takes the best shift (translation tolerance +-4 px) and
   the best polarity (hand lighter or darker than background - the data is mixed 16%/84%), and
   a final `dense` averages the 16 variant scores per class. Kernel values are tapered towards
   the frame edges and strongly tapered in the bottom sleeve zone, where all classes look alike
   and only background/wrist noise lives.

**Why this shape.** My first designs (Gabor edge energy, background-level segmentation,
raw-image correlation) each scored 8%, 8%, 4% on the training set: the confusion tables showed
predictions collapsing onto whichever templates resemble the *noise* structure of real photos
(strong interior knuckle/finger shading of +-0.3-0.6, background patches, sleeves). The final
design counters each identified failure mode: DoG features cancel background gradients;
coarse 7x7 pooling averages away interior shading; the taper removes the sleeve/background
border; the mean over a pose ensemble suppresses per-variant noise maxima; and the +/- pair
handles polarity. All weights come from closed-form filters and my own drawn templates;
build.py reads no files and uses a fixed seed.

**Self-test**: on a synthetic photo generator calibrated to the measured image statistics
(background spread, hand contrast, interior shading, sleeves, +-4 deg rotation, elastic pose
warp), the design scores ~13-15%, versus 4.2% for random guessing; templates match their own
class 315/384 under the full pipeline. On the real training set (5 allowed checks, used to
diagnose the earlier failures above) the previous iteration scored 4.9%.
