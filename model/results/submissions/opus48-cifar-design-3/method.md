# Method

A hand-designed **colour voting network**. No training, no fitting, no statistic of the
training data is used; every weight comes from formulas and hand-chosen rules based on general
knowledge of what the ten classes look like in small photos. I used the training set only to
*look* at examples (montage, class-mean thumbnails) and for the 5 allowed accuracy checks.

## Architecture
```
image(3,32,32)
  color : 1x1 conv -> 4 channels  LUM, BLU(blue-excess), GRN(green-excess), RMG(red-excess)
  poolc : avgpool 8x8/8x8 -> 4x4 grid of region colours
  det   : dense -> 9 evidence detectors ; relu
  out   : dense -> 10 class scores (votes)
```
All convolution weights are fixed formulas: luminance `0.299R+0.587G+0.114B` and the three
colour-opponent axes. Pooling to a 4x4 grid gives region colours, from which the dense `det`
layer computes 9 ReLU **evidence detectors**, each a self-calibrating comparison (opponent
channels are zero-centred, so ReLU at 0 is a natural threshold):
bluesky, bluewater, bright-top, green-ground, green-everywhere, animal-on-grass
(`green surround − green centre`, i.e. a non-green body on green grass), red/warm object in
the centre, warm-overall, dark-overall.

The `out` layer is a hand-written **vote table**: each class receives a balanced budget of
positive votes from the detectors that support it (e.g. ship ← blue water + blue sky; frog ←
green-everywhere + dark; deer/horse ← animal-on-grass + green-ground; airplane ← blue sky +
bright; cat/dog ← warm + warm-centre) and a few negative votes from detectors that contradict
it. Because every detector output is non-negative and bounded and every class has a similar
total vote budget, no class can dominate — the prediction is decided by which detectors fire.

## Why no texture/edge features
I experimented with Sobel-gradient "texture" detectors, but their magnitude cannot be set
without measuring the data, and a zero-padding border artefact plus the unknown real texture
scale made them collapse the predictions onto the high-texture classes. Since I may not tune
constants on the data, I removed them and kept only the robust, self-calibrating colour cues.

## Rules compliance
`build.py` reads no files and derives every number from code (verified by
`check_rebuild.py`: max relative difference 0.0). Constants are either formula coefficients
(luminance/opponent mixes) or hand-chosen perceptual values (mid-grey 0.5/0.46 references,
detector gains reflecting the natural size of each formula, and the vote table), none copied
or measured from training images.
