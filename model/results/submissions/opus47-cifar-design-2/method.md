# Method

## Architecture

```
image (3, 32, 32)
  -> c1    : conv 3x3, 16 feature maps, stride 1, pad 1
  -> r1    : relu
  -> p1    : avgpool 8x8, stride 8        ->  (16, 4, 4)
  -> flat  : reshape                      ->  (256,)
  -> dense : dense                        ->  (10,)
```

All weights are hand-written in `build.py`. No training-set data, no class
averages or other fitted statistics are used.

## conv1 : 16 hand-designed low-level features

Each 3x3x3 filter is written by hand from a short formula:

- 0, 1, 2  raw R, G, B at the centre pixel
- 3        blueness  `2B - R - G`      (sky, water)
- 4        greenness `2G - R - B`      (grass, foliage)
- 5        warmth    `R + G - 2B`      (brown fur, skin, earth)
- 6        redness   `2R - G - B`      (red paint, flowers)
- 7        bright    `brightness - 0.45`  (ReLU threshold)
- 8        white     `brightness - 0.75`
- 9        dark      `0.35 - brightness`
- 10..13   Sobel horizontal and vertical edges, both signs
- 14       centre - surround blob detector on brightness
- 15       metallic  `brightness - 0.55`

## dense : per-class rules over a 4x4 regional grid

After pooling, every feature survives as a 4x4 regional summary.
For each class `c` the dense weight tensor `W[c, feature, y, x]` is set by
hand from a per-class rule list that encodes, from general knowledge of the
class, where and what each class is expected to look like:

- **airplane** : strong blue above and below (sky), bright/white/metallic in
  the middle rows, large negative weights on green, warm and red.
- **automobile** : metallic middle, dark tyres at the bottom, horizontal
  edges on the lower rows, little green.
- **bird** : small dark blob centred in the frame with sky above or foliage
  around; strong `F_BLOB` and `F_DARK` weights in the middle region.
- **cat** : warm centred subject with fur-texture edges; penalises blue sky
  and pure grass.
- **deer** : brown centre with green grass below, green forest around; thin
  vertical edges in the lower rows represent legs.
- **dog** : warm or dark centred animal with grass below, less "wide" than
  horse; penalises blue sky.
- **frog** : green everywhere, strong negative on blue, warm, white and red.
- **horse** : large warm subject spanning the middle, strong grass below,
  strong negative on blue sky and metallic.
- **ship** : blue above (sky) and below (water), bright/metallic hull in the
  middle, strong negative on green and warm.
- **truck** : bright or red cab on top, dark chassis below, strong
  horizontal roof edge and vertical body edges, negative on green.

The dense bias is set so that an average natural photo gives roughly the
same score to every class.  A hand-written table `E` lists expected
feature activations on a typical natural image (not measured on training
images) and the bias is `-W[c] . E`, equalising the per-class starting
score so the discriminating features decide the winner.

## How the weights were produced

`build.py` writes `weights.npz` and `network.json` directly from the formulas
and rule lists above.  No image, pixel value, class statistic, label or any
other number from the training set appears in the source.  Running
`python3 build.py OUT_DIR` reproduces the submitted files exactly.
