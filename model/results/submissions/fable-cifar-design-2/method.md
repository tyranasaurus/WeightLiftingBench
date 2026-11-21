# Method

**Architecture** (6 layers, about 1,600 weights):
`image -> conv 3x3 (11 ch) -> relu -> conv 1x1 (12 ch) -> relu -> avgpool 10x10 (3x3 grid) -> dense (10)`.

**How the weights were produced.** Everything is typed by hand in `build.py` from general knowledge of the
classes; no training, fitting or statistics of the training images are used.

- `c1` computes rectified colour-opponent differences (R-G, G-R, G-B, B-G, B-R, R-B), luminance, and
  signed Sobel derivatives of luminance.
- `c2` combines these into 12 soft per-pixel "material" maps with hand-chosen thresholds: blue (sky/water),
  green (vegetation), saturated red (paint), warm (fur/earth), white, dark, grey (low saturation: road, overcast
  sky, metal), horizontal-edge energy, vertical-edge energy, smoothness, saturation, luminance.
- The maps are averaged over a 3x3 grid (top/middle/bottom x left/centre/right).
- The dense readout is a hand-typed evidence table (class x material x row), e.g. blue everywhere and smooth
  -> airplane; blue strongest at the bottom -> ship; green at the bottom with warm tones -> deer/horse; warm and
  dark without blue -> cat/dog/frog; grey and dark at the bottom with red in the middle and strong straight
  edges -> car/truck; vertical edges -> truck. The table is centred over classes by formula.

**Status.** Time ran out before the planned richer design (scene/material mixture model) was built, so this is a
simple first version. The training images were only viewed as PNG grids to study the classes qualitatively.
