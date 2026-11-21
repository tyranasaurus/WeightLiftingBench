# Method

## Architecture

```
image (3, 32, 32)
  -> norm                    per-channel normalization
  -> conv 1x1 out=5          colour features (Red, Blue, Yellow, Bright, Dark)
  -> relu                    thresholded by negative biases
  -> avgpool 4x4 stride=4    coarse 8x8 colour grid
  -> reshape (320)
  -> dense out=43            class scores
```

Total weights: ~14k.

## Colour features (the 5x3x1x1 conv)

Per-channel normalisation gives R, G, B channels each with mean 0 and std 1.
The 1x1 convolution then forms five balanced colour features:

- `Red = Rn - Gn - Bn` (positive only where R dominates both G and B)
- `Blue = Bn - Rn - Gn`
- `Yellow = Gn - Bn` (no green signs in GTSRB, so this picks out yellow)
- `Bright = Rn + Gn + Bn`
- `Dark = -(Rn + Gn + Bn)`

Negative biases (-0.5, -0.5, -0.4, -1.5, -1.5) threshold out the small residue
that per-channel normalization leaves at mid-saturation pixels, and the ReLU
keeps only pixels whose colour clearly stands out.  After a 4x4 average-pool
each feature becomes an 8x8 map that tells where red, blue, yellow, bright or
dark pixels concentrate in the image.

## Dense layer templates

One template of shape (5, 8, 8) per class.  Each template is hand-written from
general knowledge of German traffic signs:

- **Speed limits / prohibitions (0-5, 7-10, 15, 16)**: red at the ring and
  side-ring cells, negative at the centre, bright at the centre; no-passing
  and no-trucks have dark weight at the centre (car / truck icons).
- **End-of-limit signs (6, 32, 41, 42)**: negative weights on all saturated
  colours (gray with diagonal stripes has no red / blue / yellow).
- **Priority at next junction (11)** and **warnings (18-31)**: red on the
  upright-triangle edges, strongly negative inside the triangle interior and
  at the frame's top corners (which would otherwise be red for a disc sign).
- **Priority road (12)**: strong positive on yellow in the diamond region --
  the only class with substantial yellow.  Red is also weakly rewarded in the
  diamond because yellow pixels fire our red feature too, and red is strongly
  penalised on the ring (which belongs to disc signs).
- **Yield (13)**: red on the inverted-triangle edges, negative inside and at
  the frame's bottom corners.
- **Stop (14)**: red filling centre, mid, and ring; strong negative on yellow
  so priority road (which also fires red at centre) cannot win here.
- **No entry (17)**: red concentrated in specific hotspots above and below
  the horizontal white bar; strong white at the bar itself.
- **Mandatory (33-40)**: blue inside the disc (centre, mid, ring) with a
  strong negative at the corners; positive on bright pixels only at that
  class's arrow mask (e.g., diagonal arrow for keep-right, ring of arrows
  for roundabout), with a negative penalty on bright pixels elsewhere in
  the disc so bright-centred red signs don't win here.

A soft `log(freq / mean_freq)` bias is added so, within a visual group, the
more common class wins ties.

## How the weights were produced

Nothing in the training data was used.  The build script contains only shape
masks (radial bands, triangle edges, diamond outline, horizontal bar, arrow
glyphs) derived from the sign geometry, the colour-feature formulas and
threshold biases, and the per-class template combinations that reflect each
sign's visual signature.
