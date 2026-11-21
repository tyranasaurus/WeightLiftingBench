# Method

## Architecture

Seven layers, 148 weights total:

```
image (3, 96, 96)
  c1 : conv  3  -> 2   (1x1, pad 0)        – colour-space change
  r1 : relu
  c2 : conv  2  -> 6   (3x3, pad 1)        – gradients and smoothing
  r2 : relu
  p1 : avgpool  (48, 96) stride (48, 96)   – upper / lower half pooling
  fl : reshape to (12,)
  out: dense 12 -> 2                       – class scores
```

## Weights (all chosen by hand)

**c1** — turns the RGB image into two scalar maps:
- channel 0: luminance `Y = 0.299 R + 0.587 G + 0.114 B`
- channel 1: `warm = R + G − 2B` (positive wherever the pixel leans warm/green over blue)

**c2** — six fixed filters on top of the ReLU-preserved luminance and the
ReLU-preserved warm channel:
- channels 0–3: horizontal and vertical Sobel kernels on `Y` with both signs so
  after ReLU each pair sums to the full `|∂Y/∂x|` / `|∂Y/∂y|` magnitude;
- channel 4: 3×3 box filter on `Y` – gives smoothed brightness;
- channel 5: 3×3 box filter on `warm`.

**p1** — pools the 96×96 activations into one upper-half mean and one lower-half
mean per channel. That gives six channels × two halves = twelve features.

**out** — the dense readout:
- **class 0 (no-person)** gets the warm sum plus `1.3 × (Y_top − Y_bot)`;
- **class 1 (person)** gets `2.0 ×` (sum of the eight edge half-wave features).

## Why these weights

From general knowledge of COCO-style photos:
1. Many non-person COCO photos are close-ups of warm/organic subjects (food,
   pets, warm vehicles) or wide outdoor scenes with a bright sky above and a
   darker ground below. These produce a high "warm" channel average and a
   large top-minus-bottom luminance gap — both go into the class 0 score.
2. Person photos cover more varied scenes and a human body breaks up the clean
   sky-above / ground-below pattern, so edge density is slightly higher and
   the top-bottom asymmetry is much weaker. The class 1 score is driven by
   edge density.

Weight magnitudes balance the typical scales of the three signals in natural
photos:
- Warm fraction (R+G−2B)⁺ averages ≈ 0.15 per half – unit weight.
- Mean-luminance top–bottom gap can reach 0.05–0.1 – 1.3 × weight.
- Each Sobel half-wave averages ≈ 0.016 and eight of them sum to ≈ 0.13 –
  weighted 2.0× so the edge signal sits in the same ballpark as warm.

No training data, labels, or statistics were used to set weights. Training
images were inspected only (per the rules) to understand what the two classes
look like.
