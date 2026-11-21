# ASL letter classifier — method

## Architecture

```
image (1,28,28)
  → norm                                                    (1,28,28)
  → conv sobel_x (fixed Sobel kernel, 3x3, pad 1)           (1,28,28)
  → conv dup (W=[[+1],[-1]], 1x1) → ReLU → conv abs (W=[1,1])  (1,28,28)   = |sobel_x|
  → conv sobel_y (fixed Sobel kernel, 3x3, pad 1)           (1,28,28)
  → conv dup → ReLU → conv abs                              (1,28,28)   = |sobel_y|
  → add                                                     (1,28,28)   edge magnitude
  → conv threshold (W=1, b=-0.4, 1x1) → ReLU                (1,28,28)
  → conv templates (24 outlines, 20x20, pad 0)              (24, 9, 9)
  → maxpool 9x9                                             (24, 1, 1)
  → reshape                                                 (24,)
```

Total weights: 9,660.

## Why edges rather than the raw image

A hand produces a strong gradient at its boundary with the background regardless
of whether the hand is lighter or darker than the background, and whether the
hand fills a small or large fraction of the frame. Using
`|sobel_x| + |sobel_y|` as the hand-mask signal therefore avoids the failure
mode where `|normalised|` gives similar magnitudes to hand and background
pixels (which happens whenever the hand fills about half the image).

## How the templates were produced

I drew a canonical silhouette for each of the 24 ASL letters at 20×20
resolution using geometric primitives (ellipses, rectangles, discs, rings,
arcs) chosen from my general knowledge of the signs:

- **Upright signs** (A, B, D, E, F, I, K, L, M, N, R, S, U, V, W, X, Y) share a
  palm+wrist shape at the bottom of the template plus class-specific finger
  geometry at the top (number of fingers, spacing, bent vs straight, thumb
  position, etc.).
- **Sideways signs** (G, H, T) place the palm on the left and extend finger
  rectangles to the right.
- **Downward signs** (P, Q) place the palm at the top and extend fingers
  downward.
- **Round signs** (C, O) use an arc and a ring respectively.

Each silhouette is then eroded by 2 pixels and the difference `silhouette XOR
eroded` gives a 2-pixel-thick outline — the expected location of edge
responses when a hand in that pose is centred on the template.

Each template is converted to size-invariant weights: `+1/P` on the P outline
pixels and `-1/N` on the complementary region inside a fixed bounding box of
plausible hand positions. The convolution therefore computes

```
mean(edge magnitude along the drawn outline) -
mean(edge magnitude elsewhere inside the bounding box)
```

which is high only when a hand boundary in that specific shape lies in the
template's window. The size-invariant normalisation keeps scores comparable
across classes with different silhouette perimeters.

## Spatial tolerance

A 9×9 max-pool over the 20×20 template convolution output allows up to ±4
pixels of translation per class, absorbing the hand-centering variation in
the dataset.

## Rules compliance

All weights are produced from code in `build.py`. The Sobel kernels, dup/abs
helpers, and threshold bias come from standard image processing, not any
training-set statistic. Each class template is generated as a single drawn
image and normalised using only itself (positive-pixel count and anti-region
pixel count of the same template); no statistic is taken across multiple
templates and no training image or label influences any number.
