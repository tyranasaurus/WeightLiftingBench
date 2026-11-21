# Hand-designed MNIST classifier

## Architecture
```
image (1, 28, 28)
  → norm                                 (per-channel zero-mean / unit-std)
  → conv  60 filters, 26x26 kernel        (60, 3, 3)
  → maxpool 3x3 stride 3                  (60, 1, 1)
  → reshape to (10, 6, 1)
  → maxpool [6, 1] stride [6, 1]          (10, 1, 1)
  → reshape to (10,)
```

The 60 conv filters are 10 class × 6 hand-drawn stroke-template
variants per class. The first maxpool allows a one-pixel shift in any
direction (shift tolerance for off-centre digits). The second maxpool
picks the best-matching variant for each class, giving the 10 scores.

## How the templates are produced
`build.py` has no training-data dependency. For every (digit, variant):

1. Draw canonical strokes on a 26x26 canvas with line and arc
   primitives, using positions, radii and angles chosen by hand from
   general knowledge of how each digit is written (ovals for 0 and 8,
   two right-side bumps for 3, top horizontal + diagonal for 7, etc.).
2. Draw an *anti-mask* on the same canvas: a small set of strokes at
   places that distinguish this digit from its common confusers
   (e.g. a 7's anti-mask has a horizontal where a 2's base would be).
3. Dilate the positive strokes by three pixels and keep what the
   dilation adds (the *shadow*) to penalize nearby misplaced ink.
4. Combine: `T = positive - 0.25 * shadow - 0.5 * anti`.
5. Normalize that single template by itself alone: subtract its own
   mean and divide by its own L2 norm.

No statistic is taken across templates, no training image is read, no
number comes from a measurement of generated images. Every parameter
in the templates (coordinates, widths, angles, variant counts) is
typed in by hand based on general digit knowledge.

## Variants
Six per class covering different stroke widths, slants and shape
details (e.g. open-top vs triangle-top 4, European vs plain 7, serif
vs plain 1). Max-pooling over variants makes the class score the best
match across these written styles.
