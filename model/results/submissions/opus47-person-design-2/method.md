# Method

A small hand-crafted network that classifies 96x96 photos as person / no-person from edge density.

## Rationale

People have faces, limbs, hair, and clothing — all of which contribute to high-contrast
boundaries. No-person photos tend to contain more large uniform regions (sky, walls,
water, uniform grass, large uniform animal bodies). I therefore use a single feature:
the fraction of pixels whose local brightness gradient exceeds a threshold, averaged
over the whole image.

## Architecture

```
image (3, 96, 96)
 |
 v
conv l1  3x3, 3->4, padding 1     # forward diffs of brightness in 4 directions
 |                                 # ch0 = +dx, ch1 = -dx, ch2 = +dy, ch3 = -dy
 v
relu r1                            # after relu: ch0+ch1 = |dx|, ch2+ch3 = |dy|
 |
 v
conv l2  1x1, 4->1                 # sum(ch) - 0.1 = (|dx| + |dy|) - 0.1
 |
 v
relu r2                            # per-pixel edge intensity above 0.1 threshold
 |
 v
avgpool 96x96 (global)            # average edge intensity over the whole image
 |
 v
reshape (1,)
 |
 v
dense 1->2                        # threshold edge_density at 0.035 for person
```

## How the weights were chosen

Every weight is a hand-set constant.
- Layer 1 (3x3 conv) uses symmetric kernels for forward differences on the grayscale
  brightness channel (1/3 share per RGB channel).
- Layer 2 (1x1 conv) sums the four signed gradients to get |dx|+|dy| and subtracts 0.1.
- The dense layer thresholds the resulting average at 0.035. The multiplier 100 just
  sharpens the decision.

Both thresholds (0.1 for per-pixel gradient and 0.035 for image-wide density) come from
general knowledge about natural photos: a 25/255 gradient step is a common "strong edge"
threshold in 8-bit image processing, and typical photos have a few percent of their
pixels crossing that step. No training label or training-image statistic was used to
derive any of the weights.
