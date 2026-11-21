# Method

## Architecture

```
image (1,28,28)
 └─ feat: conv 1→9, 9x9 (formula filters) → relu                                  (9,28,28)
      ├─ fine:   conv 9→9 per-channel Gaussian σ=1.2, stride 2 → (9,14,14)
      │          → reshape (1,9,196) → norm → fine_match: dense → (35000,)
      └─ dilate: maxpool 3x3 stride 1 → coarse: conv 9→9 per-channel Gaussian σ=1.6, stride 2
                 → (9,13,13) → reshape (1,9,169) → norm → coarse_match: dense → (35000,)
 add(fine_match, coarse_match) → reshape (1,10,3500) → maxpool [1,3500] → reshape (10,)
```

35,000 templates (3,500 per class), 115 M weights, all biases zero.

## Where the numbers come from

* **feat**: channel 0 is a Gaussian blur (σ=1, gain 0.5). Channels 1–8 are oriented *line*
  detectors at 8 angles: a Gaussian along the stroke times a negative second derivative of a
  Gaussian across it (a zero-mean Mexican hat whose positive lobe is about 2.8 px wide, so it
  fires on the centre line of a stroke of any usual thickness). Closed-form formulas only.
* **fine / coarse**: per-channel Gaussian pooling kernels (formulas). The coarse branch first
  dilates with a 3x3 max-pool so that a stroke 1 px off, or thicker, still matches; the fine
  branch keeps small gaps and loops apart (open 5 vs 6, looped 2 vs 8).
* **norm** layers standardise each branch's feature vector, so each dense layer outputs a
  Pearson correlation between the image and a template; the add sums the two scales and the
  maxpool picks each class's best template.
* **Dense weights = template bank.** For every digit I typed a set of pen-path *styles* from
  general knowledge of handwriting: 0 ellipse (egg-shaped, optional inner hook); 1 bar / flag /
  base; 2 plain, looped, Z-like, S-bodied, curled head; 3 round or flat-topped; 4 open, closed,
  rounded-left; 5 joined, separate bar, sloppy stem, one-stroke S; 6 hook + loop with a straight
  or round stem; 7 plain, crossed, hooked; 8 Lissajous figure-eight with a smaller, offset top
  loop, or two stacked rings; 9 loop with a straight, curved or open tail. Each render draws its
  parameters (arc radii, hook sizes, slant −0.25..0.55, rotation, aspect ratio, stroke width
  1.5–4.2 px, a smooth sinusoidal warp) from hand-chosen ranges with a fixed seed, fits the
  stroke into a 20x20 box and centres it by its centre of mass, as the data description says the
  images were made. The render goes through the network's own feature layers, each branch's
  vector is standardised **by its own mean and std alone** and divided by its length, and that is
  one row of the dense weight. Nothing is averaged, fitted, scored or measured across renders; no
  training image, crop or statistic is used anywhere. `python3 build.py OUT_DIR` regenerates all
  files from the source alone.

## The five checks (10,000 training images each)

| check | version | accuracy | what it taught me |
|---|---|---|---|
| 1 | one Gaussian-pooled scale, first style set | 84.4 % | 2/5/8 styles under-covered; simple 1-templates attract errors |
| 2 | broader styles, one dilated scale | 85.5 % | dilation helps stroke digits (4,7,9) but merges open and closed shapes (8→2, 5→6) |
| 3 | fine + coarse scales | 85.5 % | the 8 problem is in the templates, not the pooling |
| 4 | Lissajous 8s, arcs that never close, S-style 5 | 87.3 % | 8s up from 44 % to 59 %; errors left mostly 2→3, 5→6, 8→2/9 |
| 5 | + curled 2s, higher 3 arc, shorter 6 hook; **disjoint** 10k subset | 87.3 % | same level on unseen images: no regression, no overfitting to subset 1 |

Weak classes remain 2, 3 and 8 (the most variable, curvy digits); the stroke-dominated digits
(0, 1, 4, 6, 7, 9) are at 91–98 %.
