# Method

## Architecture
```
image (3, 128, 128)
  -> mix   : conv 1x1, 3->3       (L=(R+G+B)/3, Y=R-B, B'=B-R)
  -> n1    : norm                 (per-channel zero mean / unit std)
  -> r1    : relu                 (keeps "above the mean" pixels, i.e. galaxy)
  -> p1    : avgpool 2x2 /2       -> (3, 64, 64)
  -> tmpl  : conv 64x64, 3->44    (44 hand-designed difference templates)
  -> r2    : relu
  -> flat  : reshape (44,)
  -> fc    : dense 44 -> 10
```

Total weights: 541,178 (well under 120 M). 

## Templates (all from closed-form Gaussian formulas)
- **A (5):** round difference-of-Gaussians on L at sigmas 2.5, 4, 6, 9, 13 — bandpass for round blobs of a given size.
- **B (8):** moderate elongation on L (sx=8, sy=2.5) at 8 angles minus the perpendicular — orientation-selective.
- **C (8):** thin elongation on L (sx=14, sy=1) at 8 angles minus perpendicular — edge-on shape detector.
- **D (3), E (4):** round DoG on Y and B' channels, several sizes.
- **F (4), G (4), H (4):** moderate yellow, thin yellow, thin blue at 4 angles.
- **I (4):** two offset blobs minus one central blob (merging-pair detector).

Each template is built as a *difference* of normalised Gaussians, so a uniform
background gives response zero and only galaxies of the matching shape, size and
orientation give a strong positive response (then ReLU keeps the positive ones).

## Dense classifier
Weights were set by hand from general galaxy knowledge:
- Round smooth (2, 3): compact / medium round yellow blob, strong penalty on blue.
- Cigar (4): moderate yellow elongation; penalise thin + blue.
- Barred spiral (5): yellow bar (F) + blue disk (larger E).
- Unbarred spirals (6, 7): round blue disks, tight (smaller E) vs loose (larger E).
- Edge-on (8, 9): thin shape; 8 prefers blue (H), 9 prefers yellow (G).
- Merging (1): strong two-blob (I).
- Disturbed (0): weak default + small multi-center / elongation bias.

## How the weights were produced
No training, no fitting, no statistic of the images.  `build.py` writes the
weights deterministically from the formulas above (Gaussian templates with
chosen sigmas and angles, and hand-picked dense weights).  The 5 allowed
accuracy checks were used only to look at which classes were being confused,
and the dense weights (not the templates) were adjusted from general reasoning
about which features should discriminate the confused pairs — no numbers were
copied from the data.
