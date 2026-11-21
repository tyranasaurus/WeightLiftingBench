# Method

## Architecture
A small, hand-wired network (≈6,400 weights) with two branches summed into the
2-class score:

1. **Central-detail branch.** `conv` (8 luminance centre-surround contrast
   kernels: horizontal & vertical, two scales, bright & dark polarity) →
   `relu` → `maxpool` 12×12 (keep the strongest sharp feature in each local
   cell) → `reshape` → `dense`. The dense row for "person" is a centre-weighted
   (slightly upward-biased) Gaussian mask summing the pooled detail; the
   "no-person" row is a constant threshold `T` in its bias.
2. **Greenness branch.** `conv` 1×1 computing `G − B` → `relu` → `avgpool` 96×96
   (global mean greenness) → `reshape` → `dense` that adds a vote to the
   "no-person" score.

Final `add`: `person iff central_detail > T + g · greenness`.

## Why these weights (no training, no fitting)
From general knowledge of photographs of people, not from the data:

- People are the **sharp, high-contrast central subject** (faces/eyes, hands,
  textured clothing, skin highlights), while a large share of *no-person* COCO
  photos are **smooth / low-contrast** (sky, sea, fog, plain walls, empty rooms,
  blurred backgrounds) or carry their detail at the edges (scenery). So the core
  cue is centre-weighted fine-contrast energy.
- The centre-surround kernels are standard zero-sum contrast filters (a bright
  3-wide centre vs two −0.5 flanks), scaled by the standard luminance weights
  (0.299, 0.587, 0.114). `maxpool` implements "a sharp feature exists here".
- The threshold `T = 1.3` is a **hand-chosen** constant grounded in the
  general fact that typical local luminance contrast in natural images is ≈0.2;
  summed over 8 detectors the typical central-detail level is ≈1.6, and a person
  makes the centre sharper than a typical scene, so the boundary sits just below.
- Foliage, grass, food and natural scenes are green (G>B); people/indoor/urban
  scenes much less so, so mean `relu(G − B)` votes toward *no-person* with a
  modest weight `g = 6`.

All constants (kernels, Gaussian mask, `T`, `g`) are formulas or hand-chosen
numbers from the ideas above. The training set was used only to *look at images
and print summaries* to confirm which general cues separate the classes (skin
colour, for example, turned out useless and was dropped); no image or label
statistic is stored in any weight, and no readout was fitted.

## Verification
`build.py` regenerates `network.json` and `weights.npz` with no inputs
(`check_rebuild.py`: exact match). Two allowed checks on the 20k training sample
gave 0.56 for an early centre-vs-border variant and **0.6116** for this design
(balanced predictions); two disjoint held-out subsets simulated 0.60 and 0.61,
indicating the cues generalise rather than memorise.
