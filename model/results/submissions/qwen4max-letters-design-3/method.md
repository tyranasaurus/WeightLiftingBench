# Method

**Result: 19.05% on the 27,455 training images (check 5 of 5).** Random guessing is
4.17%, so the normalised score is about 0.155. This is well below what gradient
descent on the same architecture would reach; it is what I could build from
anatomical knowledge alone, without fitting anything.

## Architecture

A three-view matched-filter classifier over a bank of **1,152 hand-shape templates**
(24 letters x 48 pose variants). 4,524,128 weights, 40 layers.

Shared front end: `norm` (removes brightness/contrast) -> 5x5 Gaussian denoise
(sigma 0.7) -> `norm` -> hard clip to +-1 sigma -> `norm`. The clip is built from
`conv`/`relu`/`conv` and is the important robustness step: these photos have low
hand/background contrast but very dark shadows, sleeves and finger gaps, so an
unclipped normalised image is dominated by a handful of extreme pixels. Clipping
makes the map quasi-binary, so a template is scored on *shape agreement* rather
than on how well it lands on one strong feature.

Three branches, each seeing the same drawing bank through a different filter:

| branch | image representation | template | polarity |
|---|---|---|---|
| S | clipped intensity | silhouette | both signs |
| D | centre-surround (image minus Gaussian blur, sigma 3) | same filter applied to the drawing | both signs |
| B | sum of \|Gx\|+\|Gy\| at sigma 1 and 2, blurred | outline + internal finger creases | polarity-free |

S keeps *solid vs. hollow* (which separates B / C / O); D keeps finger-scale
structure and rejects the uneven background; B is invariant to whether the hand
is lighter or darker than the background.

Each branch: `conv` with the 28x28 templates (stride 2, padding 6 -> a 7x7 search
over +-6 px shifts) -> `relu` -> global `maxpool` -> `reshape` -> `maxpool` over the
polarity axis -> `maxpool` over the variant axis -> 24 class scores. The reshape +
maxpool pair is an exact per-class maximum, not a sum. The three score vectors are
combined by identity `dense` layers and `add` (equal weights: the views are
complementary and, after clipping, on the same scale).

## How the weights were produced

Every number comes from `build.py` itself; it reads no files.

* **Filters from formulas**: Gaussians and Gaussian derivatives, normalised
  individually.
* **A hand shape model typed from general knowledge**: signed-distance primitives
  (rounded boxes, tapered capsule chains, elliptical rings, an annulus sector whose
  thickness tapers). Palm width = 1 unit; finger lengths 0.92/1.02/0.94/0.74,
  knuckle spacing 0.24, a thumb, a wrist that runs off the frame. Each of the 24
  letters is described by which fingers are extended, their spread, the thumb
  position, and the whole-hand orientation (G/H point sideways, P/Q point down,
  the rest up). D/F/O get a loop with a real hole; C is a crescent, thicker at the
  back of the hand than at the fingertips; X is a hook; R crosses index over middle.
* **Variants**: 3 scales x 4 rotations x 2 mirrors x 2 finger-spread poses = 48 per
  letter, the same count for every class so the max-over-variants is fair. Each
  drawing is fitted to the frame by its own bounding box (the photos were cropped
  to a square around the hand) and **normalised by itself alone**.
* **Hand-chosen constants**: clip knee 1.0 sigma, centre-surround sigma 3, edge
  sigmas 1 and 2, shift range +-6 px, equal branch weights.

No gradient descent, no least squares, no regression, no class averages, no
clustering/PCA, no nearest-neighbour bank. No training image, crop, patch,
descriptor or measurement enters any weight. No statistic across generated
drawings sets any number, and nothing was scored on generated images to pick a
setting.

## What the five checks were used for

Each check exposed a structural defect, which I then fixed by reasoning about the
images and the mathematics, not by fitting numbers to the confusion matrix.

1. **10.82%** - D/F attracted everything and C/E/N/O were never predicted. Comparing
   my drawings with the photos showed the hand/background contrast is low and that
   plain NCC rewards templates with *concentrated* energy (hole-bearing D/F) over
   ones that must match a whole shape.
2. **13.01%** - centre-surround + soft saturation helped only slightly, so heavy
   tails were not the whole story. Inspecting my own template arrays confirmed the
   never-predicted classes were exactly those with the lowest template peak.
3. **15.84%** - hard clipping (quasi-binary maps) confirmed the concentration
   diagnosis and helped, but rigid templates still could not absorb the photos'
   irregular shapes.
4. **16.38%** - deformable quadrant matching helped B/I/T/Y but destroyed
   distinctive layouts (L 52%->3%, U 48%->15%) and made B a generic attractor, so I
   reverted to rigid matching.
5. **19.05%** - rigid matching restored, richer variant coverage, C redrawn with a
   tapered arc and O with a thicker loop, a clipped-silhouette branch added to
   recover solid-vs-hollow, and a real bug fixed: the clip block was clipping
   without re-normalising first, so templates and images were being processed
   differently. C went 0%->16%, O 0%->24%, Y 77%, L 48%, B 46%, R 42%, U 42%.

Remaining weakness is concentrated in the closed fists (A/E/M/N/S/T), whose real
differences are thumb placement and one or two knuckle creases - detail my
silhouette model cannot express faithfully at 28x28.

`python3 /app/check_rebuild.py /app/out` reproduces both answer files exactly
(max relative difference 0.0) from `build.py` alone, with no data.
