# Method

## Architecture
A template-matching convolutional network, all weights hand-constructed from knowledge of
German sign design (StVO), no training involved.

1. `fe` — 3x3 conv (9 channels): luminance, four colour-opponent channels
   (red-, blue-, yellow-, green-minus) as Gaussian-smoothed combinations, and four
   oriented luminance derivatives (Sobel x, y, 45°, 135°).
2. `nfe` — `norm`: per-channel contrast normalisation over the frame, which makes the
   representation robust to the dark / washed-out lighting of the dashcam crops.
3. For each template window size k in {16, 20, 24, 28, 32} (sign diameter = k/1.2, the
   range produced by "sign + ~10% border" crops resized to 32 px), two conv layers with
   kernel k, stride 1, 43 outputs (one filter per class):
   - `t{k}` whole-sign templates, restricted to the sign silhouette (plate mask), so the
     varying photo background contributes nothing;
   - `i{k}` interior templates, restricted to the symbol-bearing interior (outline and
     anti-alias annulus eroded away), so the symbol — digits, arrows, vehicles, warning
     pictograms — carries the class identity instead of being swamped by the shared
     red ring / triangle outline. Interiors are blurred more coarsely (σ=2 px) because
     the ink *layout*, not pixel-exact strokes, survives blur and 1 px misalignment.
   Each is followed by a maxpool over all stride-1 positions (translation tolerance).
   Signs with an empty interior ("no vehicles") get a uniform-brightness interior filter,
   so dark symbol ink actively lowers their score.
4. `sum` adds the ten pooled (43,1,1) score maps (whole-sign gain 1.0, interior gain 2.5),
   `scores` reshapes to (43,).

## How the weights were produced
`build.py` procedurally draws one idealised image per class (discs, rings, rounded
triangles, octagon, diamond, stroke digits in the speed-sign proportion, arrows, car /
truck / pedestrian / bicycle / deer silhouettes, snowflake, traffic light, diagonal
cancel stripes) at 160 px, area-downsamples it to each window size, passes it through the
same fixed `fe` filters, and turns that single image into its own filter: per-channel mean
removed over the template's support, per-channel equalisation with the variance floored at
10% of the strongest channel (so structureless colour channels stay at ~0 rather than
being amplified into noise), then one joint L2 normalisation and a Gaussian blur — i.e.
each generated image is normalised by itself alone. No statistic is ever taken across
generated images, no training image, label, crop or descriptor enters `build.py` (it reads
no files at all), and no number was chosen by scoring generated images.

Total weights: 2,353,642 (limit 120,000,000).
