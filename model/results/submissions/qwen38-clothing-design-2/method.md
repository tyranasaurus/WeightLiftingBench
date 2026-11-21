# Method

## Architecture
The network is a hand-written "program as a network", about 2.7 M weights:

1. **Analytic filters** (conv + relu): a narrow gaussian blur minus a threshold gives a
   soft silhouette `M` (soft-binarised with two relus so it is brightness-invariant);
   further channels hold the raw intensity, `|laplacian|` edge energy, `|high-freq|`
   texture energy and `|d/dx|`, `|d/dy|` gradient energy. All filters are gaussian /
   gaussian-derivative / laplacian formulas.
2. **Shape descriptors** (avgpool / maxpool / reshape / dense): row and column sums and
   maxima of the silhouette, 4x4 and 7x7 regional occupancy / intensity / edge means,
   global means, and correlations with 20 hand-drawn spatial patterns (t-shirt, trouser,
   pullover, dress, coat, sandal, shirt, sneaker, bag, boot silhouettes plus leg-gap,
   handle-arc, sole-band, lapel, collar, hem and texture patterns). These are assembled
   into one flat feature vector with placement denses summed by `add`.
3. **Descriptor layer** (dense): named geometric quantities - width per row band,
   sleeve drop, skirt flare, hem widening, occupied height/width, area, top/bottom mass,
   centre-bottom gap, bottom-row asymmetry, brightness, etc.
4. **Checklist layers** (dense + relu, three stages): each condition is a saturated gate
   `clip(v,0,1) = relu(v) - relu(v-1)`, and soft conjunctions use
   `min(a,b) = a + b - relu(a-b) - relu(b-a)`. The last dense is a per-class additive
   scoring checklist (e.g. trouser = narrow + tall + open between the legs; bag = boxy
   body under a sparse handle arc with empty top corners; sandal = low, wide, strappy;
   sneaker = low, wide, solid with flat sole; boot = mid height with solid foot block;
   t-shirt = sleeves ending high; dress = narrow shoulders with flared skirt;
   coat = hem wider than shoulders and bulky; pullover = bulky with long sleeves;
   shirt = slim straight body with a collar notch).

## How the weights were produced
No training, fitting, optimisation or data-derived statistics are used. Filters come
from formulas; templates are drawn in code from general knowledge of garments, shoes and
bags (rectangles, trapezoids, arcs); gate thresholds and checklist weights were chosen by
hand after studying what the classes look like (viewing training images as PNG montages
and printing simple per-class summaries such as bounding-box size, fill ratio and row
width profiles). The rule program was debugged on idealised silhouettes drawn in code
from the same general knowledge, and the five allowed checks on the training set were
used only to confirm it works and to re-balance a few checklist weights by hand.
`build.py` reads no files: every number is generated from the code itself.
