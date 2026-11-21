# Method

## Architecture

A single-stream, fully hand-specified feature extractor followed by a four-layer
hand-written rule head. 47 layers, ~4.05M weight entries of which only ~9,840 are
nonzero — every nonzero number is a formula coefficient or a rule weight I chose
by reasoning, not by fitting.

**Front end (segmentation).** `norm` standardises the photo. Three soft steps
`(relu(x-t) - relu(x-t-s))/s` at grey levels 0.02/0.08, 0.12/0.28, 0.35/0.60 turn
the image into a brightness-independent silhouette plus two "how bright" maps.
These are merged with the raw grey and the normed grey into one 5-channel tensor
using 1x1 convs that place channels into disjoint slots, then `add` (the only way
to concatenate with these ops).

**Structure filters.** An 11x11 conv over [silhouette, mid-mask, normed grey]
whose kernels are all closed-form masks over the window: differences of means
between arms and a centre (narrow/wide dark vertical and horizontal lines, the
four arms around a dark hole, a ring-minus-disc enclosed-hole detector), Sobel-like
side differences for edge orientation, diagonal edges, local mass asymmetries, and
a period-4 check/plaid pair. A 5x5 conv over the 5-channel base adds high-pass
texture energy on normed grey, on the mid mask and on raw grey. A 1x1 conv then
forms `erode` (a soft step on the 3x3 local mean, i.e. "deep inside solid cloth")
and `thin = silhouette - erode` (straps and 1-3px bars).

**Pooling.** `avgpool` and `maxpool` (2x2, stride 2) give a 14x14 grid of both the
mean and the peak of every channel, merged the same way. Mean gives area/mass;
max preserves thin straps and handle arcs that averaging would erase.

**Rule head.** `dense` -> 167 *primitives*, each the mean of one channel over a
rectangle I named from garment anatomy (shoulder band, lower-outer sleeve box,
sleeve/body slit, centre-front placket, leg gap, handle loop, shoe shaft, toe,
heel, sole, hem). -> 6 *properties* (occupied height and width in pixels from the
empty-row/column hinges; shoulder, waist and hem widths). -> 105 *cues*, which are
thresholded rules (`relu(w.f - t)`), so a single cue can require several conditions
at once. -> 10 *shape-evidence* units, one per class. -> 10 class scores, which add
each class's evidence and subtract competing classes' evidence.

## How the weights were produced

Entirely from the stated imaging geometry (28x28, item centred, longest side 28,
light on black) and general knowledge of how these objects are built:

- trousers are two legs with real background between them, on a narrow tall
  silhouette — so the gap detector runs on the *silhouette*, and narrowness only
  counts when that gap is present;
- a coat opening or shirt placket is a fold in cloth, not a gap, so it is invisible
  in the silhouette and needs a *shading* line detector on normed grey with nine
  rows of context (a short context mistakes texture edges for lines);
- pullovers hang their sleeves beside the body, leaving two background slits and a
  wide hem; t-shirts have wide shoulders but nothing bright low down at the sides;
- dresses flare from waist to hem and have no cuff slits at the hem;
- a bag handle is a dark loop with cloth on all four sides and a solid body below
  it, and a bag's column heights are symmetric; a boot's shaft makes one end tall;
- sandals are almost all thin structure, so `thin` nearly equals the silhouette
  area, while sneakers are short and solid and boots are taller and asymmetric.

Thresholds, region rectangles and rule weights are round numbers I picked by hand
from that reasoning. No training image, patch, descriptor, class average or fitted
statistic enters any weight; `build.py` reads no files at all.

## Checks used

Four of the five allowed runs, each on the same fixed stratified 12,000-image
subset (1,200 per class):

1. 10.1% — everything fell to the bias-only default; found a sign error in the
   soft-step width that inverted all three masks.
2. 36.1% — trousers and coats never fired (their centre lines are shading, not
   silhouette gaps) and footwear was absorbed by the boot rule.
3. 30.5% — trousers reached 95.8%, but an unbounded "not full height" term made the
   shirt rule swallow every short item, and the bag rule was too generic.
4. 47.9% — current answer. Trousers 94%, sneakers 83%, t-shirts 80%, ankle boots
   72%, dresses 56%, shirts 54%; pullovers and coats still leak into shirt, and
   bags are still split across dress/pullover/shirt.

`python3 /app/check_rebuild.py /app/out` reports REBUILD OK with max relative
difference 0.0.
