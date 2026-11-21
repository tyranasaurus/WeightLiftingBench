# Method

## Architecture

A template-matching network with two small hand rules; no layer was trained or fitted.

1. **Soft mask** (`pre1`, `mask`: two 1x1 convs + relu).
   `m = 0.5*clip((x-0.015)/0.085, 0, 1) + 0.5*clip(x/0.5, 0, 1)`: the first term says "something is here" (pale items
   still give a silhouette), the second "it is solidly here" (half-dark gaps between straps, legs, sleeve and body stay
   visible).
2. **Silhouette branch** (`blur`, `sil`): 5x5 Gaussian blur (sigma 1), then `norm` (standardise the map).
3. **Edge branch** (`edge` ... `edg`): eight derivative-of-Gaussian filters (every 45 degrees), relu, 4x4 max-pool with
   stride 2, reshape to one channel, `norm` (all directions standardised together).
4. **Template bank** (`tsil`, `tedg`, `tsum`): two dense layers with 50,000 rows each (5,000 templates per class).
   Row *i* of each layer is template *i*'s own standardised map divided by its size, so the outputs are the Pearson
   correlations of the image with that template; they are added as `silhouette + 0.5 * edges`.
5. **Best template per class** (`tgrid`, `tbest`, `tcls`): reshape to (1, 10, 5000), max-pool over the 5,000.
6. **Hand rule "strappy footwear"** (`tv` ... `rule`): total variation of the mask (outline length, holes included)
   against its area; `0.08 * clip((TV - 0.6*area)/40, -1, 1)` is added to the sandal score.
7. **Hand rule "front opening"** (`z` ... `lrule`): standardise the image, apply a `[-1/2, 1, -1/2]` line filter
   averaged over 11 rows, take the strongest response in each group of 4 columns, compare the middle group with its two
   neighbours; a long thin line down the middle adds up to 0.04 to the coat and shirt scores.
8. `scores` = sum of 5, 6 and 7.  106,909,221 weights in total.

## Where the weights come from

* Filters are formulas (thresholds, Gaussian, derivative of Gaussian, finite differences, line filter).
* Templates are **drawings made by code in `build.py`** from general knowledge of the ten kinds of item: a top is a body
  polygon plus two tapering sleeve strips whose angle, length and width depend on the kind (short and sticking out for
  T-shirts, long and a little apart for pullovers, straight down and shorter than the body for coats, thin with shirt
  tails for shirts); trousers are a waist with two legs and a gap; dresses a narrow bodice and a skirt; shoes are side
  views with the toe to the left (sneaker: low wedge with a flat sole, often a pale rubber sole; ankle boot: shaft on the
  right, heel; sandal: thin sole with straps, high heel, wedge, or a cage of straps); bags are a box with a handle loop
  or strap.  Parameters are drawn from ranges I typed, with a fixed seed.  Every drawing is trimmed, shrunk so its
  longest side is 28 pixels and centred, as the photos were, then passed through layers 1-3 and normalised **by itself
  alone**.  No statistic over several drawings is used anywhere.
* Kinds whose silhouette coincides with a more common kind from another class (short-sleeved or sleeveless shirt vs
  T-shirt, short jacket vs pullover) carry a small handicap (0.03 or 0.015, in the template's bias), so they win only
  when they fit clearly better or the front-opening rule speaks for them.
* All constants were chosen by reasoning, not by measurement.  Nothing was taken from the training images; I looked at
  the ten samples only to see what the classes look like (for example that shoes point left).

## The checks on the ten training images (4 of the 5 allowed were used)

1. 8/10 - the pipeline works on real images; misses were coat->pullover and sandal->sneaker.
2. 6/10 - after adding the handicap the short-sleeved shirt sample goes to T-shirt (intended trade), and the high-top
   sneaker sample fell to ankle boot, showing that boundary is fragile.
3. 8/10 - after adding pale-sole sneaker drawings and half-tone "cage of straps" sandal drawings (both from general
   knowledge of footwear) the sneaker and sandal samples are right again.
4. 8/10 - the final network (one more shirt kind added) behaves the same; the two misses are a coat shaped like a
   pullover and a short-sleeved shirt shaped like a T-shirt.

Ten images say very little about accuracy on the 10,000 held-out ones; I have no measured estimate of that.  The
weakest classes are expected to be shirt, coat and pullover, whose silhouettes overlap.

## Reproducing

`python3 build.py OUT_DIR` writes both files in about a minute, reading nothing.  `check_rebuild.py` reports an exact
match.
