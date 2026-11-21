# Method

## Architecture

1. **Silhouette**: a 1x1 conv and relu turn the image into a soft silhouette, `clip((x - 0.04) / 0.08, 0, 1)`.
2. **Blur**: a 7x7 Gaussian (sigma 1.2) conv on the silhouette.
3. **Compare with drawings**: one dense layer with 50,000 rows, 5,000 per class. Each row is a blurred drawing of
   an item; the output is minus the squared distance to that drawing (up to a term shared by all drawings), with
   the top-centre neck zone weighted 3x. A reshape and max-pool keep the best drawing of each class.
4. **Two small detectors**, added to the class scores as bounded bonuses:
   - *centre line* (zip or button band, +12 to coat): a tall, narrow line filter on the contrast-normalised image
     (`norm`), which must be vertically coherent and lie wholly inside the silhouette;
   - *strap openings* (+12 to sandal): dark pixels with silhouette above and below in the same column, ignoring
     long horizontal runs (the dark sole band of a sneaker).

## Where the weights come from

Nothing is fitted and `build.py` reads no data. The drawings come from parametric outline models typed from
general knowledge of the items: a top is a body polygon plus two sleeve bands and a neckline, collar or hood; a
trouser is two legs under a waistband; a dress is a bodice and skirt; shoes are side profiles with the toe on
the left (sneaker, high-top, flat and heeled boot, five kinds of sandal); a bag is a body with a handle loop or
strap. Each model is drawn many times with proportions varied by a seeded generator and framed like the photos
(trimmed, longest side scaled to 28, centred). Filters, thresholds, bonuses and per-model score offsets are
hand-chosen constants.

I looked at training images to learn how the classes are photographed, and ran detector unit tests only on my
own synthetic drawings.

## The five checks (all 60,000 training images)

| Check | Accuracy | What it showed |
|---|---|---|
| 1 | 71.0% | Drawings alone work for trousers, bags, sneakers; coat drawings swallowed pullovers; sandals leaked. |
| 2 | 66.4% | A line bonus for trousers and a first gap detector misfired (dresses to trouser, sneakers to sandal). |
| 3 | 72.6% | Gated line detector and fixed models; short-sleeved shirt drawings gave more false than true hits. |
| 4 | 73.5% | Strap-opening detector, taller sneakers, shirts suppressed, coat offset rebalanced. |
| 5 | 73.6% | Offset rebalance helped; a pattern detector for shirts was wrong more often than right. |

## What is submitted

The submitted network is the check-5 network **with the pattern detector removed** and the short-sleeved shirt
offset set back to its check-4 relation to T-shirts. **This exact network was not run on training images** (no
checks were left). On synthetic drawings its scores are identical to the check-5 network for every class except
shirt, whose score is equal or lower. From the check-5 confusion table I expect about 73.7%, but that figure is
an estimate, not a measurement. Shirts remain almost entirely unrecognised (the outline does not separate them
from pullovers, coats and T-shirts).

`check_rebuild.py` passes: `build.py` reproduces `network.json` and `weights.npz` exactly (39.3M weights).
