# Method

**No training, no fitting.** Every weight is produced by `build.py` from typed-in shape models and formula
filters; it reads no file. The ten training images were only looked at (as PNGs, plus a coarse grey-level
histogram) and used for the logged checks.

## Architecture

1. **Three hand-made maps of the image** (1x1 convs + relu for clipping, a 5x5 Gaussian blur, one dense layer for
   whole-row/column sums, 2x2 average pooling):
   - *soft silhouette*: 0 below grey level 0.03 (background noise), 1 above 0.13 — pale items still count as item;
   - *filled silhouette*: a pixel with item to its left, right, above and below also counts as item, so pale or
     patterned items whose inside drops out keep their shape (gaps open to one side — between trouser legs, between
     sleeve and body — stay open);
   - *bright parts*: ramp from 0.3 to 0.6. Thin straps and handles cover only part of a pixel and come out
     mid-grey, so a strappy sandal fades here while a solid shoe stays.
2. **`norm`** standardises each map, so the dense layers that follow output Pearson correlations.
3. **Template bank** (three dense layers added together, weights 0.4 / 0.3 / 0.3): 98,400 templates, 9,840 per
   class. Each template is one picture rendered by a parametric shape model, passed through the same three maps and
   normalised by itself alone. No statistic across generated pictures is used anywhere.
4. **Class score** = max over each of 16 groups of 615 templates of a class (`maxpool`), then the mean of the 16
   group maxima (`avgpool`). The mean of group maxima rewards a class that has many near templates rather than one
   lucky one.
5. **Centre-line branch** (hand-set): the mean over 17 rows of a horizontal second difference at the centre
   columns, either polarity, must exceed 3x the same response a few columns to the side (how busy the cloth is)
   plus a floor of 0.02. If so, up to 0.03 is taken from T-shirt and pullover: a front opening or button placket
   speaks for coat or shirt.

## Shape models (general knowledge, not measurements)

Polygons and thick strokes, rasterised with 4x supersampling, then trimmed to the bounding box, scaled so the longer
side is 28 pixels and centred — the way the photos were prepared — with small random tilt, shift and brightness.

- Tops share one body/sleeve model. Classes that are easily confused use the same ranges for everything that does
  not tell them apart and differ only in specific cues: T-shirt (short sleeves, wide round neck; also vests and cap
  sleeves), shirt (narrow collar, shirt-tail hem, slimmer sleeves; long, short and sleeveless variants), pullover
  (round neck, hem pulled in, sleeves often spread, cuffs at or below the hem), coat (stand-up collar or hood,
  lapel notch, straight or flared hem, sleeves close to the body, longer body).
- Dress: straps, strapless or sleeved bodice, waist, bell or A-line skirt. Trouser: waist, hips, crotch, two legs
  together or apart, or one folded leg.
- Footwear in side view, toe to the left: sneaker (low toe box, laces rising to the ankle opening, optional high top
  and pale sole), ankle boot (shaft over the heel, optional heel with arch), sandal (flat with thin straps, heeled
  with stiletto/block/wedge and ankle strap, or thick-soled with broad straps and open toe).
- Bag: rounded trapezoid body with none, one or two top handles, a long shoulder strap, or a small top loop.

## Checks on the training images (5 allowed, all logged in `checks.log`)

1. 6/10 — pipeline works; every miss stays inside its family (T-shirt→shirt, pullover→coat, sandal→sneaker,
   sneaker→boot).
2. 6/10 — after replacing the centre-line detector with the texture-relative version and redrawing sandals and
   sneakers; same table, so the misses came from the silhouette bank, not from that branch.
3. 8/10 — after adding the bright-parts map, high-top sneaker profile and taller boot prior; sandal and sneaker
   correct.
4. 9/10 — after giving confusable tops shared body/sleeve ranges (garments are shown on a mannequin, so bodies are
   narrower than flat-lay widths); only pullover→coat left.
5. 9/10 — the final network (changes listed below); same table as check 4, so it behaves on real images as the
   version before those changes did. The one miss is still pullover→coat.

Changes between checks were general corrections to the shape models and features, reasoned from what the classes
look like; no constant was taken from an image and nothing was scored on generated pictures. After check 4 the two
large-kernel convolutions were rewritten as equivalent dense layers (same outputs to 3e-8 on generated pictures;
the reference conv code would have needed several GB per batch), the centre-line term was changed from a bonus for
coat/shirt to a penalty for T-shirt/pullover, and the bank went from 625 to 615 templates per group to stay under
the weight limit.

## Limits

Ten images say little about accuracy on 10,000. Expect the weak spots to be shirt against T-shirt/pullover/coat and
pullover against coat, where silhouettes overlap and the centre-line cue is the only extra evidence; that cue's
thresholds are untested on real data beyond these ten images.
