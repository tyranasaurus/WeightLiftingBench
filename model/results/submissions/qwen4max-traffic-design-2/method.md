# Method

## Architecture (16 layers, 1,821,150 weights)

1. **Illumination-covariant colour/contrast channels.** A 1x1 conv builds signed
   linear forms of (r,g,b): intensity, the six pairwise differences, and
   hue-sector tests such as `(r-g) - 0.12*(r+g+b)` (red), `(b-r) - 0.18*S`
   (blue), `(g-b) - 0.18*S` minus a vegetation term (yellow), a green-lamp test,
   and `0.14*S - (|r-g|+|g-b|+|b-r|)` (achromatic/white). Every form is
   homogeneous of degree 1 in the pixel values, so a multiplicative lighting
   change only scales it and an additive ambient light only shifts it.
   A centre-surround (DoG, Gaussian sigma 0.9 minus 2.2) of intensity gives a
   signed dark-stroke channel. Six channels are stacked and passed through the
   `norm` op, which removes exactly that per-channel gain and offset: the
   features are lighting-normalised without any fitted statistic.
2. **Analytic full-sign template bank.** One conv (kernel 28, stride 1, no
   padding -> a 5x5 grid of positions) holds 43 classes x 9 scale/aspect
   variants = 387 kernels over the 6 channels. Each kernel channel is
   `+1/area(pos)` on the region where the feature is expected and
   `-1/area(neg)` where it is not, so a response is a *difference of regional
   means*: a contrast score independent of region size and of brightness.
   Regions are drawn from general knowledge of German signs: red annulus
   (t = 0.24 R), white interior, equilateral / inverted warning triangles with
   rounded-border thickness, octagon, yellow diamond with white border, blue
   disc, DIN-1451-like stroke digits for 20/30/50/60/70/80/100/120, pictograms
   (exclamation, priority cross, bend arrows, double bend, bumps, skidding car,
   narrows, road worker, traffic-light lamps in red/yellow/green, pedestrian,
   children, bicycle, snowflake, leaping deer), car/truck silhouettes, and the
   five diagonal "/" stripes of the grey "end of" signs. Sign diameter ~26 px
   follows from the stated 10% border; variants span 21-29 px and 0.85 aspect
   squash in both axes. Pictograms/arrows/vehicles are scored as coarse
   full-pattern cell bitmaps (both dark and white cells specified, so partial
   matches are penalised); digits and STOP use fine strokes with a matched
   halo negative. Shape/colour family terms carry ~2.5x the weight of symbol
   terms because large-region colour contrasts are the reliable cue.
3. **Readout.** reshape to (43, 9, 25), max-pool over variants then over
   positions -> 43 class scores. No fitted readout: the decision is the argmax
   of the analytic contrast scores.

## Provenance of every number

All weights come from formulas (Gaussians, capsule/polygon/arc distance fields,
stroke fonts) and hand-chosen geometry/thresholds based on knowledge of the
sign classes and the task's stated crop geometry. `build.py` reads no files.
No training image, patch, descriptor, label statistic, fit, or regression is
used anywhere; nothing was copied or traced from photos. Design-time validation
was (a) viewing montages of training photos, (b) rendering my own templates to
PNG and comparing proportions by eye, and (c) a weight-only "ideal sign"
cross-score matrix (each template scored against pos/neg masks derived from the
templates themselves) - never a network run on data outside the 5 logged checks.
A throughput benchmark used random noise arrays, not images.

## What the five checks taught

1. 0.2178 - region-mean scores let tiny positive regions "cheat" by landing on
   any dark blob; four simple templates attracted everything.
2. 0.2862 - matched halos fixed thick-blob pictograms (bumps 72%, slippery 88%,
   deer 85%, pedestrian 94%) but thin strokes were diluted by template blur.
3. 0.1428 - sharpening made it worse: fine templates never align on blurred,
   nearest-neighbour photos, so the simplest thick shape (exclamation) won.
4. 0.3061 - coarse full-pattern cells restored shape families and several
   pictograms (yield 85%, no-entry 88%, keep-left 99%, bend-left 87%) but
   quantising digits into cells turned them into solid blobs.
5. 0.2450 - a wider position/scale search gave cross-family attractors more
   room; family evidence was under-weighted relative to noisy symbol evidence.

Final answer combines check 4's configuration with fine-stroke digits (visually
verified against photos) and family-dominant weighting; the data-free rebuild
check reproduces it exactly (max relative difference 0.0).
