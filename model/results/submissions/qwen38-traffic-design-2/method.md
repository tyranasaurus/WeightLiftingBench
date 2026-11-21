# Method

## Architecture
`image -> norm -> conv(paint, 14 fixed 5x5 filters) -> relu -> two matched-filter branches -> add -> 43 scores`

* **norm** standardises each RGB channel per image, giving invariance to the large
  lighting differences (dark / washed-out photos).
* **paint conv (c1)**: 14 hand-written filters from formulas only: red–green and
  blue–yellow opponent colours, a yellow (G−B) and a green channel, smoothed
  luminance and its negative, and luminance derivative-of-Gaussian edges at
  0/45/90/135° in both polarities (Gaussian σ=1, 5×5).
* **FULL branch (cf)**: 42 whole-sign matched filters (classes 11 and 12 share one,
  they look identical), 29×29. Each kernel is the relu-paint of a *procedurally drawn
  idealised sign* (red ring + DIN-style stroke digits, red triangle + pictogram, blue
  disc + white arrow, yellow diamond, octagon + STOP, striped end-of disc, …), drawn
  from general knowledge of the German sign designs at 4× supersampling, averaged over
  two blur levels, unit L2-normed. relu + 2×2 maxpool gives ±2 px translation tolerance;
  a dense layer sums the four pool cells.
* **PART branch (cp)**: 41 symbol filters, 17×17…13×13 crops centred on the symbol:
  digit pairs, car/truck symbols, white bar, STOP letters, the 14 triangle pictograms,
  the 8 blue-disc arrows (the "ahead only" template carries extra negative lobes where
  side branches would appear), grey-ink templates for the "end of" family (class paint
  minus stripes-disc paint so the shared stripes cancel), a truck-only template
  (42 minus 41), and a hand-formula EMPTY template (bright, ink-free interior) for the
  empty red ring (class 15). relu + 5×5 maxpool, dense over the pool cells expected for
  each symbol's position.
* **Readout**: the two dense layers are a hand-written rule table, e.g.
  `score[5] = FULL(80-sign) + 0.9·D80`, `score[6] = FULL(end-80) + 0.9·D80grey +
  0.5·STRIPES − 0.4·(grey cars/grey truck) − 0.5·TRUCK`, `score[15] = FULL(empty ring) +
  0.7·EMPTY`, plus hand-chosen mutual suppressions (stop vs no-entry, ahead-only vs
  straight-or, exclamation vs other pictograms, …) and small hand-chosen class priors
  from real-world prevalence. The two 43-vectors are summed by the final `add`.

Total 628,179 weights (limit 120,000,000).

## How the weights were produced
Every number comes from code in `build.py`: fixed filter formulas, and templates
computed by pushing the idealised sign renders (typed from general knowledge of the
sign designs — ring widths, triangle geometry, arrow and pictogram strokes, digit
stroke font) through the same norm+paint stack the network uses. No gradient descent,
no fitting, no statistics of the training images, no copied image content. Training
images were only viewed (montages) to confirm the sign designs, and the network was
run on them once (check 1, 2021 images, accuracy 0.2202) plus a data-free rebuild check.

## Honest status
On my own simulator (idealised renders perturbed with jitter, rotation ±7°, blur,
dark/washed lighting, nearest-neighbour resampling, coloured backgrounds) the network
scores ≈64.5%. Check 1 on real training images scores 0.2202: real photos carry more
in-plane rotation, perspective skew, partial shadow and motion blur than my simulator,
which the position-pooled matched filters tolerate poorly. The family-level evidence
(ring / triangle / blue disc / diamond / stripes) and the digit/arrow templates are the
parts most likely to survive on the held-out set; the reported check accuracy is the
trustworthy number.
