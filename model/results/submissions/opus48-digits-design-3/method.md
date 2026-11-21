# Method

## Architecture
A hand-designed **nearest-prototype classifier in oriented-stroke feature space**:

```
image (1,28,28)
  -> norm                         per-image contrast normalisation
  -> conv  (9 filters, 9x9, pad 4)  8 oriented ridge detectors + 1 Gaussian lowpass
  -> relu                         keep "bright stroke of orientation k present"
  -> avgpool (4x4, stride 4)      -> (9,7,7) coarse orientation map
  -> reshape (1,441,1) -> norm    standardise the 441-d feature descriptor
  -> dense (1800)                 cosine similarity to 1800 idealised prototypes
  -> reshape (1,10,180)
  -> maxpool (1x180)              best-matching prototype per class
  -> reshape (10)                 class scores
```

## Where every weight comes from (no training data, no fitting)
* **ridge.W** — 8 oriented ridge/bar detectors built from a closed-form formula
  (second-derivative-of-Gaussian across the stroke, Gaussian along it;
  `SIG_PERP=1.3`, `SIG_ALONG=2.5`), spanning orientations 0°–157.5°, plus one
  Gaussian lowpass "ink" channel. Each is zero-mean, unit-norm. These decompose an
  image into *which stroke orientation is present where*. ridge.b = 0.
* **match.W** — 1800 prototype rows. I typed shape models of the ten digits from
  general knowledge of how they are written (pen paths as line segments, arcs and
  loops: e.g. an ellipse for 0, a vertical bar with optional flag/foot for 1, two
  stacked right bulges for 3, a diagonal+crossbar+vertical for 4, ...). Each digit
  has 4 style variants, rendered across 5 slants, 3 widths and 3 stroke thicknesses
  (10×4×5×3×3 = 1800), every glyph anti-aliased and re-centred by centre of mass to
  the frame centre (as the task says the data is centred). Each rendered prototype is
  pushed through the *same* norm→ridge→relu→avgpool pipeline, the resulting 441-d
  feature vector is standardised and L2-normalised, and stored as a row. Because the
  input descriptor is standardised the same way, each dense unit outputs the cosine
  similarity between the input and that prototype. match.b = 0.

No pixel value, coordinate, average, or label from the training set is used anywhere.
Stroke positions, loop sizes and thicknesses are constants chosen by hand from general
knowledge of digit shapes (calibrated only by *looking* at example glyphs, as allowed).

## Reproducibility
`build.py` is fully self-contained and reads no files; `python3 build.py OUT` writes
`network.json` and `weights.npz` deterministically (verified by `check_rebuild.py`).

## Checks
The 5 allowed checks were used only to confirm the network runs and to read overall
accuracy while choosing between design variants (raw-pixel vs oriented-feature
matching, Euclidean vs cosine scoring, and stroke-thickness calibration). The final
oriented-feature cosine design reached ~63% accuracy on a 10,000-image training sample.
