# Method

The network is a **hand-built matched filter** — no training, no fitting, no statistic of the
training data. Every weight is derived from clean sign templates that `build.py` renders from
general knowledge of German road signs (shapes, colours, DIN-style digit strokes, arrows, symbols).

## Architecture
1. `pre` (1×1 conv): converts RGB to three opponent channels `[luminance, red−green, yellow−blue]`.
2. `norm`: per-channel zero-mean / unit-variance. This removes the DC level, so no template can
   win just by being bright/generic (a "sink"), while colour survives at edges.
3. `corr` (32×32 conv, pad 3): one filter per (class, scale). Each filter is
   `W_S·Shat[family] + Dhat[class]`, where
   - `Shat` is a **family** colour+shape template (blank sign: red ring / red triangle / inverted
     triangle / yellow diamond / red octagon / red disc / blue disc / grey ring), masked to the
     sign footprint and L2-normalised. It is *identical for every class in a family*, so it cancels
     in the within-family comparison and decides only between families.
   - `Dhat` is the **class mark** (speed digits, cars/truck, arrows, warning symbol, bar, STOP,
     derestriction stripes), masked to the sign interior, per-channel-centred (removes the family
     background colour, leaving the mark) and L2-normalised. It decides within a family.
4. `maxpool 7×7` searches ±3 px of translation; a reshape + `maxpool` over the **5 scale variants**
   (sign radius ≈ 9.6–14.4 px, matching the spec's 10%/≥5 px border) searches scale. A final
   reshape gives the 43 class scores; the argmax is the prediction.

Each template is normalised only **by itself** (its own mean / L2 norm); no average, prototype or
other statistic is ever taken across templates or across any image, and no number was chosen by
scoring the network on generated or training images. Constants (`W_S=2.5`, colour weight `CW=0.7`,
scale set, geometry) were picked by hand from knowledge of how the signs look and of how the
matched filter behaves.

## Producing the weights
`build.py` renders each template with a small vector-drawing library (supersampled 160×160, area
down-sampled to 32×32), builds the `pre` and `corr` filters as above, and writes `network.json` and
`weights.npz`. It reads no files and uses only numpy, so it reproduces the submitted weights exactly.
