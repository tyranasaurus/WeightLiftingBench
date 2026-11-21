# Method

## Idea
The network is hand-wired from general knowledge of galaxy morphology. No part
of it is trained or fitted; every weight is produced by a geometric formula in
`build.py`. The pipeline measures a handful of physically meaningful, scale- and
brightness-invariant quantities and feeds them into a small hand-written
decision tree built out of the allowed operations.

## Pipeline (see `build.py` / `network.json`)
1. **Grayscale & colour.** 1×1 convs give `gray = (r+g+b)/3` and `col = r−b`
   (z-band minus g-band → redness).
2. **Contrast normalisation.** `norm` standardises `gray` per image (mean 0,
   unit variance), so everything downstream is invariant to overall
   brightness/size. A 1×1 conv subtracts an isophote level and `relu` keeps the
   galaxy body `M` (pixels ≳1σ above background). A difference-of-Gaussians conv
   gives a band-pass map for texture.
3. **Fixed templates (dense layers).** Over `M` I sum products with fixed
   templates centred on the frame:
   - 12 oriented thin **"slits"**; `e = max_θ − min_θ` of the slit flux is a
     flux-scaled, size-robust **elongation** measure (≈0 round, large edge-on).
     `max`/`min` over orientations are done by reshaping the 12 values and
     `maxpool` (min via negation).
   - Gaussian **apertures** (σ=6 core vs σ=28 total) → **concentration** (bulge).
   - a **first-moment** → centroid offset (**asymmetry**).
   - annulus sum of the band-pass → **texture**; windowed `col` → **redness**.
4. **Literals and gates.** All cues are expressed as *fractions of the total
   flux*, so they are mutually comparable and the whole network is homogeneous
   (the prediction ignores overall scale). Thresholds give "round / edge /
   concentrated / red / blue …" literals. Conjunctions (AND) use
   `min(a,b)=a−relu(a−b)`; disjunctions (OR) are sums. A two-stage min network
   turns the literals into the ten class scores.

## Decision logic
- **Elongation** is the backbone: round → classes 2/3/5/6/7, elongated → 4/8/9.
- Within round, **colour** (red sequence vs blue cloud) splits smooth (2 round,
  3 in-between by elongation band) from spirals; concentration is a secondary cue.
- Edge-on galaxies are split **8 (blue disk) vs 9 (red bulge)** by colour, with a
  higher colour crossover than the round group.
- **Asymmetry + diffuseness** → merging (1).
- Bar/tight/loose cannot be separated by these features, so face-on spirals
  default to the plurality class **7 (loose)**; likewise classes with no reliable
  cue (0, 4, 5, 6) are left near-silent — predicting them would only add noise.

## Constants
Thresholds come from general knowledge (axis-ratio bands for elongation, the
red/blue galaxy colour split, a concentration index) typed into `build.py`. The
five permitted label checks were used only to confirm the pipeline works and to
pick reasonable scales for the colour/concentration/texture crossovers (not a
grid search, and no weight is fitted to the labels or the images).

## Reproduction
`build.py` reads no files: it computes every template from coordinate grids and
Gaussian formulas and writes `network.json` and `weights.npz`
(~0.94M weights, all finite float32). Verified with `check_rebuild.py`.
