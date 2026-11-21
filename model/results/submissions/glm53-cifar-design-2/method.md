# Method

**Architecture** (fixed filters + hand-written rule tables, no training):

1. **Front end (formula filters).** The RGB image is converted to opponent channels
   (intensity I, red–green A, blue–yellow Bb) by a 1×1 conv, then Gaussian-blurred
   (σ=1). From the blurred image, four banks of oriented first-derivative-of-Gaussian
   filters (0°/45°/90°/135°, both signs, unit-L1) give edge energy maps on all three
   colour channels; 3×3 Sobel-type filters on unblurred I give a fine-texture scale;
   difference-of-Gaussian filters at three scales (σ 1/2/3.5) give bright/dark blob
   detectors; two zero-DC annulus (ring) templates (radius 2.5 and 4.5 px) detect
   wheels (dark tyre rings / solid dark discs).

2. **Scene branch.** The edge maps and 4×4 block colour means are pooled onto an
   8×8 grid and concatenated channel-wise into one canvas (each branch is placed in
   its own canvas channels by a 1×1 conv, then all are summed with `add`).
   26 hand-written predicates read regions of the canvas: sky-blue top, blue
   corners, green bottom (grass), brown core (fur), warm overall, red/yellow core,
   brightness contrasts (core vs. average, bottom vs. average, top vs. average),
   horizon (full-width horizontal edge), smooth top/core (relative texture
   contrasts), centre-busier-than-border texture, and region-specific edge energies.

3. **Presence branch.** The max-pooled blob/ring/strong-edge maps are max-pooled
   again over three horizontal bands (rows 0–3, 2–5, 4–7), so localized evidence is
   *not* diluted by averaging. 19 thresholded predicates fire on band maxima:
   "a wheel in the lower third", "a small bright blob (eye) in the middle band",
   "a large dark body blob", "strong vertical edges below (legs)", "a strong
   horizontal edge across the middle (deck/horizon)", "diagonal strokes on top
   (ears/antlers/wing tips)", etc. Thresholds were set analytically: a matched
   template of contrast 0.3–0.5 responds ≈0.1–0.2 after unit-L1 normalisation,
   clutter ≈0.03–0.08, so predicates fire at ≈0.1.

4. **Scoring: deviation matching.** For every predicate I wrote down, from general
   knowledge of the classes, a prototype value p_ki (how much sky-blue a plane
   photo shows, how strong wheels are for an automobile, deer carry antlers,
   horses are bulkier, frogs sit in green scenes...) plus a reliability r_i
   (colour statistics are bounded and trusted, template responses less so).
   The final dense layers compute `score_k = 2 Σ_i r_i p_ki (x_i − p̄_i)`, where
   p̄_i averages the prototypes over the ten classes. Every class ties on a
   perfectly average image, so a class wins only where the image deviates from
   average in the direction its prototype predicts; features that sit at their
   average level in every photo (generic busyness) move no class. This was the
   key design decision: earlier rule-table attempts (validated with the 5 allowed
   labelled runs) collapsed onto whichever class expected the most of features
   that are in fact high in *all* photos.

**How the weights were produced.** All filter kernels are generated from closed-form
Gaussian/derivative/ring formulas with unit-L1 (and zero-DC where relevant)
normalisation. All rule tables, prototypes, thresholds and region choices were typed
by hand from general knowledge of what the 10 classes look like; nothing was fitted,
averaged, measured or read off any training image. The five allowed `run_network`
checks were used only to confirm the network computes and to observe the confusion
table; the prototype table was additionally verified to be self-consistent by pure
linear algebra (each class's own prototype vector must score highest for that
class, and does, with sensible runner-up pairs auto↔truck, cat↔dog, deer↔horse).
`build.py` reads no files and rebuilds the answer byte-for-byte.
