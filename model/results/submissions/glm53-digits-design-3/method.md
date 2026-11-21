# Method

**Architecture.** A template matcher in a hand-designed feature space:
`conv1` (32 oriented bar filters) → `relu` → `avgpool 2×2` → per-channel
`norm` → `dense1` (one row per hand-drawn template digit) → `dense2`
(per-class mean of template scores) → 10 class scores.

**Filters (from formulas).** For each of 8 orientations, four zero-sum
DoG ridge ("bar") filters in an 11×11 frame: short (5×5), medium (7×7),
long (11×11) at across-stroke σ=0.85, plus a thick 7×7 at σ=1.4. The
three lengths give a straightness cue: long filters fire only on long
straight strokes (sides of 4/1/7) while arcs (loops of 9/8/0) fire mainly
short/medium ones. The `norm` layer standardises each channel over
positions, making matching invariant to ink/activation scale.

**Templates (hand-drawn pen-path models).** Each digit is typed as pen
strokes (Catmull-Rom splines through control points from general knowledge
of how the digit is written — e.g. the dominant closed-triangle "4" with a
short crossbar, the compact-loop "9" with a long right-descending tail).
A deterministic rasterizer draws them with a Gaussian brush, mimicking the
dataset's normalisation (fit to a 20×20 box, centre of mass centred).
110 styles per class (4+ shape families × shear/brush grids + 60
seeded random styles) × 3 rasterization profiles (binarize at 4×
resolution then area-average, with/without edge softening — chosen so the
template ink statistics bracket real MNIST stroke profiles) = 3300
templates. `build.py` computes each template's own feature map through
conv1/relu/pool/norm, blurs it (σ=0.9 cells, translation tolerance), and
stores `W_t = 2·g·blur(F_t)`, `b_t = −|g·F_t|²` with hand-set channel
weights g (long channels ×3, short/thick ×0.5). Row t of `dense1` is then
`−‖g·f − g·F_t‖² + const`, i.e. a negative squared distance to template t
in normalised feature space; `dense2` averages each class's 330 template
scores. Predicted class = argmax.

**How the weights were produced.** Everything comes from the source code
in `build.py` (fixed seed 123). The training set was used only for
looking (text renderings, aggregate ink/width summaries) to check that my
stroke models and rasterization match what the classes look like in
general; no training image, crop, statistic or label became a weight, and
no fitting/optimisation of weights against data was performed. Design
choices (filter lengths, channel weights, profiles, blur) were validated
on my own synthetic renderings across rasterization profiles and
distortions, plus 5 logged whole-network checks on training images.
