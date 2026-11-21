# Method

The network is a hand-written rule system expressed with the allowed operations; no weight was
fitted, trained, or taken from any image. `build.py` writes every number from formulas and typed
constants.

**Architecture (49 layers, 318k weights, 8.4k non-zero).** Four feature branches feed a readout.

- *Colour branch*: three 1x1 convolutions with ReLU "ramps" (`relu(a) - relu(a-1)`) turn RGB into soft
  per-pixel indicators chosen from general colour knowledge: blue, green, warm/brown (1.5R - G - 0.5B),
  red, bright, dark, sky (blue AND bright), water (clearly blue AND not bright), low saturation, white,
  gray, vivid (saturated but not green/blue). A 4x4 average pool gives 8x8 area fractions.
- *Edge branch*: a fixed gray conversion, 3x3 Sobel and Laplacian filters, then 1x1 ramps producing
  per-pixel "edge", "smooth", "mild texture" (band-passed Laplacian magnitude), "axis-aligned edge"
  (|dx| and |dy| very different) and "diagonal edge" indicators, pooled to 10x10.
- *Line branch*: 11-pixel straight horizontal/vertical line detectors (mean of a signed Sobel response
  along the line minus perpendicular energy), with max pooling and row/column-mean pooling to measure
  the longest line and line density.
- *Blob/bar branch*: 9x9 templates for a dark disc with brighter ring sectors on all four sides
  (wheel, eye) and thin dark/bright vertical bars with brighter/darker columns on both sides (legs,
  masts), max-pooled per quadrant and per 3x3 cell.

**Readout.** Dense layers compute 97 named cues as region means (top, bottom, centre, border, sides,
quadrants...) of those maps, plus a few AND-combinations (field = bright top AND green bottom,
seascape, sky scene, camouflage, warm-object-on-blue...). A final dense layer holds a hand-written
table of class weights: e.g. airplane = sky everywhere + smooth + white/gray object, ship = rough
water below a bright top + long horizontal line, automobile = gray road + vivid body + horizontal lines,
truck = edges to the top + long vertical lines, horse = warm body + green bottom + legs + sky top,
frog = green centre + camouflage, cat = gray/unsaturated furry centre, dog = warm/white centre, deer =
warm body in green/brown textured surround with legs, bird = small warm/dark object on sky/green.
Animal classes share a common "animal rather than vehicle" term (texture, curved edges, no long
straight lines, no vivid colours, no road/water).

**Process.** Weights were designed from first principles, debugged on synthetic patterns and cartoon
scenes drawn from formulas (never on photos), and then checked five times on a seeded 10k training
subset (19.0%, 22.6%, 25.4%, 23.6%, 25.7%). Between checks I adjusted thresholds, cue scales and the
class bias table by hand according to the confusion tables (e.g. texture fires almost everywhere at
32x32, so it was demoted to the shared animal term; a bluish road shade was being read as water).
