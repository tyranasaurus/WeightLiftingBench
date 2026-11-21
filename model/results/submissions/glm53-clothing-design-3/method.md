# Method

The network is a hand-designed feature detector cascade; no weight was fitted
to any dataset.  Everything comes from formulas in `build.py` that encode
general knowledge of how the ten clothing classes look.

**Architecture** (see `network.json`):
1. `featS` - conv: raw copy, Gaussian blurs (sigma 1, 2) and two centre-surround
   blob kernels (for dark collar spots).
2. `cellM/cellT/cellR/cellHi/cellHiR/cellHiN/cellSat` - an occupancy grid:
   4x4 max-pool, then a per-cell threshold and rectification implemented as
   1x1 convolutions, plus a saturation circuit (`x - relu(x-cap)`) so a dim
   occupied cell reads the same as a bright one.  Channels: "cell occupied"
   and "cell strongly empty".  This makes all region detectors work equally
   for bright and dark photos.
3. `featD/featR` - rectified Gaussian-derivative channels (vertical,
   horizontal and the two diagonal directions): edge-energy maps.
4. `detC/detS/detE` - three dense "detector" layers (summed with `add`):
   about 33 hand-drawn detectors - fractions of occupied cells in regions
   (shoulder tops, under-arm corners at the hem, waist sides, leg gap,
   boot shaft left/right asymmetry, handle gap, top/bottom rows, ...),
   shape templates (T-shirt silhouette, dress flare, collar points, coat
   front panels), and edge-energy detectors (trouser inner-leg lines,
   shirt placket line, pullover knit texture, shoe sole line, bag top-edge
   line, sandal diagonal straps, handle gap darkness).
5. `detr` = relu, then `score`: a hand-written rule table (weights in
   `SCORE`) that adds evidence for and against each class, e.g.
   tee = shoulders + EMPTY under-arm corners + T-silhouette;
   pullover = shoulders + occupied under-arm corners + knit texture;
   shirt = shoulders + under-arm corners + placket line - knit texture;
   trouser = inner-leg edge lines + empty sides + narrow top;
   sandal = empty top and bottom rows + diagonal straps;
   boot = right-shaft-vs-left asymmetry + heel; bag = dark handle gap +
   body-top edge line + wide waist cells.

**Where the numbers come from**: detector thresholds and rule weights are
typed in as literals, chosen by reasoning about garment proportions.
The only derived numbers are the ten output biases: each is the mean score
of synthetic line drawings of that class, rendered by `render_probes` from
the same shape models (a T-shirt drawn from rectangles for sleeves and body,
a bag from a handle ring over a body box, etc.), with fixed seed.  No
dataset image or label was used anywhere in the build; the same drawings are
used at build time only for a printed self-check.  Two sanity runs on
training data were used (logged in checks.log) to verify the pipeline works
and to study the confusion pattern, after which detector geometry was
corrected using general knowledge of where garments sit in the frame
(confirmed by looking at printed class-mean summaries of the training set,
which the task allows).
