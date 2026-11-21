# Method

**Architecture (134 layers, 376k weights).** The network is a hand-crafted
feature extractor plus a hand-written rule layer; no weight was fitted or
trained anywhere.

1. **Illumination-robust colour maps.** A 1x1 conv splits the image into
   value / R-G / R-B / G-B channels. From these, using only `relu`, `add`
   and 1x1 convs (which implement the needed min/max/algebra), it computes
   *redness* `relu(R-max(G,B)-m)`, *blueness*, *yellowness* (rejects red and
   green), *greenness* (rejects yellow), and a *local blackness*
   `relu(mean5(V)-V-m)` (dark strokes next to bright surroundings). Each map
   is per-image normalised with the `norm` op (this kills illumination
   variation) and clamped to [-4,4] (built from relu shifts) so lens flares
   cannot dominate.

2. **Template banks.** 59 template features are 32x32 conv kernels
   (stride 1, padding 2) drawn from (a) geometric formulas — annuli for the
   prohibitory red ring, equilateral-triangle borders with an angular
   cos(3theta) mask that separates up-triangle / down-triangle / circle,
   solid-disc detectors for stop / no-entry, blue-disc detectors, a
   45-degree sinusoidal grating for end-of-limit stripes, white-rhombus
   detector for priority road — and (b) ASCII "shape models" typed from
   general knowledge of German signs: DIN-style digit strokes composed into
   the eight speed-limit numbers, the triangle pictograms (bends, bumps,
   slippery road, road works, pedestrians, children, bicycle, snowflake,
   deer, traffic-light box with three coloured dots), the truck/car blobs
   for classes 9/10/16/41/42, the no-entry bar and STOP letters, and
   white-blob quadrants for the mandatory arrows. Every kernel is a
   *mean-difference*: (mean of the map over the positive region) minus
   (mean over a negative region that expects the opposite), so any uniform
   map scores 0 and background level cancels. Each template is evaluated at
   5 scales (3 aspect variants for sign frames), shift-tolerated by a 5x5
   (or 3x3) maxpool and scale-tolerated by a maxpool over the scale
   channels (via a reshape that groups scales per template).

3. **Rule layer.** The 59 scores feed a dense+relu layer of 80 thresholded
   evidence units ("feature present above threshold", "feature absent
   below threshold") and a final dense layer into 43 class scores. The
   weights are a hand-written rule table: each class gets its group frame
   evidence (ring / triangle / blue disc / red disc / stripes / yellow
   diamond), its content evidence, negative weights on rival features
   (proportional suppression), and absence evidence where that is the
   signal (empty interior for class 15/18, no-red-ring for end signs).
   Absence units are auto-compensated in the bias so a neutral image gives
   every class exactly zero.

**How the weights were produced.** Every constant (ring radii, triangle
proportions, digit stroke layouts, pictogram shapes, thresholds) comes from
general knowledge of what German traffic signs look like, written directly
in `build.py`; the file reads no data. I debugged the pipeline entirely on
my own synthetic drawings of signs (generated in `/app/work/synth.py`, no
training images involved) and by looking at text renderings of a few
training images to check the general appearance of each class (permitted;
no pixel values or statistics were copied into any weight). The five
allowed on-data runs were not consumed.

**Known weaknesses.** The mandatory-arrow group (33-40) relies on
white-blob quadrant conjunctions that only weakly separate the eight arrow
variants; end-of-limit variants (41 vs 42) and the digit pairs 5/6, 3/8 are
the most fragile discriminators. Verification on the actual training set
(one of the five allowed checks) was still pending when time ran out.
