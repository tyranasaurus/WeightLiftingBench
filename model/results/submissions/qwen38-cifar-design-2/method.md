# Method

## Architecture
A hand-designed, fully interpretable cascade (25 layers, ~118k weights):

1. **c1 (conv 5x5, 27 filters)** — a formula-built bank: oriented luminance edges and bars
   (1st/2nd derivatives of Gaussians at 0/45/90/135 deg, both polarities), small bright/dark
   blobs (DoG), fine texture (LoG), colour-opponent low-pass channels (R-G, B-(R+G)/2, both
   signs) and smoothed R/G/B. All edge/blob filters are L2-normalised, colour channels
   sum-to-one, so their meanings and scales are known analytically.
2. **c2 (conv 5x5, 32 filters)** — hand-composed detectors written as linear combinations of
   rectified c1 channels inside a 5x5 window: 8 corner types, dark round blobs (wheels/eyes),
   highlights, eye-with-bright-surround, smooth sky, smooth flat area, water (teal + horizontal
   ripple), broadband natural texture (fur/grass), strong horizontal/vertical lines, dark patch
   with bright frame (windows), diagonal limbs, bright/dark masses, colour regions
   (red/brown/green/grey/white/black), dense vertical/horizontal edge fields, permissive sky.
   Each unit's bias is auto-computed in `build.py` from posited background/feature response
   levels per channel group (general knowledge of photographs, no data), so it fires only when
   its positive evidence exceeds background by a fixed fraction.
3. **norm + relu** on c2 makes every stage-2 map illumination/contrast relative (per-channel
   z-score), which is what the later thresholds are written against.
4. **Pooling**: 4x4 avg and max, 2x2 avg and global avg of both the rectified c1 maps (absolute
   colour/edge/texture means) and the normalised c2 maps (relative part evidence).
5. **Head**: eight dense layers (one per pooled tensor) add into 43 shared "concept" units,
   relu, then a dense read-out to 10 class scores. Each concept unit is a hand-written rule
   over named pooled features (e.g. `sky_top` = top-half bluer than rest + smooth blue quadrant
   maps - top texture; `wheels` = dark-blob channels in the bottom half + dark bottom rows;
   `face_ctr` = eye-channel maxima concentrated in the centre + fur texture there - line/wheel
   channels). Almost every concept is a *within-image contrast* (row vs row, centre vs border,
   channel vs channel) so its threshold sits near zero and cannot drift with filter scale;
   each concept's bias auto-cancels its expected feature-less baseline plus a hand-set margin.
   The read-out is a hand-written rule table (e.g. ship = water-bottom + hull + horizon + sail -
   green-background - face), each row normalised to equal positive mass with a common bias.

## How the weights were produced
Every number comes from code in `build.py`: Gaussian-derivative/DoG/LoG formulas, hand-typed
5x5 composition masks and corner/ring/row patterns, and the hand-written concept and class rule
tables. No training image, statistic of the training set, or fitted quantity enters any weight;
`build.py` reads no files at all. The training set was used only to view example images of the
classes (montage PNGs) and for the 5 allowed check runs of the finished network.

## What the checks told me
The five logged runs were used purely as calibration diagnostics, and each led to hand-made
threshold/margin corrections (no fitting, no variant search): (1) stage-2 biases were far too
high so nearly all concepts were dead and the least-penalised class won by default; (2) global
means of channel-normalised maps carry no information, so those terms were dropped and the
z-max baselines raised; (3) a cell-indexing bug let 4x4 patterns leak into the 2x2 grid;
(4) "face"-type z-max units fired on any centred object, so they were gated by co-located fur
texture and by centre-vs-border contrasts; (5) the neutrality concept was non-positive by
construction and chroma-contrast margins were unreachable, so neutrality was rewritten with
negative weights on rectified chroma channels (negative margin) and the margin/slope balance
between chroma, spatial and part evidence was re-set by hand. Synthetic feature-tensor unit
tests (no images) were used throughout to verify weight layout and that each imagined scene
activates its intended concepts and class.
