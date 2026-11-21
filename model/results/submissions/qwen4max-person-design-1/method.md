# Person / no-person at 96x96 — hand-designed weights

## Architecture
A fixed, fully hand-specified convolutional pipeline (271 layers, ~18k weights,
all from formulas or typed general knowledge — no training, no fitting, no
image-derived numbers). It is a multi-cue evidence accumulator:

1. **Colour front end (1x1 convs).** Luma `Y`, chroma differences `u=R-G`,
   `v=G-B`, `Cb`, `Cr`. A skin likelihood map `S` is a soft box (sum of hinge
   penalties, then relu) combining the classic YCbCr skin box with an
   illumination-scaled hue cone (`u ~ 0.08..0.6 Y`, `v ~ 0.25..1.25 u`, plus
   absolute floors so grey/white/B&W pixels are not skin). Channels `[S, Y,
   |u|+|v|]` are kept together and average-pooled into a 48/24/12 pyramid.

2. **Skin-blob detectors.** Zero-sum difference-of-Gaussian centre-surround
   kernels (surround std = k/4.5) at five scales, plus aspect-selective
   (tall vs wide) variants. A face/hand is a skin blob *surrounded by non-skin*;
   wide skin blobs flag animals/food.

3. **Head and face templates (on [S, Y]).** `head` = skin blob + darker hair
   above; `eyes` = bright skin disk with two dark eye spots; `fullface` = skin
   blob + hair above + two eyes together (the most specific cue). Two scales
   each. A mirror-symmetry map of the skin and a tall-blob luma detector add
   support for upright/clothed/silhouette figures.

4. **Aggregation.** Each detector map is reduced to its global max, its global
   mean, and a thresholded density (mean of relu(D-t)); region means of the skin
   grid give layout/uniformity statistics. These scalars are placed into slots
   of one feature vector (one-hot dense + add).

5. **Decision (dense(10) -> relu -> dense(2)).** Ten hand-written rules:
   whole-face, eyes, head+hair, several-people, upright figure, isolated skin
   blob, and suppressors for uniform skin sheets (food/wood/sand), horizontal
   skin (animals), gated so a detected close-up face is not suppressed.
   "Max - mean" peakiness terms favour an isolated strong face over widespread
   clutter. With no strong specific structure every unit is 0 and the tie goes
   to "no person".

## How the weights were produced
Every kernel is computed from a formula (`gk` Gaussian window, `dogk`
difference-of-Gaussians, Sobel, mirror-difference). Template geometry (eye
spacing ~0.22 of face width, hair above, head above shoulders) and the skin
chroma cone come from general knowledge of human appearance and colour science.
Decision weights/biases are a hand-written rule table. The five permitted runs
on the training sample were used only to check the pipeline works and to set a
sensible operating point (the network defaults to "no person" unless specific
person structure is present); no training statistic, image patch, or fitted
readout enters any weight. `build.py` reads no files and reproduces
`network.json` + `weights.npz` exactly (verified by check_rebuild).
