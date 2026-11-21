# Method

## Architecture
`image -> c1 (5x5 conv, 14 channels) -> norm -> relu -> binarisation + pooling
branches -> nine dense layers summed with `add` -> relu -> dense (10)`.

* **c1** is a filter bank written from formulas only: intensity, 5x5 box blur,
  four oriented Gaussian-derivative edge pairs, two zero-sum second-derivative
  kernels (vertical / horizontal dark gap lines), and a centre-surround pair
  (dark hole / thin bright line).
* **norm** makes every channel brightness-invariant; after **relu** the
  foreground channel is a contrast-normalised mask.
* **Binarisation trick:** `min(x, 0.5*max) = x - relu(x - 0.5*max)` implemented
  with two dense layers (identity / negative-identity) and `add`, applied to
  the 14x14 presence map and the row/column occupancy profiles. Every occupied
  cell then carries exactly the same known gain, so *all* shape thresholds in
  the network are exact ratios (region mean / that gain), immune to image
  brightness, fill fraction and texture.
* **Branches:** 14x14 binary presence, 7x7 fill, 28-row and 28-column
  occupancy, global fill, unclipped channel maxima (gain references for the
  gap channels), unclipped 7x7 and 14x14 maps for the gap/edge channels.
* **Hidden layer (61 units):** 40 silhouette-template matchers (ten classes x
  four geometric variants: mirror / x- / y-rescale / open-front / strap-top),
  each a normalised correlation of the binary presence map against a 14x14
  silhouette I typed from general knowledge of how these garments, shoes and
  bags look; plus ~21 semantic part detectors (height/width/fill gates, trouser
  leg gap, coat front opening, bag handle hole + arc, clutch top band, shoe
  sole line, strappy-vs-solid upper, boot shaft column, sleeve length, collar
  notch depth, skirt flare).
* **Readout:** a hand-written rule table: class score = sum of its template
  variants plus weighted part evidence (e.g. trouser = templates + narrow +
  leg-gap - sleeves - flare; sandal = templates + short + strappy - solid -
  shaft; bag = templates + handle/clutch + fill - sleeves).

## How the weights were produced
Every number comes from `build.py` alone: kernel formulas, hand-drawn ASCII
silhouettes, hand-picked region boxes and hand-set rule weights/thresholds.
No training image, statistic of training images, or generated-image score set
any number. Training images were viewed (montages, occupancy summaries) only
to confirm the qualitative geometry of the classes; five labelled runs of the
network were used purely to catch implementation bugs (a non-zero-sum gap
kernel, mis-scaled thresholds, units firing on the wrong structure), and each
fix was a design correction reasoned from what the classes look like, not a
fitted value. Final observed training accuracy of this hand-built network:
47.0% (checks.log); the last round of fixes (blurred-channel binarisation,
open-coat and strap-dress template variants, dress evidence) was made after
the final check and is verified only on synthetic drawings.
