# Method

## Architecture (245 layers, 88,066 weight entries, 3,716 non-zero)

The network is a hand-wired rule system expressed with conv / relu / pooling / add / dense layers.

1. **Per-pixel colour analysis** (two 1x1 convs + relu). Six linear inequalities in RGB describe
   photographed skin (hue about 5-30 degrees, saturation about 0.15-0.58, not too dark); each becomes a
   soft indicator via a ReLU ramp pair, and the skin map is their soft AND. The same layers compute a
   "dark" map (luminance below 0.2-0.4, for hair), a chroma map |R-G|+|G-B| and a warm-tint map relu(R-B).
2. **Pyramids.** The (skin, dark) maps and the RGB image are average-pooled into cells of 2, 3, 4, 5, 6, 8
   and 12 pixels.
3. **Face templates** at six pyramid levels (faces of about 6-36 px). On a 9x9 cell grid: a 3x3 skin box,
   non-skin flanks beside it, a non-skin cap above that is dark (hair) or at least a different colour from
   the background, and a non-skin "body" box one cell below the chin whose colour differs from the
   background beside the head. Box means are computed by single-channel 9x9 convs with box-filter weights;
   the conditions are ReLU ramps and the AND is "sum minus (n-1)", then a global max-pool per level.
4. **Standing-figure templates** at five levels (people 24-64 px tall): a 1-cell head over a 3-cell
   torso and legs, all differing in colour from the flanks, with the row above the head and the cells
   beside the head uniform, and left/right background alike (sports fields, slopes, courts, walls). No
   padding is used so the image border cannot fake contrast.
5. **Global cues**: maximum chroma (black-and-white photo), mean warm tint with low maximum chroma
   (sepia / toned photo), and the mean skin fraction (very large skin-like areas in colourful images are
   penalised: wood, sand, orange light).
6. **Readout**: scale-weighted sums clipped to [0,1] (face, isolated skin blob, figure), plus grey and
   sepia bonuses and the big-area penalty, combined with hand-chosen weights 1.0 / 0.3 / 0.85 / 0.9 / 0.9 /
   -0.6 against a fixed threshold of 0.75 for class "person".

## How the weights were produced

All weights are written by formulas in `build.py`: box filters, ramp slopes/offsets from hand-chosen
thresholds, one-hot routing matrices and the hand-chosen readout weights. No training data, statistics,
fitting or search was used; thresholds come from general knowledge of skin colour, hair, clothing and
human proportions. Mechanics were debugged on synthetic patterns I drew (coloured rectangles). The five
allowed checks on training images gave accuracies 55.4%, 58.4%, 55.6%, 59.7% and 61.0%; the last
configuration is the submitted one.
