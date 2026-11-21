# Method

**Architecture.**
1. A fixed dense layer turns the RGB image into three channels, each centred: high-passed luminance (L minus a Gaussian blur with σ = 3), R−G, and B−(R+G)/2.
2. A `norm` layer normalizes each channel per image.
3. Template matching uses 32×32 convolutions with padding 2, so every template is tried at ±2 px shifts. Each template has two parts:
   - a frame part (sign shape, border and field);
   - a pictogram part (digits or symbol), which may also shift ±1 px on its own (a second convolution followed by a 3×3 max-pool, then added to the frame part).
4. Max-pools take the best score over shifts, then over each class's 36 templates. These are 6 sign sizes (19.5–28.4 px) × 3 aspect ratios × 2 symbol sizes.

**Weights.** No weight comes from training data.
- The feature map is written as formulas.
- Every template is an individual drawing of a German road sign, typed by hand from general knowledge of StVO sign designs. Each drawing uses vector shapes (rings, triangles, octagon, diamond, DIN-style digit strokes, cars, trucks, arrows, pictograms) and camera-like colours.
- Each drawing is area-sampled at the given size, blurred (σ = 0.8), and passed through the same feature map. The chroma channels assume a neutral background.
- Each template is normalized by itself alone (zero mean and unit norm per channel), so each score is a sum of per-channel Pearson-style correlations.

**Checks.** I looked at training images only to learn what the classes look like. Three of the five allowed checks were used (checks.log): 61.7%, 73.5% and 71.4% on the training set. The submitted network is the one from check 2 (73.5%).
