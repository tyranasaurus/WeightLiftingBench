# Method

**Architecture (66,718 weights, about 3,200 of them nonzero).** Three convolution branches read the image.
- **Main branch.** A 9x9 convolution on the image (padding 3, giving 30x30) computes colour differences (R-G, G-B, B-R as relu pairs), luminance, forward differences, 9-pixel horizontal and vertical line detectors, and a centre-surround dark-blob ("wheel") detector.
- **Pixel indicators.** Two 1x1 layers turn these into soft per-pixel indicators: blue, dark blue, green vegetation, warm/brown, vivid red, vivid yellow, neutral, white, smooth, busy, sky (smooth plus blue or bright cool white), and wheel. Each indicator is a ramp `relu(u) - relu(u-1)` or a soft AND `relu(sum - (n-1))` of ramps.
- **Pooling.** A 6x6 average pool gives a 5x5 grid of "fraction of this cell that is X".
- **Shape branch.** A 9x9 valid convolution measures 45-degree straight edges and coarse vertical versus horizontal luminance change.
- **Long-line branch.** Horizontal and vertical line detectors run on the 2x2-averaged image, so they only respond to edges that stay straight for about 18 pixels.

Dense layers then turn region fractions (top row, bottom row, centre, border, whole image) into about 23 scene quantities. Examples: sky in the top band, dark water in the bottom band, grass at the bottom, a warm object in the centre, a smooth border, texture, long lines, a white centre, and bright sky at the bottom. Each quantity is split into low, mid and high bins by two soft cut points. Six soft AND features are added, such as dark water below a brighter sky, smooth green bokeh, a warm object under sky, and wheel blobs on both sides. The output layer is a naive-Bayes readout. For every quantity I typed a letter per class (very low to very high) from general knowledge of such photos, and each letter maps to a fixed bin distribution. The weights are the log-probabilities.

**How the weights were made.** Every number comes from a formula or from hand-typed thresholds and letter tables in `build.py`. There is no gradient descent, no fitting, and no statistic of the training images or labels. I only looked at grids of training images to get a feel for the classes. I tested the implementation on procedurally drawn synthetic scenes (a sky plane, a ship on water, a deer on grass, a frog texture, a car with wheels, and so on), never on training images.

**The five allowed checks** each used a different 10,000-image training subset. After each one I revised reasoning-level choices rather than searching over variants:
1. 26.2%. The line and texture thresholds were set too low.
2. 29.1%. I raised those thresholds and added a bright-sky-at-bottom cue for airplanes.
3. 29.4%. I softened the letter scale, moved uncertain truck and car letters to medium, and changed the water cue to dark blue.
4. 30.7%. I switched to long-line detectors at half resolution and weakened the wheel cue.
5. 30.9%. I softened over-confident green levels for deer and horse. This version was committed.

**Weaknesses.** Bird and dog are rarely predicted, cat is the default animal, and car vs truck and airplane vs ship are often confused.
