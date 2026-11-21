# Method

**Architecture.** A small hand-wired network: the image is average-pooled to 48x48; two 1x1 conv + relu layers
compute a soft skin-colour map (a cone in RGB given by five homogeneous linear inequalities: R > G > B, hue roughly
8-30 degrees, saturation roughly 0.18-0.68, plus a brightness floor); the map is pooled into a 48/24/12 pyramid;
at each level a 13x13 difference-of-Gaussians (sigma 1.0 / 2.5) conv with a fixed threshold detects compact or
limb-like skin-coloured blobs with non-skin surroundings (faces, arms, legs) while ignoring edges of large
skin-coloured areas (walls, tables); a global maxpool takes the strongest blob per level. A fourth branch
penalises photos where more than 35% of the area is skin-coloured (warm rooms, food, wood). Small dense layers turn
each branch into a two-class vote and an `add` sums them: "person" wins when the blob evidence exceeds a small
hand-set margin.

**How the weights were produced.** Entirely by `build.py` from formulas and constants typed from general knowledge
of skin colour and of how a centre-surround filter answers discs, bars, corners and edges (worked out analytically).
No training, fitting, or statistics of the training images were used; the training images were only viewed.

**Status.** Time ran out before the planned additional cues (upright-figure and scene-layout detectors) were built,
so this is a single-cue skin-blob detector; its thresholds are a-priori choices and were not tuned.
