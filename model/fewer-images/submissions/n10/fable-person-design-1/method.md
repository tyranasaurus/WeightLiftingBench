# Method

**Architecture.** A small hand-wired detector for isolated, face-sized skin-coloured blobs.
1. Two 1x1 convs with relus turn each pixel into a skin membership in [0, 1]: five linear rules in RGB
   (hue between red and orange-yellow, moderate saturation, not too dark); the membership is
   1 minus 25 times the summed rule violations.
2. The skin map is average-pooled to six scales (factors 1, 2, 3, 4, 6, 8). At each scale a 9x9 conv
   measures the mean skin inside an upright ellipse and in seven ring sectors around it (the sector below,
   where the neck is, is left out). A 1x1 conv + relu gives centre minus a penalty for every sector that is
   more than a quarter skin, minus 0.45, so only compact blobs surrounded by non-skin respond.
3. A global maxpool takes the peak per scale, the six peaks are added, and a dense layer outputs
   score(person) = 10 * sum - 0.5 and score(no person) = 0. With no blob the answer is "no person".

**Weights.** All typed by hand or computed from formulas in build.py (colour thresholds from general
knowledge of skin tones; ellipse and ring-sector masks from geometry). No training, fitting or data use.

**Status.** Time ran out early: this is a first, single-cue design. Planned additions (face structure,
snow / sky scene cues) were not built. None of the 5 allowed checks on training images was run, so the
network's accuracy is unmeasured and every threshold is an untested first choice.
