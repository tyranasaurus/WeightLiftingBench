# Method

**Architecture.** A small hand-wired network with no trained weights:

1. A 1x1 conv + relu computes six "violations" of linear skin-colour rules in RGB (R/G between 1.12 and 1.8,
   G >= B, G/B <= 1.85, hue below about 32 degrees, R >= 0.30). A second 1x1 conv + relu turns them into a
   soft skin membership map in [0, 1] (1 - 20 x total violation).
2. The skin map is average-pooled into a pyramid (48, 24, 12 pixels).
3. At each pyramid level two 11x11 centre-surround kernels (an upright ellipse of skin minus a surrounding ring,
   with the neck/body sector below left out) detect isolated skin blobs such as faces and limbs: six scales in all.
4. Global max-pool per scale, then a clipped ramp (0 at response 0.35, 1 at 0.65) gives one vote per scale.
   The votes are summed; "person" wins when the sum exceeds 1.0.

**How the weights were produced.** Every number is typed in `build.py` from general knowledge of skin tones
(e.g. typical light/dark skin RGB ratios versus wood, sand and painted surfaces) and from the geometry of the
kernels (ellipse formulas). Nothing was fitted, measured on, or copied from the training images.

**Check 1 (the only check used).** 7 of 20 training images right (35%); 9 of the 10 no-person photos were called
"person". What I learned: the skin-blob cue fires on most no-person photos too (warm-coloured food, wood, painted
and warm-lit surfaces), so on its own, with this threshold, it does not separate the classes. No number was
changed after the check.

**Status.** This is an early, minimal version saved when time ran out: only the skin-blob cue is implemented.
Planned cues (scene colour layout such as snow/sky, face structure, greyscale-photo handling) are not included,
and the vote threshold is an uncalibrated hand choice.
