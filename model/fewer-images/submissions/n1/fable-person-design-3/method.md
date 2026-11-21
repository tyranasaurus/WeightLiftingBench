# Method

A small hand-written network, saved as a first working answer when time ran out. It has not been run on any
training image (0 of the 5 checks used), so its accuracy is unknown.

Architecture: 1x1 conv (5 units) + relu -> 1x1 conv (1 unit) + relu -> 4x4 average pool -> 5x5 conv (2 units) + relu
-> global max pool -> dense (2 scores).

How the weights were produced: all typed by hand in `build.py` from general knowledge, with no training, fitting
or image statistics.

- The first two layers compute a per-pixel "skin-ness" in [0, 1]: 1 minus penalties for breaking a skin-colour
  rule (R > G > B with moderate R-G and G-B differences, and not dark).
- The average pool reduces the skin map to 24 x 24.
- The 5x5 conv holds two centre-minus-surround blob detectors (3x3 centre vs 5x5 ring, one cell vs 3x3 ring),
  so compact skin patches such as faces and arms respond, while large warm areas (wood, sand, warm light) cancel.
- The global max pool takes the strongest blob at each scale; the dense layer says "person" when their sum
  exceeds a hand-chosen threshold of 0.45, and "no person" otherwise.

The thresholds are untuned guesses.
