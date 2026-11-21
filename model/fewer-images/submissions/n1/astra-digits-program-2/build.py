"""Data-free handwritten digit recognizer builder. The paths below use pen coordinates
in tenths of a nominal glyph; they are descriptions, not digitized examples.
"""
import sys
from pathlib import Path
import math
import numpy as np

# M: lift/move; L: line; Q: quadratic curve; C: cubic curve. A loop which
# reconnects at a point repeats the same anchor. Randomized positions of
# repeated anchors remain identical within a single drawing.
styles = [
    [ # 0 -- variations on a single, sometimes imperfect, circuit.
        (8, 'M 6.4 .5 C 2 -.8 -.6 3.3 .5 7.4 C 1.5 12 7 11 9 6.8 C 11.1 2.2 9 -.5 5 .4'),
        (6, 'M 5 0 C 2.1 -.2 .1 1.9 0 5.6 C -.2 8.9 1.9 10.2 4.6 10 C 7.9 10 10 7.7 10 4.6 C 10.2 1.5 8 .1 5 0'),
        (4, 'M 4.7 .1 Q .3 1.3 .4 6.8 Q .2 10.4 3.3 10 Q 9.7 10.1 9.6 3.7 Q 9.4 -1.7 4.7 .1'),
        (3, 'M 7.6 .2 C 3.2 -.7 .3 2.7 .5 6.6 Q .3 10.3 3.7 10 C 7.9 10.5 10.1 6.1 9.6 3.1 Q 9.3 .1 7.2 1.2 Q 6.4 1.7 6.1 2.4'),
        (2, 'M 6.7 .4 Q 3.8 -.6 2.5 1.7 Q -.3 4.7 .7 8.1 Q 1.3 10.5 4.3 10 Q 8.3 9.8 9.5 5.5 Q 10.8 1.8 8.1 .9'),
        (2, 'M 3.3 .4 C .1 1.4 -.3 7.9 1.2 9.3 Q 3.7 11.1 7.6 8.8 C 11.5 5.9 10.6 -.9 6.5 .1 Q 4.1 .1 2.8 1.7'),
        (1, 'M 2.7 .6 C .5 1.7 -.4 7.7 1.2 9 Q 3.4 11.8 7.1 9.4 C 10 7.4 11 2.2 8.7 .6 Q 6 -1.4 2.7 .6'),
    ],
    [ # 1 -- pen descends almost straight; entry flag and base are optional.
        (7, 'M 5.3 0 L 4.9 10'),
        (5, 'M 5 0 Q 4.2 5.8 4.9 10'),
        (5, 'M 5.1 0 Q 6.5 4.1 5.1 10'),
        (4, 'M 3.7 1.4 Q 4.9 .2 5.2 0 Q 5.9 1.6 5.1 10'),
        (4, 'M 2 2.6 L 5.7 0 L 5.7 10'),
        (3, 'M 1.3 3.7 Q 4.2 1.9 5.7 .1 Q 6.6 2.3 5.7 10'),
        (2, 'M 2 3 L 5.7 .1 L 5.7 9.8 M 2.1 10 Q 5.3 9.6 9 9.8'),
        (1, 'M 5 0 L 5.2 10 M 1.7 10 Q 5.6 9.7 8.4 10'),
        (1, 'M 4.8 .8 Q 6.7 -2 6 2.7 L 5.3 10 Q 6.5 9.7 7.5 9.5'),
    ],
    [ # 2 -- arch, descending diagonal, rightward baseline. Cursive foot may loop.
        (7, 'M .3 2.4 C 2.3 -.9 7.7 -.8 8.7 1.7 C 9.6 3.6 6.3 5.9 3.3 8 L .5 10 Q 4.2 9.5 9.9 10'),
        (5, 'M 1.4 1.1 Q 8.8 -1.8 8.2 2.4 Q 7.9 4.7 .2 9.8 Q 4.5 10.1 10 9'),
        (4, 'M 1.2 3.4 C -.5 .2 5.8 -1.2 7.3 1.1 Q 9.5 3.8 1 9.6 Q 2.1 10.9 5.5 9.8 Q 8 8.4 10 8.9'),
        (5, 'M .8 2 Q 1.7 -.6 4.6 .2 C 9.4 1.1 8.4 5.3 3.6 8.7 Q .3 11.2 .4 9 Q .8 7.4 4.5 8.4 Q 8.5 10.7 10.5 9.2'),
        (4, 'M 1.4 .8 Q 4.5 -1.6 6.5 1.9 C 8.7 6.5 -.2 12.8 .6 9.2 Q 1.4 6.4 4.9 7.8 Q 9.2 9.4 10 8.8'),
        (3, 'M 1.2 1.3 C 4.2 -1.3 7.8 .6 7.5 3.1 C 7.1 6.5 .8 8.4 .2 9.9 Q 2.6 8.9 5.3 9.6 Q 8 10.4 10 9.9'),
        (2, 'M 1.6 3.7 Q -.8 .5 3.5 .1 Q 8.3 -.2 8.3 2.9 Q 8 5 .6 9.4 Q 1.3 10.8 4.4 9.9 L 10 8.6'),
        (2, 'M .3 2.6 Q 4.7 -.4 8.3 .2 Q 9.7 .8 8.3 2.4 L .2 9.9 L 10 9.7'),
        (2, 'M .4 .4 Q 4.3 -.8 6.3 2.7 Q 8 5.8 4.7 8.5 C 1 11.9 -1 9.8 1 8.1 Q 3.4 6.2 6.1 7.5 Q 8.2 8.6 10.6 8.8'),
        (1, 'M .4 2.5 C 1.9 -.9 7.6 -.8 8.6 1.6 Q 9.8 4.4 3.4 8 Q -.1 10.7 1.2 10.1 Q 4.5 8.5 7.3 9.4 L 10.3 10.5'),
    ],
    [ # 3 -- two right-facing lobes, round or angular, with or without a retrace.
        (7, 'M .5 1.5 C 3.5 -.3 8.7 -.8 8.5 1.9 Q 8.4 3.9 3.3 4.8 C 9.6 3.6 11 7 7.7 8.9 Q 3.6 11.6 .1 9'),
        (5, 'M .5 .8 Q 4.8 -.5 8.7 .2 Q 8.4 2.2 3.4 4.9 Q 8.9 3.7 9.4 6.8 Q 9.9 10.6 1 9.9'),
        (4, 'M 1 1 Q 5.4 -1.5 7.8 1.2 Q 9.1 2.6 5 4.8 Q 8.3 5 9 7 Q 9.8 9.8 5.8 10 Q 2.1 10.3 .3 8.2'),
        (4, 'M 1.2 .2 Q 5.1 -.4 8.4 .5 L 3.8 4.4 C 7.1 4.1 10.6 5.6 9.3 8 Q 7.4 11.1 .7 9.6'),
        (3, 'M 1.1 .8 Q 6.5 -1.2 7.8 1.4 Q 8.9 3.2 4.4 4.8 Q 1.1 5.9 2.6 4.5 Q 5.5 3.4 8.5 5.6 C 12 9.2 3.4 12.1 .3 8.6'),
        (2, 'M 2.2 .4 Q 5.2 -.7 7.4 .8 Q 9.3 2.2 4.9 4.3 Q 7 5.5 8.2 7.4 Q 10.1 11.3 1.4 9.8'),
        (2, 'M .7 2.4 Q 1.1 -.1 5.1 .1 Q 9.4 .2 8.1 2.6 Q 7.3 4.4 3.2 4.6 Q 7 4.7 8.6 6.1 Q 10.8 8.9 5.2 9.9 Q 2.1 10.3 .5 9.2'),
        (1, 'M .4 .2 Q 4.9 -.4 8.8 .3 L 4.1 4.9 Q 9.4 5.2 9.2 7.3 Q 9 9.6 2.2 10'),
    ],
    [ # 4 -- open chair, rounded cup, closed triangle, continuous or two strokes.
        (7, 'M 2.2 .3 Q .6 3.3 .4 5.7 Q 1.1 6.8 4.3 5.9 L 9.5 4.8 M 7.7 .2 Q 7.1 4.9 7.6 10'),
        (6, 'M 2 .4 L .6 5.8 Q 2.4 6.4 8 5.9 M 8 .2 L 7.6 10'),
        (5, 'M 1.1 .5 Q -.8 6.1 2.1 6 Q 5.2 5.9 7.8 3.5 M 8.1 .1 Q 6.7 4.2 7.4 10'),
        (4, 'M 5.5 .2 L .2 6.1 L 9.8 5.9 M 7.2 .4 L 7.1 10'),
        (3, 'M 7.2 .1 L .3 6.2 L 10 6 M 7.2 .1 L 7.4 10'),
        (4, 'M 1.8 .6 L .6 4.9 Q .3 7.1 4 5.5 Q 7.1 4.2 7.9 .1 Q 7 5.5 8.4 10'),
        (2, 'M .7 .4 Q .2 2.6 .6 4.6 C 1 7.3 6.8 5 8.4 3.7 M 7.6 .7 Q 7.5 5.2 8.1 10'),
        (2, 'M 3.7 .1 Q .8 3.4 .2 6.1 Q 4.7 6.4 10.6 4.9 M 7.2 2 L 6.5 10'),
        (2, 'M 7.7 10 L 7 .1 L .4 6.2 Q 5.2 5.7 10.4 5.8'),
        (1, 'M .5 .8 Q 1.1 4.9 2.4 5.1 Q 4.4 5.5 8.4 1 M 8.4 1 Q 7.6 6.2 8.9 10'),
    ],
    [ # 5 -- capital cap and left stem above a right-facing bowl. In many hands
      # the cap is attached last, or the convex part extends to the left of it.
        (7, 'M 9.2 .3 Q 5.2 .9 2.2 .6 L 1.2 4.7 C 4.5 3.7 9.7 4.5 9.1 7.3 Q 8.1 11.2 .6 9.6'),
        (5, 'M 9.4 .2 L 2.8 .6 Q 1.9 2.8 1.2 4.8 Q 5.7 3 8.1 5.4 Q 10.9 8 6.5 9.7 Q 2.9 11.2 .6 8.8'),
        (5, 'M 3.6 .6 Q 1.5 1.6 1.8 3.3 C 2 4.6 8.8 5 8.2 7.9 Q 7.6 10.7 .8 9.8 M 2.6 .7 Q 5.5 .8 9.7 -.2'),
        (4, 'M 8.8 .3 C 4.7 1.9 2.2 -.2 1.5 2 Q .8 3.5 4.7 4.7 C 12.4 7.3 6.7 11.5 .3 9.4'),
        (3, 'M 9.5 .1 Q 6.5 .5 3.9 .3 Q 2.8 .3 2 3.9 Q 7.1 4.2 8.8 6.3 C 11 9.3 3.2 11.6 .9 9'),
        (3, 'M 2.7 .2 L 1.9 3.6 Q 4.4 3.4 7.1 5.2 Q 10.5 7.8 6.5 9.4 Q 2.5 11 .3 9.1 M 2.6 -.1 Q 6.1 1.1 10.3 .1'),
        (3, 'M 8.2 .3 Q 5.1 .1 3.3 1.3 C -.8 3.8 2.3 4.3 5.4 5.1 Q 11.1 6.9 6.9 9.3 Q 3.6 11.1 .4 9.8'),
        (2, 'M 8.3 .1 L 3.1 .7 L .8 5.1 Q 3.8 3.3 7.2 4.4 C 11.7 6.4 7 11.5 1.6 9.7'),
        (2, 'M 5.5 .1 Q .6 .8 1 2.6 Q 1.3 3.6 4.4 4.6 C 12.4 7.7 6.3 11.5 .8 9.7'),
        (2, 'M 10 .1 Q 7.6 1.4 4.2 .8 L 3.2 4.5 Q 10.1 4.1 8.8 7.3 C 7.9 9.8 3.2 11.6 .3 9.4'),
    ],
    [ # 6 -- long descending entry curls under into a low counter.
        (7, 'M 7.8 .1 C 3.7 1.7 .3 6 .9 8.3 C 1.4 11.4 7.7 10.5 9 7.9 C 10.8 4.4 6 3.8 2.5 6.1 Q 1.4 6.8 1.2 7.5'),
        (5, 'M 7.8 .1 Q 2 .9 .7 6.4 Q -.4 10.2 4 10 C 7.9 10.3 10.1 7.7 8.8 6 Q 6.8 3.4 1.1 7.1'),
        (4, 'M 5.8 .1 C 2.1 2.3 .3 5.1 1.1 8.3 Q 2.5 11.1 6.7 9.5 Q 10.6 8 8.2 6 Q 6 4.4 2.3 7'),
        (4, 'M 8.8 .3 Q 5.9 -1.1 3.4 1.9 Q -.2 6.3 1.2 8.7 Q 3.2 11.7 7.5 9.1 C 11.5 6.3 7.1 2.9 2.7 6.9 L 1 8.2'),
        (3, 'M 6.1 .1 Q 2 2.7 1.1 6.2 Q .2 9.7 4.2 9.9 Q 8.9 10.4 9.3 7.7 Q 9.8 5 6.6 5.6 Q 4.6 6.1 3.6 7.4'),
        (3, 'M 5.8 .2 Q .7 4.7 1.9 8.1 C 3 11.5 9.5 10.3 9.1 7.3 C 8.7 4.1 4.2 4.8 1.6 6.8'),
        (2, 'M 6.4 0 Q 2.1 2.4 1 6.2 Q .2 8.8 2.4 9.7 Q 5.7 11 8.4 9 Q 11.4 6.4 7 5.5 Q 4.9 5.4 3.2 6.5'),
        (2, 'M 6.2 .2 C 1.8 3.3 .3 6.1 2 8.5 Q 3.5 11 7.7 9.4 Q 10.5 8.2 8.2 6.3 Q 5.5 4.4 1.8 6 Q .7 6.3 .1 7.1'),
        (1, 'M 8.5 .7 Q 7.1 -.7 4.2 .9 C -.4 3.6 -.3 10.8 4.4 10 Q 9.7 9.8 8.5 6.2 Q 7.5 3.7 3 5 Q 1.5 5.5 1.2 6.9'),
    ],
    [ # 7 -- cap then a descending leg. Initial tick and crossbar are optional.
        (7, 'M .3 .6 Q 4 .2 9 .4 L 3.5 10'),
        (5, 'M .3 .1 Q 3.6 1.3 8.9 .6 Q 5.4 5 4.7 10'),
        (5, 'M .1 1.3 Q 3 -.4 8.6 .2 Q 9.4 .3 8.3 2 Q 4.7 7.3 3.7 10'),
        (4, 'M .3 2.3 L .7 .2 Q 4.4 1 9.2 .5 L 4.3 10'),
        (3, 'M .3 .6 Q 4.2 .1 9.2 .3 L 3.8 10 M 2 5.5 Q 5.1 5.1 8.9 5.1'),
        (3, 'M .2 .2 Q 4.2 1.3 9.1 .5 Q 6.4 4.8 5.6 10 M 3.3 4.9 L 8.6 5.1'),
        (2, 'M 1.3 2.5 Q .6 .1 2.4 .2 Q 7.3 .2 8.7 -.2 Q 7.2 3.6 4.6 10'),
        (2, 'M .5 .8 Q 4.5 -.6 8.5 .6 Q 8.9 3.6 5.8 7 Q 3.5 9.6 4 10'),
        (2, 'M .4 .1 Q 4.4 1 8.7 .5 Q 8.1 6.4 4.9 10'),
        (1, 'M .3 1.6 Q .5 -.2 1.6 .3 Q 6 1.1 8.9 .3 Q 5.9 4.5 3.6 10 M 2.4 5 L 8.9 5.2'),
    ],
    [ # 8 -- figure-eight path, usually asymmetric; either starting direction.
        (7, 'M 6.7 .4 C 3.4 -1.1 -.1 1.5 2.1 3.5 C 4.4 5.6 10.9 6.3 8.4 9 C 5.9 11.5 -1.6 10.4 .9 7.2 C 3.2 4.4 10.5 3.1 8.2 1 Q 7.6 .4 6.1 .2'),
        (6, 'M 6 .2 C 2.9 -.7 .4 1.8 2 3.6 C 4 5.7 9.5 7.1 8.6 9 Q 7.7 11 3.9 9.9 C -.7 8.7 .3 6.4 4.6 4.8 C 9.5 3.1 10 -.1 6 .2'),
        (4, 'M 4.6 4.9 C -.4 2.8 1.1 -.7 6.2 .2 C 10.3 .8 9.3 3.2 4.6 4.9 C -.7 7.3 -1 10.2 4.9 10 C 11.8 9.6 9.1 6.4 4.6 4.9'),
        (4, 'M 8 .7 C 4.7 -.8 .8 .4 1.3 2.4 C 1.8 5.1 9.2 5.8 9.3 8.7 C 9.3 11.5 .3 10.3 .6 7.9 C 1 5.2 7.4 4 8.2 2.3 Q 8.7 .9 6.4 .6'),
        (3, 'M 6.8 .2 C 3.4 -.7 -.2 1.9 2.4 4 C 4.1 5.7 9.1 7.5 8.3 9.2 Q 7.6 10.9 4 9.7 C -1 7.4 2.3 5.4 4.7 4.5 Q 9.5 2.4 7.5 .7'),
        (3, 'M 2.8 .6 Q 6.6 -1.3 8.7 1.2 Q 10.2 3.1 4.8 5 C -.4 7.1 -.9 10.2 4.2 10 C 10.9 9.8 10 7.2 4.8 5 C 1 3.4 .3 1.7 3.8 .3'),
        (2, 'M 4.6 5.2 C .5 4.5 -.8 1.8 3.3 .5 C 7.5 -1 10.5 1.2 8 3.4 Q 6.2 4.7 4.6 5.2 C .7 6.2 .2 9.8 4.4 10 C 9.5 10.5 10 6.2 4.6 5.2'),
        (2, 'M 8.9 1.3 Q 5.6 -1.5 2.6 .6 Q -.2 2.5 3.5 4.7 C 6.8 6.6 9.2 7.5 8 9.4 C 6.2 11.4 -.4 9.2 1.4 6.8 Q 2.5 5.2 5.7 3.9 Q 8.4 2.7 8 1.5'),
        (1, 'M 4.8 4.8 C -.1 4.5 -.4 .6 4.2 .3 C 9.8 -.4 10.2 3.9 4.8 4.8 M 4.8 4.8 C -.5 5.1 -1 9.9 4.1 10 C 10.6 10.4 11.1 5.6 4.8 4.8'),
    ],
    [ # 9 -- high counter followed by a downstroke, sometimes hooked or open.
        (7, 'M 8.6 1.7 C 7.2 -.9 3.3 -.1 1.6 1.8 C -.9 5.4 2.5 6.6 5.4 5 Q 7.3 4 8.5 .7 Q 7.4 4.3 7.9 10'),
        (5, 'M 7.9 1 C 5.1 -1.1 1.4 .6 .8 3.1 Q .1 6 3.7 5.7 Q 7.6 5 8.3 1.3 L 7.6 10'),
        (4, 'M 8 .5 C 3.9 -1.1 -.1 2.2 1.4 4.5 Q 3.5 6.7 7.7 3 M 8.2 .5 Q 7.2 4.4 7.8 7.7 Q 8 9.4 7.5 10'),
        (4, 'M 8.1 1.1 C 5.9 -.9 2 .1 .8 2.7 C -.4 6 4.9 6.2 7.6 2.8 Q 8.5 1.6 8.3 .8 Q 9 5.7 5.4 9.2 Q 4.6 10.1 3.3 10'),
        (4, 'M 7.4 1.1 Q 5.8 -1.4 2.6 1 Q -.8 4.2 1.9 5.4 Q 4.6 6.6 7.9 2.5 Q 7.4 6.2 8.9 10'),
        (3, 'M 6.6 .5 C 3.2 -.4 -.9 3.6 1.5 5 Q 3.6 6.3 6.8 3.2 L 8.3 1 Q 7.5 5.2 7.6 10'),
        (3, 'M 7.3 .1 C 3.4 -.1 .3 2.1 1 4.3 C 1.9 7 7.3 5.2 8 2.5 Q 9 .2 7.3 .1 M 8.1 1.4 Q 9.6 5.1 7 8.8 Q 5.5 10.8 2.8 9.7'),
        (2, 'M 6.4 .2 Q 2.9 -.3 1.4 2.4 Q -.1 5.7 2.9 5.7 Q 6.3 5.8 8 2.4 Q 8.8 .7 8.2 .4 L 7.9 7.6 Q 7.7 9.6 9.2 10'),
        (2, 'M 7.5 1.3 C 6.9 -.7 3.9 -.4 1.6 1.4 Q -.9 4.1 2.4 4.9 Q 5.6 5.5 8.1 1.5 Q 9.1 5 6.1 10'),
        (2, 'M 6.4 .2 Q 2.1 .1 .8 2.8 Q -.2 5.5 2.9 5.5 Q 5.2 5.4 7.8 2.9 M 8.1 .4 Q 7.4 4.8 7.9 10'),
    ],
]

# Below, the same per-image normalization and descriptor code is embedded in
# the answer and used offline on each separate template. Reductions in these
# functions never mix rows. No features are learned from the drawings.
features = r'''import numpy as np

axis = np.arange(28, dtype=np.float32)
sy, sx = np.mgrid[:28, :28].astype(np.float32)
grid = np.arange(24, dtype=np.float32) - 11.5

def extent(a):
    s = np.cumsum(a, axis=1)
    total = s[:, -1:]
    left = np.sum(s < total * .009, axis=1)
    right = np.sum(s < total * .991, axis=1)
    return left.astype(np.float32), right.astype(np.float32)

def sample(a, y, x):
    x = np.clip(x, 0, 26.999)
    y = np.clip(y, 0, 26.999)
    ix = x.astype(np.int32)
    iy = y.astype(np.int32)
    dx = (x-ix).astype(np.float32)
    dy = (y-iy).astype(np.float32)
    n = np.arange(len(a))[:,None,None]
    return ((1-dy)*((1-dx)*a[n,iy,ix] + dx*a[n,iy,ix+1]) + dy*((1-dx)*a[n,iy+1,ix] + dx*a[n,iy+1,ix+1]))

def canonical(a):
    a = np.maximum(a - .025, 0)
    a = a / np.maximum(.01, a.max(axis=(1,2),keepdims=True))
    ry = a.sum(axis=2)
    mass = np.maximum(.001, ry.sum(axis=1))
    cy = (ry*axis).sum(axis=1)/mass
    cx = (a.sum(axis=1)*axis).sum(axis=1)/mass
    dys = sy-cy[:,None,None]
    sl = (a*dys*(sx-cx[:,None,None])).sum(axis=(1,2)) / np.maximum(1,(a*dys*dys).sum(axis=(1,2)))
    sl = .9*np.clip(sl,-1.6,1.6)
    # Expand the sheared canvas when estimating its width (no clipping).
    proj = np.maximum(0, a-.1)
    kx = np.floor(sx-sl[:,None,None]*dys+32.5).astype(np.int32).clip(0,95)
    offset = 96*np.arange(len(a))[:,None,None]
    hist = np.bincount((kx+offset).ravel(), weights=proj.ravel(), minlength=96*len(a)).reshape(-1,96)
    l,r = extent(hist)
    t,b = extent(proj.sum(axis=2))
    xc = .76*((l+r)*.5-32)+.24*cx
    yc = .76*((t+b)*.5)+.24*cy
    w = np.maximum(1,r-l+1)
    h = np.maximum(1,b-t+1)
    # Partial aspect normalization: do not make a bare 1 fill a square.
    yy = yc[:,None,None] + grid[None,:,None] * (h/19)[:,None,None]
    xs = np.maximum(.28*h, (.76*w+.24*h*.75))/18
    xx = xc[:,None,None] + grid[None,None,:]*xs[:,None,None] + (yy-cy[:,None,None])*sl[:,None,None]
    norm = sample(a, yy, xx)
    aspect = np.clip(np.log(w/h), -2.7, 1.2)
    return norm, aspect.astype(np.float32)

def smooth(a):
    b = np.pad(a, ((0,0),(0,0),(1,1)),mode='constant')
    b = .25*(b[:,:,:-2]+2*b[:,:,1:-1]+b[:,:,2:])
    b = np.pad(b, ((0,0),(1,1),(0,0)),mode='constant')
    return .25*(b[:,:-2]+2*b[:,1:-1]+b[:,2:])

def pooling(a,side):
    b = a.reshape(len(a),24//side,side,24//side,side)
    return b.mean(axis=(2,4)).reshape(len(a),-1)

def unit(a):
    return a / np.sqrt(np.maximum(.00001,(a*a).sum(axis=1,keepdims=True)))

def holes(a, cut):
    # Mark the outside by expanding through background. The area and vertical
    # position of unreached pockets distinguish bowls from open arches.
    bg = a < cut
    bg = np.pad(bg,((0,0),(1,1),(1,1)),constant_values=True)
    wet = np.zeros_like(bg)
    wet[:,0] = True
    wet[:,-1] = True
    wet[:,:,0] = True
    wet[:,:,-1] = True
    for step in range(56):
        old = wet.copy()
        wet[:,1:] |= old[:,:-1]
        wet[:,:-1] |= old[:,1:]
        wet[:,:,1:] |= old[:,:,:-1]
        wet[:,:,:-1] |= old[:,:,1:]
        wet &= bg
        if step%8 == 7 and np.array_equal(wet,old):
            break
    pocket = (bg & ~wet)[:,1:-1,1:-1].astype(np.float32)
    # Smoothed vertical histogram rather than exact component count/centroids.
    row = pocket.sum(axis=2)
    out = []
    for y in [4,8,12,16,20]:
        mask = np.maximum(0,1-abs(grid+11.5-y)/5)
        area = (row*mask).sum(axis=1)
        out.append(np.sqrt(area/(area+6)))
    return np.stack(out,axis=1)

def describe(a):
    a, asp = canonical(a)
    b = smooth(a)
    c = smooth(b)
    # Multiscale, sign-sensitive gradient distributions and soft occupancy.
    features = [unit(pooling(c,3))*.57]
    hog = []
    for level,block,weight in [(b,4,.65),(c,6,.50)]:
        pad = np.pad(level,((0,0),(1,1),(1,1)))
        gx = pad[:,1:-1,2:] - pad[:,1:-1,:-2]
        gy = pad[:,2:,1:-1] - pad[:,:-2,1:-1]
        mag = np.sqrt(gx*gx+gy*gy+.000001)
        chans = []
        for k in range(8):
            ang = k*np.pi/4
            response = np.maximum(0, gx*np.float32(np.cos(ang)) + gy*np.float32(np.sin(ang)) - .67*mag)
            chans.append(pooling(smooth(response),block))
        vectors = np.stack(chans,axis=2)
        local = np.sqrt((vectors*vectors).sum(axis=2,keepdims=True))
        vectors = vectors / np.sqrt(local+.026)
        features.append(weight*unit(vectors.reshape(len(a),-1)))
    f = unit(np.concatenate(features,axis=1))
    hole = .42*holes(a,.31) + .38*holes(a,.50) + .20*holes(a,.68)
    # A lower contrast medial map: fill small pen-width differences without
    # losing the sharp silhouette needed by the deformable local matcher.
    ink = np.minimum(1, .65*a+.55*b)
    aux = np.concatenate([asp[:,None],hole],axis=1)
    return f, ink, aux
'''

# Build-time primitive renderer.
def parse(s):
    tokens = iter(s.split())
    result = []
    for cmd in tokens:
        pts = tuple((float(next(tokens)),float(next(tokens))) for j in range({'M':1,'L':1,'Q':2,'C':3}[cmd]))
        result.append((cmd,pts))
    return result

styles = [[(w,parse(path)) for w,path in rows] for rows in styles]

def penpath(commands, rng):
    points = {}
    strokes=[]
    for cmd, pts in commands:
        for p in pts:
            if p not in points:
                points[p] = np.array(p)+np.clip(rng.normal(size=2),-2,2)*.26
        args = [points[p] for p in pts]
        if cmd == 'M':
            cur = args[0]
            strokes.append([cur[None,:]])
        else:
            ctrl = np.stack([cur]+args)
            d = np.linalg.norm(np.diff(ctrl,axis=0),axis=1).sum()
            t = np.linspace(0,1,max(4,int(d*6)),dtype=np.float64)[:,None]
            if cmd == 'L': curve = (1-t)*cur+t*args[0]
            elif cmd == 'Q': curve = (1-t)**2*cur+2*(1-t)*t*args[0]+t*t*args[1]
            else: curve = (1-t)**3*cur+3*(1-t)**2*t*args[0]+3*(1-t)*t*t*args[1]+t**3*args[2]
            strokes[-1].append(curve[1:])
            cur=args[-1]
    return [np.concatenate(s) for s in strokes]

def render(commands,rng, digit):
    strokes=penpath(commands,rng)
    lengths=[len(s) for s in strokes]
    p=np.concatenate(strokes)
    u,v=(p.T-5)/5
    # Smooth correlated nonlinear warps, varied aspect/rotation, nib width and
    # pressure. All distributions here are explicit priors on pen movements.
    c = np.clip(rng.normal(size=(2,6)), -2.3,2.3)
    basis = np.stack([u*v, u*u, v*v, np.cos(np.pi*v), np.sin(np.pi*v), np.sin(np.pi*u)])
    p = p + (c @ (basis*np.array([.42,.42,.60,.45,.38,.34])[:,None])).T
    p -= 5
    horizontal = .73*math.exp(np.clip(rng.normal(0,.26),-.62,.58))
    p[:,0] *= horizontal
    rot = rng.normal(0,.16)
    px = math.cos(rot)*p[:,0] - math.sin(rot)*p[:,1]
    py = math.sin(rot)*p[:,0] + math.cos(rot)*p[:,1]
    shear = np.clip(rng.normal(-.12,.28),-.82,.62)
    px += shear*py
    radius=np.clip(math.exp(rng.normal(math.log(.62),.38)),.22,1.45)
    radii=[]
    for le in lengths:
        z = np.linspace(0,1,le)
        angle = rng.uniform(0,2*np.pi,size=2)
        envelope = 1+rng.uniform(0,.19)*np.sin(2*np.pi*z+angle[0])+rng.uniform(0,.13)*np.sin(np.pi*z+angle[1])
        taper = rng.uniform(.02,.48)*(np.exp(-z/.065)+np.exp(-(1-z)/.08))
        radii.extend(radius*(envelope-taper))
    r=np.array(radii)
    rx=np.clip(rng.normal(1,.13),.68,1.35)
    lo,hi=(px-r*rx).min(),(px+r*rx).max()
    top,bot=(py-r).min(),(py+r).max()
    scale=19.8/max(hi-lo,bot-top)
    # Center by bounds here: canonical() will remove residual translation.
    px=(px-(lo+hi)/2)*scale+13.5+rng.uniform(-.65,.65)
    py=(py-(top+bot)/2)*scale+13.5+rng.uniform(-.65,.65)
    r*=scale
    # Paint many overlapping antialiased discs using small vectorized stencils.
    ix=np.floor(px).astype(np.int32)
    iy=np.floor(py).astype(np.int32)
    k=int(np.ceil(r.max()*max(1,rx)+.8))
    off=np.arange(-k,k+1,dtype=np.int32)
    xx=ix[:,None,None]+off[None,None,:]
    yy=iy[:,None,None]+off[None,:,None]
    d=np.sqrt(((xx-px[:,None,None])/rx)**2+(yy-py[:,None,None])**2)
    edge=rng.uniform(.65,1.45)
    val=np.clip(.5+(r[:,None,None]-d)/edge,0,1).astype(np.float32)
    image=np.zeros(28*28,dtype=np.float32)
    np.maximum.at(image,(28*yy.clip(0,27)+xx.clip(0,27)).ravel(),val.ravel())
    return image.reshape(28,28)

matcher = r'''
bank = CONSTANTS['f'].astype(np.float32)*(1/127)
images = CONSTANTS['ink']
auxiliary = CONSTANTS['aux']
count = bank.shape[0]//10

def localdist(q, t):
    # Compare 4x4 blocks, allowing neighboring blocks to move by up to two
    # pixels. Compute the symmetric distance; a short displacement has a cost.
    pad = np.pad(q, ((2,2),(2,2)))
    tp = np.pad(t, ((0,0),(2,2),(2,2)))
    norm = np.maximum(1, (q*q).sum()+(t*t).sum(axis=(1,2)))
    best1 = np.zeros((len(t),6,6),dtype=np.float32)+100
    best2 = best1.copy()
    for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1),(-2,0),(2,0),(0,-2),(0,2)]:
        z = t-pad[2+dy:26+dy,2+dx:26+dx]
        z *= z
        cost = z.reshape(-1,6,4,6,4).sum(axis=(2,4)) + .060*(dx*dx+dy*dy)
        best1 = np.minimum(cost,best1)
        z = q-tp[:,2+dy:26+dy,2+dx:26+dx]
        z *= z
        cost = z.reshape(-1,6,4,6,4).sum(axis=(2,4)) + .060*(dx*dx+dy*dy)
        best2 = np.minimum(cost,best2)
    return .5*(best1.sum(axis=(1,2))+best2.sum(axis=(1,2)))/norm

def classify(a):
    n = len(a)
    f, ink, aux = describe(a)
    # Fixed dimensions for the candidate search, including for singleton calls.
    fq = np.zeros((48,bank.shape[1]),dtype=np.float32)
    fq[:n] = np.rint(f*127)
    sim = (fq @ bank.T)[:n]/127
    # Quantization served only to accelerate search; correct the template norms.
    sim *= correction
    sim /= np.maximum(.01,np.sqrt((fq[:n]*fq[:n]).sum(axis=1,keepdims=True))/127)
    diff = np.minimum(1.4,np.abs(aux[:,:1]-auxiliary[None,:,0]))
    cost = 1-sim + .039*diff*diff
    for j in range(1,6):
        delta = aux[:,j:j+1]-auxiliary[None,:,j]
        cost += .014*delta*delta
    cost = cost.reshape(n,10,count)
    res = np.zeros(n,dtype=np.int32)
    for i in range(n):
        # Shortlist independently for each class, then rerank the strongest
        # hypotheses. Each cost remains a distance to one particular drawing.
        v = cost[i]
        ids = np.argpartition(v,7,axis=1)[:,:8]
        fixed = np.take_along_axis(v,ids,axis=1)
        cls = np.argsort(fixed.min(axis=1),kind='stable')[:4]
        idx = (ids[cls]+cls[:,None]*count).ravel()
        flexible = localdist(ink[i], images[idx].astype(np.float32)*(1/255)).reshape(len(cls),8)
        total = np.sort(.66*fixed[cls] + .53*flexible,axis=1)
        res[i] = cls[np.argmin(.66*total[:,0]+.23*total[:,1]+.11*total[:,2])]
    return res

correction = 1/np.sqrt(np.maximum(.0001,(bank*bank).sum(axis=1)))

def predict(x):
    result = np.zeros(len(x),dtype=np.int32)
    for start in range(0,len(x),48):
        result[start:start+48] = classify(x[start:start+48,0])
    return result
'''

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(features+matcher)
    scope={}
    exec(features,scope)
    desc=scope['describe']
    nclass=8192
    # Allocate separate descriptors; no across-drawing normalization, pruning,
    # averaging, or calibration. Even the feature quantizer is a fixed formula.
    n=nclass*10
    fs=np.empty((n,64+8*36+8*16),dtype=np.int8)
    inks=np.empty((n,24,24),dtype=np.uint8)
    auxs=np.empty((n,6),dtype=np.float32)
    for digit,rows in enumerate(styles):
        rng = np.random.default_rng(17011+307*digit)
        bag = [p for w,p in rows for j in range(w)]
        for start in range(0,nclass,64):
            stop=min(start+64,nclass)
            batch=np.stack([render(bag[j%len(bag)],rng,digit) for j in range(start,stop)])
            f,ink,aux=desc(batch)
            dest=slice(digit*nclass+start,digit*nclass+stop)
            fs[dest]=np.rint(f*127).clip(0,127).astype(np.int8)
            inks[dest]=np.rint(ink*255).clip(0,255).astype(np.uint8)
            auxs[dest]=np.rint(aux*4096)/4096
        print('Finished pen model',digit,flush=True)
    np.savez(out/'constants.npz', f=fs,ink=inks,aux=auxs)

if __name__=='__main__':
    main(Path(sys.argv[1]))
