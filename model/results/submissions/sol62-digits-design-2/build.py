"""Compile a hand-specified pen-shape matcher into the allowed network ops.

No input files. Coordinates below are imaginary pen trajectories on an
arbitrary drawing square; pixel weights are integrals of geometric kernels.
"""
import json
import sys
from pathlib import Path

import numpy as np


# M starts a stroke, L is straight, Q and C are quadratic and cubic Beziers.
# The positive y direction is down. Styles include both stroke-order choices
# and genuinely different ways to construct the same numeral.
GLYPHS = [
    [  # 0: oval, egg, rounded rectangle, pointed oval
        'M .03 -1 C .54 -1 .70 -.53 .67 .10 C .65 .67 .34 1 -.06 1 C -.58 1 -.71 .50 -.67 -.07 C -.64 -.59 -.38 -1 .03 -1',
        'M .19 -1 C .58 -1 .75 -.49 .64 .15 C .54 .76 .19 1 -.18 1 C -.66 .99 -.70 .52 -.62 -.05 C -.54 -.64 -.13 -1 .19 -1',
        'M -.08 -1 C .39 -1 .60 -.89 .65 -.48 C .74 .07 .68 .66 .44 .88 C .21 1.09 -.30 1.02 -.51 .87 C -.78 .66 -.74 .05 -.69 -.40 C -.65 -.77 -.49 -1 -.08 -1',
        'M .06 -1 C .43 -.92 .63 -.53 .61 .05 C .59 .56 .30 .88 -.03 1 C -.42 .88 -.62 .50 -.61 .04 C -.59 -.50 -.31 -.93 .06 -1',
        'M .12 -.96 C -.33 -1 -.62 -.62 -.65 -.04 C -.69 .56 -.45 .98 -.06 1 C .33 1 .64 .60 .66 .02 C .68 -.52 .49 -.91 .14 -1',
        'M .05 -1 C .39 -1 .54 -.59 .54 -.06 C .54 .54 .38 1 .01 1 C -.37 1 -.53 .55 -.53 -.07 C -.53 -.64 -.32 -1 .05 -1',
    ],
    [  # 1: single stem, entry stroke, foot, and variations on the serif
        'M .17 -1 L -.12 1',
        'M .11 -1 C -.06 -.49 -.17 .49 -.13 1',
        'M -.08 -1 Q .16 -.03 .08 1',
        'M -.36 -.47 Q -.02 -.62 .20 -1 Q .10 -.19 .02 1',
        'M -.59 -.41 L .22 -1 L .08 1',
        'M -.28 -.77 Q -.02 -.82 .11 -1 L -.01 1',
        'M -.48 -.50 Q -.07 -.69 .20 -1 Q .09 -.17 .08 .96 M -.57 .99 Q .02 .92 .60 .97',
        'M -.61 .99 L .16 .96 L .14 -1 L -.58 -.54 M .16 .96 L .65 .94',
    ],
    [  # 2: swooping, angular, rounded-corner and looped bases
        'M -.61 -.62 C -.51 -.89 -.24 -1 .05 -1 C .48 -1 .66 -.75 .61 -.42 C .55 -.02 -.16 .49 -.64 .97 Q -.02 .89 .68 .96',
        'M -.63 -.65 C -.43 -1.00 .33 -1.11 .58 -.81 C .93 -.32 -.21 .53 -.62 .89 Q -.32 1.08 .67 .90',
        'M -.63 -.89 Q .13 -1.09 .60 -.89 Q .65 -.64 .43 -.32 L -.65 .91 L .69 .91',
        'M -.60 -.50 C -.58 -.91 -.31 -1.06 .06 -1 C .45 -.99 .65 -.70 .60 -.37 C .52 .13 -.30 .53 -.53 .84 C -.68 1.05 .01 .97 .24 .93 L .68 .80',
        'M -.67 -.64 C -.42 -1.01 .28 -1.12 .56 -.83 C .96 -.38 -.10 .45 -.55 .68 C -.89 .87 -.46 1.15 -.23 .92 C .04 .63 -.32 .53 -.51 .68 Q .10 .88 .67 .97',
        'M -.67 -.39 C -.75 -.72 -.33 -1.03 .07 -1 C .42 -.98 .62 -.80 .62 -.55 C .62 -.13 .03 .27 -.37 .71 L -.60 1 Q -.03 .77 .68 .81',
        'M -.55 -.75 C -.35 -1.04 .43 -1.02 .57 -.74 C .76 -.31 .04 .32 -.45 .79 Q -.65 1.01 -.32 .99 Q .29 .96 .65 .76',
        'M -.55 -.88 C .04 -1.07 .71 -.98 .62 -.61 Q .57 -.26 .06 .24 L -.59 .85 L .68 .99',
        'M -.53 -.49 C -.63 -.74 -.20 -1.07 .13 -.98 C .73 -.94 .66 -.51 .40 -.20 C .12 .18 -.57 .54 -.57 .83 C -.56 1.15 .05 .77 -.14 .53 Q .08 .60 .68 .96',
        'M -.54 -.72 C -.23 -1.08 .41 -1.07 .53 -.77 C .72 -.22 -.41 .57 -.56 .87 Q -.22 .94 .15 .87 Q .49 .82 .66 .96',
        'M -.55 -.55 C -.50 -.99 -.08 -1.04 .23 -.97 C .71 -.86 .55 -.30 .24 .06 Q -.04 .45 -.27 .75 Q .11 .91 .59 .96',
        'M -.51 -.87 C -.15 -1.04 .36 -1.03 .54 -.77 Q .78 -.38 .15 .18 Q -.15 .49 -.65 .77 L .50 .80 Q .61 .85 .58 1.00',
    ],
    [  # 3: two bows, flat-top and pointed-waist variants
        'M -.57 -.78 C -.33 -1.03 .35 -1.09 .54 -.81 C .82 -.39 .29 -.10 -.11 -.05 C .41 -.09 .75 .15 .66 .56 C .58 .97 -.09 1.17 -.60 .79',
        'M -.60 -.91 Q -.02 -1.08 .63 -.92 L -.09 -.18 C .42 -.27 .71 .02 .65 .46 C .60 .96 -.17 1.15 -.62 .81',
        'M -.57 -.54 C -.51 -.90 -.22 -1.03 .11 -.98 C .64 -.97 .78 -.24 .18 -.03 L -.05 .02 C .51 -.03 .82 .24 .64 .66 C .47 1.03 -.20 1.09 -.60 .68',
        'M -.60 -.85 C -.30 -1.02 .45 -1.14 .60 -.77 C .74 -.45 .33 -.10 -.20 -.08 C .34 -.07 .76 .16 .67 .49 C .58 .88 .10 1.05 -.29 1.00 Q -.56 .96 -.67 .83',
        'M -.58 -.70 C -.13 -1.13 .51 -1.07 .62 -.70 C .72 -.35 .24 -.15 -.13 -.13 C .50 -.14 .81 .26 .62 .72 C .46 1.04 -.32 1.13 -.54 .70',
        'M -.51 -.94 C -.20 -1.07 .31 -.98 .45 -.74 C .59 -.51 .37 -.27 -.01 -.13 C .48 -.03 .69 .15 .68 .46 C .62 .90 .08 1.14 -.56 .85',
        'M -.61 -.63 C -.41 -.92 .09 -1.07 .45 -.95 C .73 -.87 .65 -.51 .46 -.30 Q .24 -.05 -.18 .04 C .25 -.02 .68 .07 .67 .41 C .67 .84 .16 1.12 -.68 .92',
        'M -.33 -.91 Q .12 -1.06 .49 -.87 Q .62 -.77 .14 -.13 C .43 -.03 .72 .22 .62 .60 C .52 .98 .05 1.14 -.43 .83',
        'M -.68 -.66 C -.36 -1.04 .37 -1.10 .58 -.83 C .77 -.56 .58 -.25 .21 -.15 C .08 -.13 -.03 -.19 .08 -.26 C .38 -.38 .84 .17 .61 .62 C .37 1.09 -.24 1.09 -.56 .76',
    ],
    [  # 4: open forks and closed triangles
        'M -.21 -1 L -.66 .19 L .71 .18 M .34 -.85 L .27 1',
        'M -.40 -1 Q -.62 -.47 -.65 -.17 Q -.53 .10 .70 .13 M .43 -.69 L .25 1',
        'M .34 -1 L -.72 .23 L .70 .22 M .34 -1 L .27 1',
        'M .34 -1 Q -.30 -.34 -.67 .23 Q -.04 .25 .72 .12 M .34 -1 Q .28 .07 .35 1',
        'M -.46 -.88 Q -.54 -.32 -.63 .13 L .45 .12 M .42 -1 Q .30 -.13 .32 1',
        'M -.14 -1 L -.61 .27 L .69 .14 M .16 -.73 L .11 1',
        'M .25 -1 C -.02 -.71 -.58 -.22 -.61 .08 Q -.40 .29 .61 .13 M .25 -1 Q .36 .02 .25 1',
        'M -.32 -1 L -.71 .07 Q -.41 .23 .62 .25 M .51 -.80 Q .24 .00 .33 1',
    ],
    [  # 5: various top bars, elbows and bottom bowls
        'M .62 -.98 L -.43 -.96 L -.59 -.15 C -.10 -.38 .52 -.30 .61 .23 C .74 .75 .13 1.19 -.57 .82',
        'M .63 -.95 Q .23 -1.05 -.45 -.99 Q -.52 -.58 -.56 -.15 C -.10 -.45 .59 -.22 .66 .21 C .76 .72 .15 1.15 -.52 .87',
        'M -.47 -.93 L -.63 -.14 C -.23 -.42 .55 -.37 .64 .20 C .70 .72 .12 1.15 -.59 .82 M -.41 -.95 Q .10 -.79 .66 -1',
        'M .61 -.97 L -.52 -.89 L -.57 -.24 C -.04 -.38 .69 -.02 .65 .39 C .61 .89 .06 1.16 -.52 .82',
        'M .59 -.96 Q .13 -.85 -.42 -.97 L -.58 -.22 C -.07 -.23 .47 -.30 .61 .12 C .81 .60 .09 1.13 -.55 .97',
        'M .65 -.98 L -.47 -.96 Q -.47 -.60 -.59 -.21 C -.12 -.36 .55 -.30 .66 .12 C .75 .56 .36 .94 .00 .99 Q -.35 1.02 -.55 .70',
        'M .56 -.94 Q .09 -.99 -.43 -.99 Q -.54 -.65 -.58 -.29 C -.18 -.46 .38 -.35 .57 -.02 C .83 .43 .42 .89 .09 .99 Q -.27 1.06 -.58 .86',
        'M .62 -.99 Q .15 -.98 -.42 -.80 Q -.60 -.54 -.62 -.22 C -.20 -.45 .53 -.27 .56 .19 C .63 .70 .19 1.10 -.54 .82',
        'M .70 -.97 L -.31 -.89 Q -.62 -.68 -.55 -.32 C -.29 -.41 .26 -.42 .54 -.08 C .77 .23 .62 .62 .36 .78 Q .06 1.00 -.59 .97',
        'M -.40 -.90 Q -.56 -.49 -.54 -.10 C -.08 -.29 .44 -.19 .53 .17 C .63 .65 .15 1.05 -.46 .89 M -.35 -.91 Q .16 -.98 .67 -.98',
        'M .64 -.97 Q -.12 -.82 -.54 -.86 Q -.68 -.52 -.61 -.31 C -.04 -.42 .39 -.34 .53 .04 C .74 .64 .17 .99 -.39 .88 Q -.62 .74 -.48 .49',
    ],
    [  # 6: hooked, upright, diagonal and backtracking downstrokes
        'M .45 -.98 C -.16 -.91 -.65 -.33 -.66 .30 C -.69 .81 -.40 1.02 .02 1 C .43 .99 .67 .68 .64 .35 C .59 -.12 -.06 -.24 -.57 .14',
        'M .30 -1 C -.25 -.80 -.64 -.23 -.62 .40 C -.62 .83 -.38 1.01 .03 .99 C .52 .97 .69 .69 .63 .34 C .56 -.13 -.09 -.17 -.57 .15',
        'M .09 -1 C -.39 -.83 -.67 -.28 -.66 .36 C -.65 .83 -.34 1.03 .07 .99 C .54 .94 .68 .54 .57 .26 C .44 -.09 -.26 -.11 -.65 .30',
        'M .50 -.98 Q -.31 -.65 -.62 .27 C -.76 .66 -.54 1.00 -.05 1.00 C .37 1.00 .68 .73 .65 .38 C .62 -.04 .01 -.15 -.56 .10',
        'M -.03 -.99 Q -.58 -.50 -.56 .36 C -.56 .78 -.37 1.01 .06 1.00 C .45 .99 .66 .72 .63 .37 C .60 .04 .26 -.11 -.04 -.04 Q -.39 .04 -.56 .34',
        'M .55 -.82 C .27 -1.15 -.31 -.99 -.55 -.41 C -.74 .07 -.72 .66 -.36 .90 C -.05 1.11 .45 1.03 .59 .67 C .81 .17 .42 -.11 .10 -.14 Q -.26 -.19 -.58 .09',
        'M .43 -.98 L -.33 -.24 C -.54 -.01 -.67 .43 -.59 .69 C -.50 .99 -.14 1.10 .22 .94 C .54 .81 .68 .57 .62 .35 Q .51 -.03 .00 .02 Q -.42 .06 -.60 .31',
        'M .38 -.98 C -.10 -.82 -.55 -.45 -.63 .04 C -.76 .62 -.47 .97 -.06 1 C .44 1.04 .74 .62 .58 .25 C .45 -.07 .07 -.10 -.25 -.01',
    ],
    [  # 7: caps, bent descenders, optional midbar and entry hook
        'M -.69 -.89 Q -.13 -.99 .68 -.95 L -.19 1',
        'M -.68 -.92 L .64 -.94 Q .03 -.10 -.30 1',
        'M -.65 -.91 Q .21 -1.04 .66 -.86 Q .05 -.31 -.02 1',
        'M -.69 -.93 L .69 -.93 Q .12 -.20 -.31 1 M -.55 .09 L .53 .14',
        'M -.64 -.44 L -.63 -.91 Q -.08 -1.01 .66 -.91 Q .07 -.24 -.35 1',
        'M -.68 -.78 Q -.20 -1.07 .67 -.96 L -.28 1 M -.57 .15 L .45 .04',
        'M -.72 -.93 L .66 -.89 Q .38 -.61 .20 -.15 L -.28 1',
        'M -.68 -.89 Q -.13 -.85 .68 -.99 Q .04 -.25 -.38 1',
        'M -.55 -.86 Q -.14 -.88 .47 -.97 Q .22 -.37 -.24 1',
        'M -.54 -.68 L -.53 -.91 L .55 -.96 Q .29 -.51 .26 -.09 Q .28 .47 .11 1',
        'M -.40 -.89 Q -.04 -.96 .41 -.98 L -.49 1',
    ],
    [  # 8: one-crossing and stacked-loop constructions
        'M .03 -.10 C -.42 -.30 -.60 -.54 -.43 -.82 C -.26 -1.09 .38 -1.09 .52 -.76 C .70 -.40 .16 -.10 -.23 .12 C -.73 .40 -.82 .68 -.50 .90 C -.19 1.13 .51 1.00 .64 .70 C .81 .31 .43 .06 .03 -.10',
        'M -.06 -.06 C -.53 -.26 -.70 -.62 -.45 -.85 C -.21 -1.13 .49 -1.02 .57 -.67 C .65 -.35 .23 -.13 -.09 .05 C -.53 .30 -.73 .64 -.45 .90 C -.16 1.14 .49 1.02 .63 .69 C .78 .36 .44 .11 -.06 -.06',
        'M .13 -.06 C -.54 -.11 -.70 -.50 -.52 -.78 C -.35 -1.10 .33 -1.08 .50 -.77 C .68 -.40 .41 -.15 .13 -.06 C -.44 .04 -.72 .27 -.68 .60 C -.62 1 .04 1.10 .46 .92 C .87 .74 .72 .08 .13 -.06',
        'M .29 -.98 C -.28 -1.02 -.57 -.72 -.49 -.41 C -.38 -.02 .57 .06 .66 .52 C .77 .98 .08 1.09 -.35 .94 C -.81 .76 -.72 .40 -.43 .15 C -.08 -.14 .89 -.37 .54 -.81 Q .45 -.96 .29 -.98',
        'M -.07 -.08 C .26 -.24 .64 -.47 .56 -.75 C .44 -1.07 -.13 -1.10 -.44 -.89 C -.83 -.62 -.49 -.24 -.07 -.08 C .38 .09 .78 .37 .63 .70 C .46 1.08 -.25 1.08 -.55 .84 C -.85 .56 -.65 .19 -.07 -.08',
        'M .02 -.17 C -.35 -.21 -.50 -.37 -.46 -.65 C -.39 -.94 .03 -1.12 .33 -.94 C .73 -.69 .48 -.33 .02 -.17 C -.40 -.03 -.73 .26 -.68 .59 C -.63 1.00 -.04 1.14 .40 .90 C .85 .64 .75 .22 .02 -.17',
        'M -.01 .08 C -.31 -.09 -.64 -.40 -.55 -.67 C -.46 -.99 .04 -1.12 .40 -.93 C .80 -.70 .63 -.22 .22 .04 C -.23 .28 -.69 .39 -.66 .70 C -.62 1.00 -.20 1.04 .14 .97 C .63 .88 .77 .48 .51 .29 Q .30 .14 -.01 .08',
        'M .10 -.08 C -.30 -.25 -.56 -.44 -.53 -.69 C -.49 -.97 -.11 -1.06 .21 -.98 C .76 -.86 .68 -.45 .34 -.22 C -.09 .06 -.67 .24 -.66 .66 C -.64 .98 -.24 1.08 .17 .96 C .70 .82 .83 .34 .47 .13 L .10 -.08',
        'M .57 -.95 C .08 -1.11 -.62 -.97 -.53 -.68 C -.47 -.45 .02 -.24 .39 -.04 C .92 .27 .75 .85 .25 .97 C -.20 1.08 -.68 .87 -.60 .54 C -.54 .24 -.03 .04 .30 -.29 Q .57 -.56 .57 -.95',
        'M .31 -.99 C -.22 -1.02 -.53 -.79 -.43 -.55 C -.30 -.24 .45 -.17 .59 .21 C .77 .72 .25 1.11 -.22 1 C -.64 .91 -.70 .60 -.59 .31 C -.42 -.02 .24 -.16 .42 -.46 C .58 -.74 .38 -.92 .31 -.99',
        'M .38 -.96 C -.07 -1.10 -.47 -.86 -.40 -.52 C -.30 -.14 .27 -.23 .39 .10 C .61 .66 .17 1.11 -.24 .98 C -.68 .85 -.64 .46 -.40 .23 C -.19 .02 .24 -.20 .39 -.51 Q .52 -.79 .38 -.96',
    ],
    [  # 9: hanging tail, straight stem, curl and open head
        'M .57 -.35 C .55 -.80 .32 -1.00 -.07 -1.00 C -.51 -1.00 -.69 -.70 -.63 -.35 C -.54 .09 .15 .14 .57 -.23 C .64 .15 .42 .67 -.03 1',
        'M .51 -.30 C .58 -.66 .41 -1.00 .03 -1.00 C -.38 -1.00 -.68 -.73 -.65 -.43 C -.63 -.01 .04 .18 .51 -.22 Q .54 .46 .34 1',
        'M .54 -.31 C .60 -.75 .33 -1.02 -.05 -1 C -.46 -.98 -.65 -.72 -.65 -.39 C -.62 .00 -.06 .13 .52 -.23 C .52 .43 .31 .82 -.24 1',
        'M .53 -.37 C .62 -.73 .34 -1.06 -.07 -.99 C -.43 -.95 -.67 -.66 -.64 -.34 C -.59 -.02 .11 .04 .53 -.25 C .55 .50 .23 .96 -.36 .98',
        'M .44 -.90 C -.08 -1.13 -.61 -.93 -.65 -.53 C -.71 -.05 -.08 .16 .43 -.15 M .51 -1 Q .50 .04 .25 1',
        'M .59 -.42 C .43 -.99 -.18 -1.09 -.49 -.83 C -.81 -.54 -.60 -.13 -.25 -.09 Q .21 -.03 .55 -.37 Q .39 .49 .00 1',
        'M .51 -.53 C .45 -.92 .08 -1.10 -.26 -.94 C -.62 -.77 -.67 -.42 -.50 -.19 C -.30 .09 .24 .02 .53 -.30 Q .59 .27 .07 .98',
        'M .55 -.38 C .59 -.83 .23 -1.09 -.23 -.96 C -.56 -.87 -.74 -.57 -.59 -.29 Q -.36 .00 .06 -.11 Q .38 -.20 .55 -.38 L .24 1',
    ],
]


def curves(spec):
    """Sample elementary line segments for analytic distance calculations."""
    tokens = iter(spec.split())
    last = None
    a, b = [], []
    def point():
        return np.array([float(next(tokens)), float(next(tokens))])
    for cmd in tokens:
        if cmd == 'M':
            last = point()
            continue
        p = [last] + [point() for _ in range({'L': 1, 'Q': 2, 'C': 3}[cmd])]
        last = p[-1]
        n = max(2, int(np.ceil(sum(np.linalg.norm(v-u) for u,v in zip(p,p[1:])) * 8)))
        t = np.linspace(0., 1., n+1)[:,None]
        while len(p) > 1:
            p = [(1-t)*u+t*v for u,v in zip(p,p[1:])]
        a.extend(p[0][:-1]); b.extend(p[0][1:])
    return np.array([a, b], np.float32)


def sample(n, seed):
    """Scrambled stratified coordinates; zero is the unwarped midrange."""
    rng = np.random.default_rng(seed)
    # Limit the tails without excluding wide or bent handwriting entirely.
    u = np.array([(rng.permutation(n)+rng.random(n))/n for _ in range(19)]).T
    q = 2*u-1
    q = .76*q + .24*q**3
    return q.astype(np.float32)


def transform(ab, q):
    q = q.T[:, :, None, None]
    x, y = ab.transpose(3,0,1,2)
    xx = np.exp(.54*q[0])*x + .58*q[1]*y + .35*q[2]*(y*y-.34)
    xx += .43*q[3]*x*y + .23*q[4]*x*(1-y*y) + .20*q[5]*np.sin(np.pi*y)
    yy = y + .29*q[6]*x + .25*q[7]*(1-y*y) + .23*q[8]*x*y
    yy += .21*q[9]*(x*x-.2) + .11*q[10]*np.sin(np.pi*y)
    xx += .15*q[11]*np.sin(np.pi*(x+.43*y))
    yy += .15*q[12]*np.sin(np.pi*(x-.32*y))
    p = np.stack([xx,yy], axis=-1)
    radius = 1.42 + .72*q[13].ravel()
    bbox = np.ptp(p, axis=(1,2)).max(axis=1)
    p *= ((20.-1.8*radius+1.45*q[14].ravel())/bbox)[:,None,None,None]
    l = np.sqrt(np.sum((p[:,1]-p[:,0])**2, axis=-1))
    mids = p.mean(axis=1)
    width = np.clip(1 + .35*q[15,:,0,:]*mids[:,:,1]/9 + .24*q[16,:,0,:]*mids[:,:,0]/7, .58, 1.45)
    centroid = (mids*(l*width)[:,:,None]).sum(axis=1)/(l*width).sum(axis=1)[:,None]
    p += (13.90-centroid + .79*q[17:19,:,0,0].T)[:,None,None,:]
    return p[:,0], p[:,1], l, radius[:,None]*width


def xy_grid(size, step):
    z = np.arange(size, dtype=np.float32)*step + (step-1)/2
    return np.stack(np.meshgrid(z,z), axis=-1).reshape(-1,2)


# Every ray is a narrow sector of a half-plane, stopping at the digit's
# maximum feasible span. A separate opposite-ray response detects a bay.
RAYS = []
for dx,dy in ((1,0), (0,1), (-1,0), (0,-1)):
    RAYS.append([(dy*r+dx*t, dx*r-dy*t) for r in range(1,8)
                 for t in range(-1,2) if t == 0 or r >= 3])


def enclosure(ink):
    """Compute the same cavity / opposing-wall maps as the graph branch."""
    p = np.pad(ink.reshape(-1,14,14),((0,0),(7,7),(7,7)))
    rays = [np.maximum.reduce([p[:,7+y:21+y,7+x:21+x] for y,x in r]) for r in RAYS]
    z = p[:,7:21,7:21]
    walls = np.maximum(np.stack([np.minimum(rays[0],rays[2])-z,
                                 np.minimum(rays[1],rays[3])-z],axis=1),0)
    hole = np.minimum(walls[:,0],walls[:,1])
    return walls.reshape(len(ink),2,7,2,7,2).mean(axis=(3,5)).reshape(len(ink),98), hole.reshape(len(ink),7,2,7,2).mean(axis=(2,4)).reshape(len(ink),49)


# Fixed Gaussian used for a broad, contrast-normalized similarity term.
xy = np.arange(-2,3,dtype=np.float32)
SMOOTH = np.exp(-.5*(xy[:,None]**2+xy[None,:]**2)/.77**2)
SMOOTH /= SMOOTH.sum()


def blur(ink):
    a = np.pad(ink.reshape(-1,14,14),((0,0),(2,2),(2,2)))
    z = sum(SMOOTH[y,x]*a[:,y:y+14,x:x+14] for y in range(5) for x in range(5))
    return z.reshape(-1,196)


def filters(spec, n, seed):
    """Compute matched-filter coefficients from continuous stroke geometry."""
    path = curves(spec)[None]
    grid = xy_grid(14,2)[None,:,None,:]
    raster = np.empty((n,196), np.float32)
    proximity = np.empty_like(raster)
    tangent = np.empty((n,196), np.float32)
    cavities = np.empty((n,147), np.float32)
    correlations = np.empty((n,196), np.float32)
    bias = np.empty(n, np.float32)
    samples = sample(n,seed)
    for i in range(0,n,48):
        j = slice(i,min(n,i+48))
        a,b,length,radii = transform(path,samples[j])
        v = b-a
        d = grid-a[:,None,:,:]
        t = np.clip((d*v[:,None]).sum(-1)/np.maximum(length[:,None]**2,1e-8),0,1)
        dsq = np.sum((d-t[:,:,:,None]*v[:,None])**2,axis=-1)
        ix = np.argmin(dsq,axis=2)
        dist = np.sqrt(np.take_along_axis(dsq,ix[:,:,None],axis=2)[:,:,0])
        r = np.take_along_axis(radii,ix,axis=1)
        # Approximate the area of each 2px cell covered by a round pen.
        ink = np.clip(.5+(r-dist)/1.9,0,1)
        outside = np.maximum(dist-.9,0)
        # Input-to-model part of a symmetric tolerant distance.
        penalty = np.minimum((outside/5)**1.45,1)
        # Model-to-input part: integrate near the pen centerline, giving
        # each glyph (not each pixel) equal total weight.
        core = np.exp(-.5*(dist/1.07)**2)
        core /= core.sum(axis=1)[:,None]
        # Keep both unnormalized so a short sub-stroke does not get a free
        # pass for covering only a small fraction of a complicated digit.
        area = ink.sum(axis=1)
        size = (area/30)**.62
        raster[j] = -penalty/34 + (.34*ink)/34
        proximity[j] = .65*size[:,None]*core
        bias[j] = -.65*size - .17*(ink*ink).sum(axis=1)/34
        smooth = blur(ink)
        smooth -= smooth.mean(axis=1,keepdims=True)
        correlations[j] = (2.50/np.sqrt(196))*smooth/np.maximum(1e-5,np.linalg.norm(smooth,axis=1,keepdims=True))
        # Tangent-field correlation at lower spatial resolution. Using
        # doubled angles makes this independent of stroke direction.
        unit = v/np.maximum(length[:,:,None],1e-6)
        uv = np.take_along_axis(unit,ix[:,:,None],axis=1)
        z = np.stack([uv[:,:,0]**2-uv[:,:,1]**2,2*uv[:,:,0]*uv[:,:,1]],axis=1)
        z = (z*core[:,None,:]).reshape(-1,2,7,2,7,2).sum(axis=(3,5))
        tangent[j] = (1.40*np.concatenate([z,-z],axis=1)).reshape(-1,196)
        walls,holes = enclosure(ink)
        # A squared feature-distance, omitting the observed-feature norm
        # (constant across hypotheses). Pooling tolerates slight shifts.
        f = np.concatenate([walls,holes],axis=1)
        gain = np.r_[np.full(98,.055),np.full(49,.16)]
        cavities[j] = f*gain
        bias[j] -= .5*(f*f*gain).sum(axis=1)
    return raster, proximity, tangent, cavities, correlations, bias


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def op(self, name, op, x, **kw):
        self.layers.append(dict(name=name, op=op, inputs=[x] if isinstance(x,str) else x, **kw))
        return name

    def affine(self, name, x, w, b, stride=1, padding=0):
        self.weights[name+'.W'] = np.asarray(w,np.float32)
        self.weights[name+'.b'] = np.asarray(b,np.float32)
        if w.ndim == 2:
            return self.op(name,'dense',x,out=len(w))
        return self.op(name,'conv',x,out=len(w),kernel=list(w.shape[-2:]),stride=stride,padding=padding)

    def pool(self, name, x, size, stride, mode='max'):
        return self.op(name,mode+'pool',x,size=size,stride=stride)


def cavity_branch(g, ink):
    # Parallel sheared rays, with channels and rows rearranged so that
    # one ordinary max pool implements the union of each ray's offsets.
    offsets = [off for ray in RAYS for off in ray]
    shifts = np.zeros((len(offsets),1,15,15),np.float32)
    for i,(y,x) in enumerate(offsets):
        shifts[i,0,y+7,x+7] = 1
    nr = len(RAYS[0])
    x = g.affine('ray_shift',ink,shifts,np.zeros(len(shifts)),padding=7)
    x = g.op('ray_offsets','reshape',x,shape=[1,len(offsets),196])
    x = g.pool('ray_max',x,[nr,1],[nr,1])
    x = g.op('four_sides','reshape',x,shape=[4,14,14])
    # min(left,right) = left - relu(left-right), and likewise for up/down.
    signs = np.array([[1,0,-1,0],[0,1,0,-1],[1,0,0,0],[0,1,0,0]],np.float32)
    x = g.affine('opposite_ray_differences',x,signs[:,:,None,None],np.zeros(4))
    x = g.op('positive_ray_differences','relu',x)
    signs = np.array([[-1,0,1,0],[0,-1,0,1],[0,0,0,0]],np.float32)
    x = g.affine('opposite_ray_minima',x,signs[:,:,None,None],np.zeros(3))
    blank = g.affine('exclude_centers',ink,np.array([-1,-1,0]).reshape(3,1,1,1),np.zeros(3))
    x = g.op('bay_raw','add',[x,blank])
    x = g.op('bays','relu',x)
    signs = np.array([[1,0,0],[0,1,0],[1,-1,0]],np.float32)
    x = g.affine('bay_difference',x,signs[:,:,None,None],np.zeros(3))
    x = g.op('positive_bay_difference','relu',x)
    signs = np.array([[1,0,0],[0,1,0],[1,0,-1]],np.float32)
    x = g.affine('bays_and_cavities',x,signs[:,:,None,None],np.zeros(3))
    return g.pool('cavity_cells',x,[2,2],[2,2],'avg')


def build(output):
    g = Network()
    ink = g.pool('mean_ink','image',[2,2],[2,2],'avg')
    topo = cavity_branch(g,ink)
    smooth = g.affine('blurred_ink',ink,SMOOTH.reshape(1,1,5,5),np.zeros(1),padding=2)
    smooth = g.op('contrast_normalized','norm',smooth)
    peaks = g.pool('ink_peaks','image',[2,2],[2,2])
    # Local distance-to-ink as the upper envelope of translated ink maps
    # minus the geometric price of moving each offset.
    off = [(dy,dx) for dy in range(-2,3) for dx in range(-2,3)]
    shifts = np.zeros((len(off),1,5,5),np.float32)
    costs = np.array([-.42*np.hypot(y,x) for y,x in off],np.float32)
    for k,(y,x) in enumerate(off):
        shifts[k,0,y+2,x+2] = 1
    d = g.affine('spread_ink',peaks,shifts,costs,padding=2)
    d = g.op('offsets','reshape',d,shape=[1,len(off),196])
    d = g.pool('near_ink',d,[len(off),1],[len(off),1])
    # Positive responses of a difference of two anisotropic Gaussians.
    v = np.arange(-4,5)-.5
    x,y = np.meshgrid(v,v)
    oriented = []
    for a in np.arange(4)*np.pi/4:
        along, across = x*np.cos(a)+y*np.sin(a), y*np.cos(a)-x*np.sin(a)
        def kernel(sigma):
            z = np.exp(-.5*((along/1.72)**2+(across/sigma)**2))
            return z/z.sum()
        oriented.append(kernel(.59)-kernel(1.65))
    o = g.affine('oriented_ridges','image',np.array(oriented)[:,None],np.zeros(4),stride=2,padding=4)
    o = g.op('bright_ridges','relu',o)
    o = g.pool('ridge_cells',o,[2,2],[2,2],'avg')

    per_class = 12800
    blocks = 4
    width = per_class//blocks
    count = 10*per_class
    inks, distances, directions, correlations = [np.empty((count,196),np.float32) for _ in range(4)]
    cavities = np.empty((count,147),np.float32)
    biases = np.empty(count,np.float32)
    for bank in range(blocks):
        for label,styles in enumerate(GLYPHS):
            for k,s in enumerate(styles):
                lo,hi = (k*width)//len(styles), ((k+1)*width)//len(styles)
                base = (label*blocks+bank)*width
                rows = slice(base+lo,base+hi)
                inks[rows], distances[rows], directions[rows], cavities[rows], correlations[rows], biases[rows] = filters(s,hi-lo,72113+1049*label+537*k+9277*bank)
    a = g.affine('ink_fit',ink,inks,biases)
    b = g.affine('stroke_coverage',d,distances,np.zeros(count))
    c = g.affine('tangent_fit',o,directions,np.zeros(count))
    t = g.affine('enclosure_fit',topo,cavities,np.zeros(count))
    h = g.affine('broad_shape_fit',smooth,correlations,np.zeros(count))
    s = g.op('shape_match','add',[a,b,c,t,h])
    # The maximum is a profile likelihood. Mix in a small amount of local
    # support, represented by four independently sampled banks per class.
    s = g.op('shape_families','reshape',s,shape=[1,10*blocks,width])
    s = g.pool('best_in_banks',s,[1,width],[1,width])
    s = g.op('group_bank_scores','reshape',s,shape=[1,10,blocks])
    m = g.pool('best_in_family',s,[1,blocks],[1,blocks])
    v = g.pool('bank_consensus',s,[1,blocks],[1,blocks],'avg')
    m = g.affine('strongest_match',m,np.eye(10)*.68,np.zeros(10))
    v = g.affine('consistent_match',v,np.eye(10)*.32,np.zeros(10))
    g.op('scores','add',[m,v])

    output.mkdir(parents=True,exist_ok=True)
    (output/'network.json').write_text(json.dumps({'layers':g.layers},indent=2)+'\n')
    np.savez(output/'weights.npz',**g.weights)


if __name__ == '__main__':
    build(Path(sys.argv[1]))
