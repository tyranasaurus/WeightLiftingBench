"""Compile an analytic pen-and-paper model of handwriting into a neural net.
No fonts, data, fitted parameters or input files are used.
"""
import json
import os
import sys
import numpy as np

# Control points in a unit writing box. Each row is one way to move the pen;
# repeated M commands lift it. Q and C are quadratic and cubic Beziers.
# The first two numbers are relative frequency and nominal width/height.
PENS = [
 [
 (5,.69,'M .48 .015 C .22 -.04 -.02 .17 .015 .54 C .015 .84 .20 1 .47 1 C .80 1.01 1.01 .78 1 .45 C .99 .17 .80 -.02 .48 .015'),
 (3,.72,'M .68 .065 C .45 -.12 .10 .10 .025 .48 C -.07 .82 .15 1 .40 1 C .69 1.01 1 .73 .99 .38 C 1 .12 .84 -.02 .64 .02 Q .49 .02 .39 .13'),
 (2,.64,'M .66 .025 C .38 -.05 .08 .28 .02 .57 Q -.10 .98 .34 1 C .65 1.04 .95 .70 1 .39 Q 1.07 .08 .66 .025'),
 (1,.77,'M .46 .02 Q .19 .005 .065 .31 Q -.02 .52 .03 .81 Q .06 1.01 .31 1 Q .71 1.02 .94 .76 Q 1.05 .54 .95 .20 Q .84 -.055 .46 .02'),
 ],[
 (10,.34,'M .54 0 Q .48 .48 .54 1'),
 (4,.40,'M .63 0 Q .57 .67 .40 1'),
 (3,.43,'M .20 .19 Q .55 .08 .69 0 Q .61 .51 .54 1'),
 (2,.51,'M .04 .29 L .68 0 Q .65 .47 .55 1'),
 (1,.58,'M .0 .31 L .59 0 L .54 .98 M .07 1 Q .53 .95 1 1'),
 (1,.43,'M .23 .12 Q .53 -.03 .71 0 Q .61 .38 .53 .65 Q .45 .90 .57 1'),
 ],[
 (4,.86,'M .035 .225 C .18 .045 .47 -.055 .67 .025 C 1.04 .16 .74 .45 .49 .65 L .105 .965 Q .49 .89 1 .97'),
 (3,.88,'M .04 .26 C .08 .08 .34 -.04 .54 .01 C .91 .08 .86 .32 .68 .50 Q .49 .70 .05 .99 Q .50 .83 1 .98'),
 (4,.87,'M .06 .20 Q .28 -.04 .53 .015 C .81 .07 .85 .27 .66 .53 Q .46 .78 .19 .91 Q -.01 1.02 .03 .87 Q .075 .72 .34 .77 Q .68 .90 1 .85'),
 (2,.92,'M .12 .23 C .03 -.015 .49 -.10 .68 .10 C .92 .34 .51 .74 .18 .92 C -.02 1.07 -.12 .84 .12 .77 C .43 .67 .67 .98 1 .85'),
 (2,.78,'M 0 .26 C .16 .01 .60 -.12 .82 .13 Q 1 .36 .54 .62 L .09 .99 L 1 .95'),
 (1,.78,'M .07 .30 Q .13 -.01 .46 0 C .83 -.02 1.02 .22 .77 .46 Q .58 .68 .15 .98 Q .63 .98 1 .89'),
 (1,.90,'M .02 .17 C .31 -.07 .78 -.06 .79 .21 Q .83 .46 .47 .69 L .09 .96 Q .32 .77 .56 .91 Q .86 1.05 1 .87'),
 ],[
 (4,.73,'M .035 .145 C .28 -.04 .87 -.07 .90 .16 Q .91 .35 .40 .48 C 1.08 .35 1.13 .74 .83 .90 Q .41 1.12 .035 .87'),
 (3,.71,'M .06 .14 Q .53 -.05 .96 .045 Q .72 .30 .37 .47 C .74 .35 1.06 .48 .93 .74 C .83 .97 .38 1.09 .01 .92'),
 (3,.74,'M .08 .12 Q .53 -.08 .84 .065 Q 1.08 .28 .49 .46 Q .30 .55 .36 .53 C .77 .36 1.10 .57 .87 .82 Q .51 1.13 .01 .90'),
 (2,.75,'M .02 .165 Q .45 -.07 .83 .055 Q 1.12 .20 .55 .43 L .28 .55 Q .62 .39 .87 .55 C 1.22 .81 .54 1.13 .14 .98 Q .05 .96 .02 .85'),
 (2,.70,'M .035 .09 Q .56 -.06 .88 .09 Q 1 .24 .44 .49 Q .97 .45 .99 .65 Q 1.04 .90 .60 .98 Q .21 1.08 .015 .84'),
 (1,.81,'M .025 .16 Q .50 .015 .96 .01 L .34 .46 Q .75 .41 .92 .60 Q 1.12 .89 .55 1 Q .25 1.03 .065 .95'),
 ],[
 (5,.79,'M .31 .015 Q .21 .24 .025 .54 Q .30 .63 .98 .48 M .85 .04 Q .77 .48 .75 1'),
 (4,.78,'M .27 0 Q .15 .20 .015 .46 Q -.045 .70 .38 .59 L .98 .465 M .88 .055 L .73 1'),
 (3,.80,'M .27 .015 Q -.10 .48 .04 .56 Q .16 .66 .91 .48 M .83 .10 Q .76 .56 .80 1'),
 (2,.74,'M .80 0 L .015 .59 L 1 .52 M .81 .01 L .79 1'),
 (3,.82,'M .49 .02 Q .33 .26 .02 .56 Q .48 .54 1 .48 M .77 .065 Q .77 .51 .74 1'),
 (2,.77,'M .15 0 Q .04 .26 .045 .47 C .045 .69 .47 .66 .80 .38 L .91 .08 L .74 1'),
 (1,.74,'M .32 .025 L .025 .555 L .95 .56 M .75 .015 L .74 1'),
 ],[
 (4,.78,'M .96 .01 Q .62 .055 .18 .04 L .075 .48 C .42 .33 .99 .41 .90 .70 C .83 .96 .38 1.10 .025 .93'),
 (4,.78,'M .23 .04 Q .18 .24 .095 .44 C .49 .34 .99 .43 .92 .68 C .87 .93 .37 1.12 .02 .90 M .17 .055 Q .58 .075 1 .005'),
 (3,.79,'M 1 .02 Q .63 .12 .18 .075 Q .11 .28 .08 .43 Q .56 .36 .79 .53 C 1.17 .80 .47 1.12 .04 .92'),
 (2,.77,'M .94 .02 L .22 .04 Q .17 .19 .11 .37 Q .71 .43 .88 .64 C 1.04 .91 .43 1.07 .03 .95'),
 (2,.75,'M .26 .055 Q .14 .26 .085 .37 C .58 .38 1.10 .53 .84 .79 Q .52 1.09 .055 .92 M .26 .02 Q .58 .09 .99 -.01'),
 (2,.77,'M 1 0 L .19 .08 L .12 .52 C .44 .32 .89 .36 .92 .62 C 1 .90 .47 1.13 .10 .92 Q .035 .86 .03 .78'),
 (1,.80,'M .96 .035 Q .58 .03 .15 .09 L .075 .44 C .60 .43 1 .59 .72 .84 Q .41 1.04 .03 .98'),
 ],[
 (5,.74,'M .76 .005 C .42 .14 .06 .52 .06 .77 C .05 1.08 .64 1.07 .85 .85 C 1.08 .60 .83 .45 .60 .50 Q .28 .55 .13 .78'),
 (4,.70,'M .74 .00 Q .23 .24 .075 .64 C -.09 .99 .40 1.09 .70 .95 C 1.08 .78 .94 .51 .68 .51 Q .39 .49 .13 .76'),
 (3,.69,'M .81 .025 Q .54 -.05 .23 .39 Q -.06 .82 .19 .96 C .39 1.11 .94 .94 .94 .73 C .96 .49 .67 .42 .43 .55 Q .24 .65 .20 .81'),
 (2,.76,'M .77 .015 Q .26 .28 .09 .61 Q -.03 .91 .32 .99 Q .62 1.10 .90 .86 Q 1.04 .68 .83 .53 Q .52 .37 .12 .70'),
 (2,.67,'M .72 .0 C .39 .12 .05 .58 .105 .82 Q .16 1.09 .67 .95 C .99 .83 1.0 .55 .72 .50 Q .57 .47 .45 .53'),
 (1,.70,'M .89 .10 C .76 -.10 .52 -.02 .29 .27 C .02 .56 .00 .85 .24 .96 C .56 1.12 .97 .89 .92 .68 Q .83 .35 .30 .63'),
 ],[
 (6,.78,'M .025 .10 Q .36 .025 .93 .055 Q .67 .44 .40 1'),
 (4,.77,'M .025 .075 Q .45 .015 .93 .07 Q .68 .48 .45 .99'),
 (3,.84,'M .0 .12 Q .26 -.035 .53 .04 Q .75 .096 .95 .05 Q .58 .54 .43 1'),
 (3,.75,'M .02 .23 L .075 .055 Q .42 .005 .95 .06 L .38 1'),
 (2,.84,'M .075 .13 Q .49 .015 .94 .035 L .36 1 M .24 .53 Q .51 .52 .80 .49'),
 (1,.73,'M .015 .07 L .96 .04 Q .58 .52 .55 .99 M .30 .51 L .86 .48'),
 (1,.74,'M .0 .09 Q .36 -.06 .69 .065 Q .88 .12 .96 .045 Q .72 .32 .55 .61 Q .40 .94 .29 1'),
 ],[
 (5,.70,'M .59 .04 C .39 -.09 .02 .09 .12 .28 C .22 .46 .92 .56 .93 .78 C .96 1.04 .27 1.13 .105 .86 C -.04 .65 .27 .52 .59 .38 C 1.12 .13 .89 -.025 .59 .04'),
 (4,.69,'M .53 .025 C .22 -.03 .035 .18 .18 .36 C .33 .51 .91 .60 .92 .78 C .94 1.04 .32 1.11 .12 .90 C -.07 .67 .26 .52 .58 .36 Q 1.12 .045 .65 .01 Q .47 -.025 .36 .055'),
 (3,.71,'M .56 .025 C .82 -.055 1 .17 .81 .34 C .58 .52 .075 .58 .08 .81 C .085 1.08 .76 1.06 .90 .86 C 1.08 .59 .67 .51 .40 .37 C .005 .17 .20 -.03 .56 .025'),
 (2,.71,'M .51 .48 C .19 .40 -.02 .29 .15 .11 C .37 -.10 .93 -.01 .90 .20 Q .87 .35 .51 .48 C .17 .61 -.03 .79 .15 .93 C .37 1.12 .96 .99 .93 .78 Q .91 .60 .51 .48'),
 (2,.72,'M .67 .10 C .39 -.13 .05 .10 .16 .30 C .27 .50 .87 .60 .90 .77 Q 1 1.02 .51 1 Q -.02 1.05 .085 .76 Q .18 .59 .58 .39 C .99 .19 .91 .03 .68 .02'),
 (1,.68,'M .73 .045 C .46 -.07 .095 .07 .12 .26 Q .14 .39 .58 .55 Q 1.15 .79 .83 .95 Q .57 1.11 .22 .95 Q -.05 .80 .32 .54 L .70 .28 Q .97 .04 .73 .045'),
 ],[
 (5,.67,'M .83 .11 C .66 -.10 .24 .025 .075 .25 C -.11 .52 .23 .63 .52 .53 Q .72 .46 .88 .14 Q .85 .51 .73 1'),
 (4,.70,'M .83 .13 Q .62 -.095 .29 .07 C -.01 .20 -.075 .51 .15 .55 Q .51 .67 .85 .22 L .91 .07 Q .77 .63 .60 1'),
 (3,.69,'M .86 .15 C .69 -.095 .32 -.03 .12 .20 C -.105 .46 .13 .64 .46 .53 Q .69 .46 .86 .21 Q .99 .65 .70 1'),
 (3,.74,'M .86 .15 C .68 -.095 .25 -.01 .08 .22 Q -.11 .49 .14 .56 Q .48 .68 .82 .26 Q .91 .13 .89 .08 Q .95 .54 .72 .81 Q .51 1.04 .21 .98'),
 (2,.68,'M .86 .12 C .62 -.085 .18 .06 .09 .33 Q -.02 .59 .23 .60 C .56 .61 .81 .36 .87 .13 Q .83 .50 .83 .76 Q .82 .98 .99 1'),
 (2,.70,'M .75 .10 C .48 -.09 .11 .10 .07 .35 Q -.015 .56 .25 .54 Q .63 .48 .86 .105 M .86 .04 Q .83 .55 .58 1'),
 (1,.71,'M .69 .03 Q .31 -.04 .095 .26 Q -.065 .52 .16 .56 Q .49 .65 .84 .25 L .91 .07 Q .89 .63 .65 .91 Q .46 1.07 .23 .96'),
 ]
]

# Additional writing motions: oval with a seam, Z with a hooked entry or
# looped foot, B-like 3, U-like 4, script S, b-like 6, angled cap, stacked
# bowls, and a loop on a long or curved descending stem.
PENS[0] += [
 (2,.70,'M .66 .13 C .54 -.12 .26 .02 .10 .28 C -.11 .63 .02 .98 .35 1 C .70 1.03 .95 .72 .98 .43 Q 1.02 .04 .68 .02 Q .48 .02 .32 .16'),
 (1,.76,'M .72 .09 Q .43 -.10 .19 .18 Q -.015 .38 .01 .63 Q .035 1.03 .38 1 Q .71 1.00 .94 .67 Q 1.08 .42 .86 .18 Q .76 .045 .54 .045'),
 (1,.65,'M .57 .02 Q .07 .16 .025 .57 Q -.015 .91 .32 .99 Q .61 1.09 .89 .78 Q 1 .60 .94 .20 Q .88 -.085 .57 .02'),
 (1,.62,'M .55 .10 C .25 .05 .01 .39 .04 .71 Q .10 1.06 .46 .99 Q .82 .93 .94 .59 C 1.07 .28 .81 -.12 .56 .015'),
]
PENS[1] += [
 (2,.36,'M .30 .06 Q .57 -.04 .62 .045 Q .55 .33 .51 .57 Q .45 .85 .38 .99'),
 (1,.49,'M .015 .37 L .69 .015 Q .60 .61 .62 1'),
]
PENS[2] += [
 (4,.77,'M .09 .45 C -.04 .28 .13 -.02 .44 .015 C .84 .02 .98 .29 .72 .56 Q .48 .80 .075 .97 Q .52 .83 .98 .91'),
 (3,.84,'M .25 .49 C -.08 .33 .095 .055 .35 .015 C .61 -.04 .91 .105 .85 .36 Q .80 .57 .13 .96 Q .54 .88 .98 .84'),
 (3,.84,'M .05 .33 Q .21 -.01 .48 .01 C .88 .035 .82 .35 .58 .61 Q .38 .83 .18 .93 Q -.02 1.03 .025 .83 Q .12 .61 .44 .75 Q .76 .91 .98 .77'),
 (2,.87,'M .10 .36 C .015 .12 .24 -.095 .51 .055 C .79 .20 .79 .48 .49 .73 Q .22 .99 .035 .95 C -.12 .84 .13 .64 .35 .71 Q .71 .87 .99 .91'),
 (2,.73,'M .01 .22 Q .14 -.02 .41 .005 Q .90 .01 .85 .29 Q .79 .57 .13 .95 L .98 .99'),
 (1,.73,'M .11 .32 C .13 .04 .31 -.05 .56 .045 Q .98 .215 .67 .57 Q .41 .83 .08 .92 Q .54 1.03 .98 .77'),
]
PENS[3] += [
 (2,.72,'M .08 .12 Q .61 -.10 .88 .09 Q 1.05 .29 .40 .45 Q .27 .48 .37 .50 C .80 .41 1.00 .60 .87 .79 Q .62 1.10 .14 .91 Q .06 .88 .03 .76'),
 (1,.71,'M .04 .04 Q .57 .01 .94 .055 L .37 .50 Q .68 .34 .90 .51 Q 1.13 .75 .70 .95 Q .35 1.10 .05 .92'),
 (1,.69,'M .07 .17 C .39 -.09 .80 -.02 .88 .15 Q .98 .31 .54 .46 C 1.15 .52 1.02 .76 .74 .92 Q .42 1.09 .035 .90'),
]
PENS[4] += [
 (2,.75,'M .12 .00 Q .10 .28 .04 .51 Q -.02 .68 .25 .62 L 1 .46 M .83 .015 Q .75 .45 .82 .99'),
 (2,.78,'M .69 .015 Q .37 .26 .05 .55 L .97 .52 M .71 .05 Q .65 .50 .73 1'),
 (2,.79,'M .24 .035 C .07 .31 -.04 .59 .13 .61 Q .51 .65 .85 .42 M .83 .26 L .79 1'),
]
PENS[5] += [
 (2,.80,'M .99 .015 Q .62 .01 .32 .065 Q .09 .15 .09 .39 Q .44 .40 .70 .52 C 1.17 .74 .58 1.12 .04 .94'),
 (2,.77,'M .20 .055 Q .20 .23 .07 .47 C .44 .34 .92 .43 .88 .70 Q .83 .99 .43 .99 Q .18 1.00 .055 .84 M .12 .055 Q .56 .10 .97 .00'),
 (1,.74,'M .98 .03 L .23 .05 L .065 .47 C .54 .26 1.03 .48 .88 .73 C .77 .98 .36 1.08 .12 .92 Q .02 .85 .085 .72'),
]
PENS[6] += [
 (2,.74,'M .57 .00 Q .41 .19 .21 .47 Q -.01 .80 .23 .94 Q .48 1.10 .78 .94 C 1.09 .77 .94 .47 .67 .50 Q .40 .50 .20 .67'),
 (2,.69,'M .90 .02 C .58 .11 .14 .48 .10 .73 Q .04 1.02 .43 .99 Q .80 1.01 .92 .74 Q 1 .53 .73 .50 Q .54 .50 .43 .64'),
 (1,.69,'M .63 .015 Q .34 .27 .23 .59 L .16 .82 C .24 1.11 .78 1.02 .88 .80 C 1.01 .52 .69 .40 .41 .53 Q .28 .61 .23 .71'),
]
PENS[7] += [
 (3,.83,'M .01 .22 Q .25 .035 .56 .045 Q .85 .07 .95 .02 Q .67 .37 .48 .66 Q .34 .90 .32 1'),
 (2,.76,'M .01 .28 Q -.005 .01 .24 .04 Q .63 .12 .95 .035 Q .66 .45 .51 .74 Q .44 .91 .38 1'),
 (1,.70,'M .04 .26 Q .22 .095 .87 .015 L .40 1'),
]
PENS[8] += [
 (3,.72,'M .72 .10 C .42 -.12 .08 .015 .15 .25 Q .20 .42 .69 .54 Q 1.06 .68 .88 .87 Q .66 1.09 .26 .96 C -.11 .84 .16 .63 .44 .51 Q .88 .33 .87 .16 Q .88 .08 .76 .05'),
 (2,.65,'M .53 .49 C .25 .41 .02 .26 .16 .105 C .34 -.095 .85 -.005 .83 .21 Q .80 .36 .53 .49 C .22 .64 .02 .83 .21 .94 C .57 1.12 .93 .94 .89 .75 Q .84 .58 .53 .49'),
 (2,.75,'M .41 .49 C 1.02 .30 .92 .04 .61 .015 C .21 -.035 -.01 .20 .19 .36 Q .30 .45 .62 .55 C 1.08 .72 .89 .93 .62 .99 C .30 1.05 -.02 .91 .13 .73 Q .21 .60 .41 .49'),
 (1,.68,'M .71 .02 C .37 -.045 .075 .105 .16 .30 Q .22 .43 .65 .57 C 1.04 .70 .92 .96 .60 .99 Q .15 1.09 .105 .82 Q .08 .64 .41 .50 Q .86 .26 .85 .10'),
]
PENS[9] += [
 (3,.72,'M .84 .19 Q .73 -.11 .38 .055 Q .02 .19 .05 .42 Q .11 .63 .39 .54 Q .69 .45 .83 .21 Q .73 .67 .84 1'),
 (3,.79,'M .86 .09 C .60 -.08 .27 .08 .11 .27 Q -.04 .47 .14 .56 Q .41 .68 .78 .36 L .91 .11 Q .89 .56 .58 1'),
 (2,.70,'M .74 .07 Q .53 -.065 .30 .075 Q .02 .23 .08 .41 Q .17 .60 .48 .48 Q .71 .38 .82 .16 M .80 .03 Q .80 .61 .61 1'),
 (2,.74,'M .82 .18 C .53 -.13 .13 .075 .05 .35 Q -.01 .55 .21 .60 Q .52 .66 .84 .26 Q .98 .70 .58 .94 Q .43 1.04 .26 .99'),
 (2,.72,'M .83 .16 C .70 -.09 .38 -.03 .15 .20 Q -.06 .42 .12 .55 Q .33 .70 .65 .45 L .86 .25 Q .81 .53 .86 .78 Q .88 .98 1 .99'),
]

def path_points(text):
    tokens = iter(text.split()); pts = []
    for cmd in tokens:
        def pt(): return np.array([float(next(tokens)),float(next(tokens))])
        if cmd == 'M':
            a=pt(); pts.append(a[None,:]); continue
        control=[a]+[pt() for _ in range({'L':1,'Q':2,'C':3}[cmd])]
        n=int(np.ceil(64*sum(np.linalg.norm(v-u) for u,v in zip(control,control[1:]))))+1
        t=np.linspace(0,1,n)[:,None]
        vs=control
        while len(vs)>1: vs=[(1-t)*u+t*v for u,v in zip(vs,vs[1:])]
        pts.append(vs[0]); a=control[-1]
    return np.concatenate(pts)

rng=np.random.default_rng(72571)
styles=[[(f,w,path_points(s)) for f,w,s in cls] for cls in PENS]
freq=[np.cumsum([f for f,_,_ in cls]) for cls in styles]
oy,ox=np.mgrid[-4:5,-4:5]; ox=ox.ravel(); oy=oy.ravel()
Y,X=np.mgrid[:28,:28]

def script(k, aspect_regime=None, slant_regime=None):
    """Vary pen geometry in continuous coordinates, not an image corpus."""
    _,aspect,points=styles[k][np.searchsorted(freq[k],rng.integers(freq[k][-1])+1)]
    x,y=points.T; x=x-.5; y=y-.5
    g=np.clip(rng.normal(size=21),-2.35,2.35)
    # Changes of bowl width and of waist height, arch vs straight strokes,
    # and an oblique writing axis. Shared smooth warps preserve crossings.
    wide=aspect*np.exp(.245*g[0]); slant=.12+.29*g[1]
    if k==1: slant=.19+.33*g[1]
    if aspect_regime is not None:
        wide=aspect*np.exp(aspect_regime+.17*g[0]); slant=slant_regime+.15*g[1]
    xw=(1+.42*y*g[2]+.14*g[3]*np.cos(2*np.pi*y))*x
    xw+=.082*g[4]*np.cos(np.pi*2*y)+.032*g[5]*np.sin(np.pi*2*y)+.08*g[6]*(x*x-.1)
    yw=y+.128*g[7]*np.cos(np.pi*y)**2+.13*g[8]*x+.14*g[9]*x*y+.068*g[10]*x*np.cos(2*np.pi*y)
    xx=wide*xw-slant*yw; yy=yw+.08*g[11]*(x*x-.1)
    radius=np.clip(1.36+.40*g[12],.61,2.34)
    rx=radius*np.exp(.115*g[13]); ry=radius*np.exp(-.115*g[13])
    pressure=np.clip(1+.095*g[14]*np.sin(np.pi*2*y)+.14*g[15]*x,.65,1.35)
    scale=(20-.25*g[16]-2*radius)/max(np.ptp(xx),np.ptp(yy),.05)
    xx=(xx-(xx.max()+xx.min())/2)*scale+13.5
    yy=(yy-(yy.max()+yy.min())/2)*scale+13.5
    xi=np.floor(xx+.5).astype(int)[:,None]+ox; yi=np.floor(yy+.5).astype(int)[:,None]+oy
    dist=np.sqrt(((xi-xx[:,None])/rx)**2+((yi-yy[:,None])/ry)**2)
    alpha=np.clip(.5+(pressure[:,None]-dist)*radius/.85,0,1).astype(np.float32)
    a=np.zeros((28,28),np.float32)
    np.maximum.at(a.ravel(),np.clip(yi*28+xi,0,783),alpha)
    mass=a.sum(); dx=13.7+.40*g[17]-(a*X).sum()/mass; dy=13.7+.40*g[18]-(a*Y).sum()/mass
    x0=int(np.floor(dx)); y0=int(np.floor(dy)); fx=dx-x0; fy=dy-y0
    a=(1-fx)*(1-fy)*np.roll(a,(y0,x0),(0,1))+fx*(1-fy)*np.roll(a,(y0,x0+1),(0,1))+(1-fx)*fy*np.roll(a,(y0+1,x0),(0,1))+fx*fy*np.roll(a,(y0+1,x0+1),(0,1))
    return a.astype(np.float32)

# Network compiler. Every entry of a linear layer is either a geometric
# operator or a direct evaluation of a shape formula.
layers=[]; weights={}
def layer(n,op,s,**opts):
    layers.append(dict(name=n,op=op,inputs=[s] if isinstance(s,str) else s,**opts)); return n

def affine(n,s,W,b=0,conv=False,pad=0):
    W=np.asarray(W,dtype=np.float32); weights[n+'.W']=W
    weights[n+'.b']=np.broadcast_to(np.float32(b),(len(W),)).copy()
    if conv: return layer(n,'conv',s,out=len(W),kernel=list(W.shape[-2:]),stride=1,padding=pad)
    return layer(n,'dense',s,out=len(W))
def view(n,s,shape): return layer(n,'reshape',s,shape=list(shape))
def pool(n,s,size,step,op='avgpool'): return layer(n,op,s,size=list(size),stride=list(step))
def normvector(n,s,d): return layer(n,'norm',view(n+'_flat',s,(1,1,d)))
def sumlayer(n,*ss): return layer(n,'add',list(ss))
def relu(n,s): return layer(n,'relu',s)

# Tangentially elongated edge detectors and bright centreline detectors.
# Mean subtraction gives each kernel zero response to constant ink level.
def kernels():
    y,x=np.mgrid[-3:4,-3:4]; ks=[]
    for typ,N in [('edge',8),('ridge',4)]:
        for j in range(N):
            angle=2*np.pi*j/8 if typ=='edge' else np.pi*j/4
            u=x*np.cos(angle)+y*np.sin(angle); v=-x*np.sin(angle)+y*np.cos(angle)
            sig=1.14 if typ=='edge' else 1.27
            gauss=np.exp(-.5*((u/sig)**2+(v/1.85)**2))
            w=gauss*(u if typ=='edge' else 1-(u/sig)**2); w-=w.mean()
            ks.append(w / np.abs(w).sum()*2)
    return np.array(ks,dtype=np.float32)
ks=kernels(); response=relu('line_evidence',affine('line_filters','image',ks[:,None],conv=True,pad=3))
linegrid=pool('line_cells',response,(4,4),(3,3))
# Give spiky strong edges less leverage by adding a clipped response.
# min(z,t) = z - relu(z-t), the baseline z is retained too.
C=len(ks); I=np.eye(C,dtype=np.float32)[:,:,None,None]
ex=relu('excess',affine('line_soft_threshold',linegrid,I,-.26,conv=True))
linegrid=sumlayer('line_compress',linegrid,affine('line_trim',ex,-.66*I,conv=True))
line=normvector('line_vector',linegrid,12*81)
inkgrid=pool('ink_cells','image',(4,4),(2,2)); ink=normvector('ink_vector',inkgrid,169)

# Look from each empty cell along the eight compass rays. Ray sums clipped
# to [0,1] are soft "blocked by ink" flags.
small=pool('coarse_ink','image',(2,2),(2,2)); H=14
rays=[(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1)]
R=np.zeros((9,H,H,H,H),np.float32)
for i in range(H):
    for j in range(H):
        for r,(dy,dx) in enumerate(rays):
            a=i+dy; b=j+dx
            while 0<=a<H and 0<=b<H:
                R[r,i,j,a,b]=1.45 if r<4 else 1.7; a+=dy; b+=dx
        R[-1,i,j,i,j]=1
R=R.reshape(9*H*H,H*H)
rs=view('ray_planes',affine('sightlines',small,R),(9,H,H))
J=np.eye(9,dtype=np.float32)[:,:,None,None];
over=relu('ray_excess',affine('ray_cap',rs,J,-1,conv=True))
occ=sumlayer('blocked_rays',rs,affine('ray_clip',over,-J,conv=True))
CW=[]; CB=[]
def cavity(pos,neg,th,scale=1):
    q=np.zeros(9); q[pos]=1; q[neg]=-1; q[8]=-1.3
    CW.append(q*scale); CB.append(-th*scale)
for u,v in [(0,2),(1,3),(4,6),(5,7)]: cavity([u,v],[],1.15)
for o in range(4): cavity([j for j in range(4) if j!=o],[o],2.15)
cavity([0,1,2,3],[],3.13)
cavity(list(range(8)),[],6.55,.72)
CW=np.asarray(CW,dtype=np.float32); CB=np.asarray(CB,dtype=np.float32)
cav=relu('bays',affine('bay_tests',occ,CW[:,:,None,None],CB,conv=True))
cgrid=pool('bay_cells',cav,(2,2),(2,2)); bays=normvector('bay_vector',cgrid,len(CW)*49)

# Grayscale flood fill: propagate the easiest route in from the border.
# Negative ink is the maximum admissible value at a pixel. Maxpool expands
# visited background by one step, with padding (=0) as the exterior source.
scalar=lambda x: np.array([[[[x]]]],np.float32)
neg=affine('negative_ink',small,scalar(-1),conv=True)
front=affine('interior_seed',small,scalar(0),-1,conv=True)
FLOOD=16
for t in range(FLOOD):
    s=str(t)
    pp=affine('rim'+s,front,scalar(1),conv=True,pad=1)
    spread=pool('reach'+s,pp,(3,3),(1,1),'maxpool')
    nr=affine('depth'+s,spread,scalar(-1),conv=True)
    dif=relu('gap'+s,sumlayer('excess_depth'+s,neg,nr))
    front=sumlayer('visit'+s,neg,affine('constraint'+s,dif,scalar(-1),conv=True))
# dif = (filled surface - ink) at convergence; use area and strongest
# enclosure in overlapping horizontal bands as complementary loop cues.
holegrid=pool('enclosed_cells',dif,(2,2),(2,2))
holeband=pool('enclosed_bands',dif,(6,14),(4,1),'maxpool')

# Degree-one stroke ends: Gaussian disk of ink with a continuation in one
# direction and little ink on the exterior semicircle. A smooth unbroken
# loop has no ends, unlike a 6, 9, or an open digit. Measured at two scales.
def end_kernels():
    y,x=np.mgrid[-3:4,-3:4]; out=[]
    for r in (1.75,2.6):
        for theta in np.arange(8)*np.pi/4:
            def disk(dist,ang,sig):
                xx=dist*np.cos(theta+ang); yy=dist*np.sin(theta+ang)
                w=np.exp(-((x-xx)**2+(y-yy)**2)/(2*sig**2)); return w/w.sum()
            w=.58*disk(0,0,.52)+.42*disk(r*.70,0,.65)
            for ang,wt in [(90,.38),(-90,.38),(135,.20),(-135,.20),(180,.38)]:
                w-=wt*disk(r,np.deg2rad(ang),.61)
            out.append(w)
    return np.array(out,dtype=np.float32)
ek=end_kernels(); ep=relu('end_evidence',affine('end_detectors',small,ek[:,None],-.29,conv=True,pad=3))
# Eight directed end maps; keep stronger of the two physical scales.
ep=pool('end_scales_join',view('end_scales_first',ep,(1,2,8*196)),(2,1),(2,1),'maxpool')
ep=view('oriented_ends',ep,(8,14,14))
endgrid=pool('end_cells',ep,(5,5),(3,3),'maxpool')

# Compare vertical support of ink with vertical support of inside boundary.
# An ink pixel is far from a loop if there is no filled interior anywhere
# in nearby rows. Spread the boundary over the width at each such row.
bound=pool('enclosed_rows',dif,(1,14),(1,1),'maxpool')
rowD=np.zeros((14,14,14),np.float32)
for i in range(14): rowD[i,:,i]=1
rowmap=view('rows_over_plane',affine('broadcast_rows',bound,rowD.reshape(196,14)),(1,14,14))
rowpad=affine('soft_border_pad',rowmap,scalar(1),conv=True,pad=2)
near=pool('near_enclosure',rowpad,(5,5),(1,1),'maxpool')
stem=relu('protruding_ink',sumlayer('ink_beyond_enclosure',small,affine('discount_nearby',near,scalar(-2.5),conv=True)))
stemgrid=pool('extension_cells',stem,(4,4),(2,2))

# A compact compiler-side evaluation of the feature FORMULAS on ink drawn by
# script(), never on external images. Blocked rays and filled cavities are
# deterministic geometric functions of the chosen control points.
def means(a,h,w,sy,sx):
    hi=(a.shape[-2]-h)//sy+1; wi=(a.shape[-1]-w)//sx+1
    return sum(a[...,i:i+sy*hi:sy,j:j+sx*wi:sx] for i in range(h) for j in range(w))/(h*w)

def features(a):
    B=len(a); ap=np.pad(a,((0,0),(3,3),(3,3)))
    patches=np.lib.stride_tricks.sliding_window_view(ap,(7,7),axis=(-2,-1))
    res=patches.reshape(B*28*28,49)@ks.reshape(12,49).T
    zz=means(np.maximum(res.reshape(B,28,28,12).transpose(0,3,1,2),0),4,4,3,3)
    zz-=.66*np.maximum(zz-.26,0)
    aa=means(a,4,4,2,2)
    sm=means(a,2,2,2,2); ray=np.minimum(sm.reshape(B,196)@R.T,1).reshape(B,9,196)
    cc=np.maximum(CW@ray+CB[None,:,None],0).reshape(B,len(CW),14,14); cc=means(cc,2,2,2,2)
    filled=np.ones_like(sm)
    for _ in range(FLOOD):
        pp=np.pad(filled,((0,0),(1,1),(1,1)))
        erosion=np.minimum.reduce([pp[:,i:i+14,j:j+14] for i in range(3) for j in range(3)])
        diff=np.maximum(erosion-sm,0); filled=sm+diff
    hh=means(diff,2,2,2,2); hb=np.stack([diff[:,i:i+6].max((1,2)) for i in (0,4,8)],axis=1)
    sp=np.pad(sm,((0,0),(3,3),(3,3)))
    sh=np.lib.stride_tricks.sliding_window_view(sp,(7,7),axis=(-2,-1))
    e=sh.reshape(-1,49)@ek.reshape(16,49).T
    e=np.maximum(e.reshape(B,14,14,2,8).max(3).transpose(0,3,1,2)-.29,0)
    e=np.maximum.reduce([e[:,:,i:i+10:3,j:j+10:3] for i in range(5) for j in range(5)])
    rows=np.pad(diff.max(2),((0,0),(2,2)))
    near=np.maximum.reduce([rows[:,i:i+14] for i in range(5)])
    st=means(np.maximum(sm-2.5*near[:,:,None],0),4,4,2,2)
    return zz,aa,cc,hh,hb,e,st

# Cosine comparisons after joint centering, one descriptor block at a
# time. At runtime norm() yields a vector of length sqrt(d), so the analytic
# unit-length template has an additional 1/sqrt(d) factor in its weights.
# Loop locations use an absolute (not normalized) squared-distance metric.
N=4096; GROUPS=4
featnames=['line_match','ink_match','bay_match','loop_area_match','loop_band_match','end_match','extension_match']
dims=[12*81,169,len(CW)*49,49,3,128,36]
source=[line,ink,bays,holegrid,holeband,endgrid,stemgrid]
for n,d,src in zip(featnames,dims,source):
    affine(n,src,np.empty((10*N,d),dtype=np.float32))
B=64
for digit in range(10):
    for i in range(0,N,B):
        a=np.stack([script(digit) for _ in range(B)])
        desc=features(a); sl=slice(digit*N+i,digit*N+i+B)
        for j,(nm,x) in enumerate(zip(featnames,desc)):
            x=x.reshape(B,-1)
            if j<3:
                x=x-x.mean(1,keepdims=True)
                importance=[.52,.19,.29][j]; d=x.shape[1]
                w=x*(importance/(np.sqrt(d)*np.maximum(np.sqrt((x*x).sum(1,keepdims=True)),1e-7)))
            else:
                # Equal units per squared displacement within each descriptor.
                importance={3:.012,4:.022,5:.035,6:.024}[j]
                w=2*importance*x
                weights[nm+'.b'][sl]=-importance*(x*x).sum(1)
            weights[nm+'.W'][sl]=w
    print('compiled digit',digit,flush=True)

combined=sumlayer('writing_agreement',*featnames)
layout=view('alternative_shapes',combined,(10,GROUPS,N//GROUPS))
votes=pool('best_shapes',layout,(1,N//GROUPS),(1,N//GROUPS),'maxpool')
vote=pool('typical_best',votes,(GROUPS,1),(GROUPS,1))
best=pool('closest_shape',votes,(GROUPS,1),(GROUPS,1),'maxpool')
score=sumlayer('consensus',vote,best)

# Product shape model. Tops and bottoms may be selected independently but
# must have compatible width and writing axis, and agree on a central strip.
# Allows an unfamiliar combination of familiar stroke fragments.
parts=[]
for label,sg in [('cap',0),('foot',1)]:
    for typ,src,ch,h,nrow in [('oriented',linegrid,12,9,6),('concave',cgrid,10,7,4)]:
        name=label+'_'+typ
        start=sg*(h-nrow); ids=np.arange(ch*h).reshape(ch,h)[:,start:start+nrow].ravel()
        select=np.eye(ch*h,dtype=np.float32)[ids,:,None,None]
        patch=affine(name+'_slice',view(name+'_rows',src,(ch*h,1,h)),select,conv=True)
        d=ch*nrow*h; unit=normvector(name+'_unit',patch,d)
        parts.append((name,unit,ch,h,start,nrow))
REGIMES=9; M=256; P=REGIMES*M
for name,unit,ch,h,start,nrow in parts:
    affine(name+'_compare',unit,np.empty((10*P,ch*nrow*h),np.float32))
for k in range(10):
    for r in range(REGIMES):
        width=(-.24,0,.24)[r//3]; axis=(-.20,.12,.46)[r%3]
        for i in range(0,M,B):
            a=np.stack([script(k,width,axis) for _ in range(B)])
            desc=features(a)
            for n,unit,ch,h,start,nrow in parts:
                samples=desc[0 if ch==12 else 2][:,:,start:start+nrow,:].reshape(B,-1)
                x=samples-samples.mean(1,keepdims=True)
                importance=.70 if ch==12 else .30; d=x.shape[1]
                w=x*(importance/(np.sqrt(d)*np.maximum(np.sqrt((x*x).sum(1,keepdims=True)),1e-7)))
                ix=(k*REGIMES+r)*M+i
                weights[n+'_compare.W'][ix:ix+B]=w
    print('compiled compatible parts',k,flush=True)
partvotes=[]
for label in ['cap','foot']:
    joined=sumlayer(label+'_agreement',label+'_oriented_compare',label+'_concave_compare')
    groups=view(label+'_by_regime',joined,(10,REGIMES,M))
    partvotes.append(pool(label+'_best',groups,(1,M),(1,M),'maxpool'))
pair=sumlayer('compatible_parts',*partvotes)
both=pool('best_compatible_pair',pair,(REGIMES,1),(REGIMES,1),'maxpool')
score=sumlayer('joint_decision',affine('whole_vote',score,.61*np.eye(10)[:,:,None,None],conv=True),affine('parts_vote',both,.39*np.eye(10)[:,:,None,None],conv=True))
view('digit_scores',score,(10,))

out=sys.argv[1]; os.makedirs(out,exist_ok=True)
with open(os.path.join(out,'network.json'),'w') as f: json.dump({'layers':layers},f,indent=1)
np.savez(os.path.join(out,'weights.npz'),**weights)
