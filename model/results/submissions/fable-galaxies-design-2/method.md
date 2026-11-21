# Method

The network is a classical image-measurement pipeline written as conv / pool / relu / add / reshape / norm / dense layers.
Every weight is produced by `build.py` from formulas and hand-chosen constants; nothing was trained or fitted, and no
training image, statistic or measurement is stored in the weights.

**Architecture**

1. 2x2 average pooling to 64x64, then a 5x5 binomial blur giving a smoothed luminance and a smoothed red-minus-blue map.
2. Sky level of each image = the darkest 16x16 block mean (avgpool, then max of the negated map); it is broadcast back
   with a dense layer and subtracted, together with a fixed isophote level (0.04 above the sky).
3. A soft threshold gives the footprint mask. Sixteen steps of geodesic dilation (pad, 5x5 maxpool, add `mask - 1`, relu)
   grow a small central seed inside the mask, which keeps only the connected object at the image centre and drops
   other stars and galaxies in the frame.
4. One full-frame conv takes moments of that object: second moments along 16 directions for the mask and for the
   flux, first moments along 16 directions, plus area, flux and red-minus-blue flux. Max / min over directions
   (reshape + maxpool) give major and minor moments and the centroid offset.
5. Ratios are formed with the norm layer: a channel holding `[x, -x, K*y, -K*y]` normalises to about `sqrt(2)*x/(K*y)`.
   This gives squared axis ratio (mask and flux-weighted), size, colour and centroid offset.
6. A hand-written rule table (dense, relu, dense): hinge penalties for each class on axis ratio (round / in-between /
   cigar / edge-on), size (small smooth vs. resolved discs), colour (red-sequence vs. blue discs, from the g, r, z
   display colours of red and blue galaxies), bar evidence (inner light more elongated than the footprint) and
   off-centre footprints (merging / disturbed).

**Status and limits**

This is a first version saved when time ran out. It was not run on any training image (none of the 5 checks
was used), so its accuracy is unmeasured and the hand-chosen thresholds are uncalibrated. Planned parts that are not
in it: ring/angular-harmonic detection of spiral arms, a second-nucleus detector for mergers, and a bulge test for
edge-on discs; spiral sub-classes, merging and disturbed galaxies are therefore separated only by colour, size and
centroid offset.
