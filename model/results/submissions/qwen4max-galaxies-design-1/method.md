# Method

Everything in `build.py` is analytic: Gaussian/annulus windows, ellipses, logarithmic
spirals, Fourier modes, bar and double-nucleus shape models, and hand-chosen thresholds.
No training image, label, statistic, crop or fitted number is used, and `build.py` reads
no files.

**Architecture.** Two 1x1 convs form a red-weighted luminance map and a blue-weighted map.
Three "functional" convs (kernel 128, stride 128, so each output is one centred linear
measurement of the whole frame) evaluate a bank of hand-designed kernels: compensated
radial apertures at 8 scales, traceless quadrupoles plus the trace moment at 4 scales
(16 orientations each), angular Fourier modes m=1,2,4 in 3 annuli (8 orientations),
logarithmic-spiral matched filters at pitch 12/22/35/90 degrees (16 orientations),
symmetric and centre+offset double-nucleus matched filters at 3 separations, and
three-band aperture photometry in 3 apertures. Every kernel is made exactly zero-sum so
a constant sky contributes nothing. A difference-of-Gaussians branch on the blue map gives
clumpiness, and a 32x32 branch with a fixed permutation dense layer gives the 180-degree
asymmetry.

**Orientation** is marginalised by max-pooling over each orientation block.
**Brightness** is divided out without a division op: for a pair (a,b) the group
[a,-a,b,-b] has zero mean and std sqrt((a^2+b^2)/2), so after `norm` its first entry is
sqrt(2)*(a/b)/sqrt(1+(a/b)^2) — a bounded monotone encoding of the ratio. This yields 34
interpretable features: axis ratio at four scales (via (M-2Qmax)/(M+2Qmax) = q^2),
arm strength per pitch angle, lopsidedness/boxiness, colour indices, concentration
ratios, bar proxy (inner elongation excess), bulge proxy (core rounder than disk),
double-nucleus excess, clumpiness and asymmetry.

**Readout.** A hand-written fuzzy rule table: 36 clipped ramps/hinges (dense+relu) on
those features, then a hand-set 10x36 score table. Thin disks (q<0.3) go to the edge-on
classes, split by whether the core is rounder than the disk (bulge); smooth galaxies are
split into round / in-between / cigar by axis ratio; disk galaxies go to barred (inner
elongation excess), tight (short-pitch arms, concentrated, red core) or loose (open arms,
blue, extended); high asymmetry/clumpiness/lopsidedness goes to disturbed, and a
double-nucleus excess goes to merging.

Thresholds were derived algebraically from the encoding formula and from general
knowledge of Galaxy Zoo morphology (e.g. q=0.8/0.5/0.3 boundaries), not from the data.

## Verification

`check_network.py` reports 35 layers, 6,263,700 weights (limit 120M), all shapes valid.
`check_rebuild.py` reproduces both files exactly (max relative difference 0.0) with no
data present. Five checks were run on 1500 training images: 0.101, 0.106, 0.189, 0.219,
0.230. The first two exposed two real bugs, both fixed from the algebra rather than from
the data: (1) the sky reference was a centred Gaussian, so the r^2-weighted trace moment
subtracted the galaxy's own flux and went negative for compact galaxies — replaced by an
annulus at r=52 px; (2) the double-nucleus reference blob was much broader than the
template blobs, making every centred galaxy look like a pair — replaced by a same-scale
central blob. The third exposed unnormalised ramps in the score table, which made some
classes unreachable; all memberships are now normalised to [0,1]. After the last check,
one knowledge-based correction was applied without re-checking: a thin disk (q<0.26) must
take the edge-on classes, so the spiral/barred scores carry a strong thin-disk penalty.
