# Method

A small hand-designed network; every weight comes from a formula or a hand-typed constant in `build.py`.
No training, fitting or statistic of any image was used.

**Architecture** (`conv -> relu -> dense -> relu -> dense -> reshape -> norm -> dense -> relu -> dense`)

1. `c1` (3x3 conv) blurs with a binomial kernel and makes three maps: luminance, blue (g band) and red (z band),
   each minus a floor of 0.2. The following relu removes the sky, whose level (about 0.127) follows from the
   survey viewer's asinh rendering formula.
2. `d1` (dense) takes Gaussian-windowed sums about the image centre: core flux, wide flux, blue flux, red flux,
   the m=2 angular moment (cos 2θ, 8 phases) and a core-suppressed m=1 angular moment (cos θ, 8 phases).
   After a relu, summing the 8 phases gives the moment's magnitude whatever the galaxy's angle.
3. `d2` arranges four triples (a, b, 0); the `norm` op over each triple returns a function of a/b alone, which
   makes each cue independent of brightness:
   - elongation `|m2|/flux`, which equals (1-q)/(1+q) for elliptical isophotes of axis ratio q;
   - compactness `core flux / wide flux`;
   - colour `blue / red`;
   - lopsidedness `|m1|/flux` (companions, mergers, disturbance).
4. `d3`/`d4` turn each ratio into three soft states with two relu steps (thresholds typed by hand, e.g. q of about
   0.79 and 0.45) and add hand-typed log-likelihoods per class plus log class frequencies, as a naive-Bayes table.

**Where the numbers come from:** thresholds and the likelihood table are my own judgement of the classes
(round / in-between / cigar by axis ratio, edge-on galaxies elongated and extended, spirals bluer and more
extended, mergers lopsided); class frequencies are the published Galaxy10 DECaLS class counts.

**Status:** this is a first, minimal version saved when time ran out. Spiral sub-classes (barred / tight / loose)
are separated only weakly, by colour and size, and no thresholds were tuned.
