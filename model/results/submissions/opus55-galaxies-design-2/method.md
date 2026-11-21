# Method

**Architecture.** A fixed image-analysis pipeline written as a network of conv / dense / relu / pool / add /
reshape / norm layers, followed by a hand-written soft decision tree that mirrors the Galaxy Zoo question tree.
No weight was trained, fitted, or taken from images. Every weight is a formula or a constant chosen by hand in `build.py`.

1. **Front end.** An exact 2x2 phase split (stride-2 conv with one-hot kernels) turns the 128x128 image into a
   64x64x12 tensor, so per-pixel operations at full resolution become cheap channel operations. Gaussian smoothing
   (sigma = 1 px) and a luminance high-pass follow.
2. **Background and noise.** These are measured per image: the minimum over 8x8 blocks of the block means (avgpool, then
   maxpool of the negated maps), and the same for |high-pass|.
3. **Isophote thresholds.** Outer: background + max(2 x noise, 0.015). Mid and inner: 25% and 60% of the way
   from the background to the central peak.
4. **Segmentation of the central galaxy.** Geodesic region growing from a central seed disk, done as 30 iterations of
   a 3x3 conv plus clamp (relu pairs). This separates the galaxy from stars and neighbours that do not touch it.
   Soft masks at the three isophote levels are gated by the grown region.
5. **Features.** All features are scale-free, computed with dense layers and with ratios done exactly by the
   `norm` layer on vectors of the form [lambda D, -lambda D, u, -u, ...].
   - Second moments of each mask, centred with piecewise-linear squares, give axis ratios, orientations and a
     multiplicity index (4 pi <r^2>/area against the (1+q^2)/q of an ellipse).
   - g/z colour (B/R) of the outer annulus. The colour boundaries come from the Legacy Survey viewer's asinh RGB
     scales together with textbook colours of ellipticals and of Sb and Sc disks.
   - 180-degree rotational asymmetry of light and of the outer mask (dense permutation, minimum over 25 centre shifts).
   - An extended-secondary-nucleus detector: local maximum, contrast against ring maxima, with point sources and
     plateau edges penalised.
   - CAS-style clumpiness.
   - A log-spiral (Kalnajs / 2DFFT) decomposition (m = 1, 2, 3; several radial frequencies p) in a bank of 41
     deprojected frames (6 axis ratios x 8 position angles). The minimum is taken over the frames consistent with
     the measured outer shape. Arm strength is the chirality excess |S(+p) - S(-p)|, since inclination leakage is
     symmetric in p.
   - Bar indicators: inner isophote more elongated than, or misaligned with, the disk.
6. **Decision.** Node logits (merging, disturbed, edge-on, bulge, featured, round, cigar, bar, tight) are sums of
   clipped linear terms with hand-set thresholds (Galaxy Zoo / CAS conventions, colours from stellar populations).
   Each class score is minus the summed violation of the conditions on its path through the tree, so the argmax
   is a soft version of the tree.

**Checks.** Synthetic renders (Sersic ellipses, log spirals, bars, pairs, stars) were used only to verify that
each computation measures what it should, such as recovering axis ratios or the spiral pitch. They were never used
to choose numbers. The 5 labelled checks on training images were used as diagnostics, with changes justified by a
physical failure mechanism (for example, saturated flat-topped cores faking second nuclei).

**Status at submission.** The 2 labelled checks I used both scored about 15%. In both, the merger branch fired on
roughly 85% of images. The cause turned out to be a float32 bug in region growing: M = relu(d) - relu(d-1) is not
exactly 1, and the residue grows by up to 9x per step, so every bright star in the frame joined the galaxy region.
The submitted network fixes this with a 0.02 margin. It also grows the mid-isophote region separately (field stars
no longer fake second nuclei) and uses grayscale reconstruction so stars do not contaminate the arm and clumpiness
features. These fixes were verified only on synthetic scenes with stars and correlated noise. The fixed network has
not been run on labelled training images, so its accuracy is unmeasured.
