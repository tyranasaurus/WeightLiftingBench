# Method

**Architecture (444 layers, 4.53M weights, all hand-designed).** The network is a
classical galaxy-morphology pipeline expressed in the eight allowed blocks:

1. **Sky removal.** Luminance sky = minimum of the 16x16 block means (sky is the
   darkest thing in frame); colour sky = mean over an outer annulus (52-64 px),
   because for colour the minimum-block trick fails (a red galaxy's centre is the
   most negative colour block). The sky value is broadcast with a dense layer and
   subtracted with `add`.
2. **Normalisation and threshold ladder.** `norm` standardises the sky-subtracted
   luminance per image; `relu(x - t)` at t = 0.7, 1, 2, 4 sigma gives four maps
   (body, arms, core, nucleus). Since the image contrast ratio sigma_L/sky-noise
   is about 3.8, the 0.7-sigma map is already speckle-free.
3. **Global linear features** (conv, kernel 128, stride 128 -> one scalar per
   kernel): Gaussian-windowed light sums; ring masses (r = 0-6, 6-14, 14-26,
   26-42 px); 8-sector masses in two annuli; second moments (a = <x^2-y^2>,
   b = <2xy>, t = <x^2+y^2>) at window scales 8/16/24 px, in coordinates scaled
   by the window width; 128 log-spiral m=2 templates (2 annuli x 4 pitch angles
   x 2 hands x 8 phases); 8 rotated bar-ridge templates; colour sums (g-z, r-z)
   paired with light sums. Flip (180 deg) asymmetry is computed on the pooled
   body map with a dense permutation. Clump energy = relu(DoG) masses at two
   scale pairs.
4. **Whitening (the key trick).** The op set has no division, but ratios
   (axis ratio, concentration, colour index, size) are needed. Selected scalars
   are broadcast onto orthonormal zero-mean DCT patterns in a 16x16 map via a
   dense layer; `norm` then divides by the group RMS (patterns are orthogonal,
   so the channel variance is exactly sum(feature^2)/256 plus a floor constant
   supplied through the dense bias); a conv projects back. The result is
   feature_i / sqrt(sum features^2 + floor^2) - a bounded, scale-free,
   brightness-invariant version of the feature group. Pairwise whitening turns
   any two scalars into a bounded monotone function of their ratio. Max-pooling
   over arranged scalars gives rotation-invariant elongation magnitude
   (max over 8 directional projections of (a, b)) and Fourier amplitudes of the
   sector masses (m=1, m=2 modes).
5. **Decision rule.** ~35 hand-written evidence units (relu gates, including
   linear conjunction gates such as "thin at core AND body AND extended AND
   well-detected" for edge-on, or "thin AND red" for an edge-on bulge) followed
   by a linear class score. The rule mirrors the Galaxy Zoo decision tree:
   edge-on? -> bulge (colour + core dominance); else smooth vs featured ->
   axis-ratio bands (round / in-between / cigar use elongation thresholds
   derived from physical axis ratios q = 0.7, 0.55, 0.45, 0.3 mapped through
   e = epsilon/sqrt(1+epsilon^2), epsilon = (1-q^2)/(1+q^2)); featured ->
   bar (inner-vs-outer elongation contrast + ridge anisotropy) and winding
   (tight arms partially cancel the m=2 azimuthal mode, so tight-wound spirals
   show low m=2 with arms present; loose spirals show strong m=2). Irregulars
   use 180-deg asymmetry, lopsidedness (m=1) and clump energy, gated on
   detection significance so noise cannot fake them.

**Where the numbers come from.** Every kernel is an analytic function (Gaussians,
difference-of-Gaussians, rings, sectors, log-spirals, ridges, DCT basis). All
thresholds are physical (axis-ratio values, arm-contrast levels, significance
gates) or noise-floor guards. No statistics of the training images or labels were
fitted or copied into weights; the five allowed runs of run_network.py were used
only to verify the pipeline and to catch gross miscalibrations (e.g. that real
normal galaxies are far more asymmetric than idealised models, which required the
merger/disturbed gates to be raised).

**Result on the training set (check 5): 31.5% accuracy** (random = 10%).
Strongest classes: edge-on without bulge (67% recall), round smooth (55%),
in-between smooth (36%), merging (31%). The architecture deliberately embodies
the whole solution - fixed analytic feature extractors plus a rigid hand-written
rule - so gradient descent on this same architecture starts from a rigid
functional form whose free parameters are the kernel weights; the hand weights
are already at the intended operating point of that form.
