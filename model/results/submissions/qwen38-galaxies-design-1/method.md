# Method

**Architecture.** A hand-designed, fully analytic feature network (no training, no fitting):
luminance (0.30 R + 0.55 G + 0.15 B) and colour (B−R) are extracted with 1×1 convs; the
luminance is downsampled (avgpool) to 64² and 32² and sky-subtracted with a wide negative
Gaussian. A filter bank of formula-built kernels is applied at the three scales: Gaussians
(scale space), annulus means (ring flux / radial band-pass), sector and sector-pair sums
with a central hole (azimuthal structure), cos 2φ-weighted annulus triplets (m=2 ridge
projection for spiral arms and bars), and second-moment quadrupole kernels
((x²−y²)·G, 2xy·G, r²·G) for elongation. Each shape/arm/asymmetry question is posed as a
*relu ratio comparator* (a·F1 − b·F2 with zero bias, or zero-sum kernels), so its response
scales linearly with galaxy brightness and the comparisons are brightness-invariant.
Comparator maps are block-pooled (maxpool; tiny centre windows for centre-anchored
quantities, wider rings for arms/companions/stars; avgpool for signed colour and flux
maps). One dense layer per pooled map implements hand-written block masks that sum the
blocks into ~22 physical measurements (flux, size, elongation ladder el1–el4, core
concentration, bulge dominance, bar m=2, arm m=2 inner/outer, colour, m=1 asymmetry,
off-centre clumps, field stars); their sum is relu'd and a final dense layer is a
hand-written rule table giving the 10 class scores. All readout biases are 0 and every
term scales with flux, so the argmax is brightness-invariant; a small flux-proportional
per-class prior breaks ties for uninformative (very faint) images.

**How the weights were produced.** Every weight is generated in `build.py` from closed-form
expressions (Gaussians, anisotropic Gaussians, annuli, sectors, moments) and hand-picked
constants (ratio thresholds, block masks, rule coefficients). Nothing is fitted: no
training-set statistic, image, crop or descriptor enters any weight. The design was
developed by (a) viewing training images per class as PNGs to understand what the classes
look like (colour, thinness, arms, bars, bulges, companions), and (b) unit-testing the
filter bank on *formula-generated phantom galaxies* (Sérsic bulges, exponential discs,
log-spiral arms, bars, thin lines, pairs — typed from general knowledge, with sky and
noise), printing the resulting measurements so thresholds and rule coefficients could be
chosen by hand. Real training images were only ever run through the network via the
allowed logged checks (all 5 used); the confusion tables from those checks drove the
hand rebalancing of the rule table (e.g. replacing a centre-sensitive sector elongation
detector with the offset-robust quadrupole-moment comparator after check 2 showed it
misfired on real cutouts, and damping the bulge/clump terms after checks 3-5 showed
their real-data scale exceeds the phantom predictions). The submitted weights are the
final hand-rebalanced rule table; its last adjustment was made blind after check 5.
