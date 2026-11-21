# Method

## Architecture
A fully hand-designed, formula-driven network (no training, no data-derived numbers):

1. **Normalisation / colour**: per-channel `norm` of the image (removes sky offset and
   per-image gain), then 1x1 convs for luminance `(r+g+b)/3` and redness `(r-b)/2`.
2. **Gaussian / DoG pyramid**: luminance smoothed (sigma~1.1) and re-normalised so all
   downstream features live in per-image contrast units; four further levels
   (sigma ~ 2.8, 5.6, 11, 22 px) built from separable binomial smoothing + 2x2 avgpool.
   At every level: derivative-of-Gaussian pair -> 1x1 mixes into 4 signed directional
   derivatives -> rectified orientation-anisotropy channels (`|D_perp|-|D_par|` for
   0/45/90/135 deg), and a difference-of-Gaussian band-pass (rectified) as blob/texture
   energy at that scale.
3. **Regional pooling**: every map is pooled into 16x16 / 8x8 / 4x4 / 2x2 grids
   (mean and max), placed as channels of one fixed "canvas" tensor via 1x1 slot convs
   with zero padding, then flattened.
4. **Rule network** (3 dense layers, all weights typed from general knowledge):
   - dense1 = ~128 *semantic features*: flux/profile (central vs ring luminance),
     ring/annulus texture at three scales (spiral arms, star-forming knots),
     signed orientation-anisotropy sums (elongation / thin-line evidence),
     second-moment descriptors of a star-gated body map, central mid-band-pass excess
     (bulge), 180-deg pair differences of the coarse luminance grid (asymmetry),
     off-centre coarse band-pass peaks (merger companions), diagonal-imbalance of
     central band-pass peaks (close double nuclei), coarse band-pass ring (tidal
     tails), fine-vs-mid band-pass peak ratio (stars), orthogonal-anisotropy minimum
     (diffraction spikes), ring redness (red sequence vs blue star-forming).
   - dense2 = ~80 *concepts*: SMOOTH, spiral-texture (total / near / far / knots),
     elongation cascade (EL2 < ELbig < ELvbig), thin-line (EOraw2), roundness,
     bulge, asymmetry, merger count, tails, red/blue, star/spike suppression.
   - dense3 = class scores: a hand-written priority table (smooth round /
     intermediate / cigar by elongation cascade and redness; spirals by arm texture,
     arm winding radius and blue ring; edge-on with/without bulge by strong
     anisotropy plus/minus central bulge; disturbed by asymmetry+tails; merging by
     multiple extended nuclei).

## How the weights were produced
Every number comes from closed-form formulas (Gaussians, Gaussian derivatives, DoG,
orientation mixes, geometric pooling masks, moment weights) and a hand-written rule
table with hand-chosen thresholds and coefficients. `build.py` contains all of it and
reads no files. Threshold magnitudes were sanity-reasoned against responses of the
same formulas to synthetic galaxy templates typed from general knowledge (Sersic
spheroids, exponential discs, log-spiral arms, bars, tidal arcs, PSF stars) evaluated
by an independent analysis script in `/app/work` (no training images involved).

## Verification (honest report)
`check_network.py`: usable, 2.28 M weights. `check_rebuild.py`: exact rebuild.
Five logged checks on training subsets (3000 stratified images each) measured
accuracy 0.101, 0.096, 0.093, 0.097, 0.096 for successive revisions: the hand-set
thresholds of the texture and anisotropy families proved much too tight for the real
DESI images (real galaxies are smaller/fainter relative to the normalised noise floor
than my synthetic templates), so those revisions stayed near chance level; the
confusion tables did show the signed-anisotropy elongation family responding
monotonically to true elongation (edge-on and cigar rows shifting with it). The
submitted table is the last checked configuration plus a final elongation cascade
(ELvbig routing the strongest anisotropy to the edge-on classes), which was not
re-checked because the 5-run budget was exhausted. No gradient descent, no fitting,
no training-image statistics entered any weight.
