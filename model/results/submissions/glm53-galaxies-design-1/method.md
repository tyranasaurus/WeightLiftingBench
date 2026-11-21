# Method

The network is a hand-designed feature-detector pipeline. No weights were
trained, fit or copied from data: `build.py` writes every weight from
formulas (Gaussians, Gabors, geometric masks) and hand-chosen rule tables.
The five allowed training-set runs were used only to see that it works; all
numbers come from astrophysics reasoning about what the classes look like.

Pipeline:

1. **lum / L** — 1x1 convs mixing RGB into luminance plus two colour axes
   (b-r, g-mid), and a single-channel light image.
2. **Texture bank (gabf/gabc)** — 44 L2-normalised even Gabor filters
   (sigma = 1.5/3 px at full resolution, 6/12 px at half resolution, 11
   orientations each), per-channel `norm` (brightness invariance) + `relu`.
   Pooled three ways: frame means per (scale, orientation) feeding
   orientation-Fourier rows (cos/sin of 2 and 4 times the orientation angle =
   2-fold/4-fold texture anisotropy); per-scale orientation means; and means
   over the central 64x64 px only, where the galaxy lives (fine / arm-scale /
   coarse energy contrasts against the fine-scale noise floor).
3. **Elongation bank (axf/axc)** — 22 DC-free elongated-Gaussian matched
   filters (zero-sum kernels, so the grey sky gives exactly zero response),
   fine and coarse scales. Per position, the max and mean over the 11
   orientations are pooled; contrasting them detects thin anisotropic
   structure at any position angle. The max and mean maps are block-pooled to
   4x4 grids read by "bar" (central anisotropy, negative on outer blocks so
   lines and wide pairs do not fire) and "diskline" (a line of blocks through
   the centre, max-map minus mean-map) detectors, plus frame-wide max-vs-mean
   contrast ("thin").
4. **Centred geometry** — the galaxy is centred by construction, so
   `prof`/`fine`/`mom`/`skyc` are centred mask convolutions on 8px/4px grids:
   core/ring means, H/V/diagonal strips, quadrant and 8 azimuthal-sector
   means (12-48px ring), windowed light moments <Lx^2>, <Ly^2>, <Lxy>, <L>,
   <Lr^2>, and a corner-mean sky estimate. Every moment row subtracts the
   exact contribution a flat sky would make (build-time constants), so
   elongation (Dx/S = (1-q^2)/(1+q^2)), size and concentration are
   background- and brightness-free.
   A dedicated max-minus-min over the 8 sector means ("rough") measures
   azimuthal structure — spiral arms and lopsidedness — robustly against
   noise and radial profile.
5. **Hidden layer (80 units)** — thresholded detectors (relu of hand-weighted
   sums across descriptor groups, so a unit can require several conditions at
   once): three elongation tiers (axial ratio < 0.25 / 0.38 / 0.60, four sign
   combinations each for position-angle invariance, each with an absolute
   margin because sky noise scatters the quadrupole of faint galaxies), size
   and concentration units, strip/quadrant/sector asymmetry (disturbed,
   merging), off-centre bright blob (companions), profile-shape units (core
   vs ring, steepness, envelope), red-core / blue-disk colour units, bar and
   edge-on line units, and texture-energy contrasts.
6. **out** — a hand-written 10x80 rule table turns detector firings into
   class scores following the Galaxy Zoo decision tree: smooth round /
   in-between / cigar by axial ratio and compactness; edge-on with/without
   bulge by core concentration and colour along the line; barred/tight/loose
   spirals by azimuthal roughness, arm-scale texture, 2-fold anisotropy, blue
   outskirts and size; disturbed/merging by asymmetry, off-centre blobs and
   fine texture, with negative weights on symmetric arm patterns.

The mid-resolution split (fine filters at full res, coarse at half res)
keeps inference cheap (~9 ms/image in numpy).
