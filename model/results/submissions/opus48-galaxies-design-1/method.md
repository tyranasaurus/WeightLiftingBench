# Method

A hand-wired feature classifier expressed as a small network. **No training, fitting, or
data-derived numbers**: every weight comes from an analytic formula or a reasoned rule
table. `build.py` reads no files and reproduces `network.json` + `weights.npz` exactly.

## Idea
Each galaxy sits at the image centre. From the contrast/background-normalised light
(`norm` of luminance, then `relu` so sky → 0) I measure a few rotation-, contrast- and
scale-robust quantities and classify with an explicit decision rule:

- **Colour** — smooth ellipticals are red, star-forming disks are blue. Measured as a
  homogeneous test `redness > θ·luminance` on the raw g/z bands over a central window.
- **Elongation ratio** `r = ANISO/Tr` of the core-clipped (isophotal) light.
  `Tr`, `A=Σ(x²−y²)·I`, `B=Σ2xy·I` are quadrupole moments from fixed templates;
  `ANISO = max_a(cos a·A + sin a·B) = √(A²+B²)` is formed rotation-invariantly with a
  bank of direction projections and a `maxpool` (so it is 0 for any round galaxy at any
  size). Thresholds come from ellipse geometry, `r=(1−q²)/(1+q²)`.
- **Size baseline** `Tr`, used so that every score term is proportional to galaxy size;
  hence the argmax is scale-free and depends only on `r` and the colour sign.

## Decision (relu "gates" + a scorecard)
Colour picks a red group {round-smooth 2, in-between 3, cigar 4, edge-with-bulge 9} or a
blue group {barred 5, tight 6, loose 7, edge-no-bulge 8}; being identical within a group
it cancels in the intra-group argmax. Within a group, an elongation staircase
`g_t = Tr·max(r−t,0)` is combined into triangular "bumps" so the class is routed by its
typical axis ratio: round → in-between → cigar/edge-on. The core-clip makes the moments
reflect the galaxy outline so thin disks read as elongated.

Classes 0 (disturbed) and 1 (merging) are left dormant: on real images no reliable
hand-built feature separated them from their colour group without stealing other classes,
so a clumpiness detector was removed rather than hurt overall accuracy.

## Verification
Filter behaviour (rotation invariance of elongation, colour sign, correct routing of
round/in-between/cigar/edge-on) was confirmed on self-generated synthetic shapes; no
number, threshold or design choice was taken from them. The five allowed label checks were
used only to confirm the network runs and to diagnose implementation choices (e.g. that
the clumpiness detector was unreliable), not to fit weights.
