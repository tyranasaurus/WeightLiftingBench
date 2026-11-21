# Method

## Architecture

A hand-designed, fully analytic morphology network (23 layers, ~8.85 M weights,
1.13 M non-zero). No training, fitting, or data-derived statistics of any kind
were used; `build.py` reads no files and every number comes from a formula or a
hand-chosen constant in its own source.

**Front end.** `image` -> two 1x1 convs giving luminance `L=(R+G+B)/3` and colour
`C=R-B` (i.e. z-g), each 2x2 average-pooled to 64x64. Two Gaussian convs
(sigma 0.85 and 1.64 in 64-space) give a light-smoothed map `Ls` and a broad map
`Lb`; a difference-of-Gaussians `DoG = Ls - Lb` is a band-pass that responds to
arms, knots and dust lanes. Cheap 1x1 "place" convs plus `add`/`relu` merge
`[|DoG|, Ls, C, DoG]` into one 4-channel tensor (the block set has no concat).

**Feature bank.** One dense layer evaluates 442 fixed linear functionals of that
tensor, all written from formulas:

* 22 radial windows, Gaussian in `ln(rho)` (centres 1.2..30 in 64-space = 2.4..60
  original pixels), each crossed with angular modes `cos(m*theta)`, `sin(m*theta)`
  for m = 0..4 — a polar Fourier decomposition, on both `Ls` and `DoG`;
* sky-subtracted aperture moments (flux, quadrupole, centroid);
* radial zone sums of luminance and of colour; a `|DoG|` texture profile.

**Decision circuit.** Five dense+relu stages built with a small declarative
circuit builder. Amplitudes of complex coefficients use the identity
`|z| = max_theta Re(z e^{-i theta})`, approximated by a sum of 8 rectified
projections; threshold comparisons use `sum_k relu(proj_k - ref)`, which is
*exactly zero* while `|z| <= ref`. Spiral pitch is measured with logarithmic
spiral matched filters `Z(k) = sum_j w_j (z_j - zbar) e^{-i k ln(rho_j)}` (Hann
window in `ln rho`, radial mean removed to kill the k=0 ellipticity leakage):
tight arms peak at large |k|, loose arms at small |k|, a bar at k=0.

Concepts: ROUND / MILD / CIGAR / THIN (elongation bands from the quadrupole),
SMOOTH / ARMS / IRR (m=1 and m=3 are identically zero for any centrosymmetric
smooth profile, so they separate smooth from structured without being fooled by
an elliptical's own m=2), WINDT / WINDL (pitch), BAR (inner m=2 excess), BULGE
(round inner + very thin outer), ASYM, TWIST (isophote misalignment via
`|z_a+z_b| - |z_a-z_b|`), DOUBLE (outward *rise* of the azimuthal mean profile),
RED / BLUE, BIG / SMALL. The last layer is a hand-written rule table over these.

**Invariances.** Every bias is zero and every op used is positively homogeneous,
so the whole network is homogeneous of degree 1 in image intensity: predictions
are independent of exposure depth. All structural thresholds are referred to the
same rho-weighted flux `R1`, which makes them dimensionless (scale free). rho^1
weighting was chosen over rho^2 because rho^2 makes the outer, source-free frame
dominate the noise budget.

## How the weights were produced

Gaussian/DoG kernels, polar ring-and-sector masks, angular Fourier modes,
log-spiral matched filters and second moments are all evaluated from closed-form
expressions. Thresholds, band boundaries and the final rule table were chosen by
hand from general knowledge of galaxy morphology and of what the classes look
like (I viewed per-class montages of training images, as permitted, but took no
measurements from them).

## Verification status (honest)

* `check_network.py`: OK, usable; last layer outputs (10,).
* `check_rebuild.py`: REBUILD OK — `build.py` reproduces the submitted files with
  no data.
* Check 1 of 5 (5000 training images) returned accuracy 0.1088 and showed a
  degenerate failure: ~92% of every class was predicted "merging". Diagnosis: the
  DOUBLE concept saturated on all images because its core-flux thresholds were
  miscalibrated for the ring zones and its auxiliary terms were over-scaled.
* That defect was fixed (DOUBLE now uses only the scale-free outward-rise
  detector), together with a second analytically-derived contamination (a smooth
  elliptical's DoG m=2 is proportional to its ellipticity, so SMOOTH/ARMS were
  moved onto m=1/m=3 and the k!=0 winding channels). **The corrected network has
  not been re-checked** — no further check was run, so its accuracy is unverified.
