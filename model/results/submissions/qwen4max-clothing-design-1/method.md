# Method

**Architecture** (62 layers, 367,777 weights, all hand-designed; no training, no
fitting, no training-image statistics used as weights).

1. **Silhouette front end.** The photo is blurred with three Gaussian kernels
   (sigma 0.9 / 1.6 / 2.6, from the closed-form Gaussian). Each scale is
   thresholded at `0.30 * (its own global blurred maximum)` — the backdrop is
   black, so the photo's own peak is the right reference — and clipped to [0,1]
   with a pair of ReLUs (`relu(10d) - relu(10d-1)`). This yields three soft
   occupancy masks. A separate `norm` layer feeds a texture bank.
2. **Texture bank.** Laplacian-of-Gaussian plus even Gabor filters at four
   orientations (wavelength 4) and two fine-scale oriented filters (wavelength 3,
   for a button placket's short vertical rhythm). Absolute values via
   `relu(f)+relu(-f)`, pooled to 7x7.
3. **Diagnostics.** ~80 scalar features, each a difference of mean occupancies
   over rectangles placed from general knowledge of garment layout: collar rows
   0-3, shoulders/sleeves rows 3-12, torso rows 12-25, hem/cuffs rows 25-28.
   Includes sleeve length (outer columns occupied low down), cuff gap below the
   body hem, trouser leg gap, dress flare, bag handle-hole, boot shaft asymmetry,
   sole line, and bbox height/width from saturated row/column profiles.
4. **Shape models.** 92 silhouettes rasterised from parametric geometry I typed
   from general knowledge: a T-shirt's collar + sleeve trapezoid + torso, a
   pullover's boxy block with cuffs, a coat's long trapezoid, an A-line skirt, two
   trouser legs with a crotch gap, a sole + upper curve, a shaft + foot + heel,
   and a bag body with a handle loop. Matched against the 14x14 occupancy grid
   with a Tversky-style overlap `[<T,obs> - 0.6<1-T,obs>]/<T,1>`, made
   left/right invariant by `max(a,b) = b + relu(a-b)`.
5. **Rule readout.** Features -> 116 hand-set thresholds and 0..1 ramps -> 36
   AND/sum combinations -> four linear heads summed with the shape evidence.
   Every threshold and weight was chosen by hand from reasoning about the
   classes; nothing was fitted.

**Checks used (5 of 5).** 26.8% -> 28.1% -> 31.4% -> 35.2% -> **36.4%**.
What each taught me: (1) thresholding at the image *mean* destroyed sparse
sandals; (2) a max-relative threshold alone did not fix it; (3) viewing
thresholded silhouettes revealed my row bands were wrong — the collar occupies
rows 0-3, so shoulders/sleeves sit at rows 3-12, and long sleeves are detected by
occupied *outer columns low down*, not by shoulder width; (4) a top/bottom-border
occupancy bonus I added for boots also fired on every tall garment; (5) removing
that bias and rebalancing the coat/shirt/pullover evidence gave the final 36.4%.

**Honest assessment.** This is well below what gradient descent would reach on
the same architecture. The remaining errors are concentrated where silhouette
alone is genuinely ambiguous: coat/pullover/shirt (all solid long-sleeved
trapezoids whose distinguishing placket, buttons and knit texture my hand-set
texture thresholds do not separate reliably), and boot/sneaker/bag. The
trouser class is also weak because a coat's front opening and a trouser's crotch
gap produce similar bottom-row signatures. Given more checks I would recalibrate
the texture ramps and the coat/shirt split first.

`build.py` is self-contained, reads no files, and reproduces both answer files
exactly (`check_rebuild.py`: max relative difference 0.0).
