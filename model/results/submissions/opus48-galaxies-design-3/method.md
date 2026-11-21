# Method

A hand-designed morphological classifier. Every weight comes from a formula or a
hand-chosen constant grounded in general knowledge of galaxy morphology; no training,
fitting, or statistic of the training images/labels is turned into a weight. The training
set was used only to *look at* the classes and to confirm that the standard morphology
statistics behave as expected (and for the 5 allowed network checks).

## Architecture (12 layers)
1. `pre` (1×1 conv, 3→2): luminance `L=(R+G+B)/3` and colour `R−B`.
2. `zn` (norm): per-channel standardisation → `Zl`, `Zc`. This removes each image's
   brightness/contrast scale, so all downstream features are scale-free without any division.
3. `feat` (15×15 conv, 2→20): a fixed filter bank —
   - 8 zero-mean **elongated Gaussian (oriented)** filters at 0…157° and their negatives,
     applied to `Zl` (their rectified responses give oriented energy);
   - centre taps giving `Zl` (brightness), `Zl−1` (bright-area/extent), `+Zc`, `−Zc` (colour).
4. `fr` (relu): rectifies responses; `relu(k)+relu(−k)=|k|` recovers oriented energy, and
   `relu(Zl−1)` gives bright-pixel area.
5. `fp` (avgpool 16×16): pools to an 8×8 grid; its central 2×2 cells are the galaxy centre,
   the surrounding ring is the disk.
6. `sfeat` (dense → 48): fixed linear read-outs forming standard morphology statistics:
   the **2nd angular harmonic** `C2,S2` of the centre oriented energy (rotation-invariant
   **elongation** `|C2|+|S2|`), total oriented energy, central & ring brightness
   (**concentration**), bright-area extent, central/ring **colour**, and **rotational
   asymmetry** (centre minus its 180°-rotation over the central 6×6 cells).
7. `sfr` (relu): turns the signed harmonic/colour read-outs into non-negative evidence
   (e.g. `|C2|=relu(C2)+relu(−C2)`).
8–11. Head: `lin` (linear score part) + `hid`→`hr`(relu)→`gat` (gated/AND-style part),
    combined by `out` (add). The head encodes hand-written rules about each class.

## Classification rules (physical reasoning)
- **Elongation** (standard axis-ratio idea, via oriented-energy harmonic) isolates the thin
  **edge-on** galaxies (8,9) and the **cigar** (4); within these, redness + concentration
  mark the bulge (→9), ring extent marks the thin bulgeless disk (→8).
- Among non-elongated galaxies, **colour** (`norm(R−B)`) separates red **smooth ellipticals**
  (2,3) from blue **spirals** (5,6,7); low oriented energy + bright compact core marks the
  tight spiral (6), bluest marks the loose spiral (7), redder/neutral the barred spiral (5).
- The smooth ellipticals are split by **elongation**: round (→2) vs in-between (→3).
- **Rotational asymmetry** marks **mergers** (1); blue + asymmetric marks **disturbed** (0).

All filter kernels are closed-form (oriented Gaussians, coordinate/ring masks, cos/sin of 2θ),
and all head coefficients/thresholds are hand-set in a standardised feature space whose
centre/scale constants come from the general ranges of these morphology statistics.

`build.py` constructs `network.json` and `weights.npz` deterministically from these formulas
and constants, reading no files.
