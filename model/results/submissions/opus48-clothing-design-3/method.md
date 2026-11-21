# Method

A hand-designed, training-free classifier. No gradient descent, no fitting, no
statistic of the training set is ever turned into a weight: every number in
`build.py` is a round threshold or archetype coordinate chosen from general
knowledge of how these ten kinds of item look. The training set was used only to
*look at* (save PNGs, print region-average summaries) and for the 5 allowed checks.

## Architecture (all weights set analytically in `build.py`)
1. **`feat`** — a `dense` layer whose rows are fixed region-averaging masks. It reads
   ~10 generic geometric features of a centred garment photo: how far up it reaches
   (`TOPROW`), whether it spans the full image width (`WIDE`), where the sleeves lie
   (`SIDE2`, `SIDELO`, `SIDEHI`), the two trouser legs vs. the dark central gap
   (`LEGS`-`CEN`), how far it reaches the bottom (`BOTRCH`), and how solid it is in
   the middle/lower band (`MID`, `LOWMID`, `TOTAL`).
2. **`vpre`+`viol` (relu)** — one unit per constraint, computing `relu` of how far a
   feature is on the wrong side of a hand-chosen threshold (a "violation").
3. **`smain`** — each class score is minus the sum of its own violations, so an image
   is assigned to the class whose generic description it best fits (a conjunctive
   "box" per class). Thresholds first split garments / footwear-&-bags, then split
   within each group (trouser gap, sleeve length, narrow dress, sandal sparsity,
   compact sneaker, solid bag, tall boot).
4. **symmetry path (`fdiff`→`fdr`→`aviolp`→`aviol`→`sasym`)** — computes left–right
   asymmetry `mean|x - fliplr(x)|`; ankle boots point sideways (asymmetric), bags are
   symmetric, which cleanly separates that pair.
5. **`sproto`** — pullover, coat and shirt share one near-identical silhouette, so
   they are given identical membership rules and split by a zero-sum nearest-archetype
   linear discriminant over (MID, BOTRCH, TOTAL, SIDE2); the archetypes encode that a
   coat is the most solid/long/heavy, a pullover the shortest with the bulkiest
   sleeves, and a shirt the thinnest and dimmest.
6. **`out`** — elementwise sum of the three score paths; the highest score is the
   prediction.

## Notes
`build.py` reads no files and reproduces `network.json` and `weights.npz` exactly
(verified with `check_rebuild.py`). Measured accuracy on a balanced 10k training
sample was 0.61 (check 5). The design relies on contrast-robust region averages and
left–right symmetry, which are properties of the item classes in general rather than
of any particular image, so it should transfer to the held-out set.
