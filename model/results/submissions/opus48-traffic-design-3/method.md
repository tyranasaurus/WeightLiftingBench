# Method

Hand-designed (no training) template-matching network for GTSRB 43-class signs.

## Architecture (network.json)
1. `norm` on the RGB image — per-channel zero-mean/unit-variance, giving lighting/contrast
   invariance (handles the dark/washed-out images) and encoding colour as a cross-channel pattern.
2. **Coarse branch** `cc`: a 32×32 conv (centre-only, pad 0) whose kernels are matched filters for
   7 sign *families* (red ring, red-fill=stop/no-entry, red triangle-up=warnings, red triangle-down=yield,
   yellow diamond, blue disc, grey/derestriction). Each family template is written directly in colour-opponent
   feature space (luminance, red-, blue-, green-opponent) with **positive evidence for the family and strong
   negative evidence against others** (e.g. ring requires a non-red interior and red at 3/9 o'clock; triangles
   use red at the correct corners and penalise the wrong corners; grey strongly penalises any colour).
   Two scale variants are max-pooled. Output: 7 family scores.
3. **Fine branch** `fc`: 32×32 convs (pad 2) matching per-class *interior* content (digits, pictograms,
   arrows) drawn from general knowledge, masked to the sign interior, mean-subtracted and unit-normalised.
   Two scale variants max-pooled → 43 fine scores. Empty signs (no-vehicles, end-of-all) get a negative bias
   so they are not default sinks.
4. Final score per class = 3·(coarse score of its family) + (its fine score), combined with a fixed
   family→class dense layer and an `add`.

## Weights (build.py)
All kernels are generated procedurally: signs are drawn at 4× supersampling from geometric primitives
(rings, discs, triangles, diamond, octagon, stroke-drawn digits and pictograms), converted through a fixed
colour-opponent transform, each template normalised by itself alone, and packed as conv/dense weights.
`build.py` reads no files and reproduces `weights.npz`/`network.json` exactly (rebuild check: 0.0 difference).
No training data, statistics, or fitting were used to set any number; the 5 allowed checks were used only
to verify the network runs and to find structural bugs (family leaks / sinks).
