# Analytic person-presence network

The network uses a four-level image pyramid, a piecewise-linear skin-colour membership rule, and standardized luminance. Gaussian regional comparisons describe an idealized face (separate eyes, cheeks, nose bridge, mouth, and chin) and a head above bilateral shoulders and a filled torso, with several generic scales, widths, contrast polarities, and small rotations; a stricter low-saturation mode uses a bright head over dark clothing. ReLU hinge penalties implement conjunctions of these regional rules, and exact maximum circuits select candidate locations, templates, and scales before producing the two class scores.

All weights are deterministic formulas or manually specified general anatomical/colour constants in `build.py`; the script reads no files and uses no randomness, pretrained weights, optimization, image-derived templates, fitted statistics, or stored examples. Five labeled checks were used to diagnose overly permissive rules and assess structural changes; no network or partial network was evaluated on images outside the prescribed runner. The final architecture has 451 layers and 631,974 float32 weight entries, including biases.

## Checks

| Check | Examples | Accuracy | Observation |
|---|---:|---:|---|
| 1 | 1,000 balanced | 49.4% | Permissive local cues predicted person almost everywhere. |
| 2 | Same 1,000 | 51.6% | More complete local anatomy reduced false positives, but remained weak. |
| 3 | Same 1,000 | 58.3% | Co-located head/shoulder/torso context improved discrimination. |
| 4 | Same 1,000 | 53.7% | Additional full-figure/portrait branches did not offset losses from scale restrictions. |
| 5, final | 4,000 other balanced examples | 56.0% | Restored all-scale context with rotations, a neck cue, and a low-saturation mode; false positives remain substantial. |

The final check's confusion matrix (rows true class, columns predicted class) was `[[808, 1192], [568, 1432]]`. These are training-sample checks, not held-out scoring results, and they do not establish superiority over either comparison baseline. The weights were left unchanged after the fifth check.
