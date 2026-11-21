# Analytic pen-path classifier

The network applies a fixed 5×5 binomial Gaussian convolution, spatial normalization, a dense bank of analytic digit models, and per-class maximum pooling to produce ten scores. Its 14,336 models per class are generated directly from handwritten line, Bezier, oval, and figure-eight formulas, with deterministic low-discrepancy variations in writing style, width, slant, curvature, stroke thickness, and placement. Each model is rasterized by distance to its pen path, centered using its own rendered ink mass, smoothed, and standardized; dense weights then compute normalized shape agreement, including both ink and empty space.

`build.py` reads no files and uses only Python's standard library and NumPy. Every weight comes from the typed geometric models and formulas: there is no gradient descent, synthetic-data training, fitted readout, class averaging, or copying of training images or descriptors. Training examples were viewed as raw PNGs and evaluated only through the official labeled checker; no measurements from individual training images were incorporated into the weights.

## Validation

Five official labeled checks were used. The final network achieved **84.72%** on 10,000 training examples (indices 3000 through 12999), separate from the 3,000 examples used for the earlier checks. This is a training-set diagnostic, not a measurement on the hidden scoring set. The network has **112,537,626 weight entries**, below the 120,000,000-entry limit; all structural checks pass. The isolated data-free rebuild check passed with **zero difference** between submitted and rebuilt weights and matching network JSON; the builder is 12,832 bytes.
