# Method

This is a fully hand-engineered, rotation-tolerant morphology network; no optimization, fitting, image statistics, class averages, or image-derived weights were used. The first branch forms anti-aliased thresholded luminance/red/blue isophote maps. A second branch applies analytic Gaussian-derivative ridge/edge filters and difference-of-Gaussian knot filters at twelve evenly spaced orientations.

Large fixed convolutions then measure concentric aperture flux, concentration, colour gradients, oriented ellipticity, angular asymmetry, double-nucleus patterns, logarithmic-spiral correlations, and central/annular texture. A dense ReLU rule bank implements bounded threshold ramps for general astronomical priors: red concentrated systems are smooth or bulged; coherent thin profiles are edge-on; central elongation suggests a bar; blue annular texture and spiral correlations indicate spiral arms; odd moments, knots, and paired nuclei indicate disturbance or merging. The final dense layer combines these rules into the ten class scores with modest hand-selected class priors.

All numerical weights are generated from formulas and fixed constants in `build.py`. The submitted network has 35,530,575 stored weights (896,610 nonzero), and `check_rebuild.py` confirms an exact data-free reconstruction.
