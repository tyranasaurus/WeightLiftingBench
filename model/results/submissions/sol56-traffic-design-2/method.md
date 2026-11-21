# Method

This is a data-free, hand-designed template network. A small fixed 1×1 front end computes luminance and red/blue/yellow opponent coordinates, applies per-image channel normalization, and implements clipped threshold responses with pairs of ReLUs. The resulting five semantic maps are averaged to 16×16.

The classifier weights are generated analytically from idealized German traffic signs. `build.py` draws circles, rings, triangles, diamonds, an octagon, DIN-like stroke digits, arrows, vehicles, warning pictograms, and cancellation bars using equations and hand-chosen constants based on general knowledge of the sign classes. It supersamples these geometric models and turns them into zero-mean matched filters. For robustness, each class has 19 hand-specified variants spanning scale, small translations, and mild aspect distortion; max pooling selects the best variant.

No training image, crop, pixel value, fitted statistic, optimization, regression, gradient descent, or learned parameter was used to produce any weight. The training data was used only for visual inspection and the five permitted end-to-end checks. The final files are reconstructed deterministically by `build.py` without reading any file.
