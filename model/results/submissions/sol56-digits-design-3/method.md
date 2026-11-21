# Method

This is a fixed analytic nearest-shape network; no optimization, fitting, or training-image statistics are used. The first stage normalizes each image and applies a formula-generated Gaussian ink filter plus eight oriented Gaussian-derivative filters. ReLU and 2×2 average pooling produce a compact antialiased shape descriptor.

The reference bank is generated entirely inside `build.py` from generic handwritten pen paths for digits 0–9. A fixed-seed generator varies stroke width, aspect, slant, rotation, smooth bend, and common writing conventions such as open/closed 4s, crossed 7s, and one- or two-loop 8s. Each synthetic path is analytically rasterized and center-of-mass aligned. Dense weights are the resulting analytic descriptors, with biases equal to minus half their squared norms, so the layer computes nearest-reference squared-distance scores up to an input-only constant. A reshape and max-pool select the best generic shape for each class.

The only class offsets are small hand-chosen priors intended to balance generic prototype families. No weight or constant comes from a training image, label fit, class statistic, regression, clustering, or gradient descent.