# Method

This is a data-free shape-template convolutional network. `build.py` constructs idealized ASL hands from analytic ellipses, capsules, rings, and hand-written crease segments based on general knowledge of each letter. It generates small rotation/aspect variants and both contrast polarities, then expresses correlation with those models through a bank of fixed impulse, Gaussian-derivative, and Laplacian filters followed by large convolutions. Spatial max pooling supplies translation tolerance, and exact pairwise `max(a,b) = relu(a-b)+b` trees select the best analytic pose for each class.

No training image, crop, pixel value, descriptor, class statistic, fitted parameter, optimization, or gradient descent is used to produce the weights. All numbers are generated from formulas and semantic hand geometry in the standalone deterministic `build.py` with a fixed random seed used only for analytic pose perturbations.
