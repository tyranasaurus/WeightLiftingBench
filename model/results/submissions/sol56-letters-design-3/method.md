# Method

This is a fully analytic, data-free shape matcher rather than a fitted model. `build.py` rasterizes general ASL hand models from ellipses, capsules, curved paths, holes, and crease lines that I specified from knowledge of the alphabet; it also creates hand-chosen scale, aspect, and small-rotation variants. No training image, image measurement, class statistic, fitted parameter, or learned weight is read by the builder.

The network first instance-normalizes brightness and contrast. Four Gaussian-derivative filters, each represented with both signs and combined after ReLU, produce polarity-independent oriented edge magnitudes. A bank of convolution kernels matches normalized edge maps against the analytic models, and spatial max pooling provides translation tolerance. A second, coarse branch matches low-frequency signed silhouettes in both intensity polarities. Fixed dense matrices group the nuisance variants by letter, and small semantic-family calibration constants compensate for generic geometric score differences.

I used the permitted checks only to diagnose broad architectural behavior. The final weights remain formulas and hand-authored geometric models reproducible from `build.py` without data or external files.
