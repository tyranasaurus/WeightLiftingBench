# Fixed geometric digit recognizer

The network uses only hand-specified pen geometry and analytic image operations; no learning, optimization, class statistics, or training-image-derived weights are used. Its 25,200 analytic proposals come from typed Bézier paths and oval formulas for the ten digits, with fixed grids of widths, heights, slants, and writing styles. These are generated mathematical shape models, not stored examples from the dataset.

ReLU gates first softly binarize the ink. Repeated max pooling creates multiscale proximity fields, and a fixed background flood-fill circuit supplies an enclosed-background feature. Three banks of full-frame convolutions score stroke presence, unexplained ink, and tolerant agreement of enclosed space. Max pooling chooses the strongest proposal and one-pixel translation for each digit. The output is ten class scores.

All 59,346,220 float32 weights, including zero entries and biases, are generated directly by `build.py`, which reads no files and needs only NumPy and the Python standard library. The network is below the 120,000,000-weight limit.

Exactly five labeled checks were made, each on the same first 3,000 official training images. The final network achieved 82.80% on that subset. Check results were used for manual structural revisions, not automated parameter searches or fitting. This is a training-subset sanity check, not a claim about unseen-test accuracy or the two comparison systems.
