# Method

I wrote pen paths for common ways of drawing each digit using line segments and Bézier curves. `build.py` applies a fixed, stratified set of geometric variations in slant, width, curvature, pen thickness, and placement. It turns the resulting continuous stroke models directly into filter weights; it reads no input files and uses no training or fitting.

At inference, the network compares five views of the input with each stroke hypothesis: ink placement, coverage of the modeled strokes, local ridge orientations, enclosed or partly enclosed space, and contrast-normalized coarse shape. The class score combines the strongest match with the average of the best matches from four independently sampled banks of geometric variations. All operations are expressed in the supplied network format.
