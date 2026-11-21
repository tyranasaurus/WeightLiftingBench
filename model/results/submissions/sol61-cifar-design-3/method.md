# Hand-designed analytic network

This 32-layer network has 1,522,688 float32 weight entries. It combines contrast-normalized Gaussian-derivative edge channels with 720 analytic geometric object models, using max pooling for translation, scale, view, and contrast-polarity tolerance. The geometric models are drawn from ellipses, polygons, and line segments describing generic wings, wheels, hulls, faces, bodies, and legs; a separate branch implements piecewise-linear RGB color predicates, spatial pooling, and manually specified scene rules. The two branches produce ten class scores.

`build.py` contains all formulas, drawings, and manually chosen constants and reads no data or other files. No optimization, fitting, class statistics, image-derived descriptors, copied patches, or pretrained weights were used. Training images were viewed only as a class montage. One permitted labeled check on a fixed random 2,500-image training subset achieved 24.8% accuracy; the network was not changed after that check. This is not a measurement of hidden-test accuracy, nor evidence that it beats a gradient-trained network or the other comparator.

Reproduce the submission with `python3 build.py OUTPUT_DIRECTORY`.
