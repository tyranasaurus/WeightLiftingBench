# Method

The network uses fixed color-opponent and texture filters to score broad scene cues such as sky, water, foliage, and pavement. Other branches measure coherent edges and coarse foreground contrast, and match Gaussian arrangements of parts such as wings, hulls, windshields, legs, eyes, and muzzles at several scales. A bounded, hand-written rule table combines scene and part evidence into the ten class scores.

`build.py` computes every weight from analytic formulas and manually chosen constants. It reads no data, stores no example images or descriptors, and performs no training or fitting. The final network achieved 112/600 (18.7%) on the fifth permitted training-set check; its held-out accuracy is unknown.
