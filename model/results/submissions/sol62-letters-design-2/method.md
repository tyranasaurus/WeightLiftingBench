# Method

The network removes global brightness and contrast, then uses analytic directional derivatives, two-sided strip filters, and a scanline-background residual to extract edges, finger-like ridges, and silhouette evidence. Each letter is modeled with a palm ellipse, a tapered wrist, and tapered finger and thumb segments arranged according to the ASL handshape. A fixed low-discrepancy sequence varies articulation, viewing angle, width, scale, and crop position to produce 816 hypotheses per letter.

The dense weights compare the input features with each procedural hypothesis. The score combines whole-hand agreement, three capped anatomical landmark votes, and a penalty for ridge evidence just outside the modeled silhouette. Max pooling selects the highest-scoring pose for each letter. All weights come from formulas and hand-authored anatomical geometry in `build.py`; the generator reads no data and performs no training or fitting.

The final labeled check was 75 correct out of 240 training-set images (31.25%). This is a diagnostic result, not an estimate of held-out accuracy.
