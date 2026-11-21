# Method

**Architecture.** image → 5×5 Gaussian-derivative conv at 8 orientations (both polarities) → relu → 1×1 conv forming
unsigned oriented edge energy with cross-orientation inhibition (|r(θ)| − |r(θ+90°)|) → relu → joint contrast
normalisation over all orientations and positions (`norm`, clip outliers, `norm` again) → soft threshold / saturation
(two relus) → Gaussian pooling conv, stride 2, giving an 8×12×12 edge map → one dense layer whose rows are templates
(600 per letter) → reshape + maxpool taking the best template per letter → 24 scores.

**Where the weights come from.** Nothing is trained or fitted and no training image, patch or statistic enters the
weights. The filters are formulas. The templates are the same edge features computed on *rendered* hands: `build.py`
contains a small 3D hand model (palm ellipsoid, metacarpal / finger / thumb / forearm capsules with typed joint angles
for each of the 24 letters), an orthographic z-buffer renderer with simple shading, and a fixed-seed jitter of pose,
zoom, position and lighting. Each rendered feature vector is mean-centred and normalised, so the dense layer computes a
correlation with every template. Pose angles, zoom and placement per letter were typed from knowledge of the manual
alphabet and from looking at sample sheets of each class to see how the signs are generally framed (e.g. which way the
hand faces for G, P, Q, T); no coordinates or values were read off particular images.

**Checks used.** One of the five allowed runs was used (logged in `checks.log`): an earlier version of this network
scored 33.7 % on the training images. It showed the approach transfers but that the K templates attracted fists and
several finger-up letters, and that side-view letters (C, O, Q, T, X) were weak. In response I changed the framing
model (per-letter zoom with wide zoom / position coverage), several poses and the shading. **The submitted network
includes those changes and has not been run on real images**, so its real accuracy is unmeasured; on my own synthetic
cluttered scenes it scores about 60 %.

**Known limits.** The fist-like letters (A, E, M, N, S) are barely separable with this model, and real hands carry
interior creases and shadows the renders only approximate.
