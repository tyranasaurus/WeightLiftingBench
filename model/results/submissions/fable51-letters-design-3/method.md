# Method

**Architecture.** `norm(image)` → 2×2 average pooling to 14×14 → reshape → `norm` over the 196 values →
one dense layer with 18,000 outputs → reshape to (24, 750, 1) → maxpool over the 750 variants → 24 class scores.
Every dense output is the cosine similarity between the normalised 14×14 image and one normalised hand-silhouette
template; the class score is the best-matching variant.

**Weights.** No training data was used for any weight. For each of the 24 letters I typed a geometric hand model
from general knowledge of the ASL alphabet as seen from the front (right hand, thumb on the viewer's right):
a palm/fist ellipse, a wrist capsule leaving through the bottom edge (top edge for P and Q, which point down),
finger and thumb capsules at the angles the sign requires (e.g. index up for D, index and middle spread for V,
index and thumb horizontal to the right for G, little finger up-left and thumb right for Y), a ring for C and O,
a cut-out hole for D, F and O. `build.py` renders each model as a silhouette (hand = 1, background = 0) at
5 scales × 3 rotations × 25 shifts, runs the same `norm → avgpool` front end on the render, centres and
unit-normalises the result, and stores it and its negative (hand darker than the background) as rows of the dense
layer. All constants (filter sizes, variant grid, model coordinates) are typed in the code; the rebuild
check reproduces the weights from `build.py` alone.

**What was tried and learned (5 checks on training subsets).** Gabor-energy templates with intensity (19.6%),
polarity-invariant thresholded oriented energy with discriminative templates (14.9%), a corner-background hand
mask (10.9%) and a ridge-detector "distinctive parts" matcher (12.0%) all transferred poorly from the clean
renders to real photos, whose hands are heavily shaded and textured. Plain coarse silhouette matching in both
polarities was the most robust and is the submitted network (20.4% on a 4,008-image stratified training subset;
chance is 4.2%). Letters with distinctive gross shape (A, C, O, U, P, B, L, Y) are recognised best; the fist
letters (E, M, N, S, T) and letters whose only cue is a thin finger (D, I, X) are mostly confused.
