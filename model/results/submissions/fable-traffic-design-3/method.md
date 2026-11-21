# Method

**Architecture** (8 layers, 16.5M weights): the image is reshaped to one channel and passed through `norm`, which removes the
global mean and divides by the global contrast of the whole photo (all three colours together, so colour relations survive).
A 3x3 conv then applies a small binomial blur and turns RGB into three channels: luminance, red opponent (R - (G+B)/2) and
blue opponent (B - (R+G)/2). A dense layer correlates this with 43 x 125 sign templates, and a maxpool over the 125 pose
variants of each class (5 sizes x 5 x 5 positions) gives the 43 class scores.

**Where the weights come from**: nothing is trained or fitted and no training image is used. `build.py` draws each of the 43
signs from general knowledge of German road signs (red ring with stroke-drawn digits, red-bordered triangles with simple
black pictograms, blue discs with white arrows, the stop octagon, the yield triangle, the yellow diamond, grey "end of" discs
with a diagonal band), rasterises each drawing with 3x supersampling at every size and position, passes it through the same
blur and colour formula, masks it to the sign's own outline (the background is unknown), removes the mean brightness inside
the mask and scales it to unit length. Each template is one row of the dense layer, so a score is a cosine-like match
between the photo and an ideal drawing of the sign at that pose.

**Status**: this is a first, simple version saved when time ran out. The sign proportions, colours and pose grid are
hand-chosen and were not tuned.
