# Method

**Idea.** The network is a template matcher whose templates are drawings made by `build.py` from a hand model, not
from data. A small 3-D articulated right hand (capsule fingers and thumb, slab palm and forearm, typed from general
anatomy) is posed for each of the 24 letters from knowledge of the ASL manual alphabet and of how each sign is usually
held towards the viewer (for example C, O, X, T side-on; G, H as the back of a hand pointing sideways). Each letter's
pose is drawn 1,200 times with random changes of joint angles, viewing direction, size, light and hand/background
brightness (fixed seed). Every drawing becomes one template on its own, standardised by itself alone.

**Architecture.**
1. `filt` (5x5 conv, no padding): first derivatives at 12 directions and second derivatives at 6 orientations
   (Gaussian formulas), then relu.
2. `ori` (1x1 conv + relu): 6 edge-orientation maps (contrast sign ignored, sharpened as |along| - |across|) and
   6 maps of thin dark lines (creases between fingers; a dark line is vetoed where it is just the side of a step edge).
3. Contrast normalisation: the maps and their negatives are reshaped into one channel and passed through `norm`,
   which divides by the image's RMS edge strength; a dead zone and a saturation (two relus) follow.
4. `cells`: 4x4 average pooling with stride 2, giving 12 x 11 x 11 features.
5. For each letter, a conv layer holds its 1,200 templates (kernel 11x11, padding 2, so each template is tried at
   25 shifts of up to 4 pixels); maxpool over shifts and over templates gives the letter's score; one-hot dense
   layers and an `add` assemble the 24 scores.

Total 41.8 M weights. All constants (filter scale, thresholds, pose angles, variation ranges) were chosen by hand.
No training, fitting, or statistics of training or generated images are used; `build.py` reads no files.

**Checks on the 2,400 training images (4 of the 5 allowed were used; see `checks.log`).**
46.4 %, 45.2 %, 48.5 %, 48.5 %. The saved answer is the network of the fourth check (48.5 %, score about 0.46 after
the chance correction). Between checks I compared my drawings with photos by eye and changed the framing, the
shading, the crease features and several poses.

**Known weaknesses (from the last check).** The fist letters are mostly confused with each other and pulled towards
A (S 0 %, E 11 %, N 14 %); R and U are mostly read as D or K (17 %, 20 %); Q is at 4 % because my last re-posing of it
was worse than the first one (49 %); O and X are at 24 % and 19 %. I had planned a final revision (wider range of hand
turning for the front-on letters, the original Q pose, corrected thumbs for S and E) but did not build or test it, so
it is not in this answer.

**Resources.** `build.py` takes about 1.5 minutes of CPU. Running one batch of 500 images peaked at about 2 GB of
memory here.
