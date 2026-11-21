# Method

**Architecture.** A fixed edge front-end followed by a bank of drawn templates.

1. `norm` – each image to zero mean / unit variance (removes brightness and contrast).
2. `g` (conv 5x5, 16 ch) + relu – Gaussian-derivative filters (sigma 0.9 px) in 16 directions, from the formula.
3. `e` (conv 1x1) + relu – unsigned, sharpened orientation energy for 8 orientations
   (`|d_k| - 2|d_k+90°|`), with a noise floor and a ceiling so that faint and strong outlines count
   alike and it does not matter whether the hand is lighter or darker than the background.
4. `p` (conv 8x8, stride 2) – Gaussian pooling (sigma 1.5 px) to an 8 x 11 x 11 map, for tolerance to
   small misplacement.
5. `m` (conv 11x11, padding 1, 24,000 outputs) – 1,000 templates per letter, each tried at 9 placements;
   `maxpool` takes the best placement, a reshape + `maxpool` takes the best template of each letter,
   giving the 24 scores. 23.3 M weights in total.

**Where the weights come from.** Nothing is trained or fitted. `build.py` contains a small 3-D hand
model typed from general knowledge of hand anatomy (palm, three bones per finger, thumb, forearm, as
chains of spheres) and a table of the 24 ASL handshapes (joint angles per finger, thumb bone
directions, and how the sign is turned toward the viewer – e.g. C/O/T/X seen from the side, G/H
pointing sideways, P/Q pointing down). Each letter is drawn 1,000 times with seeded random variation
(finger relaxation, turn, lean, size in the frame, hand build, light direction) using a depth buffer and
simple shading. Each drawing is passed through the same front-end arithmetic and becomes one template,
normalised by itself alone (per-channel mean removed, unit length). The front-end filters and
thresholds are formulas and hand-chosen constants.

I looked at contact sheets of the training images to understand how each letter is held and framed
(hand size in the frame, lean, which side faces the camera), and revised the handshape table and
drawing style from those general impressions; no pixel values, coordinates or statistics of training
images are used anywhere.

**Checks on the 2,400 training images (all five used).**

| check | design | accuracy |
|---|---|---|
| 1 | first version (hands drawn too small) | 39.4 % |
| 2 | larger hands, uniform lean, softer drawings, added coarse pooling level | 36.4 % |
| 3 | strong finger creases, per-family lean, coarse level weighted 1/3 | 37.6 % |
| 4 | **submitted**: outline-dominant drawings, single pooling level, corrected size / lean / shapes | 39.0 % |
| 5 | check 4 + a figure-ground ("differs from backdrop") cue | 29.1 % |

The submitted network is exactly the check-4 build (check 5 was worse, so it was discarded).
Open-hand letters are recognised well (B, L about 87 %; F, Y, C, V 70–80 %); the fist-like letters
(A, E, M, N, S) and D are mostly missed, because their differences are small inner details that the
drawn templates do not reproduce reliably.

`python3 /app/check_rebuild.py /app/out` passes (max relative difference 0.0); the build takes about
1.5 minutes of CPU.
