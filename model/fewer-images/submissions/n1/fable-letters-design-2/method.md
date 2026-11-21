# Method

**Architecture.** A fixed oriented-edge front end followed by a large bank of rendered hand templates and a voting head.

1. `norm` on the image (brightness/contrast invariance).
2. `conv` 5x5 with 16 derivative-of-Gaussian filters (8 orientations x 2 signs, from the formula), `relu`, then 1x1 convs that
   form |response|, sharpen the orientation tuning (|p_k| - tan(67.5 deg) |p_perp|, zero 22.5 deg off the preferred
   orientation), apply a dead zone and a two-step saturation (hand-chosen thresholds). The result is 8 polarity-free edge maps,
   so it does not matter whether the hand is lighter or darker than the background.
3. `avgpool` 4x4 stride 2 -> 8 x 11 x 11 cells, reshaped to one channel and `norm`-ed (z-score over the whole descriptor).
4. `dense` with 96,000 units: each row is one template, so each unit is the correlation of the image with that template.
5. `maxpool` over the 8 crops (scale/offset) of the same pose, `norm` across the 12,000 poses, then every pose that stands out
   (z > 1.8, extra weight above 2.6, thresholds chosen by hand) votes; `avgpool` over the 500 poses of each letter gives the 24 scores.

**Where the weights come from.** No training, fitting or statistics of any kind. `build.py` contains an articulated right-hand
model typed from general anatomy (palm, four fingers, thumb, skin web and forearm as tapered capsules with typical lengths and
widths) and the joint angles, thumb position and viewing direction of the 24 static ASL letters typed from knowledge of the manual
alphabet (e.g. fist with thumb at the side for A, C/O/D/X seen from the side, G/H pointing sideways, P/Q pointing down from an
arm entering on the other side). A small orthographic z-buffer renderer (Lambert shading plus darkening in creases) draws each
letter in 500 randomly perturbed poses (fixed seed) x 8 crops. Each render is passed through the same front end and becomes its
own template, standardised by itself alone; nothing is averaged or measured across renders, and no number was chosen by scoring
renders. The training images were only looked at to understand how each letter is framed and oriented.

**Checks used (1 of 5).** Check 1: 13 of the 24 training images correct (accuracy 0.54). It showed the pipeline works on real
photos; the errors are mainly among the fist-like letters (E, S, I, M, N, T), G/H read as Q, and K, P, X confused with
neighbouring shapes. No changes were made after this check; the submitted weights are the ones that were checked.

**Reproducibility.** `python3 build.py OUT_DIR` regenerates `network.json` and `weights.npz` from the code alone (about 2-3
minutes of CPU, no files read). The geometry uses only basic arithmetic (own sin/cos) so the renders do not depend on the machine.
