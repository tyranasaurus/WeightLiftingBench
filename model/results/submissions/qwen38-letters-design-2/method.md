# Method

## Architecture
`norm -> conv(24 oriented filters) -> relu -> clip -> maxpool 2x2 -> {avgpool 3x3 s1, maxpool 3x3 s1} -> conv(216 templates) + conv(absence penalties) + conv(amount penalty) -> add -> maxpool 3x3 s3 -> reshape -> dense(24)`.

* `n0` norm removes per-image brightness/contrast variation.
* `c1` = 24 filters from formulas only: 6 edge orientations x {1st derivative of Gaussian (contour), 2nd derivative of Gaussian sigma=1.3 (finger-width ridge)} x 2 polarities.
* `cl1/cl2/cl3` clip the rectified energy at TAU=4.0 (hand-set) so extreme edges (sleeves) cannot dominate; `p1` maxpool 2x2 and `s1` avgpool 3x3 give a smoothed 12x12 presence map `b`; `d1` = 3x3 maxpool of `p1` is the same map dilated by +-1 cell (+-2 px).
* `c2/c2n/c3` + `p2` implement a bidirectional chamfer match per class per variant:
  score = mean over the template's presence cells P of the image's dilated contour/ridge presence
  - GAMMA_C * mean image energy over the template's absence cells Q (contour channels)
  - GAMMA_R * same for ridge channels - GAMMA2 * mean window energy;
  max over the 3x3 shift positions gives translation tolerance, and `fc` averages 9 drawn variants per class (rotations -6/0/+6 deg x scales 0.93/1.0/1.07) for pose tolerance.

## Where the weights come from
No training data, no fitting, no gradient descent. Each of the 24x9 templates is derived from one synthetic 28x28 hand drawing that I typed from general knowledge of the ASL alphabet and of the viewing convention of this data (thumb side at viewer's right, G/H pointing right, P down-left, Q down): a palm ellipse plus finger/thumb capsules at typed angles, folded-finger bumps, crease lines, wrist stub and sleeve. Each drawing is passed through the same formula filters as the image; from its own energy map alone (self-normalised by its own peak) it yields its presence mask P (energy > 0.45*own peak), absence mask Q (energy < 0.12*own peak, away from P and from the window border) and its own normalising counts; no statistic is ever taken across drawings. All other numbers (TAU, P_REL, Q_REL, GAMMA_C, GAMMA_R, W_RIDGE, GAMMA2, filter sigmas, variant angles/scales) are hand-set constants.

## Checks
Five allowed training runs were used while developing the matcher (10.0%, 12.3%, 8.5%, 12.2%, 3.3%); their confusion tables showed that additive coverage scores collapse unless the presence map keeps amplitude information and absence penalties are strong. Those runs all used a clip value that saturated the contour map into shapeless blobs; after fixing that calibration bug (TAU 1.2 -> 4.0) the final network was verified only mechanically, on its own synthetic drawings (19/24 self-matches, remaining confusions M/N/O/S, I/Y, R/U being genuine shape similarities), since the training-check budget was exhausted. `build.py` regenerates both files from code alone (verified with check_rebuild.py).
