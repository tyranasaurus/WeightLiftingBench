# Method

**Architecture.** A fixed edge front end followed by template matching, with no trained or fitted weights.

1. `norm` on the image, then a 3x3 derivative convolution at 8 orientations (both signs), `relu`, and a 1x1
   convolution + `relu` that turns these into 8 unsigned edge-orientation energies (each response falls to zero 45
   degrees from its own orientation). No padding, so the frame border adds no false edges.
2. 2x2 average pooling to a 13x13 grid, then a local contrast normaliser built from `dense` + `reshape` + `norm`:
   for every cell a group holding its 8 energies, 8 neighbourhood averages, a constant floor and all their negatives
   is normalised, which divides each energy by the local edge strength (with a floor so that noise stays small).
   A 3x3 binomial blur gives tolerance to small misalignment.
3. One branch per letter: a convolution whose kernels are that letter's 800 templates (tried at shifts of up to 4
   pixels each way through padding), a max over shifts, a max over templates, and a one-hot `dense` that places the
   result in the letter's output slot; the 24 branches are summed with `add`.

**Where the weights come from.** Filters and the normaliser are formulas. The templates come from a small 3D hand
model typed from general anatomy (palm, three segments per finger, thumb, forearm, built from spheres) that is posed
for each of the 24 letters with joint angles and viewing directions typed from knowledge of the ASL manual alphabet.
`build.py` draws each letter 800 times with a tiny renderer under varied view angle, size, place, joint jitter and
lighting (fixed seed; view angles and size follow evenly spreading sequences). Each drawing is pushed through the
fixed front end and becomes its own template: made zero-sum using its own background region and divided by its own
norm. No statistic is taken across drawings, and nothing was chosen by scoring on generated images.

**Use of the training samples.** The 24 samples were only looked at, to learn what the classes look like in general
(which way each sign faces the camera, that hands are chunky and fill the frame, that backgrounds can hold clutter).
No pixel values, coordinates or measurements from them are in the weights.

**Checks used: 2 of 5.**
- Check 1 (first version): 6/24 correct. It showed the pipeline works on real photos, but that edge-rich templates
  attracted unrelated letters.
- Check 2 (this submitted version, after chunkier anatomy, touching fingers, less edge-on side views, even view
  coverage and background-only centring of templates): 8/24 correct (33%; chance is 4%).

**Known limits.** Accuracy on 24 samples is a rough estimate. Look-alike groups are still confused (the fists
A/E/M/N/S/T, C vs O, P vs Q), and several finger letters (B, K, R, V) were drawn to F. A further revision aimed at
that (templates made zero-sum over their own silhouette) was written but never checked on real samples, so it is not
in this answer. This network is unlikely to match the same architecture trained on the full training set.
