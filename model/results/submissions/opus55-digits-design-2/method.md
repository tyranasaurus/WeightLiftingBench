# Method

**Idea.** The network is a nearest-template classifier. Its templates are digits I drew as typed pen paths, not
training images. An input is compared with every template in a fixed, formula-defined feature space, and each
class scores its best-matching template. No weight is trained, fitted or derived from any image statistic.

**Feature layers (all weights from formulas).**
1. *Oriented edges:* 7x7 derivative-of-Gaussian filters (sigma 1 px) in 8 directions, then ReLU. A 1x1 conv
   subtracts the two perpendicular directions, which sharpens the tuning to +-45 degrees, then ReLU, 2x2 max-pool
   and a 5x5 Gaussian blur, giving 8 x 14 x 14 maps. These are flattened and standardised as one vector with
   the `norm` layer.
2. *Holes:* the background is flood-filled from the frame border in 44 one-pixel, 4-connected steps (3x3 cross
   conv, ReLU, 1x1 conv, ReLU). The ink is carried in a second channel and ink >= 1/3 blocks the fill. Background
   the fill never reaches is a hole (the loops of 0, 6, 8, 9). The hole map is 2x2 average-pooled and blurred.
3. *Concavity codes:* 22 steps of one-pixel ray propagation (3x3 conv + ReLU, clamped with
   y - relu(y-1)) mark, for every pixel, whether ink lies to its left, right, above and below. A fuzzy AND
   (ReLU of a sum) turns these into 9 background codes: bays open to one side (left, right, up, down), enclosed,
   and open on two adjacent sides. They are pooled, blurred and standardised like the edges.

**Templates.** `build.py` renders 3,000 digits per class from stroke models I typed from general knowledge of
handwriting. There are 2-3 styles per digit, for example open, closed and cup-shaped 4s, 2s with or without a
bottom loop, flag-and-belly 5s, open and closed 0s and 9s, and 8s as one crossing stroke or two loops. Each
template gets a random slant, rotation, width, pen radius and small control-point jitter, from a fixed seed. Each
is then normalised as the task describes: fitted into a 20 x 20 box keeping its proportions, then shifted so its
centre of mass is at the frame centre. Every template is passed through the same fixed feature layers in numpy
and normalised by itself alone (zero mean, unit length). Its edge and code vectors become rows of two dense
layers. Its hole map h_t gives a third dense layer computing 2g h.h_t - g|h_t|^2, which equals
-g|h - h_t|^2 up to a per-input constant (g = 1.6).

**Readout.** The three dense outputs are added, so each template scores edge correlation + code correlation
(equal weight) + hole agreement. The scores are reshaped to (1, 10, 3000), and a max-pool takes the best
template of each class.

**How the numbers were chosen.** Filter scales, step counts, ink thresholds, the code definitions, the gains
and the stroke models were all set by hand from reasoning. They were not measured on generated images or
fitted to training images. The training images were only viewed, plus the 5 logged checks. Between checks I
used the confusion tables to decide which writing styles my stroke models were missing (for example asymmetric
and open-top 8s, larger 6 loops, open 0s), and to add the hole and concavity branches.

**Submitted version.** This is the version measured by check 5: 89.7% on a fixed 10,000-image training subset.
The earlier checks scored 86.7%, 89.0%, 89.5% and 90.0%. Check 5 showed that the open-0 style made 6s get read
as 0s. I meant to remove it, but that rebuild didn't finish in time, so the submitted network still includes it.
