# Method

**Architecture.** The network first normalises the image (`norm`). It applies 16 fixed 7×7 elongated first-derivative-of-Gaussian
filters (8 edge orientations over 180°, both signs) with valid padding, so the frame border makes no edge. Pairs are combined
into polarity-free |edge| maps with a hand-chosen soft clip (slope 0.3 above 1 image-std), then 2×2 average-pooled to
8×11×11 and normalised jointly (a reshape, then `norm`). A ninth channel holds the normalised intensity blurred by a fixed
8×8 Gaussian (stride 2) on the same 11×11 grid. A single convolution holds all 5,184 templates (11×11 kernels, padding 2)
and searches ±4 px shifts. Then come a max over shifts, a max over each letter's 216 templates (reshape + `maxpool`),
and 24 scores. Total: 5.65 M weights.

**Where the weights come from.** All filters come from formulas. Each template comes from one synthetic drawing. For every
letter I typed a 2-D hand model from general knowledge of the ASL handshape as seen from the front. Each model is made of
rounded-box palms, capsule fingers and thumb, a forearm entering from the bottom, and holes for O and F, with creases drawn
where a part overlaps the hand. Each letter gets 4 hand-chosen shape variants (finger spread or angle, thumb position,
curl). Each variant is rendered on a fixed grid of 3 sizes × 3 tilts (±10°) × 3 widths. Each drawing becomes a template
using only that drawing:
- its own edge maps (blurred slightly), with 0 inside its hand (real hands have creases and shading) and a penalty of
  0.25 × its own peak edge value on its background (walls are mostly smooth);
- plus, in the intensity channel, its own silhouette: +1 on the hand, −1 on the background, 0 on the forearm, since
  sleeves flip polarity. This is weighted 0.6 and given in both signs, because the hand is only usually lighter than
  the background.

There is no fitting, averaging or statistic over training or generated images. No numbers were taken from particular
training images. I looked at training images only as PNG grids, to learn what each letter looks like in general (e.g.
G/H/T point to the viewer's right, Q hangs down).

**Checks (all 5 used, full 27,455-image training set):** 23.5% → 15.0% → 16.8% → 27.0% → 29.1%. They showed "hub"
templates, chiefly Q, soaking up the raised-finger letters. That led to two design changes based on reasoning: the
neutral-interior/penalised-background template, and the silhouette channel. The submitted network is the one from check 5.
Fist letters (A, E, M, N, S, X) stay heavily confused with each other, and Q is still over-predicted.
