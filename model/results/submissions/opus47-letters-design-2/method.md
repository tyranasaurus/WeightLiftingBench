# Method

## Architecture

The network is a small oriented-edge (HOG-style) classifier:

```
image (1, 28, 28)
  -> norm                  per-channel mean/variance normalization (brightness-invariant)
  -> conv g_pos (4x3x3)    4 Sobel-like oriented gradient filters (V, H, \, /)
  -> relu r_pos
  -> conv g_neg (= -g_pos) negated filters to pick up the opposite polarity
  -> relu r_neg
  -> add                   r_pos + r_neg == |gradient| per orientation
  -> avgpool 7x7 stride 7  pool into a 4x4 grid of cells (4 cells x 4 orient per cell)
  -> reshape (64,)
  -> dense (24,)           hand-designed per-class weights
```

Total weights: 1,640 (well under the 120 M limit).

## How the weights were produced

All weights were written by hand from general knowledge of the ASL alphabet;
no training data, no fitting, no class averages, no statistics.

* **Gradient filters** (`g_pos.W`, `g_neg.W`): the four classical 3x3
  Sobel-like kernels detecting vertical, horizontal, `\`, and `/` edges.
  Their biases are zero.
* **Class weights** (`out.W`): for each of the 24 letters I wrote a 4x4x4
  tensor (`orient x row x col`) that lists the cells and orientations in
  which that sign is expected to show edge energy, derived from a mental
  picture of the hand shape (B = four vertical fingers; L = one vertical
  finger plus a horizontal thumb; G/H = horizontal pose; O = ring; etc.).
  The wrist row (bottom row of cells) is zeroed out across all classes
  because it is a constant feature (wrist/sleeve enters from the bottom in
  nearly every image) and so carries no discriminative information.
  Each class pattern is zero-mean, with magnitudes left in their hand-chosen
  scale (not L2 normalized; experiments showed that L2-normalizing lost
  specificity).

## Design considerations

The 3x3 Sobel-like filters have measurable cross-orientation response: a
pure vertical edge activates `V` at 1.0 but `\` and `/` at 0.76 relative.
The class patterns are therefore kept "orthogonal" — a letter whose
discriminative feature is vertical fingers uses only the V channel (not
`\` or `/`); a letter whose feature is diagonals uses `\` and `/` but not V
(sometimes with a small negative V to penalize pure verticals). This gives
cleaner separation between e.g. U (vertical fingers) and R / V (diagonals).

Where other letters have a feature the current letter lacks, a small
negative weight is added so that image patterns hinting at a different
sign are penalized.

## Caveats

ASL letters that are visually very similar (A/S/E/M/N/T/X are all closed
fists with subtle thumb-position variations) are inherently hard to
discriminate with a tiny hand-designed linear readout, and no attempt is
made to fit particular training images to improve them.
