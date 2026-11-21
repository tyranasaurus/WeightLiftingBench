# Method

## Architecture

A translation-searching template matcher:

```
image (1, 28, 28)
  -> norm                                                      (1, 28, 28)
  -> conv (24 out, kernel 20x20, pad 0, stride 1)              (24, 9, 9)
  -> maxpool size 9x9 stride 9x9                               (24, 1, 1)
  -> reshape                                                   (24,)
```

The 24 conv filters are hand-designed 20x20 silhouette templates, one per ASL
letter (A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y).
The maxpool at the end is only used to let each template search for its best
alignment within the (1, 28, 28) input — the hand usually is not perfectly
centred, and a 9x9 search window gives roughly ±4 pixels of translation freedom.

## Where the weights come from

`build.py` reads no files. Every weight is produced by Python code that draws
geometric primitives (ellipses, disks, rectangles, line segments, rings) in a
20x20 canvas, following my knowledge of what each ASL hand sign looks like:

- fist classes (A, E, M, N, S, T) draw a round palm with wrist and small bumps
  for the folded fingers / thumb,
- extended-finger classes (B, D, I, U, V, W) draw a palm plus thin finger
  rectangles at the top,
- sideways classes (G, H) draw a fist on the left with fingers extending right,
- downward classes (P, Q) draw a fist on top with fingers extending down,
- shape classes (C, O) draw arcs and rings,
- L and Y draw explicit L / hang-loose shapes.

Within each template the three per-cell values have meanings tied to the
`norm` layer upstream, which centres the image at 0 mean (hand pixels become
negative because the hand is darker than the background in nearly all
training samples):

- -1  the sign's hand is expected here,
- +1  this position is expected to be background for this sign AND is a
      competing sign's hand position (so an "anti" feature),
-  0  don't care — no commitment.

The explicit +1 anti-regions are the main way templates discriminate
between classes whose main hand silhouettes overlap (e.g., the fist
family vs. B).

After drawing, each template is normalised using only its own mean and its
own L2 norm; no statistic across templates is used, no training image is
read.

## Verification

- `python3 /app/check_network.py /app/out` confirms the shapes and that the
  total weight count (9,624) is well under the 120,000,000 limit.
- `python3 /app/check_rebuild.py /app/out` confirms `build.py` reproduces
  `weights.npz` and `network.json` without any data access.
- Five logged training-subset checks were used to confirm the network
  behaves sensibly (accuracies in `/app/out/checks.log`); templates were
  then hand-edited based on my understanding of the data, not on any
  measurement of the network.
