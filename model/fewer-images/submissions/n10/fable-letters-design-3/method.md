# Method

**Idea.** The network is a template matcher whose templates are drawings I typed from general knowledge of the
ASL hand shapes; nothing in it is fitted to, copied from, or measured on the training images.

**Architecture** (`network.json`, 111.7M weights, all produced by `build.py` with no input files):

1. *Front-end, formulas only.* `norm` of the image → 5×5 derivative-of-Gaussian filters at 6 orientations (both
   signs) → `relu` → 1×1 conv that forms unsigned, sharpened orientation maps `|a_k| − 1.73·|a_perp| − dead zone`
   → `relu` → clip (`x − relu(x − c)` with an `add`) → Gaussian pooling convs at two resolutions (12×12 and 6×6
   cells per orientation) plus one orientation-free edge-density map → diagonal `dense` layers that down-weight the
   bottom rows (wrist / sleeve) → `norm`. Thresholds are hand-chosen in units of "edge step / image standard
   deviation".
2. *Two template banks.* Each bank is a set of `dense` layers with 24 × 2000 rows; every row is one drawing passed
   through the same front-end formulas and normalised by itself alone, so a row's output is the correlation between
   the photo's edge maps and that drawing's edge maps. The outputs are added, reshaped to (1, 24, 2000) and
   `maxpool`ed: a letter's score is its best-matching drawing. The two banks' 24 scores are added.

**Where the drawings come from.** `build.py` contains a small 2-D renderer (capsules, rounded boxes, ellipses,
anti-aliased, outlines slightly darkened) and, for each letter, a parametric description of the hand as a viewer
sees it: fist/palm block, finger capsules at given angles, thumb, forearm. Each drawing gets random articulation
jitter and a random affine pose (fixed seed). Bank 1 uses plain drawings; bank 2 redraws the same letters with a
tapering wrist, leaning fists for M/N/S, dark creases between folded fingers and shadows under fingertips/thumbs.
I looked at the 240 sample photos to learn how each letter is photographed (orientation, rough size, which side
the thumb is on) and described that in the shape models; no pixel values, coordinates or outlines were taken from
particular images.

**Checks (all 5 used, 240 sample photos each).**

| check | variant | accuracy |
|---|---|---|
| 1 | plain drawings, edge maps only | 44.6% |
| 2 | tapered wrists, leaning fists | 40.0% |
| 3 | + creases/shadows, + edge-density term | 40.4% |
| 4 | + hand-vs-background region branch | 35.8% |
| 5 | **final: bank 1 (as in check 1) + bank 2 (drawings of check 3), scores added** | **45.4%** |

The region branch of check 4 was removed. Remaining weaknesses, visible in every check: the fist-like letters
(A, E, M, N, S) and U/R are rarely told apart and tend to be absorbed by the O and K drawings; Q is poor. The
sample is small, so the differences between checks 1, 2, 3 and 5 are within noise; expect roughly 40–45% accuracy
(score ≈ 0.4) on the held-out images.

Generated drawings were only used as individual templates and for "does it run" sanity checks; no number or
setting was chosen from runs on generated images.
