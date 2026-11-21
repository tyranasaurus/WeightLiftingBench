# Method

**Architecture.** Fixed feature layers, then one dense layer of templates, then a vote.

1. `c1`: 24 Gaussian-derivative filters (5x5, sigma 1), written from formulas: first derivatives in 12 directions
   (edges) and second derivatives across 6 orientations with both signs (thin lines).
2. `n1`: the edge group and the line group are each divided by their own root-mean-square over the picture
   (reshape + norm), so the features do not depend on brightness or contrast.
3. `s1`, `u1`, `u2`: 1x1 convs and ReLUs that sharpen orientation tuning (subtract a fraction of the group sum),
   ignore very weak contrast, clip strong contrast, and add opposite signs together. Result: 6 unsigned edge
   orientations and 6 unsigned line orientations, the same whether the hand is lighter or darker than the background.
4. `pl`: triangular 8x8 pooling with stride 2, giving 12 channels on a 12x12 grid (1728 numbers); `n2` standardises
   the edge maps and the line maps separately.
5. `fc`: 69,240 templates (2,885 per letter). A template's score is the mean of the edge and the line correlation
   coefficient between picture and template.
6. Readout: a template votes `max(score - best score of all templates + 0.05, 0)`; a letter's output is the mean vote
   of its templates (a soft nearest-neighbour vote).

**Where the weights come from.** `build.py` reads no data. It contains a 3D hand model typed from general knowledge of
hand anatomy and the ASL manual alphabet: palm, fingers, thumb and forearm as chains of spheres, with joint angles,
thumb position, viewing direction and arm direction given per letter in a table. Each template is the feature map of
one rendered image (random small changes of joint angles, view, hand proportions, lighting and framing, fixed seeds),
standardised by itself alone. No statistic is taken across rendered images and nothing is fitted. All thresholds and
constants are hand-chosen from the derivations in the code comments.

**Use of the training set.** I looked at training photos (PNG sheets) to learn how each letter is shown in this
dataset: which way the hand faces, where the arm enters, how tightly the hand is cropped, and that folded fingers are
often held in a loose curl. I used that qualitative understanding to set the per-letter view and pose in the table.
No pixel values, coordinates or measurements from particular images were typed in.

**Checks used (3 of the 5 allowed, all on the 27,455 training images).**

- Check 1: 34.2% accuracy. Learned: the approach works for many letters (L, F, C, W, Y, A), but my P, T, X and Q
  poses did not resemble the photos and K templates absorbed D, R, U and X.
- Check 2: 37.6% accuracy. Learned: tighter framing and corrected views helped the fists, G, H, P, Q, K and Y;
  turning C to a back-of-hand view was wrong; T still never won; K still absorbed R, U and V.
- Check 3: 38.95% accuracy (from /app/out/checks.log), on the submitted version. Learned: restoring the C and O views,
  redefining T (index sideways, thumb up, seen from the back of the hand), the loose curl, new arm directions for
  G, H and T, and the soft vote together gave a further small gain. The process was killed (exit code 137) right after
  the result was logged, so the confusion table was lost and I have no per-letter breakdown for this version.

**Status of the submitted version.** It is the version measured in check 3: 38.95% on the training images, which
corresponds to a score of about 0.36 there. Accuracy on the held-out images is unknown. The fist-like letters
(A, E, M, N, S) were still largely confused with each other at check 2 and I expect that to remain.
