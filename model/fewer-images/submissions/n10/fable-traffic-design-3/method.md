# Method

**Architecture.** `image -> 1x1 conv (36) -> relu -> 1x1 conv (3) -> 7x7 conv (3) -> dense (15,093 templates) -> reshape (43, 1, 351) -> maxpool over each class's 351 templates -> 43 scores`.

**How the weights were produced (no training, no data read by build.py).**
- The first two layers are a piecewise-linear approximation of `log(x + 0.03)` per colour channel, built from 12 relu
  units per channel with hand-chosen knots, then mixed into three log-opponent channels: brightness, red-versus-rest
  and blue-versus-rest. Working in the log domain makes dark and backlit signs comparable to well-lit ones.
- The 7x7 conv holds fixed formula filters: a difference of Gaussians (sigma 0.8 minus 2.2) on brightness, and a
  Gaussian (sigma 1.0) on the two colour channels.
- The dense layer is a bank of templates. Each of the 43 signs is drawn in code from general knowledge of German signs
  (red ring with DIN-like digits drawn as strokes, red-bordered triangles with pictograms, blue discs with white arrows,
  the diamond, octagon, no-entry bar, and the grey "end of" signs with a diagonal band). Each drawing is placed at 351
  poses (sign size 60-84 percent of the frame in x and y, rotation -6/0/+6 degrees, shifts of -1/0/+1 pixel), passed
  through the same fixed filters, and stored as one template: brightness is kept only inside the sign's outline with
  that template's own mean removed, colour is kept over the whole frame with its own mean removed, and the template is
  divided by its own norm. No statistic is taken across drawings.
- The class score is the largest template response for that class (cosine-style nearest template).

All constants (knots, filter widths, colours, pose grid, colour weight 0.5) were chosen by hand in advance; none were
tuned on training or generated images.
