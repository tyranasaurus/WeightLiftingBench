# Method

**Architecture** (4 layers, 1,190 weights): `conv 3x3 (10 maps) -> relu -> avgpool 10x10 -> dense 10`.

- The conv computes ten per-pixel evidence maps from formulas: six colour categories
  (blue, green, warm, red, bright, dark; each a linear colour contrast minus a hand-chosen margin)
  and four signed luminance-step filters (horizontal and vertical edges, both signs, so that
  after the relu each pair sums to an edge magnitude).
- The average pool reduces each map to a 3x3 grid (top / middle / bottom thirds by left / centre / right).
- The dense layer is a hand-typed rule table: for each class, a weight for each feature group in the
  top, middle and bottom third of the picture (the same across the three columns), multiplied by a
  hand-chosen scale per feature group. Biases are zero.

**Where the numbers come from**: general knowledge of the classes only, e.g. airplanes sit in smooth
blue or bright sky; ships have water below and a waterline; deer and frogs are green/brown with no
blue; horses have grass below, a brown body and vertical leg edges; cars and trucks have road,
red/dark body colours and strong straight edges; cats and dogs are warm indoor neutrals.
No training, fitting or statistic of any image was used, and no generated images were used.
The 10 training photos were only viewed once as a contact sheet.

**Status**: this is a first, simple version saved when time ran out. The network was not run on the
training images (0 of the 5 checks used), so its accuracy is unmeasured and the table is uncalibrated.
