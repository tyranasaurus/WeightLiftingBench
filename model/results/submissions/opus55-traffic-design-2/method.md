# Method

**Architecture.** The image goes through a per-channel `norm`, which standardises lighting and gives a gray-world white balance. Two 32x32 convolutions then each hold 1806 templates: 43 classes x 7 sign sizes x 3 aspect ratios x 2 pictogram scales.
- The **frame** convolution (padding 3, so shifts of +-3 px) scores the sign's surface. This means its shape and colours, with the pictogram blurred to a coarse blob. Lightness is zero-meaned within the sign; colour is measured against neutral.
- The **content** convolution (padding 4, then a 3x3 max, so the pictogram may move +-1 px against the frame) scores the pictogram's band-passed detail inside the sign's interior.
- The two score maps are added. A max over shifts and then a max over each class's variants gives the 43 class scores.

**How the weights were made (no training).** Every class is drawn by code from geometric primitives typed from general knowledge of German sign designs:
- red rings, rounded triangles, octagon, diamond, blue discs;
- digits and letters drawn as strokes;
- vehicles, arrows and pictograms drawn as polygons and strokes;
- hand-chosen paint colours.

Each rendering becomes its own template, normalised by itself alone: blurred with a Gaussian, split into a coarse part and a detail part, and given separate unit-norm lightness and colour budgets. The only other numbers are a few constants chosen by hand: blur widths, size range, and a bias of 2 for classes with a plain interior (no vehicles, yield). No statistic of training images or labels, and no statistic across generated images, sets any weight.

**Checks on training data.** Check 1 (plain whole-sign templates) scored 0.56; check 2 (frame/content split) 0.71; check 3 (frameless bias for end signs, joint normalisation) 0.64. The submitted version is untested on training data: check 4 was killed before it finished. It keeps check 3's rim-free, absolute-colour frames and restores check 2's separate budgets.
