# Method

Hand-designed network (176 layers of conv / relu / pooling / add / dense, ~96k weights, none trained).
Every number in `build.py` is typed from general knowledge of skin colour, face and body proportions, and
scene context; nothing was fitted to the data, and no training image or statistic of it enters the weights.

**Pixel stage (1x1 convs).** A skin map is built as a soft AND (ReLU gates) of linear colour inequalities
typed from the usual RGB/HSV skin rules: R clearly above G, warm hue between ~5 and ~27 degrees, saturation
0.20-0.65, R above 0.30. R, G, B are carried along; luminance, two chroma channels and an edge-energy map are
derived per pyramid level (96 -> 48 -> 24 -> 12 by average pooling).

**Detectors (window convs + 1x1 ReLU logic: min / max / clip gates built from ReLUs).**
- *Skin blob*: 3x3 centre mostly skin while the maximum over the four 3-wide surround bands is mostly not skin
  (compact face/hand-sized patch), at four scales.
- *Face*: a blob gated by face structure: a cap region directly above that is darker than the blob and dark in
  absolute terms (hair), and/or a dark horizontal line through the blob (eyes), via a border-safe min of the
  two half-differences. Face evidence never exceeds the blob evidence.
- *Figure*: a 3-wide, 7-tall body whose upper box, lower box and torso both contrast (same sign) with both side
  backgrounds (min over all parts and sides), with no contrast directly above the head, on the flanks or below
  the feet, and a smooth surround (edge energy in a ring must stay low). Used at the two finer scales.
- *Snow*: fraction of bright, colourless, smooth cells in the lower 70 % of the picture (ski scenes).
- *Skin fraction*: small amounts count slightly for, large amounts (wood, food, animals) against.
All contrasts are mins over sides/parts and surrounds are maxes over bands, so zero padding at the image border
can only suppress a detection, never create one.

**Readout.** Global max pooling of each detector map, clipping to [0, 1], and a hand-set weighted sum: a face or
an isolated figure decides alone; snow and plain blobs only add partial evidence; threshold 0.95.

**Checks.** Three of the five permitted checks were used: 50.7 % (first version: detectors fired on almost every
real photo), 54.1 % (structure requirements added), 57.2 % for this submitted version (false positives on
no-person images 31 %, recall on person images 45 %). The lesson from the checks was that colour alone is far
less specific on real photos than on drawings, so later versions required face structure, body geometry and a
smooth surround before a detection could count.
