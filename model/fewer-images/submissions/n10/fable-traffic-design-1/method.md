# Method

**Architecture.** A fixed, hand-designed matched-filter network (about 60M weights, no training of any kind).

1. *Front end (formula filters only).*
   - Brightness = mean of R, G, B, then `norm`. Eight Gaussian-derivative filters (sigma 0.8 px, 45 degree steps),
     `relu`, a 1x1 conv that sharpens orientation tuning (response minus the two perpendicular ones), `relu`, then a
     4x4 tent-window pooling with stride 2 into 16x16 cells.
   - Local normalisation: a one-hot dense layer gathers the cells into 7x7 overlapping blocks of 4x4 cells, one block
     per channel; `norm` standardises each block; a second dense layer blends the blocks back to a (8,16,16) map with
     tent weights. The gain before `norm` sets a semi-saturation floor chosen by hand from the response of the filter
     to an ideal step edge.
   - The same block normalisation applied to pooled brightness gives a (1,16,16) local-brightness map.
   - Per-channel `norm` of RGB, two colour-opponent combinations (red-green, blue-yellow), pooled to cells and
     normalised jointly, give a (2,16,16) colour map.
2. *Templates.* Three 16x16 convolutions with padding 1 (one per map, 12,040 kernels each) are added, max-pooled over
   the 3x3 shifts, then max-pooled over the 280 templates of each class. The 43 maxima are the class scores.

**Where the weights come from.** `build.py` draws each of the 43 signs from geometric descriptions I typed from my
knowledge of German road signs (rings, triangles, octagon, diamond, digits and letters as strokes, arrows, simple
pictograms). Each drawing is placed at 7 sizes x 4 half-pixel shifts x 10 looks (upright, rotated by -8/+8 degrees,
two aspect ratios, extra blur, dark surround, bright surround with the sign in shade, and two proportion variants of
the inner symbol). Every drawing is passed through the same fixed front end and becomes one template, masked by the
sign's own silhouette and normalised by itself alone (zero mean over its mask, unit length). No statistic is taken
across drawings, no number was fitted, and no training image or anything derived from one is used in the weights.
The three evidence weights (edges 1.0, brightness 0.7, colour 0.5) and the normalisation floors were set by hand.

**Checks used (2 of 5).**
- Check 1: 80.0% on the 430 training images. Learned that the pipeline works on real photos and that a few of my
  pictogram drawings (priority at next junction, bends, double bend, slippery road) were too thin, and that merging
  both edge polarities at the outline hurt the white "end of" signs.
- Check 2: 81.6% after redrawing those symbols more boldly and adding the surround, blur and proportion variants.
  Learned that the bend arrows were fixed, but classes 11, 21 and 23 still lose to the snowflake and deer templates,
  my bolder pedestrian and children drawings got worse, and more dark speed-limit signs fall to the empty-ring class.

**Known weaknesses of the submitted answer.** The drawings for classes 11, 21, 23, 26, 27 and 28 still do not match
the real symbols well (on the training sample classes 11, 21 and 23 were 0/10), and sparse templates (empty ring,
exclamation mark) win when a symbol is dark or blurred. I had identified fixes for these but ran out of time before
applying or checking them, so they are not in this answer.
