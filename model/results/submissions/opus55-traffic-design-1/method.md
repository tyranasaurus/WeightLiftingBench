# Method

**Architecture.**
1. A 5x5 conv turns the photo into four feature maps: luminance, "ink" (local mean minus centre, then relu), red-green chroma and blue-yellow chroma.
2. For each sign family (circle, upward triangle, downward triangle, diamond), a stride-32 conv on the stacked feature maps resamples the photo into a canonical frame. It does this at 7 sign-size hypotheses, using Gaussian sampling weights inside a mask of that family's shape. Every row has the block mean removed, so luminance and ink are relative. The block-mean chroma is added back through a small broadcast conv, so chroma stays absolute (red cannot be confused with "less blue").
3. Each block is then normalised with `norm`, so a dot product with a unit template is a normalised cross-correlation.
4. Separate interior windows are resampled and normalised the same way. They sit clear of the frame: a disc of 0.64 or 0.73 of the sign radius, or a small triangle around the incircle of a danger sign.
5. Template convs score the following over 243 fine sub-poses per scale (3 sub-scales × 9×9 sub-pixel shifts):
   - "frames": the empty red ring, blue disc, end-of-restriction disc and red triangle;
   - pictograms: the class drawing minus its frame, unit-normalised, scored on the interior window;
   - complete templates for stop, no entry, yield and priority road.
6. Each class's score is the maximum over poses of frame correlation + 0.5 × pictogram correlation.
7. Classes whose interior is empty (no vehicles, end of all limits, yield, priority road) score their frame correlation plus a fixed allowance of 0.5 × 0.3.
8. Each group's best score is capped at 1.5 × its best frame correlation. The same excess is subtracted from every class in the group, so the order inside the group is kept.
9. Max-pooling and one-hot dense layers place the scores into the 43 outputs.

**How the weights were produced.** No weight comes from data.
- Every template is rendered by `build.py` from vector drawings I typed from general knowledge of the German StVO sign designs: ring and border proportions, DIN-style digits drawn as strokes, arrows, the "STOP" letters, the end-of-restriction stripes, and simple pictograms. Each rendered image is blurred with a fixed Gaussian and normalised on its own.
- The resampling weights are Gaussian formulas.
- The gains, blur widths, pose grid, the 0.5 pictogram weight, the 0.3 empty-interior allowance and the 1.5 cap were all set by hand.
- I looked at training images only as pictures, to judge the sign designs qualitatively (for example, that danger-sign symbols are compact and sit in the triangle's incircle). No measurements or statistics of them were used.
- Clean synthetic renders were used only to confirm the network wiring is correct.

**Checks (4 of the 5 allowed).**
- Check 1, plain NCC: 69.2%. The empty "no vehicles" template absorbed many signs, and relative chroma let blue signs look like red rings.
- Check 2, absolute chroma and frame + pictogram scoring: 66.3%. The blue classes were fixed, but pictogram evidence leaked across groups.
- Check 3, interior-normalised pictograms: 73.3%.
- Check 4, compact danger-sign pictograms, frame-clear interior windows and the group cap: 74.4%. This is the submitted version.

**Known weaknesses.** Speed-limit digits are still confused with each other and with "no vehicles", some red signs still leak to blue classes, and some danger pictograms are confused with each other: cyclists, double bend, priority-at-junction against snowflake.
