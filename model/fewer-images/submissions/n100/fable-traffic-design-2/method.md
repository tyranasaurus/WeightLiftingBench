# Method

**Architecture.** A hand-designed matched-filter network, no learned weights.

1. *Front end.* `log(x + 0.025)` per colour channel built from 16 relu pieces, then one per-image `norm` over all three
   channels. From this: (A) three opponent-colour maps (luminance, red-green, blue-yellow; colour gain 2), lightly blurred;
   (B) eight signed oriented-edge maps (Gaussian-derivative filters on luminance, relu, orientation sharpening and
   saturation with relus, Gaussian pooling, `norm`); (U) four unsigned edge maps (opposite directions added, `norm`)
   that describe the sign outline whatever the background brightness.
2. *Template banks.* Two banks of 3,612 templates (43 classes x 84 poses), each a 28x28 conv over the feature maps, so
   every template is tried at 5x5 placements. The "whole-sign" bank scores the full sign (A + B + U); the "ranking" bank
   adds an interior-only content template (weight 2) used to rank classes inside a family.
3. *Read-out.* Max over placements and poses per class (reshape + maxpool), then small dense/maxpool layers implement
   `score(c) = best whole-sign score in c's family + rank(c) - best rank in c's family`, with families = red ring,
   red triangle, blue disc, white "end of" disc, and four singletons. Families compete on the whole-sign match; classes
   inside a family compete on their interior content, which is not biased towards signs with less ink.

**Where the weights come from.** `build.py` draws all 43 signs as vector shapes from the German road-sign designs
(rings, triangles, octagon, diamond, DIN-like digits as strokes, arrows, pictograms from polygons and strokes), renders
each into the 32x32 frame at 7 sizes x 3 aspect ratios x 3 small rotations plus an upright blurred set, passes each
render through the front end, masks it to the sign, removes its own mean and normalises it by itself alone. Each render
is one template. All filters are formulas (Gaussians, Gaussian derivatives, opponent colours); all constants (blur
widths, saturation level, group weights, the family table) were chosen by hand. Nothing was fitted, no training image
or statistic of images is in the weights, and `build.py` reads no files.

**Checks used (2 of 5).**
- Check 1, 72.6 %: the approach works, but colour was under-weighted (triangles and the roundabout went to blue-arrow
  and red-ring classes), empty signs (yield, no vehicles) lost to other families, and several drawings were wrong
  (yield border too thick, digits too small, "end of" band too faint, some pictograms).
- Check 2, 83.1 % (balanced over classes): the colour group, outline-edge group, family logic and redrawn signs fixed
  the cross-family errors. Remaining errors are mostly pictograms I still draw poorly (priority-at-junction, slippery
  road, double bend, bicycle) and "end of" signs taken for red-ring signs.

The training images were otherwise only viewed as PNG sheets to see what the classes look like. Rendered signs were run
through the network only as a wiring check (does each drawn class map to itself). One such run showed blurred speed
signs falling to the empty-ring class; the remedy (a blurred pose set and the interior content term) was derived from
the algebra of cosine matching, not by scoring alternatives on rendered images.

**Not done.** I ran out of time before redrawing the weakest pictograms identified in check 2, so those classes remain
weak in the submitted network.
