# Method

## Architecture

A hand-designed, fully interpretable feature-matching network (137 layers, ~219k weights,
all drawn from geometry and from my knowledge of the German road-sign catalogue). It has
four stages:

1. **Colour front end.** `norm` standardises each RGB channel over the whole frame. Because
   mean and standard deviation both scale with a multiplicative lighting change, this makes
   the representation exactly invariant to the very dark / washed-out frames. A 1x1 conv
   then forms colour-opponent channels (redness `R-(G+B)/2`, blueness, yellowness, luminance,
   ink). The opponent channels cancel the common luminance swing, so they carry a hand-set
   gain of 2.5 to be commensurate with luminance. A parallel 1x1 conv keeps *signed*
   (un-thresholded) luminance / redness / blueness / greenness for matched filters.

2. **Outer-shape category detectors.** On a half-resolution map, 16 convolution templates
   (kernel 15, 8x8 positions near the frame centre, then max-pool) score seven sign families:
   red ring, filled red disc, upright warning triangle, inverted triangle, yellow diamond,
   colourless "end-of" disc, blue mandatory disc. Each template is a difference of region
   means (border minus core minus outside), built from analytic disc / annulus / polygon
   masks at two or three sign radii. Exact maxima over scale groups are computed with
   `max(a,b) = b + relu(a-b)` from dense/relu/add. A second branch computes, for each family,
   the strongest *competing* family, and the readout subtracts 0.7 of it, so a class only
   wins when its own outer shape is the best explanation.

3. **Symbol matched filters.** All symbols are drawn analytically as capsules, elliptical
   arcs, filled ellipses and triangles in a unit box, rasterised with 3x supersampling:
   * digits 0,1,2,3,5,6,7,8 composed into the blocks 20/30/50/60/70/80/100/120 at four
     (two-digit) and three (three-digit) heights, kernel 21, stride 1;
   * the eight blue-disc arrows (turn left/right, ahead, ahead-or-side, keep left/right,
     roundabout) at three disc radii, kernel 21;
   * fifteen warning pictograms (priority cross, exclamation, bends, double bend, bumps,
     slippery car, narrowing, road works, pedestrian, children, bicycle, snowflake, deer,
     traffic-light dots) at two scales, kernel 19;
   * traffic-light dots additionally use centre-surround red/green/yellow filters, so a red
     sign border or green foliage cannot fire them.
   Each filter is zero-sum (`mean over surround - mean over strokes`), so it responds to a
   contrast *pattern* and not to overall brightness; a hand-set bias thresholds weak matches
   to zero. Digit templates also carry a redness penalty on their surround, since speed
   digits are black on white, never on red.

4. **Interior region contrasts and edge energy.** 34 region templates (core vs. a white
   reference annulus, split left/right/upper/lower, plus a centre-band vs. off-band contrast)
   distinguish empty disc / truck / two cars / truck+car, stop lettering vs. the no-entry bar,
   and stripes-only vs. digits vs. cars on the grey discs. A `[-1,0,1]` gradient, rectified
   both ways and summed, gives vertical-edge energy (STOP lettering has many, a bar has none).
   These are read at the four central conv positions so they stay jointly aligned.

The final 43 scores are the sum of eight hand-written dense rule tables:
`score(k) = 3*(category(k) - 0.7*best_other_category) + symbol evidence(k)`, with an
"explaining away" term that suppresses the car/truck/empty-disc classes whenever any speed
number matched well.

## How the weights were produced

Every number comes from `build.py` itself: analytic shape formulas (discs, annuli, convex
polygons, Gaussian derivatives), stroke models of the DIN pictograms and digits typed from
general knowledge of the signs, and thresholds / rule coefficients chosen by hand.
`build.py` reads no files. No training image, patch, descriptor or label statistic was used
to set any weight; no fitting, regression, class averaging or template tracing was done.
The training set was used only to *look* at montages of the classes (saved as PNGs) and for
the three logged full-set checks, which were used to find calibration bugs (the opponent
channel gain, the grey-disc colour penalty, and centre-surround colour dots).

## Status

Three of the five allowed checks were used (6.3% -> 16.2% -> 18.0% on the 39,209 training
images). The network is well calibrated for the blue mandatory discs, the yellow diamond,
no-entry and the grey "end-of" discs; the speed-limit digit bank and several warning
pictograms are still under-performing, and the "no trucks" class remains an over-broad
attractor for signs with dark cores. This is the best verified state reached.
