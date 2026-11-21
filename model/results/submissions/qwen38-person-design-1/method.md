# Method

## Architecture
A hand-designed, scale-free "parts and configuration" detector, 284 layers,
11,634 weights (all produced by `build.py` from formulas and typed constants;
no file is read, no image statistic is used).

1. **Pixel front end** (conv/relu/min-max caps, every feature kept in [0,1]):
   * six unsigned gradient-orientation bins + edge energy from Sobel
     projections of luminance at 30° steps;
   * a skin mask built from hat-shaped chromaticity gates
     (R-G, R-B, G-B bands plus a luminance band, combined as a soft AND),
     chosen so yellow fruit and brown wood/bread fall outside the skin band;
   * dark/bright centre-surround (DoG) blob masks at four head-size scales
     (full, 1/2, 1/4, 1/8 resolution);
   * colour saturation and mean luminance.
2. **Cell grids**: avgpool 4x4 and 8x8 of those masks (HOG-like cells).
3. **Presence maps**: per channel, `p = min(max(z-score over cells, 0)/2.5, 1)`
   using the `norm` op, so every cue is contrast/scale free and no single
   salient cell can dominate a template sum.
4. **Six-scale template bank** (windows 16/24/32/48/72 px): convs correlate
   the presence maps with hand-stamped person models - standing figure
   (head blob above a shoulder line, vertical torso sides, leg edges with a
   gap, figure-ground suppression), sitting/upper body, face (compact skin
   core with hair blob above, no skin at the corners) and a skin-patch
   template. Each template response is thresholded at 0.5 x (sum of its
   positive weights) - a majority vote on the expected person layout - and
   capped to [0,1] ("fire" map).
5. **Readout**: global max- and mean-pool of every fire map, plus a global
   skin-fraction gist; a hand-picked linear combination (dense layers summed
   with `add`) gives the two class scores, class 0 being the constant 0.

## How the weights were produced
All filters come from formulas (Gaussian/DoG kernels, Sobel directions,
chromaticity bands from general knowledge of skin colour) and all template
taps were stamped by code from a typed stick-figure/face model (head above
shoulders, torso sides, legs; skin core with hair above). Thresholds, the
presence cap, the vote fraction and the readout weights are hand-chosen
constants. No gradient descent, no fitting, no statistics of the training
images or labels entered any weight.

## Checks (5 of 5 used, logged in checks.log)
They were used to calibrate the hand-set constants: (1) absolute-scale
calibration never fired (15% recall); (2) unbounded z-scoring fired on
everything - single salient cells dominated the sums; (3) bounded presence +
majority vote fired on ~half of both classes - generic compact objects match
edge-frame templates; (4) a stricter vote fired selectively (precision 0.58)
but missed most people; (5) loosening it again gave 0.518 with precision
0.50, showing shape-only evidence ranks poorly while skin/face evidence is
the selective cue. The submitted readout therefore up-weights face/skin-patch
and skin-gist evidence relative to the generic figure templates. Expected
accuracy is modest (roughly 0.52-0.65); the honest conclusion from the checks
is that hand-set shape templates alone separate these classes weakly, and
skin/face cues carry most of the usable signal.
