# Method

## Architecture (49 layers, 349,820 weights, all written by `build.py` from formulas)

The network is a hand-wired rule system expressed in the allowed operations.

1. **Silhouette.** Two 1x1 convolutions and a relu turn the grey image into a soft binary silhouette
   `B = clip(10*x - 0.5, 0, 1)`.
2. **Geometry of the silhouette**, all by pooling and fixed-kernel convolutions of `B`:
   row/column occupancy (max-pools over whole rows and columns, whose sums are the object's height and width),
   half-image column-fill profiles, a two-sided "bright-dark-bright" gap detector on the lower-half profile
   (realised with the identity `min(a,b) = (a+b-relu(a-b)-relu(b-a))/2`), a row-width profile with a
   "thin row suddenly followed by a wide row" detector, 7x7 and 3x3 morphological closings and a 5x5 erosion
   (max-pools on `B` and on `1-B`, padding supplied by 1x1 convolutions), horizontal/vertical transition maps,
   and a vertical-line detector on the grey image (|second horizontal derivative| integrated over 5 rows).
3. **Predicates.** One dense layer per pooled tensor (summed with `add`, then relu) computes 24 soft
   thresholds `clip((feature - t)/w, 0, 1)` of linear functionals of those tensors, each as a relu pair:
   left-right asymmetry (shoes point left), height bands (sneaker / ankle-boot / full-height), fill ratio of the
   bounding box, gap area, solid core, width bands, shoulder width, centre and side gaps, sleeve drop, flare,
   thin top (handle or straps), wide body, sleeve-hem edges, vertical line detail in the upper and lower body.
4. **Gates.** A dense layer forms 35 soft ANDs of predicates and negated predicates
   (`clip(sum p - (n-1), 0, 1)`), e.g. "asymmetric AND low AND solid AND not sparse" (sneaker),
   "full height AND symmetric AND not wide AND centre gap" (trousers), "full height AND thin top AND wide body"
   (handle bag).
5. **Scores.** The last dense layer adds hand-chosen evidence weights from each gate to the ten class scores.

## How the weights were produced

Every number is a design constant typed from general knowledge of the item shapes and of the preprocessing
(longest side 28 px, centred, light on black): e.g. trousers are about half as wide as tall and show a dark gap
between the legs in the lower half; sneakers are low, solid, and taller at the heel than at the toe; sandals are
made of thin straps on a thin sole, so a 5x5 erosion leaves almost nothing; bags are left-right symmetric with a
thin handle above a wide body; T-shirts are widest at the sleeve band and narrower below it; coats show a
dark opening or a vertical zip/button line down the centre. Thresholds were set by reasoning about pixel sizes
of these parts in a 28x28 frame. The logic was debugged on synthetic shapes drawn from rectangles (not on
training images), and then checked five times on the training set; after each check the confusion table was
read and rules were revised by hand. No weights were fitted, averaged, copied or traced from any image.

The five checks gave 54.1 %, 58.4 %, 62.2 %, 60.1 % and 64.0 % on the 60,000 training images. The submitted
version differs from the last checked one only by moving the sneaker/sandal boundary (solid-core and fill
thresholds) slightly toward sneakers, since the last check showed 2,150 sneakers called sandals against 362 sandals
called sneakers. The weak classes are shirt and coat, which share their silhouette with pullovers.
