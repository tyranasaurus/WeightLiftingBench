# Method

**Architecture (about 9.8M weights, every number written by `build.py` from formulas or typed constants).**

1. **Pixel filters (conv 5x5 → relu → 1x1 → relu → 1x1 → relu).** These give a soft "item" mask (grey level ramping from 0.02 to 0.06), a stricter mask (0.08 to 0.15), the grey level, thin dark and bright vertical-line detectors built as min(left contrast, right contrast) with relu gadgets, and |∂x| and |∂y|.
2. **Directional context (dense 784→9·784, then relu and 1x1 conv).** For every pixel the network computes whether the item lies to its left, right, above and below, using cumulative sums along rows and columns, clamped with relu(s) − relu(s−1). Combinations of these four flags give the filled outline (pixels with item on all four sides), enclosed holes, background bays that open only upward, downward or sideways (neckline, gap between the legs, sleeve gaps, gaps between sandal straps), and the corner regions outside the outline.
3. **Silhouette matching (dense → reshape → maxpool).** The filled outline is compared with 2,000 hand-drawn parametric shape models: T-shirts and tank tops, long-sleeved tops (round neck, collar or hood), trousers, dresses (sleeveless, cap or long sleeves, straight or flared), bags with or without a handle arch, sneakers, ankle boots, and flat, heeled and wedge sandals. Each model is a polygon written from general knowledge of the garment. It is drawn in natural proportions, then scaled so its longest side is 28 px and centred, as the shop photos were. The score is pixel agreement, Σ outline·(2t−1) + Σ(1−t), and each class takes its best-matching model. Pullover, coat and shirt share one outline family. Sandals may also use the shoe and boot outlines at a −10 penalty, because straps fill in a shoe-shaped outline.
4. **Evidence rules (dense/relu layers forming soft indicators clamp((Σ a·f + c)/s), AND = relu(a+b−1), then a hand-written weight table in "pixels of agreement").** The rules are:
   - an opening between the legs together with a narrow item → trouser;
   - many gaps between straps, or a full-width item that is mostly empty → sandal;
   - item height → boot rather than sneaker;
   - a long straight line down the middle → coat or shirt. It is measured on 8-row column averages, must appear in both the upper and the lower torso, and must exceed the same measure in the neighbouring columns;
   - a checked pattern (many vertical and many horizontal edges relative to brightness) → shirt;
   - short sleeves with a placket → shirt;
   - a skirt that widens toward the hem → dress; sleeves wider than the hem → T-shirt;
   - long-sleeve gaps beside the body and a neckline dip → tops rather than bag; a hole under a handle → bag;
   - small class offsets for trouser, coat and shirt.

   The final scores are the silhouette scores plus the evidence.

**How the weights were produced.** There was no training, fitting, statistics of the data or copying of images. I looked at sample images saved as PNG files to understand the classes. Filters, regions, thresholds, shape models and rule weights were then chosen by hand from that understanding and from general knowledge of garments. Between checks I unit-tested the components only on synthetic images I drew in code: the shape models themselves, a striped or checked top, a top with an artificial zip, a strappy sandal, a bag with a thin handle. No training image was used for those tests.

**Checks.** Each check was run on the same fixed random set of 10,000 training images: 53.6% → 64.4% → 64.7% → 67.6% → 68.25%. Each one led to a change I could explain: an over-eager leg-gap rule, long-sleeve outlines that were too trapezoidal, sandals whose straps fill in a shoe outline, and a zip detector that was first too weak and then not specific enough.

**Remaining weaknesses.** Shirts are rarely recognised, many pullovers are called coats, sleeveless tops are confused with dresses, and some sneakers and boots are swapped.
