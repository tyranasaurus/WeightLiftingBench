# Method

**Idea.** Analysis by synthesis: the network compares the silhouette of the photo with a large bank of
silhouettes that `build.py` *draws* from shape models I typed from general knowledge of clothing, shoes and
bags, and picks the class of the closest drawing. Nothing is trained or fitted; no training image, statistic
or measurement of one is in the weights.

**Architecture** (23.6 M weights, ops: conv, relu, dense, reshape, maxpool, add).

1. *Ink mask*: 1x1 convs + relu compute `min(1, (x - 0.02) / 0.06)` — is this pixel part of the item?
2. *Blur*: a fixed 9x9 Gaussian (sigma 1.4) so that outlines a pixel or two apart still agree.
3. *Template bank* `match` (dense, 784 -> 30,000): 3,000 units per class. Unit j holds one blurred drawing
   t_j as weights `2*w*t_j` and bias `-sum(w*t_j^2)`, i.e. minus the (row-weighted) squared distance between the
   photo's silhouette and that drawing. `w` is a fixed weight that makes the neckline rows count up to 2.5x.
4. *Class score* = max over the class's 3,000 units (reshape to (10, 1, 3000) + maxpool).
5. Two small hand-wired side branches, added to the class scores:
   * *front line*: a thin vertical line (zip / button placket) down the whole middle of the front — minimum
     over four row segments, maximum over 3 column placements x 2 polarities — gives coat and shirt +12.
     The shirt class also holds 600 copies of short-sleeved tee drawings with a handicap of 7, so a
     short-sleeved garment becomes "shirt" only when such a line is seen.
   * *openwork*: background trapped in horizontal gaps narrower than 9 px (a morphological closing built from
     padding + max-pooling), on an item tall enough to reach the top rows, gives sandal up to +25
     (strappy heeled sandal versus boot).

**Where the weights come from.** Each class has a drawing program built from polygons and strokes in the
item's own units: tee (torso, short sleeves hanging at the sides, wide neckline; also sleeveless tops),
trousers (waist, two legs, gap below the crotch; a few shorts), pullover (wide flat neckline, sloping
shoulders, sleeves beside the body with cuffs usually below the ribbed hem), dress (narrow bodice, waist,
flared skirt; straps / sleeveless / short / long sleeves), coat (same body as the pullover but a narrow
stand-up collar or hood, hem at or below the cuffs, long versions), sandal (flat sole with straps; rising
sole with heel and ankle strap; wedge with bands), shirt (thin sleeves, small collar open at the throat,
shirt-tail hem; three-quarter sleeves), sneaker (low wedge profile, toe left, tongue and heel tab), bag
(rectangle / trapezoid / sack body with or without a handle arch), ankle boot (toe box, ramp or step up to a
shaft over the heel, optional heel). A fixed-seed random generator picks proportions inside hand-chosen
ranges. Every drawing is trimmed to its bounding box, scaled so its longest side is 28 px and centred (the
framing described in the task), and becomes one template by itself, normalised by itself alone. All
thresholds, weights and bonuses are constants I chose by reasoning (listed at the top of the assembly section
of `build.py`); no number was chosen by scoring drawings or by fitting anything to the training images.

**Use of the training set.** I looked at contact sheets of the 1,000 images to understand the classes
(e.g. shoes point left and are vertically centred, pullover hems sit above the cuffs, coats have a raised
collar). I ran the network on them 5 times (`checks.log`):

| check | accuracy | what I learned |
|---|---|---|
| 1 | 69.6 % | Works on real photos; my short-sleeved "shirt" drawings swallowed T-shirts, and "coat" was predicted twice too often. |
| 2 | 75.1 % | Steeper tee sleeves, collar/neckline modelling and a stricter front-line detector fixed T-shirts, trousers, dresses; pullovers/shirts still fell to coat because my pullover shoulders were too square. |
| 3 | 75.9 % | Giving pullover/coat/shirt the same body proportions balanced those classes; plain sleeveless tops were being read as legs-together trousers. |
| 4 | 75.9 % | Sleeveless-top fix worked; a first openwork cue rescued sandals but misfired on dark sneakers with white (= background-coloured) stripes and laces. |
| 5 | 77.0 % | Restricting the openwork cue to tall shoes keeps sneakers intact and still recovers heeled sandals from "boot". This is the submitted network. |

Runs on images my own code drew were used only to confirm that the layers compute what they were designed
to compute (e.g. the trapped-area branch returns 50 for two bars 5 px apart over 10 rows); no number was
chosen from them.

**Known limits.** Shirt recall is low (short-sleeved shirts look like tees; long-sleeved ones like coats or
pullovers), chunky closed sandals look like sneakers, and long-sleeved wide dresses look like shirts/coats —
silhouettes alone cannot separate these.
