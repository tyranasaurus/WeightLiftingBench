# Method

**Architecture.** The image is divided by its own standard deviation (`norm`, then subtracting its minimum) and softly
thresholded into an item mask, so the threshold is relative to the item's brightness. The mask is cut into four 14 x 14
quadrants, the left ones mirrored so that left-right symmetric garments share templates. One convolution per quadrant
type (top / bottom, about 133,000 filters each) scores every quadrant against every template; max-pooling keeps the best
template per class and quadrant, and the four quadrant scores of each class are added. A small extra head detects a thin
vertical line through the centre of the trunk (zip, button placket, open front) and adds a hand-set bonus per class.
About 53 M weights in total.

**Where the weights come from.** No training or fitting of any kind. For every class I wrote a parametric drawing program
from general knowledge of the object: a T-shirt is a body with two short sleeves hanging beside it, trousers are a
waistband with two legs and a gap, a dress is a bodice with a waist and a flared or straight skirt, a sneaker is a low
wedge profile (optionally with a white sole, which is dark in an inverted photo), an ankle boot is a shoe profile with a
shaft at the heel end, a sandal is a thin sole with straps and possibly a heel, a bag is a rounded trapezoid with or
without a handle loop. Parameters (widths, sleeve angle, heel height, ...) are drawn from hand-chosen ranges with a fixed
seed. Each drawing is rendered as the task says the photos were made (trimmed, longest side 28 px, centred) and becomes
one template on its own: its silhouette is blurred (sigma 1.2 px) and converted to per-pixel log-odds, with the bias
set from that same template (a Bernoulli log-likelihood of the mask). No statistic is taken across drawings.
T-shirt / short-sleeved shirt and pullover / jacket / long-sleeved shirt are drawn from the same body parameters and
differ only in neckline, collar and hem, so those features decide between them. All thresholds, the blur, the outlier
probability and the cue bonuses are constants I chose by reasoning; they are named and commented in `build.py`.

**Use of the training images.** I looked at montages of the 1,000 images to understand what the classes look like
(for example that T-shirt sleeves hang down, that sharpening leaves a faint halo around items, that many sneakers have
white soles), and ran the five allowed checks (68.5 %, 72.1 %, 70.0 %, 71.6 %, 73.6 %). After each check I revised the
drawing programs or the design according to what the confusion table showed. No pixel values, coordinates or
measurements from particular images are typed into the code.

**One unchecked edit.** The submitted network differs from the one measured in check 5 (73.6 %) by a single change made
after the last check: the round-neck top edge of T-shirts and pullovers may now also be slightly raised (a neck band
standing above the shoulder points) instead of only flat or dipped. Check 5 showed round-neck items being taken by the
collar variants far more often than the reverse, which is what this corrects, but its effect has not been measured.
The exact check-5 version is the same `build.py` with `yn=U(r, B, -0.045, 0.06)` set back to `yn=U(r, B, 0.0, 0.06)`.
