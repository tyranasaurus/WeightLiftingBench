# Method

**Architecture.** The network is a fixed feature pipeline followed by hand-written fuzzy rules (about 21M weight entries, 232k nonzero).

1. **Two silhouette masks.** 1x1 convs and relus turn grey levels into soft masks: a low threshold (0.04 to 0.12) that shows the whole item, and a high one (0.15 to 0.25) where faint gaps show up.
2. **Open-direction maps.** Dense layers take running sums along every row and column. Each is clipped as relu(1 - sum), which gives four maps per mask: NL, NR, NT and NB. Each map marks whether the item has started yet when you walk in from the left, right, top or bottom.
3. **Region codes.** A 1x1 conv turns the four flags into background codes:
   - notches open to the top-left, top-right, bottom-left or bottom-right;
   - notches open only to the top or only to the bottom;
   - enclosed holes;
   - row-filled spans.

   These carry the garment knowledge. Background under short sleeves opens to the bottom-left and bottom-right (T-shirt). The gap between trouser legs opens only downward. A bag handle encloses a hole. A shoe with its toe on the left has a large top-left notch and almost no top-right notch. A flaring skirt has top-left and top-right notches reaching below the chest. A tank top has a top-only notch at the neck.
4. **Edge branch.** A 7x7 conv computes small gradients plus tall vertical line and edge filters (7-row averages of [-.5, 1, -.5] and [-.5, 0, .5]). These are gated to the eroded inside of the item, and row-wise maxpools pick out zips and sleeve seams. 3x3 erosions of a hard low-threshold mask measure solidity, since thin sandal straps disappear.
5. **Features and rules.** Dense layers sum these maps over named regions to give features in pixel units (width, height, areas, notch areas, line counts, texture). A rule layer computes predicates, clamp((linear combination - a) / (b - a), 0, 1), then AND terms of them, relu(sum of literals - (n - 1)). A linear layer gives up to 4 sub-class scores per class, and a reshape plus maxpool takes the maximum of each class's sub-class scores.

**How the weights were made.** Every number comes from formulas or rules typed in `build.py`: derivative kernels, cumulative-sum matrices, region boxes, and thresholds and rule weights chosen from general knowledge of garment shapes (e.g. shoes point left; tops are roughly square; trousers are narrow with a gap between the legs). Nothing was fitted, and no training image, crop or statistic is stored.

I looked at training images as PNG grids to understand the classes. I debugged the network only on synthetic silhouettes drawn from formulas in `/app/work/synth.py` (T-shirt, tank top, pullover, coat, dress, trousers, sneaker, boot, sandal, bag), at several brightness levels.

**Checks.** I used the 5 allowed checks on the 60,000 training images, adjusting rules by reasoning about each confusion table. Accuracy was 51.1%, 54.9%, 60.8%, 61.8%, and finally 62.1%.

The remaining weak spots are:
- coat vs pullover, which is close to a coin flip;
- shirts, which are mostly taken by T-shirt, coat and pullover;
- sneaker vs sandal vs boot.
