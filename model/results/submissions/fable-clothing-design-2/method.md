# Method

## Architecture

A hand-wired "nearest drawn shape" network (30 layers, 102.5 M weights, no trained parameters).

1. **Silhouette** (`t`, `t_r`): a 1x1 conv and relu turn the image into a soft mask, 0 below grey level 0.07 and 1 above 0.13.
2. **Filled silhouettes** (`d1`, `d2`): two dense layers with relus compute, from running sums along each row and column,
   the row-filled mask (everything between the leftmost and rightmost item pixel) and the column-filled mask. These
   ignore dark prints and stripes inside an item but keep gaps that are open to the outside (trouser legs, hem notch).
3. **Prototype bank** (`bank`): one dense layer scores the three masks against 35,520 drawn prototypes. The score is
   minus a multi-scale squared distance (pixel level plus Gaussian blurs of sigma 1 and 2), so a one-pixel shift of an
   outline costs little and a missing sleeve costs a lot.
4. **Shape groups** (`bank_g`, `grp`, `cls`): prototypes come in 592 groups of 60 that share the same attributes; a
   maxpool keeps the best match per group and a dense layer adds each class's offset for that group.
5. **Centre-line cue** (`z` ... `cue2`): on the contrast-normalised image, a 19-row filter finds a thin vertical valley
   or ridge down the middle (zip or button placket), using relus to take the minimum of the two one-sided differences
   so plain edges do not count. The same detector on the silhouette cancels lines that are real gaps (trouser legs,
   open necks). The cue adds to coat and shirt scores.
6. **Decision** (`sum`, `rs`, `mx`, `out`): the best group per class gives the ten class scores.

## Where the weights come from

`build.py` reads no files. It contains a small signed-distance renderer and shape models that I typed from general
knowledge of the items: a top is a neck, sloping shoulders, armpits, hem and optional sleeves; trousers are a waistband
and two legs with a gap; a sneaker is a low wedge with a toe box, collar peak and heel; a boot adds a shaft; a bag is
a rounded box with an optional handle arc; sandals are a sole plus straps, or a sole plus a "soft zone" of partial
value where straps of unknown layout sit. Each model's proportions are drawn at random inside hand-chosen ranges with
a fixed seed, the shape is fitted to the 28-pixel frame the way the dataset trims and centres items, and the result is
written into the bank. Pullover, coat and long-sleeved shirt share one shape family (likewise T-shirt / short-sleeved
shirt, and tank top / sleeveless shirt); which class a shape counts for is set by hand-written tables of how typical
each attribute is (collar or flat neck, hem above or below the cuffs, width, sleeves hanging apart).

No gradient descent, fitting, averaging or other statistic of the training images was used, and no pixel values or
measurements were copied from them. I looked at grids of training images to learn what the classes look like and
compared them by eye with my drawn shapes.

## The five checks (all on the 60,000 training images)

| check | accuracy | what I learned and changed by hand afterwards |
|---|---|---|
| 1 | 66.3% | Pipeline works; T-shirts were called shirts and pullovers coats. Added position jitter, rarer-variant penalties, better shoe shapes. |
| 2 | 70.0% | With the cue off the same errors remained, so the cause was the shape bank: one class's shape family covered another's. Moved to shared families with attribute tables. |
| 3 | 72.5% | Pullover/coat balanced; "shirt" guesses were right only ~30% of the time; the cue fired on trouser leg gaps. Added the gap veto, a shirt handicap, a small coat bonus. |
| 4 | 73.4% | Allowing low boot shapes cost more sneakers than it gained boots; shirt guesses still ~23% precise. Restored the check-3 shoe height ranges, strengthened the shirt handicap. |
| 5 | **73.8%** | Final configuration (the submitted one). |

The numbers changed in response to checks are a handful of hand-set constants (class offsets in `CLASS_ADJ`, family
offsets, shoe height ranges, the cue window), each changed for a stated reason; no constant was fitted numerically.

## Known limits

- **Shirts are essentially never predicted** (2 of 6,000 in check 5). Their silhouettes overlap T-shirts, pullovers and
  coats, and every attempt to predict them was wrong more often than right, so the network gives them up.
- Coat vs pullover is still weak (coat 57%, pullover 75%): the silhouettes are nearly identical and the centre-line cue
  catches only part of the coats.
- Sandals lose about 1,000 of 6,000 to sneakers (chunky, mostly filled sandals look like low shoes in outline).
- Per-class accuracy in check 5: T-shirt 83%, trouser 92%, pullover 75%, dress 84%, coat 57%, sandal 78%, shirt 0%,
  sneaker 90%, bag 93%, ankle boot 86%.
