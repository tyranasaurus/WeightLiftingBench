# Method

## Architecture
A small hand-designed matched-discriminant network (no training, no fitting):

1. **blur** — a 5×5 Gaussian conv (σ≈0.8, from a formula) with a small negative
   bias (−0.06). Followed by
2. **pos** — a `relu`, so the near-black background (≈0, the stated data range)
   is removed and what remains is a non-negative *item mask* `r` that preserves
   the full bright silhouette.
3. **tmpl** — one `dense` layer of 20 rows (10 classes × 2 pose variants). Each
   row is a hand-drawn **discriminant** `P_c/|P_c| − N_c/|N_c|`, so its score is
   `mean_{P_c}(r) − mean_{N_c}(r)`.
4. **grp → mx → out** — reshape to (10,1,2), `maxpool` over the 2 variants
   (as-drawn and left–right mirror, so shoes facing either way match), reshape
   to the 10 class scores. The prediction is the argmax.

## How the weights were produced
Every number comes from geometry typed from general knowledge of how these ten
items look — never from the training images.

For each class I drew two region masks from rectangles:
* `P_c` — where that class characteristically **has** mass (e.g. a short wide
  body with short sleeves for a T-shirt; two legs with a waistband for trousers;
  a narrow tall column for a dress; a low solid shoe; an L-shaped boot; a
  rectangular block for a bag).
* `N_c` — where its most-confusable neighbours have mass but this class does
  **not**: the centre gap between trouser legs (a filled torso covers it), the
  near-sides a wide top fills but a dress does not, the lower-side long-sleeve
  zone a pullover fills but a T-shirt does not, the strip just above a bag's
  flat top where a garment's shoulders sit, the whole empty upper half above a
  shoe, the tall ankle shaft that a boot has but a sneaker does not, a shirt's
  open collar, and a garment length ordering (pullover < shirt < coat).

The weight row `P_c/|P_c| − N_c/|N_c|` is balanced (it sums to zero, so constant
brightness and the background offset cancel) and scale-free (scaling `r` scales
every score equally, leaving the argmax unchanged), which keeps any one template
from dominating. The Gaussian kernels come from a formula; the `−0.06` bias
comes from the stated range (background near 0). No statistic of the training
data, and no statistic across generated images, sets any weight; the few
synthetic shapes rendered were used only to confirm the network runs and ranks
its own idealised silhouettes correctly.
