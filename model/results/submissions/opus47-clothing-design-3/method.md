# Method

## Architecture

```
image (1, 28, 28)
  -> pool : avgpool 2x2, stride 2    -> (1, 14, 14)   # downsample
  -> flat : reshape                   -> (196,)
  -> out  : dense 196 -> 10
```

A 14x14 average-pooled view of the input is linearly classified by ten
hand-designed templates (1970 weights total: 10 × 196 + 10 biases). Each
output k is the dot product between template k and the pooled image.

## How the weights were produced

Each template is a 14x14 pattern built from general knowledge of how each
Fashion-MNIST class fills a centred, light-on-dark 28x28 frame. I used
rectangular patches with hand-chosen signed weights to mark three kinds of
cells for each class:

* `+` where the class is **reliably bright** (e.g. T-shirt shoulders at the
  top rows, trouser leg columns, shoe sole at the bottom, bag handle uprights),
* `-` where the class is **reliably dark** (e.g. empty top rows for footwear,
  the centre column gap between a trouser's legs, the hole inside a bag
  handle, the strongly-dark sides of a dress),
* `0` where the class is **shared with others** (upper-garment body) so the
  template is non-committal.

The per-class signature features emphasised:

| Class       | Signature in the template                                         |
|-------------|--------------------------------------------------------------------|
| T-shirt     | wide top shoulders over a narrow body with empty sides            |
| Trouser     | two leg columns + strong dark centre gap in the lower half        |
| Pullover    | broad shoulders + full-width sleeves + body reaches the bottom    |
| Dress       | narrow central column all the way down, strongly dark sides      |
| Coat        | large filled silhouette with full-width sleeves + narrow collar  |
| Sandal      | empty upper half, strappy content extended up to mid, solid sole |
| Shirt       | broad shoulders and sleeves, narrower body at the bottom         |
| Sneaker     | empty top ~3/4, solid compact shoe in the bottom rows            |
| Bag         | handle on top (two uprights with a dark hole) + body, dark sides |
| Ankle boot  | empty top, ankle rising on the right, solid sole                 |

After stacking the ten templates (shape 10 × 196), each row is **zero-centred**
(subtract its mean) so the dot product is invariant to overall image
brightness, then **L2-normalised** so classes with denser templates don't
dominate, then scaled by 10 so scores land in a comfortable range.

## Why this architecture

Fashion-MNIST items are already centred and size-normalised, so a single
14x14 silhouette per class captures most of the distinguishing structure that
is hand-recoverable without training.  A simple linear classifier on the
pooled image makes every weight directly interpretable, keeps the design
tractable, and - crucially - leaves no room for the templates to accidentally
encode statistics of the training set, since every value comes from a
textual description of a class's general shape (`rect(T[k], r1, r2, c1, c2,
val)` calls) rather than from any image or summary derived from images.

No training, fitting, regression, clustering or class averages were used.
