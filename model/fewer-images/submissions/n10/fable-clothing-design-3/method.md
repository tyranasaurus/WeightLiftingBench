# Method

**Architecture.** `image -> norm -> dense (30 templates) -> reshape (1, 10, 3) -> maxpool over the 3 templates
of each class -> reshape (10,)`. The `norm` layer removes each image's brightness and contrast, so the dense
layer's outputs are (scaled) Pearson correlations between the image and each template; the score of a class is
its best-matching template. Total 23,550 weights.

**How the weights were produced.** No training, fitting or statistics of any images. `build.py` draws 30
silhouettes (3 per class) from polygons and ellipse rings whose coordinates I typed from general knowledge of how
these items look in a shop photo that fills a 28 x 28 frame: a T-shirt with short sleeves at the shoulders or a
tank top; trousers as two tall legs with a gap; a pullover with long sleeves angled away from the body; a dress
with a narrow top and widening skirt; a coat as a long block with a collar and sleeves hanging against the body;
a sandal as a thin sole with straps (flat or high-heeled); a shirt with an open V collar; a sneaker as a low wedge
with the toe on the left; a bag as a box with a handle loop; an ankle boot as a shoe with a tall shaft at the
right. Each drawing is anti-aliased, lightly blurred with a binomial kernel (shift tolerance), and then made
zero-mean and unit-norm by itself alone to become one row of the dense layer. Biases are zero.

**Use of the training images.** I viewed the 100 training images as PNG montages to see what the classes look
like (orientation of shoes, how much of the frame items fill). No pixel values, coordinates or measurements from
them were used. The design was written once and not tuned; time ran out before any refinement, so the template
set is a first draft (the upper-body classes 0/2/4/6 are expected to be confused with each other).
