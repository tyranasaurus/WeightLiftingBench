# Method

Architecture: image -> 3x3 conv (10 channels) -> relu -> 9x9 average pool, stride 7 (a 4x4 grid of cells) -> dense (160 -> 10).

The conv channels are typed formulas: six per-pixel colour categories (blue, green, warm, red, bright, dark) and
Sobel luminance edges in both polarities (vertical and horizontal). The dense layer is a hand-written rule table:
for each class and feature, one weight for each of four zones of the 4x4 grid (top row, sides of the middle rows,
centre, bottom row), chosen from general knowledge of the classes: sky all round an airplane, water below a ship,
grass around deer and under horses, warm fur indoors for cats and dogs, green/brown dark texture for frogs, painted
bodies with long edges and a dark underside for cars and trucks.

No training, fitting, statistics of the training images, or generated images were used; I only looked at contact
sheets of training images to recall what the classes look like. Time ran out before the planned larger design
(a bank of synthetic-scene templates) was built, and none of the 5 checks on training images was used, so the
accuracy of this network is unmeasured.
