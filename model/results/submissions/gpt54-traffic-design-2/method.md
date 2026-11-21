I saved a hand-built template-matching network. The build script draws idealized traffic sign prototypes from general knowledge using fixed geometric primitives: circles, triangles, an octagon, arrows, digits, and simple pictograms.

For each of the 43 classes, I made four aspect-ratio/scale variants. The network has two dense branches on the full 32x32x3 image: one matches raw color templates after subtracting a fixed gray background, and the other matches channel-normalized templates to reduce lighting sensitivity. Their scores are added, reshaped to 43 classes x 4 variants, and max-pooled over the 4 variants so each class keeps its best-matching prototype.

The weights were produced directly from those hand-coded templates, with no fitting or training.