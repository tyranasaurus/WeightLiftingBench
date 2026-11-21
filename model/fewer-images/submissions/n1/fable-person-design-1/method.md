# Method

A small hand-designed network (489 weights); every number is typed in `build.py` from general knowledge. Nothing was trained, fitted or measured on any image.

**Architecture**

1. **Skin-colour membership (1x1 convs).** Seven linear rules on (R, G, B) describe skin: red dominant over green, not pure red, not blue or purple, hue below about 30 degrees (to reject yellow-brown wood, sand and fur), not too dark, not over-saturated, not grey or white. Each rule's violation passes through a relu; membership is `relu(1 - sum of violations)`.
2. **Compact-blob detectors (9x9 convs on pooled skin maps).** The skin map is average-pooled into 4 px cells (step 2) and 8 px cells (step 4). Centre-minus-ring kernels at two sizes on each level respond to isolated skin patches about 4, 8 and 16 px across (faces, hands, arms). The 0.5 threshold was chosen analytically so that straight edges of large skin-coloured surfaces stay silent.
3. **Decision (global average pooling, two dense layers).** Blob evidence B is expressed in units of "about one clean blob", and f is the mean skin membership of the whole photo. The person score rises as B goes from 0.3 to 1.5 and falls as f goes from 0.12 to 0.30, so a photo flooded with skin-coloured pixels (wood, food, warm interiors) is not called a person. The no-person score is fixed at 0.

**Checks**

One of the 5 allowed checks was used, on the 2 training images. It showed that the network runs end to end and gets both right, which says nothing about held-out accuracy. No number was changed after it.

**Limitations**

This is a single-cue design saved when time ran out. Planned cues for scene context, clothing colour and upright-figure shape were not built. It will miss people with no visible skin (skiers, back views) and will fire on compact skin-coloured objects such as food items and brown animals.
