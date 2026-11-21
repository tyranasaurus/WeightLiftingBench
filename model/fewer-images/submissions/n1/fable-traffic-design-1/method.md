# Method

**Architecture.** A template-matching network with no trained parts:

1. `feat` – 1x1 conv turning RGB into three hand-chosen channels: luminance, red opponent (R − (G+B)/2) and blue opponent (B − (R+G)/2).
2. `featn` – `norm`, so each channel is standardised per image (copes with dark and washed-out photos).
3. `match` – one 32x32 conv kernel per class and per sign size (43 classes × 4 sizes = 172 kernels), applied with padding 3 so each template is tried at 7×7 shifts.
4. `shift` – maxpool over the 7×7 shifts; `grid`/`size` – reshape and maxpool over the 4 sizes; `scores` – reshape to the 43 class scores.

About 528,000 weights in total.

**Where the weights come from.** `build.py` draws each of the 43 signs from general knowledge of the German sign catalogue
(red ring with stroke-drawn digits, red-bordered triangles with pictograms, blue discs with white arrows, the yellow priority diamond,
the stop octagon, the "end of" signs with a diagonal band, and so on) at four sign sizes. Each drawing becomes exactly one template: it is
converted to the same three channels, mean-removed and scaled using that drawing alone, restricted to the sign's own area (the unknown
background gets zero weight), and given an extra luminance term on the inner symbol so classes that differ only in the symbol are separated.
All constants (colours, stroke shapes, sizes, the weight of the inner-symbol term) were typed by hand.

No training data, statistics of the training images, or fitting of any kind was used; nothing was tuned on generated images.

**Status.** The design was written in one pass with no tuning. One of the five allowed checks was used: 27 of the 43 training
images (62.8%) were classified correctly, which showed that the network runs end to end and that the drawn templates do match real
signs. Nothing was changed after that check. The drawings were never inspected visually and no per-class errors were analysed, so
this is a first working version rather than a refined one.
