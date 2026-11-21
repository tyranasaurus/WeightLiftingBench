# Method

A small hand-designed network; no weight was trained, fitted or taken from any image.

**Architecture**

1. `viol` / `skinp` (1x1 convs + relu): a per-pixel skin-colour rule typed from general knowledge
   (R > G >= B, R-G between 0.05 and 0.38, G-B below 0.24, not dark, hue not too yellow).
   Each relu unit measures the violation of one linear constraint; `skin = relu(1 - sum of violations)`.
2. `skin2`, `skin4` (avgpool): the skin map at 48x48 and 24x24.
3. `blobS`, `blobA`, `blobB` (convs + relu): centre-surround filters from a formula, a disc of mean +1 minus a ring
   of mean -1 (ring from 1.5 to 2.3 disc radii), at disc radii of about 2, 3, 4.5, 7, 10 and 15 pixels, with a
   hand-chosen offset of -0.30. They respond to compact skin-coloured blobs (faces, hands, limbs) that stand out
   from their surroundings, and not to large uniform skin-coloured areas such as wood or sand.
4. Global maxpool per scale, then dense layers and an `add`: person score = sum over scales of the strongest blob
   response minus 0.15; the no-person score is 0.

**How the weights were produced**

All constants are written by hand in `build.py` (skin-colour bounds, filter radii, the 0.30 and 0.15 thresholds).
The training images were only viewed as contact sheets to see what the classes look like; no statistic of them
was computed or used.

**Limitations**

This is a minimal design saved under a time limit: it has a single cue (skin-coloured blobs). It cannot see people
in black-and-white photos, people seen from behind or fully clothed, and it will fire on skin-coloured objects
(food, wood, tan animals). The thresholds were chosen by reasoning, not tuned.
