# Method

Hand-designed network, no training or fitting; every constant is typed from general knowledge and written by build.py.

- **Per-pixel colour rules (1x1 convs + relu).** A soft "skin" map from a cone of linear rules in RGB
  (hue roughly 6-36 degrees, saturation 0.15-0.65, not dark), and a soft "snow" map (bright, neutral-to-bluish white).
  Each rule's violation is a relu unit; the map is relu(1 - sum of violations). Both are average-pooled to 48 x 48.
- **Compact skin blobs.** Centre-disc-minus-surrounding-ring filters at four radii (face / limb sized) on the skin map,
  thresholded at 0.55 (a straight edge of a large skin-coloured surface gives at most 0.5), then maximum over
  top / middle / bottom bands, the bottom band counted half (floors, tables and ground lie there).
- **Against:** a large overall skin-coloured fraction (wood, sand, warm lighting) subtracts.
- **For:** snow-like white filling the lower half of the picture (ski and snowboard scenes) adds.
- The four branches each end in a dense layer giving (2,) and are summed; the default with no evidence is "no person".

Only the simple first design was finished in the time available: planned face-structure, figure-shape and further
scene cues were not built. The network was not run on the training images (no checks used).
