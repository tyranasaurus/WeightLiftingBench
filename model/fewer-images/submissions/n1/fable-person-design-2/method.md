# Method

**Architecture.** A small hand-designed network that looks for compact skin-coloured blobs.

1. A 1x1 conv + relu computes six "limit violated" amounts for a skin-colour cone in RGB
   (R > G > B, hue roughly 3-39 degrees, saturation 0.15-0.72, not too dark). A second 1x1 conv + relu
   turns them into a soft skin map: 1 minus 10 times the summed violations, clipped at 0.
2. The skin map is used at full, 1/2, 1/4 and 1/8 resolution (avgpool). At each a 7x7 centre-minus-ring
   conv (3x3 centre mean minus the outer ring mean) responds near 1 to a compact blob, at most 0.5 to the
   straight edge of a large skin-coloured surface, and 0 inside one. Only the part above 0.5 is kept (relu)
   and the strongest position is taken (global maxpool). The full-resolution scale has half weight.
3. A penalty of 2 x relu(skin share of the picture - 0.30) discounts wooden rooms, food close-ups and sepia photos.
4. The four blob evidences and the penalty are added; a dense layer outputs "person" when the sum exceeds 0.28.

**How the weights were produced.** Every number is typed in `build.py` from general knowledge of skin colour
and face-sized blobs. Nothing was trained, fitted, or measured on training or generated images. The two
training images were only viewed as PNGs; the network was never run on them (0 of the 5 checks used).

**Status.** This is a minimal answer saved when time ran out. The thresholds are untested guesses, so the
accuracy is unknown; planned scene cues (snow, courts, sky) and face / head-shoulder templates were not built.
