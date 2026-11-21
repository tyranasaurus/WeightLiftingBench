# Method

## Architecture

The network has 875 weights, all typed by hand in `build.py`. It works on 2x2 block averages of the photo (48x48). The JPEG files only keep colour at about that resolution.

1. **Colour boxes.** 1x1 convolutions compute soft membership in boxes in chromaticity space (r, g, b = R/S, G/S, B/S). Each box edge is a brightness-independent linear rule, and membership is `relu(1 - sum of relu(rule violations))`. There are five boxes:
   - a **skin** box (r 0.39–0.52, g 0.285–0.335, b 0.20–0.295, R > G > B), taken from the published skin-colour clusters of web photos;
   - three **matched control** boxes for the warm colours just outside it: the yellow side (light wood, bread, cardboard), the orange/brown side (dark wood, fur) and the red side (brick).
   - a **snow** box (bright, neutral to bluish white).
2. **Local net skin.** On 8x8-pixel cells, skin share minus control share minus a 0.2 margin, relu, then averaged over the image. Materials whose colour spreads around the skin box cancel against their control. Real skin, whose colour cluster is tight, does not.
3. **Compact skin patches.** A 5x5-cell template is applied on cells of 4, 8 and 16 pixels. It scores the centre cell's net skin minus twice the warm share of the ring two cells away. The results are summed as counts. A face, hand or arm against hair, clothes or background counts; large warm regions such as walls, tables and animal bodies don't.
4. **Scene and photo cues:**
   - a skin share above 40% counts against a person (it suggests a skin-coloured wall or object);
   - a black-and-white test (the largest colour difference in any block is near zero);
   - a snow-underfoot cue: snow covers most of the lower half while the top quarter is *not* also white.
5. **Decision.** The evidence units are relu ramps. They feed a hand-weighted log-odds sum, with the class-0 score fixed at 0.

## How the weights were produced

Every threshold and weight is typed from general knowledge: skin-colour literature, which materials sit next to skin in colour, face sizes at 96 px, and the fact that skiing is a common photo subject. Nothing is fitted, averaged or measured from training images.

Synthetic drawings were used only to check that each mechanism computed what it was meant to (for example, that uniform wood gives no evidence and a drawn face does). They were never used to pick numbers.

## Use of the 5 checks (all on the full 20,000 training images)

| Check | Design | Accuracy | What it showed |
|---|---|---|---|
| 1 | Multi-scale face templates (skin face + dark hair band, eye band), max over positions | 0.509 | The best template match over thousands of positions fires on almost every photo (86% of no-person photos). |
| 2 | A stricter eye-pair face template | 0.510 | It fires equally on both classes (37% vs 38%): faces at 96 px rarely show eye structure, while skin-coloured clutter does. |
| 3 | Regional skin evidence with matched warm-colour controls | 0.570 | Real but weak signal (54% of person photos vs 40% of no-person photos). |
| 4 | Plus compact-patch counting | 0.622 | False alarms fell from 40% to 31% while person detection held. |
| 5 | Plus the snow-scene cue (final) | 0.627 | About 145 more person photos caught for about 29 extra false alarms. |

Design changes between checks came from reasoning about why the previous design failed. No thresholds were swept.
