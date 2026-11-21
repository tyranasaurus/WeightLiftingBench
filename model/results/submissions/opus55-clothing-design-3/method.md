# Method

**Architecture.** It has two branches whose class scores are added at the end.

1. *Shape branch:*
   - A 1x1 conv + relu turns the image into a soft silhouette mask (grey 0.02 → 0, grey 0.06 → 1) plus the raw intensity.
   - A dense layer scores 750 hand-made *shape prototypes* with a per-pixel log-likelihood: Σ m·logit(P) + log(1−P), where P is the prototype's foreground probability (clipped to [0.06, 0.94]).
   - A reshape and maxpool keep the best prototype of each class.
2. *Cue branch:* three convs (13x13, 5x5, 7x7) with relus compute local interior cues from formulas:
   - a centre vertical line, i.e. a zip, front opening or button placket (dark or bright, averaged over 5 rows);
   - line energy inside the garment in the vertical and horizontal directions; "both directions" (min via relu) means a check pattern;
   - background lying between the item on its left and right, and/or above and below, within 6 px. A version that must run for 7 rows in the centre columns is the trouser leg gap. A version enclosed on all four sides marks the spaces between sandal straps.

   Two dense+relu layers turn these sums into clamped ramps. A last dense layer turns the ramps into hand-set class bonuses: coat +45·line; shirt +20·line +15·texture +45·plaid; trouser +50·leg-gap; sandal +20·wide holes +40·strap pockets.

**How the weights were made.** No weight is fitted or computed from data.
- The prototypes are polygons I typed in normalised item coordinates, drawn from general knowledge of the garments and from looking at example images: two-legged trousers; A-line, sheath and shift dresses; T-shirts with angled short sleeves; tank tops; long-sleeved tops with domed shoulders and hanging sleeves; shirt-tail shirts; sneakers; ankle boots with a shaft at the right; flat and heeled sandals with sparse straps; and bags with an optional handle loop.
- Each polygon is drawn at several aspect ratios, then placed the way the shop pipeline places items (longest side 28, centred), rasterised by supersampling and blurred by 0.9 px.
- Pullover, coat and shirt share one long-sleeved shape family, with hand-set offsets (coat −20, shirt −25), so the interior cues decide between them.
- All thresholds, offsets and bonus sizes were chosen by hand.
- `build.py` regenerates everything from its own source and reads no files.

**Checks.** I used all 5 allowed runs on the 60,000 training images, each followed by changes based on what the confusion table showed:

| Check | Accuracy | Change made afterwards |
|---|---|---|
| 1 | 58.8% | Long-sleeved tops redrawn with domed tops, T-shirt neck notch removed, leg-gap cue required to run 7 rows |
| 2 | 63.9% | One shared long-sleeved family for pullover, coat and shirt |
| 3 | 67.2% | More sensitive line cue, plaid cue, strap-pocket cue |
| 4 | 68.0% | Long-coat shapes removed, fitted tees added, a shoulder-slope rule tried |
| 5 | 67.2% | The shoulder-slope rule sent most pullovers to coat, so I set its offsets back to check 4's values |

The submitted network is the check-5 network with that one change of offsets. It was not run on images again, so its accuracy has not been measured.

**Other tests.** Plumbing and cue arithmetic were tested only on synthetic drawings I made from the same polygons and on toy patterns (a ring, stripes, a check, a zip line), never on training images.
