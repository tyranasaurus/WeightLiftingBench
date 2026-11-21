# Method

**Architecture.** A hand-built feature CNN:
`image -> conv c1 (20 filters, 5x5) -> relu -> avgpool 2x2 -> (20,14,14) -> dense fc1 (46 evidence units) -> relu -> dense fc2 (10 class scores)`.
181,356 weights in total. Nothing was trained, fitted, or copied from data: every
number comes from a formula or a hand-written table in `build.py`.

**Layer 1 filters (formulas only).** An identity ("fill") channel whose pooled
value over a region is literally the fraction of that region covered by the
garment; a Gaussian blur; derivative-of-Gaussian edge filters at 0/45/90/135
degrees in both polarities (normalised so a unit step gives 1); bright-bar and
dark-bar detectors for 2-3px strokes (a trouser leg, a placket, a sole line),
normalised so an ideal 3px bar gives 1; and a LoG dark-hole detector for a bag's
handle loop. The scorer's own `avgpool` then gives calibrated region means on a
14x14 grid, where each cell is 2x2 pixels.

**Layer 2 (fc1) - evidence concepts.** Each unit is a hand-chosen weighted sum of
region means / stroke energies plus a hand threshold, kept positive by relu. They
encode general knowledge of garment geometry: shoulders vs straps in the top rows
(`topFill`, `topWide`), sleeve length below mid-torso (`sleeveLong`), the trouser
signature `legGap` (bright legs either side of a dark centre gap), the A-line
`flare`/`diagFlare` of a dress, the bag's `wideMid`/`wideBody` (solid body, no
shoulders) and `holeSur` (bright bar over a deep dark loop), shoe structure
(`soleBar` sole line, `shaftL/R` boot-shaft asymmetry, `sparse` + `strapE` +
`shortShoe` for sandals, `shoeUpper`/`shoeMid` sneaker bands), shirt/coat detail
(`collar` notch, `placket` centre line, `hemRib` pullover hem band), and absence
evidence such as `topEmpty`/`botEmpty` (shoes leave the top and bottom rows of
the frame empty because the photo is scaled to the longer side and centred).

**Layer 3 (fc2) - rule table.** Each class score is a hand-written weighted vote
of the evidence units. Garment classes require ink at the top and bottom of the
frame and penalise sole lines and sparseness; shoe classes penalise solid
garment-like fill; the four top classes (t-shirt / pullover / shirt / coat) are
separated mainly by sleeve length (t-shirt sleeves end by mid-torso), collar +
placket lines (shirt), and a long front opening with a full-frame body (coat) -
these three are genuinely similar, so their boundary was deliberately left
balanced rather than over-tuned.

**How the weights were produced.** I wrote the filters from formulas, chose
regions/thresholds from general knowledge of how these garments are shaped, and
sanity-checked everything with two data-free testbenches in `/app/scratch`: (1) a
hand-typed table of the concept values I expect for a typical member of each
class, pushed through the rule table (the diagonal must win); (2) idealised
silhouettes drawn from general knowledge (rectangles for body and sleeves, an
A-line skirt, a leg gap, a handle loop, sole lines) run through the scorer's own
interpreter. I used my 5 allowed runs on a 10,000-image stratified subset of the
training set and read the confusion tables to find where my hand-set thresholds
were wrong (e.g. real garments are dimmer than a naive calibration assumes, the
trouser legs sit at particular columns, sandals are chunkier than expected);
after each check I corrected the offending constants by reasoning and re-verified
with the data-free testbenches.
