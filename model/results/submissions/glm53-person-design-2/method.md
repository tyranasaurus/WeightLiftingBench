# Method

**Architecture.** A hand-designed, threshold-cascade detector for faces/heads,
built only from the eight allowed blocks. Per pixel, 1x1 convs compute Y, Cr,
Cb and R-G, and relu "bump" units turn the chroma into triangular membership
functions (windows from the published YCbCr skin ranges, centred lower because
shrinking a photo to 96x96 desaturates skin). From these: a skin map s, a
strong-skin map ss (saturated membership sum above 1), a dark-spot map d
(5x5 centre-surround ring contrast: eyes/mouth/hair are darker than their
surroundings), a grey map g (near-zero chroma, for black-and-white photos),
and a local skin density sA (9x9 mean of s — a face is an *isolated* warm
blob, a wall or wooden surface is not). These are computed at 96, 48 and 24 px
(avgpool pyramid). For each of 12 head widths (4..60 px at 96-scale) a fixed
kernel encodes the geometry of a head: skin oval (forehead discounted for
hair), strong-skin bonus, eye band and mouth on the dark map, a nose-bridge
penalty (the bridge between the eyes is skin, not shadow — this rejects
flowers/wood-knot patterns), a dark hair band above, negative skin weight on
side margins, and an isolation penalty on the surroundings. Template responses
are relu-thresholded, max-pooled over position, grouped into co-localised
scale groups (adjacent scales must fire at the *same* position), plus
strong-skin block cues, skin fractions, and a colourful-photo discount
(food/flower photos saturate more than photos of people, whose palette is
skin/hair/clothing neutrals). A final dense layer with hand-set weights gives
the two class scores.

**How the weights were produced.** Every weight is a formula or a hand-chosen
constant in build.py: chroma windows from the skin-detection literature, face
geometry (eye band ~42% down the face, mouth ~75%, head aspect 1.35:1) from
general knowledge, centre-surround and isolation weights from judgement. No
training, fitting, or data-derived numbers are stored anywhere. I calibrated
and debugged the design on my own synthetic cartoon drawings (faces, wood,
sand, flowers, dogs) which I rendered and ran freely, and I used my five
allowed runs of the network on training images as coarse feedback: run 1
(labeled, 4000 images) showed a sum-of-maxes readout accumulates background
noise; runs 2-4 (diagnostic, saving score statistics) showed that on real
96x96 photos the *maximum* template response and the warm/strong-skin
fractions do not separate the classes — real faces are desaturated by the
shrink+JPEG as much as warm objects are — which led me to the isolation,
nose-bridge and co-localisation mechanisms and to a conservative readout
threshold. Run 5 (labeled, all 20000 images) verified the final network.
The architecture is deliberately a chain of hard thresholds and fixed
formula filters rather than a generic trainable CNN: its behaviour comes
from the hand-set values, not from capacity.
