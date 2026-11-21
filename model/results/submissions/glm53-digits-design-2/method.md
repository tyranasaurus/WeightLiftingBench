# Method

**Architecture.** One `conv` layer holds 480 hand-drawn digit templates (48 per
class: 8 handwriting styles x slants x 2 stroke widths; 6 styles x 4 slants for
"1"). A `maxpool` over the conv's 7x7 output takes each template's best
alignment (padding 3 gives a +-3 px search, small enough that every window
still contains the whole digit), a `reshape` + `maxpool` takes each class's
best template, and a hand-set rule-table finish (norm -> relu threshold ->
`dense` with -0.22 on classic confusable pairs, added back to the normalised
scores) sharpens the winner's margin between look-alike classes such as 8/9,
5/6 and 1/7. No weight was fitted: the templates are pen paths I typed from
general knowledge of how people write digits, and the pair table, threshold
and gamma are hand choices.

**How the weights were produced.** Each style is a list of pen strokes in a
20x20 box coordinate system, rendered to 28x28 with anti-aliased round-cap
segments. Every render is auto-scaled so its *ink* spans exactly 20 px
(mirroring the box-fit normalisation of the data), re-centred so its centre of
mass sits at the frame centre (as the data is), then zero-meaned and
L2-normalised, so a conv output is the projection of the image ink onto the
template direction and is comparable across all 480 templates. I studied text
renderings of training images (as the rules allow) to check qualitative
proportions — loop aspect ratios, slant ranges, stroke width, where the waist
of an 8 or the crossbar of a 4 sits — and adjusted my typed paths accordingly;
no pixel values, statistics or labels of training images entered the weights.
The five allowed `run_network` checks (4000 images each) were used to verify
the design end-to-end and to catch two systematic bugs (templates ~10% too
large because ink-vs-centreline extent was confused, and stroke-width/
slant-coverage gaps); the submitted configuration is the best verified one
(64.3% on a held-style subset), plus two changes justified a priori (thinner
stroke widths, whose support is covered by any thicker real stroke, and one
extra narrow-8 style). A synthetic-image test (my own renders with jittered
shift, slant and scale) passes 57/58 and the data-free rebuild check passes
with zero difference.
