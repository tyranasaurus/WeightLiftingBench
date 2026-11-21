# Method

**Architecture.** A hand-built template-matching network, 2,378,514 weights.

1. `reshape` the image to one channel of 96 x 32 and apply `norm`, so all three colour channels share one mean and
   one gain (removes overall brightness and contrast), then `reshape` back to (3, 32, 32).
2. A 1 x 1 `conv` turns RGB into brightness (R+G+B)/3 and two colour-opponent channels,
   red minus cyan and blue minus yellow (each weighted 1.3).
3. A 32 x 32 `conv` with padding 2 correlates the whole frame with 774 templates at 5 x 5 shifts (-2..2 pixels).
4. `maxpool` over the 25 shifts, `reshape` to (1, 43, 18), `maxpool` over the 18 variants of each class
   (9 sign sizes, box half-size 9.5 to 13.5 pixels, times 2 blur levels), `reshape` to the 43 class scores.

**Where the weights come from.** `build.py` draws each of the 43 signs from general knowledge of German road signs:
a red ring with digits drawn as strokes, red-bordered triangles with simple stroke pictograms, blue discs with white
arrows, the red octagon with STOP, the yellow diamond, the red disc with a white bar, and the grey "end of" discs
with a diagonal band. Each drawing is rasterised at 4x supersampling, blurred with a Gaussian, converted to the same
three channels, masked to the sign's own silhouette (the background gets zero weight), made brightness-offset
free inside the mask, and scaled to unit length. That is the filter for one template. All sizes, colours and the
channel weight were chosen by hand; no number comes from the training images, and nothing was trained or fitted.

**Status.** The work was cut short: this is the first version of the design, with no tuning. Sign geometry and
pictograms are rough first drafts. One of the 5 allowed checks was used, on every 8th training image (4,902
images): accuracy 0.5341. It showed that the network runs and that the untuned templates are far above chance
(1/43) but far from solved. Nothing was changed after that check.
