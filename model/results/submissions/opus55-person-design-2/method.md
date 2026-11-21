# Method

**Architecture.** The network is a hand-wired detector for heads and bare limbs, plus two scene priors. Output 0 is always 0, and output 1 is the summed person evidence minus a threshold of 0.10. It has about 64.6k weights.

1. **Colour memberships (1x1 convs + relu).** Hue and saturation thresholds are ratios of R, G and B, so each one is a linear half-space in RGB. I wrote fuzzy ANDs of such half-spaces from general knowledge of photographed skin:
   - `skin_core`: hue 4–28°, saturation 0.20–0.64, R ≥ 0.2–0.3.
   - `skin_loose`: anything skin-like, such as wood, tan fur, bread or blond hair. Hue up to 42°, saturation 0.1–0.8, not dark.
   - darkness: 1 − luminance.
   - daylight snow: bright, near-grey, with blue ≥ red.

   The AND is the Łukasiewicz t-norm relu(1 − Σ relu(1 − zᵢ)).
2. **Pyramid.** Binomial blur-and-subsample convs give levels 48, 32, 24, 16, 12, 8, 6 and 4. Each level also has a copy on a grid shifted by half a pixel, so blobs that straddle grid cells are still caught.
3. **Part templates at every level.** These are 7x7 convs whose weights are the area fractions of shapes, supersampled.
   - **Head:** skin over the lower face ellipse, plus a capped bonus when the top of the head is darker (hair). The head is vetoed when the ring sector above it is skin-coloured, or when the sectors to the left and right are skin-like. That rules out pieces of larger skin-coloured surfaces. The sectors read the non-skin maps, so the zero padding outside the photo counts as skin-like and big beige or wooden areas can't pass at the photo's corners.
   - **Limbs:** skin bars in 4 orientations, with non-skin on both long sides.
4. **Readout.** I take the max over positions of each response, by thirds of the photo, and the max of the plain and shifted grids. These are summed with hand-set weights:
   - heads in the bottom third count 0.6;
   - limbs count 0.5;
   - snow covering more than 30% of the lower two thirds adds evidence (skiing);
   - a small prior (+0.25) is added for true greyscale photos, where colour cues can't see skin.

**How the weights were produced.** Every number is typed in or computed by a formula in `build.py`: half-spaces, binomial kernels, ellipse, ring-sector and bar masks, and hand-chosen thresholds. Nothing is fitted to data, and no statistic of the training images or labels is used. I used training images only by viewing image grids. I debugged the components on synthetic, formula-drawn images: colour swatches, drawn faces and skin squares.

**The five checks.** I used the confusion tables to make reasoned design changes, not to search over variants:
1. Face blobs alone fired almost everywhere: 55.7%.
2. A strict head detector with a hair requirement was precise but caught little: 57.2%.
3. After fixing dark-brown hair counting as skin-like and adding graded evidence and limbs: 60.2%.
4. With a lower threshold (the measured trade-off was still about 2:1), a softer side veto and the greyscale prior: 64.5%.
5. Final, after the border fix, on 12,000 untouched images: **64.1%**, catching 50% of person photos with 22% false positives.
