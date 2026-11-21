# Method

**Submitted network = the design that was run in check 2 (62.5% on the 200 training photos: 40/100 person photos found, 15/100 false alarms).**

## Architecture (all weights typed by hand in build.py; nothing fitted, no data read)
1. **Skin-colour maps (1x1 convs + relu, full resolution).** Six linear rules from general colour knowledge
   (reddish-orange hue about -12..45 degrees, saturation 0.08..0.80, not too dark, R above G) are combined as a fuzzy AND
   (1 - largest violation, via reshape + maxpool over channels). Four "level-set" variants add one extra demand each
   (bright, very bright, pinker hue, redder), so a face that merges with brown hair or a warm wall in the loose map
   still stands out as a blob in a stricter one.
2. **Face-like blob detector at 5 scales** (avg-pooled pyramids, 13x13 template): mean skin density inside an ellipse
   minus the density in the top / left / right ring sectors (bottom sector, i.e. neck, at half weight), best of the
   5 skin maps. Ring area outside the picture counts as half skin.
3. **Support cues added to the blob score:** an eye band (luminance: eye row darker than forehead/cheeks, 3 vertical
   shifts) and head context (box above the head or at the shoulders differs in colour from the boxes beside the head;
   boxes that leave the picture give no credit). A blob low in the frame is discounted (tables, floors, food).
4. **Upright figure with skin** at 3 scales: box model of a standing person (trunk vs. left/right background, legs,
   nothing continuing above the head, some skin on the figure, torso not all skin).
5. **Decision:** best face evidence + 0.8 x best figure evidence - a small penalty when most of the picture is
   skin-coloured (wood, food, warm light) - 0.5; output = [0, 10 x that].

About 272k weights (43k non-zero).

## The 5 allowed checks (4 used) and what each taught me
- Check 1, 55.0% (12 found / 2 false alarms): first version worked but was far too strict on real photos.
- Check 2, 62.5% (40 / 15): looser skin gamut + level sets fixed most of that. **This is the submitted version.**
- Check 3, 62.0% (40 / 16): an added colour-agnostic "island figure" cue never fired on real people; removed.
- Check 4, 59.0% (33 / 15): stricter context/eye logic plus hair-cap and dark-background cues lost 7 detections and
  removed no false alarms; reverted.
- Check 5 was not used. A further variant (blob ramp 0.20..0.50) was prepared but never run on real photos, so it is
  not submitted.

## Honest limits
- Expect roughly 60-63% accuracy on the held-out set (the 200-photo estimate has a standard error of about 3.4 points).
- It relies almost entirely on visible skin: people in snow gear or wetsuits, seen from behind, in black-and-white
  photos, or very small are missed (60 of 100 person photos in check 2).
- Images drawn by code were used only to confirm that each circuit computes what it was meant to (and to fix logic
  bugs found that way); no threshold was set from them.
