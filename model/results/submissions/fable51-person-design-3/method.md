# Method

## Architecture (121 layers, 63,472 weights, all hand-set)

The network is a fixed pipeline of hand-designed detectors, written entirely from general
knowledge of how people look in photographs; no weight was measured from, fitted to, or copied
from any image.

1. **Pixel colour maps** (two 1x1 convs with ReLU). Linear colour features (YCbCr chroma,
   luminance, channel differences) are turned into soft indicators of the form
   `relu(1 - sum of violations)`, each violation being a ReLU ramp past a hand-chosen threshold.
   Seven maps result: skin (classic YCbCr skin box, not too dark, not pink, hue below ~30 deg),
   luminance, darkness, bright neutral white, saturated warm colour, green, blue.
2. **Cell pyramids** (average pooling): overlapping cells of 6/8/12/16/32 px at half-cell stride,
   and non-overlapping cells of 3/4/6/8/12 px, plus whole-image fractions.
3. **Detectors** (5x5 or 9x9 conv -> ReLU -> 1x1 conv -> ReLU -> global max), each an AND of
   clipped ramps: a compact skin blob (centre cell minus its surrounding ring); a two-eye face
   (two dark cells beside a brighter nose bridge, bright skin cheeks below, a bright forehead or
   dark fringe above, a thin band); a head as a dark hair/hat strip above a 2:3 skin column with
   non-skin sides and some darkness at eye level; narrow vertical figures against snow, water or
   grass with a vertical-isolation check so poles and door frames are rejected; a compact dark
   blob on water (crouching surfer); thin bright ridges on coloured ground integrated over 9 px
   (court lines).
4. **Readout**: placement dense layers drop each branch's maximum into named slots; a shared
   ReLU layer forms clipped ORs across scales and whole-image gates (a figure on snow only counts
   if the picture is largely white, etc.); a second dense+ReLU adds penalties for too much
   skin-coloured, saturated-warm or white area, a large skin blob on grass (animal) and brown food
   on a white plate; the last dense layer sums the evidence with hand-chosen weights against a
   fixed threshold.

## How the weights were chosen

All thresholds, templates and readout weights were typed by reasoning about the classes
(e.g. skin chromaticity ranges, hair occupying ~30% of head height, eye separation ~0.4 of face
width, skiers as dark narrow blobs on white). The detectors were unit-tested only on synthetic
drawings I generated in code (a face with two eyes and hair, a wooden plank, a skier on snow, a
court with white lines, a brown dog on grass, bread on a plate) to confirm that each fires where
intended. The five permitted checks on training images were used as follows:
check 1 (first full version) 54.5%, far too many positives; check 2 (skin/face cues only) showed
the first face template fired on 36% of non-person images; check 3 (context cues only) showed the
sports-scene cues were clean but rare (6.6% / 3.9%); check 4 (redesigned two-eye face and
hair-cap head) 56.0%; check 5 (added negative scene cues) 55.6% on the same 5,000-image subset
(recall 33.7%, false positives 23.8%). No statistics of the training images or labels were
turned into weights; build.py regenerates the identical files from code alone.
