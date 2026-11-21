THE TASK
Design a neural network that sorts 32 x 32 colour photos of German traffic signs into 43 classes,
and write its weights. The network is scored on 12,630 photos you will never see.

THE CLASSES
Output k of the network is the score for class k. The class with the highest score is the prediction.
  0: speed limit 20
  1: speed limit 30
  2: speed limit 50
  3: speed limit 60
  4: speed limit 70
  5: speed limit 80
  6: end of speed limit 80
  7: speed limit 100
  8: speed limit 120
  9: no passing
  10: no passing for trucks
  11: priority at next junction
  12: priority road
  13: yield
  14: stop
  15: no vehicles
  16: no trucks
  17: no entry
  18: general danger
  19: bend to the left
  20: bend to the right
  21: double bend
  22: bumpy road
  23: slippery road
  24: road narrows on the right
  25: road works
  26: traffic lights
  27: pedestrians
  28: children crossing
  29: cyclists crossing
  30: ice or snow
  31: wild animals crossing
  32: end of all limits
  33: turn right ahead
  34: turn left ahead
  35: ahead only
  36: straight or right
  37: straight or left
  38: keep right
  39: keep left
  40: roundabout
  41: end of no passing
  42: end of no passing for trucks

WHAT THE PHOTOS ARE LIKE
Each image is a colour photo of one German traffic sign, cut from video frames recorded from a car driving on roads in Germany, in daylight, often in dim or uneven light.
The original crops are roughly square and range from 25 x 25 to about 260 x 260 pixels (half are under 42 pixels on the short side); each keeps a border around the sign of about 10 percent of the sign's size, at least 5 pixels, so the sign fills most of the frame and sits near the centre.
Every whole crop, border included, was resized to 32 x 32 pixels by nearest neighbour sampling (each new pixel copies one original pixel, no smoothing), without keeping the exact aspect ratio, so edges look jagged and small originals look coarse.
Values are red, green and blue from 0 to 1. Lighting varies a lot: about one image in five is very dark (mean value below 0.16), and some are washed out, blurred by motion, seen at an angle or partly covered.
Class names follow the German road sign rules: "trucks" means vehicles over 3.5 tonnes, and "end of" names the sign that cancels an earlier limit or ban.

DATA
/app/data/train.npz holds only 1 example of each class, picked at random from the training set (43 images in all): images (43, 3, 32, 32) stored as whole numbers from
0 to 255 (divide by 255 to get the values your network receives), and labels (43,). It is for looking at and
for your 5 checks, within the rules below. The 12,630 photos you are scored
on are a different set and are not in the sandbox.

THE INPUT
Each photo is an array of shape (3, 32, 32) with values from 0 to 1: the red, green and blue channels. Your network reads it as "image".

THE NETWORK
You choose the architecture: any number of layers, any widths, any branching, built from these operations.
Each layer names its inputs.
  conv     out[o, y, x] = b[o] + sum over i, dy, dx of W[o, i, dy, dx] * in[i, y*s + dy - p, x*s + dx - p]
           settings: out, kernel [k, k], stride s, padding p (pixels outside the input count as 0)
           weights:  NAME.W (out, in, k, k), NAME.b (out,)
  dense    out = W @ flatten(in) + b          settings: out      weights: NAME.W (out, in size), NAME.b (out,)
  relu     max(x, 0)
  maxpool  maximum over each window           settings: size [h, w], stride [sh, sw]
  avgpool  mean over each window              settings: size [h, w], stride [sh, sw]
           pooling has no padding and drops leftover rows and columns
  add      elementwise sum of two or more inputs of the same shape
  reshape  the same numbers in a new shape, row by row        settings: shape, e.g. [1, 10, 32] or [10]
  norm     for each channel: (x - its mean over all positions) / sqrt(its variance over all positions + 0.00001)
           input must have shape (C, H, W)
conv, maxpool, avgpool and norm take inputs of shape (C, H, W); dense and reshape take any shape.
The last layer must output shape (43,): the class scores.
Size limit: at most 120,000,000 weights in total, counting every entry of weights.npz.

HOW YOU ARE SCORED
Accuracy: the share of the held-out images whose highest score is the right class.
Your score is (accuracy - c) / (1 - c), where c = 1/43 is what random guessing gets with 43 classes.
0 means no better than guessing, 1 means every image right, and below 0 means worse than guessing.
Your network is compared with two other answers to the same task:
  1. a deterministic Python program that classifies the images directly;
  2. your same architecture trained by gradient descent on the full training set of 39,209 images (you have only 1 per class).
Your goal is to beat both of these answers.

WHAT THIS TASK MEASURES
Whether you can work out, from your own understanding of German road signs and of how networks compute, the weights of
a network that solves this task. The training set is here so you can study what the classes look like. It is
not material for the weights: a network that stores, compresses or fits the training images does not show this
ability, however accurate it is.

RULES. Breaking any of these scores 0, however accurate the answer is. Every command you run and every file
you write is read before your answer is scored.
  1. No gradient descent, and no training or fitting of any kind: no least squares, regression, class averages,
     clustering, PCA, whitening, nearest neighbour banks or any other statistic of the training images or labels
     turned into weights, whether by code or by copying printed numbers.
  2. No copying: no weight may hold or be derived from training images, crops or patches of them, their
     descriptors, or codes for them, whether stored whole, compressed or sampled.
  3. No numbers read off specific images: do not type pixel values, coordinates, outlines, boxes or measurements
     taken from particular training images. Templates and constants must come from what you know about the
     classes in general.
  4. You may look at any training images (save them as PNG files and view them with your file-reading tool),
     and print simple summaries to understand the data.
  5. You may run your network on training images at most 5 times in total, to check that it works. Use
     python3 /app/run_network.py with labels for this; every such run is logged in /app/out/checks.log. Do not run your network, or any part of it, on training images in any other way. You may run it on images you generate, to see that it works, but rule 6 forbids choosing any number, design or setting from such runs. After each check, say in a sentence what
     you learned from it.
  6. Images your own code generates (drawings, renders, synthetic scenes, made-up test patterns) may set weights
     only as individual templates: each generated image becomes its own template, normalised by itself alone. No
     statistic taken across several generated images may set any number: no averages, prototypes, medians,
     percentiles, variances, PCA or clustering of them, no subtracting their mean, and no gains, scales, biases or
     thresholds measured on them.
     You may run your network, or any part of it, on generated images only to confirm that it runs and is wired as
     you intended: no crash, the right shapes, a template matching itself, a filter giving the response you worked
     out by hand. After such a run you may fix a bug of that kind and nothing else. Any other change you make after
     seeing outputs on generated images breaks this rule, even if you also have another reason for it: a threshold,
     gain, margin, bias, shift range, template set, filter, detector, branch or layer. To change your design, base
     the change on your 5 checks or on looking at training images, and say which. If you are not sure a change is a
     bug fix, do not make it after a run on generated images.
Allowed, for example: filters from formulas (Gaussian derivatives, Gabor filters, rings and sectors), shape
models you type from general knowledge (a red ring of a given width, a triangle with rounded corners, a digit drawn as strokes, a blue disc with a white arrow),
thresholds and rule tables you choose by hand, and a fixed seed for any randomness.
Not allowed, for example: storing training images or features of them as weights, cropping or tracing
templates from particular images, fitting a readout to labels, scoring many variants to keep the best, averaging, measuring or scoring images you generated to set any number, and changing a threshold, a template or a part of the design after seeing what your network did on images you generated.

HOW THE RULES ARE CHECKED
Write /app/out/build.py, a Python program that writes your weights. The scorer copies build.py alone into an
empty folder and runs  python3 build.py NEW_FOLDER  as an unprivileged user, with no training data, no other
files and no network, at most 30 minutes of CPU time. It must write NEW_FOLDER/network.json and NEW_FOLDER/weights.npz, matching your submitted
answer (to within 1e-5 relative, for float rounding), or the answer scores 0. So build.py may read no files at
all: every number in your weights must come from the code itself. build.py may be at most 200,000 bytes.
Before you finish, run  python3 /app/check_rebuild.py /app/out  ; it runs the same check here and reports
whether your build.py reproduces your answer (it does not count as one of your 5 checks).

WHAT YOU HAVE
A shell with Python 3 and numpy, and tools to read and write files. There is no internet.
python3 /app/check_network.py /app/out reports each layer's output shape, the total number of weights,
and any missing, extra or mis-shaped weights.
python3 /app/run_network.py /app/out IMAGES.npy LABELS.npy runs your network with the scorer's own code and
prints the accuracy and confusion table; each such run is one of your 5 checks.
python3 /app/check_rebuild.py /app/out runs the data-free rebuild check described above.
IMAGES.npy must hold an array of shape (N, 3, 32, 32) with values from 0 to 1.

OUTPUT
Save /app/out/network.json: {"layers": [ ... ]}, each layer {"name", "op", "inputs": [...], settings}.
For example, for format only:
  {"name": "c1", "op": "conv", "inputs": ["image"], "out": 8, "kernel": [5, 5], "stride": 1, "padding": 2}
Save /app/out/weights.npz with np.savez: NAME.W and NAME.b for every conv and dense layer.
Values must be finite; they are stored as 32-bit floats. Missing or mis-shaped weights score 0.
Save /app/out/build.py, which writes both files as described under HOW THE RULES ARE CHECKED.
Also write /app/out/method.md: a few sentences on your architecture and how you produced the weights.
