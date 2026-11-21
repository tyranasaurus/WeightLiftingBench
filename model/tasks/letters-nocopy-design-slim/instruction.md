THE TASK
Design a neural network that sorts 28 x 28 greyscale photos of hands making sign-language letters into 24 classes,
and write its weights. The network is scored on 7,172 images you will never see.

THE CLASSES
Output k of the network is the score for class k. The class with the highest score is the prediction.
  0: A
  1: B
  2: C
  3: D
  4: E
  5: F
  6: G
  7: H
  8: I
  9: K
  10: L
  11: M
  12: N
  13: O
  14: P
  15: Q
  16: R
  17: S
  18: T
  19: U
  20: V
  21: W
  22: X
  23: Y

WHAT THE IMAGES ARE LIKE
Each image is a photo of one hand making a static American Sign Language letter, 28 x 28 pixels, greyscale, with values from 0 (black) to 1 (white).
The photos were taken of several people repeating each sign in front of different backgrounds, so the background is not plain and its brightness varies; the hand is often, but not always, lighter than the background.
Each colour photo was cropped to a square around the hand, turned to greyscale and shrunk to 28 x 28; the hand fills most of the frame and is not centred by any rule, and the wrist or sleeve often enters from the bottom edge.
The images were then multiplied by making about 50 variants of each: different resizing filters, 5% random pixel noise, brightness and contrast changed by up to 15%, and rotation by up to 3 degrees.
The hand is seen from the front, as a person facing the signer sees it, and keeps the orientation the sign has, so for some letters it points sideways or down rather than up.
J and Z are left out because they need motion, so there are 24 classes.

DATA
/app/data/train.npz is the official training set: images (27455, 1, 28, 28) stored as whole numbers from
0 to 255 (divide by 255 to get the values your network receives), and labels (27455,). It is for looking at and
for your 5 checks, within the rules below. The 7,172 images you are scored
on are a different set and are not in the sandbox.

THE INPUT
Each image is an array of shape (1, 28, 28) with values from 0 to 1. Your network reads it as "image".

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
The last layer must output shape (24,): the class scores.
Size limit: at most 120,000,000 weights in total, counting every entry of weights.npz.

HOW YOU ARE SCORED
Accuracy: the share of the held-out images whose highest score is the right class.
Your score is (accuracy - c) / (1 - c), where c = 1/24 is what random guessing gets with 24 classes.
0 means no better than guessing, 1 means every image right, and below 0 means worse than guessing.
Your network is compared with two other answers to the same task:
  1. a deterministic Python program that classifies the images directly;
  2. your same architecture trained by gradient descent on the 27,455 training images.
Your goal is to beat both of these answers.

WHAT THIS TASK MEASURES
Whether you can work out, from your own understanding of hands and the American Sign Language alphabet and of how networks compute, the weights of
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
  6. Images your own code generates (drawings, renders, synthetic scenes) may set weights only as individual
     templates: each generated image becomes its own template, normalised by itself alone. No statistic taken across
     several generated images may set any number: no averages, prototypes, medians, percentiles, variances, PCA or
     clustering of them, no subtracting their mean, and no gains, scales, biases or thresholds measured on them. Do not
     score your network, or any variant of it, on generated images to choose a number, a design or a setting, whether
     with a script or by editing by hand and rechecking.
Allowed, for example: filters from formulas (Gaussian derivatives, Gabor filters, rings and sectors), shape
models you type from general knowledge (a hand outline built from a palm ellipse and finger segments at given angles, a fist, an extended thumb),
thresholds and rule tables you choose by hand, and a fixed seed for any randomness.
Not allowed, for example: storing training images or features of them as weights, cropping or tracing
templates from particular images, fitting a readout to labels, scoring many variants to keep the best, and averaging, measuring or scoring images you generated to set any number.

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
IMAGES.npy must hold an array of shape (N, 1, 28, 28) with values from 0 to 1.

OUTPUT
Save /app/out/network.json: {"layers": [ ... ]}, each layer {"name", "op", "inputs": [...], settings}.
For example, for format only:
  {"name": "c1", "op": "conv", "inputs": ["image"], "out": 8, "kernel": [5, 5], "stride": 1, "padding": 2}
Save /app/out/weights.npz with np.savez: NAME.W and NAME.b for every conv and dense layer.
Values must be finite; they are stored as 32-bit floats. Missing or mis-shaped weights score 0.
Save /app/out/build.py, which writes both files as described under HOW THE RULES ARE CHECKED.
Also write /app/out/method.md: a few sentences on your architecture and how you produced the weights.
