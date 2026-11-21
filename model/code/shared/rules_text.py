"""The no-copying rule text shared by both galaxies tasks."""
MEASURES = """WHAT THIS TASK MEASURES
Whether you can work out, from your own understanding of {subject} and of how networks compute, the weights of
a network that solves this task. The training set is here so you can study what the classes look like. It is
not material for the weights: a network that stores, compresses or fits the training images does not show this
ability, however accurate it is."""

RULES = """RULES. Breaking any of these scores 0, however accurate the answer is. Every command you run and every file
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
models you type from general knowledge ({shape_examples}),
thresholds and rule tables you choose by hand, and a fixed seed for any randomness.
Not allowed, for example: storing training images or features of them as weights, cropping or tracing
templates from particular images, fitting a readout to labels, scoring many variants to keep the best, and averaging, measuring or scoring images you generated to set any number."""

REBUILD = """HOW THE RULES ARE CHECKED
Write /app/out/build.py, a Python program that writes your weights. The scorer copies build.py alone into an
empty folder and runs  python3 build.py NEW_FOLDER  as an unprivileged user, with no training data, no other
files and no network, at most 30 minutes of CPU time. It must write NEW_FOLDER/{files}, matching your submitted
answer (to within 1e-5 relative, for float rounding), or the answer scores 0. So build.py may read no files at
all: every number in your weights must come from the code itself. build.py may be at most 200,000 bytes.
Before you finish, run  python3 /app/check_rebuild.py /app/out  ; it runs the same check here and reports
whether your build.py reproduces your answer (it does not count as one of your 5 checks)."""
