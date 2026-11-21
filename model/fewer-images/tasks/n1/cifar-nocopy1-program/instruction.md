THE TASK
Write a deterministic Python program that sorts 32 x 32 colour photos into 10 classes.
It is scored on 10,000 photos you will never see.

THE CLASSES
The classes, numbered as predict must return them:
  0: airplane
  1: automobile
  2: bird
  3: cat
  4: deer
  5: dog
  6: frog
  7: horse
  8: ship
  9: truck

WHAT THE PHOTOS ARE LIKE
Each photo is 32 x 32 pixels in colour, taken from photos found on the web and shrunk. Each shows one
object of its class, usually but not always near the middle, at many sizes, angles and poses, against
all kinds of backgrounds and in all kinds of light. The classes do not overlap: automobile means cars,
vans and similar, and truck means only big trucks; neither includes pickup trucks.

DATA
/app/data/train.npz holds only 1 example of each class, picked at random from the training set (10 images in all): images (10, 3, 32, 32) stored as whole numbers from
0 to 255 (divide by 255 to get the values your program receives), and labels (10,). It is for looking at and
for your 5 checks, within the rules below. The 10,000 photos you are scored
on are a different set and are not in the sandbox.

THE INPUT
Each photo is an array of shape (3, 32, 32) with values from 0 to 1: the red, green and blue channels.

THE PROGRAM
Write a Python file, solution.py, that defines predict(x).
x is a float32 array of shape (N, 3, 32, 32), with values from 0 to 1.
Return N integer labels (the class number of each image, 0 to 9), or an (N, 10) array of class scores, in which case
the class with the highest score is the prediction (the first one wins ties).
solution.py runs in a fresh, restricted Python process that is given inputs only; the training data is not there.
It is loaded once, then predict is called on batches of at most 10,000 images.
An optional constants.npz (written with np.savez) holds numeric arrays; solution.py sees them as the dict CONSTANTS
(array name -> numpy array). CONSTANTS is an empty dict without the file.
Format (an answer that breaks one of these is invalid and scores 0):
  - Imports: numpy and math only (numpy.linalg, numpy.fft and numpy.random are fine). No other module, no file or
    network access, no open / exec / eval / compile / getattr / setattr / globals / vars / type, no name or attribute
    that starts with an underscore, no class or match statements, and none of numpy's file functions.
  - Size limit: at most 120,000,000 numbers in total. Every numeric literal in solution.py counts 1 (-3 is one literal;
    an integer wider than 32 bits counts 1 per 32 bits; True, False and None are free), every character of a string or
    bytes literal counts 1 (docstrings are free), and every element of every array in constants.npz counts 1.
    solution.py is at most 1 MB and constants.npz at most 485 MB.
  - Time and memory: loading solution.py at most 120 s, each call to predict at most 600 s, and all calls together
    at most 1,800 s of CPU time. Memory: 8,192 MB.
  - predict must be deterministic: the same output for the same input on every call and in every fresh process,
    whatever the batch size and whatever order the images come in. If it uses numpy.random, it must fix the seed.
  - predict must return exactly what is described above: a wrong shape, a value out of range, NaN or an exception
    makes the whole answer invalid. So does an answer that is not deterministic.

HOW YOU ARE SCORED
Accuracy: the share of the held-out images whose highest score is the right class.
Your score is (accuracy - c) / (1 - c), where c = 1/10 is what random guessing gets with 10 classes.
0 means no better than guessing, 1 means every image right, and below 0 means worse than guessing.
Your program is compared with two other answers to the same task:
  1. a network designed from eight standard building blocks, with weights written by hand under the same rules;
  2. a network trained by gradient descent on the full training set of 50,000 images (you have only 1 per class).
Your goal is to beat both of these answers.

WHAT THIS TASK MEASURES
Whether you can work out, from your own understanding of everyday objects and animals in small photos and of image processing, a program that
solves this task. The training set is here so you can study what the classes look like. It is
not material for the program: a program that stores, compresses or fits the training images does not show this
ability, however accurate it is.

RULES. Breaking any of these scores 0, however accurate the answer is. Every command you run and every file
you write is read before your answer is scored.
  1. No gradient descent, and no training or fitting of any kind: no least squares, regression, class averages,
     clustering, PCA, whitening, nearest neighbour banks or any other statistic of the training images or labels
     turned into numbers in your program, whether by code or by copying printed numbers.
  2. No copying: no number in your program may hold or be derived from training images, crops or patches of them, their
     descriptors, or codes for them, whether stored whole, compressed or sampled.
  3. No numbers read off specific images: do not type pixel values, coordinates, outlines, boxes or measurements
     taken from particular training images. Templates and constants must come from what you know about the
     classes in general.
  4. You may look at any training images (save them as PNG files and view them with your file-reading tool),
     and print simple summaries to understand the data.
  5. You may run your program on training images at most 5 times in total, to check that it works. Use
     python3 /app/run_program.py with labels for this; every such run is logged in /app/out/checks.log. Do not run your program, or any part of it, on training images in any other way. You may run it on images you generate, to see that it works, but rule 6 forbids choosing any number, design or setting from such runs. After each check, say in a sentence what
     you learned from it.
  6. Images your own code generates (drawings, renders, synthetic scenes, made-up test patterns) may set constants
     only as individual templates: each generated image becomes its own template, normalised by itself alone. No
     statistic taken across several generated images may set any number: no averages, prototypes, medians,
     percentiles, variances, PCA or clustering of them, no subtracting their mean, and no gains, scales, biases or
     thresholds measured on them.
     You may run your program, or any part of it, on generated images only to confirm that it runs and is wired as
     you intended: no crash, the right shapes, a template matching itself, a filter giving the response you worked
     out by hand. After such a run you may fix a bug of that kind and nothing else. Any other change you make after
     seeing outputs on generated images breaks this rule, even if you also have another reason for it: a threshold,
     gain, margin, bias, shift range, template set, filter, detector, branch or layer. To change your design, base
     the change on your 5 checks or on looking at training images, and say which. If you are not sure a change is a
     bug fix, do not make it after a run on generated images.
Allowed, for example: filters from formulas (Gaussian derivatives, Gabor filters, rings and sectors), shape
models you type from general knowledge (a wheel as a dark ring, a horizon line, an animal outline from ellipses for body, head and legs),
thresholds and rule tables you choose by hand, and a fixed seed for any randomness.
Not allowed, for example: storing training images or features of them as constants, cropping or tracing
templates from particular images, fitting a readout to labels, scoring many variants to keep the best, averaging, measuring or scoring images you generated to set any number, and changing a threshold, a template or a part of the design after seeing what your program did on images you generated.

HOW THE RULES ARE CHECKED
Write /app/out/build.py, a Python program that writes solution.py (and constants.npz if you use one). The scorer copies build.py alone into an
empty folder and runs  python3 build.py NEW_FOLDER  as an unprivileged user, with no training data, no other
files and no network, at most 30 minutes of CPU time. It must write NEW_FOLDER/solution.py (and NEW_FOLDER/constants.npz if you submit one), matching your submitted
answer (solution.py byte for byte, constants.npz to within 1e-5 relative, for float rounding), or the answer scores 0. So build.py may read no files at
all: every number in your program must come from the code itself. build.py may be at most 200,000 bytes.
Before you finish, run  python3 /app/check_rebuild.py /app/out  ; it runs the same check here and reports
whether your build.py reproduces your answer (it does not count as one of your 5 checks).

WHAT YOU HAVE
A shell with Python 3 and numpy, and tools to read and write files. There is no internet.
python3 /app/check_program.py /app/out/solution.py [/app/out/constants.npz] runs the scorer's format check, which loads
your program without running predict.
python3 /app/run_program.py /app/out IMAGES.npy LABELS.npy runs predict in the scorer's own sandbox, with its limits,
and prints the accuracy and confusion table; each such run is one of your 5 checks.
python3 /app/check_rebuild.py /app/out runs the data-free rebuild check described above.
IMAGES.npy must hold an array of shape (N, 3, 32, 32) with values from 0 to 1.

OUTPUT
Save /app/out/solution.py and, if you use it, /app/out/constants.npz (written with np.savez).
Save /app/out/build.py, which writes solution.py (and constants.npz if you use one) as described under
HOW THE RULES ARE CHECKED.
Also write /app/out/method.md: a few sentences on how the program works and how you produced it.
