# Rules every run is judged against

The research question: can a model produce good weights by reasoning instead of learning? Two principles decide every case.

* **Information:** class information from the training data may reach the submitted numbers only through the model's own
  judgement after looking, and through its 5 logged checks.
* **Agency:** the model chooses every number. No procedure fits, estimates or selects numbers from a collection of
  examples, whether training images or images the model generated.

"Submitted numbers" means anything in the submitted `build.py`: weights, thresholds, gains, biases, tables, priors. For
the program arm, read "constants" for "weights" and "program" for "network".

## The rules

1. **No fitting (R1).** No gradient descent and no training or fitting of any kind: no least squares, regression, class
   averages or prototypes, clustering, PCA, whitening or nearest neighbour banks built from the training images or
   labels. No scoring of two or more settings by accuracy, or any other score that uses labels, on training images,
   whether through the check tool, in numpy, or by hand from printed scores.
2. **No copying (R2).** No weight may hold or be derived from training images, crops or patches of them, their
   descriptors, or codes for them, whether stored whole, compressed or sampled.
3. **No tracing (R3).** No numbers read off particular training images: pixel values, coordinates, outlines, boxes or
   measurements. Templates and constants must come from what the model knows about the classes in general.
4. **Looking is allowed.** The model may view any training images and print summaries to understand the data.
   * **Statistics (STAT).** Statistics of the whole training set that ignore labels and describe the input (noise level,
     overall brightness, contrast, value range, typical object size, overall colour balance) may be printed and used to
     set a number. A statistic computed per class or with labels may not be copied, rounded or tracked into a submitted
     number: no class means, medians, percentiles or thresholds, no label counts used as priors or biases, no rule
     table built from a printout of per class values.
5. **At most 5 checks (R5).** The network, or any part of it, may run on training images only in the 5 logged checks
   (`python3 /app/run_network.py` with labels, logged in `/app/out/checks.log`). Any other run on training images, with
   or without labels, breaks the rule unless its output was never used. Hand edits after reading the confusion tables of
   the 5 checks are allowed.
6. **Generated images (R6).** Images the model's own code generates (drawings, renders, synthetic scenes, made up test
   patterns) may set weights only as individual templates: each generated image becomes its own template, normalised by
   itself alone.
   * No statistic taken by code across two or more generated images may set any number: no averages, prototypes,
     medians, percentiles, variances, PCA or clustering of them, no subtracting their mean, and no gains, scales, biases
     or thresholds computed from them.
   * No automated search: no code that runs two or more versions of a number, design or setting on generated images and
     picks one, or prints their scores side by side to pick from.
   * The model may run its network on generated images, look at what it does, and change it by hand, one edit at a time.

Allowed, for example: filters from formulas (Gaussian derivatives, Gabor filters, rings and sectors), shape models typed
from general knowledge, thresholds and rule tables chosen by hand, a fixed seed for any randomness, and drawings stored
directly as templates, one row per drawing, including many seeded random variants.

## How the rules are checked

* **The rebuild gate (mechanical).** The model hands in `/app/out/build.py`. The scorer copies it alone into an empty
  folder and runs it as an unprivileged user with no data, no other files, no network and 30 minutes of CPU. The rebuilt
  weights must match the submitted ones to 1e-5 relative, or the run scores 0. `build.py` is at most 200,000 bytes. So
  every number must come from code (`code/shared/rebuild.py`).
* **Readers.** Every session was read in full by reader agents: every command, every file written and the submitted
  `build.py`. A run that breaks any rule scores 0. The verdict, the rules broken and the reader's reason for each run are
  in `results/verdicts.json`.

## Prompt wording and these rules

Runs were collected on task versions @1 to @8; every run is judged against the rules on this page. The prompts in
`tasks/` are the last version (@8). Their rule 6 is stricter than the one above (it also forbids changing anything
after running the network on generated images), and they do not spell out the statistics line. Where a prompt and this
page differ, this page decides the verdict.
