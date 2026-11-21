# Follow-up: no-copy design with 1, 10 or 100 training images per class

Chart: `../results/figures/4-training-images.png` (95% intervals from `code/levels_ci.py`).

## Question

How much do the written networks depend on the training data they can look at? Same rules (`../rules.md`), tasks,
tools, budget and scoring as the main experiment (`../README.md`), but `train.npz` in the sandbox holds only N images
per class.

## Setup

* **Levels:** 1, 10 and 100 images per class. Subsample with seed 0, the same images in both arms; a class with fewer
  images keeps all of them. Built by `code/make_nocopy10.py N` from the main experiment's @8 tasks; the only prompt
  change is the DATA line ("holds only N examples of each class") and a note that the 5 checks run on those images.
* **Valkyrie datasets:** `wsp/nocopy1@1`, `wsp/nocopy10@1`, `wsp/nocopy100@1` (all 7 tasks). Tasks without data are in
  `tasks/n<N>/`.
* **Models:** GPT-6.2 Astra and Fable 5.2 design networks, and GPT-6.2 Astra programs, 7 tasks x 3 runs each: 63 runs
  per level, 189 in all. Orchestrator `code/sweep10.py`.
* **Reading:** every session was read in full and judged against `../rules.md`. No run broke the rules at any level.
  One entry per run is in `verdicts.json`.

## Results (mean skill score over the 7 tasks)

| | 1 per class | 10 per class | 100 per class | Full training set (main experiment) |
|---|---|---|---|---|
| GPT-6.2 Astra programs | 0.48 | 0.54 | 0.55 | 0.57 |
| GPT-6.2 Astra networks | 0.44 | 0.47 | 0.50 | 0.52 |
| Fable 5.2 networks | 0.43 | 0.42 | 0.50 | 0.43 |

* Fewer images barely change the scores. Prior knowledge carries the networks: the rules narrow the data channel to
  looking plus 5 checks, Astra opened few images (most 1 per class networks opened none), and the checks themselves are
  the main route by which data reaches the weights.
* Tasks with quirky data (galaxies, letters) gain most from 1 to 10 images per class.
* What the models built at each level: `strategies/<task>.txt`.
* The scorer could not use the output of one program run (`astra-person-program-1` at 100 per class, `valid_output`
  False); it is left out of the averages, as is the one such run in the main experiment.

## Where things are

| What | Where |
|---|---|
| Run ids, accuracies and skill scores | `runs.csv` (one row per run, with its level) |
| Verdicts and what each run built | `verdicts.json` (keys `n<N>/<model>-<task>-<arm>-<run>`) |
| Handed-in files of every run | `submissions/n<N>/<model>-<task>-<arm>-<run>/` |
| Tasks (no data) | `tasks/n1`, `tasks/n10`, `tasks/n100` |
| Task builder, orchestrator, intervals | `code/` |
| Strategy write-ups per task | `strategies/` |
