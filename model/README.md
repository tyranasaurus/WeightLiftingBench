# Model: Weight Space Modeling, design a network with no copying

The Model sub-bench of WeightLiftingBench. Compile ([`../compile/`](../compile/README.md)) asks whether a model can put a known
program into a network's weights. This family asks the harder question: can a model write good weights for an image
task that has no short program behind it, **by reasoning instead of learning**? Models get the training set to look at
and a sandbox with code execution, design their own network, and hand in a `build.py` that writes every weight with no
access to data.

357 runs: 16 models x 7 tasks x 3 runs of network design, plus 21 program runs as a reference. A follow-up with 1, 10
and 100 training images per class is in `fewer-images/`.

## Setup

* **Tasks:** 7 image tasks: MNIST digits, Fashion-MNIST, GTSRB traffic signs (43 classes), Galaxy10 DECaLS galaxies,
  Sign Language MNIST letters (24 classes), CIFAR-10, Visual Wake Words "is there a person". Training and test sets from
  the release `write-vs-program-data-v1` (see `data/README.md`), scored on the full test set.
* **Design arm:** the model designs its own network from 8 building blocks (conv, dense, relu, max pool, average pool,
  add, reshape, norm) and writes its weights. Sandbox with code execution and the full training set to look at.
  16 models x 7 tasks x 3 runs = 336 runs.
* **Program arm (the reference):** GPT-6.2 Astra writes a plain deterministic program for each task under the same rules
  (no learned parameters, every constant written by `build.py`). 7 tasks x 3 runs = 21 runs. "Best no-copy program" is
  the best of the 3 runs per task.
* **Trained reference network:** each task's published trained network, rebuilt from the same blocks (sources below).
* **The data-free rebuild gate** (`code/shared/rebuild.py`): the scorer runs the submitted `build.py` alone, with no
  data, and the rebuilt weights must match the submitted ones, or the run scores 0. So every number must come from code.
* **Checks:** at most 5 runs of the network on training images, logged in `/app/out/checks.log`.
* **Models:** OpenAI and open-weight models through `harness/claude_code_wsp_or` (GPT-6.2 Astra, GPT-6 Astra,
  GPT-6.2 Sol, GPT-6.1 Sol, GPT-5.6 Sol, GPT-5.5, GPT-5.4, GLM-5.3, Qwen3.8 Max, Qwen 4 Max, Kimi K3.1 Pro); Anthropic
  models through `harness/claude_code_wsp_k3` (Fable 5.2, Fable 5.1, Opus 5.5, Opus 4.8, Opus 4.7). Effort `xhigh`, tools
  Bash Read Write Edit Glob Grep, 90 minutes per run, 4 CPUs, 8 GB.
* **Valkyrie:** dataset `wsp/nocopy`. Every run id and its dataset version is in `results/runs.csv`.

## Rules

The full rules are in [`rules.md`](rules.md). In short:

1. No gradient descent, training or fitting of any kind.
2. No copying of training images, or anything derived from them, into weights.
3. No numbers read off particular images.
4. Looking at training images and printing summaries is allowed. Label free statistics of the whole training set may
   set a number; per class statistics may not.
5. At most 5 logged checks on training images, and no other runs of the network on them.
6. Images the model generates may set weights only as individual templates, each normalised by itself: no statistics
   across them and no automated search on them.

Runs were collected on task versions @1 to @8; every run is judged against the final rules in `rules.md`. `tasks/` holds
the @8 task folders without data: `<task>-nocopy-design-slim` (design), `<task>-nocopy-design-text-slim` (the prompt
without image viewing, used for GLM-5.3) and `<task>-nocopy-program` (program arm). `data/assemble_tasks.py` copies the
training set, the test set and the program verifier's source into them, after which their Dockerfiles build.

## Scoring

* Accuracy on the full test set. Skill score = (accuracy minus chance) / (1 minus chance), so 0 is guessing and 1 is
  every test image right.
* Every session was read in full by reader agents and judged against `rules.md`. The verdict for each run (`clean` or
  `breach`, the rules broken and the reader's reason) is in `results/verdicts.json`, with a short description of what
  the model built.
* A run that broke the rules scores 0% accuracy, which is below chance, so its skill score is negative.
* Averages are the mean over 3 runs per task, then the mean over 7 tasks.
* 310 runs are clean and 47 broke the rules, all 47 in the design arm.

## Results (accuracy %, mean of 3 runs, a rule break counts as 0)

| Model | Digits | Fashion | Traffic | Galaxies | Letters | CIFAR-10 | Person | Average | Skill score | Rule breaks |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-6.2 Astra | 93.5 | 67.5 | 78.3 | 48.4 | 47.9 | 30.3 | 59.0 | **60.7** | 0.519 | 0 |
| Fable 5.2 | 96.2 | 72.9 | 49.8 | 39.1 | 32.7 | 17.2 | 59.4 | **52.5** | 0.433 | 0 |
| Opus 5.5 | 61.4 | 66.2 | 65.2 | 48.7 | 31.6 | 30.8 | 62.6 | **52.4** | 0.434 | 1 |
| GPT-6 Astra | 88.9 | 61.3 | 55.1 | 39.1 | 20.6 | 20.8 | 55.0 | **48.7** | 0.386 | 0 |
| GPT-6.2 Sol | 80.7 | 57.5 | 57.1 | 36.2 | 18.8 | 22.1 | 53.7 | **46.6** | 0.361 | 0 |
| GPT-6.1 Sol | 82.7 | 61.0 | 42.3 | 28.0 | 28.9 | 24.5 | 53.6 | **45.9** | 0.354 | 0 |
| Fable 5.1 | 59.2 | 60.5 | 35.2 | 32.8 | 20.1 | 24.4 | 58.0 | **41.5** | 0.312 | 3 |
| Kimi K3.1 Pro | 69.6 | 39.5 | 31.8 | 19.5 | 5.9 | 18.9 | 50.6 | **33.7** | 0.218 | 0 |
| Qwen 4 Max | 76.7 | 34.9 | 31.2 | 11.8 | 9.9 | 13.7 | 54.3 | **33.2** | 0.217 | 2 |
| GPT-5.6 Sol | 67.9 | 32.1 | 17.1 | 13.4 | 8.9 | 15.7 | 36.4 | **27.4** | 0.131 | 5 |
| GPT-5.5 | 44.7 | 29.7 | 21.6 | 16.3 | 13.7 | 18.7 | 34.4 | **25.6** | 0.108 | 2 |
| Opus 4.8 | 63.0 | 7.9 | 22.2 | 21.3 | 6.3 | 5.6 | 33.8 | **22.9** | 0.078 | 7 |
| GPT-5.4 | 57.2 | 22.9 | 10.7 | 9.8 | 10.6 | 11.6 | 33.2 | **22.3** | 0.071 | 1 |
| Qwen3.8 Max | 0.0 | 34.4 | 13.8 | 10.0 | 8.9 | 8.2 | 54.0 | **18.5** | 0.055 | 7 |
| Opus 4.7 | 34.9 | 17.3 | 6.0 | 18.8 | 7.0 | 17.5 | 17.7 | **17.0** | -0.006 | 7 |
| GLM-5.3 | 21.9 | 15.3 | 19.4 | 10.6 | 0.0 | 5.1 | 17.2 | **12.8** | -0.054 | 12 |

| | Digits | Fashion | Traffic | Galaxies | Letters | CIFAR-10 | Person |
|---|---|---|---|---|---|---|---|
| Trained reference network | 99.7 | 94.8 | 97.9 | 82.4 | 91.8 | 91.7 | 84.5 |
| Best no-copy program (GPT-6.2 Astra) | 96.9 | 75.3 | 88.3 | 60.3 | 58.3 | 32.9 | 65.7 |
| Mean no-copy program | 96.5 | 73.3 | 87.0 | 56.4 | 48.0 | 32.8 | 62.6 |
| Chance | 10.0 | 10.0 | 2.3 | 10.0 | 4.2 | 10.0 | 50.0 |

Best program average 68.3%. "Rule breaks" counts design runs (of 21) that score 0. The skill column is the mean of
`scored_skill` over a model's 21 design runs in `results/task_results.csv`.

**Main findings**
* Written networks never beat a program: GPT-6.2 Astra's networks average 60.7% against 68.3% for its own best
  programs, and the program wins on all 7 tasks. The build files read as programs in network form: drawn templates,
  colour and edge detectors, if-then rules.
* GPT-6.2 Astra leads, then Fable 5.2 and Opus 5.5, then the other recent OpenAI models. The best open-weight models
  (Kimi K3.1 Pro, Qwen 4 Max) sit behind them, and older models are near the bottom.
* All networks sit far below the trained reference networks.
* Given data, models reach for learning: 47 of the 336 design runs broke the rules, most often by turning per class
  statistics into numbers or by running the network on training images outside the 5 checks.

## Fine-tuning the written networks (GPT-6.2 Astra and Fable 5.2)

Every GPT-6.2 Astra and Fable 5.2 network that followed the rules (42 networks) was fine-tuned on the training set to
see what gradient descent changes (`code/finetune_nocopy.py`, run on Modal by `code/finetune_modal.py`). The blog chart
uses Lehmann's settings: plain Adam, learning rate 3e-4, 3 epochs, no mask, key `lehmann.standard` in
`results/finetune/<run>.dissect.json`. Mean test accuracy %:

| Task | Astra printed | Astra after 3 epochs | Change | Fable 5.2 printed | Fable 5.2 after 3 epochs | Change |
|---|---|---|---|---|---|---|
| Digits | 93.5 | 98.8 | +5.3 | 96.2 | 68.8 | -27.4 |
| Fashion | 67.5 | 88.5 | +21.0 | 72.9 | 79.3 | +6.4 |
| Traffic | 78.3 | 91.0 | +12.7 | 49.8 | 74.7 | +24.9 |
| Galaxies | 48.4 | 36.1 | -12.3 | 39.1 | 57.9 | +18.8 |
| Letters | 47.9 | 93.3 | +45.4 | 32.7 | 84.9 | +52.2 |
| CIFAR-10 | 30.3 | 62.7 | +32.4 | 17.2 | 41.4 | +24.2 |
| Person | 59.0 | 72.0 | +13.0 | 59.4 | 64.9 | +5.5 |

* Fable's digits average falls because one network (`fable-digits-design-1`, 95.9%) collapses to chance within 30
  steps. Seeds 0, 1 and 2 all collapse (8.9%, 9.7%, 9.8%; `results/finetune/fable-digits-design-1.seed1.json`,
  `.seed2.json`): adding noise of 1e-4 to every weight already drops it to chance, and the variant that keeps every
  designed zero at zero (`lehmann.structure`) climbs to 98.9%.
* `results/finetune/<run>.json` holds a gentler recipe that is not used in the blog: 3 epochs of AdamW with every
  designed zero kept at zero and each layer's learning rate at 0.1% of its median nonzero weight (key
  `curve-rel0.001-mask-wu0.3`).

## Recreating the results

Everything a score depends on is in this folder except the data files.

1. Get the 14 data files: `bash model/data/fetch.sh DATA_DIR` (1.6 GB from Hugging Face, see `data/README.md`).
2. Score any submission again, with no Docker and no evaluation platform:
   `uv run --no-project --with numpy python model/code/score_submission.py DATA_DIR astra-digits-design-1`.
   It runs the submitted `build.py` alone in an empty folder, then runs the rebuilt network or program on the full test
   set, and prints the accuracy next to the recorded one. With no run name it scores all 357.
3. Rebuild the result tables and charts: `python model/code/make_results_csv.py`, then
   `uv run --no-project --with matplotlib python model/results/figures/make_charts.py`.
4. To run a new model on a task, `python model/data/assemble_tasks.py DATA_DIR` makes every task folder buildable
   (Harbor task format: `instruction.md`, `environment/`, `tests/`). The two agent harnesses are in `harness/`.

**Trained reference networks.** The reference accuracies come from published trained networks, converted to the same
building blocks. They are not copied here:

| Task | Source |
|---|---|
| Letters | Hugging Face `namas191297/sign_language_mnist_cnn` |
| Traffic | Hugging Face `snehilsanyal/traffic_sign_detection_gtsrb` (published 97.9%) |
| Galaxies | GitHub `DaybreakWannaCode/weight-space-galaxy10` (commit `baf8121`) |
| Person | `vww_96.h5` from GitHub `mlcommons/tiny`, `benchmark/training/visual_wake_words/trained_models/` (84.5% on our test split) |
| Digits, Fashion, CIFAR-10 | source not recorded in this repository yet |

**What cannot be checked from here.** The verdicts rest on reading each session in full; the session transcripts are
not published, so a rule break ruling can be checked only against the submitted files and the reader's reason.

## Caveats

* **Prompts and rules.** Models saw task versions @1 to @8, whose rule wording differs; verdicts use `rules.md` for
  every run. The `task_version` column says which version a run saw.
* **Readers are agents.** Reasons in `results/verdicts.json` are the readers' own words, and some refer to an earlier
  reading of the same session. Fable's thinking is logged only as signatures, so its verdicts rest on commands, files
  and messages.
* **Unlogged runs of the network.** The check tool logged only runs with labels, so readers had to find other runs on
  training images in the transcript.
* **GLM-5.3** used the text-only prompt (no image viewing).
* **One program run has no score:** the scorer could not use the output of `astra-person-program-2` (`valid_output`
  False); it is left out of the program averages.
* **Orchestration ran on Valkyrie.** `code/sweep.py`, `code/finetune_modal.py` and the task builders record how the runs
  were launched and need that platform, Modal, or earlier task folders that are not here. Scoring, the analyses and
  the charts run from this folder.
* **Data.** The data sets are on Hugging Face (`data/README.md`); COCO photos (person) carry mixed licences, some
  non-commercial.

## Results files

| File | What it holds |
|---|---|
| `results/runs.csv` | one row per run (357): Valkyrie run id, model, agent, dataset version, task, arm, run number, accuracy, whether the output was valid, rebuild gate, checks used |
| `results/task_results.csv` | one row per run: task version, verdict, rules broken, accuracy and skill, and the scored accuracy and skill (a rule break counts as 0% accuracy). Rebuilt by `code/make_results_csv.py` |
| `results/verdicts.json` | one entry per run: verdict, rules broken (`R1`, `R2`, `R3`, `STAT`, `R5`, `R6` as in `rules.md`), the reader's reason, and what the model built |
| `results/submissions/<model>-<task>-<arm>-<run>/` | the files each run handed in: `build.py`, `network.json`, `method.md`, `checks.log`; `solution.py` for programs |

## Analyses behind the blog

| What | Where |
|---|---|
| Image by image agreement of Astra's networks and programs | `results/analysis/agreement.json`, `code/agreement.py`, saved program predictions in `results/analysis/program_preds/` |
| Blind algorithm comparison: 42 readers describe each Astra network and program from comment-stripped code, judges compare them per task | `results/analysis/blind_algo/` (`summary.txt`, `results.json`, `KEY.json`) |
| Weights-only reading of 4 Astra networks and 4 trained reference networks | `results/analysis/weights_only/` (`results.json`, `KEY.json`) |
| What fine-tuning changes (layer drift, dissection, Lehmann settings) | `results/analysis/*_summary.json`, `code/dissect_summary.py`, `code/drift_summary.py`, `code/lehmann_summary.py` |
| Size and shape of Astra's programs | `results/analysis/program_stats.json` |
| Blog figures and their script | `results/figures/` (`make_charts.py`) |

## Where things are

| What | Where |
|---|---|
| Rules | `rules.md` |
| Tasks (@8, no data) | `tasks/` |
| Data release and checksums | `data/` |
| Agent harnesses, as the runs used them | `harness/claude_code_wsp_or` (OpenAI and open-weight models through OpenRouter), `harness/claude_code_wsp_k3` (Anthropic models) |
| Task builders and rule text | `code/make_nocopy.py`, `code/make_nocopy_program.py`, `code/shared/rules_text.py` |
| Rebuild gate | `code/shared/rebuild.py` |
| Orchestrator | `code/sweep.py` |
| Scoring a submission again, locally | `code/score_submission.py` |
| Program verifier source | `verifier/` |
| Fine-tuning | `results/finetune/`, `code/finetune_nocopy.py`, `code/finetune_modal.py` |
| Fewer training images (1, 10, 100 per class) | `fewer-images/` (own README) |
