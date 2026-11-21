# Method

## Architecture

Two hand-built feature branches feed one bank of digit templates; the class score is the best template match in the class.

- **Edge branch** (`edge` → `dir` → `pool` → `bend` → `root`): eight Gaussian-derivative edge filters (σ = 1 px, 45° apart), a 1×1 layer that splits each gradient between its two neighbouring directions, Gaussian pooling (σ = 1.8 px) onto a 7×7 grid, and a piecewise-linear square-root curve built from ReLUs. 392 numbers per image.
- **Bay branch** (`solid` → `half` → `rays` → `bays` → `baypool`): the image is thresholded and halved to 14×14; one convolution sums the ink above, below, left and right of every pixel; a 1×1 rule table turns these into nine maps (enclosed hole, bay open to one side ×4, corner blocked on two adjacent sides ×4); Gaussian pooling onto a 5×5 grid. 225 numbers per image.
- **Template bank** (`match_edge` + `match_bay` → `match`): two dense layers whose rows are 190,000 templates (19,000 per class), added together, so each output is the dot product of the image's features with one template.
- **Decision** (`bank` → `best_a` → `best_b` → `scores`): reshape to (10, 100, 190) and two max-pools give the largest match per class.

Total 117,629,147 weights.

## How the weights were produced

No training or fitting of any kind. `build.py` reads no files.

- All front-end filters come from formulas and constants chosen by hand (filter scales, ink threshold, rule table, the weight of the bay maps against the edge maps).
- Each template is one drawing made by `build.py`: a digit written as pen strokes along control points typed from general knowledge of how digits are written (ovals, bars, stems, hooks, loops; several styles per digit such as open and closed 4, looped and Z-topped 2, crossed 7, hooked 9). A fixed-seed random generator varies proportions, slant, rise of level strokes, pen thickness and small wobble.
- Each drawing is fitted into a 20×20 box, centred by centre of mass, passed through the front end, and divided by its own length. No statistic across drawings is used anywhere.

The training set was used only for looking at digits (contact sheets) and for three logged checks.

## Checks on the training set (3 of 5 used)

| Check | Accuracy | What changed before it | What I learned |
|---|---|---|---|
| 1 | 94.37% | Edge branch only, first set of drawings | The approach works; 9→4, 8→9, 5→6 and 2→others dominate the errors. |
| 2 | 95.64% | Bay branch added; broader 2, 8, 9, 0 drawings; 4s always have the bar past the stem | Hole/bay maps fix most 8→9, 0→6, 2→3; digits 2, 3, 5 still lose about 8% each. |
| 3 | 95.99% | Rounder 5 bowls, rising level strokes, wider 3s, narrower 2s, open-top 9s | 3→5, 5→6, 2→1 fell; open-top 9s moved errors from 9→4 (231→128) to 4→9 (81→193). |

The submitted weights are the ones measured in check 3.
