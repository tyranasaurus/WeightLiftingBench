# Blog charts for Compile

The four Compile charts in the blog post, with the scripts and data that draw them. Each script reads and writes next
to itself: `uv run --no-project --with matplotlib --with numpy python plot_launch.py`.

| Chart | Script | Data |
|---|---|---|
| `figures/blog/1-easy.png`, `2-hard.png`: best model so far from each lab on the easy and hard tiers | `plot_launch.py` | `v1-cells.json` (per model and task scores), `lineup.json` (models, labs, release dates) |
| `figures/blog/3-gap.png`: the gap between labs here and on other benchmarks | `plot_vals_gap.py` | scores and sources in `vals-gap-notes.md` |
| `figures/blog/4-tb-scatter.png`: hard tier score against Terminal-Bench | `plot_tb_scatter.py` | in the script |

`chart_style.py` is the shared style. `sudoku-buckets.json` and `effort-costs.json` feed the extra charts that
`plot_launch.py --all` draws.
