"""WeightLiftingBench hard tier vs Terminal-Bench (Vals), each lab's strongest model per release.

uv run --quiet --no-project --with matplotlib --with numpy python plot_tb_scatter.py -> figures/blog/4-tb-scatter.png (shared chart_style; old figures/tb-vs-wsc.png left as it was)
Terminal-Bench: Vals TB 4.0 (vals.ai/benchmarks/terminal-bench-4, updated 11/16/2026) where a model has it; else Vals
TB 2.0 (vals.ai/benchmarks/terminal-bench-2, updated 6/4/2026) scaled by 23.23/70.04 = 0.332, the ratio for Opus 4.8,
the one model on both. Opus 4.8 is plotted once, at its TB 4.0 score. WeightLiftingBench: v1 hard tier from blog_v1_runs.csv.
"""
from pathlib import Path

import numpy as np

import chart_style as cs

HERE = Path(__file__).parent
COLOR, TEXTC, BG, MUTED = cs.COLOR, cs.TEXTC, cs.BG, cs.MUTED
SCALE = 23.23 / 70.04
# (name, lab, TB 4.0 or None, TB 2.0 or None, WSC hard tier, label offset in points)
PTS = [
    ("GPT-5.2", "OpenAI", None, 51.69, .156, (-8, 8)), ("GPT-5.4", "OpenAI", None, 58.43, .319, (-8, 8)),
    ("GPT-5.5", "OpenAI", None, 73.20, .354, (8, 6)), ("GPT-5.6 Sol", "OpenAI", 37.88, None, .371, (-8, 9)),
    ("GPT-6 Astra", "OpenAI", 59.60, None, .944, (-10, 4)), ("GPT-6.2 Astra", "OpenAI", 63.13, None, .904, (-10, -12)),
    ("Opus 4.5", "Anthropic", None, 53.93, .049, (8, -8)), ("Opus 4.7", "Anthropic", None, 68.54, .118, (9, 3)),
    ("Opus 4.8", "Anthropic", 23.23, None, .243, (8, -8)), ("Fable 5.1", "Anthropic", 58.08, None, .593, (-10, 0)),
    ("Opus 5.5", "Anthropic", 65.15, None, .601, (8, 0)), ("Fable 5.2", "Anthropic", 65.15, None, .852, (8, -8)),
    ("Kimi K2.5", "Open-weight labs", None, 40.45, .028, (-8, -6)), ("GLM-5", "Open-weight labs", None, 49.44, .067, (-6, 10)),
    ("Qwen3.8 Max", "Open-weight labs", 38.89, None, .290, (-8, 9)), ("GLM-5.3", "Open-weight labs", 38.89, None, .096, (8, -6)),
    ("Qwen4 Max", "Open-weight labs", 48.48, None, .273, (0, -15)), ("Kimi K3.1 Pro", "Open-weight labs", 63.64, None, .293, (8, -6)),
]

fig, ax = cs.canvas(grid="both")

xs, ys = [], []
for name, lab, tb4, tb2, wsc, (dx, dy) in PTS:
    x = tb4 if tb4 is not None else tb2 * SCALE
    if lab != "Open-weight labs":
        xs.append(x); ys.append(wsc * 100)
    hollow = tb4 is None
    ax.scatter([x], [wsc * 100], s=95, marker="D" if hollow else "o", zorder=3,
               color=BG if hollow else COLOR[lab], edgecolors=COLOR[lab], linewidths=2.2)
    ax.annotate(name, (x, wsc * 100), xytext=(dx, dy), textcoords="offset points", fontsize=cs.LABEL_SIZE - 0.5, color=TEXTC[lab],
                ha="left" if dx >= 0 else "right", va="center")
b, a = np.polyfit(xs, ys, 1)
rmse = float(np.sqrt(np.mean((np.array(ys) - (a + b * np.array(xs))) ** 2)))
gx = np.linspace(8, 70, 50)
ax.plot(gx, a + b * gx, color="#8d887c", linewidth=1.6, zorder=1, gid="ignore")
ax.fill_between(gx, a + b * gx - rmse, a + b * gx + rmse, color="#e8e6dc", alpha=0.6, zorder=0, linewidth=0)
ax.text(44, a + b * 44 + rmse + 1.5, "OpenAI + Anthropic trend", fontsize=11, color=MUTED, ha="center", va="bottom", rotation=27)

from matplotlib.lines import Line2D

handles = [cs.lab_handle(lab, "dot") for lab in cs.LABS] + [
    Line2D([], [], marker="o", linestyle="", markersize=10, color=MUTED, label="Terminal-Bench 4.0"),
    Line2D([], [], marker="D", linestyle="", markersize=9, color=BG, markeredgecolor=MUTED, markeredgewidth=2,
           label="Terminal-Bench 2.0 (rescaled)")]
cs.legend(ax, handles)
ax.set_xlim(8, 72); ax.set_ylim(-2, 102)
ax.set_xticks(range(10, 71, 10)); ax.set_xticklabels([f"{t}%" for t in range(10, 71, 10)])
ax.set_yticks(range(0, 101, 25)); ax.set_yticklabels([f"{t}%" for t in range(0, 101, 25)])
cs.axis_label(ax, x="Terminal-Bench score (run by Vals)", y="WeightLiftingBench hard tier score")
cs.header(fig, "Open-weight models gain on coding but not on weight space compilation",
          "Each lab's strongest model at each release: WeightLiftingBench hard tier against Terminal-Bench, an agentic coding test",
          "weight space compilation", source="Terminal-Bench scores run by Vals.ai")
fig.subplots_adjust(bottom=0.15)
cs.save(fig, HERE / "figures" / "blog" / "4-tb-scatter.png")
print("trend slope", round(b, 2), "rmse", round(rmse, 1))
