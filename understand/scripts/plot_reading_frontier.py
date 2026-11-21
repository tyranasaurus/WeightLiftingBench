"""Family 1 (weight reading) frontier chart: best model so far from each lab vs release date, as step lines.

uv run --quiet --no-project --with matplotlib python scripts/plot_reading_frontier.py
Reads results/scores.json (from score.py). Writes results/figures/5-reading-frontier.png in the
shared chart_style. A line steps up only when a lab releases a model that beats its
own best; models that didn't are smaller dots off the line. One pass per task, about +-0.07 per point.
"""
import datetime as dt
import json
from pathlib import Path

import matplotlib.dates as mdates

import chart_style as cs

HERE = Path(__file__).parent
COLOR, TEXTC = cs.COLOR, cs.TEXTC
LAB = {"GPT": "OpenAI", "Opus": "Anthropic", "Fable": "Anthropic", "Kimi": "Open-weight labs", "GLM": "Open-weight labs"}
START, END = dt.date(2026, 4, 1), dt.date(2026, 12, 15)
# label offsets in points (dx, dy[, text]); "" hides a label
LABELS = {"GPT-5.5": (-8, 12), "GPT-6 Astra": (-10, 4), "GPT-6.1 Sol": (8, -13), "GPT-6.2 Sol": (8, -14),
          "GPT-6.2 Astra": (-10, 4), "Opus 4.8": (-8, 12), "Fable 5.1": (-10, 8), "Opus 5.5": (8, -15),
          "Fable 5.2": (-10, 6), "GLM-5.3": (-10, 0), "Kimi K3.1 Pro": (8, -14)}

scores = {m: r for m, r in json.loads((HERE.parent / "results/scores.json").read_text()).items() if r["all"]["n"] == 62}
fig, ax = cs.canvas()

ends = []
for lab in cs.LABS:
    pts = sorted(((dt.date.fromisoformat(r["date"]), r["all"]["skill"], m) for m, r in scores.items()
                  if next(v for k, v in LAB.items() if m.startswith(k)) == lab))
    best, xs, ys, steps = -1.0, [], [], set()
    for d, y, m in pts:
        if y > best:
            best = y; xs.append(d); ys.append(y); steps.add(m)
    ax.step(xs + [END], ys + [ys[-1]], where="post", color=COLOR[lab], linewidth=3.2, zorder=2, solid_capstyle="round")
    for d, y, m in pts:
        on = m in steps
        ax.scatter([d], [y], s=70 if on else 40, color=COLOR[lab], zorder=3)
        dx, dy, *text = LABELS.get(m, (8, -12))
        if text == [""]:
            continue
        ax.annotate(text[0] if text else m, (d, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=cs.LABEL_SIZE if on else cs.LABEL_SIZE - 1, color=TEXTC[lab], ha="left" if dx >= 0 else "right", va="center")
    ends.append((lab, ys[-1]))

cs.legend(ax, [cs.lab_handle(lab) for lab in cs.LABS])

ax.set_xlim(START, END)
ax.set_ylim(-0.05, 1.04)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0 (guessing)", "0.25", "0.50", "0.75", "1 (all right)"])
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(4, 6, 8, 10, 12)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
cs.axis_label(ax, y="Skill score")

cs.header(fig, "GPT-6.2 Astra is a big jump in weight space classification",
          "Best model so far from each lab at naming what 62 networks were trained on, from weights alone. Skill score",
          "weight space classification")
fig.subplots_adjust(left=0.14)  # room for the "0 (guessing)" tick labels
cs.save(fig, HERE.parent / "results" / "figures" / "5-reading-frontier.png")
