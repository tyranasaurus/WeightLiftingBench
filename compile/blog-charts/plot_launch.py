"""Launch charts for WeightLiftingBench v1, built for social (16:9, single panel, big direct labels, finding-as-title).

uv run --no-project --with matplotlib --with numpy python plot_launch.py
Reads v1-cells.json, lineup.json, effort-costs.json, sudoku-by-difficulty.tsv.
Writes the blog charts 1-easy.png and 2-hard.png to figures/blog/ in the shared chart_style.
`--all` also re-renders the legacy launch-sudoku/effort/hard-heatmap PNGs in figures/.
"""
import csv
import sys
import datetime as dt
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import chart_style as cs

HERE = Path(__file__).parent
FIG = HERE / "figures"
BLOG = FIG / "blog"
D = json.loads((HERE / "v1-cells.json").read_text())
EASY, HARD, MAIN = D["easy"], D["hard"], D["main"]
HARD6 = [t for t in HARD if t != "chess-mate1"]  # the effort sweep ran these 6 only

BG, INK, MUTED, FAINT, GRID = "#fcfcfb", "#141413", "#5e5d59", "#9c9a92", "#e8e6dc"
# validated with the dataviz validator (--pairs all): passes; direct labels carry identity too
COLOR = {"OpenAI": "#139067", "Anthropic": "#e8622c", "Open-weight labs": "#2a6fd6", "Google": "#a8329e"}
# darker text shades of each lab hue (>= 4.5:1 on the background) so point labels read as their lab
TEXTC = {"OpenAI": "#0d6b4c", "Anthropic": "#b8461a", "Open-weight labs": "#1f55a8", "Google": "#7d2475"}
FOOT = ("WeightLiftingBench v1 · single-turn, no tools or code execution · ≤500k nonzero weights · "
        "hidden test inputs · 3 runs per task · data as of Nov 17, 2026\n"
        "Open-weight labs = DeepSeek, Moonshot, Zhipu, Alibaba (Qwen Max and Kimi K3.1 Pro weights not yet released)")

LINEUP = {m["id"]: (m["lab"], m["name"].replace("Claude ", ""), m["date"]) for m in json.loads((HERE / "lineup.json").read_text())}
LINEUP.update({
    "openai/gpt-6-luna": ("openai", "GPT-6 Luna", "2026-09-22"),
    "openai/gpt-6.1-terra": ("openai", "GPT-6.1 Terra", "2026-10-26"),
    "openai/gpt-6.2-sol": ("openai", "GPT-6.2 Sol", "2026-11-17"),
    "openai/gpt-6.2-astra": ("openai", "GPT-6.2 Astra", "2026-11-17"),
    "anthropic/claude-sonnet-5.5": ("anthropic", "Sonnet 5.5", "2026-09-28"),
    "anthropic/claude-haiku-5.5": ("anthropic", "Haiku 5.5", "2026-10-07"),
    "qwen/qwen3.8-max-0902": ("open", "Qwen3.8 Max", "2026-09-03"),
    "qwen/qwen4-max": ("open", "Qwen4 Max", "2026-11-15"),
    "google/gemini-4-hydrogen": ("google", "Gemini 4 Hydrogen", "2026-11-12"),  # OpenRouter listing date
})
LAB = {"openai": "OpenAI", "anthropic": "Anthropic", "open": "Open-weight labs", "google": "Google"}
# Each lab's 2 most capable models at each date. Smaller siblings and the top tier skipped on easy are out.
NOT_FRONTIER = {"openai/gpt-6-luna", "openai/gpt-6.1-terra", "anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"}
EASY_SKIPPED = {"openai/gpt-6-astra", "anthropic/claude-fable-5.2", "anthropic/claude-fable-5.1", "anthropic/claude-fable-5"}
CHINA_FRONTIER = {"deepseek/deepseek-r1-0528", "moonshotai/kimi-k2-thinking", "deepseek/deepseek-v3.2", "moonshotai/kimi-k2.5",
                  "z-ai/glm-5", "z-ai/glm-5.3", "qwen/qwen3.8-max-0902", "moonshotai/kimi-k3.1-pro", "qwen/qwen4-max"}
START, END = dt.date(2025, 4, 1), dt.date(2026, 12, 15)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 22})


def tier_mean(cells, tasks):
    vals = [np.mean(cells[t]) for t in tasks if cells.get(t)]
    return (float(np.mean(vals)) if vals else None), len(vals)


def canvas(w=12, h=6.75):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=13, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return fig, ax


def header(fig, title, subtitle, foot=FOOT):
    fig.text(0.035, 0.94, title, fontsize=21, color=INK, fontweight="bold", va="top")
    fig.text(0.035, 0.865, subtitle, fontsize=14, color=MUTED, va="top")
    fig.text(0.035, 0.015, foot, fontsize=10.5, color=FAINT, linespacing=1.4)


def spread(ys, gap):
    """Nudge label y positions apart so line-end labels don't overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


RUNSPAN = {}


def run_span(cells, tasks):
    """Min and max over runs k of the tier mean of run k (cells hold up to 3 runs per task, in run order)."""
    k_max = max((len(cells[t]) for t in tasks if cells.get(t)), default=0)
    means = []
    for k in range(k_max):
        vals = [cells[t][k] for t in tasks if cells.get(t) and len(cells[t]) > k]
        if len(vals) == sum(1 for t in tasks if cells.get(t)):
            means.append(float(np.mean(vals)))
    return (min(means), max(means)) if means else (None, None)


def history(tasks, skip, labs, name, title, subtitle, jump_labels, ylabel, show_all=False, note=None, note_xy=None, ylo=0.0):
    """Blog chart: best model so far per lab as step lines, in the shared chart_style (writes figures/blog/)."""
    fig, ax = cs.canvas()
    for lab in labs:
        color = COLOR[lab]
        pts = []
        for mid, cells in MAIN.items():
            if mid not in LINEUP or LAB.get(LINEUP[mid][0]) != lab or mid in skip:
                continue
            if lab == "Open-weight labs" and mid not in CHINA_FRONTIER:
                continue
            score, n = tier_mean(cells, tasks)
            if score is not None and n * 2 >= len(tasks):  # a point needs at least half the tier
                pts.append((LINEUP[mid][1], dt.date.fromisoformat(LINEUP[mid][2]), score, n == len(tasks)))
                RUNSPAN[(lab, LINEUP[mid][1])] = run_span(cells, tasks)
        pts.sort(key=lambda p: p[1])
        best, xs, ys, steps = -1.0, [], [], []
        for p in pts:
            if p[3] and p[2] > best:
                best = p[2]; xs.append(p[1]); ys.append(best); steps.append(p)
        if not xs:
            continue
        ax.step(xs + [END], ys + [ys[-1]], where="post", color=color, linewidth=3.2, zorder=2, solid_capstyle="round")
        # step points always; with show_all, every full-tier model too (smaller, still labeled) so e.g. GPT-6.1 Sol shows
        shown = [p for p in pts if p in steps or (show_all and p[3])]
        for p in shown:
            on = p in steps
            ax.scatter([p[1]], [p[2]], s=70 if on else 40, color=color, edgecolors=color, zorder=3)
            dx, dy, *text = jump_labels.get(p[0], (6, -15))  # optional 3rd item: label text ("" hides it)
            if text == [""]:
                continue
            ax.annotate(text[0] if text else p[0], (p[1], p[2]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=cs.LABEL_SIZE if on else cs.LABEL_SIZE - 1, color=TEXTC[lab],
                        ha="left" if dx >= 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color=color, alpha=0.35, lw=1, shrinkA=2, shrinkB=4)
                        if abs(dx) + abs(dy) >= 24 else None)
    cs.legend(ax, [cs.lab_handle(lab) for lab in labs])
    ax.set_xlim(START, END)
    ax.set_ylim(ylo, 1.04)  # ylo < 0 leaves room for labels under points that sit near 0
    if ylo < 0:  # spine would sit below 0, so draw the baseline at 0 instead (matches the easy chart)
        ax.spines["bottom"].set_visible(False)
        ax.axhline(0, color=cs.AXIS, linewidth=1, zorder=1, gid="ignore")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    cs.axis_label(ax, y=ylabel)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    if note:  # e.g. the hard chart: say why 2025 is empty, inside the empty region itself
        ax.text(*note_xy, note, fontsize=12.5, color=MUTED, style="italic", ha="center", va="center")
    cs.header(fig, title, subtitle, "weight space compilation")
    cs.save(fig, BLOG / name)


# ---------- 1. Hero: easy tier over time ----------
history(EASY, NOT_FRONTIER | EASY_SKIPPED, cs.LABS, "1-easy.png",
        "LLMs can now compile simple programs into neural net weights",
        "Best model so far from each lab on 7 easy programs (add, sort, GCD, ...), scored on hidden test inputs",
        {"o3": (8, 12), "GPT-5": (8, 12), "GPT-5.2": (-8, 10), "GPT-5.6 Sol": (-8, 14),
         "Opus 4.5": (8, -14), "Opus 4.8": (8, -14), "Opus 5.5": (-14, -4),
         "GLM-5": (8, -14), "GPT-6.1 Sol": (-6, 14), "GPT-5.5": (-8, 12), "DeepSeek R1 0528": (8, 12),
         "Opus 4.1": (-8, 0)},
        "Easy tier score")

# ---------- 2. Hard tier over time ----------
history(HARD, NOT_FRONTIER, cs.LABS, "2-hard.png",
        "OpenAI and Anthropic lead at compiling hard programs into neural nets",
        "Best model so far from each lab on 7 hard programs (mate-in-1, regex, Sudoku, ...), scored on hidden test inputs",
        {"GPT-5.4": (-8, 12), "GPT-5.5": (-8, 12), "GPT-6 Astra": (-14, 2),
         "GPT-6.2 Sol": (-2, 44, "GPT-6.2\nSol & Astra"), "GPT-6.2 Astra": (0, 0, ""), "Opus 4.8": (8, -12),
         "Fable 5.1": (-14, -6), "Opus 5.5": (4, -16), "Fable 5.2": (8, -15), "GPT-6.1 Sol": (2, -30, "GPT-6.1\nSol"), "GPT-5.6 Sol": (-8, 12), "Qwen4 Max": (-4, -18), "Kimi K3.1 Pro": (6, 16),
         "Qwen3.8 Max": (-8, 8), "GPT-5.2": (-8, 12), "Opus 4.5": (-8, 0), "Opus 4.7": (-12, -3), "GLM-5": (-8, 11), "Kimi K2.5": (-6, -14)},
        "Hard tier score", show_all=True,
        note="Earlier models weren't run\non the hard tier", note_xy=(dt.date(2025, 7, 20), 0.3), ylo=-0.06)

if "--all" not in sys.argv:  # the legacy launch charts below only re-render on request, so old PNGs stay put
    sys.exit(0)

# ---------- 3. Sudoku by difficulty (line chart) ----------
# exact-solve rate per difficulty bucket, from the verifier's per-bucket metrics (sudoku-buckets.json; budget-outs = 0)
SB = json.loads((HERE / "sudoku-buckets.json").read_text())
pick = [("openai/gpt-6.2-astra", "GPT-6.2 Astra", "OpenAI"), ("anthropic/claude-fable-5.2", "Fable 5.2", "Anthropic"),
        ("qwen/qwen4-max", "Qwen4 Max", "Open-weight labs")]  # best per lab; Sol/Opus 5.5 overlap these lines
cats = ["singles", "pairs_box_line", "search"]
fig, ax = canvas()
xs = [0, 1, 2]
ax.plot(xs, [1, 1, 1], color=INK, linestyle=(0, (4, 3)), linewidth=1.6, zorder=1)
ax.text(2.14, 1.0, "Backtracking program", fontsize=13, color=INK, va="center")
ends = []
for mid, label, lab in pick:
    ys = [SB[mid][f"solved_rate_{c}"] for c in cats]
    ls = "-" if lab in ("OpenAI", "Anthropic") else (0, (1, 2))
    mk = "o" if "6" in label or "5.2" in label else "s"
    col = COLOR[lab]
    ax.plot(xs, ys, color=col, linewidth=3, linestyle="-", marker="o", markersize=8, zorder=3,
            alpha=1 if lab in ("OpenAI", "Anthropic") else 0.85)
    # same scale, so same axis: cell accuracy as a dashed line in the same color
    yc = [SB[mid][f"cell_score_{c}"] for c in cats]
    ax.plot(xs, yc, color=col, linewidth=2.2, linestyle=(0, (5, 3)), marker="o", markersize=5, zorder=2,
            alpha=0.8, markerfacecolor=BG)
    ends.append((label, ys[1], TEXTC[lab]))
# label each line at the medium point (the hard end is 0% for everyone, so labels there would stack)
# model key in the empty upper right (inline labels kept landing on other lines)
for i, (label, y0, color) in enumerate(sorted(ends, key=lambda e: -e[1])):
    y = 0.90 - i * 0.065
    ax.plot([1.30, 1.40], [y, y], color=COLOR[next(l for m, n, l in pick if n == label)], linewidth=3)
    ax.text(1.43, y, label, fontsize=13, color=color, fontweight="bold", va="center")
ax.annotate("GPT-6.2 Astra 8%", (2, 0.08), xytext=(2.14, 0.13), fontsize=13, color=INK, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.annotate("~33% of cells", (2, 0.33), xytext=(2.14, 0.38), fontsize=12.5, color=INK, va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
# line-style key in the empty bottom-left
ax.plot([0.02, 0.07], [0.16, 0.16], transform=ax.transAxes, color=MUTED, linewidth=3, marker="o", markersize=7,
        markevery=[1])
ax.text(0.085, 0.16, "Whole puzzles solved", transform=ax.transAxes, fontsize=12.5, color=INK, va="center")
ax.plot([0.02, 0.07], [0.09, 0.09], transform=ax.transAxes, color=MUTED, linewidth=2.2, linestyle=(0, (5, 3)),
        marker="o", markersize=6, markerfacecolor=BG, markevery=[1])
ax.text(0.085, 0.09, "Empty cells filled correctly (−2 per wrong cell)", transform=ax.transAxes, fontsize=12.5, color=INK,
        va="center")
ax.set_xticks(xs)
ax.set_xticklabels(["Easy\n(single candidates)", "Medium\n(pairs, box–line)", "Hard\n(needs backtracking)"], fontsize=13)
ax.set_xlim(-0.15, 2.1)
ax.set_ylim(0, 1.06)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_ylabel("Score", fontsize=13, color=MUTED)
header(fig, "LLM-written nets almost never solve Sudokus that need backtracking",
       "Best model from each lab, by the hardest technique the puzzle needs. On puzzles that need backtracking,\n"
       "models fill in about a third of the empty cells, then leave the rest blank.",
       foot=FOOT)
fig.subplots_adjust(left=0.08, right=0.8, top=0.78, bottom=0.17)
fig.savefig(FIG / "launch-sudoku.png")
plt.close(fig)

# ---------- 4. Effort vs cost ----------
cost = json.loads((HERE / "effort-costs.json").read_text())
series = {"GPT-6.1 Sol": ("openai/gpt-6.1-sol", "sol", ["low", "medium", "high", "xhigh"], COLOR["OpenAI"]),
          "Opus 5.5": ("anthropic/claude-opus-5.5", "opus55", ["low", "medium", "high"], COLOR["Anthropic"])}
fig, ax = canvas()
ax.grid(axis="x", color=GRID, linewidth=1)
for name, (mid, key, levels, color) in series.items():
    xs, ys = [], []
    for lv in levels:
        cells = MAIN[mid] if lv == "high" else D["effort"].get(f"{mid}@{lv}", {})
        score, n = tier_mean(cells, HARD6)
        c = cost.get(f"{key}@{lv}")
        if score is None or not c:
            continue
        xs.append(c["usd_per_task"]); ys.append(score)
        capped = c["share_at_cap"] > 0
        ax.scatter([xs[-1]], [ys[-1]], s=150, color=BG if capped else color, edgecolors=color, linewidths=3, zorder=3)
        txt = (f"{lv}: {int(round(c['share_at_cap'] * 100))}% of replies\nran out of room") if capped else lv
        ax.annotate(txt, (xs[-1], ys[-1]), xytext=(-20, -58) if capped else (-16, 8), textcoords="offset points", fontsize=12.5,
                    color=INK if not capped else color, ha="center" if capped else "right")
    ax.plot(xs, ys, color=color, linewidth=3, zorder=2)
    ax.annotate(name, (xs[0], ys[0]), xytext=(-12, -26), textcoords="offset points", fontsize=15, color=color,
                fontweight="bold", ha="left")
ax.set_xscale("log")
ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2])
ax.set_xticklabels(["$0.05", "$0.10", "$0.20", "$0.50", "$1", "$2"])
ax.minorticks_off()
ax.set_xlim(0.05, 3.2)
ax.set_ylim(0, 1.04)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Cost per task (list price × tokens used, log scale)", fontsize=13, color=MUTED)
ax.set_ylabel("Hard tier score (6 tasks; mate-in-1 not in this sweep)", fontsize=12, color=MUTED)
header(fig, "More thinking helps, until the reply runs out of room",
       "Hard tier score at each reasoning effort setting. Every reply gets 128k output tokens; one that\n"
       "runs out before writing its weights scores 0.")
fig.subplots_adjust(left=0.08, right=0.95, top=0.78, bottom=0.15)
fig.savefig(FIG / "launch-effort.png")
plt.close(fig)

# ---------- 5. Hard-tier heatmap (blog) ----------
hrows = []
for mid, cells in MAIN.items():
    if mid not in LINEUP:
        continue
    score, n = tier_mean(cells, HARD)
    if n == len(HARD):
        if LINEUP[mid][0] == "google":  # Google left out of v1 charts until more Gemini models run (future work)
            continue
        hrows.append((LINEUP[mid][1], LAB.get(LINEUP[mid][0], "Other"), score, [np.mean(cells[t]) for t in HARD]))
hrows.sort(key=lambda r: -r[2])
grid = np.array([r[3] + [r[2]] for r in hrows])
fig, ax = plt.subplots(figsize=(12, 0.5 * len(hrows) + 2.4), dpi=200)
fig.patch.set_facecolor(BG)
ax.imshow(grid, cmap=plt.get_cmap("Blues"), vmin=-0.1, vmax=1.0, aspect="auto")
for i, r in enumerate(hrows):
    for j, v in enumerate(list(r[3]) + [r[2]]):
        if v < 0.05:
            continue
        txt = "✓" if (v >= 0.995 and j < len(HARD)) else f"{v:.2f}".lstrip("0")
        ax.text(j, i, txt, ha="center", va="center", fontsize=12 if txt == "✓" else 10.5,
                color="white" if v > 0.55 else INK, fontweight="bold" if j == len(HARD) else "normal")
band = sum(1 for r in hrows if r[2] >= 0.5)
ax.axhline(band - 0.5, color=INK, linewidth=1.2)
ax.axvline(len(HARD) - 0.5, color=BG, linewidth=6)
ax.set_xticks(range(len(HARD) + 1))
ax.set_xticklabels(["Mate\nin 1", "Regex\nmatch", "Decimal\nperiod", "Inverse\nBWT", "Empty\ncircumcircle", "Reed–Solomon\ndecode", "Sudoku",
                    "Mean"], fontsize=11.5, color=MUTED)
ax.xaxis.tick_top()
ax.set_yticks(range(len(hrows)))
ax.set_yticklabels([r[0] for r in hrows], fontsize=12)
for tick, r in zip(ax.get_yticklabels(), hrows):
    tick.set_color(COLOR.get(r[1], MUTED))
    tick.set_fontweight("bold")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
LEG = {k: v for k, v in COLOR.items() if k != "Google"}
handles = [plt.Line2D([], [], color=c, linewidth=0, marker="s", markersize=10) for c in LEG.values()]
fig.legend(handles, LEG.keys(), frameon=False, fontsize=12, loc="lower right", ncol=4, bbox_to_anchor=(0.98, 0.005))
fig.text(0.02, 0.985, "The top OpenAI and Anthropic models lead the hard tier", fontsize=20, color=INK,
         fontweight="bold", va="top")
fig.text(0.02, 0.945, "Mean score over 3 runs. ✓ = solved exactly every run. Blank = under 0.05. Line splits models above and below 50%.",
         fontsize=12, color=MUTED, va="top")
fig.text(0.02, 0.01, "WeightLiftingBench v1 · single-turn, tool-free: no code execution, no feedback · models that passed the easy tier · out of output budget counts as 0.", fontsize=10.5, color=FAINT)
fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.07)
fig.savefig(FIG / "launch-hard-heatmap.png")
plt.close(fig)
print("wrote launch charts")
