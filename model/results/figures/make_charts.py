"""Draw the three Weight Space Modeling charts in the blog's chart style (2400 x 1350, off-white, lab colours).
Usage: uv run --no-project --with matplotlib python make_charts.py"""
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
BG = "#FCFCFA"
INK, SUB, FOOT, GRID, AXIS = "#111111", "#555555", "#999999", "#E6E6DF", "#BBBBBB"
LINE = {"OpenAI": "#15906A", "Anthropic": "#E8622C", "Open-weight labs": "#2A6FD6"}
TEXT = {"OpenAI": "#0E6B4B", "Anthropic": "#B5451B", "Open-weight labs": "#1F4FA5"}
LEFT = 0.035


def frame(title, subtitle, footer, *, left, right=0.97, bottom=0.13, top=0.77):
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG)
    ax = fig.add_axes([left, bottom, right - left, top - bottom], facecolor=BG)
    lines = title.split("\n")
    fig.text(LEFT, 0.96, title, fontsize=28, fontweight="bold", color=INK, va="top", linespacing=1.25)
    fig.text(LEFT, 0.96 - 0.075 * len(lines) + 0.005, subtitle, fontsize=15.5, color=SUB, va="top", linespacing=1.45)
    fig.text(LEFT, 0.028, footer, fontsize=13.5, color=FOOT, va="center")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=16, labelcolor=SUB)
    return fig, ax


# ---------- chart 1: skill score by release date, best model so far per lab ----------
MODELS = [  # name, lab, release date, mean skill score over the 7 tasks (a rule break counts as 0% accuracy)
    ("GPT-5.4", "OpenAI", "2026-03-05", 0.071), ("Opus 4.7", "Anthropic", "2026-04-16", -0.006),
    ("GPT-5.5", "OpenAI", "2026-04-24", 0.108), ("Opus 4.8", "Anthropic", "2026-05-27", 0.078),
    ("GPT-5.6 Sol", "OpenAI", "2026-07-09", 0.131), ("GLM-5.3", "Open-weight labs", "2026-08-18", -0.054),
    ("Fable 5.1", "Anthropic", "2026-09-01", 0.312), ("Qwen3.8 Max", "Open-weight labs", "2026-09-02", 0.055),
    ("GPT-6 Astra", "OpenAI", "2026-09-04", 0.386), ("Opus 5.5", "Anthropic", "2026-09-22", 0.434),
    ("GPT-6.1 Sol", "OpenAI", "2026-09-29", 0.354), ("Kimi K3.1 Pro", "Open-weight labs", "2026-10-23", 0.218),
    ("Fable 5.2", "Anthropic", "2026-11-09", 0.433), ("Qwen 4 Max", "Open-weight labs", "2026-11-15", 0.217),
    ("GPT-6.2 Sol", "OpenAI", "2026-11-17", 0.361), ("GPT-6.2 Astra", "OpenAI", "2026-11-17", 0.519),
]
REFS = [("Trained reference network", 0.891), ("Best deterministic program", 0.609)]
# label offset per model: (days, skill units, horizontal alignment)
OFFSET = {
    "GPT-5.4": (0, 0.035, "center"), "GPT-5.5": (0, 0.035, "center"), "GPT-5.6 Sol": (0, 0.035, "center"),
    "GPT-6 Astra": (-4, 0.02, "right"), "GPT-6.1 Sol": (4, -0.05, "left"), "GPT-6.2 Sol": (5, 0.0, "left"),
    "GPT-6.2 Astra": (-4, 0.02, "right"), "Opus 4.7": (0, -0.065, "center"), "Opus 4.8": (4, -0.055, "left"),
    "Fable 5.1": (-4, 0.02, "right"), "Opus 5.5": (0, 0.03, "center"), "Fable 5.2": (-3, 0.03, "right"),
    "GLM-5.3": (-4, -0.012, "right"), "Qwen3.8 Max": (4, -0.06, "left"), "Kimi K3.1 Pro": (4, -0.065, "left"),
    "Qwen 4 Max": (0, 0.03, "center"),
}


def chart_skill():
    fig, ax = frame(
        "Written networks are improving, but none beats the program",
        "Best model so far from each lab on 7 image tasks, each network designed and printed by the model with no training.\n"
        "Skill score = (accuracy − chance) / (1 − chance), averaged over the tasks. A rule break counts as 0% accuracy.",
        "No-copy design · 90 minutes in a sandbox with the training set · at most 5 checks · data-free rebuild · 3 runs per task · data as of Nov 20, 2026",
        left=0.12)
    day = lambda s: dt.datetime.strptime(s, "%Y-%m-%d")
    x0, x1 = day("2026-02-15"), day("2026-12-22")
    for v in (-0.25, 0, 0.25, 0.5, 0.75, 1):
        ax.axhline(v, color=GRID, lw=1.5, zorder=1)
    for name, level in REFS:
        ax.axhline(level, color="#8A8A85", lw=1.6, ls=(0, (5, 4)), zorder=2)
        ax.text(x0 + dt.timedelta(days=4), level + 0.014, f"{name} {level:.2f}", fontsize=14.5, color=SUB, va="bottom")
    for lab in LINE:
        ms = sorted((day(d), s, n) for n, l, d, s in MODELS if l == lab)
        best, xs, ys = None, [], []
        for d, s, n in ms:
            record = best is None or s > best
            tie = not record and best - s < 0.005  # within rounding of the lab's best so far: drawn as large
            if record:
                best = s; xs.append(d); ys.append(s)
            ax.scatter([d], [s], s=300 if record or tie else 120, color=LINE[lab], zorder=5, edgecolors="none")
            dx, dy, ha = OFFSET[n]
            ax.text(d + dt.timedelta(days=dx), s + dy, n, fontsize=16 if record or tie else 14, color=TEXT[lab], ha=ha, va="bottom" if dy > 0 else "center", zorder=6)
        ax.step(xs + [x1], ys + [ys[-1]], where="post", color=LINE[lab], lw=5, zorder=4, solid_joinstyle="round", solid_capstyle="butt")
    for i, lab in enumerate(LINE):
        y = 0.52 - 0.085 * i
        ax.plot([x0 + dt.timedelta(days=6), x0 + dt.timedelta(days=18)], [y, y], color=LINE[lab], lw=5, solid_capstyle="round", zorder=4)
        ax.text(x0 + dt.timedelta(days=23), y, lab, fontsize=20, fontweight="bold", color=TEXT[lab], va="center")
    ax.set_xlim(x0, x1); ax.set_ylim(-0.29, 1.04)
    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["-0.25", "0 (guessing)", "0.25", "0.50", "0.75", "1 (all right)"])
    ax.set_ylabel("Skill score", fontsize=16, color=SUB, labelpad=14)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 5, 7, 9, 11])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", pad=10)
    fig.savefig(OUT / "1-skill-score-by-release.png", facecolor=BG); plt.close(fig)


# ---------- chart 2: how alike a network and a program answer, against two programs ----------
SIM = [  # task, network against program score, program against program score (agreement beyond chance)
    ("Digits", 0.298, 0.350), ("Fashion", 0.392, 0.476), ("Traffic signs", 0.176, 0.237), ("Galaxies", 0.240, 0.397),
    ("Letters", 0.112, 0.123), ("CIFAR-10", 0.173, 0.209), ("Person", 0.334, 0.315),
]


def chart_similarity():
    fig, ax = frame(
        "How similarly Astra networks and programs answer, by task",
        "How alike a network and a program answer, image by image, as a share of how alike two programs answer.\n"
        "All written separately by GPT-6.2 Astra for the same task, under the same rules.\n"
        "Inside each bar: the two scores, agreement beyond chance on the full test set (0 = unrelated, 1 = identical answers).",
        "No-copy design · GPT-6.2 Astra's 3 programs and 3 networks per task · every pair of runs compared on the full test set, then averaged · data as of Nov 20, 2026",
        left=0.10, bottom=0.15, top=0.72)
    rows = sorted(SIM, key=lambda r: -r[1] / r[2])
    for v in (0, 25, 50, 75):
        ax.axhline(v, color=GRID if v else "#8A8A85", lw=1.5 if v else 2, zorder=1 if v else 4)
    ax.axhline(100, color="#8A8A85", lw=1.8, ls=(0, (5, 4)), zorder=2)
    ax.text(len(rows) - 0.55, 102.5, "As alike as two Astra programs", fontsize=15, color=SUB, ha="right", va="bottom")
    for i, (task, net, prog) in enumerate(rows):
        share = 100 * net / prog
        ax.bar(i, share, width=0.62, color=LINE["OpenAI"], zorder=3)
        ax.text(i, share + 1.5, f"{share:.0f}", fontsize=26, fontweight="bold", color=TEXT["OpenAI"], ha="center", va="bottom", zorder=5,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))
        ax.text(i, share - 4, f"{net:.2f}\nof {prog:.2f}", fontsize=15, fontweight="bold", color="white", ha="center", va="top", linespacing=1.3, zorder=5)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=19, color=INK)
    ax.tick_params(axis="x", pad=14)
    ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.spines["bottom"].set_visible(False)
    fig.savefig(OUT / "2-network-vs-program-similarity.png", facecolor=BG); plt.close(fig)


# ---------- chart 3: fine-tuning gains ----------
FT = [  # task, model, lab, printed weights accuracy, accuracy after 3 epochs
    ("MNIST", "GPT-6.2 Astra", "OpenAI", 0.935, 0.988), ("MNIST", "Fable 5.2", "Anthropic", 0.962, 0.688),
    ("GTSRB", "GPT-6.2 Astra", "OpenAI", 0.783, 0.910), ("GTSRB", "Fable 5.2", "Anthropic", 0.498, 0.747),
    ("Fashion-MNIST", "GPT-6.2 Astra", "OpenAI", 0.675, 0.885), ("Fashion-MNIST", "Fable 5.2", "Anthropic", 0.729, 0.793),
    ("Sign Language MNIST", "GPT-6.2 Astra", "OpenAI", 0.479, 0.933), ("Sign Language MNIST", "Fable 5.2", "Anthropic", 0.327, 0.849),
    ("Galaxy10", "GPT-6.2 Astra", "OpenAI", 0.484, 0.361), ("Galaxy10", "Fable 5.2", "Anthropic", 0.391, 0.579),
    ("CIFAR-10", "GPT-6.2 Astra", "OpenAI", 0.303, 0.627), ("CIFAR-10", "Fable 5.2", "Anthropic", 0.172, 0.414),
    ("Visual Wake Words", "GPT-6.2 Astra", "OpenAI", 0.590, 0.720), ("Visual Wake Words", "Fable 5.2", "Anthropic", 0.594, 0.649),
]
TASK_REF = {  # task: (trained reference network, best deterministic program)
    "MNIST": (0.997, 0.969), "GTSRB": (0.979, 0.883), "Fashion-MNIST": (0.948, 0.753), "Sign Language MNIST": (0.918, 0.583),
    "Galaxy10": (0.824, 0.603), "CIFAR-10": (0.917, 0.329), "Visual Wake Words": (0.845, 0.657),
}


def chart_finetune():
    ups = sum(a > b for *_, b, a in FT); gain = 100 * sum(a - b for *_, b, a in FT) / len(FT)
    fig, ax = frame(
        "Gradient descent improves almost every printed network",
        f"{ups} of {len(FT)} model and task averages improve after 3 epochs of fine-tuning, by {gain:.1f} points on average.\n"
        "Hollow dot: accuracy of the weights as the model printed them. Filled dot: after fine-tuning. Right: change in points.",
        "No-copy design · all 42 rule-following GPT-6.2 Astra and Fable 5.2 networks, averaged per task · Adam, learning rate 3 × 10⁻⁴, 3 epochs · data as of Nov 20, 2026",
        left=0.20, right=0.90, bottom=0.12, top=0.70)
    tasks = list(TASK_REF)
    for v in (0, 0.25, 0.5, 0.75, 1):
        ax.axvline(v, color=GRID, lw=1.5, zorder=1)
    for i, t in enumerate(tasks):
        trained, prog = TASK_REF[t]
        ax.plot([trained, trained], [i - 0.36, i + 0.36], color="#8A8A85", lw=3, zorder=2)
        ax.plot([prog, prog], [i - 0.36, i + 0.36], color="#8A8A85", lw=3, ls=(0, (2, 1.6)), zorder=2)
    for t, model, lab, before, after in FT:
        y = tasks.index(t) + (-0.16 if lab == "OpenAI" else 0.16)
        ax.plot([before, after], [y, y], color=LINE[lab], lw=5.5, alpha=0.45, zorder=3, solid_capstyle="butt")
        ax.scatter([before], [y], s=230, facecolors=BG, edgecolors=LINE[lab], linewidths=3.2, zorder=4)
        ax.scatter([after], [y], s=260, color=LINE[lab], zorder=5, edgecolors="none")
        d = 100 * (after - before)
        ax.text(1.115, y, f"{'+' if d >= 0 else '−'}{abs(d):.1f}", fontsize=17, fontweight="bold", color=TEXT[lab], ha="right", va="center", clip_on=False)
    ax.text(1.115, -0.72, "Change", fontsize=15, color=SUB, ha="right", va="center", clip_on=False)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks, fontsize=18, color=INK)
    ax.tick_params(axis="y", pad=16)
    ax.set_ylim(len(tasks) - 0.45, -0.55); ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Test accuracy", fontsize=16, color=SUB, labelpad=10)
    ax.spines["bottom"].set_visible(False)
    handles = [
        Line2D([], [], marker="o", ls="", ms=13, color=LINE["OpenAI"], label="GPT-6.2 Astra"),
        Line2D([], [], marker="o", ls="", ms=13, color=LINE["Anthropic"], label="Fable 5.2"),
        Line2D([], [], color="#8A8A85", lw=3, label="Trained reference network"),
        Line2D([], [], color="#8A8A85", lw=3, ls=(0, (2, 1.6)), label="Best deterministic program"),
    ]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(LEFT - 0.008, 0.80), ncol=4, frameon=False, fontsize=16,
                     handlelength=1.6, columnspacing=2.2, handletextpad=0.6)
    for txt, lab in zip(leg.get_texts()[:2], ("OpenAI", "Anthropic")):
        txt.set_color(TEXT[lab]); txt.set_fontweight("bold")
    for txt in leg.get_texts()[2:]:
        txt.set_color(SUB)
    fig.savefig(OUT / "3-fine-tuning-gains.png", facecolor=BG); plt.close(fig)


# ---------- chart 4: skill score against training images per class ----------
LEVELS = ["1", "10", "100", "full set"]
TRAIN = [  # name, lab, dashed, mean skill score over the 7 tasks at 1, 10, 100 per class and the full set
    ("GPT-6.2 Astra programs", "OpenAI", True, [0.48, 0.54, 0.55, 0.57]),
    ("GPT-6.2 Astra networks", "OpenAI", False, [0.44, 0.47, 0.50, 0.52]),
    ("Fable 5.2 networks", "Anthropic", False, [0.43, 0.42, 0.50, 0.43]),
]
# 95% intervals (low, high) per level: runs resampled within each task, tasks fixed (nocopy-galaxies/scripts/levels_ci.py)
TRAIN_CI = {
    "GPT-6.2 Astra programs": [(0.461, 0.496), (0.530, 0.549), (0.537, 0.569), (0.551, 0.593)],
    "GPT-6.2 Astra networks": [(0.423, 0.468), (0.456, 0.486), (0.490, 0.516), (0.505, 0.532)],
    "Fable 5.2 networks": [(0.402, 0.450), (0.377, 0.455), (0.476, 0.519), (0.421, 0.445)],
}


def chart_training():
    fig, ax = frame(
        "Fewer training images barely change the scores",
        "Mean skill score on the 7 image tasks when the sandbox holds only 1, 10 or 100 training images per class,\n"
        "against the full training set. Same rules as before; 3 runs per task at each level. Bars: 95% interval from run to run noise.",
        "No-copy design · GPT-6.2 Astra and Fable 5.2 networks and GPT-6.2 Astra programs · full set from the main experiment · data as of Nov 20, 2026",
        left=0.09, right=0.70)
    xs = range(len(LEVELS))
    for v in (0, 0.25, 0.5, 0.75, 1):
        ax.axhline(v, color=GRID, lw=1.5, zorder=1)
    for name, level in (("Trained reference network, full training set", 0.891), ("Chance", 0.0)):
        ax.axhline(level, color="#8A8A85", lw=1.6, ls=(0, (5, 4)), zorder=2)
        ax.text(-0.05, level + 0.015, f"{name} {level:.2f}", fontsize=14.5, color=SUB, va="bottom")
    ends = []
    for i, (name, lab, dashed, vals) in enumerate(TRAIN):
        xs = [x + (i - 1) * 0.06 for x in range(len(LEVELS))]  # small sideways shift so the interval bars don't overlap
        ax.plot(xs, vals, color=LINE[lab], lw=5, ls=(0, (4, 2.5)) if dashed else "-", zorder=4, solid_capstyle="round")
        ax.scatter(xs, vals, s=220, color=LINE[lab], zorder=5, edgecolors="none")
        lo, hi = zip(*TRAIN_CI[name])
        ax.errorbar(xs, vals, yerr=[[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]], fmt="none",
                    ecolor=LINE[lab], elinewidth=2.5, capsize=9, capthick=2.5, zorder=6)
        ends.append([vals[-1], name, lab])
    ends.sort(reverse=True)
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 0.05)
    for y, name, lab in ends:
        ax.text(len(LEVELS) - 1 + 0.17, y, f"{name} {dict((n, v) for n, _, _, v in TRAIN)[name][-1]:.2f}", fontsize=17, fontweight="bold",
                color=TEXT[lab], va="center", clip_on=False)
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS, fontsize=18, color=INK)
    ax.set_xlabel("Training images per class in the sandbox", fontsize=16, color=SUB, labelpad=12)
    ax.set_xlim(-0.15, len(LEVELS) - 0.85); ax.set_ylim(-0.08, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0 (guessing)", "0.25", "0.50", "0.75", "1 (all right)"])
    ax.set_ylabel("Skill score", fontsize=16, color=SUB, labelpad=14)
    fig.savefig(OUT / "4-training-images.png", facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    chart_skill(); chart_similarity(); chart_finetune(); chart_training()
    print("wrote 4 charts to", OUT)
