"""Draw results/figures/hard-tier-heatmap.png from results/blog_v1_runs.csv.

One row per OpenAI, Anthropic or open-weight lab model with replies on all 7 hard tasks, one column per task. A cell is
the mean reward_fixed_count of that model's replies on the task (at most 3; out of output budget counts as 0).
Usage: uv run --with matplotlib python scripts/hard_tier_heatmap.py
"""

import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "results" / "blog_v1_runs.csv"
FIGURE = ROOT / "results" / "figures" / "hard-tier-heatmap.png"

TASKS = {
    "chess-mate1": "Mate\nin 1",
    "regex-match": "Regex\nmatch",
    "decimal-period": "Decimal\nperiod",
    "inverse-bwt": "Inverse\nBWT",
    "empty-circumcircle": "Empty\ncircumcircle",
    "reed-solomon-decode": "Reed-Solomon\ndecode",
    "sudoku-any": "Sudoku",
}
LABS = {
    "openai": ("OpenAI", "#12906a"),
    "anthropic": ("Anthropic", "#e8602c"),
    "open": ("Open-weight labs", "#2a6fdb"),
}
OPEN_WEIGHT_PREFIXES = ("deepseek/", "moonshotai/", "z-ai/", "qwen/")
NAME_PARTS = {"gpt": "GPT", "glm": "GLM", "claude": "", "max": "Max", "pro": "Pro"}
BLANK_BELOW = 0.05
SURFACE = "#fcfcfb"
INK, MUTED = "#1a1a19", "#6b6b68"
SCALE = LinearSegmentedColormap.from_list("score", ["#e9f0fa", "#9cc3e6", "#2f7dc0", "#0b3d87", "#072a63"])


def lab_of(model: str) -> str | None:
    if model.startswith("openai/"):
        return "openai"
    if model.startswith("anthropic/"):
        return "anthropic"
    return "open" if model.startswith(OPEN_WEIGHT_PREFIXES) else None


def display_name(model: str) -> str:
    slug = model.split("/")[1].removesuffix("-0902")
    words = [NAME_PARTS.get(w, w.capitalize()) for w in slug.split("-")]
    words = [w for w in words if w]
    if words[0] in ("GPT", "GLM"):
        return f"{words[0]}-{words[1]}" + "".join(f" {w}" for w in words[2:])
    return " ".join(words)


def load() -> list[dict]:
    cells = collections.defaultdict(list)
    for row in csv.DictReader(ROWS.open()):
        if row["tier"] == "hard":
            cells[(row["model"], row["task"])].append(float(row["reward_fixed_count"]))
    models = []
    for model in sorted({m for m, _ in cells}):
        lab = lab_of(model)
        if lab is None or any((model, task) not in cells for task in TASKS):
            continue
        rewards = [cells[(model, task)] for task in TASKS]
        means = [sum(r) / len(r) for r in rewards]
        models.append({"name": display_name(model), "lab": lab, "means": means,
                       "exact": [min(r) == 1.0 for r in rewards], "mean": sum(means) / len(means)})
    return sorted(models, key=lambda m: -m["mean"])


def cell_text(*, value: float, exact: bool) -> str:
    if exact:
        return "✓"
    return "" if value < BLANK_BELOW else f"{value:.2f}".lstrip("0")


def main() -> None:
    models = load()
    n = len(models)
    fig, ax = plt.subplots(figsize=(12, 2.9 + 0.5 * n), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    gap = 0.12
    for i, model in enumerate(models):
        values = model["means"] + [model["mean"]]
        exact = model["exact"] + [False]
        for j, value in enumerate(values):
            x = j + (gap if j == len(TASKS) else 0)
            ax.add_patch(Rectangle((x, i), 1, 1, facecolor=SCALE(value), edgecolor=SURFACE, linewidth=1.5))
            ax.text(x + 0.5, i + 0.5, cell_text(value=value, exact=exact[j]), ha="center", va="center", fontsize=11,
                    color="white" if value >= 0.45 else INK, fontweight="bold" if j == len(TASKS) else "normal")
        ax.add_patch(Rectangle((-0.16, i + 0.32), 0.09, 0.36, facecolor=LABS[model["lab"]][1], clip_on=False))
        ax.text(-0.24, i + 0.5, model["name"], ha="right", va="center", fontsize=12, color=INK, fontweight="bold")
    split = sum(m["mean"] >= 0.5 for m in models)
    ax.plot([0, len(TASKS) + 1 + gap], [split, split], color=INK, linewidth=1.2)
    for j, label in enumerate([*TASKS.values(), "Mean"]):
        ax.text(j + 0.5 + (gap if j == len(TASKS) else 0), -0.15, label, ha="center", va="bottom", fontsize=9.5,
                color=MUTED)
    ax.set_xlim(-2.2, len(TASKS) + 1 + gap)
    ax.set_ylim(n + 1.1, -2.6)
    ax.axis("off")
    ax.text(-2.2, -2.3, "The top OpenAI and Anthropic models lead the hard tier", fontsize=19, fontweight="bold",
            color=INK, va="center")
    ax.text(-2.2, -1.65, "Mean score over up to 3 replies. ✓ = solved exactly on every reply. Blank = under 0.05. "
            "Line splits models above and below 50%.", fontsize=11, color=MUTED, va="center")
    for k, (label, color) in enumerate(LABS.values()):
        x = 0.1 + 1.9 * k
        ax.add_patch(Rectangle((x, n + 0.42), 0.12, 0.3, facecolor=color))
        ax.text(x + 0.2, n + 0.57, label, fontsize=11, color=INK, va="center")
    ax.text(-2.2, n + 1.0, "WSC-Bench v1 · one reply, no tools, no code execution, no feedback · models with replies "
            "on all 7 hard tasks · sparse nonzero count fixed · out of output budget counts as 0", fontsize=8.5,
            color=MUTED, va="center")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"{n} models -> {FIGURE.relative_to(ROOT)}")
    for model in models:
        print(f"  {model['name']:16} {model['mean']:.3f}")


if __name__ == "__main__":
    main()
