"""Weight reading by family: newest OpenAI, Anthropic and open-weight model on each family (Atari, BERT, style).

uv run --quiet --no-project --with matplotlib python scripts/plot_reading_families.py
Reads results/scores.json (from score.py). Writes results/figures/6-reading-families.png in the shared
chart_style. LoRA is not scored (a negative control), so its panel is drawn in grey.
A model is eligible for a family once >= COVER of that family's tasks are scored (no lucky partial runs).
Ties go to the newer model. No gap labels: the bars show it.
"""
import json
from pathlib import Path

import chart_style as cs

HERE = Path(__file__).parent
COLOR, TEXTC, INK = cs.COLOR, cs.TEXTC, cs.INK
FAMS = [("atari", "Atari game\npolicies", 40), ("bert", "BERT\nfine-tunes", 17),
        ("style", "Style-transfer\nnets", 5), ("lora", "LoRA adapters\n(not scored)", 8)]
GREY, GREY_TEXT = "#c9c7bf", "#8a887f"
COVER = 0.85
# Each lab's current generation only: for similar scores prefer the newer model, never an older one
CURRENT = {"GPT-6.2 Astra", "Fable 5.2", "Kimi K3.1 Pro"}  # one model per lab


def lab(m: str) -> str:
    return "OpenAI" if m.startswith("GPT") else "Anthropic" if m.startswith(("Opus", "Fable")) else "Open-weight labs"


scores = json.loads((HERE.parent / "results/scores.json").read_text())
panels = []
for key, title, n_all in FAMS:
    best = {}
    for m, r in scores.items():
        if key not in r or r[key]["n"] < COVER * n_all or m not in CURRENT: continue
        L = lab(m); cand = (r[key]["skill"], r["date"], m, r[key]["n"])
        if L not in best or cand[:2] > best[L][:2]: best[L] = cand
    panels.append((key, title, n_all, [(L, *best[L]) for L in COLOR if L in best]))

fig, ax = cs.canvas()
ax.spines["bottom"].set_visible(False)
ax.axhline(0, color="#8a887f", linewidth=1.2, zorder=3)
w = 0.27
for i, (key, title, n_all, bars) in enumerate(panels):
    grey = key == "lora"
    for k, (L, sk, date, m, n) in enumerate(bars):
        x = i + (k - 1) * (w + 0.015); v = 100 * sk
        ax.bar(x, v, w, color=GREY if grey else COLOR[L], zorder=2)
        top = max(v, 0)
        ax.text(x, top + 2.5, f"{v:+.0f}" if v < 0 else f"{v:.0f}", ha="center", fontsize=15, fontweight="bold", color=GREY_TEXT if grey else TEXTC[L])
        name = m.replace(" ", "\n", 1)
        if v >= 14:
            ax.text(x, v - 2.5, name, ha="center", va="top", fontsize=8.5, color="white", fontweight="bold", linespacing=1.3,
                    gid="inside")
        else:  # bar too short for an inside label: name sits below the axis line, under the bar
            ax.text(x, min(v, 0) - 2.5, name, ha="center", va="top", fontsize=8.5, color=GREY_TEXT if grey else TEXTC[L], fontweight="bold",
                    linespacing=1.3)
ax.set_xticks(range(len(panels)))
ax.set_xticks([]); [ax.text(i, -34, f"{t.replace(chr(10), ' ')}\n({n} tasks)", ha="center", va="top", fontsize=12.5, color=INK, linespacing=1.3)
                    for i, (_, t, n, _) in enumerate(panels)]
ax.tick_params(axis="x", pad=8)
ax.set_ylim(-32, 140)  # headroom above the bars for the legend; ax.set_xlim(-0.55, len(panels) - 0.45)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0 (guessing)", "25", "50", "75", "100"])
cs.legend(ax, [cs.lab_handle(lab, "square") for lab in cs.LABS])
cs.header(fig, "Breakdown by type of reading task",
          "Newest model from each lab on each type of trained network. Skill score × 100 (0 = guessing).",
          "weight space classification")
fig.subplots_adjust(left=0.1, bottom=0.15)
cs.save(fig, HERE.parent / "results" / "figures" / "6-reading-families.png")
for _, t, n, b in panels: print(t.replace("\n", " "), [(L[:4], m, round(sk, 2), nn) for L, sk, d, m, nn in b])
