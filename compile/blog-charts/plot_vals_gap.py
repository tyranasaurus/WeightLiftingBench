"""Best OpenAI, Anthropic and open-weight model: Vals.ai Index, Artificial Analysis, Terminal-Bench, WeightLiftingBench hard tier.

uv run --quiet --no-project --with matplotlib python plot_vals_gap.py
Writes figures/blog/3-gap.png in the shared chart_style.
Scores from vals-gap-notes.md. The open-weight bar is Kimi K3.1 Pro in every group for consistency
(Vals.ai Index: Kimi K3.1 Pro 62.52, raw-confirmed in the notes; the best open-weight entry there is Qwen 4 Max 63.20).
"""
from pathlib import Path

import chart_style as cs

HERE = Path(__file__).parent
OPEN = "Open-weight labs"

# (group label, [(best model, lab, score) for OpenAI, Anthropic, open-weight labs])
PANELS = [
    ("Vals.ai\nIndex", [("GPT-6.2\nAstra", "OpenAI", 67.66), ("Fable\n5.2", "Anthropic", 71.31),
                         ("Kimi\nK3.1 Pro", OPEN, 62.52)]),
    ("Artificial Analysis\nIntelligence Index", [("GPT-6.2\nAstra", "OpenAI", 62.1), ("Fable\n5.2", "Anthropic", 64.56),
                                                  ("Kimi\nK3.1 Pro", OPEN, 46.5)]),
    ("Terminal-Bench\n4.1", [("GPT-6.1\nSol", "OpenAI", 60.9), ("Fable\n5.2", "Anthropic", 65.8),
                              ("Kimi\nK3.1 Pro", OPEN, 59.7)]),
    ("WeightLiftingBench\nhard tier", [("GPT-6\nAstra", "OpenAI", 94.4), ("Fable\n5.2", "Anthropic", 85.2),
                               ("Kimi\nK3.1 Pro", OPEN, 29.3)]),
]

fig, ax = cs.canvas()
w = 0.27
for i, (name, bars) in enumerate(PANELS):
    for k, (m, lab, v) in enumerate(bars):
        x = i + (k - 1) * (w + 0.015)
        ax.bar(x, v, w, color=cs.COLOR[lab], zorder=2)
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=15, fontweight="bold", color=cs.TEXTC[lab])
        t = ax.text(x, v - 2.5, m, ha="center", va="top", fontsize=8.5, color="white", fontweight="bold", linespacing=1.3)
        t.set_gid("inside")
    last = i == len(PANELS) - 1
    gap = max(v for _, lab, v in bars if lab != OPEN) / bars[2][2]
    print(name.replace("\n", " "), f"gap {gap:.3f}")
    ax.text(i, max(v for *_, v in bars) + 10, f"{gap:.1f}x gap", ha="center", fontsize=19 if last else 16, fontweight="bold",
            color="#b3261e" if last else cs.INK)
ax.set_xticks(range(len(PANELS)))
ax.set_xticklabels([p[0] for p in PANELS], fontsize=12.5, color=cs.INK)
ax.set_ylim(0, 128); ax.set_xlim(-0.55, len(PANELS) - 0.45)
ax.set_yticks([0, 25, 50, 75, 100])
ax.axvline(len(PANELS) - 1.5, color=cs.AXIS, linewidth=1, linestyle=(0, (4, 3)), gid="ignore")
cs.legend(ax, [cs.lab_handle(lab, "square") for lab in cs.LABS])
cs.header(fig, "Open-weight models trail far behind at weight space compilation",
          "Best OpenAI, Anthropic and open-weight model on each benchmark. Gap = best of OpenAI and Anthropic ÷ open-weight.",
          "weight space compilation", source="other scores: Vals.ai, Artificial Analysis, Terminal-Bench")
fig.subplots_adjust(bottom=0.16)
cs.save(fig, HERE / "figures" / "blog" / "3-gap.png")
