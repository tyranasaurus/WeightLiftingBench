"""Shared style for the WeightLiftingBench blog charts: canvas, colours, fonts, header, footer, legend and a layout check.

Every blog chart script imports this, so the look lives in one place. 12 x 6.75 in at 200 dpi = 2400 x 1350.
Colours come from plot_launch.py (validated with the dataviz validator, --pairs all).
"""
from __future__ import annotations

import itertools

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cbook import STEP_LOOKUP_MAP
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

BG, INK, MUTED, FAINT, GRID = "#fcfcfb", "#141413", "#5e5d59", "#9c9a92", "#e8e6dc"
AXIS = "#c3c2b7"  # baseline / spine
REF = "#8a887f"  # grey for reference lines (trained network, program, chance)
LABS = ("OpenAI", "Anthropic", "Open-weight labs")  # fixed order, never by rank
COLOR = {"OpenAI": "#139067", "Anthropic": "#e8622c", "Open-weight labs": "#2a6fd6", "Google": "#a8329e"}
# darker text shades of each lab hue (>= 4.5:1 on the background) so point labels read as their lab
TEXTC = {"OpenAI": "#0d6b4c", "Anthropic": "#b8461a", "Open-weight labs": "#1f55a8", "Google": "#7d2475"}

W, H, DPI = 12, 6.75, 200
LEFT_X = 0.035  # left edge of title, subtitle and footer
TITLE_SIZE, SUB_SIZE, FOOT_SIZE = 19, 13, 10.5  # 19 pt fits the longest blog title on one line
TICK_SIZE, AXLABEL_SIZE, LEGEND_SIZE, LABEL_SIZE = 12, 12.5, 13, 11.5
# data freeze per family: compilation scores and outside benchmarks were read Nov 16-17; reading and modeling Nov 20
DATE = {"weight space compilation": "data as of Nov 17, 2026", "weight space classification": "data as of Nov 20, 2026",
        "weight space modeling": "data as of Nov 20, 2026"}
FAMILIES = ("weight space compilation", "weight space classification", "weight space modeling")
# plot area that leaves room for a one-line title + subtitle above and the footer below
PLOT = dict(left=0.08, right=0.965, top=0.80, bottom=0.13)

plt.rcParams.update({"font.family": "DejaVu Sans"})


def canvas(grid: str = "y"):
    """Figure + axes with the light background, recessive grid and no top/right/left spines."""
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=TICK_SIZE, length=0)
    ax.tick_params(axis="y", pad=8)
    ax.tick_params(axis="x", pad=10)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    fig.subplots_adjust(**PLOT)
    return fig, ax


def header(fig, title: str, subtitle: str, family: str, source: str | None = None):
    """One-line bold title, one-line muted subtitle, one-line faint footer. Raises if any of them would not fit."""
    assert family in FAMILIES, family
    assert "\n" not in title + subtitle, "title and subtitle are one line each"
    texts = [
        fig.text(LEFT_X, 0.955, title, fontsize=TITLE_SIZE, color=INK, fontweight="bold", va="top"),
        fig.text(LEFT_X, 0.893, subtitle, fontsize=SUB_SIZE, color=MUTED, va="top"),
        fig.text(LEFT_X, 0.022, f"WeightLiftingBench v1 · {family} · {DATE[family]}" + (f" · {source}" if source else ""), fontsize=FOOT_SIZE, color=FAINT, va="bottom"),
    ]
    r = fig.canvas.get_renderer()
    for t in texts:
        w = t.get_window_extent(r).width / (W * DPI)
        assert LEFT_X + w <= 1 - LEFT_X + 1e-3, f"does not fit on one line ({w:.3f}): {t.get_text()!r}"
    return texts


def axis_label(ax, x: str | None = None, y: str | None = None):
    if x:
        ax.set_xlabel(x, fontsize=AXLABEL_SIZE, color=MUTED, labelpad=8)
    if y:
        ax.set_ylabel(y, fontsize=AXLABEL_SIZE, color=MUTED, labelpad=10)


def lab_handle(lab: str, kind: str = "line"):
    """Legend handle for a lab: 'line' (step/line charts), 'square' (bars) or 'dot' (scatter)."""
    if kind == "line":
        return Line2D([], [], color=COLOR[lab], linewidth=3.2, solid_capstyle="round", label=lab)
    marker = "s" if kind == "square" else "o"
    return Line2D([], [], color=COLOR[lab], marker=marker, markersize=11, linestyle="", label=lab)


def legend(ax, handles, *, loc="upper left", anchor=(0.005, 1.0), ncol=1, **kw):
    """Same legend treatment everywhere: no frame (gridlines run through), top-left inside the plot. Lab entries in bold lab text colour,
    other entries (encodings, reference lines) in muted ink."""
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=ncol, frameon=False,
                    fontsize=LEGEND_SIZE,
                    handlelength=1.8, handletextpad=0.6, labelspacing=0.45, columnspacing=1.6, borderaxespad=0.3, **kw)
    for t in leg.get_texts():
        lab = t.get_text()
        if lab in TEXTC:
            t.set_color(TEXTC[lab]); t.set_fontweight("bold")
        else:
            t.set_color(MUTED)
    leg.set_zorder(10)
    return leg


def save(fig, path):
    check_layout(fig)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


# ---------- layout check: text clipped by the figure, text on text, text on points / lines ----------
def _texts(fig):
    out = list(fig.texts)
    for ax in fig.axes:
        out += ax.texts + [ax.xaxis.label, ax.yaxis.label]
        out += [t for t in ax.get_xticklabels() + ax.get_yticklabels()]
        leg = ax.get_legend()
        if leg:
            out += leg.get_texts()
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def check_layout(fig, pad: float = 1.0):
    """Print every collision this can see. Text marked t.set_gid('inside') (labels drawn inside a bar) may sit on
    its own bar. It can't see meaning, so still look at the PNG."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.bbox.width, fig.bbox.height
    texts = _texts(fig)
    # Text.get_window_extent, not the Annotation override: that one also wraps the leader arrow
    boxes = [(t, Text.get_window_extent(t, r).padded(-pad)) for t in texts]
    problems = []
    for t, b in boxes:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fw or b.y1 > fh:
            problems.append(f"clipped: {t.get_text()!r}")
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if b1.overlaps(b2):
            problems.append(f"text/text: {t1.get_text()!r} x {t2.get_text()!r}")
    for ax in fig.axes:
        pts = []
        for c in ax.collections:
            if isinstance(c, PathCollection) and len(c.get_offsets()):
                sizes = c.get_sizes()
                for i, xy in enumerate(c.get_offset_transform().transform(c.get_offsets())):
                    rad = (sizes[i % len(sizes)] ** 0.5) / 2 * DPI / 72
                    pts.append((xy, rad))
        segs = []
        for ln in ax.get_lines():
            if ln.get_gid() == "ignore" or not ln.get_visible():
                continue
            data = ln.get_xydata()
            if ln.get_drawstyle() in STEP_LOOKUP_MAP and ln.get_drawstyle() != "default":  # step lines: draw-time path
                data = np.column_stack(STEP_LOOKUP_MAP[ln.get_drawstyle()](data[:, 0], data[:, 1]))
            xy = ln.get_transform().transform(data)
            lw = ln.get_linewidth() * DPI / 72 / 2
            for a, b in zip(xy, xy[1:]):
                segs.append((a, b, lw))
        for t, b in boxes:
            if t.axes is not ax or t.get_gid() == "free":
                continue
            for (x, y), rad in pts:
                if b.x0 - rad < x < b.x1 + rad and b.y0 - rad < y < b.y1 + rad:
                    problems.append(f"text/point: {t.get_text()!r}"); break
            for a, c, lw in segs:
                n = int(max(abs(c[0] - a[0]), abs(c[1] - a[1])) // 3) + 2
                if any(b.x0 - lw < a[0] + (c[0] - a[0]) * k / n < b.x1 + lw and
                       b.y0 - lw < a[1] + (c[1] - a[1]) * k / n < b.y1 + lw for k in range(n + 1)):
                    problems.append(f"text/line: {t.get_text()!r}"); break
            if t.get_gid() != "inside":
                for p in ax.patches:
                    if isinstance(p, Rectangle) and p.get_visible():
                        pb = p.get_window_extent(r)
                        if pb.width > 1 and pb.height > 1 and b.overlaps(pb):
                            problems.append(f"text/bar: {t.get_text()!r}"); break
        leg = ax.get_legend()
        if leg:
            lb = leg.get_window_extent(r)
            for (x, y), rad in pts:
                if lb.x0 - rad < x < lb.x1 + rad and lb.y0 - rad < y < lb.y1 + rad:
                    problems.append("legend/point"); break
            for a, c, lw in segs:
                n = int(max(abs(c[0] - a[0]), abs(c[1] - a[1])) // 3) + 2
                if any(lb.x0 < a[0] + (c[0] - a[0]) * k / n < lb.x1 and lb.y0 < a[1] + (c[1] - a[1]) * k / n < lb.y1
                       for k in range(n + 1)):
                    problems.append("legend/line"); break
            for t, b in boxes:
                if t.axes is ax and t not in leg.get_texts() and b.overlaps(lb):
                    problems.append(f"legend/text: {t.get_text()!r}")
    for p in problems:
        print("  LAYOUT:", p)
    if not problems:
        print("  layout check: clean")
    return problems
