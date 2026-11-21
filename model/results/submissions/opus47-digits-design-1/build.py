"""Build weights for a hand-designed MNIST classifier.

For each digit we draw several stroke-templates. Each template is a
single generated image that contains both positive strokes (where ink
is expected) and "anti-strokes" drawn at negative value (where ink
indicates a different digit). We also put a thin negative shadow
around the positive strokes. Each template is normalized by itself
alone (zero-mean, unit-norm).

Architecture: norm the input, convolve with (10 * V) 26x26 templates
(padding 0, so output is 3x3 shifts), spatial maxpool, group by class,
maxpool across variants, reshape to (10,).
"""
import json
import os
import sys

import numpy as np

TH = 26  # template size
H, W = 28, 28
V = 6
STROKE_W = 2.5
SHADOW_WEIGHT = 0.25
ANTI_WEIGHT = 0.5

C = (TH - 1) / 2.0  # 12.5


def stamp(canvas, r, c, radius, val):
    ir, ic = int(np.floor(r)), int(np.floor(c))
    rr = int(radius) + 2
    for dr in range(-rr, rr + 1):
        for dc in range(-rr, rr + 1):
            yr, yc = ir + dr, ic + dc
            if 0 <= yr < canvas.shape[0] and 0 <= yc < canvas.shape[1]:
                d = np.sqrt((yr - r) ** 2 + (yc - c) ** 2)
                if d < radius:
                    v = val * (1.0 - d / radius)
                    if v > canvas[yr, yc]:
                        canvas[yr, yc] = v


def line(canvas, r0, c0, r1, c1, width=STROKE_W, val=1.0):
    length = max(abs(r1 - r0), abs(c1 - c0))
    n = max(3, int(4 * length) + 1)
    for t in np.linspace(0.0, 1.0, n):
        stamp(canvas, r0 + t * (r1 - r0), c0 + t * (c1 - c0), width, val)


def arc(canvas, cr, cc, ry, rx, a0, a1, width=STROKE_W, val=1.0, slant=0.0):
    length = np.pi * max(ry, rx) * abs(a1 - a0) / 180.0
    n = max(16, int(4 * length) + 1)
    for a in np.linspace(a0, a1, n):
        rad = np.radians(a)
        dr = ry * np.sin(rad)
        dc = rx * np.cos(rad) + slant * dr
        stamp(canvas, cr + dr, cc + dc, width, val)


def draw_pos(digit, variant):
    """Positive strokes for a given digit variant. 26x26 canvas, center (12.5, 12.5)."""
    t = np.zeros((TH, TH), dtype=np.float32)
    w = [STROKE_W, STROKE_W + 0.4, STROKE_W - 0.3, STROKE_W, STROKE_W, STROKE_W + 0.2][variant]
    slant = [0.0, 0.0, 0.0, 0.15, -0.12, 0.08][variant]
    scale = [1.0, 1.05, 0.95, 1.0, 1.0, 0.98][variant]

    if digit == 0:
        ry = 9.5 * scale
        rx = 6.3 * scale
        if variant == 1: ry = 9.8 * scale
        if variant == 2: rx = 5.4 * scale
        arc(t, C, C, ry, rx, 0, 360, width=w, slant=slant)

    elif digit == 1:
        top = (2.0, C + slant * 10)
        bot = (23.0, C - slant * 10)
        line(t, top[0], top[1], bot[0], bot[1], width=w)
        if variant == 1:
            line(t, 2.0, top[1], 5.0, top[1] - 3.5, width=w)
        if variant == 5:
            line(t, 23.0, bot[1] - 3, 23.0, bot[1] + 3, width=w)

    elif digit == 2:
        ry2 = 5.3 * scale
        rx2 = 5.2 * scale
        cr2 = 7.8
        arc(t, cr2, C, ry2, rx2, 180, 390, width=w, slant=slant)
        r_up = cr2 + ry2 * np.sin(np.radians(390))
        c_up = C + rx2 * np.cos(np.radians(390))
        line(t, r_up, c_up, 22.0, 6.5, width=w)
        line(t, 22.0, 6.5, 22.0, 19, width=w)

    elif digit == 3:
        top_cr = 7.0
        bot_cr = 17.5
        if variant == 1:
            top_cr, bot_cr = 6.5, 18.0
        ry3 = 4.8 * scale
        rx3 = 5.0 * scale
        arc(t, top_cr, 12, ry3, rx3, 205, 420, width=w, slant=slant)
        arc(t, bot_cr, 12, ry3 + 0.3, rx3, 320, 530, width=w, slant=slant)
        if variant == 5:
            line(t, 12, 8, 12, 14, width=w)  # middle crossing

    elif digit == 4:
        if variant == 0:
            line(t, 2.0, 10, 14, 10, width=w)
            line(t, 14, 10, 14, 21, width=w)
            line(t, 2.0, 18, 23.0, 18, width=w)
        elif variant == 1:
            line(t, 2.0, 14, 14, 8, width=w)
            line(t, 14, 8, 14, 21, width=w)
            line(t, 2.0, 18, 23.0, 18, width=w)
        elif variant == 2:
            line(t, 2.0, 16, 14, 8, width=w)
            line(t, 14, 8, 14, 19, width=w)
            line(t, 2.0, 17, 23.0, 17, width=w)
        elif variant == 3:
            line(t, 2.0, 18, 14, 10, width=w)
            line(t, 14, 10, 14, 21, width=w)
            line(t, 2.0, 18, 23.0, 18, width=w)
        elif variant == 4:
            line(t, 2.0, 10, 14, 10, width=w)
            line(t, 14, 10, 14, 20, width=w)
            line(t, 2.0, 17, 23.0, 17, width=w)
        else:
            line(t, 2.0, 12, 14, 12, width=w)
            line(t, 14, 12, 14, 21, width=w)
            line(t, 2.0, 18, 23.0, 18, width=w)

    elif digit == 5:
        top_r = [2.5, 2.0, 3.0, 2.5, 3.0, 2.5][variant]
        top_c0 = [6.0, 6.5, 5.5, 6.0, 6.5, 6.0][variant]
        top_c1 = [18.0, 17.5, 18.5, 18.0, 18.5, 18.5][variant]
        mid_r = [11.5, 11.0, 12.0, 11.5, 12.0, 11.5][variant]
        line(t, top_r, top_c0, top_r, top_c1, width=w)
        line(t, top_r, top_c0, mid_r, top_c0 - 0.5, width=w)
        if variant in (0, 2, 3, 5):
            line(t, mid_r, top_c0 - 0.5, mid_r, 12, width=w)
        arc(t, 17.0, 12.5, 5.3 * scale, 5.2 * scale, 265, 490, width=w, slant=slant)

    elif digit == 6:
        top_r = [3.0, 2.5, 3.5, 2.5, 3.5, 3.0][variant]
        top_c = [17.0, 16.5, 17.5, 16.0, 17.5, 17.0][variant]
        knee_r = [9.0, 8.5, 9.5, 9.0, 8.5, 9.0][variant]
        knee_c = [11.0, 10.5, 11.5, 11.0, 11.0, 10.5][variant]
        mid_r = [13.0, 12.5, 13.5, 13.0, 12.5, 13.0][variant]
        mid_c = [7.5, 7.0, 8.0, 7.5, 7.5, 7.0][variant]
        line(t, top_r, top_c, knee_r, knee_c, width=w)
        line(t, knee_r, knee_c, mid_r, mid_c, width=w)
        bot_cr = 16.5
        bot_cc = 12.5
        arc(t, bot_cr, bot_cc, 4.8 * scale, 5.0 * scale, 0, 360, width=w)

    elif digit == 7:
        top_r = [2.5, 2.0, 3.0, 2.5, 3.0, 2.5][variant]
        top_c0 = [6.0, 6.5, 5.5, 6.0, 6.5, 5.5][variant]
        top_c1 = [18.5, 18.0, 19.0, 18.5, 18.0, 19.0][variant]
        line(t, top_r, top_c0, top_r, top_c1, width=w)
        end_col = [8.0, 7.5, 8.5, 8.0, 7.0, 7.5][variant]
        line(t, top_r, top_c1, 23.0, end_col, width=w)
        if variant == 1:
            line(t, 12, 8, 12, 14, width=w)

    elif digit == 8:
        top_cr = 7.5
        bot_cr = 17.5
        top_ry = [4.5, 4.2, 4.8, 4.5, 4.4, 4.7][variant]
        top_rx = [4.5, 4.2, 4.8, 4.5, 4.4, 4.7][variant]
        bot_ry = [5.4, 5.7, 5.2, 5.4, 5.2, 5.5][variant]
        bot_rx = [5.4, 5.7, 5.2, 5.4, 5.2, 5.5][variant]
        top_cc = C + ([0, 0, 0, 0.3, -0.3, 0][variant])
        bot_cc = C + ([0, 0, 0, -0.3, 0.3, 0][variant])
        arc(t, top_cr, top_cc, top_ry, top_rx, 0, 360, width=w)
        arc(t, bot_cr, bot_cc, bot_ry, bot_rx, 0, 360, width=w)

    elif digit == 9:
        top_cr = [7.5, 7.0, 8.0, 7.5, 7.5, 7.0][variant]
        top_cc = C + ([0, 0.3, -0.3, 0.3, -0.3, 0][variant])
        top_ry = [5.0, 5.0, 5.0, 5.0, 5.2, 5.0][variant]
        top_rx = [5.3, 5.3, 5.0, 5.3, 5.0, 5.3][variant]
        arc(t, top_cr, top_cc, top_ry, top_rx, 0, 360, width=w)
        tail_top_c = top_cc + top_rx
        tail_bot_c = [tail_top_c - 0.8, tail_top_c - 1.5, tail_top_c,
                      tail_top_c - 2.5, tail_top_c + 0.5, tail_top_c - 1.0][variant]
        line(t, top_cr, tail_top_c, 23.0, tail_bot_c, width=w)

    return t


def draw_anti(digit, variant):
    """Anti-mask: places where ink would indicate a DIFFERENT digit.
    Drawn from general knowledge of digit shapes (not training images)."""
    anti = np.zeros((TH, TH), dtype=np.float32)
    if digit == 0:
        # 0 has hollow center (no ink), no middle horizontal crossing (not 8)
        line(anti, C, 8, C, 17, width=2.5)
    elif digit == 1:
        # 1 has no top horizontal (not 7) and no bottom horizontal (not 2)
        line(anti, 2.5, 6, 2.5, 19, width=2.5)
        line(anti, 22, 6, 22, 19, width=2.5)
        # No loops either
        arc(anti, 7, 12, 3.5, 3.5, 0, 360, width=2)
        arc(anti, 17, 12, 3.5, 3.5, 0, 360, width=2)
    elif digit == 2:
        # 2 has no top loop closure on the left (not 8 or 9)
        line(anti, 2, 7, 12, 7, width=2.0)  # left side of top region
    elif digit == 3:
        # 3 is open on the left (no left vertical)
        line(anti, 4, 6, 20, 6, width=2.5)
    elif digit == 4:
        # 4 has no top arc (not 9, 0, 6)
        arc(anti, 6, 12, 4, 4, 180, 360, width=2.5)
        # No bottom loop
        arc(anti, 18, 12, 4, 4, 0, 360, width=2.0)
    elif digit == 5:
        # 5 top is open (no closed top like 6 or 0 or 8 or 9)
        arc(anti, 6, 13, 4, 4, 180, 360, width=2.0)
        # right side upper no ink (not 2, 3, 7)
        line(anti, 2.5, 18, 8, 18, width=2.0)
    elif digit == 6:
        # 6 has no closed top loop (not 8 or 9 or 0)
        arc(anti, 6, 12, 3.5, 3.5, 180, 360, width=2.5)
        # 6 has no top-right vertical (not 7, 0)
        line(anti, 2.5, 18, 8, 18, width=2.0)
        # 6 has no bottom straight (not 2)
        line(anti, 22, 7, 22, 18, width=2.0)
    elif digit == 7:
        # 7 has no bottom horizontal (not 2)
        line(anti, 22, 7, 22, 18, width=2.5)
        # 7 has no bottom loop (not 3, 5, 6, 8, 9)
        arc(anti, 17, 12, 4, 4, 0, 360, width=2.5)
        # 7 has no middle crossbar (not 4) - but variant 1 adds one, so skip
    elif digit == 8:
        # 8 has top closed on left (not 3)
        # Not much anti, since 8 has ink almost everywhere in its shape
        pass
    elif digit == 9:
        # 9 has no bottom loop (not 6 or 8)
        arc(anti, 18, 12, 4, 4, 0, 360, width=2.5)
        # 9 has no bottom horizontal (not 2)
        line(anti, 22, 7, 22, 18, width=2.0)
    return anti


def dilate(mask, radius):
    out = np.zeros_like(mask)
    r_int = int(radius)
    for dr in range(-r_int, r_int + 1):
        for dc in range(-r_int, r_int + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            src_r0 = max(0, -dr); src_r1 = mask.shape[0] - max(0, dr)
            src_c0 = max(0, -dc); src_c1 = mask.shape[1] - max(0, dc)
            dst_r0 = max(0, dr);  dst_r1 = mask.shape[0] - max(0, -dr)
            dst_c0 = max(0, dc);  dst_c1 = mask.shape[1] - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] = np.maximum(
                out[dst_r0:dst_r1, dst_c0:dst_c1],
                mask[src_r0:src_r1, src_c0:src_c1],
            )
    return out


def make_all_templates():
    out = []
    for d in range(10):
        for v in range(V):
            pos = draw_pos(d, v)
            anti = draw_anti(d, v)
            dilated = dilate(pos, 3.0)
            shadow = np.clip(dilated - pos, 0.0, 1.0)
            # Also don't let anti overlap with positive ink
            anti = np.clip(anti - pos, 0.0, 1.0)
            T = pos - SHADOW_WEIGHT * shadow - ANTI_WEIGHT * anti
            T = T - float(T.mean())
            n = float(np.sqrt((T * T).sum()))
            if n > 0:
                T = T / n
            out.append(T.astype(np.float32))
    return np.stack(out, axis=0)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    T = make_all_templates()
    conv_W = T[:, None, :, :].astype(np.float32)
    conv_b = np.zeros(10 * V, dtype=np.float32)

    np.savez(
        os.path.join(out_dir, "weights.npz"),
        **{"conv.W": conv_W, "conv.b": conv_b},
    )

    net = {
        "layers": [
            {"name": "n0", "op": "norm", "inputs": ["image"]},
            {
                "name": "conv",
                "op": "conv",
                "inputs": ["n0"],
                "out": 10 * V,
                "kernel": [TH, TH],
                "stride": 1,
                "padding": 0,
            },
            {
                "name": "spool",
                "op": "maxpool",
                "inputs": ["conv"],
                "size": [3, 3],
                "stride": [3, 3],
            },
            {
                "name": "grouped",
                "op": "reshape",
                "inputs": ["spool"],
                "shape": [10, V, 1],
            },
            {
                "name": "pool",
                "op": "maxpool",
                "inputs": ["grouped"],
                "size": [V, 1],
                "stride": [V, 1],
            },
            {
                "name": "scores",
                "op": "reshape",
                "inputs": ["pool"],
                "shape": [10],
            },
        ]
    }
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=2)


if __name__ == "__main__":
    main()
