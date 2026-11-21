"""Builds a template-matching network for MNIST digit classification.

The weights are produced by rendering hand-written stroke paths for each digit
class based on general knowledge of how digits are drawn. Each template is a
28x28 image with positive values along the expected stroke path and small
negative values in surrounding "forbidden" regions where ink should not appear.
Multiple variants per class (slants, sizes, stroke widths) are pooled with max
to give one score per class.
"""
import sys, os, json
from pathlib import Path
import numpy as np

PI = np.pi
H = 28
CX, CY = 13.5, 13.5
HALF = 9.0  # digit half-size in pixels (fits in 20x20 box)


# ---------- drawing primitives ----------
def add_spot(img, cx, cy, radius, value):
    H, W = img.shape
    r = int(radius * 3) + 1
    y0, y1 = max(0, int(cy - r)), min(H, int(cy + r + 1))
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r + 1))
    if y1 <= y0 or x1 <= x0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    w = value * np.exp(-d2 / (2 * radius ** 2))
    if value >= 0:
        img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], w)
    else:
        img[y0:y1, x0:x1] = np.minimum(img[y0:y1, x0:x1], w)


def draw_segment(img, x0, y0, x1, y1, radius=1.5, value=1.0):
    L = max(2, int(np.hypot(x1 - x0, y1 - y0) * 4))
    for t in np.linspace(0, 1, L):
        add_spot(img, x0 + t * (x1 - x0), y0 + t * (y1 - y0), radius, value)


def transform(nx, ny, slant, scale_x=1.0, scale_y=1.0):
    y = CY + ny * HALF * scale_y
    x = CX + (nx + slant * ny) * HALF * scale_x
    return x, y


def draw_line_n(img, nx0, ny0, nx1, ny1, slant, scale_x, scale_y, radius, value):
    x0, y0 = transform(nx0, ny0, slant, scale_x, scale_y)
    x1, y1 = transform(nx1, ny1, slant, scale_x, scale_y)
    draw_segment(img, x0, y0, x1, y1, radius, value)


def draw_arc_n(img, cx_n, cy_n, rx_n, ry_n, a0, a1, slant, scale_x, scale_y, radius, value):
    n = max(12, int(abs(a1 - a0) * max(abs(rx_n), abs(ry_n)) * HALF * 2))
    angles = np.linspace(a0, a1, n)
    pts = []
    for a in angles:
        nx = cx_n + rx_n * np.cos(a)
        ny = cy_n + ry_n * np.sin(a)
        pts.append(transform(nx, ny, slant, scale_x, scale_y))
    for i in range(len(pts) - 1):
        draw_segment(img, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], radius, value)


# ---------- digit stroke paths ----------
# Each entry is a list of ("line", nx0, ny0, nx1, ny1) or ("arc", cx, cy, rx, ry, a0, a1).
# Coordinates are in normalized [-1, 1] space (y positive = down).
# These shapes come from my general knowledge of how each digit is written,
# not from training images.
def digit_strokes(digit):
    if digit == 0:
        return [("arc", 0.0, 0.0, 0.55, 0.9, 0.0, 2 * PI)]
    if digit == 1:
        return [
            ("line", 0.0, -0.9, 0.0, 0.9),
            ("line", -0.35, -0.55, 0.0, -0.9),
            ("line", -0.35, 0.9, 0.35, 0.9),  # bottom serif
        ]
    if digit == 2:
        return [
            ("arc", 0.0, -0.4, 0.5, 0.45, PI, 2 * PI),
            ("line", 0.5, -0.4, 0.4, 0.0),
            ("line", 0.4, 0.0, -0.55, 0.85),
            ("line", -0.55, 0.85, 0.65, 0.85),
        ]
    if digit == 3:
        return [
            ("arc", 0.0, -0.5, 0.45, 0.4, -PI / 2, PI / 2),
            ("arc", 0.0, 0.4, 0.5, 0.5, -PI / 2, PI / 2),
        ]
    if digit == 4:
        return [
            ("line", -0.4, -0.9, -0.55, 0.1),
            ("line", -0.55, 0.1, 0.55, 0.1),
            ("line", 0.3, -0.9, 0.3, 0.9),
        ]
    if digit == 5:
        # top horizontal + left top vertical + right-opening curve for the lower bulge
        return [
            ("line", 0.4, -0.9, -0.5, -0.9),
            ("line", -0.5, -0.9, -0.45, 0.05),
            # bottom bulge: C-shape opening LEFT from top-mid, right, bottom-mid
            ("arc", -0.1, 0.45, 0.55, 0.5, -PI / 2, PI / 2),
            # connect vertical to arc top
            ("line", -0.45, 0.05, -0.1, -0.05),
        ]
    if digit == 6:
        # top stroke: from upper-right curving down-left toward the loop
        # loop at bottom
        return [
            ("line", 0.25, -0.9, -0.3, 0.0),
            ("line", -0.3, 0.0, -0.45, 0.4),
            ("arc", 0.0, 0.4, 0.5, 0.5, 0.0, 2 * PI),
        ]
    if digit == 7:
        return [
            ("line", -0.55, -0.9, 0.7, -0.9),
            ("line", 0.7, -0.9, -0.35, 0.9),
        ]
    if digit == 8:
        return [
            ("arc", 0.0, -0.42, 0.4, 0.4, 0.0, 2 * PI),
            ("arc", 0.0, 0.42, 0.45, 0.45, 0.0, 2 * PI),
        ]
    if digit == 9:
        return [
            ("arc", 0.0, -0.35, 0.45, 0.45, 0.0, 2 * PI),
            ("line", 0.45, -0.35, 0.3, 0.9),
        ]
    raise ValueError(digit)


def render_strokes(digit, slant, scale_x, scale_y, radius, value=1.0):
    img = np.zeros((H, H), dtype=np.float32)
    for s in digit_strokes(digit):
        if s[0] == "line":
            _, nx0, ny0, nx1, ny1 = s
            draw_line_n(img, nx0, ny0, nx1, ny1, slant, scale_x, scale_y, radius, value)
        elif s[0] == "arc":
            _, cx_n, cy_n, rx_n, ry_n, a0, a1 = s
            draw_arc_n(img, cx_n, cy_n, rx_n, ry_n, a0, a1, slant, scale_x, scale_y, radius, value)
    return img


def anti_features(digit):
    """Extra negative regions per digit: where OTHER similar digits would have ink.
    Each entry is (kind, x, y, radius, strength_multiplier).
    """
    if digit == 0:
        # strong negative in center column where 1, 7 would have ink
        return [("disk", 0.0, 0.0, 0.22, 1.2),
                ("disk", 0.0, -0.4, 0.18, 0.8),
                ("disk", 0.0, 0.4, 0.18, 0.8)]
    if digit == 1:
        # sides already covered by box, but add explicit penalty for ink elsewhere
        return [("disk", 0.4, -0.6, 0.3, 0.9),
                ("disk", -0.5, 0.2, 0.3, 0.9),
                ("disk", 0.5, 0.5, 0.25, 0.8),
                ("disk", -0.5, -0.4, 0.25, 0.8)]
    if digit == 2:
        # upper-left side (where 3 has nothing but 8/0 have ink)
        return [("disk", -0.5, -0.1, 0.22, 0.8),
                ("disk", -0.5, 0.4, 0.22, 0.6)]
    if digit == 3:
        # left side where 8, 0 would have the left arc/ring
        return [("disk", -0.55, -0.3, 0.25, 0.9),
                ("disk", -0.55, 0.3, 0.25, 0.9),
                ("disk", -0.55, 0.0, 0.2, 0.8)]
    if digit == 4:
        # upper middle where 9 has loop top closure
        return [("disk", 0.0, -0.5, 0.22, 0.8),
                ("disk", -0.2, 0.6, 0.2, 0.6)]
    if digit == 5:
        # lower-left (where 6 has loop left side) and upper right
        return [("disk", -0.5, 0.55, 0.22, 0.8),
                ("disk", 0.5, -0.5, 0.22, 0.7)]
    if digit == 6:
        # upper-right area where 0/5/9 top ink is; top center (anti-5 top horiz)
        return [("disk", 0.45, -0.5, 0.25, 1.1),
                ("disk", 0.0, -0.85, 0.3, 0.8),
                ("disk", 0.4, -0.1, 0.2, 0.7)]
    if digit == 7:
        # anti-features: where 1 (center vertical) has ink
        return [("disk", -0.05, 0.3, 0.25, 1.2),
                ("disk", -0.1, 0.6, 0.25, 1.2),
                ("disk", 0.0, 0.0, 0.2, 0.7),
                ("disk", 0.55, 0.4, 0.22, 0.8),
                ("disk", -0.55, 0.4, 0.22, 0.7)]
    if digit == 8:
        # keep minimal extra
        return [("disk", 0.55, -0.0, 0.18, 0.5),
                ("disk", -0.55, 0.0, 0.18, 0.5)]
    if digit == 9:
        # lower-left area where 6/0/4 have ink
        return [("disk", -0.4, 0.5, 0.28, 1.1),
                ("disk", -0.5, -0.0, 0.2, 0.7),
                ("disk", -0.3, 0.8, 0.2, 0.6)]
    return []


def apply_anti_features(img, digit, slant, scale_x, scale_y, strength):
    """Subtract anti-feature disks from the template."""
    for feat in anti_features(digit):
        kind, nx, ny, r, w = feat
        if kind == "disk":
            cx, cy = transform(nx, ny, slant, scale_x, scale_y)
            radius_pix = r * HALF * max(scale_x, scale_y)
            # subtract a Gaussian blob with strength*w
            yy, xx = np.mgrid[0:H, 0:H]
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            disk = np.exp(-d2 / (2 * radius_pix ** 2)) * (strength * w)
            img -= disk


def make_template(digit, slant=0.0, scale_x=1.0, scale_y=1.0, stroke_radius=1.6,
                  pos_weight=1.0, neg_weight=0.55, tol_radius=2.8, anti_strength=0.6):
    """Make a 28x28 template with positive stroke values and negative background."""
    # Tolerance mask: dilated stroke region (where ink IS allowed, no penalty)
    tol = render_strokes(digit, slant, scale_x, scale_y, radius=tol_radius, value=1.0)
    # Forbidden region: 20x20 box minus tolerance zone
    template = np.zeros((H, H), dtype=np.float32)
    template[4:24, 4:24] = -neg_weight
    template[tol > 0.25] = 0.0
    # Draw positive stroke on top
    stroke = render_strokes(digit, slant, scale_x, scale_y, radius=stroke_radius, value=pos_weight)
    template = template + stroke
    # Apply class-specific anti-features (extra negative in discriminative areas)
    apply_anti_features(template, digit, slant, scale_x, scale_y, anti_strength)
    # Keep positive stroke intact; anti-features might over-subtract on stroke
    template = np.maximum(template, -0.9)
    template = np.minimum(template, 1.2)
    # Restore stroke if anti-feature reduced it
    template = np.where(stroke > 0.3, np.maximum(template, stroke), template)
    # L2 normalize so templates are comparable across classes
    l2 = float(np.sqrt((template.astype(np.float64) ** 2).sum()))
    if l2 > 1e-8:
        template = template / l2
    return template.astype(np.float32)


# ---------- build weights ----------
def build_templates():
    """Returns (K, 1, 28, 28) kernel weights, (K,) biases."""
    # Variant parameters: (slant, scale_x, scale_y, stroke_radius)
    variants = [
        (0.00, 1.00, 1.00, 1.6),    # standard upright, normal thickness
        (-0.25, 1.00, 1.00, 1.6),   # strong left italic (top-right to bottom-left)
        (-0.12, 1.00, 1.00, 1.6),   # mild left italic
        (0.15, 1.00, 1.00, 1.6),    # mild right italic
        (0.00, 0.85, 1.00, 1.6),    # narrower
        (0.00, 1.10, 1.00, 1.6),    # wider
        (-0.15, 1.00, 1.00, 2.1),   # left italic thick
        (0.00, 1.00, 1.00, 2.1),    # upright thick
        (-0.05, 0.95, 0.90, 1.6),   # slightly smaller, slight slant
        (0.00, 1.00, 1.05, 1.6),    # taller
    ]
    K = 10 * len(variants)
    W = np.zeros((K, 1, H, H), dtype=np.float32)
    B = np.zeros((K,), dtype=np.float32)
    for c in range(10):
        for v, (slant, sx, sy, r) in enumerate(variants):
            tmpl = make_template(c, slant, sx, sy, r)
            W[c * len(variants) + v, 0] = tmpl
    return W, B, len(variants)


# ---------- network definition ----------
def build_network(num_variants):
    return {
        "layers": [
            {"name": "n1", "op": "norm", "inputs": ["image"]},
            {"name": "c1", "op": "conv", "inputs": ["n1"], "out": 10 * num_variants,
             "kernel": [28, 28], "stride": 1, "padding": 2},
            {"name": "mp1", "op": "maxpool", "inputs": ["c1"], "size": [5, 5], "stride": [5, 5]},
            {"name": "r1", "op": "reshape", "inputs": ["mp1"], "shape": [10, 1, num_variants]},
            {"name": "mp2", "op": "maxpool", "inputs": ["r1"], "size": [1, num_variants],
             "stride": [1, num_variants]},
            {"name": "out", "op": "reshape", "inputs": ["mp2"], "shape": [10]},
        ]
    }


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    W, B, num_variants = build_templates()
    net = build_network(num_variants)

    with open(out_dir / "network.json", "w") as f:
        json.dump(net, f, indent=2)
    np.savez(out_dir / "weights.npz", **{"c1.W": W, "c1.b": B})
    print(f"wrote network ({num_variants} variants, {10 * num_variants} templates)")


if __name__ == "__main__":
    main()
