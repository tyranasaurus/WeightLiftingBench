"""Writes network.json and weights.npz for a hand-designed GTSRB template matcher (no data read)."""
import types, sys
RENDER_SRC = '"""Hand-written vector drawings of the 43 German sign designs, from general knowledge of the StVO signs.\nCoordinates: u to the right, v downward, sign bounding box half-width = 1 (circle radius 1)."""\nimport numpy as np\n\nHR = 1.25          # half extent of the high-res canvas\nNHR = 500          # canvas resolution\n_ax = (np.arange(NHR) + 0.5) / NHR * 2 * HR - HR\nU, V = np.meshgrid(_ax, _ax)          # U[i, j] = x of column j, V[i, j] = y of row i\n\n# colours as a camera typically records them (0..1)\nWHITE = (0.82, 0.82, 0.82)\nRED = (0.62, 0.10, 0.12)\nBLUE = (0.05, 0.22, 0.56)\nYELLOW = (0.90, 0.66, 0.06)\nBLACK = (0.07, 0.07, 0.07)\nGREY = (0.45, 0.45, 0.45)\nDGREY = (0.25, 0.25, 0.25)\nTL_RED = (0.75, 0.10, 0.10)\nTL_YEL = (0.88, 0.68, 0.08)\nTL_GRN = (0.10, 0.58, 0.40)\n\n\n# ---------------------------------------------------------------- primitives (boolean masks on the canvas)\ndef disc(cx, cy, r):\n    return (U - cx) ** 2 + (V - cy) ** 2 <= r * r\n\n\ndef ellipse(cx, cy, rx, ry, ang=0.0):\n    c, s = np.cos(ang), np.sin(ang)\n    x, y = U - cx, V - cy\n    xr, yr = c * x + s * y, -s * x + c * y\n    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0\n\n\ndef rect(x0, y0, x1, y1):\n    return (U >= x0) & (U <= x1) & (V >= y0) & (V <= y1)\n\n\ndef polygon(pts):\n    """even-odd point in polygon"""\n    pts = np.asarray(pts, float)\n    inside = np.zeros(U.shape, bool)\n    n = len(pts)\n    for i in range(n):\n        x0, y0 = pts[i]\n        x1, y1 = pts[(i + 1) % n]\n        if y0 == y1:\n            continue\n        cond = (V >= min(y0, y1)) & (V < max(y0, y1))\n        xc = x0 + (V - y0) * (x1 - x0) / (y1 - y0)\n        inside ^= cond & (U < xc)\n    return inside\n\n\ndef seg_dist(x0, y0, x1, y1):\n    dx, dy = x1 - x0, y1 - y0\n    L2 = dx * dx + dy * dy\n    if L2 == 0:\n        return np.hypot(U - x0, V - y0)\n    t = np.clip(((U - x0) * dx + (V - y0) * dy) / L2, 0, 1)\n    return np.hypot(U - x0 - t * dx, V - y0 - t * dy)\n\n\ndef stroke(pts, w):\n    """thick polyline with round caps and joins"""\n    pts = np.asarray(pts, float)\n    d = np.full(U.shape, np.inf)\n    if len(pts) == 1:\n        return np.hypot(U - pts[0, 0], V - pts[0, 1]) <= w / 2\n    for i in range(len(pts) - 1):\n        d = np.minimum(d, seg_dist(*pts[i], *pts[i + 1]))\n    return d <= w / 2\n\n\ndef poly_dist(pts):\n    pts = np.asarray(pts, float)\n    d = np.full(U.shape, np.inf)\n    n = len(pts)\n    for i in range(n):\n        d = np.minimum(d, seg_dist(*pts[i], *pts[(i + 1) % n]))\n    return d\n\n\ndef rounded_polygon(pts, r):\n    """convex polygon with corners rounded by radius r (outline stays where pts says)"""\n    pts = np.asarray(pts, float)\n    c = pts.mean(0)\n    # shrink the polygon by r (move every edge inward by r), then grow it back with a disc\n    n = len(pts)\n    lines = []\n    for i in range(n):\n        p, q = pts[i], pts[(i + 1) % n]\n        e = q - p\n        nrm = np.array([e[1], -e[0]]) / np.hypot(*e)\n        if np.dot(nrm, c - p) < 0:\n            nrm = -nrm\n        lines.append((p + nrm * r, e))\n    inner = []\n    for i in range(n):\n        (p1, e1), (p2, e2) = lines[i - 1], lines[i]\n        A = np.array([e1, -e2]).T\n        t = np.linalg.solve(A, p2 - p1)\n        inner.append(p1 + t[0] * e1)\n    inner = np.array(inner)\n    return polygon(inner) | (poly_dist(inner) <= r)\n\n\ndef arc(cx, cy, rx, ry, a0, a1, n=40):\n    """points on an elliptical arc, angles in degrees, v downward (90 = bottom)"""\n    a = np.radians(np.linspace(a0, a1, n))\n    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)\n\n\ndef chain(*parts):\n    return np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in parts], 0)\n\n\ndef tri_up(s=1.0):\n    h = np.sqrt(3) * s\n    return [(0, -h / 2), (s, h / 2), (-s, h / 2)]\n\n\ndef tri_down(s=1.0):\n    h = np.sqrt(3) * s\n    return [(-s, -h / 2), (s, -h / 2), (0, h / 2)]\n\n\ndef octagon(apothem=1.0):\n    R = apothem / np.cos(np.pi / 8)\n    a = np.radians(22.5 + 45 * np.arange(8))\n    return list(zip(R * np.cos(a), R * np.sin(a)))\n\n\ndef offset_polygon(pts, d):\n    """move every edge of a convex polygon inward by d"""\n    pts = np.asarray(pts, float)\n    c = pts.mean(0)\n    n = len(pts)\n    lines = []\n    for i in range(n):\n        p, q = pts[i], pts[(i + 1) % n]\n        e = q - p\n        nrm = np.array([e[1], -e[0]]) / np.hypot(*e)\n        if np.dot(nrm, c - p) < 0:\n            nrm = -nrm\n        lines.append((p + nrm * d, e))\n    out = []\n    for i in range(n):\n        (p1, e1), (p2, e2) = lines[i - 1], lines[i]\n        t = np.linalg.solve(np.array([e1, -e2]).T, p2 - p1)\n        out.append(p1 + t[0] * e1)\n    return out\n\n\n# ---------------------------------------------------------------- glyphs (centre-line paths, then stroked)\ndef glyph_paths(ch, w, h):\n    """centre lines of a DIN-1451-like character in a box [0, w] x [0, h] (centre-line box)"""\n    P = []\n    if ch == \'0\':\n        a = w / 2\n        P.append(chain(arc(a, a, a, a, 180, 360, 30), [(w, h - a)], arc(a, h - a, a, a, 0, 180, 30), [(0, a)]))\n    elif ch == \'1\':\n        P.append([(w * 0.75, 0), (w * 0.75, h)])\n        P.append([(w * 0.75, 0), (0.0, h * 0.30)])\n    elif ch == \'2\':\n        a = w / 2\n        top = arc(a, a * 1.0, a, a, 195, 360 + 25, 30)\n        P.append(chain(top, [(0, h), (w, h)]))\n    elif ch == \'3\':\n        yj = h * 0.46\n        top = arc(w * 0.48, yj / 2, w * 0.46, yj / 2, 200, 450, 40)\n        bot = arc(w / 2, (yj + h) / 2, w / 2, (h - yj) / 2, -90, 160, 40)\n        P.append(chain(top, [(w * 0.42, yj)], bot))\n    elif ch == \'5\':\n        ry = h * 0.30\n        cy = h - ry\n        bowl = arc(w / 2, cy, w / 2, ry, -150, 150, 40)\n        P.append(chain([(w * 0.95, 0), (w * 0.12, 0), (w * 0.06, bowl[0, 1] + 0.02 * h)], bowl))\n    elif ch == \'6\':\n        ry = h * 0.31\n        cy = h - ry\n        P.append(arc(w / 2, cy, w / 2, ry, 0, 360, 50))\n        up = arc(w * 1.05, cy, w * 1.05, cy, 180, 258, 30)\n        P.append(up)\n    elif ch == \'7\':\n        P.append([(0, 0), (w, 0), (w * 0.62, h * 0.38), (w * 0.35, h)])\n    elif ch == \'8\':\n        yj = h * 0.45\n        P.append(arc(w / 2, yj / 2, w * 0.43, yj / 2, 0, 360, 40))\n        P.append(arc(w / 2, (yj + h) / 2, w / 2, (h - yj) / 2, 0, 360, 40))\n    elif ch == \'S\':\n        yj = h * 0.48\n        top = arc(w / 2, yj / 2, w / 2, yj / 2, -10, -270, 40)     # from upper right, over the top, down the left\n        bot = arc(w / 2, (yj + h) / 2, w / 2, (h - yj) / 2, -90, 170, 40)\n        P.append(chain(top, bot))\n    elif ch == \'T\':\n        P.append([(0, 0), (w, 0)])\n        P.append([(w / 2, 0), (w / 2, h)])\n    elif ch == \'O\':\n        a = w / 2\n        P.append(chain(arc(a, a, a, a, 180, 360, 30), [(w, h - a)], arc(a, h - a, a, a, 0, 180, 30), [(0, a)]))\n    elif ch == \'P\':\n        P.append([(0, h), (0, 0)])\n        bh = h * 0.55\n        P.append(chain([(0, 0), (w * 0.5, 0)], arc(w * 0.5, bh / 2, w * 0.5, bh / 2, -90, 90, 30), [(0, bh)]))\n    return P\n\n\nGLYPH_W = {\'0\': 0.56, \'1\': 0.34, \'2\': 0.54, \'3\': 0.54, \'5\': 0.54, \'6\': 0.56, \'7\': 0.52, \'8\': 0.56,\n           \'S\': 0.50, \'T\': 0.48, \'O\': 0.54, \'P\': 0.46}\n\n\ndef text_mask(txt, cx, cy, h, t, wscale=1.0, gap=0.10):\n    """black text centred at (cx, cy) with glyph height h and stroke t"""\n    widths = [GLYPH_W[c] * h * wscale for c in txt]\n    gaps = gap * h\n    total = sum(widths) + gaps * (len(txt) - 1)\n    x = cx - total / 2\n    m = np.zeros(U.shape, bool)\n    for c, gw in zip(txt, widths):\n        cw, chh = gw - t, h - t\n        for p in glyph_paths(c, cw, chh):\n            p = np.asarray(p, float) + np.array([x + t / 2, cy - h / 2 + t / 2])\n            m |= stroke(p, t)\n        x += gw + gaps\n    return m\n\n\n# ---------------------------------------------------------------- pictogram pieces\ndef car_rear(cx, cy, w, h):\n    """car seen from behind: cabin on top, wide body, wheels; (cx, cy) = centre of its box"""\n    x0, y0 = cx - w / 2, cy - h / 2\n    m = rounded_polygon([(x0 + 0.18 * w, y0), (x0 + 0.82 * w, y0), (x0 + 0.95 * w, y0 + 0.45 * h),\n                         (x0 + 0.05 * w, y0 + 0.45 * h)], 0.03)\n    m |= rounded_polygon([(x0, y0 + 0.40 * h), (x0 + w, y0 + 0.40 * h), (x0 + w, y0 + 0.82 * h),\n                          (x0, y0 + 0.82 * h)], 0.03)\n    m |= rect(x0 + 0.04 * w, y0 + 0.8 * h, x0 + 0.24 * w, y0 + h)\n    m |= rect(x0 + 0.76 * w, y0 + 0.8 * h, x0 + 0.96 * w, y0 + h)\n    return m\n\n\ndef truck_rear(cx, cy, w, h):\n    x0, y0 = cx - w / 2, cy - h / 2\n    m = rect(x0, y0, x0 + w, y0 + 0.86 * h)\n    m |= rect(x0 + 0.05 * w, y0 + 0.84 * h, x0 + 0.27 * w, y0 + h)\n    m |= rect(x0 + 0.73 * w, y0 + 0.84 * h, x0 + 0.95 * w, y0 + h)\n    return m\n\n\ndef truck_side(cx, cy, w, h, face_left=True):\n    """lorry in side view, cab on the left"""\n    x0, y0 = cx - w / 2, cy - h / 2\n    cab = polygon([(x0 + 0.02 * w, y0 + 0.40 * h), (x0 + 0.12 * w, y0 + 0.30 * h), (x0 + 0.27 * w, y0 + 0.30 * h),\n                   (x0 + 0.27 * w, y0 + 0.82 * h), (x0, y0 + 0.82 * h), (x0, y0 + 0.45 * h)])\n    box = rect(x0 + 0.30 * w, y0, x0 + w, y0 + 0.78 * h)\n    chassis = rect(x0, y0 + 0.72 * h, x0 + w, y0 + 0.84 * h)\n    wheels = disc(x0 + 0.15 * w, y0 + 0.86 * h, 0.14 * h) | disc(x0 + 0.62 * w, y0 + 0.86 * h, 0.14 * h) | \\\n        disc(x0 + 0.84 * w, y0 + 0.86 * h, 0.14 * h)\n    m = cab | box | chassis | wheels\n    if not face_left:\n        m = m[:, ::-1]\n    return m\n\n\ndef person(hx, hy, s, pose=\'walk\'):\n    """stick figure with head centre (hx, hy) and overall height ~ s"""\n    t = 0.13 * s\n    m = disc(hx, hy, 0.11 * s)\n    neck = (hx, hy + 0.14 * s)\n    hip = (hx - 0.02 * s, hy + 0.52 * s)\n    m |= stroke([neck, hip], t * 1.25)\n    m |= stroke([hip, (hx - 0.24 * s, hy + 0.95 * s)], t)\n    m |= stroke([hip, (hx + 0.06 * s, hy + 0.70 * s), (hx + 0.20 * s, hy + 0.95 * s)], t)\n    m |= stroke([(hx, hy + 0.22 * s), (hx - 0.20 * s, hy + 0.48 * s)], t * 0.85)\n    m |= stroke([(hx, hy + 0.22 * s), (hx + 0.20 * s, hy + 0.45 * s)], t * 0.85)\n    return m\n\n\n# ---------------------------------------------------------------- sign bases\nimport contextlib\n_U0, _V0 = U, V\n\n\n@contextlib.contextmanager\ndef scaled(s, cx=0.0, cy=0.0):\n    """draw the enclosed shapes scaled by s about (cx, cy)"""\n    global U, V\n    U, V = (_U0 - cx) / s + cx, (_V0 - cy) / s + cy\n    try:\n        yield\n    finally:\n        U, V = _U0, _V0\n\n\nclass Canvas:\n    """sign drawing: colour, sign coverage, and the colour the background is assumed to have"""\n    def __init__(self, bg_l):\n        self.rgb = np.zeros(_U0.shape + (3,), np.float32)\n        self.cov = np.zeros(_U0.shape, np.float32)\n        self.picto = np.zeros(_U0.shape + (3,), np.float32)   # colour change made by the pictogram layer\n        self.bg_l = bg_l\n        self.layer = \'frame\'\n\n    def paint(self, mask, col, base=False):\n        col = np.asarray(col, np.float32)\n        if self.layer == \'picto\':\n            self.picto[mask] += col - self.rgb[mask]\n        self.rgb[mask] = col\n        if base:\n            self.cov[mask] = 1.0\n\n\ndef lum(c):\n    return float(np.mean(c))\n\n\ndef red_ring_sign(ring_in=0.72):\n    c = Canvas(lum(RED))\n    c.paint(disc(0, 0, 1.0), WHITE, base=True)\n    c.paint(disc(0, 0, 0.965) & ~disc(0, 0, ring_in), RED)\n    c.layer = \'picto\'\n    return c\n\n\ndef white_end_sign():\n    c = Canvas(lum(WHITE))\n    c.paint(disc(0, 0, 1.0), DGREY, base=True)\n    c.paint(disc(0, 0, 0.955), WHITE)\n    c.layer = \'picto\'\n    return c\n\n\ndef end_band(c, width=0.30, col=DGREY):\n    d = np.abs(U + V) / np.sqrt(2)        # distance to the line u = -v ("/" from top right to bottom left)\n    c.paint(disc(0, 0, 0.955) & (d <= width / 2), col)\n\n\ndef warning_sign(border=0.22):\n    c = Canvas(lum(RED))\n    T = tri_up()\n    c.paint(rounded_polygon(T, 0.10), WHITE, base=True)\n    c.paint(rounded_polygon(offset_polygon(T, 0.035), 0.08), RED)\n    c.paint(rounded_polygon(offset_polygon(T, 0.035 + border), 0.04), WHITE)\n    c.layer = \'picto\'\n    return c\n\n\ndef blue_sign():\n    c = Canvas(lum(BLUE))\n    c.paint(disc(0, 0, 1.0), WHITE, base=True)\n    c.paint(disc(0, 0, 0.955), BLUE)\n    c.layer = \'picto\'\n    return c\n\n\ndef arrow_head(tip, direction, length, width):\n    d = np.asarray(direction, float)\n    d = d / np.hypot(*d)\n    n = np.array([-d[1], d[0]])\n    tip = np.asarray(tip, float)\n    base = tip - d * length\n    return polygon([tip, base + n * width / 2, base - n * width / 2])\n\n\ndef flat_car(cx, cy, w, h):\n    """car seen from behind, wide and low: roof, body, wheels"""\n    x0, y0 = cx - w / 2, cy - h / 2\n    m = polygon([(x0 + 0.22 * w, y0), (x0 + 0.78 * w, y0), (x0 + 0.90 * w, y0 + 0.40 * h), (x0 + 0.10 * w, y0 + 0.40 * h)])\n    m |= rect(x0, y0 + 0.36 * h, x0 + w, y0 + 0.80 * h)\n    m |= rect(x0 + 0.04 * w, y0 + 0.78 * h, x0 + 0.22 * w, y0 + h)\n    m |= rect(x0 + 0.78 * w, y0 + 0.78 * h, x0 + 0.96 * w, y0 + h)\n    return m\n\n\n# ---------------------------------------------------------------- the classes\nSPEED = {0: \'20\', 1: \'30\', 2: \'50\', 3: \'60\', 4: \'70\', 5: \'80\', 7: \'100\', 8: \'120\'}\nDIGIT_VARIANTS = [(0.86, 0.15), (0.96, 0.17)]      # (digit height, stroke as a share of the height)\n\n\ndef digits(c, txt, var, col=BLACK):\n    dh, ts = DIGIT_VARIANTS[var]\n    if len(txt) == 2:\n        c.paint(text_mask(txt, 0.0, 0.0, dh, dh * ts, 1.0, 0.08), col)\n    else:\n        c.paint(text_mask(txt, 0.0, 0.0, dh, dh * ts, 0.72, 0.06), col)\n\n\ndef draw(cls, var=0):\n    """returns a Canvas for class cls; var 0/1 = two sizes of the inner symbol"""\n    ps = [1.0, 1.12][var]                  # pictogram scale for the non-digit classes\n    if cls in SPEED:\n        c = red_ring_sign()\n        digits(c, SPEED[cls], var)\n        return c\n    if cls == 6:\n        c = white_end_sign()\n        digits(c, \'80\', var, DGREY)\n        end_band(c, 0.30)\n        return c\n    if cls == 9:\n        c = red_ring_sign()\n        with scaled(ps):\n            c.paint(flat_car(-0.27, 0.0, 0.46, 0.26), RED)\n            c.paint(flat_car(0.27, 0.0, 0.46, 0.26), BLACK)\n        return c\n    if cls == 10:\n        c = red_ring_sign()\n        with scaled(ps):\n            c.paint(rect(-0.56, -0.26, -0.10, 0.16), RED)\n            c.paint(rect(-0.52, 0.14, -0.40, 0.22) | rect(-0.26, 0.14, -0.14, 0.22), RED)\n            c.paint(flat_car(0.30, 0.08, 0.40, 0.24), BLACK)\n        return c\n    if cls == 15:\n        c = red_ring_sign()\n        return c\n    if cls == 16:\n        c = red_ring_sign()\n        with scaled(ps):\n            x0, y0, w, h = -0.60, -0.30, 1.20, 0.60\n            cab = polygon([(x0, y0 + 0.42 * h), (x0 + 0.08 * w, y0 + 0.28 * h), (x0 + 0.26 * w, y0 + 0.28 * h),\n                           (x0 + 0.26 * w, y0 + 0.84 * h), (x0, y0 + 0.84 * h)])\n            box = rect(x0 + 0.29 * w, y0, x0 + w, y0 + 0.80 * h)\n            chassis = rect(x0, y0 + 0.70 * h, x0 + w, y0 + 0.84 * h)\n            wh = disc(x0 + 0.14 * w, y0 + 0.86 * h, 0.13 * h) | disc(x0 + 0.64 * w, y0 + 0.86 * h, 0.13 * h) | \\\n                disc(x0 + 0.84 * w, y0 + 0.86 * h, 0.13 * h)\n            c.paint(cab | box | chassis | wh, BLACK)\n        return c\n    if cls == 17:\n        c = Canvas(lum(RED))\n        c.paint(disc(0, 0, 1.0), WHITE, base=True)\n        c.paint(disc(0, 0, 0.965), RED)\n        c.layer = \'picto\'\n        with scaled([1.0, 0.92][var]):\n            c.paint(rect(-0.78, -0.17, 0.78, 0.17), WHITE)\n        return c\n    if cls == 32:\n        c = white_end_sign()\n        end_band(c, [0.40, 0.48][var], (0.32, 0.32, 0.32))\n        return c\n    if cls == 41:\n        c = white_end_sign()\n        with scaled(ps):\n            c.paint(flat_car(-0.27, 0.0, 0.46, 0.26), DGREY)\n            c.paint(flat_car(0.27, 0.0, 0.46, 0.26), DGREY)\n        end_band(c, 0.30)\n        return c\n    if cls == 42:\n        c = white_end_sign()\n        with scaled(ps):\n            c.paint(rect(-0.56, -0.26, -0.10, 0.16), DGREY)\n            c.paint(flat_car(0.30, 0.08, 0.40, 0.24), DGREY)\n        end_band(c, 0.30)\n        return c\n    if cls == 12:\n        c = Canvas(lum(WHITE))\n        L1 = np.abs(U) + np.abs(V)\n        c.paint(L1 <= 1.0, BLACK, base=True)\n        c.paint(L1 <= 0.95, WHITE)\n        c.layer = \'picto\'\n        c.paint(L1 <= [0.60, 0.66][var], YELLOW)\n        return c\n    if cls == 13:\n        c = Canvas(lum(RED))\n        T = tri_down()\n        c.paint(rounded_polygon(T, 0.10), WHITE, base=True)\n        c.paint(rounded_polygon(offset_polygon(T, 0.035), 0.08), RED)\n        c.paint(rounded_polygon(offset_polygon(T, 0.035 + [0.22, 0.27][var]), 0.04), WHITE)\n        return c\n    if cls == 14:\n        c = Canvas(lum(RED))\n        c.paint(polygon(octagon(1.0)), WHITE, base=True)\n        c.paint(polygon(octagon(0.94)), RED)\n        c.layer = \'picto\'\n        with scaled([1.0, 0.92][var]):\n            c.paint(text_mask(\'STOP\', 0.0, 0.0, 0.62, 0.14, 1.20, 0.06), WHITE)\n        return c\n    if 33 <= cls <= 40:\n        c = blue_sign()\n        W = WHITE\n        with scaled(ps):\n            if cls == 35:\n                c.paint(stroke([(0, 0.70), (0, -0.20)], 0.19), W)\n                c.paint(arrow_head((0, -0.68), (0, -1), 0.46, 0.62), W)\n            elif cls in (33, 34):\n                s = 1 if cls == 33 else -1\n                a = (180, 270) if s > 0 else (0, -90)\n                p = chain([(-0.15 * s, 0.70)], arc(0.15 * s, 0.05, 0.30, 0.30, a[0], a[1], 20), [(0.26 * s, -0.25)])\n                c.paint(stroke(p, 0.19), W)\n                c.paint(arrow_head((0.66 * s, -0.25), (s, 0), 0.42, 0.60), W)\n            elif cls in (36, 37):\n                s = 1 if cls == 36 else -1\n                c.paint(stroke([(-0.12 * s, 0.70), (-0.12 * s, -0.25)], 0.17), W)\n                c.paint(arrow_head((-0.12 * s, -0.68), (0, -1), 0.40, 0.50), W)\n                a = (180, 270) if s > 0 else (0, -90)\n                br = chain(arc(0.13 * s, 0.40, 0.25, 0.25, a[0], a[1], 16), [(0.28 * s, 0.15)])\n                c.paint(stroke(br, 0.16), W)\n                c.paint(arrow_head((0.66 * s, 0.15), (s, 0), 0.38, 0.48), W)\n            elif cls in (38, 39):\n                s = 1 if cls == 38 else -1\n                c.paint(stroke([(-0.46 * s, -0.46), (0.12 * s, 0.12)], 0.23), W)\n                c.paint(arrow_head((0.52 * s, 0.52), (s, 1), 0.50, 0.68), W)\n            elif cls == 40:\n                for k in range(3):\n                    a0 = 90 + 120 * k            # arrows run counter-clockwise on screen (angle decreasing)\n                    c.paint(stroke(arc(0, 0, 0.52, 0.52, a0 + 30, a0 - 40, 16), 0.16), W)\n                    at = np.radians(a0 - 72)\n                    tip = (0.52 * np.cos(at), 0.52 * np.sin(at))\n                    c.paint(arrow_head(tip, (np.sin(at), -np.cos(at)), 0.30, 0.40), W)\n        return c\n    # warning triangles: pictograms sit around the middle of the white field\n    c = warning_sign()\n    B = BLACK\n    with scaled(ps, 0.0, 0.20):\n        if cls == 11:\n            c.paint(polygon([(-0.04, -0.24), (0.04, -0.24), (0.13, 0.62), (-0.13, 0.62)]), B)\n            c.paint(rect(-0.36, 0.13, 0.36, 0.25), B)\n        elif cls == 18:\n            c.paint(polygon([(-0.085, -0.28), (0.085, -0.28), (0.055, 0.30), (-0.055, 0.30)]), B)\n            c.paint(disc(0, 0.47, 0.085), B)\n        elif cls in (19, 20):\n            s = -1 if cls == 19 else 1\n            a = (180, 270) if s > 0 else (0, -90)\n            p = chain([(-0.10 * s, 0.64)], arc(0.10 * s, 0.18, 0.20, 0.20, a[0], a[1], 16), [(0.32 * s, -0.04)])\n            c.paint(stroke(p, 0.16), B)\n        elif cls == 21:\n            p = chain([(0.10, 0.64), (0.10, 0.38), (-0.10, 0.18), (-0.10, -0.02), (0.06, -0.20)])\n            c.paint(stroke(p, 0.14), B)\n        elif cls == 22:\n            c.paint(rect(-0.52, 0.46, 0.52, 0.60), B)\n            c.paint(disc(-0.24, 0.48, 0.16) & (V <= 0.52), B)\n            c.paint(disc(0.24, 0.48, 0.16) & (V <= 0.52), B)\n        elif cls == 23:\n            c.paint(flat_car(0.0, 0.04, 0.52, 0.30), B)\n            for x in (-0.20, 0.14):\n                c.paint(stroke(chain(arc(x, 0.30, 0.09, 0.07, -90, 90, 10), arc(x, 0.44, 0.09, 0.07, -90, -270, 10)), 0.06), B)\n        elif cls == 24:\n            c.paint(stroke([(-0.16, 0.64), (-0.16, -0.18)], 0.12), B)\n            c.paint(stroke([(0.26, 0.64), (0.26, 0.30), (0.08, 0.10), (0.08, -0.18)], 0.12), B)\n        elif cls == 25:\n            c.paint(disc(-0.12, -0.20, 0.09), B)\n            c.paint(stroke([(-0.10, -0.10), (0.0, 0.22)], 0.16), B)\n            c.paint(stroke([(0.0, 0.22), (-0.16, 0.62)], 0.11), B)\n            c.paint(stroke([(0.0, 0.22), (0.12, 0.42), (0.10, 0.62)], 0.11), B)\n            c.paint(stroke([(-0.06, -0.02), (0.18, 0.18), (0.30, 0.42)], 0.08), B)\n            c.paint(disc(0.38, 0.64, 0.20) & (V <= 0.62), B)\n        elif cls == 26:\n            box = [(-0.17, -0.38), (0.17, -0.38), (0.17, 0.62), (-0.17, 0.62)]\n            c.paint(rounded_polygon(box, 0.05) & ~rounded_polygon(offset_polygon(box, 0.045), 0.03), B)\n            c.paint(disc(0, -0.21, 0.11), TL_RED)\n            c.paint(disc(0, 0.12, 0.11), TL_YEL)\n            c.paint(disc(0, 0.45, 0.11), TL_GRN)\n        elif cls == 27:\n            c.paint(person(0.0, -0.24, 0.86), B)\n        elif cls == 28:\n            c.paint(person(-0.22, 0.0, 0.62), B)\n            c.paint(person(0.16, -0.24, 0.86), B)\n        elif cls == 29:\n            c.paint(disc(-0.28, 0.42, 0.18) & ~disc(-0.28, 0.42, 0.11), B)\n            c.paint(disc(0.28, 0.42, 0.18) & ~disc(0.28, 0.42, 0.11), B)\n            c.paint(stroke([(-0.28, 0.42), (-0.06, 0.16), (0.20, 0.16), (0.28, 0.42)], 0.06), B)\n            c.paint(stroke([(-0.06, 0.16), (0.04, 0.42), (0.20, 0.16)], 0.06), B)\n            c.paint(disc(0.06, -0.24, 0.08), B)\n            c.paint(stroke([(0.02, -0.14), (-0.06, 0.14)], 0.11), B)\n            c.paint(stroke([(0.0, -0.08), (0.20, 0.08)], 0.07), B)\n            c.paint(stroke([(-0.06, 0.14), (0.06, 0.32)], 0.08), B)\n        elif cls == 30:\n            cx, cy, R = 0.0, 0.18, 0.32\n            for k in range(6):\n                a = np.radians(90 + 60 * k)\n                c.paint(stroke([(cx, cy), (cx + R * np.cos(a), cy + R * np.sin(a))], 0.08), B)\n                bx, by = cx + 0.6 * R * np.cos(a), cy + 0.6 * R * np.sin(a)\n                for da in (-50, 50):\n                    a2 = a + np.radians(da)\n                    c.paint(stroke([(bx, by), (bx + 0.13 * np.cos(a2), by + 0.13 * np.sin(a2))], 0.06), B)\n            c.paint(disc(cx, cy, 0.08), B)\n        elif cls == 31:\n            # deer leaping to the left: head up-left, rump and hind legs down-right, fore legs stretched forward\n            c.paint(ellipse(0.04, 0.19, 0.27, 0.095, np.radians(30)), B)\n            c.paint(stroke([(-0.16, 0.07), (-0.26, -0.14)], 0.10), B)\n            c.paint(ellipse(-0.30, -0.18, 0.09, 0.055, np.radians(-25)), B)\n            c.paint(stroke([(-0.27, -0.22), (-0.22, -0.36)], 0.04), B)\n            c.paint(stroke([(-0.32, -0.23), (-0.36, -0.36)], 0.04), B)\n            c.paint(stroke([(-0.10, 0.12), (-0.36, 0.16), (-0.50, 0.12)], 0.06), B)\n            c.paint(stroke([(0.22, 0.30), (0.34, 0.46), (0.50, 0.58)], 0.07), B)\n            c.paint(stroke([(0.18, 0.30), (0.26, 0.50), (0.36, 0.62)], 0.06), B)\n    return c\n\n\nN_VARIANTS = {c: 2 for c in range(43)}\nN_VARIANTS[15] = 1\n'
R = types.ModuleType("render"); exec(compile(RENDER_SRC, "render", "exec"), R.__dict__)
import json, sys, os
import numpy as np

# ---------------------------------------------------------------- template sampling
SS = 6
_o = (np.arange(32 * SS) + 0.5) / SS
_X, _Y = np.meshgrid(_o, _o)


def gauss1d(sigma):
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    g = np.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def gauss_matrix(sigma, n=32):
    """row-normalised gaussian smoothing matrix for one axis of an n-pixel frame (edges renormalised)"""
    x = np.arange(n)
    M = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * sigma * sigma))
    return M / M.sum(1, keepdims=True)


def blur2(img, sigma):
    """gaussian blur of (..., 32, 32) with weights renormalised at the frame edge"""
    M = gauss_matrix(sigma)
    return np.einsum('ij,...jk,lk->...il', M, img, M)


def sample(arr, D, ax=1.0, ay=1.0, cx=16.0, cy=16.0):
    """area-sample a canvas array (H, W, ...) into a 32x32 frame; sign half-width D/2 pixels"""
    u = (_X - cx) / (D / 2 * ax)
    v = (_Y - cy) / (D / 2 * ay)
    iu = np.floor((u + R.HR) / (2 * R.HR) * R.NHR).astype(int)
    iv = np.floor((v + R.HR) / (2 * R.HR) * R.NHR).astype(int)
    ok = (iu >= 0) & (iu < R.NHR) & (iv >= 0) & (iv < R.NHR)
    iu, iv = np.clip(iu, 0, R.NHR - 1), np.clip(iv, 0, R.NHR - 1)
    a = arr[iv, iu]
    a = a * ok.reshape(ok.shape + (1,) * (a.ndim - 2))
    a = a.reshape((32, SS, 32, SS) + a.shape[2:]).mean((1, 3))
    return a


# ---------------------------------------------------------------- the fixed feature map (also used inside the network)
SIG_HP = 3.0
LP = np.kron(gauss_matrix(SIG_HP), gauss_matrix(SIG_HP))     # (1024, 1024) separable low-pass, rows sum to 1
HP = np.eye(1024) - LP
FEAT_W = np.array([1.0, 1.0, 1.0])
BLUR = 0.8


def feature_matrix():
    """dense map from flattened RGB (3*1024) to flattened [L high-pass, R-G, B-(R+G)/2]"""
    I = np.eye(1024)
    W = np.zeros((3072, 3072))
    for c in range(3):
        W[0:1024, c * 1024:(c + 1) * 1024] = HP / 3.0
    W[1024:2048, 0:1024] = I
    W[1024:2048, 1024:2048] = -I
    W[2048:3072, 0:1024] = -0.5 * I
    W[2048:3072, 1024:2048] = -0.5 * I
    W[2048:3072, 2048:3072] = I
    return W


def features(rgb):
    """rgb (3, 32, 32) -> (3, 32, 32) features, same linear map as the network's first layer"""
    L = rgb.mean(0).reshape(1024)
    return np.stack([(HP @ L).reshape(32, 32), rgb[0] - rgb[1], rgb[2] - 0.5 * (rgb[0] + rgb[1])])


def make_template(cv, D, ax, ay):
    rgbp = sample(cv.rgb * cv.cov[..., None], D, ax, ay).transpose(2, 0, 1)
    cov = sample(cv.cov, D, ax, ay)
    pic = sample(cv.picto, D, ax, ay).transpose(2, 0, 1)
    full = rgbp + cv.bg_l * (1 - cov)[None]
    full, pic = blur2(full, BLUR), blur2(pic, BLUR)
    Tf, Tp = features(full), features(pic)
    frame = np.zeros_like(Tf)
    picto = np.zeros_like(Tf)
    for f in range(3):
        tz = Tf[f] - Tf[f].mean()
        n = np.sqrt((tz * tz).sum())
        if n < 1e-4:
            continue
        pz = Tp[f] - Tp[f].mean()
        frame[f] = FEAT_W[f] * (tz - pz) / n / 32.0
        picto[f] = FEAT_W[f] * pz / n / 32.0
    return frame, picto


SCALES = [19.5, 21.5, 23.5, 25.2, 26.8, 28.4]
ASPECTS = [(1.0, 1.0), (0.92, 1.0), (1.0, 0.92)]


def build_templates():
    per_class = []
    for k in range(43):
        ts = []
        for var in range(R.N_VARIANTS[k]):
            cv = R.draw(k, var)
            for D in SCALES:
                for ax, ay in ASPECTS:
                    ts.append(make_template(cv, D, ax, ay))
        per_class.append(ts)
    V = max(len(t) for t in per_class)
    F, P = [], []
    for ts in per_class:
        ts = ts + [ts[i % len(ts)] for i in range(V - len(ts))]
        F.extend(t[0] for t in ts)
        P.extend(t[1] for t in ts)
    return np.array(F, np.float32), np.array(P, np.float32), V


def build(out):
    F, P, V = build_templates()
    NT = len(F)
    layers = [
        {"name": "feat", "op": "dense", "inputs": ["image"], "out": 3072},
        {"name": "feat3", "op": "reshape", "inputs": ["feat"], "shape": [3, 32, 32]},
        {"name": "nrm", "op": "norm", "inputs": ["feat3"]},
        {"name": "frame", "op": "conv", "inputs": ["nrm"], "out": NT, "kernel": [32, 32], "stride": 1, "padding": 2},
        {"name": "picto", "op": "conv", "inputs": ["nrm"], "out": NT, "kernel": [32, 32], "stride": 1, "padding": 3},
        {"name": "pmax", "op": "maxpool", "inputs": ["picto"], "size": [3, 3], "stride": [1, 1]},
        {"name": "sum", "op": "add", "inputs": ["frame", "pmax"]},
        {"name": "shiftmax", "op": "maxpool", "inputs": ["sum"], "size": [5, 5], "stride": [5, 5]},
        {"name": "flat", "op": "reshape", "inputs": ["shiftmax"], "shape": [1, NT, 1]},
        {"name": "hypmax", "op": "maxpool", "inputs": ["flat"], "size": [V, 1], "stride": [V, 1]},
        {"name": "scores", "op": "reshape", "inputs": ["hypmax"], "shape": [43]},
    ]
    weights = {
        "feat.W": feature_matrix().astype(np.float32), "feat.b": np.zeros(3072, np.float32),
        "frame.W": F, "frame.b": np.zeros(NT, np.float32),
        "picto.W": P, "picto.b": np.zeros(NT, np.float32),
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **weights)


if __name__ == "__main__":
    build(sys.argv[1])
