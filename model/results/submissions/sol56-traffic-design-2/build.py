#!/usr/bin/env python3
"""Build a fully hand-designed traffic-sign template network without reading data files."""
import json
import math
import os
import sys
import numpy as np

# Semantic material indices used only while constructing analytic templates.
BG, WHITE, RED, BLUE, BLACK, YELLOW = range(6)


def line_mask(X, Y, x1, y1, x2, y2, width):
    vx, vy = x2 - x1, y2 - y1
    den = vx * vx + vy * vy
    if den == 0:
        return (X - x1) ** 2 + (Y - y1) ** 2 <= width ** 2
    t = np.clip(((X - x1) * vx + (Y - y1) * vy) / den, 0.0, 1.0)
    return (X - (x1 + t * vx)) ** 2 + (Y - (y1 + t * vy)) ** 2 <= width ** 2


def poly_mask(X, Y, points):
    # Even/odd polygon fill. Vertices are mathematical constants, not image measurements.
    inside = np.zeros(X.shape, bool)
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i]
        xj, yj = points[j]
        cross = ((yi > Y) != (yj > Y)) & (X < (xj - xi) * (Y - yi) / (yj - yi + 1e-20) + xi)
        inside ^= cross
        j = i
    return inside


def ellipse_ring(X, Y, cx, cy, rx, ry, width):
    # A polyline ellipse gives a stroke of nearly constant Euclidean thickness.
    a = np.linspace(0, 2 * np.pi, 65)
    mask = np.zeros(X.shape, bool)
    for i in range(64):
        mask |= line_mask(X, Y, cx + rx * np.cos(a[i]), cy + ry * np.sin(a[i]),
                          cx + rx * np.cos(a[i + 1]), cy + ry * np.sin(a[i + 1]), width)
    return mask


def stroke(X, Y, points, width):
    m = np.zeros(X.shape, bool)
    for p, q in zip(points[:-1], points[1:]):
        m |= line_mask(X, Y, p[0], p[1], q[0], q[1], width)
    return m


def digit_mask(X, Y, digit, cx, cy, halfw, halfh, width):
    """A compact rounded DIN-like numeral made from general geometric strokes."""
    def tr(points):
        return [(cx + halfw * x, cy + halfh * y) for x, y in points]
    w = width
    if digit == "0":
        return ellipse_ring(X, Y, cx, cy, halfw, halfh, w)
    if digit == "1":
        return stroke(X, Y, tr([(-.58, -.55), (0, -1), (0, 1), (-.55, 1), (.55, 1)]), w)
    if digit == "2":
        a = np.linspace(np.pi * 1.08, np.pi * 1.93, 15)
        pts = [(0.82 * np.cos(t), -.48 + .52 * np.sin(t)) for t in a]
        pts += [(.80, -.38), (-.82, 1), (.86, 1)]
        return stroke(X, Y, tr(pts), w)
    if digit == "3":
        a1 = np.linspace(-np.pi / 2, np.pi / 2, 15)
        a2 = np.linspace(-np.pi / 2, np.pi / 2, 15)
        pts1 = [(-.08 + .82 * np.cos(t), -.50 + .50 * np.sin(t)) for t in a1]
        pts2 = [(-.08 + .82 * np.cos(t), .50 + .50 * np.sin(t)) for t in a2]
        return stroke(X, Y, tr(pts1), w) | stroke(X, Y, tr(pts2), w)
    if digit == "5":
        return stroke(X, Y, tr([(.82, -1), (-.72, -1), (-.82, -.05),
                                (.45, -.05), (.80, .20), (.78, .72),
                                (.48, 1), (-.62, 1), (-.84, .80)]), w)
    if digit == "6":
        a = np.linspace(0, 2 * np.pi, 34)
        loop = [(.00 + .78 * np.cos(t), .43 + .57 * np.sin(t)) for t in a]
        lead = [(.72, -.80), (.36, -1), (-.30, -.88), (-.68, -.35), (-.73, .38)]
        return stroke(X, Y, tr(lead), w) | stroke(X, Y, tr(loop), w)
    if digit == "7":
        return stroke(X, Y, tr([(-.86, -1), (.86, -1), (.15, 1)]), w)
    if digit == "8":
        return (ellipse_ring(X, Y, cx, cy - halfh * .50, halfw * .82, halfh * .50, w) |
                ellipse_ring(X, Y, cx, cy + halfh * .50, halfw * .91, halfh * .52, w))
    return np.zeros(X.shape, bool)


def put_speed(mat, X, Y, text, material=BLACK):
    n = len(text)
    if n == 2:
        centers, hw = [-.205, .205], .155
    else:
        centers, hw = [-.255, 0, .255], .110
    for c, d in zip(centers, text):
        mat[digit_mask(X, Y, d, c, .02, hw, .34, .055)] = material


def put_car(mat, X, Y, cx, cy, scale=.42, material=BLACK):
    body = poly_mask(X, Y, [(cx - .50 * scale, cy + .10 * scale),
                            (cx - .40 * scale, cy - .12 * scale),
                            (cx - .18 * scale, cy - .30 * scale),
                            (cx + .20 * scale, cy - .30 * scale),
                            (cx + .40 * scale, cy - .08 * scale),
                            (cx + .52 * scale, cy), (cx + .48 * scale, cy + .23 * scale),
                            (cx - .50 * scale, cy + .23 * scale)])
    mat[body] = material
    for wx in (-.30, .30):
        mat[(X - (cx + wx * scale)) ** 2 + (Y - (cy + .23 * scale)) ** 2 < (.11 * scale) ** 2] = BLACK


def put_truck(mat, X, Y, cx, cy, scale=.48, material=BLACK):
    box = (np.abs(X - (cx - .12 * scale)) < .40 * scale) & (np.abs(Y - (cy - .08 * scale)) < .27 * scale)
    cab = poly_mask(X, Y, [(cx + .28 * scale, cy - .25 * scale), (cx + .47 * scale, cy - .25 * scale),
                            (cx + .58 * scale, cy), (cx + .58 * scale, cy + .20 * scale),
                            (cx + .25 * scale, cy + .20 * scale)])
    mat[box | cab] = material
    for wx in (-.28, .38):
        mat[(X - (cx + wx * scale)) ** 2 + (Y - (cy + .22 * scale)) ** 2 < (.10 * scale) ** 2] = BLACK


def put_arrow(mat, X, Y, kind):
    white = WHITE
    if kind == "up":
        m = line_mask(X, Y, 0, .58, 0, -.38, .095)
        m |= poly_mask(X, Y, [(0, -.72), (-.27, -.34), (.27, -.34)])
    elif kind in ("right", "left"):
        sg = 1 if kind == "right" else -1
        pts = [(0, .58), (0, .10), (.08 * sg, -.08), (.40 * sg, -.20)]
        m = stroke(X, Y, pts, .09)
        m |= poly_mask(X, Y, [(.62 * sg, -.22), (.30 * sg, -.43), (.31 * sg, .02)])
    elif kind in ("up_right", "up_left"):
        sg = 1 if kind == "up_right" else -1
        m = line_mask(X, Y, 0, .62, 0, -.48, .075)
        m |= poly_mask(X, Y, [(0, -.72), (-.22, -.40), (.22, -.40)])
        m |= stroke(X, Y, [(0, .08), (.36 * sg, -.16)], .075)
        m |= poly_mask(X, Y, [(.54 * sg, -.28), (.24 * sg, -.30), (.43 * sg, -.02)])
    elif kind in ("down_right", "down_left"):
        sg = 1 if kind == "down_right" else -1
        m = line_mask(X, Y, -.34 * sg, -.34, .30 * sg, .32, .10)
        m |= poly_mask(X, Y, [(.51 * sg, .53), (.43 * sg, .13), (.10 * sg, .45)])
    else:
        m = np.zeros(X.shape, bool)
    mat[m] = white


def person_mask(X, Y, cx, cy, scale=1.0, running=True):
    m = (X - cx) ** 2 + (Y - (cy - .32 * scale)) ** 2 < (.085 * scale) ** 2
    m |= line_mask(X, Y, cx, cy - .22 * scale, cx - .04 * scale, cy + .12 * scale, .045 * scale)
    m |= line_mask(X, Y, cx - .02 * scale, cy - .10 * scale, cx - .25 * scale, cy + .02 * scale, .038 * scale)
    m |= line_mask(X, Y, cx - .01 * scale, cy - .08 * scale, cx + .20 * scale, cy - .20 * scale, .038 * scale)
    m |= line_mask(X, Y, cx - .04 * scale, cy + .10 * scale, cx - .23 * scale, cy + .40 * scale, .044 * scale)
    m |= line_mask(X, Y, cx - .04 * scale, cy + .10 * scale, cx + .22 * scale, cy + .36 * scale, .044 * scale)
    return m


def bicycle_mask(X, Y, cx=0, cy=.14, scale=1.0):
    m = ellipse_ring(X, Y, cx - .27 * scale, cy + .18 * scale, .20 * scale, .20 * scale, .035 * scale)
    m |= ellipse_ring(X, Y, cx + .28 * scale, cy + .18 * scale, .20 * scale, .20 * scale, .035 * scale)
    p1 = (cx - .27 * scale, cy + .18 * scale)
    p2 = (cx - .03 * scale, cy - .08 * scale)
    p3 = (cx + .10 * scale, cy + .18 * scale)
    p4 = (cx + .28 * scale, cy + .18 * scale)
    m |= stroke(X, Y, [p1, p2, p3, p1], .028 * scale)
    m |= stroke(X, Y, [p2, p4], .028 * scale)
    m |= line_mask(X, Y, p2[0] - .07 * scale, p2[1], p2[0] + .07 * scale, p2[1], .025 * scale)
    return m


def cancellation(mat, X, Y):
    # Parallel lower-left to upper-right cancellation bars.
    for off in (-.54, -.18, .18, .54):
        mat[line_mask(X, Y, -.72 + off, .72, .72 + off, -.72, .055)] = BLACK


def warning_icon(mat, X, Y, cls):
    m = np.zeros(X.shape, bool)
    if cls == 11:  # priority at next junction: thick major road crossing a minor one
        m = line_mask(X, Y, 0, -.42, 0, .43, .075) | line_mask(X, Y, -.36, .03, .36, .03, .055)
    elif cls == 18:
        m = line_mask(X, Y, 0, -.34, 0, .16, .055) | ((X ** 2 + (Y - .34) ** 2) < .065 ** 2)
    elif cls in (19, 20):
        sg = -1 if cls == 19 else 1
        pts = [(0, .43), (.02 * sg, .18), (-.08 * sg, -.02), (-.31 * sg, -.18), (-.35 * sg, -.39)]
        m = stroke(X, Y, pts, .065)
    elif cls == 21:
        m = stroke(X, Y, [(0, .44), (-.03, .22), (.20, -.02), (.22, -.23), (.10, -.42)], .065)
    elif cls == 22:
        # Two rounded humps above a short baseline.
        a = np.linspace(np.pi, 2 * np.pi, 18)
        p1 = [(-.38 + .20 * np.cos(t), .20 + .19 * np.sin(t)) for t in a]
        p2 = [(.16 + .26 * np.cos(t), .20 + .25 * np.sin(t)) for t in a]
        m = stroke(X, Y, p1 + p2, .055)
    elif cls == 23:
        put_car(mat, X, Y, 0, -.10, .55, BLACK)
        m = stroke(X, Y, [(-.30, .23), (-.12, .29), (-.31, .38)], .028)
        m |= stroke(X, Y, [(0, .22), (.22, .28), (.02, .39)], .028)
    elif cls == 24:
        m = stroke(X, Y, [(-.28, -.40), (-.28, .40)], .070)
        m |= stroke(X, Y, [(.42, -.40), (.18, .40)], .070)
    elif cls == 25:
        # Road worker with head, torso, limbs and shovel.
        m = (X ** 2 + (Y + .34) ** 2 < .075 ** 2)
        m |= stroke(X, Y, [(0, -.24), (-.08, .06), (-.30, .33)], .048)
        m |= stroke(X, Y, [(-.05, -.08), (.23, .05), (.37, .36)], .040)
        m |= stroke(X, Y, [(-.06, -.07), (-.31, .08)], .040)
        m |= line_mask(X, Y, .22, -.18, .35, .39, .027)
        m |= poly_mask(X, Y, [(.25, -.22), (.37, -.26), (.27, -.08)])
    elif cls == 26:
        m = (np.abs(X) < .15) & (np.abs(Y) < .42)
        # Colored lamps are put on top of the dark housing below.
        mat[m] = BLACK
        mat[X ** 2 + (Y + .25) ** 2 < .075 ** 2] = RED
        mat[X ** 2 + Y ** 2 < .075 ** 2] = YELLOW
        # Green is represented as a dark lamp because no green semantic channel is used.
        mat[X ** 2 + (Y - .25) ** 2 < .075 ** 2] = BLACK
        return
    elif cls == 27:
        m = person_mask(X, Y, 0, .02, 1.0)
    elif cls == 28:
        m = person_mask(X, Y, -.13, .07, .82) | person_mask(X, Y, .16, -.03, .65)
    elif cls == 29:
        m = bicycle_mask(X, Y, 0, .03, 1.0)
    elif cls == 30:
        for a in (0, np.pi / 3, 2 * np.pi / 3):
            dx, dy = .42 * np.cos(a), .42 * np.sin(a)
            m |= line_mask(X, Y, -dx, -dy, dx, dy, .032)
            for sg in (-1, 1):
                ex, ey = sg * dx, sg * dy
                ang = a + (0 if sg > 0 else np.pi)
                for da in (-.55, .55):
                    m |= line_mask(X, Y, ex, ey, ex - .13 * np.cos(ang + da), ey - .13 * np.sin(ang + da), .025)
    elif cls == 31:
        # Leaping deer: body, neck, head, four legs and antlers.
        m = ((X + .03) / .29) ** 2 + ((Y + .02) / .14) ** 2 < 1
        m |= stroke(X, Y, [(.16, -.08), (.31, -.29), (.41, -.30)], .045)
        m |= (X - .44) ** 2 + (Y + .31) ** 2 < .055 ** 2
        m |= stroke(X, Y, [(-.18, .07), (-.34, .33)], .035)
        m |= stroke(X, Y, [(-.02, .09), (-.07, .37)], .035)
        m |= stroke(X, Y, [(.12, .07), (.28, .31)], .035)
        m |= stroke(X, Y, [(.30, -.31), (.25, -.45), (.18, -.50)], .022)
        m |= stroke(X, Y, [(.32, -.33), (.39, -.46), (.47, -.48)], .022)
    mat[m] = BLACK


def render_class(cls, radius, shift_x=0.0, shift_y=0.0, aspect_x=1.0, aspect_y=1.0, out_size=16, ss=5):
    """Render one idealized sign from elementary geometric primitives."""
    n = out_size * ss
    q = (np.arange(n) + .5) / n * 2 - 1
    gx, gy = np.meshgrid(q, q)
    X = (gx - shift_x) / (radius * aspect_x)
    Y = (gy - shift_y) / (radius * aspect_y)
    mat = np.zeros((n, n), np.uint8)

    circle = X * X + Y * Y <= 1
    sign = circle.copy()

    if cls in list(range(0, 11)) + [15, 16]:
        mat[circle] = RED
        mat[X * X + Y * Y <= .76 ** 2] = WHITE
        if cls in [0, 1, 2, 3, 4, 5, 7, 8]:
            put_speed(mat, X, Y, {0:"20", 1:"30", 2:"50", 3:"60", 4:"70", 5:"80", 7:"100", 8:"120"}[cls])
        elif cls == 9:
            put_car(mat, X, Y, -.20, .04, .50, RED)
            put_car(mat, X, Y, .23, .04, .50, BLACK)
        elif cls == 10:
            put_truck(mat, X, Y, -.17, .03, .48, RED)
            put_car(mat, X, Y, .28, .06, .43, BLACK)
        elif cls == 16:
            put_truck(mat, X, Y, 0, .02, .82, BLACK)
    elif cls == 17:
        mat[circle] = RED
        mat[(np.abs(Y) < .16) & (np.abs(X) < .72)] = WHITE
    elif cls in [6, 32, 41, 42]:
        mat[circle] = WHITE
        # a restrained dark rim helps define the cancellation disc
        mat[(X * X + Y * Y <= 1) & (X * X + Y * Y >= .91 ** 2)] = BLACK
        if cls == 6:
            put_speed(mat, X, Y, "80", BLACK)
        elif cls == 41:
            put_car(mat, X, Y, -.20, .04, .50, BLACK)
            put_car(mat, X, Y, .23, .04, .50, BLACK)
        elif cls == 42:
            put_truck(mat, X, Y, -.17, .03, .48, BLACK)
            put_car(mat, X, Y, .28, .06, .43, BLACK)
        cancellation(mat, X, Y)
    elif cls == 12:
        outer = poly_mask(X, Y, [(0, -.98), (.98, 0), (0, .98), (-.98, 0)])
        inner = poly_mask(X, Y, [(0, -.72), (.72, 0), (0, .72), (-.72, 0)])
        sign = outer
        mat[outer] = WHITE
        mat[inner] = YELLOW
    elif cls == 13:
        outer = poly_mask(X, Y, [(-.91, -.61), (.91, -.61), (0, .98)])
        inner = poly_mask(X, Y, [(-.64, -.43), (.64, -.43), (0, .70)])
        sign = outer
        mat[outer] = RED
        mat[inner] = WHITE
    elif cls == 14:
        ang = np.pi / 8 + np.arange(8) * np.pi / 4
        pts = [(np.cos(a), np.sin(a)) for a in ang]
        sign = poly_mask(X, Y, pts)
        mat[sign] = RED
        # Geometric block lettering for STOP.
        def L(a, b, w=.040):
            return line_mask(X, Y, a[0], a[1], b[0], b[1], w)
        m = L((-.52,-.12),(-.31,-.12)) | L((-.52,-.12),(-.52,.02)) | L((-.52,.02),(-.31,.02)) | L((-.31,.02),(-.31,.16)) | L((-.31,.16),(-.52,.16))
        m |= L((-.24,-.12),(.02,-.12)) | L((-.11,-.12),(-.11,.16))
        m |= ellipse_ring(X,Y,.16,.02,.13,.15,.040)
        m |= L((.34,.16),(.34,-.12)) | L((.34,-.12),(.51,-.12)) | ellipse_ring(X,Y,.43,-.04,.09,.08,.040)
        mat[m] = WHITE
    elif cls in [11] + list(range(18, 32)):
        outer = poly_mask(X, Y, [(0, -.98), (-.91, .66), (.91, .66)])
        inner = poly_mask(X, Y, [(0, -.69), (-.62, .48), (.62, .48)])
        sign = outer
        mat[outer] = RED
        mat[inner] = WHITE
        warning_icon(mat, X, Y, cls)
    elif cls in list(range(33, 41)):
        mat[circle] = BLUE
        if cls == 33: put_arrow(mat, X, Y, "right")
        elif cls == 34: put_arrow(mat, X, Y, "left")
        elif cls == 35: put_arrow(mat, X, Y, "up")
        elif cls == 36: put_arrow(mat, X, Y, "up_right")
        elif cls == 37: put_arrow(mat, X, Y, "up_left")
        elif cls == 38: put_arrow(mat, X, Y, "down_right")
        elif cls == 39: put_arrow(mat, X, Y, "down_left")
        elif cls == 40:
            # Three clockwise arrows tangent to a circle.
            m = np.zeros(X.shape, bool)
            for a in (-np.pi/2, np.pi/6, 5*np.pi/6):
                a0, a1 = a - .52, a + .52
                aa = np.linspace(a0, a1, 12)
                pts = [(.38*np.cos(t), .38*np.sin(t)) for t in aa]
                m |= stroke(X, Y, pts, .075)
                ex, ey = pts[-1]
                tangent = a1 + np.pi/2
                tip = (ex + .20*np.cos(tangent), ey + .20*np.sin(tangent))
                left = (ex + .10*np.cos(tangent+2.25), ey + .10*np.sin(tangent+2.25))
                right = (ex + .10*np.cos(tangent-2.25), ey + .10*np.sin(tangent-2.25))
                m |= poly_mask(X, Y, [tip, left, right])
            mat[m] = WHITE

    # Supersampled material coverages and sign support.
    cov = np.stack([(mat == k).reshape(out_size, ss, out_size, ss).mean((1, 3)) for k in range(6)])
    support = sign.reshape(out_size, ss, out_size, ss).mean((1, 3))
    target = np.zeros((5, out_size, out_size), np.float64)
    target[0] = cov[BLACK]
    target[1] = cov[WHITE] + .35 * cov[YELLOW]
    target[2] = 2.0 * cov[RED]
    target[3] = 2.0 * cov[BLUE]
    target[4] = 2.0 * cov[YELLOW]
    return target, support


def make_template_weights():
    variants = []
    for radius in (.76, .85, .94):
        for dx, dy in ((0,0), (-.085,0), (.085,0), (0,-.085), (0,.085)):
            variants.append((radius, dx, dy, 1.0, 1.0))
    variants += [(.85,0,0,.86,1.0), (.85,0,0,1.14,1.0),
                 (.85,0,0,1.0,.86), (.85,0,0,1.0,1.14)]
    rows = []
    # Relative reliability after each semantic template plane is normalized.
    rel = np.array([2.35, .90, 1.45, 1.55, 1.20], np.float64)
    for cls in range(43):
        for radius, dx, dy, ax, ay in variants:
            t, support = render_class(cls, radius, dx, dy, ax, ay)
            allw = np.zeros_like(t)
            domain = support > .08
            for ch in range(5):
                plane = t[ch]
                if np.ptp(plane[domain]) < 1e-5:
                    continue
                mean = plane[domain].mean()
                w = np.where(domain, plane - mean, 0.0)
                # A tiny 3x3 binomial blur gives tolerance to jagged resize edges.
                p = np.pad(w, 1)
                w = (p[:-2,:-2] + 2*p[:-2,1:-1] + p[:-2,2:] +
                     2*p[1:-1,:-2] + 4*p[1:-1,1:-1] + 2*p[1:-1,2:] +
                     p[2:,:-2] + 2*p[2:,1:-1] + p[2:,2:]) / 16.0
                norm = np.sqrt((w*w).sum())
                if norm > 1e-8:
                    allw[ch] = w / norm
            norm = np.sqrt((allw*allw).sum())
            if norm > 0:
                allw /= norm
            rows.append(allw.reshape(-1))
    # Hand-set class priors compensate only for analytic template complexity and
    # reliability, never for observed frequencies or fitted statistics.
    prior = np.zeros(43, np.float32)
    prior[[14, 26]] = -.15
    prior[[6, 32, 41, 42]] = -.05
    prior[list(range(0,6)) + [7,8,9,10,16,17]] = .03
    prior[15] = -.12
    prior[[12,13]] = .07
    prior[list(range(18,26)) + list(range(27,32))] = .05
    prior[list(range(33,41))] = .08
    return np.asarray(rows, np.float32), np.repeat(prior, len(variants)), len(variants)


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    layers = [
        {"name":"base", "op":"conv", "inputs":["image"], "out":4, "kernel":[1,1], "stride":1, "padding":0},
        {"name":"zn", "op":"norm", "inputs":["base"]},
        {"name":"nthr", "op":"conv", "inputs":["zn"], "out":10, "kernel":[1,1], "stride":1, "padding":0},
        {"name":"nrelu", "op":"relu", "inputs":["nthr"]},
        {"name":"nclip", "op":"conv", "inputs":["nrelu"], "out":5, "kernel":[1,1], "stride":1, "padding":0},
        {"name":"rawthr", "op":"conv", "inputs":["image"], "out":6, "kernel":[1,1], "stride":1, "padding":0},
        {"name":"rawrelu", "op":"relu", "inputs":["rawthr"]},
        {"name":"rawclip", "op":"conv", "inputs":["rawrelu"], "out":5, "kernel":[1,1], "stride":1, "padding":0},
        {"name":"semantic", "op":"add", "inputs":["nclip", "rawclip"]},
        {"name":"small", "op":"avgpool", "inputs":["semantic"], "size":[2,2], "stride":[2,2]},
    ]
    Wt, bt, nvar = make_template_weights()
    layers += [
        {"name":"matches", "op":"dense", "inputs":["small"], "out":43*nvar},
        {"name":"match_grid", "op":"reshape", "inputs":["matches"], "shape":[43,nvar,1]},
        {"name":"best", "op":"maxpool", "inputs":["match_grid"], "size":[nvar,1], "stride":[nvar,1]},
        {"name":"scores", "op":"reshape", "inputs":["best"], "shape":[43]},
    ]
    weights = {}
    # Gray luminance and three opponent-color coordinates.
    baseW = np.array([[.299,.587,.114], [1,-.5,-.5], [-.5,-.5,1], [.5,.5,-1]], np.float32)
    weights["base.W"] = baseW[:,:,None,None]
    weights["base.b"] = np.zeros(4, np.float32)
    # Two ReLUs per semantic coordinate make clip(q-threshold, 0, 1).
    qrows = [(-1,0), (1,0), (1,1), (1,2), (1,3)]
    nw = np.zeros((10,4,1,1), np.float32)
    nb = np.zeros(10, np.float32)
    for j,(sg,src) in enumerate(qrows):
        nw[2*j,src,0,0] = sg
        nw[2*j+1,src,0,0] = sg
        nb[2*j] = -.12
        nb[2*j+1] = -1.12
    weights["nthr.W"], weights["nthr.b"] = nw, nb
    cw = np.zeros((5,10,1,1), np.float32)
    for j in range(5):
        cw[j,2*j,0,0], cw[j,2*j+1,0,0] = 1, -1
    weights["nclip.W"], weights["nclip.b"] = cw, np.zeros(5,np.float32)
    # Saturation-sensitive raw color evidence. Dark/white rely on normalized luminance only.
    raww = np.zeros((6,3,1,1), np.float32)
    rawb = np.zeros(6,np.float32)
    opp = [np.array([1,-.5,-.5])/.22, np.array([-.5,-.5,1])/.20, np.array([.5,.5,-1])/.25]
    thresholds = [.03/.22, .03/.20, .05/.25]
    for j in range(3):
        raww[2*j,:,0,0] = opp[j]
        raww[2*j+1,:,0,0] = opp[j]
        rawb[2*j] = -thresholds[j]
        rawb[2*j+1] = -thresholds[j]-1
    weights["rawthr.W"], weights["rawthr.b"] = raww, rawb
    rcw = np.zeros((5,6,1,1), np.float32)
    for j in range(3):
        rcw[j+2,2*j,0,0], rcw[j+2,2*j+1,0,0] = 1,-1
    weights["rawclip.W"], weights["rawclip.b"] = rcw, np.zeros(5,np.float32)
    weights["matches.W"], weights["matches.b"] = Wt, bt
    with open(os.path.join(out_dir,"network.json"),"w") as f:
        json.dump({"layers":layers},f,separators=(",",":"))
    np.savez(os.path.join(out_dir,"weights.npz"), **weights)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: build.py OUTPUT_DIRECTORY")
    build(sys.argv[1])
