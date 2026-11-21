import json, os, sys
import numpy as np

# Hand-built traffic-sign template network.  The generated numbers are all from
# geometric formulae and simple stroke drawings in this file.

CEN = 15.5
BLACK = np.array([-1.0, -1.0, -1.0], np.float32)
WHITE = np.array([1.0, 1.0, 1.0], np.float32)
RED = np.array([2.2, -1.05, -1.05], np.float32)
BLUE = np.array([-1.0, -1.0, 2.25], np.float32)
YELLOW = np.array([1.25, 1.10, -0.95], np.float32)
GREEN = np.array([-0.8, 1.5, -0.8], np.float32)

DIGITS = {
    '0': 'abcfed', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc',
    '5': 'afgcd', '6': 'afgecd', '7': 'abc', '8': 'abcdefg', '9': 'abfgcd'
}


def grid(dx=0.0, dy=0.0, sc=1.0, rot=0.0):
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    xr = xx - CEN - dx
    yr = yy - CEN - dy
    co, si = np.cos(rot), np.sin(rot)
    x = (co * xr + si * yr) / sc + CEN
    y = (-si * xr + co * yr) / sc + CEN
    return x, y


def add(t, mask, color, w=1.0):
    m = mask.astype(np.float32) * np.float32(w)
    for c in range(3):
        t[c] += color[c] * m


def inside_circle(x, y, r, cx=CEN, cy=CEN, soft=0.7):
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.clip((r + soft - d) / (2 * soft), 0.0, 1.0)


def ring(x, y, r0, r1):
    return np.clip(inside_circle(x, y, r1) - inside_circle(x, y, r0), 0.0, 1.0)


def ellipse(x, y, cx, cy, rx, ry, soft=0.55):
    d = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    return np.clip((1.0 + soft / max(rx, ry) - d) / (2 * soft / max(rx, ry)), 0.0, 1.0)


def rect(x, y, x0, y0, x1, y1, soft=0.45):
    dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
    d = np.sqrt(dx * dx + dy * dy)
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    m = np.clip((soft - d) / soft, 0.0, 1.0)
    return np.maximum(m, inside.astype(np.float32))


def poly_mask(x, y, pts):
    pts = np.asarray(pts, np.float32)
    sign = None
    ok = np.ones_like(x, dtype=bool)
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if sign is None:
            s = np.sign(np.mean(cross))
            sign = 1.0 if s == 0 else s
        ok &= (cross * sign >= -0.1)
    return ok.astype(np.float32)


def scale_poly(pts, s, cx=CEN, cy=CEN):
    return [(cx + (px - cx) * s, cy + (py - cy) * s) for px, py in pts]


def dist_seg(x, y, x0, y0, x1, y1):
    vx, vy = x1 - x0, y1 - y0
    wx, wy = x - x0, y - y0
    den = vx * vx + vy * vy + 1e-6
    u = np.clip((wx * vx + wy * vy) / den, 0.0, 1.0)
    px, py = x0 + u * vx, y0 + u * vy
    return np.sqrt((x - px) ** 2 + (y - py) ** 2)


def line(x, y, x0, y0, x1, y1, width=1.0, soft=0.7):
    d = dist_seg(x, y, x0, y0, x1, y1)
    return np.clip((width + soft - d) / soft, 0.0, 1.0)


def polyline(x, y, pts, width=1.0):
    m = np.zeros_like(x, dtype=np.float32)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        m = np.maximum(m, line(x, y, x0, y0, x1, y1, width))
    return m


def tri_head(x, y, tip, base, halfw):
    tx, ty = tip; bx, by = base
    vx, vy = tx - bx, ty - by
    n = np.sqrt(vx * vx + vy * vy) + 1e-6
    px, py = -vy / n * halfw, vx / n * halfw
    return poly_mask(x, y, [(tx, ty), (bx + px, by + py), (bx - px, by - py)])


def arrow(x, y, start, end, width=1.35, head=4.2, headw=3.2):
    sx, sy = start; ex, ey = end
    vx, vy = ex - sx, ey - sy
    n = np.sqrt(vx * vx + vy * vy) + 1e-6
    bx, by = ex - vx / n * head, ey - vy / n * head
    return np.maximum(line(x, y, sx, sy, bx, by, width), tri_head(x, y, (ex, ey), (bx, by), headw))


def seg_digit_mask(x, y, d, x0, y0, w, h, thick):
    # Seven-stroke road-number approximation.
    xL, xR, xM = x0, x0 + w, x0 + 0.5 * w
    yT, yB, yM = y0, y0 + h, y0 + 0.5 * h
    segs = {
        'a': (xL + .18*w, yT, xR - .18*w, yT),
        'b': (xR, yT + .10*h, xR, yM - .08*h),
        'c': (xR, yM + .08*h, xR, yB - .10*h),
        'd': (xL + .18*w, yB, xR - .18*w, yB),
        'e': (xL, yM + .08*h, xL, yB - .10*h),
        'f': (xL, yT + .10*h, xL, yM - .08*h),
        'g': (xL + .18*w, yM, xR - .18*w, yM),
    }
    m = np.zeros_like(x, dtype=np.float32)
    for s in DIGITS[d]:
        a,b,c,dv = segs[s]
        m = np.maximum(m, line(x, y, a, b, c, dv, thick))
    if d == '1':
        m = np.maximum(m, line(x, y, xM, yT, xM, yB, thick*1.05))
    if d == '0':
        m = np.maximum(m, ring((x-xM)/(w*.50)*6+CEN, (y-(y0+h/2))/(h*.50)*6+CEN, 4.4, 6.1))
    return m


def draw_number(t, x, y, s, wt=2.7):
    n = len(s)
    if n == 2:
        w, h, gap = 5.5, 12.2, 1.4
        start, yy, th = CEN - (2*w+gap)/2, 9.6, 0.85
    else:
        w, h, gap = 4.4, 10.3, 0.75
        start, yy, th = CEN - (3*w+2*gap)/2, 10.7, 0.70
    for i, ch in enumerate(s):
        m = seg_digit_mask(x, y, ch, start + i*(w+gap), yy, w, h, th)
        add(t, m, BLACK, wt)


def draw_circle_base(t, x, y, red=True, blue=False, white=True):
    if blue:
        add(t, inside_circle(x, y, 14.0), BLUE, 1.4)
    elif red:
        add(t, ring(x, y, 10.7, 14.5), RED, 1.35)
        if white:
            add(t, inside_circle(x, y, 10.7), WHITE, 0.22)
    else:
        add(t, inside_circle(x, y, 13.5), WHITE, 0.42)
        add(t, ring(x, y, 12.7, 14.2), BLACK, 0.75)


def draw_cancel(t, x, y):
    draw_circle_base(t, x, y, red=False)
    # German cancellation: several descending black bands.
    for off in [-10, -5, 0, 5, 10]:
        add(t, line(x, y, 4 + off, 26, 25 + off, 5, 0.95), BLACK, 1.45)


def draw_car(t, x, y, cx, cy, color=BLACK, wt=2.2, truck=False):
    if truck:
        m = np.maximum(rect(x, y, cx-5.2, cy-2.3, cx+1.5, cy+1.7), rect(x, y, cx+1.0, cy-1.0, cx+5.0, cy+1.7))
        m = np.maximum(m, line(x, y, cx+1.5, cy-2.3, cx+4.7, cy+1.7, 0.8))
    else:
        m = np.maximum(rect(x, y, cx-4.7, cy-1.2, cx+4.7, cy+1.8), ellipse(x, y, cx, cy-1.8, 4.1, 1.8))
    m = np.maximum(m, ellipse(x, y, cx-3.0, cy+2.0, 1.0, 1.0))
    m = np.maximum(m, ellipse(x, y, cx+3.0, cy+2.0, 1.0, 1.0))
    add(t, m, color, wt)


def draw_triangle_base(t, x, y, inverted=False):
    if inverted:
        outer = [(3, 5), (29, 5), (16, 29)]
        ctr = (16, 13)
    else:
        outer = [(16, 3), (3, 28), (29, 28)]
        ctr = (16, 19)
    inner = scale_poly(outer, 0.72, ctr[0], ctr[1])
    om = poly_mask(x, y, outer)
    im = poly_mask(x, y, inner)
    add(t, np.clip(om - im, 0, 1), RED, 1.45)
    add(t, im, WHITE, 0.22 if not inverted else 0.55)


def draw_person(t, x, y, cx, cy, scale=1.0, wt=2.3, lean=0.0):
    add(t, ellipse(x, y, cx, cy-4.2*scale, 1.25*scale, 1.25*scale), BLACK, wt)
    add(t, line(x, y, cx, cy-3.0*scale, cx+lean*scale, cy+1.5*scale, 0.75*scale), BLACK, wt)
    add(t, line(x, y, cx-0.2*scale, cy-1.2*scale, cx-3.0*scale, cy+1.0*scale, 0.55*scale), BLACK, wt)
    add(t, line(x, y, cx+0.2*scale, cy-1.0*scale, cx+3.0*scale, cy-2.3*scale, 0.55*scale), BLACK, wt)
    add(t, line(x, y, cx+lean*scale, cy+1.3*scale, cx-2.5*scale, cy+5.0*scale, 0.65*scale), BLACK, wt)
    add(t, line(x, y, cx+lean*scale, cy+1.3*scale, cx+3.0*scale, cy+5.0*scale, 0.65*scale), BLACK, wt)


def draw_bicycle(t, x, y):
    add(t, ring(x, y, 2.1, 3.0) * ellipse(x, y, 11.5, 21.0, 3.0, 3.0), BLACK, 2.5)
    add(t, ring(x, y, 2.1, 3.0) * ellipse(x, y, 21.0, 21.0, 3.0, 3.0), BLACK, 2.5)
    add(t, polyline(x, y, [(11.5,21),(15.5,16),(18.2,21),(11.5,21),(21,21),(15.5,16)], 0.65), BLACK, 2.5)
    add(t, line(x, y, 15.5, 16, 17.2, 13.5, 0.55), BLACK, 2.5)
    add(t, line(x, y, 17.2, 13.5, 20.5, 14.2, 0.55), BLACK, 2.5)
    add(t, ellipse(x, y, 15.5, 11.7, 1.2, 1.2), BLACK, 2.5)


def draw_octagon(t, x, y):
    pts = [(10,3.5),(22,3.5),(28.5,10),(28.5,22),(22,28.5),(10,28.5),(3.5,22),(3.5,10)]
    add(t, poly_mask(x, y, pts), RED, 1.7)
    add(t, poly_mask(x, y, scale_poly(pts, 0.83)), WHITE, -0.25)  # slight red hole to emphasize edge
    # white border and block STOP-like strokes
    for p0,p1 in zip(pts, pts[1:]+pts[:1]):
        add(t, line(x,y,p0[0],p0[1],p1[0],p1[1],0.8), WHITE, 1.4)
    # STOP: four bright block glyphs, deliberately coarse.
    add(t, line(x,y,7,13,11,13,0.7), WHITE, 1.5); add(t, line(x,y,7,13,7,16,0.7), WHITE, 1.5); add(t, line(x,y,7,16,11,16,0.7), WHITE, 1.5); add(t, line(x,y,11,16,11,19,0.7), WHITE, 1.5); add(t, line(x,y,7,19,11,19,0.7), WHITE, 1.5)
    add(t, line(x,y,12,13,17,13,0.7), WHITE, 1.5); add(t, line(x,y,14.5,13,14.5,19,0.7), WHITE, 1.5)
    add(t, ring(x,y,2.2,3.1) * ellipse(x,y,20,16,3.2,3.6), WHITE, 1.4)
    add(t, line(x,y,24,13,24,19,0.75), WHITE, 1.5); add(t, line(x,y,24,13,28,13,0.75), WHITE, 1.5); add(t, line(x,y,28,13,28,16,0.75), WHITE, 1.5); add(t, line(x,y,24,16,28,16,0.75), WHITE, 1.5)


def draw_priority(t, x, y):
    outer = [(16,2.2),(29.8,16),(16,29.8),(2.2,16)]
    mid = scale_poly(outer,0.78)
    inner = scale_poly(outer,0.55)
    add(t, poly_mask(x,y,outer), WHITE, 1.0)
    add(t, np.clip(poly_mask(x,y,mid)-poly_mask(x,y,inner),0,1), BLACK, 0.45)
    add(t, poly_mask(x,y,inner), YELLOW, 1.7)


def warning_symbol(t, x, y, k):
    if k == 18:  # general danger
        add(t, line(x,y,16,11,16,20,1.1), BLACK, 3.0)
        add(t, ellipse(x,y,16,24,1.25,1.25), BLACK, 3.0)
    elif k == 19:
        add(t, polyline(x,y,[(19,24),(18,18),(14,14),(12,10)],1.35), BLACK, 3.0)
        add(t, tri_head(x,y,(11.2,8.8),(13.1,11.5),2.3), BLACK, 3.0)
    elif k == 20:
        add(t, polyline(x,y,[(13,24),(14,18),(18,14),(20,10)],1.35), BLACK, 3.0)
        add(t, tri_head(x,y,(20.8,8.8),(18.9,11.5),2.3), BLACK, 3.0)
    elif k == 21:
        add(t, polyline(x,y,[(14,24),(19,20),(13,15),(18,10)],1.25), BLACK, 3.0)
        add(t, tri_head(x,y,(18.7,8.7),(16.5,11.1),2.0), BLACK, 3.0)
    elif k == 22:
        add(t, polyline(x,y,[(8,22),(11,17),(14,22),(18,16),(23,22)],1.15), BLACK, 3.0)
    elif k == 23:
        draw_car(t,x,y,16,15,BLACK,2.5,False)
        add(t, polyline(x,y,[(10,23),(13,21),(15,24),(18,22),(22,24)],0.75), BLACK, 2.8)
        add(t, polyline(x,y,[(12,26),(15,24),(17,27),(20,25)],0.75), BLACK, 2.8)
    elif k == 24:
        add(t, line(x,y,11,23,11,11,1.0), BLACK, 3.0)
        add(t, polyline(x,y,[(22,23),(19,19),(19,11)],1.0), BLACK, 3.0)
    elif k == 25:
        draw_person(t,x,y,16,17,0.9,2.5,lean=-0.6)
        add(t, line(x,y,20,12,12,25,0.55), BLACK, 2.8)
        add(t, polyline(x,y,[(9,24),(15,21),(23,24)],1.1), BLACK, 2.8)
    elif k == 26:
        add(t, rect(x,y,13,9,19,24), BLACK, 2.8)
        add(t, ellipse(x,y,16,12,1.35,1.35), RED, 2.0)
        add(t, ellipse(x,y,16,16.5,1.35,1.35), YELLOW, 2.0)
        add(t, ellipse(x,y,16,21,1.35,1.35), GREEN, 2.0)
    elif k == 27:
        draw_person(t,x,y,16,18,1.05,2.7,lean=0.5)
        add(t, line(x,y,9,25,23,25,0.75), BLACK, 2.0)
    elif k == 28:
        draw_person(t,x,y,13.2,18.5,0.85,2.6,lean=0.2)
        draw_person(t,x,y,19,17.7,0.72,2.6,lean=-0.3)
    elif k == 29:
        draw_bicycle(t,x,y)
    elif k == 30:
        for ang in [0, np.pi/3, -np.pi/3]:
            dx, dy = np.cos(ang)*6.5, np.sin(ang)*6.5
            add(t, line(x,y,16-dx,17-dy,16+dx,17+dy,0.65), BLACK, 3.0)
        add(t, ellipse(x,y,16,17,1.0,1.0), BLACK, 3.0)
    elif k == 31:
        add(t, ellipse(x,y,15,16,4.7,2.2), BLACK, 2.8)
        add(t, line(x,y,18.5,14.8,22,12.2,1.0), BLACK, 2.8)
        add(t, ellipse(x,y,23,11.8,1.8,1.2), BLACK, 2.8)
        for lx in [12,15,18]:
            add(t, line(x,y,lx,18,lx-1.5,24,0.7), BLACK, 2.8)
        add(t, line(x,y,22.2,10.8,20.5,7.5,0.55), BLACK, 2.8)
        add(t, line(x,y,22.8,10.8,25.0,7.8,0.55), BLACK, 2.8)


def draw_blue_arrow(t, x, y, k):
    draw_circle_base(t,x,y,red=False,blue=True)
    if k == 33:
        add(t, arrow(x,y,(9,16),(23.5,16),1.6,4.5,3.7), WHITE, 2.6)
    elif k == 34:
        add(t, arrow(x,y,(23,16),(8.5,16),1.6,4.5,3.7), WHITE, 2.6)
    elif k == 35:
        add(t, arrow(x,y,(16,24),(16,7.5),1.7,4.8,3.8), WHITE, 2.6)
    elif k == 36:
        add(t, arrow(x,y,(15,24),(15,7.5),1.35,4.4,3.4), WHITE, 2.35)
        add(t, arrow(x,y,(15,17),(24,17),1.25,4.0,3.1), WHITE, 2.35)
    elif k == 37:
        add(t, arrow(x,y,(17,24),(17,7.5),1.35,4.4,3.4), WHITE, 2.35)
        add(t, arrow(x,y,(17,17),(8,17),1.25,4.0,3.1), WHITE, 2.35)
    elif k == 38:
        add(t, arrow(x,y,(9.2,9.2),(22.8,22.8),1.75,4.7,3.7), WHITE, 2.6)
    elif k == 39:
        add(t, arrow(x,y,(22.8,9.2),(9.2,22.8),1.75,4.7,3.7), WHITE, 2.6)
    elif k == 40:
        # three rough circular arrows
        pts1=[(16,7),(22,10),(23,16)]
        pts2=[(23,17),(18,23),(12,21)]
        pts3=[(11,21),(8,14),(12,9)]
        add(t, polyline(x,y,pts1,1.25), WHITE, 2.4); add(t, tri_head(x,y,(23,16),(21,13.8),2.6), WHITE, 2.4)
        add(t, polyline(x,y,pts2,1.25), WHITE, 2.4); add(t, tri_head(x,y,(12,21),(15,22.2),2.6), WHITE, 2.4)
        add(t, polyline(x,y,pts3,1.25), WHITE, 2.4); add(t, tri_head(x,y,(12,9),(9.6,11.4),2.6), WHITE, 2.4)


def make_template(cls, dx=0.0, dy=0.0, sc=1.0, rot=0.0, mode='norm', inner=False):
    x, y = grid(dx, dy, sc, rot)
    t = np.zeros((3,32,32), np.float32)
    if cls in [0,1,2,3,4,5,7,8]:
        nums = {0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}
        if not inner: draw_circle_base(t,x,y,red=True)
        draw_number(t,x,y,nums[cls],3.45 if inner else 2.85)
    elif cls == 6:
        if not inner: draw_cancel(t,x,y)
        draw_number(t,x,y,'80',2.4 if inner else 2.0)
        if inner: draw_cancel(t,x,y)
    elif cls == 9:
        if not inner: draw_circle_base(t,x,y,red=True)
        draw_car(t,x,y,12.5,18.0,RED,2.0,False); draw_car(t,x,y,20.0,18.2,BLACK,2.5 if inner else 2.1,False)
    elif cls == 10:
        if not inner: draw_circle_base(t,x,y,red=True)
        draw_car(t,x,y,12.0,18.0,RED,2.0,True); draw_car(t,x,y,21.0,18.2,BLACK,2.5 if inner else 2.1,False)
    elif cls == 11:
        if not inner: draw_triangle_base(t,x,y,False)
        add(t, line(x,y,16,23,16,10,1.15), BLACK, 3.4 if inner else 3.0)
        add(t, line(x,y,16,16,10,20,0.85), BLACK, 3.4 if inner else 3.0)
        add(t, line(x,y,16,16,22,20,0.85), BLACK, 3.4 if inner else 3.0)
    elif cls == 12:
        draw_priority(t,x,y)
    elif cls == 13:
        draw_triangle_base(t,x,y,True)
    elif cls == 14:
        draw_octagon(t,x,y)
    elif cls == 15:
        draw_circle_base(t,x,y,red=True)
        add(t, inside_circle(x,y,9.5), WHITE, 0.65)
    elif cls == 16:
        if not inner: draw_circle_base(t,x,y,red=True)
        draw_car(t,x,y,16,18.2,BLACK,3.1 if inner else 2.7,True)
    elif cls == 17:
        add(t, inside_circle(x,y,14.0), RED, 1.8)
        add(t, rect(x,y,7.5,13.2,24.5,18.2), WHITE, 2.3)
    elif 18 <= cls <= 31:
        if not inner: draw_triangle_base(t,x,y,False)
        warning_symbol(t,x,y,cls)
    elif cls == 32:
        draw_cancel(t,x,y)
    elif 33 <= cls <= 40:
        if inner:
            # white arrow symbol without relying on the blue disk.
            if cls == 33: add(t, arrow(x,y,(9,16),(23.5,16),1.8,4.8,3.9), WHITE, 3.0)
            elif cls == 34: add(t, arrow(x,y,(23,16),(8.5,16),1.8,4.8,3.9), WHITE, 3.0)
            elif cls == 35: add(t, arrow(x,y,(16,24),(16,7.5),1.9,5.0,4.0), WHITE, 3.0)
            elif cls == 36:
                add(t, arrow(x,y,(15,24),(15,7.5),1.45,4.4,3.5), WHITE, 2.7); add(t, arrow(x,y,(15,17),(24,17),1.35,4.0,3.2), WHITE, 2.7)
            elif cls == 37:
                add(t, arrow(x,y,(17,24),(17,7.5),1.45,4.4,3.5), WHITE, 2.7); add(t, arrow(x,y,(17,17),(8,17),1.35,4.0,3.2), WHITE, 2.7)
            elif cls == 38: add(t, arrow(x,y,(9.2,9.2),(22.8,22.8),1.9,4.9,3.9), WHITE, 3.0)
            elif cls == 39: add(t, arrow(x,y,(22.8,9.2),(9.2,22.8),1.9,4.9,3.9), WHITE, 3.0)
            elif cls == 40:
                pts1=[(16,7),(22,10),(23,16)]; pts2=[(23,17),(18,23),(12,21)]; pts3=[(11,21),(8,14),(12,9)]
                add(t, polyline(x,y,pts1,1.45), WHITE, 2.9); add(t, tri_head(x,y,(23,16),(21,13.8),2.8), WHITE, 2.9)
                add(t, polyline(x,y,pts2,1.45), WHITE, 2.9); add(t, tri_head(x,y,(12,21),(15,22.2),2.8), WHITE, 2.9)
                add(t, polyline(x,y,pts3,1.45), WHITE, 2.9); add(t, tri_head(x,y,(12,9),(9.6,11.4),2.8), WHITE, 2.9)
        else:
            draw_blue_arrow(t,x,y,cls)
    elif cls == 41:
        draw_cancel(t,x,y)
        draw_car(t,x,y,12.5,18.0,BLACK,1.7,False); draw_car(t,x,y,20.0,18.2,BLACK,1.7,False)
    elif cls == 42:
        draw_cancel(t,x,y)
        draw_car(t,x,y,16.0,18.0,BLACK,2.0,True)
    # Ignore the far border/background: signs are cropped with a little surround.
    r = np.sqrt((x-CEN)**2 + (y-CEN)**2)
    t[:, r > 15.8] *= 0.2
    if mode == 'feat':
        mean = (t[0] + t[1] + t[2]) / 3.0
        redm = np.maximum(t[0] - 0.50 * (t[1] + t[2]), 0)
        bluem = np.maximum(t[2] - 0.50 * (t[0] + t[1]), 0)
        yellowm = np.maximum(0.50 * (t[0] + t[1]) - t[2], 0)
        whitem = np.maximum(mean, 0)
        darkm = np.maximum(-mean, 0)
        f = np.stack([redm, bluem, yellowm, 0.75 * whitem, 1.35 * darkm]).astype(np.float32)
        n = np.sqrt(np.sum(f*f))
        if n > 1e-6:
            f = f / n * 5.5
        return f.reshape(-1)
    # Some banks read per-channel normalized imagery for shape contrast; others
    # read raw colour so that blue/red/yellow decisions survive normalization.
    if mode == 'raw':
        t = t * 0.018
    else:
        t = t - t.mean(axis=(1,2), keepdims=True)
        n = np.sqrt(np.sum(t*t))
        if n > 1e-6:
            t = t / n * 8.0
    return t.reshape(-1)


def build(outdir):
    # Fixed, hand-chosen pose bank: the sign is usually centred and fills most of
    # the crop, but nearest-neighbour resize and the video crop leave a little
    # translation/scale uncertainty.
    variants = []
    for sc in [0.92, 1.00, 1.08]:
        for dy in [-1.25, 0.0, 1.25]:
            for dx in [-1.25, 0.0, 1.25]:
                variants.append((dx, dy, sc, 0.0))
    rows = []
    for cls in range(43):
        for dx,dy,sc,rot in variants:
            rows.append(make_template(cls, dx, dy, sc, rot, 'norm', False))
    W = np.stack(rows).astype(np.float32)
    b = np.zeros((len(rows),), np.float32)
    vcount = len(variants)
    net = {"layers": [
        {"name":"norm", "op":"norm", "inputs":["image"]},
        {"name":"tmpl", "op":"dense", "inputs":["norm"], "out": int(W.shape[0])},
        {"name":"grid", "op":"reshape", "inputs":["tmpl"], "shape": [43, int(vcount), 1]},
        {"name":"mx", "op":"maxpool", "inputs":["grid"], "size": [int(vcount), 1], "stride": [int(vcount), 1]},
        {"name":"base", "op":"reshape", "inputs":["mx"], "shape": [43]},
        {"name":"col", "op":"conv", "inputs":["image"], "out": 4, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name":"cpos", "op":"relu", "inputs":["col"]},
        {"name":"fam", "op":"dense", "inputs":["cpos"], "out": 7},
        {"name":"fpos", "op":"relu", "inputs":["fam"]},
        {"name":"fs", "op":"dense", "inputs":["fpos"], "out": 43},
        {"name":"scores", "op":"add", "inputs":["base", "fs"]}
    ]}
    # Simple colour-family features from raw RGB: red, blue, dark, and bright.
    Wcol = np.array([[1.0,-0.55,-0.55],[-0.55,-0.55,1.0],[-0.8,-0.8,-0.8],[0.55,0.55,0.55]], np.float32).reshape(4,3,1,1)
    bcol = np.array([-0.08,-0.08,0.62,-0.38], np.float32)
    yy, xx = np.mgrid[0:32,0:32].astype(np.float32)
    rr = np.sqrt((xx-CEN)**2 + (yy-CEN)**2)
    outer = ((rr > 9.5) & (rr < 15.2)).astype(np.float32)
    centre = (rr < 10.5).astype(np.float32)
    allm = (rr < 15.5).astype(np.float32)
    Wfam = np.zeros((7, 4*32*32), np.float32)
    def put(row, ch, mask, val): Wfam[row, ch*32*32:(ch+1)*32*32] = (mask.reshape(-1) * val).astype(np.float32)
    put(0,0,outer, 0.016)          # red-ring warning/prohibition/speed signs
    put(1,1,allm, 0.018)           # blue mandatory signs
    put(2,2,allm, 0.014)           # dark cancellation signs
    put(3,3,centre,0.012)          # pale interior signs
    put(4,3,allm, 0.010); put(4,0,allm,-0.010); put(4,1,allm,-0.010)  # white/black cancellation
    put(5,0,allm, 0.010); put(5,3,allm,0.006)  # red filled stop/no-entry
    put(6,1,allm, 0.012); put(6,3,centre,0.010) # blue+white arrows
    bfam = np.array([-0.08,-0.10,-0.16,-0.05,-0.10,-0.10,-0.10], np.float32)
    Wfs = np.zeros((43,7), np.float32)
    red_classes = list(range(0,11))+[13,15,16,17]+list(range(18,32))
    for k in red_classes: Wfs[k,0] += 0.18
    for k in range(33,41): Wfs[k,1] += 0.28; Wfs[k,6] += 0.18
    for k in [6,32,41,42]: Wfs[k,2] += 0.34; Wfs[k,4] += 0.20
    for k in [12]: Wfs[k,3] += 0.18
    for k in [14,17]: Wfs[k,5] += 0.16
    bofs = np.zeros(43, np.float32)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'network.json'), 'w') as f:
        json.dump(net, f, separators=(',', ':'))
    np.savez(os.path.join(outdir, 'weights.npz'), **{
        'tmpl.W': W, 'tmpl.b': b, 'col.W': Wcol, 'col.b': bcol,
        'fam.W': Wfam, 'fam.b': bfam, 'fs.W': Wfs, 'fs.b': bofs})


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '/app/out')
