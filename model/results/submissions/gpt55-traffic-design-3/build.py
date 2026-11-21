import sys, os, json
import numpy as np

K = 43
V = 72
D = 32
N = 96
F = N // D
CFEAT = 5

RED = np.array([1.25, -0.55, -0.55], np.float32)
BLUE = np.array([-0.45, -0.15, 1.25], np.float32)
WHITE = np.array([0.72, 0.72, 0.72], np.float32)
BLACK = np.array([-1.0, -1.0, -1.0], np.float32)
YELLOW = np.array([1.05, 0.86, -0.55], np.float32)
GREY = np.array([-0.12, -0.12, -0.12], np.float32)

_py, _px = np.mgrid[0:N, 0:N].astype(np.float32)
PX = (_px + 0.5) * D / N - 0.5
PY = (_py + 0.5) * D / N - 0.5


def coords(cx, cy, rad, rot=0.0):
    x = (PX - cx) / rad
    y = (PY - cy) / rad
    if rot:
        c, s = np.cos(rot), np.sin(rot)
        return c * x + s * y, -s * x + c * y
    return x, y


def paint(img, mask, color, strength=1.0):
    if mask.dtype != bool:
        mask = mask > 0.5
    img[:, mask] = (color * strength)[:, None]


def circle(x, y, r=1.0):
    return x * x + y * y <= r * r


def ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def rect(x, y, x1, x2, y1, y2):
    return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)


def line(x, y, x1, y1, x2, y2, t):
    vx, vy = x2 - x1, y2 - y1
    den = vx * vx + vy * vy + 1e-9
    u = ((x - x1) * vx + (y - y1) * vy) / den
    u = np.clip(u, 0.0, 1.0)
    dx = x - (x1 + u * vx)
    dy = y - (y1 + u * vy)
    return dx * dx + dy * dy <= t * t


def polyline(x, y, pts, t):
    m = np.zeros_like(x, dtype=bool)
    for a, b in zip(pts[:-1], pts[1:]):
        m |= line(x, y, a[0], a[1], b[0], b[1], t)
    return m


def polygon(x, y, pts):
    pts = list(pts)
    inside = np.zeros_like(x, dtype=bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        inside ^= cross
        j = i
    return inside


def tri(x, y, inv=False, scale=1.0):
    yy = -y if inv else y
    xx = x
    top, bot, hw = -0.98 * scale, 0.86 * scale, 1.03 * scale
    half = (yy - top) / (bot - top) * hw
    return (yy >= top) & (yy <= bot) & (np.abs(xx) <= half)


def octagon(x, y, s=1.0):
    ax, ay = np.abs(x), np.abs(y)
    return (ax <= 0.92 * s) & (ay <= 0.92 * s) & (ax + ay <= 1.28 * s)


def diamond(x, y, s=1.0):
    return np.abs(x) + np.abs(y) <= s


def down(img):
    return img.reshape(3, D, F, D, F).mean(axis=(2, 4)).astype(np.float32)


def arrow_head(x, y, tip, ang, length=0.28, width=0.32):
    tx, ty = tip
    ux, uy = np.cos(ang), np.sin(ang)
    bx, by = tx - ux * length, ty - uy * length
    px, py = -uy, ux
    pts = [(tx, ty), (bx + px * width / 2, by + py * width / 2), (bx - px * width / 2, by - py * width / 2)]
    return polygon(x, y, pts)


def arrow_poly(x, y, pts, t=0.105, head=True):
    m = polyline(x, y, pts, t)
    if head and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        m |= arrow_head(x, y, (x2, y2), np.arctan2(y2 - y1, x2 - x1), max(0.22, 2.4 * t), max(0.30, 3.1 * t))
    return m


def draw_digit_mask(x, y, ch, cx, cy, w, h, thick, style=0):
    if style:
        # Slight slant and bolder strokes give a second, still generic, road-font variant.
        x = x + 0.10 * (y - cy)
        thick *= 1.18
    m = np.zeros_like(x, dtype=bool)
    if ch == '0':
        outer = ellipse(x, y, cx, cy, w / 2, h / 2)
        inner = ellipse(x, y, cx, cy, w / 2 - thick * 1.1, h / 2 - thick * 1.1)
        return outer & ~inner
    if ch == '1':
        return line(x, y, cx + w * 0.12, cy - h / 2, cx + w * 0.12, cy + h / 2, thick)
    segs = {
        '2': 'abged', '3': 'abgcd', '4': 'fgbc', '5': 'afgcd',
        '6': 'afgecd', '7': 'abc', '8': 'abcdefg', '9': 'abfgcd'
    }.get(ch, '')
    left, right = cx - w / 2, cx + w / 2
    top, mid, bot = cy - h / 2, cy, cy + h / 2
    if 'a' in segs: m |= line(x, y, left + thick, top, right - thick, top, thick)
    if 'b' in segs: m |= line(x, y, right, top + thick, right, mid - thick, thick)
    if 'c' in segs: m |= line(x, y, right, mid + thick, right, bot - thick, thick)
    if 'd' in segs: m |= line(x, y, left + thick, bot, right - thick, bot, thick)
    if 'e' in segs: m |= line(x, y, left, mid + thick, left, bot - thick, thick)
    if 'f' in segs: m |= line(x, y, left, top + thick, left, mid - thick, thick)
    if 'g' in segs: m |= line(x, y, left + thick, mid, right - thick, mid, thick)
    return m


def draw_digits(img, x, y, text, color=BLACK, strength=2.1, style=0):
    n = len(text)
    if n == 2:
        centers = [-0.26, 0.26]
        w, h, t = 0.34, 0.62, 0.040
    else:
        centers = [-0.42, 0.00, 0.42]
        w, h, t = 0.27, 0.58, 0.034
    m = np.zeros_like(x, dtype=bool)
    for ch, cx in zip(text, centers):
        m |= draw_digit_mask(x, y, ch, cx, 0.05, w, h, t, style)
    paint(img, m, color, strength)


def draw_stop_text(img, x, y):
    c = WHITE; t = 0.035; sy = 0.05; h = 0.32; w = 0.18
    starts = [-0.43, -0.16, 0.10, 0.34]
    # S
    x0 = starts[0]; m = line(x,y,x0-w/2,sy-h/2,x0+w/2,sy-h/2,t)|line(x,y,x0-w/2,sy,x0+w/2,sy,t)|line(x,y,x0-w/2,sy+h/2,x0+w/2,sy+h/2,t)|line(x,y,x0-w/2,sy-h/2,x0-w/2,sy,t)|line(x,y,x0+w/2,sy,x0+w/2,sy+h/2,t)
    paint(img,m,c,1.9)
    # T
    x0 = starts[1]; m = line(x,y,x0-w/2,sy-h/2,x0+w/2,sy-h/2,t)|line(x,y,x0,sy-h/2,x0,sy+h/2,t); paint(img,m,c,1.9)
    # O
    x0 = starts[2]; m = ellipse(x,y,x0,sy,w/2,h/2)&~ellipse(x,y,x0,sy,w/2-t*1.4,h/2-t*1.4); paint(img,m,c,1.9)
    # P
    x0 = starts[3]; m = line(x,y,x0-w/2,sy-h/2,x0-w/2,sy+h/2,t)|line(x,y,x0-w/2,sy-h/2,x0+w/2,sy-h/2,t)|line(x,y,x0-w/2,sy,x0+w/2,sy,t)|line(x,y,x0+w/2,sy-h/2,x0+w/2,sy,t); paint(img,m,c,1.9)


def draw_car(img, x, y, cx, cy, sc=1.0, col=BLACK, strength=1.8):
    m = rect(x, y, cx - 0.28*sc, cx + 0.28*sc, cy - 0.04*sc, cy + 0.12*sc)
    m |= polygon(x, y, [(cx-0.18*sc,cy-0.04*sc),(cx-0.08*sc,cy-0.18*sc),(cx+0.13*sc,cy-0.18*sc),(cx+0.23*sc,cy-0.04*sc)])
    paint(img, m, col, strength)
    wheels = ellipse(x,y,cx-0.17*sc,cy+0.14*sc,0.055*sc,0.055*sc)|ellipse(x,y,cx+0.17*sc,cy+0.14*sc,0.055*sc,0.055*sc)
    paint(img, wheels, BLACK, max(1.6, strength))


def draw_truck(img, x, y, cx, cy, sc=1.0, col=BLACK, strength=1.8):
    m = rect(x,y,cx-0.32*sc,cx+0.08*sc,cy-0.12*sc,cy+0.10*sc)|rect(x,y,cx+0.08*sc,cx+0.28*sc,cy-0.04*sc,cy+0.10*sc)
    paint(img,m,col,strength)
    wheels = ellipse(x,y,cx-0.18*sc,cy+0.13*sc,0.055*sc,0.055*sc)|ellipse(x,y,cx+0.17*sc,cy+0.13*sc,0.055*sc,0.055*sc)
    paint(img,wheels,BLACK,max(1.6,strength))


def stick_person(img, x, y, cx, cy, sc=1.0, col=BLACK, strength=1.9):
    m = ellipse(x,y,cx,cy-0.22*sc,0.06*sc,0.06*sc)
    m |= line(x,y,cx,cy-0.15*sc,cx,cy+0.08*sc,0.035*sc)
    m |= line(x,y,cx,cy-0.02*sc,cx-0.15*sc,cy+0.04*sc,0.030*sc)
    m |= line(x,y,cx,cy-0.02*sc,cx+0.15*sc,cy+0.02*sc,0.030*sc)
    m |= line(x,y,cx,cy+0.08*sc,cx-0.13*sc,cy+0.30*sc,0.032*sc)
    m |= line(x,y,cx,cy+0.08*sc,cx+0.15*sc,cy+0.30*sc,0.032*sc)
    paint(img,m,col,strength)


def warning_symbol(img, x, y, cls, style=0):
    t = 0.045 * (1.18 if style else 1.0)
    c = BLACK
    if cls == 11:
        m = line(x,y,0,-0.55,0,0.55,0.070)|line(x,y,-0.45,-0.12,0,0.02,0.050)|line(x,y,0,0.02,0.45,-0.12,0.050)
        paint(img,m,c,2.0)
    elif cls == 18:
        paint(img, line(x,y,0,-0.35,0,0.20,0.055)|ellipse(x,y,0,0.40,0.07,0.07), c, 2.2)
    elif cls == 19:
        pts=[(0.25,0.48),(0.10,0.15),(-0.16,-0.05),(-0.30,-0.42)]
        paint(img, polyline(x,y,pts,0.075), c, 2.0)
    elif cls == 20:
        pts=[(-0.25,0.48),(-0.10,0.15),(0.16,-0.05),(0.30,-0.42)]
        paint(img, polyline(x,y,pts,0.075), c, 2.0)
    elif cls == 21:
        pts=[(-0.25,0.52),(0.10,0.25),(0.23,0.02),(-0.10,-0.18),(-0.25,-0.42)]
        paint(img, polyline(x,y,pts,0.070), c, 2.0)
    elif cls == 22:
        pts=[(-0.55,0.28),(-0.35,0.02),(-0.15,0.28),(0.10,0.02),(0.35,0.28),(0.55,0.12)]
        paint(img, polyline(x,y,pts,0.070), c, 2.0)
    elif cls == 23:
        draw_car(img,x,y,0,-0.08,0.78,c,1.7)
        paint(img, polyline(x,y,[(-0.35,0.25),(-0.10,0.18),(0.15,0.27)],0.030)|polyline(x,y,[(-0.25,0.40),(0.04,0.32),(0.34,0.42)],0.030), c, 2.0)
    elif cls == 24:
        m = line(x,y,-0.28,-0.50,-0.28,0.55,0.055)|line(x,y,0.36,-0.50,0.12,0.55,0.055)
        paint(img,m,c,2.0)
    elif cls == 25:
        stick_person(img,x,y,-0.03,-0.02,0.85,c,1.9)
        paint(img, line(x,y,0.23,-0.28,-0.35,0.45,0.032)|polygon(x,y,[(0.24,-0.31),(0.34,-0.20),(0.18,-0.17)]), c, 2.0)
    elif cls == 26:
        m = rect(x,y,-0.16,0.16,-0.48,0.45)
        paint(img,m,c,1.7)
        paint(img, ellipse(x,y,0,-0.27,0.065,0.065), RED, 1.4)
        paint(img, ellipse(x,y,0,0.00,0.065,0.065), YELLOW, 1.2)
        paint(img, ellipse(x,y,0,0.27,0.065,0.065), np.array([-0.5,1.0,-0.5],np.float32), 1.2)
    elif cls == 27:
        stick_person(img,x,y,0,-0.03,0.90,c,2.0)
        paint(img, line(x,y,-0.42,0.45,0.35,0.28,0.025), c, 1.5)
    elif cls == 28:
        stick_person(img,x,y,-0.13,-0.02,0.75,c,1.9)
        stick_person(img,x,y,0.17,0.05,0.60,c,1.9)
    elif cls == 29:
        m = ellipse(x,y,-0.25,0.24,0.16,0.16)&~ellipse(x,y,-0.25,0.24,0.12,0.12)
        m |= ellipse(x,y,0.25,0.24,0.16,0.16)&~ellipse(x,y,0.25,0.24,0.12,0.12)
        m |= line(x,y,-0.25,0.24,0.00,-0.02,0.028)|line(x,y,0.00,-0.02,0.25,0.24,0.028)|line(x,y,-0.25,0.24,0.10,0.24,0.028)|line(x,y,0.00,-0.02,0.10,0.24,0.028)
        m |= line(x,y,0.00,-0.02,0.05,-0.18,0.028)|line(x,y,0.04,-0.18,0.18,-0.18,0.028)
        paint(img,m,c,2.0)
    elif cls == 30:
        m = np.zeros_like(x,dtype=bool)
        for ang in [0, np.pi/3, -np.pi/3]:
            m |= line(x,y,-0.38*np.cos(ang),-0.38*np.sin(ang),0.38*np.cos(ang),0.38*np.sin(ang),0.025)
        paint(img,m,c,2.0)
    elif cls == 31:
        m = ellipse(x,y,-0.05,0.02,0.28,0.12)|ellipse(x,y,0.22,-0.10,0.11,0.08)
        m |= line(x,y,-0.22,0.10,-0.30,0.43,0.030)|line(x,y,-0.02,0.11,-0.04,0.43,0.030)|line(x,y,0.12,0.08,0.20,0.40,0.030)
        m |= line(x,y,0.29,-0.15,0.43,-0.32,0.025)|line(x,y,0.29,-0.15,0.38,-0.02,0.025)
        paint(img,m,c,2.0)


def circle_red_white(img, x, y, inner=0.72):
    paint(img, circle(x,y,1.0), RED, 0.95)
    paint(img, circle(x,y,inner), WHITE, 0.25)


def sign_template(cls, rad, dx, dy, style):
    cx, cy = 15.5 + dx, 15.5 + dy
    img = np.zeros((3, N, N), np.float32)
    x, y = coords(cx, cy, rad, 0.0)

    speed = {0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}
    if cls in speed:
        circle_red_white(img,x,y,0.70)
        draw_digits(img,x,y,speed[cls],BLACK,2.25,style)
    elif cls == 6:
        paint(img,circle(x,y,1.0),WHITE,0.34)
        paint(img,circle(x,y,0.92)&~circle(x,y,0.78),GREY,0.9)
        draw_digits(img,x,y,'80',GREY,1.25,style)
        for off in [-0.42,-0.16,0.10,0.36]:
            paint(img,line(x,y,off-0.38,0.72,off+0.34,-0.72,0.040),BLACK,1.7)
    elif cls == 9:
        circle_red_white(img,x,y,0.73)
        draw_car(img,x,y,-0.23,0.02,0.70,BLACK,1.7)
        draw_car(img,x,y,0.27,0.02,0.70,RED,1.4)
    elif cls == 10:
        circle_red_white(img,x,y,0.73)
        draw_car(img,x,y,-0.27,0.04,0.62,BLACK,1.7)
        draw_truck(img,x,y,0.24,0.04,0.62,RED,1.5)
    elif cls == 15:
        circle_red_white(img,x,y,0.73)
    elif cls == 16:
        circle_red_white(img,x,y,0.73)
        draw_truck(img,x,y,0.00,0.04,0.95,BLACK,2.0)
    elif cls == 17:
        paint(img,circle(x,y,1.0),RED,1.05)
        paint(img,rect(x,y,-0.62,0.62,-0.15,0.15)&circle(x,y,0.95),WHITE,1.3)
    elif cls in [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]:
        paint(img, tri(x,y,False,1.0), RED, 0.95)
        paint(img, tri(x,y,False,0.70), WHITE, 0.18)
        warning_symbol(img,x,y,cls,style)
    elif cls == 12:
        paint(img, diamond(x,y,1.02), WHITE, 0.8)
        paint(img, diamond(x,y,0.72), YELLOW, 1.25)
    elif cls == 13:
        paint(img, tri(x,y,True,1.0), RED, 1.1)
        paint(img, tri(x,y,True,0.68), WHITE, 0.35)
    elif cls == 14:
        paint(img, octagon(x,y,1.0), RED, 1.1)
        paint(img, octagon(x,y,0.78), RED, 1.25)
        paint(img, octagon(x,y,0.94)&~octagon(x,y,0.80), WHITE, 0.9)
        draw_stop_text(img,x,y)
    elif cls == 32:
        paint(img,circle(x,y,1.0),WHITE,0.32)
        paint(img,circle(x,y,0.94)&~circle(x,y,0.82),GREY,0.8)
        for off in [-0.46,-0.18,0.10,0.38]:
            paint(img,line(x,y,off-0.42,0.74,off+0.36,-0.74,0.045),BLACK,1.8)
    elif 33 <= cls <= 40:
        paint(img,circle(x,y,1.0),BLUE,1.05)
        if cls == 33:
            m = arrow_poly(x,y,[(-0.25,0.55),(-0.25,-0.20),(0.55,-0.20)],0.090,True)
        elif cls == 34:
            m = arrow_poly(x,y,[(0.25,0.55),(0.25,-0.20),(-0.55,-0.20)],0.090,True)
        elif cls == 35:
            m = arrow_poly(x,y,[(0.0,0.55),(0.0,-0.62)],0.105,True)
        elif cls == 36:
            m = arrow_poly(x,y,[(0.0,0.56),(0.0,-0.62)],0.085,True) | arrow_poly(x,y,[(0.0,-0.05),(0.55,-0.05)],0.085,True)
        elif cls == 37:
            m = arrow_poly(x,y,[(0.0,0.56),(0.0,-0.62)],0.085,True) | arrow_poly(x,y,[(0.0,-0.05),(-0.55,-0.05)],0.085,True)
        elif cls == 38:
            m = arrow_poly(x,y,[(-0.45,-0.45),(0.48,0.48)],0.115,True)
        elif cls == 39:
            m = arrow_poly(x,y,[(0.45,-0.45),(-0.48,0.48)],0.115,True)
        else:
            m = np.zeros_like(x,dtype=bool)
            for a0,a1 in [(-2.8,-1.05),(-0.70,0.95),(1.35,3.10)]:
                ts = np.linspace(a0,a1,18)
                pts=[(0.43*np.cos(a),0.43*np.sin(a)) for a in ts]
                m |= polyline(x,y,pts,0.060)
                m |= arrow_head(x,y,pts[-1],a1+np.pi/2,0.18,0.22)
        paint(img,m,WHITE,2.15)
    elif cls in [41,42]:
        paint(img,circle(x,y,1.0),WHITE,0.30)
        if cls == 41:
            draw_car(img,x,y,-0.20,0.05,0.65,GREY,1.0); draw_car(img,x,y,0.24,0.05,0.65,GREY,1.0)
        else:
            draw_car(img,x,y,-0.25,0.05,0.58,GREY,1.0); draw_truck(img,x,y,0.22,0.05,0.58,GREY,1.0)
        for off in [-0.35,-0.08,0.20]:
            paint(img,line(x,y,off-0.40,0.70,off+0.35,-0.70,0.045),BLACK,1.8)
    return down(img)


def symbol_template(cls, rad, dx, dy, style):
    cx, cy = 15.5 + dx, 15.5 + dy
    img = np.zeros((3, N, N), np.float32)
    x, y = coords(cx, cy, rad, 0.0)
    speed = {0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}
    if cls in speed:
        draw_digits(img,x,y,speed[cls],BLACK,3.5,style)
    elif cls == 6:
        draw_digits(img,x,y,'80',BLACK,1.8,style)
        for off in [-0.42,-0.16,0.10,0.36]:
            paint(img,line(x,y,off-0.38,0.72,off+0.34,-0.72,0.045),BLACK,2.2)
    elif cls == 9:
        draw_car(img,x,y,-0.23,0.02,0.70,BLACK,2.1)
        draw_car(img,x,y,0.27,0.02,0.70,RED,1.6)
    elif cls == 10:
        draw_car(img,x,y,-0.27,0.04,0.62,BLACK,2.0)
        draw_truck(img,x,y,0.24,0.04,0.62,RED,1.8)
    elif cls == 16:
        draw_truck(img,x,y,0.00,0.04,0.95,BLACK,2.4)
    elif cls == 17:
        paint(img,rect(x,y,-0.62,0.62,-0.15,0.15)&circle(x,y,0.95),WHITE,2.0)
    elif cls in [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]:
        warning_symbol(img,x,y,cls,style)
    elif cls == 12:
        paint(img, diamond(x,y,0.78), YELLOW, 1.0)
    elif cls == 13:
        paint(img, tri(x,y,True,1.0)&~tri(x,y,True,0.65), RED, 1.4)
    elif cls == 14:
        draw_stop_text(img,x,y)
    elif cls == 15:
        paint(img, circle(x,y,1.0)&~circle(x,y,0.72), RED, 1.2)
    elif cls == 32:
        for off in [-0.46,-0.18,0.10,0.38]:
            paint(img,line(x,y,off-0.42,0.74,off+0.36,-0.74,0.050),BLACK,2.2)
    elif 33 <= cls <= 40:
        if cls == 33:
            m = arrow_poly(x,y,[(-0.25,0.55),(-0.25,-0.20),(0.55,-0.20)],0.105,True)
        elif cls == 34:
            m = arrow_poly(x,y,[(0.25,0.55),(0.25,-0.20),(-0.55,-0.20)],0.105,True)
        elif cls == 35:
            m = arrow_poly(x,y,[(0.0,0.55),(0.0,-0.62)],0.120,True)
        elif cls == 36:
            m = arrow_poly(x,y,[(0.0,0.56),(0.0,-0.62)],0.095,True) | arrow_poly(x,y,[(0.0,-0.05),(0.55,-0.05)],0.095,True)
        elif cls == 37:
            m = arrow_poly(x,y,[(0.0,0.56),(0.0,-0.62)],0.095,True) | arrow_poly(x,y,[(0.0,-0.05),(-0.55,-0.05)],0.095,True)
        elif cls == 38:
            m = arrow_poly(x,y,[(-0.45,-0.45),(0.48,0.48)],0.125,True)
        elif cls == 39:
            m = arrow_poly(x,y,[(0.45,-0.45),(-0.48,0.48)],0.125,True)
        else:
            m = np.zeros_like(x,dtype=bool)
            for a0,a1 in [(-2.8,-1.05),(-0.70,0.95),(1.35,3.10)]:
                ts = np.linspace(a0,a1,18)
                pts=[(0.43*np.cos(a),0.43*np.sin(a)) for a in ts]
                m |= polyline(x,y,pts,0.070)
                m |= arrow_head(x,y,pts[-1],a1+np.pi/2,0.20,0.25)
        paint(img,m,WHITE,2.5)
    elif cls in [41,42]:
        if cls == 41:
            draw_car(img,x,y,-0.20,0.05,0.65,BLACK,1.4); draw_car(img,x,y,0.24,0.05,0.65,BLACK,1.4)
        else:
            draw_car(img,x,y,-0.25,0.05,0.58,BLACK,1.4); draw_truck(img,x,y,0.22,0.05,0.58,BLACK,1.4)
        for off in [-0.35,-0.08,0.20]:
            paint(img,line(x,y,off-0.40,0.70,off+0.35,-0.70,0.050),BLACK,2.2)
    return down(img)


def norm_weight(w, scale):
    w = w.astype(np.float32).copy()
    w -= w.mean(axis=(1,2), keepdims=True)
    n = np.sqrt(np.sum(w*w)) + 1e-6
    return (w * (scale / n)).reshape(-1)


def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    variants = []
    for rad in [10.8, 12.0, 13.1, 14.2]:
        for dy in [-1.5, 0.0, 1.5]:
            for dx in [-1.5, 0.0, 1.5]:
                for style in [0, 1]:
                    variants.append((rad, dx, dy, style))
    assert len(variants) == V
    T = K * V
    Wf = np.zeros((T, 3*32*32), np.float32)
    bf = np.zeros(T, np.float32)
    Wl = np.zeros((T, 3*16*16), np.float32)
    bl = np.zeros(T, np.float32)
    for c in range(K):
        for i, v in enumerate(variants):
            tmpl = sign_template(c, *v)
            Wf[c*V+i] = norm_weight(tmpl, 7.0)
            lo = tmpl.reshape(3,16,2,16,2).mean(axis=(2,4))
            Wl[c*V+i] = norm_weight(lo, 3.0)
    class_bias = np.zeros(K, np.float32)
    # Generic rings/slashes/diamonds have little interior structure and should be
    # conservative; distinctive red, triangular and blue-arrow signs get small priors.
    for c in [0,1,2,3,4,5,7,8,9,10,14,16,17]:
        class_bias[c] += 0.45
    for c in [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]:
        class_bias[c] += 0.10
    for c in [33,34,35,36,37,38,39]:
        class_bias[c] += 0.20
    class_bias[6] = -1.80
    class_bias[12] = -2.20
    class_bias[15] = -1.80
    class_bias[32] = -1.30
    class_bias[40] = -1.00
    class_bias[41] = -1.10
    class_bias[42] = -1.10
    Wo = np.eye(K, dtype=np.float32)
    bo = class_bias
    net = {"layers": [
        {"name":"n", "op":"norm", "inputs":["image"]},
        {"name":"a", "op":"avgpool", "inputs":["n"], "size":[2,2], "stride":[2,2]},
        {"name":"tf", "op":"dense", "inputs":["n"], "out":T},
        {"name":"tl", "op":"dense", "inputs":["a"], "out":T},
        {"name":"t", "op":"add", "inputs":["tf","tl"]},
        {"name":"r", "op":"reshape", "inputs":["t"], "shape":[K,1,V]},
        {"name":"m", "op":"maxpool", "inputs":["r"], "size":[1,V], "stride":[1,V]},
        {"name":"f", "op":"reshape", "inputs":["m"], "shape":[K]},
        {"name":"out", "op":"dense", "inputs":["f"], "out":K}
    ]}
    with open(os.path.join(outdir, 'network.json'), 'w') as f:
        json.dump(net, f, separators=(',', ':'))
    np.savez(os.path.join(outdir, 'weights.npz'), **{
        'tf.W': Wf, 'tf.b': bf, 'tl.W': Wl, 'tl.b': bl, 'out.W': Wo, 'out.b': bo
    })

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '/app/out')
