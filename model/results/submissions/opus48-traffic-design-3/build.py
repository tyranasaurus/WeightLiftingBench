import numpy as np, json, os, sys

# ===== draw.py =====
"""Procedural drawing of German traffic-sign templates (general knowledge only).
Renders at high resolution with supersampling, returns 32x32 RGB float arrays."""

SS = 4                 # supersample factor
R = 32 * SS            # hi-res canvas
# pixel-center coordinates in 32-space
_lin = (np.arange(R) + 0.5) / SS
YY, XX = np.meshgrid(_lin, _lin, indexing='ij')   # each (R,R) in [0,32]
CX = CY = 16.0

# colours (approximate, from general knowledge of the signs)
RED   = (0.72, 0.08, 0.10)
WHITE = (0.98, 0.98, 0.98)
BLACK = (0.10, 0.10, 0.10)
BLUE  = (0.11, 0.21, 0.62)
YELLOW= (0.96, 0.75, 0.08)
GRAY  = (0.60, 0.60, 0.60)   # neutral sign plate (derestriction)
DGRAY = (0.30, 0.30, 0.30)

def canvas(color=(0,0,0)):
    c = np.zeros((R, R, 3), np.float32)
    c[:] = color
    return c

def _clip01(a): return np.clip(a, 0.0, 1.0)

def disc_mask(cx, cy, r):
    d = np.sqrt((XX-cx)**2 + (YY-cy)**2)
    # soft edge ~0.6 px
    return _clip01((r - d) * SS * 1.2 + 0.5)

def ring_mask(cx, cy, r_out, r_in):
    return np.clip(disc_mask(cx,cy,r_out) - disc_mask(cx,cy,r_in), 0, 1)

def line_mask(p0, p1, width):
    """thick segment with round caps, 'width' is full width in 32-space."""
    x0,y0 = p0; x1,y1 = p1
    vx,vy = x1-x0, y1-y0
    L2 = vx*vx+vy*vy + 1e-9
    t = ((XX-x0)*vx + (YY-y0)*vy)/L2
    t = np.clip(t,0,1)
    px = x0+t*vx; py=y0+t*vy
    d = np.sqrt((XX-px)**2+(YY-py)**2)
    return _clip01((width/2 - d)*SS*1.2 + 0.5)

def arc_mask(cx, cy, r, a0, a1, width, n=60):
    """thick arc from angle a0 to a1 (radians), approximated by segments."""
    angs = np.linspace(a0, a1, n)
    pts = [(cx+r*np.cos(a), cy+r*np.sin(a)) for a in angs]
    m = np.zeros((R,R), np.float32)
    for i in range(len(pts)-1):
        m = np.maximum(m, line_mask(pts[i], pts[i+1], width))
    return m

def poly_mask(points):
    """filled convex-ish polygon via even-odd / winding using half-plane for convex.
    points: list of (x,y) in order. Uses crossing-number test (works for simple polygons)."""
    px = np.array([p[0] for p in points]); py = np.array([p[1] for p in points])
    n = len(points)
    inside = np.zeros((R,R), bool)
    j = n-1
    for i in range(n):
        cond = ((py[i]>YY) != (py[j]>YY)) & \
               (XX < (px[j]-px[i])*(YY-py[i])/(py[j]-py[i]+1e-12) + px[i])
        inside ^= cond
        j = i
    # light AA by small supersample already via SS grid
    return inside.astype(np.float32)

def paint(cv, mask, color):
    m = mask[...,None]
    col = np.array(color, np.float32)[None,None,:]
    cv[:] = cv*(1-m) + col*m
    return cv

def downsample(cv):
    """average SSxSS blocks -> 32x32x3 ; this is a within-one-image operation (antialiasing)."""
    return cv.reshape(32, SS, 32, SS, 3).mean(axis=(1,3))

def tri_up(cx, cy, half, h):
    """equilateral-ish triangle pointing up; returns the three vertices."""
    return [(cx, cy-h*2/3), (cx-half, cy+h/3), (cx+half, cy+h/3)]

def tri_down(cx, cy, half, h):
    return [(cx, cy+h*2/3), (cx-half, cy-h/3), (cx+half, cy-h/3)]


# ===== symbols.py =====
"""Digits and pictograms drawn from general knowledge, as ink masks on the hi-res grid."""

def ellipse_ring(cx, cy, rx, ry, th):
    d = np.sqrt(((XX-cx)/rx)**2 + ((YY-cy)/ry)**2)
    outer = _clip01((1 - d)*SS*rx*1.0 + 0.5)
    d2 = np.sqrt(((XX-cx)/(rx-th))**2 + ((YY-cy)/(ry-th))**2)
    inner = _clip01((1 - d2)*SS*(rx-th)*1.0 + 0.5)
    return np.clip(outer-inner,0,1)

SEG = {
 '0':'abcdef','1':'','2':'abged','3':'abgcd','4':'fgbc',
 '5':'afgcd','6':'afgcde','7':'abc','8':'abcdefg','9':'abcdfg'}

def draw_digit(ch, cx, cy, w, h, th):
    L=cx-w/2; Rt=cx+w/2; T=cy-h/2; B=cy+h/2; M=cy
    m=np.zeros((R,R),np.float32)
    if ch=='1':
        m=np.maximum(m, line_mask((cx, T),(cx, B), th))
        m=np.maximum(m, line_mask((cx-w*0.35, T+h*0.18),(cx, T), th))
        return m
    if ch=='0':
        return ellipse_ring(cx,cy,w/2,h/2,th)
    segs=SEG[ch]
    S={'a':((L,T),(Rt,T)),'b':((Rt,T),(Rt,M)),'c':((Rt,M),(Rt,B)),
       'd':((L,B),(Rt,B)),'e':((L,M),(L,B)),'f':((L,T),(L,M)),'g':((L,M),(Rt,M))}
    for s in segs:
        p0,p1=S[s]; m=np.maximum(m, line_mask(p0,p1,th))
    return m

def draw_number(s, cx, cy, totalw, h, th):
    n=len(s); gap=totalw*0.12
    dw=(totalw-gap*(n-1))/n
    m=np.zeros((R,R),np.float32)
    x0=cx-totalw/2+dw/2
    for i,ch in enumerate(s):
        x=x0+i*(dw+gap)
        m=np.maximum(m, draw_digit(ch,x,cy,dw*0.82,h,th))
    return m

# ---- pictograms (schematic, from general knowledge) ----
def sym_exclaim(cx,cy,h):      # 18 danger "!"
    m=line_mask((cx,cy-h*0.5),(cx,cy+h*0.12), h*0.16)
    m=np.maximum(m, disc_mask(cx,cy+h*0.42, h*0.11))
    return m

def sym_bend(cx,cy,h,right=True):   # 19/20 curve
    s=1 if right else -1
    # vertical stem then hook turning to the side at top
    m=line_mask((cx-0.0,cy+h*0.5),(cx,cy-h*0.1), h*0.14)
    m=np.maximum(m, arc_mask(cx+s*h*0.28, cy-h*0.1, h*0.28, np.pi if right else 0, np.pi*1.5 if right else -np.pi*0.5, h*0.14))
    # arrow head
    tipx=cx+s*h*0.28; tipy=cy-h*0.38
    m=np.maximum(m, line_mask((tipx,tipy),(tipx-s*h*0.16,tipy+h*0.02),h*0.13))
    m=np.maximum(m, line_mask((tipx,tipy),(tipx-s*h*0.02,tipy+h*0.18),h*0.13))
    return m

def sym_double_bend(cx,cy,h,right_first=True):  # 21
    s=1 if right_first else -1
    m=arc_mask(cx+s*h*0.2, cy-h*0.1, h*0.22, 0, np.pi, h*0.12)
    m=np.maximum(m, arc_mask(cx-s*h*0.2, cy+h*0.25, h*0.22, np.pi, 2*np.pi, h*0.12))
    m=np.maximum(m, line_mask((cx,cy+h*0.47),(cx,cy+h*0.2),h*0.12))
    return m

def sym_bumps(cx,cy,h):   # 22 bumpy
    m=arc_mask(cx-h*0.22, cy+h*0.1, h*0.2, np.pi, 2*np.pi, h*0.1)
    m=np.maximum(m, arc_mask(cx+h*0.22, cy+h*0.1, h*0.2, np.pi, 2*np.pi, h*0.1))
    m=np.maximum(m, line_mask((cx-h*0.5,cy+h*0.12),(cx+h*0.5,cy+h*0.12),h*0.08))
    return m

def sym_slippery(cx,cy,h):  # 23 car + skid
    # car body
    m=line_mask((cx-h*0.3,cy-h*0.1),(cx+h*0.3,cy-h*0.1),h*0.18)
    m=np.maximum(m, line_mask((cx-h*0.15,cy-h*0.25),(cx+h*0.15,cy-h*0.25),h*0.14))
    # skid wavy lines
    m=np.maximum(m, arc_mask(cx-h*0.25,cy+h*0.35,h*0.12,np.pi,2*np.pi,h*0.07))
    m=np.maximum(m, arc_mask(cx+h*0.05,cy+h*0.35,h*0.12,0,np.pi,h*0.07))
    return m

def sym_narrows(cx,cy,h):  # 24 road narrows right
    m=line_mask((cx-h*0.3,cy+h*0.5),(cx-h*0.18,cy-h*0.5),h*0.1)
    m=np.maximum(m, line_mask((cx+h*0.3,cy+h*0.5),(cx+h*0.05,cy-h*0.5),h*0.1))
    return m

def sym_roadworks(cx,cy,h):  # 25 worker + mound
    # worker: head + body
    m=disc_mask(cx-h*0.05,cy-h*0.35,h*0.1)
    m=np.maximum(m, line_mask((cx-h*0.05,cy-h*0.25),(cx-h*0.05,cy+h*0.05),h*0.12))
    m=np.maximum(m, line_mask((cx-h*0.05,cy-h*0.15),(cx+h*0.22,cy-h*0.3),h*0.1)) # shovel arm
    # mound
    m=np.maximum(m, arc_mask(cx,cy+h*0.42,h*0.3,np.pi,2*np.pi,h*0.14))
    return m

def sym_lights(cx,cy,h):  # 26 traffic lights (3 dots vertical) in a box
    m=disc_mask(cx,cy-h*0.3,h*0.12)
    m=np.maximum(m, disc_mask(cx,cy,h*0.12))
    m=np.maximum(m, disc_mask(cx,cy+h*0.3,h*0.12))
    m=np.maximum(m, np.clip(ring_outline(cx,cy,h*0.22,h*0.55,h*0.07),0,1))
    return m

def ring_outline(cx,cy,halfw,halfh,th):
    outer=((np.abs(XX-cx)<halfw)&(np.abs(YY-cy)<halfh)).astype(np.float32)
    inner=((np.abs(XX-cx)<halfw-th)&(np.abs(YY-cy)<halfh-th)).astype(np.float32)
    return outer-inner

def sym_person(cx,cy,h):  # 27 pedestrian
    m=disc_mask(cx,cy-h*0.38,h*0.11)
    m=np.maximum(m, line_mask((cx,cy-h*0.27),(cx,cy+h*0.1),h*0.12))
    m=np.maximum(m, line_mask((cx,cy+h*0.1),(cx-h*0.18,cy+h*0.5),h*0.1)) # legs
    m=np.maximum(m, line_mask((cx,cy+h*0.1),(cx+h*0.18,cy+h*0.5),h*0.1))
    m=np.maximum(m, line_mask((cx,cy-h*0.15),(cx+h*0.22,cy+h*0.05),h*0.09)) # arm
    return m

def sym_children(cx,cy,h):  # 28 two figures
    m=sym_person(cx-h*0.22,cy,h*0.8)
    m=np.maximum(m, sym_person(cx+h*0.22,cy,h*0.8))
    return m

def sym_bike(cx,cy,h):  # 29 bicycle
    m=ellipse_ring(cx-h*0.28,cy+h*0.18,h*0.22,h*0.22,h*0.06)
    m=np.maximum(m, ellipse_ring(cx+h*0.28,cy+h*0.18,h*0.22,h*0.22,h*0.06))
    m=np.maximum(m, line_mask((cx-h*0.28,cy+h*0.18),(cx+h*0.1,cy-h*0.1),h*0.06))
    m=np.maximum(m, line_mask((cx+h*0.28,cy+h*0.18),(cx+h*0.1,cy-h*0.1),h*0.06))
    m=np.maximum(m, line_mask((cx-h*0.28,cy+h*0.18),(cx+h*0.28,cy+h*0.18),h*0.05))
    return m

def sym_snow(cx,cy,h):  # 30 snowflake / asterisk
    m=np.zeros((R,R),np.float32)
    for a in range(0,180,60):
        ang=np.deg2rad(a)
        dx=np.cos(ang)*h*0.42; dy=np.sin(ang)*h*0.42
        m=np.maximum(m, line_mask((cx-dx,cy-dy),(cx+dx,cy+dy),h*0.1))
    return m

def sym_deer(cx,cy,h):  # 31 deer (schematic body + legs + antler)
    m=line_mask((cx-h*0.3,cy),(cx+h*0.25,cy-h*0.05),h*0.16)  # body
    m=np.maximum(m, line_mask((cx+h*0.25,cy-h*0.05),(cx+h*0.42,cy-h*0.35),h*0.1)) # neck/head
    m=np.maximum(m, line_mask((cx-h*0.22,cy+h*0.05),(cx-h*0.28,cy+h*0.45),h*0.08)) # legs
    m=np.maximum(m, line_mask((cx+h*0.15,cy+h*0.05),(cx+h*0.2,cy+h*0.45),h*0.08))
    m=np.maximum(m, line_mask((cx+h*0.42,cy-h*0.35),(cx+h*0.55,cy-h*0.55),h*0.06)) # antler
    return m

def sym_priority_junction(cx,cy,h):  # 11 thick vertical + thin horizontal
    m=line_mask((cx,cy-h*0.5),(cx,cy+h*0.5),h*0.2)
    m=np.maximum(m, line_mask((cx-h*0.4,cy-h*0.05),(cx+h*0.4,cy-h*0.05),h*0.1))
    return m

# cars for no-passing / derestriction
def car(cx,cy,w,hh,th):
    m=line_mask((cx-w/2,cy),(cx+w/2,cy),hh)   # body block (thick line)
    return m

def truck(cx,cy,w,hh,th):
    m=line_mask((cx-w/2,cy),(cx+w/2,cy),hh)   # longer body
    return m

# blue arrows (white). Returned as ink mask; caller colours white on blue.
def arrow_straight(cx,cy,h):
    m=line_mask((cx,cy+h*0.45),(cx,cy-h*0.35),h*0.18)
    m=np.maximum(m, line_mask((cx,cy-h*0.45),(cx-h*0.22,cy-h*0.18),h*0.16))
    m=np.maximum(m, line_mask((cx,cy-h*0.45),(cx+h*0.22,cy-h*0.18),h*0.16))
    return m

def arrow_turn(cx,cy,h,right=True):   # up then turn (33/34)
    s=1 if right else -1
    m=line_mask((cx-s*h*0.2,cy+h*0.45),(cx-s*h*0.2,cy),h*0.16)
    m=np.maximum(m, arc_mask(cx, cy, h*0.2, np.pi if right else 0, np.pi*1.5 if right else -np.pi*0.5, h*0.16))
    tipx=cx+s*h*0.0; tipy=cy-h*0.2
    m=np.maximum(m, line_mask((cx,cy-h*0.2),(cx-s*h*0.0,cy-h*0.2),h*0.01))
    # head pointing up-ish at end of arc (end at top)
    ex,ey=cx, cy-h*0.2
    m=np.maximum(m, line_mask((ex,ey),(ex-s*h*0.18,ey+h*0.08),h*0.14))
    m=np.maximum(m, line_mask((ex,ey),(ex+s*h*0.02,ey+h*0.22),h*0.14))
    return m

def arrow_fork(cx,cy,h,right=True):  # straight + branch (36/37)
    s=1 if right else -1
    m=arrow_straight(cx,cy,h)
    m=np.maximum(m, line_mask((cx,cy+h*0.05),(cx+s*h*0.33,cy-h*0.28),h*0.15))
    ex,ey=cx+s*h*0.33,cy-h*0.28
    m=np.maximum(m, line_mask((ex,ey),(ex-s*h*0.2,ey-h*0.02),h*0.12))
    m=np.maximum(m, line_mask((ex,ey),(ex-s*h*0.02,ey+h*0.2),h*0.12))
    return m

def arrow_diag(cx,cy,h,right=True):  # keep right/left (38/39) diagonal down
    s=1 if right else -1
    m=line_mask((cx-s*h*0.3,cy-h*0.4),(cx+s*h*0.3,cy+h*0.4),h*0.2)
    ex,ey=cx+s*h*0.3,cy+h*0.4
    m=np.maximum(m, line_mask((ex,ey),(ex-s*h*0.28,ey-h*0.02),h*0.16))
    m=np.maximum(m, line_mask((ex,ey),(ex-s*h*0.02,ey-h*0.28),h*0.16))
    return m

def arrow_roundabout(cx,cy,h):  # 40 three curved arrows
    m=np.zeros((R,R),np.float32)
    for a in range(0,360,120):
        a0=np.deg2rad(a); a1=np.deg2rad(a+80)
        m=np.maximum(m, arc_mask(cx,cy,h*0.42,a0,a1,h*0.1))
        ex=cx+h*0.42*np.cos(a1); ey=cy+h*0.42*np.sin(a1)
        tx=-np.sin(a1); ty=np.cos(a1)
        m=np.maximum(m, line_mask((ex,ey),(ex-h*0.14*tx+h*0.06*np.cos(a1),ey-h*0.14*ty+h*0.06*np.sin(a1)),h*0.08))
    return m


# ===== classes.py =====
RED,WHITE,BLACK,BLUE,YELLOW,GRAY,DGRAY = RED,WHITE,BLACK,BLUE,YELLOW,GRAY,DGRAY

ROUT=15.0; RIN=11.3                       # ring geometry
def base_ring():
    cv=canvas((0,0,0))
    paint(cv, disc_mask(16,16,ROUT), RED)
    paint(cv, disc_mask(16,16,RIN), WHITE)
    return cv
def base_blue():
    cv=canvas((0,0,0)); paint(cv, disc_mask(16,16,ROUT), BLUE); return cv
def base_reddisc():
    cv=canvas((0,0,0)); paint(cv, disc_mask(16,16,ROUT), RED); return cv
def base_tri(up=True):
    cv=canvas((0,0,0))
    if up: V=tri_up(16,17,16.5,25.5)
    else:  V=tri_down(16,15,16.5,25.5)
    paint(cv, poly_mask(V), RED)
    if up: Vi=tri_up(16,17.6,12.2,19.0)
    else:  Vi=tri_down(16,14.4,12.2,19.0)
    paint(cv, poly_mask(Vi), WHITE)
    return cv
def base_diamond():
    cv=canvas((0,0,0))
    V=[(16,2.5),(29.5,16),(16,29.5),(2.5,16)]
    paint(cv, poly_mask(V), WHITE)
    Vi=[(16,5.2),(26.8,16),(16,26.8),(5.2,16)]
    paint(cv, poly_mask(Vi), YELLOW)
    return cv
def base_gray():
    cv=canvas((0,0,0))
    paint(cv, disc_mask(16,16,ROUT), WHITE)
    paint(cv, ring_mask(16,16,ROUT,ROUT-1.4), DGRAY)
    paint(cv, line_mask((7.0,25.0),(25.0,7.0), 2.2), DGRAY)  # diagonal slash
    return cv
def base_octagon():
    cv=canvas((0,0,0))
    a=15.0; k=a*0.4142
    V=[(16-k,16-a),(16+k,16-a),(16+a,16-k),(16+a,16+k),(16+k,16+a),(16-k,16+a),(16-a,16+k),(16-a,16-k)]
    paint(cv, poly_mask(V), RED)
    return cv

def ink_white(cv, mask):   # black ink on white interior
    return paint(cv, mask, BLACK)
def ink_blue(cv, mask):    # white ink on blue
    return paint(cv, mask, WHITE)

# number drawing params
def speed(cv, s):
    if len(s)==2: SY_ink=draw_number(s,16,16.5,15.0,12.5,2.2)
    else:         SY_ink=draw_number(s,16,16.5,19.0,11.5,1.9)
    return paint(cv, SY_ink, BLACK)

def render(k):
    if k in (0,1,2,3,4,5,7,8):
        cv=base_ring(); return downsample(speed(cv, {0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[k]))
    if k==9:   # no passing: red car (left) + black car (right), clear gap
        cv=base_ring(); paint(cv, line_mask((8.8,16.5),(13.8,16.5),5.0), RED)
        paint(cv, line_mask((18.2,16.5),(23.2,16.5),5.0), BLACK); return downsample(cv)
    if k==10:  # no passing trucks: small red car (left) + long black truck (right)
        cv=base_ring(); paint(cv, line_mask((8.8,16.5),(12.6,16.5),5.0), RED)
        paint(cv, line_mask((14.4,16.5),(23.2,16.5),5.6), BLACK); return downsample(cv)
    if k==15:  cv=base_ring(); return downsample(cv)   # no vehicles (empty)
    if k==16:  cv=base_ring(); paint(cv, line_mask((9.8,16.5),(22.2,16.5),6.0), BLACK); return downsample(cv) # single truck
    if k==17:  cv=base_reddisc(); paint(cv, line_mask((8.5,16),(23.5,16),4.6), WHITE); return downsample(cv) # no entry
    if k==14:  # stop octagon + STOP text band
        cv=base_octagon(); paint(cv, line_mask((8.5,16),(23.5,16),7.0), RED)
        # white STOP-ish band: four blocks
        for i,x in enumerate([10.2,13.6,17.0,20.4]):
            paint(cv, line_mask((x,13.5),(x,18.5),1.6), WHITE)
        return downsample(cv)
    if k==11:  cv=base_tri(True); return downsample(paint(cv, sym_priority_junction(16,17.5,10.5), BLACK))
    if k==18:  cv=base_tri(True); return downsample(paint(cv, sym_exclaim(16,17.0,12.0), BLACK))
    if k==19:  cv=base_tri(True); return downsample(paint(cv, sym_bend(16,17.5,11.0,right=False), BLACK))
    if k==20:  cv=base_tri(True); return downsample(paint(cv, sym_bend(16,17.5,11.0,right=True), BLACK))
    if k==21:  cv=base_tri(True); return downsample(paint(cv, sym_double_bend(16,17.5,11.0,right_first=True), BLACK))
    if k==22:  cv=base_tri(True); return downsample(paint(cv, sym_bumps(16,17.5,11.0), BLACK))
    if k==23:  cv=base_tri(True); return downsample(paint(cv, sym_slippery(16,17.5,11.5), BLACK))
    if k==24:  cv=base_tri(True); return downsample(paint(cv, sym_narrows(16,17.5,11.0), BLACK))
    if k==25:  cv=base_tri(True); return downsample(paint(cv, sym_roadworks(16,17.5,11.5), BLACK))
    if k==26:  cv=base_tri(True); return downsample(paint(cv, sym_lights(16,17.5,11.5), BLACK))
    if k==27:  cv=base_tri(True); return downsample(paint(cv, sym_person(16,17.5,12.0), BLACK))
    if k==28:  cv=base_tri(True); return downsample(paint(cv, sym_children(16,17.5,12.0), BLACK))
    if k==29:  cv=base_tri(True); return downsample(paint(cv, sym_bike(16,17.5,12.0), BLACK))
    if k==30:  cv=base_tri(True); return downsample(paint(cv, sym_snow(16,17.5,11.5), BLACK))
    if k==31:  cv=base_tri(True); return downsample(paint(cv, sym_deer(16,17.5,12.0), BLACK))
    if k==13:  cv=base_tri(False); return downsample(cv)  # yield
    if k==12:  cv=base_diamond(); return downsample(cv)   # priority road
    if k==33:  cv=base_blue(); return downsample(paint(cv, arrow_turn(16,16,13.5,right=True), WHITE))
    if k==34:  cv=base_blue(); return downsample(paint(cv, arrow_turn(16,16,13.5,right=False), WHITE))
    if k==35:  cv=base_blue(); return downsample(paint(cv, arrow_straight(16,16,14.0), WHITE))
    if k==36:  cv=base_blue(); return downsample(paint(cv, arrow_fork(16,16,13.5,right=True), WHITE))
    if k==37:  cv=base_blue(); return downsample(paint(cv, arrow_fork(16,16,13.5,right=False), WHITE))
    if k==38:  cv=base_blue(); return downsample(paint(cv, arrow_diag(16,16,13.0,right=True), WHITE))
    if k==39:  cv=base_blue(); return downsample(paint(cv, arrow_diag(16,16,13.0,right=False), WHITE))
    if k==40:  cv=base_blue(); return downsample(paint(cv, arrow_roundabout(16,16,13.0), WHITE))
    if k==6:   cv=base_gray(); return downsample(paint(cv, draw_number('80',16,16.5,12.0,10.0,1.8), DGRAY))
    if k==32:  cv=base_gray(); return downsample(cv)
    if k==41:  cv=base_gray(); paint(cv, line_mask((10.5,16.5),(14.5,16.5),3.4), DGRAY); paint(cv, line_mask((17.0,16.5),(21.0,16.5),3.4), DGRAY); return downsample(cv)
    if k==42:  cv=base_gray(); paint(cv, line_mask((10.5,16.5),(14.0,16.5),3.2), DGRAY); paint(cv, line_mask((16.0,16.5),(21.5,16.5),4.0), DGRAY); return downsample(cv)
    raise ValueError(k)


# ===== buildcore.py =====

# ---------------- families ----------------
FRING,FREDFILL,FTRIUP,FTRIDN,FDIAM,FBLUE,FGRAY = range(7)
NFAM=7
FAMILY={}
for k in [0,1,2,3,4,5,7,8,9,10,15,16]: FAMILY[k]=FRING
FAMILY[14]=FREDFILL; FAMILY[17]=FREDFILL
for k in [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]: FAMILY[k]=FTRIUP
FAMILY[13]=FTRIDN; FAMILY[12]=FDIAM
for k in [33,34,35,36,37,38,39,40]: FAMILY[k]=FBLUE
for k in [6,32,41,42]: FAMILY[k]=FGRAY

# ---------------- colour transform ----------------
M = np.array([[1/3,1/3,1/3],[1,-.5,-.5],[-.5,-.5,1.],[-.5,1.,-.5]],np.float32)  # (4,3) f->rgb rows
def feat_to_rgb(Tf):           # Tf (4,32,32) -> (3,32,32): K_rgb[i]=sum_f M[f,i]Tf[f]
    return np.tensordot(M, Tf, axes=(0,0)).astype(np.float32)
def colorA(gl,gc):
    g=np.diag([gl,gc,gc,gc]).astype(np.float32); return (M.T@g@M).astype(np.float32)

# ---------------- region masks (32x32 floats) ----------------
def ds(m2d): return downsample(np.repeat(m2d[...,None],3,2))[...,0]
def reg_ring(ro,ri): return ds(ring_mask(16,16,ro,ri))
def reg_disc(r):     return ds(disc_mask(16,16,r))
def reg_disc_at(cx,cy,r): return ds(disc_mask(cx,cy,r))
def reg_triup_out():
    o=poly_mask(tri_up(16,17,16.8,26.0)); i=poly_mask(tri_up(16,17.4,12.0,18.6))
    return ds(np.clip(o-i,0,1))
def reg_tridn_out():
    o=poly_mask(tri_down(16,15,16.8,26.0)); i=poly_mask(tri_down(16,14.6,12.0,18.6))
    return ds(np.clip(o-i,0,1))
def reg_diamond():
    return ds(poly_mask([(16,4),(28,16),(16,28),(4,16)]))
def reg_diag():
    return ds(np.clip(line_mask((8,24),(24,8),2.6)*disc_mask(16,16,13.5),0,1))
def reg_band(y0,y1,r=14.0):
    b=((YYg>=y0)&(YYg<=y1)).astype(np.float32)
    return ds(np.clip(b*disc_mask(16,16,r),0,1))

# ---------------- coarse feature templates ----------------
def blob(cx,cy,r=4.3): return reg_disc_at(cx,cy,r)
# red-location blobs: triangle/ring corners (y increases downward)
TL,TR=(6.5,6.5),(25.5,6.5)        # top corners
BLc,BRc=(6.5,25.5),(25.5,25.5)    # bottom corners
TC,BC=(16,4.5),(16,27.5)          # top/bottom centre (apexes)
def coarse_feat(fam):
    Tf=np.zeros((4,32,32),np.float32)
    L,RE,BL,GR=0,1,2,3
    if fam==FRING:
        Tf[RE]+= reg_ring(15.2,11.0)*1.0      # full red circle...
        Tf[RE]-= reg_disc(9.5)*1.5            # ...with non-red interior
        Tf[RE]+= (blob(3.2,16)+blob(28.8,16))*0.5   # red at 3 & 9 o'clock (ring, not triangle)
    elif fam==FREDFILL:                        # stop + no-entry: red interior
        Tf[RE]+= reg_disc(10.5)*1.0
    elif fam==FTRIUP:                          # triangle outline + base corners, NOT top corners (vs ring/tri-down)
        Tf[RE]+= reg_triup_out()*1.0
        Tf[RE]+= (blob(*BLc)+blob(*BRc)+blob(*TC))*0.4
        Tf[RE]-= (blob(*TL)+blob(*TR))*1.5
    elif fam==FTRIDN:                          # triangle outline + top corners, NOT bottom corners (vs ring/tri-up)
        Tf[RE]+= reg_tridn_out()*1.0
        Tf[RE]+= (blob(*TL)+blob(*TR)+blob(*BC))*0.4
        Tf[RE]-= (blob(*BLc)+blob(*BRc))*1.5
    elif fam==FDIAM:
        dm=reg_diamond(); Tf[RE]+=dm*1.0; Tf[GR]+=dm*1.0; Tf[BL]-=dm*1.0
    elif fam==FBLUE:
        Tf[BL]+= reg_disc(14.3)*1.0; Tf[RE]-=reg_disc(14.3)*0.4
    elif fam==FGRAY:
        d=reg_disc(13.5)
        Tf[L]-= reg_diag()*1.0            # dark centred diagonal
        Tf[RE]-= d*1.1; Tf[BL]-= d*1.1    # strongly penalise coloured signs
    return Tf

# ---------------- fine (interior) ----------------
def fine_mask(fam):
    if fam==FRING:    return reg_disc(10.6)
    if fam==FREDFILL: return reg_disc(12.0)
    if fam==FTRIUP:   return reg_disc_at(16,17.4,8.8)
    if fam==FBLUE:   return reg_disc(12.2)
    if fam==FGRAY:   return reg_disc(10.6)
    return None

Af=colorA(1.0,1.0)
def blur32(img,k=np.array([0.25,0.5,0.25],np.float32)):
    a=img.astype(np.float32).copy()
    for ax in (0,1): a=np.apply_along_axis(lambda m:np.convolve(m,k,mode='same'),ax,a)
    return a
def warp32(img,f,deg):
    """scale by f and rotate by deg about centre; bilinear; outside->0. img HxWx3"""
    if f==1.0 and deg==0.0: return img.astype(np.float32)
    H,Wd=img.shape[:2]; cy=(H-1)/2; cx=(Wd-1)/2
    a=np.deg2rad(deg); ca,sa=np.cos(a),np.sin(a)
    ys=np.arange(H)-cy; xs=np.arange(Wd)-cx
    GY,GX=np.meshgrid(ys,xs,indexing='ij')
    # source = R(-deg) @ out / f
    SY=( ca*GY + sa*GX)/f + cy
    SX=(-sa*GY + ca*GX)/f + cx
    y0=np.floor(SY).astype(int); x0=np.floor(SX).astype(int); wy=SY-y0; wx=SX-x0
    def samp(yy,xx):
        m=((yy>=0)&(yy<H)&(xx>=0)&(xx<Wd))
        v=img[np.clip(yy,0,H-1),np.clip(xx,0,Wd-1)]
        return v*m[...,None]
    v00=samp(y0,x0);v01=samp(y0,x0+1);v10=samp(y0+1,x0);v11=samp(y0+1,x0+1)
    wy_=wy[...,None];wx_=wx[...,None]
    return (v00*(1-wy_)*(1-wx_)+v01*(1-wy_)*wx_+v10*wy_*(1-wx_)+v11*wy_*wx_).astype(np.float32)
def zoom32(img,f): return warp32(img,f,0.0)

def unit(K):
    n=np.sqrt((K*K).sum()); return K/n if n>1e-8 else K*0

def fine_kernel(k,scale,deg):
    fam=FAMILY[k]; mask=fine_mask(fam)
    if mask is None: return np.zeros((3,32,32),np.float32)
    rgb=warp32(render(k),scale,deg)
    mask=warp32(np.repeat(mask[...,None],3,2),scale,deg)[...,0]
    rgb=blur32(rgb)
    col=(rgb.reshape(-1,3)@Af.T).reshape(32,32,3)
    mm=(mask>0.5).astype(np.float32); col*=mm[...,None]
    area=mm.sum()+1e-9
    for c in range(3): col[:,:,c]=(col[:,:,c]-col[:,:,c].sum()/area)*mm
    return unit(np.transpose(col,(2,0,1)).astype(np.float32))

CBLUR=np.array([0.12,0.22,0.32,0.22,0.12],np.float32)   # wider blur -> shift tolerance
def coarse_kernel(fam,scale,deg):
    K=feat_to_rgb(coarse_feat(fam))                      # (3,32,32)
    img=warp32(np.transpose(K,(1,2,0)),scale,deg)
    img=blur32(img,CBLUR)
    return unit(np.transpose(img,(2,0,1)))

VARIANTS=[(1.0,0.0),(0.9,0.0)]
SCALES=VARIANTS
WC=3.0; WF=1.0
CPAD=0; CO=32+2*CPAD-32+1    # coarse: centre-only (preserves negative evidence)
FPAD=2; FO=32+2*FPAD-32+1    # fine: small translation search
EMPTY_BIAS={15:-3.0, 32:-3.0}

def build(outdir):
    os.makedirs(outdir,exist_ok=True)
    nS=len(VARIANTS)
    ccW=np.zeros((NFAM*nS,3,32,32),np.float32)
    for fam in range(NFAM):
        for si,(s,dg) in enumerate(VARIANTS): ccW[fam*nS+si]=coarse_kernel(fam,s,dg)*WC
    fcW=np.zeros((43*nS,3,32,32),np.float32)
    fcb=np.zeros(43*nS,np.float32)
    for k in range(43):
        for si,(s,dg) in enumerate(VARIANTS):
            fcW[k*nS+si]=fine_kernel(k,s,dg)*WF
            if k in EMPTY_BIAS: fcb[k*nS+si]=EMPTY_BIAS[k]
    d=np.zeros((43,NFAM),np.float32)
    for c in range(43): d[c,FAMILY[c]]=1.0
    weights={'cc.W':ccW,'cc.b':np.zeros(NFAM*nS,np.float32),
             'fc.W':fcW,'fc.b':fcb,
             'coarse43.W':d,'coarse43.b':np.zeros(43,np.float32)}
    np.savez(os.path.join(outdir,'weights.npz'),**weights)
    net={"layers":[
      {"name":"n","op":"norm","inputs":["image"]},
      {"name":"cc","op":"conv","inputs":["n"],"out":NFAM*nS,"kernel":[32,32],"stride":1,"padding":CPAD},
      {"name":"cp","op":"maxpool","inputs":["cc"],"size":[CO,CO],"stride":[CO,CO]},
      {"name":"cr","op":"reshape","inputs":["cp"],"shape":[NFAM,nS,1]},
      {"name":"cm","op":"maxpool","inputs":["cr"],"size":[nS,1],"stride":[nS,1]},
      {"name":"csuper","op":"reshape","inputs":["cm"],"shape":[NFAM]},
      {"name":"fc","op":"conv","inputs":["n"],"out":43*nS,"kernel":[32,32],"stride":1,"padding":FPAD},
      {"name":"fp","op":"maxpool","inputs":["fc"],"size":[FO,FO],"stride":[FO,FO]},
      {"name":"fr","op":"reshape","inputs":["fp"],"shape":[43,nS,1]},
      {"name":"fm","op":"maxpool","inputs":["fr"],"size":[nS,1],"stride":[nS,1]},
      {"name":"ffine","op":"reshape","inputs":["fm"],"shape":[43]},
      {"name":"coarse43","op":"dense","inputs":["csuper"],"out":43},
      {"name":"out","op":"add","inputs":["coarse43","ffine"]},
    ]}
    json.dump(net,open(os.path.join(outdir,'network.json'),'w'))
    return weights,net





if __name__=='__main__':
    build(sys.argv[1])
