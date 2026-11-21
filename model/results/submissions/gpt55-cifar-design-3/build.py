import json, sys, os
import numpy as np

classes = 10
H8 = W8 = 8

# Utility masks on an 8x8 grid, using pixel centers in grid coordinates 0..7.
def grid():
    y, x = np.mgrid[0:H8, 0:W8].astype(np.float32)
    return y, x
GY, GX = grid()

def rect(x0, x1, y0, y1, soft=0.35):
    # soft rectangular mask
    xleft = 1/(1+np.exp(-(GX-x0)/soft)); xright = 1/(1+np.exp((GX-x1)/soft))
    ytop = 1/(1+np.exp(-(GY-y0)/soft)); ybot = 1/(1+np.exp((GY-y1)/soft))
    return (xleft*xright*ytop*ybot).astype(np.float32)

def gauss(cx, cy, sx, sy=None):
    if sy is None: sy = sx
    return np.exp(-0.5*(((GX-cx)/sx)**2 + ((GY-cy)/sy)**2)).astype(np.float32)

def ellipse(cx, cy, sx, sy):
    r = ((GX-cx)/sx)**2 + ((GY-cy)/sy)**2
    return np.clip(1-r, 0, 1).astype(np.float32)

def line_mask(cx, cy, length, width, angle):
    ca, sa = np.cos(angle), np.sin(angle)
    xp = (GX-cx)*ca + (GY-cy)*sa
    yp = -(GX-cx)*sa + (GY-cy)*ca
    return (np.exp(-0.5*(yp/width)**2) * np.exp(-0.5*(xp/(length/2))**8)).astype(np.float32)

def tri_mask(points):
    # simple soft union of vertices/edges by rasterizing coarse polygon with matplotlib-free winding test
    pts = np.array(points, np.float32)
    x = GX; y = GY
    inside = np.zeros_like(x, dtype=bool)
    j = len(pts)-1
    for i in range(len(pts)):
        xi, yi = pts[i]; xj, yj = pts[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj-xi)*(y-yi)/(yj-yi+1e-6)+xi)
        inside ^= cond
        j = i
    # soften with nearby vertices/edges
    m = inside.astype(np.float32)
    for i in range(len(pts)):
        x1,y1=pts[i]; x2,y2=pts[(i+1)%len(pts)]
        vx=x2-x1; vy=y2-y1; den=vx*vx+vy*vy+1e-6
        t=np.clip(((GX-x1)*vx+(GY-y1)*vy)/den,0,1)
        d=((GX-(x1+t*vx))**2+(GY-(y1+t*vy))**2)**0.5
        m=np.maximum(m, np.exp(-0.5*(d/0.45)**2)*0.6)
    return m.astype(np.float32)

def zone_top(): return rect(-1, 8, -1, 3.6, 0.6)
def zone_bottom(): return rect(-1, 8, 3.4, 8, 0.6)
def zone_center(): return gauss(3.5, 3.8, 2.3, 2.0)
def zone_lower_center(): return gauss(3.5, 5.0, 2.4, 1.6)
def zone_whole(): return np.ones((8,8), np.float32)

# Color directions for linear RGB evidence. They are hand chosen semantic directions.
lum = np.array([1/3,1/3,1/3], np.float32)
dark = -lum
bright = lum
sky = np.array([-0.35, 0.10, 0.90], np.float32)
blue = np.array([-0.25, -0.05, 0.75], np.float32)
water = np.array([-0.10, 0.20, 0.70], np.float32)
green = np.array([-0.45, 0.85, -0.35], np.float32)
veg = np.array([-0.55, 1.05, -0.35], np.float32)
brown = np.array([0.75, 0.36, -0.55], np.float32)
yellow = np.array([0.65, 0.55, -0.45], np.float32)
red = np.array([0.90, -0.35, -0.35], np.float32)
white = np.array([0.45,0.45,0.45], np.float32)
black = -np.array([0.55,0.55,0.55], np.float32)


def add_raw(W, c, mask, vec, coef):
    for ch in range(3):
        W[c, ch] += coef * mask * vec[ch]

# ---------- raw low-resolution linear score branch ----------
rawW = np.zeros((10, 3, 8, 8), np.float32); rawb = np.zeros(10, np.float32)
whole=zone_whole(); top=zone_top(); bot=zone_bottom(); cen=zone_center(); low=zone_lower_center()
# 0 airplane: blue/bright sky, little vegetation, thin central object.
add_raw(rawW,0, whole, sky, 1.15); add_raw(rawW,0, top, sky, 1.05); add_raw(rawW,0, whole, green, -0.95); add_raw(rawW,0, cen, dark, 0.28); add_raw(rawW,0, top, bright, 0.25)
# 1 automobile: lower centered, dark road/wheels, colorful body, not sky/grass dominated.
add_raw(rawW,1, low, dark, 1.10); add_raw(rawW,1, rect(1.0,7.0,4.0,7.5), dark, 0.65); add_raw(rawW,1, cen, red, 0.38); add_raw(rawW,1, cen, blue, 0.25); add_raw(rawW,1, whole, sky, -0.45); add_raw(rawW,1, whole, green, -0.35)
# 2 bird: compact natural subject, often green or sky with brown/dark center.
add_raw(rawW,2, whole, green, 0.35); add_raw(rawW,2, whole, sky, 0.24); add_raw(rawW,2, gauss(3.6,3.6,1.5,1.3), brown, 0.78); add_raw(rawW,2, gauss(3.6,3.6,1.2,1.2), dark, 0.42); add_raw(rawW,2, whole, blue, -0.10)
# 3 cat: close furry face, gray/brown/white center, less grass/sky.
add_raw(rawW,3, cen, brown, 0.55); add_raw(rawW,3, cen, bright, 0.30); add_raw(rawW,3, gauss(3.5,3.4,1.8,1.8), dark, 0.36); add_raw(rawW,3, whole, sky, -0.55); add_raw(rawW,3, whole, green, -0.28)
# 4 deer: brown animal in green/brown scenery.
add_raw(rawW,4, whole, veg, 0.78); add_raw(rawW,4, gauss(3.7,4.0,2.0,1.5), brown, 0.86); add_raw(rawW,4, rect(1.0,7.0,4.0,7.5), dark, 0.25); add_raw(rawW,4, whole, sky, -0.45)
# 5 dog: furry face/body, brown/white/dark, sometimes green outdoors.
add_raw(rawW,5, cen, brown, 0.68); add_raw(rawW,5, cen, bright, 0.24); add_raw(rawW,5, cen, dark, 0.45); add_raw(rawW,5, whole, green, 0.12); add_raw(rawW,5, whole, sky, -0.45)
# 6 frog: strong green/yellow, squat dark spotted subject.
add_raw(rawW,6, whole, green, 1.20); add_raw(rawW,6, cen, green, 0.92); add_raw(rawW,6, cen, yellow, 0.35); add_raw(rawW,6, cen, dark, 0.36); add_raw(rawW,6, whole, sky, -0.55)
# 7 horse: large brown/black side-on quadruped, green lower outdoors.
add_raw(rawW,7, whole, green, 0.52); add_raw(rawW,7, gauss(3.6,4.3,2.4,1.2), brown, 1.08); add_raw(rawW,7, rect(1.0,7.0,4.5,7.8), dark, 0.38); add_raw(rawW,7, whole, sky, -0.35)
# 8 ship: water below, sky above, bright hull, little vegetation.
add_raw(rawW,8, bot, water, 1.62); add_raw(rawW,8, top, sky, 0.66); add_raw(rawW,8, rect(1.0,7.0,3.4,6.3), bright, 0.58); add_raw(rawW,8, whole, green, -1.10); add_raw(rawW,8, whole, blue, 0.28)
# 9 truck: large tall vehicle, dark road/wheels, colorful/white box.
add_raw(rawW,9, rect(0.6,7.4,2.4,7.3), dark, 0.80); add_raw(rawW,9, rect(0.6,7.4,2.0,6.4), bright, 0.35); add_raw(rawW,9, cen, red, 0.55); add_raw(rawW,9, cen, blue, 0.20); add_raw(rawW,9, whole, green, -0.45); add_raw(rawW,9, whole, sky, -0.42)
# Mild priors: avoid the broad natural classes swallowing everything.
rawb += np.array([0.05, 0.00, -0.10, -0.05, -0.10, -0.08, -0.06, -0.08, 0.06, 0.02], np.float32)
rawW = rawW.reshape(10, -1)

# ---------- thresholded color branch ----------
color_defs = [
    ('green',      [-1.0,  2.0, -1.0], -0.10),
    ('blue',       [-0.8, -0.2,  1.6], -0.06),
    ('red',        [ 1.6, -0.8, -0.8], -0.08),
    ('yellow',     [ 0.8,  0.6, -1.0], -0.20),
    ('dark',       [-1/3, -1/3, -1/3],  0.55),
    ('verydark',   [-1/3, -1/3, -1/3],  0.34),
    ('bright',     [ 1/3,  1/3,  1/3], -0.55),
    ('cyan_sky',   [-0.5,  0.45, 1.0], -0.35),
    ('water',      [-0.2,  0.28, 0.75], -0.42),
    ('orange',     [ 1.1,  0.20,-0.9], -0.25),
    ('white',      [ 0.45, 0.45,0.45], -0.72),
    ('black',      [-0.55,-0.55,-0.55], 0.30),
    ('vegetation', [-0.7,  1.5, -0.45], -0.24),
    ('brown',      [ 1.0,  0.42,-0.95], -0.28),
    ('deepblue',   [-0.2, -0.1, 0.80], -0.28),
    ('pinkred',    [ 1.2, -0.15,-0.55], -0.35),
]
Ccol=len(color_defs)
colW=np.zeros((Ccol,3,1,1), np.float32); colb=np.zeros(Ccol,np.float32)
for i,(_,v,b) in enumerate(color_defs):
    colW[i,:,0,0]=np.array(v,np.float32); colb[i]=b
colScore=np.zeros((10,Ccol,8,8),np.float32); colBias=np.zeros(10,np.float32)
def add_col(c,name,mask,coef):
    idx=[i for i,(n,_,_) in enumerate(color_defs) if n==name][0]
    colScore[c,idx]+=coef*mask
# airplanes and ships: blue/cyan/bright vs green
for nm,coef in [('blue',0.9),('cyan_sky',1.0),('bright',0.25),('deepblue',0.25)]: add_col(0,nm,whole,coef)
add_col(0,'cyan_sky',top,0.9); add_col(0,'vegetation',whole,-1.0); add_col(0,'green',whole,-0.55); add_col(0,'dark',cen,0.25)
add_col(8,'water',bot,1.8); add_col(8,'blue',bot,1.1); add_col(8,'cyan_sky',top,0.55); add_col(8,'bright',rect(1,7,3,6),0.8); add_col(8,'white',rect(1,7,3,6),0.65); add_col(8,'vegetation',whole,-1.2); add_col(8,'green',whole,-0.8)
# vehicles
for c in [1,9]:
    add_col(c,'dark',rect(0.5,7.5,4.3,7.8),0.95); add_col(c,'black',rect(0.5,7.5,4.8,7.8),0.65)
    add_col(c,'red',cen,0.35); add_col(c,'pinkred',cen,0.25); add_col(c,'blue',cen,0.18); add_col(c,'white',cen,0.45); add_col(c,'bright',cen,0.20)
    add_col(c,'vegetation',whole,-0.45); add_col(c,'green',whole,-0.25); add_col(c,'cyan_sky',whole,-0.25)
add_col(1,'dark',rect(1.0,7.0,4.5,6.6),0.75); add_col(1,'bright',rect(1.0,7.0,3.6,5.8),0.25)
add_col(9,'dark',rect(0.5,7.5,2.5,7.4),0.95); add_col(9,'bright',rect(0.5,7.5,2.1,6.2),0.45); add_col(9,'red',rect(0.5,7.5,2.0,6.2),0.38)
# animals by color/background
add_col(2,'vegetation',whole,0.40); add_col(2,'green',whole,0.22); add_col(2,'brown',cen,0.55); add_col(2,'orange',cen,0.45); add_col(2,'dark',cen,0.42); add_col(2,'blue',whole,0.10)
add_col(3,'brown',cen,0.65); add_col(3,'orange',cen,0.25); add_col(3,'dark',cen,0.45); add_col(3,'white',cen,0.35); add_col(3,'vegetation',whole,-0.20); add_col(3,'cyan_sky',whole,-0.35)
add_col(5,'brown',cen,0.70); add_col(5,'orange',cen,0.30); add_col(5,'dark',cen,0.50); add_col(5,'white',cen,0.42); add_col(5,'vegetation',whole,0.10); add_col(5,'cyan_sky',whole,-0.30)
add_col(4,'vegetation',whole,0.90); add_col(4,'green',whole,0.52); add_col(4,'brown',gauss(3.6,4.0,2.1,1.4),0.90); add_col(4,'dark',rect(1,7,4.0,7.7),0.22); add_col(4,'cyan_sky',whole,-0.40)
add_col(7,'vegetation',whole,0.55); add_col(7,'green',whole,0.34); add_col(7,'brown',gauss(3.6,4.4,2.5,1.2),1.05); add_col(7,'dark',rect(1,7,4.6,7.8),0.45); add_col(7,'cyan_sky',whole,-0.32)
add_col(6,'green',whole,1.20); add_col(6,'vegetation',whole,1.00); add_col(6,'green',cen,0.85); add_col(6,'yellow',cen,0.42); add_col(6,'dark',cen,0.30); add_col(6,'cyan_sky',whole,-0.45)
colScore = colScore.reshape(10,-1)

# ---------- edge/line branch on per-channel normalized image ----------
edge_kernels=[]; edge_names=[]
def add_edge(name,k):
    k=np.array(k,np.float32); edge_names.append(name); edge_kernels.append(k)
sobx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)/4.0
soby=sobx.T
diag1=np.array([[0,1,2],[-1,0,1],[-2,-1,0]],np.float32)/4.0
diag2=np.array([[2,1,0],[1,0,-1],[0,-1,-2]],np.float32)/4.0
hline=np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]],np.float32)/3.0
vline=hline.T
dl1=np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]],np.float32)/3.0
dl2=np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]],np.float32)/3.0
lap=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.float32)/6.0
for nm,k in [('vx+',sobx),('vx-',-sobx),('hy+',soby),('hy-',-soby),('d1+',diag1),('d1-',-diag1),('d2+',diag2),('d2-',-diag2),('hline',hline),('vline',vline),('diagline1',dl1),('diagline2',dl2),('spot',lap),('valley',-lap)]:
    add_edge(nm,k)
Ce=len(edge_kernels)
edgeW=np.zeros((Ce,3,3,3),np.float32); edgeb=np.zeros(Ce,np.float32)
for i,k in enumerate(edge_kernels): edgeW[i,:,:,:]=k[None,:,:]/3.0
edgeScore=np.zeros((10,Ce,8,8),np.float32); edgeBias=np.zeros(10,np.float32)
def ei(name): return edge_names.index(name)
def add_edge_score(c,names,mask,coef):
    if isinstance(names,str): names=[names]
    for n in names: edgeScore[c,ei(n)] += coef*mask
edge_all=['vx+','vx-','hy+','hy-','d1+','d1-','d2+','d2-','hline','vline','diagline1','diagline2','spot']
# airplane: diagonal/horizontal high-contrast slender object in sky
add_edge_score(0,['d1+','d1-','d2+','d2-'],cen,0.55); add_edge_score(0,['hline','diagline1','diagline2'],rect(0.5,7.5,1.8,5.9),0.45); add_edge_score(0,['spot'],whole,-0.10)
# car/truck/ship: horizontal/vertical box edges lower half
add_edge_score(1,['hline','hy+','hy-'],rect(0.5,7.5,3.6,6.6),0.75); add_edge_score(1,['vline','vx+','vx-'],rect(0.5,7.5,3.2,6.8),0.35); add_edge_score(1,['spot'],rect(1,7,4.2,7.5),0.25)
add_edge_score(9,['hline','hy+','hy-'],rect(0.2,7.8,2.2,6.8),0.65); add_edge_score(9,['vline','vx+','vx-'],rect(0.2,7.8,1.8,7.2),0.55); add_edge_score(9,['spot'],rect(0.5,7.5,3.0,7.5),0.22)
add_edge_score(8,['hline','hy+','hy-'],rect(0.0,8.0,3.0,6.4),0.95); add_edge_score(8,['vline'],rect(2.0,6.0,1.0,5.6),0.28); add_edge_score(8,['spot'],top,-0.15)
# animals: more texture/spot in center; quadrupeds have lower vertical strokes
for c in [2,3,5,6]: add_edge_score(c,edge_all,cen,0.18)
add_edge_score(2,['d1+','d1-','d2+','d2-'],gauss(3.5,3.5,1.8,1.5),0.25)
add_edge_score(3,['spot','valley','vx+','vx-','hy+','hy-'],gauss(3.5,3.4,2.0,2.0),0.28)
add_edge_score(5,['spot','valley','vx+','vx-','hy+','hy-'],gauss(3.5,3.7,2.1,2.0),0.30)
add_edge_score(6,['spot','valley'],cen,0.45); add_edge_score(6,['hline'],rect(1,7,4.0,6.5),0.28)
for c in [4,7]:
    add_edge_score(c,['vline','vx+','vx-'],rect(1.0,7.0,4.0,7.8),0.45); add_edge_score(c,['hline','hy+','hy-'],rect(0.5,7.5,3.0,5.8),0.40); add_edge_score(c,['spot'],cen,0.12)
add_edge_score(4,['vline'],rect(2.0,6.5,3.2,7.5),0.20)
add_edge_score(7,['hline'],gauss(3.5,4.5,2.8,1.0),0.35)
edgeScore=edgeScore.reshape(10,-1)

# ---------- local part/color blob branch (max-pooled) ----------
part_names=[]; part_filters=[]; part_bias=[]
ky,kx=np.mgrid[-2:3,-2:3].astype(np.float32)
g5=np.exp(-0.5*((kx/1.15)**2+(ky/1.15)**2)); g5=g5/g5.sum()
lineh=np.exp(-0.5*(ky/0.6)**2)*np.exp(-0.5*(kx/2.0)**4); lineh=lineh/lineh.sum()
linev=lineh.T
ring=np.exp(-0.5*((np.sqrt(kx*kx+ky*ky)-1.3)/0.45)**2); ring=ring/(ring.sum()+1e-6)
center=np.exp(-0.5*((kx/0.75)**2+(ky/0.75)**2)); center=center/center.sum()
def add_part(name, colorvec, spatial, bias):
    W=np.zeros((3,5,5),np.float32)
    for ch in range(3): W[ch]=colorvec[ch]*spatial
    part_names.append(name); part_filters.append(W); part_bias.append(bias)
add_part('dark_blob', np.array([-1/3,-1/3,-1/3],np.float32), g5, 0.42)
add_part('black_blob', np.array([-0.55,-0.55,-0.55],np.float32), g5, 0.30)
add_part('bright_blob', np.array([1/3,1/3,1/3],np.float32), g5, -0.52)
add_part('green_blob', np.array([-0.5,1.0,-0.4],np.float32), g5, -0.12)
add_part('blue_blob', np.array([-0.3,0.0,0.8],np.float32), g5, -0.10)
add_part('brown_blob', np.array([0.8,0.35,-0.65],np.float32), g5, -0.18)
add_part('red_blob', np.array([1.0,-0.35,-0.35],np.float32), g5, -0.12)
add_part('dark_hline', np.array([-1/3,-1/3,-1/3],np.float32), lineh, 0.42)
add_part('dark_vline', np.array([-1/3,-1/3,-1/3],np.float32), linev, 0.42)
add_part('bright_hline', np.array([1/3,1/3,1/3],np.float32), lineh, -0.50)
add_part('dark_ring', np.array([-0.45,-0.45,-0.45],np.float32), ring+0.3*center, 0.35)
Cp=len(part_names)
partW=np.stack(part_filters,axis=0).astype(np.float32); partb=np.array(part_bias,np.float32)
partScore=np.zeros((10,Cp,8,8),np.float32); partBias=np.zeros(10,np.float32)
def pi(n): return part_names.index(n)
def add_part_score(c,names,mask,coef):
    if isinstance(names,str): names=[names]
    for n in names: partScore[c,pi(n)] += coef*mask
# wheels/road
wheelmask = gauss(2.2,5.8,0.85,0.65)+gauss(5.8,5.8,0.85,0.65)
add_part_score(1,['black_blob','dark_ring','dark_blob'],wheelmask,1.10); add_part_score(1,['dark_hline','bright_hline'],rect(1,7,3.5,5.6),0.42)
add_part_score(9,['black_blob','dark_ring','dark_blob'],wheelmask,0.95); add_part_score(9,['dark_hline','bright_hline','dark_vline'],rect(0.5,7.5,2.3,6.3),0.58)
# sky/water objects
add_part_score(0,['blue_blob'],whole,0.35); add_part_score(0,['dark_hline','bright_hline'],rect(0.5,7.5,1.8,5.6),0.50); add_part_score(0,['dark_blob'],cen,0.24)
add_part_score(8,['blue_blob'],bot,0.80); add_part_score(8,['bright_hline','dark_hline'],rect(0.5,7.5,3.2,5.8),0.90); add_part_score(8,['bright_blob'],rect(1,7,2.5,5.8),0.45)
# animal parts
add_part_score(2,['brown_blob','dark_blob','red_blob'],gauss(3.5,3.4,1.7,1.5),0.50); add_part_score(2,['green_blob','blue_blob'],whole,0.14)
add_part_score(3,['brown_blob','dark_blob','bright_blob'],cen,0.62); add_part_score(3,['black_blob'],gauss(2.6,3.0,0.7,0.5)+gauss(4.4,3.0,0.7,0.5),0.55)
add_part_score(5,['brown_blob','dark_blob','bright_blob'],cen,0.68); add_part_score(5,['black_blob'],gauss(3.5,4.1,0.7,0.55),0.58)
add_part_score(4,['green_blob'],whole,0.45); add_part_score(4,['brown_blob'],gauss(3.6,4.0,2.1,1.2),0.85); add_part_score(4,['dark_vline'],rect(1.2,6.8,4.2,7.8),0.55)
add_part_score(7,['green_blob'],whole,0.25); add_part_score(7,['brown_blob','dark_blob'],gauss(3.6,4.4,2.6,1.1),0.85); add_part_score(7,['dark_vline'],rect(1.0,7.0,4.3,7.8),0.72)
add_part_score(6,['green_blob'],whole,1.05); add_part_score(6,['green_blob','brown_blob'],cen,0.72); add_part_score(6,['black_blob','dark_blob'],gauss(2.6,3.0,0.65,0.55)+gauss(4.4,3.0,0.65,0.55),0.54)
partScore=partScore.reshape(10,-1)

# ---------- low-resolution semantic template branch ----------
# Hidden templates are hand-drawn from generic object/background shapes, not data.
tmpl_rows=[]; tmpl_class=[]; tmpl_gain=[]
def add_component(W, mask, vec, coef):
    s=float(mask.sum())
    if s < 1e-6: return
    for ch in range(3): W[ch] += coef * (mask/s) * vec[ch]

def new_template(c, comps, bias=-0.05, gain=1.0):
    W=np.zeros((3,8,8),np.float32)
    for mask,vec,coef in comps: add_component(W,mask,vec,coef)
    tmpl_rows.append((W.reshape(-1), np.float32(bias)))
    tmpl_class.append(c); tmpl_gain.append(gain)

def vehicle_templates():
    for dx in [-0.4,0,0.4]:
        body=rect(1.2+dx,6.8+dx,3.5,5.6,0.35)
        wheels=gauss(2.3+dx,5.8,0.55,0.45)+gauss(5.5+dx,5.8,0.55,0.45)
        road=rect(0,8,5.5,8,0.5)
        new_template(1,[(body,red,0.8),(body,white,0.5),(wheels,black,1.0),(road,dark,0.4),(whole,green,-0.4),(whole,sky,-0.25)],-0.12,1.25)
        body2=rect(0.8+dx,7.2+dx,2.6,5.9,0.35); cab=rect(0.8+dx,3.5+dx,2.2,5.6,0.3); wh=gauss(2.0+dx,6.0,0.55,0.45)+gauss(5.9+dx,6.0,0.55,0.45)
        new_template(9,[(body2,white,0.65),(body2,red,0.55),(cab,bright,0.3),(wh,black,1.0),(road,dark,0.38),(whole,green,-0.45),(whole,sky,-0.25)],-0.11,1.25)

def sky_water_templates():
    for ang in [0,0.35,-0.35,0.75,-0.75]:
        fus=line_mask(3.5,3.3,5.0,0.25,ang); wing=line_mask(3.5,3.4,3.2,0.22,ang+np.pi/2)
        new_template(0,[(whole,sky,0.9),(top,sky,0.6),(fus,dark,0.65),(wing,dark,0.45),(whole,green,-0.8)],-0.04,1.1)
        new_template(0,[(whole,sky,0.9),(top,sky,0.6),(fus,white,0.55),(wing,white,0.35),(whole,green,-0.8)],-0.03,1.0)
    for dy in [-0.3,0.2,0.6]:
        hull=rect(1.0,7.0,4.2+dy,5.5+dy,0.35); waterm=rect(-1,9,4.4+dy,9,0.5); mast=line_mask(3.8,3.2+dy,2.6,0.18,np.pi/2)
        new_template(8,[(waterm,water,1.0),(top,sky,0.55),(hull,white,0.70),(hull,dark,0.35),(mast,dark,0.20),(whole,green,-0.9)],-0.05,1.2)
        new_template(8,[(waterm,water,1.1),(top,sky,0.45),(hull,red,0.40),(hull,dark,0.55),(whole,green,-0.9)],-0.07,1.1)

def animal_templates():
    # Bird: small oval body plus head in natural or sky background.
    for bgvec,bgcoef in [(veg,0.45),(sky,0.35),(green,0.35)]:
        for dx in [-0.4,0.3]:
            body=ellipse(3.4+dx,3.7,1.0,0.75); head=ellipse(4.25+dx,3.2,0.45,0.45); tail=line_mask(2.5+dx,3.8,1.2,0.22,-0.5)
            new_template(2,[(whole,bgvec,bgcoef),(body,brown,0.85),(body,dark,0.35),(head,dark,0.40),(tail,dark,0.25)],-0.06,1.0)
    # Cat and dog face templates.
    for facevec in [brown,white,dark]:
        face=ellipse(3.5,3.8,2.0,2.1); ears=tri_mask([(2.0,2.7),(2.7,0.8),(3.2,2.7)])+tri_mask([(4.0,2.7),(4.7,0.8),(5.2,2.7)])
        eyes=gauss(2.75,3.35,0.35,0.25)+gauss(4.25,3.35,0.35,0.25); nose=gauss(3.5,4.3,0.38,0.30)
        new_template(3,[(face,facevec,0.75),(ears,facevec,0.45),(eyes,black,0.70),(nose,black,0.35),(whole,sky,-0.45),(whole,green,-0.25)],-0.08,1.1)
        snout=ellipse(3.5,4.45,0.95,0.65); ears2=ellipse(1.8,3.5,0.75,1.25)+ellipse(5.2,3.5,0.75,1.25)
        new_template(5,[(face,facevec,0.70),(ears2,dark,0.35),(eyes,black,0.55),(snout,white,0.25),(snout,brown,0.35),(nose,black,0.55),(whole,sky,-0.40)],-0.08,1.15)
    # Deer / horse side bodies.
    for scale in [0.9,1.05,1.2]:
        body=ellipse(3.4,4.2,2.2*scale,0.85*scale); neck=line_mask(5.0,3.4,1.6,0.28,-1.05); head=ellipse(5.5,2.8,0.65,0.45)
        legs=line_mask(2.3,5.7,2.1,0.17,np.pi/2)+line_mask(3.4,5.8,2.0,0.17,np.pi/2)+line_mask(4.7,5.8,2.0,0.17,np.pi/2)+line_mask(5.5,5.7,1.8,0.17,np.pi/2)
        new_template(7,[(whole,green,0.42),(body,brown,1.05),(neck,brown,0.45),(head,brown,0.45),(legs,dark,0.55),(top,sky,-0.15)],-0.08,1.2)
        ant=line_mask(5.7,2.1,0.9,0.12,-0.75)+line_mask(5.4,2.1,0.9,0.12,-2.35)
        slim=ellipse(3.5,4.0,1.75*scale,0.70*scale)
        new_template(4,[(whole,veg,0.62),(slim,brown,0.95),(neck,brown,0.55),(head,brown,0.45),(legs,dark,0.45),(ant,dark,0.20),(top,sky,-0.25)],-0.08,1.15)
    # Frog: low green oval, dark eyes/spots.
    for gy in [4.0,4.4]:
        body=ellipse(3.5,gy,2.15,1.25); eyes=gauss(2.6,gy-1.0,0.45,0.35)+gauss(4.4,gy-1.0,0.45,0.35); legs=line_mask(2.0,5.0,1.7,0.22,-0.45)+line_mask(5.0,5.0,1.7,0.22,0.45)
        new_template(6,[(whole,green,0.70),(body,green,1.10),(body,yellow,0.30),(eyes,black,0.75),(legs,green,0.35),(whole,sky,-0.45)],-0.06,1.25)
vehicle_templates(); sky_water_templates(); animal_templates()
H=len(tmpl_rows)
tmplW=np.stack([r[0] for r in tmpl_rows],axis=0).astype(np.float32); tmplb=np.array([r[1] for r in tmpl_rows],np.float32)
tmplScore=np.zeros((10,H),np.float32); tmplScore_b=np.zeros(10,np.float32)
for i,(c,g) in enumerate(zip(tmpl_class,tmpl_gain)):
    tmplScore[c,i]=g
    # A small inhibitory vote to broad rivals helps the template act discriminatively.
    if c in (0,8):
        for cc in (4,6,7): tmplScore[cc,i]-=0.15*g
    if c in (1,9):
        for cc in (4,6,7,8): tmplScore[cc,i]-=0.10*g
    if c in (4,7):
        for cc in (1,8,9): tmplScore[cc,i]-=0.08*g

# Output branch scales. Hand-normalize so no branch dominates too much.
rawW *= 0.48; colScore *= 0.72; edgeScore *= 0.30; partScore *= 0.55; tmplScore *= 1.05
# Broad semantic row multipliers: reduce very generic sky/boat votes and give more
# weight to animal/vehicle part templates that are more specific.
row_mult = np.array([0.72,0.88,1.22,1.38,1.35,1.38,1.05,1.25,0.88,1.18], np.float32)
rawW *= row_mult[:,None]; rawb *= row_mult
colScore *= row_mult[:,None]; edgeScore *= row_mult[:,None]
partScore *= row_mult[:,None]; tmplScore *= row_mult[:,None]
# Calibrate only against a synthetic uniform grey picture so brightness alone does not vote for
# airplanes/ships.  This uses no data; it is just a hand baseline for the rule system.
grey_raw = np.full((3,8,8), 0.5, np.float32).reshape(-1)
rawb -= rawW @ grey_raw
grey_col = np.maximum(colW[:, :, 0, 0] @ np.array([0.5,0.5,0.5], np.float32) + colb, 0)
colBias -= colScore @ np.tile(grey_col, 64)
grey_part = np.maximum(partW.sum(axis=(2,3)) @ np.array([0.5,0.5,0.5], np.float32) + partb, 0)
partBias -= partScore @ np.tile(grey_part, 64)
grey_tmpl = np.maximum(tmplW @ grey_raw + tmplb, 0)
tmplScore_b -= tmplScore @ grey_tmpl
# Class biases chosen by general breadth/specificity, then adjusted only coarsely by hand.
branch_bias = np.array([-6.10,7.65,9.20,5.80,10.00,3.45,1.95,12.05,0.85,-4.05],np.float32)
rawb += branch_bias

net={"layers":[
 {"name":"raw8","op":"avgpool","inputs":["image"],"size":[4,4],"stride":[4,4]},
 {"name":"raw_score","op":"dense","inputs":["raw8"],"out":10},
 {"name":"color","op":"conv","inputs":["image"],"out":Ccol,"kernel":[1,1],"stride":1,"padding":0},
 {"name":"color_relu","op":"relu","inputs":["color"]},
 {"name":"color8","op":"avgpool","inputs":["color_relu"],"size":[4,4],"stride":[4,4]},
 {"name":"color_score","op":"dense","inputs":["color8"],"out":10},
 {"name":"nimg","op":"norm","inputs":["image"]},
 {"name":"edge","op":"conv","inputs":["nimg"],"out":Ce,"kernel":[3,3],"stride":1,"padding":1},
 {"name":"edge_relu","op":"relu","inputs":["edge"]},
 {"name":"edge8","op":"avgpool","inputs":["edge_relu"],"size":[4,4],"stride":[4,4]},
 {"name":"edge_score","op":"dense","inputs":["edge8"],"out":10},
 {"name":"part","op":"conv","inputs":["image"],"out":Cp,"kernel":[5,5],"stride":1,"padding":2},
 {"name":"part_relu","op":"relu","inputs":["part"]},
 {"name":"part8","op":"maxpool","inputs":["part_relu"],"size":[4,4],"stride":[4,4]},
 {"name":"part_score","op":"dense","inputs":["part8"],"out":10},
 {"name":"tmpl","op":"dense","inputs":["raw8"],"out":H},
 {"name":"tmpl_relu","op":"relu","inputs":["tmpl"]},
 {"name":"tmpl_score","op":"dense","inputs":["tmpl_relu"],"out":10},
 {"name":"scores","op":"add","inputs":["raw_score","color_score","edge_score","part_score","tmpl_score"]}
]}
weights={
 'raw_score.W':rawW.astype(np.float32), 'raw_score.b':rawb.astype(np.float32),
 'color.W':colW.astype(np.float32), 'color.b':colb.astype(np.float32),
 'color_score.W':colScore.astype(np.float32), 'color_score.b':colBias.astype(np.float32),
 'edge.W':edgeW.astype(np.float32), 'edge.b':edgeb.astype(np.float32),
 'edge_score.W':edgeScore.astype(np.float32), 'edge_score.b':edgeBias.astype(np.float32),
 'part.W':partW.astype(np.float32), 'part.b':partb.astype(np.float32),
 'part_score.W':partScore.astype(np.float32), 'part_score.b':partBias.astype(np.float32),
 'tmpl.W':tmplW.astype(np.float32), 'tmpl.b':tmplb.astype(np.float32),
 'tmpl_score.W':tmplScore.astype(np.float32), 'tmpl_score.b':tmplScore_b.astype(np.float32),
}

def main(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out,'network.json'),'w') as f: json.dump(net,f,separators=(',',':'))
    np.savez(os.path.join(out,'weights.npz'), **weights)

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '/app/out')
