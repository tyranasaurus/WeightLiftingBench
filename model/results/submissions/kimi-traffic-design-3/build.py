"""Build a hand-designed German traffic-sign network.

All templates are drawn from general road-sign geometry by this file.  No data
files, font files, measurements, fitted parameters, or randomness are used.
"""
import json
import math
import os
import sys
import numpy as np

EPS = 1e-5
RED = (0.72, 0.035, 0.045)
DARKRED = (0.48, 0.025, 0.035)
WHITE = (0.93, 0.93, 0.90)
BLACK = (0.025, 0.03, 0.035)
GREY = (0.36, 0.37, 0.38)
BLUE = (0.025, 0.16, 0.62)
YELLOW = (0.98, 0.72, 0.035)
GREEN = (0.03, 0.38, 0.12)
BACK = (0.48, 0.48, 0.48)
SS = 4


class Drawing:
    """Tiny supersampled vector renderer, in logical 32 by 32 coordinates."""
    def __init__(self, scale=1.0, aspect=1.0, dx=0.0, dy=0.0):
        n = 32 * SS
        q = (np.arange(n, dtype=np.float64) + 0.5) / SS - 0.5
        x, y = np.meshgrid(q, q)
        self.x = (x - 15.5 - dx) / (scale * aspect) + 15.5
        self.y = (y - 15.5 - dy) / scale + 15.5
        self.im = np.empty((n, n, 3), dtype=np.float64)
        self.im[:] = BACK

    def paint(self, mask, color):
        self.im[mask] = color

    def disc(self, cx, cy, r, color):
        self.paint((self.x-cx)**2 + (self.y-cy)**2 <= r*r, color)

    def ellipse(self, cx, cy, rx, ry, color):
        self.paint(((self.x-cx)/rx)**2 + ((self.y-cy)/ry)**2 <= 1, color)

    def ring(self, cx, cy, rx, ry, width, color):
        d = np.sqrt(((self.x-cx)/rx)**2 + ((self.y-cy)/ry)**2)
        self.paint(np.abs(d-1) * min(rx, ry) <= width/2, color)

    def arc(self, cx, cy, rx, ry, width, a0, a1, color):
        x, y = self.x-cx, self.y-cy
        d = np.sqrt((x/rx)**2 + (y/ry)**2)
        a = np.arctan2(y/ry, x/rx)
        # Angles increase downwards, matching image coordinates.
        lo, hi = math.radians(a0), math.radians(a1)
        if hi-lo >= 2*math.pi-1e-8:
            sector = np.ones(a.shape,dtype=bool)
        else:
            span = (hi-lo) % (2*math.pi)
            sector = ((a-lo) % (2*math.pi)) <= span
        self.paint((np.abs(d-1)*min(rx, ry) <= width/2) & sector, color)

    def polygon(self, points, color):
        x, y = self.x, self.y
        inside = np.zeros(x.shape, dtype=bool)
        for i in range(len(points)):
            ax, ay = points[i-1]
            bx, by = points[i]
            crossing = ((ay > y) != (by > y)) & (x < (bx-ax)*(y-ay)/(by-ay+1e-20)+ax)
            inside ^= crossing
        self.paint(inside, color)

    def line(self, a, b, width, color):
        ax, ay = a; bx, by = b
        vx, vy = bx-ax, by-ay
        t = ((self.x-ax)*vx + (self.y-ay)*vy) / (vx*vx+vy*vy+1e-20)
        t = np.clip(t, 0, 1)
        d2 = (self.x-(ax+t*vx))**2 + (self.y-(ay+t*vy))**2
        self.paint(d2 <= (width/2)**2, color)

    def polyline(self, points, width, color):
        for a, b in zip(points[:-1], points[1:]):
            self.line(a, b, width, color)

    def output(self):
        return self.im.reshape(32, SS, 32, SS, 3).mean((1, 3)).transpose(2, 0, 1).astype(np.float32)


def upright(d, fill=WHITE, border=RED):
    d.polygon([(16,2.2),(29.3,27.3),(2.7,27.3)], border)
    d.polygon([(16,6.1),(25.4,24.3),(6.6,24.3)], fill)


def inverted(d):
    d.polygon([(2.5,4.0),(29.5,4.0),(16,29.3)], RED)
    d.polygon([(6.3,7.8),(25.7,7.8),(16,25.0)], WHITE)


def circle(d, interior=WHITE, rim=RED):
    d.disc(16,16,13.0,rim)
    d.disc(16,16,10.1,interior)


def blue_circle(d):
    d.disc(16,16,13.0,WHITE)
    d.disc(16,16,12.4,BLUE)


def digit(d, n, cx, cy, w, h, width=1.35, color=BLACK):
    """A simple rounded DIN-like stroke digit, not a font or image glyph."""
    left, right = cx-w*0.38, cx+w*0.38
    top, bot = cy-h*0.47, cy+h*0.47
    mid = cy
    if n == 0:
        d.ring(cx,cy,w*0.38,h*0.47,width,color)
    elif n == 1:
        d.polyline([(cx-w*.25,top+h*.18),(cx,top),(cx,bot)],width,color)
        d.line((cx-w*.25,bot),(cx+w*.28,bot),width,color)
    elif n == 2:
        pts=[]
        for a in np.linspace(190,355,13):
            aa=math.radians(a); pts.append((cx+w*.39*math.cos(aa),cy-h*.20+h*.28*math.sin(aa)))
        pts += [(left,bot),(right,bot)]
        d.polyline(pts,width,color)
    elif n == 3:
        d.arc(cx-w*.03,cy-h*.23,w*.38,h*.24,width,-135,135,color)
        d.arc(cx-w*.03,cy+h*.23,w*.38,h*.24,width,-135,135,color)
    elif n == 4:
        d.polyline([(right,top),(right,bot)],width,color)
        d.polyline([(right,top),(left,mid+h*.10),(right+w*.08,mid+h*.10)],width,color)
    elif n == 5:
        d.polyline([(right,top),(left,top),(left,mid-h*.03),(cx,mid-h*.06)],width,color)
        d.arc(cx-w*.02,cy+h*.20,w*.40,h*.27,width,-135,135,color)
    elif n == 6:
        d.arc(cx,cy+h*.18,w*.39,h*.29,width,0,360,color)
        d.polyline([(right,top),(cx-w*.05,top-h*.01),(left,cy),(left,cy+h*.20)],width,color)
    elif n == 7:
        d.polyline([(left,top),(right,top),(cx-w*.08,bot)],width,color)
    elif n == 8:
        d.ring(cx,cy-h*.22,w*.36,h*.245,width,color)
        d.ring(cx,cy+h*.23,w*.39,h*.27,width,color)
    elif n == 9:
        d.ring(cx,cy-h*.18,w*.39,h*.29,width,0,360,color)
        d.polyline([(right,cy-h*.15),(right,cy+h*.15),(cx-w*.05,bot)],width,color)


def number(d, value, cy=16.0):
    s = str(value)
    if len(s) == 2:
        w, h, off = 6.4, 11.5, 3.55
        for c, x in zip(s, (16-off,16+off)):
            digit(d,int(c),x,cy,w,h,1.45)
    else:
        w, h = 4.6, 11.3
        for c, x in zip(s, (11.2,16.0,20.8)):
            digit(d,int(c),x,cy,w,h,1.18)


def car(d, cx, cy, color, truck=False):
    if truck:
        d.polygon([(cx-4.4,cy-3.4),(cx+1.0,cy-3.4),(cx+1.0,cy+2.4),(cx-4.4,cy+2.4)],color)
        d.polygon([(cx+1,cy-1.5),(cx+3.2,cy-1.5),(cx+4.5,cy+.2),(cx+4.5,cy+2.4),(cx+1,cy+2.4)],color)
        d.disc(cx-2.5,cy+2.5,1.0,BLACK); d.disc(cx+2.8,cy+2.5,1.0,BLACK)
    else:
        d.polygon([(cx-4,cy-1.3),(cx-2.5,cy-3.2),(cx+2.1,cy-3.2),(cx+3.7,cy-1.2),(cx+4.2,cy+2),(cx-4.2,cy+2)],color)
        d.disc(cx-2.5,cy+2.1,.9,BLACK); d.disc(cx+2.5,cy+2.1,.9,BLACK)


def person(d, cx, cy, size=1.0, color=BLACK, child=False):
    d.disc(cx,cy-5.0*size,1.25*size,color)
    d.polyline([(cx,cy-3.4*size),(cx-.4*size,cy+1.2*size)],1.65*size,color)
    d.polyline([(cx-.1*size,cy-2.3*size),(cx-2.6*size,cy-.2*size)],1.25*size,color)
    d.polyline([(cx-.1*size,cy-2.2*size),(cx+2.4*size,cy-.9*size)],1.25*size,color)
    d.polyline([(cx-.4*size,cy+1.0*size),(cx-2.2*size,cy+5.3*size)],1.5*size,color)
    d.polyline([(cx-.4*size,cy+1.0*size),(cx+2.2*size,cy+4.8*size)],1.5*size,color)


def arrow_head(d, tip, direction, size, color=WHITE):
    dx,dy=direction
    norm=math.hypot(dx,dy); dx/=norm; dy/=norm
    px,py=-dy,dx
    bx,by=tip[0]-dx*size,tip[1]-dy*size
    d.polygon([tip,(bx+px*size*.62,by+py*size*.62),(bx-px*size*.62,by-py*size*.62)],color)


def up_arrow(d, color=WHITE, x=16, bottom=23, top=8, width=2.4):
    d.line((x,bottom),(x,top+2),width,color)
    arrow_head(d,(x,top),(0,-1),4.0,color)


def diagonal_stripes(d, color=BLACK, spacing=4.0, width=1.5):
    inside=(d.x-16)**2+(d.y-16)**2 <= 11.8**2
    stripe=np.zeros(inside.shape,bool)
    for off in np.arange(-16,17,spacing):
        stripe |= np.abs((d.x+d.y-32)-off) < width/2
    d.paint(inside & stripe,color)


def make_sign(k, scale=1.0, aspect=1.0, dx=0.0, dy=0.0):
    d=Drawing(scale,aspect,dx,dy)
    if 0 <= k <= 8:
        if k == 6:
            circle(d,WHITE,BLACK)
            number(d,80)
            diagonal_stripes(d,BLACK,4.0,1.15)
        else:
            circle(d)
            number(d,[20,30,50,60,70,80,80,100,120][k])
    elif k in (9,10):
        circle(d)
        car(d,10.5,16,BLACK,k==10)
        car(d,21.5,16,RED,k==10)
    elif k == 11:
        upright(d,YELLOW,BLACK)
        up_arrow(d,BLACK,16,22.5,9.0,2.1)
        d.line((10.5,18.5),(21.5,18.5),2.0,BLACK)
    elif k == 12:
        d.polygon([(16,2),(30,16),(16,30),(2,16)],WHITE)
        d.polygon([(16,4),(28,16),(16,28),(4,16)],BLACK)
        d.polygon([(16,5.5),(26.5,16),(16,26.5),(5.5,16)],YELLOW)
    elif k == 13:
        inverted(d)
    elif k == 14:
        pts=[(16+13*math.cos(math.radians(22.5+i*45)),16+13*math.sin(math.radians(22.5+i*45))) for i in range(8)]
        d.polygon(pts,DARKRED)
        # Compact, hand-drawn STOP lettering.
        for ch,cx in zip('STOP',(9.1,13.6,18.1,22.7)):
            if ch=='S':
                d.polyline([(cx+1.4,12),(cx-1.1,12),(cx-1.4,15.5),(cx+1.2,16.2),(cx+1.1,20),(cx-1.4,20)],.85,WHITE)
            elif ch=='T':
                d.line((cx-1.5,12),(cx+1.5,12),.9,WHITE); d.line((cx,12),(cx,20),.9,WHITE)
            elif ch=='O':
                d.ring(cx,16,1.55,4.0,.9,WHITE)
            else:
                d.polyline([(cx-1.1,20),(cx-1.1,12),(cx+1.2,12),(cx+1.4,15.5),(cx-1.1,15.8)],.9,WHITE)
    elif k == 15:
        circle(d)
    elif k == 16:
        circle(d); car(d,16,16,BLACK,True)
    elif k == 17:
        d.disc(16,16,13,RED)
        d.polygon([(6,13.7),(26,13.7),(26,18.3),(6,18.3)],WHITE)
    elif 18 <= k <= 31:
        upright(d)
        if k == 18:
            d.line((16,9),(16,18.5),2.2,BLACK); d.disc(16,22,1.25,BLACK)
        elif k in (19,20):
            pts=[(18,23),(18,17),(16.5,13.5),(13,12),(10.5,12)]
            if k==20: pts=[(32-x,y) for x,y in pts]
            d.polyline(pts,2.5,BLACK)
        elif k == 21:
            d.polyline([(18.5,23),(18.5,19),(12.5,15.5),(12.5,10)],2.5,BLACK)
        elif k == 22:
            d.line((9,22),(24,22),1.7,BLACK)
            d.arc(12.5,21.5,3.0,3.2,1.8,180,360,BLACK)
            d.arc(20,21.5,3.0,3.2,1.8,180,360,BLACK)
        elif k == 23:
            d.polygon([(13,10),(19,10),(20.5,16),(11.5,16)],BLACK)
            d.disc(13,15.5,1,BLACK); d.disc(19,15.5,1,BLACK)
            d.polyline([(12,18),(10.5,20),(12,23)],1.2,BLACK)
            d.polyline([(19,18),(21,20),(19.5,23)],1.2,BLACK)
        elif k == 24:
            d.polyline([(11,10),(11,23)],2.5,BLACK)
            d.polyline([(22,10),(17.5,16),(17.5,23)],2.5,BLACK)
        elif k == 25:
            person(d,14,15,0.9,BLACK)
            d.line((18,12),(22,23),1.1,BLACK)
            d.polygon([(20,22),(25,22),(24,24),(19,24)],BLACK)
        elif k == 26:
            d.polygon([(12.5,8),(19.5,8),(19.5,24),(12.5,24)],BLACK)
            d.disc(16,11,1.55,RED); d.disc(16,16,1.55,YELLOW); d.disc(16,21,1.55,GREEN)
        elif k == 27:
            person(d,16,15.5,1.2,BLACK)
        elif k == 28:
            person(d,12.5,15.5,.88,BLACK); person(d,19.2,16.2,.75,BLACK)
            d.line((13.5,14),(18,15),1.0,BLACK)
        elif k == 29:
            d.ring(10.5,20,3.1,3.1,1.0,BLACK); d.ring(21.5,20,3.1,3.1,1.0,BLACK)
            d.polyline([(10.5,20),(14,14),(17.5,20),(10.5,20),(15.5,17),(21.5,20),(18.5,13)],1.0,BLACK)
            d.line((13,13),(16,13),1.0,BLACK); d.disc(17.5,9.5,1.15,BLACK)
            d.polyline([(17,11),(15.5,14),(18.5,16)],1.3,BLACK)
        elif k == 30:
            for a in (0,60,120):
                aa=math.radians(a); vx,vy=math.cos(aa),math.sin(aa)
                p1=(16-6*vx,16-6*vy); p2=(16+6*vx,16+6*vy)
                d.line(p1,p2,1.15,BLACK)
                for sign in (-1,1):
                    bx,by=16+sign*3.8*vx,16+sign*3.8*vy
                    for side in (-1,1):
                        bb=aa+math.pi+side*.75
                        d.line((bx,by),(bx+2*math.cos(bb),by+2*math.sin(bb)),.9,BLACK)
        elif k == 31:
            d.ellipse(15.5,17,5.0,2.8,BLACK)
            d.polyline([(12,18),(11,23),(10,23)],1.2,BLACK)
            d.polyline([(17,18),(18,23),(19,23)],1.2,BLACK)
            d.polyline([(19,17),(20.5,12),(22,11)],2.0,BLACK)
            d.ellipse(22.2,10.5,2.0,1.25,BLACK)
            d.polyline([(21.5,9.5),(20,7.5),(19.5,6.5)],.8,BLACK)
            d.polyline([(22,9),(23,7),(24,6.5)],.8,BLACK)
    elif k == 32:
        circle(d,WHITE,BLACK); diagonal_stripes(d,BLACK,4.0,1.5)
    elif k in (33,34):
        blue_circle(d)
        pts=[(13,23),(13,15),(15,11),(21,11)]
        if k==34: pts=[(32-x,y) for x,y in pts]
        d.polyline(pts,2.5,WHITE)
        tip=pts[-1]; arrow_head(d,tip,(1 if k==33 else -1,0),4.0,WHITE)
    elif k == 35:
        blue_circle(d); up_arrow(d)
    elif k in (36,37):
        blue_circle(d); up_arrow(d,WHITE,16,23,8,2.3)
        if k==36:
            d.line((16,17),(23,17),2.3,WHITE); arrow_head(d,(24,17),(1,0),3.8,WHITE)
        else:
            d.line((16,17),(9,17),2.3,WHITE); arrow_head(d,(8,17),(-1,0),3.8,WHITE)
    elif k in (38,39):
        blue_circle(d)
        tip=(22,22) if k==38 else (10,22)
        start=(10,9) if k==38 else (22,9)
        d.line(start,tip,3.0,WHITE)
        arrow_head(d,tip,(1 if k==38 else -1,1),4.5,WHITE)
    elif k == 40:
        blue_circle(d)
        for a in (-90,30,150):
            aa=math.radians(a)
            cx,cy=16+4.8*math.cos(aa),16+4.8*math.sin(aa)
            d.arc(16,16,6.3,6.3,2.0,a-50,a+35,WHITE)
            tip=(16+6.3*math.cos(aa+.65),16+6.3*math.sin(aa+.65))
            arrow_head(d,tip,(-math.sin(aa+.65),math.cos(aa+.65)),3.2,WHITE)
    elif k in (41,42):
        circle(d,WHITE,BLACK)
        car(d,11,16,BLACK,k==42); car(d,21,16,GREY,k==42)
        diagonal_stripes(d,GREY,4.0,1.35)
    else:
        raise ValueError(k)
    return d.output()


# A fixed set of geometric variants gives tolerance to crop size and centring.
# These are independent drawings, not augmentations aggregated into a prototype.
VARIANTS = [
    (1.00,1.00,0,0), (0.85,1.00,0,0), (1.14,1.00,0,0),
    (1.00,1.00,-1.2,0), (1.00,1.00,1.2,0),
    (1.00,1.00,0,-1.2), (1.00,1.00,0,1.2),
]


def norm(x):
    mu=x.mean(axis=(1,2),keepdims=True)
    return (x-mu)/np.sqrt(((x-mu)**2).mean(axis=(1,2),keepdims=True)+EPS)


def avgpool2(x):
    return x.reshape(x.shape[0],16,2,16,2).mean((2,4))


def conv3(x,kernels):
    xp=np.pad(x,((0,0),(1,1),(1,1)))
    out=[]
    for ker in kernels:
        y=np.zeros(x.shape,dtype=np.float64)
        for dy in range(3):
            for dx in range(3):
                y += ker[dy,dx]*xp[:,dy:dy+32,dx:dx+32]
        out.append(y)
    return np.stack(out).astype(np.float32)


SOBEL=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64)/4
ORIENTS=[]
for a in np.arange(8)*math.pi/8:
    ORIENTS.append(math.cos(a)*SOBEL+math.sin(a)*SOBEL.T)


def features(im):
    """Numpy equivalent of the network's fixed preprocessing, for one drawing."""
    rgb=norm(im)
    gray=(np.array([.299,.587,.114],dtype=np.float32)[:,None,None]*im).sum(0,keepdims=True)
    gray=norm(gray)
    gp=norm(avgpool2(gray))
    rp=norm(avgpool2(rgb))
    r,g,b=im
    cp=np.stack([r-(g+b)/2,b-(r+g)/2,(r+g)/2-b])
    cp=norm(cp)
    cp=np.maximum(cp,0)
    cp=norm(avgpool2(cp))
    dirs=conv3(gray,ORIENTS)
    hog=norm(avgpool2(np.abs(dirs)))
    return rp,gp,cp,hog


def normalized_template(x, mask=None):
    """Normalize one rendered template only; never pool information across templates."""
    z=x.astype(np.float64).copy()
    if mask is not None:
        z=z*mask[None]
        count=float(mask.sum())
        for c in range(z.shape[0]):
            vals=z[c][mask]
            mu=vals.mean()
            sd=math.sqrt(float(((vals-mu)**2).mean())+EPS)
            z[c]=np.where(mask,(z[c]-mu)/sd,0)
        z/=math.sqrt(z.shape[0]*count)
    else:
        z=norm(z.astype(np.float32)).astype(np.float64)
        z/=math.sqrt(z.size)
    return z.astype(np.float32).ravel()


def normalized_nonnegative_template(x, mask=None):
    """Unit-overlap template for a nonnegative feature map, scaled by its dimensions."""
    z=x.astype(np.float64).copy()
    if mask is not None:
        z=z*mask[None]
        count=z.shape[0]*float(mask.sum())
    else:
        count=z.size
    length=math.sqrt(float((z*z).sum()))
    if length>1e-12:
        z/=length
    # A normalized pre-pooling map has unit variance; 0.45 is a hand-chosen
    # post-ReLU/pooling response scale, not a measurement on any images.
    z/=math.sqrt(count)*.45
    return z.astype(np.float32).ravel()


def normalized_silhouette(x,k,mask):
    """A single drawing's dark (or white-on-blue) interior silhouette."""
    z=np.zeros_like(x,dtype=np.float64)
    channel=1 if (33<=k<=40 or k==14) else 0
    z[channel]=x[channel]*mask
    length=math.sqrt(float((z*z).sum()))
    if length>1e-12:
        z/=length
    z/=math.sqrt(float(mask.sum()))*.35
    return z.astype(np.float32).ravel()


def inner_mask(k):
    yy,xx=np.mgrid[:16,:16]
    x=xx+.5; y=yy+.5
    if 18 <= k <= 31 or k==11:
        # Interior of an upright triangular warning sign at half resolution.
        return (y>=3.2)&(y<=12.2)&(x>=8-(y-3.2)*.52)&(x<=8+(y-3.2)*.52)
    if k==13:
        return (y>=4)&(y<=12)&(x>=8-(12-y)*.52)&(x<=8+(12-y)*.52)
    if k==12:
        return np.abs(x-8)+np.abs(y-8)<=5.4
    if k==14:
        return np.abs(x-8)<=5.5
    return (x-8)**2+(y-8)**2<=5.6**2


def add_template_branch(layers,weights,name,input_name,flat_templates,branch_weights,nonnegative=False):
    nv=len(VARIANTS); nt=43*nv
    layers.append({'name':name+'_match','op':'dense','inputs':[input_name],'out':nt})
    weights[name+'_match.W']=np.stack(flat_templates).astype(np.float32)
    weights[name+'_match.b']=np.zeros(nt,np.float32)
    layers.append({'name':name+'_grid','op':'reshape','inputs':[name+'_match'],'shape':[43,nv,1]})
    layers.append({'name':name+'_max','op':'maxpool','inputs':[name+'_grid'],'size':[nv,1],'stride':[nv,1]})
    layers.append({'name':name+'_flat','op':'reshape','inputs':[name+'_max'],'shape':[43]})
    score_input=name+'_flat'
    if nonnegative:
        layers.append({'name':name+'_positive','op':'relu','inputs':[score_input]})
        score_input=name+'_positive'
    layers.append({'name':name+'_score','op':'dense','inputs':[score_input],'out':43})
    weights[name+'_score.W']=np.diag(np.asarray(branch_weights,dtype=np.float32))
    weights[name+'_score.b']=np.zeros(43,np.float32)
    return name+'_score'


def build(out):
    layers=[]; weights={}
    layers.append({'name':'nrgb','op':'norm','inputs':['image']})
    layers.append({'name':'gray','op':'conv','inputs':['image'],'out':1,'kernel':[1,1],'stride':1,'padding':0})
    weights['gray.W']=np.array([.299,.587,.114],np.float32).reshape(1,3,1,1)
    weights['gray.b']=np.zeros(1,np.float32)
    layers.append({'name':'ngray','op':'norm','inputs':['gray']})
    layers.append({'name':'gpool','op':'avgpool','inputs':['ngray'],'size':[2,2],'stride':[2,2]})
    layers.append({'name':'gn','op':'norm','inputs':['gpool']})
    layers.append({'name':'rpool','op':'avgpool','inputs':['nrgb'],'size':[2,2],'stride':[2,2]})
    layers.append({'name':'rn','op':'norm','inputs':['rpool']})
    layers.append({'name':'color','op':'conv','inputs':['image'],'out':3,'kernel':[1,1],'stride':1,'padding':0})
    weights['color.W']=np.array([[1,-.5,-.5],[-.5,-.5,1],[.5,.5,-1]],np.float32).reshape(3,3,1,1)
    weights['color.b']=np.zeros(3,np.float32)
    layers.append({'name':'color_norm','op':'norm','inputs':['color']})
    layers.append({'name':'color_relu','op':'relu','inputs':['color_norm']})
    layers.append({'name':'cpool','op':'avgpool','inputs':['color_relu'],'size':[2,2],'stride':[2,2]})
    layers.append({'name':'cn','op':'norm','inputs':['cpool']})
    layers.append({'name':'directions','op':'conv','inputs':['ngray'],'out':16,'kernel':[3,3],'stride':1,'padding':1})
    dw=[]
    for f in ORIENTS:
        dw.extend([f,-f])
    weights['directions.W']=np.asarray(dw,np.float32).reshape(16,1,3,3)
    weights['directions.b']=np.zeros(16,np.float32)
    layers.append({'name':'dir_relu','op':'relu','inputs':['directions']})
    layers.append({'name':'absolute','op':'conv','inputs':['dir_relu'],'out':8,'kernel':[1,1],'stride':1,'padding':0})
    aw=np.zeros((8,16,1,1),np.float32)
    for i in range(8): aw[i,2*i:2*i+2,0,0]=1
    weights['absolute.W']=aw; weights['absolute.b']=np.zeros(8,np.float32)
    layers.append({'name':'hpool','op':'avgpool','inputs':['absolute'],'size':[2,2],'stride':[2,2]})
    layers.append({'name':'hn','op':'norm','inputs':['hpool']})

    banks={n:[] for n in ('rgb','gray','color','hog','inner')}
    for k in range(43):
        mask=inner_mask(k)
        for variant in VARIANTS:
            im=make_sign(k,*variant)
            rp,gp,cp,hp=features(im)
            banks['rgb'].append(normalized_template(rp))
            banks['gray'].append(normalized_template(gp))
            banks['color'].append(normalized_template(cp))
            banks['hog'].append(normalized_template(hp))
            banks['inner'].append(normalized_template(hp,mask))

    # Hand-selected relative importance of appearance, outline and pictogram evidence.
    rgbw=np.full(43,.20,np.float32)
    grayw=np.full(43,.55,np.float32)
    colorw=np.full(43,.35,np.float32)
    hogw=np.full(43,.55,np.float32)
    innerw=np.full(43,1.10,np.float32)
    # Outer colour is especially diagnostic for the yellow priority signs and blue mandates.
    colorw[[11,12]]=0.65
    colorw[33:41]=0.65
    # Fine internal geometry deserves greater weight in the circular regulatory families.
    innerw[:11]=1.30
    innerw[33:43]=1.30
    outputs=[]
    outputs.append(add_template_branch(layers,weights,'rgb','rn',banks['rgb'],rgbw))
    outputs.append(add_template_branch(layers,weights,'gray','gn',banks['gray'],grayw))
    outputs.append(add_template_branch(layers,weights,'color','cn',banks['color'],colorw))
    outputs.append(add_template_branch(layers,weights,'hog','hn',banks['hog'],hogw))
    outputs.append(add_template_branch(layers,weights,'inner','hn',banks['inner'],innerw,nonnegative=True))
    layers.append({'name':'scores','op':'add','inputs':outputs})
    os.makedirs(out,exist_ok=True)
    with open(os.path.join(out,'network.json'),'w') as f:
        json.dump({'layers':layers},f,indent=2)
    np.savez(os.path.join(out,'weights.npz'),**weights)


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('usage: python3 build.py OUTPUT_DIRECTORY')
    build(sys.argv[1])
