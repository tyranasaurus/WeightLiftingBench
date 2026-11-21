import json, math, sys
from pathlib import Path
import numpy as np

WHITE=np.array([1.0,1.0,1.0],np.float32)
BLACK=np.array([0.06,0.06,0.06],np.float32)
RED=np.array([0.88,0.10,0.10],np.float32)
BLUE=np.array([0.16,0.42,0.92],np.float32)
YELLOW=np.array([0.95,0.80,0.12],np.float32)
BG=np.array([0.50,0.50,0.50],np.float32)
EPS=1e-5
SCALES=[0.82,0.90,1.00]
RGB_SCALE=1.0
GRAY_SCALE=1.15
PAD=1
POOL=2*PAD+1

FONT={
 '0':['01110','10001','10011','10101','11001','10001','01110'],
 '1':['00100','01100','00100','00100','00100','00100','01110'],
 '2':['01110','10001','00001','00010','00100','01000','11111'],
 '3':['11110','00001','00001','01110','00001','00001','11110'],
 '5':['11111','10000','10000','11110','00001','00001','11110'],
 '6':['01110','10000','10000','11110','10001','10001','01110'],
 '7':['11111','00001','00010','00100','01000','01000','01000'],
 '8':['01110','10001','10001','01110','10001','10001','01110'],
 'S':['01111','10000','10000','01110','00001','00001','11110'],
 'T':['11111','00100','00100','00100','00100','00100','00100'],
 'O':['01110','10001','10001','10001','10001','10001','01110'],
 'P':['11110','10001','10001','11110','10000','10000','10000'],
 '!':['00100','00100','00100','00100','00100','00000','00100'],
}


def hi_grid(size=32, ss=8):
    t=(np.arange(size*ss)+0.5)/(size*ss)*2-1
    y,x=np.meshgrid(t,t,indexing='ij')
    return x,y,ss


def downsample(mask, ss):
    n=mask.shape[0]//ss
    if mask.ndim==2:
        return mask.reshape(n,ss,n,ss).mean((1,3))
    c=mask.shape[2]
    return mask.reshape(n,ss,n,ss,c).mean((1,3))


def dist_to_segment(x,y,x1,y1,x2,y2):
    vx,vy=x2-x1,y2-y1
    wx,wy=x-x1,y-y1
    vv=vx*vx+vy*vy+1e-12
    t=np.clip((wx*vx+wy*vy)/vv,0,1)
    px=x1+t*vx
    py=y1+t*vy
    return np.sqrt((x-px)**2+(y-py)**2)


def stroke_polyline(x,y,pts,width):
    d=np.full(x.shape,1e9,np.float32)
    for (x1,y1),(x2,y2) in zip(pts[:-1],pts[1:]):
        d=np.minimum(d, dist_to_segment(x,y,x1,y1,x2,y2))
    return (d<=width).astype(np.float32)


def fill_convex_poly(x,y,pts):
    pts=np.asarray(pts,np.float32)
    area=0.0
    for i in range(len(pts)):
        x1,y1=pts[i]
        x2,y2=pts[(i+1)%len(pts)]
        area += x1*y2-x2*y1
    sign=1.0 if area>0 else -1.0
    ok=np.ones(x.shape,bool)
    for i in range(len(pts)):
        x1,y1=pts[i]
        x2,y2=pts[(i+1)%len(pts)]
        c=sign*((x-x1)*(y2-y1)-(y-y1)*(x2-x1))
        ok &= c>=-1e-9
    return ok.astype(np.float32)


def fill_circle(x,y,r,cx=0,cy=0):
    return (((x-cx)**2+(y-cy)**2)<=r*r).astype(np.float32)


def fill_ellipse(x,y,rx,ry,cx=0,cy=0,angle=0):
    ca,sa=math.cos(angle),math.sin(angle)
    xr=(x-cx)*ca+(y-cy)*sa
    yr=-(x-cx)*sa+(y-cy)*ca
    return ((xr/rx)**2+(yr/ry)**2<=1).astype(np.float32)


def ring(x,y,r0,r1,cx=0,cy=0):
    d=(x-cx)**2+(y-cy)**2
    return ((d>=r0*r0)&(d<=r1*r1)).astype(np.float32)


def draw(canvas, mask, color):
    if mask.ndim==2:
        mask=mask[...,None]
    canvas[:]=canvas*(1-mask)+color*mask


def triangle_pts(scale=1.0, upside=False):
    pts=[(0,-0.82*scale),(0.74*scale,0.46*scale),(-0.74*scale,0.46*scale)]
    if upside:
        pts=[(a,-b) for a,b in pts]
    return pts


def bitmap_mask(x,y,text,cx,cy,sx,sy):
    rows=max(len(FONT[ch]) for ch in text)
    cols=sum(len(FONT[ch][0]) for ch in text)+len(text)-1
    mask=np.zeros(x.shape,np.float32)
    x0=cx-cols*sx/2
    y0=cy-rows*sy/2
    cur=x0
    for ch in text:
        pat=FONT[ch]
        h=len(pat)
        w=len(pat[0])
        for r in range(h):
            for c in range(w):
                if pat[r][c]=='1':
                    cell=fill_convex_poly(x,y,[(cur+c*sx,y0+r*sy),(cur+(c+1)*sx,y0+r*sy),(cur+(c+1)*sx,y0+(r+1)*sy),(cur+c*sx,y0+(r+1)*sy)])
                    mask=np.maximum(mask,cell)
        cur += w*sx + sx
    return mask


def arrow_straight(x,y,scale=1.0):
    shaft=fill_convex_poly(x,y,[(-0.08*scale,0.50*scale),(0.08*scale,0.50*scale),(0.08*scale,-0.05*scale),(-0.08*scale,-0.05*scale)])
    head=fill_convex_poly(x,y,[(-0.26*scale,-0.05*scale),(0,-0.55*scale),(0.26*scale,-0.05*scale)])
    return np.maximum(shaft,head)


def arrow_turn(x,y,right=True,scale=1.0):
    s=1 if right else -1
    shaft=stroke_polyline(x,y,[(0.0,0.55*scale),(0.0,-0.18*scale),(0.38*s,-0.18*scale)],0.10*scale)
    head=fill_convex_poly(x,y,[(0.38*s,-0.38*scale),(0.60*s,-0.18*scale),(0.38*s,0.02*scale)])
    return np.maximum(shaft,head)


def arrow_keep(x,y,right=True,scale=1.0):
    s=1 if right else -1
    shaft=stroke_polyline(x,y,[(-0.30*s,-0.45*scale),(-0.04*s,-0.18*scale),(0.16*s,0.02*scale),(0.32*s,0.18*scale)],0.10*scale)
    head=fill_convex_poly(x,y,[(0.08*s,-0.08*scale),(0.42*s,0.18*scale),(0.16*s,0.52*scale)])
    return np.maximum(shaft,head)


def arrow_fork(x,y,right=True,scale=1.0):
    s=1 if right else -1
    base=stroke_polyline(x,y,[(0,0.52*scale),(0,-0.05*scale)],0.10*scale)
    branch=stroke_polyline(x,y,[(0,-0.05*scale),(0.34*s,-0.05*scale)],0.10*scale)
    head1=fill_convex_poly(x,y,[(-0.24*scale,-0.05*scale),(0,-0.52*scale),(0.24*scale,-0.05*scale)])
    head2=fill_convex_poly(x,y,[(0.18*s,-0.24*scale),(0.52*s,-0.05*scale),(0.18*s,0.14*scale)])
    return np.maximum.reduce([base,branch,head1,head2])


def roundabout_symbol(x,y,scale=1.0):
    mask=np.zeros(x.shape,np.float32)
    for ang in [0,2*math.pi/3,4*math.pi/3]:
        ca,sa=math.cos(ang),math.sin(ang)
        xr=x*ca+y*sa
        yr=-x*sa+y*ca
        arc=stroke_polyline(xr,yr,[(-0.06*scale,0.42*scale),(0.26*scale,0.22*scale),(0.22*scale,-0.10*scale)],0.08*scale)
        head=fill_convex_poly(xr,yr,[(0.12*scale,-0.02*scale),(0.36*scale,-0.22*scale),(0.42*scale,0.06*scale)])
        mask=np.maximum(mask,np.maximum(arc,head))
    return mask


def truck_symbol(x,y,scale=1.0):
    body=fill_convex_poly(x,y,[(-0.42*scale,0.05*scale),(0.16*scale,0.05*scale),(0.16*scale,0.28*scale),(-0.42*scale,0.28*scale)])
    cab=fill_convex_poly(x,y,[(0.16*scale,0.05*scale),(0.42*scale,0.05*scale),(0.42*scale,0.28*scale),(0.28*scale,0.28*scale),(0.16*scale,0.18*scale)])
    wheels=np.maximum(fill_circle(x,y,0.08*scale,-0.22*scale,0.33*scale),fill_circle(x,y,0.08*scale,0.22*scale,0.33*scale))
    return np.maximum.reduce([body,cab,wheels])


def car_symbol(x,y,scale=1.0):
    base=fill_convex_poly(x,y,[(-0.36*scale,0.14*scale),(0.36*scale,0.14*scale),(0.28*scale,0.30*scale),(-0.28*scale,0.30*scale)])
    roof=fill_convex_poly(x,y,[(-0.18*scale,-0.02*scale),(0.08*scale,-0.02*scale),(0.22*scale,0.14*scale),(-0.30*scale,0.14*scale)])
    wheels=np.maximum(fill_circle(x,y,0.07*scale,-0.18*scale,0.33*scale),fill_circle(x,y,0.07*scale,0.18*scale,0.33*scale))
    return np.maximum.reduce([base,roof,wheels])


def person_symbol(x,y,scale=1.0,step=-1):
    head=fill_circle(x,y,0.10*scale,0,-0.34*scale)
    body=stroke_polyline(x,y,[(0,-0.22*scale),(0,0.06*scale)],0.06*scale)
    leg1=stroke_polyline(x,y,[(0,0.06*scale),(-0.18*scale,0.34*scale)],0.06*scale)
    leg2=stroke_polyline(x,y,[(0,0.06*scale),(0.16*scale,0.34*scale if step>0 else 0.10*scale)],0.06*scale)
    arm1=stroke_polyline(x,y,[(0,-0.06*scale),(-0.18*scale,0.02*scale)],0.05*scale)
    arm2=stroke_polyline(x,y,[(0,-0.06*scale),(0.20*scale,0.02*scale)],0.05*scale)
    return np.maximum.reduce([head,body,leg1,leg2,arm1,arm2])


def bike_symbol(x,y,scale=1.0):
    w1=ring(x,y,0.16*scale,0.23*scale,-0.24*scale,0.22*scale)
    w2=ring(x,y,0.16*scale,0.23*scale,0.24*scale,0.22*scale)
    frame=np.maximum.reduce([
        stroke_polyline(x,y,[(-0.24*scale,0.22*scale),(-0.02*scale,0.02*scale),(0.06*scale,0.22*scale),(-0.12*scale,0.22*scale)],0.05*scale),
        stroke_polyline(x,y,[(-0.02*scale,0.02*scale),(0.24*scale,0.22*scale)],0.05*scale),
        stroke_polyline(x,y,[(-0.02*scale,0.02*scale),(0.06*scale,-0.16*scale)],0.05*scale),
        stroke_polyline(x,y,[(0.06*scale,-0.16*scale),(0.20*scale,-0.16*scale)],0.05*scale),
        stroke_polyline(x,y,[(-0.06*scale,-0.16*scale),(0.06*scale,-0.16*scale)],0.05*scale),
    ])
    return np.maximum.reduce([w1,w2,frame])


def snowflake_symbol(x,y,scale=1.0):
    mask=np.zeros(x.shape,np.float32)
    for ang in [0,math.pi/3,2*math.pi/3]:
        ca,sa=math.cos(ang),math.sin(ang)
        x1,y1=-0.34*ca*scale,-0.34*sa*scale
        x2,y2=0.34*ca*scale,0.34*sa*scale
        mask=np.maximum(mask,stroke_polyline(x,y,[(x1,y1),(x2,y2)],0.05*scale))
    return mask


def traffic_light_symbol(x,y,scale=1.0):
    box=fill_convex_poly(x,y,[(-0.14*scale,-0.34*scale),(0.14*scale,-0.34*scale),(0.14*scale,0.34*scale),(-0.14*scale,0.34*scale)])
    lights=np.maximum.reduce([fill_circle(x,y,0.09*scale,0,-0.20*scale),fill_circle(x,y,0.09*scale,0,0.0),fill_circle(x,y,0.09*scale,0,0.20*scale)])
    return np.maximum(box,lights)


def narrows_right_symbol(x,y,scale=1.0):
    return np.maximum(stroke_polyline(x,y,[(-0.20*scale,0.38*scale),(-0.20*scale,-0.34*scale)],0.07*scale),stroke_polyline(x,y,[(0.22*scale,0.38*scale),(0.08*scale,-0.34*scale)],0.07*scale))


def cross_symbol(x,y,scale=1.0):
    return np.maximum(stroke_polyline(x,y,[(0,-0.36*scale),(0,0.36*scale)],0.10*scale),stroke_polyline(x,y,[(-0.30*scale,0),(0.30*scale,0)],0.10*scale))


def bumpy_symbol(x,y,scale=1.0):
    return stroke_polyline(x,y,[(-0.34*scale,0.12*scale),(-0.18*scale,0.0),(0.0,0.10*scale),(0.18*scale,-0.02*scale),(0.34*scale,0.08*scale)],0.08*scale)


def curve_symbol(x,y,scale=1.0,right=True):
    s=1 if right else -1
    return stroke_polyline(x,y,[(-0.26*s*scale,0.36*scale),(-0.14*s*scale,0.10*scale),(0.06*s*scale,-0.06*scale),(0.22*s*scale,-0.30*scale)],0.09*scale)


def double_curve_symbol(x,y,scale=1.0):
    return stroke_polyline(x,y,[(-0.28*scale,0.34*scale),(-0.18*scale,0.08*scale),(0.0,-0.02*scale),(0.18*scale,-0.16*scale),(0.28*scale,-0.36*scale)],0.08*scale)


def roadworks_symbol(x,y,scale=1.0):
    head=fill_circle(x,y,0.08*scale,-0.05*scale,-0.18*scale)
    body=stroke_polyline(x,y,[(-0.05*scale,-0.10*scale),(0.10*scale,0.10*scale)],0.06*scale)
    leg=stroke_polyline(x,y,[(0.05*scale,0.10*scale),(-0.12*scale,0.32*scale)],0.06*scale)
    shovel=stroke_polyline(x,y,[(-0.04*scale,0.08*scale),(0.26*scale,0.24*scale)],0.05*scale)
    heap=fill_convex_poly(x,y,[(-0.26*scale,0.28*scale),(0.06*scale,0.28*scale),(-0.08*scale,0.16*scale)])
    return np.maximum.reduce([head,body,leg,shovel,heap])


def slippery_symbol(x,y,scale=1.0):
    return np.maximum(car_symbol(x,y,0.78*scale),np.maximum(stroke_polyline(x,y,[(-0.24*scale,0.34*scale),(-0.10*scale,0.14*scale),(-0.18*scale,-0.04*scale)],0.04*scale),stroke_polyline(x,y,[(0.08*scale,0.34*scale),(0.20*scale,0.18*scale),(0.10*scale,0.00)],0.04*scale)))


def deer_symbol(x,y,scale=1.0):
    body=fill_ellipse(x,y,0.22*scale,0.12*scale,-0.02*scale,0.06*scale,-0.15)
    neck=stroke_polyline(x,y,[(0.06*scale,-0.02*scale),(0.18*scale,-0.22*scale)],0.05*scale)
    head=fill_ellipse(x,y,0.10*scale,0.06*scale,0.24*scale,-0.28*scale,-0.2)
    antlers=np.maximum(stroke_polyline(x,y,[(0.28*scale,-0.32*scale),(0.40*scale,-0.42*scale)],0.03*scale),stroke_polyline(x,y,[(0.24*scale,-0.34*scale),(0.18*scale,-0.46*scale)],0.03*scale))
    legs=np.maximum.reduce([
        stroke_polyline(x,y,[(-0.12*scale,0.16*scale),(-0.18*scale,0.42*scale)],0.04*scale),
        stroke_polyline(x,y,[(0.00,0.16*scale),(-0.02*scale,0.42*scale)],0.04*scale),
        stroke_polyline(x,y,[(0.10*scale,0.14*scale),(0.10*scale,0.40*scale)],0.04*scale),
        stroke_polyline(x,y,[(0.18*scale,0.10*scale),(0.22*scale,0.32*scale)],0.04*scale),
    ])
    return np.maximum.reduce([body,neck,head,antlers,legs])


def render_class(cls, sign_scale=1.0):
    x,y,ss=hi_grid()
    canvas=np.ones((x.shape[0],x.shape[1],3),np.float32)*BG
    def put(mask,color):
        draw(canvas,mask,color)
    def tri(inner_mask=None):
        outer=fill_convex_poly(x,y,triangle_pts(0.92*sign_scale))
        inner=fill_convex_poly(x,y,triangle_pts(0.72*sign_scale))
        put(outer,RED); put(inner,WHITE)
        if inner_mask is not None: put(inner_mask,BLACK)
    def invtri(inner_mask=None):
        outer=fill_convex_poly(x,y,triangle_pts(0.92*sign_scale,upside=True))
        inner=fill_convex_poly(x,y,triangle_pts(0.72*sign_scale,upside=True))
        put(outer,RED); put(inner,WHITE)
        if inner_mask is not None: put(inner_mask,WHITE)
    def circ(inner_mask=None, inner_color=BLACK):
        put(fill_circle(x,y,0.88*sign_scale),RED)
        put(fill_circle(x,y,0.68*sign_scale),WHITE)
        if inner_mask is not None: put(inner_mask,inner_color)
    def blue_disc(inner_mask=None):
        put(fill_circle(x,y,0.84*sign_scale),BLUE)
        if inner_mask is not None: put(inner_mask,WHITE)
    if cls in [0,1,2,3,4,5,7,8]:
        txt={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[cls]
        circ(bitmap_mask(x,y,txt,0,0,0.18 if len(txt)==2 else 0.12,0.20))
    elif cls==6:
        circ(bitmap_mask(x,y,'80',0,0,0.18,0.20))
        put(stroke_polyline(x,y,[(-0.56,-0.52),(0.56,0.52)],0.12),BLACK)
    elif cls==9:
        circ(); put(car_symbol(x+0.22,y,0.70),RED); put(car_symbol(x-0.22,y,0.70),BLACK)
    elif cls==10:
        circ(); put(car_symbol(x+0.24,y,0.62),RED); put(truck_symbol(x-0.14,y,0.82),BLACK)
    elif cls==11:
        tri(cross_symbol(x,y,1.0))
    elif cls==12:
        outer=fill_convex_poly(x,y,[(0,-0.76*sign_scale),(0.76*sign_scale,0),(0,0.76*sign_scale),(-0.76*sign_scale,0)])
        mid=fill_convex_poly(x,y,[(0,-0.62*sign_scale),(0.62*sign_scale,0),(0,0.62*sign_scale),(-0.62*sign_scale,0)])
        put(outer,WHITE); put(mid,YELLOW)
    elif cls==13:
        invtri()
    elif cls==14:
        pts=[(0.78*math.cos(math.pi/8+k*math.pi/4)*sign_scale,0.78*math.sin(math.pi/8+k*math.pi/4)*sign_scale) for k in range(8)]
        put(fill_convex_poly(x,y,pts),RED); put(bitmap_mask(x,y,'STOP',0,0,0.16,0.18),WHITE)
    elif cls==15:
        circ()
    elif cls==16:
        circ(truck_symbol(x,y,1.0))
    elif cls==17:
        put(fill_circle(x,y,0.86*sign_scale),RED); put(fill_convex_poly(x,y,[(-0.50,-0.12),(0.50,-0.12),(0.50,0.12),(-0.50,0.12)]),WHITE)
    elif cls==18:
        tri(bitmap_mask(x,y,'!',0,0,0.18,0.20))
    elif cls==19:
        tri(curve_symbol(x,y,1.0,right=False))
    elif cls==20:
        tri(curve_symbol(x,y,1.0,right=True))
    elif cls==21:
        tri(double_curve_symbol(x,y,1.0))
    elif cls==22:
        tri(bumpy_symbol(x,y,1.0))
    elif cls==23:
        tri(slippery_symbol(x,y,0.95))
    elif cls==24:
        tri(narrows_right_symbol(x,y,1.0))
    elif cls==25:
        tri(roadworks_symbol(x,y,1.0))
    elif cls==26:
        tri(traffic_light_symbol(x,y,1.0))
    elif cls==27:
        tri(person_symbol(x,y,1.0,-1))
    elif cls==28:
        tri(np.maximum(person_symbol(x+0.14,y+0.02,0.80,-1),person_symbol(x-0.14,y-0.02,0.60,-1)))
    elif cls==29:
        tri(bike_symbol(x,y,0.95))
    elif cls==30:
        tri(snowflake_symbol(x,y,1.0))
    elif cls==31:
        tri(deer_symbol(x,y,1.0))
    elif cls==32:
        put(fill_circle(x,y,0.86*sign_scale),WHITE)
        for d in (-0.28,0.0,0.28):
            put(stroke_polyline(x,y,[(-0.72+d,-0.62),(0.42+d,0.52)],0.10),BLACK)
    elif cls==33:
        blue_disc(arrow_turn(x,y,True,1.0))
    elif cls==34:
        blue_disc(arrow_turn(x,y,False,1.0))
    elif cls==35:
        blue_disc(arrow_straight(x,y,1.0))
    elif cls==36:
        blue_disc(arrow_fork(x,y,True,1.0))
    elif cls==37:
        blue_disc(arrow_fork(x,y,False,1.0))
    elif cls==38:
        blue_disc(arrow_keep(x,y,True,1.0))
    elif cls==39:
        blue_disc(arrow_keep(x,y,False,1.0))
    elif cls==40:
        blue_disc(roundabout_symbol(x,y,1.0))
    elif cls==41:
        circ(); put(car_symbol(x+0.22,y,0.70),np.array([0.45,0.45,0.45],np.float32)); put(car_symbol(x-0.22,y,0.70),BLACK); put(stroke_polyline(x,y,[(-0.56,-0.52),(0.56,0.52)],0.12),BLACK)
    elif cls==42:
        circ(); put(car_symbol(x+0.24,y,0.62),np.array([0.45,0.45,0.45],np.float32)); put(truck_symbol(x-0.14,y,0.82),BLACK); put(stroke_polyline(x,y,[(-0.56,-0.52),(0.56,0.52)],0.12),BLACK)
    else:
        raise ValueError(cls)
    return np.transpose(downsample(canvas,ss),(2,0,1))


def sign_mask(img):
    mask=np.max(np.abs(img-BG[:,None,None]),axis=0)>0.04
    # dilate by one pixel
    d=mask.copy()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==0 and dx==0:
                continue
            ys=slice(max(0,dy),min(32,32+dy))
            yt=slice(max(0,-dy),min(32,32-dy))
            xs=slice(max(0,dx),min(32,32+dx))
            xt=slice(max(0,-dx),min(32,32-dx))
            d[yt,xt] |= mask[ys,xs]
    return d.astype(np.float32)


def gray_of(img):
    return 0.299*img[0]+0.587*img[1]+0.114*img[2]


def normalize_channels(img):
    mu=img.mean(axis=(1,2),keepdims=True)
    var=((img-mu)**2).mean(axis=(1,2),keepdims=True)
    return (img-mu)/np.sqrt(var+EPS)


def unit(v):
    n=float(np.sqrt((v*v).sum()))
    return v/(n+1e-8)


def build_filters():
    m=43*len(SCALES)
    wrgb=np.zeros((m,3,32,32),np.float32)
    wgray=np.zeros((m,1,32,32),np.float32)
    for cls in range(43):
        for si,scale in enumerate(SCALES):
            idx=cls*len(SCALES)+si
            img=render_class(cls,scale).astype(np.float32)
            mask=sign_mask(img)
            trgb=normalize_channels(img)*mask[None]
            wrgb[idx]=unit(trgb)*RGB_SCALE
            g=gray_of(img)[None]
            g=(g-g.mean())/np.sqrt(((g-g.mean())**2).mean()+EPS)
            wgray[idx]=unit(g*mask[None])*GRAY_SCALE
    return wrgb,wgray


def max_stage_weights(group_sizes):
    inp=sum(group_sizes)
    out=sum((g+1)//2 for g in group_sizes)
    wd=np.zeros((out,inp),np.float32)
    wb=np.zeros((out,inp),np.float32)
    new=[]
    r=0
    base=0
    for g in group_sizes:
        for j in range((g+1)//2):
            a=base+2*j
            b=base+2*j+1
            if b<base+g:
                wd[r,a]=1.0; wd[r,b]=-1.0
                wb[r,b]=1.0
            else:
                wb[r,a]=1.0
            r+=1
        new.append((g+1)//2)
        base+=g
    return wd,wb,new


def make_network_and_weights():
    m=43*len(SCALES)
    layers=[]
    weights={}
    layers.append({"name":"n_rgb","op":"norm","inputs":["image"]})
    layers.append({"name":"gray","op":"conv","inputs":["image"],"out":1,"kernel":[1,1],"stride":1,"padding":0})
    layers.append({"name":"n_gray","op":"norm","inputs":["gray"]})
    layers.append({"name":"c_rgb","op":"conv","inputs":["n_rgb"],"out":m,"kernel":[32,32],"stride":1,"padding":PAD})
    layers.append({"name":"c_gray","op":"conv","inputs":["n_gray"],"out":m,"kernel":[32,32],"stride":1,"padding":PAD})
    layers.append({"name":"c_sum","op":"add","inputs":["c_rgb","c_gray"]})
    layers.append({"name":"shiftmax","op":"maxpool","inputs":["c_sum"],"size":[POOL,POOL],"stride":[POOL,POOL]})
    layers.append({"name":"flat","op":"reshape","inputs":["shiftmax"],"shape":[m]})
    weights["gray.W"]=np.array([[[[0.299]],[[0.587]],[[0.114]]]],np.float32)
    weights["gray.b"]=np.zeros(1,np.float32)
    wrgb,wgray=build_filters()
    weights["c_rgb.W"]=wrgb
    weights["c_rgb.b"]=np.zeros(m,np.float32)
    weights["c_gray.W"]=wgray
    weights["c_gray.b"]=np.zeros(m,np.float32)
    group_sizes=[len(SCALES)]*43
    inp="flat"
    stage=0
    while max(group_sizes)>1:
        wd,wb,group_sizes=max_stage_weights(group_sizes)
        dname=f"d{stage}"
        rname=f"r{stage}"
        bname=f"b{stage}"
        aname=f"m{stage}"
        layers.append({"name":dname,"op":"dense","inputs":[inp],"out":wd.shape[0]})
        layers.append({"name":rname,"op":"relu","inputs":[dname]})
        layers.append({"name":bname,"op":"dense","inputs":[inp],"out":wb.shape[0]})
        layers.append({"name":aname,"op":"add","inputs":[rname,bname]})
        weights[dname+".W"]=wd
        weights[dname+".b"]=np.zeros(wd.shape[0],np.float32)
        weights[bname+".W"]=wb
        weights[bname+".b"]=np.zeros(wb.shape[0],np.float32)
        inp=aname
        stage+=1
    net={"layers":layers}
    return net,weights


def main(out_dir):
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    net,weights=make_network_and_weights()
    (out/"network.json").write_text(json.dumps(net,separators=(",",":")))
    np.savez(out/"weights.npz",**weights)


if __name__=="__main__":
    main(sys.argv[1])
