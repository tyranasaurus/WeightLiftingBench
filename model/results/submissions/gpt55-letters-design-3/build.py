#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np
H=W=28; CLASSES=24
Y,X=np.mgrid[0:H,0:W].astype(np.float32)

def smooth_step(z,aa=.8): return np.clip(.5-z/aa,0,1)
def capsule(x1,y1,x2,y2,r,aa=.9):
    vx,vy=x2-x1,y2-y1; t=((X-x1)*vx+(Y-y1)*vy)/(vx*vx+vy*vy+1e-6); t=np.clip(t,0,1)
    px=x1+t*vx; py=y1+t*vy
    return smooth_step(np.sqrt((X-px)**2+(Y-py)**2)-r,aa)
def ellipse(cx,cy,rx,ry,aa=.10):
    q=((X-cx)/rx)**2+((Y-cy)/ry)**2
    return np.clip((1-q)/aa+.5,0,1)
def add(m,s,strength=1.0): np.maximum(m,np.clip(s*strength,0,1),out=m)
def erase(m,s,strength=1.0): m*=1-np.clip(s*strength,0,1)
def finger(m,x,y0=15,y1=3.8,r=1.35,lean=0): add(m,capsule(x,y0,x+lean,y1,r))
def palm(m,cx=14,cy=18,rx=6,ry=6.4): add(m,ellipse(cx,cy,rx,ry))
def wrist(m,cx=14,y0=22,y1=29,r=3.8): add(m,capsule(cx,y0,cx,y1,r))
def fist(m,cx=14,cy=18):
    wrist(m,cx,22,29,3.9); palm(m,cx,cy,6.2,6.5)
    for x in [9.7,12.7,15.7,18.3]: add(m,ellipse(x,12.6,2.2,2.3))
    add(m,capsule(8.7,15.5,19.5,16.5,1.6),.7)
def ring(m,cx,cy,rx,ry,hx,hy): add(m,ellipse(cx,cy,rx,ry)); erase(m,ellipse(cx,cy,hx,hy,aa=.18),1)

def base_template(k):
    m=np.zeros((H,W),np.float32)
    if k==0: # A: fist, thumb on viewer right
        fist(m); add(m,capsule(20.5,19,22,13.5,1.55)); add(m,ellipse(21,16,2,3.2))
    elif k==1: # B: flat vertical fingers, folded thumb on right
        wrist(m); palm(m,14,18,6.2,6.8)
        for x in [9.7,12.5,15.3,18.1]: finger(m,x,16,3,1.35,.05*(x-14))
        add(m,capsule(20.5,19.2,12.2,20.5,1.45))
    elif k==2: # C
        add(m,ellipse(13.8,14.3,8.5,10)); erase(m,ellipse(15.3,14.4,5,6.3,aa=.18)); erase(m,ellipse(21,14.3,4.6,4.8,aa=.20)); add(m,capsule(8.5,21,11,28,2.8))
    elif k==3: # D
        wrist(m); palm(m,14,18.5,5.8,5.8); finger(m,14.2,15.5,2.8,1.35); ring(m,14.5,17.2,5.2,4.2,2.4,2.0); add(m,capsule(19,17.5,14,20,1.4))
    elif k==4: # E
        wrist(m); palm(m,14,18,6.3,6.4)
        for x in [9.5,12.6,15.7,18.5]: add(m,ellipse(x,12.7,2.2,2.7)); add(m,capsule(x-.8,13,x+.9,17,1.15))
        add(m,capsule(19.8,19,10.5,20.2,1.5))
    elif k==5: # F: OK circle on right, three fingers left/up
        wrist(m); palm(m,14.5,18.2,5.4,6.2)
        for x in [13.8,11,8.5]: finger(m,x,15.5,3.3,1.25,.10*(x-11.5))
        ring(m,18.3,16.2,4.5,4.0,2.0,1.8); add(m,capsule(19.5,18,14.5,20.6,1.25))
    elif k==6: # G: index and thumb point right
        add(m,ellipse(9.5,18,5.5,5.1)); add(m,capsule(6.5,21.5,0,24,3.4)); add(m,capsule(10,12,23,12.5,1.35)); add(m,capsule(10,16.3,21.5,17.8,1.25))
    elif k==7: # H: two fingers point right
        add(m,ellipse(9.8,18.3,5.8,5.2)); add(m,capsule(6.5,21.5,0,24,3.4)); add(m,capsule(10,11,23.2,11.6,1.25)); add(m,capsule(10,14,23.2,14.3,1.25)); add(m,capsule(9.5,17.6,19.7,18.8,1.15))
    elif k==8: # I: fist with pinky on left
        fist(m); finger(m,9.8,13.8,2.8,1.25,-.5)
    elif k==9: # K
        wrist(m); palm(m,14,18.5,5.8,5.9); add(m,capsule(13,16.2,10.6,3.2,1.25)); add(m,capsule(15.2,16.2,18.2,4,1.25)); add(m,capsule(19.5,17.8,12,15,1.35))
    elif k==10: # L: index up, thumb right
        wrist(m); palm(m,13,18.5,5.8,6); finger(m,11.5,15,3,1.35); add(m,capsule(12,16.2,23.2,17,1.55))
    elif k==11: # M
        fist(m)
        for x in [10.5,13.8,17.1]: add(m,ellipse(x,13.5,2.35,3),.95)
        add(m,capsule(19,18.5,9,20.5,1.3))
    elif k==12: # N
        fist(m)
        for x in [12,15.5]: add(m,ellipse(x,13.2,2.5,3.1),.95)
        add(m,capsule(19.5,18.5,11.5,20.2,1.35))
    elif k==13: # O
        ring(m,14,14.2,7.8,9,3.8,4.7); add(m,capsule(11.2,21,13,28.3,2.9)); add(m,capsule(18,20.5,15,27.5,2.3),.7)
    elif k==14: # P: downward/side K
        add(m,ellipse(15.8,12.4,5.7,5.6)); add(m,capsule(18.5,7,22,0,3.3)); add(m,capsule(14.5,13.5,9.7,25,1.25)); add(m,capsule(17.2,13.5,14.2,26,1.25)); add(m,capsule(21,14.5,13.5,16.2,1.3))
    elif k==15: # Q: downward/side G
        add(m,ellipse(15.8,11.8,5.6,5.4)); add(m,capsule(19,6.7,22,0,3.1)); add(m,capsule(13.3,13.4,9.5,26,1.35)); add(m,capsule(16.4,13.2,14.7,25,1.25))
    elif k==16: # R
        wrist(m); palm(m,14,18.2,5.8,6); add(m,capsule(12.3,16,16,3.5,1.25)); add(m,capsule(16,16,12.8,3.6,1.25)); add(m,capsule(19.5,18.5,13,20,1.35))
    elif k==17: # S
        fist(m); add(m,capsule(20.8,18,8.8,20,1.85))
    elif k==18: # T: side fist, thumb projecting right
        add(m,ellipse(9.7,18,5.7,5.6)); add(m,capsule(6.5,22,0,24,3.5)); add(m,capsule(10,14.2,23.5,14.6,1.65)); add(m,capsule(10.5,18.0,10.5,8.0,1.55))
    elif k==19: # U
        wrist(m); palm(m,14,18.5,5.8,6); finger(m,12.8,15.3,3.1,1.3,-.1); finger(m,15.7,15.3,3.1,1.3,.1); add(m,capsule(19.2,18.5,13,20,1.25))
    elif k==20: # V
        wrist(m); palm(m,14,18.5,5.8,6); add(m,capsule(13,16,9.5,3.5,1.25)); add(m,capsule(15.6,16,19.2,3.5,1.25)); add(m,capsule(19.5,18.4,13,20,1.25))
    elif k==21: # W
        wrist(m); palm(m,14,18.6,5.9,6); add(m,capsule(12,16,8.8,3.4,1.18)); add(m,capsule(14.5,16,14.5,3,1.18)); add(m,capsule(17,16,20.2,3.7,1.18)); add(m,capsule(19.7,18,13.5,20,1.2))
    elif k==22: # X
        fist(m); add(m,capsule(14.2,15,13,8,1.35)); add(m,capsule(13,8,17,8.5,1.30))
    elif k==23: # Y: pinky left, thumb right
        wrist(m); palm(m,14,19,5.8,5.8); add(m,capsule(23,16.5,14.5,18.5,1.6)); add(m,capsule(10,16,8,3.5,1.35))
    return np.clip(m,0,1)

def proto_image(k):
    p=.80*base_template(k).astype(np.float32)
    def dark(s,a=.85):
        nonlocal_p[0]-=a*s
    nonlocal_p=[p]
    # Hand-drawn dark crease/finger-gap marks: generic ASL anatomy, not data-derived pixels.
    if k in [0,8,11,12,17,22]:
        for x in [11.2,14.2,17.2]: dark(capsule(x,10.2,x-.3,15.0,.45),.55)
        dark(capsule(9,15.2,19.5,16.1,.55),.55)
    if k==0:
        dark(capsule(19.5,14.5,22.0,18.5,.55),.65)
    elif k==1:
        for x in [11.1,14.0,16.9]: dark(capsule(x,3.2,x,16.0,.38),.75)
        dark(capsule(19.8,15,13.5,20.5,.55),.65)
    elif k==2:
        dark(capsule(18,6,21,13,.45),.45); dark(capsule(18,22,22,16,.45),.45)
    elif k==3:
        dark(ellipse(14.5,17.1,2.7,2.1,aa=.20),.75); dark(capsule(14.2,3,14.2,14,.35),.45)
    elif k==4:
        for x in [10.8,13.8,16.8,19.0]: dark(capsule(x,8,x-.4,14.5,.55),.80)
        dark(capsule(8.5,14.0,20.5,14.8,.65),.85)
    elif k==5:
        for x in [9.8,12.4]: dark(capsule(x,3.5,x+.2,15,.36),.70)
        dark(ellipse(18.3,16.2,2.2,1.9,aa=.20),.90)
    elif k==6:
        dark(capsule(10,14.2,22,14.4,.42),.65); dark(capsule(10,16.4,17,20,.45),.45)
    elif k==7:
        dark(capsule(10,12.5,23,12.7,.45),.80); dark(capsule(10,15.3,19,18.5,.42),.50)
    elif k==8:
        dark(capsule(9.8,3,9.8,13,.38),.45); dark(capsule(20,14,21,18,.5),.60)
    elif k==9:
        dark(capsule(14.2,5,14.2,16,.45),.65); dark(capsule(18.5,16.5,12,15,.55),.75)
    elif k==10:
        dark(capsule(11.5,3,11.5,14.5,.38),.45); dark(capsule(12.5,16.6,23,16.8,.45),.65)
    elif k==11:
        for x in [10.8,13.8,16.8]: dark(capsule(x,9.5,x,16.8,.55),.85)
        dark(capsule(18.8,18.4,9.5,20.3,.55),.70)
    elif k==12:
        for x in [12.2,15.4]: dark(capsule(x,9.5,x,16.8,.58),.85)
        dark(capsule(19.0,18.4,11.3,20.1,.55),.70)
    elif k==13:
        dark(ellipse(14,14.2,3.9,4.8,aa=.20),.35); dark(capsule(17,7,20,13,.35),.35)
    elif k==14:
        dark(capsule(15,13,11,25,.45),.65); dark(capsule(17.0,13.5,14,25,.45),.65); dark(ellipse(16,16,2.2,1.8,aa=.25),.55)
    elif k==15:
        dark(capsule(13.3,13.4,9.5,26,.45),.75); dark(capsule(16.4,13.2,14.7,25,.42),.55)
    elif k==16:
        dark(capsule(13.5,6,15.5,15.8,.42),.75); dark(capsule(15.0,6,12.7,15.8,.42),.75)
    elif k==17:
        dark(capsule(20.5,17.8,9.5,19.8,.70),.90)
    elif k==18:
        dark(capsule(10.5,8,10.5,18,.48),.70); dark(capsule(10,14.5,23.5,14.5,.48),.55)
    elif k==19:
        dark(capsule(14.25,3.2,14.25,15.5,.45),.85); dark(capsule(19,18,13,20,.45),.55)
    elif k==20:
        dark(capsule(14.3,15.0,14.3,8.0,.55),.80); dark(capsule(19,18,13,20,.45),.50)
    elif k==21:
        for x in [13.1,15.8]: dark(capsule(x,4,x,16,.45),.75)
        dark(capsule(19.4,18,13.5,20,.45),.50)
    elif k==22:
        dark(capsule(13.5,8,17.5,8.5,.55),.80); dark(capsule(14,9,14,15,.45),.65)
    elif k==23:
        dark(capsule(9,3.5,10,16,.38),.45); dark(capsule(14.5,18.4,23,16.4,.45),.55)
    return nonlocal_p[0]

def transform_mask(m,dx=0,dy=0,scale=1,theta=0):
    cx=cy=(H-1)/2
    xr=X-cx-dx; yr=Y-cy-dy; co=np.cos(theta); si=np.sin(theta)
    sx=(co*xr+si*yr)/scale+cx; sy=(-si*xr+co*yr)/scale+cy
    x0=np.floor(sx).astype(int); y0=np.floor(sy).astype(int); x1=x0+1; y1=y0+1; wx=sx-x0; wy=sy-y0
    out=np.zeros_like(m,np.float32)
    for yy,wyv in ((y0,1-wy),(y1,wy)):
        for xx,wxv in ((x0,1-wx),(x1,wx)):
            ok=(xx>=0)&(xx<W)&(yy>=0)&(yy<H); out[ok]+=m[yy[ok],xx[ok]]*wxv[ok]*wyv[ok]
    return out

def gkern(k=7,s=1.3):
    ax=np.arange(k,dtype=np.float32)-(k-1)/2; yy,xx=np.meshgrid(ax,ax,indexing='ij'); g=np.exp(-(xx*xx+yy*yy)/(2*s*s)).astype(np.float32); g/=g.sum(); return g,xx.astype(np.float32),yy.astype(np.float32)
def make_filters():
    g,xx,yy=gkern(7,1.35); filters=[]
    g2,_,_=gkern(7,2.2); dog=g-g2; dog-=dog.mean(); dog/=np.abs(dog).sum()+1e-6
    filters += [dog.astype(np.float32),(-dog).astype(np.float32)]
    for typ in range(2):
        for oi in range(8):
            th=np.pi*oi/8; u=np.cos(th)*xx+np.sin(th)*yy; v=-np.sin(th)*xx+np.cos(th)*yy
            if typ==0:
                f=u*g; f-=f.mean(); f/=np.abs(f).sum()+1e-6
            else:
                f=np.exp(-(u*u)/(2*.55*.55)-(v*v)/(2*2.2*2.2)).astype(np.float32); f-=.55*np.exp(-(u*u)/(2*1.9*1.9)-(v*v)/(2*2.3*2.3)).astype(np.float32); f-=f.mean(); f/=np.abs(f).sum()+1e-6
            filters += [f.astype(np.float32),(-f).astype(np.float32)]
    Wc=np.stack(filters).astype(np.float32)[:,None,:,:]; return Wc,np.zeros((Wc.shape[0],),np.float32)
def conv2(x,f):
    k=f.shape[0]; p=k//2; xp=np.pad(x,((p,p),(p,p))); out=np.zeros((H,W),np.float32)
    for dy in range(k):
        for dx in range(k): out += f[dy,dx]*xp[dy:dy+H,dx:dx+W]
    return out
def pool2(a): return a.reshape(a.shape[0],14,2,14,2).mean(axis=(2,4))
def variants():
    shifts=[(-2,0),(0,0),(2,0),(0,-2),(0,2),(-1.4,-1.4),(1.4,-1.4),(-1.4,1.4),(1.4,1.4)]
    for sc in [.94,1.0,1.07]:
        for dx,dy in shifts: yield dx,dy,sc

def make_rows(Wc):
    raw=[]; edge=[]
    for k in range(CLASSES):
        p=proto_image(k)
        for dx,dy,sc in variants():
            pp=transform_mask(p,dx,dy,sc)
            for pol in (1,-1):
                u=pol*(pp-pp.mean()); u=u/(np.sqrt((u*u).sum()+1e-6)); raw.append((3.0*u.reshape(-1)).astype(np.float32))
                ch=[]; q=pol*pp
                for j in range(Wc.shape[0]): ch.append(pool2(np.maximum(conv2(q,Wc[j,0]),0)[None,:,:])[0])
                e=np.stack(ch).reshape(-1).astype(np.float32); e-=e.mean(); e=e/(np.sqrt((e*e).sum()+1e-6)); edge.append((8.0*e).astype(np.float32))
    return np.stack(raw).astype(np.float32),np.stack(edge).astype(np.float32),54

def build(out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    Wc,bc=make_filters(); Wr,We,v=make_rows(Wc); br=np.zeros((Wr.shape[0],),np.float32); be=np.zeros((We.shape[0],),np.float32)
    Wcal=np.eye(CLASSES,dtype=np.float32); bcal=np.zeros((CLASSES,),np.float32)
    # Hand-set deconfusion: closed-loop templates are deliberately broad, so make O
    # compete against fist-like letters instead of absorbing them all.
    for j in [0,4,11,12,17,22]: Wcal[13,j]-=.15
    Wcal[13,2]+=.08; bcal[13]=-.85
    Wcal[0,13]+=.10; Wcal[17,13]+=.10; Wcal[11,13]+=.08; Wcal[12,13]+=.07; Wcal[4,13]+=.05
    bcal[4]+=.35; bcal[11]+=.40; bcal[12]+=.35; bcal[9]+=.15; bcal[23]+=.15; bcal[22]-=.25
    layers=[
        {"name":"n","op":"norm","inputs":["image"]},
        {"name":"raw","op":"dense","inputs":["n"],"out":int(Wr.shape[0])},
        {"name":"raw_r","op":"reshape","inputs":["raw"],"shape":[CLASSES,int(v),1]},
        {"name":"raw_m","op":"maxpool","inputs":["raw_r"],"size":[int(v),1],"stride":[int(v),1]},
        {"name":"raw_s","op":"reshape","inputs":["raw_m"],"shape":[CLASSES]},
        {"name":"ef","op":"conv","inputs":["n"],"out":int(Wc.shape[0]),"kernel":[7,7],"stride":1,"padding":3},
        {"name":"ea","op":"relu","inputs":["ef"]},
        {"name":"ep","op":"avgpool","inputs":["ea"],"size":[2,2],"stride":[2,2]},
        {"name":"em","op":"dense","inputs":["ep"],"out":int(We.shape[0])},
        {"name":"er","op":"reshape","inputs":["em"],"shape":[CLASSES,int(v),1]},
        {"name":"ex","op":"maxpool","inputs":["er"],"size":[int(v),1],"stride":[int(v),1]},
        {"name":"edge_s","op":"reshape","inputs":["ex"],"shape":[CLASSES]},
        {"name":"sum","op":"add","inputs":["raw_s","edge_s"]},
        {"name":"scores","op":"dense","inputs":["sum"],"out":CLASSES}
    ]
    (out/'network.json').write_text(json.dumps({"layers":layers},separators=(',',':')))
    np.savez(out/'weights.npz',**{"raw.W":Wr,"raw.b":br,"ef.W":Wc,"ef.b":bc,"em.W":We,"em.b":be,"scores.W":Wcal,"scores.b":bcal})
if __name__=='__main__': build(sys.argv[1] if len(sys.argv)>1 else '/app/out')
