#!/usr/bin/env python3
"""Build a traffic-sign classifier entirely from analytic, hand-drawn sign models."""
import json, math, sys
from pathlib import Path
import numpy as np

# The renderer uses only geometric primitives and conventional sign semantics.  It
# deliberately has no data-loading code.
S = 3
N = 32 * S
q = (np.arange(N, dtype=np.float32) + .5) / N * 2 - 1
YY, XX = np.meshgrid(q, q, indexing="ij")
BG = np.array([.43, .45, .42], np.float32)
WHITE = np.array([.94, .94, .91], np.float32)
BLACK = np.array([.025, .025, .022], np.float32)
RED = np.array([.88, .035, .025], np.float32)
BLUE = np.array([.025, .23, .70], np.float32)
YELLOW = np.array([1.0, .72, .035], np.float32)
GRAY = np.array([.67, .68, .65], np.float32)


def paint(im, m, col):
    im[m] = col


def ellipse(x, y, cx, cy, rx, ry):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1


def polygon(x, y, pts):
    # Even/odd fill rule.
    inside = np.zeros(x.shape, bool)
    xj, yj = pts[-1]
    for xi, yi in pts:
        hit = ((yi > y) != (yj > y)) & (x < (xj-xi)*(y-yi)/(yj-yi + 1e-12) + xi)
        inside ^= hit
        xj, yj = xi, yi
    return inside


def segment(x, y, a, b, width):
    ax, ay = a; bx, by = b
    vx, vy = bx-ax, by-ay
    t = np.clip(((x-ax)*vx + (y-ay)*vy)/(vx*vx + vy*vy + 1e-12), 0, 1)
    return (x-(ax+t*vx))**2 + (y-(ay+t*vy))**2 <= width*width


def pathmask(x, y, pts, width, closed=False):
    m = np.zeros(x.shape, bool)
    pp = list(pts)
    if closed: pp.append(pp[0])
    for a, b in zip(pp[:-1], pp[1:]):
        m |= segment(x, y, a, b, width)
    return m


def arrowhead(x, y, tip, direction, size=.20):
    dx, dy = direction
    z = math.hypot(dx,dy); dx,dy=dx/z,dy/z
    px,py=-dy,dx
    back=(tip[0]-dx*size,tip[1]-dy*size)
    return polygon(x,y,[tip,(back[0]+px*size*.58,back[1]+py*size*.58),
                         (back[0]-px*size*.58,back[1]-py*size*.58)])

# Rounded, stroke-based glyphs, authored from ordinary sans-serif letterforms.
DIG = {
 '0': [[(-.34,-.64),(-.48,-.40),(-.48,.39),(-.32,.64),(.30,.64),(.47,.39),(.47,-.39),(.30,-.64),(-.34,-.64)]],
 '1': [[(-.28,-.43),(0,-.66),(0,.64)],[(-.27,.64),(.28,.64)]],
 '2': [[(-.43,-.40),(-.34,-.61),(.22,-.66),(.44,-.46),(.42,-.22),(-.42,.62),(.47,.62)]],
 '3': [[(-.40,-.58),(.27,-.62),(.44,-.40),(.34,-.12),(.02,0),(.34,.10),(.45,.38),(.28,.61),(-.38,.57)]],
 '4': [[(.25,.66),(.25,-.66),(-.46,.22),(.48,.22)]],
 '5': [[(.40,-.62),(-.38,-.62),(-.43,-.05),(.24,-.08),(.43,.12),(.39,.43),(.20,.62),(-.39,.57)]],
 '6': [[(.35,-.55),(.12,-.66),(-.25,-.51),(-.43,-.12),(-.39,.43),(-.19,.63),(.24,.58),(.43,.37),(.38,.04),(.18,-.12),(-.38,-.05)]],
 '7': [[(-.44,-.61),(.45,-.61),(-.03,.65)]],
 '8': [[(-.27,-.64),(.27,-.61),(.43,-.39),(.34,-.10),(0,0),(-.35,-.10),(-.44,-.39),(-.27,-.64)],
       [(0,0),(.35,.10),(.43,.38),(.27,.62),(-.27,.62),(-.43,.38),(-.35,.10),(0,0)]],
 '9': [[(-.35,.55),(-.10,.66),(.27,.50),(.43,.11),(.39,-.43),(.19,-.63),(-.24,-.58),(-.43,-.37),(-.38,-.04),(-.18,.12),(.38,.05)]],
}
LET = {
 'S': [[(.38,-.50),(.20,-.64),(-.25,-.60),(-.40,-.39),(-.30,-.10),(.28,.07),(.40,.30),(.30,.55),(-.25,.62),(-.42,.48)]],
 'T': [[(-.43,-.62),(.43,-.62)],[(0,-.62),(0,.63)]],
 'O': DIG['0'],
 'P': [[(-.36,.64),(-.36,-.64),(.20,-.64),(.42,-.42),(.38,-.10),(.18,.06),(-.34,.04)]],
}


def textmask(x,y,text,cx,cy,height,width,stroke):
    # width is the total requested text width.
    n=len(text); cell=width/n
    m=np.zeros(x.shape,bool)
    for j,ch in enumerate(text):
        gx=cx-width/2+(j+.5)*cell
        sx=cell*.78/.96; sy=height/1.30
        glyph = DIG.get(ch, LET.get(ch, []))
        for p in glyph:
            pts=[(gx+px*sx,cy+py*sy) for px,py in p]
            m |= pathmask(x,y,pts,stroke)
    return m


def circle_sign(im,x,y,kind='red',thick=1.0):
    r=.78
    outer=ellipse(x,y,0,0,r,r)
    paint(im,outer,WHITE)
    if kind=='red':
        ring=outer & ~ellipse(x,y,0,0,r-.115*thick,r-.115*thick)
        paint(im,ring,RED)
    elif kind=='blue':
        paint(im,outer,BLUE)
    elif kind=='gray':
        paint(im,outer,np.array([.84,.84,.80],np.float32))
        edge=outer & ~ellipse(x,y,0,0,r-.025,r-.025)
        paint(im,edge,np.array([.28,.28,.27],np.float32))
    return outer


def triangle_sign(im,x,y,inverted=False,thick=1.0):
    if not inverted:
        po=[(0,-.83),(-.79,.58),(.79,.58)]
        pi=[(0,-.63),(-.59,.46),(.59,.46)]
    else:
        po=[(-.79,-.58),(.79,-.58),(0,.83)]
        pi=[(-.59,-.45),(.59,-.45),(0,.62)]
    outer=polygon(x,y,po); inner=polygon(x,y,pi)
    paint(im,outer,RED); paint(im,inner,WHITE)
    return outer


def carmask(x,y,cx,cy,scale=1.0):
    body=polygon(x,y,[(cx-.25*scale,cy+.10*scale),(cx-.22*scale,cy-.08*scale),
                      (cx-.11*scale,cy-.20*scale),(cx+.13*scale,cy-.20*scale),
                      (cx+.25*scale,cy-.04*scale),(cx+.26*scale,cy+.12*scale)])
    body |= ellipse(x,y,cx-.14*scale,cy+.12*scale,.065*scale,.065*scale)
    body |= ellipse(x,y,cx+.15*scale,cy+.12*scale,.065*scale,.065*scale)
    return body


def truckmask(x,y,cx=0,cy=0,scale=1.0):
    m=polygon(x,y,[(cx-.34*scale,cy-.18*scale),(cx+.08*scale,cy-.18*scale),
                    (cx+.08*scale,cy-.02*scale),(cx+.30*scale,cy-.02*scale),
                    (cx+.37*scale,cy+.20*scale),(cx-.34*scale,cy+.20*scale)])
    m |= ellipse(x,y,cx-.20*scale,cy+.22*scale,.07*scale,.07*scale)
    m |= ellipse(x,y,cx+.24*scale,cy+.22*scale,.07*scale,.07*scale)
    return m


def stripes(im,x,y,clip,width=.035):
    # Parallel cancellation bars rising from lower left to upper right.
    for a in np.linspace(-1.0,1.0,5):
        m=segment(x,y,(a-.65,.78),(a+.65,-.78),width) & clip
        paint(im,m,np.array([.16,.16,.15],np.float32))


def render(k, scale=1.0, aspect=1.0, rot=0.0, tx=0.0, ty=0.0, style=0):
    a=math.radians(rot); ca,sa=math.cos(a),math.sin(a)
    # Coordinates in the canonical sign frame.
    X=(XX-tx); Y=(YY-ty)
    x=(ca*X+sa*Y)/(scale*aspect)
    y=(-sa*X+ca*Y)/scale
    im=np.empty((N,N,3),np.float32); im[:]=BG
    thick=(.87 if style==0 else 1.13)
    support=np.zeros((N,N),bool)

    if k in [0,1,2,3,4,5,7,8]:
        support=circle_sign(im,x,y,'red',thick)
        val={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[k]
        ww=.83 if len(val)==2 else 1.12
        paint(im,textmask(x,y,val,0,.015,.86,ww,.065*thick),BLACK)
    elif k==6:
        support=circle_sign(im,x,y,'gray',thick)
        paint(im,textmask(x,y,'80',0,.015,.80,.78,.054*thick),np.array([.28,.28,.27]))
        stripes(im,x,y,support,.028*thick)
    elif k in [9,10,15,16,17]:
        support=circle_sign(im,x,y,'red',thick)
        if k==9:
            paint(im,carmask(x,y,-.22,.05,.75),RED)
            paint(im,carmask(x,y,.23,.05,.75),BLACK)
        elif k==10:
            paint(im,carmask(x,y,-.24,.12,.62),RED)
            paint(im,truckmask(x,y,.19,.02,.72),BLACK)
        elif k==16:
            paint(im,truckmask(x,y,0,.02,1.18),BLACK)
        elif k==17:
            paint(im,ellipse(x,y,0,0,.66,.66),RED)
            paint(im,polygon(x,y,[(-.48,-.105),(.48,-.105),(.48,.105),(-.48,.105)]),WHITE)
    elif k==11:
        support=triangle_sign(im,x,y,False,thick)
        paint(im,pathmask(x,y,[(0,.40),(0,-.40)],.105*thick),BLACK)
        paint(im,pathmask(x,y,[(-.38,.16),(0,-.03),(.38,.16)],.072*thick),BLACK)
    elif k==12:
        po=[(0,-.83),(.83,0),(0,.83),(-.83,0)]
        p1=[(0,-.72),(.72,0),(0,.72),(-.72,0)]
        p2=[(0,-.57),(.57,0),(0,.57),(-.57,0)]
        support=polygon(x,y,po); paint(im,support,BLACK); paint(im,polygon(x,y,p1),WHITE); paint(im,polygon(x,y,p2),YELLOW)
    elif k==13:
        support=triangle_sign(im,x,y,True,thick)
    elif k==14:
        pts=[(-.34,-.79),(.34,-.79),(.79,-.34),(.79,.34),(.34,.79),(-.34,.79),(-.79,.34),(-.79,-.34)]
        support=polygon(x,y,pts); paint(im,support,RED)
        edge=pathmask(x,y,pts,.025,True); paint(im,edge,WHITE)
        paint(im,textmask(x,y,'STOP',0,.03,.37,1.20,.037*thick),WHITE)
    elif 18 <= k <= 31:
        support=triangle_sign(im,x,y,False,thick)
        if k==18:
            paint(im,pathmask(x,y,[(0,-.34),(0,.16)],.075*thick),BLACK); paint(im,ellipse(x,y,0,.35,.085,.085),BLACK)
        elif k in [19,20]:
            sg=-1 if k==19 else 1
            pts=[(0,.40),(0,.17),(sg*.08,-.02),(sg*.29,-.22),(sg*.31,-.42)]
            paint(im,pathmask(x,y,pts,.085*thick),BLACK)
        elif k==21:
            pts=[(-.18,.42),(-.18,.22),(.20,-.02),(.18,-.25),(-.09,-.44)]
            paint(im,pathmask(x,y,pts,.080*thick),BLACK)
        elif k==22:
            pts=[(-.43,.30),(-.30,.08),(-.14,.28),(0,.02),(.18,.27),(.34,.08),(.44,.27)]
            paint(im,pathmask(x,y,pts,.065*thick),BLACK)
        elif k==23:
            paint(im,carmask(x,y,0,-.17,.92),BLACK)
            paint(im,pathmask(x,y,[(-.28,.17),(-.42,.29),(-.30,.41)],.035*thick),BLACK)
            paint(im,pathmask(x,y,[(.18,.17),(.34,.29),(.22,.42)],.035*thick),BLACK)
        elif k==24:
            paint(im,pathmask(x,y,[(-.38,.39),(-.38,-.36)],.065*thick),BLACK)
            paint(im,pathmask(x,y,[(.38,.39),(.17,-.05),(.17,-.36)],.065*thick),BLACK)
        elif k==25:
            # Worker, shovel and mound.
            paint(im,ellipse(x,y,-.10,-.28,.09,.09),BLACK)
            paint(im,pathmask(x,y,[(-.10,-.19),(-.02,.06),(-.20,.31)],.055*thick),BLACK)
            paint(im,pathmask(x,y,[(-.05,-.06),(.25,.08),(.40,.36)],.040*thick),BLACK)
            paint(im,pathmask(x,y,[(-.03,.03),(.18,.30)],.050*thick),BLACK)
            paint(im,pathmask(x,y,[(-.38,.40),(.38,.40)],.055*thick),BLACK)
        elif k==26:
            m=polygon(x,y,[(-.17,-.42),(.17,-.42),(.17,.39),(-.17,.39)]); paint(im,m,BLACK)
            for cy in [-.27,0,.26]: paint(im,ellipse(x,y,0,cy,.095,.095),WHITE)
        elif k in [27,28]:
            centers=[0] if k==27 else [-.18,.17]
            for j,cx in enumerate(centers):
                cy=-.27+(j*.09 if k==28 else 0)
                paint(im,ellipse(x,y,cx,cy,.075,.075),BLACK)
                paint(im,pathmask(x,y,[(cx,cy+.08),(cx,cy+.34)],.045*thick),BLACK)
                paint(im,pathmask(x,y,[(cx,cy+.16),(cx-.14,cy+.27)],.035*thick),BLACK)
                paint(im,pathmask(x,y,[(cx,cy+.16),(cx+.14,cy+.26)],.035*thick),BLACK)
                paint(im,pathmask(x,y,[(cx,cy+.34),(cx-.13,cy+.52)],.038*thick),BLACK)
                paint(im,pathmask(x,y,[(cx,cy+.34),(cx+.13,cy+.50)],.038*thick),BLACK)
        elif k==29:
            paint(im,ellipse(x,y,-.25,.25,.18,.18) & ~ellipse(x,y,-.25,.25,.13,.13),BLACK)
            paint(im,ellipse(x,y,.27,.25,.18,.18) & ~ellipse(x,y,.27,.25,.13,.13),BLACK)
            paint(im,pathmask(x,y,[(-.25,.25),(-.05,-.02),(.12,.25),(-.25,.25),(.02,.23),(.27,.25),(.08,-.10)],.035*thick),BLACK)
            paint(im,ellipse(x,y,.10,-.25,.07,.07),BLACK)
        elif k==30:
            for ang in [0,math.pi/3,2*math.pi/3]:
                dx,dy=.43*math.cos(ang),.43*math.sin(ang)
                paint(im,pathmask(x,y,[(-dx,-dy),(dx,dy)],.035*thick),BLACK)
                for sg in [-1,1]:
                    ex,ey=sg*dx,sg*dy
                    paint(im,pathmask(x,y,[(ex,ey),(ex-sg*.12*math.cos(ang-.55),ey-sg*.12*math.sin(ang-.55))],.027),BLACK)
                    paint(im,pathmask(x,y,[(ex,ey),(ex-sg*.12*math.cos(ang+.55),ey-sg*.12*math.sin(ang+.55))],.027),BLACK)
        elif k==31:
            # Stylised leaping deer silhouette.
            body=ellipse(x,y,-.02,.04,.28,.15); paint(im,body,BLACK)
            paint(im,pathmask(x,y,[(.17,-.02),(.33,-.25),(.42,-.28)],.055*thick),BLACK)
            paint(im,ellipse(x,y,.43,-.29,.08,.06),BLACK)
            paint(im,pathmask(x,y,[(-.18,.12),(-.38,.39)],.040*thick),BLACK)
            paint(im,pathmask(x,y,[(-.02,.14),(-.02,.42)],.040*thick),BLACK)
            paint(im,pathmask(x,y,[(.14,.10),(.34,.35)],.040*thick),BLACK)
            paint(im,pathmask(x,y,[(.35,-.31),(.30,-.46),(.42,-.39)],.025),BLACK)
    elif k==32:
        support=circle_sign(im,x,y,'gray',thick); stripes(im,x,y,support,.035*thick)
    elif 33 <= k <= 40:
        support=circle_sign(im,x,y,'blue',thick)
        if k in [33,34]:
            sg=1 if k==33 else -1
            pts=[(0,.48),(0,.03),(sg*.12,-.18),(sg*.43,-.35)]
            paint(im,pathmask(x,y,pts,.085*thick),WHITE)
            paint(im,arrowhead(x,y,pts[-1],(sg,-.45),.22),WHITE)
        elif k==35:
            paint(im,pathmask(x,y,[(0,.49),(0,-.43)],.09*thick),WHITE); paint(im,arrowhead(x,y,(0,-.52),(0,-1),.24),WHITE)
        elif k in [36,37]:
            sg=1 if k==36 else -1
            paint(im,pathmask(x,y,[(0,.50),(0,-.45)],.075*thick),WHITE); paint(im,arrowhead(x,y,(0,-.53),(0,-1),.21),WHITE)
            paint(im,pathmask(x,y,[(0,.10),(sg*.40,-.20)],.075*thick),WHITE); paint(im,arrowhead(x,y,(sg*.46,-.25),(sg,-.65),.20),WHITE)
        elif k in [38,39]:
            sg=1 if k==38 else -1
            a=(-sg*.35,-.38); b=(sg*.35,.36)
            paint(im,pathmask(x,y,[a,b],.095*thick),WHITE); paint(im,arrowhead(x,y,b,(sg,1),.23),WHITE)
        else:
            # Three tangential arrows around a roundabout.
            for ang in [-math.pi/2, math.pi/6, 5*math.pi/6]:
                a1=ang-.55; a2=ang+.45
                pts=[(.35*math.cos(t),.35*math.sin(t)) for t in np.linspace(a1,a2,7)]
                paint(im,pathmask(x,y,pts,.065*thick),WHITE)
                tip=pts[-1]; direction=(-math.sin(a2),math.cos(a2))
                paint(im,arrowhead(x,y,tip,direction,.15),WHITE)
    elif k in [41,42]:
        support=circle_sign(im,x,y,'gray',thick)
        if k==41:
            paint(im,carmask(x,y,-.22,.05,.72),np.array([.30,.30,.29]))
            paint(im,carmask(x,y,.23,.05,.72),BLACK)
        else:
            paint(im,truckmask(x,y,0,.02,1.12),BLACK)
        stripes(im,x,y,support,.030*thick)

    # Area-average the analytic supersampled drawing to 32 square pixels.
    im=im.reshape(32,S,32,S,3).mean((1,3))
    sup=support.reshape(32,S,32,S).mean((1,3))
    return im, sup


def feature_kernels():
    yy,xx=np.mgrid[-2:3,-2:3].astype(np.float32)
    def g(sig):
        z=np.exp(-(xx*xx+yy*yy)/(2*sig*sig)); return z/z.sum()
    def deriv(axis,sig):
        z=(-xx if axis=='x' else -yy)*np.exp(-(xx*xx+yy*yy)/(2*sig*sig))
        z-=z.mean(); return z/(np.abs(z).sum()+1e-9)
    def lap(sig):
        r=xx*xx+yy*yy; z=(r-2*sig*sig)*np.exp(-r/(2*sig*sig))
        z-=z.mean(); return z/(np.abs(z).sum()+1e-9)
    K=[]; alpha=[]
    def add(rgb,sp,a):
        K.append(np.asarray(rgb,np.float32)[:,None,None]*sp[None]); alpha.append(a)
    gf=g(.82)
    add([1,0,0],gf,.72); add([0,1,0],gf,.72); add([0,0,1],gf,.72)
    add([.299,.587,.114],gf,1.05)
    ro=np.array([1,-.5,-.5]); bo=np.array([-.5,-.5,1.])
    add(ro,gf,1.05); add(-ro,gf,.78); add(bo,gf,1.05); add(-bo,gf,.78)
    lum=np.array([.299,.587,.114])
    for axis in ['x','y']:
        d=deriv(axis,.82); add(lum,d,.82); add(lum,-d,.82)
    lp=lap(.9); add(lum,lp,.58); add(lum,-lp,.58)
    for axis in ['x','y']:
        d=deriv(axis,1.35); add(lum,d,.58); add(lum,-d,.58)
    return np.stack(K), np.asarray(alpha,np.float32)


def conv_features(images,K):
    n=images.shape[0]; p=np.pad(images,((0,0),(0,0),(2,2),(2,2)))
    out=np.zeros((n,len(K),32,32),np.float32)
    for dy in range(5):
        for dx in range(5):
            out += np.einsum('nchw,oc->nohw',p[:,:,dy:dy+32,dx:dx+32],K[:,:,dy,dx],optimize=True)
    mu=out.mean((2,3),keepdims=True)
    out=(out-mu)/np.sqrt(((out-mu)**2).mean((2,3),keepdims=True)+1e-5)
    out=np.maximum(out,0)
    return out.reshape(n,len(K),16,2,16,2).mean((3,5))


def make_templates(K,alpha):
    # Equal-size, class-major transformation bank. Aspect, scale, centring,
    # modest camera roll, and two conventional stroke weights are represented.
    scales=[.84,1.02]
    geos=[(1.0,0),(.90,0),(1.10,0),(1.0,-7),(1.0,7)]
    shifts=[(0,0),(-.105,0),(.105,0),(0,-.105),(0,.105)]
    pars=[(sc,asp,rot,tx,ty,sty) for sc in scales for asp,rot in geos
          for tx,ty in shifts for sty in [0,1]]
    nv=len(pars); D=len(K)*32*32
    W=np.empty((43*nv,D),np.float32)
    batch_images=[]; batch_support=[]; batch_rows=[]
    def flush():
        if not batch_images: return
        F=conv_features(np.stack(batch_images).transpose(0,3,1,2),K)
        for z,(f,sup,row) in enumerate(zip(F,batch_support,batch_rows)):
            # Undo the helper's 2x pool so its values match the scorer-side
            # unpooled feature grid; interpolation is unnecessary for these
            # low-pass analytic channels.
            f=np.repeat(np.repeat(f,2,axis=1),2,axis=2)
            # The scorer-side input to this readout is already standardized per
            # channel over the whole frame. Correlate against the corresponding
            # globally centered template, while gently emphasizing the centre.
            gy,gx=np.mgrid[:32,:32]; rr=((gx-15.5)/9.8)**2+((gy-15.5)/9.8)**2
            # Interior symbols carry the class identity; the outer silhouette is
            # deliberately given less leverage because it is shared by families.
            importance=.42+2.10*np.exp(-rr/2)
            parts=[]
            for c in range(len(K)):
                v=(f[c]-f[c].mean())*importance
                norm=np.sqrt((v*v).sum())
                if norm>1e-6: v=v/norm*alpha[c]
                parts.append(v)
            v=np.stack(parts)
            W[row]=v.reshape(-1)
        batch_images.clear(); batch_support.clear(); batch_rows.clear()
    row=0
    for k in range(43):
        for par in pars:
            im,sup=render(k,*par)
            batch_images.append(im); batch_support.append(sup); batch_rows.append(row); row+=1
            if len(batch_images)>=96: flush()
    flush()
    return W,nv


def build(dest):
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    K,alpha=feature_kernels()
    T,nv=make_templates(K,alpha)
    layers=[
      {"name":"features","op":"conv","inputs":["image"],"out":int(len(K)),"kernel":[5,5],"stride":1,"padding":2},
      {"name":"standardized","op":"norm","inputs":["features"]},
      {"name":"positive_features","op":"relu","inputs":["standardized"]},
      {"name":"template_scores","op":"dense","inputs":["positive_features"],"out":int(43*nv)},
      {"name":"score_bank","op":"reshape","inputs":["template_scores"],"shape":[43,int(nv),1]},
      {"name":"best_view","op":"maxpool","inputs":["score_bank"],"size":[int(nv),1],"stride":[int(nv),1]},
      {"name":"scores","op":"reshape","inputs":["best_view"],"shape":[43]}
    ]
    (dest/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':')))
    np.savez(dest/'weights.npz',**{
      'features.W':K.astype(np.float32),'features.b':np.zeros(len(K),np.float32),
      'template_scores.W':T.astype(np.float32),'template_scores.b':np.zeros(43*nv,np.float32)})

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: build.py OUTPUT_FOLDER')
    build(sys.argv[1])
