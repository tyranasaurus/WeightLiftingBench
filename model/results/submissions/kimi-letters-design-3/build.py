"""Build a data-free, hand-designed ASL shape-correlation network.
All anatomy, poses, filters and constants are specified below from general
knowledge of the ASL alphabet.  No data files are read.
"""
import sys, json
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
S = 3                         # antialiasing supersampling
N = 28
unit = 10.5

# Each primitive is a capsule: x0,y0,x1,y1,radius.  Coordinates describe a
# generic right hand, palm facing the viewer, with the thumb on the right.
def palm(w=1.0, h=1.0):
    return [('e', 0.0, .24, .38*w, .46*h),
            ('c', -.15*w, .46, -.15*w, .98, .20*w),
            ('c',  .15*w, .46,  .15*w, .98, .20*w)]

bases = {'p':(-.27,-.04), 'r':(-.09,-.12), 'm':(.09,-.13), 'i':(.26,-.06)}
ups = {'p':(-.32,-.62), 'r':(-.12,-.88), 'm':(.09,-.97), 'i':(.30,-.82)}
widths = {'p':.105, 'r':.12, 'm':.125, 'i':.12}

def finger(shapes, name, end=None, width=None, start=None):
    if start is None: start = bases[name]
    if end is None: end = ups[name]
    shapes.append(('c', start[0],start[1],end[0],end[1],
                   widths[name] if width is None else width))

def folded(shapes, names='prmi', height=.34):
    for name in names:
        x,y = bases[name]
        finger(shapes, name, (x, -height), widths[name]*1.08)
        finger(shapes, name, (x-.025,-.08), widths[name]*1.05,
               (x,-height+.025))

def thumb(shapes, points, r=.145):
    for a,b in zip(points[:-1],points[1:]):
        shapes.append(('c',a[0],a[1],b[0],b[1],r))

def anatomy(k, variant):
    """Return silhouette primitives and dark anatomical crease primitives."""
    q = palm(1.0 + .08*(variant-.5), 1.0)
    lines = []
    # Upright open-hand letters.
    if k == 1:                         # B, four fingers together
        for f in 'prmi': finger(q,f)
        thumb(q,[(.36,.25),(.12,.03),(-.20,-.02)],.13)
        lines += [('c',.30,.23,-.18,-.01,.025)]
    elif k == 2:                       # C, curved fingers and thumb
        q = palm(1.0,.88)
        pts=[(-.28,-.15),(-.35,-.55),(-.10,-.82),(.30,-.80),(.58,-.58)]
        thumb(q,pts,.16)
        thumb(q,[(-.27,.28),(-.38,.02),(-.15,-.10),(.25,-.13),(.51,-.35)],.15)
        # Add the parallel finger mass along the upper arc.
        thumb(q,[(-.15,-.20),(-.18,-.51),(.03,-.66),(.34,-.64),(.55,-.49)],.19)
    elif k == 3:                       # D, index up, remaining fingers make a ring
        folded(q,'prm',.25); finger(q,'i')
        thumb(q,[(.36,.23),(.32,-.18),(.08,-.37),(-.15,-.26)],.14)
        thumb(q,[(-.20,-.08),(-.25,-.31),(-.04,-.42),(.18,-.30)],.13)
        lines += [('c',.25,.08,-.05,-.29,.025)]
    elif k == 5:                       # F, index-thumb circle, three fingers up
        for f in 'prm': finger(q,f)
        thumb(q,[(.34,.22),(.47,-.10),(.36,-.32),(.17,-.26),(.18,-.08)],.13)
        finger(q,'i',(.29,-.25),.12)
        lines += [('c',.23,-.04,.36,-.20,.022)]
    elif k == 8:                       # I, little finger up
        folded(q,'rmi',.29); finger(q,'p')
        thumb(q,[(.35,.22),(.12,-.04),(-.08,-.12)],.14)
        lines += [('c',.27,.09,-.08,-.09,.025)]
    elif k == 9:                       # K, index up, middle diagonal, thumb between
        folded(q,'pr',.28); finger(q,'i')
        finger(q,'m',(.52,-.70),.125)
        thumb(q,[(.35,.25),(.28,-.13),(.37,-.40)],.14)
        lines += [('c',.22,.07,.30,-.34,.025)]
    elif k == 10:                      # L
        folded(q,'prm',.30); finger(q,'i')
        thumb(q,[(.32,.20),(.67,.02),(.94,-.02)],.15)
    elif k == 13:                      # O, all fingers curve to thumb
        q = palm(.95,.90)
        thumb(q,[(.31,.25),(.43,-.08),(.30,-.43),(.04,-.62),(-.23,-.48),(-.26,-.19)],.16)
        thumb(q,[(-.25,-.10),(-.34,-.43),(-.12,-.72),(.17,-.72),(.37,-.48)],.17)
        lines += [('c',-.20,-.17,-.18,-.46,.022)]
    elif k == 16:                      # R, crossed index and middle fingers
        folded(q,'pr',.29)
        finger(q,'i',(.02,-.93),.12)
        finger(q,'m',(.30,-.87),.125)
        thumb(q,[(.36,.24),(.12,-.04),(-.12,-.10)],.135)
        lines += [('c',.20,-.26,.15,-.66,.022)]
    elif k == 19:                      # U, two fingers together
        folded(q,'pr',.29); finger(q,'m'); finger(q,'i',(.25,-.86))
        thumb(q,[(.36,.25),(.10,-.04),(-.13,-.08)],.135)
        lines += [('c',.18,-.22,.18,-.72,.022)]
    elif k == 20:                      # V
        folded(q,'pr',.29)
        finger(q,'m',(-.02,-.96),.125)
        finger(q,'i',(.48,-.83),.12)
        thumb(q,[(.36,.25),(.12,-.02),(-.10,-.08)],.135)
    elif k == 21:                      # W, three fingers spread
        folded(q,'p',.28)
        finger(q,'r',(-.30,-.87),.12)
        finger(q,'m',(.06,-.99),.125)
        finger(q,'i',(.43,-.83),.12)
        thumb(q,[(.36,.25),(.06,-.02),(-.25,-.18)],.135)
    elif k == 23:                      # Y, thumb and little finger extended
        folded(q,'rmi',.28)
        finger(q,'p',(-.62,-.63),.115)
        thumb(q,[(.32,.25),(.68,.35),(.93,.52)],.16)
    elif k == 6:                       # G, index and thumb point right
        q = palm(.95,.90); folded(q,'prm',.25)
        finger(q,'i',(.91,-.20),.125, (.16,-.13))
        thumb(q,[(.26,.20),(.62,.10),(.91,.03)],.145)
    elif k == 7:                       # H, two fingers together point right
        q = palm(.95,.90); folded(q,'pr',.25)
        finger(q,'m',(.92,-.13),.125,(.08,-.12))
        finger(q,'i',(.90,.10),.12,(.18,.05))
        thumb(q,[(.31,.29),(.04,.06),(-.14,-.03)],.135)
    elif k == 15:                      # Q, index and thumb point down
        q = palm(.95,.83); folded(q,'prm',.20)
        finger(q,'i',(.27,.75),.125)
        thumb(q,[(-.28,.12),(-.35,.52),(-.30,.79)],.15)
    elif k == 14:                      # P, the K configuration pointing down/left
        q = palm(.95,.88); folded(q,'pr',.22)
        finger(q,'i',(-.49,.55),.12)
        finger(q,'m',(-.70,.19),.125)
        thumb(q,[(.28,.18),(-.12,.25),(-.37,.36)],.14)
        lines += [('c',.15,.12,-.29,.30,.025)]
    elif k == 22:                      # X, hooked index
        folded(q,'prm',.28)
        finger(q,'i',(.29,-.48),.12)
        thumb(q,[(.29,-.46),(.48,-.38),(.45,-.21)],.12)
        thumb(q,[(.35,.24),(.11,-.03),(-.10,-.10)],.135)
    else:                              # A,E,M,N,S,T: fist family
        heights = {0:.36,4:.25,11:.31,12:.32,17:.34,18:.32}
        folded(q,'prmi',heights[k])
        if k == 0:                     # A, thumb beside the fist
            thumb(q,[(.34,.40),(.43,.06),(.39,-.30)],.15)
        elif k == 4:                   # E, thumb tightly folded across palm
            thumb(q,[(.31,.22),(.02,.02),(-.25,-.05)],.125)
            lines += [('c',.22,.16,-.22,-.03,.024)]
        elif k == 11:                  # M, thumb beneath three fingers
            thumb(q,[(.34,.30),(.16,-.04),(-.16,-.17)],.135)
            lines += [('c',-.24,-.30,-.20,.02,.023),
                      ('c',-.04,-.34,-.03,.01,.023),
                      ('c',.16,-.32,.15,.00,.023)]
        elif k == 12:                  # N, thumb beneath two fingers
            thumb(q,[(.34,.29),(.15,-.06),(-.09,-.20)],.135)
            lines += [('c',-.12,-.32,-.10,.01,.023),
                      ('c',.10,-.33,.10,.00,.023)]
        elif k == 17:                  # S, thumb wrapped over the fingers
            thumb(q,[(.37,.24),(.19,-.12),(-.22,-.18)],.15)
            lines += [('c',.29,-.08,-.22,-.15,.028)]
        elif k == 18:                  # T, thumb tip between index and middle
            thumb(q,[(.34,.29),(.21,-.05),(.17,-.38)],.135)
            lines += [('c',.20,.05,.17,-.31,.023)]
    return q, lines

def render(k, variant, scale, dx, dy, angle, mirror):
    """Independently draw one generic hand template; no cross-template stats."""
    shapes, lines = anatomy(k, 0)
    # Independent anatomical variation: palm/finger widths, lengths and small
    # joint changes.  The fixed seed only reproduces the specified drawings.
    rng=np.random.default_rng(7919*k+variant)
    sx=float(rng.uniform(.85,1.15)); sy=float(rng.uniform(.90,1.10))
    varied=[]
    for sh in shapes:
        if sh[0]=='e':
            _,cx,cy,rx,ry=sh
            varied.append(('e',cx*sx,cy*sy,rx*float(rng.uniform(.88,1.12)),
                           ry*float(rng.uniform(.90,1.10))))
        else:
            _,x0,y0,x1,y1,r=sh
            varied.append(('c',x0*sx+float(rng.uniform(-.025,.025)),
                           y0*sy+float(rng.uniform(-.025,.025)),
                           x1*sx+float(rng.uniform(-.035,.035)),
                           y1*sy+float(rng.uniform(-.035,.035)),
                           r*float(rng.uniform(.88,1.12))))
    shapes=varied
    lines=[(sh[0],sh[1]*sx,sh[2]*sy,sh[3]*sx,sh[4]*sy,sh[5]) for sh in lines]
    yy,xx = np.mgrid[0:N*S,0:N*S].astype(np.float32)
    xx = (xx+.5)/S; yy=(yy+.5)/S
    ca,sa = np.cos(angle),np.sin(angle)
    # Fit this one drawing's own bounding box to the hand-crop convention.
    # This uses no information from any other drawing or photograph.
    bounds=[]
    for sh in shapes:
        if sh[0]=='e':
            _,x,y,rx,ry=sh
            if mirror: x=-x
            X,Y=x*ca-y*sa,x*sa+y*ca
            ex=np.sqrt((rx*ca)**2+(ry*sa)**2)
            ey=np.sqrt((rx*sa)**2+(ry*ca)**2)
            bounds.append((X-ex,X+ex,Y-ey,Y+ey))
        else:
            _,x0,y0,x1,y1,r=sh
            if mirror: x0=-x0; x1=-x1
            X0,Y0=x0*ca-y0*sa,x0*sa+y0*ca
            X1,Y1=x1*ca-y1*sa,x1*sa+y1*ca
            bounds.append((min(X0,X1)-r,max(X0,X1)+r,min(Y0,Y1)-r,max(Y0,Y1)+r))
    left=min(b[0] for b in bounds); right=max(b[1] for b in bounds)
    top=min(b[2] for b in bounds); bottom=max(b[3] for b in bounds)
    cx0=(left+right)/2; cy0=(top+bottom)/2
    u=25.0*scale/max(right-left,bottom-top)
    def project(x,y):
        X,Y=x*ca-y*sa,x*sa+y*ca
        return 14+dx+(X-cx0)*u,14+dy+(Y-cy0)*u
    def points():
        for sh in shapes:
            if sh[0]=='e':
                _,x,y,rx,ry=sh
                if mirror: x=-x
                a,b=project(x,y)
                yield sh[0],a,b,rx*u,ry*u,0,0
            else:
                _,x0,y0,x1,y1,r=sh
                if mirror: x0=-x0; x1=-x1
                a,b=project(x0,y0); c,d=project(x1,y1)
                yield ('c',a,b,c,d,r*u,0)
    mask=np.zeros((N*S,N*S),np.float32)
    for typ,a,b,c,d,r,_ in points():
        if typ=='e':
            v=1-((xx-a)/c)**2-((yy-b)/d)**2
        else:
            vx,vy=c-a,d-b
            t=np.clip(((xx-a)*vx+(yy-b)*vy)/(vx*vx+vy*vy+1e-8),0,1)
            v=r-np.sqrt((xx-a-t*vx)**2+(yy-b-t*vy)**2)
        mask=np.maximum(mask,(v>0).astype(np.float32))
    img=.12+.76*mask
    # Creases are part of the independently specified anatomical drawing.
    for sh in lines:
        _,x0,y0,x1,y1,r=sh
        if mirror: x0=-x0; x1=-x1
        a,b=project(x0,y0)
        c,d=project(x1,y1)
        vx,vy=c-a,d-b
        t=np.clip(((xx-a)*vx+(yy-b)*vy)/(vx*vx+vy*vy+1e-8),0,1)
        ink=(np.sqrt((xx-a-t*vx)**2+(yy-b-t*vy)**2)<r*u)
        img[ink & (mask>0)]=.36
    return img.reshape(N,S,N,S).mean((1,3)).astype(np.float32)

def conv(a,k):
    p=k.shape[-1]//2
    z=np.pad(a,((0,0),(p,p),(p,p))) if a.ndim==3 else np.pad(a,((p,p),(p,p)))
    if a.ndim==2: z=z[None]
    win=np.lib.stride_tricks.sliding_window_view(z,(k.shape[-1],k.shape[-1]),axis=(1,2))
    return np.einsum('chwkl,okl->ohw',win,k,optimize=True)

def normalize(a):
    a=a-a.mean(axis=(-2,-1),keepdims=True)
    return a/np.sqrt(a.var(axis=(-2,-1),keepdims=True)+1e-5)

def pool2(a):
    return a.reshape(a.shape[0],14,2,14,2).mean((2,4))

g_y,g_x=np.mgrid[-2:3,-2:3]
gauss=np.exp(-(g_x*g_x+g_y*g_y)/(2*1.1**2)); gauss/=gauss.sum()
gauss=np.float32(gauss)
dog=np.exp(-(g_x*g_x+g_y*g_y)/(2*.75**2)); dog/=dog.sum()
dog2=np.exp(-(g_x*g_x+g_y*g_y)/(2*1.65**2)); dog2/=dog2.sum()
dog=np.float32(dog-dog2)
gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)/4
gy=gx.T.copy()
kernels=np.stack([gx,gy,(gx+gy)/np.sqrt(2),(gx-gy)/np.sqrt(2)]).astype(np.float32)

# Equal numbers of templates for every class.  Variants are explicit general
# pose possibilities, never selected by their scores.
poses=[(v,s,dx,dy,a,m)
       for v in range(10)
       for s in (.90,1.08)
       for dx,dy in ((0,0),(-2,0),(2,0),(0,-2),(0,2))
       for a in np.deg2rad((-10.,0.,10.))
       for m in (False,True)]
V=len(poses)*2
raw=[]; coarse=[]; seg_light=[]; seg_dark=[]
for k in range(24):
    for pose in poses:
        im=render(k,*pose)[None]
        sm=conv(im,gauss[None])
        silhouette=np.clip((sm-.12)/.76,0,1)
        # Fists need a stronger exclusion term because their silhouettes are
        # subsets of many open-hand poses.
        penalty=3.0 if k in (0,4,11,12,17,18) else 2.0
        fs=normalize((1+penalty)*silhouette-penalty)[0]
        fc=normalize(pool2((1+penalty)*silhouette-penalty))
        seg=normalize(np.maximum(normalize(sm)-.25,0))[0]
        for sign in (1.,-1.):
            raw.append(sign*fs.ravel()/(3*fs.size))
            coarse.append(sign*fc.ravel()/(3*fc.size))
            # Alternate rows are the light-hand and dark-hand interpretations.
            seg_light.append(seg.ravel()/(3*seg.size) if sign>0 else np.zeros(seg.size,np.float32))
            seg_dark.append(seg.ravel()/(3*seg.size) if sign<0 else np.zeros(seg.size,np.float32))
raw=np.asarray(raw,np.float32); coarse=np.asarray(coarse,np.float32)
seg_light=np.asarray(seg_light,np.float32); seg_dark=np.asarray(seg_dark,np.float32)

layers=[
 {'name':'smooth','op':'conv','inputs':['image'],'out':1,'kernel':[5,5],'stride':1,'padding':2},
 {'name':'smooth_n','op':'norm','inputs':['smooth']},
 {'name':'raw_corr','op':'dense','inputs':['smooth_n'],'out':24*V},
 {'name':'pool','op':'avgpool','inputs':['smooth'],'size':[2,2],'stride':[2,2]},
 {'name':'pool_n','op':'norm','inputs':['pool']},
 {'name':'coarse_corr','op':'dense','inputs':['pool_n'],'out':24*V},
 {'name':'light_shift','op':'conv','inputs':['smooth_n'],'out':1,'kernel':[1,1],'stride':1,'padding':0},
 {'name':'light_relu','op':'relu','inputs':['light_shift']},
 {'name':'light_n','op':'norm','inputs':['light_relu']},
 {'name':'dark_shift','op':'conv','inputs':['smooth_n'],'out':1,'kernel':[1,1],'stride':1,'padding':0},
 {'name':'dark_relu','op':'relu','inputs':['dark_shift']},
 {'name':'dark_n','op':'norm','inputs':['dark_relu']},
 {'name':'seg_light_corr','op':'dense','inputs':['light_n'],'out':24*V},
 {'name':'seg_dark_corr','op':'dense','inputs':['dark_n'],'out':24*V},
 {'name':'raw_grid','op':'reshape','inputs':['raw_corr'],'shape':[24,V,1]},
 {'name':'coarse_grid','op':'reshape','inputs':['coarse_corr'],'shape':[24,V,1]},
 {'name':'seg_light_grid','op':'reshape','inputs':['seg_light_corr'],'shape':[24,V,1]},
 {'name':'seg_dark_grid','op':'reshape','inputs':['seg_dark_corr'],'shape':[24,V,1]},
 {'name':'combined','op':'add','inputs':['raw_grid','coarse_grid','seg_light_grid','seg_dark_grid']},
 {'name':'best','op':'maxpool','inputs':['combined'],'size':[V,1],'stride':[V,1]},
 {'name':'scores','op':'reshape','inputs':['best'],'shape':[24]}
]
weights={
 'smooth.W':gauss[None,None], 'smooth.b':np.zeros(1,np.float32),
 'raw_corr.W':raw, 'raw_corr.b':np.zeros(24*V,np.float32),
 'coarse_corr.W':coarse, 'coarse_corr.b':np.zeros(24*V,np.float32),
 'light_shift.W':np.ones((1,1,1,1),np.float32), 'light_shift.b':np.array([-.25],np.float32),
 'dark_shift.W':-np.ones((1,1,1,1),np.float32), 'dark_shift.b':np.array([-.25],np.float32),
 'seg_light_corr.W':seg_light, 'seg_light_corr.b':np.zeros(24*V,np.float32),
 'seg_dark_corr.W':seg_dark, 'seg_dark_corr.b':np.zeros(24*V,np.float32)
}
(OUT/'network.json').write_text(json.dumps({'layers':layers},indent=2))
np.savez(OUT/'weights.npz',**weights)
print('wrote',len(poses),'individual poses per class;',sum(x.size for x in weights.values()),'weights')
