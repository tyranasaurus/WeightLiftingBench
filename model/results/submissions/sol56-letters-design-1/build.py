#!/usr/bin/env python3
"""Build a data-free, hand-designed ASL shape-template network."""
import json, math, os, sys
import numpy as np


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def ellipse(X, Y, cx, cy, rx, ry, angle=0.0, sharp=28.0):
    a = math.radians(angle); ca, sa = math.cos(a), math.sin(a)
    x = ca*(X-cx) + sa*(Y-cy); y = -sa*(X-cx) + ca*(Y-cy)
    q = np.sqrt((x/rx)**2 + (y/ry)**2)
    return sigmoid((1.0-q)*sharp)


def capsule(X, Y, x1, y1, x2, y2, r, sharp=32.0):
    vx, vy = x2-x1, y2-y1
    t = np.clip(((X-x1)*vx + (Y-y1)*vy)/(vx*vx+vy*vy+1e-9), 0.0, 1.0)
    d = np.sqrt((X-(x1+t*vx))**2 + (Y-(y1+t*vy))**2)
    return sigmoid((r-d)*sharp)


def ring(X,Y,cx,cy,rx,ry,t=.10):
    outer=ellipse(X,Y,cx,cy,rx,ry,sharp=35)
    inner=ellipse(X,Y,cx,cy,max(.02,rx-t),max(.02,ry-t),sharp=35)
    return np.clip(outer-inner,0,1)


def rotate_coords(X,Y,ang):
    # Coordinates in the unrotated object for an object displayed at angle ang.
    a=math.radians(ang); ca,sa=math.cos(a),math.sin(a)
    return ca*X+sa*Y, -sa*X+ca*Y


def palm(X,Y,cx=0,cy=-.17,rx=.30,ry=.38):
    m=ellipse(X,Y,cx,cy,rx,ry)
    m=np.maximum(m,capsule(X,Y,cx,-.40,cx,-.93,.155))
    return m


def fist(X,Y):
    m=palm(X,Y,0,-.18,.33,.38)
    xs=[-.235,-.08,.08,.235]; ys=[.185,.235,.24,.18]
    rs=[.13,.135,.135,.13]
    for x,y,r in zip(xs,ys,rs): m=np.maximum(m,ellipse(X,Y,x,y,r,r*.90))
    return m


def add_line(c,X,Y,x1,y1,x2,y2,r=.022,a=1.0):
    return np.maximum(c,a*capsule(X,Y,x1,y1,x2,y2,r,sharp=50))


def upright_model(k,X,Y):
    """Generic hand silhouette and crease map, from ASL anatomy rather than samples."""
    c=np.zeros_like(X)
    if k in (0,4,11,12,17,18):
        m=fist(X,Y)
        if k==0: # A: thumb lies diagonally along the closed fingers
            c=add_line(c,X,Y,-.22,-.22,.26,.13,.030,1.0)
            c=add_line(c,X,Y,.23,.12,.29,-.23,.020,.7)
        elif k==4: # E: all fingertips curl down
            for x in [-.22,-.075,.075,.22]: c=add_line(c,X,Y,x,.25,x-.025,-.05,.025,.9)
            c=add_line(c,X,Y,-.23,-.08,.20,-.18,.026,.7)
        elif k==11: # M: three fingers cover the thumb
            for x in [-.16,0,.16]: c=add_line(c,X,Y,x,.25,x-.03,-.15,.030,1.0)
            c=add_line(c,X,Y,-.23,-.12,.22,-.02,.023,.65)
        elif k==12: # N: two fingers cover the thumb
            for x in [-.09,.10]: c=add_line(c,X,Y,x,.27,x-.035,-.17,.032,1.0)
            c=add_line(c,X,Y,-.22,-.08,.22,.02,.023,.7)
        elif k==17: # S: thumb crosses the front of the fist
            c=add_line(c,X,Y,-.25,.03,.22,-.12,.040,1.0)
            c=add_line(c,X,Y,-.12,.18,.24,.04,.020,.65)
        else: # T: thumb emerges between index and middle
            c=add_line(c,X,Y,.00,-.18,.04,.25,.045,1.0)
            c=add_line(c,X,Y,-.22,.04,.22,-.02,.021,.65)
        return m,c
    if k==1: # B
        m=palm(X,Y)
        specs=[(-.225,.02,-.255,.66,.078),(-.085,.03,-.10,.86,.078),(.065,.03,.065,.91,.080),(.205,.01,.22,.76,.078)]
        for x1,y1,x2,y2,r in specs: m=np.maximum(m,capsule(X,Y,x1,y1,x2,y2,r))
        c=add_line(c,X,Y,-.20,-.05,.25,.08,.030,.8)
        return m,c
    if k==2: # C: curved fingers and thumb leave an opening to the right
        m=np.zeros_like(X)
        pts=[]
        for a in np.linspace(math.radians(55),math.radians(305),12): pts.append((-.02+.40*math.cos(a),.12+.48*math.sin(a)))
        for p,q in zip(pts[:-1],pts[1:]): m=np.maximum(m,capsule(X,Y,p[0],p[1],q[0],q[1],.105))
        m=np.maximum(m,capsule(X,Y,-.17,-.30,-.20,-.92,.16))
        c=add_line(c,X,Y,-.22,-.18,.08,-.30,.024,.5)
        return m,c
    if k==3: # D
        m=palm(X,Y)
        for x,y in [(-.22,.16),(-.07,.21),(.07,.17)]: m=np.maximum(m,ellipse(X,Y,x,y,.135,.12))
        m=np.maximum(m,capsule(X,Y,.15,.08,.16,.91,.080))
        hole=ellipse(X,Y,-.02,.13,.11,.13,sharp=40)
        m=np.clip(m*(1-.90*hole),0,1)
        return m,c
    if k==5: # F: three fingers up, index and thumb form a ring
        m=palm(X,Y)
        for vals in [(-.23,.0,-.27,.68,.072),(-.09,.02,-.10,.88,.073),(.05,.02,.06,.78,.073)]: m=np.maximum(m,capsule(X,Y,*vals))
        m=np.maximum(m,ring(X,Y,.245,.18,.18,.19,.075))
        hole=ellipse(X,Y,.245,.18,.09,.10,sharp=45); m=np.clip(m*(1-.95*hole),0,1)
        return m,c
    if k in (6,7): # G and H, pointing horizontally
        m=ellipse(X,Y,-.27,-.10,.30,.31)
        m=np.maximum(m,capsule(X,Y,-.42,-.28,-.86,-.58,.17))
        m=np.maximum(m,capsule(X,Y,-.12,.15,.78,.17,.080))
        if k==6:
            m=np.maximum(m,capsule(X,Y,-.05,-.01,.45,-.08,.075))
            c=add_line(c,X,Y,-.12,.05,.42,.03,.018,.7)
        else:
            m=np.maximum(m,capsule(X,Y,-.12,.00,.76,.01,.078))
            c=add_line(c,X,Y,-.12,.075,.65,.09,.018,.6)
        return m,c
    if k==8: # I, little finger
        m=fist(X,Y); m=np.maximum(m,capsule(X,Y,-.225,.16,-.30,.88,.075))
        c=add_line(c,X,Y,-.12,.12,.22,-.06,.025,.8)
        return m,c
    if k==9: # K
        m=palm(X,Y)
        m=np.maximum(m,capsule(X,Y,-.07,.02,-.25,.88,.078))
        m=np.maximum(m,capsule(X,Y,.08,.03,.34,.78,.078))
        m=np.maximum(m,capsule(X,Y,.02,-.06,.39,.29,.075))
        c=add_line(c,X,Y,.02,-.04,.23,.34,.022,.8)
        return m,c
    if k==10: # L
        m=palm(X,Y); m=np.maximum(m,capsule(X,Y,.09,.03,.10,.91,.080)); m=np.maximum(m,capsule(X,Y,.05,-.04,.75,.02,.085))
        return m,c
    if k==13: # O
        m=ring(X,Y,0,.18,.37,.49,.13)
        m=np.maximum(m,ellipse(X,Y,0,-.18,.31,.34))
        m=np.maximum(m,capsule(X,Y,0,-.38,0,-.93,.16))
        hole=ellipse(X,Y,.0,.22,.20,.28,sharp=45); m=np.clip(m*(1-.98*hole),0,1)
        return m,c
    if k==16: # R, crossed index and middle fingers
        m=palm(X,Y)
        m=np.maximum(m,capsule(X,Y,-.07,.02,.17,.89,.078))
        m=np.maximum(m,capsule(X,Y,.09,.02,-.10,.86,.078))
        c=add_line(c,X,Y,-.10,.19,.14,.62,.020,.7)
        return m,c
    if k==19: # U
        m=palm(X,Y)
        m=np.maximum(m,capsule(X,Y,-.075,.02,-.08,.86,.080)); m=np.maximum(m,capsule(X,Y,.085,.02,.09,.84,.080))
        c=add_line(c,X,Y,.0,.15,.0,.73,.016,.5)
        return m,c
    if k==20: # V
        m=palm(X,Y)
        m=np.maximum(m,capsule(X,Y,-.07,.02,-.31,.86,.078)); m=np.maximum(m,capsule(X,Y,.08,.02,.36,.82,.078))
        return m,c
    if k==21: # W
        m=palm(X,Y)
        for vals in [(-.10,.01,-.39,.78,.073),(0,.03,.01,.91,.075),(.10,.01,.42,.72,.073)]: m=np.maximum(m,capsule(X,Y,*vals))
        return m,c
    if k==22: # X, hooked index finger
        m=fist(X,Y)
        m=np.maximum(m,capsule(X,Y,.10,.12,.10,.66,.082))
        m=np.maximum(m,capsule(X,Y,.10,.66,.35,.51,.080))
        c=add_line(c,X,Y,.10,.54,.28,.48,.020,.7)
        return m,c
    if k==23: # Y, thumb and little finger extended
        m=fist(X,Y)
        m=np.maximum(m,capsule(X,Y,-.20,.12,-.48,.76,.078))
        m=np.maximum(m,capsule(X,Y,.18,-.03,.72,.30,.085))
        return m,c
    raise ValueError(k)


def model(k,X,Y):
    if k==14: # P is K turned downward and diagonally
        U,V=rotate_coords(X,Y,-145); return upright_model(9,U,V)
    if k==15: # Q is G turned downward
        U,V=rotate_coords(X,Y,-88); return upright_model(6,U,V)
    return upright_model(k,X,Y)


def make_filters():
    fs=[]
    q=np.zeros((5,5),np.float64); q[2,2]=1; fs += [q.copy(),-q.copy()]
    yy,xx=np.mgrid[-2:3,-2:3]; g=np.exp(-(xx*xx+yy*yy)/2.2)
    for a in [0,45,90,135]:
        t=math.radians(a); f=(xx*math.cos(t)+yy*math.sin(t))*g; f/=np.sqrt((f*f).sum())
        fs += [f,-f]
    lap=(xx*xx+yy*yy-2.2)*g; lap-=lap.mean(); lap/=np.sqrt((lap*lap).sum()); fs += [lap,-lap]
    return np.asarray(fs,dtype=np.float64)[:,None]


def correlate_same(a,f):
    # a is H,W and f is K,K; this is the scorer's cross-correlation.
    p=f.shape[0]//2; z=np.pad(a,p); out=np.zeros_like(a,dtype=np.float64)
    for j in range(f.shape[0]):
        for i in range(f.shape[1]): out += f[j,i]*z[j:j+a.shape[0],i:i+a.shape[1]]
    return out


def synthetic_features(k,par,filters):
    n=31; axis=(np.arange(n)-(n-1)/2)/10.0; X,Y=np.meshgrid(axis,axis[::-1])
    rot,sx,sy=par
    U,V=rotate_coords(X,Y,rot); U/=sx; V/=sy
    m,c=model(k,U,V)
    raw=-1.0+2.0*m-1.05*c*m
    raw=(raw-raw.mean())/(raw.std()+1e-6)
    maps=[]
    for f in filters[:,0]: maps.append(np.maximum(correlate_same(raw,f),0))
    maps=np.asarray(maps); d=(n-23)//2
    return raw[d:d+23,d:d+23],maps[:,d:d+23,d:d+23]


def conv_layer(name,inp,out,k=1,pad=0):
    return {'name':name,'op':'conv','inputs':[inp],'out':out,'kernel':[k,k],'stride':1,'padding':pad}


def build(dest):
    os.makedirs(dest,exist_ok=True)
    filters=make_filters(); weights={}
    layers=[{'name':'n0','op':'norm','inputs':['image']},conv_layer('feat','n0',len(filters),5,2),{'name':'feat_relu','op':'relu','inputs':['feat']}]
    weights['feat.W']=filters.astype(np.float32); weights['feat.b']=np.zeros(len(filters),np.float32)
    rng=np.random.default_rng(24051991)
    pars=[(0.0,1.0,1.0)]
    for _ in range(15): pars.append((rng.uniform(-6,6),rng.uniform(.91,1.16),rng.uniform(.92,1.13)))
    # Analytic pose variants in two contrast polarities.
    templates=[]; owners=[]
    for k in range(24):
        for par in pars:
            T,F=synthetic_features(k,par,filters)
            T=T-T.mean(); te=float((T*T).sum())+1e-9
            for polarity in (1.0,-1.0):
                w=np.zeros_like(F)
                # Exact signed intensity correlation through positive/negative decomposition.
                w[0]=polarity*T/te; w[1]=-polarity*T/te
                # Oriented edges carry pose and internal-finger evidence.
                E=F[2:] if polarity>0 else np.concatenate([F[3:10:2],F[2:10:2],F[11:12],F[10:11]],axis=0)
                # The explicit permutation above is awkward to align; derive the reversed-polarity maps directly.
                if polarity<0:
                    E=np.empty_like(F[2:]); E[0::2]=F[3::2]; E[1::2]=F[2::2]
                ee=float((E*E).sum())+1e-9
                w[2:]=.72*E/ee
                templates.append(w); owners.append(k)
    W=np.asarray(templates,np.float32)
    C=len(W)
    layers += [conv_layer('match','feat_relu',C,23,0),{'name':'match_relu','op':'relu','inputs':['match']},{'name':'place_max','op':'maxpool','inputs':['match_relu'],'size':[6,6],'stride':[6,6]}]
    weights['match.W']=W; weights['match.b']=np.full(C,-.50,np.float32)
    # Exact binary max tree over 16 templates belonging to each class.
    cur='place_max'; per=len(pars)*2; stage=0
    while per>1:
        outc=24*(per//2); dn=f'diff{stage}'; rn=f'diff_relu{stage}'; bn=f'base{stage}'; mn=f'variant_max{stage}'
        layers += [conv_layer(dn,cur,outc),{'name':rn,'op':'relu','inputs':[dn]},conv_layer(bn,cur,outc),{'name':mn,'op':'add','inputs':[rn,bn]}]
        A=np.zeros((outc,24*per,1,1),np.float32); B=np.zeros_like(A)
        for k in range(24):
            for j in range(per//2):
                o=k*(per//2)+j; a=k*per+2*j; b=a+1
                A[o,a,0,0]=1; A[o,b,0,0]=-1; B[o,b,0,0]=1
        weights[dn+'.W']=A; weights[dn+'.b']=np.zeros(outc,np.float32)
        weights[bn+'.W']=B; weights[bn+'.b']=np.zeros(outc,np.float32)
        cur=mn; per//=2; stage+=1
    layers.append({'name':'scores','op':'dense','inputs':[cur],'out':24})
    weights['scores.W']=np.eye(24,dtype=np.float32)
    weights['scores.b']=np.linspace(0,-2.3e-5,24,dtype=np.float32)
    with open(os.path.join(dest,'network.json'),'w') as f: json.dump({'layers':layers},f,separators=(',',':'))
    np.savez(os.path.join(dest,'weights.npz'),**weights)

if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '.')
