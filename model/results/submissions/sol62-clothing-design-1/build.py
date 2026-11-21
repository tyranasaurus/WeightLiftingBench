import json, sys
from pathlib import Path
import numpy as np

# Coordinates describe idealized garment patterns, not individual products.
rng = np.random.default_rng(127)
def poly(*pts): return ('p', pts)
def box(l,t,r,b): return poly((l,t),(r,t),(r,b),(l,b))
def ellipse(x,y,a,b): return ('e',(x,y,a,b))
def stroke(x,y,u,v,w):
    d=np.hypot(u-x,v-y); a=(v-y)*w/d/2; b=(x-u)*w/d/2
    return poly((x+a,y+b),(u+a,v+b),(u-a,v-b),(x-a,y-b))
def pattern(c):
    u=rng.uniform
    holes=[]
    if c in (0,2,3,4,6):
        h=u(1.27,1.98) if c in (0,6) else (u(1.12,1.65) if c in (2,4) else u(1.80,2.65))
        torso=u(.32,.49) if c!=3 else u(.29,.43)
        shoulder=torso*u(.88,1.08)
        hem=torso*u(.81,1.13) if c!=3 else torso*u(.85,1.73)
        waist=torso*u(.72,.98) if c in (0,3,6) else torso*u(.89,1.05)
        parts=[poly((-shoulder,.045),(-torso,.25),(-waist,h*.58),(-hem,h),(hem,h),(waist,h*.58),(torso,.25),(shoulder,.045),(.14,0),(-.14,0))]
        neck=u(.14,.28) if c in (0,3) else u(.10,.19)
        holes=[ellipse(0,-.012,neck,u(.075,.20) if c in (0,3) else u(.04,.115))]
        short=c==0 or (c in (3,6) and u(0,1)<(.74 if c==3 else .28))
        if short:
            end=u(.18,.36); spread=u(.11,.31); cuff=u(.09,.15)
        else:
            end=h*u(.83,1.02) if c!=3 else u(.45,.74); spread=u(.04,.23); cuff=u(.055,.11)
        if not (c in (0,3) and u(0,1)<(.15 if c==0 else .20)):
            for s in (-1,1):
                parts.append(poly((s*(shoulder-.035),.044),(s*(shoulder+spread*.52),.11),(s*(shoulder+spread+cuff),end),(s*(shoulder+spread-cuff),end+u(-.027,.037)),(s*(torso-.057),.235)))
    elif c==1:
        w=u(.19,.35); crotch=u(.24,.42); foot=w*u(.35,.86); sep=u(.07,.17)
        parts=[poly((-w,0),(w,0),(w*1.07,.23),(sep+foot,1),(sep,1),(0,crotch),(-sep,1),(-sep-foot,1),(-w*1.07,.23))]
    elif c==8:
        w=u(.65,1.15); h=u(.39,.86); handle=u(0,1)<.65
        t=u(.035,.16); b=u(.0,.19)
        parts=[poly((-w/2+t,0),(w/2-t,0),(w/2,h*.23),(w/2-b,h),(-w/2+b,h),(-w/2,h*.27))]
        if handle:
            a=w*u(.19,.37); d=u(.12,.42); thickness=u(.021,.057)
            parts.insert(0,ellipse(0,.018,a,d)); holes=[ellipse(0,.018,a-thickness,d-thickness)]
    else:
        sole=u(.045,.108); toe=u(.075,.19); heel=u(.07,.16)
        if c==7:
            high=u(.20,.39); top=u(.56,.78)
            parts=[poly((0,.87),(.06,.80),(.24,.78),(.43,.71),(top,1-high),(top+.09,1-high*.81),(.88,1-high*.73),(.97,1-high*u(.82,1.12)),(1,.93),(.97,1),(.07,1),(0,.96))]
        elif c==9:
            high=u(.37,.82); shaft=u(.37,.65); taper=u(-.05,.08)
            parts=[poly((0,1-toe*.67),(.08,1-toe),(.30,1-toe*1.04),(shaft-.08,.73),(shaft,1-high),(.94+taper,1-high+u(-.05,.065)),(.95,.73),(1,.93),(.97,1),(.66,1),(.61,1-u(.015,.13)),(.29,1),(.06,1),(0,.97))]
        elif u(0,1)<.48:
            high=u(.12,.49); strap=u(.027,.085)
            parts=[poly((0,.93),(.25,.96),(.43,.91),(.65,1-high*.72),(.82,1-high),(1,1-high),(1,1-high+sole),(.94,1),(.88,1),(.86,1-high+sole),(.67,1-high*.57),(.48,.99),(.1,1),(0,.976)),stroke(.23,.95,.38,.78,strap),stroke(.38,.78,.53,.87,strap)]
            if u(0,1)<.66:
                parts += [stroke(.68,1-high*.69,.77,1-high-u(.09,.24),strap),stroke(.77,1-high-.14,.98,1-high-.18,strap),stroke(.98,1-high-.18,.96,1-high,strap)]
        else:
            parts=[poly((0,1-sole),(.78,1-sole*.85),(1,1-sole*1.4),(.974,1),(.03,1))]
            k=u(.27,.50); ht=u(.085,.23); strap=u(.045,.17)
            parts.extend([stroke(.12,1-sole,k,1-sole-ht,strap),stroke(k,1-sole-ht,.61,1-sole,strap)])
            if u(0,1)<.72: parts += [stroke(.56,1-sole,.81,1-heel-ht,strap*.67),stroke(.81,1-heel-ht,.98,1-sole,strap*.67)]
    return parts,holes

def render(c):
    parts,holes=pattern(c)
    bounds=[]
    for typ,p in parts:
        if typ=='p': bounds.extend(p)
        else:
            x,y,a,b=p; bounds.extend([(x-a,y-b),(x+a,y+b)])
    lo=np.min(bounds,axis=0); hi=np.max(bounds,axis=0)
    scale=rng.uniform(25.8,28.6)/max(hi-lo)
    y,x=np.mgrid[:84,:84]/3+1/6
    flip=rng.choice([-1,1])
    center=(lo+hi)/2
    x=(x-14-rng.normal(0,.52))*flip/scale+center[0]
    if c in (5,7,9):
        # Allow either centered or sole-aligned crops.
        anchor=rng.choice([14+(hi[1]-lo[1])*scale/2,27.7],p=[.65,.35])
        y=(y-anchor-rng.normal(0,.55))/scale+hi[1]
    else:
        y=(y-14-rng.normal(0,.48))/scale+center[1]
    def inside(item):
        typ,p=item
        if typ=='e':
            X,Y,a,b=p; return ((x-X)/a)**2+((y-Y)/b)**2<=1
        m=np.zeros(x.shape,bool)
        for (a,b),(d,e) in zip(p,p[1:]+p[:1]):
            if b!=e: m^=((y>b)!=(y>e)) & (x<(d-a)*(y-b)/(e-b)+a)
        return m
    m=np.logical_or.reduce([inside(p) for p in parts])
    for p in holes:
        hole=inside(p)
        if c==8: hole &= ~inside(parts[-1])
        m &= ~hole
    return m.reshape(28,3,28,3).mean(axis=(1,3))

def build(folder):
    layers=[]; weights={}
    def layer(name,op,inputs,**kw):
        layers.append(dict(name=name,op=op,inputs=inputs,**kw))
    def affine(name,op,inputs,w,b,**kw):
        layer(name,op,inputs,out=len(w),**kw)
        weights[name+'.W']=np.asarray(w,np.float32); weights[name+'.b']=np.asarray(b,np.float32)
    affine('thresholds','conv',['image'],np.ones((3,1,1,1)),[-.035,-.17,-.45],kernel=[1,1],stride=1,padding=0)
    layer('positive','relu',['thresholds'])
    affine('occupancy','conv',['positive'],np.array([.94/.135,.06/.28-.94/.135,-.06/.28]).reshape(1,3,1,1),[0],kernel=[1,1],stride=1,padding=0)
    # A small spatial blur makes near-aligned boundaries less costly.
    affine('blur','conv',['occupancy'],np.outer([.18,.64,.18],[.18,.64,.18]).reshape(1,1,3,3),[0],kernel=[3,3],stride=1,padding=1)
    n=960
    W=[]; B=[]
    for c in range(10):
        for _ in range(n):
            m=render(c).ravel(); area=m.sum()
            # Centered unit-energy silhouette correlation discounts overall
            # brightness and compares sparse and broad shapes on equal terms.
            template=m-area/m.size
            W.append(template/max(np.linalg.norm(template),1))
            B.append(0)
    affine('match','dense',['blur'],W,B)
    layer('by_class','reshape',['match'],shape=[1,10,n])
    layer('best','maxpool',['by_class'],size=[1,n],stride=[1,n])
    layer('scores','reshape',['best'],shape=[10])
    dest=Path(folder); dest.mkdir(parents=True,exist_ok=True)
    (dest/'network.json').write_text(json.dumps(dict(layers=layers),indent=2)+'\n')
    np.savez(dest/'weights.npz',**weights)

if __name__=='__main__': build(sys.argv[1])
