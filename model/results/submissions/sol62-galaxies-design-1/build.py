import sys, json
from pathlib import Path
import numpy as np


def build(folder):
    layers, weights = [], {}
    def op(kind, inputs, **kw):
        name = 'l'+str(len(layers))
        layers.append(dict(name=name, op=kind, inputs=inputs, **kw))
        return name
    def linear(kind, src, w, b=0, **kw):
        w = np.asarray(w, np.float32)
        name = op(kind,[src],out=len(w),**kw)
        weights[name+'.W'] = w
        weights[name+'.b'] = np.broadcast_to(b,(len(w),)).astype(np.float32).copy()
        return name
    def dense(src,w,b=0): return linear('dense',src,w,b)
    def rel(src): return op('relu',[src])
    def add(*src): return op('add',list(src)) if len(src)>1 else src[0]
    def conv(src,w,b=0,s=1):
        k=np.shape(w)[-1]
        return linear('conv',src,w,b,kernel=[k,k],stride=s,padding=0)

    # Box antialiasing, then a bank of piecewise-linear photometric maps.
    rgb=op('avgpool',['image'],size=[2,2],stride=[2,2])
    light=np.array([.30,.48,.22]); chrom=np.array([1,0,-1.])
    phot=rel(conv(rgb,np.array([light]*5+[light+chrom*.45,light-chrom*.45,light+chrom*.45,light-chrom*.45])[:,:,None,None],
                   -np.array([.125,.205,.145,.355,.51,.15,.15,.28,.28])))
    # Two brightness bands suppress sky and prevent saturated nuclei from
    # determining the apparent shape of the entire object.
    low=np.array([1,-1,0,0,0,0,0,0,0.]); hi=np.array([0,0,1,-1,0,0,0,0,0.])
    bright=np.eye(9)[4]; red=np.array([0,0,0,0,0,1,-1,-1,1.])
    maps=conv(phot,np.stack([low,hi,bright,red])[:,:,None,None])
    y,x=np.mgrid[:64,:64]*2-63
    r=np.hypot(x,y); t=np.arctan2(y,x)
    g=lambda s: np.exp(-.5*(r/s)**2)
    masks={}
    def measure(name,mask,channel):
        a=np.zeros((4,64,64),np.float32); a[channel]=mask; masks[name]=a.ravel()
    for s in (4,7,11,17,25):
        a=g(s)
        measure('f'+str(s),a/(2*np.pi*s*s/4),0)
        measure('h'+str(s),a,1)
        if s in (7,11,17):
            measure('b'+str(s),a,2); measure('c'+str(s),a,3)
            for m in (1,2,3,4):
                rad=a*r**2/(r**2+(.45*s)**2)
                measure(f'd{s}',rad,1)
                for j in range(8):
                    measure(f'm{s}_{m}_{j}',rad*np.cos(m*t-j*np.pi/4),1)
    # Compensated winding filters: subtract the weighted radial mean of
    # the complex phase so a constant-angle oval has little response.
    for s in (9,16):
        a=g(s)*(1-g(s*.42))
        measure(f'wd{s}',a,1)
        for pitch in (18,34):
            for hand in (-1,1):
                phase=np.exp(-2j*hand*np.log(np.maximum(r,1)/s)/np.tan(np.deg2rad(pitch)))
                phase-=np.sum(a*phase)/np.sum(a)
                for j in range(8):
                    measure(f'w{s}_{pitch}_{hand}_{j}',a*np.real(phase*np.exp(1j*(2*t-j*np.pi/4))),1)
    keys=list(masks)
    feats=dense(maps,np.stack(list(masks.values())))
    # Scalar ReLU arithmetic, with projection and addition instead of concatenation.
    def lin(terms,b=0):
        pieces=[dense(n,[[a]],b if i==0 else 0) for i,(n,a) in enumerate(terms) if a or i==0]
        return add(*pieces)
    def sc(n,a=1,b=0): return lin([(n,a)],b)
    def sub(a,b): return lin([(a,1),(b,-1)])
    def maximum(a,b): return add(b,rel(sub(a,b)))
    def minimum(a,b): return sub(a,rel(sub(a,b)))
    def mx(vals):
        while len(vals)>1:
            vals=[maximum(*vals[i:i+2]) if i+1<len(vals) else vals[i] for i in range(0,len(vals),2)]
        return vals[0]
    def ramp(a,lo,hi):
        v=sc(a,1/(hi-lo),-lo/(hi-lo))
        return sub(rel(v),rel(sc(v,1,-1)))
    def inv(a): return sc(a,-1,1)
    def ratio(a,b,lim=1,steps=20,soft=.13):
        # Sum soft comparisons a > t*b. Units here are integrated light.
        t=(np.arange(steps)+.5)*lim/steps
        va=dense(a,np.ones((steps,1))/soft)
        vb=dense(b,-t[:,None]/soft,.5)
        v=add(va,vb)
        return add(dense(rel(v),np.full((1,steps),lim/steps)),
                   dense(rel(dense(v,np.eye(steps),-1)),np.full((1,steps),-lim/steps)))
    f={k:dense(feats,np.eye(1,len(keys),i)) for i,k in enumerate(keys)}
    def harmonic(s,m): return ratio(mx([f[f'm{s}_{m}_{j}'] for j in range(8)]),f[f'd{s}'])
    q7,q11,q17=[harmonic(s,2) for s in (7,11,17)]
    q=lin([(q7,.28),(q11,.52),(q17,.20)])
    odd=lin([(harmonic(11,1),.53),(harmonic(11,3),.30),(harmonic(17,3),.17)])
    outer=ratio(f['h17'],f['h7'],lim=5,steps=40)
    core=ratio(f['h7'],f['h17'])
    peak=ratio(f['b7'],f['h7'],lim=2,steps=40)
    redness=ratio(add(f['c11'],sc(f['h11'],.5)),f['h11'],lim=1.5,steps=40)
    outerred=ratio(add(sub(f['c17'],f['c7']),sc(sub(f['h17'],f['h7']),.5)),sub(f['h17'],f['h7']),lim=1.5,steps=40)
    color=lin([(redness,.35),(outerred,.65)])
    def winding(p):
        return mx([ratio(mx([f[f'w{s}_{p}_{h}_{j}'] for h in (-1,1) for j in range(8)]),f[f'wd{s}']) for s in (9,16)])
    t18,t34=winding(18),winding(34)
    ext=ramp(outer,1.3,2.5)
    blue=inv(ramp(color,.43,.63))
    tex=ramp(lin([(t18,.45),(t34,.55)]),.055,.20)
    disk=lin([(ext,.48),(blue,.31),(tex,.21)])
    edge=minimum(ramp(q,.38,.65),ramp(outer,1.30,1.85))
    bulge=lin([(ramp(peak,.40,1.20),.47),(ramp(redness,.47,.75),.35),(ramp(core,.33,.65),.18)])
    bar=lin([(ramp(q7,.11,.38),.56),(ramp(sub(q7,q17),-.025,.14),.29),(ramp(peak,.25,1),.15)])
    loose=lin([(ramp(sub(t34,t18),-.04,.095),.34),(blue,.34),(inv(ramp(core,.32,.66)),.32)])
    asym=ramp(odd,.085,.27)
    # Decision surfaces: competing morphology scores with continuous margins.
    smooth=sc(disk,-1.45,1.04)
    spiral=sc(disk,1.45,-.51)
    rounded=inv(ramp(q,.055,.22))
    mid=minimum(ramp(q,.055,.17),inv(ramp(q,.26,.49)))
    cigar=ramp(q,.24,.49)
    scores=[lin([(asym,.83),(blue,.29),(ext,.22),(edge,-.36)],-.45),
            lin([(asym,1.30),(blue,-.20),(edge,-.23),(ext,.09)],-.51),
            lin([(smooth,1),(rounded,.47),(asym,-.47)],.025),
            lin([(smooth,1),(mid,.46),(asym,-.47)],.015),
            lin([(smooth,1),(cigar,.49),(asym,-.42),(edge,-.38)],-.10),
            lin([(spiral,1),(bar,.70),(edge,-.65),(asym,-.25)],-.01),
            lin([(spiral,1),(loose,-.64),(edge,-.65),(asym,-.24)],.47),
            lin([(spiral,1),(loose,.67),(edge,-.65),(asym,-.20)],-.02),
            lin([(edge,1.65),(bulge,-.62),(asym,-.27)],-.13),
            lin([(edge,1.65),(bulge,.66),(asym,-.28)],-.73)]
    add(*(dense(s,np.eye(10)[:,i:i+1]) for i,s in enumerate(scores)))
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    (folder/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':')))
    np.savez(folder/'weights.npz',**weights)


if __name__=='__main__':
    build(sys.argv[1])
