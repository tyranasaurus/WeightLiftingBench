"""Build isolated, individually normalized exemplars from a parametric pen alphabet."""
import sys
from pathlib import Path
import numpy as np

# This coordinate system is a writing box: (0,0) is the upper left and
# (1,1) the lower right. The descriptions are pen motions, not bitmaps.
def alphabet(d, rng):
    paths = []
    p = None
    t = np.linspace(0., 1., 25)[1:, None]
    u = rng.uniform
    r = rng.random
    def M(x,y):
        nonlocal p
        paths.append([np.array([[x,y]])]); p = np.array([x,y])
    def C(x1,y1,x2,y2,x,y):
        nonlocal p
        q = np.array([x,y]); a = 1-t
        seg = a**3*p + 3*a*a*t*np.array([x1,y1]) + 3*a*t*t*np.array([x2,y2]) + t**3*q
        paths[-1].append(seg); p=q
    def L(x,y):
        nonlocal p
        q=np.array([x,y]); paths[-1].append(p+(q-p)*t); p=q
    def oval(cx,cy,rx,ry,begin,end,tilt,angular):
        a=np.linspace(begin,end,113); c=np.cos(a); s=np.sin(a)
        xx=cx+rx*np.sign(c)*abs(c)**angular+tilt*s
        yy=cy+ry*np.sign(s)*abs(s)**angular
        paths.append([np.stack([xx,yy],axis=1)])
    pi=np.pi
    mode=r()
    if d==0:
        rot=u(-2.6,-.45) if r()<.86 else u(-pi,pi)
        gap=u(-.6,.20) if r()<.85 else u(.2,.79)
        oval(.5,.5,.49,.49,rot,rot+2*pi-gap,u(-.12,.12),u(.70,1.20))
        if r()<.17:
            # A second touch of the pen at the closing seam.
            z=paths[-1][-1][-1]; M(*z); L(z[0]+u(-.16,.16),z[1]+u(.02,.13))
    elif d==1:
        # A downstroke, sometimes a lead-in and sometimes feet.
        bend=u(-.12,.12); tip=u(-.08,.08)
        if mode<.57: M(.53+tip,0)
        else:
            M(u(-.05,.39),u(.13,.47)); C(.27,.27,.43,.13,.53+tip,0)
        C(.51+bend,.27,.49+bend,.70,.50,1)
        if mode>.85:
            M(u(-.02,.27),1); C(.4,.99,.64,1,u(.81,1.05),u(.95,1.01))
        elif r()<.16: L(.50+u(-.13,.13),u(.94,1.00))
    elif d==2:
        # Arch, falling diagonal, written baseline. The baseline can turn
        # through a loop when drawn without lifting the pen.
        b=u(.83,1.03); a=u(-.02,.20)
        M(a,u(.12,.39))
        if mode<.11:
            L(u(.44,.67),0); L(u(.85,1.01),u(.04,.18)); L(.08,.93)
        else:
            C(a+u(-.12,.12),u(-.06,.10),.81,-.12,b,u(.12,.27))
            if mode<.49:
                C(b+.14,.42,.34,.76,u(.01,.16),u(.91,1.00))
            elif mode<.86:
                C(b+.06,.49,.22,.99,.02,.94)
                C(-.19,.89,u(-.12,.02),u(.64,.82),.21,u(.77,.91))
            else:
                C(b+.22,.63,.12,.76,-.02,.87)
                C(-.19,.96,.09,1.07,.37,.96)
        C(.27,u(.78,1.03),.72,u(.90,1.10),u(.91,1.14),u(.78,1.03))
    elif d==3:
        h=u(.36,.58); m=u(.17,.58); start=u(-.05,.22)
        M(start,u(.02,.21))
        if mode<.27:
            C(.26,.04,.66,-.01,.90,u(-.01,.07)); L(m,h)
        else:
            C(.28,-.06,.93,-.07,u(.84,1.02),u(.10,.23))
            C(1.04,.31,.68,h-.04,m,h)
        if .46<mode<.79:
            C(m-.29,h+.09,m-.24,h-.12,m+.05,h-.035)
        C(.66,h-.07,1.04,h+.06,u(.88,1.02),u(.65,.82))
        C(.93,1.01,.24,1.13,u(-.10,.17),u(.77,1.00))
    elif d==4:
        h=u(.45,.69); stem=u(.61,.94); left=u(-.07,.12)
        if mode<.27:
            # Triangular 4, sometimes written without taking the pen off.
            M(stem,u(-.03,.08)); L(left,h); L(u(.95,1.22),h+u(-.10,.05))
            M(stem+u(-.08,.06),0); L(stem+u(-.09,.06),1)
        elif mode<.58:
            # Open U form: slow turn on the left, up to the right, then down.
            M(u(.08,.31),u(0,.16)); C(left,.32,left-.08,h+.05,left+.11,h)
            C(.39,h+.06,stem,h-.12,stem+u(-.08,.10),u(-.03,.22))
            C(stem,.32,stem-.02,.71,stem+u(-.07,.10),1)
        else:
            M(u(.05,.47),u(-.03,.14)); C(.10,.28,left,h-.06,left,h)
            C(.26,h+.08,.70,h+.02,u(.84,1.18),h+u(-.13,.03))
            M(stem+u(-.08,.08),u(-.06,.18)); C(stem,.26,stem-.03,.71,stem+u(-.05,.08),1)
    elif d==5:
        left=u(.03,.27); h=u(.36,.54)
        if mode<.71:
            # A separate cap is common, and can over- or under-run the stem.
            M(u(.83,1.17),u(-.07,.11)); C(.69,.07,.39,.00,left+u(-.12,.09),u(-.03,.08))
            M(left+u(-.08,.10),u(-.02,.13)); C(left,.16,left-.07,.35,left-.045,h)
            C(.44,h-.17,.84,h-.20,.95,h+.025)
        else:
            M(u(.71,1.07),u(.01,.14)); C(.47,-.10,left-.16,.03,left-.01,u(.22,.33))
            C(left,.39,.72,.39,.90,.56)
        C(1.16,.81,.72,1.04,.35,.99)
        C(.15,1.00,-.03,.91,u(-.08,.16),u(.76,1.00))
    elif d==6:
        # Descending hook followed by a counterclockwise lower bowl.
        top=u(.48,.93); h=u(.40,.64)
        if mode<.18:
            M(u(.85,1.06),u(.08,.22)); C(.86,-.09,.50,-.04,.29,.18)
        else:
            M(top,u(-.04,.06)); C(top-.12,.05,.38,.18,.25,.28)
        C(-.10,.62,-.07,.99,.42,1.00)
        C(.81,1.09,1.12,.85,.94,h+.14)
        C(.88,h-.06,.54,h-.10,.32,h+.005)
        if mode<.68:
            C(.16,h+.05,.10,h+.13,u(-.025,.15),u(.66,.87))
        elif mode<.86:
            L(u(.19,.32),h+u(.01,.11))
        else:
            C(.20,h+.05,.01,h+.17,u(.00,.22),u(.85,1.01))
    elif d==7:
        # Horizontal cap (possibly waved), then a diagonal downstroke.
        if mode<.24:
            M(u(-.06,.08),u(.13,.31)); L(u(-.03,.09),u(-.04,.08))
        else: M(u(-.06,.14),u(-.02,.12))
        C(.32,u(-.11,.12),.72,u(-.03,.18),u(.87,1.04),u(-.01,.12))
        end=u(.14,.48); bend=u(-.15,.12)
        C(.82+bend,.28,end+bend,.81,end,1.00)
        if .19<mode<.40:
            M(u(.16,.39),u(.45,.58)); L(u(.75,.91),u(.42,.59))
        elif mode>.92:
            L(end+u(-.11,.12),u(.92,.99))
    elif d==8:
        h=u(.36,.62); tr=u(.29,.52); br=u(.35,.55); mid=u(.40,.60)
        if mode<.18:
            oval(.50,h*.50,tr,h*.50,-pi/2,1.51*pi,u(-.07,.07),u(.8,1.2))
            oval(.50,(1+h)*.5,br,(1-h)*.50,-pi/2,1.52*pi,u(-.08,.08),u(.8,1.2))
        else:
            # Start at the top, turn left, cross the waist, curl around the
            # bottom and return to the top from the other side.
            sx=u(.48,.77); M(sx,u(-.02,.07))
            C(.37,-.09,.50-tr-.11,.07,.50-tr,h*.48)
            C(.50-tr,h*.74,mid-.17,h-.07,mid,h)
            C(mid+.22,h+.09,.50+br,h+.13,.50+br,h+(1-h)*.55)
            C(.50+br,1.06,.50-br,1.08,.50-br,h+(1-h)*.56)
            C(.50-br-.025,h+.14,mid-.10,h+.04,mid,h)
            C(mid+.15,h-.09,.50+tr+.04,h*.61,.50+tr,h*.32)
            if mode<.87: C(.50+tr,-.025,sx+.04,-.028,sx+u(-.14,.07),u(-.025,.07))
            else: L(sx+u(.02,.17),u(.015,.12))
    elif d==9:
        h=u(.41,.66); right=u(.78,1.00); end=u(.40,.89)
        if mode<.36:
            # The loop and descending stem can be drawn separately.
            st=u(-1.35,-.32); gap=u(-.25,.75)
            oval(.46,h/2,.46,h/2,st,st+2*pi-gap,u(-.05,.06),u(.77,1.18))
            M(right+u(-.14,.04),u(.0,.22)); C(right,.35,right-.14,.70,end,1)
        else:
            M(right+u(-.15,.06),u(.08,.27)); C(right-.11,-.08,.29,-.09,.095,h*.25)
            C(-.16,h*.55,.02,h+.06,.32,h)
            C(.59,h+.04,right-.05,h*.67,right,u(.025,.23))
            if mode>.87:
                C(right+.05,.52,right+.04,.93,.63,.98)
                C(.45,1.04,.24,1.01,u(.08,.34),u(.83,1.00))
            else:
                C(right+.02,.44,end+.07,.76,end,1.0)
    return [np.concatenate(path) for path in paths]


def draw(d,rng):
    paths=alphabet(d,rng); u=rng.uniform; r=rng.random
    if d==8 and r()<.45:
        paths=[np.column_stack([1-p[:,0],p[:,1]]) for p in paths]
    if d==3 and r()<.17:
        paths=[np.column_stack([p[:,0],1-p[:,1]]) for p in paths]
    # Smooth changes of penmanship including relative lobe size, rotation,
    # bent stroke axis, tilt of the crossbars and narrow/wide bowls.
    aspect=u(.47,1.04) if r()<.88 else u(.32,1.28)
    shear=np.clip(rng.normal(.10,.27),-.70,.85)
    htilt=np.clip(rng.normal(0.,.115),-.31,.31)
    waist=u(-.56,.56); bow=u(-.17,.17); taper=u(-.49,.49); arc=u(-.065,.065)
    ripple=rng.normal(0.,.034,(2,3,3)); stretch=u(-.07,.07)
    # Fixed smooth basis functions; coefficients come directly from the seed.
    chunks=[]; lengths=[]
    for path in paths:
        x=path[:,0]-.5; y=path[:,1]-.5
        g=np.stack([np.ones_like(x),np.sin(np.pi*x),np.cos(2.8*x)],1)
        k=np.stack([np.ones_like(y),np.sin(np.pi*y),np.cos(3.4*y)],1)
        noise=np.einsum('ni,aij,nj->na',g,ripple,k)
        xx=aspect*(x*(1+taper*y) + bow*(1-4*y*y)+stretch*np.sin(2*np.pi*y)+noise[:,0])-shear*y
        yy=y+htilt*x+waist*(.25-y*y)+arc*(1-4*x*x)+noise[:,1]
        xy=np.stack([xx,yy],axis=1)
        chunks.append(xy); lengths.append(len(xy))
    allxy=np.concatenate(chunks); lo=allxy.min(0); hi=allxy.max(0)
    thick=u(.95,1.92) if r()<.80 else u(.63,2.65)
    scale=(20.-2*thick)/max(np.max(hi-lo),.1)
    ax=u(.86,1.15); ay=u(.88,1.12); press=u(.03,.27); phase=u(0,2*np.pi)
    buf=np.zeros((28*28,),np.float32)
    off=np.arange(-4,5); ox=off[None,None,:]; oy=off[None,:,None]
    for xy in chunks:
        pos=(xy-(lo+hi)*.5)*scale+13.5
        s=np.concatenate([[0],np.cumsum(np.sqrt(np.sum(np.diff(pos,axis=0)**2,axis=1)))])
        ss=np.linspace(0,s[-1],max(2,int(s[-1]*2.8)+1))
        x=np.interp(ss,s,pos[:,0])[:,None,None]; y=np.interp(ss,s,pos[:,1])[:,None,None]
        ix=np.floor(x).astype(int)+ox; iy=np.floor(y).astype(int)+oy
        rad=thick*(1+press*np.sin(phase+ss[:,None,None]*.19))
        rad=rad*(1-u(0,.32)*np.exp(-np.minimum(ss,s[-1]-ss)[:,None,None]/1.8))
        dist=np.sqrt(((ix-x)/ax)**2+((iy-y)/ay)**2)
        ink=np.clip(.5+(rad-dist)/1.05,0.,1.)
        idx=np.clip(iy,0,27)*28+np.clip(ix,0,27)
        np.maximum.at(buf,idx.ravel(),ink.ravel().astype(np.float32))
    return buf.reshape(28,28)


FEATURES = '''import numpy as np

pixel = np.arange(28, dtype=np.float32)
xx = pixel[None,None,:]
yy = pixel[None,:,None]

def normalize(a):
    # Partial moment deskew, then normalize position and partially normalize
    # aspect. A straight one should stay narrow, not become a thick column.
    mass = np.maximum(a.sum((1,2)), 1e-6)
    cy = (a*yy).sum((1,2))/mass
    cx = (a*xx).sum((1,2))/mass
    dy = yy-cy[:,None,None]
    dx = xx-cx[:,None,None]
    slope = .81*np.clip((a*dx*dy).sum((1,2))/(1e-5+(a*dy*dy).sum((1,2))), -1.15,1.15)
    pos = xx-slope[:,None,None]*dy
    mask = a > np.maximum(1e-6,.19*a.max((1,2),keepdims=True))
    left = np.where(mask,pos,64).min((1,2)); right = np.where(mask,pos,-64).max((1,2))
    top = np.where(mask,yy,64).min((1,2)); bot = np.where(mask,yy,-64).max((1,2))
    h = np.maximum(bot-top+1,1); w=np.maximum(right-left+1,1)
    sy=22/h
    sx=np.minimum(sy*1.65,np.minimum(24,17.5*(w/(.74*h))**.42)/w)
    y = (yy-13.5)/sy[:,None,None]+((top+bot)*.5)[:,None,None]
    x = (xx-13.5)/sx[:,None,None]+((right+left)*.5)[:,None,None]+slope[:,None,None]*(y-cy[:,None,None])
    i=np.floor(x).astype(np.int32); j=np.floor(y).astype(np.int32); fx=x-i; fy=y-j
    n=np.arange(len(a))[:,None,None]
    def sample(j,i):
        return a[n,j.clip(0,27),i.clip(0,27)]*((i>=0)&(i<28)&(j>=0)&(j<28))
    out=(1-fy)*((1-fx)*sample(j,i)+fx*sample(j,i+1))+fy*((1-fx)*sample(j+1,i)+fx*sample(j+1,i+1))
    return out.astype(np.float32),np.log(np.maximum(w/h,.05))

def gauss(s,n):
    centers=13.5 + (np.arange(n,dtype=np.float32)-(n-1)*.5)*(27./n)
    g=np.exp(-.5*((centers[:,None]-pixel[None,:])/s)**2)
    return (g/g.sum(1,keepdims=True)).astype(np.float32)

finegrid=gauss(1.38,10)
coarsegrid=gauss(2.15,6)
voidgrid=gauss(2.5,5)

def pool(a,g):
    return (g @ a @ g.T)

def smooth(a):
    p=np.pad(a,((0,0),(1,1),(1,1)))
    return (.25*p[:,1:-1,:-2]+.50*p[:,1:-1,1:-1]+.25*p[:,1:-1,2:])*.5 + (.25*p[:,:-2,1:-1]+.50*p[:,1:-1,1:-1]+.25*p[:,2:,1:-1])*.5

def holes(a):
    # Flood exterior paper, not ink. Small isolated pinholes carry little
    # weight. Padding models the unbounded region outside the writing box.
    bg=np.pad(a,((0,0),(1,1),(1,1)),constant_values=False)
    reach=np.zeros_like(bg); reach[:,1,1:-1]=True; reach[:,-2,1:-1]=True
    reach[:,1:-1,1]=True; reach[:,1:-1,-2]=True; reach &= bg
    for step in range(60):
        prev=reach[:,1:-1,1:-1]
        nex=(prev|reach[:,:-2,1:-1]|reach[:,2:,1:-1]|reach[:,1:-1,:-2]|reach[:,1:-1,2:]) & a
        if np.array_equal(prev,nex): break
        reach[:,1:-1,1:-1]=nex
    return a & ~reach[:,1:-1,1:-1]

def unit(a):
    return a/np.maximum(np.sqrt(np.sum(a*a,axis=1,keepdims=True)),1e-6)

def describe(arr):
    a,aspect=normalize(arr)
    a=a/np.maximum(a.max((1,2),keepdims=True),.001)
    b=smooth(smooth(a)); p=np.pad(b,((0,0),(1,1),(1,1)))
    gx=p[:,1:-1,2:]-p[:,1:-1,:-2]; gy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    mag=np.sqrt(gx*gx+gy*gy)
    co=[]; fi=[]
    angle=(np.arctan2(gy,gx)*(4/np.pi))%8
    for k in range(8):
        d=np.abs(angle-k); d=np.minimum(d,8-d)
        vote=mag*(np.maximum(1-d,0)+.22*np.maximum(np.abs(d-2)-1,0))
        co.append(pool(vote,coarsegrid)); fi.append(pool(vote,finegrid))
    def hog(ch):
        z=np.stack(ch,axis=-1)
        # Local normalization tempers differences in width, ink, and contrast.
        return np.sqrt(z/(.14+z.sum(-1,keepdims=True)))
    cg=hog(co); fg=hog(fi)
    # Visibility from the four page edges: a notch open to the left is
    # different from a hole and from a notch open to the right.
    near=a>.30
    l=np.maximum.accumulate(near,axis=2); r=np.maximum.accumulate(near[:,:,::-1],axis=2)[:,:,::-1]
    t=np.maximum.accumulate(near,axis=1); btm=np.maximum.accumulate(near[:,::-1],axis=1)[:,::-1]
    paper=1-a
    h1=holes(a<.45).astype(np.float32); h2=holes(a<.17).astype(np.float32)
    hole=.55*h1+.45*h2
    hollow=pool(hole,voidgrid)
    notches=[]
    for v in ((l&r),(t&btm),(~l&r&t&btm),(l&~r&t&btm),(l&r&~t&btm),(l&r&t&~btm)):
        notches.append(pool(paper*v,voidgrid))
    cavity=np.stack(notches,-1)
    n=len(a)
    cv=cavity.reshape(n,-1); hv=hollow.reshape(n,-1)
    # Weights are fixed from the semantics of each channel. Everything is
    # normalized within one drawing, never across drawings.
    main=unit(cg.reshape(n,-1)); area=unit(pool(a,coarsegrid).reshape(n,-1))
    extra=cv/np.sqrt(7.+(cv*cv).sum(1,keepdims=True))
    hcode=np.sqrt(np.maximum(hv,0)); hcode=hcode/np.sqrt(2.5+(hcode*hcode).sum(1,keepdims=True))
    aspectcode=np.exp(-.5*((aspect[:,None]-np.linspace(-2.5,.65,10)[None,:])/.37)**2).astype(np.float32)
    desc=np.concatenate([main,.47*area,.62*extra,.48*hcode,.22*unit(aspectcode)],axis=1)
    # Euclidean distance includes the lengths of the notch and hole codes.
    coarse=np.rint(desc*1450).astype(np.int16)
    fm=pool(a,finegrid); fh=pool(hole,finegrid)
    fine=np.concatenate([fg,.72*np.sqrt(fm[:,:,:,None]),.56*np.sqrt(fh[:,:,:,None])],axis=3)
    fine=np.rint(np.clip(fine,0,1)*200).astype(np.uint8)
    return coarse,fine
'''

PREDICTOR = '''
# All rows in the alphabet remain separate; there is no pooled class model.
coarseBank=CONSTANTS['coarse'].astype(np.float32)
coarseLen=(coarseBank*coarseBank).sum(1)
fineBank=CONSTANTS['fine']
perDigit=len(fineBank)//10

def localcost(q,bank):
    # Local elastic matching: 25 overlapping-context cells grouped into a
    # 5-by-5 mesh. Compare 2-by-2 patches at each neighboring displacement.
    # A second, coarser mesh encourages nearby patches to move together.
    m=len(bank)
    b=np.pad(bank.astype(np.float32)/200,((0,0),(1,1),(1,1),(0,0)))
    q=q.astype(np.float32)/200
    ds=[]
    for y in range(3):
        for x in range(3):
            dif=b[:,y:y+10,x:x+10,:]-q
            e=np.sum(dif*dif,axis=3).reshape(m,5,2,5,2).mean((2,4))
            ds.append(e)
    dist=np.stack(ds,-1)
    local=dist[:,:,:,4]
    # Penalize displacements but don't penalize zero displacement. The cost
    # is expressed in units of squared normalized patch-feature differences.
    pen=np.array([.072,.045,.072,.045,0,.045,.072,.045,.072],np.float32)
    best=(dist+pen).min(3)
    pad=np.pad(dist,((0,0),(1,1),(1,1),(0,0)),mode='edge')
    neighbor=(2*dist+pad[:,:-2,1:-1]+pad[:,2:,1:-1]+pad[:,1:-1,:-2]+pad[:,1:-1,2:])/6
    group=(neighbor+pen*.76).min(3)
    return (.43*best+.42*group+.15*local).mean((1,2))

def predict(x):
    out=np.empty(len(x),np.int32)
    for start in range(0,len(x),64):
        stop=min(len(x),start+64); n=stop-start
        c,f=describe(x[start:stop,0]); c=c.astype(np.float32)
        # Integer descriptors ensure all inner products are exact in float32.
        dots=c @ coarseBank.T
        raw=(coarseLen[None,:]+(c*c).sum(1,keepdims=True)-2*dots)/(1450*1450)
        raw=raw.reshape(n,10,perDigit)
        for i in range(n):
            # Rerank a small, separate shortlist from every class.
            ix=np.argpartition(raw[i],8,axis=1)[:,:8]
            ci=np.arange(10)[:,None]
            cd=raw[i][ci,ix]; ids=(ci*perDigit+ix).ravel()
            ld=localcost(f[i],fineBank[ids]).reshape(10,-1)
            dist=.50*cd+1.6*ld
            # A little corroboration from other independent pen renditions.
            first=np.sort(dist,axis=1)
            out[start+i]=np.argmin(.72*first[:,0]+.19*first[:,1]+.09*first[:,2])
    return out
'''


def build(folder):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'solution.py').write_text(FEATURES+PREDICTOR)
    # Execute our own arithmetic feature definitions; no input files.
    ns={}; exec(FEATURES,ns)
    rng=np.random.default_rng(913571)
    count=4608
    coarse=[]; fine=[]
    for digit in range(10):
        for start in range(0,count,64):
            a=np.stack([draw(digit,rng) for j in range(64)])
            c,f=ns['describe'](a); coarse.append(c); fine.append(f)
        print('finished pen models for',digit,flush=True)
    np.savez(folder/'constants.npz',coarse=np.concatenate(coarse),fine=np.concatenate(fine))

if __name__=='__main__':
    build(Path(sys.argv[1]))
