"""Build individual analytic silhouettes and a fixed shape-and-appearance classifier.

No input files.  All distances in the pattern descriptions are fractions of a
pattern's own height or width.  Each design is independently rendered and encoded.
"""
import math
import sys
from pathlib import Path
import numpy as np

DEST = Path(sys.argv[1])
DEST.mkdir(parents=True, exist_ok=True)
SIDE = 56
axis = (np.arange(SIDE) + .5) / SIDE
X, Y = np.meshgrid(axis, axis)
bank = [[] for k in range(10)]
priors = [[] for k in range(10)]

def polygon(x, y, vertices):
    inside = np.zeros(x.shape, bool)
    u, v = vertices[-1]
    for s, t in vertices:
        if t != v:
            inside ^= ((y >= min(v, t)) & (y < max(v, t)) & (x < u + (s-u)*(y-v)/(t-v)))
        u, v = s, t
    return inside

def stroke(x,y,xp,yp,r):
    d = np.full(x.shape, 1e3)
    for a,b,c,e in zip(xp,yp,xp[1:],yp[1:]):
        dx,dy = c-a,e-b
        u = np.clip(((x-a)*dx+(y-b)*dy)/(dx*dx+dy*dy+1e-9),0,1)
        d = np.minimum(d,(x-a-u*dx)**2+(y-b-u*dy)**2)
    return d < r*r

def plane(aspect):
    return (X-.5)*max(1.,1./aspect)+.5, (Y-.5)*max(1.,aspect)+.5

def encode(mask):
    m = mask.reshape(28,2,28,2).mean(axis=(1,3)).astype(np.float32)
    # Unit Pearson feature, centering only this rendering, not the collection.
    def unit(x):
        x = x.ravel() - x.mean()
        return x / max(.5, np.sqrt((x*x).sum()))
    p = np.pad(m,1)
    sm = (4*m+2*(p[1:-1,:-2]+p[1:-1,2:]+p[:-2,1:-1]+p[2:,1:-1]) + p[:-2,:-2]+p[:-2,2:]+p[2:,:-2]+p[2:,2:])/16
    cells = sm.reshape(14,2,14,2).mean(axis=(1,3))
    projections = np.concatenate([m.mean(axis=0), m.mean(axis=1)])
    return np.concatenate([.78*unit(cells), .22*unit(projections)]).astype(np.float32)

def add(k, mask, p=0.):
    bank[k].append(encode(recenter(mask)))
    priors[k].append(p)

# Cut a bodice and stitch on two tapered sleeves. Widths refer to the complete
# garment including sleeves, heights to shoulder-to-hem length. A hidden/flattish
# neckline models a visible back panel behind a front collar.
def upper(k, aspect, shoulder, chest, waist, hem, sleeve, neck, reach, cuff, end=.995, curve=.02, tip=.98, gap=0., prior=0., rounded=False):
    x,y=plane(aspect)
    a=abs(x-.5)*2
    width=np.interp(y,[0,.065,.15,.28,.52,.82,end],[neck+.045,shoulder*.91,shoulder,chest,waist,hem*.98,hem])
    if neck>0:
        dip=.032+(.09 if rounded else .0)
        top=dip*np.maximum(0,1-(a/(neck+.0001))**2)**.65
    else:
        top=0.
    bottom=end-curve*np.minimum(1.,(a/(hem+.001))**2)
    body=(a<=width)&(y>=top)&(y<=bottom)&(a>gap*np.maximum(0,(y-.25)/.75))
    # raglan/set-in shoulder, upper arm, and outer and inner cuff
    if sleeve>0:
        outer=polygon(a,y,[(max(.1,shoulder-.19),.052),(shoulder,.105),
            (shoulder+.35*(reach-shoulder),.14+.19*(sleeve-.14)),
            (reach,sleeve-.045),(tip,sleeve),(max(0.,tip-cuff),sleeve+.006),
            (max(0.,tip-cuff-.01),sleeve-.046),(chest-.055,.25),
            (chest-.11,.17)])
        body |= outer
    add(k,body,prior)

def make_uppers():
    # Short sleeves: dropped shoulders, close and loose fits, and cap sleeves.
    for ar in [.63,.76,.89,1.02,1.18]:
        for torso in [.51,.63,.74,.84]:
            for sleeve in [.23,.32,.41,.49]:
                for shape in [0,1]:
                    upper(0,ar,.57 if torso<.7 else .66,torso,torso*(.89 if shape else .99),torso*(.96 if shape else 1.03),sleeve,.24,.995,.24,curve=.014 if not shape else .05,tip=.91,rounded=bool(shape),prior=.005 if sleeve>.45 else 0)
    for ar in [.47,.57,.69,.80]:
        for neck in [.20,.35,.53]:
            for waist in [.62,.81,.92]:
                for end in [.82,1.]:
                    x,y=plane(ar)
                    a=abs(x-.5)*2
                    w=np.interp(y,[0,.07,.23,.32,.58,.94,1],[.70,.72,.79,.94,waist,end,end])
                    top=(.22 if neck>.3 else .12)*np.maximum(0,1-(a/neck)**2)
                    add(0,(a<w)&(y>=top)&(y<=1-.025*a**2),.005)
    # Long-sleeved garments. A knitted ribbed waistband typically ends level
    # with or above the cuffs; an overcoat extends below its sleeves. A blouse
    # has a less regular shirttail and a wider allowance of sleeve lengths.
    for k,ars,lens,ends in [
        (2,[.69,.80,.91,1.04,1.17],[.86,.95,.99],[.88,.965,1.]),
        (4,[.54,.64,.75,.86,.97],[.70,.81,.91,.98],[.97,1.]),
        (6,[.58,.70,.82,.94,1.06],[.61,.78,.93,.99],[.94,1.])]:
        for ar in ars:
            for sl in lens:
                for end in ends:
                    if max(end,sl)<.96: continue
                    for chest in [.57,.69,.77]:
                        for drape in range(3):
                            # sometimes sleeves touch the sides, sometimes separated by air
                            reach=[.99,.98,.91][drape]
                            tip=[.98,.88,.79][drape]
                            cuff=[.20,.23,.25][drape]
                            for style in range(2):
                                if k==2:
                                    hem=chest*[.90,1.04][style]
                                    waist=chest*[1.04,1.07][style]
                                    curve=[.012,.032][style]
                                elif k==4:
                                    hem=chest*[1.14,1.03][style]
                                    waist=chest*[.96,1.02][style]
                                    curve=.023
                                else:
                                    waist=chest*[.93,1.00][style]
                                    hem=chest*[1.06,1.02][style]
                                    curve=[.095,.048][style]
                                upper(k,ar,.62 if chest<.7 else .72,chest,waist,hem,sl,.21,reach,cuff,end,curve,tip,prior=0.)
    # Sleeves frequently hang almost vertically from a rounded shoulder, as on
    # an invisible form. The silhouette of a triangular, spread-out sleeve is
    # not sufficient for these poses. A few looser/batwing cuts are also here.
    for k in [2,4,6]:
        for ar in ([[.74,.86,.99,1.14],[.59,.71,.83,.94],[.64,.76,.88,1.]][[2,4,6].index(k)]):
            x,y=plane(ar); a=abs(2*x-1)
            for cw in [.53,.65,.75]:
                for length in ([.86,.96,.995] if k==2 else ([.65,.78,.90,.98] if k==4 else [.54,.75,.93,.99])):
                    for pose in range(4):
                        for cut in range(2):
                            end=([.89,.975] if k==2 and length>.95 else [.97,1.])[cut]
                            hem=cw*([.91,1.02] if k==2 else ([1.17,1.04] if k==4 else [1.04,1.12]))[cut]
                            curve=.014 if k==2 else (.029 if k==4 else .078)
                            sidebody=np.interp(y,[0,.07,.2,.35,.54,.83,1],[.3,.68,.74,cw,cw*(.91 if k==6 else 1),hem,hem])
                            body=(a<sidebody)&(y>=.037*np.maximum(0,1-(a/.24)**2))&(y<end-curve*np.minimum(1,a/hem)**2)
                            reach=[.98,.92,.98,.99][pose]; cuff=[.95,.83,.72,.86][pose]; thick=[.22,.24,.27,.28][pose]
                            armpit=[.23,.28,.35,.48][pose]
                            sleeve=polygon(a,y,[(.44,.035),(.65,.054),(.76,.085),(.82,.14),
                                (reach,.45*length+.085),(min(1,reach+.012),.68*length+.035),
                                (cuff+.025,length-.065),(cuff,length),(cuff-thick,length-.01),
                                (cuff-thick*.98,length-.083),(cw-.012,armpit),(.51,.19)])
                            add(k,body|sleeve)
    # Button-down short sleeve shirts, open jacket panels and longer cardigans.
    for ar in [.68,.82,.96]:
        for ch in [.54,.67,.76]:
            for sl in [.31,.42,.52]:
                upper(6,ar,.60,ch,ch*.96,ch*1.04,sl,.15,.99,.23,curve=.08,tip=.91,prior=.012)
    for ar in [.58,.72,.87]:
        for sl in [.74,.87,.98]:
            for gap in [.07,.22,.36]:
                upper(4,ar,.63,.68,.69,.76,sl,.23,.99,.22,gap=gap,curve=.015)

def make_pants():
    for ar in [.27,.34,.41,.49,.58,.70]:
        x,y=plane(ar)
        for crotch in [.24,.33,.43,.54]:
            for cut in range(5):
                for spread in [.04,.16,.29]:
                    a=abs(2*x-1)
                    outer=np.interp(y,[0,.12,.24,.57,.92,1],
                        [[.84,1,.99,.85,.72,.72], [.91,1,.99,.95,.89,.90],
                         [.84,.96,1,.90,.97,.99],[.82,1,.96,.77,.56,.55],
                         [.84,.97,1,.81,.84,.92]][cut])
                    inside=np.interp(y,[0,crotch,crotch+.06,.69,.98,1],[0,0,.035,.62*spread,spread,spread])
                    add(1,(a<outer)&(a>=inside)&(y>=.012*(1-a))&(y<=1-.014*a),.012 if ar>.62 else 0.)
    # A relaxed stance, or both legs photographed on a slight slant.
    for ar in [.35,.44,.56]:
        x,y=plane(ar)
        for skew in [-.12,.10]:
            for flare in [.55,.76,1.]:
                a=abs((x-.5-skew*(y-.45))*2)
                w=np.interp(y,[0,.15,.30,.6,1],[.72,.9,.90,.75,flare])
                g=np.interp(y,[0,.33,.48,1],[0,0,.04,.15])
                add(1,(a<w)&(a>=g)&(y>=0)&(y<=1-.02*a))

def make_dress():
    for ar in [.31,.41,.51,.62,.75,.88]:
        x,y=plane(ar)
        a=abs(2*x-1)
        for bod in [.44,.59,.76,.92]:
            for waist in [.70,.87,1.]:
                for hem in [.72,.98]:
                    for sleeve in [0,.15,.28,.58,.80]:
                        sh=bod*.88
                        w=np.interp(y,[0,.04,.105,.21,.38,.48,.72,.98,1],
                            [sh*.77,sh,sh*.96,bod,bod*waist,bod*waist+.03,(bod*waist+hem)*.5+.03,hem,hem])
                        neck=.16 if sleeve==0 else .13
                        top=(.115 if sleeve==0 else .039)*np.maximum(0,1-(a/(sh*.61))**2)
                        body=(a<w)&(y>=top)&(y<=1-.035*a*a)
                        if sleeve:
                            reach=min(.995,bod+(.24 if sleeve<.3 else .32))
                            body|=polygon(a,y,[(sh*.62,.024),(sh+.03,.045),(reach,sleeve-.025),(reach-.04,sleeve),(reach-bod*.24,sleeve+.009),(bod-.05,.176)])
                        add(3,body,.008 if sleeve>.5 and ar>.7 else 0.)
    # Dress with narrow straps and full skirt, and a sleeveless shift dress.
    for ar in [.36,.48,.60,.75]:
        x,y=plane(ar); a=abs(x*2-1)
        for strap in [.42,.61,.80]:
            for waist in [.37,.58,.77]:
                w=np.interp(y,[0,.1,.18,.32,.43,.68,.98,1],[strap,strap*.96,strap+.10,waist,waist,waist*.5+.48,.98,1])
                body=(a<w)&(y>=.145*np.maximum(0,1-(a/(strap*.79))**6))&(y<=1-.035*a*a)
                add(3,body)

# Totes, satchels, clutches, shopping bags, backpacks and flap bags. Rails of
# the handles are thin, with an actual opening above the body.
def make_bags():
    for ar in [.61,.77,.93,1.10,1.32,1.60,1.95,2.35]:
        x,y=plane(ar)
        for top in [0,.12,.25,.39,.53,.66]:
            for contour in range(5):
                a=abs(2*x-1)
                v=np.clip((y-top)/(1-top),0,1)
                w=np.interp(v,[0,.07,.24,.7,.91,1],
                        [[.83,.97,1,1,.99,.88],[.97,1,.97,.85,.81,.74],
                         [.73,.84,.91,1,.97,.86],[.41,.69,.91,.99,.82,.45],
                         [.66,.80,.92,1,.99,.92]][contour])
                body=(y>=top)&(y<=1)&(a<w)
                if top==0:
                    add(8,body,.009)
                    continue
                for handle in [.42,.65,.85]:
                    arch=top*(1-np.sqrt(np.maximum(0,1-(a/handle)**2)))**.65
                    # ellipse arcs as tubular rings, unioned to the bag body
                    radius=np.sqrt((a/handle)**2+((y-top)/top)**2)
                    hoop=(radius<1)&(radius>1-min(.31,.085/max(top,.24)))&(y<=top)
                    add(8,body|hoop,.004 if handle>.8 else 0.)
                if contour in [0,2,4]:
                    # diagonal shoulder strap
                    strap=stroke(x,y,[.05,.18,.42,.67,.83,.96],[top+.12,top*.71,.10,.018,top*.29,top+.1],.026)
                    add(8,body|strap,.008)
    for ar in [.70,.85,1.05,1.35,1.7]:
        x,y=plane(ar)
        for dep in [.19,.32,.48,.59]:
            a=abs(2*x-1)
            w=np.interp(y,[0,.16,.32,.62,.84,1],[.67,.86,.98,1,.96,.78])
            bag=(a<w)&(y<=1-.03*a)&(y>=0)
            hole=(a<.72)&(y<dep*np.sqrt(np.maximum(0,1-(a/.75)**2)))
            add(8,bag&~hole,.005)

# Side views of footwear, with toes pointing left (also reflected at inference).
# Upper and outsole profiles are continuous broken lines. Shoe absolute aspect
# ratio, instep height, footwear shaft width, and slope are independent knobs.
shoeaxis=[0,.035,.10,.22,.36,.49,.60,.70,.80,.94,1.]
def make_shoes():
    for ar in [1.40,1.65,1.90,2.20,2.55,3.,3.6]:
        x,y=plane(ar)
        for style in range(5):
            top0=[[.83,.71,.63,.59,.47,.26,.06,.17,.37,.27,.34],
                  [.83,.68,.60,.57,.45,.29,.15,.21,.32,.09,.18],
                  [.80,.62,.58,.53,.41,.27,.18,.24,.32,.25,.36],
                  [.78,.65,.56,.51,.43,.44,.54,.62,.46,.22,.28],
                  [.82,.66,.59,.54,.47,.30,.08,.06,.13,.04,.10]][style]
            for rise in [-.10,0,.12]:
                top=np.clip(np.array(top0)+rise*np.array([0,0,0,.3,.6,1,1,.6,.2,0,0]),0,1)
                for sole in [0,.09,.20]:
                    bottom=np.interp(x,[0,.06,.18,.48,.64,.83,.96,1],[.82,.94,.99,1,.996-sole*.38,.97-sole,.95-sole,.86-sole])
                    # Normalize independently to the full extent, as if cropped before resizing.
                    z=y*(1-top.min())+top.min()
                    m=(x>=0)&(x<=1)&(z>=np.interp(x,shoeaxis,top))&(z<=bottom)
                    add(7,m,.01 if ar<1.55 and style!=4 else 0.)
    for ar in [.79,.96,1.12,1.31,1.55,1.8,2.07,2.38]:
        x,y=plane(ar)
        for start in [.38,.47,.56,.65]:
            for shaft in [.12,.24,.38]:
                for heel in [0.,.14,.32,.48]:
                    for lean in [-.055,.04]:
                        hi=[(0,.84),(.025,.76),(.11,.71+heel*.20),(.24,.68+heel*.16),
                            (start-.13,.57),(start-.04,.47),(start+.025,.34),
                            (start+lean,.045),(start+.035+lean,.008),(1-shaft*.12,.035),
                            (.97,.19),(.95-shaft*.12,.42),(.98,.57),(.997,.70),
                            (.98, .985-heel*.10),(.89 if heel>.2 else .79,.99-heel*.10),
                            (.84 if heel>.2 else .79,.98-heel),(.72,.96-heel),
                            (.52,.965-heel*.40),(.35,.99),(.16,.995),(.035,.95)]
                        add(9,polygon(x,y,hi),.013 if ar>2.2 else 0.)
    for ar in [1.18,1.45,1.75,2.1,2.55,3.10,3.9,4.75]:
        x,y=plane(ar)
        for heel in [0,.20,.39,.57]:
            for straps in range(7):
                for ankle in [.17,.36,.55]:
                    floor=np.interp(x,[0,.12,.38,.60,.78,1],[.87,.90,.90,.84-heel*.47,.81-heel,.77-heel])
                    bottom=floor+.10+.025*np.sin(np.pi*x)
                    m=(x>=.008)&(x<=.985)&(y>=floor)&(y<bottom)
                    if heel:
                        m|=polygon(x,y,[(.83,.84-heel),(.98,.80-heel),(.96,.978),(.89 if heel>.24 else .79,.98)])
                    h=.98 if straps==0 else .83
                    if straps in [0,1,3,5]:
                        toe=polygon(x,y,[(.036,.88),(.09,.64),(.19,.57),(.31,.61),(.35,.69),(.35,.94),(.23,.94),(.21,.73),(.16,.67),(.12,.88)])
                    else:
                        toe=polygon(x,y,[(.10,.91),(.16,.51),(.28,.47),(.42,.80),(.43,.91),(.30,.92),(.23,.66),(.23,.91)])
                    m|=toe
                    if straps in [0,2,4,5,6]:
                        m |= stroke(x,y,[.32,.43,.61,.72,.91],[.88,.58,.37,ankle,ankle*.78],.049)
                    if straps in [1,3,4,6]:
                        m |= polygon(x,y,[(.40,.89),(.45,.53),(.59,.27),(.68,.28),(.61,.58),(.55,.84)])
                    if straps not in [1,2]:
                        m |= polygon(x,y,[(.69,ankle+.01),(.80,ankle-.07),(.945,ankle-.11),(.985,ankle-.04),(.97,.80-heel),(.91,.84-heel),(.895,ankle+.06),(.71,ankle+.14)])
                    # Crop vertically because not all sandals have ankle straps.
                    add(5,m,.006)
    # Pumps, slingbacks and wedges; slide sandals and flip-flops.
    for ar in [1.32,1.64,2.,2.4,3.,3.7,4.6]:
        x,y=plane(ar)
        for heel in [.09,.29,.51,.66]:
            for style in range(4):
                top=[(0,.77),(.06,.62),(.20,.57),(.35,.54),(.42,.69),(.52,.69-heel*.31),(.65,.57-heel*.44),(.82,.36-heel*.3),(.91,.14),(.98,.12),(.98,.35)]
                if style==1:
                    top[5:9]=[(.53,.74-heel*.32),(.65,.68-heel*.54),(.82,.44-heel*.42),(.93,.19)]
                if style==2:
                    top[:5]=[(0,.88),(.07,.87),(.18,.56),(.28,.52),(.45,.67)]
                low=[(.96,.55),(.94,.99),(.88 if style!=3 else .70,.99),(.86,.93-heel),(.71,.95-heel),(.53,.93-heel*.52),(.35,.99),(.16,.99),(.03,.93)]
                if style==3: low=[(.97,.91),(.91,.98),(.75,.99),(.49,.99),(.32,.99),(.14,.99),(.025,.9)]
                add(5,polygon(x,y,top+low),.012 if heel<.15 else .003)
        for band in [.22,.36,.49]:
            m=polygon(x,y,[(0,.83),(.12,.76),(.21,.35),(band+.17,.26),(band+.31,.69),(.8,.69),(.98,.51),(.99,.78),(.76,.91),(.37,.99),(.09,.96),(0,.90)])
            add(5,m)

def recenter(mask):
    # Only this render's bounding box is used. Preserve its proportions.
    ys,xs=np.nonzero(mask)
    if not xs.size: return mask
    h=ys.max()-ys.min()+1; w=xs.max()-xs.min()+1
    scale=max(h,w)/SIDE
    iy=np.floor((Y-.5)*SIDE*scale+(ys.max()+ys.min()+1)/2).astype(int)
    ix=np.floor((X-.5)*SIDE*scale+(xs.max()+xs.min()+1)/2).astype(int)
    return mask[np.clip(iy,0,SIDE-1),np.clip(ix,0,SIDE-1)]&(iy>=0)&(iy<SIDE)&(ix>=0)&(ix<SIDE)

make_uppers()
make_pants()
make_dress()
make_bags()
make_shoes()
lengths=np.array([len(b) for b in bank],np.int32)
np.savez(DEST/'constants.npz', patterns=np.column_stack([p for group in bank for p in group]),
         lengths=lengths, costs=np.array([p for group in priors for p in group],np.float32))

SOURCE = r'''import numpy as np

def clip(x):
    return np.clip(x,0,1)


def ramp(x,low,high):
    return clip((x-low)/(high-low))


def smooth(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return (4*a+2*(p[:,1:-1,:-2]+p[:,1:-1,2:]+p[:,:-2,1:-1]+p[:,2:,1:-1])+p[:,:-2,:-2]+p[:,:-2,2:]+p[:,2:,:-2]+p[:,2:,2:])/16


def bounds(m):
    px=m.sum(1); py=m.sum(2)
    cols=px>np.maximum(.48,px.max(1,keepdims=True)*.043)
    rows=py>np.maximum(.48,py.max(1,keepdims=True)*.043)
    x0=cols.argmax(1); x1=27-cols[:,::-1].argmax(1)
    y0=rows.argmax(1); y1=27-rows[:,::-1].argmax(1)
    return x0,x1,y0,y1


def sample(a,cx,cy,w,h):
    grid=(np.arange(28,dtype=np.float32)-13.5)/28
    xx=cx[:,None]+w[:,None]*grid
    yy=cy[:,None]+h[:,None]*grid
    return lookup(a,xx,yy)


def lookup(a,xx,yy):
    xx=np.clip(xx+1,0,28.9999); yy=np.clip(yy+1,0,28.9999)
    ix=xx.astype(np.int32); iy=yy.astype(np.int32)
    rx=(xx-ix)[:,None,:].astype(np.float32); ry=(yy-iy)[:,:,None].astype(np.float32)
    p=np.pad(a,((0,0),(1,1),(1,1)))
    n=np.arange(len(a))[:,None,None]
    i=iy[:,:,None]; j=ix[:,None,:]
    return (1-ry)*((1-rx)*p[n,i,j]+rx*p[n,i,j+1])+ry*((1-rx)*p[n,i+1,j]+rx*p[n,i+1,j+1])


def unit(a):
    v=a.reshape(len(a),-1)
    v=v-v.mean(1,keepdims=True)
    return v/np.maximum(.5,np.sqrt((v*v).sum(1,keepdims=True)))


def descriptor(m):
    p=np.pad(m,((0,0),(1,1),(1,1)))
    s=(4*m+2*(p[:,1:-1,:-2]+p[:,1:-1,2:]+p[:,:-2,1:-1]+p[:,2:,1:-1])+p[:,:-2,:-2]+p[:,:-2,2:]+p[:,2:,:-2]+p[:,2:,2:])/16
    cells=s.reshape(len(m),14,2,14,2).mean(axis=(2,4))
    proj=np.concatenate([m.mean(1),m.mean(2)],axis=1)
    return np.concatenate([unit(cells),unit(proj)],axis=1)


def match(views):
    # Use a fixed GEMM row shape even for a singleton so roundoff does not
    # depend on which other items have been supplied by the caller.
    n=len(views[0]); nv=len(views)
    v=np.zeros((24*nv,252),np.float32)
    for j,m in enumerate(views):
        v[24*j:24*j+n]=descriptor(m)
    sims=v @ CONSTANTS['patterns']
    sims-=CONSTANTS['costs']
    out=np.zeros((n,10),np.float32); pos=0
    for k,size in enumerate(CONSTANTS['lengths']):
        s=sims[:,pos:pos+size].max(1).reshape(nv,24)[:,:n]
        # Only mirror the query, never infer a label from other batch members.
        main=np.maximum(s[0],s[1]); alt=np.maximum(s[2],s[3])
        out[:,k]=.79*np.maximum(main,alt-.018)+.21*np.minimum(main,alt)
        pos+=size
    return out


def extent(b):
    live=b.any(axis=2)
    l=b.argmax(axis=2); r=27-b[:,:,::-1].argmax(axis=2)
    return l,r,(r-l+1)*live


def regionmean(a,b):
    return (a*b).sum(axis=(1,2))/np.maximum(1,b.sum(axis=(1,2)))


def appearance(scores, gray, mask, w, h):
    # Here the axes are fractional bounding-box coordinates, independent of
    # item size. Measurements refer to this query alone.
    b=mask>.43
    l,r,span=extent(b)
    rowwidth=span/28
    asp=w/h
    col=np.arange(28)[None,None,:]
    row=np.arange(28)[None,:,None]
    sym=1-np.abs(mask-mask[:,:,::-1]).sum((1,2))/(mask.sum((1,2))+2)
    asym=ramp(1-sym,.16,.62)
    wide=ramp(asp,1.08,1.80); tall=ramp(1/asp,1.05,1.9)
    # Thin handles, straps and gaps: distinguish a convex hull from its contents.
    hull=(col>=l[:,:,None])&(col<=r[:,:,None])&(span[:,:,None]>0)
    top=b.argmax(axis=1); bot=27-b[:,::-1].argmax(axis=1)
    vhull=(row>=top[:,None,:])&(row<=bot[:,None,:])&b.any(1)[:,None,:]
    holes=hull&vhull
    void=regionmean(1-mask,holes)
    below=regionmean(1-mask,holes&(row>=11))
    shapefill=mask.mean(axis=(1,2))
    upper=mask[:,2:12,8:20].mean((1,2)); core=mask[:,13:22,8:20].mean((1,2))
    corebottom=mask[:,20:26,10:18].mean((1,2))
    mid=mask[:,13:20,11:17].mean((1,2))
    gap=1-mask[:,17:26,12:16].mean((1,2))
    legs=(mask[:,18:26,4:11].mean((1,2))+mask[:,18:26,17:24].mean((1,2)))*.5
    trouser=ramp(gap,.26,.80)*ramp(legs,.33,.78)*ramp(upper,.72,.96)
    # Side view of shoe: the toe is lower than the ankle. Measure the two
    # ends away from the rounded-over pixels at the extremities.
    columns=b.any(1)
    lo=(top[:,3:10]*columns[:,3:10]).sum(1)/np.maximum(1,columns[:,3:10].sum(1))
    hi=(top[:,18:25]*columns[:,18:25]).sum(1)/np.maximum(1,columns[:,18:25].sum(1))
    slope=np.abs(hi-lo)/28
    toeslope=ramp(slope,.14,.50)
    footwear=np.maximum(wide*.50,asym*toeslope)
    shoeprior=.045*wide+.052*footwear-.052*(1-wide)*(1-asym)-.035*tall
    for k in [5,7,9]: scores[:,k]+=shoeprior
    for k in [0,1,2,3,4,6]: scores[:,k]-=.053*wide+.035*asym
    scores[:,1]+=.109*trouser+.037*ramp(1/asp,1.8,2.8)+.028*ramp(upper,.84,.98)-.047*ramp(.63-upper,0,.4)
    # High narrow handles above a substantial, mostly filled bag body.
    handle=ramp(core-upper,.22,.68)*ramp(core,.65,.94)*(1-.8*toeslope)
    scores[:,8]+=.085*handle+.036*wide*(1-toeslope)+.020*ramp(shapefill,.77,.91)-.025*tall*sym*(1-handle)
    for k in [0,2,3,4,6]: scores[:,k]-=.040*handle
    # Upright heel with an air space ahead of it and under the arch.
    bottom = np.where(columns, bot, 0)
    groundleft=bottom[:,2:11].max(1); groundright=bottom[:,18:27].max(1)
    arch=(np.minimum(groundleft,groundright)-bottom[:,12:21].min(1))/28
    arch=ramp(arch,.13,.40)
    strap=ramp(void,.045,.22); openarea=ramp(1-shapefill,.40,.69)
    scores[:,5]+=.105*strap+.047*ramp(below,.08,.25)+.040*openarea+.034*arch-.022*(1-strap)*(1-openarea)
    scores[:,7]+=.022*(1-strap)-.082*strap-.044*arch+.021*ramp(asp,1.85,3.0)
    scores[:,9]+=.026*(1-wide)+.018*toeslope-.062*strap-.015*ramp(asp,1.75,2.40)
    # A long level topline over the entire back quarter is characteristic of a shaft.
    back=np.where((hi<lo)[:,None],top[:,17:25],top[:,3:11][:,::-1])
    shaft=ramp((back<4).mean(1),.38,.88)*ramp(slope,.26,.60)
    scores[:,9]+=.047*shaft
    scores[:,7]-=.031*shaft
    # A dangling backpack strap or a second handle can make the article
    # asymmetric, but the body walls are often nearly vertical and turn into
    # a broad, level base. An outsole instead follows a much shallower body.
    flatbody=ramp(rowwidth[:,17:24].mean(1),.77,.95)*ramp(mask[:,16:24,4:24].mean((1,2)),.80,.98)
    scores[:,8]+=.038*flatbody*ramp(1-slope,.65,.90)*ramp(1.1-tall,.04,.65)
    # Shape cues for dresses and short-sleeved garments.
    bust=rowwidth[:,5:10].mean(1); waist=rowwidth[:,11:16].mean(1)
    hem=rowwidth[:,21:26].mean(1); low=rowwidth[:,18:23].mean(1)
    flare=ramp(hem-waist,.055,.31)
    dressform=flare*ramp(corebottom,.65,.98)
    scores[:,3]+=.066*dressform+.043*ramp(1/asp,1.62,2.45)*(1-trouser)-.055*trouser-.022*ramp(asp,.82,1.13)
    shortarm=ramp(bust-low,.13,.34)*ramp(core,.58,.91)
    scores[:,0]+=.031*shortarm-.038*dressform
    for k in [2,4,6]: scores[:,k]-=.025*dressform
    # Fabric evidence uses original tonal differences, not the binary silhouettes.
    # Search for a persistent narrow central opening/placket across several
    # heights. Subtract broad curvature and normalize by local fabric contrast.
    inner=b&(col>l[:,:,None]+1)&(col<r[:,:,None]-1)
    sm=smooth(gray)
    avg=regionmean(gray,inner&(row>4)&(row<25))
    norm=.16+.65*avg
    centers=[]; edges=[]
    for j in range(11,17):
        q=(sm[:,:,j]+sm[:,:,j+1])*.5
        left=(sm[:,:,j-3]+sm[:,:,j-2])*.5
        right=(sm[:,:,j+3]+sm[:,:,j+4])*.5
        contrast=(left+right)*.5-q
        lr=np.abs(left-right)
        sections=[]; side=[]
        for a,z in [(7,13),(13,19),(19,25)]:
            sections.append(contrast[:,a:z].mean(1)); side.append(lr[:,a:z].mean(1))
        sec=np.stack(sections,1); ed=np.stack(side,1)
        centers.append((np.abs(sec.mean(1))*.63+np.minimum(np.abs(sec).min(1),np.abs(sec.mean(1)))*.37)/norm)
        edges.append(ed.min(1)/norm)
    pl=np.stack(centers,1).max(1); ed=np.stack(edges,1).max(1)
    seam=ramp(pl,.048,.20); opening=ramp(pl,.15,.42); step=ramp(ed,.09,.30)
    # Difference between shoulders and center at the neck; V-shaped lapels often
    # continue from the neck almost to mid-chest, unlike a pullover crew neck.
    neck=sm[:,2:8,11:17].mean((1,2)); chest=sm[:,9:15,10:18].mean((1,2))
    ns=.5*(sm[:,2:8,6:10].mean((1,2))+sm[:,2:8,18:22].mean((1,2)))
    notch=ramp((.56*ns+.44*chest-neck)/norm,.11,.55)
    # Texture of interior cloth. Directional coherence separates checks and
    # prints from slowly changing shading and generally uniform knitwear.
    hp=np.abs(gray-sm)
    grain=regionmean(hp,inner&(row>=6)&(row<24))/norm
    grain=ramp(grain,.051,.145)
    dx=np.abs(sm[:,:,2:]-sm[:,:,:-2]); dy=np.abs(sm[:,2:]-sm[:,:-2])
    dx=regionmean(dx,inner[:,:,2:]&inner[:,:,:-2]&(row>=8)&(row<23))/norm
    dy=regionmean(dy,inner[:,2:]&inner[:,:-2]&(row[:,2:]>=8)&(row[:,2:]<23))/norm
    plaid=ramp(dx,.11,.29)*ramp(dy,.11,.27)
    # Compare the shirt-tail with the ends of the sleeves, measured relative to
    # the item box. A ribbed jumper often finishes a little above the cuffs.
    middle=bot[:,10:18].mean(1)
    sidebot=(bot[:,1:5].max(1)+bot[:,23:27].max(1))*.5
    cuffbelow=ramp((sidebot-middle)/28,-.025,.145)
    longcoat=ramp((middle-sidebot)/28,.08,.27)*(1-shortarm)
    # A curved bottom hem is common in shirts. Look at the central body, not
    # the end of the sleeves, for a shirttail that hangs lower in the middle.
    sidetail=(bot[:,6:9].mean(1)+bot[:,19:22].mean(1))*.5
    tail=ramp((middle-sidetail)/28,.034,.15)
    boxy=ramp(low,.76,.96)
    longarm=ramp(low/(bust+.03),.73,.99)
    scores[:,2]+=.034*(1-seam)+.020*(1-step)+.028*cuffbelow+.015*boxy-.042*opening-.021*notch-.021*plaid-.016*tail
    scores[:,4]+=.048*seam+.046*opening+.026*step+.021*notch+.030*longcoat+.013*tall-.022*shortarm-.025*plaid-.012
    scores[:,6]+=.033*seam+.012*step+.027*plaid+.028*grain+.028*tail+.020*notch-.015*cuffbelow-.014*boxy-.006
    # Track the two converging inner edges of open lapels. Local gradient
    # direction matters: scattered printed marks give less consistent support
    # than paired left/right diagonals which extend downwards toward the center.
    gx=(np.roll(sm,-1,2)-np.roll(sm,1,2))*.5
    gy=(np.roll(sm,-1,1)-np.roll(sm,1,1))*.5
    lapels=[]
    for depth in [8,12,16]:
        rr=np.arange(2,depth+1); t=(rr-2)/(depth-2)
        for radius in [4.5,6.5]:
            rad=radius*(1-t)+.8*t; sl=(radius-.8)/(depth-2)
            lcol=np.rint(13.5-rad).astype(np.int32); rcol=27-lcol
            evidence=[]
            for shift in [-1,0,1]:
                la=gx[:,rr,lcol+shift]-sl*gy[:,rr,lcol+shift]
                ra=-gx[:,rr,rcol+shift]-sl*gy[:,rr,rcol+shift]
                evidence.append((la+ra)*.5)
            # Minor alignment deviations are allowed; polarity must be
            # consistent up and down both lapels, whether dark or light.
            e=np.stack(evidence,1)/(1+sl*sl)
            a=e.max(1); d=-e.min(1); split=len(rr)//2
            support=np.maximum(np.minimum(a[:,:split].mean(1),a[:,split:].mean(1)),np.minimum(d[:,:split].mean(1),d[:,split:].mean(1)))
            lapels.append(support/norm)
    lapel=ramp(np.stack(lapels,1).max(1),.085,.24)
    scores[:,4]+=.031*lapel*ramp(longarm,.18,.80)
    scores[:,6]+=.012*lapel
    scores[:,2]-=.019*lapel
    # A short plain tee is the default short-sleeved garment. Distinguish a
    # woven collared short-sleeve shirt when central opening/texture is visible.
    scores[:,0]+=.014*(1-seam)*shortarm-.028*opening*shortarm-.013*tail
    # We only compare the fine fabric and hem evidence inside the upper-garment
    # family. Avoid raising a poor coat match above a good bag or shoe match
    # just because a zipper or strap produced a strong placket response.
    return scores


def block(a):
    # Robust foreground segmentation with a permissive and a stricter view.
    n=len(a); peak=np.maximum(.055,np.sort(a.reshape(n,-1),axis=1)[:,-18])
    sm=smooth(a); frac=.65*a+.35*sm
    base=.008+.033*peak
    first=clip((frac-base[:,None,None])/(.020+.091*peak[:,None,None]))
    # Smooth away isolated accent marks, but keep the edges of thin straps.
    m=.77*first+.23*smooth(first)
    second=clip((frac-(.014+.082*peak[:,None,None]))/(.030+.14*peak[:,None,None]))
    x0,x1,y0,y1=bounds(m)
    w=(x1-x0+1).astype(np.float32); h=(y1-y0+1).astype(np.float32)
    cx=(x0+x1).astype(np.float32)*.5; cy=(y0+y1).astype(np.float32)*.5
    size=np.maximum(w,h)
    m1=sample(m,cx,cy,size,size); m2=sample(second,cx,cy,size,size)
    shape=match([m1,m1[:,:,::-1],m2,m2[:,:,::-1]])
    scores=appearance(shape.copy(),sample(a,cx,cy,w,h),sample(m,cx,cy,w,h),w,h)
    ids=[0,2,4,6]
    # Within-family cues are useful even when all four silhouette correlations
    # are nearly tied. Remove their shared uplift in cross-family comparison.
    uplift=scores[:,ids].max(1)-shape[:,ids].max(1)
    scores[:,ids]-=np.maximum(.025,uplift-.048)[:,None]
    return scores.argmax(1)


def predict(x):
    result=np.empty(len(x),np.int64)
    for i in range(0,len(x),24):
        result[i:i+24]=block(np.ascontiguousarray(x[i:i+24,0],dtype=np.float32))
    return result
'''
(DEST/'solution.py').write_text(SOURCE)
