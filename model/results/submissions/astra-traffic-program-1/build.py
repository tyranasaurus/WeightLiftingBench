"""Data-free traffic sign renderer and classifier builder."""
import sys
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
K={}
# Render in a square coordinate system; a sign face spans [-1,1].
side=192
v=(np.arange(side)+.5)/side*2-1
Y,X=np.meshgrid(v,v,indexing='ij')

def poly(pts):
    a=np.zeros(X.shape,bool)
    for (x,y),(u,v) in zip(pts,pts[1:]+pts[:1]):
        if y!=v: a ^= ((Y>y)!=(Y>v)) & (X<(u-x)*(Y-y)/(v-y)+x)
    return a

def ellipse(x,y,rx,ry):return ((X-x)/rx)**2+((Y-y)/ry)**2<=1

def stroke(pts,width):
    a=np.zeros(X.shape,bool)
    for (x,y),(u,v) in zip(pts,pts[1:]):
        dx=u-x; dy=v-y
        t=np.clip(((X-x)*dx+(Y-y)*dy)/(dx*dx+dy*dy+1e-25),0,1)
        a |= (X-x-t*dx)**2+(Y-y-t*dy)**2 <= (width*.5)**2
    return a

def curve(pts,n=16):
    t=np.linspace(0,1,n)[:,None]; a,b,c,d=np.array(pts)
    return (a*(1-t)**3+3*b*t*(1-t)**2+3*c*t*t*(1-t)+d*t**3).tolist()

def curved(pts,width):return stroke(curve(pts),width)

def rect(x,y,u,v):return (X>=x)&(X<=u)&(Y>=y)&(Y<=v)

# DIN-like numerals, built from open/closed centerlines. Digit coordinates
# below are in a unit typographic em, independent of pixels in the input.
def digit(d,variant):
    # Original geometry transformed by the caller to make tall and condensed type.
    def path(*p):
        res=[p[0]]
        for q in p[1:]:
            if len(q)==2: res.append(q)
            else:res+=curve((res[-1],q[:2],q[2:4],q[4:6]))[1:]
        return res
    if d=='0':
        return [path((.5,.075),(.82,.075,.90,.19,.90,.38),(.90,.64),(.90,.84,.81,.925,.50,.925),(.20,.925,.10,.82,.10,.64),(.10,.38),(.10,.17,.22,.075,.5,.075))]
    if d=='1':return [path((.18,.235),(.67,.08),(.67,.925))]
    if d=='2':return [path((.11,.26),(.15,.02,.85,.005,.88,.235),(.96,.41,.73,.52,.52,.64),(.12,.91),(.93,.91))]
    if d=='3':return [path((.12,.18),(.34,.01,.85,.035,.86,.245),(.88,.40,.70,.47,.43,.48),(.72,.47,.91,.57,.89,.72),(.89,.95,.34,.995,.10,.80))]
    if d=='5':return [path((.88,.09),(.22,.09),(.17,.48),(.35,.36,.83,.36,.88,.58),(.95,.74,.83,.925,.52,.925),(.30,.93,.17,.87,.10,.78))]
    if d=='6':
        loop=path((.32,.49),(.57,.33,.90,.45,.90,.67),(.91,.86,.77,.93,.51,.93),(.23,.93,.10,.83,.11,.66),(.11,.57,.17,.51,.32,.49))
        top=path((.14,.60),(.17,.44,.43,.23,.69,.075)) if variant!=2 else path((.11,.64),(.09,.28,.20,.04,.69,.09),(.82,.11))
        return [loop,top]
    if d=='7':return [path((.10,.095),(.91,.095),(.35,.92))]
    if d=='8':return [path((.51,.465),(.25,.465,.13,.37,.13,.27),(.13,.10,.34,.07,.51,.07),(.75,.07,.87,.15,.87,.27),(.87,.39,.71,.46,.51,.465),(.25,.47,.095,.57,.095,.705),(.095,.865,.25,.93,.50,.93),(.78,.93,.91,.84,.91,.705),(.91,.57,.74,.47,.51,.465))]

def number(text,style):
    # Font alternatives have different condensation, weights and tracking.
    w=(.38 if len(text)==2 else .305)*[1,.95,1.08][style]; h=.79
    gap=.09; widths=[w*(.65 if c=='1' else 1) for c in text]; left=-(sum(widths)+(len(text)-1)*gap)*.5
    a=np.zeros(X.shape,bool)
    for c,cw in zip(text,widths):
        for p in digit(c,style):
            a |=stroke([(left+u*cw,(v-.5)*h) for u,v in p],[.099,.081,.111][style])
        left+=cw+gap
    return a

def car(x,y,w=0.48):
    # Rear silhouette, with bright windscreen and lamps.
    def pp(pts):return poly([(x+u*w,y+v*w) for u,v in pts])
    a=pp([(-.48,.25),(-.48,-.04),(-.37,-.13),(-.28,-.36),(.25,-.36),(.38,-.13),(.48,-.04),(.48,.25),(.37,.25),(.35,.34),(.22,.34),(.21,.24),(-.24,.24),(-.24,.34),(-.38,.34),(-.39,.25)])
    a &=~pp([(-.29,-.12),(-.22,-.28),(.19,-.28),(.27,-.12)])
    return a

def truck(x,y,w,h,mirror=False):
    def p(pts): return poly([(x+u*w*(-1 if mirror else 1),y+v*h) for u,v in pts])
    a=p([(-.48,.25),(-.48,-.105),(-.34,-.31),(-.19,-.31),(-.19,-.43),(.48,-.43),(.48,.26)])
    a &=~p([(-.41,-.09),(-.32,-.25),(-.24,-.25),(-.24,-.09)])
    for t in [-.32,.29]:a|=ellipse(x+(-1 if mirror else 1)*t*w,y+.26*h,.13*h,.13*h)
    return a

def arrowhead(x,y,dx,dy,w=.30,h=.29,th=.12):
    # Open-sided arrow head, centered at the tip.
    q=np.hypot(dx,dy);dx/=q;dy/=q
    def at(s,t):return(x-dx*s-dy*t,y-dy*s+dx*t)
    return poly([at(0,0),at(h,w),at(h+th,w-th*.65),at(th*1.25,0),at(h+th,-w+th*.65),at(h,-w)])

def arrow(c):
    if c in (38,39):
        a=stroke([(-.44,-.49),(.40,.43)],.165)|arrowhead(.49,.52,1,1,.39,.38,.125)
    if c in (35,36,37):
        x=0 if c==35 else -.16
        a=rect(x-.083,-.54,x+.083,.72)|arrowhead(x,-.75,0,-1,.31,.29,.12)
        if c!=35:a|=rect(x,.07,.46,.235)|arrowhead(.66,.15,1,0,.26,.24,.12)
    if c in (33,34):
        a=rect(-.22,-.04,-.055,.66)|curved([(-.138,-.03),(-.14,-.31),(-.11,-.37),(.18,-.37)],.165)|rect(.15,-.452,.46,-.287)|arrowhead(.62,-.37,1,0,.31,.29,.13)
    if c==40:
        a=np.zeros(X.shape,bool)
        for k in range(3):
            ts=np.linspace(-.3+k*2*np.pi/3,-1.62+k*2*np.pi/3,20)
            pts=list(zip(.51*np.cos(ts),.51*np.sin(ts))); t=ts[-1]
            a|=stroke(pts,.095)|arrowhead(pts[-1][0],pts[-1][1],np.sin(t),-np.cos(t),.19,.19,.08)
    if c in (34,37,39):a=a[:,::-1]
    return a

def person(x,y,s,lean=0,flip=1):
    # Walking silhouette positioned by the top of the head.
    def pts(p):return [(x+flip*(u+lean*(1-v))*s,y+v*s) for u,v in p]
    def st(p,w):return stroke(pts(p),s*w)
    hx,hy=pts([(0,.09)])[0];a=ellipse(hx,hy,s*.092,s*.095)
    a|=st([(-.035,.24),(.025,.50)],.195)|st([(.035,.52),(-.07,.73),(-.24,.96)],.10)|st([(.025,.54),(.13,.73),(.22,.95)],.115)
    a|=st([(-.025,.27),(-.16,.42),(-.22,.51)],.075)|st([(.006,.27),(.14,.43),(.24,.46)],.082)
    return a

def symbol(c,style):
    nil=np.zeros(X.shape,bool); red=nil.copy(); a=nil.copy()
    if c in (0,1,2,3,4,5,7,8,6):a=number({0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}[c],style)
    if c in (9,41):red=car(-.29,-.028,.52); a=car(.27,.005,.54)
    if c in (10,42):
        if style==2:red=truck(-.32,-.015,.54,.51,True)
        else:red=rect(-.56,-.32,-.13,.12)|rect(-.53,-.04,-.16,.22)
        a=car(.25,.04,.51)
    if c==16:a=truck(0,0,1.18,.65,style==2)
    if c in (41,42):a |=red;red=nil.copy()
    if c in (6,32,41,42):
        # End-of-restriction hatch: often blurred into a single gray band.
        diag=X+Y; a |= (abs(diag)<(.245 if style!=2 else .31)) & (X*X+Y*Y<1) & ((np.cos(diag*66)>(-.40 if style==0 else -.82)) if style!=2 else True)
    if c==11:
        a=poly([(-.095,.52),(.095,.52),(.095,.19),(.34,.19),(.15,.075),(.095,.02),(.095,-.10),(0,-.36),(-.095,-.10),(-.095,.02),(-.34,.19),(-.095,.19)])
    if c==18:
        a=poly([(-.073,-.35),(.073,-.35),(.041,.28),(-.041,.28)])|ellipse(0,.45,.07,.077)
    if c in (19,20):
        # A strip of constant width with two perpendicular straight ends.
        cy=.30; cx=-.065; r=.24; thick=.15
        a=rect(-.26,cy-r-thick/2,cx,cy-r+thick/2)|rect(cx+r-thick/2,cy,cx+r+thick/2,.56)
        dr=np.hypot(X-cx,Y-cy);a |= (abs(dr-r)<thick/2)&(X>=cx)&(Y<=cy)
        if c==20:a=a[:,::-1]
    if c==21:
        # Two bends between parallel vertical ends (a reverse curve).
        a=rect(.105,.28,.265,.56)|rect(-.215,-.28,-.055,.12)
        a|=curved([(.185,.29),(.18,.10),(-.13,.28),(-.135,.11)],.16)
        if style==2:a=a[:,::-1]
    if c==22:
        a=rect(-.45,.41,.46,.55)|ellipse(-.23,.47,.18,.19)|ellipse(.18,.45,.22,.22);a&=Y<=.55
    if c==23:
        # The skidding car has a dark body above a pair of much finer tracks.
        a=car(0,.005,.61)
        for x in (-.19,.19):a|=curved([(x+.06,.13),(x-.31,.32),(x+.30,.30),(x-.055,.57)],.041)
        if style>=1:
            a=car(0,.025,.65)
            for x in (-.20,.20):a|=curved([(x,.19),(x-.27,.36),(x+.23,.26),(x-.09,.57)],.063)
        if style>=2:
            # In the tilted-car version the short body is a nearly horizontal
            # solid sweep, with dotted or broken tire marks beneath it.
            body=poly([(-.32,-.12),(.25,-.12),(.30,.06),(.27,.115),(-.29,.115)])
            ix=np.clip(np.rint((X+(.22 if style==2 else -.22)*Y+1)*side/2-.5).astype(int),0,side-1)
            iy=np.clip(np.rint((Y-(.22 if style==2 else -.22)*X+1)*side/2-.5).astype(int),0,side-1)
            a=body[iy,ix]
            for z in (-.19,.19):a |= curved([(z,.075),(z+.20,.28),(z-.26,.30),(z,.52)],.052)&((Y<.29)|(Y>.35))
    if c==24:
        a=rect(-.20,-.265,-.13,.53)|stroke([(.04,-.25),(.04,-.11),(.18,.22),(.18,.51)],.077)
    if c==25:
        a=ellipse(.00,-.22,.087,.09)|stroke([(-.04,-.085),(-.16,.10),(-.17,.23)],.19)|stroke([(-.13,.21),(-.16,.30),(-.28,.51)],.125)|stroke([(-.13,.20),(.005,.305),(.07,.51)],.13)|stroke([(-.05,-.025),(.045,.135),(.14,.17)],.08)
        a|=stroke([(-.25,-.03),(.30,.47)],.049)|poly([(.10,.55),(.29,.33),(.37,.30),(.57,.55)])
    if c==26:
        a=ellipse(0,-.16,.102,.103)|ellipse(0,.105,.103,.105)|ellipse(0,.38,.105,.105);red=ellipse(0,-.16,.10,.10)
    if c==27:
        # International walking figure: head, solid trunk, swinging arms and
        # diagonal legs. Alternatives vary stride independently of height.
        a=person(0,-.34,.92,-.05 if style==0 else -.16,-1)
        if style==1:a |= stroke([(-.015,-.085),(-.08,.215)],.16)
        if style>=2:
            a=ellipse(-.065,-.23,.086,.10)|stroke([(-.065,-.064),(.005,.20)],.17)
            a|=stroke([(.0,.18),(.105,.35),(.21,.55)],.115)|stroke([(.0,.19),(-.065,.345),(-.22,.525)],.115)
            a|=stroke([(-.065,-.04),(-.15,.12),(-.205,.18)],.063)|stroke([(-.035,-.065),(.11,.035),(.16,.175)],.068)
            if style==3:a=a[:,::-1]
    if c==28:
        # The crossing pair consists of a smaller child at the left and the
        # larger running figure at the right, joined by an outstretched arm.
        a=person(.14,-.25,.85,-.13,1)|person(-.215,.075,.52,.035,1)|stroke([(.065,.04),(-.035,.20),(-.17,.21)],.080)
        if style>=2:
            a=person(.055,-.26,.85,-.21,1)|person(-.245,.11,.49,.07,-1)
            a|=stroke([(-.07,.035),(-.18,.125),(-.22,.25)],.090)|poly([(-.07,.04),(.07,.035),(.18,.31),(-.01,.30)])
        if style in (1,3):a=a[:,::-1]
    if c==29:
        for x in [-.275,.272]:a|=ellipse(x,.355,.171,.17)&~ellipse(x,.355,.119,.118)
        a|=stroke([(-.275,.35),(-.13,.08),(.00,.35),(-.275,.35)],.047)|stroke([(0,.35),(.17,.08),(.27,.35)],.049)|ellipse(.12,-.245,.083,.084)|stroke([(.025,-.12),(-.125,.028),(.00,.145),(-.03,.295)],.093)|stroke([(.015,-.105),(.12,.035),(.228,.058)],.067)
        if style in (1,3):a=a[:,::-1]
        if style>=2:
            # Also model the bicycle symbol without a rider: two circular
            # rims, seat, raised handlebars, fork and a triangulated frame.
            a=nil.copy()
            for x in [-.275,.272]:a |=ellipse(x,.34,.192,.19)&~ellipse(x,.34,.145,.143)
            a|=stroke([(-.275,.34),(-.11,.07),(.025,.34),(-.275,.34)],.044)|stroke([(-.11,.07),(.153,.06),(.025,.34)],.044)|stroke([(.27,.34),(.12,-.02),(.20,-.035)],.047)|stroke([(-.13,.08),(-.16,-.015),(-.23,-.015)],.043)
            if style==3:a=a[:,::-1]
    if c==30:
        for t in np.arange(6)*np.pi/3:
            u=np.sin(t);v=np.cos(t)
            def pos(r,b):return (u*r-v*b,.17+v*r+u*b)
            a|=stroke([pos(0,0),pos(.37,0)],.049)
            for b in [-.10,.10]:a|=stroke([pos(.19,0),pos(.30,b)],.045)
    if c==31:
        a=ellipse(-.005,.09,.257,.106)|poly([(-.24,.105),(-.32,-.22),(-.235,-.19),(-.18,-.045),(-.04,.055)])|poly([(-.28,-.13),(-.41,-.21),(-.29,-.28),(-.205,-.23)])
        a|=stroke([(-.15,.13),(-.14,.29),(-.32,.39)],.068)|stroke([(-.13,.12),(-.05,.26),(-.17,.34)],.065)|stroke([(.18,.12),(.30,.20),(.34,.38),(.48,.47)],.078)|stroke([(.15,.13),(.24,.28),(.22,.39)],.068)|stroke([(-.25,-.21),(-.16,-.37),(-.18,-.41)],.035)
        if style==1:
            # A stretched-out leaping deer, with the rear hooves trailing.
            a=ellipse(.035,.12,.255,.106)|poly([(-.16,.17),(-.285,-.12),(-.26,-.245),(-.18,-.21),(-.14,-.045),(.06,.045)])
            a|=poly([(-.28,-.14),(-.39,-.21),(-.32,-.28),(-.19,-.22)])|stroke([(-.13,.135),(-.24,.17),(-.31,.04)],.066)|stroke([(-.06,.16),(-.12,.24),(-.25,.24)],.064)
            a|=stroke([(.20,.14),(.32,.24),(.45,.40)],.087)|stroke([(.21,.11),(.35,.18),(.50,.33)],.070)|stroke([(-.24,-.20),(-.20,-.38)],.043)
        if style>=3:
            # Graphic alternative: upright head and neck, a sloping back,
            # forelegs bent below the chest, long outstretched hind legs.
            a=poly([(-.235,-.27),(-.15,-.23),(-.115,-.07),(.005,-.02),(.17,.065),(.26,.16),(.245,.258),(.13,.235),(-.035,.16),(-.15,.16),(-.23,.075)])
            a|=poly([(-.22,-.24),(-.31,-.17),(-.25,-.10),(-.15,-.175)])|stroke([(-.205,-.22),(-.15,-.37)],.042)
            a|=stroke([(-.13,.11),(-.22,.24),(-.30,.21)],.071)|stroke([(-.05,.14),(-.10,.25),(-.22,.31)],.066)
            a|=stroke([(.17,.165),(.27,.26),(.36,.39),(.47,.45)],.094)|stroke([(.17,.14),(.32,.22),(.39,.35)],.079)
            if style==4:a|=stroke([(-.19,-.18),(-.18,.085)],.15)|stroke([(-.15,.063),(.19,.17)],.16)
        if style>=5:
            # Low-resolution silhouette of a leaping animal. The neck rises
            # ahead of the body; a flat belly ends in tapered, swept-back legs.
            a=poly([(-.34,-.19),(-.235,-.255),(-.19,-.37),(-.153,-.28),(-.08,-.34),(-.135,-.19),(-.12,-.055),(-.01,-.015),(.20,.042),(.275,.146),(.30,.25),(.46,.44),(.405,.445),(.24,.327),(.115,.220),(-.025,.16),(-.20,.16),(-.25,.11),(-.25,-.115),(-.34,-.135)])
            a |=stroke([(-.14,.095),(-.23,.22),(-.31,.18)],.058)|stroke([(.18,.14),(.20,.29),(.34,.42)],.058)
            if style==6:
                # Second posture: high shoulders and long, straight hind leg.
                a=poly([(-.29,-.30),(-.17,-.35),(-.12,-.25),(-.14,-.11),(-.065,-.065),(.16,.005),(.255,.095),(.30,.20),(.44,.41),(.38,.45),(.19,.28),(.07,.20),(-.15,.155),(-.27,.13),(-.28,-.09),(-.32,-.13)])|stroke([(-.12,.08),(-.24,.21),(-.32,.18)],.065)
        if style==2:a=a[:,::-1]
    if c in range(33,41):a=arrow(c)
    if c==17:a=rect(-.72,-.145,.72,.145)
    if c==14:
        # STOP word in white capitals
        def glyph(ox,w,paths):
            z=nil.copy()
            for p in paths:z|=stroke([(ox+xx*w,(yy-.5)*.52) for xx,yy in p],.066)
            return z
        a=glyph(-.71,.27,[[(.90,.08),(.65,.02),(.2,.04),(.05,.19),(.1,.36),(.85,.55),(.97,.76),(.84,.94),(.25,.97),(.03,.87)]])|glyph(-.37,.29,[[(0,0),(1,0)],[(.5,0),(.5,1)]])|glyph(-.01,.27,[[tuple((.50+.46*np.cos(t),.5+.48*np.sin(t))) for t in np.linspace(0,2*np.pi,22)]])|glyph(.38,.28,[[(.03,.97),(.03,.03),(.72,.03),(.96,.13),(.96,.37),(.78,.49),(.03,.49)]])
    return a.astype(float),red.astype(float)

# Outer face models (separate from graphics so border registration isn't
# dominated by fine lettering). Each stencil compares regions of the expected
# face, trim and surroundings, across candidate locations in the photo.
families=['ring','triangle','yield','diamond','blue','end','stop','entry']

def extent(f,x,y):
    if f=='triangle':return np.maximum(y,(2*abs(x)-y)) # apex y=-1, base y=1
    if f=='yield':return np.maximum(-y,(2*abs(x)+y))
    if f=='diamond':return abs(x)+abs(y)
    if f=='stop':return np.maximum(np.maximum(abs(x),abs(y)),(abs(x)+abs(y))/np.sqrt(2))
    return np.hypot(x,y)

def frac(f,x,y):
    if f in ('triangle','yield'):
        if f=='yield':y=-y
        return np.maximum((y-1/3)*1.5,(2*abs(x)-y+1/3)*.75)
    return extent(f,x,y)

def avg(a):return a.reshape(16,4,16,4).mean(axis=(1,3)).ravel()

def mkernel(f,rx,ry,dx,dy,shear):
    t=(np.arange(64)+.5)*.5-16; yy,xx=np.meshgrid(t,t,indexing='ij'); y=(yy-dy)/ry; x=(xx-dx)/rx-shear*y
    d=frac(f,x,y)
    if f in ('triangle','yield'):inner=.68
    elif f=='diamond':inner=.69
    else:inner=.765
    ring=(d<.975)&(d>inner+.025); core=(d<inner-.035); outside=(d>1.05)&(d<1.33)
    if f=='blue':ring=(d<.92)&(d>.51);core=(d<.52)
    if f=='stop' or f=='entry':ring=(d<.91)&(abs(y)>.34);core=(d<.91)&(abs(y)<.22)
    def n(m):return m/(m.sum()+1e-10)
    if f=='ring' or f=='triangle' or f=='yield':
        # Don't let black pictograms determine the white level.
        core=(d<inner-.025)&(d>inner*.48)
        hue=1.20*n(ring)-.60*n(core)-.43*n(outside);light=.91*(n(core)-n(ring)); chrom=n(ring)*.23
    elif f=='diamond':
        hue=1.00*n(core)-.66*n(ring)-.25*n(outside);light=.64*(n(ring)-.6*n(core)-.4*n(outside));chrom=.32*n(core)
    elif f=='blue':
        hue=1.12*n(ring)+.04*n(core)-.85*n(outside);light=.12*(n(core)-n(ring));chrom=.53*n(ring)
    elif f=='end':
        b=abs(x+y)<.18; core=(d<.89)&(~b);ring=(d<.85)&b
        hue= -0.46*n(core);light=.93*(n(core)-n(ring))+.40*(n(core)-n(outside));chrom=-.2*n(core)
    else:
        hue=1.18*n(ring)-.53*n(core)-.48*n(outside);light=.78*(n(core)-n(ring));chrom=.39*n(ring)
    # sum rather than mean after supersampling to preserve sum of coefficients
    return np.concatenate((avg(hue),avg(light),avg(chrom)))*16

poses=[(rx,ry,x,y,s) for x in (-1.6,0,1.6) for y in (-1.6,0,1.6) for rx in (10,12.5,15) for ry in (10,12.5,15) for s in (0,)]
K['poses']=np.array(poses,np.float32)
for i,f in enumerate(families):K['outer'+str(i)]=np.array([mkernel(f,*p) for p in poses],np.float32)

# Second registration pass at full photo resolution searches nearby scales,
# shifts and slants. Sample just the three characteristic regions of the sign.
# Stratified low-discrepancy locations are data independent.
for i,f in enumerate(families):
    t=(np.arange(84)+.5)/84*2.8-1.4; y,x=np.meshgrid(t,t,indexing='ij');d=frac(f,x,y)
    inner=.68 if i in (1,2,3) else .765
    regs=[(d>inner+.035)&(d<.955),(d<inner-.035)&(d>inner*.53),(d>1.055)&(d<1.24)]
    if i==4:regs=[(d>.50)&(d<.93),d<.50,regs[2]]
    if i in (6,7):regs=[(d<.93)&(abs(y)>.32),(d<.89)&(abs(y)<.20),regs[2]]
    if i==5:regs=[(d<.90)&(abs(x+y)<.16),(d<.90)&(abs(x+y)>.31),regs[2]]
    vals=[]
    for reg in regs:
        pts=np.flatnonzero(reg); count=80
        # Use a jittered grid of scan lines, spread across each line.
        ix=pts[np.minimum(len(pts)-1,np.floor((np.arange(count)+(np.arange(count)*.6180339)%1)*len(pts)/count).astype(int))]
        vals.append(np.stack([x.ravel()[ix],y.ravel()[ix]],axis=0))
    K['land'+str(i)]=np.array(vals,np.float32)

# Class-specific inner face comparisons. Each class competes over small
# geometric deformations, sign typesetting styles, and degrees of defocus.
# Features are normalized within a common window for the family. Normalized
# cross-correlation is implemented via dot products using precentered masks.
classes=[[0,1,2,3,4,5,7,8,9,10,15,16],[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],[13],[12],list(range(33,41)),[6,32,41,42],[14],[17]]

def sample(a,x,y):
    xx=np.clip((x+1)*side/2-.5,0,side-1.001);yy=np.clip((y+1)*side/2-.5,0,side-1.001);ix=xx.astype(int);iy=yy.astype(int);fx=xx-ix;fy=yy-iy
    return (a[iy,ix]*(1-fx)+a[iy,ix+1]*fx)*(1-fy)+(a[iy+1,ix]*(1-fx)+a[iy+1,ix+1]*fx)*fy

def smooth(a,s):
    r=int(np.ceil(3*s));t=np.arange(-r,r+1);k=np.exp(-.5*(t/s)**2);k/=k.sum()
    for axis in (0,1):
        a=sum(b*np.roll(a,int(ti),axis=axis) for ti,b in zip(t,k))
    return a

for i,cl in enumerate(classes):
    # Work mainly at native resolution, taking more sample points on glyphs.
    t=(np.arange(30)+.5)/30*2-1;y,x=np.meshgrid(t,t,indexing='ij');d=frac(families[i],x,y)
    if i==0:mask=(d<.703)&(abs(y)<.53)
    elif i==1:mask=d<.695
    elif i==4:mask=d<.87
    elif i==5:mask=d<.90
    else:mask=d<.88
    x=x[mask];y=y[mask];K['grid'+str(i)]=np.array([x,y],np.float32)
    if i in (2,3):continue
    arr=[];colors=[];starts=[];renderwidth=[]
    for c in cl:
        starts.append(len(arr))
        if c==15: # uniformity is tested directly in inference
            arr.append(x*0);colors.append(x*0);renderwidth.append(.05);continue
        styles=range(7) if c==31 else range(4) if c in (23,27,28,29) else range(3) if c in (0,1,2,3,4,5,6,7,8,10,16,21,32,41,42) else range(1)
        for style in styles:
            a,b=symbol(c,style)
            for blur in ((.031,.068,.102) if i==1 else (.031,.068)):
                if c in (6,41,42):
                    # Cancelled graphics are gray behind black hatching, not
                    # the same black as on a normal red-ring sign.
                    base=(number('80',style) if c==6 else np.maximum(*symbol(9 if c==41 else 10,style)))
                    stripe=symbol(32,style)[0];g=smooth(np.maximum(stripe,base*(.48 if style!=1 else .75)),blur*side/2)
                else:g=smooth(np.maximum(a,b*(.66 if c in (9,10) else 1)),blur*side/2)
                col=smooth(b,blur*side/2) if b.any() and c in (9,10,26) else None
                for sx in (.84,1,1.17):
                    for sy in (.84,1,1.17):
                        for sh in (0,):
                            for dx in (-.066,0,.066):
                                for dy in ((-.15,-.065,.02,.105,.19) if i==1 else (-.067,.00,.067)):
                                    cy=1/3 if i==1 else 0
                                    xx=(x-dx-sh*y)/sx; yy=(y-dy-cy)/sy+cy
                                    p=sample(g,xx,yy)
                                    n=p-p.mean();n/=np.sqrt(np.sum(n*n))+.00001
                                    arr.append(n);colors.append(x*0 if col is None else sample(col,xx,yy));renderwidth.append(blur*(sx+sy)*.5)
    K['class'+str(i)]=np.array(cl,np.int32);K['start'+str(i)]=np.array(starts,np.int32);K['blur'+str(i)]=np.array(renderwidth,np.float32)
    # Oriented differences of sign interior stencils provide a second test of
    # the edges, less sensitive to overall contrast and shading.
    arr=np.array(arr,dtype=np.float32)
    if i in (0,1,5):
        flatidx=np.flatnonzero(mask);edges=[]
        for ds in (1,30):
            ok=np.isin(flatidx+ds,flatidx);edges.append(np.stack([np.flatnonzero(ok),np.searchsorted(flatidx,flatidx[ok]+ds)]))
        edge=np.concatenate(edges,axis=1);K['edge'+str(i)]=edge.astype(np.int32)
        dif=arr[:,edge[0]]-arr[:,edge[1]]; dif/=np.sqrt((dif*dif).sum(1,keepdims=True))+1e-7
        sc=np.max(abs(dif),axis=1)+1e-8;K['edtpl'+str(i)]=np.rint(dif/sc[:,None]*127).astype(np.int8);K['edamp'+str(i)]=sc/127
    # Quantization to one byte saves copy and setup time in the worker.
    gamma=np.max(abs(arr),1)+1e-8
    K['tpl'+str(i)]=np.rint(arr/gamma[:,None]*127).astype(np.int8);K['amp'+str(i)]=gamma/127
    colors=np.array(colors,np.float32);idx=np.flatnonzero(colors.max(axis=1)>0)
    if len(idx):
        co=colors[idx]; co=co-co.mean(axis=1,keepdims=True); co/=np.sqrt(np.sum(co**2,axis=1,keepdims=True))+.00001
        K['coloridx'+str(i)]=idx.astype(np.int32);K['colortpl'+str(i)]=co.astype(np.float32)

# A second, compositional OCR pass. The rightmost 0 is the same on every
# speed sign; register the numeral string to that glyph, then compare its
# other digits in their own windows. This prevents the shared zero or a
# nearly invisible stroke of the 1 from dominating the decision.
t=(np.arange(29)+.5)/29*2-1; y,x=np.meshgrid(t,t,indexing='ij'); mask=(abs(y)<.54)&(np.hypot(x,y)<.72);x=x[mask];y=y[mask];K['ocrgrid']=np.array([x,y],np.float32)
ocr=[];digitlabels=[];K['ocrpose']=[]
for text in ('20','30','50','60','70','80','100','120'):
    digitlabels.append(len(ocr)); num=len(text)
    w=.38 if num==2 else .305; gap=.09;ww=[w*(.65 if c=='1' else 1) for c in text];left=-(sum(ww)+gap*(num-1))*.5
    cuts=[]
    for width in ww[:-1]:left+=width+gap;cuts.append(left-gap*.5)
    for style,bl in [(0,.021),(1,.041),(2,.06)]:
        g=smooth(number(text,style),bl*side/2)
        for sx in (.80,.94,1.08,1.22):
            for sy in (.86,1.04):
                for dx in (-.10,-.05,0,.05,.10):
                    for dy in (-.07,0,.07):
                        for sl in (-.105,.05):
                            xx=(x-dx-sl*y)/sx;yy=(y-dy)/sy;p=sample(g,xx,yy)
                            # Score only after registration in a shared word window;
                            # the segment windows below are used for a separate pass.
                            a=p-p.mean();a/=np.linalg.norm(a)+1e-7
                            ocr.append(a)
                            # Store statistics for the maximizer only at runtime.
                            # The best glyph string is decoded by subword agreement.
                            # Cuts in rectified x; store geometry to recover words.
                            K['ocrpose'].append((sx,dx,sl,cuts[-1],cuts[0]))
ocr=np.array(ocr,np.float32);amp=np.max(abs(ocr),axis=1)+1e-7;K['ocr']=np.rint(ocr/amp[:,None]*127).astype(np.int8);K['ocramp']=amp/127;K['ocrpos']=np.array(digitlabels,np.int32);K['ocrpose']=np.array(K['ocrpose'],np.float32)

# Elastic pictogram comparison. Predict sign graphics in five overlapping
# parts; allow individual parts a small slip while keeping their common scale
# and center. This models articulation of limbs, wheels, and antlers without
# requiring the exact outline of any one sign edition.
x,y=K['grid1'];base=[];layers=[];labels=[]
for c in classes[1]:
    st=range(7) if c==31 else range(4) if c in (23,27,28,29) else range(3) if c==21 else range(1)
    for k in st:
        a,b=symbol(c,k);image=smooth(np.maximum(a,b),.045*side/2); labels.append(c)
        for dx,dy in [(0,0),(-.055,0),(.055,0),(0,-.075),(0,.075)]:
            for sx,sy in [(1,1),(.8,.9),(1.17,.95),(1.08,1.18)]:
                xx=(x-dx)/sx; yy=(y-dy-1/3)/sy+1/3
                lay=[]
                for ox,oy in [(0,0),(-.085,0),(.085,0),(0,-.08),(0,.08)]:lay.append(sample(image,xx-ox,yy-oy))
                layers.append(lay)
lay=np.array(layers,np.float32);K['partlabels']=np.array(labels,np.int32);K['partpos']=np.flatnonzero(np.r_[True,np.diff(labels)!=0]).astype(np.int32)
for q,mask in enumerate([y==y,y<.19,y>.16,(x<.035)&(y>.10),(x>-.035)&(y>.10)]):
    K['partidx'+str(q)]=np.flatnonzero(mask).astype(np.int32);v=lay[:,:,mask].copy();v-=v.mean(axis=2,keepdims=True);v/=np.sqrt((v*v).sum(2,keepdims=True))+.00001;K['parts'+str(q)]=np.rint(v*120).astype(np.int8)

# Whole-face renderer. The comparison channels are green reflectance and two
# color opponents. All values specify a synthetic palette, not observations.
q=(np.arange(34)+.5)/34*2-1; yy,xx=np.meshgrid(q,q,indexing='ij');blueids=set(range(33,41));grayids={6,32,41,42}
for i,cl in enumerate(classes):
    e=frac(families[i],xx,yy);mask=e<.982;x=xx[mask];y=yy[mask];K['facegrid'+str(i)]=np.array([x,y],np.float32)
    starts=[];ws=[]
    for c in cl:
        starts.append(len(ws))
        st=range(7) if c==31 else range(4) if c in (23,27,28,29) else range(3) if c in (0,1,2,3,4,5,6,7,8,10,16,21,32,41,42) else range(1)
        for k in st:
            a,r=symbol(c,k);de=frac(families[i],X,Y);nil=X*0
            for inner in ((.65,.74) if i in (1,2) else (.72,.80) if i==0 else (.69,)):
                border=de>inner
                green=1+nil;red=nil.copy();blue=nil.copy()
                if i in (0,1,2):
                    green[border]=.20;red[border]=1
                    green=np.minimum(green,1-.94*a);green=np.minimum(green,1-.78*r);red=np.maximum(red,r)
                    if c==26:
                        for tx,ty,v,hue in [(0,-.16,.21,1),(0,.105,.72,.23),(0,.38,.46,-.6)]:
                            m=ellipse(tx,ty,.103,.104);green[m]=v;red[m]=hue
                elif i==3:green[:]=.75;green[border]=1;blue[~border]=-1
                elif i==4:green[:]=.19;green+=.81*a;blue=1-a
                elif i==5:
                    g1=number('80',k) if c==6 else np.maximum(*symbol(9 if c==41 else 10,k)) if c in (41,42) else nil
                    green=1-np.maximum(symbol(32,k)[0],g1*.55)*.92
                elif i in (6,7):green=.19+.81*a;red=1-a
                if i==6:green[de>.956]=.95;red[de>.956]=0
                sigma=.047*side/2;gg=smooth(green,sigma);rr=smooth(red,sigma) if red.any() else red;bb=smooth(blue,sigma) if blue.any() else blue
                for dx in (-.045,.045):
                    for dy in (-.045,.045):
                        for scale in (.96,1.04):
                            posx=x/scale-dx;posy=y/scale-dy
                            cha=[]
                            for ar in [gg,rr,bb]:
                                p=sample(ar,posx,posy);p=p-p.mean();p/=np.sqrt((p*p).sum())+1e-6;cha.append(p)
                            ws.append(np.concatenate(cha))
    ws=np.array(ws,np.float32);ss=np.max(abs(ws),axis=1)+1e-9;K['whole'+str(i)]=np.rint(ws/ss[:,None]*127).astype(np.int8);K['wholeamp'+str(i)]=ss/127;K['wholestart'+str(i)]=np.array(starts,np.int32)

# Deployment code, emitted verbatim by a clean rebuild.
SOLUTION = r'''import numpy as np

bank = CONSTANTS
outer = [bank['outer'+str(i)] for i in range(8)]
land = [bank['land'+str(i)] for i in range(8)]
grid = [bank['grid'+str(i)] for i in range(8)]
templates = {}
for family in (0,1,4,5,6,7):
    templates[family]=(bank['tpl'+str(family)].astype(np.float32)*bank['amp'+str(family)][:,None]).T.copy()

parts=[bank['parts'+str(i)].reshape(-1,len(bank['partidx'+str(i)])).T.astype(np.float32).copy()/120 for i in range(5)]
wholefilters=[(bank['whole'+str(i)].astype(np.float32)*bank['wholeamp'+str(i)][:,None]).T.copy() for i in range(8)]
edgefilters={}
for family in (0,1,5):edgefilters[family]=(bank['edtpl'+str(family)].astype(np.float32)*bank['edamp'+str(family)][:,None]).T.copy()
ocrbank=(bank['ocr'].astype(np.float32)*bank['ocramp'][:,None]).T.copy()

# All matrix products use the same row blocking, including for singleton
# calls. This fixes the arithmetic path independently of the caller's batch.
def dots(a,b):
    n=len(a);z=np.zeros((32,a.shape[1]),np.float32);z[:n]=a
    return (z@b)[:n]

def digitread(im,rx,ry,ox,oy,sh):
    x,y=bank['ocrgrid']; pix=interp(im,15.5+ox+rx*(x+sh*y),15.5+oy+ry*y)
    so=np.sort(pix,axis=1);lo=so[:,int(.045*len(x)),None];hi=so[:,int(.88*len(x)),None]
    z=unit(-np.clip((pix-lo)/(hi-lo+.004),-.3,1.25));co=dots(z,ocrbank)
    n=len(z);pos=bank['ocrpos'];step=int(pos[1]);co=co.reshape(n,8,step);k=co.argmax(axis=2);j=k+pos
    ref=ocrbank.T[j];p=bank['ocrpose'][j];px=(x[None,None,:]-p[:,:,1,None]-p[:,:,2,None]*y)/p[:,:,0,None]
    z=z[:,None,:];m=px<p[:,:,3,None]
    def region(mask):
        count=mask.sum(axis=2,keepdims=True);z1=(z-(mask*z).sum(axis=2,keepdims=True)/(count+1e-6))*mask
        r1=(ref-(mask*ref).sum(axis=2,keepdims=True)/(count+1e-6))*mask
        return (z1*r1).sum(axis=2)/(np.sqrt((z1*z1).sum(2)*(r1*r1).sum(2))+.000001)
    first=region(m);lead=region(px<p[:,:,4,None]);last=region(~m)
    first[:,6:]=.68*first[:,6:]+.32*lead[:,6:]
    score=.48*co.max(axis=2)+.40*first+.12*last
    # Test the putative leading 1 relative to the intensity just outside the
    # string. With normalized correlation alone, a weak vertical noise edge
    # or a clearly printed narrow 1 can both have unit agreement.
    for jn in (6,7):
        region1=px[:,jn]<p[:,jn,4,None];rs=ref[:,jn];zs=z[:,0];pk=np.max(np.where(region1,rs,-10),axis=1)
        ink=region1&(rs>pk[:,None]*.40);paper=region1&(rs<0)
        dif=(zs*ink).sum(1)/(ink.sum(1)+.01)-(zs*paper).sum(1)/(paper.sum(1)+.01)
        score[:,jn]+=.095*np.clip(dif*np.sqrt(len(x))-.68,-1,1.25)
    return score

# Sample a batch of images at floating-point coordinates, independently.
def interp(a,x,y):
    x=np.clip(x,0,30.9999); y=np.clip(y,0,30.9999)
    ix=x.astype(np.int32);iy=y.astype(np.int32);fx=(x-ix).astype(np.float32);fy=(y-iy).astype(np.float32)
    b=np.arange(len(a)).reshape((len(a),)+(1,)*(ix.ndim-1)); ix=iy*32+ix
    return (a[b,ix]*(1-fx)+a[b,ix+1]*fx)*(1-fy)+(a[b,ix+32]*(1-fx)+a[b,ix+33]*fx)*fy

def flat(a):return a.reshape(len(a),-1)
def pool(a):return a.reshape(-1,16,2,16,2).mean(axis=(2,4)).reshape(-1,256)
def unit(a):
    a=a-a.mean(axis=1,keepdims=True)
    return a/(np.sqrt((a*a).sum(axis=1,keepdims=True))+.00001)

def spots(a,xx,yy):return interp(a,xx,yy).mean(axis=-1)

def familyfields(f,r,b,y,lum):
    if f==3:return y,lum,y
    if f==4:return b,lum,b
    if f==5:return np.maximum(np.maximum(r,b),y),lum,np.maximum(r,b)
    return r,lum,r

def poseval(f,h,l):
    # registered region order: band, field, immediate exterior.
    a,b,c=h[:,:,0],h[:,:,1],h[:,:,2];u,v,w=l[:,:,0],l[:,:,1],l[:,:,2]
    if f in (0,1,2):return 1.43*a-.60*b-.43*c+.91*(v-u)
    if f==3:return 1.32*b-.66*a-.25*c+.64*(u-.60*v-.40*w)
    if f==4:return 1.65*a+.04*b-.85*c+.12*(v-u)
    if f==5:return -.66*b+.93*(v-u)+.40*(v-w)
    return 1.57*a-.53*b-.48*c+.78*(v-u)

steps=np.array([(ds,dt,dx,dy,sh) for ds,dt in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)) for dx,dy in ((0,0),(-.85,0),(.85,0),(0,-.85),(0,.85)) for sh in (0,)],np.float32)
stepsfine=np.array([(0,0,dx,dy,sh) for dx in (-.48,.0,.48) for dy in (-.48,.0,.48) for sh in (-.12,0,.12)],np.float32)

def refine(f,pose,h,l,delta):
    opts=pose[:,None,:]+delta[None,:,:];g=land[f];rx,ry,ox,oy,sh=[opts[:,:,k,None,None] for k in range(5)]
    x=g[None,None,:,0,:]; y=g[None,None,:,1,:];xx=15.5+ox+rx*(x+sh*y);yy=15.5+oy+ry*y
    v=poseval(f,spots(h,xx,yy),spots(l,xx,yy));v-=.007*(abs(opts[:,:,2])+abs(opts[:,:,3]))+.10*abs(opts[:,:,4])
    best=v.argmax(axis=1);rows=np.arange(len(v));return opts[rows,best],v[rows,best]

def mainchunk(a):
    n=len(a); r,g,b=a[:,0],a[:,1],a[:,2]
    # Stretch luminance for this image only. Red, yellow and blue measures
    # combine a relative channel difference with an absolute saturation floor.
    gray=.24*r+.52*g+.24*b
    t=np.sort(gray.reshape(n,-1),axis=1);lo=t[:,65,None,None];hi=t[:,940,None,None];span=np.maximum(hi-lo,.047)
    lum=np.clip((gray-lo)/span,-.22,1.28)
    def color(p,q):return np.clip((p-q)/(.025+.53*p+.47*q),-.14,1.15)
    red=color(r,.75*g+.25*b);blue=color(b,.70*r+.30*g);yellow=color(.58*r+.42*g,b)-.46*np.maximum(red,0)
    # Use both global and sign-centered exposure: a bright sky outside a
    # deeply shadowed face should not flatten all of the sign's contrast.
    mid=np.sort(gray[:,5:27,5:27].reshape(n,-1),axis=1);floor=mid[:,15,None,None];ceil=mid[:,-20,None,None]
    lum=.4*lum+.6*np.clip((gray-floor)/np.maximum(ceil-floor,.034),-.25,1.25)
    l=flat(lum); R=flat(red); B=flat(blue); Y=flat(yellow); po=bank['poses'];val=[]; poses=[]
    for f in range(8):
        h,light,ch=familyfields(f,red,blue,yellow,lum)
        feat=np.concatenate([pool(h),pool(light),pool(ch)],axis=1)
        s=dots(feat,outer[f].T)-.008*(abs(po[:,2])+abs(po[:,3]))
        j=s.argmax(axis=1); val.append(s[np.arange(n),j]);poses.append(po[j])
    coarse=np.stack(val,axis=1)
    # A neutral circular end sign needs a diagonal, not merely neutral color.
    # Compare its evidence to chromatic faces, keeping multiple hypotheses.
    result=np.full((n,43),-20,np.float32)
    for f in range(8):
        rows=np.flatnonzero(coarse[:,f]>coarse.max(axis=1)-.63)
        if not len(rows):continue
        h=familyfields(f,R,B,Y,l)[0][rows];light=l[rows];p=poses[f][rows]
        p,s=refine(f,p,h,light,steps);p,s=refine(f,p,h,light,stepsfine)
        if f in (0,1,2,3):
            # Test agreement around the entire perimeter. A ring and a
            # triangle can both hit two strong red patches; the third side
            # and the clear field just inside the boundary distinguish them.
            gx,gy=land[f][0];cy=1/3 if f==1 else -1/3 if f==2 else 0
            levels=[];chs=[]
            for scale in (1,.68):
                xx=gx*scale;yy=cy+(gy-cy)*scale
                ax=15.5+p[:,2,None]+p[:,0,None]*(xx+p[:,4,None]*yy);ay=15.5+p[:,3,None]+p[:,1,None]*yy
                levels.append(interp(light,ax,ay));chs.append(interp(h,ax,ay))
            v=(chs[0]-chs[1])+.72*(levels[1]-levels[0]);v=v if f!=3 else -v
            penalty=np.sort(v,axis=1)[:,int(.25*v.shape[1])];s+=.29*np.clip(penalty,-.75,.8)*np.clip(chs[0].mean(1)*2 if f!=3 else chs[1].mean(1)*2,0,1)
        # Re-evaluate polarity and graphics using only central face samples.
        x,y=grid[f];rx,ry,ox,oy,sh=[p[:,k,None] for k in range(5)];xx=15.5+ox+rx*(x+sh*y); yy=15.5+oy+ry*y
        # Independent whole-face check, using the entire field and its trim.
        fx,fy=bank['facegrid'+str(f)];ax=15.5+ox+rx*(fx+sh*fy);ay=15.5+oy+ry*fy
        green=interp(flat(.08*r+.78*g+.14*b)[rows],ax,ay);gs=np.sort(green,axis=1)
        green=unit(np.clip(green,gs[:,int(.035*len(fx)),None]-.02,gs[:,int(.96*len(fx)),None]+.02))
        rc=interp(R[rows],ax,ay);bc=interp((-Y if f==3 else B)[rows],ax,ay)
        ra=np.clip(rc.std(1,keepdims=True)*6.5,0,1);ba=np.clip(bc.std(1,keepdims=True)*6.5,0,1)
        if f==3 or f==4:ra=ra*0
        if f!=3 and f!=4:ba=ba*0
        whole=dots(np.concatenate((green*.73,unit(rc)*ra*.27,unit(bc)*ba*.27),axis=1),wholefilters[f]);whole=np.maximum.reduceat(whole,bank['wholestart'+str(f)],axis=1)
        if f==3:
            # Yellow is defined by both red and green exceeding blue. On a
            # dark yellow diamond, green alone may hardly differ from the
            # border; the blue channel supplies the complementary edge.
            shape=abs(fx)+abs(fy);inside=shape<.58;trim=(shape>.75)&(shape<.92)
            bc0=interp(flat(b)[rows],ax,ay);patch=bc0[:,trim].mean(1)-bc0[:,inside].mean(1)
            facecol=interp(Y[rows],ax,ay);diff=facecol[:,inside].mean(1)-facecol[:,trim].mean(1)
            s+=.80*np.clip(diff,0,.8)+.45*np.clip(patch/(.035+bc0.mean(1)),0,.65)
        if f in (2,3):
            # Yield/priority have an empty field; reward uniformity only if
            # present rather than giving every inverted triangle a fixed bonus.
            inside=(np.maximum(1.5*(-y-1/3),.75*(2*abs(x)+y+1/3))<.58) if f==2 else (abs(x)+abs(y)<.56)
            face=interp(light,xx[:,inside],yy[:,inside]);ss=np.sort(face,axis=1)
            scatter=ss[:,int(len(ss[0])*.91)]-ss[:,int(len(ss[0])*.08)]
            result[rows,13 if f==2 else 12]=.83*s+.18-.31*np.minimum(scatter,1)+.96*(whole[:,0]-.39);continue
        # Green emphasizes the black and red ink against white; on blue signs
        # the red channel makes the white arrow stand out from the blue field.
        if f==4 or f in (6,7):
            im=flat((.67*r+.27*g+.06*b) if f==4 else (.11*r+.52*g+.37*b))[rows];sign=1
        else:im=flat(.25*r+.53*g+.22*b)[rows];sign=-1
        pix=interp(im,xx,yy); sort=np.sort(pix,axis=1);low=sort[:,int(.055*len(x))];high=sort[:,int(.91*len(x))];rng=np.maximum(high-low,.012)
        pic=np.clip((pix-low[:,None])/rng[:,None],-.22,1.18)
        z=unit(sign*pic); corr=dots(z,templates[f])
        if f in (0,1,5):
            edge=bank['edge'+str(f)];d=z[:,edge[0]]-z[:,edge[1]];d/=np.sqrt((d*d).sum(1,keepdims=True))+.00001
            ec=dots(d,edgefilters[f]);corr=.74*corr+.26*ec
        if f==1:
            # Approximate optical blur from the inner transition of the red
            # border, which should have the same optics as the graphic. A
            # washed-out image may lose fine legs but a sharp one should not.
            gx,gy=land[1][0];fy=1/3;rad=np.maximum((gy-fy)*1.5,(2*abs(gx)-gy+fy)*.75)
            cuts=[]
            for level in (.54,.60,.66,.72,.78,.84,.90):
                tx=gx*level/rad;ty=fy+(gy-fy)*level/rad
                ax=15.5+ox+rx*(tx+sh*ty);ay=15.5+oy+ry*ty
                cuts.append((interp(h,ax,ay)-.7*interp(light,ax,ay)).mean(1))
            cuts=np.stack(cuts,axis=1);am=cuts.max(1)-cuts.min(1);steep=np.diff(cuts,axis=1).max(1)
            width=np.clip(.025*(am+.025)/(steep+.025),.030,.11)
            corr-=.075*np.minimum(abs(np.log(bank['blur1'][None,:]/width[:,None])),1.1)
        if f in (0,1):
            col=interp(R[rows],xx,yy);colz=unit(col);ids=bank['coloridx'+str(f)]
            colscore=dots(colz,bank['colortpl'+str(f)].T)
            strength=np.clip((col.max(1)-col.min(1)-.13)*1.1,0,1)
            if f==0:corr[:,ids]+=.29*strength[:,None]*np.maximum(colscore,0)
            else:
                # Traffic lights should have a colored stack, not just a dark
                # vertical stroke. Compare the actual channel opponency at
                # the predicted red aspect with the neutral inner field.
                centered=col-col.mean(axis=1,keepdims=True)
                redsignal=np.maximum(0,dots(centered,bank['colortpl1'].T)/3)
                corr[:,ids]+=.32*np.minimum(redsignal,.8)-.045
                # Red/yellow/green ordered on a white field, separate from
                # the red rim. Measure chroma relative to nearby white.
                signals=[]
                for tx,ty in ((0,-.18),(0,.09),(0,.37)):
                    responses=[]
                    for dx in (-.055,0,.055):
                        ax=15.5+ox+rx*(tx+dx+sh*ty);ay=15.5+oy+ry*ty
                        left=ax-rx*.18;right=ax+rx*.18
                        if ty<0:ch=flat(np.maximum(0,np.minimum(r-g,r-b)/(.03+gray)))[rows]
                        elif ty<.2:ch=flat(np.maximum(0,np.minimum(r-b,g-b)/(.03+gray)))[rows]
                        else:ch=flat(np.maximum(0,np.minimum(g-r,g-b)/(.03+gray)))[rows]
                        responses.append(interp(ch,ax,ay)-.5*(interp(ch,left,ay)+interp(ch,right,ay)))
                    signals.append(np.maximum.reduce(responses)[:,0])
                aspects=np.minimum(np.maximum(signals,0),.7)
                lights=np.sort(aspects,axis=0);corr[:,ids]+=(.22*lights[2]+.82*lights[1]+.38*lights[0])[:,None]
        ss=np.maximum.reduceat(corr,bank['start'+str(f)],axis=1)
        if f==0:
            ss[:,:8]=.35*ss[:,:8]+.65*digitread(im,rx,ry,ox,oy,sh)
            # A blank interior must lack both contrast and coherent strokes.
            # Test relative to the field's OWN brightness, not background sky.
            rel=(high-low)/(high+.02);ss[:,10]=np.clip(.91-1.45*rel,-.3,.93)
            line=np.maximum(ss[:,:8].max(1),ss[:,11]);ss[:,10]-=.90*np.maximum(line-.36,0)
            # A coherent numeral in a dark photo can have very little absolute
            # contrast. In that situation a blank-field hypothesis should not
            # defeat the independently recognized digit string.
            med=(pix[:,abs(x)<.52]).mean(axis=1)
            localcontrast=(med-low)/(med+.005)
            ss[:,10]-=.72*np.maximum(line-.25,0)+.42*np.maximum(localcontrast-.12,0)
        if f==5:
            # Suppress the dominant common diagonal and explicitly inspect
            # the two remaining crescents for a speed or passing symbol.
            use=abs(x+y)>.35;zz=unit(-pic[:,use]);tt=templates[5][use].T.copy();tt=unit(tt).T.copy()
            cres=np.maximum.reduceat(dots(zz,tt),bank['start5'],axis=1)
            wing=pix[:,use];spread=np.sort(wing,axis=1);contrast=(spread[:,int(.84*len(spread[0]))]-spread[:,int(.12*len(spread[0]))])/(rng+.015)
            ss[:,[0,2,3]]+=.27*(cres[:,[0,2,3]]-.45)-.28*np.maximum(.45-contrast[:,None],0)
            ss[:,1]+=.38*np.maximum(.50-contrast,0)
        if f==1:
            pos=bank['start1'];idx=[]
            for j in range(len(pos)):
                end=pos[j+1] if j+1<len(pos) else corr.shape[1]
                idx.append(pos[j]+corr[:,pos[j]:end].argmax(1))
            ref=templates[1].T[np.stack(idx,axis=1)]
            # The feet of a pedestrian, bicycle wheels and the bent animal's
            # legs are small relative to the mass of ink at the torso. Compare
            # the lower half and side parts on their own contrast scales.
            for part in (y>.275,y<.18,x<-.01,x>.01):
                target=unit(-pic[:,part]);refpart=ref[:,:,part];refpart=refpart-refpart.mean(2,keepdims=True)
                mag=np.sqrt((refpart*refpart).sum(2));match=(refpart*target[:,None,:]).sum(2)/(mag+.00001)
                ss+=.13*(match-.40)*np.clip((mag-.06)*4,0,1)
            # Part displacements are coupled through a common face pose.
            pc=0
            for k in range(5):
                pp=unit(-pic[:,bank['partidx'+str(k)]]);part=dots(pp,parts[k]).reshape(len(rows),-1,5)
                pscore=part[:,:,0] if k==0 else .25*part[:,:,0]+.75*part.max(axis=2)
                pc+=pscore*(.40 if k==0 else .15)
            pc=pc.reshape(len(rows),-1,20).max(axis=2);pc=np.maximum.reduceat(pc,bank['partpos'],axis=1)
            ss+=.62*(pc-.43)

        # Alignment of the inner symbol as well as the rim should contribute to
        # the face decision. Correlations also resolve lookalike silhouettes.
        weight=.57 if f in (0,1,5) else .69
        if f==5:s-=.01
        if f in (6,7):
            # A red ring is also red above and below its center. Check that
            # the field behind a stop legend or the crossbar is filled in.
            inner=(abs(y)>.30)&(abs(y)<.65)&(abs(x)<.35)
            cap=interp(h,xx[:,inner],yy[:,inner]).mean(1)
            sides=(abs(y)<.14)&(abs(x)>.60)
            rim=interp(h,xx[:,sides],yy[:,sides]).mean(1)
            s-=.7*np.maximum(rim-cap,0)
        scores=.83*s[:,None]+weight*(ss-.43)+.96*(whole-.39)
        if f==5:scores+=.15 # its neutral band supplies no chromatic correlation
        result[rows[:,None],bank['class'+str(f)][None,:]]=scores
    return result.argmax(axis=1)

def predict(x):
    out=np.empty(len(x),np.int32)
    for i in range(0,len(x),32):out[i:i+32]=mainchunk(x[i:i+32])
    return out
'''
(out/'solution.py').write_text(SOLUTION)
np.savez(out/'constants.npz',**K)
