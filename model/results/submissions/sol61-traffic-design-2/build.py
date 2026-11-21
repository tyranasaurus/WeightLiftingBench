"""Data-free traffic-sign network. All geometry is idealized, not image-derived.
No data files are opened. Only NumPy and the Python standard library are used.
"""
import json
import sys
from pathlib import Path
import numpy as np

F = np.float32
layers = []
weights = {}

def layer(name, op, source, **kw):
    layers.append(dict(name=name, op=op, inputs=[source] if isinstance(source, str) else source, **kw))
    return name

def affine(name, source, w, b=None, conv=False, stride=1, padding=0):
    w = np.asarray(w, dtype=F)
    if b is None:
        b = np.zeros(len(w), F)
    weights[name+'.W'] = w
    weights[name+'.b'] = np.asarray(b, dtype=F)
    if conv:
        return layer(name, 'conv', source, out=len(w), kernel=list(w.shape[-2:]), stride=stride, padding=padding)
    return layer(name, 'dense', source, out=len(w))

# Geometry primitives. Shapes are drawn on a fixed quadrature grid in ideal
# sign coordinates; there is no raster font and no photograph in this program.
Q = 2
v = (np.arange(32*Q, dtype=F)+.5)/Q
Y, X = np.meshgrid(v, v, indexing='ij')

def disk(x, y, cx, cy, r):
    return (x-cx)**2+(y-cy)**2 <= r*r

def polygon(x, y, pts):
    inside = np.zeros(x.shape, bool)
    for a, b in zip(pts, pts[1:]+pts[:1]):
        if a[1] == b[1]:
            continue
        inside ^= ((a[1]>y) != (b[1]>y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
    return inside

def line(x, y, pts, width):
    ans = np.zeros(x.shape, bool)
    for (ax,ay),(bx,by) in zip(pts[:-1],pts[1:]):
        d = (bx-ax)**2+(by-ay)**2
        t = np.clip(((x-ax)*(bx-ax)+(y-ay)*(by-ay))/max(d,1.e-10),0,1)
        ans |= (x-ax-t*(bx-ax))**2+(y-ay-t*(by-ay))**2 <= (width/2)**2
    return ans

def ellipse(x,y,cx,cy,rx,ry,width=0):
    r = ((x-cx)/rx)**2+((y-cy)/ry)**2
    if width == 0:
        return r<=1
    ri = ((x-cx)/max(rx-width, .001))**2+((y-cy)/max(ry-width,.001))**2
    return (r<=1)&(ri>=1)

def patharc(cx,cy,rx,ry,a,b,n=24):
    t=np.linspace(a,b,n)
    return [(cx+rx*np.cos(z),cy+ry*np.sin(z)) for z in t]

# The numerals are constructed from elementary arcs and line segments in a
# unit character cell. Two conventional rounded/straight stroke variants.
def digit(x,y,d,style=0):
    # Work in physical, not stretched cell coordinates: stroke thickness
    # remains the same for vertical and horizontal parts of a narrow glyph.
    x=x*.48
    w=[.090,.115,.075][style%3]
    def stroke(pts):
        return line(x,y,[(a*.48,b) for a,b in pts],w)
    def oval(cx,cy,rx,ry):
        return ellipse(x,y,cx*.48,cy,rx*.48,ry,w)
    if d=='0':
        return oval(.5,.5,.37,.48)
    if d=='1':
        pts=[(.20,.23),(.49,.035),(.49,.96)]
        z=stroke(pts)
        if style==1:z |= stroke([(.22,.96),(.76,.96)])
        return z
    if d=='2':
        pts=patharc(.49,.265,.35,.24,np.pi,2*np.pi)+[(.83,.38),(.18,.95),(.86,.95)]
        return stroke(pts)
    if d=='3':
        return stroke(patharc(.43,.265,.37,.24,-2.65,np.pi/2))+stroke(patharc(.43,.735,.38,.245,-np.pi/2,2.65))
    if d=='5':
        pts=[(.84,.06),(.22,.06),(.18,.47),(.48,.44)]+patharc(.47,.704,.36,.27,-np.pi/2,2.65)
        return stroke(pts)
    if d=='6':
        return oval(.50,.70,.35,.28)|stroke(patharc(.51,.49,.36,.46,-.8,-np.pi)+[(.15,.68)])
    if d=='7':
        return stroke([(.13,.055),(.86,.055),(.39,.96)])
    if d=='8':
        return oval(.5,.27,.32,.255)|oval(.5,.735,.36,.265)
    return np.zeros(x.shape,bool)

def numeral(x,y,text,style=0):
    width = 1.32 if len(text)==3 else 1.28
    height = 1.08
    if style>=3:
        width*=.88;height*=1.12
    pitch=width/len(text)
    ink=np.zeros(x.shape,bool)
    for i,d in enumerate(text):
        u=(x+width/2-i*pitch)/(pitch*.94)
        v=(y+height/2)/height
        ink |= digit(u,v,d,style)
    return ink

def arrow(x,y,pts,width=.16,head=.34):
    z=line(x,y,pts,width)
    a,b=pts[-2],pts[-1]
    dx,dy=b[0]-a[0],b[1]-a[1]
    l=np.hypot(dx,dy); dx/=l; dy/=l
    pts=[b,(b[0]-head*dx-head*.82*dy,b[1]-head*dy+head*.82*dx),
         (b[0]-head*dx+head*.82*dy,b[1]-head*dy-head*.82*dx)]
    return z|polygon(x,y,pts)

def car(x,y,cx=0,cy=0,s=1):
    u=(x-cx)/s; v=(y-cy)/s
    z=polygon(u,v,[(-.24,-.28),(.24,-.28),(.32,-.02),(.34,.30),(-.34,.30),(-.32,-.02)])
    z &= ~polygon(u,v,[(-.16,-.20),(.16,-.20),(.21,-.035),(-.21,-.035)])
    z |= line(u,v,[(-.23,.26),(-.23,.39)],.13)|line(u,v,[(.23,.26),(.23,.39)],.13)
    z &= ~(disk(u,v,-.21,.11,.055)|disk(u,v,.21,.11,.055))
    return z

def truck(x,y,cx=0,cy=0,s=1,front=False):
    u=(x-cx)/s;v=(y-cy)/s
    if front:
        z=(abs(u)<.32)&(v>-.42)&(v<.35)
        z &= ~((abs(u)<.23)&(v>-.29)&(v<-.05))
        z |= line(u,v,[(-.24,.30),(-.24,.44)],.14)|line(u,v,[(.24,.30),(.24,.44)],.14)
        return z
    z=polygon(u,v,[(-.62,-.36),(.13,-.36),(.13,-.19),(.41,-.19),(.62,.10),(.62,.34),(-.62,.34)])
    z &= ~polygon(u,v,[(.23,-.12),(.37,-.12),(.49,.07),(.23,.07)])
    z |= disk(u,v,-.39,.34,.13)|disk(u,v,.39,.34,.13)
    return z

def person(x,y,cx=0,cy=0,s=1,running=False):
    u=(x-cx)/s;v=(y-cy)/s
    z=disk(u,v,0,-.44,.115)
    z |= line(u,v,[(0,-.26),(-.07,.05),(.09,.20)],.15)
    z |= line(u,v,[(-.03,-.16),(-.24,.06),(-.38,.07)],.08)
    z |= line(u,v,[(-.01,-.15),(.22,-.02),(.30,-.12)],.08)
    z |= line(u,v,[(.02,.12),(-.23,.46),(-.40,.46)],.10)
    z |= line(u,v,[(.04,.14),(.19,.42),(.36,.45)],.10)
    return z

SPEED={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
BLUE=set(range(33,41))
TRI={11,*range(18,32)}

def sign_masks(k,x,y,style=0):
    rr=x*x+y*y
    disc=rr<=1
    inner=rr<=.77**2
    ring=disc & ~inner
    red=np.zeros(x.shape,bool);blue=red.copy();yellow=red.copy();ink=red.copy();white=red.copy()
    symbol=red.copy()
    if k in TRI:
        outer=polygon(x,y,[(0,-1.04),(-1.02,.79),(1.02,.79)])
        inside=polygon(x,y,[(0,-.72),(-.76,.64),(.76,.64)])
        red=outer&~inside;white=inside
        support=outer
        factor=[1.,.85,1.15][style%3]
        x=x/factor;y=(y-.10)/factor+.10
        if k==11:
            symbol=arrow(x,y,[(0,.54),(0,-.46)],.18,.22)|line(x,y,[(-.40,.25),(0,-.07),(.40,.25)],.10)
        elif k==18:
            symbol=polygon(x,y,[(-.085,-.43),(.085,-.43),(.06,.20),(-.06,.20)])|disk(x,y,0,.41,.09)
        elif k in (19,20):
            u=x if k==20 else -x
            pts=[(-.15,.54),(-.15,.07)]+patharc(.10,.07,.25,.28,np.pi,1.5*np.pi)+[(.38,-.21)]
            symbol=line(u,y,pts,.17)
        elif k==21:
            yy=np.linspace(.54,-.42,35)
            xx=.20*np.cos((yy-.54)/.96*2*np.pi)
            symbol=line(x,y,list(zip(xx,yy)),.19)
        elif k==22:
            pts=[(-.58,.51),(-.58,.43)]
            for cx in [-.28,.28]:
                pts += patharc(cx,.43,.28,.22,np.pi,2*np.pi)
            pts += [(.56,.51)]
            symbol=polygon(x,y,pts)
        elif k==23:
            # Front-view car and serpentine tyre marks.
            symbol=car(x,y,-.05,-.12,.78)
            symbol |= line(x,y,[(-.23,.14),(-.36,.28),(-.15,.40),(-.30,.53)],.065)
            symbol |= line(x,y,[(.17,.14),(.05,.28),(.27,.40),(.11,.53)],.065)
        elif k==24:
            symbol=line(x,y,[(-.34,.55),(-.34,-.39)],.13)|line(x,y,[(.40,.55),(.40,.22),(.13,-.20),(.13,-.39)],.13)
        elif k==25:
            symbol=disk(x,y,.05,-.31,.105)
            symbol |= line(x,y,[(.03,-.15),(-.14,.10),(-.17,.29)],.15)
            symbol |= line(x,y,[(-.15,.22),(-.39,.53)],.11)|line(x,y,[(-.13,.22),(.06,.51)],.11)
            symbol |= line(x,y,[(0,-.10),(.22,.08),(.30,.24)],.09)
            symbol |= line(x,y,[(-.24,-.05),(.37,.50)],.045)
            symbol |= polygon(x,y,[(.13,.54),(.38,.30),(.61,.54)])
        elif k==26:
            symbol=(abs(x)<.17)&(y>-.45)&(y<.56)
            symbol &= ~(disk(x,y,0,-.27,.070)|disk(x,y,0,.055,.070)|disk(x,y,0,.38,.070))
            red |= disk(x,y,0,-.27,.075)
            yellow |= disk(x,y,0,.055,.075)
        elif k==27:
            symbol=person(x,y,0,.10,.88)
            for yy in [.43,.57]:symbol |= line(x,y,[(-.48,yy),(.48,yy)],.035)
        elif k==28:
            symbol=person(x,y,-.23,.16,.78)|person(x,y,.28,.29,.54)
            symbol |= line(x,y,[(-.06,.15),(.19,.13)],.06)
        elif k==29:
            symbol=ellipse(x,y,-.33,.33,.23,.23,.045)|ellipse(x,y,.33,.33,.23,.23,.045)
            symbol |= line(x,y,[(-.33,.33),(-.13,-.03),(.10,.33),(-.33,.33)],.05)
            symbol |= line(x,y,[(.10,.33),(.26,-.12),(.33,.33)],.05)
            symbol |= line(x,y,[(.16,-.14),(.32,-.14)],.05)
            symbol |= disk(x,y,-.03,-.40,.09)|line(x,y,[(-.03,-.28),(-.17,-.08),(.12,-.04),(.25,-.13)],.09)
            symbol |= line(x,y,[(-.16,-.08),(.03,.10),(-.10,.26)],.08)
        elif k==30:
            for a in np.arange(6)*np.pi/3:
                dx,dy=np.cos(a),np.sin(a)
                symbol |= line(x,y,[(0,.12),(.48*dx,.12+.48*dy)],.065)
                for side in [-1,1]:
                    ex=.31*dx+.14*np.cos(a+side*np.pi/3)
                    ey=.12+.31*dy+.14*np.sin(a+side*np.pi/3)
                    symbol |= line(x,y,[(.31*dx,.12+.31*dy),(ex,ey)],.045)
        elif k==31:
            symbol=ellipse(x,y,0,.15,.31,.13)
            symbol |= line(x,y,[(.20,.14),(.37,-.15),(.42,-.24)],.13)
            symbol |= ellipse(x,y,.43,-.23,.15,.065)
            symbol |= line(x,y,[(.16,.21),(.31,.42),(.44,.51)],.06)
            symbol |= line(x,y,[(-.22,.18),(-.33,.42),(-.52,.48)],.065)
            symbol |= line(x,y,[(.04,.23),(-.02,.52)],.055)
            symbol |= line(x,y,[(.37,-.28),(.28,-.44),(.18,-.48)],.035)
            symbol |= line(x,y,[(.32,-.39),(.39,-.46)],.03)
        ink=symbol&inside
        white &= ~ink
    elif k in BLUE:
        blue=disc.copy();support=disc
        factor=[1.,.85,1.15][style%3]
        x=x/factor;y=y/factor
        if k==33:
            symbol=arrow(x,y,[(-.22,.65),(-.22,-.15),(-.08,-.35),(.60,-.35)],.17,.34)
        elif k==34:
            symbol=arrow(-x,y,[(-.22,.65),(-.22,-.15),(-.08,-.35),(.60,-.35)],.17,.34)
        elif k==35:
            symbol=arrow(x,y,[(0,.69),(0,-.69)],.17,.37)
        elif k in (36,37):
            u=x if k==36 else -x
            symbol=arrow(u,y,[(0,.69),(0,-.69)],.16,.34)
            symbol |= arrow(u,y,[(0,.18),(.22,-.03),(.68,-.03)],.16,.32)
        elif k in (38,39):
            u=x if k==38 else -x
            symbol=arrow(u,y,[(-.49,-.57),(.51,.53)],.19,.42)
        elif k==40:
            for a in [0,2*np.pi/3,4*np.pi/3]:
                pts=patharc(0,0,.48,.48,a,a-1.43,16)
                symbol |= arrow(x,y,pts,.13,.27)
        white=symbol&disc;blue &= ~white
    elif k==12:
        support=(abs(x)+abs(y)<1.20)
        yellow=abs(x)+abs(y)<.94
        white=support&~yellow
    elif k==13:
        support=polygon(x,y,[(-1.03,-.72),(1.03,-.72),(0,1.04)])
        white=polygon(x,y,[(-.76,-.56),(.76,-.56),(0,.78)])
        red=support&~white
    elif k==14:
        ang=np.arange(8)*np.pi/4+np.pi/8
        pts=[(np.cos(a),np.sin(a)) for a in ang]
        support=polygon(x,y,pts)
        red=support.copy()
        # Stroke-letter STOP from a conventional sans-serif construction.
        u=(x+.72)/.36; v=(y+.26)/.52
        symbol=line(u,v,[(.80,.07),(.30,.07),(.13,.23),(.24,.44),(.73,.56),(.86,.76),(.69,.94),(.12,.94)],.13)
        u=(x+.36)/.36
        symbol |= line(u,v,[(.08,.07),(.90,.07)],.13)|line(u,v,[(.49,.07),(.49,.94)],.13)
        u=x/.36
        symbol |= ellipse(u,v,.49,.50,.36,.46,.13)
        u=(x-.36)/.36
        symbol |= line(u,v,[(.16,.94),(.16,.07),(.61,.07),(.84,.22),(.84,.42),(.61,.54),(.16,.54)],.13)
        white=symbol&support
        # Thin inset white octagonal outline.
        in1=polygon(x/.92,y/.92,pts);in2=polygon(x/.86,y/.86,pts)
        white |= in1&~in2;red &= ~white
    elif k==17:
        support=disc;white=(abs(x)<.80)&(abs(y)<.145)&disc;red=disc&~white
    else:
        support=disc.copy()
        white=inner.copy()
        if k not in (6,32,41,42):red=ring.copy()
        if k not in SPEED:
            factor=[1.,.85,1.15][style%3]
            x=x/factor;y=y/factor
        if k in SPEED:
            symbol=numeral(x,y,SPEED[k],style)
        elif k==9:
            symbol=car(x,y,.34,0,.70)
            red |= car(x,y,-.35,0,.70)
        elif k==10:
            symbol=car(x,y,.38,.06,.62)
            red |= truck(x,y,-.29,-.02,.88,True)
        elif k==16:
            symbol=truck(x,y,0,0,1.02)
        elif k in (41,42):
            symbol=car(x,y,.34,.04,.66)
            symbol |= car(x,y,-.34,.04,.66) if k==41 else truck(x,y,-.29,0,.82,True)
        if k in (6,32,41,42):
            white=disc.copy()
            stripes=np.zeros(x.shape,bool)
            for off in [-.24,-.12,0,.12,.24]:
                stripes |= (abs(y+x-off)<.045)&(x*x+y*y<.96**2)
            if k==32:
                symbol=stripes
            else:
                symbol |= stripes
        ink=symbol&support
        white &= ~(ink|red)
    return support,white,ink,red,blue,yellow,symbol

def down(a):
    return a.astype(F).reshape(32,Q,32,Q).mean((1,3))

def contrast(pos,neg):
    a=down(pos);b=down(neg)
    sa=a.sum();sb=b.sum()
    if sa<.05 or sb<.05:return np.zeros((32,32),F)
    return a/sa-b/sb

def symbol_contrast(pos,neg):
    # Unit-energy, zero-DC correlation. Balancing by the energy (rather than
    # only by the two areas) makes sparse and dense symbols comparable.
    c=contrast(pos,neg)
    t=np.arange(-2,3,dtype=F);g=np.exp(-t*t/(2*.60**2));g/=g.sum()
    p=np.pad(c,((0,0),(2,2)))
    c=sum(g[i]*p[:,i:i+32] for i in range(5))
    p=np.pad(c,((2,2),(0,0)))
    c=sum(g[i]*p[i:i+32] for i in range(5))
    area=down(pos|neg).sum()
    return c/max(float(np.sqrt((c*c).sum()*area)),1.e-8)

# Feature extraction: five invariant, bounded opponent channels. Gray and its
# local high-pass separate printed ink from illumination. RGB opponents keep
# the red border, blue disc, and yellow priority diamond distinct.
# Saturated colour requires BOTH opponent inequalities; a red surface must
# not count as yellow merely because its blue value is small.
mat=np.array([[.299,.587,.114],[1,-1.10,0],[-1.10,0,1],[1,0,-1.15]],F)
affine('colour_base','image',mat[:,:,None,None],conv=True)
delta=np.array([[0,-1.10,1.10],[-1.10,1.05,0],[1,-1,0]],F)
affine('colour_delta','image',delta[:,:,None,None],conv=True)
layer('colour_delta_positive','relu','colour_delta')
join=np.zeros((4,3,1,1),F)
for i in range(3):join[i+1,i,0,0]=-1
affine('colour_minimum_correction','colour_delta_positive',join,conv=True)
layer('colour_minimum','add',['colour_base','colour_minimum_correction'])
layer('opponents','relu','colour_minimum')
layer('standard','norm','opponents')
# Gray high-pass: center minus a fixed Gaussian, reflected only by zero padding
# as specified by the network's convolution primitive.
v3=np.arange(-4,5,dtype=F)
g=np.exp(-v3*v3/(2*2.0**2));g/=g.sum()
k=-np.outer(g,g);k[4,4]+=1
hw=np.zeros((1,4,9,9),F);hw[0,0]=k
affine('local_contrast','opponents',hw,conv=True,padding=4)
layer('local_standard','norm','local_contrast')
# Two clipped branches supply feature planes, stacked by addition into the
# same five-channel array. clip(x,-1.5,2) = ReLU(x+1.5)-ReLU(x-2)-1.5.
def bound(source,n,prefix):
    w=np.zeros((2*n,n,1,1),F)
    for i in range(n):w[i,i,0,0]=w[n+i,i,0,0]=1
    affine(prefix+'_hinges',source,w,np.r_[np.full(n,1.5),np.full(n,-2)],True)
    layer(prefix+'_positive','relu',prefix+'_hinges')
    w=np.zeros((7,2*n,1,1),F)
    for i in range(n):
        j=i if n==4 else 4
        w[j,i,0,0]=1;w[j,n+i,0,0]=-1
    b=np.zeros(7,F);b[:4 if n==4 else 0]=-1.5
    if n==1:b[4]=-1.5
    affine(prefix+'_bounded',prefix+'_positive',w,b,True)
    return prefix+'_bounded'
a=bound('standard',4,'global')
b=bound('local_standard',1,'local')
affine('dark_hinges','local_standard',np.array([[[[-1]]],[[[-1]]]],F),[-.1,-1.1],True)
layer('dark_positive','relu','dark_hinges')
dw=np.zeros((1,2,1,1),F);dw[0,:,0,0]=[1,-1]
affine('dark_map','dark_positive',dw,conv=True)
affine('dark_pad','dark_map',np.ones((1,1,1,1),F),conv=True,padding=1)
layer('dark_tolerance','maxpool','dark_pad',size=[3,3],stride=[1,1])
layer('dark_tolerance_normal','norm','dark_tolerance')
dw=np.zeros((7,1,1,1),F);dw[5,0,0,0]=1
affine('darkness','dark_map',dw,conv=True)
dw=np.zeros((7,1,1,1),F);dw[6,0,0,0]=1
affine('tolerant_darkness','dark_tolerance_normal',dw,conv=True)
layer('features','add',[a,b,'darkness','tolerant_darkness'])

# Analytic matched filters. Variations are declared in advance, never selected
# by fitting. All classes receive the same center/size/aspect family.
centers=[(0,0),(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(-2,2),(2,-2),(2,2)]
radii=[9.0,11.0,13.0]
aspects=[.85,1.,1.16]
# One extra font form for the speed signs; two blue-symbol proportions.
models=[]
for k in range(43):
    rows=[];geometry=[];bs=[]
    styles=list(range(6)) if k in SPEED else [0,1,2]
    if k in (12,13,14,15,17):styles=[0]
    for style in styles:
        for radius in radii:
            for asp in aspects:
                for dx,dy in centers:
                    x=(X-16-dx)/(radius*asp)
                    y=(Y-16-dy)/radius
                    support,white,ink,red,blue,yellow,symbol=sign_masks(k,x,y,style)
                    out=np.zeros((7,32,32),F)
                    geom=np.zeros_like(out)
                    # Shape is a separate gate: a better border cannot
                    # compensate for a wrong symbol. Opponent gating also
                    # prevents pale cancellation discs matching blue discs.
                    outside=(~support)&(x*x+y*y<1.22**2)
                    if k in BLUE:
                        out[0]=.65*symbol_contrast(white,blue)
                        out[4]=.65*symbol_contrast(white,blue)
                        geom[2]=contrast(blue,white|outside)
                    elif k==14:
                        out[0]=.65*symbol_contrast(white,red)
                        out[4]=.65*symbol_contrast(white,red)
                        geom[1]=contrast(red,white|outside)
                    elif k==12:
                        geom[3]=contrast(yellow,white|outside)
                    elif k in (13,15):
                        geom[1]=contrast(red,white)
                        # A blank interior is evidence, rather than a bonus
                        # that can dominate a correctly matched numeral.
                        blank=down(white)
                        out[5]=-blank/max(blank.sum(),1)
                    else:
                        region=support&(x*x+y*y<.74**2)
                        symbol_ink=ink&region
                        surround=region&~symbol_ink&~red
                        sc=symbol_contrast(surround,symbol_ink)
                        out[0]=.45*sc
                        out[4]=.45*sc
                        # One-pixel max dilation supplies stroke tolerance;
                        # comparison of its unit-energy map still penalizes
                        # unexpected strokes in the symbol's white spaces.
                        out[6]=-.40*sc
                        if k in (6,32,41,42):
                            rim=support&~(x*x+y*y<.80**2)
                            geom[0]=.25*contrast(white,outside)
                            a=down(rim);geom[1]=-a/max(a.sum(),1)
                        else:
                            geom[1]=contrast(red,white|ink)
                        if k==17:
                            out[0]=.65*symbol_contrast(white,red);out[4]=.65*symbol_contrast(white,red)
                    # All hypotheses at a pose use that same pose's border.
                    # Give a blank interior a low stroke-count likelihood,
                    # not an extra positive copy of the border likelihood.
                    base=0.
                    out[6]*=.25
                    if k not in (12,13,15):
                        out[0]*=1.30;out[4]*=1.30
                    if k in (13,15):
                        out[5]*=4.5
                    if k in (6,32,41,42):
                        base=-.28
                    out += .45*geom
                    rows.append(out.reshape(-1));geometry.append(geom.reshape(-1));bs.append(base)
    models.append((np.array(rows,F),np.array(geometry[:81],F),np.array(bs,F)))
# Each class is an independent, fixed template bank and a maximum reduction.
# Keeping branches separate avoids allocating an enormous block-diagonal
# readout. The interpreter's arrays remain small even for its batch of 500.
score_names=[]
for k,(w,g,bias) in enumerate(models):
    name='templates_%02d'%k
    affine(name,'features',w,bias)
    threshold=.55 if k not in (6,32,41,42) else -.10
    # Fold the negative identity into the geometry filters, then use ReLU
    # for a family-violation penalty. This needs no fitted readout.
    affine(name+'_geometry','features',-g,np.full(len(g),threshold,F))
    layer(name+'_violation','relu',name+'_geometry')
    penalty=np.zeros((len(w),len(g)),F)
    for i in range(len(w)):penalty[i,i%len(g)]=-2.5
    affine(name+'_penalty',name+'_violation',penalty)
    layer(name+'_gated','add',[name,name+'_penalty'])
    layer(name+'_view','reshape',name+'_gated',shape=[1,len(w),1])
    layer(name+'_best','maxpool',name+'_view',size=[len(w),1],stride=[1,1])
    out=np.zeros((43,1),F);out[k,0]=1
    score_names.append(affine('place_%02d'%k,name+'_best',out))
layer('scores','add',score_names)

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n')
np.savez(out/'weights.npz',**weights)
print('Wrote',len(layers),'layers and',sum(a.size for a in weights.values()),'float32 weights.')
