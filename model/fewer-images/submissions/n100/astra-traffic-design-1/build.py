"""Data-free construction of a part-based traffic sign recognizer. No font files or image assets."""
import sys, json, math
from pathlib import Path
import numpy as np

# The drawing plane uses units of one sign radius (x to the right, y down).
S=128
u=((np.arange(S)+.5)/S*2-1)*1.09
X,Y=np.meshgrid(u,u)

def poly(points):
    inside=np.zeros(X.shape,bool)
    x1,y1=points[-1]
    for x2,y2 in points:
        if y1!=y2:
            inside ^= ((Y>=min(y1,y2))&(Y<max(y1,y2))&(X < x1+(Y-y1)*(x2-x1)/(y2-y1)))
        x1,y1=x2,y2
    return inside

def ellipse(cx,cy,rx,ry): return ((X-cx)/rx)**2+((Y-cy)/ry)**2<=1

def stroke(points,width,closed=False):
    mask=np.zeros(X.shape,bool)
    for (a,b),(c,d) in zip(points, points[1:]+([points[0]] if closed else [])):
        dx,dy=c-a,d-b
        t=np.clip(((X-a)*dx+(Y-b)*dy)/max(dx*dx+dy*dy,1e-12),0,1)
        mask|=(X-a-t*dx)**2+(Y-b-t*dy)**2<=(.5*width)**2
    return mask

def rect(a,b,c,d): return (X>=a)&(X<=c)&(Y>=b)&(Y<=d)

def curve(start,*sections):
    pts=[start]
    for p,q,r in sections:
        a,b=pts[-1];c,d=p;e,f=q;g,h=r
        for t in np.linspace(0,1,9)[1:]:
            v=1-t; pts.append((v**3*a+3*t*v*v*c+3*t*t*v*e+t**3*g,v**3*b+3*t*v*v*d+3*t*t*v*f+t**3*h))
    return pts

# Single-line glyphs, with curves like the condensed sans lettering used on signs.
# Nominal glyph design box 0..1 in x and y (stroke extends outside the box).
def glyph(letter):
    if letter in '0O':
        return [curve((.50,.00),((.15,0),(.02,.13),(.02,.40)),((.02,.53),(.02,.60),(.02,.65)),((.02,.90),(.18,1),(.50,1)),((.84,1),(.98,.85),(.98,.62)),((.98,.53),(.98,.48),(.98,.38)),((.98,.12),(.81,0),(.50,0)))]
    if letter=='1': return [[(.06,.21),(.61,0),(.61,1)]]
    if letter=='2':return [curve((.03,.22),((.06,-.08),(.95,-.10),(.97,.22)),((1,.43),(.69,.52),(.54,.64)))+[(.015,.995),(.99,.995)]]
    if letter=='3':return [curve((.04,.14),((.26,-.07),(.93,-.05),(.94,.20)),((.97,.38),(.79,.47),(.42,.47)),((.74,.47),(1,.54),(.98,.76)),((.97,1.04),(.29,1.09),(.02,.87)))]
    if letter=='5':return [[(.95,0),(.10,0),(.07,.46)]+curve((.07,.46),((.33,.32),(.94,.34),(.96,.59)),((1,.69),(1,.79),(.93,.86)),((.77,1.07),(.19,1.07),(.01,.86)))]
    if letter=='6':return [curve((.80,-.005),((.58,.14),(.13,.43),(.06,.59)),((-.10,.89),(.16,1.01),(.47,1.01)),((.82,1.01),(.99,.90),(.96,.70)),((.95,.48),(.63,.43),(.44,.46)),((.20,.46),(.11,.54),(.05,.61)))]
    if letter=='7':return [[(0,.14),(0,0),(.99,0),(.32,1)]]
    if letter=='8':return [curve((.5,.48),((-.1,.48),(-.14,.01),(.48,0)),((1.16,-.02),(1.12,.47),(.5,.48)),((-.13,.49),(-.19,1.01),(.49,1)),((1.18,1),(1.15,.49),(.5,.48)))]
    if letter=='S':return [curve((.95,.15),((.56,-.17),(-.02,.01),(.03,.24)),((.03,.47),(.81,.48),(.92,.65)),((1.20,.99),(.48,1.17),(.02,.88)))]
    if letter=='T':return [[(0,0),(1,0)],[(.5,0),(.5,1)]]
    if letter=='P':return [[(.06,1),(.06,0),(.54,0)]+curve((.54,0),((1.10,-.02),(1.12,.49),(.54,.49)))+[(.06,.49)]]
    raise ValueError(letter)

class Sketch:
    def __init__(self): self.z=np.zeros_like(X)
    def put(self,mask,level=1.): self.z[mask]=level
    def line(self,points,width,level=1.): self.put(stroke(points,width),level)
    def polygon(self,points,level=1.): self.put(poly(points),level)
    def oval(self,x,y,rx,ry,level=1.): self.put(ellipse(x,y,rx,ry),level)
    def text(self,word,width,height,thick,cy=0):
        gap=.14 if len(word)==2 else .20
        units=[.70 if c=='1' else 1 for c in word]; gw=(width-thick)/(sum(units)+(len(word)-1)*gap*2)
        x=-width/2+thick/2
        for c,l in zip(word,units):
            for path in glyph(c):
                self.line([(x+xx*l*gw,cy+(yy-.5)*(height-thick)) for xx,yy in path],thick)
            x+=(l+gap*2)*gw

def car(d,x,y,w,h,level=1.):
    def xy(a,b):return (x+a*w,y+b*h)
    d.polygon([xy(a,b) for a,b in [(-.49,.32),(-.49,-.07),(-.39,-.16),(-.27,-.46),(-.18,-.5),(.25,-.5),(.40,-.16),(.49,-.06),(.49,.32)]],level)
    d.polygon([xy(a,b) for a,b in [(-.28,-.16),(-.19,-.39),(.20,-.39),(.29,-.16)]],0.)
    for pos in [-.33,.33]: d.polygon([xy(pos-.09,.24),xy(pos+.09,.24),xy(pos+.09,.5),xy(pos-.09,.5)],level)

def truck(d,x,y,w,h,frontleft=True):
    def xy(a,b):return (x+a*w*(1 if frontleft else -1),y+b*h)
    d.polygon([xy(-.20,-.48),xy(.50,-.48),xy(.50,.20),xy(-.20,.20)])
    d.polygon([xy(-.21,-.25),xy(-.35,-.25),xy(-.50,.03),xy(-.50,.29),xy(.46,.29),xy(.46,.17),xy(-.21,.17)])
    for a in [-.33,.30]:d.oval(*xy(a,.31),h*.17,h*.17)

def top_arrow(d,x,y,dx,dy,shaft=.17,head=.47,point=.32):
    length=math.hypot(dx,dy); e=(dx/length,dy/length); n=(-e[1],e[0])
    def tr(a,b):return(x+e[0]*a+n[0]*b,y+e[1]*a+n[1]*b)
    d.polygon([tr(a,b) for a,b in [(0,-shaft/2),(length-point,-shaft/2),(length-point,-head/2),(length,0),(length-point,head/2),(length-point,shaft/2),(0,shaft/2)]])

def symbol(k,variant):
    d=Sketch(); v=variant
    if k in [0,1,2,3,4,5,7,8,6]:
        s=['20','30','50','60','70','80','80','100','120'][k]
        w=(1.13 if len(s)==3 else 1.08)*[1.,.92,1.07][v%3]
        h=[.88,.76,.90][v%3]; th=([.118,.105,.138] if len(s)==2 else [.097,.087,.107])[v%3]
        d.text(s,w,h,th)
        if k==6: d.put((abs(X+Y)<.18)&(X**2+Y**2<.98**2),1.5)
    elif k in (9,10,16,41,42):
        if k in (9,41):
            car(d,-.33,-.035,.52,.42,.84);car(d,.31,.045,.50,.42)
        elif k in (10,42):
            if v%2==0:
                d.put(rect(-.62,-.34,-.15,.20),.88);d.put(rect(-.55,.18,-.22,.28),.88)
            else: truck(d,-.33,-.04,.62,.61)
            car(d,.29,.10,.50,.36)
        else:truck(d,0,0,1.25,.75)
        if k in (41,42):d.put((abs(X+Y)<.21)&(X**2+Y**2<.99**2),1.8)
    elif k in (11,24):
        if k==11:
            d.polygon([(-.13,.49),(.13,.49),(.13,-.05),(.24,-.05),(0,-.52),(-.24,-.05),(-.13,-.05)])
            d.polygon([(-.42,.40),(-.38,.22),(-.10,.01),(.10,.01),(.38,.22),(.42,.40),(0,.17)])
        else:
            d.line([(-.20,-.36),(-.20,.50)],.112);d.line([(.05,-.36),(.05,-.16),(.22,.09),(.22,.50)],.112)
    elif k==14:d.text('STOP',1.69,.69,.084)
    elif k==15:pass
    elif k==17:d.put(rect(-.85,-.14,.85,.14))
    elif k==18:
        d.polygon([(-.09,-.51),(.09,-.51),(.065,.23),(-.065,.23)]);d.oval(0,.405,.079,.081)
    elif k in (19,20,21):
        if k in (19,20):
            side=1 if k==19 else -1
            pts=curve((.22,.49),((.22,.37),(.23,.18),(.20,.07)),((.18,-.07),(-.02,-.14),(-.23,-.20)))
            d.line([(side*a,b) for a,b in pts],.165); d.polygon([(side*a,b) for a,b in [(-.34,-.23),(-.05,-.38),(-.14,-.04)]])
        else:
            d.line(curve((.16,.50),((.16,.35),(.19,.20),(.15,.12)),((.08,-.01),(-.16,.10),(-.17,-.06)),((-.18,-.16),(-.16,-.22),(-.16,-.26))),.16)
            top_arrow(d,-.16,-.09,0,-.33,.14,.32,.19)
    elif k==22:
        d.put(rect(-.52,.35,.52,.45));
        for x in [-.23,.22]:d.oval(x,.35,.25,.17)
    elif k==23:
        # rear of a car at a slant, above a pair of wavy skid marks
        for x in [-.20,.12]:d.line(curve((x,-.01),((x+.23,.07),(x+.23,.14),(x,.22)),((x-.14,.28),(x-.15,.36),(x+.09,.46))),.045)
        auto=Sketch();car(auto,0,0,.60,.46)
        d.put(resample(auto.z,.94*X+.34*(Y+.15),-.34*X+.94*(Y+.15))>.45)
        if v!=0:
            # Alternative with the whole vehicle and its skid path tilted
            # together, or with an alternating left-right fishtail.
            carpic=Sketch();car(carpic,0,-.13,.57,.48)
            for a in [-.19,.19]:carpic.line(curve((a,.04),((a+.17,.13),(a+.18,.20),(a+.05,.27)),((a-.09,.34),(a-.07,.44),(a+.08,.50))),.065)
            rot=math.radians(-18 if v==1 else 20);co=math.cos(rot);si=math.sin(rot)
            d.z=resample(carpic.z,co*X+si*(Y-.06),-si*X+co*(Y-.06))
    elif k==25:
        d.oval(-.035,-.29,.082,.084)
        d.line([(-.10,-.14),(-.22,-.02),(-.25,.14)],.131)
        d.line([(-.24,.10),(-.28,.30),(-.39,.47)],.098)
        d.line([(-.24,.11),(-.11,.22),(-.11,.43)],.086)
        d.line([(-.09,-.13),(-.032,.005),(.078,.046)],.075)
        d.line([(-.19,-.09),(.30,.31)],.043)
        d.polygon([(.02,.47),(.21,.21),(.29,.18),(.49,.47)])
    elif k==26:
        for y,l in [(-.30,.90),(.04,.45),(.37,.78)]:d.oval(0,y,.14,.145,l)
        if v%2: d.put((abs(X)<.04)&(Y>-.30)&(Y<.40),.45)
    elif k in (27,28):
        def walk(x,y,s,mirror=1.):
            def tr(a,b):return(x+mirror*a*s,y+b*s)
            d.oval(*tr(.075,-.76),.092*s,.096*s)
            d.line([tr(.02,-.57),tr(-.07,-.28)],.147*s)
            for pts in [[(-.07,-.28),(-.15,-.10),(-.25,.025)],[(-.07,-.26),(.04,-.15),(.13,.016)],[(.01,-.53),(.10,-.40),(.21,-.34)],[(-.01,-.55),(-.19,-.42),(-.24,-.29)]]:
                d.line([tr(a,b) for a,b in pts],.071*s)
        if k==27:walk(0,.46,1.10,-1 if v%2 else 1)
        else:walk(.12,.46,.92,-1);walk(-.24,.44,.61,1)
    elif k==29:
        for x in [-.28,.28]:
            d.oval(x,.29,.185,.185);d.oval(x,.29,.137,.137,0.)
        d.line([(-.28,.29),(-.13,.02),(.007,.285),(.22,.035),(-.13,.02)],.041);d.line([(.28,.29),(.19,-.01),(.12,-.04)],.046)
        if v==1: # Cycle sign with a rider, rather than the bicycle alone.
            d.oval(.06,-.32,.079,.079);d.line([(.005,-.21),(-.14,-.05),(.017,.07),(-.014,.23)],.084);d.line([(.012,-.19),(.10,-.08),(.21,-.08)],.06)
        else:
            d.line([(-.18,-.035),(-.055,-.035)],.056);d.line([(.19,.01),(.15,-.09),(.22,-.09)],.04);d.z=resample(d.z,X/1.12,Y+.13,0)
    elif k==30:
        for i in range(6):
            t=math.pi*i/3+math.pi/6; a,b=math.cos(t),math.sin(t)
            def tr(p,q):return (p*a-q*b,.10+p*b+q*a)
            d.line([tr(0,0),tr(.44,0)],.052)
            for h in [.20,.30]:
                for s in [-1,1]:d.line([tr(h,0),tr(h+.080,s*.097)],.043)
    elif k==31:
        d.polygon([(-.39,-.22),(-.30,-.18),(-.22,-.30),(-.16,-.24),(-.14,-.10),(.17,.01),(.29,.07),(.31,.18),(.20,.22),(-.04,.10),(-.21,.04),(-.25,-.12),(-.40,-.14)])
        for pts in [[(-.12,-.02),(-.27,.07),(-.39,.04)], [(-.05,.04),(-.23,.14),(-.29,.27)],[(.20,.14),(.27,.28),(.44,.33)],[(.19,.16),(.20,.30),(.32,.45)],[(-.24,-.25),(-.32,-.35),(-.30,-.44)]]:d.line(pts,.048)
    elif k==32:d.put((abs(X+Y)<.21)&(X**2+Y**2<1),1.)
    elif k in (33,34):
        s=1 if k==33 else -1
        pts=curve((-.17,.73),((-.17,.51),(-.17,.17),(-.17,-.01)),((-.17,-.28),(.01,-.35),(.33,-.35)))
        d.line([(s*x,y) for x,y in pts],.175)
        d.polygon([(s*x,y) for x,y in [( .26,-.68),(.65,-.35),(.26,-.035)]])
        d.put(rect(-.27 if s==1 else .08,.65,-.07 if s==1 else .27,.77))
    elif k==35:top_arrow(d,0,.77,0,-1.55,.19,.64,.40)
    elif k in (36,37):
        s=1 if k==36 else -1
        top_arrow(d,-.16*s,.76,0,-1.51,.17,.55,.35)
        pts=curve((-.16,.59),((-.15,.20),(.08,.01),(.38,.01)))
        d.line([(s*x,y) for x,y in pts],.17)
        d.polygon([(s*x,y) for x,y in [(.31,-.27),(.69,.005),(.31,.30)]])
    elif k in (38,39):
        s=1 if k==38 else -1
        top_arrow(d,-.48*s,-.54,1.02*s,1.08,.18,.84,.38)
    elif k==40:
        for i in range(3):
            a=math.pi/2+2*math.pi*i/3
            ang=np.linspace(a+.93,a-.46,15)
            points=[(.54*math.cos(t),.54*math.sin(t)) for t in ang]
            d.line(points,.135); ex,ey=points[-1];t=ang[-1];top_arrow(d,ex-.13*math.sin(t),ey+.13*math.cos(t),.29*math.sin(t),-.29*math.cos(t),.128,.39,.24)
    else:raise ValueError(k)
    if k in (19,20,21):
        # Signs warning of a curve may show the bend as a bent road rather
        # than a literal turn-arrow, including the square-cornered convention.
        side=-1 if k==20 else 1
        if v:
            d=Sketch(); bend=[(.18,.50),(.18,.02),(.12,-.09),(-.18,-.22),(-.26,-.31)]
            if k==21:bend=[(.19,.50),(.19,.12),(.15,.07),(-.16,.07),(-.16,-.24),(-.22,-.35)]
            d.line([(side*x,y) for x,y in bend],.18 if v==1 else .145)
    return d.z

redring=[0,1,2,3,4,5,7,8,9,10,15,16]
triangles=[11]+list(range(18,32))
ends=[6,32,41,42]
blue=list(range(33,41))

def family(k):
    if k in redring:return 'ring'
    if k in triangles:return 'triangle'
    if k in ends:return 'end'
    if k in blue:return 'blue'
    return {12:'diamond',13:'yield',14:'stop',17:'entry'}[k]

# Distance from each boundary, in units of the apothem. Round triangle tips slightly.
def field(f):
    if f in ('triangle','yield'):
        yy=Y if f=='triangle' else -Y
        g=np.maximum(yy,2*abs(X)-yy-1)
        # quadratic rounding of the top apex and the two obtuse-angle coordinates
        g=np.maximum(g, -yy-.022);g=np.maximum(g,abs(X)*1.26-.30*yy-.015)
        return g
    if f=='diamond':return abs(X)+abs(Y)
    if f=='stop':return np.maximum(np.maximum(abs(X),abs(Y)),(abs(X)+abs(Y))*.70710678)
    return np.hypot(X,Y)

def family_art(f):
    a=field(f); mask=(a<1.025); rim=(a<=1)&(a>=({'triangle':.57,'yield':.53,'ring':.76}.get(f,.93)))
    lumin=np.where(a<=1,.94,.52);colour=np.zeros_like(X)
    if f in ('ring','triangle','yield'):
        lumin[rim]=.23;colour[rim]=1
    elif f=='blue':
        lumin[a<=1]=.94;lumin[a<.955]=.21;colour[a<.955]=1
    elif f in ('stop','entry'):
        m=a<(.935 if f=='stop' else .965);lumin[m]=.28;colour[m]=1
        if f=='entry':
            bar=rect(-.85,-.145,.85,.145);lumin[bar]=.95;colour[bar]=0
    elif f=='diamond':
        m=a<.70;colour[m]=1;lumin[m]=.70;lumin[(a<1)&(a>.957)]=.20
    else:
        m=(abs(Y+X)<.20)&(a<.99);lumin[m]=.30
    if f in ('ring','triangle','yield'):
        # White edge outside the red when present on a reflective sign.
        lumin[(a>.975)&(a<=1)]=.63
    return lumin,colour

layers=[]; weights={}
def layer(name,op,inp,**settings):
    layers.append(dict(name=name,op=op,inputs=inp if isinstance(inp,list) else [inp],**settings));return name

def aff(name,inp,w,b=None,conv=False,p=0):
    w=np.asarray(w,dtype=np.float32);b=np.zeros(len(w),np.float32) if b is None else np.asarray(b,np.float32)
    weights[name+'.W']=w;weights[name+'.b']=b
    return layer(name,'conv' if conv else 'dense',inp,out=len(w),**(dict(kernel=list(w.shape[-2:]),stride=1,padding=p) if conv else {}))

def pool(name,op,inp,win,step):return layer(name,op,inp,size=list(win),stride=list(step))
def reshape(name,inp,*s):return layer(name,'reshape',inp,shape=list(s))
def point(name,inp,w,b=None):return aff(name,inp,np.array(w)[:,:,None,None],b,True)

def contrast_maps():
    # A compressed radiance map makes dark and bright crops more comparable. A
    # piecewise-linear approximation is used instead of a table of fitted values.
    knees=[0,.035,.15,.40]; slopes=[3.4,1.85,1.06,.64]; w=[];b=[]
    for j in range(3):
        for x in knees:w.append(np.eye(3)[j]);b.append(-x)
    a=point('tone_knots','image',w,b);a=layer('tone_relu','relu',a)
    w=np.zeros((3,len(b)))
    for j in range(3):w[j,4*j:4*j+4]=np.diff([0]+slopes)
    a=point('radiance',a,w); a=point('opponency',a,[[.24,.53,.23],[1,-.65,-.35],[-.70,-.30,1],[.45,.55,-1]])
    # Spatial high pass on the luminance and opponent channels. Denominator is
    # the energy in the current photograph, never a population statistic.
    a=layer('contrast','norm',a)
    return a

# Build a template in patch coordinates. Each individual dot product is
# a zero-DC contrast within its mask, with optional surrounds to reject clutter.
# A regularized mean absolute contrast norm does not privilege bulky pictograms.
def unit(im,m,how=1):
    w=m*(im-(im*m).sum()/max(m.sum(),1e-8))
    if how==1:div=np.abs(w).sum()
    else:div=np.sqrt((w*w).sum()*m.sum())
    return w/max(div,1e-7)

# Patch size, sign radii in the image, and translation lattice. Pose is shared
# between colour border and pictogram, not inferred from dataset measurements.
PS=28; SHIFT=5
# Slightly oversample location even when a change of one pixel moves a thin
# numeral completely off its matching stroke. Full BLAS matrices are used for
# evaluating the same filter bank at all candidate centers in a batch.
scales=[8.5,10.5,12.8,15.0]
rr,cc=np.meshgrid(np.arange(PS)-.5*(PS-1),np.arange(PS)-.5*(PS-1),indexing='ij')

def resample(im,x,y,extend=0):
    xx=(x/2.18+.5)*S-.5;yy=(y/2.18+.5)*S-.5
    ix=xx.astype(int).clip(0,S-2);iy=yy.astype(int).clip(0,S-2)
    ax=(xx-ix).clip(0,1);ay=(yy-iy).clip(0,1)
    z=(1-ay)*(im[iy,ix]*(1-ax)+im[iy,ix+1]*ax)+ay*(im[iy+1,ix]*(1-ax)+im[iy+1,ix+1]*ax)
    return np.where((abs(x)<=1.09)&(abs(y)<=1.09),z,extend)

def raster(im,rx,ry,angle,sx=0.,sy=0.,extend=0):
    t=math.radians(angle);co=math.cos(t);si=math.sin(t);a=np.zeros_like(cc)
    for dx in [-.32,.32]:
        for dy in [-.32,.32]:
            xx=(cc+dx)/rx-sx; yy=(rr+dy)/ry-sy
            a+=resample(im,co*xx+si*yy,co*yy-si*xx,extend)*.25
    return a

def avg2(z):return z.reshape(PS//2,2,PS//2,2).mean((1,3))

def surround(z):
    # Small center-surround filter applied to one template, i.e. it stays a
    # linear template match. Removes flat expanses and very slow shading.
    e=np.pad(z,2,mode='edge');sm=np.zeros_like(z)
    for i in range(5):
        for j in range(5):sm+=e[i:i+PS,j:j+PS]/25
    return z-sm

def ftemplate(f,R,aspect=1,angle=0):
    l,c=arts[f]; shapes=field(f);m0=(shapes<1.10).astype(float)
    if f in ('triangle','yield','ring'):
        # Mask the pictogram area down; it is accounted for by a different part.
        m0*=np.where(shapes<(.40 if f!='ring' else .58),.45,1.)
    m=avg2(raster(m0,R*aspect,R,angle));l=avg2(raster(l,R*aspect,R,angle,extend=.52));c=avg2(raster(c,R*aspect,R,angle))
    # Pixels just outside the disc supply a weak contextual baseline for solid
    # coloured signs. Unlike a texture, their brightness is unspecified.
    z=np.zeros((4,PS//2,PS//2))
    # Same neutral red-vs-blue cue for all flat fields. For a blue disc the
    # light arrow also supplies evidence, separately, in the part bank.
    if f in ('ring','triangle','yield'):
        z[0]=unit(l,m)*.61;z[1]=unit(c,m)*.50
        z[1]+=.24*m*c/max((m*c).sum(),1e-8)
    elif f=='diamond':
        z[0]=unit(l,m)*.30;z[3]=unit(c,m)*.71
    elif f in ('stop','entry','blue'):
        ext=.45*((cc/R)**2+(rr/R)**2<=1.30**2);m=np.maximum(m,avg2(ext));
        z[0]=unit(l,m)*(.50 if f=='entry' else .23);ch=2 if f=='blue' else 1
        z[ch]=unit(c,m)*(.50 if f=='entry' else .61)
        z[ch]+=.42*m*c/max((m*c).sum(),1e-8)
    else:
        # A neutral disk with an uninterrupted rising diagonal extending
        # *outside* its central lettering or vehicles. Reject spurious dark
        # stripes made from two ordinary glyphs by emphasizing both ends.
        flank=avg2(raster(((abs(X)>.35)|(abs(Y)>.39)).astype(float),R,R,0))
        z[0]=unit(l,m*(.17+.83*flank))*.90
    return z.astype('f')

def itemplate(k,art,R,aspect,angle,mag,dx=0,dy=0):
    f=family(k);rp=R*mag
    # Evidence is primarily in the sign face, not the surrounding sky/road.
    if k==14: roi=(field('stop')<.93)
    elif f=='blue':roi=(field(f)<.90)
    elif f=='triangle':roi=(field(f)<.548)
    elif f=='end':roi=(field(f)<.88)
    else:roi=field('ring')<.716
    if k in [0,1,2,3,4,5,7,8]:roi=roi & (abs(Y)<.60)
    m=raster(roi.astype(float),R*aspect,R,angle)
    if k==15 or k in (12,13):return np.zeros_like(m,dtype='f')
    z=raster(art,rp*aspect,rp,angle,dx/rp,dy/rp)
    if k in [6,41,42]:
        # Read the faded original symbol on either side of the overprint.
        m*=raster(np.clip((abs(X+Y)-.16)*6,0,1),rp*aspect,rp,angle,dx/rp,dy/rp)
    if k in [14,17]+blue:pol=1
    else:pol=-1
    # Combine a face contrast with an edge-enhanced contrast from the same art.
    u0=unit(z,m,2); u1=m*surround(z);u1-=m*(u1.sum()/max(m.sum(),1e-8));u1/=max(np.sqrt(m.sum()*(u1*u1).sum()),1e-7)
    return (pol*(.88*u0+.12*u1)).astype('f')

def kernel_to_row(kernel,h,dy,dx):
    # Scatter a patch into a fixed crop, zero padding instead of wrapping.
    c,k,_=kernel.shape;off=(h-k)//2; y=off+dy;x=off+dx
    dest=np.zeros((c,h,h),np.float32)
    a,b=max(0,-y),max(0,-x);p,q=min(k,h-y),min(k,h-x)
    dest[:,y+a:y+p,x+b:x+q]=kernel[:,a:p,b:q];return dest.ravel()

arts={f:family_art(f) for f in sorted(set(family(k) for k in range(43)))}
pictos={k:[symbol(k,i) if k not in (12,13) else np.zeros_like(X) for i in range(3)] for k in range(43)}

def stripe_and_blank(lum,poses,F):
    # The symbol-free outer segments of the cancellation stripe form two
    # independent tests. The minimum of their responses rejects unrelated
    # center graphics; a third test reads residual ink *beside* the stripe.
    W=[]
    for typ in range(3):
        for R in scales:
            inside=field('ring')<.97;t=(X-Y)/1.4142;d=(X+Y)/1.4142
            if typ<2:
                m=inside & (abs(d)<.58) & (t*(2*typ-1)>.41)
                z=-(abs(d)<.165).astype(float)
            else:
                m=inside & (abs(d)>.22) & (abs(X)<.72) & (abs(Y)<.74)
                z=(Y**2 < .36**2).astype(float)
            w=unit(raster(z,R,R,0),raster(m*1.,R,R,0))[None]
            for dy,dx in poses:W.append(kernel_to_row(w,32,dy,dx))
    S0=aff('slash_parts',lum,W);S0=reshape('slash_maps',S0,3,len(scales),len(poses));w=np.zeros((43,3));w[ends,0]=1.15;w[32,2]=1.02;w[[6,41,42],2]=-.38
    a=point('slash_part_a',S0,w);w=np.zeros((43,1));w[ends]=-1.15
    delta=point('slash_difference',S0,[[1,-1,0]])
    delta=layer('unequal_parts','relu',delta);delta=point('slash_balance',delta,w)
    return layer('slash_support','add',[a,delta])

def warning_parts(inp,poses,Q,P):
    # Besides the full interior, recognise two overlapping pictogram parts
    # independently. Small differences in the scale of a car, skid marks,
    # bicycle wheels, a person, or the base of a curve then need not defeat a
    # correct match to the other half. No weights shared between drawings.
    parts=[]; nv=24
    for k in triangles:
        for R in scales:
            for side in [0,1]:
                for v,mag,ar,ang in [(0,.93,1,0),(0,1.13,1,0),(1,1,1,0),(2,.98,1,0),(0,1.,.84,-6),(0,.99,1.06,6)]:
                    for dx,dy in [(-.7,-.82),(.7,-.82),(-.7,.82),(.7,.82)]:
                        # A vertically centered symbol on a triangular plaque
                        # is lower in the bounding box than one on a circle.
                        down=.135
                        m0=(field('triangle')<.57) & ((Y<.27) if side==0 else (Y>-.01))
                        m=raster(m0*1.,R*ar,R,ang)
                        z=raster(pictos[k][v],R*mag*ar,R*mag,ang,dx/R,dy/R+down)
                        parts.append(-unit(z,m,2).ravel())
    a=point('warning_parts',inp,parts);a=reshape('part_views',a,len(triangles)*Q*2,nv,P)
    a=pool('best_part','maxpool',a,[nv,1],[nv,1]);a=reshape('part_pairs',a,len(triangles)*Q,2,P)
    a=pool('part_mean','avgpool',a,[2,1],[2,1]);a=reshape('warning_grid',a,len(triangles),Q,P)
    w=np.zeros((43,len(triangles)))
    for i,k in enumerate(triangles):w[k,i]=.69
    return point('warning_support',a,w)

def local_contrast(lum,poses,Q,P):
    # Adapt the contrast to the central region of *this input*. A dark sign
    # against a bright sky can otherwise contribute next to nothing to a global
    # contrast measurement. The 20x20 crop deliberately excludes the rim of
    # the photograph, not the rim of a rendered template.
    coords=poses
    n=20;h=32;g=[]
    for dy,dx in coords:
        for j in range(n):
            for i in range(n):
                w=np.zeros((h,h),'f');w[(h-n)//2+j+dy,(h-n)//2+i+dx]=1;g.append(w.ravel())
    a=aff('local_windows',lum,g);a=reshape('local_images',a,len(coords),n,n);a=layer('local_contrast','norm',a)
    # Tile the cropped images into a tall strip; a single shared kernel bank
    # with stride equal to the tile width scores one whole tile at a time.
    a=reshape('local_strip',a,1,len(poses)*n,n)
    bank=[]
    for k in range(43):
        for R in scales:
            for ar,ang,mag,v in [(1,0,1,0),(.96,0,.87,1),(1,0,1.10,2),(.88,-5,1,0),(1.05,5,1,0)]:
                dy=R*.085 if k in triangles else 0
                for dx,dyy in [(-.70,-.68),(.70,-.68),(-.70,.68),(.70,.68)]:
                    z=itemplate(k,pictos[k][v],R,ar,ang,mag,dx,dy+dyy)
                    off=(PS-n)//2;bank.append(z[off:off+n,off:off+n][None])
    w=np.asarray(bank,dtype='f');name='local_symbol_bank';weights[name+'.W']=w;weights[name+'.b']=np.zeros(len(w),'f');
    a=layer(name,'conv',a,out=len(w),kernel=[n,n],stride=n,padding=0)
    a=reshape('local_symbol_grid',a,43*Q,20,P);a=pool('local_best','maxpool',a,[20,1],[20,1]);a=reshape('local_evidence',a,43,Q,P)
    st=np.ones(43)*.37;st[[12,13,15]]=0;st[[17,32]]=.18
    return point('local_support',a,np.diag(st))

def achromatic_end(lum,poses,Q,P):
    # A standardized gray cancellation plaque is distinctive even when both
    # the pigment and reflective background are dark. Measure the stripe
    # relative to two physically separate parallel clear regions, using a
    # rectified difference that vanishes for uniform gray.
    opts=[(ang,shift,th) for ang in [-10,0,10] for shift in [-.11,.0,.11] for th in [.13,.205]];mats=[]
    for R in scales:
        for ang,shift,th in opts:
            t=math.radians(45+ang);d=X*math.cos(t)+Y*math.sin(t)-shift
            m=(X*X+Y*Y<.93**2)&(abs(d)<.72);z=-(abs(d)<th).astype(float)
            w=unit(raster(z,R,R,0),raster(m*1.,R,R,0))[None]
            for y,x in poses:mats.append(kernel_to_row(w,32,y,x))
    a=aff('diagonal_variants',lum,mats);a=reshape('diagonal_grid',a,Q,len(opts),P);a=pool('diagonal_peak','maxpool',a,[len(opts),1],[len(opts),1]);a=reshape('continuous_slash',a,1,Q,P)
    g=np.zeros((43,1));g[ends]=.66;g[32]=.86
    # Any residual symbol should increase dark-on-light contrast outside the
    # slash. An empty white face gets credit when residual texture is absent.
    return point('diagonal_support',a,g)

def digit_parts(inp,poses,Q,P):
    # Factor the speed legend: separately match the first (non-shared)
    # numeral and the suffix. This admits kerning and stroke variations without
    # having to hand-specify a new full legend for every font.
    speeds=[0,1,2,3,4,5,7,8]; bank=[]
    for k in speeds:
        word=['20','30','50','60','70','80','','100','120'][k]
        for R in scales:
            for part in range(2):
                for wide,h,mag,ang in [(1.,.84,1.,0),(.85,.87,1.,0),(.98,.87,.85,0),(1.10,.90,1.06,0),(.90,.87,1.,-5),(1.06,.82,1.,5)]:
                    ww=(1.17 if len(word)==3 else 1.08)*wide
                    d=Sketch();d.text(word,ww,h,.106 if len(word)==2 else .094)
                    # Two digits have a decoupled left glyph. Three digits
                    # share the outer 1 and 0, so the middle glyph is decisive.
                    if len(word)==2:m=(X<0) if part==0 else (X>=0)
                    else:m=(abs(X-.04)<ww*.19) if part==0 else (abs(X-.04)>=ww*.19)
                    m=m & (abs(X)<ww*.5+.095)&(abs(Y)<h*.5+.09)&(np.hypot(X,Y)<.76)
                    for dx,dy in [(0,0),(-.78,-.55),(.78,-.55),(-.78,.55),(.78,.55)]:
                        z=raster(d.z,R*mag,R*mag,ang,dx/R,dy/R);mask=raster(m*1.,R*mag,R*mag,ang,dx/R,dy/R)
                        bank.append(-unit(z,mask,2).ravel())
    a=point('digit_parts',inp,bank);a=reshape('digit_hypotheses',a,len(speeds)*Q*2,30,P)
    a=pool('best_glyph','maxpool',a,[30,1],[30,1]);a=reshape('separate_glyphs',a,len(speeds)*Q,2,P)
    a=pool('legend_parts','avgpool',a,[2,1],[2,1]);a=reshape('legend_grid',a,len(speeds),Q,P)
    m=np.zeros((43,len(speeds)))
    for i,k in enumerate(speeds):m[k,i]=.80
    return point('digit_support',a,m)

def build():
    a=contrast_maps()
    # A mild saturation clamp on each input channel (fixed knots at +/-2).
    t=point('clip_knots',a,np.vstack((np.eye(4),np.eye(4))),[2.]*4+[-2.]*4);t=layer('clip_ramp','relu',t)
    t=point('bounded',t,np.hstack((np.eye(4),-np.eye(4))),[-2.]*4)
    lum=point('light',t,[[1,0,0,0]])
    low=pool('coarse','avgpool',t,[2,2],[2,2])
    # Two-pixel grid for position, one dense layer handles all subwindows in
    # parallel. Pool across alternative symbols/aspect/rotation first, then
    # across translations and sizes. Family evidence shares the latter pose.
    poses=[(y,x) for y in [-2,0,2] for x in [-2,0,2]]
    F=sorted(arts);K=43;P=len(poses);Q=len(scales)
    M=[]
    rim_poses=[(1,0),(.84,0),(1.15,0),(1,-8),(1,8)]
    for f in F:
        for R in scales:
            for aspect,angle in rim_poses:
                z=ftemplate(f,R,aspect,angle)
                for y,x in poses:M.append(kernel_to_row(z,16,y//2,x//2))
    fam=aff('rim_bank',low,M);del M
    fam=reshape('rim_views',fam,len(F)*Q,len(rim_poses),P)
    fam=pool('rim_view','maxpool',fam,[len(rim_poses),1],[len(rim_poses),1]);fam=reshape('rim_grid',fam,len(F),Q,P)
    # A class must agree with the type of sign, as well as matching a small
    # patch of black marks. Blank faces give all their evidence at this level.
    mapping=np.zeros((K,len(F)))
    for k in range(K):mapping[k,F.index(family(k))]=1.45
    fam=point('rim_evidence',fam,mapping)
    # Penalty for an appreciably non-uniform face: fixed bandpass followed by
    # absolute value and averaging. Essential for empty-ring and yield signs.
    b=np.zeros((2,1,5,5));b[0,0].fill(-1/25);b[0,0,2,2]+=1;b[1]=-b[0]
    hi=aff('fine_variation',lum,b,conv=True,p=2);hi=layer('abs_halves','relu',hi);hi=point('face_texture',hi,[[1,1]])
    eb=[]
    for f in ['ring','yield','diamond']:
        for R in scales:
            mask=(field(f)<{'ring':.67,'yield':.46,'diamond':.58}[f]);z=raster(mask*1.,R,R,0);z/=max(z.sum(),1)
            for y,x in poses:eb.append(kernel_to_row(z[None],32,y,x))
    en=aff('texture_bank',hi,eb);en=reshape('texture_grid',en,3,Q,P)
    w=np.zeros((K,3));w[15,0]=-2.0;w[13,1]=-.48;w[12,2]=-.38
    en=point('uniformity',en,w)
    # Vary body artwork size independently of border size, and allow small
    # perspective distortions. All hypotheses are hand specified.
    variations=[(aspect,angle,mag,v,dx,dy)
        for aspect,angle,mag,v in [(1.,0,1.,0),(.95,0,.88,1),(1.,0,1.10,2),(.86,-5,1.,0),(1.05,5,1.,0)]
        for dx,dy in [(0,0),(-.84,-.84),(.84,-.84),(-.84,.84),(.84,.84)]]
    # To avoid an enormous replicated dense layer, gather all translated
    # luminance patches once. Reshape so that a 1x1 convolution scores them.
    gather=[]
    for j in range(PS):
        for i in range(PS):
            for y,x in poses:
                row=np.zeros((32,32),np.float32);row[(32-PS)//2+j+y,(32-PS)//2+i+x]=1;gather.append(row.ravel())
    inp=aff('extract_windows',lum,gather);del gather
    inp=reshape('window_columns',inp,PS*PS,1,P)
    bank=[]
    for k in range(K):
        for R in scales:
            for aspect,angle,mag,v,dx,dy in variations:
                bank.append(itemplate(k,pictos[k][v],R,aspect,angle,mag,dx,dy).ravel())
    sym=point('pictogram_bank',inp,bank);del bank
    sym=reshape('pictogram_grid',sym,K*Q,len(variations),P)
    sym=pool('best_art','maxpool',sym,[len(variations),1],[len(variations),1]);sym=reshape('symbol_grid',sym,K,Q,P)
    strength=np.ones(K)*1.15
    strength[[6,41,42]]=1.10;strength[32]=.74;strength[17]=.38
    strength[[0,1,2,3,4,5,7,8]]=.52
    strength[triangles]=.74
    strength*=.82
    bias=np.zeros(K);bias[15]=.59;bias[12]=.40;bias[13]=.45;bias[32]=.20
    sym=point('symbol_evidence',sym,np.diag(strength),bias)
    digits=digit_parts(inp,poses,Q,P);slash=stripe_and_blank(lum,poses,F)
    parts=warning_parts(inp,poses,Q,P);loc=local_contrast(lum,poses,Q,P);ach=achromatic_end(lum,poses,Q,P)
    comb=layer('pose_scores','add',[fam,sym,en,digits,slash,parts,loc,ach]);comb=pool('best_pose','maxpool',comb,[Q,P],[Q,P]);reshape('scores',comb,43)

if __name__=='__main__':
    build();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(out/'weights.npz',**weights)
