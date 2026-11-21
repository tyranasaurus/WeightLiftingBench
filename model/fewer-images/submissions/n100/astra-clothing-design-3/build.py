"""Build a clothing-recognition circuit from parametric cut patterns and hand rules.
No input files, fonts, image resources or learned parameters.
"""
import sys, json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(72914)
def u(a,b): return rng.uniform(a,b)
def pick(seq): return seq[rng.integers(len(seq))]
def sometimes(p): return rng.random()<p

# Coordinates in units of the image side. Each garment is described in its
# own bounding rectangle before being placed in the camera frame.
N=56
Y,X=np.mgrid[:N,:N].astype(float); X=(X+.5)/N;Y=(Y+.5)/N

def frame(aspect,kind):
    full=u(.93,1.005)
    w,h=full*min(1.,aspect),full*min(1.,1/aspect)
    dx=u(-.028,.028);dy=u(-.035,.035)
    # Horizontal articles may have a little more headroom than footroom.
    if kind=='foot': dy+=u(-.06,.095)
    if kind=='bag': dy+=u(-.04,.045)
    y=(Y-.5-dy)/h+.5; x=(X-.5-dx)/w+.5
    x=x+u(-.045,.045)*(y-.5); y=y+u(-.045,.045)*(x-.5)
    return x,y

def curve(t,knots,vals): return np.interp(t,knots,vals)
def between(t,lo,hi): return (t>=lo)&(t<=hi)

def ellipse(x,y,cx,cy,rx,ry):return ((x-cx)/rx)**2+((y-cy)/ry)**2<1

def line(x,y,a,b,th):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=np.clip(((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy+1e-12),0,1)
    return (x-a[0]-dx*t)**2+(y-a[1]-dy*t)**2<th**2

def fabric(m,x,y):
    # Cast in soft material with subdued folds, not solid cardboard.
    phase=u(0,7); f=u(1,4); amp=u(0,.095)
    return m*(1-amp*(1+np.cos(6.283*f*x+phase+y*u(-3,3))))

# A symmetric bodice with attached sleeves. All horizontal measurements below
# are fractions of the width, vertical measurements fractions of the height.
def body(k):
    style=rng.random(); collar=False;opening=False; tails=0.; sleeve='long'
    if k==0:                       # knit tee, tank or long-sleeved jersey
        sleeve='short' if style<.83 else 'tank' if style<.96 else 'long'
        aspect=u(.72,1.08) if sleeve!='tank' else u(.48,.73)
        if sleeve=='long':aspect=u(.66,.86)
        bw=u(.26,.325) if sleeve=='short' else u(.28,.39)
        waist=bw*u(.87,1.05); hem=bw*u(.9,1.09)
        shoulder=u(.28,.36); bottom=1.; end=u(.25,.395) if sleeve=='short' else u(.88,1.)
        dep=u(.035,.14); neck=u(.11,.19); slope=u(.05,.13)
        tails=u(0,.03)
    elif k==2:                     # sweater / sweatshirt, often a ribbed waistband
        aspect=u(.7,.98);bw=u(.285,.365);waist=bw*u(.92,1.045);hem=waist*u(.87,1.02)
        shoulder=u(.285,.36);bottom=u(.82,1.);end=1.;dep=u(.035,.135);neck=u(.10,.19);slope=u(.065,.145)
        if sometimes(.55):                # an opaque back neck behind the front scoop
            dep*=u(.12,.55);slope*=u(.8,1.15)
        if sometimes(.23):bottom=1.;end=u(.88,.995)
        if style<.11: dep=u(.0,.03);neck=u(.075,.12);slope=u(.1,.17) # rollneck
        if style>.88:sleeve='three';end=u(.62,.85);bottom=1.
    elif k==4:                     # structured jacket, parka, or longer overcoat
        short=style<.57
        aspect=u(.72,.95) if short else u(.51,.81)
        bw=u(.29,.36);waist=bw*u(.90,1.03);hem=waist*(u(.91,1.02) if short else u(1.01,1.21))
        shoulder=u(.27,.345);bottom=u(.86,1.) if short else 1.;end=(1. if bottom<1. else u(.65,.95))
        if short and sometimes(.23):bottom=1.;end=u(.88,1.)
        dep=u(.0,.06);neck=u(.07,.125);slope=u(.11,.195);collar=True;opening=True
    elif k==6:                     # button-fronts, blouses, overshirts, smocks
        sleeve='short' if style<.29 else 'three' if style<.50 else 'long'
        aspect=u(.72,1.02) if sleeve=='short' else u(.62,.94)
        bw=u(.285,.38);waist=bw*u(.82,1.04);hem=waist*u(.96,1.18)
        shoulder=u(.265,.37);bottom=1.;end={'short':u(.3,.5),'three':u(.52,.83),'long':u(.82,1)}[sleeve]
        if sleeve=='long' and sometimes(.24):end=1.;bottom=u(.86,.99)
        collar=sometimes(.54);opening=collar or sometimes(.34)
        neck=u(.075,.13) if collar else u(.115,.235);dep=u(.03,.105) if collar else u(.07,.19)
        slope=u(.09,.165);tails=u(.015,.12)
    else:                          # one-piece dress: bodice, fitted waist, then skirt
        sleeve='tank' if style<.30 else 'short' if style<.66 else 'three' if style<.82 else 'long'
        aspect=u(.38,.69) if sleeve!='long' else u(.42,.72)
        bw=u(.22,.335);waist=bw*u(.72,.95);hem=u(.32,.49)
        shoulder=u(.21,.34);bottom=1.;end={'short':u(.18,.31),'long':u(.52,.77),'three':u(.34,.49),'tank':0}[sleeve]
        neck=u(.1,.22);dep=u(.045,.165);slope=u(.04,.10);tails=u(.01,.055)
    # A standard-length top covers a torso wider than half its sleevespan
    # when displayed with the arms hanging down. Vary this breadth at fixed
    # length rather than forcing all hems to be narrow.
    if k==0 and sleeve=='short':
        sb=u(1.02,1.25);bw*=sb;waist*=sb;hem*=sb
    if k==0 and sleeve=='tank':
        aspect=u(.56,.80);bw=u(.385,.475);waist=bw*u(.87,1.02);hem=waist*u(.99,1.12);shoulder=u(.29,.41)
    if k==3:
        # A skirt has room for the hips; sleeveless/summer dresses often
        # reach their greatest breadth at the bottom, not at the shoulder.
        if sometimes(.60):
            hem=u(.40,.50);bw=min(bw,hem*u(.48,.77));waist=min(waist,bw*u(.72,.92))
    if k==3 and sometimes(.3):
        # Straight shift/sheath dress, not necessarily an exaggerated A-line.
        hem=waist*u(1.0,1.35);aspect*=u(.80,1.0)
    if k==4 and sometimes(.4):
        # Tailored overcoats have their own elongated pattern, not just a
        # recolored sweater. The sleeves stop above the bottom of the coat.
        aspect=u(.49,.78);bottom=1.;end=u(.67,.91);hem=waist*u(1.06,1.26)
    x,y=frame(aspect,'top');q=np.abs(x-.5)
    # Top edge is a scooped neckline between two shoulders, then slopes down.
    shoulder_y=curve(q,[0,neck,shoulder,.55],[0,0,slope,slope+.075]); top=shoulder_y.copy()
    neckshape=max(.5,u(.65,2.8)); top+=dep*(np.maximum(1-(q/neck)**neckshape,0))**.65
    bust_y=.21 if k==3 else .27; waist_y=u(.34,.52) if k==3 else u(.49,.70)
    shape=curve(y,[0,slope,bust_y,waist_y,bottom-.065,bottom],[neck,shoulder,bw,waist,hem*1.015,hem]);
    edge=bottom-tails*np.clip(q/hem,0,1)**2
    core=(q<=shape)&between(y,top,edge)
    if sleeve!='tank':
        # Outer contour, cuff edge and the line from the armpit to the wrist.
        if sleeve=='short':
            tip=u(.455,.505) if k!=3 else u(.36,.49); thick=u(.095,.16)
            sideoff=u(.04,.115); inner=max(bw+.008,tip-thick)
            if k==3:sideoff*=.65
            outer=curve(y,[slope,end-sideoff,end],[shoulder,tip,inner]);
            low=curve(q,[bw,inner,tip],[end+u(-.025,.055),end,end-sideoff]);
            arm=(q>=np.minimum(shape,shoulder)-.045)&(q<=outer)&(y>=shoulder_y)&(y<=low)&(y<=end+.045)
        else:
            tip=u(.453,.50); thick=u(.071,.115) if k!=4 else u(.09,.135)
            if k==3:tip=u(.35,.48);thick=u(.055,.095)
            if k==6 and sleeve=='three':thick=u(.08,.144)
            armpit=.21 if k==3 else u(.24,.43)
            mid=shoulder+(tip-shoulder)*u(.39,.76)
            outer=curve(y,[slope,armpit,end-.08,end],[shoulder,mid,tip,tip-u(0,.018)])
            inner=curve(y,[slope,armpit,end],[shoulder-.07,bw-u(.0,.05),tip-thick]);
            arm=between(q,inner,outer)&(y>=shoulder_y)&(y<=end-u(0,.033)*(q-inner)/thick)
        m=core|arm
    else:
        # tanks have deep armholes and shoulder straps, not cap sleeves
        shoulder=u(.245,.345); neck=min(shoulder-.04,neck*u(1.,1.34));
        shape=curve(y,[0,.07,bust_y,waist_y,bottom],[shoulder,shoulder*.97,bw,waist,hem]);
        top=dep*1.55*np.maximum(1-(q/neck)**2,0)**.55
        m=(q<shape)&between(y,top,edge)
    z=fabric(m,x,y)
    if collar:                     # turned lapels or folded points
        cy=u(.105,.245); lap=abs(q-neck*np.clip(1-y/cy,0,1))
        z*=1-u(.08,.26)*np.exp(-(lap/.024)**2)*between(y,0,cy)
    if opening:
        breadth=u(.008,.035) if k!=4 or sometimes(.65) else u(.035,.095)
        z*=1-u(.16,.60)*np.exp(-(q/breadth)**4)*between(y,dep,edge)
    if k==2 and sometimes(.58):z*=1-u(.025,.12)*between(y,bottom-u(.04,.09),bottom)
    # Reframe short/long versions of the major components after assembly.
    # Unit rectangles are only cutting-pattern coordinates; the shop crop
    # surrounds the garment itself. Centering uses this drawing alone.
    return z

def pants():
    shorts=sometimes(.08); aspect=u(.33,.60) if not shorts else u(.55,.89)
    x,y=frame(aspect,'leg');q=abs(x-.5); waist=u(.38,.485);hips=u(.445,.50);crotch=u(.26,.46) if not shorts else u(.4,.69)
    inner_end=u(.026,.19); outer_end=u(.25,.455);
    if sometimes(.16):inner_end=u(0,.045);outer_end=u(.42,.50)
    outer=curve(y,[0,.12,crotch,.72,1],[waist,hips,hips*u(.91,1.015),outer_end*u(.95,1.17),outer_end]);
    inner=inner_end*np.maximum((y-crotch)/(1-crotch),0)**u(.52,1.4)
    m=(q<outer)&(q>=inner)&between(y,0,1-u(0,.06)*q);return fabric(m,x,y)

def bag():
    style=rng.random(); handled=style<.72; pack=style>.92
    if handled:
        aspect=u(.73,1.52); high=u(.17,.52)
    elif pack:aspect=u(.62,.93); high=u(.015,.12)
    else:aspect=u(1.08,2.5);high=u(.01,.06)
    x,y=frame(aspect,'bag');q=abs(x-.5);top=u(.36,.495);bot=u(.32,.49)
    if pack:top=u(.23,.39)
    shape=curve(y,[high,high+.08,(high+1)/2,.93,1],[top,top+u(.005,.055),u(.44,.5),bot+.028,bot]);
    edge=high+u(-.045,.055)*(q/.45)**2; bottom=1-u(.015,.11)*(q/.46)**4
    m=(q<shape)&between(y,edge,bottom)
    z=fabric(m,x,y)
    if handled or pack:
        width=u(.115,.365); rad=u(.011,.033) if handled else .037
        if sometimes(.70):
            outer=(q/width)**2+((y-high)/(high+.001))**2;inner=(q/max(.025,width-rad))**2+((y-high)/max(.02,high-1.6*rad))**2
            strap=(outer<1)&(inner>1)&between(y,0,high+.04)
        else:strap=line(x,y,(.5-width,high),(.5-u(0,.075),0),rad)|line(x,y,(.5,0),(.5+width,high),rad)
        z=np.maximum(z,strap*u(.50,1.))
    return z

def shoe(k):
    # Side profile in toe-to-heel coordinates. Heights measured as fractions
    # of the shoe's height. The heel block and sole join below the instep.
    style=rng.random();highheel=False
    if k==7:
        hi=style<.13; aspect=u(1.35,1.98) if hi else u(2.0,3.48)
        rise=u(.40,.64) if not hi else u(.19,.4); toe=u(.58,.79);peak=u(.52,.72)
        knots=[0,.045,.18,peak-.16,peak,peak+.09,.85,.955,1]
        knotvals=[.83,toe+.07,toe,rise,u(0,.09),u(.14,.36),u(.19,.43),u(.06,.29),u(.35,.6)]
        arch=u(.0,.11); heel=u(.01,.10)
    elif k==9:
        aspect=u(.82,1.68) if style<.75 else u(1.55,2.03);highheel=sometimes(.4)
        peak=u(.48,.68); toe=u(.63,.83); ankle=u(.85,.985)
        knots=[0,.045,.18,peak-.145,peak-.055,peak,ankle,ankle+.013,1]
        knotvals=[.86,toe+.05,toe,u(.36,.56),u(.20,.36),u(0,.075),u(.00,.115),u(.30,.48),u(.51,.7)]
        arch=u(.02,.22) if not highheel else u(.24,.52);heel=u(.04,.145) if not highheel else u(.22,.49)
    else:
        highheel=style<.43;aspect=u(.97,1.93) if highheel else u(1.9,3.65)
        arch=u(.19,.55) if highheel else u(.00,.11)
        heel=(u(.20,.51) if highheel else u(.02,.09)); toe=u(.71,.86)
    x,y=frame(aspect,'foot');
    if sometimes(.04): x=1-x
    start=u(.36,.53);stop=u(.76,.91)
    # Underside seen in profile. A cut-away heel has a vertical rear support.
    bottom=curve(x,[0,.06,.16,start,start+.14,stop,stop+.024,.965,1.],
                 [u(.86,.94),.965,1.,.985,.98-arch*.65,.98-arch,.985,.98,u(.81,.94)])
    if k!=5:
        roof=curve(x,knots,knotvals)
        m=between(x,0,1)&between(y,roof,bottom)
        z=fabric(m,x,y)
        if k==7 or sometimes(.45):              # contrast foxing / sole
            outsole=curve(x,[0,.18,start,stop,1],[.81,.89,.91,.9-heel*.53,.79-heel*.40])
            z*=1-u(.025,.155)*(y<outsole)
        return z
    # Sandals have a footbed, sometimes a heel, and one or more straps with
    # air between them. Some casual slides have a single broad vamp.
    sole=curve(x,[0,.10,.30,.47,.82,1],[u(.78,.89),.865,.90,.88,.88-heel,.85-heel])
    m=between(y,sole,bottom)&between(x,0,1);z=fabric(m,x,y)
    rad=u(.017,.047);front=u(.18,.38);back=u(.67,.89)
    if style>.84:                       # pool slide / mule, open at the back
        band=line(x,y,(front-.16,.88),(front+.07,.07),u(.039,.105))|line(x,y,(front+.07,.07),(front+.31,.86),u(.035,.105))
    else:
        junction=(u(.40,.62),u(.25,.53));
        band=line(x,y,(front-.12,.89),junction,rad*u(.8,2.5))|line(x,y,junction,(front+.15,.88),rad*u(.75,1.45));
        if sometimes(.82):band|=line(x,y,junction,(back,.09),rad)
        band|=line(x,y,(back-.13,.12),(back+.095,.045),rad*u(.95,1.6))
        if sometimes(.78):band|=line(x,y,(back+.075,.09),(back+.08,.88-heel),rad*u(.85,2.0))
        if sometimes(.4):band|=line(x,y,(back-.1,.09),(back+.09,.74-heel),rad*u(.7,1.5))
        if sometimes(.30):band|=line(x,y,(.045,.88),(.17,.61),rad*1.45)|line(x,y,(.17,.61),(.31,.87),rad)
    return np.maximum(z,band*u(.67,1.))

# Network-writing functions. All data-dependent normalizations in the network
# operate on the single item being classified, never on a batch of items.
layers=[];weights={}
def layer(name,op,ins,**kw):
    layers.append(dict(name=name,op=op,inputs=[ins] if isinstance(ins,str) else ins,**kw));return name

def dense(name,inp,W,b=None):
    w=np.asarray(W,np.float32);n=len(w);weights[name+'.W']=w.reshape(n,-1);weights[name+'.b']=np.zeros(n,np.float32) if b is None else np.asarray(b,np.float32)
    return layer(name,'dense',inp,out=n)

def conv(name,inp,W,b=None):
    w=np.asarray(W,np.float32);n,_,k,_=w.shape;weights[name+'.W']=w;weights[name+'.b']=np.zeros(n,np.float32) if b is None else np.asarray(b,np.float32)
    return layer(name,'conv',inp,out=n,kernel=[k,k],padding=k//2,stride=1)

def smooth(a):
    p=np.pad(a,1);return (4*a+2*(p[1:-1,:-2]+p[1:-1,2:]+p[:-2,1:-1]+p[2:,1:-1])+p[:-2,:-2]+p[:-2,2:]+p[2:,:-2]+p[2:,2:])/16

def finish_drawing(a,k):
    # Per-template framing: fit the longest foreground extent to the frame.
    # Most clothes are centered; small variations in centering model loose
    # hanging details and natural asymmetry. No other drawing is involved.
    yy,xx=np.nonzero(a>.08)
    if len(xx)==0:return a
    left,right=xx.min(),xx.max();top,bot=yy.min(),yy.max()
    factor=u(.91,1.)*N/max(right-left+1,bot-top+1)
    dx=u(-.026,.026);dy=u(-.03,.03)
    if k in (5,7,9):dy+=u(-.065,.095)
    if k==8:dy+=u(-.035,.055)
    px=(X-.5-dx)*N/factor+(left+right)*.5;py=(Y-.5-dy)*N/factor+(top+bot)*.5
    ix=np.floor(px).astype(int);iy=np.floor(py).astype(int);fx=px-ix;fy=py-iy
    def at(j,i):return a[j.clip(0,N-1),i.clip(0,N-1)]*between(i,0,N-1)*between(j,0,N-1)
    return (1-fy)*((1-fx)*at(iy,ix)+fx*at(iy,ix+1))+fy*((1-fx)*at(iy+1,ix)+fx*at(iy+1,ix+1))

def stencil_templates():
    bank=[]; n=1024
    for k in range(10):
        for i in range(n):
            if k in (0,2,3,4,6): a=body(k)
            elif k==1:a=pants()
            elif k==8:a=bag()
            else:a=shoe(k)
            a=finish_drawing(a,k)
            a=smooth(a.reshape(28,2,28,2).mean((1,3)))
            a=a-a.mean();bank.append((a/(np.sqrt((a*a).mean()+1.e-6)*a.size)).astype(np.float32))
    corr=dense('silhouette_agreement','standard_shape',bank)
    layout=layer('by_cut_pattern','reshape',corr,shape=[10,1,n]);
    best=layer('best_cut','maxpool',layout,size=[1,n],stride=[1,n])
    return layer('shape_scores','reshape',best,shape=[10])

def input_maps():
    # Concave grayscale map: foreground membership saturates for bright pixels,
    # but weak boundaries of dark garments can still take part in the match.
    th=[0,.115,.46];w=np.ones((3,1,1,1));l=conv('photometric_breakpoints','image',w,-np.array(th));l=layer('positive_light','relu',l)
    a=.72/th[1];b=.28/th[2];l=conv('cloth_support',l,np.array([a+b,-a,-b]).reshape(1,3,1,1))
    blur=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
    l=conv('rounded_edges',l,blur.reshape(1,1,3,3));layer('standard_shape','norm',l)

# A library of contrast/texture measurements. Only analytic filters and
# hand-specified regions of interest enter here; the filters compare evidence
# for types of fabric and construction within the *current* image.
def make_cues():
    l=layer('standard_greys','norm','image')
    kernels=[]; names=[]
    def filt(name,taps):
        a=np.zeros((5,5));
        for dy,dx,v in taps:a[2+dy,2+dx]+=v
        names.append(name);kernels.append(a)
    for name,ofs in [('v',(0,1)),('h',(1,0))]:
        dy,dx=ofs
        for d in (1,2):
            taps=[(0,0,1),(dy*d,dx*d,-.5),(-dy*d,-dx*d,-.5)]
            filt(name+str(d)+'p',taps);filt(name+str(d)+'m',[(i,j,-v) for i,j,v in taps]);
        filt(name+'step+',[(dy,dx,.5),(-dy,-dx,-.5)]);filt(name+'step-',[(dy,dx,-.5),(-dy,-dx,.5)])
    f=conv('weave_and_seam_filters',l,np.array(kernels)[:,None]);f=layer('directional_contrast','relu',f)
    # Neighborhoods below are expressed in fractions of the crop, with the
    # origin at its top center. The texture fields stay inside the bodice.
    yy,xx=(np.mgrid[:28,:28]+.5)/28;xx-=.5;q=abs(xx)
    def box(xlo,xhi,ylo,yhi):return between(xx,xlo,xhi)&between(yy,ylo,yhi)
    def region(qlo,qhi,ylo,yhi):return between(q,qlo,qhi)&between(yy,ylo,yhi)
    def normmask(m):return m/max(m.sum(),1.e-12)
    feats=[];fn=[]
    def feat(name,chs,ms):
        a=np.zeros((len(names),28,28))
        for m,coeff in ms:
            for channel,c in chs.items():a[names.index(channel)]+=c*coeff*normmask(m)
        feats.append(a);fn.append(name)
    def channels(*nm):return {s:1. for s in nm}
    panel=region(.065,.178,.28,.76); chest=region(.0,.18,.22,.55);low=region(.0,.18,.58,.80)
    for d in ('h','v'):
        feat(d+'weave',channels(d+'1p',d+'1m'),[(panel,1.)]);
        feat(d+'broad',channels(d+'2p',d+'2m'),[(panel,1.)]);
        feat(d+'detail',channels(d+'1p',d+'1m'),[(chest,1.),(low,-1.)]);
    lower_sides=region(.09,.205,.55,.80); upper_sides=region(.09,.205,.28,.53)
    for d in ('h','v'):
        feat(d+'lower',channels(d+'1p',d+'1m'),[(lower_sides,1.)]);
        feat(d+'upper',channels(d+'1p',d+'1m'),[(upper_sides,1.)]);
    # Detect long continuous bright OR dark opening. Detect at several center
    # positions; keep polarity consistent along its length.
    for t in range(-2,3):
        cx=(t+.5)/28
        for pol in 'pm':
            mid=box(cx-.02,cx+.02,.23,.84)
            feat('zip'+str(t)+pol,channels('v2'+pol),[(mid,1.),(panel,-.52)])
    for t in range(-2,3):
        cx=(t+.5)/28; mid=box(cx-.037,cx+.037,.22,.82)
        feat('placket'+str(t),channels('vstep+','vstep-'),[(mid,1.),(panel,-.70)])
    # Broad local concentration of contrast at the neck, including diagonal
    # collar edges and the opening of a V-neck.
    feat('collar',channels('vstep+','vstep-','hstep+','hstep-'),[(region(0,.11,.085,.25),1.),(chest,-.8)])
    out=dense('tailoring_measurements',f,feats)
    return out,fn

def extra_scores(feats,names):
    # A small ReLU rule circuit. Rules are in comparable units to normalized
    # silhouette agreement (+1 for a perfect match).
    flows=[]; coefficients=[]
    def rule(terms,th,coeff):
        a=np.zeros(len(names));
        for k,v in terms.items(): a[names.index(k)]=v
        flows.append((a,-th));coefficients.append(coeff)
    # Joint vertical and cross-grain detail suggests a woven patterned blouse.
    # Use min(V,H) = H - relu(H-V) rather than firing fully on a one-way stripe.
    # Restrict this test to the lower side panels: a logo across the chest
    # is not evidence of an all-over printed or checked fabric.
    rule({'hlower':1},0,{6:.26,2:-.045,4:-.03})
    rule({'hlower':1,'vlower':-1},0,{6:-.26,2:.045,4:.03})
    rule({'vlower':1,'hlower':-.82},.035,{6:.095})
    # A high-contrast illustration localized on the chest favors tees/sweats.
    rule({'vdetail':.5,'hdetail':.5},.035,{0:.082,2:.095})
    # Horizontal banding is typical of a knitted jumper rather than a jacket.
    rule({'hbroad':1,'vbroad':-.88},.055,{2:.11})
    rule({'collar':1},.06,{4:.052,6:.026,2:-.017})
    l=dense('fabric_evidence',feats,[a for a,b in flows],[b for a,b in flows]);l=layer('fabric_rules','relu',l)
    def readout(name,inp,cs,b=None):
        A=np.zeros((10,len(cs)))
        for j,c in enumerate(cs):
            for i,s in c.items(): A[i,j]=s
        return dense(name,inp,A,b)
    style=readout('fabric_scores',l,coefficients)
    # Best sustained vertical line from center positions, compared to the two
    # side panels. A coat's opening can be a bright zipper OR a dark slit.
    ids=[i for i,s in enumerate(names) if s.startswith(('zip','placket'))];a=np.eye(len(names))[ids];
    z=dense('opening_candidates',feats,a,[-.062]*len(ids));z=layer('salient_openings','relu',z)
    z=layer('opening_array','reshape',z,shape=[1,1,len(ids)]);z=layer('front_opening','maxpool',z,size=[1,len(ids)],stride=[1,len(ids)])
    seam=readout('opening_scores',z,[{4:.24,2:-.12,6:.068,0:-.038}])
    # Shape of the topmost edge is a separate diagnostic from the shape
    # of the front surface: an upstanding collar, round neck and lapel leave
    # different profiles near the hanger. Measure them relative to shoulder
    # tips and the wearer's shoulders, not to an image-wide average.
    yy,xx=(np.mgrid[:28,:28]+.5)/28;q=abs(xx-.5)
    def band(a,b,c,d):return between(q,a,b)&between(yy,c,d)
    def avg(m):return m/max(m.sum(),1)
    neck=avg(band(0,.071,0,.095)); lobe=avg(band(.09,.17,.012,.105)); outer=avg(band(.19,.265,.04,.14));
    cap=avg(band(0,.071,.02,.135))-avg(band(.16,.26,.02,.135)); scoop=lobe-neck
    slope=lobe-outer
    narrowhem=avg(band(0,.12,.81,.96))-avg(band(.31,.47,.81,.96))
    # Shoulder width balanced against cuff/forearm width. A tee's rectangular
    # body is narrower than its upper sleeves; shirts also come in intermediate
    # and rolled sleeve lengths, uncommon in plain tees and basic knits.
    outerarm=band(.34,.49,.21,.32);midarm=band(.34,.49,.44,.57);cuff=band(.34,.49,.76,.90)
    cap_sleeve=avg(outerarm)-avg(midarm);three_quarter=avg(midarm)-avg(cuff)
    jacket_hem=avg(band(.19,.34,.70,.82))-avg(band(.19,.34,.91,.98));
    tilt=avg(band(.24,.34,.71,.89))-avg(band(.24,.34,.36,.54))
    features=[cap,scoop,slope,cap_sleeve,three_quarter,jacket_hem,tilt]
    threshold=[.085,.055,.12,.115,.125,.15,.10]
    h=dense('outline_details','cloth_support',features,-np.array(threshold));h=layer('outline_evidence','relu',h)
    shape_detail=readout('cut_details',h,[{4:.052},{2:.075,0:.05,6:.035,4:-.060},{4:.022},{0:.13,2:-.08,4:-.07},{6:.14,0:-.045},{2:.038},{6:.055}])
    # A folded coat collar usually accompanies a placket or a zipper; a plain
    # round-neck sweater may have almost the same top outline. Reward their
    # conjunction, not just the collar alone: min(A,B) = (A+B-|A-B|)/2.
    stand=dense('standing_collar',h,[[1,0,0,0,0,0,0]]);front=dense('opening_vector',z,[[1.]])
    minus=dense('negative_opening',z,[[-1.]]);dif=layer('collar_without_opening','add',[stand,minus]);dif=layer('unmatched_collar','relu',dif)
    no=dense('remove_unmatched',dif,[[-1.]]);joint=layer('collar_with_zip','add',[stand,no]);joint=readout('structured_front',joint,[{4:.25,6:.025,2:-.055}]);
    # Detect a rounded shirttail at any of several hem lengths. Look for the
    # *same hem edge* at different heights near the panel sides and center.
    # A mere textured patch, or straight hem shifted up, need not satisfy both.
    tails=[]
    for cy in [.72,.79,.86,.93]:
        mid=avg(band(0,.14,cy,cy+.06)); corners=avg(band(.18,.25,cy,cy+.06));
        lower=avg(band(0,.16,cy+.074,cy+.14));upper=avg(band(.17,.25,cy-.09,cy-.026));
        tails.append(.5*(mid-corners)+.25*(upper-corners)+.25*(mid-lower))
    tail=dense('shirttail_edges','cloth_support',tails,[-.36]*len(tails));tail=layer('positive_tail','relu',tail);tail=layer('tail_positions','reshape',tail,shape=[1,1,len(tails)]);tail=layer('rounded_shirttail','maxpool',tail,size=[1,len(tails)],stride=[1,len(tails)]);tail=readout('shirttail_score',tail,[{6:.13,2:-.018}]);
    # Do the lower side pixels end long before the body, or do they continue
    # beside it? Read the spatial distribution of positive material responses.
    M=layer('material_extent','maxpool','cloth_support',size=[1,1],stride=[1,1]);
    def sideband(lo,hi,yt,yb):return avg(band(lo,hi,yt,yb))
    arm=sideband(.33,.475,.62,.83);capband=sideband(.33,.475,.24,.34);bosom=sideband(0,.17,.25,.57);hemfield=sideband(0,.17,.72,.86)
    short=dense('short_sleeve_extent',M,[capband-1.4*arm+.30*hemfield],[-.39]);short=layer('short_sleeve_cut','relu',short);tees=readout('short_sleeve_score',short,[{0:.115,2:-.04,4:-.04}]);
    # Make local specialists comparable. Structural details rank the possible
    # labels *within* a family; they do not vote to send a bag to the coat family.
    base=np.array([.020,0,.015,0,-.008,0,-.041,0,0,0])
    priors=dense('shape_with_priors','shape_scores',np.eye(10),base)
    # A pattern on both upper and lower, off-center panels is all-over
    # evidence. A separate check below tests for plaid by looking for aligned
    # vertical and horizontal lines extending across patches of the torso.
    R=[]
    for orient in ['vertical','horizontal']:
        parts=[]
        for cx in [-.14,0,.14]:
            for cy in [.365,.54,.715]:
                for gap in [1.45/28,2.5/28]:
                    if orient=='vertical':
                        center=between(xx-.5,cx-.022,cx+.022)&between(yy,cy-.095,cy+.095)
                        sides=between(abs(xx-.5-cx),gap-.020,gap+.020)&between(yy,cy-.095,cy+.095)
                    else:
                        center=between(xx-.5,cx-.095,cx+.095)&between(yy,cy-.022,cy+.022)
                        sides=between(xx-.5,cx-.095,cx+.095)&between(abs(yy-cy),gap-.020,gap+.020)
                    a=avg(center)-avg(sides);parts.extend([a,-a])
        R.extend(parts)
    r=dense('woven_lines','standard_greys',R);r=layer('line_contrast','relu',r);nr=len(R)//2
    v=[1/nr]*nr+[0]*nr;horiz=[0]*nr+[1/nr]*nr
    r=dense('woven_directions',r,[v,horiz]);r=dense('crossed_directions',r,[[1,0],[1,-1]]);r=layer('positive_direction_difference','relu',r);r=dense('plaid',r,[[1,-1]],[-.08]);r=layer('repeated_plaid','relu',r);plaid=readout('plaid_score',r,[{6:.32,2:-.025,4:-.025}])
    # Surface tests are alternate explanations of the same material. Max
    # rather than sum their evidence for a shirt, and bound that evidence:
    # even a very strong pattern, alone, cannot establish the garment's cut.
    weave_votes=layer('combine_surface_votes','add',[style,plaid]);row=np.eye(10)[[6]]
    sh=dense('shirt_texture',style,row);pl=dense('shirt_plaid',plaid,row);neg=dense('negative_plaid',pl,[[-1.]])
    diff=layer('texture_difference','add',[sh,neg]);diff=layer('unique_texture','relu',diff);maxtex=layer('one_surface_vote','add',[pl,diff]);lim=.095
    excess=dense('surface_excess',maxtex,[[1]],[-lim]);excess=layer('too_much_texture','relu',excess);sub=dense('bound_vote',excess,[[-1.]])
    bounded=layer('bounded_texture','add',[maxtex,sub]);mask=np.eye(10);mask[6,6]=0
    surf=dense('other_texture_scores',weave_votes,mask);surface_sh=readout('shirt_surface_score',bounded,[{6:1.}]);
    # Strength of upper-body material evidence is evaluated relative to the
    # alternative of a straight unadorned sweatshirt or jacket panel.
    basefabric=dense('unbroken_interiors',feats,[-np.eye(len(names))[names.index('vweave')]-np.eye(len(names))[names.index('hweave')]],[.22]);basefabric=layer('plain_fabric','relu',basefabric);simple=readout('quiet_panel_score',basefabric,[{2:.09,0:.080,4:.065}]);
    # Straight-edge shoulders and squared collar top: run a small explicit
    # bank of neck cut profiles, applied only near the hanger. These measure
    # peaked support over the center against the air just below shoulder slope.
    cuts=[]
    for cx in [-.045,0.,.045]:
        for ytop in [0.,.04,.075]:
            Q=abs(xx-.5-cx);m=between(Q,0,.092)&between(yy,ytop,ytop+.07);air=between(Q,.14,.24)&between(yy,ytop,ytop+.07);under=between(Q,.14,.24)&between(yy,ytop+.14,ytop+.22)
            cuts.append(avg(m)-avg(air)+.28*avg(under))
    nc=dense('collar_tip_locations','cloth_support',cuts,[-.61]*len(cuts));nc=layer('collar_tips','relu',nc);nc=layer('tip_location_list','reshape',nc,shape=[1,1,len(cuts)]);nc=layer('raised_neckline','maxpool',nc,size=[1,len(cuts)],stride=[1,len(cuts)]);nc=readout('coat_neck_shape',nc,[{4:.105,2:-.028}]);
    # Cross-sections of the actual object: test whether fabric still extends
    # beyond each of a series of distances from the midline. This yields a
    # length-sensitive sleeve test even when cuffs are separated from the body
    # by a dark vertical space. Use low-threshold ramps rather than counting
    # all the dark interior space between a dangling arm and the body.
    ext=[];zones=[(.19,.32),(.45,.57),(.70,.82),(.83,.945)]
    tests=[.22,.27,.32,.37,.42]
    for lo,hi in zones:
        for dist in tests:ext.append(avg(band(dist,.50,lo,hi)))
    e=dense('cross_section_probes','cloth_support',ext,[-.042]*len(ext));e=layer('nonempty_extent','relu',e)
    sat=dense('extent_excess',e,np.eye(len(ext)),[-.17]*len(ext));sat=layer('occupied_extent','relu',sat);sat=dense('subtract_extent_excess',sat,-np.eye(len(ext)));e=layer('extent_flags','add',[e,sat]);
    Mx=np.zeros((4,len(ext)))
    for j in range(4):Mx[j,j*len(tests):(j+1)*len(tests)]=1/(.17*len(tests))
    e=dense('widths_by_height',e,Mx);
    e=dense('sleeve_width_changes',e,[[1,-.65,-.35,0],[0,1,-.45,-.55],[-.65,-.35,.65,.35]],[-.085,-.16,-.08]);e=layer('sleeve_width_evidence','relu',e);width=readout('length_sensitive_cuts',e,[{0:.28,2:-.065,4:-.055},{6:.19,0:-.05},{2:.043,0:-.035}]);
    # The floating waistband of a pullover can end just above the sleeve
    # cuffs. A longer straight body hiding the thighs is more typical of a coat.
    flt=avg(band(.32,.47,.80,.94))-avg(band(0,.17,.80,.94));
    cuffgap=.5*avg(band(.36,.47,.7,.88))+.5*avg(band(.11,.20,.7,.88))-avg(band(.24,.32,.7,.88));
    sleeves=dense('cuff_separation','cloth_support',[flt,cuffgap],[-.01,-.18]);sleeves=layer('hanging_cuffs','relu',sleeves);cuffs=readout('pullover_cuffs',sleeves,[{2:.205,4:-.045},{2:.077}]);
    construction=layer('upper_body_recognition','add',[priors,surf,surface_sh,seam,shape_detail,joint,tail,tees,simple,nc,width,cuffs])
    yy,xx=(np.mgrid[:28,:28]+.5)/28;q=abs(xx-.5)
    def mask(xl,xr,yt,yb):return between(xx,xl,xr)&between(yy,yt,yb)
    def avg(m):return m/max(m.sum(),1)
    # Pants have a dark separating space with cloth immediately to either
    # side. A dress instead flares continuously outwards from a narrow waist.
    split=avg(between(q,.077,.145)&between(yy,.62,.89))-avg((q<.04)&between(yy,.62,.89))
    widen=avg(between(q,.168,.32)&between(yy,.71,.915))-avg(between(q,.168,.32)&between(yy,.29,.52))
    # Air trapped between the sole and a strap is characteristic of open
    # footwear. Search multiple heights, since the sole is not registered.
    # Use absolute, low-light foreground support for these topological cues:
    # the dark material inside a closed shoe is not an opening to background.
    solid=conv('look_for_background','image',np.ones((1,1,1,1))*20,[-1]);solid=layer('bright_excess','relu',solid)
    light=conv('absolute_support','image',np.ones((1,1,1,1))*20);excess=conv('remove_excess',solid,-np.ones((1,1,1,1)))
    solid=layer('occupied_pixels','add',[light,excess])
    cutout=conv('enclosed_gaps','cloth_support',np.array([[[[0.,0.,.5,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,-1.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,.5,0.,0.]]]]));cutout=layer('gap_evidence','relu',cutout)
    # Also test larger spaces between footbed and vamp, in a whole section
    # of the shoe (not just on a single pixel or on a pale lace).
    rows=[]
    for cx in [.22,.39,.56,.73,.85]:
        for cy in np.arange(.32,.81,.09):
            for dy in [.085,.17]:
                above=avg(mask(cx-.065,cx+.065,cy-dy-.035,cy-dy+.035));below=avg(mask(cx-.065,cx+.065,cy+dy-.035,cy+dy+.035))
                mid=avg(mask(cx-.055,cx+.055,cy-.03,cy+.03));rows.extend([above-mid,below-mid])
    gaps=dense('space_brackets',solid,rows)
    # min(above - middle, below - middle) = -max(negated evidence)
    inv=dense('space_missing_side',gaps,-np.eye(len(rows)));inv=layer('bracket_pairs','reshape',inv,shape=[1,len(rows)//2,2]);inv=layer('least_supported_side','maxpool',inv,size=[1,2],stride=[1,2]);
    pos=dense('enclosed_spaces',inv,-np.eye(len(rows)//2),[-.25]*(len(rows)//2));pos=layer('strap_spaces','relu',pos);pos=layer('space_list','reshape',pos,shape=[1,1,len(rows)//2]);pos=layer('strongest_space','maxpool',pos,size=[1,len(rows)//2],stride=[1,len(rows)//2])
    open_shoe=readout('open_footwear',pos,[{5:.57,7:-.09,9:-.06}])
    fine_holes=dense('fine_footwear_space',cutout,[avg(mask(.10,.91,.33,.78))],[0]);fine_holes=readout('fine_sandal_evidence',fine_holes,[{5:.88}])
    # A sneaker often has an unbroken solid sidewall all along the foot,
    # outside of which sits a thin sole. Open straps do not fill such a band.
    # Require coverage in all three sections, not just the ankle or the toe.
    pieces=[]
    for cy in np.linspace(.43,.72,8):
        for cx,deltay in [(.205,.055),(.48,0.),(.77,-.014)]:
            pieces.append(-avg(mask(cx-.095,cx+.095,cy+deltay-.060,cy+deltay+.060)))
    wall=dense('sidewall_parts',solid,pieces);wall=layer('sidewall_heights','reshape',wall,shape=[1,8,3]);wall=layer('incomplete_wall','maxpool',wall,size=[1,3],stride=[1,3]);wall=dense('wall_support',wall,-np.eye(8),[-.80]*8);wall=layer('solid_sidewall','relu',wall);wall=layer('sidewall_positions','reshape',wall,shape=[1,1,8]);wall=layer('best_sidewall','maxpool',wall,size=[1,8],stride=[1,8]);wall=readout('closed_footwear',wall,[{7:.44,9:.44,5:-.16}]);
    # A boot's shaft rises above the toe for a broad span and has a cuff at
    # the back; a sneaker normally has only a narrow tongue above the vamp.
    # A dark heel scoop below a solid upper is further evidence for a boot.
    shaft=avg(mask(.59,.91,.245,.41));fore=avg(mask(.12,.40,.245,.41));
    instep=avg(mask(.69,.85,.54,.71));arch=avg(mask(.62,.80,.78,.89));heelblock=avg(mask(.84,.94,.72,.88));
    featboot=dense('boot_profile',solid,[shaft-.55*fore,instep+.7*heelblock-1.35*arch],[-.38,-.73]);featboot=layer('shaft_or_heel','relu',featboot);boot=readout('shaft_evidence',featboot,[{9:.19,7:-.075},{9:.115}]);
    foot_scores=layer('footwear_recognition','add',['shape_scores',open_shoe,fine_holes,wall,boot])
    geom=dense('skirt_and_legs','cloth_support',[split,widen],[-.12,-.055]);geom=layer('leg_gap_and_flare','relu',geom)
    geo_scores=readout('lengthwise_construction',geom,[{1:.32,3:-.045},{3:.34}]);
    # Pants and dresses are long garments but differ in topology. Check for
    # two separate legs all the way down, combined with a straight waistband.
    # In particular, a printed dress is not two legs: check presence of near-
    # black background between legs, not local darkness from dyes or folds.
    waist=avg(band(0,.115,.035,.14)); arms=avg(band(.32,.47,.47,.86));
    common_waist=avg(band(.08,.16,.01,.09))-avg(band(.08,.16,.18,.28))
    noneck=avg(band(0,.055,.01,.105))-avg(band(.09,.16,.01,.105))
    linehem=avg(band(0,.055,.72,.92))-avg(band(.095,.17,.72,.92))
    absf=dense('leg_and_waist_anatomy',solid,[split,-split,common_waist+noneck*.50,-linehem-arms+waist,arms],[-.15,-.20,-.04,-1.01,-.20]);absf=layer('leg_and_waist_rules','relu',absf)
    leg=readout('pant_construction',absf,[{1:.27},{1:-.10},{1:.082},{1:.16},{1:-.14}])
    # The waistband of ordinary trousers runs straight across the top; a
    # bodice or dress has shoulders rising above its neckline instead.
    shoulders=dense('neck_or_waist','cloth_support',[scoop],[-.08]);shoulders=layer('visible_neck','relu',shoulders);shoulders=readout('not_trousers',shoulders,[{1:-.27}]);
    # Width at several lengths distinguishes a torso with a skirt from a
    # bag or a man's top. Favor a dress when a continuous skirt hem extends
    # outwards from a narrow midsection. Penalize lateral cuffs well away
    # from the waist, evidence for two sleeves beside a plain body instead.
    narrow=1.0 # Full foreground membership for a covered position.
    ratio=avg(band(0,.105,.3,.6))-1.3*avg(band(.225,.46,.18,.66))
    dshape=dense('dress_proportions','cloth_support',[ratio,widen-avg(band(.32,.47,.45,.67))*.4,avg(band(0,.12,.78,.96))-avg(band(.18,.34,.33,.55))-avg(band(.33,.48,.58,.86))],[-.62,-.075,-.51]);dshape=layer('skirt_proportions','relu',dshape)
    dress=readout('length_and_skirt',dshape,[{3:.13},{3:.19},{3:.14}]);
    # A bag often has a large vacant region overhead with handles on its
    # flanks, or a wide, almost rectangular body without shoulders. Model
    # these alternatives as conjunctions of local occupancy tests.
    handle=avg(band(.155,.36,.09,.36))-avg(band(0,.105,.10,.32));
    broad_body=avg(band(.26,.435,.52,.77));full_bottom=avg(band(0,.36,.68,.86));
    upper_void=avg(band(0,.37,.07,.27)); topcloth=avg(band(0,.145,.00,.17));
    top_heavy=avg(band(.28,.42,.15,.38))-avg(band(.28,.42,.66,.86))
    handbag=dense('container_profile',solid,[handle+.40*broad_body+.20*full_bottom, broad_body+.45*full_bottom-1.4*upper_void, top_heavy, cap_sleeve, three_quarter],[-.56,-.89,-.15,-.2,-.15]);handbag=layer('container_parts','relu',handbag);hand=readout('body_or_container',handbag,[{8:.34,3:-.055},{8:.27},{3:-.16},{3:-.12},{8:-.12}]);
    # A near-empty band across the full width at the top is strong evidence
    # *against* any full-length dress/pants shape. Horizontally elongated bags
    # can have such a band; using the waist or a good lower-hem match alone
    # would not distinguish them. Allow a large empty center under a handle.
    top_band=avg(band(0,.34,0,.19));upper_center=avg(band(0,.135,.18,.33));
    emptytop=dense('headroom',solid,[-top_band,-upper_center],[.18,.30]);emptytop=layer('visible_headroom','relu',emptytop);emptytop=readout('missing_shoulders',emptytop,[{3:-.72,1:-.66},{3:-.38,1:-.28}]);
    # Distinguish an A-line skirt from the straight torso and shoulder wings
    # of a tee. Compare material to each side at the bust, waist, and hem.
    teecut=avg(band(.24,.43,.19,.333))-avg(band(.24,.43,.68,.89)); arms_top=avg(band(.31,.48,.19,.33))-avg(band(.31,.48,.47,.8));
    teetaper=dense('shoulders_broader_than_skirt',solid,[teecut,arms_top],[-.09,-.22]);teetaper=layer('upper_breadth','relu',teetaper);tee_shape=readout('torso_breadth',teetaper,[{3:-.22},{3:-.18}]);
    coarse=layer('overall_form','add',['shape_scores',geo_scores,leg,shoulders,dress,hand,emptytop,tee_shape]);
    def peak(inp,ids,key):
        a=dense(key+'_collect',inp,np.eye(10)[ids]);a=layer(key+'_array','reshape',a,shape=[1,1,len(ids)]);return layer(key+'_max','maxpool',a,size=[1,len(ids)],stride=[1,len(ids)])
    # For each expert family: max(final member score) == family shape score.
    # Subtracting the max changes neither the order within the family nor the
    # decision of the independent family selector.
    routed=[coarse];
    for key,ids,expert in [('bodices',[0,2,4,6],construction),('shoes',[5,7,9],foot_scores)]:
        keep=np.zeros((10,10));keep[ids,ids]=1
        delta1=dense(key+'_detail',expert,keep);delta2=dense(key+'_unassign',coarse,-keep)
        a=peak(coarse,ids,key+'_form');b=peak(expert,ids,key+'_fine');
        col=np.zeros((10,1));col[ids]=1
        routed.extend([delta1,delta2,dense(key+'_family_score',a,col),dense(key+'_relative',b,-col)])
    layer('class_scores','add',routed)

def build(folder):
    input_maps();stencil_templates();f,n=make_cues();extra_scores(f,n)
    folder.mkdir(exist_ok=True,parents=True)
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(folder/'weights.npz',**weights)

if __name__=='__main__':build(Path(sys.argv[1]))
