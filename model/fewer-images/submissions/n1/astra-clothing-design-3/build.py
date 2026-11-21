"""Build an analytic shape-and-detail classifier. No input files or learned parameters."""
import json
import math
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
layers = []; weights = {}

def layer(name, op, src, **settings):
    layers.append(dict(name=name, op=op, inputs=src if isinstance(src,list) else [src], **settings))
    return name

def weighted(name, op, src, w, b, **kw):
    w = np.asarray(w, dtype=np.float32)
    weights[name+'.W'] = w
    weights[name+'.b'] = np.broadcast_to(np.asarray(b,dtype=np.float32),(len(w),)).copy()
    return layer(name,op,src,out=len(w),**kw)

def point(name, src, w, b):
    return weighted(name,'conv',src,np.asarray(w)[:,:,None,None],b,kernel=[1,1],padding=0,stride=1)

def dense(name, src, w, b): return weighted(name,'dense',src,w,b)
def pool(name, op, src, h, w, sh, sw): return layer(name,op,src,size=[h,w],stride=[sh,sw])
def reshape(name,src,*dims): return layer(name,'reshape',src,shape=list(dims))
def rel(name,src): return layer(name,'relu',src)

# Soft material/background separation, retaining evidence from faint outlines as
# well as stronger fabric. The slopes make a concave piecewise linear tone curve.
z=point('tone_knots','image',[[1],[1],[1]],[0,-.045,-.20]); z=rel('tone_positive',z)
mask=point('fabric',z,[[.69/.045+.26/.20+.05,-.69/.045,-.26/.20]],0)
small=pool('footprint','avgpool',mask,2,2,2,2)
unit=layer('standard_footprint','norm',small)

# Single-instance parametric drawings. Coordinates in the model's own frame
# are in units of garment length (tops, pants) or sole length (footwear).
# All poses, pattern cuts, and camera variations are chosen *before* rendering.
rng=np.random.default_rng(80479)
u=rng.uniform

def choose(p): return rng.random()<p
q=42
axis=(np.arange(q,dtype=float)+.5)/q-.5
xx=axis[None,:]; yy=axis[:,None]

def frame(w,h,shoe=False):
    # Trim scale is determined by the designed object dimensions, not a dataset.
    scale=u(.90,1.035)/max(w,h)
    dx=u(-.042,.042); dy=u(-.052,.052) + (u(-.025,.025) if shoe else 0)
    y=(yy-dy)/scale + h*.5
    shear=u(-.045,.045)
    x=(xx-dx)/scale+(y-h*.5)*shear
    if choose(.5): x=-x
    return x, y

def interp(y, ys, widths): return np.interp(y,ys,widths)

def upper(k):
    # Half body widths are measured from centre line. 'arm' is cuff-centre
    # distance from that line. Garments can be hanging straight or slightly spread.
    penalty=0.; rounded=0.; open_amt=0.; tank=False; hood=False
    t=rng.random(); collar='crew'; cuff_slope=u(.012,.050)
    if k==0:  # jersey tee or sleeveless top
        width=u(.275,.405); sh=width*u(.87,1.12); waist=width*u(.76,1.05); hem=width*u(.81,1.12)
        under=width; shoulder_y=u(.045,.125); neck=u(.085,.180); nd=u(.048,.17)
        s_len=u(.245,.46); arm=sh+u(.09,.22); cuff=u(.048,.098); bend=u(.02,.07)
        if t<.20:
            tank=True; sh=width*u(.60,1.03); shoulder_y=.018; s_len=.0; nd=u(.11,.27); neck=sh*u(.45,.80)
            penalty=-.018
        elif t<.235:
            s_len=u(.48,.70); arm=sh+u(.07,.16); penalty=-.042
        if choose(.23): collar='v'
        rounded=u(0,.055)
    elif k==2:  # long knitted top, full sleeves, often drawn in at wrists and hem
        width=u(.28,.408); sh=width*u(.77,1.02); waist=width*u(.88,1.07); hem=width*u(.71,1.0)
        under=width; shoulder_y=u(.052,.15); neck=u(.070,.158); nd=u(.025,.125)
        s_len=u(.82,1.045); cuff=u(.040,.080); arm=hem+cuff+u(-.006,.12); bend=u(.012,.07)
        if t<.11: s_len=u(.62,.83); penalty=-.022
        elif t<.22: collar='high'; shoulder_y+=.05
        elif t<.31: collar='v'; nd=u(.11,.21)
        elif t<.40: hood=True; shoulder_y+=.06
        rounded=u(0,.045)
    elif k==4:  # overcoat, blazer or jacket
        width=u(.255,.36); sh=width*u(.85,1.10); under=width
        waist=width*u(.83,1.02); hem=width*u(.92,1.22)
        s_len=u(.70,.96); cuff=u(.048,.078); arm=hem+cuff+u(-.07,.068); bend=u(.025,.080)
        shoulder_y=u(.082,.173); nd=u(.018,.105); neck=u(.066,.135)
        if t<.33:  # short jacket
            width=u(.32,.405); under=width; sh=width*u(.84,1.10); waist=width*u(.88,1.04); hem=width*u(.85,1.08)
            s_len=u(.85,1.045); arm=hem+cuff+u(-.02,.10)
        elif t<.50:  # long coat
            width=u(.205,.278); sh=width*u(.84,1.05); waist=width*u(.82,.98); under=width
            hem=width*u(1.02,1.40); s_len=u(.54,.76); arm=hem+u(-.01,.065)
        if choose(.36): open_amt=u(.025,.13)
        if choose(.29): collar='v'; nd=u(.11,.25)
        elif choose(.28): hood=True
        else: collar='high'
        rounded=u(0,.055)
    elif k==6:  # shirt, shirt blouse. Tail below sleeve, frequent centre fastening.
        width=u(.265,.372); sh=width*u(.82,1.09); waist=width*u(.76,1.045); hem=width*u(.84,1.11)
        shoulder_y=u(.066,.136); under=width; neck=u(.062,.132); nd=u(.042,.16)
        s_len=u(.75,1.015); cuff=u(.037,.065); arm=hem+cuff+u(-.02,.116); bend=u(.02,.072)
        rounded=u(.042,.14); collar='v' if choose(.53) else 'crew'
        if t<.21:
            s_len=u(.285,.47); arm=sh+u(.07,.16); cuff=u(.046,.09); penalty=-.018
        elif t<.42:
            s_len=u(.49,.78); arm=hem+cuff+u(.008,.125)
        elif t<.48:
            tank=True; sh=width*u(.62,1.02); shoulder_y=.025; neck=sh*u(.45,.72); nd=u(.08,.24); penalty=-.06
        if choose(.30): rounded=u(.0,.05)
    else:  # dress. Bodice, waist, and skirt can have very different widths.
        width=u(.153,.254); under=width; waist=width*u(.59,.93); sh=width*u(.73,1.07)
        hem=waist*u(1.38,2.9); neck=u(.053,.12); nd=u(.035,.14); shoulder_y=u(.032,.093)
        s_len=u(.13,.29); cuff=u(.032,.072); arm=sh+u(.038,.119); bend=u(.015,.047)
        if t<.51:
            tank=True; sh=width*u(.63,1.13); shoulder_y=u(.0,.033); neck=sh*u(.42,.87); nd=u(.070,.20)
        elif t<.76:
            s_len=u(.37,.73); arm=width+cuff+u(.015,.095)
        if choose(.20):  # straight skirt / shift
            hem=width*u(.90,1.38); waist=width*u(.81,1.03)
        if choose(.27): collar='v'
        rounded=u(.0,.08)
    if tank: arm=0; cuff=0
    hb=max(hem,waist,under,sh,arm+cuff)
    fullh=max(1,s_len+.025)
    x,y=frame(2*hb,fullh); x=np.abs(x)
    # Torso side seam and bottom hem.
    pit_y=u(.20,.35) if k!=3 else u(.13,.225)
    waist_y=u(.47,.69) if k!=3 else u(.28,.50)
    side=interp(y,[0,shoulder_y,pit_y,waist_y,.87,1.],[sh*.84,sh,under,waist,.21*waist+.79*hem,hem])
    top=shoulder_y*np.clip((x-neck)/max(.02,sh-neck),0,1)
    if collar=='high' or hood:
        top=np.where(x<neck*1.04,.005, np.maximum(top,.045))
    if collar=='v': notch=nd*np.clip(1-x/neck,0,1)
    elif collar=='high': notch=nd*.25*np.clip(1-(x/neck)**2,0,1)
    else: notch=nd*np.sqrt(np.clip(1-(x/neck)**2,0,1))
    # Collar at the back gives a partly filled rather than wholly empty neckline.
    notch_fill=u(.05,.6) if not tank else u(0,.15)
    hem_y=1-rounded*np.minimum((x/max(.02,hem))**2,1.)
    torso=(x<=side)&(y>=top)&(y<=hem_y)
    val=torso*np.where(y<notch,notch_fill,1.)
    if open_amt:
        slit=open_amt*np.clip((y-u(.48,.78))/.38,0,1)
        val*=np.where(x<slit,u(0,.35),1.)
    if not tank:
        # Outer sleeve line turns at the elbow; inner line rejoins the side seam at the armpit.
        sl=max(shoulder_y+.076,s_len); elbow=shoulder_y+.52*(sl-shoulder_y)
        root=sh*.88; joint=.45*sh+.55*(arm+cuff)+bend
        outer=interp(y,[shoulder_y*.77,shoulder_y,elbow,sl,sl+cuff_slope],[root,sh,joint,arm+cuff,arm-cuff])
        in_y=min(pit_y,sl-.035)
        inner=interp(y,[in_y,(sl+in_y)*.5,sl+cuff_slope],[under*.92,(under*.92+arm-cuff)*.5+u(-.012,.025),arm-cuff])
        sleeve=(x<=outer)&((x>=inner)|(y<in_y))&(y>=shoulder_y*.77)&(y<=sl+cuff_slope)
        val=np.maximum(val,sleeve)
    return val,penalty


def trousers():
    # Waist, hips and two independently outlined leg tubes joined at the crotch.
    t=rng.random(); wb=u(.135,.235); hip=wb*u(1.00,1.24); crotch=u(.245,.45)
    spread=u(.012,.09); leg=hip*u(.43,.68); foot=leg*u(.47,.91); knee=leg*u(.63,.94)
    pen=0.
    if t<.18: foot=leg*u(.9,1.37); knee=leg*u(.84,1.08); pen=-.015
    if t>.94: crotch=u(.38,.51); foot=leg*u(.77,1.1); pen=-.023
    end=foot+spread; x,y=frame(2*max(hip,end+foot),1); x=np.abs(x)
    outer=interp(y,[0,.14,crotch,.64,1],[wb,hip,hip*.99,.52*hip+.48*(end+foot),end+foot]); inner=interp(y,[crotch,crotch+.14,.70,1],[0,spread*.27,max(.005,end-knee),spread]);
    v=(x<=outer)&((x>=inner)|(y<crotch))&(y>=.013*(1-np.minimum(x/wb,1)))&(y<=1-.022*np.abs(x-end)/foot)
    return v,pen


def bag():
    t=rng.random(); pen=0.
    if t<.69:  # tote, shopper, handbag, satchel
        w=u(.68,1.36); top=u(.11,.49); wb=w*.5*u(.78,1.0); wt=w*.5*u(.72,1.0)
        x,y=frame(w,1.); a=np.abs(x)
        side=interp(y,[top,top+.085,.82,.935,1],[wt*.92,wt,w*.50,wb,wb*u(.78,.96)])
        lip=top+u(-.025,.14)*np.clip(1-(x/wt)**2,0,1)
        v=(a<side)&(y>=lip)&(y<=1-.022*(a/wb)**3)
        # Handle: upper half of an elliptical loop, with ends attached to rim.
        hw=wt*u(.47,.93); ht=top+u(.012,.12); thick=u(.019,.052)
        radial=np.sqrt((x/hw)**2+((y-ht)/ht)**2)
        loop=(np.abs(radial-1)<thick/max(.08,min(hw,ht)))&(y<=ht+.016)
        if choose(.15):  # a second, offset handle on the far side
            far=np.sqrt(((x+.045)/hw)**2+((y-ht)/(.85*ht))**2)
            loop|=(np.abs(far-1)<thick/max(.08,min(hw,ht)))&(y<=ht)
        return v|loop,pen
    if t<.82:  # backpack or upright travel bag
        w=u(.57,.92); top=u(.035,.15); x,y=frame(w,1); a=np.abs(x)
        co=interp(y,[top,top+.10,top+.25,.61,.92,1],[.19,.72,.93,1.,.96,.70])*w/2
        loop=np.abs(np.sqrt((x/(.15*w))**2+((y-top*1.25)/(top*1.25))**2)-1)<.22
        return ((a<=co)&(y>=top)&(y<1))|loop, -.012
    if t<.92:  # clutch, wallet, briefcase; slight rounding
        h=u(.46,.92); x,y=frame(1,h); a=np.abs(x)
        co=interp(y,[0,h*.08,.50*h,.91*h,h],[.42,.49,.50,.49,.42]); return (a<=co)&(y>=0)&(y<=h),-.012
    # Cross-body bag with a high, sometimes sloping strap.
    w=u(.58,1.1); top=u(.36,.65); x,y=frame(w,1); a=np.abs(x); wt=.5*w
    co=interp(y,[top,top+.05,.88,1],[.87,1,.98,.74])*wt
    body=(a<co)&(y>top+.05*(1-(x/wt)**2))&(y<=1)
    loop=np.abs(np.sqrt(((x+.14*w*(1-y/top))/(wt*.87))**2+((y-top)/top)**2)-1)<u(.043,.12)
    return body|(loop&(y<top+.02)), -.02


def shoe(k):
    t=rng.random(); pen=0.
    if k==7:  # low lace-up, runner, slip-on, or high-top
        h=u(.33,.59); tongue=u(.46,.69); ankle=u(.70,.89); back=u(.66,.99); notch=u(.40,.76)
        toe=u(.24,.46); sole=u(.09,.23); rise=u(.004,.047); arch=u(.0,.04)
        if t<.14: h=u(.54,.77); back=u(.85,1.); notch=u(.62,.89); tongue=u(.51,.68); pen=-.02
        if t>.85: h=u(.25,.41); notch=u(.28,.61); toe=u(.30,.50)
    elif k==9:  # boot: raised shaft over the heel, optionally with a separate heel
        h=u(.55,1.02); toe=u(.10,.32); sole=u(.045,.17); rise=u(.007,.06); arch=u(.0,.085)
        if t<.16: h=u(.42,.60)
        if t>.89: h=u(.96,1.23); pen=-.016
    else:  # sandal styles range from flat slides to heeled multi-strap sandals
        h=u(.32,.68); toe=u(.10,.25); sole=u(.055,.16); rise=u(.01,.065); arch=u(.0,.065)
        if t<.27: h=u(.20,.43); sole=u(.105,.27)
        elif t>.65: h=u(.48,.87)
    x,y=frame(1,h,True); x=x+.5; z=h-y # height above lowest point
    bot=interp(x,[0,.045,.16,.43,.59,.74,.90,.98,1],[h*.16,rise+.013,rise*.23,0,arch*.42,arch,0,rise*.47,h*.13])
    if k==7:
        top=h*interp(x,[0,.045,.19,.37,tongue,ankle,.95,1],[.14,toe*.85,toe,toe+u(.08,.21),1.,notch,back,back*.75]);
        # Slant back seam slightly and round the tip. Collar sometimes dipping low.
        v=(z>=bot)&(z<=top)&(x>=0)&(x<=1.-.047*np.maximum(z/h-.2,0))
        return v,pen
    heelcut=(k==9 and choose(.45)) or (k==5 and t>.54 and choose(.81))
    heelheight=h*(u(.13,.36) if k==9 else u(.22,.54)) if heelcut else arch
    archx=u(.66,.81); joint=u(.48,.63); heelx=u(.77,.94)
    floor=bot+interp(x,[joint,archx,heelx,min(heelx+.036,.985)],[0,heelheight,.92*heelheight,0])
    if k==9:
        shaft=u(.39,.67); heel_back=u(.92,1.01); high_front=u(.81,1.00); high_back=u(.87,1.)
        # A projecting toe, instep, and near-vertical rise at the front of the shaft.
        a=shaft+u(-.09,.04); b=shaft+u(-.04,.095); c=shaft+u(-.01,.12)
        front=np.interp(z,[0,h*toe*.55,h*toe,h*(toe+.13),h*.52,h*.78,h],[.018,0,.14,.36,a,b,c])
        backx=interp(z,[0,.2*h,.47*h,h],[.96,1,heel_back,heel_back+u(-.09,.02)])
        top=h*interp(x,[min(b,c)-.06,c+.04,heel_back],[high_front,high_front,high_back]);
        return (x>=front)&(x<=backx)&(z>=floor)&(z<=top),pen
    # Base of the sandal: thick sole with or without heel. Footbed on a slight ramp.
    ramp=heelheight*interp(x,[.33,archx,.92],[0.,1.,1.])
    bed=bot+h*sole+ramp
    base=(z>=floor)&(z<=bed)&(x>=0)&(x<=1)
    # Tubular straps drawn as tilted bands. Instead of filling the entire upper,
    # leave the gaps between these individual pieces as negative space.
    def strap(x0,z0,x1,z1,rad):
        vx=x1-x0; vz=z1-z0; f=np.clip(((x-x0)*vx+(z-z0)*vz)/(vx*vx+vz*vz+1e-9),0,1)
        return (x-x0-f*vx)**2+(z-z0-f*vz)**2<=rad**2
    foot=h*sole; rr=u(.022,.058); apex=u(.27,.51)
    if t<.27: # slide or thong: low triangular gap across vamp
        v=base|strap(u(.06,.22),foot,apex,h*.90,rr)|strap(apex,h*.90,u(.50,.79),foot,rr*u(.90,1.6))
        if choose(.38): v|=(x>.095)&(x<apex)&(z>foot)&(z<h*(.31+.45*x))
        return v,pen
    # Toe strap over a small peep-toe opening.
    f_end=u(.26,.51); frontheight=h*u(.25,.57)
    v=base|strap(u(.04,.20),foot,apex,frontheight,rr)|strap(apex,frontheight,f_end,foot+.02,rr*u(.90,1.7))
    hx=u(.77,.95); ax=u(.53,.77); heel_base=h*sole+heelheight
    if t>.54 or choose(.66):
        v|=strap(hx,heel_base,hx+u(-.12,.015),h*.89,rr*u(.7,1.6))
        v|=strap(ax,h*.85,hx-.025,h*.93,rr*u(.65,1.18))
        v|=strap(ax,h*.85,u(.36,.69),foot+.035,rr*u(.6,1.16))
        if choose(.54): v|=strap(ax,h*.85,apex,frontheight,rr*u(.7,1.7))
        if choose(.26): v|=(x>hx-.10)&(x<hx+.02)&(z>=heel_base)&(z<h*.77)
    else:
        v|=strap(apex,frontheight,u(.58,.74),h*.85,rr)|strap(u(.58,.74),h*.85,hx,heel_base,rr)
    return v,pen

# Each dense row is ONE separately antialiased drawing. Normalized correlation
# with the input, not a classifier fitted to a collection of labeled pictures.
groups=4; pergroup=768; D=14*14
W=np.empty((10*groups*pergroup,D),dtype=np.float32); B=np.empty(len(W),np.float32)
for i in range(len(W)):
    k=i//(groups*pergroup)
    if k in (0,2,3,4,6): v,pen=upper(k)
    elif k==1: v,pen=trousers()
    elif k==8: v,pen=bag()
    else: v,pen=shoe(k)
    pic=v.reshape(14,3,14,3).mean(axis=(1,3)).astype(np.float64)
    pad=np.pad(pic,1); pic=.76*pic+.06*(pad[2:,1:-1]+pad[:-2,1:-1]+pad[1:-1,2:]+pad[1:-1,:-2])
    pic-=pic.mean(); pic/=D*np.sqrt(np.mean(pic*pic)+1.e-8)
    W[i]=pic.ravel(); B[i]=pen
matches=dense('drawing_correlations',unit,W,B)
z=reshape('by_cut_and_class',matches,10,groups,pergroup)
z=pool('best_cut','maxpool',z,1,pergroup,1,pergroup)
best=reshape('best_evidence',pool('best_pose','maxpool',z,groups,1,groups,1),10)
robust=reshape('repeated_evidence',pool('consistent_pose','avgpool',z,groups,1,groups,1),10)
shape=dense('shape_max_weight',best,.62*np.eye(10),0)
shape=layer('shape_score','add',[shape,dense('shape_consistency_weight',robust,.38*np.eye(10),0)])

# The remaining weights are spatial measurement formulas and a handwritten rule
# table. All thresholds and gains are fixed assumptions, not measurements.
raw=layer('standard_light','norm','image')
# Make zero-sum derivative and thin-ridge filters. We use both contrast polarities.
def stencil(a):
    a=np.asarray(a,float); return a/np.maximum(np.sum(a[a>0]),1e-8)
vertical=stencil(np.outer([1,2,1],[-1,2,-1])); horizontal=vertical.T
vx=stencil(np.outer([1,2,1],[-1,0,1])); vy=vx.T
bank=[vertical,horizontal,vx,vy]
f=np.array([sign*k for k in bank for sign in [1,-1]])
z=weighted('cloth_filters','conv',raw,f[:,None],0,kernel=[3,3],padding=1,stride=1)
z=rel('cloth_halfwaves',z)
energy=point('cloth_edges',z,np.repeat(np.eye(4),2,axis=1),0)

# Assemble compact measurement vectors (multiple branches lifted into the same vector space).
values=[]; terms={}
def save(name,src,arr,bias=0.):
    assert name not in terms
    terms[name]=(src,np.asarray(arr,dtype=float).ravel(),bias)
    values.append(name)

def region(n,x0,x1,y0,y1):
    v=np.zeros((n,n)); v[y0:y1,x0:x1]=1.; return v/max(1,v.sum())

def rect(n,x0,x1,y0,y1): return region(n,x0,x1,y0,y1)
# Contrast between two panels and a thin placket: long-range coherence is
# more specific to a fastening than simply counting texture in the middle.
for x in range(7,21):
    for w in [1,3]:
        mid=rect(28,x-w//2,x+w//2+1,9,23)
        fl=.5*(rect(28,x-w//2-3,x-w//2,9,23)+rect(28,x+w//2+1,x+w//2+4,9,23))
        save(f'line{x}_{w}',raw,mid-fl)
# A pair of collar wings / V-neck contrasts diagonally with the chest below.
for deep in [4,7,10]:
    for xc in [12,14,16]:
        a=np.zeros((28,28)); b=np.zeros_like(a)
        for j in range(deep):
            off=round(1.+(deep-j)*.45)
            for xp in [xc-off,xc+off]:
                a[j+2,xp-1:xp+1]+=1; b[j+5,xp-1:xp+1]+=1
        save(f'collar{deep}_{xc}',raw,a/a.sum()-b/b.sum())
# Knit ribbing, woven stripes/checks and texture; features from central fabric,
# away from most silhouette boundaries. Also a dedicated sole contrast.
for d in range(4):
    a=np.zeros((4,28,28)); a[d]=rect(28,9,19,10,21); save('grain'+str(d),energy,a)
for d in range(4):
    a=np.zeros((4,28,28)); a[d]=rect(28,5,23,13,18); save('upperedge'+str(d),energy,a)
for y in range(15,26):
    save('sole'+str(y),raw,rect(28,4,24,y,y+1)-rect(28,4,24,y-3,y-1))
# Bilateral differences in block occupancy, split between shoulders and hem.
for y in range(0,14,2):
    for x in range(0,6,2):
        save(f'asym{y}_{x}',small,rect(14,x,x+2,y,y+2)-rect(14,12-x,14-x,y,y+2))
# Opposed strips test for a dark interval *bounded on both sides* by material.
# Test vertical, diagonal, and horizontal directions on several spatial scales.
# This is a local description of negative space, not a particular sandal outline.
for y in range(5,11):
    for x in range(2,12):
        for dx,dy in [(0,2),(2,0),(1,2),(-1,2),(0,3)]:
            if not (0<=y-dy and y+dy<14 and 0<=x-dx<14 and 0<=x+dx<14): continue
            a=np.zeros((14,14)); a[y,x]=-1.; a[y-dy,x-dx]=.5; a[y+dy,x+dx]=.5
            save(f'gap{y}_{x}_{dx}_{dy}',small,a)
# Relative distribution of foreground in garment sections.
def mr(name,*areas):
    save(name,small,sum(g*rect(14,*r) for g,r in areas))
mr('legs',(.5,(4,6,9,13)),(.5,(8,10,9,13)),(-1,(6,8,9,13)))
mr('hem_spread',(.5,(1,4,11,13)),(.5,(10,13,11,13)),(-.5,(1,4,5,7)),(-.5,(10,13,5,7)))
mr('yoke_spread',(.5,(0,3,3,5)),(.5,(11,14,3,5)),(-.5,(0,3,9,12)),(-.5,(11,14,9,12)))
mr('handle',(.5,(2,5,2,5)),(.5,(9,12,2,5)),(-1,(5,9,2,5)))
mr('midheight',(1,(3,11,5,8)))
mr('short_sides',(.5,(0,3,7,11)),(.5,(11,14,7,11)))
mr('solid_belly',(1,(4,10,7,11)))
# Regional texture in normalized-light units on the body, including local front seam.
for j,(a,b) in enumerate([(11,17),(5,10),(18,23)]):
    A=np.zeros((4,28,28)); A[0]=rect(28,a,b,9,22); save('seam_zone'+str(j),energy,A)

def combine_sources(name,entries):
    # entries are scalar linear measurements on one of a few fixed feature maps.
    sources=list(dict.fromkeys(s for s,a,b in entries)); parts=[]
    for j,s in enumerate(sources):
        size=next(len(a) for source,a,b in entries if source==s)
        mat=np.zeros((len(entries),size)); bias=np.zeros(len(entries))
        for r,(source,a,b) in enumerate(entries):
            if source==s: mat[r]=a; bias[r]=b
        parts.append(dense(name+'_from_'+str(j),s,mat,bias))
    return layer(name,'add',parts) if len(parts)>1 else parts[0]
measure=combine_sources('measurements',[terms[v] for v in values]); vi={v:i for i,v in enumerate(values)}

# Small algebra for assembling one-sided and saturated evidence. Each named
# quantity is a linear combination of the preceding layer's rectified units.
class Stage:
    def __init__(self,dim): self.W=[]; self.B=[]; self.dim=dim
    def hinge(self,terms,bias=0):
        row=np.zeros(self.dim)
        for i,c in terms.items(): row[i]=c
        j=len(self.B); self.W.append(row); self.B.append(bias); return {j:1.}
    def clip(self,terms,lo,hi):
        p=self.hinge(terms,-lo); q=self.hinge(terms,-hi)
        return plus((1/(hi-lo),p),(-1/(hi-lo),q))
    def run(self,name,src): return rel(name+'_positive',dense(name,src,self.W,self.B))

def plus(*pieces):
    ans={}
    for c,p in pieces:
        for j,x in p.items(): ans[j]=ans.get(j,0.)+c*x
    return ans

def avg(ps): return plus(*[(1/len(ps),p) for p in ps])

s=Stage(len(values)); F={}
def v(n): return {vi[n]:1.}
def absclip(n,lo,hi): return plus((1,s.clip(v(n),lo,hi)),(1,s.clip({vi[n]:-1.},lo,hi)))
# Saturating many weak measurements makes individual shadows or holes less decisive.
for w in [1,3]:
    for x in range(7,21): F[f'l{x}_{w}']=absclip(f'line{x}_{w}',.06,.84)
for n in values:
    if n.startswith('collar'): F[n]=absclip(n,.11,1.0)
for n in ['legs','hem_spread','yoke_spread','handle','midheight','short_sides','solid_belly']:
    F[n]=s.clip(v(n),0,1)
for j in range(4):
    for prefix in ['grain','upperedge']:
        n=prefix+str(j); F[n]=s.clip(v(n),.055 if j<2 else .12,.46 if j<2 else .82)
F['seam_centre']=s.clip(plus((1,v('seam_zone0')),(-.5,v('seam_zone1')),(-.5,v('seam_zone2'))),.008,.32)
F['collar']=avg([p for n,p in F.items() if n.startswith('collar')]); F['gaps']=avg([s.clip(v(n),.35,.78) for n in values if n.startswith('gap')]);
F['asym']=avg([absclip(n,.08,.74) for n in values if n.startswith('asym')]); F['sole']=avg([s.clip(v(n),.22,1.35) for n in values if n.startswith('sole')]);
for n,subset in [('central',range(11,17)),('flank',[7,8,9,18,19,20])]:
    F[n]=avg([F[f'l{x}_{w}'] for x in subset for w in [1,3]])
front=s.run('detail_tests',measure)
t=Stage(len(s.W)); H={}
H['placket']=t.clip(plus((1,F['central']),(-.67,F['flank']),(.26,F['seam_centre'])),.055,.40)
H['lapel']=t.clip(plus((.62,F['collar']),(.80,F['central']),(-.42,F['flank'])),.20,.74)
H['collar']=t.clip(F['collar'],.18,.57)
H['weave']=t.clip(plus((.52,F['grain0']),(.48,F['grain1'])),.18,.73)
H['stripe']=t.clip(plus((.60,F['grain1']),(.4,F['grain3']),(-.36,F['grain0'])),.19,.64)
H['gap']=t.clip(F['gaps'],.013,.105)
H['shoe_texture']=t.clip(plus((.5,F['upperedge0']),(.5,F['upperedge1'])),.15,.74)
H['sole']=t.clip(F['sole'],.045,.25)
H['legs']=t.clip(F['legs'],.12,.74)
H['hem']=t.clip(F['hem_spread'],.06,.51)
H['handle']=t.clip(F['handle'],.12,.64)
H['asym']=t.clip(F['asym'],.115,.46)
H['sleevecut']=t.clip(plus((1,F['yoke_spread']),(-.40,F['short_sides'])),.12,.67)
H['thin']=t.clip(plus((-1,F['solid_belly'])), -.85,-.40)
logic=t.run('part_combinations',front)

# Corrections are deliberately small in correlation units. In particular a
# fastening helps a shirt or jacket over a sweater, but cannot make a boot a coat.
C=np.zeros((10,len(t.W))); cb=np.array([.009,0,.022,0,-.011,-.006,-.007,.010,0,.005])
def rule(c, **kw):
    for n,a in kw.items():
        for j,w in H[n].items(): C[c,j]+=a*w
rule(0,sleevecut=.019,placket=-.035,lapel=-.012,hem=-.012)
rule(1,legs=.089,hem=-.020,asym=-.023)
rule(2,placket=-.059,lapel=-.016,stripe=.012,legs=-.026,sleevecut=-.021)
rule(3,hem=.067,legs=-.045,asym=-.013)
rule(4,placket=.061,lapel=.033,weave=-.012,legs=-.020,sleevecut=-.032)
rule(5,gap=.108,thin=.033,asym=.022)
rule(6,placket=.034,collar=.026,weave=.023,hem=-.009)
rule(7,gap=-.046,sole=.036,shoe_texture=.019,asym=.009)
rule(8,handle=.077,asym=-.028,legs=-.031)
rule(9,gap=-.050,asym=.031,shoe_texture=-.015)
correction=dense('construction_details',logic,C,cb)
layer('scores','add',[shape,correction])
(out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
np.savez(out/'weights.npz',**weights)
