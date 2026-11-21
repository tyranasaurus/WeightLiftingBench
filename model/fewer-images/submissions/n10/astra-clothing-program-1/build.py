"""Data-free builder. Each drawing describes just one possible manufactured item.
The rasterizer uses height units for the front-view patterns and a unit square
plus a specified aspect ratio for side-view / bag patterns.
"""
from pathlib import Path
import sys
import numpy as np

rng = np.random.default_rng(51872)
u = rng.uniform
n = 48
t = (np.arange(n) + .5) / n
x, y = np.meshgrid(t, t)
f = abs(2*x-1)

def curve(pos, values):
    return np.interp(y, pos, values)

def choice(values):
    return values[rng.integers(len(values))]

def front(k):
    # a flat or lightly draped garment hanging by its shoulders
    style = u()
    if k == 0:
        tank = style < .20
        length = u(.22,.44) if not tank else 0
        if style > .97: length = u(.48,.75)
        width = u(.49,.78)
        waist = u(.84,1.02)
        hip = u(.94,1.12)
        shoulder = u(.80,1.06)
        neck = u(.31,.61) if not tank else u(.51,.82)
        dip = u(.005,.064) if not tank else u(.06,.25)
    elif k == 3:
        tank = style < .48
        length = u(.12,.30) if style < .79 else u(.32,.70)
        width = u(.24,.43)
        waist = u(.66,1.03)
        hip = u(1.12,1.88) if u()<.68 else u(.90,1.29)
        shoulder = u(.80,1.11)
        neck = u(.33,.65) if not tank else u(.48,.90)
        dip = u(.005,.10) if not tank else u(.035,.20)
    elif k == 2:
        tank = False
        length = u(.84,1.08) if style<.91 else u(.61,.85)
        width = u(.48,.73)
        waist = u(.91,1.06)
        hip = u(.84,1.05)
        shoulder = u(.84,1.06)
        neck = u(.25,.55)
        dip = u(.004,.054)
    elif k == 4:
        tank = False
        length = u(.85,1.08) if style<.57 else u(.54,.88)
        width = u(.47,.69) if style<.57 else u(.37,.59)
        waist = u(.84,1.02)
        hip = u(.95,1.16)
        shoulder = u(.87,1.05)
        neck = u(.24,.48)
        dip = u(.0,.048)
    else:
        tank = style < .09
        length = u(.19,.44) if style<.23 else (u(.57,.84) if style<.44 else u(.82,1.02))
        width = u(.45,.70)
        waist = u(.83,1.02)
        hip = u(.94,1.13)
        shoulder = u(.82,1.04)
        neck = u(.28,.60) if not tank else u(.44,.77)
        dip = u(.0,.065) if not tank else u(.04,.18)
    # Use an isotropic physical frame; sleeves can extend beyond the torso.
    ar = 1.40 if k != 3 else .90
    a = abs((x-.5)*ar + u(-.011,.011)*(2*y-1))
    b = width/2
    shoulder *= b
    waist *= b
    hip *= b
    neck *= shoulder
    cap = u(.035,.11) * (1 if k!=3 else .72)
    top = cap * np.clip((a-neck) / max(shoulder-neck,.01),0,1) + dip * np.maximum(0,1-(a/neck)**2)
    if k == 4 and u() < .7 or k == 6 and u() < .43 or k == 2 and u() < .14:
        # raised stand collar or hood: shoulder line starts below neck
        top += u(.026,.098)*np.clip((a-.64*neck)/(.45*neck),0,1)
    if tank:
        armhole = b * (u(.18,.37) if k!=3 else u(.14,.33))
        body = curve([.03,.10,.19,.30,.55,.80,1], [shoulder,shoulder-armhole*.48,b-armhole,b*.97,waist,hip*.96,hip])
    else:
        underarm = u(.21,.38) if k!=3 else u(.12,.22)
        body = curve([.005,cap,underarm,.54,.79,1], [shoulder*.68,shoulder,b,waist,(waist+hip)/2,hip])
    if k == 3:
        seam = u(.23,.52)
        skirt = curve([0,seam,min(seam+.23,.77),.98,1], [b,waist,waist+(hip-waist)*u(.34,.71),hip,hip*.99])
        body = np.where(y>seam,skirt,curve([.0,cap,.16,seam],[shoulder*(.71 if not tank else 1),shoulder*(.78 if tank else 1),b,waist]))
    if k == 6:
        fall = u(.008,.14) if u()<.78 else u(-.024,.025)
    elif k == 4:
        fall = u(-.025,.045)
    else:
        fall = u(-.025,.043)
    hem = 1-fall*(a/max(hip,.03))**3 - max(0,-fall)
    hem -= u(0,.024)*(.5+np.sin(x*14+u(0,5))*.5)
    mask = (a<body)&(y>=top)&(y<=hem)
    if not tank:
        long = length > .48 and (k!=3 or length>.32)
        # sleeve axis from shoulder to wrist; the cuff is cut crosswise
        slant = u(.010,.126) if long else u(.046,.18)
        if k==3: slant *= .64
        s0 = shoulder*.91
        s1 = max(hip if length>.73 else waist, b)*u(.94,1.12) + slant if long else shoulder + slant
        w0 = b*u(.36,.60)
        w1 = b*(u(.16,.29) if long else u(.32,.53))
        if k == 4: w1 *= 1.14
        z0 = cap + w0*.55
        z1 = length - (max(0,length-1) * .1)
        # If a cuff hangs below the hem, compress vertically to keep it in frame.
        if z1>1: extra=z1-1; z1=1.; mask=(a<body)&(y>=top)&(y<=hem-extra)
        for side in [0,1]:
            bias = u(-.023,.023)
            reach = s1+u(-.022,.022)
            end = z1 + bias
            if not long: end = max(end,z0+.035)
            dy=max(end-z0,.045); dx=reach-s0; norm=np.hypot(dy,dx)
            ax=((a-s0)*dx + (y-z0)*dy)/(norm*norm)
            cross=((a-s0)*dy-(y-z0)*dx)/norm
            w=w0+(w1-w0)*np.clip(ax,0,1)
            sleeve=(ax>=0)&(ax<=1)&(abs(cross)<w)
            shoulderhead = ((a-s0)/w0)**2+((y-z0)/(z0-cap*.7+.008))**2 < 1
            mask |= (sleeve | shoulderhead) & ((x<.5)==bool(side))
    return mask, ar

def trousers():
    ar=.75
    w=u(.135,.25)
    a=(x-.5)*ar
    rise=u(.19,.43)
    crotch=u(.003,.020)
    waist=w*u(.72,.98)
    hem=w*u(.51,1.18)
    gap=u(.02,.55)*w
    if u()<.13: gap=u(-.04,.03)*w
    outer=curve([0,.04,.18,rise,.63,.86,1],[waist,waist*1.02,w,w*.99,(w+hem)/2,hem,hem*u(.90,1.05)])
    inner=np.maximum(0,(y-rise)/(1-rise))**u(.45,1.5)*gap+crotch
    drift = u(-.021,.021)*y
    if u()<.27:
        # Skinny jeans and leggings: the feet separate while the legs narrow.
        inner += np.maximum(0,y-rise)**.8*w*u(.18,.43)
        outer *= 1-y*u(.0,.14)
    if u()<.21:
        # Draping can bring the inner seams into contact except at the ankles.
        inner *= np.clip((y-u(.62,.93))*8,0,1)
    v=abs(a-drift)
    mask=(v<outer)&((v>=inner)|(y<rise))&(y>u(.0,.022)*(a/w)**2)&(y<1-u(0,.023)*(x>.5))
    return mask, ar

def ribbon(xa,ya,xb,yb,w,ar):
    dx=(xb-xa)*ar; dy=yb-ya
    m=np.clip(((x-xa)*ar*dx+(y-ya)*dy)/max(.00001,dx*dx+dy*dy),0,1)
    return ((x-xa-m*(xb-xa))*ar)**2+(y-ya-m*dy)**2<w*w

def shoe(k):
    r=u()
    if k==7:
        high=r<.15
        ar=u(1.28,2.00) if high else u(1.90,3.40)
        toe=u(.50,.74) if not high else u(.64,.83)
        tongue=u(.47,.71); rear=u(.12,.47) if not high else u(.09,.39)
        tongueHt=u(0,.21); hollow=u(.10,.39)
        up=np.interp(x,[0,.05,.17,tongue-.18,tongue-.035,tongue,tongue+.06,.84,.965,1],
                   [toe+.105,toe,toe-u(0,.08),toe*.64,tongueHt+.09,tongueHt,rear+hollow,rear+hollow*.84,rear,rear+.16])
        if r>.88: # loafers / canvas slip-ons
            up=np.interp(x,[0,.07,.24,.43,.54,.62,.82,.95,1],[toe+.14,toe,toe-.08,.18,.08,.46,.49,.12,.29])
        heelcurve = u(.0,.11); tip=u(.06,.20)
        bottom=np.interp(x,[0,.06,.18,.48,.73,.94,1],[1-tip,1-tip*.51,1,.99,1-heelcurve,1-heelcurve,1-heelcurve-.08])
        mask=(y>=up)&(y<=bottom)&(x>np.maximum(0,(y-.80)*.08))&(x<1-np.maximum(0,.66-y)*.08)
    elif k==9:
        ar=u(.96,1.83) if r<.83 else u(1.62,2.35)
        shaft=u(.39,.68); topback=u(.85,.99); toel=u(.66,.84)
        neck=u(.42,.68)
        up=np.interp(x,[0,.065,.19,shaft-.12,shaft,shaft+.035,topback-.05,topback,1],
                     [toel+.085,toel,toel-.02,neck+.08,neck-u(.13,.24),u(.0,.09),u(.0,.07),.03,neck+.14])
        # rear of the shaft can be straight or bowed, not a triangular wedge
        back=curve([0,.15,neck,.74,.91,1],[topback,topback-.01,u(.89,.98),.995,.99,.965])
        high=r<.46
        heel=u(.18,.48) if high else u(.022,.14)
        heelwidth=u(.045,.24)
        bottom=np.interp(x,[0,.09,.27,.39,.62,.98-heelwidth,.99-heelwidth,1],
                       [.91,.986,1,.986,1-heel*.72,1-heel,.985,.97]);
        mask=(y>=up)&(y<=bottom)&(x<back)
    else:
        high=r<.30; backstrap=u()<.62
        ar=u(1.22,2.02) if high else u(1.85,3.98)
        lift=u(.33,.72) if high else u(.005,.17)
        s=np.interp(x,[0,.09,.27,.40,.67,.85,1],[.86,.92,.96,.95,.95-lift*.65,.93-lift,.90-lift])
        thick=u(.04,.17) * (1.35 if not high else 1)
        mask=(y>s-thick*.5)&(y<s+thick*.5)
        heel=u(.04,.20)
        if high: mask |= (x>.97-heel)&(x<.98)&(y>=s)&(y<=.98)
        if high and u()<.17: mask |= (x>.33)&(y>=s)&(y<.98)
        base=u(.06,.30)
        if not backstrap and u()<.55:
            # moulded vamp of a slide, open at the heel
            top=np.interp(x,[.0,base,base+.11,.59,.72,.99],[.84,.64,.43,.17,.50,.83])
            mask |= (x<.72)&(x>.025)&(y>=top)&(y<=s)
        else:
            peak=u(.36,.64); py=u(.27,.58) if not high else u(.42,.68)
            for j in range(int(choice([1,2,2,3]))):
                xa=base+u(-.10,.13); w=u(.032,.095)
                mask |= ribbon(xa,.90,peak,py, w,ar) | ribbon(peak,py,peak+u(-.01,.17),.93-lift*.35,w*u(.9,1.2),ar)
            if backstrap:
                strapx=u(.68,.92); sy=u(.02,.33); sw=u(.025,.085)
                mask |= ribbon(.94,.87-lift,strapx,sy,sw,ar)
                mask |= ribbon(peak,py,strapx,sy+u(.0,.23),sw,ar)
                if u()<.58: mask |= ribbon(strapx,sy,.52,.91-lift*.3,sw,ar)
                if u()<.46: mask |= ribbon(.93,.83-lift,.96,.31,sw*1.2,ar) | ribbon(.96,.31,strapx,sy,sw,ar)
        if u()<.35 and not high: mask |= (x>.86)&(y<s)&(y>.36+(x-.90)**2*9)
        # Flip-flops, mules, sport slides, and clogs are low at the heel.
        # Their side elevation is quite different from a buckled ankle strap.
        if not high and u()<.48:
            ar=u(2.25,4.6)
            heel=1-u(.06,.20)*x
            sole=heel-u(.10,.24)
            mask=(y<heel-.07*(abs(x-.5)*2)**6)&(y>sole-.07*(abs(x-.5)*2)**6)
            peak=u(.37,.65); py=u(.01,.27); rear=u(.69,.9)
            thick=u(.09,.29)
            upper=np.interp(x,[.015,.09,.24,peak,rear,.98],[.75,.68,.56,py,.65,.74])
            lower=np.interp(x,[.015,.09,.24,peak,rear,.98],[.83,.86,.77,py+thick,.81,.84])
            if u()<.5: lower=np.where(x<peak,sole+.02,lower)
            mask |= (x>.02)&(x<rear)&(y>=upper)&(y<lower)
            if u()<.48: mask |= ribbon(peak,py,peak+.08,.85,thick*.24,ar)
    return mask, ar

def bag():
    have = u()<.77
    start = u(.20,.59) if have else u(.0,.09)
    ar = u(.74,1.37) if have else u(.98,1.84)
    if u()<.13: ar*=.76
    topw=u(.33,.49); base=u(.34,.49); widest=u(.44,.50)
    p=u(.06,.24)
    a=abs(x-.5-u(-.046,.046)*(y-.58))
    size=curve([start,start+(1-start)*p,start+(1-start)*.68,.94,1], [topw,widest,(widest+base)/2,base,base*u(.78,1)])
    top=start+u(-.08,.16)*(1-np.minimum(1,(a/topw)**2))
    mask=(a<size)&(y>=top)&(y < 1-u(.005,.06)*(a/base)**4)
    if have:
        half=u(.13,.40); midpoint=.5+u(-.08,.08); sw=u(.014,.056)
        z=np.clip((x-midpoint)/half,-1,1)
        if u()<.71:
            # inverted U with nearly vertical sides: an elliptic arch
            radius=np.sqrt(((x-midpoint)/half)**2+((y-start)/(start+.006))**2)
            handle=(abs(radius-1)<sw/max(.1,(half+start)/2))&(y<=start+.04)
        else:
            handle=ribbon(midpoint-half,start+.02,midpoint,.015,sw,ar)|ribbon(midpoint,.015,midpoint+half,start+.02,sw,ar)
        mask |= handle
    return mask, ar

# The same descriptor implementation is used on each drawing and on each input.
common = '''import numpy as np

def sample(im, box, side):
    left,top,right,bottom = box
    t=(np.arange(side)+.5)/side
    xx=left-.5+(right-left+1)*t; yy=top-.5+(bottom-top+1)*t
    xx=np.clip(xx,0,im.shape[1]-1); yy=np.clip(yy,0,im.shape[0]-1)
    ix=xx.astype(int); iy=yy.astype(int)
    ax=(xx-ix).astype(np.float32); ay=(yy-iy).astype(np.float32)[:,None]
    p=im[:,ix]*(1-ax)+im[:,np.minimum(ix+1,im.shape[1]-1)]*ax
    return p[iy]*(1-ay)+p[np.minimum(iy+1,im.shape[0]-1)]*ay

def bounds(binary):
    rows=np.flatnonzero(binary.any(1)); cols=np.flatnonzero(binary.any(0))
    if len(rows)==0 or len(cols)==0: return (0,0,binary.shape[1]-1,binary.shape[0]-1)
    return (cols[0],rows[0],cols[-1],rows[-1])

def pool(a,s):
    return a.reshape(a.shape[0]//s,s,a.shape[1]//s,s).mean((1,3))

def describe(a):
    p=pool(a,2)
    b=a>.44
    h=np.maximum.accumulate(b,axis=1)&np.maximum.accumulate(b[:,::-1],axis=1)[:,::-1]
    v=np.maximum.accumulate(b,axis=0)&np.maximum.accumulate(b[::-1],axis=0)[::-1]
    # A blur for silhouette agreement, plus envelopes which ignore interior detail.
    padded=np.pad(p,1,mode='edge')
    blur=.5*p+.125*(padded[:-2,1:-1]+padded[2:,1:-1]+padded[1:-1,:-2]+padded[1:-1,2:])
    h=h.astype(np.float32); v=v.astype(np.float32)
    pieces=[blur.ravel()*12, pool(h,4).ravel()*13, pool(v,4).ravel()*13,
            a.mean(0).reshape(16,2).mean(1)*19, a.mean(1).reshape(16,2).mean(1)*19]
    return np.rint(np.concatenate(pieces)).astype(np.int8)

def fabric(a,v):
    # Contrast-normalized gradient histograms of the neck, front closure, hem,
    # and two contiguous fabric panels. Each descriptor is image-local.
    lum=np.log(.07+v)
    collar=np.maximum(a,.07)
    pieces=[]
    for stride in (1,2):
        p=np.pad(lum,stride,mode='edge'); n=len(lum)
        gx=p[stride:stride+n,2*stride:]-p[stride:stride+n,:n]
        gy=p[2*stride:,stride:stride+n]-p[:n,stride:stride+n]
        mag=np.sqrt(gx*gx+gy*gy)+.00001
        strength=np.minimum(.9,mag)*collar
        angle=np.arctan2(gy,gx)
        orient=np.stack([strength*np.maximum(0,np.abs(np.cos(angle-th*np.pi/4))-.65)/.35 for th in range(4)])
        for y0,y1,x0,x1,rows,cols,weight in ((0,12,4,28,3,6,1.20),(8,28,10,22,4,3,1.28),(24,32,8,24,2,4,.68),(8,24,4,28,2,6,.77)):
            blocks=orient[:,y0:y1,x0:x1].reshape(4,rows,(y1-y0)//rows,cols,(x1-x0)//cols).mean((2,4))
            # The balance of orientations matters even with faint stitching.
            # Retain some absolute contrast; a plain chest is also evidence.
            norm=.053+.48*blocks.sum(0)+.18*blocks.sum(0).mean()
            pieces.append(np.minimum(2.0,blocks/norm).ravel()*weight)
    return np.rint(22*np.concatenate(pieces)).astype(np.int8)
'''

# Parametric textile drawings, used only in the front-view garment subgroup.
# Coordinates here are fractions of the individual garment's trimmed box.
def materialmask(z,k):
    h,w=z.shape; py,px=np.indices(z.shape,dtype=float); py=(py+.5)/h; px=(px+.5)/w
    p=px-.5-u(-.024,.024); q=py
    def diffuse(lo,hi): return u(lo,hi)*choice([-1,1])
    val=np.ones_like(z)*u(.48,.76)
    # Broad illumination, cylindrical torso, and independently hanging sleeves.
    val += u(-.16,.16)*p+u(-.10,.10)*q-u(.025,.20)*(2*p)**2
    val += u(.04,.17)*np.cos((abs(p)-.39)*32)*(abs(p)>.28)
    def stripe(center,width): return np.exp(-.5*((p-center)/width)**2)
    def horizontal(center,width): return np.exp(-.5*((q-center)/width)**2)
    def segment(ax,ay,bx,by,width):
        dx=bx-ax; dy=by-ay; t=np.clip(((p-ax)*dx+(q-ay)*dy)/(dx*dx+dy*dy+.000001),0,1)
        return np.exp(-.5*((p-ax-t*dx)**2+(q-ay-t*dy)**2)/width**2)
    # Fabric folds: short, tapered and not restricted to center positions.
    for j in range(int(u(2,8))):
        cx=u(-.36,.36)+u(-.12,.12)*(q-.5)+u(.005,.055)*np.sin(q*u(3,9)+u(0,6))
        val+=diffuse(.035,.20)*stripe(cx,u(.012,.043))*horizontal(u(.18,1),u(.08,.36))
    r=u()
    if k in (0,2):
        # Neckband forms an arc, V or a shallow crew. It has a back facing and
        # a line of stitches. Hood/turtleneck variants are higher necked.
        width=u(.115,.26); depth=u(.036,.192); off=u(-.025,.015)
        if u()<.28: neck=off+depth*(1-abs(p)/width)
        else: neck=off+depth*np.sqrt(np.maximum(0,1-(p/width)**2))
        valve=(abs(p)<width)
        val+=diffuse(.12,.40)*horizontal(neck,u(.010,.026))*valve
        val+=u(-.42,.10)*(q<neck)*valve
        if k==2:
            height=u(.86,.96)
            val+=diffuse(.035,.21)*(q>height)+diffuse(.022,.085)*horizontal(height,.014)
            if u()<.27:
                val+=u(.025,.11)*np.cos(q*u(50,170)+u(0,6))
        if u()<(.43 if k==0 else .15):
            # A chest print with a boundary and simple light/dark motifs.
            dy=u(.25,.48); sx=u(.085,.24); sy=u(.105,.25)
            emblem=((p/sx)**2+((q-dy)/sy)**2<1) if u()<.55 else ((abs(p)<sx)&(abs(q-dy)<sy))
            val+=emblem*(diffuse(.12,.41)+u(.035,.13)*np.cos(p*u(32,95)+q*u(30,95)))
    else:
        # Slender overlapped fronts on woven shirts; single/double-breasted
        # jackets, zips and open fronts on coats. Each is a separate template.
        shirt=k==6; spread=u(.085,.20); depth=u(.10,.21) if shirt else u(.10,.40)
        low=u(-.017,.017)+u(-.034,.034)*(q-.5)+u(.001,.014)*np.sin(5*q+u(0,6))
        buttons=r<(.78 if shirt else .65)
        if not shirt and r>.80: low+=u(.055,.18)*choice([-1,1])
        placket=u(.014,.043) if shirt else u(.018,.066)
        val+=(p<low)*u(-.14,.14)
        val+=(abs(p-low)<placket)*diffuse(.045,.22)
        val+=stripe(low+placket,u(.006,.016))*diffuse(.06,.27)
        if u()<.65: val+=stripe(low-placket,.011)*diffuse(.03,.15)
        if buttons:
            spacing=u(.10,.16) if shirt else u(.15,.27)
            for j in range(9):
                yb=depth+spacing*j
                val+=stripe(low,u(.010,.024))*horizontal(yb,.014 if shirt else .019)*diffuse(.19,.51)
        else: val+=diffuse(.11,.34)*stripe(low,.012)
        kind=u()
        if shirt and kind<.75:
            # Two collar points on the upper chest; a short open V down to
            # the first button and, sometimes, a patch pocket.
            for sign in (-1,1):
                ax=sign*spread; tip=sign*spread*u(1.04,1.39); ht=depth*u(.53,1.2)
                val+=segment(ax,.021,tip,ht,.014)*diffuse(.15,.30)
                val+=segment(tip,ht,sign*.018,depth*.45,.015)*diffuse(.10,.26)
                val+=segment(ax,.021,sign*.017,depth,.019)*diffuse(.15,.33)
            if u()<.3: val+=segment(.095,.28,.245,.27,.013)*diffuse(.08,.25)
        elif not shirt and kind<.64:
            # Broad lapel facing, with notched outward edge and turned front.
            for sign in (-1,1):
                a=spread*sign; b=u(.13,.25)*sign; end=depth*u(.85,1.24)
                val+=segment(a,.015,b,end*.39,.020)*diffuse(.13,.33)
                val+=segment(b,end*.39,sign*.015,end,.019)*diffuse(.16,.44)
                val+=segment(a,.03,sign*.012,end,.028)*diffuse(.08,.28)
        else:
            # Mandarin / stand collar or raised jacket collar.
            val+=horizontal(depth*.52,u(.008,.021))*(abs(p)<spread)*diffuse(.13,.37)
            val+=horizontal(.027,.015)*(abs(p)<spread)*diffuse(.1,.30)
        if k==4 and u()<.55:
            ht=u(.50,.73)
            for sign in (-1,1): val+=segment(.12*sign,ht+.06,.25*sign,ht+u(-.06,.07),.014)*diffuse(.08,.33)
        if shirt and u()<.34:
            # Fine check or pinstripe pattern across the two panels.
            rate=u(38,112); val+=u(.045,.19)*np.cos(p*rate+.6*(p<low)+q*u(-4,4))
            if u()<.80: val+=u(.04,.19)*np.cos(q*u(44,131)+.5*(p>low))
    # Tone-on-tone construction: a placket or facing may be barely visible
    # on light fabric. Keep these versions as independent drawings too.
    base=u(.40,.90)
    val=base+(val-base)*np.exp(u(np.log(.08),np.log(1.16)))
    return np.clip(val,0,1.2)*z

# Only computation on an individual template is performed here. Aspect, mask
# descriptor and squared norm each belong to that same template.
helpers={}
exec(common,helpers)

def makebank():
    desc=[]; aspects=[]; classes=[]; textile=[]; ids=[]
    for k in range(10):
        count=[2600,1200,2600,3200,2800,3400,3000,2000,2000,2200][k]
        for j in range(count):
            if k==1: mask,ar=trousers()
            elif k==8: mask,ar=bag()
            elif k in (5,7,9): mask,ar=shoe(k)
            else: mask,ar=front(k)
            box=helpers['bounds'](mask)
            asp=ar*(box[2]-box[0]+1)/(box[3]-box[1]+1)
            a=helpers['sample'](mask.astype(np.float32),box,32)
            if k in (0,2,4,6):
                v=materialmask(a,k)
                # Optical blur and sampling down to the problem resolution.
                p=np.pad(v,1,mode='edge'); v=.52*v+.12*(p[2:,1:-1]+p[:-2,1:-1]+p[1:-1,2:]+p[1:-1,:-2])
                v/=max(.01,np.sort(v.ravel())[-20]); v=np.minimum(1.4,v)
                textile.append(helpers['fabric'](a,v)); ids.append(len(desc))
            desc.append(helpers['describe'](a)); aspects.append(np.log(asp)); classes.append(k)
    return dict(pattern=np.stack(desc), aspect=np.array(aspects,np.float32), label=np.array(classes,np.int8),
                textile=np.stack(textile), links=np.array(ids,np.int32))

solution = common + '''

def sat(x):
    return np.clip(x,0,1)

def largest(v):
    # Keep the main article, not isolated sharpening specks or a detached tag.
    h,w=v.shape
    seed=np.flatnonzero(v)
    seen=np.zeros((h,w),bool)
    keep=np.zeros((h,w),bool); best=0
    for s in seed:
        row=int(s//w); col=int(s%w)
        if seen[row,col]: continue
        stack=[(row,col)]; seen[row,col]=True; group=[]
        while stack:
            r,c=stack.pop(); group.append((r,c))
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                rr=r+dr; cc=c+dc
                if 0<=rr<h and 0<=cc<w and v[rr,cc] and not seen[rr,cc]:
                    seen[rr,cc]=True; stack.append((rr,cc))
        if len(group)>best:
            best=len(group); keep[:]=False
            for rr,cc in group: keep[rr,cc]=True
    return keep

def readitem(raw):
    level=np.sort(raw.ravel())[-16]
    im=np.minimum(1.4,raw/max(.005,level))
    p=np.pad(im,1,mode='edge')
    sm=.64*im+.09*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])
    weak=sm>.029; m=largest(weak)
    # Reattach details near the main component. Do not discard a detached strap
    # or a second piece of a dark object that is otherwise well supported.
    p=np.pad(m,2); near=m.copy()
    for r in range(5):
        for c in range(5): near |= p[r:r+28,c:c+28]
    m |= weak & near
    if m.sum()<.77*weak.sum(): m=weak
    left,top,right,bottom=bounds(m & (sm>.056))
    # Estimate placement from the substantial silhouette of the item. A thread
    # or one projecting corner should not change the aspect ratio very much.
    row=m[:,left:right+1].sum(1); col=m[top:bottom+1].sum(0)
    rr=np.flatnonzero(row>=2); cc=np.flatnonzero(col>=2)
    if len(rr)>0:
        top=max(top,rr[0]-1); bottom=min(bottom,rr[-1]+1)
    if len(cc)>0:
        left=max(left,cc[0]-1); right=min(right,cc[-1]+1)
    box=(left,top,max(left,right),max(top,bottom))
    ar=(box[2]-box[0]+1)/(box[3]-box[1]+1)
    soft=sat((sm-.019)/.13)*m
    a=sample(soft,box,32)
    val=sample(im,box,32)
    return a,val,ar

def detect(a,v,ar):
    # Semantic evidence, in dimensionless units within the object's own box.
    # Texture and placket detectors use local contrast, not garment colour.
    m=a>.46; yy,xx=np.indices((32,32)); pos=np.arange(32)
    rev=m[:,::-1]
    sym=1-np.mean(abs(a-a[:,::-1]))/(.16+a.mean())
    b=m.any(0); start=np.where(b,m.argmax(0),32)
    end=np.where(b,32-m[::-1].argmax(0),0)
    ymin=m.argmax(1); ymax=31-rev.argmax(1)
    widths=np.where(m.any(1),ymax-ymin+1,0)/32
    mh=np.maximum.accumulate(m,1)&np.maximum.accumulate(rev,1)[:,::-1]
    mv=np.maximum.accumulate(m,0)&np.maximum.accumulate(m[::-1],0)[::-1]
    inside=mh & mv
    holes=(1-a)*inside
    topvoid=np.maximum(0, (1-a[:16,8:24]).mean()-(1-a[16:28,8:24]).mean())
    shoulder=widths[5:11].mean(); waist=widths[14:20].mean(); hips=widths[24:29].mean()
    # A handbag has a loop over a wide, largely contiguous pouch. A backpack
    # with no loop typically has rounded shoulders rather than upper sleeves.
    sides=(a[18:27,:6].mean()+a[18:27,26:].mean())*.5
    loop=sat((topvoid-.18)/.48)*sat((holes[:19,6:26].mean()-.09)/.42)
    topcenter=start[12:20].mean(); flanks=(start[4:10].mean()+start[22:28].mean())/2
    # Footwear: low forward toe box, long sole, raised quarters at the rear.
    heights=np.maximum(0,end-start)
    hLeft=heights[2:10].mean(); hRight=heights[22:30].mean()
    diff=abs(hRight-hLeft)/32
    symmetry=sat((sym-.70)/.21)
    shoeShape=sat((diff-.13)/.30)*sat((.93-sym)/.22)
    if hLeft>hRight: a=a[:,::-1]; v=v[:,::-1]; m=m[:,::-1]; start=start[::-1]; end=end[::-1]; mv=mv[:,::-1]; holes=holes[:,::-1]
    # Is the sole separated from the upper by open air? Vertical fill is
    # especially informative about sandals: it doesn't count the arch below.
    windows=np.sum((1-a[4:28,3:29])*mv[4:28,3:29])/max(10,mv[4:28,3:29].sum())
    apertures=sat((windows-.14)/.36)
    # Monotone shaft atop the heel versus a padded heel below a sneaker tongue.
    roofFront=start[2:10].mean()/32; roofBack=start[23:29].mean()/32
    roofMid=start[16:22].mean()/32
    shaft=sat((.30-roofBack)/.25)*sat((roofFront-.45)/.26)
    collarDip=sat((roofBack-roofMid-.025)/.23)
    arch=sat(((.5*(end[3:10].mean()+end[27:31].mean())-end[18:25].mean())/32-.055)/.20)
    mass=a.mean(); shoe=sat((ar-1.06)/.85)*sat((.97-sym)/.20)
    score=np.zeros(10,np.float64)
    score[5]+=3.8*apertures+1.3*sat((.42-mass)/.22)
    score[7]+=.9*collarDip+1.2*sat((ar-1.85)/.78)*(1-apertures)-1.8*apertures
    score[9]+=1.7*shaft+1.0*arch-.9*collarDip-2.6*apertures
    score[[5,7,9]]+=1.4*shoeShape+.8*shoe-2.6*symmetry*sat((2-ar)/.75)
    score[8]+=4*loop+1.4*symmetry*sat((ar-.88)/.45)+1.1*sides*sat((ar-.8)/.55)-2.4*shoeShape
    score[[0,2,3,4,6]]-=2.3*shoeShape+2.8*loop
    # Trouser leg separation extends upwards from the hem. Unlike a handle,
    # the bottom of this gap is open, and it has material on BOTH sides.
    mid=a[19:31,14:18].mean(); legs=min(a[19:31,5:12].mean(),a[19:31,20:27].mean())
    split=sat((legs-mid-.12)/.55)
    score[1]+=4.3*split+1.1*sat((.54-ar)/.18)-2*sat((ar-.65)/.30)
    score[[0,2,3,4,6]]-=2.2*split
    # Dress proportions. A waist indentation with a flaring skirt is strong
    # evidence, even in a relatively short dress with cap sleeves.
    flare=sat((hips-waist-.04)/.25)
    dress=sat((.64-ar)/.24)
    score[3]+=2.3*dress+1.7*flare*sat((.80-ar)/.3)-.8*sat((ar-.80)/.18)
    # Re-establish left/right garment coordinates; the statistics below are
    # mirror invariant apart from rounding of the central strip.
    p=np.pad(v,((0,0),(3,3)),mode='edge')
    axial=np.zeros(32)
    for dx in (-2,0,2):
        strip=v[:,15+dx:17+dx].mean(1)
        l=v[:,12+dx:14+dx].mean(1); r=v[:,18+dx:20+dx].mean(1)
        # A vertical ridge or trough, not just a light half and dark half.
        d=((l+r)*.5-strip)/(.15+(strip+l+r)/3)
        axial[8:28]=np.maximum(axial[8:28],abs(d[8:28]))
    # Compare the sustained central opening with general vertical folds.
    smooth=(p[:,1:33]+2*p[:,2:34]+3*p[:,3:35]+2*p[:,4:36]+p[:,5:37])/9
    field=(np.abs(.5*(smooth[:,8:22]+smooth[:,14:28])-smooth[:,11:25]))/(smooth[:,11:25]+.17)
    ridge=(.5*(p[:,:32]+p[:,6:38])-smooth)/(.15+smooth)
    def line(lo,hi,c1,c2):
        reg=ridge[lo:hi,c1:c2]
        energy=(np.abs(reg).mean(0)*.52 + np.abs(reg.mean(0))*.48)
        return energy.max()
    background=(line(9,27,7,12)+line(9,27,20,25))/2
    plack=line(9,28,13,19)
    persistence=min(line(9,18,13,19),line(18,28,13,19))
    opening=sat((.62*plack+.38*persistence-.42*background-.046)/.19)
    # Paired diagonals running down from either side of the collar (lapels).
    lap=0.
    for depth in (7,11,15):
        ry=np.arange(2,depth+2); off=np.rint(np.linspace(4.7,.8,depth)).astype(int)
        for center in (15,16):
            left=ridge[ry,center-off]; right=ridge[ry,center+off]
            lap=max(lap,.5*abs(left.mean()+right.mean())+.25*(abs(left).mean()+abs(right).mean()))
    lapel=sat((lap-.095)/.29)
    raised=sat((flanks-topcenter-1.9)/4.4)
    collar=raised*sat((.75-holes[:6,11:21].mean())/.45)
    # Neck band vs broad open shirt collar: an open top is either V-shaped
    # or cut deeply into the body, not a high-standing neck.
    lowneck=sat((topcenter+.5-flanks)/5.0)
    # Repeated checks or stripes are more typical of shirting than outerwear.
    lum=v[9:27,9:23]; cont=lum+.19
    vx=np.abs(lum[:,2:]-lum[:,:-2])/(.5*(cont[:,2:]+cont[:,:-2]))
    vy=np.abs(lum[2:]-lum[:-2])/(.5*(cont[2:]+cont[:-2]))
    checks=sat((min(vx.mean(),vy.mean())-.12)/.28)
    gx=np.abs(v[:,2:]-v[:,:-2]); gy=np.abs(v[2:]-v[:-2]);
    vdetail=np.mean(gx[9:25,8:22])/(.13+lum.mean()); hdetail=np.mean(gy[9:25,9:23])/(.13+lum.mean())
    smoothKnit=sat((.16-max(vdetail,hdetail))/.13)
    # Cuffs, high hem at the sides, and soft shirttails.
    centerHem=np.median(end[12:20]); edgeHem=.5*(np.median(end[8:11])+np.median(end[21:24]))
    tail=sat((centerHem-edgeHem-.8)/2.8)
    sleeveBottom=a[21:28,:5].mean()+a[21:28,27:].mean()
    short=sat((shoulder-hips-.11)/.28)*sat((.5-sleeveBottom)/.32)
    score[0]+=1.2*short+.45*lowneck+.48*(1-opening)-.8*collar*opening
    score[2]+=1.12*(1-opening)+.57*smoothKnit+.55*lowneck-.95*opening-.62*lapel*collar
    score[4]+=2.05*opening+1.30*collar+1.05*lapel+.52*sat((.81-ar)/.24)-.72*tail-.42*checks-.8*lowneck
    score[6]+=.95*opening+.69*tail+1.30*checks+.40*lapel+.38*sat((.8-ar)/.28)-.42*collar*lapel
    return score

def construction(a,v,ar):
    # Long garment fasteners can be visible as just one edge: not every coat
    # has a dark stripe bounded on both sides. Look for a coherent edge along
    # gently tilted center lines. Contrast is relative to noise on that image.
    eps=.035
    lum=np.log(.09+v)
    lum=(lum+np.roll(lum,1,axis=0)*.5+np.roll(lum,-1,axis=0)*.5)/2
    y=np.arange(8,28)
    gx=np.roll(lum,-1,axis=1)-np.roll(lum,1,axis=1)
    def tracks(centers):
        strengths=[]
        for slope in (-.10,0,.10):
            xx=np.rint((y[:,None]-17)*slope+centers[None,:]).astype(int)
            d=gx[y[:,None],xx]
            # Consecutive sections should agree. Separate sign before pooling
            # so that a shaded side of the placket also counts as a fastener.
            for sign in (-1,1):
                val=d*sign
                parts=np.maximum(0,val.reshape(4,5,-1).mean(1))
                strengths.append(.5*np.sort(parts,axis=0)[1]+.5*np.minimum(.6,parts).mean(0))
        return np.max(np.array(strengths),axis=0)
    strength=tracks(np.arange(7,25)); center=strength[6:13].max()
    flank=np.median(np.concatenate((strength[:5],strength[-5:])))
    vertical=sat((center-.038-flank*.85)/(.12+flank*.45))
    # Local bilateral symmetry of a visible button/fly strip. A two-panel
    # closure extends through a majority of the torso, below a potential lapel.
    collarArea=np.mean(a[:3,12:20]); shoulders=np.mean(a[:3,5:10])+np.mean(a[:3,22:27])
    raised=sat((collarArea-shoulders*.5-.10)/.63)
    sect=a>.47; st=sect.argmax(0); en=np.where(sect.any(0),32-sect[::-1].argmax(0),0)
    st=np.where(sect.any(0),st,32)
    # Contrast along mirror-paired obliques: coat lapels and pressed collars
    # usually run into the opening rather than lying horizontally.
    diag=0.
    for depth in (7,12,17):
        ry=np.arange(2,depth+2)
        for w in (4,7):
            off=np.rint(np.linspace(w,1,depth)).astype(int)
            le=gx[ry,15-off]; ri=gx[ry,16+off]
            paired=.5*(le-ri)
            diag=max(diag, .60*abs(paired.mean())+.40*np.abs(paired).mean())
    lapel=sat((diag-.09)/.30)
    # A pullover can have a collar but has no front fastener. A crew-neck or
    # V-neck sweater usually has its contrast confined to the top of the chest.
    neckContrast=np.std(lum[1:8,11:21]); torsoContrast=np.std(lum[10:26,12:20])
    neckOnly=sat((neckContrast-torsoContrast-.03)/.22)*(1-vertical)
    # Cloth pattern over the two front panels, omitting center, cuffs and hem.
    body=np.concatenate((lum[8:26,9:14],lum[8:26,18:23]),1)
    dx=np.concatenate((np.diff(body[:,:5],axis=1),np.diff(body[:,5:],axis=1)),axis=1)
    dy=np.diff(body,axis=0)
    printStrength=sat((min(abs(dx).mean(),abs(dy).mean())-.054)/.16)
    # Dissimilar button-front panels and rounded shirttails favor a shirt.
    panel=np.mean(abs(lum[8:26,10:15]-lum[8:26,17:22][:,::-1]))
    asymmetric=sat((panel-.14)/.26)
    centerHem=np.median(en[13:19]); tail=sat((centerHem-(np.median(en[8:12])+np.median(en[20:24]))*.5-.6)/2.9)
    hips=a[22:27].mean(0); chest=a[6:11].mean(0)
    # Short sleeves abruptly end to either side of the chest. Tank tops are
    # handled mainly by the outline matching, with a broad cut-out neck.
    sideLow=.5*(hips[:5].mean()+hips[-5:].mean()); sideHi=.5*(chest[:5].mean()+chest[-5:].mean())
    short=sat((sideHi-sideLow-.14)/.40)
    openNeck=sat((np.median(st[12:20])-.5*(np.median(st[7:11])+np.median(st[21:25]))-.4)/4.5)
    # Measure the *shape* of the collar pedestal, not just whether its
    # central pixels are filled. Neighboring neck and shoulder contours move
    # together when a high collar is added, unlike a scooped ribbed neck.
    shoulderline=np.r_[st[6:10],st[22:26]]
    leftNeck=min(st[10:16]); rightNeck=min(st[16:22])
    abrupt=sat((np.median(shoulderline)-(leftNeck+rightNeck)/2-1.65)/3.15)
    notch=sat((np.mean(st[14:18])-.5*(leftNeck+rightNeck)-.6)/2.6)
    stand=abrupt*(1-.7*notch)
    # Lapel edges are not necessarily parallel to the center-line. Look for
    # an inverted chevron on either side of the neck (points of a collar) and
    # downward-converging lines of longer lapels on both fronts.
    fold=0.; pointed=0.
    for depth in (5,9,14):
        ry=np.arange(1,depth+2); frac=np.arange(depth+1)/depth
        for spread in (3,6):
            for ctr in (15,16):
                margin=1+spread*(1-frac)
                lx=np.rint(ctr-margin).astype(int); rx=np.rint(ctr+margin).astype(int)
                lsign=gx[ry,lx]; rsign=gx[ry,rx]
                fold=max(fold,.5*abs(lsign.mean()-rsign.mean())+.25*(abs(lsign).mean()+abs(rsign).mean()))
                if depth<10:
                    lx=np.rint(ctr-2-spread*frac).astype(int); rx=np.rint(ctr+2+spread*frac).astype(int)
                    pointed=max(pointed,min(abs(gx[ry,lx].mean()),abs(gx[ry,rx].mean())))
    folds=sat((fold-.085-flank*.25)/(.24+flank))
    points=sat((pointed-.07)/.22)
    # Plackets can be low-contrast, interrupted by dots (buttons), or curved.
    # Compare local ridge evidence near the midline with adjacent fabric at
    # the same rows, and require support down the front, not just on the chest.
    ridge=np.abs(lum-.5*(np.roll(lum,-2,axis=1)+np.roll(lum,2,axis=1)))
    lines=(ridge[9:29,:]*.5+abs(gx[9:29,:])*.5).reshape(4,5,32).mean(1)
    stripe=np.median(lines[:,13:19].max(1)); textile=np.median(lines[:,[8,9,10,11,20,21,22,23]])
    seam=sat((stripe-.035-1.2*textile)/(.095+.7*textile))
    button=1-(1-vertical)*(1-.78*seam)
    # Flat cuffs form a band crossing the sleeve near its lower end. A ribbed
    # waistband often echoes it on the torso of a pullover.
    band=0.
    for ry in range(23,29):
        delta=lum[ry-1,11:21]-lum[ry+1,11:21]
        band=max(band,abs(delta.mean())-.6*np.std(delta))
    rib=sat((band-.09)/.27)*(1-tail)
    # Strong, spatially repeated contrasts and a gently rounded shirttail are
    # useful even on a shirt whose placket is concealed. Sleeve and torso
    # panels may be lit asymmetrically on softly draped woven fabric.
    woven=sat((min(abs(dx).mean(),abs(dy).mean())-.040)/.12)
    fine=.5*(abs(dx).mean()+abs(dy).mean()); panelStd=np.std(body)
    drape=sat((panelStd-fine*1.35-.07)/.22)
    collar=stand
    # Style proportions from the actual hem, rather than the row envelope:
    # sleeves can project below a cropped sweater, or a long body can extend
    # down well past the sleeves of a coat. On shirting the body often ends
    # in a rounded tail, while the straight ribbed hem of a knit stops abruptly.
    torsoHem=sat((np.median(en[11:21])-(np.quantile(en[2:7],.75)+np.quantile(en[25:30],.75))*.5-1.3)/5)
    straight=sat((1.9-np.std(en[10:22]))/1.6)
    crop=sat(((np.quantile(en[2:7],.75)+np.quantile(en[25:30],.75))*.5-np.median(en[11:21])-.60)/3.6)
    slight=sat((.83-ar)/.24); boxy=sat((ar-.70)/.21)
    slim=slight*(1-.65*short); longLine=torsoHem*(1-short)
    # Differentiate a weak but continuous front closure from a genuinely
    # uniform center panel. A contrast ratio within this one image is more
    # reliable on a light coat than a fixed minimum ridge amplitude.
    ratio=sat((stripe-.009-1.4*textile)/(.015+1.4*textile))
    closure=max(button,.82*ratio)
    # Compare how well the item satisfies each construction, without treating
    # an invisible fastener alone as evidence for a pullover.
    knit=.50*(1-closure)+.52*neckOnly+.45*notch+.65*rib+.45*boxy+.57*crop+.30*straight-.46*collar-.39*folds
    outer=1.22*closure+1.05*collar+.65*folds+.49*closure*collar+.22*straight+.68*slim+.48*longLine-.42*short-.52*woven+.58*points*collar
    shirt=.78*closure+1.04*woven+.80*tail+.56*asymmetric+.42*points+.66*drape+.47*points*(1-collar)+.72*closure*short+.58*slim+.42*longLine-.3*collar
    tee=1.05*short+.70*(1-button)+.43*openNeck+.4*(1-tail)-.6*points*button
    wear=np.array([tee,knit,outer,shirt]); wear-=wear.max()
    # Leg fork: search a connected opening entering from the lower hem. The
    # center of a cropped pair of pants need not run down the column midline.
    lower=a[15:31]
    off=np.arange(9,23)
    gap=1-lower[:,off]
    flank1=lower[:,off-5]; flank2=lower[:,off+5]
    gap=np.maximum(0,np.minimum(flank1,flank2)-lower[:,off])
    perRow=gap.max(1); fork=.6*perRow[-8:].mean()+.4*perRow.mean()
    belt=a[:4,5:27].mean(); neck=a[:3,13:19].mean()
    # Evidence for narrow unbroken legs (front-view trousers with touching
    # ankles) can be a flat waistband and a taper from hip to hem.
    breadth=lambda z: np.mean(np.sum(np.maximum.accumulate(z>.5,axis=1)&np.maximum.accumulate(z[:,::-1]>.5,axis=1)[:,::-1],axis=1))/32
    taper=breadth(a[6:13])-breadth(a[23:30]); belt=sat((belt-.40)/.43)
    pants=sat((fork-.24)/.53)*sat((.91-ar)/.35)
    pants=max(pants,.70*belt*sat((taper+.03)/.23)*sat((.61-ar)/.20))
    # Air under a strap indicates a sandal. Measure holes also at a higher
    # contrast level, for nearly black voids adjacent to sharpened straps.
    air=0.
    for mask in (a>.5,v>.12):
        col=np.maximum.accumulate(mask,0)&np.maximum.accumulate(mask[::-1],0)[::-1]
        row=np.maximum.accumulate(mask,1)&np.maximum.accumulate(mask[:,::-1],1)[:,::-1]
        inside=col[3:28,3:29]; dark=(~mask)[3:28,3:29]
        # A connected gap with a finite span across and down is stronger than
        # isolated dark pixels in a patterned textile or gaps along the laces.
        g=inside&dark
        square=g[1:,1:]&g[:-1,1:]&g[1:,:-1]&g[:-1,:-1]
        air=max(air,sat((g.sum()+square.sum()*1.5-18)/max(28,.33*inside.sum())))
    # Side-view closed shoes have one continuous vamp. An open heel leaves a
    # tail of bare sole, often with its top nearly parallel to its bottom.
    tall=(en-st)/32
    if tall[3:9].mean()>tall[23:29].mean(): tall=tall[::-1]; st=st[::-1]; en=en[::-1]
    tip=np.mean(tall[2:7]); back=np.mean(tall[24:30]); highest=np.mean(tall[12:22]); rearLow=sat((highest-back-.05)/.32)
    soleOnly=sat((.46-back)/.26)*rearLow
    # Use the shape of the rear collar as well as absolute proportions. A boot
    # shaft tends to be an upright plateau rather than a tongue and heel cup.
    shaft=sat((back-tip-.20)/.36)*sat((.35-np.mean(st[24:29])/32)/.28)
    flat=sat((ar-1.77)/.86); upright=sat((2.15-ar)/.71)
    heelArch=sat(((.5*(en[4:10].mean()+en[27:31].mean())-en[18:24].mean())/32-.035)/.24)
    # Sneakers usually have a concave topline between the tongue and heel
    # counter. Even high tops often have this ankle opening. Boot shafts stay
    # high over a continuous interval; merely having a tall tongue is weak
    # evidence. Sandals can have an almost bare midfoot over a continuous sole.
    dip=sat((st[20:27].mean()-.5*(st[15:20].min()+st[26:31].min())-1.8)/5)
    plateau=sat((.28-st[19:29].mean()/32)/.23)
    midLow=sat((.43-tall[13:21].min())/.30)*sat((tall[6:14].max()-.50)/.30)
    boot=1.10*shaft+.72*upright+.74*heelArch+.81*plateau-.84*dip-2.25*air-1.05*soleOnly
    sneaker=.95*flat+.82*(1-air)+.42*(1-shaft)+1.0*dip-1.45*soleOnly-.70*heelArch
    sandal=2.90*air+2.30*soleOnly+.45*heelArch+.76*midLow
    foot=np.array([sandal,sneaker,boot]); foot-=foot.max()
    return wear,foot,pants

# Exact integer squared distances to individual drawings. The aspect ratio is
# matched separately in log space so that a proportional resizing costs the
# same for e.g. a narrow dress and a wide shoe.
patterns=CONSTANTS['pattern'].astype(np.float32)
norms=(patterns*patterns).sum(1)
logasp=CONSTANTS['aspect']
sections=[np.flatnonzero(CONSTANTS['label']==k) for k in range(10)]
bank=patterns.T.copy()
tiles=CONSTANTS['textile'].astype(np.float32)
tileNorm=(tiles*tiles).sum(1)
tileBank=tiles.T.copy(); links=CONSTANTS['links']
tileSections=[np.flatnonzero(CONSTANTS['label'][links]==k) for k in (0,2,4,6)]

def classify(x):
    num=len(x)
    features=np.empty((num*2,bank.shape[0]),np.float32)
    scores=np.empty((num,10),np.float64)
    shape=np.empty((num,10),np.float64)
    cloth=np.empty((num,4)); shoes=np.empty((num,3)); pants=np.empty(num)
    ar=np.empty(num,np.float32); textile=np.empty((num*2,tileBank.shape[0]),np.float32)
    for i in range(num):
        a,v,aspect=readitem(x[i,0]); ar[i]=np.log(aspect)
        features[2*i]=describe(a); features[2*i+1]=describe(a[:,::-1]); scores[i]=detect(a,v,aspect)
        cloth[i],shoes[i],pants[i]=construction(a,v,aspect)
        textile[2*i]=fabric(a,v); textile[2*i+1]=fabric(a[:,::-1],v[:,::-1])
    dist=features@bank
    dist *= -2
    dist += (features*features).sum(1)[:,None]
    dist=norms+np.minimum(dist[::2],dist[1::2])
    delta=np.abs(ar[:,None]-logasp)
    dist += 46000*delta**2/(1+delta)
    for k,indices in enumerate(sections):
        best=dist[:,indices].min(1)
        shape[:,k]=-best/165
    scores+=shape
    dtex=-2*(textile@tileBank)+(textile*textile).sum(1)[:,None]
    dtex=tileNorm+np.minimum(dtex[::2],dtex[1::2])
    # Compare edge patterns with a modest alignment cost from the corresponding
    # drawing, so that a neckline is not matched to the end of a short sleeve.
    dtex+=.34*dist[:,links]
    fabricCost=np.stack([dtex[:,sec].min(1) for sec in tileSections],1)
    # Simulated appearance is a supporting vote; style and lighting vary much
    # more in the photographs than in the schematic textile drawings.
    cloth-=1.15*np.tanh((fabricCost-fabricCost.min(1,keepdims=True))/970)
    # Group first: a jacket and a sweater can have indistinguishable outlines.
    # Once inside a group, an imperfect match to its closest pattern is weak
    # evidence. Bound its effect and use the independent construction cues.
    groups=([0,2,4,6],[5,7,9])
    for cols,details,scale in zip(groups,(cloth,shoes),(1.85,2.20)):
        sh=shape[:,cols]; old=scores[:,cols]; prior=old-sh; peak=old.max(1,keepdims=True)
        loss=sh.max(1,keepdims=True)-sh
        if len(cols)==4:
            # Distinguishing a tee from a coat by its short sleeves is much
            # more reliable than distinguishing a long-sleeve shirt from one.
            longLoss=sh[:,1:].max(1,keepdims=True)-sh[:,1:]
            # A long-sleeved garment has two separable shape decisions: sleeve
            # length versus a tee, and a much weaker silhouette preference
            # among collared shirts, jackets and knits of similar proportions.
            commonLoss=sh.max(1,keepdims=True)-sh[:,1:].max(1,keepdims=True)
            penalty=np.empty_like(sh)
            penalty[:,1:]=.68*commonLoss+1.35*np.tanh(longLoss/4)
            penalty[:,0]=.60*loss[:,0]+.40*1.8*np.log1p(loss[:,0]/2.5)
        else:
            penalty=2.7*np.log1p(loss/3.5)
        final=scale*details+.56*prior-penalty
        scores[:,cols]=peak+final-final.max(1,keepdims=True)
    # Discount an accidental dress-outline match when there is a leg fork.
    scores[:,1]+=5*pants
    scores[:,3]-=5*pants
    return np.argmax(scores,axis=1)

def predict(x):
    out=np.empty(len(x),np.int64)
    for i in range(0,len(x),64): out[i:i+64]=classify(x[i:i+64])
    return out
'''

def main():
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(solution)
    np.savez(out/'constants.npz',**makebank())

if __name__=='__main__': main()
