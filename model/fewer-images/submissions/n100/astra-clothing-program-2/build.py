"""Build individual paper-cutout models of shop garments. No input files."""
import math
from pathlib import Path
import sys
import numpy as np

# All pattern dimensions for clothing below are nominal centimetres, not pixels.
# The footwear uses foot length as the unit. Coordinates on screen increase down.
rng = np.random.default_rng(31077)
def u(a,b): return rng.uniform(a,b)
def coin(p): return rng.random()<p
def pick(a): return a[rng.integers(len(a))]
def tri(a,m,b): return rng.triangular(a,m,b)
def sym(edge): return edge+[(-x,y) for x,y in reversed(edge)]
def positive(points): return (points,1)
def negative(points): return (points,0)

def upper(k):
    # A paper sewing pattern: neckline, shoulder, armhole, waist and hem.
    # Sleeves are sewn on separately, and may hang apart from the body.
    penalty=0
    neckstyle='crew'; hemcurve=u(0,2.5); slit=0; flare=u(-2,2)
    rib=0; waistin=u(-1,3); tank=False; shoulderstrap=False
    short=False; longsleeve=True
    if k==0:
        height=tri(52,67,82); chest=tri(19,24.5,32)
        t=u(0,1)
        if t<.81: arm=tri(9,19,30); short=True
        elif t<.91: arm=u(0,6); tank=True; penalty=.015
        else: arm=u(40,63); penalty=.049
        neckstyle=pick(['crew','crew','crew','scoop','vee'])
        if coin(.2): flare=u(2,7); waistin=u(1,5)
    elif k==2:
        # Knit sleeves can also be short or reach only the elbow.
        height=tri(49,63,79); chest=tri(20,25,33); arm=tri(51,62,71)
        rib=u(0,4.5); neckstyle=pick(['crew','crew','crew','wide','roll','vee','hood'])
        if coin(.13): arm=u(33,52); penalty=.015
    elif k==4:
        height=tri(53,70,98) if coin(.81) else u(88,118)
        chest=tri(21,26,32); arm=tri(52,60,67)
        neckstyle=pick(['stand','stand','lapel','lapel','hood','crew'])
        flare=u(-1,5); waistin=u(-1,3)
        if coin(.26): slit=u(1,7)
        if coin(.20): height=u(49,65); rib=u(0,4)
    elif k==6:
        height=tri(53,70,86); chest=tri(19,24,32)
        neckstyle=pick(['collar','collar','collar','vee','scoop','crew','wide'])
        hemcurve=tri(0,4,10); flare=u(-1,4)
        t=u(0,1)
        if t<.51: arm=u(51,64)
        elif t<.70: arm=u(29,49)
        elif t<.95: arm=u(10,25); short=True
        else: arm=0; tank=True
        if neckstyle in ['scoop','wide']: waistin=u(0,4); flare=u(0,7)
    else:
        height=tri(75,96,145); chest=tri(17,21,27)
        waistin=u(1,7); flare=tri(1,10,29); hemcurve=u(0,3)
        neckstyle=pick(['scoop','crew','vee','collar'])
        t=u(0,1)
        if t<.45: tank=True; shoulderstrap=coin(.40); arm=0
        elif t<.75: arm=u(6,26); short=True
        else: arm=u(32,64)
        if coin(.17): flare=u(-2,4); waistin=u(-1,3)
        if coin(.09): # skirt or strapless maxi
            return [(sym([(chest*u(.62,.85),0),(chest,10),(chest+flare,height*.76),(chest+flare*u(.8,1.12),height-1),(chest+flare-3,height)]),1)],.015
    shoulder=chest*u(.72,.94); shoulderY=u(3.4,8)
    neck=u(6.3,10); drop=u(.6,3)
    if neckstyle=='scoop': neck=u(8,13); drop=u(2,9)
    if neckstyle=='wide': neck=u(9,14)
    if neckstyle=='vee': drop=u(2,10)
    if tank:
        shoulder=chest*u(.56,.86); shoulderY=u(0,3)
        neck=min(neck,shoulder-u(1,3)); drop=u(4,12)
        if shoulderstrap: neck=shoulder-u(1,3); drop=u(9,17)
    neck=min(neck,shoulder-1)
    raised=0
    if neckstyle in ['collar','lapel']: raised=u(1.0,4.5); neck=u(5.8,9.2); drop=u(0,2.5)
    if neckstyle=='stand' or neckstyle=='roll': raised=u(3,8); neck=u(6.4,10)
    if neckstyle=='hood': raised=u(4,13); neck=u(8,12); drop=u(0,2)
    neckedge=[(0,drop-raised),(neck*.45,drop*(.58 if neckstyle=='vee' else .9)-raised),(neck*.83,drop*.46-raised),(neck,-raised)]
    if raised>0: neckedge+= [(neck+u(-1,2),0)]
    pitY=u(17,25); waistY=min(height*.63,44); hem=chest+flare
    bodyside=[(shoulder,shoulderY),(shoulder*.5+chest*.5,shoulderY*.5+pitY*.5),(chest,pitY),(chest-waistin,waistY),(hem,height-max(hemcurve,4)),(hem-rib,height-hemcurve),(hem-rib-2,height-hemcurve*.7),(hem*.65,height-hemcurve*.15),(hem*.22,height)]
    if k==3: # smooth waist-to-skirt line
        bodyside[4]=(hem,height-u(1,4))
    body=sym(neckedge+bodyside)
    # sym expects the last point at the bottom: close the centre of the hem
    # first, then return around the opposite side to the top.
    polys=[positive(body)]
    if not tank:
        if short:
            spread=u(.48,.92); cuff=tri(9,13,18); tilt=u(.26,.95)
        else:
            spread=tri(.11,.22,.43); cuff=tri(6,9,13); tilt=u(-.14,.28)
            if k==4: cuff+=u(.5,2.5)
            if k==6 and arm<50: cuff+=u(0,4)
        for sign in [-1,1]:
            tiltarm=np.clip(spread+u(-.038,.038),.06,.97)
            wristX=shoulder+arm*tiltarm; wristY=shoulderY+arm*math.sqrt(1-tiltarm**2)+u(-1,1)
            # A short sleeve is wide at the armpit; for a hanging sleeve its
            # edge can be almost horizontal. Avoid making a long thin spike.
            if short: tilt=u(.15,.72)
            wristY2=wristY+cuff*tilt; width=cuff*math.sqrt(max(.12,1-min(.95,abs(tilt))**2))
            elbowX=.52*wristX+.48*shoulder+u(.3,3.4)
            sleeve=[(shoulder-.6,shoulderY-.1),(shoulder+u(1,3),shoulderY+1.6),(elbowX,(wristY+shoulderY)*.51),(wristX,wristY),(wristX-width,wristY2),((wristX-width+chest)*.5+u(-.8,.9),(wristY2+pitY)*.5),(chest-.5,pitY+1),(shoulder-3,shoulderY+3)]
            polys.append(positive([(x*sign,y) for x,y in sleeve]))
    if slit>0:
        polys.append(negative([(-slit*.45,height+1),(u(-1,1),height-u(6,23)),(slit*.55,height+1)]))
    # Slight overall tilt / catalogue drape. One draw, not shared across patterns.
    skew=u(-.042,.042); squash=u(.95,1.05)
    return [([(x*squash+skew*y,y) for x,y in poly],v) for poly,v in polys],penalty

def belt(k):
    # Straight-on construction rather than separate sewn pieces. Draft a
    # complete short-sleeve side contour in units of back length, so there
    # is an unbroken side seam where a kimono/tee sleeve meets the trunk.
    height=tri(54,67,80) if k==0 else tri(55,69,84)
    chest=tri(18,24,31); shoulder=chest*u(.66,.86); neck=u(6,10)
    side=u(.17,.27)*height; under=side+u(0,.065)*height
    width=chest*u(1.30,1.63); armpit=chest*u(.92,1.04)
    hem=chest*u(.90,1.13); waist=chest*u(.88,1.04); rise=u(0,.043)*height
    if k==6: rise=u(1,9); hem=chest*u(.95,1.28)
    contour=[(0,u(.3,3)),(neck*.7,u(.1,2)),(neck,0),(shoulder,u(1.8,4.1)),(shoulder+2,u(3.7,5.6)),(width,side),(width-2,side+2),(armpit+u(1.2,4),under+u(0,2.4)),(armpit,under-1.5),(waist,height*.62),(hem,height-rise-2),(hem-1,height-rise),(hem*.52,height-rise*.13),(0,height)]
    shear=u(-.045,.045)
    return [positive([(x+y*shear,y) for x,y in sym(contour)])],0

def coatdraft():
    # Set-in sleeves of a structured jacket. A coat has a distinct raised
    # centre neck, a fairly straight/wide cuff and thick, often touching,
    # side panels. Include a raglan shoulder and a short bomber version.
    h=tri(52,69,98); chest=tri(21,26,32); neck=tri(5,8,11)
    if coin(.16): h=u(48,61)
    shoulder=chest*u(.71,.95); sy=u(6,12); armpit=u(19,27)
    wx=chest+u(2.5,14); wy=sy+tri(48,56,64); cuff=u(8,14)
    hem=chest*u(.91,1.15); waist=chest*u(.87,1.01); stand=u(1.7,7)
    hood=coin(.13)
    neckedge=[(0,u(.2,2.8)),(neck*.7,0),(neck,1),(neck+u(-1,1),stand)]
    if hood: neckedge=[(0,0),(neck*.8,u(0,1.2)),(neck+1,3),(neck-1,stand+2)]
    torso=neckedge+[(shoulder,sy),(chest,armpit),(waist,h*.58),(hem,h-4),(hem-1,h-1),(hem*.64,h),(hem*.19,h)]
    polys=[positive(sym(torso))]
    for sign in [-1,1]:
        bend=u(-1.8,2.2); a=wx+bend
        sleeve=[(shoulder-1,sy-.6),(shoulder+2,sy+.5),(chest+1,armpit*.88),((a+chest)*.5+1.7,(wy+armpit)*.5),(a,wy),(a-cuff,wy+u(-1.7,2.2)),((a-cuff+chest)*.5,(wy+armpit)*.5),(chest-1,armpit+1)]
        polys.append(positive([(sign*x,y) for x,y in sleeve]))
    if coin(.30): polys.append(negative([(-u(.4,3),h+.2),(u(-1,1),h-u(5,22)),(u(.4,3),h+.2)]))
    return polys,0

def trousers():
    h=tri(73,102,117); hip=tri(17,22,29); waist=hip*u(.77,1.01)
    crotch=h*u(.19,.43); hipline=crotch*u(.33,.72)
    gap=tri(0,4.8,16); leg=tri(6,10,19)
    if coin(.16): leg=u(12,21); gap=u(0,6)
    edge=[(0,u(0,1.7)),(waist,0),(hip,hipline),(hip*u(.96,1.04),crotch),(hip*.48+(gap*.5+leg)*.52,(h+crotch)*.53),(gap*.5+leg,h),(gap*.5,h-u(0,2)),(gap*.29,(h+crotch)*.57),(.6,crotch+3),(0,crotch)]
    skew=u(-.06,.06)
    return [positive([(x+skew*y,y) for x,y in sym(edge)])],0

def shoe(k):
    # Positive y is upwards in this drawing. The profile is flipped for rasterisation.
    polygons=[]; penalty=0
    heel=0 if coin(.52 if k==9 else .82) else u(.05,.36 if k!=7 else .10)
    if k==5: heel=0 if coin(.46) else u(.09,.60)
    toe=u(.07,.155); spring=u(.015,.068); sole=u(.025,.082)
    if k==5: toe=u(.025,.095); sole=u(.018,.070)
    back=u(.92,.99); heelwidth=u(.07,.22)
    ground=[(0,spring+.028),(.018,spring*.58),(.12,spring*.16),(.31,0),(.48,u(0,.018)),(.67,heel*.82),(back-heelwidth,heel),(back-heelwidth+u(-.025,.025),.004),(back,.008),(back+.014,heel+sole*.6)]
    if heel<.045: ground=[(0,spring+.023),(.019,spring*.6),(.12,.012),(.3,0),(.56,u(0,.012)),(back-.07,u(0,.025)),(back,.02),(back+.012,.047)]
    if k==7:
        top=tri(.23,.34,.49); heelH=top*u(.63,.99)+heel*.5
        if coin(.12): top=u(.44,.63); heelH=top*u(.77,1); penalty=.022
        tongue=u(.50,.71); throat=top*u(.57,.85); notch=u(.72,.86)
        roof=[(0,spring+.035),(.032,toe),(.14,toe+u(.002,.033)),(.29,toe+u(.028,.075)),(tongue-.09,top-u(.025,.080)),(tongue-.038,top+.009),(tongue,top),(tongue+.035,top*.83),(notch,throat),(back-.044,heelH),(back-.008,heelH-u(0,.045)),(back+.015,heelH*.53),(back+.012,heel+sole*.6)]
        polygons.append(positive(roof+ground[::-1]))
    elif k==9:
        top=tri(.38,.69,1.21)+heel*.65; front=u(.40,.66); backtop=back+u(-.095,.045)
        roof=[(0,spring+.035),(.023,toe),(.13,toe+u(0,.038)),(.27,toe+u(.018,.075)),(front-.095,toe+.10+heel*.20),(front-.015,toe+.16+heel*.43),(front+u(-.032,.045),top*.78),(front+u(-.075,.05),top+u(-.05,.035)),((front+backtop)*.5,top+u(-.046,.02)),(backtop,top+u(-.02,.068)),(backtop+u(-.02,.025),top*.86),(back-.043,heel+.17),(back+.012,heel+sole*.6)]
        polygons.append(positive(roof+ground[::-1]))
    else:
        def floor(x): return spring*max(0,1-6*x)+sole+heel*min(1,max(0,(x-.34)/.49))
        xs=[0,.035,.15,.32,.48,.63,.81,back,back+.012]
        polygons.append(positive([(x,floor(x)) for x in xs]+ground[::-1]))
        # Straps are bands with an arch across the instep. Some sandals have a counter.
        def strap(base,span,h,w,lean):
            outer=[(base,floor(base)-.012),(base+span*.12+lean*.35,floor(base)+h*.56),(base+span*.44+lean,floor(base+span*.44)+h),(base+span*.68+lean,floor(base+span*.68)+h*1.015),(base+span*.91+lean*.36,floor(base+span*.91)+h*.64),(base+span,floor(base+span)-.014)]
            inner=[(x+(1 if j<3 else -1)*w*.44,y-(w*.84 if j not in (0,5) else 0)) for j,(x,y) in enumerate(outer)]
            polygons.append(positive(outer+inner[::-1]))
        strap(u(.018,.22),u(.21,.50),u(.055,.185),u(.025,.115),u(-.02,.11))
        kind=u(0,1)
        if kind<.20: # slide, pool sandal, mule
            if coin(.5): strap(.20,u(.25,.44),u(.085,.2),u(.062,.14),.05)
        elif kind<.66:
            # In side view a pump has a rising footbed, often with empty air
            # over the arch. An ankle strap need not connect to the toe strap.
            ah=u(.12,.40); rear=back+u(-.04,.025); w=u(.018,.10)
            counter=u(.075,.22)
            polygons.append(positive([(rear-counter,floor(rear)-.018),(rear-.058,floor(rear)+ah*.53),(rear-.06,floor(rear)+ah),(rear+.012,floor(rear)+ah+u(-.04,.02)),(back+.015,floor(back)-.016)]))
            xf=u(.59,.8); yf=floor(rear)+ah+u(-.012,.069)
            polygons.append(positive([(xf,yf),(rear,yf+u(-.041,.005)),(rear,yf-w),(xf+.018,yf-w)]))
            if coin(.48): polygons.append(positive([(.36,floor(.36)),(xf,yf),(xf+.054,yf-.018),(.43,floor(.43))]))
            if coin(.38): # wedge rather than a separate block/stiletto heel
                polygons.append(positive([(.29,.005),(back,0),(back,floor(back)),(.64,floor(.64)),(.29,floor(.29))]))
        else:
            ankle=tri(.14,.28,.53); center=u(.67,.87); w=u(.025,.093)
            strap(u(.29,.48),u(.32,.5),ankle,w,u(.005,.12))
            if coin(.72):
                rear=back-u(.005,.065)
                polygons.append(positive([(center-.08,floor(center)+ankle*.84),(rear,floor(rear)+ankle),(rear+.015,floor(rear)+ankle-w),(center-.09,floor(center)+ankle*.84-w)]))
                if coin(.66): polygons.append(positive([(rear-.02,floor(rear)+ankle),(rear+.02,floor(rear)+ankle),(back,floor(back)-.01),(back-u(.05,.14),floor(back)-.01)]))
            if coin(.34): strap(u(.12,.26),u(.31,.47),ankle*.68,w*.8,u(.04,.11))
    sign=pick([-1,1]); skew=u(-.038,.038)
    return [([(x*sign,-y+x*skew) for x,y in poly],v) for poly,v in polygons],penalty

def bag():
    h=tri(.42,.76,1.23); top=u(.0,.14); bot=u(.005,.17)
    if coin(.12): top=u(.12,.29)
    body=[(top,0),(.4,u(-.016,.06)),(1-top,u(-.025,.025)),(1-top+.035,h*.06),(1,h*u(.37,.66)),(1-bot*.57,h*.93),(1-bot,h),(.5,h+u(0,.032)),(bot,h),(bot*.60,h*.93),(0,h*u(.37,.66)),(top-.028,h*.08)]
    polys=[positive(body)]
    if coin(.73):
        rise=tri(.08,.39,1.13); width=u(.36,.88); cx=.5+u(-.070,.070); thick=u(.012,.060); curv=u(.65,1.65)
        outer=[]; inner=[]
        for t in np.linspace(0,math.pi,25):
            outer.append((cx+width*.5*math.cos(t),.025-rise*math.sin(t)**curv))
            inner.append((cx+(width*.5-thick)*math.cos(t),.04-(rise-thick)*math.sin(t)**curv))
        polys.append(positive(outer+inner[::-1]))
    if coin(.13): # messenger bag side gusset or backpack side
        polys.append(positive([(.06,.08),(-.07,h*.64),(-.035,h*.88),(.11,h*.96),(.14,h*.54)]))
    skew=u(-.07,.07); tilt=u(-.045,.045)
    return [([(x+y*skew,y+x*tilt) for x,y in poly],v) for poly,v in polys],0

DESCRIPTOR='''def unit(v):
    return v / np.sqrt(np.maximum(np.sum(v*v,axis=1,keepdims=True),.01))

def describe(mask):
    # Anti-aliased occupancy and two signed components of the contour normal.
    n=len(mask)
    m=mask.reshape(n,14,2,14,2).mean(axis=(2,4))
    p=np.pad(m,((0,0),(1,1),(1,1)))
    dx=(p[:,1:-1,2:]-p[:,1:-1,:-2])*.5
    dy=(p[:,2:,1:-1]-p[:,:-2,1:-1])*.5
    area=unit((m-.5).reshape(n,-1))
    normal=unit(np.concatenate((dx.reshape(n,-1),dy.reshape(n,-1)),axis=1))
    return np.rint(512*np.concatenate((.89*area,.456*normal),axis=1)).astype(np.float32)
'''
# Use the identical formula to encode the individual drawings.
exec(DESCRIPTOR)

def render(polys):
    # Scan conversion. The box and descriptor of each pattern depend only on it.
    edges=[(np.array(poly),v) for poly,v in polys]
    vertices=np.concatenate([p for p,v in edges if v])
    left,top=vertices.min(0); right,bottom=vertices.max(0)
    side=56
    xs=left+(np.arange(side)+.5)*(right-left)/side
    ys=top+(np.arange(side)+.5)*(bottom-top)/side
    m=np.zeros((side,side),np.bool_)
    for p,v in edges:
        x,y=p.T; xe=np.roll(x,1); ye=np.roll(y,1)
        den=ye-y; yy=ys[:,None]; active=(yy<y)!=(yy<ye)
        intersections=x+(yy-y)*(xe-x)/np.where(den!=0,den,1)
        # Each crossing toggles the inside bit for all following samples.
        cuts=np.clip(np.searchsorted(xs,intersections),0,side)
        r,c=np.nonzero(active); changes=np.zeros((side,side+1),np.uint8)
        np.bitwise_xor.at(changes,(r,cuts[r,c]),1)
        inside=np.bitwise_xor.accumulate(changes[:,:-1],axis=1).astype(bool)
        m[inside]=v
    small=m.reshape(28,2,28,2).mean(axis=(1,3)).astype(np.float32)
    # Thin straps and handles have at most a pixel or two across them on the
    # imaging grid. Model the footprint of thresholded antialiased boundaries
    # in the observation; each drawing gets its own camera/pose perturbation.
    aspect=(right-left)/(bottom-top)
    pad=np.pad(small,1,mode='edge'); thick=small.copy()
    for dy,dx in [(0,1),(2,1),(1,0),(1,2)]:
        expansion=u(.06,.41)*(max(1,aspect) if dx==1 else max(1,1/aspect))
        thick=np.maximum(thick,small+min(.91,expansion)*(pad[dy:dy+28,dx:dx+28]-small))
    silhouette=small>.46
    hi=np.argmax(silhouette,axis=0); lo=small.shape[0]-np.argmax(silhouette[::-1],axis=0)
    le=np.argmax(silhouette,axis=1); ri=small.shape[1]-np.argmax(silhouette[:,::-1],axis=1)
    hw=hi[None,:]<=np.arange(28)[:,None]; hw &= np.arange(28)[:,None]<lo[None,:]
    hw &= np.any(silhouette,axis=0)[None,:]
    vw=le[:,None]<=np.arange(28)[None,:]; vw &= np.arange(28)[None,:]<ri[:,None]
    vw &= np.any(silhouette,axis=1)[:,None]
    outer=thick.copy()
    outer[hw & vw]=1
    return describe(thick[None])[0],math.log(aspect),describe(outer[None])[0]

RUNTIME='''import numpy as np

'''+DESCRIPTOR+'''
def blur(x):
    p=np.pad(x,((0,0),(1,1),(1,1)),mode='edge')
    a=(p[:,:-2]+2*p[:,1:-1]+p[:,2:])*.25
    return (a[:,:,:-2]+2*a[:,:,1:-1]+a[:,:,2:])*.25

def morph(x,op):
    p=np.pad(x,((0,0),(1,1),(1,1)),mode='edge')
    a=op(op(p[:,:-2],p[:,1:-1]),p[:,2:])
    return op(op(a[:,:,:-2],a[:,:,1:-1]),a[:,:,2:])

def bounds(profile):
    yes=profile>np.minimum(.65,profile.max(1,keepdims=True)*.095)
    a=np.argmax(yes,axis=1)
    b=27-np.argmax(yes[:,::-1],axis=1)
    return a,np.maximum(a,b)

def resize(x,l,r,t,b):
    f=(np.arange(28,dtype=np.float32)+.5)/28
    cx=np.clip(l[:,None]-.5+(r-l+1)[:,None]*f,0,27)
    cy=np.clip(t[:,None]-.5+(b-t+1)[:,None]*f,0,27)
    xx=cx.astype(np.int32); yy=cy.astype(np.int32)
    fx=(cx-xx).astype(np.float32)[:,None,:]; fy=(cy-yy).astype(np.float32)[:,:,None]
    k=np.arange(len(x))[:,None,None]; ya=yy[:,:,None]; xa=xx[:,None,:]
    yb=np.minimum(27,ya+1); xb=np.minimum(27,xa+1)
    return ((1-fy)*((1-fx)*x[k,ya,xa]+fx*x[k,ya,xb])+fy*((1-fx)*x[k,yb,xa]+fx*x[k,yb,xb]))

def sat(x):
    return np.clip(x,0,1)

def rules(p,m,aspect):
    n=len(p); s=np.zeros((n,10),np.float32)
    # Coordinates here are fractions of the observation's own bounding box.
    # Measure a persistent front opening, not just any single shadow or logo.
    linear=p
    p=np.power(np.maximum(0,p),.8)
    cloth=p[:,7:24,:]
    light=np.maximum(.16,cloth[:,:,8:20].mean((1,2)))
    cloth=cloth/light[:,None,None]
    def line(z):
        step=z[:,:,4:]-z[:,:,:-4]
        ridge=(z[:,:,4:]+z[:,:,:-4])*.5-z[:,:,2:-2]
        return (np.abs(step.mean(1))*.60+np.abs(ridge.mean(1))*.85+.35*np.abs(ridge).mean(1))
    lines=line(cloth); middle=lines[:,9:15].max(1)
    side=.5*(lines[:,5:9].max(1)+lines[:,15:19].max(1))
    seam=sat((middle-.50*side-.055)*3.6)
    # Regular fabric pattern and small-scale detail (measured away from the silhouette).
    core=cloth[:,1:-1,7:21]
    gx=np.abs(core[:,:,2:]+core[:,:,:-2]-2*core[:,:,1:-1]).mean((1,2))
    gy=np.abs(core[:,2:,:]+core[:,:-2,:]-2*core[:,1:-1,:]).mean((1,2))
    detail=sat((np.minimum(gx,gy)-.105)*3.7)
    busy=sat((.5*(gx+gy)-.11)*2.5)
    # Do the two sloping revers outline a V that extends onto the chest?
    q=p/light[:,None,None]; pairs=[]
    for shift in [-1,0,1]:
        for end in [7,10,13]:
            y=np.arange(2,end+1); off=np.rint(1.4+(end-y)*.52).astype(np.int32)
            a=13+shift-off; b=14+shift+off
            left=q[:,y,a-1]-q[:,y,a+1]; right=q[:,y,b+1]-q[:,y,b-1]
            pairs.append(np.maximum(np.minimum(left.mean(1),right.mean(1)),np.minimum(-left.mean(1),-right.mean(1))))
    lapel=sat((np.maximum.reduce(pairs)-.13)*2.6)
    # Standing collar versus a broad crew neckline. A narrow upper tower must widen below.
    width=m.mean(2)
    stand=sat((width[:,4:8].mean(1)-width[:,:3].mean(1)-.22)*2.8)
    cowl=sat((width[:,1:4].mean(1)-.39)*3.5)
    # A rounded shirttail: a lower centre compared with the flanks of the body.
    yy=np.arange(28)[None,:,None]
    bottom=np.max(np.where(m>.47,yy,-1),axis=1)
    roundhem=sat((bottom[:,11:17].mean(1)-.5*(bottom[:,7:10].mean(1)+bottom[:,18:21].mean(1))-.7)/3.5)
    # A zipper/placket produces a slender coherent edge down the front.
    # Test both transition polarities at multiple widths, allowing the line to
    # slant by a few pixels from chest to hem. Compare against the same filter
    # on the side panels to cancel all-over stripes and shadow contrast.
    y=np.arange(6,24); openings=[]; sidelines=[]
    for run in [-2,0,2]:
        axis=np.arange(6,22)[None,:]+np.rint((y-14)*run/17).astype(np.int32)[:,None]
        for radius in [1,2,3]:
            mid=q[:,y[:,None],axis]; ll=q[:,y[:,None],axis-radius]; rr=q[:,y[:,None],axis+radius]
            step=(ll-rr).mean(1); line=((ll+rr)*.5-mid).mean(1)
            lineval=np.maximum(np.abs(line),np.abs(step)*.55)
            openings.append(lineval[:,5:11].max(1)); sidelines.append((lineval[:,:4].max(1)+lineval[:,12:].max(1))*.5)
    zipval=np.maximum.reduce(openings); other=np.maximum.reduce(sidelines)
    fastener=sat((zipval-.48*other-.035)*3.6)
    # Lapel or shirt collar surrounds the neck and changes tone locally, even
    # on a plain garment. A longer coat leaves cloth *below* the cuffs.
    neckpatch=q[:,2:9,9:19]; neckcontrast=np.abs(neckpatch-blur(neckpatch)).mean((1,2))
    collar=sat((neckcontrast-.04)*5)
    cuffs=np.maximum(bottom[:,:5].max(1),bottom[:,23:].max(1))
    coatlength=sat((bottom[:,10:18].mean(1)-cuffs-1)/5)
    # Tee sleeves end abruptly high up; under the armpits the silhouette of
    # an untucked tee stays close to the same width all the way to the hem.
    row=(m>.48); span=(row.shape[2]-np.argmax(row,2)-np.argmax(row[:,:,::-1],2))/28
    shortarm=sat((span[:,3:10].max(1)-span[:,12:18].mean(1)-.07)*6)
    flare=span[:,22:26].mean(1)-span[:,12:18].mean(1)
    straight=sat(1-np.abs(flare)*7)
    block=shortarm*straight
    # Thin, loose cloth develops fine vertical folds over the lower torso.
    # Use the outer panels, as opposed to the single central coat fastening.
    panel=q[:,12:24,6:22]; wav=panel[:,:,2:] + panel[:,:,:-2]-2*panel[:,:,1:-1]
    folds=sat((np.abs(wav.mean(1)).mean(1)-.045)*4)
    # A graphic printed centrally on an otherwise plain tee is not plaid.
    ht=np.abs(q-blur(q)); printval=ht[:,5:17,9:19].mean((1,2)); outside=ht[:,14:23,7:21].mean((1,2))
    graphic=sat((printval-1.5*outside-.015)*7)
    # A long coat and a blouse may both have buttons. Distinguish a single
    # nearly uninterrupted closure from a row of little shirt buttons or a
    # placket on plaid fabric. Use agreement of three separate vertical bands.
    def bandlines(z):
        a=z[:,:,:-4]; b=z[:,:,2:-2]; c=z[:,:,4:]
        return np.stack(((a-c).mean(1)*.60,((a+c)*.5-b).mean(1)),axis=2)
    bands=np.stack([bandlines(q[:,i:i+6]) for i in (6,12,18)],axis=1)
    pos=np.min(bands,axis=1); neg=np.min(-bands,axis=1)
    persistence=np.maximum(pos,neg).max(2)
    main=persistence[:,9:15].max(1); sides=(persistence[:,5:9].max(1)+persistence[:,15:19].max(1))*.5
    continuous=sat((main-.55*sides-.022)*5)
    # Knit tops tend to have a broad, shallow neck opening; a coat puts
    # collar material high up in the centre. Read the relative height of
    # the neck and both shoulders, not just the total width of the top row.
    rooftop=np.argmax(m>.5,axis=1)
    neckheight=(rooftop[:,6:10].mean(1)+rooftop[:,18:22].mean(1))*.5-rooftop[:,12:16].mean(1)
    highneck=sat((neckheight-1)/3); scoop=sat((1-neckheight)/3)
    # Detect checked or floral fabric by energy over BOTH flanks and multiple
    # rows. In comparison a printed sweatshirt often has a flat back panel,
    # sleeves and hem around a single graphic on the chest.
    checker=ht[:,8:23,6:22]
    allover=sat((np.minimum(checker[:,:,:6].mean((1,2)),checker[:,:,10:].mean((1,2)))-.045)*7)
    # All evidence below is relative to the chosen garment family. In
    # particular a shadow on a skirt or split between legs isn't a zipper.
    s[:,0]+=.012-.039*seam-.018*lapel+.076*block+.061*graphic-.021*folds+.010*scoop
    # Paired, touching banks of a zip can be VERY low contrast. A dotted
    # row of buttons instead makes several compact local extrema surrounded
    # by similar cloth. Look at each row in relation to its neighbours.
    sm=blur(q); marks=np.abs(q-sm)
    dots=marks[:,7:24,11:17].max(2)
    flanks=(marks[:,7:24,6:11].max(2)+marks[:,7:24,17:22].max(2))*.5
    rows=np.maximum(0,dots-flanks*.8)
    button=sat((np.minimum(rows[:,:8].max(1),rows[:,8:].max(1))-.038)*4)
    # Wide jacket cuffs and thick body panels leave little visible space
    # between arm and body; relaxed sweater sleeves taper toward the wrist.
    wleft=m[:,17:24,:8].sum(2); wright=m[:,17:24,20:].sum(2)
    filled=sat(((wleft+wright).mean(1)/16-.56)*3)
    # A vertical crease of the cloth can be as strong as a front fastening.
    # Use a line-vs-parallel-lines contrast ratio to discount such folds.
    prominence=sat((zipval-other*.87)/(.06+other))
    outer=.48*fastener*(.38+.62*prominence)+.38*continuous+.14*seam
    button=button*sat(1-allover)*(.35+.65*prominence)
    # Thin/fluid blouses have curved hems, and a thinner draping sleeve
    # ending above the wrist. Sleeve length is apparent on both sides.
    mediumarm=sat((23-cuffs)*.35)*sat((cuffs-9)*.35)
    openfront=outer*(.8+.2*filled)
    s[:,2]+=.012-.080*openfront-.030*lapel-.012*collar-.014*detail-.018*roundhem-.034*coatlength+.016*scoop-.012*button
    s[:,4]+=-.012+.098*openfront+.037*lapel+.012*stand+.016*highneck+.040*coatlength-.044*allover-.027*roundhem-.015*folds+.016*button
    s[:,6]+=-.014+.028*fastener+.018*lapel+.026*detail+.056*allover+.045*roundhem+.044*folds+.013*collar+.014*button-.027*graphic+.023*mediumarm
    local=np.array([0,2,4,6]); dressgroup=s[:,local].copy()
    s[:,local]-=.90*np.max(dressgroup,axis=1,keepdims=True)
    s[:,3]+=.050*sat(flare*4)*sat((-.03-aspect)*2)-.065*block-.030*sat((aspect+.16)*3)
    # Distinguish rectangular tees from straight shift dresses of similar
    # outline using the length from clavicle to hem in shoulder-width units.
    # Front-view tees end around the hip; a dress extends past the seat.
    trunk=span[:,13:21].mean(1)*np.exp(aspect)
    splitlow=(m[:,16:25,6:10].mean((1,2))+m[:,16:25,18:22].mean((1,2)))*.5-m[:,16:25,12:16].mean((1,2))
    s[:,3]+=.038*sat((.52-trunk)*5)-.075*sat((trunk-.54)*5)-.047*graphic-.074*sat((splitlow-.12)*3)
    # A sleeve ends at an outward-pointing corner on BOTH sides. Its extent
    # as a fraction of total torso length separates a tee from a shift dress.
    tip=np.argmax(span[:,2:14],axis=1)+2
    tips=sat((span[:,2:14].max(1)-span[:,15:21].mean(1)-.10)*5)
    tshape=tips*sat((tip-3)/4)*sat((trunk-.52)*6)*(1-sat(flare*5))
    s[:,0]+=.066*tshape; s[:,3]-=.057*tshape
    # Split legs give evidence even when one leg leans inwards.
    low=m[:,18:26]; split=(low[:,:,4:10].mean((1,2))+low[:,:,18:24].mean((1,2)))*.5-low[:,:,12:16].mean((1,2))
    s[:,1]+=.027*sat((split-.2)*2)
    # Negative spaces bounded above, below and to both sides are characteristic
    # of straps. Ignore the borders and the top, where sneaker laces can be dark.
    above=np.maximum.accumulate(m,axis=1); below=np.maximum.accumulate(m[:,::-1],axis=1)[:,::-1]
    left=np.maximum.accumulate(m,axis=2); right=np.maximum.accumulate(m[:,:,::-1],axis=2)[:,:,::-1]
    hole=np.maximum(0,np.minimum(np.minimum(above,below),np.minimum(left,right))-m)
    air=hole[:,8:24,2:26].mean((1,2)); straps=sat((air-.044)*6)
    # Open toes, T-straps and slingbacks may not enclose a hole in 2-D, but
    # they remove material from the hull of a shoe. Measure column solidity,
    # sparse regions between the straps, and upper material over the footbed.
    level=np.maximum(.07,np.sort(linear.reshape(n,-1),axis=1)[:,675])
    ink=sat((linear/level[:,None,None]-.055)*4.2)
    aircol=np.minimum(np.maximum.accumulate(ink,1),np.maximum.accumulate(ink[:,::-1],1)[:,::-1])-ink
    gap=sat((aircol[:,3:24,2:26].mean((1,2))-.045)*7)
    # How much of the toe is bare? Only consider the toe, not the back end,
    # and allow the more massive ankle/heel to be at either side.
    flip=m[:,:14,:14].sum((1,2))>m[:,:14,14:].sum((1,2))
    oriented=np.where(flip[:,None,None],ink[:,:,::-1],ink)
    profile=oriented.sum(1)/28
    toeheight=profile[:,2:10].mean(1)
    sparse=sat((.51-ink.mean((1,2)))*4.4)
    baretoe=sat((.26-toeheight)*5.5)
    # In a sneaker the tongue then the low collar make a notch; a boot has
    # a much higher continuous shaft, normally almost vertical at the front.
    topedge=np.argmax(oriented>.42,axis=1)
    stepheight=sat((np.median(topedge[:,5:11],axis=1)-np.median(topedge[:,18:24],axis=1)-10)/9)
    # If a strap opens sideways it leaves an indentation instead of an
    # enclosed hole. The convex hull of a column-height graph bridges it.
    # Account for both the arch under the foot and gaps above the sole.
    solid=oriented>.42
    roof=np.argmax(solid,axis=1).astype(np.float32)
    floor=27-np.argmax(solid[:,::-1],axis=1).astype(np.float32)
    roof=np.where(solid.any(1),roof,27); floor=np.where(solid.any(1),floor,0)
    bridged=roof.copy(); base=floor.copy()
    for spanlen in (4,8,12,16,20,24):
        for part in (.25,.5,.75):
            cut=int(part*spanlen); remain=spanlen-cut
            bridged[:,cut:28-remain]=np.minimum(bridged[:,cut:28-remain],(1-part)*roof[:,:28-spanlen]+part*roof[:,spanlen:])
            base[:,cut:28-remain]=np.maximum(base[:,cut:28-remain],(1-part)*floor[:,:28-spanlen]+part*floor[:,spanlen:])
    emptytop=(roof-bridged)[:,2:26]; arch=(base-floor)[:,3:25]
    sunk=sat((emptytop.mean(1)-1.35)*.62); heel=sat((arch.mean(1)-.8)*.65)
    column=aircol[:,3:24].mean(1)
    biggap=sat((np.sort(column[:,2:26],axis=1)[:,-7:-2].mean(1)-.105)*4.0)
    s[:,5]+=.25*straps+.21*gap+.24*sparse+.115*baretoe+.16*sunk+.14*biggap+.078*heel-.046
    s[:,7]+=.010-.102*straps-.085*gap-.125*sparse-.040*baretoe-.055*stepheight-.052*heel
    s[:,9]+=.005-.098*straps-.105*gap-.110*sparse+.034*stepheight-.044*sunk-.046*biggap
    # A bag with a handle has most of its interior space above the pouch.
    grip=hole[:,1:12,:].mean((1,2)); pouch=m[:,15:24,4:24].mean((1,2))
    s[:,8]+=.082*sat((grip-.07)*5)*sat((pouch-.62)*3)
    # A clutch has a broad closed top without shoulders, whereas a handbag
    # can have a narrow top but a near-rectangular base below the handle.
    rectangle=np.minimum(width[:,4:9].mean(1),width[:,19:24].mean(1))
    s[:,8]+=.047*sat((rectangle-.82)*8)*sat((aspect+.18)*3)
    return s

def batch(images):
    raw=images[:,0]
    # Black fabric still has a visible outline. Use a soft dark-pixel test and
    # a one-pixel closing, keeping genuine spaces between straps and sleeves.
    scale=np.sort(raw.reshape(len(raw),-1),axis=1)[:,735,None,None]
    weak=sat((raw-.012)/(.015+.105*scale)); firm=sat((raw-.025)/(.028+.22*scale))
    mask=weak*.73+firm*.27
    closed=morph(morph(mask,np.maximum),np.minimum)
    mask=.64*mask+.36*closed
    l,r=bounds(mask.sum(1)); t,b=bounds(mask.sum(2))
    aspect=np.log((r-l+1)/(b-t+1)).astype(np.float32)
    m=resize(mask,l,r,t,b); picture=resize(raw,l,r,t,b)
    f=describe(m)
    # Look at both the full footwear pattern and the filled side profile.
    above=np.maximum.accumulate(m,axis=1); below=np.maximum.accumulate(m[:,::-1],axis=1)[:,::-1]
    left=np.maximum.accumulate(m,axis=2); right=np.maximum.accumulate(m[:,:,::-1],axis=2)[:,:,::-1]
    boundary=describe(np.minimum(np.minimum(above,below),np.minimum(left,right)))
    footstart=0
    bank=CONSTANTS['patterns']; rat=CONSTANTS['aspect']; cost=CONSTANTS['cost']; ends=CONSTANTS['ends']
    adjustment=rules(picture,m,aspect)
    shape=np.empty_like(adjustment)
    start=0
    for k in range(10):
        last=int(ends[k]); d=f @ bank[start:last].T
        if k in (5,7,9):
            footend=footstart+last-start
            contour=boundary @ CONSTANTS['outlines'][footstart:footend].T
            d=.55*d+.45*contour; footstart=footend
        error=np.abs(aspect[:,None]-rat[None,start:last]); d=d*(1/262144)-.26*error-cost[None,start:last]
        # Require support from several similar sewing patterns as well as the
        # single best correspondence; one lucky extreme pose is weak evidence.
        best=np.partition(d,-8,axis=1)[:,-8:]
        shape[:,k]=.5*(best.max(1)+best.mean(1)); start=last
    scores=shape+adjustment
    clothing=[0,2,4,6]; shoes=[5,7,9]
    # The family comparison should reflect the garment's cut rather than its
    # surface detail. Keep a little evidence for a particularly clear detail.
    bestcut=shape[:,clothing].max(1); bestscore=scores[:,clothing].max(1)
    scores[:,clothing]+=.82*(bestcut-bestscore)[:,None]
    gain=scores[:,shoes].max(1)-shape[:,shoes].max(1)
    scores[:,shoes]-=.85*np.maximum(0,gain-.045)[:,None]
    return np.argmax(scores,axis=1)

def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),64):
        result[i:i+64]=batch(x[i:i+64])
    return result
'''

def main():
    dest=Path(sys.argv[1]); dest.mkdir(parents=True,exist_ok=True)
    patterns=[]; shoeoutlines=[]; aspects=[]; costs=[]; ends=[]
    for k,count in enumerate([2700,1100,3600,3600,4200,3800,4400,2400,2100,2900]):
        for j in range(count):
            if (k==0 and j%2==0) or (k==6 and j%7==0): polys,cost=belt(k)
            elif k==4 and j%2==0: polys,cost=coatdraft()
            elif k in (0,2,3,4,6): polys,cost=upper(k)
            elif k==1: polys,cost=trousers()
            elif k==8: polys,cost=bag()
            else: polys,cost=shoe(k)
            v,a,hull=render(polys); patterns.append(v); aspects.append(a); costs.append(cost)
            if k in (5,7,9): shoeoutlines.append(hull)
        ends.append(len(patterns))
    (dest/'solution.py').write_text(RUNTIME)
    np.savez(dest/'constants.npz',patterns=np.stack(patterns),outlines=np.stack(shoeoutlines),aspect=np.array(aspects,np.float32),cost=np.array(costs,np.float32),ends=np.array(ends,np.int32))

if __name__=='__main__': main()
