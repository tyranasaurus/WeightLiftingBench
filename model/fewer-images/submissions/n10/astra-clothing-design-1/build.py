"""Construct matched-filter and garment-detail units.  No input files."""
import json
import math
import sys
from pathlib import Path
import numpy as np

# The drawing coordinates are fractions of the object's major dimension, not
# landmarks in a photograph. y increases from the neckline to the hem, or
# from the ankle to the sole. Each drawing has its own nuisance variables.
r = np.random.default_rng(27031)
U = r.uniform

def chance(p):
    return U() < p

def pick(*vs):
    return vs[r.integers(len(vs))]

def clip(x):
    return np.clip(x, 0, 1)

N = 42
axis = (np.arange(N) + .5)/N
X0, Y0 = np.meshgrid(axis-.5, axis-.5)

def grid(w, h, flip=False, asymmetric=False):
    # The trimming/centering of a catalogue photo leaves small margins;
    # allow geometric centering as well as a shift toward the thick part.
    scale = U(.9,1.01)/max(w,h)
    y = (Y0-U(-.021,.021))/(scale*h) + .5
    x = (X0-U(-.021,.021))/(scale*w) + .5
    x = x + U(-.038,.038)*(y-.5)
    if asymmetric:
        x = x + U(-.022,.022)*np.sin(np.pi*y)
    if flip: x = 1-x
    return x,y

def polygon(x,y,v):
    # An even/odd scan conversion, independently for this one polygon.
    inside = np.zeros_like(x,dtype=bool)
    x1,y1 = v[-1]
    for x2,y2 in v:
        if y1 != y2:
            inside ^= ((y1>y)!=(y2>y)) & (x < (x2-x1)/(y2-y1)*(y-y1)+x1)
        x1,y1 = x2,y2
    return inside

def symmetric(x,y,right):
    # Polygon described by the clockwise outline of its right half.
    return polygon(np.abs(x-.5),y,[(0,right[0][1])]+right+[(0,right[-1][1])])

def line(x,y,a,b,width):
    dx,dy = b[0]-a[0], b[1]-a[1]
    t = clip(((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy+1e-14))
    return ((x-a[0]-t*dx)**2+(y-a[1]-t*dy)**2) < width**2

def ellipse(x,y,cx,cy,rx,ry):
    return ((x-cx)/rx)**2+((y-cy)/ry)**2 < 1

def shaded(mask,value):
    return mask.astype(float)*value

# Front-facing sewn garments. Coordinates here, unlike grid(), use the
# garment's height as the unit for both axes, preserving sleeve angles.
def garment(k,i):
    if k==0:
        typ=pick('tee','tee','tee','tee','tank','tank','cap')
    elif k==2:
        typ=pick('knit','knit','knit','knit','knit','hood','wideknit')
    elif k==4:
        typ=pick('jacket','jacket','jacket','overcoat','overcoat','puffer','blazer')
    else:
        typ=pick('button','button','button','button','blouse','blouse','shortbutton','tunic','sleeveless')
    tank=typ in ('tank','sleeveless'); short=typ in ('tee','cap','shortbutton')
    knit=k==2; coat=k==4
    if tank:
        body=U(.20,.32); sh=body*U(.65,.95); pit=U(.23,.39); end=body*U(.84,1.12)
        if typ=='sleeveless': sh=body*U(.80,1.02); pit=U(.17,.32); end=body*U(1.01,1.23)
        edge=sh; cuff=0.; el=0.; span=max(end,body,sh)
    elif short:
        body=U(.255,.37); sh=body*U(.76,.98); pit=U(.23,.37)
        end=body*U(.80,1.04); el=U(.20,.38) if typ!='cap' else U(.14,.24)
        edge=sh+U(.14,.26); cuff=U(.10,.19); span=max(edge,body,end)
    else:
        if knit:
            body=U(.265,.36); sh=body*U(.84,1.04); end=body*U(.87,1.06)
            if typ=='wideknit': body*=1.12; end*=1.15
        elif coat:
            body=U(.23,.33) if typ=='overcoat' else U(.26,.37)
            sh=body*U(.79,1.03); end=body*U(.93,1.13)
        else:
            body=U(.24,.34); sh=body*U(.83,1.01); end=body*U(.88,1.11)
        pit=U(.20,.36)
        el=U(.82,1.11) if knit else U(.71,1.01) if coat else U(.72,1.06)
        if typ=='overcoat': el=U(.60,.85)
        if typ=='blouse' and chance(.45): el=U(.42,.73)
        cuff=U(.071,.119); edge=max(body,sh)+U(.073,.16)
        span=max(edge,end,body)
    totalh=max(1.,el+.016)
    # The shoulder of a tailored garment is fitted rather than dropped. A
    # shaped collar/stand projects above the collar bones. Its height also
    # changes the outside contour around the neck, even in black fabric.
    # A pullover can have a high neck too, but usually has a flat neckline.
    collar_stand = U(.045,.113) if coat else U(.028,.094) if typ in ('button','shortbutton') else 0.
    # A little air around a neckline; collars protrude above the shoulder.
    hascollar=typ in ('button','shortbutton','jacket','blazer','overcoat','puffer')
    dx,dy=grid(span*2,totalh+collar_stand,asymmetric=True)
    x=(dx-.5)*span*2; y=dy*(totalh+collar_stand)-collar_stand
    ax=np.abs(x); shoulder_y=U(.055,.125) if hascollar else U(.035,.10)
    neck=min(sh*.82,U(.085,.157) if not tank else U(.083,.154))
    top_y=U(0,.027) if not hascollar else U(.029,.071)
    waist=body*(U(.77,.93) if typ in ('tank','sleeveless','tunic') else U(.91,1.025))
    side = np.interp(y,[0,shoulder_y,pit,.61,.87,1.01],[neck,sh,body,waist,(waist+end)*.5,end])
    hem=1.-U(.00,.023)*(1-np.minimum(ax/end,1)**2)
    if typ in ('button','blouse','shortbutton','tunic','sleeveless'):
        # Shirttails hang below the side seams.
        hem=1.-U(.035,.14)*np.minimum(ax/end,1)**2
    top = top_y+(shoulder_y-top_y)*clip((ax-neck)/(sh-neck+.00001))
    torso=(ax<=side)&(y>=top)&(y<=hem)
    if tank:
        # Armhole scoops below the shoulder strap.
        arm=sh+(body-sh)*clip(y/pit)-U(.022,.065)*np.sin(np.pi*clip(y/pit))
        torso &= (y>pit)|(ax<arm)
    cut=U(.10,.27) if tank else U(.042,.134) if not hascollar else U(.014,.075)
    if typ=='hood': cut=U(.008,.045)
    neck_shape=pick('round','round','v')
    frac=clip(ax/neck)
    bowl=np.sqrt(clip(1-frac**2)) if neck_shape=='round' else 1-frac
    opening=(ax<neck)&(y<top_y+cut*bowl)
    inside_neck=torso & opening
    torso &= ~opening
    a=shaded(torso,U(.55,.83)); sleeve=np.zeros_like(torso)
    if not tank:
        if short:
            sleeve=polygon(ax,y,[(sh-.025,shoulder_y+.002),(sh,shoulder_y-.004),((sh+edge)*.5,shoulder_y+.035),(edge,el-.04),(edge-cuff,el+.025),(body-.018,pit+.02),(body-.06,pit)])
        else:
            elbow=U(.39,.64)*el
            shoulder_out=sh+U(.035,.071)
            sleeve=polygon(ax,y,[(sh-.045,shoulder_y+.018),(sh,shoulder_y-.006),(shoulder_out,shoulder_y+.045),((edge+shoulder_out)*.51,elbow),(edge,el-.015),(edge-.015,el+.013),(edge-cuff,el+.004),(edge-cuff-.003,el-.077),(body+U(-.038,.020),pit+.08),(body-.048,pit-.026)])
        a=np.where(sleeve,a.max()*U(.73,1.13),a)
    if not tank and chance(.76):
        a[inside_neck]=a.max()*U(.16,.59)  # back of the collar, inside a neck opening
    if collar_stand:
        stand=polygon(ax,y,[(0,-collar_stand+.005),(neck*.60,-collar_stand),(neck*.86,-collar_stand+.014),(neck*1.05,top_y+.023),(neck*1.09,.084),(0,.085)])
        a[stand]=a.max()*U(.51,.91)
    if k in (4,6):
        # Front fastening and its two lapels/collar points.
        if hascollar or chance(.32):
            depth=U(.16,.39) if coat else U(.09,.22)
            spread=min(neck, U(.062,.115)); band=U(.007,.024) if coat else U(.005,.015)
            if chance(.22) and coat: band=U(.025,.073)
            placket=(ax<band*(1+.65*clip((y-.65)/.35)))&(y>depth*.50)
            a[placket & torso]*=U(.18,.74) if coat else U(.48,.91)
            c=polygon(ax,y,[(.002,depth),(.003,.056),(spread*.67,top_y-.025),(spread,top_y+.005),(spread*1.22,depth*.61),(spread*.82,depth*.48)])
            if coat and chance(.65):
                v=(ax<spread*clip(1-y/depth)) & (y>top_y)
                a[v]*=U(.12,.67)
            a[c]=a.max()*U(.49,.98)
            if chance(.65):
                for by in np.arange(depth+.04,.97,U(.095,.173)):
                    a[ellipse(x,y,0,by,U(.008,.015),.009)&torso]*=U(.22,.70)
        if typ=='puffer':
            a*=1-U(.06,.24)*np.cos(2*np.pi*y/U(.11,.19))**8
    if knit:
        a[(y>hem-.044)&torso]*=U(.7,.96)
        if not tank: a[(y>el-.055)&sleeve]*=U(.65,.97)
    return a

def trousers():
    width=U(.29,.57); x,y=grid(width,1,asymmetric=True); ax=np.abs(x-.5)
    hip=U(.43,.50); waist=hip*U(.70,.99); crotch=U(.22,.42)
    outer_end=U(.31,.51); inner_end=U(.035,.18)
    if chance(.22): outer_end=U(.45,.55); inner_end=U(.012,.065)
    if chance(.22): inner_end=U(.002,.05); outer_end=U(.35,.46)
    outer=np.interp(y,[0,.14,.34,.64,1],[waist,hip,hip*U(.90,1.01),(hip+outer_end)*.5*U(.88,1.02),outer_end]); inner=inner_end*clip((y-crotch)/(1-crotch))**U(.53,1.35)
    a=shaded((ax<outer)&((y<crotch)|(ax>=inner))&(y>0)&(y<1),U(.53,.86)); a*=1-U(.09,.31)*np.exp(-((ax-inner)/.115)**2)*clip((y-.13)/.24)
    return a

def dress():
    typ=pick('a','a','a','shift','column','full','fitted'); neck=U(.042,.094); sh=U(.11,.205)
    bust=sh*U(.86,1.05); waist=bust*U(.62,.97); wy=U(.29,.50)
    hip=waist*U(1.07,1.53); end=hip*U(.78,1.16) if typ in ('fitted','column','shift') else U(.205,.39) if typ!='full' else U(.32,.51)
    arm=pick('none','none','cap','short','short','long','strap'); edge=sh+U(.043,.115) if arm not in ('strap','none') else sh
    ext=max(edge,end,hip); x,y=grid(2*ext,1,asymmetric=True); ax=2*ext*np.abs(x-.5)
    n=U(.04,.13); sy=U(.023,.064); pit=U(.11,.24)
    contour=np.interp(y,[0,sy,pit,wy,.59,.75,1],[sh*.73,sh,bust,waist,hip,(hip+end)*.5,end]); bot=1-U(.002,.04)*clip(ax/end)**2
    upper=sy*(.25+.75*clip((ax-neck)/(max(sh-neck,.01))))
    a=shaded((ax<contour)&(y>upper)&(y<bot),U(.53,.82)); a[(ax<neck)&(y<n*np.sqrt(clip(1-(ax/neck)**2)))]=0
    if arm=='strap': a[(y<.12)&(ax>sh*.82)]=0
    elif arm not in ('none',):
        el=U(.08,.14) if arm=='cap' else U(.12,.22) if arm=='short' else U(.44,.70)
        sleeve=polygon(ax,y,[(sh-.035,sy+.008),(sh,sy),(sh+.035,sy+.022),(edge,el-.01),(edge-.05,el+.017),(bust-.007,pit+.014),(bust-.044,pit)])
        a[sleeve]=a.max()*U(.75,1.05)
    if chance(.4): a*=1-.15*np.exp(-((y-wy)/.02)**2)
    return a

# Shoes: the forefoot is on the left. Each entry in a side profile is a
# control point, in shoe-length and shoe-height units. Material parts are
# explicit: outsole, upper, tongue, counter, shaft, heel, and straps.
def shoe(k):
    if k==7:
        high=chance(.15); ht=U(.46,.68) if high else U(.25,.50); heel=U(.02,.17) if high else U(.06,.39)
        x,y=grid(1,ht,chance(.045)); t=U(.51,.71); drop=U(.16,.48)
        upper=[(0,.73),(.025,.61),(.12,U(.56,.65)),(.24,U(.46,.58)),(t-.16,.25),(t-.08,.10),(t,0),(t+.044,.055),(t+.07,drop*.80),(.80,drop),(.89,drop*.66),(.95,heel),(.984,heel+.07),(.986,.58),(1.,.85)]
        sole_y=U(.70,.84); top=upper+[(.97,.91),(.74,.95),(.23,.965),(.08,.92),(.012,.83)]
        a=shaded(polygon(x,y,top),U(.44,.82)); sole=polygon(x,y,[(0,.73),(.09,sole_y+.05),(.27,sole_y+.095),(.58,sole_y+.072),(.84,sole_y+.014),(.99,sole_y-.031),(1,.96),(.77,.998),(.45,.995),(.15,.973),(.034,.887),(.002,.829)])
        a[sole]=U(.53,.94)
        for j in range(4):
            xx=t-.21+.045*j; yy=.33-.054*j
            a[line(x,y,(xx,yy),(xx+.049,yy+.085),.010)&(a>0)]*=U(.65,1.25)
        return a
    if k==9:
        ht=U(.45,1.05); highheel=chance(.34); x,y=grid(1,ht,chance(.04))
        front=U(.43,.65); back=U(.86,.995); lip=U(-.025,.07); ankle=U(.44,.70); sole=U(.79,.88) if not highheel else U(.59,.73)
        upper=[(.007,.83),(.032,.765),(.14,.728),(.25,.69),(front-.14,ankle+.08),(front-.041,ankle-.008),(front,ankle-.104),(front+U(-.029,.067),lip+.026),(front+.06,.01),(back-.06,.009),(back,lip+.025),(back-.018,.31),(back-.025,ankle-.045),(.991,sole-.065),(.978,sole+.075),(.80,sole+.077),(.64,sole+.087),(.45,.94),(.28,.979),(.091,.965),(.017,.918)]
        a=shaded(polygon(x,y,upper),U(.46,.83)); heel_width=U(.075,.21) if highheel else U(.19,.35)
        soleline=[(0,.848),(.055,.899),(.19,.931),(.36,.922),(.54,sole+.04),(.73,sole-.022),(.978,sole-.022),(.989,sole+.052),(.961,1.),(.98-heel_width,1),(.978-heel_width,sole+.067),(.68,sole+.074),(.49,.938),(.35,.98),(.17,.988),(.044,.954),(0,.919)]
        a[polygon(x,y,soleline)]=U(.35,.80); return a
    style=pick('flat','flat','sport','heeled','heeled','wedge','slide','mule','flip')
    high=style in ('heeled','wedge'); ht=U(.47,.78) if high else U(.14,.30) if style in ('flip','slide') else U(.26,.52)
    x,y=grid(1,ht,chance(.055)); lift=U(.33,.61) if high else U(.009,.16); thick=U(.079,.20)
    profile=[(0,.79),(.055,.84),(.20,.887),(.33,.88),(.48,.84-lift*.25),(.69,.828-lift*.85),(.90,.82-lift),(.985,.805-lift)]
    sole=profile+[(.996,.84-lift+thick),(.92,.865-lift+thick),(.66,.872-lift*.78+thick),(.44,.867-lift*.12+thick),(.21,.903+thick*.65),(.07,.904+thick*.48),(.009,.86+thick*.4)]
    a=shaded(polygon(x,y,sole),U(.49,.84)); col=U(.47,.82)
    if high:
        hw=U(.037,.17) if style=='heeled' else U(.31,.55)
        a[polygon(x,y,[(.965-hw,.81-lift),(.988,.804-lift),(.96,1.),(.959-hw*.67,1.),(.91-hw*.59,.73)])]=col
    # Front vamp: a diagonal broad band across the toes, sometimes woven.
    f=U(.08,.23); fw=U(.09,.29); fy=U(.31,.64) if not high else U(.58,.74)
    if style=='mule': fw=U(.27,.49); fy=U(.17,.35)
    if style=='flip': fw=.055
    a[polygon(x,y,[(f-.03,.885),(f+.012,fy+.07),(f+fw*.47,fy),(f+fw*.84,fy+.025),(f+fw+.038,.876),(f+fw*.49,.87),(f+fw*.36,fy+.13),(f+fw*.19,fy+.15),(f+.038,.883)])]=col
    if style not in ('mule','slide','flip'):
        tx=U(.59,.82); ty=U(.013,.16); w=U(.026,.065)
        a[line(x,y,(f+fw,.85),(tx,ty+.085),w)]=col
        a[line(x,y,(tx-.035,ty+.082),(.961,ty),w)]=col
        a[line(x,y,(.958,ty+.012),(.94,.80-lift),w*U(.83,1.55))]=col
        if chance(.55): a[line(x,y,(.65,.84-lift*.65),(.925,ty+.06),w)]=col
    elif style=='flip':
        a[line(x,y,(.13,.82),(.38,.17),.037)]=col; a[line(x,y,(.38,.17),(.65,.84),.043)]=col
    elif style=='slide': a[polygon(x,y,[(.08,.85),(.16,.19),(.31,.03),(.39,.13),(.48,.87)])]=col
    # More closed sport sandals and clogs have a continuous upper but
    # large side windows. A thin strap-only outline is not appropriate.
    if style=='sport' or (style in ('mule','flat') and chance(.28)):
        upper=polygon(x,y,[(.01,.84),(.058,.60),(.24,.48),(.40,.21),(.58,.045),(.68,.095),(.75,.38),(.93,.16),(.984,.19),(.98,.84),(.68,.88),(.38,.96),(.09,.935)])
        for cx,cy,rx,ry in ((.17,.67,.096,.153),(.39,.57,.089,.159),(.58,.47,.071,.212),(.832,.58,.107,.199)):
            if chance(.76): upper &= ~ellipse(x,y,cx,cy,rx*U(.69,1.28),ry*U(.70,1.18))
        a[upper]=col
    return a

def bag():
    typ=pick('tote','tote','bowler','handbag','handbag','shoulder','shoulder','clutch','satchel','rucksack')
    if typ=='clutch': ratio=U(.41,.77); top=U(.014,.09)
    elif typ=='rucksack': ratio=U(1.03,1.49); top=U(.045,.17)
    elif typ=='shoulder': ratio=U(.82,1.62); top=U(.28,.62)
    else: ratio=U(.67,1.25); top=U(.15,.46)
    x,y=grid(1,ratio,asymmetric=True); xc=x-.5; ax=np.abs(xc)
    taper=U(.36,.49); foot=U(.33,.47); bulge=U(.44,.50); h=1-top; sag=U(-.04,.075) if typ!='rucksack' else -.08
    side=np.interp(y,[top-.09,top+.02*h,top+.14*h,top+.51*h,1-.12*h,1-.03*h,1],[taper*.76,taper*.95,taper,bulge,foot+.022,foot,foot*.85]);
    upper=top+sag*(1-(ax/.48)**2)+.015*xc
    if typ in ('rucksack','bowler'): upper+=U(.15,.36)*(ax/.48)**2
    body=(ax<side)&(y>upper)&(y<1-.022*(ax/foot)**2)
    a=shaded(body,U(.52,.83))
    if typ!='clutch':
        reach=U(.20,.39) if typ!='shoulder' else U(.365,.49); width=U(.020,.048) if typ!='shoulder' else U(.014,.038)
        if typ=='rucksack': reach=U(.085,.17)
        arch=top+.077; nx=xc+U(-.055,.055)*(1-clip(y/arch)); power=U(1.55,2.6); ht=arch-U(.002,.033)
        rad=(np.abs(nx)/reach)**power + (np.abs(y-arch)/ht)**power
        inner=(np.abs(nx)/(reach-width))**power + (np.abs(y-arch)/max(.018,ht-width/ratio))**power
        a[(rad<1)&(inner>=1)&(y<arch+.04)]=U(.35,.81)
    return a

def drawing(k,i):
    if k in (0,2,4,6): return garment(k,i)
    if k==1: return trousers()
    if k==3: return dress()
    if k==8: return bag()
    return shoe(k)

layers=[]; weights={}

def op(name,what,src,**kw):
    layers.append(dict(name=name,op=what,inputs=[src] if isinstance(src,str) else src,**kw)); return name

def affine(name,src,W,b=None,conv=False,pad=0):
    W=np.asarray(W,dtype=np.float32); b=np.zeros(len(W),dtype=np.float32) if b is None else np.asarray(b,dtype=np.float32)
    weights[name+'.W']=W; weights[name+'.b']=b
    if conv: return op(name,'conv',src,out=len(W),kernel=list(W.shape[-2:]),stride=1,padding=pad)
    return op(name,'dense',src,out=len(W))

def pool(name,what,src,shape,stride):
    return op(name,what,src,size=list(shape),stride=list(stride))

def act(name,src):return op(name,'relu',src)

def extra_evidence(norm,masked):
    # Relative darkness along a coherent fastening. A normalized Laplacian
    # down the whole torso tends to cancel the two halves of a zipper with
    # different lighting. Use unsigned, local differences on either side,
    # then pool ALONG, not across, the line of fastening. The nuisance
    # location is pooled out only after testing for continuity.
    xx=np.broadcast_to((np.arange(28)+.5)/28-.5,(28,28)); yy=xx.T+.5
    def avg(a):return a/(np.sum(a)+1.e-16)
    def ridge(c,w):return np.exp(-.5*((xx-c)/w)**2)
    F=[]; indices=[]
    regions=[(.22,.37),(.37,.52),(.52,.67),(.67,.82)]
    for centre in (-.055,-.018,.018,.055):
        for slope in (-.07,0,.07):
            idx=[]
            for lo,hi in regions:
                window=(yy>=lo)&(yy<hi); pos=centre+slope*(yy-.5)
                a=avg(window*ridge(pos,.019)); l=avg(window*ridge(pos-.066,.023)); rt=avg(window*ridge(pos+.066,.023))
                ids=[]
                for m in (a-.5*(l+rt),.5*(l-rt)):
                    ids.append(len(F));F.extend([m,-m]);
                idx.append(ids)
            indices.append(idx)
    s=act('unsigned_front_contrasts',affine('local_front_edges',norm,np.reshape(F,(len(F),-1))))
    A=np.zeros((len(indices),len(F)))
    for t,idx in enumerate(indices):
        for a,b in idx:
            A[t,a:a+2]=.25;A[t,b:b+2]=.20
    s=affine('fastening_continuity',s,A);s=op('fastening_paths','reshape',s,shape=[1,1,len(indices)]);s=pool('best_fastening','maxpool',s,(1,len(indices)),(1,len(indices)))
    s=act('fastening_test',affine('fastening_amplitude',s,[[1],[1]],[-.10,-1.18]));
    A=np.zeros((10,2));A[2]=[-.075,.075];A[4]=[.105,-.105]; A[6]=[.058,-.058]
    fastening=affine('fastening_evidence',s,A)

    # Reach of a *connected collar*: count the delay from the top of the
    # neckline to the top of each shoulder. A cutaway crew or scoop neck
    # has the opposite sign. Multiple shoulder spacings accommodate size.
    F=[]
    for cen in (-.035,0,.035):
        for radius in (.17,.215,.26):
            for lim in (.12,.17,.22):
                win=(yy<lim); cap=avg(win*(ridge(cen-.052,.035)+ridge(cen+.052,.035)))
                shoulder=avg(win*(ridge(cen-radius,.033)+ridge(cen+radius,.033)))
                F.append(cap-shoulder)
    s=affine('proud_collar',masked,np.reshape(F,(len(F),-1)));s=op('collar_scales','reshape',s,shape=[1,3,9]);s=pool('shoulder_spacings','avgpool',s,(1,9),(1,9));s=pool('collar_center','maxpool',s,(3,1),(3,1))
    s=act('collar_votes',affine('raised_neck',s,[[1],[1]],[-.09,-.66]));A=np.zeros((10,2));A[4]=[.22,-.22];A[2]=[-.095,.095];A[6]=[.058,-.058]
    collar=affine('collar_evidence',s,A)

    # Direct support for a sandal is visible sky IN the shoe and around
    # the sparse vamp. For each part of its horizontal extent compare the
    # material depth with the projected vertical span. These are features
    # of the current input (not statistics of a bank of drawings).
    rows=pool('occupied_rows','maxpool',masked,(1,28),(1,28));cols=pool('occupied_columns','maxpool',masked,(28,1),(28,1))
    areas=[(.07,.93),(.07,.48),(.50,.90)]
    def vertarea(lo,hi):
        inside=((xx+.5>=lo)&(xx+.5<=hi));return avg(inside.astype(float))
    F=np.array([vertarea(lo,hi) for lo,hi in areas]); area=affine('material_in_shoe_sections',masked,F.reshape(3,-1));
    dep=affine('projected_height',rows,np.ones((3,28))/28);dif=affine('depth_fraction',area,-np.diag([1/.54,1/.39,1/.56]));
    v=op('open_space','add',[dep,dif]);v=act('open_profile',v);A=np.zeros((10,3));A[5]=[.59,.24,.22];A[7]=[-.17,-.040,-.070]; A[9]=[-.115,-.025,-.040];
    # Narrow items are not horizontal footwear. In particular a handbag
    # with a long thin handle must not receive the shoe-profile correction.
    spans=affine('shoe_width_test',cols,np.ones((3,28))/28,np.full(3,-.80));vs=act('narrow_item',affine('not_wide',spans,-np.eye(3)));neg=affine('suppress_vertical_profiles',vs,-3*np.eye(3));v=act('open_wide_profile',op('horizontal_profile','add',[v,neg]));
    g=affine('open_profile_evidence',v,A)

    # Long trousers have two bars of cloth (or a long valley in the
    # lighting, for closed legs). The skirt of a dress bridges the centre.
    F=[]
    for c in (-.046,0,.046):
        for spread in (.082,.125,.166):
            win=(yy>.56)&(yy<.91); F.append(avg(win*(ridge(c-spread,.029)+ridge(c+spread,.029)))-avg(win*ridge(c,.030)))
    leg=affine('leg_separation_tests',norm,np.reshape(F,(len(F),-1)));leg=op('leg_scales','reshape',leg,shape=[1,1,len(F)]);leg=pool('leg_valley','maxpool',leg,(1,len(F)),(1,len(F)));leg=act('weak_leg_valley',affine('no_bifurcation',leg,[[-1]],[.45]));A=np.zeros((10,1));A[1]=-.29;nosplit=affine('skirt_or_trousers',leg,A)
    # Shirttails round off into the side seams. Compare side-to-middle
    # support near the bottom with the same comparison higher on the body
    # to discount narrow but straight garments.
    F=[]
    for side in (.15,.20,.25):
        ends=ridge(-side,.030)+ridge(side,.030);mid=ridge(0,.067)
        top=(yy>.56)&(yy<.72)
        for bottom in (.82,.90,.97):
            low=(yy>bottom-.095)&(yy<=bottom)
            F.append(avg(low*mid)-avg(low*ends)-.8*(avg(top*mid)-avg(top*ends)))
    t=affine('curved_shirttail',masked,np.reshape(F,(len(F),-1)));t=op('tail_scales','reshape',t,shape=[1,1,len(F)]);t=pool('round_hem','maxpool',t,(1,len(F)),(1,len(F)));t=act('tail_test',affine('nonstraight_hem',t,[[1]],[-.19]));A=np.zeros((10,1));A[6]=.14;A[2]=-.047;t=affine('tail_evidence',t,A)

    # Pervasive pattern vs an isolated central graphic. Require contrast
    # in all four quadrants of the bodice. A woven check, vertical folds
    # in a blouse, and repeated floral prints will survive this operation.
    kernels=[]
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        f=np.zeros((3,3)); f[1,1]=-1;f[1+dy,1+dx]=1;kernels.extend([f,-f]);
    z=act('oriented_variation',affine('fabric_differences',norm,np.array(kernels)[:,None],conv=True,pad=1));
    z=affine('local_weave',z,np.ones((1,len(kernels),1,1))/4,conv=True)
    F=[avg(((np.abs(xx-c)<.09)&(np.abs(yy-h)<.114)).astype(float)) for c in [-.11,.11] for h in [.36,.68]]
    z=affine('quietest_region',z,-np.reshape(F,(4,-1)));z=op('weave_regions','reshape',z,shape=[1,1,4]);z=pool('texture_consensus','maxpool',z,(1,4),(1,4));z=act('pattern_present',affine('pattern_strength',z,[[-1],[-1]],[-.15,-.78]));A=np.zeros((10,2));A[6]=[.32,-.32];A[2]=[-.054,.054];A[4]=[-.030,.030];z=affine('pattern_evidence',z,A)
    # Distinguish a narrow shirt stand and its paired fold-over points from
    # a rounded neckline or the wide lapels of outerwear. Test the signed
    # transition at the two downward-pointing collar tips, with either
    # contrast polarity, plus the shallow triangle between the folds.
    F=[]
    for c in (-.037,0,.037):
        for start in (.04,.082):
            for depth in (.091,.14):
                for width in (.079,.122):
                    u=(yy-start)/depth; win=(u>0)&(u<1); tip=width*(.47+.53*u)
                    line1=avg(win*(ridge(c+tip,.022)+ridge(c-tip,.022)))
                    surround=avg(win*(ridge(c+tip+.054,.026)+ridge(c-tip-.054,.026)))
                    F.extend([line1-surround,surround-line1]);
    ct=act('collar_points',affine('paired_point_filters',norm,np.reshape(F,(len(F),-1)),np.full(len(F),-.29)));ct=op('collar_point_scales','reshape',ct,shape=[1,1,len(F)]);ct=pool('folded_collar','maxpool',ct,(1,len(F)),(1,len(F)));ct=act('collar_point_limit',affine('clip_collar_points',ct,[[1],[1]],[0,-1.]));A=np.zeros((10,2));A[6]=[.054,-.054];A[2]=[-.013,.013];ct=affine('pointed_collar_evidence',ct,A)
    # A shoelace deck has a succession of short two-sided contrasts on the
    # *diagonal* ascent from the toe to the tongue. This shoe part is less
    # characteristic of a smooth boot vamp or of a sandal with flat straps.
    F=[];counts=[]
    for flip in [False,True]:
        ex=-xx if flip else xx
        for xpos in (-.058,.014,.085):
            for level in (.39,.46,.53):
                ids=[]
                for s0 in (-.10,-.018,.069):
                    loc=xpos+s0;sy=level-s0*.53; band=ridge(0,1)*np.exp(-.5*((ex-loc)/.050)**2-.5*((yy-sy)/.060)**2)
                    coord=(yy-sy)+.55*(ex-loc)
                    f=avg(band*np.exp(-.5*(coord/.016)**2))-avg(band*(np.exp(-.5*((coord+.042)/.017)**2)+np.exp(-.5*((coord-.042)/.017)**2)))
                    ids.append(len(F));F.extend([f,-f])
                counts.append(ids)
    lace=act('lace_contrasts',affine('instep_filters',norm,np.reshape(F,(len(F),-1)),np.full(len(F),-.14)));A=np.zeros((len(counts),len(F)))
    for i,ids in enumerate(counts):
        for j in ids:A[i,j:j+2]=1./len(ids)
    lace=affine('lacing_deck',lace,A);lace=op('instep_locations','reshape',lace,shape=[1,1,len(counts)]);lace=pool('best_lace_deck','maxpool',lace,(1,len(counts)),(1,len(counts)));A=np.zeros((10,1));A[7]=.067; lace=affine('lacing_evidence',lace,A)
    return [fastening,collar,g,nosplit,t,z,ct,lace]

def build():
    # Brightness-independent soft occupancy. A low contrast edge can enclose
    # a dark garment; the raw intensities get a separate, lower-weight vote.
    v=affine('soft_foreground_hinges','image',np.array([1,1,1,1.]).reshape(4,1,1,1),[-.012,-.11,-.025,-.36],conv=True)
    v=act('positive_hinges',v)
    v=affine('foreground',v,np.array([.70/.098,-.70/.098,.30/.335,-.30/.335]).reshape(1,4,1,1),conv=True)
    masked=v
    # Approximate a mild morphological closing to suppress dark fabric folds.
    blur=np.outer([1,2,1],[1,2,1])/16.
    v=affine('neighbourhood',v,blur.reshape(1,1,3,3),conv=True,pad=1)
    f=pool('shape_cells','avgpool',v,(2,2),(2,2)); f=op('normalized_shape','norm',f)
    raw=pool('tone_cells','avgpool','image',(2,2),(2,2)); raw=op('normalized_tone','norm',raw)

    # Each neuron below uses ONE drawing, normalized relative to its own
    # pixel mean and contrast. No bank-level centering, statistics or fitting.
    count=1536; shapes=np.empty((10*count,14*14),np.float32); tones=np.empty_like(shapes)
    def filt(im):
        cell=im.reshape(14,3,14,3).mean(axis=(1,3)); d=cell-cell.mean()
        return d.ravel()/(196*math.sqrt(np.mean(d*d)+1.e-5))
    for k in range(10):
        for j in range(count):
            im=drawing(k,j); idx=k*count+j
            m=.70*clip((im-.012)/.098)+.30*clip((im-.025)/.335)
            # Gentle anti-alias filter on this silhouette alone.
            p=np.pad(m,1); sm=sum(blur[a,b]*p[a:a+N,b:b+N] for a in range(3) for b in range(3))
            shapes[idx]=.76*filt(sm); tones[idx]=.24*filt(im)
    shape_scores=affine('shape_correlations',f,shapes); tone_scores=affine('tone_correlations',raw,tones)
    match=op('individual_matches','add',[shape_scores,tone_scores]); match=op('class_and_views','reshape',match,shape=[10,6,count//6]);
    match=pool('best_view','maxpool',match,(1,count//6),(1,count//6)); match=pool('typical_best_view','avgpool',match,(6,1),(6,1)); match=op('silhouette_evidence','reshape',match,shape=[10]);

    # Detail evidence. Full-length openings, narrow plackets, V lapels,
    # crew necklines, welt pockets and hem bands distinguish garments
    # which are almost indistinguishable by silhouette alone. All filters
    # below are analytic strips or differences of strips.
    norm=op('contrast','norm','image'); yy,xx=np.meshgrid((np.arange(28)+.5)/28,(np.arange(28)+.5)/28-.5,indexing='ij');
    def region(w):return w/(np.sum(w)+1.e-20)
    def stripe(pos,width):return np.exp(-.5*((xx-pos)/width)**2)
    def vertical(c,wd,delta,lo,hi):
        win=(yy>=lo)&(yy<=hi)
        return region(win*.5*(stripe(c+delta,wd)+stripe(c-delta,wd)))-region(win*stripe(c,wd))
    def band(lo,hi):return ((yy>=lo)&(yy<=hi)).astype(float)
    groups=[]; mats=[]; bs=[]
    def group(name,arr,threshold):
        start=len(mats); mats.extend(arr); bs.extend([-threshold]*len(arr)); groups.append((name,start,len(mats)))
    centres=np.array([-.046,0,.046])
    group('front_opening',[vertical(c+sl*(yy-.5),w,d,.29,.85) for c in centres for sl in (-.055,0,.055) for w,d in ((.026,.095),(.043,.113),(.057,.14))],.14)
    group('placket',[vertical(c+sl*(yy-.5),.018,.052,.28,.85) for c in centres for sl in (-.045,0,.045)],.115)
    group('light_fastening',[-vertical(c,.023,.078,.26,.84) for c in centres],.25)
    group('deep_v',[region(band(.105,.38)*(stripe(c-.115,.033)+stripe(c+.115,.033)))-region(band(.105,.38)*stripe(c,np.maximum(.018,.083*(.44-yy)/.35))) for c in centres],.22)
    group('lapel_edges',[sum(vertical(c+sgn*(tip-yy)*slope,.019,.045,.12,tip) for sgn in (-1,1))*.5 for c in centres for tip in (.28,.42,.53) for slope in (.34,.53,.71)],.13)
    group('hem_band',[(region(((np.abs(xx)<.19)&(yy>h-.15)&(yy<h-.05)).astype(float))-region(((np.abs(xx)<.19)&(yy>=h-.04)&(yy<=h)).astype(float))) for h in (.82,.89,.96)],.19)
    group('collar_contrast',[region(band(.11,.20)*(stripe(c-.12,.035)+stripe(c+.12,.035)))-region(band(.075,.15)*stripe(c,.048)) for c in centres],.23)
    # A dip in the centre of the neck that recovers to a uniform chest.
    group('scoop',[region(band(.18,.29)*stripe(c,.065))-region(band(.056,.129)*stripe(c,.065)) for c in centres],.21)
    # Small repeated near-horizontal contrasts in the sleeve fabric: a cue
    # to the baffles of a quilted coat. No dependency on a specific garment.
    group('baffles',[region(band(.28,.76)*stripe(s,.050)*np.maximum(np.cos(2*np.pi*yy/period+phase),0)**2)-region(band(.28,.76)*stripe(s,.050)*np.maximum(-np.cos(2*np.pi*yy/period+phase),0)**2) for s in [-.32,.32] for period in (.12,.17) for phase in (0,2.1,4.2)],.24)
    detail=act('positive_details',affine('detail_filters',norm,np.array(mats).reshape(len(mats),-1),bs))
    # Max within each filter family; the size of a family does not change
    # its vote. ReLU hinges below implement saturation of strong responses.
    longest=max(stop-start for _,start,stop in groups); align=np.zeros((len(groups)*longest,len(mats)))
    for n,(_,start,stop) in enumerate(groups):
        for j in range(longest):align[n*longest+j,start+j%(stop-start)]=1
    q=affine('arrange_details',detail,align); q=op('detail_groups','reshape',q,shape=[len(groups),1,longest]); q=pool('strongest_detail','maxpool',q,(1,longest),(1,longest));
    ng=len(groups); q=act('detail_hinges',affine('limit_detail',q,np.tile(np.eye(ng),(2,1)),np.r_[np.zeros(ng),-np.ones(ng)*1.45]));
    D=np.zeros((10,ng))
    # Evidence expressed in units of silhouette correlation. At matching
    # silhouettes the prior favours a plain knit unless a fastening shows.
    def votes(name, pairs):
        idx=[g[0] for g in groups].index(name)
        for k,a in pairs:D[k,idx]=a
    votes('front_opening',[(2,-.046),(4,.069),(6,.026)])
    votes('placket',[(2,-.016),(4,.004),(6,.037)])
    votes('light_fastening',[(2,-.011),(4,.043),(6,.016)])
    votes('deep_v',[(2,-.019),(4,.032),(6,.012)])
    votes('lapel_edges',[(2,-.016),(4,.036),(6,.022)])
    votes('hem_band',[(2,.011)])
    votes('collar_contrast',[(4,.005),(6,.010)])
    votes('scoop',[(0,.017),(2,.004),(6,-.004)])
    votes('baffles',[(4,.019),(2,-.004)])
    detail_scores=affine('garment_details',q,np.concatenate([D,-D],axis=1));

    # Open spaces, mostly of interest for distinguishing open footwear
    # from sneakers. A black channel continuing UPWARD from the arch is
    # not an enclosed hole; measure fill on opposite sides of a dark spot.
    # Use large crossing chords to avoid treating shoe texture as holes.
    H=[];B=[]
    def cavity(dx,dy,sgn):
        for y in range(9,21,2):
            for x in range(4,25,2):
                if 0<=y-dy and y+dy<28 and 0<=x-dx<28 and 0<=x+dx<28:
                    m=np.zeros((28,28));m[y-dy,x-dx]=.5;m[y+dy,x+dx]=.5;m[y,x]=-1
                    H.append(m.ravel()); B.append(-.64)
    cavity(3,0,1); cavity(0,3,1); cavity(3,2,1); cavity(3,-2,1)
    hole=act('open_cells',affine('bounded_dark_spots',masked,H,B)); C=np.zeros((10,len(H)));C[5]=.0027;C[7]=-.00055
    hole_scores=affine('openwork_evidence',hole,C)
    # Detect the broad category with silhouette evidence. Detail cues then
    # divide that evidence inside the category rather than competing for
    # its magnitude. E.g. trouser legs must not vote for a front opening.
    fastening,collar,openprofile,nosplit,tail,pattern,points,lace=extra_evidence(norm,masked)
    base=op('outline_and_leg_evidence','add',[match,nosplit]);
    # Placket, lapel, and pattern cues are correlated, so do not count
    # them all at full strength for the upper-garment tie break.
    details=op('interior_evidence','add',[detail_scores,fastening,collar,tail,pattern,points]);details=affine('interior_vote',details,.5*np.eye(10));
    upper=op('upper_item_choices','add',[base,details]);
    foot=op('footwear_choices','add',[base,openprofile,hole_scores,lace]);
    def family(tag,classes,choice):
        n=len(classes);select=np.eye(10)[classes];expand=select.T@np.ones((n,1));
        def maximum(name,src):
            v=affine(name+'_select',src,select);v=op(name+'_channels','reshape',v,shape=[1,1,n]);return pool(name,'maxpool',v,(1,n),(1,n))
        initial=maximum(tag+'_outline_confidence',base);winner=maximum(tag+'_detail_winner',choice)
        anchor=affine(tag+'_family_anchor',initial,expand);ref=affine(tag+'_reference',winner,-expand);score=affine(tag+'_mask',choice,select.T@select)
        return op(tag+'_scores','add',[anchor,ref,score])
    garm=family('long_sleeved',[2,4,6],upper);shoes=family('shoe',[5,7,9],foot);other=[0,1,3,8];sel=np.eye(10)[other];other_score=affine('other_outlines',base,sel.T@sel)
    op('scores','add',[garm,shoes,other_score]);

if __name__=='__main__':
    build(); dst=Path(sys.argv[1]); dst.mkdir(parents=True,exist_ok=True)
    (dst/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(dst/'weights.npz',**weights)
