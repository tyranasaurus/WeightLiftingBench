"""A bank of independently rendered shape hypotheses plus analytic detail detectors.
Only the standard library and numpy are needed. No input files.
"""
import json
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(872491)
def U(a,b): return rng.uniform(a,b)
def coin(p): return rng.random() < p

def interp(v, knots):
    return np.interp(v, [p[0] for p in knots], [p[1] for p in knots])

# Coordinates at subpixel centers, in fractions of the input window.
ww=(np.arange(56)+.5)/56-.5
xx0,yy0=np.meshgrid(ww,ww)

def field(ratio, signed=True, anchor=None):
    # Fit the posed object in a square, allowing modest margin and centering variation.
    size=U(.91,1.005) if not coin(.08) else U(.84,.94)
    w=size*min(1.,ratio); h=size*min(1.,1/ratio)
    cx=U(-.027,.027); cy=U(-.027,.027)
    if anchor=='shoe': cy+=U(.005,.07)*(1-h)
    if anchor=='handle': cy-=U(.015,.095)*h
    y=(yy0-cy)/h+.5
    x=(xx0-cx)/w-(y-.5)*U(-.055,.055)-U(-.012,.012)*np.sin(y*6)
    # Slight left/right asymmetry in drape.
    if signed: return x, y
    x=x+.5
    if coin(.045): x=1-x
    return x,y

# Horizontal scanline profiles. A garment consists of pieces with separate
# geometry: this preserves the real gap between the sleeve and the body.
def union(a,mask,v): np.maximum(a,mask*v,out=a)

def upper(c,kind):
    # T-shirts and tanks; pullovers; coats; blouses / button-front shirts.
    tank=kind=='tank'; short=kind=='short'; med=kind=='elbow'; long=not(tank or short or med)
    if c==0:
        ratio=U(.55,.79) if tank else U(.78,1.16)
        hem=U(.67,.89) if not tank else U(.87,1.0)
        shoulder=U(.62,.79); hip=U(.78,1.07); nwd=U(.21,.42); nd=U(.065,.19)
    elif c==2:
        ratio=U(.70,1.01); hem=U(.56,.78); shoulder=U(.62,.81); hip=U(.85,1.06)
        nwd=U(.16,.35); nd=U(.028,.13)
    elif c==4:
        ratio=U(.55,.91); hem=U(.62,.84); shoulder=U(.60,.80); hip=U(.92,1.17)
        nwd=U(.13,.28); nd=U(.012,.09)
    else:
        ratio=U(.61,.93) if long or med else U(.67,1.08)
        hem=U(.60,.86); shoulder=U(.59,.80); hip=U(.9,1.12)
        nwd=U(.15,.38); nd=U(.04,.18)
    if c==2 and kind=='wide': ratio=U(.89,1.17); hem=U(.70,.86)
    if tank:
        shoulder=U(.69,.88); nwd=U(.33,.58); nd=U(.13,.30); hem=U(.88,1.)
    x,y=field(ratio); a=np.zeros_like(x); t=abs(x)
    # Neckline and shoulder seam slope downwards away from the neck.
    V=coin(.20); cap=U(.0,.025); pitch=U(.042,.105)
    if kind=='collar': cap=U(.075,.13); pitch=U(.04,.083)
    neckline=cap+nd*np.maximum(0,1-(2*t/nwd)**(1 if V else 2.4))
    slope=cap+pitch*np.maximum(0,(2*t-nwd)/(shoulder-nwd))
    top=np.where(t<nwd/2,neckline,slope)
    if tank:
        bw=interp(y,[(0,shoulder*.5),(.12,shoulder*.50),(.32,U(.35,.44)),(.63,U(.31,.44)),(1,hem*.5)])
        # Curved armholes dive inwards and return to the chest.
        bw-=U(.035,.08)*np.maximum(0,1-((y-.20)/.17)**2)
        end=1-U(.0,.045)*(2*t)**2
        union(a,(y>=top)&(y<end)&(t<bw),1)
        return a
    sleeve_end=(U(.23,.43) if short else U(.47,.73) if med else U(.865,1.0))
    body_end=(U(.92,1.0) if long and c!=4 else 1.)
    if c==4 and coin(.5): sleeve_end=U(.72,.90)
    if c==2 and coin(.22): body_end=U(.78,.94); sleeve_end=U(.94,1.)
    # Ensure one piece reaches item length.
    if long and body_end<1 and sleeve_end<1:
        if sleeve_end>body_end:sleeve_end=1.
        else:body_end=1.
    pit=U(.22,.36); width=hem*.5
    bw=interp(y,[(0,shoulder*.46),(pit,width*U(.97,1.09)),(.63,width*U(.87,1.04)),(.91*body_end,width*hip),(body_end,width*hip*U(.94,1.035))])
    bend=U(-.027,.055) if c!=6 else U(.0,.11)
    body=(t<bw)&(y>=top)&(y<=body_end-bend*(t/width)**2) & (y>=0)
    # Sleeve is widest on top, with elbow and cuff farther out than the shoulder.
    hand=U(.425,.495); cuff=U(.082,.152)
    out=interp(y,[(0,shoulder/2),(cap+pitch,shoulder/2+U(.0,.025)),(min(.37,sleeve_end*.76),U(.421,.48)),(sleeve_end,(hand+cuff*.4) if long else .49)])
    if short or med:
        inside=interp(y,[(0,shoulder*.32),(pit*.77,width*.92),(sleeve_end,U(.345,.405))]); out+=U(.0,.024)*np.sin(np.pi*np.clip(y/sleeve_end,0,1))
        cuffline=sleeve_end-U(.045,.116)*(t-.345)/.15
    else:
        inside=interp(y,[(0,shoulder*.34),(pit,width*U(.81,1.00)),(.53,width*U(.96,1.16)),(sleeve_end,hand-cuff*.6)])
        cuffline=sleeve_end+(t-hand)*U(-.34,.34)
    sl=(t<out)&(t>inside)&(y>=slope)&(y<=cuffline)
    union(a,sl,U(.78,.97)); union(a,body,1.)
    # Vertical seams in open-front garments. Width and contrast are uncertain.
    open_front=(c==4 and coin(.80)) or (c==6 and coin(.59))
    if open_front:
        mid=.014*np.sin(y*5+U(-3,3))+(y-.4)*U(-.025,.025)
        slit=U(.005,.025)+U(0,.065)*np.maximum(0,(y-.72)/.28)
        a*=np.where((y>cap)&(abs(x-mid)<slit),U(.15,.74),1.)
    if kind=='collar' or ((c==6 or c==4) and coin(.55)):
        high=cap if kind=='collar' else U(.0,.035)
        collar=(y>=0)&(y<high+U(.08,.17)*(1-np.minimum(1,abs(t-.055)/.11)))&(t<nwd*.54)
        union(a,collar,U(.59,1.))
        if c in (6,4):
            cut=abs(t-interp(y,[(0,.045),(.09,.094),(.21,.02)]))<.012
            a*=np.where(cut&(y<.21),.51,1.)
    # Knit cuffs/ribbed waistband, contrasting neck binding.
    if c==2 and coin(.54): a*=np.where((body&(y>body_end-U(.038,.085)))|(sl&(y>sleeve_end-.048)),U(.72,.95),1.)
    return a

def trousers():
    ratio=U(.27,.49) if not coin(.12) else U(.45,.59)
    x,y=field(ratio); t=abs(x); a=np.zeros_like(x)
    crotch=U(.25,.44); endgap=U(.04,.21); wide=coin(.22)
    out=interp(y,[(0,U(.35,.46)),(.11,U(.46,.50)),(.28,U(.46,.50)),(.58,U(.36,.48)),(1,U(.29,.47) if not wide else U(.43,.53))]);
    inside=interp(y,[(0,0),(crotch,0),(.55,endgap*U(.12,.57)),(1,endgap)])
    union(a,(y>U(.0,.03)*(1-2*t))&(y<1-.024*(t-.2))&(t<out)&((y<crotch)|(t>inside)),1.0)
    a*=np.where((abs(t-(out+inside)*.50)<.018)&(y>.18),.82,1.)
    return a

def dress(kind):
    tank=kind in ('straps','column'); long=kind=='long'; full=kind=='full'
    ratio=U(.37,.68) if not full else U(.62,.87)
    x,y=field(ratio); t=abs(x); a=np.zeros_like(x)
    hem=U(.41,.5) if not full else .5
    waist=U(.24,.47); hip=min(.495,hem*U(.81,1.12))
    wa=hem*U(.43,.73) if not kind=='column' else hem*U(.67,.89)
    bust=wa*U(1.10,1.51); bust=min(.447,bust)
    sh=bust*U(.78,1.02); neck=sh*U(.33,.66)
    if kind=='column': hip=min(.5,hem*U(1.02,1.36))
    bw=interp(y,[(0,sh),(.12,bust*.89),(.21,bust),(waist,wa),(.62,hip if kind=='column' else .57*wa+.43*hem),(.86,hem*.97),(1,hem)])
    top=interp(t,[(0,U(.07,.23) if tank else U(.04,.10)),(neck,.005),(sh,U(.022,.060)),(.5,.16)])
    union(a,(t<bw)&(y>top)&(y<1-U(.0,.06)*(t/hem)**2),1.0)
    if not tank:
        sl=U(.18,.32) if not long else U(.46,.80)
        outer=interp(y,[(0,sh*.82),(.1,sh+U(.04,.09)),(sl,min(.5,bust+U(.08,.19)))]); inner=interp(y,[(0,sh*.75),(.18,bust*.9),(sl,bust+U(.0,.095))]);
        union(a,(y>top)&(y<sl-(t-bust)*.26)&(t>inner)&(t<outer),.91)
    if coin(.48): a*=np.where(abs(y-waist)<U(.009,.027),U(.55,.9),1.)
    return a

def shoe(c,kind):
    # Side elevation: last (toe, vamp, quarter), counter/shaft, topline, outsole.
    high=kind in ('hightop','shaft','tall','heel','wedgestrap','heeledboot')
    heel=kind in ('heel','heeledboot','wedgestrap')
    if c==5:
        if kind in ('flat','slides','flip','ballet'): ratio=U(2.65,4.6)
        elif kind=='sport': ratio=U(1.85,3.10)
        elif kind in ('heel','wedgestrap'): ratio=U(1.35,2.35)
        else: ratio=U(1.6,2.8)
    elif c==7: ratio=U(1.50,2.13) if high else U(2.12,3.54)
    else: ratio=U(.91,1.33) if kind=='tall' else U(1.19,1.86) if kind!='lowboot' else U(1.75,2.23)
    x,y=field(ratio,False,'shoe'); a=np.zeros_like(x); u=np.clip(x,0,1)
    toe=U(.012,.068); back=U(.92,.997); arch=U(.58,.74); heelx=U(.83,.94)
    lift=U(.30,.58) if heel else U(.02,.10) if c!=9 else U(.055,.17)
    if kind=='wedgestrap': lift=U(.20,.41)
    fore=U(.86,.955)
    bed=interp(u,[(0,fore-.06),(.08,fore),(.30,fore+.015),(.47,fore-.005),(arch,fore-lift*.57),(heelx,fore-lift),(1,fore-lift-.012)])
    th=U(.06,.18) if c!=9 else U(.038,.102)
    solelo=bed+th*np.minimum(1,np.maximum(.17,(u-toe)/.073)); solelo=np.minimum(1,solelo)
    solelt=toe+np.maximum(0,y-bed)*.2
    sole=(x>solelt)&(x<back-.13*(y-bed))&(y>=bed)&(y<solelo)
    union(a,sole,U(.73,1.))
    if heel or c==9:
        slim=kind=='heel' and coin(.65)
        left=heelx+(y-fore+lift)*(U(.032,.19) if slim else .045)-(.10 if not slim else .005)
        right=back-(y-fore+lift)*(U(.18,.30) if slim else .035)
        if kind=='wedgestrap':left=.32+(y-.8)*.40
        union(a,(x>left)&(x<right)&(y>bed)&(y<1-.04*(x-.85)),U(.78,1.))
    if c==5:
        # Individual bands, not a solid foot: a toe bridge, crossed instep
        # straps and optional ankle band form negative spaces above the footbed.
        if kind=='flip':
            strap=interp(u,[(.1,fore),(.44,U(.055,.24)),(.74,fore-.09)])
            union(a,(u>.10)&(u<.76)&(abs(y-strap)<U(.063,.125)),.93)
        elif kind in ('slides','flat','ballet'):
            st=U(.04,.17); peak=U(.30,.48); end=U(.52,.72)
            vtop=interp(u,[(st,fore-.06),(st+.08,fore-.39),(peak,U(.025,.19)),(end,U(.30,.58))]); bot=interp(u,[(st,fore),(peak-.06,fore),(end,fore-U(.15,.38))]); union(a,(x>st)&(x<end)&(y>vtop)&(y<bot),1.)
            if kind=='ballet':
                top=interp(u,[(.015,fore),(.12,.50),(.26,.34),(.48,.60),(.68,.72),(.88,.52),(.955,.19),(1,.38)])
                union(a,(x>toe)&(x<back)&(y>=top)&(y<bed+.025),.90)
        else:
            # Toe box may be open. Rounded fore-strap running obliquely into sole.
            st=U(.065,.24); end=U(.33,.52); peak=st+(end-st)*U(.55,.87)
            roof=interp(u,[(st,fore-.11),(peak,fore-U(.17,.43)),(end,fore-.22)])
            union(a,(u>st)&(u<end)&(y>=roof)&(y<bed+.025),.92)
        if kind in ('anklestrap','sport','heel','wedgestrap','cage'):
            ax=U(.68,.90); ay=U(.02,.15); px=U(.36,.62); wid=U(.021,.058)
            for start,end in [(px,ax),(min(.96,ax+.105),ax+.052)]:
                line= start+(end-start)*np.clip((bed-y)/(bed-ay+.001),0,1)
                union(a,(y<bed+.03)&(y>ay)&(abs(x-line)<wid),.94)
            union(a,(x>ax-.085)&(x<min(1.,ax+.14))&(y>ay)&(y<ay+U(.055,.154)),.91)
            if kind in ('sport','cage'):
                for ax in [.34,.54,.73]:
                    top=bed-.34*np.maximum(0,1-abs(u-ax)/.17)
                    union(a,(abs(u-ax)<.11)&(y>=top)&(y<bed+.02)&(abs(u-ax-(bed-y)*.23)<.052),.87)
        return a
    toeheight=U(.18,.38) if c==7 else U(.105,.22)
    start=U(.28,.43); peak=U(.55,.72) if c==7 else U(.58,.74)
    if c==7:
        crest=U(.005,.13); opening=U(.20,.48); backhi=U(.095,.39)
        if high:
            top=interp(u,[(toe,fore-.018),(.085,fore-toeheight*.64),(start,fore-toeheight),(peak-.10,.42),(peak,crest),(peak+.08,.18),(.82,.21),(back-.032,.095),(back,.20)])
        else:
            top=interp(u,[(toe,fore-.012),(.07,fore-toeheight*.79),(.21,fore-toeheight),(start,fore-toeheight*1.24),(peak-.085,.23),(peak,crest),(peak+.072,opening),(.84,opening+.018),(back-.048,backhi),(back,backhi+.10)])
    else:
        slope=U(.33,.54); shaftw=U(.30,.47); sx=U(.48,.67); rx=min(1.,sx+shaftw); bootTop=U(0,.06)
        # Toe rises gradually to instep. Shaft front then rises steeply from it.
        upperEnd=rx; bootFront=sx+U(-.09,.065)
        top=interp(u,[(toe,fore-.012),(.10,fore-toeheight*.88),(.24,fore-toeheight),(.34,fore-toeheight*1.30),(sx-.045,slope),(sx+.027,slope-.07),(rx,slope-.10)])
        # Main foot ends at the heel; separate shaft rectangle can lean or flare.
        front=sx+(bootFront-sx)*(slope-y)/max(.10,slope-bootTop)
        rear=back+(rx-back)*(slope-y)/max(.10,slope-bootTop)
        neckline=bootTop+.064*(1-(x-bootFront)/(rx-bootFront)) + U(-.05,.05)*np.sin(np.pi*(x-bootFront)/(rx-bootFront))
        union(a,(x>front)&(x<rear)&(y>neckline)&(y<slope+.06),.98)
    foot=(y>=top)&(y<bed+.02)&(x>toe+.10*np.maximum(0,fore-y-toeheight*.9))&(x<back+.028*np.maximum(0,bed-y))
    union(a,foot,1.)
    if c==7:
        # Tongue and lace line, heel counter and a midsole boundary.
        bound=interp(u,[(start,.67),(peak-.04,.25),(peak,.15)])
        a*=np.where(foot&(u>start)&(u<peak)&(abs(y-bound)<.035),.70,1.)
        if coin(.59): a*=np.where(foot&(y>bed-.04)&(y<bed),.62,1.)
    return a

def bag(kind):
    handle=kind not in ('clutch','satchel','pack')
    ratio=U(.71,1.39) if handle else U(1.10,1.90) if kind!='pack' else U(.61,.92)
    x,y=field(ratio,True,'handle' if handle else None); t=abs(x); a=np.zeros_like(x)
    lip=U(.23,.53) if handle else U(.02,.095); bot=U(.37,.49); mouth=U(.35,.49)
    if kind in ('hobo','shoulder'):lip=U(.37,.62)
    if kind=='pack': lip=U(.07,.17); mouth=U(.25,.36)
    hw=interp(y,[(lip-.1,mouth*.93),(lip,mouth),(lip+.10,mouth*U(.98,1.10)),(.78,U(.43,.5)),(.94,bot+.014),(1,bot*U(.82,.95))]);
    droop=U(-.03,.11) if kind!='hobo' else U(.12,.23)
    top=lip+droop*(1-np.minimum(1,t/mouth)**2)
    body=(y>=top)&(t<hw)&(y<1-.037*(t/bot)**4)
    union(a,body,1.)
    if handle or kind=='pack':
        hw=U(.14,.34) if kind not in ('shoulder','hobo') else U(.33,.47); bw=U(.022,.056) if kind!='hobo' else U(.055,.093)
        dy=lip+U(.03,.12); base=dy; yy=(y-base)/dy
        r=np.sqrt((x/hw)**2+yy**2); loop=(abs(r-1)<bw/hw)&(y<=base+.02)
        union(a,loop,U(.72,1.))
    if kind in ('satchel','clutch'):
        curve=lip+.33+.17*(1-(t/mouth)**2)
        a*=np.where((abs(y-curve)<.018)&body,.67,1.)
    return a

# Architecture helpers
layers=[]; weights={}
def node(name,op,arg,**kw):
    layers.append(dict(name=name,op=op,inputs=arg if isinstance(arg,list) else [arg],**kw));return name

def conv(name,arg,w,b):
    w=np.asarray(w,dtype='f'); weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(b,(len(w),)).astype('f'); k=w.shape[-1];return node(name,'conv',arg,out=len(w),kernel=[k,k],stride=1,padding=k//2)

def dense(name,arg,w,b=0):
    w=np.asarray(w,dtype='f'); weights[name+'.W']=w;weights[name+'.b']=np.broadcast_to(b,(len(w),)).astype('f'); return node(name,'dense',arg,out=len(w))

def avg2(name,arg):return node(name,'avgpool',arg,size=[2,2],stride=[2,2])

def softmask():
    # A compressive piecewise linear response: give even dark fabrics a vote
    # for foreground. The zero/background value maps to exactly zero.
    knots=np.array([.008,.105,.34,.69]); val=np.array([0,.76,.94,1.]); slopes=np.diff(val)/np.diff(knots); co=np.diff(np.r_[0,slopes,0]);
    h=conv('intensity_knots','image',np.ones((4,1,1,1)), -knots)
    h=node('above_knots','relu',h); return conv('foreground',h,co.reshape(1,-1,1,1),0)

def selfnorm(im):
    f=im.ravel(); f=f-f.mean(); return f/(np.sqrt(np.dot(f,f))*np.sqrt(f.size)+1e-12)

groups=[['short','tank','elbow'],['legs'],['plain','wide','collar'],['straps','short','long','column','full'],['plain','collar','jacket'],['flat','slides','flip','ballet','sport','anklestrap','cage','heel','wedgestrap'],['plain','short','elbow','tank','collar'],['low','hightop'],['tote','shoulder','hobo','satchel','pack','clutch'],['shaft','lowboot','tall','heeledboot']]
# Relative frequencies here are design choices about apparel styles. They
# do not depend on any images. The bank is padded to equal length per class.
probs=[[.72,.23,.05],[1],[.65,.20,.15],[.22,.31,.14,.15,.18],[.45,.34,.21],[.09,.14,.09,.07,.13,.16,.09,.14,.09],[.44,.16,.15,.07,.18],[.84,.16],[.34,.16,.13,.16,.07,.14],[.38,.23,.20,.19]]

def make_bank():
    per=2048; W=np.empty((10*per,196),'f'); bs=np.zeros(10*per,'f')
    for c,opts in enumerate(groups):
        for j in range(per):
            kind=rng.choice(opts,p=probs[c]);
            if c in (0,2,4,6): f=upper(c,kind)
            elif c==1: f=trousers()
            elif c==3: f=dress(kind)
            elif c==8: f=bag(kind)
            else: f=shoe(c,kind)
            small=f.reshape(14,4,14,4).mean((1,3)); i=c*per+j
            W[i]=selfnorm(small)
            # Generic style priors only; the overlap of fine-grained clothing
            # labels is real. Short sleeves without blouse details favor tee.
            if (c==6 and kind in ('short','tank')): bs[i]=-.038
            if c==0 and kind=='elbow':bs[i]=-.025
            if c==7 and kind=='hightop':bs[i]=-.021
            if c==2 and kind=='collar':bs[i]=-.013
            if c==5 and kind=='ballet':bs[i]=-.028
            if c==9 and kind=='lowboot':bs[i]=-.012
    return W,bs,per

# Hand-coded, independent evidence for the ambiguous classes: central
# closures, spread collars, vents, fabric evenness, and thin straps.
# These measurements are performed at inference, not during weight design.
def detail_branch(mask, silhouette5):
    axis=(np.arange(28)+.5)/28; X,Y=np.meshgrid(axis-.5,axis)
    def box(l,r,t,b): return ((X>=l)&(X<r)&(Y>=t)&(Y<b)).astype(float)
    def unit(z):return z/max(1e-10,z.sum())
    def absconv(name,arg,kernels):
        k=np.array(kernels,dtype=float); u=conv(name,arg,np.concatenate([k,-k])[:,None],0); return node(name+'_rect','relu',u)
    def sparse(k,entries):
        a=np.zeros((k,k));
        for x,y,v in entries:a[k//2+y,k//2+x]+=v
        return a
    # Use relative brightness for interior contrast, absolute for garment tone.
    lum=node('relative_tone','norm','image')
    # Direction-selective local second differences. These respond to a bright
    # or a dark center line without requiring dark buttons specifically.
    linekern=[sparse(5,[(0,0,1),(-d,0,-.5),(d,0,-.5)]) for d in (1,2)]
    linekern += [sparse(5,[(0,0,1),(-1,-1,-.5),(1,1,-.5)]),sparse(5,[(0,0,1),(-1,1,-.5),(1,-1,-.5)])]
    edge=absconv('line_contrast',lum,linekern)
    features=[]; names=[]; sources={'relative_tone':1,edge:8,'image':1,mask:1}
    def feat(name,src,w,b=0):features.append((src,np.asarray(w).reshape(sources[src],28,28),b));names.append(name)
    def lineweight(a,chans):
        w=np.zeros((8,28,28));
        for k in chans:w[k]=w[k+4]=a/len(chans)
        return w
    # Offset and slope hypotheses for an open front, contrasted with nearby
    # parallel pieces of cloth (to suppress response on printed fabrics).
    fronts=[]
    for k,mid in enumerate([-.057,-.029,0.,.029,.057]):
        for j,slope in enumerate([-.058,0.,.058]):
            u=X-mid-(Y-.5)*slope; midline=(abs(u)<.035)&(Y>.23)&(Y<.86); flank=(abs(u)>.061)&(abs(u)<.18)&(Y>.23)&(Y<.86)
            nm='fastening%d_%d'%(k,j); fronts.append(nm); feat(nm,edge,lineweight(unit(midline)-.8*unit(flank),[0,1]))
    gaps=[]
    for k,cx in enumerate([-.035,0,.035]):
        u=abs(X-cx); b=unit((u<.04)&(Y>.49)&(Y<.95)); s=unit((u>.07)&(u<.18)&(Y>.49)&(Y<.95));
        nm='part%d'%k; gaps.append(nm);feat(nm,mask,s-b)
    # Lapel edges: test for converging obliques on the two sides of a V,
    # and for a standing collar filled above the shoulder line.
    lap=[]
    for j,depth in enumerate([.17,.26,.35]):
        u=abs(X)-.12*(1-(Y-.035)/depth); z=(Y>.035)&(Y<.035+depth); ring=(abs(u)<.04)&z; outside=(abs(u)>.06)&(abs(u)<.13)&z
        nm='lapel%d'%j; lap.append(nm);feat(nm,edge,lineweight(unit(ring)-.65*unit(outside),[0,1,2,3]))
    local=unit(box(-.25,.25,.22,.79)); feat('pattern',edge,lineweight(local,[0,1,2,3]));
    feat('tone','image',local)
    feat('fill',mask,local)
    feat('neck',mask,unit(box(-.085,.085,0,.075))-.65*unit(box(-.19,-.09,.04,.14))-.35*unit(box(.09,.19,.04,.14)))
    # Texture continuity along the center: buttons and stitching against a
    # smooth or incoherent shirt fabric. Contrast between sleeves and body.
    feat('neck_dark',lum,unit(box(-.18,.18,.13,.22))-unit(box(-.071,.071,.035,.11)))
    # Is the shoe upper perforated? Detect horizontal / diagonal strands
    # surrounded above AND below by background, not just a high edge energy.
    # Below the uppermost outline a solid sneaker has little such response.
    k=sparse(5,[(0,0,1),(0,-2,-.5),(0,2,-.5)])
    hole=node('open_strand','relu',conv('strap_edges',mask,k[None,None],-.54))
    sources[hole]=1
    feat('strap',hole,unit(box(-.42,.43,.31,.71)))
    # Compare a detached sleeve with the shadow of the armhole further in.
    limb=unit((abs(X)>.34)&(abs(X)<.46)&(Y>.56)&(Y<.86)); under=unit((abs(X)>.25)&(abs(X)<.33)&(Y>.56)&(Y<.86))
    feat('separate_sleeve',mask,limb-under)
    # Button plackets are patterns down the whole front, unlike a local
    # graphic or a crease. Integrate signed contrast along the front, then
    # allow either polarity (consistent light or consistent dark placket).
    plackets=[]
    for i,dx in enumerate([-.035,0,.035]):
        z=abs(X-dx); region=(Y>.24)&(Y<.84)
        w=unit((z<.036)&region)-unit((z>.059)&(z<.13)&region)
        for j,sgn in enumerate([-1,1]):
            nm='placket%d_%d'%(i,j); plackets.append(nm);feat(nm,lum,w*sgn)
    # Aggregate input-specific measurements; unused rows in each branch are zero.
    branches=[]; n=len(features)
    for src,ch in sources.items():
        rows=np.zeros((n,ch,28,28)); bias=np.zeros(n)
        for i,(s,w,b) in enumerate(features):
            if s==src:rows[i]=w; bias[i]=b
        branches.append(dense('measure_'+src,src,rows.reshape(n,-1),bias))
    raw=node('measurements','add',branches)
    # Signed features stay linear until tested by the thresholds below.
    def take_max(nm,items):
        v=np.zeros((len(items),n)); v[np.arange(len(items)),[names.index(t) for t in items]]=1
        q=dense(nm+'_select',raw,v); q=node(nm+'_axis','reshape',q,shape=[1,1,len(items)]);return node(nm+'_max','maxpool',q,size=[1,len(items)],stride=[1,1])
    merged=names+['front','vent','vee','seam','sandality']; ex=[]
    q=np.zeros((len(merged),1)); q[-1]=1
    ex.append(dense('sandal_affinity',silhouette5,q))
    e=np.zeros((len(merged),n));e[:n]=np.eye(n);ex.append(dense('direct_cues',raw,e))
    for nm,items in [('front',fronts),('vent',gaps),('vee',lap),('seam',plackets)]:
        m=take_max(nm,items); v=np.zeros((len(merged),1));v[merged.index(nm),0]=1; ex.append(dense(nm+'_cue',m,v))
    allfeat=node('garment_cues','add',ex)
    # Each ramp is clipped to [0,1]. Score changes are in units of
    # normalized template similarity, not logits estimated on examples.
    tests=[]; biases=[]; votes=[]
    def ramp(form,lo,hi,votes_dict):
        w=np.zeros(len(merged));
        for key,val in form.items():w[merged.index(key)]+=val/(hi-lo)
        g=np.zeros(10)
        for c,val in votes_dict.items(): g[c]=val
        tests.extend([w,w]);biases.extend([-lo/(hi-lo),-hi/(hi-lo)]);votes.extend([g,-g])
    # Strong center line with respect to adjacent material suggests an opening.
    ramp({'front':1},.12,.52,{2:-.072,4:.051,6:.029,0:-.020})
    ramp({'vent':1},.095,.52,{4:.052,2:-.042,6:.015})
    ramp({'vee':1},.14,.48,{4:.028,6:.026,2:-.018})
    ramp({'neck':1},-.04,.54,{4:.048,6:.014,2:-.013,0:-.024})
    ramp({'neck_dark':1},.19,1.3,{6:.016,0:.010,4:-.011})
    # Cloth and construction: a smooth crew-neck with no longitudinal
    # seam favors a pullover; printed/wrinkled blouses favor a shirt.
    ramp({'pattern':1},.22,.59,{6:.067,2:-.018,4:-.010})
    ramp({'pattern':1,'front':.35},.26,.66,{6:.026})
    ramp({'seam':1},.26,.86,{4:.017,6:.022,2:-.021,0:-.007})
    ramp({'separate_sleeve':1},.14,.60,{2:.021,4:.006,0:-.027})
    ramp({'front':-1,'pattern':-.23},-.20,-.085,{2:.031,4:-.016})
    ramp({'fill':.48,'tone':-1},.15,.34,{4:.011,6:.012})
    # The physical cue overrides a silhouette only for plausible footwear.
    # Against a solid sneaker, separated straps at the same location are
    # positive sandal evidence; merely having sleeves is not evidence.
    ramp({'strap':1,'sandality':.090},.006,.045,{5:.145,7:-.041,9:-.025})
    h=node('cue_strength','relu',dense('cue_thresholds',allfeat,tests,biases))
    return dense('detail_scores',h,np.transpose(votes),[.010,0,.014,0,-.008,-.016,-.019,.010,0,.005])

def main(out):
    fg=softmask()
    # Mix normalized luminance and normalized soft foreground. A dark item
    # still has a recognizable silhouette, but internal negative spaces count.
    sh=node('shape_unit','norm',avg2('shape_small',fg)); lum=node('tone_unit','norm',avg2('tone_small','image'))
    lum=conv('tone_fraction',lum,np.array([[[[.16]]]]),0); sh=conv('shape_fraction',sh,np.array([[[[.84]]]]),0)
    q=node('appearance','add',[sh,lum]); w,b,per=make_bank(); t=dense('drawings',q,w,b)
    t=node('by_class','reshape',t,shape=[10,1,per]);
    # Max in each independent sub-bank, followed by a blend of best match
    # and consensus. Pool scores only at inference, never pool drawings.
    t=node('style_matches','maxpool',t,size=[1,per//8],stride=[1,per//8]); m=node('closest','maxpool',t,size=[1,8],stride=[1,8]); av=node('stable','avgpool',t,size=[1,8],stride=[1,8]);
    m=dense('best_match_scores',m,np.eye(10)*.64); av=dense('consensus_scores',av,np.eye(10)*.36)
    shape=node('shape_scores','add',[m,av])
    def cluster(nm,indices):
        tab=np.eye(10)[indices]; sel=dense(nm+'_classes',shape,tab)
        axis=node(nm+'_shape_axis','reshape',sel,shape=[1,1,len(indices)])
        return node(nm+'_shape_best','maxpool',axis,size=[1,len(indices)],stride=[1,1])
    # Other footwear shapes can be similar to sandals: let straps decide
    # within the footwear group, but not across bags or long garments.
    foot=cluster('footwear',[5,7,9]); other=cluster('not_footwear',[0,1,2,3,4,6,8]);
    affinity=node('footwear_lead','add',[dense('footwear_side',foot,[[1]]),dense('object_side',other,[[-1]])]);
    deficit=node('footwear_deficit','relu',dense('footwear_disagreement',affinity,[[-1]]));
    gate=dense('footwear_gate',deficit,[[-1]])
    detail=detail_branch(fg,gate)
    # A hierarchical vote. The broad silhouette chooses between lower-body
    # garments, upper-body garments, footwear and bags. Collar and button
    # evidence can then change WHICH top wins, without turning a pair of
    # trousers into a shirt. Subtracting a common group score preserves the
    # order of subclasses. These maxima are only computed on the test input.
    scores=node('fine_scores','add',[shape,detail]); corrections=[]
    for nm,ix in [('upper',[0,2,4,6]),('shoe',[5,7,9])]:
        parent=cluster(nm,ix); w=np.eye(10)[ix]
        v=dense(nm+'_fine_select',scores,w); v=node(nm+'_fine_axis','reshape',v,shape=[1,1,len(ix)]); v=node(nm+'_fine_best','maxpool',v,size=[1,len(ix)],stride=[1,1]);
        send=np.zeros((10,1)); send[ix,0]=1
        corrections.extend([dense(nm+'_coarse_route',parent,send),dense(nm+'_fine_route',v,-send)])
    node('scores','add',[scores]+corrections)
    path=Path(out);path.mkdir(parents=True,exist_ok=True);(path/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n');np.savez(path/'weights.npz',**weights)

if __name__=='__main__': main(sys.argv[1])
