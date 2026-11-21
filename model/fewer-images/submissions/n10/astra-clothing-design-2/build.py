"""An image-free pattern book and a small set of analytical image operators.

All dimensions in the pattern book are in units of garment length (clothes),
sole length (footwear), or body width (luggage), not source-image pixels.
"""
import json
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(202511)
U = rng.uniform

def chance(p):
    return U() < p


class Drawing:
    def __init__(self):
        self.paths = []

    def poly(self, pts, ink=1.):
        self.paths.append((np.asarray(pts, dtype=float), ink))

    def band(self, path, width, ink=1.):
        # Ribbon about a polyline, using a normal to its tangent.
        p = np.array(path, dtype=float)
        tangent = np.gradient(p, axis=0)
        normal = np.stack((-tangent[:, 1], tangent[:, 0]), 1)
        normal *= (.5 * width / np.maximum(np.linalg.norm(normal, axis=1), 1e-8))[:, None]
        self.poly(np.concatenate([p + normal, (p - normal)[::-1]]), ink)

    def reflect_poly(self, left_to_mid, ink=1.):
        # A half contour running along the right side of a symmetric object.
        q = list(left_to_mid)
        self.poly(q + [(-x, y) for x, y in q[::-1]], ink)

    def raster(self, flip=False):
        # Centre and scale this drawing using only its own geometry.
        both = np.concatenate([p for p, ink in self.paths if ink > 0.])
        lo, hi = both.min(0), both.max(0)
        size = (hi-lo).max()
        centre = (lo+hi)*.5
        scale = U(.93,1.015) / size
        # Small pose, hanger asymmetry and framing variation.
        lean = U(-.037,.037); tip = U(-.025,.025)
        tx, ty = U(-.026,.026), U(-.029,.029)
        n = 42
        result = np.zeros((n,n), dtype=float)
        rows = np.arange(n) + .5
        for q, value in self.paths:
            a = (q - centre) * scale
            x,y = a.T
            x = -x if flip else x
            xx = (.5 + tx + x + lean*y)*n
            yy = (.5 + ty + y + tip*x)*n
            # Scan-convert all edges in parallel, using even/odd fill.
            nx, ny = np.roll(xx,-1), np.roll(yy,-1)
            active = (rows[:,None] >= np.minimum(yy,ny)) & (rows[:,None] < np.maximum(yy,ny))
            cross = xx + (rows[:,None]-yy)*((nx-xx)/np.where(ny!=yy,ny-yy,1.))
            edges = np.clip(np.ceil(cross-.5),0,n).astype(np.int64)
            r,e = np.nonzero(active)
            toggles = np.zeros((n,n+1),np.uint8)
            np.bitwise_xor.at(toggles,(r,edges[r,e]),1)
            mask = (np.cumsum(toggles[:,:n],axis=1) % 2).astype(bool)
            result[mask] = value
        return result.reshape(14,3,14,3).mean(axis=(1,3))


def pants():
    d = Drawing()
    waist = U(.143,.275); hip = waist*U(.98,1.27)
    crotch=U(.24,.47)
    split=U(.001,.095); ankle=hip*U(.42,.91)
    hem=U(.96,1.0); kneeGap=split*U(.16,1.05)
    d.reflect_poly([(0,U(0,.019)),(.84*waist,0),(waist,.012),
                    (hip,U(.12,.24)),(hip*U(.77,1.04),.61),
                    (ankle+split,hem),(split,1.),(kneeGap,.63),
                    (split*.28,crotch+.09),(0,crotch)])
    # Leg-separating seam can remain visible where the two legs touch.
    if chance(.48):
        d.poly([(-.009,crotch),(0,crotch-.075),(.009,crotch),(split+.006,1.),(-split-.006,1.)],U(.12,.6))
    return d


def garment(style):
    d = Drawing()
    # shoulder endpoint, half chest/waist/hem, side of neck, sleeve-end height
    # The slimmer torso of a dress includes a skirt in its unit length.
    dress = style in ('frock','sundress','sleeved_dress','shift')
    outer = style in ('jacket','coat')
    woven = style in ('shirt','blouse','short_shirt')
    knit = style in ('knit','sweatshirt')
    tank = style in ('tank','sundress') or (style=='blouse' and chance(.21))
    if dress:
        sh = U(.123,.215); bust = sh*U(.83,1.17); neck = sh*U(.36,.67)
        waist=bust*U(.64,.99); hem=waist*U(1.45,2.85)
        under=U(.17,.31); waist_y=U(.28,.52); hip_y=U(.53,.76)
        hip=max(waist, min(hem,waist*U(1.13,1.61)))
        if style=='shift':
            waist = bust*U(.78,1.10); hip = waist*U(1.03,1.37); hem=hip*U(.8,1.14)
        if style=='sundress' and chance(.5): sh*=.85
        arm = U(.10,.25) if style!='sleeved_dress' else U(.33,.83)
        arm_w = U(.042,.088); slope=U(.04,.11)
        depth = U(.035,.14); collar_y=0.; tail=U(-.012,.022)
    else:
        if outer:
            sh=U(.235,.369) if style=='jacket' else U(.175,.295)
            bust=sh*U(.95,1.24); waist=bust*U(.85,1.04)
            hem=bust*(U(.96,1.19) if style=='coat' else U(.81,1.08))
            arm_w=U(.094,.162)*(sh/.3)**.55; arm=U(.86,1.04) if style=='jacket' else U(.64,.93)
            slope=U(.105,.215); neck=sh*U(.29,.55); depth=U(.006,.081); collar_y=U(-.053,0)
        elif knit:
            sh=U(.255,.397); bust=sh*U(.89,1.18); waist=bust*U(.88,1.02); hem=bust*U(.78,1.02)
            arm_w=U(.085,.147); arm=U(.87,1.075); slope=U(.055,.155)
            neck=sh*U(.29,.59); depth=U(.019,.096); collar_y=0.
            if style=='sweatshirt':
                sh*=1.065; arm_w*=1.12; hem*=.94
                if chance(.38): slope+=.06; neck*=1.06; collar_y=-.026
        elif woven:
            sh=U(.219,.36); bust=sh*U(.94,1.23); waist=bust*U(.78,1.04); hem=bust*U(.91,1.19)
            arm_w=U(.071,.129); arm=U(.765,1.035); slope=U(.065,.166)
            neck=sh*U(.30,.59); depth=U(.014,.091); collar_y=-U(0,.035)
            if style=='blouse':
                depth=U(.036,.18); collar_y=0.; arm=U(.38,1.015); arm_w*=U(.88,1.55)
            if style=='short_shirt': arm=U(.24,.56)
        else:
            sh=U(.247,.375); bust=sh*U(.96,1.24); waist=bust*U(.79,1.01); hem=bust*U(.89,1.12)
            arm_w=U(.099,.174); arm=U(.23,.455); slope=U(.051,.136); neck=sh*U(.37,.62)
            depth=U(.03,.15); collar_y=0.
        if tank:
            sh*=U(.74,.95); neck=sh*U(.53,.86); depth=U(.115,.268)
        under = U(.26,.38); waist_y=U(.47,.68); hip_y=U(.76,.89); hip=.52*hem+.48*waist
        tail=U(-.025,.025) if not woven else U(-.02,.096)
    if tank and dress: depth*=1.35
    uarm_y=under
    # Neck line: distinguish collar stands, scoop, V, and narrow straps.
    neck_mid = depth*U(.45,.86) if chance(.76) else depth*.46
    right = [(0,depth),(.54*neck,neck_mid),(.84*neck,depth*.40), (neck,collar_y)]
    if (outer or style in ('shirt','short_shirt')) and not tank:
        right += [(neck*1.04,collar_y),(neck*1.12,slope*U(.30,.64))]
    cap_y = slope if not tank else slope*.24
    right += [(.73*sh,cap_y*.61),(sh,cap_y)]
    if tank:
        right += [(sh*.96,under*.54),(bust*.93,under*.89),(bust,under)]
    else:
        # Sleeve root is incorporated into the torso; lower sleeve may hang
        # away, leaving a dark slit, or overlap the side of the garment.
        right += [(max(sh,bust*.98),slope+.034),(bust,under)]
    hem_y = 1.
    right += [(waist,waist_y),(hip,hip_y),(hem,hem_y-tail),(.74*hem,hem_y-.24*tail),(.28*hem,hem_y),(0,hem_y-.004)]
    if not tank:
        short = arm < under+.18
        spread = U(.055,.18) if short else U(.028,.16)
        if dress: spread *= .64
        if chance(.17) and not short: spread *= -.22
        end_inner=(bust if short else (.36*waist+.64*hem if not dress else bust)) + spread - (arm_w*.53 if not short else arm_w*.10)
        # shorter sleeves project laterally, and have an oblique cut at the cuff
        end_inner=max(end_inner, sh*.75)
        if short: end_inner=max(bust-.012,sh+spread*.3)
        for side in [-1,1]:
            ex=end_inner+U(-.014,.014); sy=arm+U(-.016,.016)
            cuff=arm_w*U(.70,1.06) if not short else arm_w
            till=sy - (cuff*U(.38,.87) if short else U(-.01,.018))
            pts=[(.81*sh,slope*.70),(sh,slope),(sh+arm_w*.35,slope+.057),
                 ((sh+ex+cuff)*.5+arm_w*.33,(slope+till)*.51),
                 (ex+cuff,till),(ex,sy),((ex+bust)*.5,(sy+uarm_y)*.5),
                 (bust*.92,uarm_y),(bust*.75,uarm_y-.034)]
            d.poly([(side*x,y) for x,y in pts])
    d.reflect_poly(right)
    if outer or style in ('shirt','short_shirt'):
        # Major construction details only, muted relative to the outline.
        if chance(.65):
            w=U(.006,.020); bottom=hem_y-U(.002,.026)
            d.poly([(0,depth),(w,depth+.09),(w*.70,bottom),(-w*.58,bottom),(-w*.32,depth+.08)],U(.42,.82))
        if chance(.48):
            end=depth+U(.07,.19)
            for side in [-1,1]:
                d.poly([(side*neck,collar_y),(side*neck*1.13,depth+.052),
                        (side*neck*.6,end),(0,depth+.024)],U(.62,.92))
        if outer and chance(.15):
            # Splayed lower ends of an unfastened jacket.
            w=hem*U(.05,.40)
            d.poly([(0,U(.54,.86)),(w,1.015),(-w,1.015)],0.)
    elif knit and chance(.44):
        # Rib knit at the waist.
        d.reflect_poly([(0,.941),(.99*hem,.941),(hem*.99,.99),(0,.99)],U(.80,.95))
    return d


def shoe(style):
    d=Drawing()
    if style in ('trainer','high_top','loafer'):
        toe=U(.087,.18); collar=U(.175,.33); tongue=collar*U(.94,1.42)
        if style=='high_top':
            collar=U(.32,.54); tongue=collar*U(.97,1.26)
        if style=='loafer':
            collar=U(.127,.27); tongue=collar*U(.85,1.36); toe*=.89
        spring=U(.016,.064); heel=U(.009,.095)
        low=[(0,-spring),(.036,-.49*spring),(.2,0),(.40,.003),(.65,-heel*.35),(.86,-heel*.32),(1,-heel*.56)]
        tx=U(.48,.68); ankle=U(.74,.85); notch=collar*U(.49,.84)
        if style=='loafer': notch*=.79
        upper=[(.994,-collar*.85),(U(.92,.985),-collar),(.9,-collar*U(.93,1.02)),
               (ankle,-notch),((ankle+tx)*.50,-notch*1.04), (tx,-tongue),
               (tx-.055,-tongue*U(.81,.98)),(.28,-toe*U(.96,1.36)),(.10,-toe),(.025,-toe*.80),(0,-toe*.55)]
        d.poly(low+upper)
        if chance(.64):
            thick=U(.028,.072)
            d.poly(low+[(x,y-thick) for x,y in low[::-1]],U(.73,1.))
        if style!='loafer' and chance(.54):
            for t in [.22,.45,.70]:
                x=.27+(tx-.27)*t; y=-(toe+(tongue*.91-toe)*t)
                d.band([(x-.019,y+.006),(x+.014,y+.044)],.015,U(.54,.85))
    elif style in ('boot','low_boot','heel_boot'):
        shaft=U(.47,.99) if style=='boot' else U(.29,.56)
        lift=U(.014,.106)
        if style=='heel_boot': shaft=U(.38,.91); lift=U(.105,.325)
        top=U(.53,.74); rear=U(.88,1.015); instep=top*U(.82,1.03)
        toe=U(.086,.19); spring=U(.011,.055); heel_x=U(.73,.91)
        d.poly([(0,-spring),(.08,-spring*.35),(.25,0),(.41,-.006),(.56,-lift*.53),
                (.73,-lift*.91),(heel_x,-lift),(heel_x+U(-.025,.035),-.009),(.965,-.023),
                (.987,-lift-.049),(.961,-lift-.125),(rear,-shaft*U(.87,1.04)),
                ((rear+top)*.5,-shaft*U(.955,1.025)),(top,-shaft),
                (top+U(-.017,.056),-shaft*.72),(instep,-max(toe+.032,shaft*.49)),
                (instep-.12,-max(toe*.97,shaft*.32)),(.28,-toe*.96),(.14,-toe),(.043,-toe*.88),(0,-toe*.57)])
    else:
        flipflop=style=='slides'; high=style=='heels'
        lift=U(.12,.39) if high else (U(.006,.057) if flipflop else U(.009,.16))
        thick=U(.025,.087); toe=U(.015,.046)
        xp=np.array([0,.06,.19,.36,.47,.61,.78,.92,1]); floor=np.array([-toe,-.44*toe,0,0,-.13*lift,-.58*lift,-lift,-1.02*lift,-.96*lift])
        foot=floor-thick; foot[-1]-=.012
        def bed(x): return float(np.interp(x,xp,foot))
        wedge=chance(.31) and not flipflop
        lower=floor.copy()
        if wedge: lower[4:]=[0,-.004,-.009,-.012,-.018]
        d.poly(list(zip(xp,lower))+list(zip(xp[::-1],foot[::-1])))
        if not wedge and lift>.06:
            wx=U(.03,.21); d.poly([(.965-wx,-lift-.012),(.997,-lift),(.972,-.007),(.968-wx*.69,-.006)])
        # Forefoot straps: a broad band folded over the toes or criss-cross strips.
        x1=U(.035,.16); apex=U(.20,.36); h=U(.072,.195); x2=U(.35,.54)
        if chance(.46) or flipflop:
            band=U(.035,.11); d.poly([(x1,bed(x1)+.006),(x1+.06,bed(x1)-.61*h),(apex,bed(apex)-h),
                                      (x2,bed(x2)-h*.65),(x2+band,bed(x2)+.01),
                                      (x2,bed(x2)+.008),(apex-.012,bed(apex)-h*.51),(x1+band,bed(x1)+.008)])
        else:
            for a,b in [(x1,apex+.095),(x2,apex)]:
                d.band([(a,bed(a)+.005),(.5*(a+b),bed(.5*(a+b))-.58*h),(b,bed(b)-h)],U(.034,.076))
        if not flipflop:
            ank=U(.20,.47) if high else U(.12,.31)
            xa=U(.61,.79); xb=U(.90,.99); cy=bed(.87)-ank
            if chance(.72):
                d.band([(xa,cy+.01),((xa+xb)*.5,cy-.012),(xb,cy+.002)],U(.029,.084))
                d.band([(xb,cy+.006),(xb-.045,(cy+bed(.84))*.5),(.84,bed(.84)+.01)],U(.029,.083))
                d.band([(xa+.015,cy),(.64,bed(.64)-ank*.56),(U(.43,.61),bed(.52)+.016)],U(.029,.084))
            else:
                d.band([(.36,bed(.36)-h*.8),(.66,bed(.66)-ank*.54),(.96,cy+.054),(.975,cy+.103)],U(.035,.088))
            if chance(.44):
                d.poly([(.80,bed(.8)+.01),(.87,bed(.87)-ank*.44),(xb,cy+.04),(.986,bed(.986)+.012)])
    return d


def bag(style):
    d=Drawing()
    h=U(.49,.91); bottom=U(.34,.49); mouth=U(.345,.495); bulge=U(.47,.53)
    if style=='pack': h=U(.92,1.55); mouth*=.82
    if style=='clutch': h=U(.38,.79); bottom=U(.38,.49); mouth=U(.39,.49)
    strap=U(.21,.80) if style=='tote' else U(.12,.54)
    if style=='pack': strap=U(.039,.24)
    if style=='clutch': strap=U(.015,.16)
    sep=mouth*U(.53,.94); w=U(.025,.073)
    spread=U(.57,1.) if style=='purse' else U(.76,1.06)
    if style!='clutch' or chance(.29):
        theta=np.linspace(0,np.pi,19); power=U(.68,1.35)
        x=sep*np.cos(theta)*(1-(1-spread)*np.sin(theta)); y=.09-(strap+.09)*np.sin(theta)**power
        if style=='purse' and chance(.39):
            x=np.array([-sep,-.04,sep]); y=np.array([.08,-strap,.08])
        d.band(list(zip(x,y)),w)
    if style=='pack':
        d.reflect_poly([(0,0),(mouth*.62,.015*h),(mouth,.13*h),(bulge*.95,.32*h),(bulge,.72*h),
                        (bottom,.955*h),(.72*bottom,h),(0,h)])
    else:
        dip=U(-.038,.075)*h if style!='purse' else U(.016,.18)*h
        d.reflect_poly([(0,dip),(.63*mouth,dip*.66),(mouth,0),(bulge,U(.08,.27)*h),
                        (bulge*U(.90,1.02),.51*h),(bottom,.94*h),(bottom*.88,.99*h),(.53*bottom,h),(0,h)])
    return d


styles=[('tee',0),('tank',0),('pants',1),('knit',2),('sweatshirt',2),('frock',3),('sundress',3),
        ('sleeved_dress',3),('shift',3),('jacket',4),('coat',4),('straps',5),('heels',5),('slides',5),
        ('shirt',6),('blouse',6),('short_shirt',6),('trainer',7),('high_top',7),('loafer',7),
        ('tote',8),('purse',8),('pack',8),('clutch',8),('boot',9),('low_boot',9),('heel_boot',9)]

layers=[]; weights={}

def lay(name,op,ins,**kw):
    layers.append(dict(name=name,op=op,inputs=[ins] if isinstance(ins,str) else ins,**kw));return name

def affine(name,inp,W,b=0):
    W=np.asarray(W,np.float32); weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype(np.float32)
    return lay(name,'dense',inp,out=len(W))

def conv(name,inp,W,b=0,p=0):
    W=np.asarray(W,np.float32);weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype(np.float32)
    return lay(name,'conv',inp,out=len(W),kernel=list(W.shape[2:]),stride=1,padding=p)

def rel(name,inp):return lay(name,'relu',inp)
def shape(name,inp,*s):return lay(name,'reshape',inp,shape=list(s))
def pool(name,inp,h,w,mode='maxpool'):return lay(name,mode,inp,size=[h,w],stride=[h,w])
def add(name,*ins):return lay(name,'add',list(ins))


def silhouettes(n):
    # Foreground matting: weighted sum of saturating ramps. It retains faint
    # material and discounts changes in albedo inside a garment.
    lifts=conv('soft_thresholds','image',np.ones((2,1,1,1)),[-.085,-.32])
    clipped=conv('albedo_subtraction',rel('above_thresholds',lifts),np.array([[-.78/.085,-.22/.32]])[:,:,None,None])
    linear=conv('shadow_gain','image',[[[[.78/.085+.22/.32]]]])
    mask=add('silhouette',linear,clipped)
    small=pool('antialiased_mask',mask,2,2,'avgpool');unit=lay('outline_norm','norm',small)
    blur=np.outer([1,2,1],[1,2,1])/16
    hp=np.zeros((3,3)); hp[1,1]=1.; hp-=blur
    edge=conv('surround_contrast',small,hp[None,None,:,:],p=1); edgen=lay('contour_norm','norm',edge)
    f1=conv('mask_channel',unit,[[[[1]]],[[[0]]]])
    f2=conv('edge_channel',edgen,[[[[0]]],[[[1]]]])
    descriptor=add('shape_descriptor',f1,f2)
    ng=len(styles); bank=np.empty((ng*n,392),np.float32)
    for g,(style,k) in enumerate(styles):
        for j in range(n):
            if k==1: d=pants()
            elif k in (5,7,9): d=shoe(style)
            elif k==8: d=bag(style)
            else: d=garment(style)
            a=d.raster(flip=k in (5,7,9) and chance(.14))
            # Identical mask/edge channels, normalized within THIS drawing only.
            pad=np.pad(a,1); h=a.copy()
            for dy in range(3):
                for dx in range(3): h-=blur[dy,dx]*pad[dy:dy+14,dx:dx+14]
            def coef(z): return ((z-z.mean())/(np.sqrt(z.var()+.00001)*196)).ravel()
            bank[g*n+j,:196]=.78*coef(a); bank[g*n+j,196:]=.22*coef(h)
    sims=affine('pattern_correlations',descriptor,bank)
    patterns=shape('pattern_book',sims,1,ng,n)
    return mask,small,shape('best_each_style',pool('match_alternatives',patterns,1,n),ng)


def region(x,y,x0,x1,y0,y1):
    a=((x>=x0)&(x<x1)&(y>=y0)&(y<y1)).astype(float)
    return a/max(1.,a.sum())


def grid(n):
    y,x=np.mgrid[:n,:n];return (x+.5)/n,(y+.5)/n


class Evidence:
    """Discriminants below use scalar named observations, not data-derived weights."""
    def __init__(self): self.parts=[]; self.names=[]

    def fields(self,src,pairs):
        names,mats=zip(*pairs); mats=np.array(mats).reshape(len(mats),-1)
        out=affine('observe_'+src,src,mats)
        self.parts.append((out,len(self.names),len(names)));self.names.extend(names)

    def gather(self):
        n=len(self.names); parts=[]
        for out,start,k in self.parts:
            W=np.zeros((n,k));W[start:start+k]=np.eye(k);parts.append(affine('index_'+out,out,W))
        return add('observations',*parts)


def centre_lines(e,photo):
    # Vertical contrast, measured over long segments. Coherent contrast of a
    # line is more indicative of a placket than a few isolated bright pixels.
    x,y=grid(28); patterns=[]; mids=np.arange(-2,3)+13.5
    for slope in [-.065,0,.065]:
        for middle in mids:
            for y0,y1 in [(.22,.86),(.27,.56),(.51,.82)]:
                delta=x*28-.5 - middle-slope*(y-.54)*28
                left=np.exp(-.5*((delta+2.1)/.94)**2);right=np.exp(-.5*((delta-2.1)/.94)**2)
                narrow=np.exp(-.5*(delta/.67)**2);span=(y>=y0)&(y<y1)
                def unit(a):
                    a=a*span;return a/a.sum()
                patterns.extend([unit(narrow)-.5*(unit(left)+unit(right)),.62*(unit(right)-unit(left))])
    # Also measure background vertical pattern strength away from the placket.
    side=[]
    for middle in [8.5,10.,17.,18.5]:
        dist=x*28-.5-middle
        span=(y>.31)&(y<.76)
        def bell(offset):
            a=np.exp(-.5*((dist-offset)/.7)**2)*span;return a/a.sum()
        side.append(bell(0)-.5*(bell(-1.8)+bell(1.8)))
    W=np.array(patterns+side);m=len(W)
    val=affine('line_integrals',photo,np.concatenate((W,-W)).reshape(2*m,-1))
    val=pool('line_magnitude',shape('line_polarities',val,1,2,m),2,1)
    val=affine('organize_lines',val,np.eye(m)[:len(patterns)])
    # For each pose, use both the intensity ridge and the intensity step.
    val=pool('ridge_or_step',shape('paired_operators',val,1,len(patterns)//2,2),1,2)
    val=shape('segment_responses',val,1,len(patterns)//6,3)
    allsegs=val
    # Dimensionless magnitudes in units of image contrast.
    for j,name in enumerate(['placket','upper_line','lower_line']):
        which=np.eye(3)[j]
        W=np.kron(np.eye(len(patterns)//6),which[None,:]); z=affine(name+'_by_pose',allsegs,W)
        z=pool(name+'_max',shape(name+'_positions',z,1,1,len(W)),1,len(W))
        e.fields(z,[(name,[[1.]])])
    W=np.zeros(m);W[len(patterns):]=1/len(side);e.fields('line_magnitude',[('other_lines',W)])


def details(e,photo,mask):
    x,y=grid(28); featurekern=[]
    vertical=np.zeros((3,3));vertical[1]=[-.5,1,-.5]
    diagonal=(np.eye(3)[::-1]-.5*(np.eye(3)+np.eye(3)[::-1])) # opposite diagonals
    diagonal/=np.sum(np.maximum(diagonal,0))
    for k in [vertical,vertical.T,diagonal,(vertical+vertical.T)*.5]:featurekern.extend([k,-k])
    z=conv('small_detail_filters',photo,np.array(featurekern)[:,None,:,:])
    mag=rel('half_wave_details',z);xx=x[1:-1,1:-1];yy=y[1:-1,1:-1]
    def field(op,regionmask):
        a=np.zeros((8,26,26));a[2*op:2*op+2]=regionmask;return a
    chest=region(xx,yy,.35,.65,.30,.74); off=region(xx,yy,.32,.68,.29,.74)
    middle=region(xx,yy,.452,.548,.28,.78)
    e.fields(mag,[('vertical_texture',field(0,chest)),('horizontal_texture',field(1,chest)),
                  ('collar_angles',field(2,region(xx,yy,.34,.66,.08,.255))),
                  ('button_detail',field(3,middle)),('field_detail',field(3,off))])
    # A neckline notch (shoulder straps without a high central collar),
    # or the converse (a lapel stand, hood, turtleneck, etc.).
    mid=region(x,y,.43,.57,.0,.16)
    side=(region(x,y,.28,.43,.0,.16)+region(x,y,.57,.72,.0,.16))*.5
    neck=side-mid
    rail=region(x,y,.40,.60,.16,.26)
    low_mid=region(x,y,.38,.62,.89,.985); hip_mid=region(x,y,.38,.62,.69,.84)
    e.fields(mask,[('neck_recess',neck),('deep_neck',side-rail),
                   ('lower_split',hip_mid-low_mid)])
    # Dark center or bright insert under a V-shaped opening at the neck.
    vfront=region(x,y,.447,.553,.065,.17)-.5*(region(x,y,.31,.437,.08,.175)+region(x,y,.563,.69,.08,.175))
    e.fields(photo,[('throat_contrast',vfront)])
    centre_lines(e,photo)

    # Horizontal rib band on the lower trunk. Look for a cross-body line,
    # not the edge against the backdrop. Discard locations near empty space.
    rows=[];inkrows=[]
    for t in np.arange(19,26):
        midx=(abs(x-.5)<.155);single=(abs(y*28-.5-t)<.1)*midx;single=single/single.sum()
        rows.append(.5*(np.roll(single,1,axis=0)+np.roll(single,-1,axis=0))-single)
        inkrows.append(np.roll(single,2,axis=0))
    k=len(rows);z=affine('ribbing_test',photo,np.array(rows+[-a for a in rows]).reshape(2*k,-1))
    z=pool('ribbing_magnitude',shape('ribbing_signs',z,1,2,k),2,1)
    bg=affine('ribbing_inside',mask,1.8*np.array(inkrows).reshape(k,-1),-1.8)
    cand=add('ribbing_candidates',shape('rib_strength',z,k),bg);z=pool('hem_rib',shape('hem_positions',cand,1,1,k),1,k)
    e.fields(z,[('hem_band',[[1.]])])


def gaps(e,small):
    n=14;m=n*n; yy,xx=np.divmod(np.arange(m),n);a=xx[:,None]-xx;b=yy[:,None]-yy
    # Four soft existence predicates per cell: any occupied cells on the
    # same ray? Crop out the cell itself. This measures interruptions in
    # material, which a smooth shoe upper usually lacks.
    rays=np.concatenate([(a==0)&(b>0),(a==0)&(b<0),(b==0)&(a>0),(b==0)&(a<0)]).astype(float)
    t=affine('empty_rays_test',small,-1.35*rays,1.0);z=rel('empty_rays',t)
    I=np.eye(m);Z=I*0
    W=np.block([[-I,-I,Z,Z],[Z,Z,-I,-I],[-I,-I,-I,-I]])
    room=affine('opposing_ray_support',z,W,1.0)
    ink=affine('exclude_object',small,np.tile(-I,(3,1)))
    holes=rel('between_material',add('test_gaps',room,ink))
    x,y=grid(n);one=np.ones((n,n))/m; zero=one*0
    def means(a,b,c):return np.stack((a,b,c))
    e.fields(holes,[('vertical_void',means(one,zero,zero)),('internal_void',means(zero,zero,one)),
                    ('handle_void',means(zero,zero,region(x,y,.22,.78,.06,.53))),
                    ('sole_void',means(zero,zero,region(x,y,.04,.96,.39,.75)))])
    # Mirror imbalance: compare both occupancy and row width in the halves.
    # A shoe has a toe and a heel; a purse usually has bilateral symmetry.
    flip=np.eye(m).reshape(m,n,n)[:,:,::-1].reshape(m,m)
    d=rel('asymmetric_mass',affine('compare_mirrors',small,I-flip))
    e.fields(d,[('asymmetry',np.full(m,2/m))])
    # Global silhouette proportions and upper/leg divergence landmarks.
    e.fields(small,[('above_centre',region(x,y,0,1,0,.43)),('below_centre',region(x,y,0,1,.57,1)),
                    ('top_mass',region(x,y,0,1,0,.22)),('belt_width',region(x,y,0,1,.35,.50)),
                    ('chest_width',region(x,y,0,1,.20,.35)),('hip_width',region(x,y,0,1,.65,.80)),
                    ('hem_width',region(x,y,0,1,.80,.94)),('hem_centre',region(x,y,.43,.57,.74,.97))])


def collars(e,photo):
    # Collar folds make bilateral, oblique ridges inside the shoulders.
    # Test mirror pairs of line segments, using the weaker response so
    # that one accidental edge (or a single sleeve) is not a collar.
    x,y=grid(28); W=[]; outside=[]; tall=[]
    for high in [.025,.075]:
        for length in [.14,.235]:
            for wide in [.082,.137]:
                for offset,reverse in [(0,False),(0,True)]:
                    for side in [-1,1]:
                        t=(y-high)/length; t=t.clip(0,1)
                        pos=.5+side*(wide*(.32+.68*(t if reverse else 1-t)))
                        dist=(x-pos)*28; seg=np.exp(-.5*((y-high-.5*length)/(.37*length))**2)
                        def stripe(dx):
                            a=seg*np.exp(-.5*((dist-dx)/.66)**2);return a/a.sum()
                        W.append(stripe(0)-.5*(stripe(1.65)+stripe(-1.65)))
                        # Do not confuse the outer item/background boundary
                        # at the neck with an interior fold. Shift the same
                        # line to its two sides to test material coverage.
                        outside.extend([stripe(-1.65),stripe(1.65)])
    W=np.array(W);m=len(W);j=m//2
    z=affine('lapel_ridge',photo,np.concatenate([W,-W]).reshape(m*2,-1));z=pool('lapel_polarity',shape('lapel_halves',z,1,2,m),2,1)
    pres=np.array(outside).reshape(2*m,-1);one=np.eye(m)
    bright=rel('missing_lapel_material',affine('check_fold_interior','silhouette',-pres,1.))
    penalty=affine('seam_at_boundary',bright,-.86*np.repeat(one,2,axis=1))
    z=add('inner_lapel',shape('fold_strength',z,m),penalty)
    q=affine('weaker_pair_sign',z,-one);q=pool('weakest_pair',shape('collar_pairs',q,1,j,2),1,2)
    q=affine('symmetrical_folds',q,-np.eye(j));q=pool('collar_found',shape('collar_choices',q,1,1,j),1,j)
    e.fields(q,[('lapel',[[1.]])])


def extent_and_notches(e,small):
    # Bounding-box height without locating the first/last row: in each
    # column count the cells that have ink above AND below (inclusive).
    # Side elevations of footwear can then be described in three bands,
    # independently of their vertical position in the photo.
    n=14; idx=np.arange(n*n); yy,xx=np.divmod(idx,n)
    x,y=grid(n);same=(xx[:,None]==xx);dy=yy[:,None]-yy
    above=same&(dy>=0);below=same&(dy<=0)
    upper=rel('no_material_above',affine('column_up',small,-2.0*above,1.))
    lower=rel('no_material_below',affine('column_down',small,-2.0*below,1.))
    span=rel('column_extent',affine('column_occupied',add('column_exterior',upper,lower),-np.eye(n*n),1.))
    e.fields(span,[('left_height',region(x,y,.07,.34,0,1)),('middle_height',region(x,y,.38,.66,0,1)),
                   ('right_height',region(x,y,.65,.93,0,1))])
    # A shaft fills a contiguous high block at one end. A high-top sneaker
    # is more often tallest at a forward-leaning tongue, not at the back.
    halves=affine('end_heights',span,[region(x,y,lo,hi,0,1).ravel() for lo,hi in [(.07,.34),(.65,.93)]])
    high=pool('taller_end',shape('both_ends',halves,1,1,2),1,2);e.fields(high,[('shaft_height',[[1.]])])
    # Count occupied rows. Saturating coverage across three columns avoids
    # mistaking a stray faint pixel for a tall object.
    W=np.repeat(np.eye(n),n,axis=1)
    rowempty=rel('rows_without_item',affine('row_coverage',small,-.63*W,1.))
    H=affine('item_height',rowempty,np.full((1,n),-1/n),1.);e.fields(H,[('height',[[1.]])])
    # Forks open to the lower edge, under a crotch or between a dangling
    # sleeve and the body. A split off to the side should not count as legs.
    side=(np.arange(n)[None,:] < np.arange(n)[:,None]).astype(float)
    sides=np.concatenate([np.kron(np.eye(n),side),np.kron(np.eye(n),side.T)])
    no=rel('one_side_empty',affine('fork_sides',small,-1.55*sides,1.))
    span=affine('fork_coverage',no,-np.tile(np.eye(n*n),(1,2)),1.)
    ink=affine('fork_empty',small,-np.eye(n*n))
    fork=rel('forks',add('fork_candidates',span,ink));center=region(x,y,.425,.575,.57,.97)
    e.fields(fork,[('leg_gap',center)])


def reason(e,base):
    src=e.gather(); tests=[];bounds=[]; votes=[];ng=len(styles)
    # A vote ramps from zero at lo to full strength at hi; negative
    # coefficients oppose a style. All endpoints are a priori thresholds.
    def cue(obs,lo,hi,scores):
        if isinstance(obs,str): obs={obs:1.}
        w=np.array([obs.get(k,0.) for k in e.names]); v=[]
        for s,k in styles:v.append(scores.get(k,0.)+scores.get(s,0.))
        tests.extend([w/(hi-lo),w/(hi-lo)]);bounds.extend([-lo/(hi-lo),-hi/(hi-lo)]);votes.extend([v,[-a for a in v]])
    cue({'placket':1,'other_lines':-.60},.105,.46,{2:-.061,4:.058,6:.053,0:-.011})
    cue({'upper_line':.38,'lower_line':.62,'other_lines':-.86},.12,.48,{4:.025,6:.019,2:-.024})
    cue({'button_detail':1,'field_detail':-.9},.035,.20,{6:.025,4:.019,2:-.017})
    cue('neck_recess',.11,.58,{0:.046,2:.004,4:-.060,'shirt':-.022,'blouse':.024,'sundress':.018})
    cue({'neck_recess':-1},.06,.40,{4:.049,6:.009,2:-.016,0:-.021})
    cue({'throat_contrast':-1},.20,.87,{4:.029,6:.023,2:-.021,'tee':.009})
    cue('throat_contrast',.15,.95,{4:.029,6:.020,2:-.015})
    cue('collar_angles',.21,.65,{4:.032,6:.024,2:-.012,0:-.010})
    cue({'lapel':1,'field_detail':-.6},.09,.42,{4:.040,'shirt':.048,'short_shirt':.051,'blouse':.017,2:-.032,0:-.010})
    cue('hem_band',.10,.48,{2:.030,4:.006,6:-.013})
    cue({'vertical_texture':.5,'horizontal_texture':.5},.24,.55,{6:.028,4:-.017})
    cue({'hem_width':1,'belt_width':-1},.055,.28,{3:.042,2:-.015})
    cue({'chest_width':1,'belt_width':-1},.10,.32,{0:.025,3:.012})
    cue('lower_split',.22,.77,{1:.022,4:.011,2:-.012})
    cue('internal_void',.015,.081,{5:.074,7:-.043,9:-.021})
    cue('vertical_void',.043,.143,{5:.057,7:-.039,9:-.027})
    cue('sole_void',.034,.155,{5:.046,7:-.028,9:-.018})
    cue('handle_void',.12,.52,{8:.126})
    cue('asymmetry',.10,.28,{8:-.085,9:.014,6:.008})
    cue('asymmetry',.035,.125,{8:-.023,3:-.012})
    cue({'below_centre':1,'above_centre':-1},.18,.53,{8:.041,3:.018})
    # High boots also differ from ordinary low trainers by the amount
    # of material extending toward the upper edge of the frame.
    cue('top_mass',.061,.23,{9:.055,7:-.046})
    cue('leg_gap',.16,.67,{1:.085,3:-.021})
    cue('height',.50,.68,{9:.055,7:-.029})
    cue({'shaft_height':1,'middle_height':-.54},.23,.45,{9:.049,7:-.031})
    cue({'height':-1},-.48,-.32,{7:.039,9:-.095})
    cue({'shaft_height':-1},-.31,-.21,{9:-.041})
    h=rel('condition_ramps',affine('cue_thresholds',src,tests,bounds))
    # Modest style priors: short-sleeve button-downs, ankle-height trainers,
    # and worn-open blouses are alternatives to more diagnostic patterns.
    priors={'tee':.019,'tank':.016,'pants':.017,'knit':.045,'sweatshirt':.032,
            'frock':.013,'sundress':.005,'sleeved_dress':-.009,'shift':-.005,
            'coat':.007,'jacket':.001,'straps':.006,'slides':-.008,'heels':.006,
            'shirt':-.012,'blouse':-.032,'short_shirt':-.049,'trainer':.034,'loafer':.002,'high_top':-.039,
            'tote':.015,'purse':.008,'pack':-.017,'clutch':.004,'boot':.014,'low_boot':-.024,'heel_boot':.006}
    vote=affine('construction_evidence',h,np.array(votes).T,[priors[s] for s,k in styles])
    per=add('style_scores',base,vote)
    # One max per class, not a sum biased toward the number of patterns.
    slots=4;W=np.zeros((10,slots,ng));bias=np.full((10,slots),-4.);counts=[0]*10
    for i,(s,k) in enumerate(styles):
        p=counts[k];W[k,p,i]=1;bias[k,p]=0;counts[k]+=1
    bytype=affine('group_styles',per,W.reshape(10*slots,-1),bias.ravel())
    s=pool('best_construction',shape('classes_and_styles',bytype,1,10,slots),1,slots)
    shape('scores',s,10)


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    mask,small,base=silhouettes(1024)
    photo=lay('photo_contrast','norm','image');e=Evidence();details(e,photo,mask);collars(e,photo);gaps(e,small);extent_and_notches(e,small);reason(e,base)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(out/'weights.npz',**weights)


if __name__=='__main__':main()

