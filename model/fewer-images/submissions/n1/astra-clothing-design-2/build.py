"""A data-free Fashion silhouette and construction-detail classifier.

All spatial parameters below are in garment units (body length, shoe length,
handbag width, etc.). Each sample is a separate analytic model, not an update
of a set of model parameters. There is no reference or calibration dataset.
"""
import json
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(418097)

def u(a,b): return rng.uniform(a,b)
def chance(p): return rng.random()<p
def choose(a): return a[rng.integers(len(a))]
def bound(x,a,b): return max(a,min(b,x))

class Drawing:
    def __init__(self):
        self.parts=[]
        self.flip_y=False
    def poly(self, pts, value):
        pts=np.asarray(pts,dtype=np.float64)
        self.parts.append((pts, bound(value,0,1)))
    def sym(self, pts, value):
        self.poly(pts+[(-x,y) for x,y in pts[::-1]],value)
    def line(self, pts, width, value):
        # A strip about a polygonal path, with mitered joins.
        pts=np.asarray(pts,dtype=np.float64)
        d=np.diff(pts,axis=0)
        normals=np.stack([-d[:,1],d[:,0]],axis=1)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1.e-9)
        n=np.concatenate([normals[:1],(normals[:-1]+normals[1:])*.5,normals[-1:]])
        n/=np.maximum(np.sum(n*np.vstack([normals,normals[-1:]]),axis=1,keepdims=True),.35)
        self.poly(np.vstack([pts+n*width*.5,(pts-n*width*.5)[::-1]]),value)

# A neck opening is part silhouette (back of the neck) and part interior
# detail (the inside face of the garment, visible inside the front neckline).
def upper(kind):
    g=Drawing(); tee=kind==0; knit=kind==2; coat=kind==4; shirt=kind==6
    fabric=u(.36,.93)
    short=tee and chance(.86) or shirt and chance(.19) or knit and chance(.018)
    tank=tee and not short and chance(.9) or shirt and not short and chance(.06)
    blouse=shirt and chance(.27)
    collar=shirt and not blouse or coat and chance(.68)
    cardigan=knit and chance(.06)
    opening=(coat or shirt and not blouse or cardigan)
    jacket=coat and chance(.48)
    if tee:
        chest=u(.30,.46); hem=chest*u(.82,1.12); shoulder=chest*u(.77,1.02)
    elif knit:
        chest=u(.32,.49); hem=chest*u(.74,1.01); shoulder=chest*u(.75,1.03)
    elif shirt:
        chest=u(.28,.432); hem=chest*u(.88,1.16); shoulder=chest*u(.78,1.02)
    elif jacket:
        chest=u(.33,.46); hem=chest*u(.83,1.06); shoulder=chest*u(.87,1.04)
    else:
        chest=u(.235,.36); hem=chest*u(.96,1.25); shoulder=chest*u(.84,1.03)
    waist=(chest+hem)*u(.425,.505)
    armhole=u(.26,.40); nw=shoulder*(u(.24,.45) if not tank else u(.54,.83))
    sy=u(.055,.135) if coat or shirt and not blouse else u(.075,.19)
    if tank: sy=u(.013,.06); shoulder*=u(.82,.99)
    back=u(.008,.07); front=u(.065,.23)
    if tank or blouse: back=u(.025,.125);front=u(.12,.32)
    roll=knit and chance(.14) or coat and not collar and chance(.32)
    if roll: back=.01;front=u(.035,.10)
    # Bottom shape: a shirttail drops below the side seams; ribbed hems
    # and waistbands are much closer to horizontal.
    tail=u(.045,.14) if shirt and not blouse else u(-.012,.05)
    y_neck=-u(.025,.074) if roll or collar else u(-.008,.012)
    # Arms lie behind the torso. Give the two arms almost the same cut,
    # but do not force them to hang at precisely the same angle.
    if not tank:
        endy=(u(.24,.46) if short else (u(.58,.87) if coat and not jacket else
               u(.78,1.00) if shirt else u(.83,1.08)))
        if not short and (blouse or tee) and chance(.35): endy=u(.49,.78)
        sweep=u(.13,.285) if short else u(.00,.20)
        cuff=u(.105,.188) * (1 if (jacket or not coat) else .85)
        elbowbulge=u(.01,.086); taper=u(.99,1.48)
        for s in [-1,1]:
            ey=endy+u(-.026,.026); sleeve=fabric*u(.83,1.035)
            if short:
                xo=shoulder+sweep*u(.92,1.08); yo=ey-u(.045,.12)
                xi=xo-u(.075,.18); yi=ey+u(.0,.045)
                outline=[(nw,.045),(shoulder,sy),(shoulder+sweep*.53,(sy+yo)*.5-.014),
                         (xo,yo),(xi,yi),(chest*.94,armhole),(chest*.7,armhole+.018)]
            else:
                wrist=(hem*.38+shoulder*.62)+sweep+u(-.023,.023)
                xo=wrist+cuff/2; xi=wrist-cuff/2
                tilt=u(-.028,.025); yo=ey+tilt; yi=ey-tilt
                elbow=(shoulder+xo)*.5+elbowbulge
                outline=[(nw,.048),(shoulder,sy),(shoulder+u(.043,.092),sy+u(.08,.16)),
                         (elbow,.53*ey+.04),(xo+u(.0,.025),.82*ey),(xo,yo),(xi,yi),
                         (xi-u(-.009,.02),ey-.065),(elbow-cuff*taper,.53*ey+.04),
                         (chest*.94,armhole),(chest*.70,armhole+.025)]
            g.poly([(s*x,y) for x,y in outline],sleeve)
            if not short and chance(.72):
                g.poly([(s*xo,yo),(s*xi,yi),(s*xi,yi-.046),(s*xo,yo-.046)],sleeve*u(.70,1.06))
    shoulder_top=(shoulder, sy)
    cx=hem*u(.44,.74)
    neckside=[(0,y_neck+back),(nw*.54,y_neck+back*.82),(nw*.86,y_neck+back*.5),(nw,y_neck)]
    top=neckside+[(shoulder*.72,sy*.51),shoulder_top]
    if tank:
        top=neckside+[(shoulder,sy),(shoulder*.91,sy+.10),(chest*.85,armhole*.82)]
    g.sym(top+[(chest,armhole),(waist,.62),(hem,.99-tail),
               (hem*.90,1-tail*.54),(cx,1.0),(0,1+u(-.012,.012))],fabric)
    # Inside of collar.
    collar_col=fabric*(u(.32,.76) if chance(.84) else u(1.05,1.32))
    v=chance(.20 if not collar else .6)
    g.sym(neckside+[(nw*.91,y_neck+front*.44),
                    (nw*(.24 if v else .58),y_neck+front*(.77 if v else .88)),
                    (0,y_neck+front)],collar_col)
    if opening:
        span=u(.009,.027) if shirt else u(.012,.045)
        cx=u(-.016,.016); tx=cx+u(-.018,.018)
        g.poly([(cx-span,front*.77),(cx+span,front*.77),(tx+span,.98),
                (tx-span,1)], fabric*(u(.32,.79) if chance(.68) else u(1.04,1.44)))
        if coat and chance(.29):
            # Open lapels / an open jacket, over what is underneath.
            op=u(.065,.185);t=u(.25,.58)
            g.poly([(-nw*.92,.025),(nw*.92,.025),(op*.42,t),(op,1),
                    (-op*.85,1),(-op*.38,t)],fabric*(u(.19,.70) if chance(.69) else u(1.12,1.71)))
    if collar:
        spread=nw*u(1.11,1.50); length=u(.09,.17) if shirt else u(.12,.29)
        for s in [-1,1]:
            g.poly([(s*nw*.84,y_neck),(s*spread,.025),(s*spread*.96,length*.57),
                    (s*nw*.65,length),(s*nw*.09,front*.87)],fabric*u(.63,1.19))
    if knit and chance(.50):
        g.sym([(0,1-tail),(hem*.94,.986-tail),(hem,.942-tail),(0,.945)],fabric*u(.75,1.04))
    if not opening and chance(.15):
        # A printed panel breaks up some otherwise plain pull-on garments.
        w=u(.09,.22);yy=u(.20,.34);hh=u(.065,.18)
        g.poly([(-w,yy),(w,yy+.012),(w*.90,yy+hh),(-w*.89,yy+hh*.91)],fabric*u(.57,1.22))
    return g

def pants():
    g=Drawing(); fabric=u(.35,.93);short=chance(.035)
    hip=u(.155,.237) if not short else u(.28,.45); wb=hip*u(.78,.99)
    crotch=u(.23,.45) if not short else u(.44,.66)
    lc=hip*u(.54,.98); leg=hip*u(.27,.54)
    wide=chance(.18)
    if wide: leg=hip*u(.49,.72)
    knee=hip*u(.81,1.04)
    for s in [-1,1]:
        l=lc+u(-.021,.021);foot=leg*u(.94,1.06);y=u(.981,1.016)
        g.poly([(0,0),(s*wb,.005),(s*hip,.145),(s*hip*1.01,crotch-.02),
                (s*knee,.64),(s*(l+foot),y-.006),(s*(l+foot*.1),y+.006),
                (s*(l-foot),y),(s*max(.004,l-foot*.88),.72),
                (s*max(.001,(l-foot)*.44),.51 if not short else .78),(0,crotch),(-s*.018,.14)],fabric*u(.90,1.05))
    g.sym([(0,0),(wb,.005),(wb*1.015,.042),(0,.044)],fabric*u(.56,1.10))
    return g

def dress():
    g=Drawing();fabric=u(.35,.9); waist_y=u(.32,.50)
    chest=u(.15,.235);shoulder=chest*u(.80,1.06);waist=chest*u(.63,.96)
    hem=waist*u(1.29,2.86); hip=max(waist*1.12,hem*u(.60,.88))
    if chance(.22):
        # Shift, sheath or sweater dress; no wide A-line skirt.
        waist=chest*u(.81,1.04);hip=chest*u(1.00,1.27);hem=hip*u(.84,1.10)
    mode=choose(['strap','strap','sleeveless','sleeveless','short','short','short','long','long','threequarter','bandeau'])
    nw=shoulder*u(.30,.55); sy=u(.037,.083); ay=u(.145,.22)
    if mode in ['strap','sleeveless']:nw=shoulder*u(.52,.82);sy=u(.00,.035)
    by=u(.012,.055);front=u(.067,.166)
    if mode in ['strap','bandeau']:by=u(.077,.158);front=by+u(.015,.04)
    if mode in ['long','threequarter','short']:
        ey=(u(.51,.69) if mode=='long' else u(.31,.51) if mode=='threequarter' else u(.17,.255))
        arm=chest*u(.31,.55); sweep=u(.025,.13)
        for s in [-1,1]:
            outer=shoulder+sweep+u(-.01,.01); end=ey+u(-.014,.014)
            g.poly([(s*nw,.02),(s*shoulder,sy),(s*(shoulder+.045),sy+.055),
                    (s*(outer+.014),end-.07),(s*outer,end),(s*(outer-arm),end+.009),
                    (s*chest*.94,ay),(s*chest*.65,ay+.023)],fabric*u(.83,1.05))
    nt=[(0,by),(nw*.51,by*.91),(nw*.88,by*.52),(nw,.004)]
    if mode=='strap': nt=[(0,by),(nw*.87,by*.90),(nw,.012)]
    top=nt+[(shoulder,sy)]
    if mode in ['strap','sleeveless']: top+=[(shoulder*.84,ay*.68)]
    if mode=='bandeau': top=[(0,.037),(chest*.53,.0),(chest*.90,.026)]
    g.sym(top+[(chest,ay),(waist,waist_y),(waist*.64,.54),(0,.58)],fabric)
    if mode!='bandeau':
        g.sym(nt+[(nw*.77,front*.67),(nw*.44,front*.91),(0,front)],fabric*u(.41,.84))
    g.sym([(0,waist_y-.012),(waist,waist_y-.012),(hip,waist_y+(1-waist_y)*.44),
           (hem,.97),(hem*.82,.989),(hem*.39,.996),(0,1.)],fabric*u(.72,1.12))
    if chance(.36):g.sym([(0,waist_y-.014),(waist,waist_y-.014),(waist,waist_y+.020),(0,waist_y+.020)],fabric*u(.42,.90))
    return g

# Shoes are drawn with positive height measured upwards from the floor. The
# coordinate reversal in the renderer makes the toe/heel equations easier to read.
def shoe(kind):
    g=Drawing(); g.flip_y=True
    sandal=kind==5; sneaker=kind==7; boot=kind==9
    color=u(.32,.89);rubber=u(.22,.9)
    pump=sandal and chance(.30); fancy=boot and chance(.48)
    heeled=(pump or fancy and chance(.82) or sandal and chance(.13))
    wedge=heeled and chance(.22 if sandal else .10)
    heel=(u(.135,.39) if heeled else u(.00,.063)); heel_start=u(.65,.84)
    platform=(u(.024,.087) if not sneaker else u(.046,.126))
    if sandal and chance(.14):platform=u(.085,.17)
    toe=u(.025,.084) if not heeled else u(.008,.058)
    arch=u(.32,.56); rear=u(.94,.99)
    def bed(x):
        return platform + heel*bound((x-arch)/(.79-arch),0,1) + toe*max(0,1-x/.25)**1.7
    sole=[(.006,bed(.006)-platform*.43),(.036,bed(.036)),(.15,bed(.15)),(.30,bed(.30)),
          (arch,bed(arch)),(.59,bed(.59)),(.76,bed(.76)),(.90,bed(.90)),(rear+.010,bed(rear)),
          (rear+.017,bed(rear)-platform*.74)]
    if heeled and not wedge:
        hwidth=u(.042,.11) if pump and chance(.7) else u(.12,.28)
        sole +=[(rear-.006,.008),(rear-hwidth,.006),(heel_start,heel*.90),(.58,bed(.58)-platform*.54)]
    else: sole +=[(rear,.016),(.87,.005),(.68,.009),(.52,.011)]
    sole +=[(arch,0),(.18,.002),(.063,toe*.39),(.012,toe*.74)]
    g.poly(sole,rubber)
    tongue=u(.53,.70);frontankle=u(.14,.31); lowtoe=u(.072,.155)
    if sneaker or boot:
        if sneaker:
            counter=u(.19,.34);peak=counter+u(.005,.122)
            high=chance(.085)
            if high:peak+=u(.08,.19);counter+=u(.06,.16)
            top=[(.006,bed(0)-.03),(.024,bed(.024)+lowtoe*.42),(.093,bed(.093)+lowtoe*.87),
                 (.22,bed(.22)+lowtoe),(.34,bed(.34)+lowtoe*1.18),
                 (tongue-.115,bed(tongue-.115)+peak*.78),(tongue-.042,bed(tongue)+peak),
                 (tongue+.011,bed(tongue)+peak*.96),(tongue+.047,bed(tongue)+peak*.68),
                 (.79,bed(.79)+counter*.66),(.894,bed(.894)+counter*.90),
                 (.94,bed(.94)+counter),(.969,bed(.97)+counter*.90),(rear+.004,bed(rear)+.080)]
        else:
            shaft=u(.30,.65) if not chance(.16) else u(.62,.91)
            toeupper=lowtoe*u(.82,1.25); throat=u(.48,.68); ftop=throat+u(-.114,.104);rtop=u(.90,1.03)
            top=[(.006,bed(.006)-.025),(.024,bed(.024)+toeupper*.43),(.090,bed(.09)+toeupper*.83),
                 (.24,bed(.24)+toeupper),(.38,bed(.38)+toeupper*1.28),
                 (throat-.06,bed(throat-.06)+shaft*.46),(throat,bed(throat)+shaft*.64),
                 (ftop-.008,bed(.9)+shaft*u(.90,1.01)),(ftop+.045,bed(.9)+shaft),
                 ((rtop+ftop)*.5,bed(.9)+shaft*u(.90,1.03)),(rtop,bed(.9)+shaft*u(.88,1.04)),
                 (rear-.03,bed(.9)+shaft*.45),(rear+.003,bed(.9)+shaft*.18)]
        g.poly(top+[(rear+.010,bed(rear)-.010),(.81,bed(.81)-.018),
                    (.62,bed(.62)-.018),(arch,bed(arch)-.018),(.21,bed(.21)-.018),(.056,bed(.056)-.014)],color)
        if sneaker:
            # Eyestay, toe cap, heel panel and laces do not have a universal
            # polarity. The outsole may similarly be either dark or light.
            g.poly([(.26,bed(.26)+lowtoe),(.36,bed(.36)+lowtoe*1.12),(tongue-.035,bed(tongue)+peak*.96),
                    (tongue+.034,bed(tongue)+peak*.72),(.41,bed(.41)+lowtoe*.59)],color*u(.54,1.23))
            for t in [.24,.45,.66,.84]:
                x=.31+(tongue-.32)*t; h=bed(x)+lowtoe+(peak-lowtoe)*t
                g.line([(x-.023,h-.007),(x+.022,h-.050)],u(.010,.024),u(.32,.94))
            g.line([(.06,bed(.06)-.029),(.21,bed(.21)-.034),(.45,bed(.45)-.033),(.71,bed(.71)-.034),(.96,bed(.96)-.026)],u(.008,.018),rubber*u(.59,1.1))
        elif chance(.47):
            g.line([(ftop+.095,bed(.88)+shaft*.92),(throat+.084,bed(.88)+shaft*.43),
                    (throat+.106,bed(.80)+.085)],u(.009,.024),color*u(.39,1.31))
    else:
        # Open shoes: a vamp over the toes, then one or more independent
        # supports (instep, heel counter, slingback or ankle strap).
        slide=not pump and chance(.24); flip=not pump and not slide and chance(.17)
        h=u(.108,.228); xx=u(.055,.202); length=u(.17,.33)
        if flip:
            g.line([(.13,bed(.13)),(.28,bed(.28)+h*.60),(.46,bed(.46)+h),(.66,bed(.66))],u(.032,.079),color)
            g.line([(.42,bed(.42)+h*.87),(.31,bed(.31)+h*.63),(.235,bed(.235))],u(.022,.054),color)
        else:
            g.poly([(xx-.04,bed(xx-.04)-.013),(xx-.014,bed(xx)+h*.60),(xx+.036,bed(xx)+h*.87),
                    (xx+length*.77,bed(xx)+h*1.1),(xx+length,bed(xx+length)+h*.93),
                    (xx+length*.78,bed(xx+length*.78)+h*.48),(xx+length*.72,bed(xx+length*.72)-.018)],color)
        if not (slide or flip):
            ankle=u(.69,.82); ah=u(.17,.34) if not pump else u(.18,.37);hh=bed(.85)+ah
            strap=u(.035,.094);x=u(.40,.56)
            if chance(.83):
                g.line([(x,bed(x)),(x+.065,bed(x)+h*.75),(ankle,hh*.94),
                        (ankle+.094,hh*.91)],strap,color*u(.73,1.09))
            if chance(.60):
                g.poly([(.79,bed(.79)-.02),(.85,bed(.85)+ah*.64),(.94,hh),(.975,hh*.92),
                        (rear+.004,bed(.96)+.046),(rear,bed(rear)-.01)],color*u(.80,1.06))
            else:
                g.line([(.68,bed(.68)),(.77,hh*.85),(.92,hh),(.958,hh*.88),(.88,hh*.67)],strap,color)
            if chance(.52):g.line([(ankle-.048,hh*.91),(ankle+.004,hh*1.038),(.922,hh*1.015),(.964,hh*.94)],strap*.84,color*u(.83,1.12))
    return g

def bag():
    g=Drawing(); style=choose(['hand','hand','hand','tote','tote','satchel','shoulder','clutch','duffel','pack'])
    color=u(.32,.91);trim=u(.25,.92)
    bh=u(.48,.83) if style not in ['pack','clutch','duffel'] else (u(.87,1.30) if style=='pack' else u(.34,.58))
    top=u(.37,.52); bot=u(.39,.50); bulge=u(.006,.037)
    handle= u(.20,.54) if style not in ['clutch','duffel','shoulder','pack'] else (
        u(.54,.96) if style=='shoulder' else u(.02,.25))
    if style in ['satchel','clutch'] and chance(.48):handle=0
    anch=(u(.18,.32) if style not in ['shoulder','satchel'] else u(.32,.46)); hw=u(.024,.067)
    if handle:
        n=14;t=np.linspace(-1,1,n+1);lean=u(-.12,.12);corner=u(.38,1.15)
        points=[(anch*v+lean*(1-v*v),.055-(handle+.055)*max(0,1-v*v)**corner) for v in t]
        if chance(.41) and style not in ['pack','shoulder']:
            g.line([(x+.034*(1-v*v),y+handle*.20*(1-v*v)) for (x,y),v in zip(points,t)],hw*.81,trim*u(.68,1.12))
        g.line(points,hw,trim)
    dip=u(-.020,.096) if style!='pack' else -bh*.13
    if style=='duffel':top=u(.29,.40);bot=.46
    if style=='pack':top=u(.22,.39);bot=u(.39,.47)
    g.sym([(0,dip),(top*.79,dip*.49),(top,.012),(top+.033,bh*.093),
           (max(top,bot)+bulge,bh*u(.47,.76)),(bot+.012,bh*.91),(bot-.024,bh*.982),
           (bot*.70,bh),(0,bh+.010)],color)
    if style in ['satchel','clutch','pack'] or chance(.27):
        w=top*u(.78,.985);y=bh*u(.28,.59);start=max(.034,dip)
        g.sym([(0,start),(w,start+.018),(w*.96,y*.73),(w*.84,y),(0,y+.050)],color*u(.60,1.18))
    if style in ['hand','tote'] and handle and chance(.72):
        for s in [-1,1]:g.line([(s*anch,.03),(s*anch*.98,bh*u(.21,.59))],hw*.93,trim)
    return g

# Rasterize the component polygons of ONE article onto a 2x oversampled
# canvas. Tighten the bounding rectangle and center it as in a catalog photo.
N=56
axis=np.arange(N)+.5

def render(g,kind):
    points=np.concatenate([p for p,c in g.parts]);counts=[len(p) for p,c in g.parts]
    if g.flip_y:points[:,1]*=-1
    x,y=points.T; h=max(np.ptp(y),.1);w=max(np.ptp(x),.1);t=(y-y.min())/h
    # Slightly rumpled hem, leaning sleeve, camera pitch etc.
    amount=.022 if kind not in [5,7,9] else .014
    x += w*(u(-amount,amount)*np.sin(np.pi*t)+u(-.050,.050)*(t-.5))
    y += h*(u(-amount,amount)*np.sin(2*np.pi*x/w+u(0,6.28))+u(-.038,.038)*x/w)
    if chance(.5 if not g.flip_y else .055):x*=-1
    x*=u(.94,1.07); y*=u(.97,1.04)
    scale=N*u(.915,.996)/max(np.ptp(x),np.ptp(y))
    x-=(x.max()+x.min())*.5;y-=(y.max()+y.min())*.5
    x*=scale;y*=scale
    x+=N*.5+u(-.85,.85);y+=N*.5+u(-.95,.95)
    gray=np.zeros((N,N));i=0
    for m,(_,c) in zip(counts,g.parts):
        v=points[i:i+m];i+=m
        x0=max(0,int(np.floor(v[:,0].min())));x1=min(N,1+int(v[:,0].max()))
        y0=max(0,int(np.floor(v[:,1].min())));y1=min(N,1+int(v[:,1].max()))
        if x1<=x0 or y1<=y0:continue
        nxt=np.roll(v,-1,axis=0);ys=axis[y0:y1,None]
        crossing=(v[:,1]>ys)!=(nxt[:,1]>ys)
        dxdy=(nxt[:,0]-v[:,0])/np.where(abs(nxt[:,1]-v[:,1])>1.e-10,nxt[:,1]-v[:,1],1.)
        intercept=v[:,0]+dxdy*(ys-v[:,1]);intercept[~crossing]=np.inf
        # Sorted crossings come in pairs bounding filled spans. Using two
        # intervals handles concave outlines / multiple folds on a scanline.
        intercept.sort(axis=1);xx=axis[None,x0:x1];ink=np.zeros((y1-y0,x1-x0),bool)
        for j in range(0,m-1,2):
            if not np.isfinite(intercept[:,j]).any():break
            ink|=(xx>=intercept[:,j:j+1])&(xx<intercept[:,j+1:j+2])
        gray[y0:y1,x0:x1][ink]=c
    return gray

def project(a): return a.reshape(14,4,14,4).mean(axis=(1,3))
def unit(a):
    # Mean and energy of ONE template only; normalized cross-correlation
    # uses the same imagewise standardization on the observation.
    z=a-a.mean();return (z/np.sqrt(np.mean(z*z)+1e-5)/z.size).flatten()

def detail(a):
    p=np.pad(a,1)
    sm=sum(w*v*p[j:j+14,i:i+14] for j,w in enumerate([.25,.5,.25]) for i,v in enumerate([.25,.5,.25]))
    return a-sm

def model(k):
    if k in (0,2,4,6):return upper(k)
    if k==1:return pants()
    if k==3:return dress()
    if k==8:return bag()
    return shoe(k)

layers=[]; weights={}
def op(name,kind,inputs,**settings):
    if isinstance(inputs,str):inputs=[inputs]
    layers.append(dict(name=name,op=kind,inputs=inputs,**settings));return name

def linear(name,kind,source,W,b=0,padding=0):
    W=np.asarray(W,dtype=np.float32);b=np.broadcast_to(np.asarray(b,np.float32),(W.shape[0],)).copy()
    weights[name+'.W']=W;weights[name+'.b']=b
    args=dict(out=W.shape[0]);
    if kind=='conv':args.update(kernel=list(W.shape[2:]),stride=1,padding=padding)
    return op(name,kind,source,**args)

def pool(name,kind,src,size,stride):return op(name,kind,src,size=list(size),stride=list(stride))

def foreground():
    # Turn brightness into soft material occupancy. A morphological close
    # partially repairs isolated black weave lines and resampling artifacts.
    a=np.array([1/.085,1/.28]); z=linear('deficit','conv','image',(-a).reshape(2,1,1,1),[1,1]);z=op('positive_deficit','relu',z)
    z=linear('occupancy','conv',z,np.array([-.70,-.30]).reshape(1,2,1,1),1)
    neg=linear('padded_ink','conv',z,[[[[1]]]],padding=2)
    dil=pool('dilate','maxpool',neg,(3,3),(1,1));neg=linear('invert','conv',dil,[[[[-1]]]])
    clo=pool('erode_inverse','maxpool',neg,(3,3),(1,1));z2=linear('repair','conv',clo,[[[[-.55]]]])
    z1=linear('original_fraction','conv',z,[[[[.45]]]]);z=op('repaired_occupancy','add',[z1,z2]);return z

def front_detail():
    # Detect sustained plackets and center openings. Paired contrasts detect
    # a dark line, a bright line, or a step between the two front panels.
    src=op('contrast_input','norm','image');Y,X=np.mgrid[:28,:28]+.5
    bank=[];groups=4
    for band in range(groups):
        left,right=[(-2,2),(-2,2),(-2,2),(-2,2)][band]
        for location in [0,-5,5]:
            for dx in np.linspace(left,right,5):
                for slope in [-.09,0,.09]:
                    for typ in ['thin','thick','step']:
                        cy,sy=[(15,5),(19,4.3),(11.7,3.8),(16,6.7)][band]
                        cx=14+dx+location+(Y-cy)*slope
                        env=np.exp(-.5*((Y-cy)/sy)**2)
                        def blob(offset,width):
                            a=env*np.exp(-.5*((X-cx-offset)/width)**2);return a/a.sum()
                        if typ=='step': filt=blob(-1.10,.70)-blob(1.10,.70)
                        else:
                            w=.49 if typ=='thin' else .77;sep=1.50 if typ=='thin' else 2.15
                            filt=blob(0,w)-.5*(blob(-sep,w)+blob(sep,w))
                        bank.extend([filt.flatten(),-filt.flatten()])
    d=linear('front_panel_filters','dense',src,bank);d=op('panel_arrangement','reshape',d,shape=[1,groups*3,90]);d=pool('strongest_panel_edge','maxpool',d,(1,90),(1,90))
    # Lack of a stripe favors pullovers; a particularly strong stripe down
    # the middle, compared with random wrinkles, favors a coat / button-up.
    coef=np.zeros((3,groups,3)); b=[]
    for j,(th,sub) in enumerate([(.17,.40),(.42,.48),(.82,.52)]):
        coef[j,:,0]=1/groups;coef[j,:,1:]=-sub/(2*groups);b.append(-th)
    d=linear('closure_levels','dense',d,coef.reshape(3,-1),b);d=op('closure_evidence','relu',d)
    # Saturating, piecewise-linear semantic score corrections in NCC units.
    A=np.zeros((10,3));bi=np.zeros(10)
    for k,off,ww in [(0,.003,[-.020,.015,.005]),(2,.017,[-.094,.041,.046]),
                     (4,-.006,[.053,-.007,-.040]),(6,-.009,[.065,-.049,-.015])]:
        A[k]=ww;bi[k]=off
    return linear('construction_cues','dense',d,A,bi)

def shoe_detail(occ):
    # SIDE-VIEW footwear: the inside of an ankle boot shaft normally contains
    # material, whereas the space inside a buckled ankle strap does not. Scan
    # at several shaft heights, always relative to the (flat) sole. The sole
    # and the absence of material above/below it on the toe end gate this cue.
    Y,X=np.mgrid[:28,:28]+.5
    def box(x0,x1,y0,y1):
        m=((X>=x0)&(X<x1)&(Y>=y0)&(Y<y1)).astype(float)
        # A window can project beyond the raster. Divide by its nominal
        # lattice area so the invisible part is exactly zero-padded.
        area=(np.ceil(x1-.5)-np.ceil(x0-.5))*(np.ceil(y1-.5)-np.ceil(y0-.5))
        return m/max(1,area)
    occupancy=[];bias=[];reg=[]
    for flip in [False,True]:
        for base in (18.5,20.5,22.5,24.5,26.5):
            for raised in (0,2):
                for height in (4,7,10,13,16):
                    toe=box(3,12,base-2-raised,base-raised)
                    void_under=box(3,12,base,min(29,base+2))
                    void_over=box(2,9,base-10-raised,base-7-raised)
                    shaft=box(16,26,base-height-3,base-height+2)
                    for m,complement,b in [(toe,False,.75),(void_under,True,.78),(void_over,True,.80),(shaft,False,.36)]:
                        if flip:m=m[:,::-1]
                        occupancy.append(m.flatten()*(1 if complement else -1))
                        # For a desired-empty box the condition is 1-mean >= b.
                        bias.append(b-1 if complement else b)
                    core=box(17,25,base-height-2,base-height+2)
                    around=(box(14,17,base-height-3,base-height+2)+box(25,27,base-height-3,base-height+2)+box(17,25,base-height-5,base-height-2))/3
                    if flip: core=core[:,::-1];around=around[:,::-1]
                    # A deep open region with supporting straps nearby.
                    reg.append((.63*around-core).flatten())
    z=linear('foot_conditions','dense',occ,occupancy,bias);z=op('foot_shortfalls','relu',z)
    n=len(bias)//4
    z=op('foot_rule_groups','reshape',z,shape=[1,n,4]);z=pool('foot_shortfall_mean','avgpool',z,(1,4),(1,4))
    q=linear('open_upper_contrast','dense',occ,reg,-.035);z=linear('foot_context','dense',z,-4.2*np.eye(n));z=op('open_shoe_evidence','add',[q,z]);z=op('open_shoe_positive','relu',z)
    z=op('shoe_views','reshape',z,shape=[1,1,n]);z=pool('strongest_open_upper','maxpool',z,(1,n),(1,n))
    w=np.zeros((10,1));w[5]=.25;w[9]=-.065;w[7]=-.030
    return linear('strap_cue','dense',z,w)

def make():
    occ=foreground();a=pool('shape_grid','avgpool',occ,(2,2),(2,2));z=op('normalized_shape','norm',a)
    hp=-np.outer([.25,.5,.25],[.25,.5,.25]);hp[1,1]+=1
    ed=linear('contour','conv',a,hp[None,None],padding=1);ed=op('normalized_contour','norm',ed)
    val=pool('tone_grid','avgpool','image',(2,2),(2,2));val=op('normalized_tone','norm',val)
    blocks=4;per=480;nt=blocks*per
    shape=np.empty((10*nt,196),np.float32);edges=np.empty_like(shape);tone=np.empty_like(shape)
    for k in range(10):
        for j in range(nt):
            article=render(model(k),k); silhouette=project(article>0); intensity=project(article)
            i=k*nt+j
            shape[i]=.68*unit(silhouette);edges[i]=.15*unit(detail(silhouette));tone[i]=.17*unit(intensity)
    sh=linear('individual_shapes','dense',z,shape);de=linear('individual_contours','dense',ed,edges);gr=linear('individual_surfaces','dense',val,tone)
    allscore=op('joint_match','add',[sh,de,gr]);mat=op('template_groups','reshape',allscore,shape=[10,blocks,per]);mx=pool('best_in_group','maxpool',mat,(1,per),(1,per));best=pool('best_match','maxpool',mx,(blocks,1),(blocks,1));support=pool('match_support','avgpool',mx,(blocks,1),(blocks,1))
    best=linear('peak_vote','dense',best,.71*np.eye(10));sup=linear('support_vote','dense',support,.29*np.eye(10));cues=front_detail();straps=shoe_detail(occ);op('scores','add',[best,sup,cues,straps])

if __name__=='__main__':
    dest=Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=True);make()
    (dest/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n');np.savez(dest/'weights.npz',**weights)
