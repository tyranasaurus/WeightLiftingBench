"""Generate a clothing-recognition circuit. Its dictionaries are drawings, not photographs."""
import sys, json
from pathlib import Path
import numpy as np

F=np.float32
r=np.random.default_rng(987321)
u=r.uniform
pick=r.choice
layers=[]; weights={}
def layer(name,op,inp,**settings):
    if isinstance(inp,str): inp=[inp]
    layers.append(dict(name=name,op=op,inputs=inp,**settings)); return name
def conv(name,inp,w,b,p=0):
    w=np.array(w,F); k=w.shape[-1]; weights[name+'.W']=w; weights[name+'.b']=np.array(b,F)
    return layer(name,'conv',inp,out=len(w),kernel=[k,k],stride=1,padding=p)
def dense(name,inp,w,b=0):
    w=np.array(w,F); weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(np.array(b,F),(len(w),)).copy()
    return layer(name,'dense',inp,out=len(w))
def relu(name,inp):return layer(name,'relu',inp)
def pool(name,inp,size,mode='avgpool',stride=None):return layer(name,mode,inp,size=size,stride=size if stride is None else stride)
def reshape(name,inp,s):return layer(name,'reshape',inp,shape=s)

# A small planar vector drawing language. Coordinates in these drawings are garment
# design coordinates (e.g. fractions of the garment length), not image samples.
class Drawing:
    def __init__(self): self.parts=[]
    def poly(self,pts,ink=True):self.parts.append((ink,np.array(pts,float)))
    def bilateral(self,pts):self.poly(list(pts)+[(-x,y) for x,y in pts[::-1]])
    def band(self,x,y,half,ink=True):
        self.poly(list(zip(np.asarray(x)-half,y))+list(zip((np.asarray(x)+half)[::-1],y[::-1])),ink)
    def tube(self,pts,radius):
        a=np.array(pts); t=np.gradient(a,axis=0); t/=np.sqrt((t*t).sum(axis=1,keepdims=True))+1e-12
        n=np.stack([-t[:,1],t[:,0]],1)*radius
        self.poly(list(a+n)+list((a-n)[::-1]))
    def render(self,aspect=(1.,1.)):
        ps=np.concatenate([p for ink,p in self.parts if ink]); lo=ps.min(0); hi=ps.max(0)
        # Normalize the longer dimension, with a little variable clearance at the edge.
        scale=u(.925,.998)*56/max((hi-lo)*aspect)*np.asarray(aspect)
        skew=u(-.026,.026); tilt=u(-.018,.018)
        shift=np.array([28,28])+u(-.65,.65,2)
        row=np.arange(56)+.5; canvas=np.zeros((56,56),bool)
        for ink,p in self.parts:
            q=(p-(hi+lo)*.5)*scale
            x=q[:,0]+skew*q[:,1]+shift[0]; y=q[:,1]+tilt*q[:,0]+shift[1]
            # Scan conversion by horizontal ray crossings (vectorized over rows).
            ax=np.roll(x,1); ay=np.roll(y,1); dy=y-ay
            dy=np.where(dy==0,1e-16,dy)
            valid=(y[None,:]>row[:,None])!=(ay[None,:]>row[:,None]); cross=ax+(row[:,None]-ay)*(x-ax)/dy
            xx=np.sort(np.where(valid,cross,np.inf),axis=1)
            mask=np.zeros_like(canvas)
            for i in range(0,xx.shape[1]-1,2):
                if not np.isfinite(xx[:,i]).any(): break
                mask|=(row[None,:]>=xx[:,i,None])&(row[None,:]<xx[:,i+1,None])
            if ink:canvas|=mask
            else:canvas&=~mask
        return canvas.reshape(14,4,14,4).mean((1,3))

def garment(kind):
    d=Drawing(); dress=kind.startswith('dress'); long=kind in ('knit','longshirt','jacket','coat','dresslong','longtee')
    tank=kind in ('tank','dresstank'); cap=kind in ('cap','dresscap'); medium=kind in ('blouse','dresselbow'); short=not (long or tank or cap or medium)
    if dress:
        sh=u(.105,.217); hem=sh*u(.98,2.05); waist=sh*u(.64,.95); hip=sh*u(.88,1.38); wy=u(.29,.48); body_end=1.
    elif kind=='coat':
        sh=u(.19,.34);hem=sh*u(.96,1.32);waist=sh*u(.82,1.05);hip=sh*u(.94,1.15);wy=.46;body_end=1.
    elif kind=='jacket':
        sh=u(.28,.46);hem=sh*u(.86,1.20);waist=sh*u(.9,1.09);hip=(hem+waist)/2;wy=.5;body_end=u(.82,1.)
    elif kind=='knit':
        sh=u(.245,.40);hem=sh*u(.83,1.18);waist=sh*u(.85,1.11);hip=(hem+waist)/2;wy=.50;body_end=u(.82,1.)
    elif kind=='longshirt':
        sh=u(.225,.38);hem=sh*u(.87,1.22);waist=sh*u(.81,1.06);hip=(hem+waist)/2;wy=.49;body_end=u(.91,1.)
    else:
        sh=u(.21,.38) if not tank else u(.15,.27);hem=sh*u(.84,1.24);waist=sh*u(.80,1.03);hip=(waist+hem)/2;wy=.53;body_end=1.
    # Collar, shoulder seam, armscye, side seam, hem. A scooped collar dips in the middle.
    neck=sh*u(.31,.62) if not tank else sh*u(.53,.82)
    if kind in ('jacket','coat'):neckdepth=u(0,.037);shouldery=u(.05,.16)
    elif kind=='longshirt':neckdepth=u(0,.10);shouldery=u(.04,.13)
    elif tank:neckdepth=u(.085,.28);shouldery=u(0,.025)
    else:neckdepth=u(.017,.115);shouldery=u(.025,.105)
    seam=u(.19,.34) if not dress else u(.14,.23)
    bust=sh*(u(.85,1.06) if not tank else u(1.0,1.38));
    if tank and not dress:waist*=1.13;hem*=1.15
    if tank and dress:bust=sh*u(.92,1.15)
    # Boundary down the right side of the bodice, mirrored to obtain the left side.
    hemcurve=(u(.008,.06) if kind in ('longshirt','blouse') else u(0,.026))*body_end
    edge=[(0,neckdepth),(neck*.57,neckdepth*.79),(neck,0),(sh*.83,shouldery*.69),(sh,shouldery)]
    if tank:edge.extend([(sh*.91,seam*.61),(bust,seam)])
    else:edge.append((bust,seam))
    edge += [(waist,wy),(hip,wy+(body_end-wy)*.48),(hem,body_end-hemcurve),(hem*.68,body_end-hemcurve*.2),(0,body_end)]
    d.bilateral(edge)
    if not tank:
        if long:
            if dress:sy=u(.51,.77)
            elif kind=='coat':sy=u(.66,.94)
            elif kind=='longtee':sy=u(.65,1.00)
            else:sy=u(.83,1.00)
            cuff=sh*(u(.14,.34) if kind not in ('coat','jacket') else u(.20,.39))
            tipx=max(waist,hip)*u(.93,1.23)+cuff; elbow_x=(sh+tipx)*.5+u(.015,.075); ey=(shouldery+sy)*.52
            upper=sh*u(.24,.39); cuffslant=u(-.012,.027)
            sleeve=[(sh*.82,shouldery*.73),(sh,shouldery),(sh+upper*.58,shouldery+.066),(elbow_x+upper*.71,ey),(tipx+cuff*.5,sy-cuffslant),(tipx-cuff*.5,sy),(elbow_x-upper*.67,ey+.035),(bust*.92,seam)]
        else:
            if cap:sy=u(.16,.27);reach=u(.12,.42)*sh
            elif medium:sy=u(.36,.69);reach=u(.22,.63)*sh
            else:sy=u(.22,.405) if not dress else u(.14,.29);reach=u(.27,.76)*sh
            sleeve=[(sh*.82,shouldery*.73),(sh,shouldery),(sh+reach*.42,shouldery+(sy-shouldery)*.23),(sh+reach,sy-u(.005,.067)),(bust*u(.95,1.12),sy+u(.0,.045)),(bust*.88,seam)]
        for side in (-1,1):d.poly([(side*x,y) for x,y in sleeve])
    return d

def pants():
    d=Drawing(); h=u(.14,.275); waist=h*u(.78,1.02); crotch=u(.25,.49); skinny=u(.40,1.18); gap=u(0,.105); sway=u(-.028,.028)
    d.bilateral([(0,.012),(waist*.76,.005),(waist,0),(h,.17),(h*.99,crotch),(0,crotch+.03)])
    for side in (-1,1):
        c=h*.52; rad=h*.47; leg=rad*skinny; bottom=gap+leg; knee=(c+bottom)*.52
        d.poly([(side*x+sway*y,y) for x,y in [(0,crotch),(c*.18,.18),(h,.15),(h,crotch),(knee+rad*.86,.62),(bottom+leg,.99),(bottom-leg*.32,1),(bottom-leg,.99),(knee-rad*.83,.62),(0,crotch+.02)]])
    return d

def shoe(kind):
    d=Drawing(); thick=u(.045,.108); high=kind in ('boot','heel','highsandal'); sneaker=kind in ('sneaker','hightop')
    heelheight=(u(.13,.48) if kind in ('heel','highsandal') else (u(.015,.14) if kind=='boot' else u(0,.055)))
    start=u(.28,.48); end=u(.67,.85); heelx=u(.78,.94); heelfoot=u(.035,.18) if kind in ('highsandal','heel') else u(.10,.24)
    def sole(x):return 1.-.045*max(1-x/.21,0)**2-heelheight*np.clip((x-start)/(end-start),0,1)
    top=lambda x:sole(x)-thick
    xx=np.array([0,.035,.14,.27,start,(start+end)/2,end,.96,1]); yy=np.array([sole(x) for x in xx]);yt=yy-thick;yt[0]+=.5*thick
    d.poly(list(zip(xx,yy))+list(zip(xx[::-1],yt[::-1])))
    d.poly([(heelx-heelfoot*.55,top(heelx)),(.995,top(1)),(.977,sole(1)+heelheight-.01),(.98-heelfoot,sole(1)+heelheight), (heelx-heelfoot*.51,sole(heelx)+heelheight*.38)])
    if kind in ('boot','heel'):
        tall=u(.39,.86) if kind=='boot' else u(.30,.70); tall=max(tall,heelheight+.25)
        toe=u(.065,.15); ankle=u(.46,.68); back=u(.89,1.0); front=ankle+u(-.025,.11); notch=u(-.065,.09)
        d.poly([(0,top(0)+.028),(.038,top(.038)-toe*.63),(.13,top(.13)-toe),(.28,top(.28)-toe*1.18),(ankle-.13,top(ankle-.13)-toe-.05),(ankle-.038,1-tall*.68),(front,1-tall),(back,1-tall+notch),(back+.018,1-tall*.71),(.985,top(1)-.072),(.99,sole(1)-.015)]+[(x,sole(x)-.016) for x in xx[::-1]])
    elif sneaker:
        tongue=u(.22,.40) if kind=='sneaker' else u(.33,.51); back=u(.16,.32) if kind=='sneaker' else u(.24,.46); tonguex=u(.48,.72); toe=u(.057,.13)
        d.poly([(0,top(0)+.022),(.027,top(.027)-toe*.48),(.13,top(.13)-toe*.85),(.29,top(.29)-toe*1.18),(tonguex-.105,1-tongue+.076),(tonguex,1-tongue),(tonguex+.038,1-tongue+.012),(tonguex+.089,1-tongue+u(.04,.115)),(.82,1-min(tongue,back)+.064),(.94,1-back),(.975,1-back+.028),(.997,top(1)+.032)]+[(x,sole(x)-.01) for x in xx[::-1]])
    else:
        # Toe strap, sloping instep strap, and optional back or ankle strap.
        tall=(heelheight+u(.17,.46)) if high else u(.12,.39); a=u(.16,.31); b=u(.44,.64); c=u(.68,.83); w=u(.018,.060); strap=1-tall
        d.tube([(u(.01,.09),top(.04)+.02),(a*.65,top(a)-u(.03,.085)),(a,top(a)-u(.06,.13)),(a+u(.05,.18),top(a+.1)+.02)],w)
        if kind!='slide':
            d.tube([(b-u(.12,.26),top(b-.18)+.02),(b,top(b)-.10),(c,strap+.065),(.95,strap+u(-.025,.075))],w)
            if r.random()<.84:d.tube([(.99,top(1)+.026),(.982,(top(1)+strap)*.52),(.944,strap),(.85,strap-.005)],w*u(.7,1.6))
        else:
            d.tube([(b-.20,top(b-.20)+.015),(b-.08,strap),(b+.09,strap+.035),(b+.19,top(b+.19)+.014)],w*u(1,2))
    return d

def bag(kind):
    d=Drawing(); bw=1.0; h=u(.43,.96) if kind!='tote' else u(.85,1.38); top=u(.73,1.06); bottom=u(.68,.99); radius=u(.025,.15)
    if kind=='pack':top=u(.35,.82);h=u(.86,1.45)
    d.bilateral([(0,0),(top*.4,.008),(top*.5,radius*.48),(.5,h*.28),(.505,h*.62),(bottom*.5+radius*.3,h-radius),(bottom*.5,h-radius*.21),(bottom*.4,h),(0,h)])
    if kind not in ('clutch','pack') or r.random()<.33:
        hw=u(.14,.48);hh=u(.12,.60) if kind not in ('tote','strap') else u(.36,1.12); t=np.linspace(0,np.pi,19);x=hw*np.cos(t); lean=u(-.12,.12);y=.05-hh*np.sin(t)**u(.70,1.7);x+=lean*np.sin(t)
        d.tube(list(zip(x,y)),u(.012,.045) if kind!='strap' else u(.008,.020))
    return d

# Match silhouettes on a coarse grid and with multi-scale pooling. Matching both
# light fabric and black space helps reject shapes that only coincide in one part.
def silhouette_input():
    # Combine two foreground ramps. Soft thresholding handles charcoal fabric and
    # antialiased contour pixels; no statistic of any set of images is needed.
    z=conv('foreground_hinges','image',np.ones((3,1,1,1)),[0,.0-.045,-.145]);z=relu('foreground_pos',z)
    a=.70/.045; b=.30/.145
    mask=conv('fabric',z,np.array([a+b,-a,-b]).reshape(1,3,1,1),[0])
    coarse=pool('silhouette_grid',mask,[2,2]);return mask,coarse

styles=[
 ['tee']*6+['tank']*3+['cap']*2+['longtee'],
 ['pants'],
 ['knit'],
 ['dresscap','dressshort','dresstank','dresslong','dresselbow','dresstank'],
 ['jacket']*3+['coat']*2,
 ['highsandal']*3+['sandal']*4+['slide'],
 ['longshirt']*4+['blouse']*3+['tee']*2+['tank'],
 ['sneaker']*5+['hightop'],
 ['tote','bag','clutch','pack','strap','bag'],
 ['boot']*5+['heel']*2]

def smooth(a):
    # Adjoint of this symmetric averaging operator is itself (zero extension).
    p=np.pad(a,1);return sum(p[i:i+14,j:j+14]*(.5 if i==1 else .25)*(.5 if j==1 else .25) for i in range(3) for j in range(3))

def directional_envelopes(coarse):
    # For each scanline, its filled envelope consists of the points with fabric
    # BOTH before and after them. Do this for rows and columns in parallel and
    # leave the original occupancy grid as a third, independent measurement.
    m=14; ext=3*m-2; line=2*m-1; eye=np.eye(m*m).reshape(m,m,-1)
    gathers=np.zeros((2,m,ext,m*m)); gathers[0,:,m-1:2*m-1]=eye; gathers[1,:,m-1:2*m-1]=eye.transpose(1,0,2)
    z=dense('scanline_padding',coarse,gathers.reshape(-1,m*m));z=reshape('scanlines',z,[1,2*m,ext]);z=pool('scanline_maxima',z,[1,m],'maxpool',[1,1])
    # For a point j, prefix = window j, suffix = window (m-1+j).
    ids=np.arange(2*m*line).reshape(2,m,line); left=ids[:,:,:m].copy();right=ids[:,:,m-1:].copy();left[1]=left[1].T.copy();right[1]=right[1].T.copy();left=left.reshape(-1);right=right.reshape(-1);ix=np.arange(len(left))
    p=np.zeros((2,len(left),2*m*line));p[0,ix,left]=1;p[1,ix,left]=1;p[1,ix,right]-=1
    z=dense('envelope_sides',z,p.reshape(-1,2*m*line));z=relu('scanline_positive',z)
    w=np.zeros((3*m*m,2*len(left)));w[m*m+ix,ix]=1;w[m*m+ix,len(left)+ix]=-1
    z=dense('extent_grids',z,w);v=np.zeros((3*m*m,m*m));v[:m*m]=np.eye(m*m);a=dense('unfilled_grid',coarse,v)
    return layer('shape_measurements','add',[z,a])

def shape_filters(coarse):
    coarse=directional_envelopes(coarse)
    W=[];B=[]; n=1024
    for k,family in enumerate(styles):
        for i in range(n):
            kind=family[i%len(family)]
            d=(pants() if k==1 else shoe(kind) if k in (5,7,9) else bag(kind) if k==8 else garment(kind))
            # Garment proportions refer to body length vs. shoulder span, trouser
            # length vs. hip breadth, etc. Include slender and oversized cuts.
            stretch = (u(.55,.98) if k==1 else u(.77,1.14) if k==3 else
                       u(.86,1.29) if k in (0,6) else u(.89,1.36) if k in (2,4) else 1.)
            rise=u(.98,1.67) if k==7 else u(.85,1.42) if k==5 else u(.90,1.18) if k==9 else 1.
            t=d.render((stretch,rise))
            # Fractional occupancy prototypes, treated as drawings with uncertain edges.
            # Negative squared discrepancy, omitting the class-independent input norm.
            # Shoulder/hem distinctions and sole/ankle distinctions are positional.
            soft=smooth(t)
            # Scanline envelopes are less affected by an open placket, printed
            # stripes, a highlight or lettering than direct fabric occupancy.
            w=[.53*(.67*t+.33*smooth(soft))]; b=-.5*.53*(.67*(t*t).sum()+.33*(soft*soft).sum())
            for ax,amount in [(1,.23),(0,.24)]:
                ed=np.minimum(np.maximum.accumulate(t,axis=ax),np.flip(np.maximum.accumulate(np.flip(t,ax),axis=ax),ax))
                ed_s=smooth(ed);w.append(amount*(.75*ed+.25*smooth(ed_s)));b-=.5*amount*(.75*(ed*ed).sum()+.25*(ed_s*ed_s).sum())
            w=np.stack(w)
            # Scores expressed in 'mismatching coarse pixel' units.
            W.append(2*w);B.append(2*b)
    z=dense('paper_patterns',coarse,np.array(W).reshape(len(W),-1),B)
    z=reshape('pattern_families',z,[10,1,n]);z=pool('best_cut',z,[1,n],'maxpool');return reshape('outline_vote',z,[10])

# Texture, vertical opening, neckline, hem and shoe-strap cues. Use a common
# coordinate system centred on the object. All intervals are fractions of the frame.
# The only across-position reductions here take place at classification time.
class Features:
    def __init__(self):self.rows={}; self.names=[];self.grid=(np.arange(28)+.5)/28
    def add(self,src,channel,x1,x2,y1,y2):
        g=self.grid;win=((g>=y1)&(g<y2))[:,None]*((g>=x1)&(g<x2))[None,:];return self.custom(src,channel,win)
    def custom(self,src,channel,win):
        win=np.asarray(win,float); win=win/win.sum();key=(src,channel,win.tobytes())
        if key not in self.names:
            self.names.append(key); self.rows.setdefault(src,[]).append((len(self.names)-1,channel,win))
        return self.names.index(key)
    def finish(self,sources):
        branches=[];n=len(self.names)
        for src,rows in self.rows.items():
            h,x=rows[0][2].shape; w=np.zeros((n,sources[src],h,x),F)
            for j,c,win in rows:w[j,c]=win
            branches.append(dense('read_'+src,src,w.reshape(n,-1)))
        return layer('observations','add',branches)

# A tiny symbolic compiler for continuous piecewise-linear evidence rules.
class Expr:
    def __init__(self,terms=None,b=0):self.terms=terms or {};self.b=b
    def __add__(self,other):
        if not isinstance(other,Expr):other=Expr(b=other)
        t=self.terms.copy()
        for k,v in other.terms.items():t[k]=t.get(k,0)+v
        return Expr(t,self.b+other.b)
    __radd__=__add__
    def __mul__(self,a):return Expr({k:a*v for k,v in self.terms.items()},self.b*a)
    __rmul__=__mul__
    def __sub__(self,other):return self+(-1*other)
    def __rsub__(self,other):return other+self*(-1)

def detail_votes(mask):
    # Use both raw intensity and contrast relative to the image's own contrast.
    normal=layer('exposure','norm','image'); raw=conv('linear_light','image',[[[[1.1]]]], [0]);rel=conv('relative_light',normal,[[[[.24]]]], [0]); light=layer('light','add',[raw,rel]);
    # Light need not be positive: all photometric features below are differences.
    k=5; fs=[]; bs=[];ch={}
    def pair(name,kernel):
        a=np.array(kernel);ch[name]=len(fs); fs.extend([a,-a]);bs.extend([0.,0.]);
    dx=np.zeros((k,k));dx[1:4,3]=[.25,.50,.25]; dx[1:4,1]=[-.25,-.50,-.25];pair('dx',dx);pair('dy',dx.T)
    ridge=np.zeros((k,k)); ridge[:,2]=.2;ridge[:,0]=ridge[:,4]=-.1;pair('ridge',ridge)
    detail=np.zeros((k,k));detail[2,2]=1;detail[1:4,1:4]-=1/9;pair('detail',detail)
    g=conv('local_contrasts',light,np.array(fs)[:,None],bs,2);g=relu('rectified_contrasts',g)
    shape,rr='fabric',g
    feet=Features();sources={shape:1,g:len(fs),light:1,'image':1};
    # A soft morphological closing fills a dark motif within a panel but largely
    # preserves its *outer* contour. Compare that envelope to undilated fabric.
    eye=np.ones((1,1,1,1)); pad=conv('border_for_envelope',shape,eye,[0],2)
    dil=pool('nearby_fabric',pad,[5,5],'maxpool',[1,1]); inv=conv('inverted_envelope',dil,-eye,[0],2)
    er=pool('trim_envelope',inv,[5,5],'maxpool',[1,1]);env=conv('fabric_envelope',er,-eye,[0]);sources[env]=1
    # Compare upper/lower boundary directions along scanlines. Ordered endpoint
    # masks distinguish a neck/handle opening from a dark print crossing the body.
    right=np.zeros((1,1,3,3));right[0,0,1,0]=-1;right[0,0,1,2]=1
    ed=conv('silhouette_flanks',env,np.concatenate((right,-right)),[0,0],1);ed=relu('oriented_flanks',ed);sources[ed]=2
    def box(src,x1,x2,y1,y2):return Expr({feet.add(src,0,x1,x2,y1,y2):1})
    def region(src,c,x1,x2,y1,y2):return Expr({feet.add(src,c,x1,x2,y1,y2):1})
    def rect(x1,x2,y1,y2):return region(shape,0,x1,x2,y1,y2)
    def tex(name,x1,x2,y1,y2):return sum(region(g,ch[name]+s,x1,x2,y1,y2) for s in (0,1))
    def lum(x1,x2,y1,y2):return region(light,0,x1,x2,y1,y2)
    # Hinge layer then votes. clip translates physical differences into bounded evidence.
    hinges=[]; second=[]; coef=[]
    def second_relu(e):
        j=len(second);second.append(e);return Expr({-1-j:1})
    def activation_cut(e,start,end):
        a=(e-start)*(1/(end-start));return second_relu(a)-second_relu(a-1)
    def clip(e,start,end):
        i=len(hinges);e=(e-start)*(1/(end-start));hinges.extend([e,e-1]);return Expr({i:1,i+1:-1})
    def ge(e,a,b):return clip(e,a,b)
    score=[Expr() for i in range(10)]
    # Distinct physical proportions. These observations are translation tolerant:
    # a max along a scanline measures whether a row/column contains *any* fabric.
    # For very thin legs it is the split topology, not their total mass, that matters.
    def projection(name,axis):
        z=pool(name,shape,[28,1] if axis==0 else [1,28],'maxpool');
        sources[z]=1
        return z
    wide=projection('columns_used',0); tall=projection('rows_used',1)
    def portion(src,lo,hi):
        v=feet.grid; a=((v>=lo)&(v<hi)); a=a.reshape(1,28) if src==wide else a.reshape(28,1)
        return Expr({feet.custom(src,0,a):1})
    width=portion(wide,0,1);height=portion(tall,0,1)
    def available(y1,y2):
        v=feet.grid; return ((v>=y1)&(v<y2)).sum()
    # Thin vertical legs, with a continuous belt/seat above and space below.
    leftleg=rect(.28,.46,.58,.91);rightleg=rect(.54,.72,.58,.91)
    midleg=rect(.46,.54,.58,.91);split=(leftleg+rightleg)*.5-midleg
    seat=rect(.42,.58,.08,.24)
    splitgap=clip(split,.04,.57); narrow=clip(.59-width,0,.27)
    score[1]+=16*narrow+13*splitgap+7*clip(seat-midleg,.18,.80)
    score[1]-=10*clip(width,.59,.82)+7*clip(1-height,.13,.44)
    for c in (0,2,4,6):score[c]-=7*narrow
    # A long skirt has a smaller bust-to-length ratio than a separate top.
    slim=clip(.80-width,0,.36); pelvis=rect(.43,.57,.65,.90)
    score[3]+=8.5*slim
    score[3]+=4*clip(2*pelvis-width,1.24,1.61)
    # Across a long-sleeved garment rows, a pair of arms is separate from
    # the body. Facing flank edges *inside* the outside contour identify the
    # underarm channels, which are absent in a single uninterrupted skirt.
    def inner_edges(y0,y1):
        return .5*(region(ed,1,.05,.42,y0,y1)+region(ed,0,.58,.95,y0,y1))
    armgap=clip(inner_edges(.51,.80),.025,.15)
    score[3]-=5.2*armgap;score[1]-=4.5*armgap
    # At a skirt hem the forks of two trouser legs cannot both be present.
    score[1]-=8*clip(pelvis,.64,.97)
    # Two sleeves stop abruptly either side of a tee. A narrow, sleeveless top
    # instead has a low scoop neck and cut-in armholes. Both cues distinguish
    # upper-body wear from a more gradual skirt flare.
    topwidth=box(env,0,1,.18,.31); lowwidth=box(env,0,1,.51,.78)
    steptaper=clip(topwidth-lowwidth,.095,.30);score[0]+=4.1*steptaper;score[3]-=1.8*steptaper
    underarm=(rect(.14,.30,.30,.42)+rect(.70,.86,.30,.42))*.5
    shoulderpatch=(rect(.14,.30,.15,.28)+rect(.70,.86,.15,.28))*.5
    tee_sleeve=clip(shoulderpatch-underarm,.25,.77);score[0]+=2.6*tee_sleeve
    # Full-width rectangular bags can resemble knits without a handle. Garment
    # shoulders slope outwards below a high point at the central neck. A handle
    # instead encloses black sky above a wide, usually solid body.
    uppercentre=rect(.40,.60,.12,.35); inner=rect(.34,.66,.46,.72)
    shoulders=(rect(.22,.39,.16,.32)+rect(.61,.78,.16,.32))*.5
    topcorners=(rect(.045,.23,.14,.28)+rect(.77,.955,.14,.28))*.5
    opening=clip(inner-uppercentre,.35,.89); score[8]+=4*opening
    fullheight=clip(height,.84,.98)
    # Totes and purses have a disconnected-looking pair of straps, with sky
    # between them. Clutches have no handle but boxy sides and wide flat tops.
    handle=clip(shoulders-uppercentre,.14,.64); score[8]+=6*handle
    bagtop=clip(topcorners,.26,.76);score[8]+=6*bagtop
    squat=clip(width-height,.10,.49);score[8]+=6*squat
    # Rather than rewarding every dark neck as a bag handle, check how abruptly
    # the handle changes to a fabric panel below it.
    abrupt=clip(lowwidth-topwidth,.28,.71);score[8]+=5*abrupt
    score[8]-=3*armgap
    # A T-shaped garment and a bag can both occupy a wide rectangle. Distinguish
    # them by the voids under the sleeves and the continuous lower bag panel.
    # Do not rely on morphological closing here: it could seal an underarm gap.
    rawdx=conv('raw_flank_signs',shape,np.concatenate((right,-right)),[0,0],1);rawdx=relu('inner_fabric_edges',rawdx);sources[rawdx]=2
    rawedges=.5*(region(rawdx,1,.04,.42,.48,.84)+region(rawdx,0,.58,.96,.48,.84))
    clefts=clip(rawedges,.047,.20)
    score[8]-=6.5*clefts; score[2]+=1.3*clefts;score[4]+=1.3*clefts;score[6]+=.75*clefts
    # Footwear is asymmetric: tongue and shaft rise on the rear half, not
    # in the flat handle arch or symmetrical shoulders of a tote.
    rearupper=box(env,.59,.91,.28,.52); frontupper=box(env,.09,.41,.28,.52)
    lean=clip(rearupper-frontupper,.12,.68);score[8]-=10*lean
    score[7]+=1.8*lean;score[9]+=1.5*lean
    even_top=clip(uppercentre-shoulders,.13,.70);score[8]-=4.5*even_top
    # A waist/seat spreads into two hems rather than into a single skirt hem.
    # Inspect the crotch fork over several depths, including splits which start
    # near the feet. A long narrow object without sleeves is trouser-like even
    # if only a fine vertical shadow separates the two legs.
    fork=[]
    for y0,y1 in [(.42,.61),(.61,.79),(.79,.95)]:
        a=rect(.34,.465,y0,y1);b=rect(.535,.66,y0,y1);slot=rect(.465,.535,y0,y1)
        fork.append(clip(.5*(a+b)-slot,.065,.56))
    pantlike=activation_cut(narrow+sum(fork)*.5,.72,1.53)
    score[1]+=6.5*pantlike;score[3]-=5*pantlike
    # Identify the too-long sleeves of an outer layer. In a long coat there is
    # cloth outside a narrow central skirt even though the full shape is slim.
    longarm=activation_cut(clefts+slim,.85,1.45);score[3]-=4.4*longarm;score[4]+=2.2*longarm
    # Sneaker: heel counter is low with a dip in front of it and the tongue is
    # further forward; a boot shaft rises solidly at the heel. A sandal has
    # straps rather than a continuous upper and a near-empty toe box.
    tongue=rect(.47,.64,.35,.51);ankle=rect(.71,.91,.30,.49)
    notch=clip(tongue-ankle,.085,.62); score[7]+=7*notch;score[9]-=3.5*notch
    fullness=rect(.22,.84,.50,.65);toe=rect(.08,.36,.52,.63)
    # A low closed upper (as opposed to an equally low sandal) encloses fabric
    # right across the vamp. Weighted conjunction, made before clipping.
    flatclosed=clip(.75*fullness-height,.05,.42);score[7]+=3*flatclosed
    lowboot=clip(ankle-tongue,.065,.55);score[9]+=7.0*lowboot;score[7]-=3.2*lowboot
    anklecap=rect(.64,.88,.21,.39);score[9]+=5*clip(anklecap,.25,.90)
    # Internal cut-outs tend to be much deeper than the grooves in a sneaker.
    # The morphological envelope bridges straps and toe holes. The intensity
    # boundary cue is independent: it also sees contrasting straps on a footbed.
    holes=box(env,.09,.93,.38,.71)-rect(.09,.93,.38,.71)
    strapvoid=clip(holes,.04,.25);score[5]+=9.0*strapvoid;score[7]-=3.5*strapvoid;score[9]-=3.0*strapvoid
    # A sandal can have an open box even with a thick platform sole.
    frontroof=rect(.08,.29,.44,.57);heelroof=rect(.59,.83,.40,.58)
    score[5]+=3*clip(.72-frontroof-.7*heelroof,0,.65)
    score[7]+=2.0*clip(fullness,.65,.98);score[5]+=6*clip(.66-fullness,.0,.55)
    score[5]+=4*clip(.45-toe,0,.42)
    # A coat has a full-height central opening, unlike a knit. Evaluate at
    # several horizontal offsets so a slightly off-centre zip counts too.
    center_ridge=tex('ridge',.44,.56,.27,.82); side_ridge=(tex('ridge',.30,.43,.27,.82)+tex('ridge',.57,.70,.27,.82))*.5
    center_dx=tex('dx',.43,.57,.28,.82); side_dx=(tex('dx',.29,.43,.28,.82)+tex('dx',.57,.71,.28,.82))*.5
    zip_strength=ge(.6*(center_dx-side_dx)+.4*(center_ridge-side_ridge),.018,.145)
    zipper=[]
    for j in (12,13,14,15):
        middle=lum(j/28,(j+1)/28,.30,.86)
        flanks=(lum((j-2)/28,(j-1)/28,.30,.86)+lum((j+2)/28,(j+3)/28,.30,.86))*.5
        zipper.append(clip(flanks-middle,.046,.20)+clip(middle-flanks,.067,.25))
    vertical=sum(zipper)*.40+zip_strength
    score[4]+=3.7*vertical;score[2]-=2.9*vertical;score[6]+=1.35*vertical
    # Lapels and plackets are long, approximately straight contrast boundaries.
    # Average *before* taking magnitude: random knit texture tends to cancel out.
    # The paired oblique lines approach the buttoning point from both shoulders.
    gv=feet.grid; X,Y=np.meshgrid(gv,gv)
    def strip(x0,x1,y0,y1,wide=1.):
        line=x0+(x1-x0)*(Y-y0)/(y1-y0);w=np.maximum(0,1-abs(X-line)*28/wide)*((Y>=y0)&(Y<=y1))
        return Expr({feet.custom(light,0,w):1})
    def signed_line(x0,x1,y0,y1,s):
        c=strip(x0,x1,y0,y1);l=strip(x0-s,x1-s,y0,y1);rr=strip(x0+s,x1+s,y0,y1)
        return c-.5*(l+rr)
    def absclip(x,a,b):return clip(x,a,b)+clip(-1*x,a,b)
    diag=Expr()
    for outer,inner,end in [(.32,.48,.39),(.37,.49,.32),(.33,.46,.49),(.285,.46,.35)]:
        dl=signed_line(outer,inner,.10,end,.075)
        dr=signed_line(1-outer,1-inner,.10,end,.075)
        diag+=absclip((dl+dr)*.5,.046,.23)
    score[4]+=1.45*diag;score[6]+=.90*diag;score[2]-=.95*diag
    # Crisp central asymmetry (an overlapping panel or a shadow to one side of a
    # button placket) often lasts even when both panels have the same colour.
    overlap=absclip(lum(.38,.47,.34,.80)-lum(.53,.62,.34,.80),.10,.35)
    score[4]+=2.9*overlap;score[6]+=1.0*overlap;score[2]-=1.5*overlap
    # Persistent centre-front structure. Break the front into disjoint bands:
    # contrast must follow a common near-vertical track *in multiple bands*.
    # Thus a chest graphic or a waistband alone doesn't become a fake zipper.
    plackets=Expr()
    for j in (12.5,13.5,14.5,15.5):
        trace=Expr()
        for a,b in [(.25,.42),(.42,.60),(.60,.79),(.79,.90)]:
            st=signed_line(j/28,j/28,a,b,2/28)
            trace+=absclip(st,.032,.19)
        plackets+=activation_cut(trace,1.63,3.05)
    # Lower-contrast uniform surfaces are consistent with both blouses and
    # knitwear. A placket alone does not imply coat: look also for structured
    # collars/large lapels or other front-panel detail as independent evidence.
    score[4]+=2.8*plackets;score[6]+=1.6*plackets;score[2]-=1.65*plackets
    # Knit cuffs and waistbands. Average an across-body transition at the bottom
    # before rectifying it; this tolerates texture and ignores vertical seams.
    hems=Expr()
    for y in (.72,.79,.86):
        heml=lum(.31,.69,y,y+.055)-lum(.31,.69,y-.083,y-.027)
        line=absclip(heml,.090,.29);hems+=line
    score[2]+=2.0*hems;score[4]-=.6*hems;score[6]-=.8*hems
    # A lapel frames a V of contrasting fabric, while a crew-neck knit normally
    # has little contrast below the neckline. Shirt collars share the V cue.
    collartex=tex('dx',.32,.68,.08,.29)+tex('dy',.32,.68,.08,.29)
    chesttex=tex('dx',.28,.72,.34,.67)+tex('dy',.28,.72,.34,.67)
    lapels=clip(collartex-chesttex,.04,.32)
    score[4]+=2.4*lapels;score[6]+=1.9*lapels;score[2]-=1.8*lapels
    # Shirt patterns often extend down to the hem, in both warp and weft;
    # a print on a tee or a pullover is often confined to the chest. Horizontal
    # stripes alone are not evidence for a button-up.
    tx=tex('dx',.31,.69,.32,.86); ty=tex('dy',.31,.69,.32,.86)
    txlo=tex('dx',.31,.69,.61,.86); tylo=tex('dy',.31,.69,.61,.86)
    pattern=clip(txlo,.078,.25)+clip(tylo,.080,.26)+clip(tx+ty,.22,.53)
    score[6]+=2.6*pattern;score[2]-=.6*pattern;score[4]-=.38*pattern
    # Plaid and woven prints repeat in both directions. Use continuity in
    # several quadrants rather than assuming that one centre-front patch is
    # representative. Separate recurrent weave from large jacket panels.
    repeat=Expr()
    for x0,x1,y0,y1 in [(.29,.47,.29,.47),(.53,.71,.29,.47),(.29,.47,.56,.78),(.53,.71,.56,.78)]:
        qx=tex('dx',x0,x1,y0,y1);qy=tex('dy',x0,x1,y0,y1);fine=tex('detail',x0,x1,y0,y1)
        repeat+=clip(qx+qy+fine,.24,.54)
    printcloth=activation_cut(repeat,1.30,3.10); score[6]+=4.7*printcloth;score[4]-=1.5*printcloth
    # A blouse's curved shirt tail is bright in the centre and absent near
    # either side of its hem, unlike a square cut tee or rib-knit waistband.
    shirttail=rect(.40,.60,.91,.985)-.5*(rect(.23,.35,.91,.985)+rect(.65,.77,.91,.985))
    curved=clip(shirttail,.32,.85);score[6]+=2.1*curved;score[2]-=.7*curved
    # An actual collar has two light/dark transitions near the shoulders, with
    # the contrasting patches mirrored to either side of the inside neck.
    collar=activation_cut(diag*.33+lapels,.75,1.42); score[6]+=2.4*collar;score[4]+=1.1*collar
    logo=clip(chesttex-(txlo+tylo),.066,.29)
    score[0]+=1.8*logo;score[2]+=2.4*logo;score[4]-=1.5*logo;score[6]-=1.5*logo
    stripes=clip(ty-tx,.045,.22);score[2]+=2.0*stripes;score[0]+=1.2*stripes
    # A jacket can close with a broad front panel instead of a thin zipper.
    front=lum(.42,.58,.34,.78); flanks=(lum(.275,.415,.34,.78)+lum(.585,.725,.34,.78))*.5
    panel=clip(front-flanks,.09,.36)+clip(flanks-front,.09,.36)
    score[4]+=2.2*panel;score[2]-=1.8*panel;score[6]+=.4*panel
    # Shoulder-to-hem proportions: the sleeves of a pullover often reach beyond
    # the waistband. A tunic/blouse tends to end in a curved shirt tail.
    centerhem=rect(.34,.66,.88,.98); cuff=(rect(.03,.20,.88,.98)+rect(.80,.97,.88,.98))*.5
    cuffs=clip(cuff-centerhem,-.10,.48);score[2]+=3.4*cuffs;score[4]-=1.65*cuffs;score[6]-=2.4*cuffs
    # When a top's skirt fans out below a narrow waist it is probably a dress.
    lower=rect(0,1,.78,.91);waist=rect(0,1,.39,.51)
    flare=clip(lower-waist,.06,.26);score[3]+=6*flare
    # Sole plus open air interspersed between straps: use true internal holes
    # (foreground on opposed sides), not just darkness or overall aspect ratio.
    mk=np.zeros((4,1,5,5));mk[:,0,2,2]=-1.
    for i,(dy,dx) in enumerate([(2,0),(0,2),(2,2),(2,-2)]):mk[i,0,2+dy,2+dx]=mk[i,0,2-dy,2-dx]=.5
    z=conv('gap_boundaries',shape,mk,[-.51]*4,2);hh=relu('open_spaces',z);sources[hh]=4
    void=sum(region(hh,i,.06,.95,.37,.70) for i in range(4)); cutouts=clip(void,.013,.090)
    score[5]+=4.8*cutouts;score[7]-=2.1*cutouts;score[9]-=1.4*cutouts
    # Sneaker outsole usually has a distinct light band under a solid upper.
    # A slide that has only a strap often has an almost unoccupied heel.
    rear=rect(.77,.93,.48,.58);fore=rect(.10,.28,.50,.58)
    openrear=clip(fore-rear,.3,.80);score[5]+=3.5*openrear;score[7]-=2.*openrear
    # Locate the tongue relative to the heel top using column blocks, rather
    # than only a fixed-height ankle test. Compare the *highest occupied row*
    # in forward and rear quarters: sneakers often dip behind the laces, boots
    # rise to a nearly level shaft, and sandals have isolated thin strap tips.
    # A direct 1-D scanline integration converts max signal above y to height.
    m=14; ids=np.arange(m*m).reshape(m,m); offsets=[(2,5),(5,7),(7,9),(9,11),(11,13)]
    w=[]
    for lo,hi in offsets:
        for y in range(3,10):
            # max-region computed below; layout groups entries in scan order.
            sel=ids[:y,lo:hi].ravel();w.append(sel)
    maxlen=max(len(t) for t in w);mat=np.zeros((len(w),maxlen,m*m),F)
    for i,sel in enumerate(w):mat[i,np.arange(len(sel)),sel]=1
    a=dense('shoe_height_samples','silhouette_grid',mat.reshape(-1,m*m));a=reshape('shoe_scan_sets',a,[1,len(w),maxlen]);a=pool('shoe_top_lines',a,[1,maxlen],'maxpool')
    sources[a]=1
    def altitude(i):
        win=np.zeros((len(w),1));win[7*i:7*i+7]=1;return Expr({feet.custom(a,0,win):1})
    nose,instep,laces,throat,heel=[altitude(i) for i in range(5)]
    # Rising from nose to tongue then falling towards heel suggests sneaker.
    spike=clip(laces-throat,.015,.19)+clip(laces-heel,.015,.22)+clip(instep-heel,.005,.21)
    stair=clip(heel-instep,.13,.47)
    score[7]+=2.8*spike;score[9]-=2.8*spike;score[9]+=2.5*stair;score[7]-=1.5*stair
    # A boot shaft is near level for its entire rear third, not a skinny heel
    # counter. A sneaker's closed upper reaches the sole under the tongue.
    solid=clip(fullness,.64,.93); sneaker_vamp=activation_cut(solid+notch,.95,1.54)
    score[7]+=3.0*sneaker_vamp;score[5]-=2.*sneaker_vamp
    footwear=clip(heel-nose,.10,.47);score[8]-=5*footwear
    # Distinguish air from the shadow on a dark closed shoe. Each proposed
    # hole needs background in a patch bracketed by fabric on *two* sides.
    # All offsets describe generic strap spacing; they are not photo crops.
    enclosed=Expr()
    for cx in (.25,.39,.53,.67,.81):
        for cy in (.45,.54,.63):
            empty=rect(cx-.036,cx+.036,cy-.037,cy+.037)
            lr=.5*(rect(cx-.16,cx-.09,cy-.055,cy+.055)+rect(cx+.09,cx+.16,cy-.055,cy+.055))
            tb=.5*(rect(cx-.06,cx+.06,cy-.14,cy-.065)+rect(cx-.06,cx+.06,cy+.065,cy+.14))
            enclosed+=clip(.52*lr+.48*tb-empty,.36,.71)
    bigopen=activation_cut(enclosed,.68,2.55)
    upper_air=clip(.80*heel+ .15 - fullness,.08,.60)
    score[5]+=7*bigopen+4*upper_air;score[7]-=3.5*bigopen;score[9]-=2.5*bigopen
    # A sneaker has an unbroken, roughly horizontal sole. The instep arch of
    # a high-heeled sandal instead leaves a space in front of its narrow heel.
    # Test the void just ABOVE the bottom contour, at a series of possible
    # sole heights, requiring a separate heel and a forefoot at that level.
    heelvoid=Expr()
    for cy in (.62,.71,.80,.89):
        l=rect(.12,.38,cy-.04,cy+.04);c=rect(.56,.75,cy-.04,cy+.04);rr=rect(.82,.94,cy-.04,cy+.04)
        heelvoid+=clip(.5*(l+rr)-c,.32,.71)
    openheel=activation_cut(heelvoid,.39,1.38);score[5]+=3.8*openheel;score[9]+=1.7*openheel;score[7]-=3.8*openheel
    # Actual enclosed background, using the two scanline envelopes. A shoe
    # with a closed upper has nearly convex vertical slices: from the outsole
    # to the topline there is material all the way. A strap bridging the sole
    # is different even if both have exactly the same outside profile.
    m=14;eye2=np.eye(m*m);wm=np.zeros((2*m*m,3*m*m));wm[:m*m,:m*m]=-eye2;wm[:m*m,m*m:2*m*m]=eye2;wm[m*m:,:m*m]=-eye2;wm[m*m:,2*m*m:]=eye2
    gaps=dense('enclosed_space_maps','shape_measurements',wm);gaps=reshape('enclosed_spaces',gaps,[2,m,m]);sources[gaps]=2
    grid=(np.arange(m)+.5)/m
    def gap(axis,xlo,xhi,ylo,yhi):
        sel=((grid>=ylo)&(grid<yhi))[:,None]*((grid>=xlo)&(grid<xhi))[None,:]
        return Expr({feet.custom(gaps,axis,sel):1})
    vent=gap(1,.08,.93,.30,.74); middlevent=gap(1,.28,.86,.36,.66)
    perforated=clip(vent+.5*middlevent,.048,.23)
    score[5]+=12*perforated;score[7]-=4.5*perforated;score[9]-=4.5*perforated
    finger=gap(0,.39,.61,.48,.91);score[1]+=5*clip(finger,.16,.63);score[3]-=3.6*clip(finger,.22,.68)
    # The enclosed parts of an underarm slot differ from a bag handle in
    # *where* they appear: near the flanks below the chest, not high and central.
    arms=.5*(gap(0,.10,.36,.53,.83)+gap(0,.64,.90,.53,.83)); armpits=clip(arms,.11,.36)
    headroom=gap(0,.33,.67,.14,.38); handlehole=clip(headroom,.18,.63)
    score[8]+=5.0*handlehole-6.0*armpits;score[3]-=3.1*armpits
    # A horizontal contrast band across the natural waist suggests a belt or a
    # skirt/bodice join. In combination with a lengthwise cut it supports dress.
    belt=Expr()
    for cy in (.32,.41,.50):
        l=lum(.35,.65,cy-.033,cy+.033); a=lum(.35,.65,cy-.10,cy-.045);b=lum(.35,.65,cy+.045,cy+.10)
        belt+=absclip(l-.5*(a+b),.12,.37)
    dressbelt=activation_cut(belt+slim,.92,1.65);score[3]+=3*dressbelt
    # Separate COARSE construction evidence (height, split legs, broad body,
    # handle) from the details relevant only to comparisons *within* a family.
    # In particular a printed dress shouldn't lose to a shirt just because it
    # has the right textile. Maxima of finer votes will be subtracted below.
    coarse_score=[Expr() for i in range(10)]
    coarse_score[1]=score[1];coarse_score[3]=score[3];coarse_score[8]=score[8]
    garment_prior=1.5*clefts-6*narrow-2.5*abrupt
    coarse_score[0]=garment_prior+3.2*steptaper+2*tee_sleeve
    for j in (2,4,6):coarse_score[j]=garment_prior+1.3*armgap
    # Directional top and sole profiles distinguish shoes from bags. Don't let
    # the strongest of several style-specific tests decide if the object is a shoe.
    footprior=3.5*lean+2.8*footwear+1.8*clip(.68-height,.0,.35)-2.5*opening
    for j in (5,7,9):coarse_score[j]=footprior
    # A modest bonus for a real collar opening in the coat/knit comparison:
    # a tall raised collar occupies the centre above sloping, low shoulder seams.
    upright=clip(rect(.43,.57,.037,.14)-.5*(rect(.27,.40,.037,.14)+rect(.60,.73,.037,.14)),.27,.77)
    score[4]+=3.1*upright;score[2]-=1.4*upright
    # Constant priors are in drawing discrepancy units (one grid cell per unit).
    # 'Shirt' is an overlapping category: use textile/collar evidence rather
    # than letting its generic patterns claim every plain tee or sweater.
    score[0]+=.8;score[2]+=1.6;score[4]-=1.25;score[6]-=3.7;score[5]-=1.0
    obs=feet.finish(sources);n=len(feet.names)
    def make(es,n):
        w=np.zeros((len(es),n));b=[]
        for i,e in enumerate(es):
            for j,v in e.terms.items():w[i,j]=v
            b.append(e.b)
        return w,b
    z=dense('evidence_ramps',obs,*make(hinges,n));z=relu('evidence_positive',z)
    nh=len(hinges); z=dense('local_combinations',z,*make([Expr({i:1}) for i in range(nh)]+second,nh));z=relu('construction_positive',z)
    def remap(seq):return [Expr({(i if i>=0 else nh-i-1):v for i,v in e.terms.items()},e.b) for e in seq]
    return (dense('construction_vote',z,*make(remap(score),nh+len(second))),
            dense('family_evidence',z,*make(remap(coarse_score),nh+len(second))))

def main(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    mask,coarse=silhouette_input();outline=shape_filters(coarse);detail,gate=detail_votes(mask)
    fine=layer('style_hypotheses','add',[outline,detail]);broad=layer('shape_hypotheses','add',[outline,gate]);
    families=[[0,2,4,6],[1],[3],[5,7,9],[8]]
    order=np.array([fam+[fam[-1]]*(4-len(fam)) for fam in families]).ravel();broadcast=np.zeros((10,len(families)))
    for i,f in enumerate(families):broadcast[f,i]=1
    def best(inp,name):
        z=dense(name+'_collect',inp,np.eye(10)[order]);z=reshape(name+'_sets',z,[1,len(families),4]);
        z=pool(name+'_max',z,[1,4],'maxpool');return dense(name+'_broadcast',z,broadcast)
    f=best(fine,'style');g=best(broad,'family')
    # Within-family rankings retain all detail; between families use primarily
    # the coarse model (small residual for genuinely diagnostic construction).
    factor=np.diag([.80]*10);correction=layer('family_calibration','add',[dense('remove_local_max',f,-factor),dense('restore_shape_max',g,factor)])
    layer('scores','add',[fine,correction]);np.savez(folder/'weights.npz',**weights)
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1])
