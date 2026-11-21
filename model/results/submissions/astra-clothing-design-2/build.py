"""Construct analytic clothing-shape matched filters and a small fabric-detail circuit.
No examples, features from examples, or learned parameters are used by this file.
"""
import json, sys
from pathlib import Path
import numpy as np

# All drawing dimensions are proportions of a generic item's own bounding rectangle.
# A mask is integrated on subpixels for antialiasing; it represents occupancy, not a photo.
rng = np.random.default_rng(271828)
U = rng.uniform
X0,Y0 = np.meshgrid((np.arange(56)+.5)/56, (np.arange(56)+.5)/56)

def frame(aspect, *, hanging=False, shoe=False):
    scale=U(.935,1.035)
    w=scale*min(1.,aspect); h=scale*min(1.,1/aspect)
    cx=.5+U(-.018,.018)+U(-.08,.08)*max(0,1-w)
    # Vary the vertical framing of low objects and long handles. The whole
    # longest dimension is in view; allow play in the other dimension.
    cy=.5+U(-.014,.014)+max(0,1-h)*U(-.10,.36 if shoe else .15)
    y=(Y0-cy)/h+.5
    x=(X0-cx)/w+(y-.5)*U(-.055,.055)
    return x,y

def interp(t,knots,values):
    return np.interp(t,knots,values)

def poly(x,y,points):
    inside=np.zeros(x.shape,bool)
    x1,y1=points[-1]
    for x2,y2 in points:
        if y1!=y2:
            inside ^= ((y1>y)!=(y2>y)) & (x < x1+(x2-x1)*(y-y1)/(y2-y1))
        x1,y1=x2,y2
    return inside

def top(kind):
    # Type-specific styles share the same bodice and set-in sleeve construction.
    # b = body half-width in units of total garment width; e = cuff depth.
    t=U(); q=U(); rounded=0.; neck=U(.00,.055); shoulder_y=U(.06,.14)
    collar=False; cuff_slope=0.; sleeve='long'
    if kind=='tee':
        if t<.72:
            sleeve='short'; aspect=U(.70,1.18); b=U(.275,.38); e=U(.245,.43)
        elif t<.94:
            sleeve='tank'; aspect=U(.38,.78); b=U(.39,.49); e=.23; neck=U(.07,.25)
        else:
            sleeve='mid'; aspect=U(.65,1.12); b=U(.24,.36); e=U(.43,.65)
        hem=U(.955,1.); waist=b*U(.84,1.01); bottom=b*U(.90,1.08)
        rounded=U(-.015,.055); sh=b*U(.95,1.15)
    elif kind=='knit':
        aspect=U(.66,1.03); b=U(.26,.372); waist=b*U(.94,1.07); bottom=b*U(.92,1.10)
        hem=U(.81,1.); e=1. if hem<.955 else U(.84,1.); sh=b*U(.93,1.16)
        if q<.09: e=U(.62,.85); hem=1.
        if q>.90: collar=True
        rounded=U(-.018,.04)
    elif kind=='outer':
        # A jacket ends around the hip, a coat can extend far below the wrists.
        short=t<.45
        aspect=U(.73,.99) if short else U(.47,.78)
        b=U(.285,.39); waist=b*U(.85,1.04); bottom=b*U(.99,1.24)
        hem=U(.87,1.) if short else 1.; e=1. if hem<.97 else U(.59,.98)
        if not short: e=U(.53,.89)
        sh=b*U(.91,1.11); collar=q<.84; rounded=U(-.02,.055)
    elif kind=='shirt':
        # Includes dress shirts and flowing blouses, with several sleeve lengths.
        if t<.43:
            aspect=U(.60,.98); b=U(.26,.37); hem=1.; e=U(.75,1.02)
        elif t<.78:
            sleeve='short'; aspect=U(.81,1.32); b=U(.31,.425); hem=1.; e=U(.27,.46)
        elif t<.90:
            sleeve='mid'; aspect=U(.68,1.14); b=U(.26,.39); hem=1.; e=U(.48,.76)
        else:
            sleeve='tank'; aspect=U(.54,.88); b=U(.38,.49); hem=1.; e=.22; neck=U(.06,.25)
        sh=b*U(.90,1.12); waist=b*U(.86,1.10); bottom=b*U(.98,1.25)
        rounded=U(.01,.14); collar=q<.62 and sleeve!='tank'
        if t<.43 and U()<.52: e=1.;hem=U(.85,.99)
    else: # dress, bodice, distinct waist and skirt
        aspect=U(.38,.84); hem=1.
        b=U(.22,.35) if t<.8 else U(.34,.43)
        waist=b*U(.67,.95); bottom=U(max(waist,.31),.50); sh=b*U(.91,1.12)
        if t<.32: sleeve='tank'; neck=U(.05,.21); e=.20
        elif t<.68: sleeve='short'; e=U(.15,.30)
        else: sleeve='long'; e=U(.42,.75)
        if q<.42:
            # Straight shift and sheath dresses do not have a separate flared skirt.
            b=U(.30,.41);waist=b*U(.80,1.01);bottom=b*U(.91,1.17);sh=b*U(.90,1.12)
        shoulder_y=U(.035,.10); rounded=U(-.035,.06)
    x,y=frame(aspect)
    # Scale x later, so that the widest of a skirt, a hem or a cuff fills the box.
    tip=.5 if sleeve!='tank' else 0.
    if kind=='dress' and sleeve!='tank': tip=U(.32,.49)
    if sleeve=='tank': sh=b*U(.64,.99)
    sh=min(sh,.44)
    extent=max(bottom,b,waist,tip,sh)
    x=x*(2*extent)
    ax=np.abs(x+U(-.012,.012)*np.sin(y*np.pi*2))
    nc=U(.080,.160) if sleeve!='tank' else sh*U(.45,.76)
    # Raised collar or hood, slopes of shoulders, and hollow neck in tanks/tees.
    lift=U(.035,.115) if collar else U(0.,.025)
    neckpoint=nc+.020 if collar else nc
    sy=shoulder_y+lift*.70
    under=U(.19,.33) if kind!='dress' else U(.13,.23)
    under=max(under,sy+.045)
    half=interp(y,[0.,lift,lift+.009,sy,under,.46,.71,hem],
                [nc,neckpoint,neckpoint+.025,sh,b,waist,(bottom+waist)/2,bottom])
    if collar: neck*=.32
    depth=neck*np.maximum(0.,1-(ax/nc)**2)
    mask=(ax<half)&(y>=depth)&(y<=hem-rounded*(ax/max(.1,bottom))**2)
    if kind=='tee' and sleeve in ('mid','short') and U()<.18:
        # Raglan and batwing tees can have particularly wide slanting sleeves.
        e+=U(.015,.085)
    if sleeve!='tank':
        if sleeve=='short':
            width=U(.115,.19); cuff_slope=U(.005,.088)
        else:
            width=U(.068,.122) if kind!='outer' else U(.084,.138)
            cuff_slope=U(-.028,.036)
        armpity=under+U(.018,.10)
        for sgn in [-1,1]:
            sx=sgn*x; ex=tip+U(-.014,.014); ey=e+U(-.022,.022)
            mid=(sy+ey)*.5; outer_mid=(ex+sh)*.5+U(.005,.047)
            if sleeve=='long':
                inner_mid=(ex-width+b-.025)*.5+U(-.009,.028)
            else: inner_mid=(ex-width+b-.02)*.5
            sleeve_piece=[(sh-.075,sy-.024),(sh+.03,sy+.006),(outer_mid+(.018 if sleeve=='short' else 0.),mid-.012),
                   (ex,ey-cuff_slope),(ex-width,ey),(inner_mid,(ey+armpity)*.5),
                   (b-.025,armpity),(b-.060,under)]
            mask |= poly(sx,y,sleeve_piece)
    return mask

def trousers():
    style=U(); aspect=U(.29,.62) if style<.91 else U(.55,.90)
    x,y=frame(aspect); ax=np.abs(x)
    hip=U(.40,.50); belt=hip*U(.77,1.02); crotch=U(.27,.53)
    if style>.91: crotch=U(.44,.66)
    spread=U(.13,.34); calf=U(.07,.152) if style<.82 else U(.15,.23)
    outer=interp(y,[0.,.16,crotch,.67,1.],[belt,hip,hip*U(.88,1.),spread+calf*U(1.0,1.23),spread+calf]);
    inner=interp(y,[crotch,.65,1.],[0.,(spread-calf)*U(.32,.80),spread-calf]);
    return (ax<=outer)&((y<=crotch)|(ax>inner))&(y>=0)&(y<1.-.014*ax)

def bag():
    # A textile/leather pouch with a separate arch, two straps, or no handles.
    mode=U(); drop=0. if mode<.23 else U(.03,.29) if mode<.54 else U(.30,.70)
    body_aspect=U(1.,2.0) if mode<.45 else U(.72,1.65)
    x,y=frame(body_aspect*(1-drop)); ax=np.abs(x); v=(y-drop)/(1.-drop)
    topw=U(.37,.50); botw=U(.34,.50); midw=max(topw,botw,U(.46,.50))
    w=interp(v,[0.,.17,.50,.88,1.],[topw,midw,midw,(midw+botw)*.5,botw])
    curve=U(.012,.17); rim=U(-.08,.065)
    mask=(ax<w)&(v>rim*(1-(ax/.5)**2))&(v<1.-curve*(ax/.5)**6)
    if drop:
        spread=U(.135,.44); thick=U(.010,.047); power=U(1.35,2.65)
        rail=((ax/spread)**power+(np.maximum(drop-y,0)/drop)**power)**(1/power)
        mask |= (np.abs(rail-1)<thick/spread)&(y<drop+.035)&(y>-.01)
    return mask

def shoe(kind):
    # Side view: toe, vamp, instep/tongue, throat, heel counter, outsole.
    # Coordinates use x=0 at toe, x=1 at heel; heights measured up from sole.
    r=U(); tall=False; wedge=False
    if kind=='sneaker':
        aspect=U(2.05,3.5) if r<.90 else U(1.50,2.15); highheel=U(.00,.095)
    elif kind=='boot':
        aspect=U(.80,1.92); highheel=U(.19,.49) if r<.48 else U(.018,.17)
    else:
        tall=r<.40; wedge=r<.14
        aspect=U(.97,1.94) if tall else U(1.65,3.90)
        highheel=U(.41,.74) if tall else U(.0,.27)
    x,y=frame(aspect,shoe=True); x=x+.5
    fore=U(.22,.38); back=U(.69,.88); tip=U(.035,.12)
    base=interp(x,[0.,tip,fore,.50,back,1.],[U(.08,.20),U(.015,.065),.01,highheel*.20,highheel,highheel+U(-.02,.042)])
    h=1.-y; sole=U(.073,.175) if kind!='boot' else U(.065,.11)
    inset=U(.010,.05); heelw=U(.12,.28) if kind!='sandal' else U(.035,.20)
    toe_round=np.maximum(1-x/.075,0)**2*.08
    domain=(x>=0)&(x<=1.); floor=base+toe_round
    if kind=='sneaker':
        tongue=U(.52,.70); counter=U(.75,1.); dip=U(.52,.82); toe=U(.29,.51)
        roof=interp(x,[0.,.045,.15,.31,tongue-.05,tongue,tongue+.045,.82,.925,1.],
                    [toe*.49,toe*.86,toe,toe*U(1.0,1.20),.90,1.,.91,dip,counter,counter*.82]);
        mask=(h>floor)&(h<roof)&(x<1-inset*np.maximum(0,h-.34))
    else:
        thickness=sole*(1+U(-.35,.50)*x)
        mask=(h>floor)&(h<base+thickness)
        # Arch below the shank; an independent heel can be block, spike, or wedge.
        heel=(x>(1-heelw+(.5-h)*U(-.065,.060)))&(x<.99-np.maximum(.45-h,0)*U(0,.14))
        if wedge: heel=(x>fore)
        mask |= heel & (h>=interp(x,[fore,.75,1.],[0.,.025,.045])) & (h<=base+thickness)
        if kind=='boot':
            ankle=U(.48,.71); toe=U(.20,.38); shaft=U(.34,.63)
            roof=interp(x,[0.,.045,.19,.38,ankle-.045,ankle,ankle+U(.014,.06),.86,.98,1.],
                    [toe*.55,toe*.91,toe,toe*1.15,shaft,shaft+.13,U(.86,1.03),U(.82,1.0),U(.87,1.03),.78]);
            mask |= (h>=base+.015)&(h<roof)&(x<=1.-inset*np.maximum(h-.24,0))
        else:
            # Sandals expose skin between the toe/instep straps and above the footbed.
            nstrap=rng.integers(1,4); toe=U(.22,.55) if not tall else U(.15,.33)
            arch=np.clip(x/.77,0,1); roof=base+thickness+toe*(.60+.55*arch)
            for j in range(nstrap):
                loc=U(.09,.36) if j==0 else U(.34,.83)
                wd=U(.06,.27) if j==0 else U(.030,.135)
                line=loc+(h-base)*U(-.52,.55)
                mask |= (np.abs(x-line)<wd*.5)&(h>=base)&(h<roof)
            if U()<.80:
                ah=U(.77,1.0) if tall else U(.55,1.0)
                start=U(.33,.65); end=U(.83,.98); thick=U(.035,.105)
                p0=interp(start,[0.,fore,back,1.],[sole,sole,highheel+sole,highheel+sole]);
                line=p0+(ah-p0)*(x-start)/(end-start)
                mask |= (x>=start)&(x<end)&(np.abs(h-line)<thick)&(h>base)
                mask |= (x>end-U(.035,.19))&(x<end+.04)&(h>base)&(h<ah)&((x>end-.032)|(h>ah-.09) | (h<base+.20))
    return mask&domain

# Network construction utilities.
layers=[]; weights={}
def layer(name,op,inputs,**kw):
    layers.append(dict(name=name,op=op,inputs=[inputs] if isinstance(inputs,str) else inputs,**kw));return name
def dense(name,src,W,b=0):
    W=np.asarray(W,dtype='f'); weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype('f').copy()
    return layer(name,'dense',src,out=len(W))
def conv(name,src,W,b=0,p=0):
    W=np.asarray(W,dtype='f'); weights[name+'.W']=W; weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype('f').copy()
    return layer(name,'conv',src,out=len(W),kernel=list(W.shape[-2:]),padding=p,stride=1)
def relu(name,src): return layer(name,'relu',src)
def pool(name,op,src,hw,st):return layer(name,op+'pool',src,size=hw,stride=st)
def shape(name,src,sh):return layer(name,'reshape',src,shape=sh)
def plus(name,*args):return layer(name,'add',list(args))

def silhouettes():
    # Soft foreground membership. A weak blur reduces sensitivity to isolated
    # noise and small dark knit/stitch texture while largely retaining edges.
    k=np.array([[1,2,1],[2,20,2],[1,2,1]],float)/32
    starts=np.array([.009,.045,.16]); slopes=np.array([19.,(1.-19.*(.045-.009))/(.16-.045),0]);
    z=conv('ink_thresholds','image',k[None,None,:,:]*np.ones((3,1,1,1)),-starts,p=1)
    z=relu('ink_positive',z); z=conv('soft_foreground',z,np.diff(slopes,prepend=0)[None,:,None,None]);
    # Remove isolated flecks but do not require a thin strap to fill a 2x2 bin.
    med=pool('occupancy','avg',z,[2,2],[2,2]); ink=pool('fine_ink','max',z,[2,2],[2,2]);
    z=plus('antialiased_occupancy',conv('area_part',med,[[[[.84]]]]),conv('line_part',ink,[[[[.16]]]]))
    z=relu('ignore_specks',conv('noise_floor',z,[[[[1.]]]],-.06))
    eye=np.eye(196,dtype='f').reshape(14,14,196)
    shadows=[eye/(1-.06),np.cumsum(eye,axis=1)*1.50,np.cumsum(eye[:,::-1],axis=1)[:,::-1]*1.50,
             np.cumsum(eye,axis=0)*1.50,np.cumsum(eye[::-1],axis=0)[::-1]*1.50]
    u=dense('directional_sweeps',z,np.stack(shadows).reshape(-1,196));u=shape('sweep_planes',u,[1,5,196]);
    # min(u,1) = 1 - relu(1-u), supplied as the complementary map instead.
    return relu('empty_space',conv('remaining_space',u,[[[[-1.]]]],1.))

AREA=np.array([.51,.1125,.1125,.15,.115],dtype=np.float32)
def complement_features(mask):
    # Integrate the geometric sketch into pixels before the anti-aliased pooling.
    m=mask.reshape(28,2,28,2).mean((1,3));b=np.pad(m,1)
    m=.625*m+.0625*(b[:-2,1:-1]+b[2:,1:-1]+b[1:-1,:-2]+b[1:-1,2:])+.03125*(b[:-2,:-2]+b[2:,:-2]+b[:-2,2:]+b[2:,2:])
    # The clamp in the image pathway acts as a soft dilation of dark-on-black
    # boundaries. Apply the same geometric footprint to the contour sketch.
    m=np.minimum(m*2.5,1.);tiles=m.reshape(14,2,14,2)
    a=(.84*tiles.mean((1,3))+.16*tiles.max((1,3))).astype('f');a=np.maximum(a-.06,0)
    planes=[a/(1-.06),1.5*np.cumsum(a,axis=1),1.5*np.cumsum(a[:,::-1],axis=1)[:,::-1],
            1.5*np.cumsum(a,axis=0),1.5*np.cumsum(a[::-1],axis=0)[::-1]]
    return np.clip(1.-np.stack(planes),0,1)

def template_match(src):
    # For each class, max over silhouettes varying in style, size and pose.
    # 2<t,z> - ||t||^2 is a squared-error comparison up to a common offset.
    kinds=['tee','pants','knit','dress','outer','sandal','shirt','sneaker','bag','boot']
    n=1536; ng=4; W=np.empty((len(kinds)*n,980),'f');B=np.empty(len(W),'f');
    for i,kind in enumerate(kinds):
        for j in range(n):
            if kind=='bag': m=bag()
            elif kind=='pants': m=trousers()
            elif kind in ('boot','sneaker','sandal'):m=shoe(kind)
            else:m=top(kind)
            t=complement_features(m); ww=t*AREA[:,None,None]*(100./196)
            ix=i*n+j; W[ix]=(2*ww).ravel();B[ix]=-np.sum(t*ww)
    s=dense('cut_and_profile_matches',src,W,B);s=shape('by_class_and_group',s,[10,ng,n//ng]);
    v=pool('group_matches','max',s,[1,n//ng],[1,n//ng]); v1=pool('best_match','max',v,[ng,1],[ng,1]);
    v2=pool('consensus_match','avg',v,[ng,1],[ng,1]);
    v1=dense('best_cut',v1,np.eye(10)*.55);v2=dense('consensus_cut',v2,np.eye(10)*.45)
    return plus('silhouette_evidence',v1,v2)

# A second pathway looks inside the garment. It does not compare photos:
# filters examine center-front openings, surface pattern, and placket/lapel detail.
def appearance():
    z=layer('contrast','norm','image')
    # Cap the effect of small brilliant marks on extremely dark fabric.
    z=conv('tone_levels',z,[[[[1.]]],[[[1.]]]], [2.,-2.8]); z=relu('tone_hinges',z)
    z=conv('scaled_tone',z,[[[[1.]], [[-1.]]]],-2.)
    k=7; mid=k//2; fs=[];names=[]
    def filt(n,rows):
        a=np.zeros((k,k));
        for y,x,v in rows:a[mid+y,mid+x]+=v
        fs.append(a);names.append(n)
    def line(n,sep,extent):
        filt(n,[(y,x,v/(2*extent+1)) for y in range(-extent,extent+1) for x,v in [(-sep,-.5),(0,1),(sep,-.5)]])
    line('seam',2,3);line('wide',3,2)
    filt('vertical',[(0,-1,-.5),(0,0,1),(0,1,-.5)]);filt('horizontal',[(-1,0,-.5),(0,0,1),(1,0,-.5)])
    filt('vert_fold',[(y,x,v/5) for y in range(-2,3) for x,v in [(-2,-.5),(2,.5)]])
    filt('horiz_band',[(y,x,v/5) for x in range(-2,3) for y,v in [(-2,-.5),(2,.5)]])
    # Two signs rectify into absolute strength.
    fs=np.stack(fs)[:,None];f=len(fs); a=relu('oriented_responses',conv('cloth_filters',z,np.concatenate([fs,-fs]),p=3));
    a=conv('absolute_edges',a,np.concatenate([np.eye(f),np.eye(f)],axis=1)[:,:,None,None]);
    yy,xx=np.mgrid[:28,:28];xx=(xx+.5)/28-.5;yy=(yy+.5)/28
    w=[]; tags=[]
    def feat(n,parts):
        rows=np.zeros((f,28,28));
        for k,x0,x1,y0,y1,val in parts:
            m=(abs(xx)>=x0)&(abs(xx)<x1)&(yy>=y0)&(yy<y1);rows[names.index(k)]+=val*m/max(1,m.sum())
        tags.append(n);w.append(rows.ravel())
    for key,lo,hi in [('lower',.38,.82),('upper',.15,.43)]:
        feat('zip_'+key,[(k,x0,x1,lo,hi,t*fac) for k,fac in [('seam',.65),('wide',.35)] for x0,x1,t in [(0,.066,1),(.085,.205,-.75)]])
    for k in ['vertical','horizontal','vert_fold','horiz_band']:
        feat(k,[(k,.055,.225,.27,.73,1)])
    feat('lapel', [('vert_fold',.035,.17,.08,.34,1),('vert_fold',.045,.19,.41,.76,-.65)])
    feat('rib', [('horiz_band',0,.20,.77,.92,1),('horiz_band',0,.20,.42,.70,-.7)])
    a=dense('material_measurements',a,w)
    # Each cue saturates; none overrides a large disagreement with the cut.
    rules=[]; biases=[]; votes=[]
    def vote(expr,start,end,values):
        co=np.array([expr.get(n,0.) for n in tags])/(end-start)
        for threshold,sign in [(start,1),(end,-1)]:
            rules.append(co);biases.append(-threshold/(end-start));votes.append([values.get(i,0)*sign for i in range(10)])
    vote({'zip_lower':1},.035,.29,{2:-2.1,4:2.65,6:.95,0:-.55})
    vote({'zip_upper':1},.07,.38,{2:-.85,4:1.25,6:.6,0:-.5})
    vote({'lapel':1},.10,.35,{2:-.4,4:.8,6:.32})
    vote({'vertical':1},.12,.34,{6:2.3,0:-.65,2:-.75,4:-.45})
    vote({'vertical':1,'horizontal':.5},.22,.55,{6:1.15,0:-.25,2:-.25})
    vote({'horizontal':1,'vertical':-1},.075,.24,{2:1.0,6:-.3})
    vote({'rib':1},.18,.48,{2:.50})
    v=relu('material_cues',dense('cue_thresholds',a,rules,biases));
    # Plain knit and tee are the unmarked cases; the other tops have specific cues.
    fabric=dense('material_evidence',v,np.transpose(votes),[.55,0,.6,0,-.45,0,-1.1,0,0,0])
    # Coherent front fastening: a persistent bright OR dark thin ridge, a two-tone
    # split, or the edge of a lapel. Take magnitude AFTER integration along a line,
    # rather than confusing isolated pattern/knit marks with a central fastener.
    px=(xx+.5)*28; py=yy*28
    def gauss_strip(center,y1,y2,sigma=.8):
        g=np.exp(-.5*((px-center)/sigma)**2)*((py>=y1*28)&(py<y2*28));return g/max(1e-9,g.sum())
    def ridge(c,y1,y2,sep):
        return gauss_strip(c,y1,y2)-.5*(gauss_strip(c+sep,y1,y2)+gauss_strip(c-sep,y1,y2))
    groups=[]
    for y1,y2 in [(.15,.80),(.35,.85),(.10,.45)]:
        wf=[]
        for off in [-.09,-.055,-.025,0.,.025,.055,.09]:
            for sl in [-.075,0.,.075]:
                c=14.+28*off+28*sl*(yy-.5)
                for sep in (1.5,2.75):
                    rr=ridge(c,y1,y2,sep);wf.extend([rr,-rr])
                step=.7*(gauss_strip(c+1.45,y1,y2)-gauss_strip(c-1.45,y1,y2));wf.extend([step,-step])
        groups.append(wf)
    wf=[]
    for wide in [.085,.14,.205]:
        for depth in [.23,.35,.46]:
            for o in [-.046,0.,.046]:
                dx=28*wide*(depth-yy)/(depth-.04)
                rr=.5*(ridge(14+28*o+dx,.05,depth,2)+ridge(14+28*o-dx,.05,depth,2));wf.extend([rr,-rr])
    groups.append(wf)
    # For a knit crewneck or low tee collar the rim is a short curved band under the neck.
    wf=[]
    for wid in [.095,.14,.20]:
        for dep in [.075,.12,.19]:
            for off in [-.028,.0,.028]:
                dx=(xx-off)/wid;cy=(.015+dep*np.sqrt(np.maximum(0.,1.-dx**2)))*28
                def band(c):
                    g=(np.abs(dx)<1)*np.exp(-.5*((py-c)/.76)**2);return g/max(1e-9,g.sum())
                rr=band(cy)-.5*(band(cy+2)+band(cy-2));wf+= [rr,-rr]
    groups.append(wf)
    maxima=[]
    for j,wf in enumerate(groups):
        name='tailoring'+str(j);r=dense(name,z,np.array(wf).reshape(len(wf),-1));r=shape(name+'_axes',r,[1,1,len(wf)]);r=pool(name+'_max','max',r,[1,len(wf)],[1,len(wf)])
        maxima.append(dense(name+'_place',r,np.eye(len(groups))[:,j:j+1]))
    zz=plus('tailoring_strength',*maxima)
    # Use the same straight-line test off-center: all-over stripes and folds
    # should not be counted as a center-front fastening.
    refs=[]
    for off in [-.18,.18]:
        wf=[]
        for sl in [-.06,0.,.06]:
            for sep in (1.5,2.75):
                rr=ridge(14+28*off+28*sl*(yy-.5),.16,.80,sep);wf.extend([rr,-rr])
        refs.append(wf)
    ref=dense('parallel_fabric_lines',z,np.array(refs).reshape(2*len(refs[0]),-1));ref=shape('left_right_lines',ref,[1,2,len(refs[0])]);ref=pool('strong_flank_lines','max',ref,[1,len(refs[0])],[1,len(refs[0])]);ref=pool('flank_line_strength','avg',ref,[2,1],[2,1]);
    zz=plus('center_specific',zz,dense('discount_pattern',ref,np.array([-.80,-.6,-.50,-.20,0])[:,None]))
    co=[];bb=[];out=[]
    def evidence(coeff,lo,hi,vals):
        for sign,threshold in [(1,lo),(-1,hi)]:
            co.append(np.array(coeff)/(hi-lo));bb.append(-threshold/(hi-lo));out.append([sign*vals.get(i,0.) for i in range(10)])
    evidence([.75,.25,0,0,0],.11,.45,{2:-1.65,4:2.8,6:.6,0:-.8})
    evidence([.4,.1,.5,0,0],.16,.52,{2:-.65,4:1.4,6:.35})
    evidence([.35,0,0,.65,-.18],.18,.60,{2:-.5,4:1.,6:.65,0:-.5})
    evidence([-.35,-.25,0,0,.8],.22,.75,{2:.7,0:.8,4:-.35})
    extra=dense('fastening_and_neck',relu('tailoring_cues',dense('tailoring_thresholds',zz,co,bb)),np.transpose(out))
    # A checked woven shirt often has BOTH vertical and horizontal repeated
    # stripes that extend across sizeable regions. A central print on a tee or
    # horizontal knit stripes alone need not exhibit a warp-and-weft grid.
    checker=[]
    for along in [xx,yy]:
        rows=[]
        for period in [2.8,3.5,4.4,5.5,7.0,9.0]:
            for phase in [0,np.pi/4,np.pi/2,3*np.pi/4]:
                for lo,hi in [(.22,.53),(.40,.76),(.26,.74)]:
                    window=np.exp(-.5*(xx/.132)**2)*np.exp(-.5*((yy-(lo+hi)/2)/((hi-lo)/3))**2)
                    wave=np.cos(2*np.pi*along*28/period+phase);w=window*(wave-np.sum(window*wave)/window.sum());w=w/max(1e-8,np.sum(np.abs(w))); rows.extend([w,-w])
        checker+=rows
    rr=dense('woven_repeats',z,np.array(checker).reshape(len(checker),-1));rr=shape('warp_and_weft',rr,[1,2,len(checker)//2]);rr=pool('warp_and_weft_strength','max',rr,[1,len(checker)//2],[1,len(checker)//2]);
    d=relu('unbalanced_stripes',dense('grid_imbalance',rr,[[1,-1]]));vx=dense('warp',rr,[[1,0]]);weak=plus('crosshatch',vx,dense('remove_single_axis',d,[[-1]]))
    pr=relu('plaid_thresholds',dense('plaid_filter',weak,[[1],[1]],[-.070,-.235]));pat=dense('plaid_evidence',pr,np.outer(np.array([-.45,0,-.85,0,-.20,0,1.65,0,0,0])/.165,[1,-1]))
    return plus('cloth_and_tailoring',fabric,extra,pat)

def parts():
    # Row/column envelopes preserve topology even for a one-pixel strap or slit.
    # Use a full-resolution scan, independent of the coarse cut comparison.
    p=relu('firm_ink',conv('firm_threshold','soft_foreground',[[[[1.]]]],-.15))
    e=np.eye(784,dtype='f').reshape(28,28,784)
    a=[e/.85,np.cumsum(e,1)*1.8,np.cumsum(e[:,::-1],1)[:,::-1]*1.8,
       np.cumsum(e,0)*1.8,np.cumsum(e[::-1],0)[::-1]*1.8]
    v=dense('full_sweeps',p,-np.stack(a).reshape(-1,784),1.);v=shape('full_resolution',v,[5,28,28]);v=relu('background_scans',v)
    # a pixel in the horizontal chord has foreground on both left and right;
    # similarly for vertical chords. A gap has background at the pixel itself.
    wu=[[0,-1,-1,0,0],[0,0,0,-1,-1],[1,-1,-1,0,0],[1,0,0,-1,-1],
        [-1,0,0,0,0],[0,0,0,1,0],[0,0,0,0,1]]
    v=relu('part_maps',conv('chord_logic',v,np.array(wu)[:,:,None,None],[1,1,0,0,1,0,0]))
    yy,xx=np.mgrid[:28,:28];xx=(xx+.5)/28-.5;yy=(yy+.5)/28
    rows=[];names=[]
    def reg(n,ch,xlo,xhi,ylo,yhi):
        m=(xx>=xlo)&(xx<xhi)&(yy>=ylo)&(yy<yhi);r=np.zeros((len(wu),28,28));r[ch]=m/max(1,m.sum());rows.append(r.ravel());names.append(n)
    for n,y0,y1 in [('rim',.025,.11),('shoulder',.11,.26),('chest',.22,.37),('waist',.38,.58),('hip',.66,.82),('hem',.81,.95)]:
        reg(n,0,-.5,.5,y0,y1)
    reg('height',1,-.5,.5,0,1);reg('foot_area',4,-.5,.5,0,1)
    reg('vspace',3,-.46,.46,0,1);reg('vspace_toe',3,-.46,.02,0,1);reg('vspace_arch',3,-.18,.43,0,1)
    reg('front_height',1,-.45,-.15,0,1);reg('rear_height',1,.14,.44,0,1)
    reg('instep_top',5,.015,.16,0,1);reg('throat_top',5,.20,.33,0,1);reg('back_top',5,.31,.43,0,1)
    reg('toe_top',5,-.39,-.21,0,1);reg('archer_top',5,-.18,-.02,0,1)
    reg('leg_split',2,-.15,.15,.50,.96);reg('center_split',2,-.072,.072,.54,.94)
    for j in range(7):
        t=(j-3)*.038;reg('trouser_slit'+str(j),2,t-.031,t+.031,.58,.94)
    reg('neck_top',5,-.076,.076,0,.25); reg('L_sh_top',5,-.24,-.15,0,.25); reg('R_sh_top',5,.15,.24,0,.25)
    reg('sleeve_holes',2,-.5,.5,.62,.89)
    reg('arm_span',0,-.5,.5,.49,.76);reg('high_span',0,-.5,.5,.16,.33)
    reg('arm_background',2,-.5,.5,.44,.75)
    reg('underarm',0,-.5,.5,.335,.55)
    # A true split is connected to the outside below the two cuffs. Integrate
    # evidence for empty space along several outward or leaning crotch-to-cuff rays.
    raw=relu('near_black',conv('empty_pixel_test','image',[[[[-48.]]]],1.))
    wf=[]
    for dest in np.linspace(-.125,.125,9):
        for lean in [-.075,0.,.075]:
            c=dest+lean*(yy-.96)/.40
            path=np.exp(-.5*((xx-c)/.031)**2)*((yy>.54)&(yy<.978));wf.append(path.ravel()/path.sum())
    rr=dense('crotch_to_cuff',raw,wf)
    # Allow several independent leg spacings; the ray stays fixed. Require fabric
    # on both sides of that ray, not at one particular distance from it.
    def adjacent_leg(sgn,name):
        candidates=[]
        for dest in np.linspace(-.125,.125,9):
            for lean in [-.075,0.,.075]:
                c=dest+lean*(yy-.96)/.40
                for spacing in [.055,.085,.12,.16,.20]:
                    path=np.exp(-.5*((xx-c-spacing*sgn)/.029)**2)*((yy>.54)&(yy<.95));candidates.append(path.ravel()/path.sum())
        side=dense(name+'_spacings',raw,-np.array(candidates),1.);side=shape(name+'_rays',side,[1,len(wf),5]);side=pool(name+'_best','max',side,[1,5],[1,5]);return shape(name,side,[len(wf)])
    l=adjacent_leg(-1,'left_leg');r=adjacent_leg(1,'right_leg')
    ident=np.eye(len(wf));diff=relu('leg_asymmetry',plus('leg_diff',l,dense('minus_right',r,-ident)))
    both=plus('both_legs',l,dense('subtract_asymmetry',diff,-ident));rr=plus('bounded_slit',rr,dense('require_both_legs',both,ident*.95,-.95))
    rr=relu('positive_slit',rr);rr=shape('leg_ray_bank',rr,[1,1,len(wf)]);rr=pool('open_split','max',rr,[1,len(wf)],[1,len(wf)])
    names.append('to_cuff');rows.append(np.zeros_like(rows[0]))
    # Even when trouser legs touch, a waistband is straight and wide; across
    # the same span a dress or shirt usually slopes down toward the shoulders.
    belts=[]
    for wid in [.11,.15,.195]:
        for dx in [-.045,0.,.045]:
            for tilt in [-.12,0.,.12]:
                span=(np.abs(xx-dx)<wid);sl=yy-tilt*(xx-dx)
                before=span&(sl>=0)&(sl<.09);after=span&(sl>=.09)&(sl<.17)
                belts.append((before/max(1,before.sum())*.65+after/max(1,after.sum())*.35).ravel())
    bt=dense('waistband_fabric',raw,-np.array(belts),1.);bt=shape('belt_alternatives',bt,[1,1,len(belts)]);bt=pool('full_waistband','max',bt,[1,len(belts)],[1,len(belts)])
    names.append('belt_fill');rows.append(np.zeros_like(rows[0]));src=plus('part_measurements',dense('regional_means',v,rows),dense('place_ray',rr,np.eye(len(names))[:,-2:-1]),dense('place_belt',bt,np.eye(len(names))[:,-1:]))
    # A small arithmetic circuit: add, scale and clip continuous hand-written rules.
    # Nodes are affine functions of inputs and earlier ReLUs, compiled by depth.
    S={n:({i:1.},0.) for i,n in enumerate(names)}; ops=[]
    def lin(*terms,b=0.):
        d={}
        for k,c in terms:
            for i,v in S[k][0].items():d[i]=d.get(i,0.)+c*v
            b+=c*S[k][1]
        return d,b
    def hinge(expr):
        i=len(names)+len(ops);ops.append(expr);return ({i:1.},0.)
    def ramp(n,terms,lo,hi):
        t=lin(*[(k,c/(hi-lo)) for k,c in terms],b=-lo/(hi-lo));a,b=t
        S[n+'+']=hinge(t);S[n+'-']=hinge((a,b-1));S[n]=lin((n+'+',1),(n+'-',-1))
    def lowest(n,a,b):
        S[n+'+']=hinge(lin((a,1),(b,-1))); S[n]=lin((a,1),(n+'+',-1))
    def highest(n,a,b):
        S[n+'+']=hinge(lin((a,1),(b,-1))); S[n]=lin((b,1),(n+'+',1))
    ramp('long_gap',[('leg_split',.65),('center_split',.35)],.045,.30)
    # Localize an off-center split without requiring a memorized leg position.
    pairs=['trouser_slit'+str(i) for i in range(7)]
    while len(pairs)>1:
        out=[]
        for i in range(0,len(pairs),2):
            a=pairs[i]
            if i+1<len(pairs):
                b=pairs[i+1];n=a+'m';highest(n,a,b);a=n
            out.append(a)
        pairs=out
    ramp('slit',[(pairs[0],1)],.13,.65)
    ramp('waistband',[('rim',1),('chest',-.75)],-.04,.075)
    ramp('narrow',[('waist',-1)],-.67,-.44)
    ramp('no_arms',[('chest',-.50),('waist',-.50)],-.67,-.43)
    ramp('skirt',[('hem',.45),('hip',.55),('waist',-1)],.01,.14)
    ramp('sleeveless_length',[('waist',-.7),('chest',-.3)],-.52,-.33)
    ramp('very_flared',[('hip',1),('chest',-1)],.03,.21)
    ramp('boxy_shoulder',[('chest',1),('waist',-1)],.13,.36)
    ramp('cuff_above_hem',[('waist',.6),('hip',.4),('hem',-1)],.065,.23)
    ramp('neck_rise',[('L_sh_top',.5),('R_sh_top',.5),('neck_top',-1)],.16,.54)
    ramp('split_to_outside',[('to_cuff',1)],.26,.77)
    lowest('fly_gap','long_gap','split_to_outside');lowest('two_legs','slit','split_to_outside');lowest('flat_belt','waistband','narrow')
    ramp('solid_dress',[('to_cuff',-1)],-.42,-.085);lowest('straight_dress','solid_dress','narrow')
    ramp('full_belt',[('belt_fill',1)],.76,.94)
    ramp('tapered_legs',[('chest',.65),('waist',.35),('hem',-1)],.025,.15)
    lowest('taper_with_belt','tapered_legs','full_belt');lowest('pant_cut','taper_with_belt','flat_belt')
    ramp('gentle_slit',[('to_cuff',1)],.09,.38);lowest('possible_split','gentle_slit','flat_belt')
    ramp('flat_top',[('rim',1),('shoulder',-1)],-.065,-.012);lowest('belted','flat_belt','flat_top')
    lowest('bounded_skirt','skirt','no_arms');lowest('long_bodice','sleeveless_length','skirt')
    ramp('thin',[('height',-1)],-.33,-.17)
    ramp('thin_with_air',[('foot_area',-1)],-.22,-.10)
    ramp('strapspace',[('vspace',1),('height',-.038)],.001,.043)
    ramp('toe_open',[('vspace_toe',1),('front_height',-.035)],.001,.054)
    ramp('arch_open',[('vspace_arch',1),('rear_height',-.045)],.001,.083)
    ramp('rear_shaft',[('rear_height',1),('front_height',-.8)],.21,.42)
    ramp('low_counter',[('rear_height',-1)],-.48,-.31)
    ramp('ankle_notch',[('throat_top',1),('instep_top',-.65),('back_top',-.35)],.011,.075)
    ramp('rising_shaft',[('instep_top',1),('back_top',-1)],.014,.12)
    ramp('solid_upper',[('vspace',-1)],-.05,-.007);lowest('solid_low_counter','solid_upper','low_counter')
    ramp('flat_sole_profile',[('front_height',.4),('rear_height',.6),('foot_area',-1)],.070,.21)
    # A man's tee is wide across the sleeve ends and narrows abruptly; a tank
    # has no projecting cuffs. Full-length knitwear has the opposite envelope:
    # the arms descend outside the torso and may end below its ribbed hem.
    ramp('tee_step',[('high_span',1),('arm_span',-1)],.095,.28)
    ramp('broad_tee',[('high_span',1)],.69,.90)
    lowest('tee_cut','tee_step','broad_tee')
    ramp('long_wrist_span',[('arm_span',1)],.70,.88)
    ramp('outside_sleeve',[('arm_span',1),('shoulder',-1)],.045,.20)
    lowest('extended_sleeve','long_wrist_span','outside_sleeve')
    ramp('long_full_body',[('arm_span',.3),('hip',.7),('high_span',-.5)],.34,.53)
    ramp('tail_above_cuffs',[('sleeve_holes',1)],.065,.19)
    ramp('constricted_waist',[('hip',.53),('high_span',.47),('waist',-1)],.04,.14)
    lowest('hourglass','constricted_waist','no_arms')
    terms=[lin(('boxy_shoulder',.75),('tee_cut',2.6)),
           lin(('fly_gap',4),('two_legs',3.5),('flat_belt',1.5),('belted',1.),('split_to_outside',3.),('solid_dress',-1.5),('skirt',-1.6),('pant_cut',5.),('possible_split',2.0),b=-.85),
           lin(('neck_rise',-.65),('cuff_above_hem',-.8),('extended_sleeve',1.5),('long_full_body',.9),('tail_above_cuffs',.6)),
           lin(('bounded_skirt',3.2),('long_bodice',2.2),('very_flared',1.4),('sleeveless_length',1.25),('straight_dress',1.15),('hourglass',1.45),('split_to_outside',-3.5),('waistband',-.6),b=-.15),
           lin(('neck_rise',1.6),('cuff_above_hem',1.25),('extended_sleeve',1.),('long_full_body',.55)),
           lin(('strapspace',6.5),('toe_open',4.2),('arch_open',3.7),('thin',2.0),('thin_with_air',2.8),b=-.9),
           lin(('neck_rise',.35)),lin(('low_counter',.9),('strapspace',-.8),('ankle_notch',2.1),('solid_low_counter',1.3)),({},0.),
           lin(('rear_shaft',2.0),('low_counter',-.85),('strapspace',-1.0),('rising_shaft',1.4))]
    # Compile signed affine input measurements and nonnegative hidden nodes.
    # All original measurements are nonnegative; copy them through hidden ReLUs.
    depth=[0]*len(names);m=len(names)
    for a,b in ops:depth.append(1+max(depth[i] for i in a))
    ids=list(range(m));start=src
    for d in range(1,max(depth)+1):
        new=[i for i,v in enumerate(depth) if v==d];w=np.zeros((len(ids)+len(new),len(ids)));w[:len(ids)]=np.eye(len(ids));bias=np.zeros(len(w))
        for j,i in enumerate(new,len(ids)):
            a,b=ops[i-m];bias[j]=b
            for k,c in a.items():w[j,ids.index(k)]=c
        start=relu('logic'+str(d),dense('part_rule'+str(d),start,w,bias));ids+=new
    w=np.zeros((10,len(ids)));bias=np.zeros(10)
    for j,(a,b) in enumerate(terms):
        bias[j]=b
        for k,c in a.items():w[j,ids.index(k)]=c
    return dense('construction_evidence',start,w,bias)

def group_max(src,idx,name):
    # Broadcast the maximum of a subset back to exactly that subset.
    eye=np.eye(10);s=dense(name+'_select',src,eye[idx]);s=shape(name+'_group',s,[1,1,len(idx)])
    m=pool(name+'_max','max',s,[1,len(idx)],[1,len(idx)])
    return dense(name+'_broadcast',m,eye[idx].sum(0)[:,None])

def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
    space=silhouettes(); cut=template_match(space); cloth=appearance(); geom=parts()
    # Opening/fabric and strap cues decide the kind of garment or footwear,
    # not whether the object is a garment or a shoe. Preserve the coarse vote.
    tops=[0,2,4,6]; shoes=[5,7,9]; E=np.eye(10); garment=E.copy();garment[shoes]=0
    garment[tops]*=.38 # Sleeve details refine the garment subclass, not whether it is a bag.
    gc=dense('garment_parts',geom,garment);coarse=plus('object_and_cut',cut,gc)
    # An enclosed handle space and a pouch are both needed for this bonus.
    # Estimate a handle by looking for ink above a void connected to the bag's
    # filled lower body. The two sides may be angled and far apart at the base.
    yy,xx=np.mgrid[:28,:28];xx=(xx+.5)/28-.5;yy=(yy+.5)/28
    w=[];b=[]
    for wid in [.10,.18,.28,.38]:
        for top in [.025,.09,.16,.23,.30]:
            for dep in [.13,.22,.34,.48]:
                for dx in [-.04,.025]:
                    rim=top+dep*np.maximum(0,1-np.sqrt(np.maximum(0,1-((xx-dx)/wid)**2)))
                    crown=np.exp(-.5*((yy-rim)/.041)**2)*(np.abs(xx-dx)<wid)
                    left=np.exp(-.5*((xx-dx+wid)/.037)**2)*(yy>top+dep*.67)*(yy<top+dep+.026)
                    right=np.exp(-.5*((xx-dx-wid)/.037)**2)*(yy>top+dep*.67)*(yy<top+dep+.026)
                    hole=(np.abs(xx-dx)<wid*.62)&(yy>top+.05)&(yy<top+dep-.024)
                    body=(np.abs(xx-dx)<min(.39,wid+.12))&(yy>=top+dep+.04)&(yy<min(.94,top+dep+.26))
                    if not hole.any() or not body.any():continue
                    def avg(mask):return mask/max(.0001,mask.sum())
                    ww=.42*avg(crown)+.10*(avg(left)+avg(right))+.47*avg(body)-.95*avg(hole)
                    w.append(ww.ravel());b.append(-.53)
    br=relu('handle_shapes',dense('bag_handle_detector','soft_foreground',w,b));br=shape('different_handles',br,[1,1,len(w)]);br=pool('best_handle','max',br,[1,len(w)],[1,len(w)])
    br=dense('handle_evidence',br,E[:,8:9]*8.)
    allscores=plus('unconditional_evidence',cut,geom,cloth,br)
    delta=[]
    for nm,group in [('upper',tops),('foot',shoes)]:
        base=group_max(coarse,group,nm+'_coarse'); detail=group_max(allscores,group,nm+'_style')
        delta += [base,dense(nm+'_center',detail,-E)]
    plus('scores',allscores,*delta)
    (root/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(root/'weights.npz',**weights)

if __name__=='__main__':main()
