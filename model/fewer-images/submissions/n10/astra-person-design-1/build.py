"""Construct a person detector from colour inequalities and anatomical geometry.

No inputs except the name of the directory to write into. Spatial distances
in the part filters are in units of projected face height or torso width.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


class Net:
    def __init__(self):
        self.layers = []
        self.weights = {}
        self.channels = {'image': 3}

    def layer(self, name, op, inputs, channels, **kw):
        if isinstance(inputs, str):
            inputs = [inputs]
        assert name not in self.channels
        self.layers.append(dict(name=name, op=op, inputs=inputs, **kw))
        self.channels[name] = channels
        return name

    def conv(self, name, src, w, bias=0., stride=1, padding=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        co, ci, k, kw = w.shape
        assert ci == self.channels[src] and k == kw
        self.weights[name+'.W'] = w
        self.weights[name+'.b'] = np.broadcast_to(bias,(co,)).astype(np.float32).copy()
        return self.layer(name, 'conv', src, co, out=co, kernel=[k,k], stride=stride,
                          padding=k//2 if padding is None else padding)

    def dense(self, name, src, w, b=0.):
        w = np.array(w, dtype=np.float32, ndmin=2)
        n = len(w)
        self.weights[name+'.W'] = w
        self.weights[name+'.b'] = np.broadcast_to(b,(n,)).astype(np.float32).copy()
        return self.layer(name,'dense',src,n,out=n)

    def add(self, name, *src):
        assert len(set(self.channels[s] for s in src)) == 1
        return self.layer(name, 'add', list(src), self.channels[src[0]])

    def relu(self, src, name=None):
        return self.layer(name or src+'_pos', 'relu', src, self.channels[src])

    def pool(self, name, src, k, stride=None, kind='avg'):
        if isinstance(k,int): k=(k,k)
        if stride is None: stride=k
        if isinstance(stride,int): stride=(stride,stride)
        return self.layer(name,kind+'pool',src,self.channels[src],size=list(k),stride=list(stride))

    def reshape(self,name,src,shape):
        return self.layer(name,'reshape',src,shape[0],shape=list(shape))

    def maxchannels(self,name,src,h):
        # Move the filter index onto a poolable axis; no change of element order.
        c=self.channels[src]
        t=self.reshape(name+'_planes',src,[1,c,h*h])
        t=self.pool(name+'_best',t,(c,1),kind='max')
        return self.reshape(name,t,[1,h,h])

    def save(self, out):
        out.mkdir(parents=True,exist_ok=True)
        (out/'network.json').write_text(json.dumps({'layers':self.layers},indent=1)+'\n')
        np.savez(out/'weights.npz',**self.weights)


LUMA = np.array([.30,.59,.11])
U = np.array([1.,-1.,0])             # red minus green
V = np.array([0.,1.,-1.])            # green minus blue


def colours(net):
    rgb=net.pool('two_pixel_colour','image',2)
    # A soft convex gamut around the skin locus. Red excess alone responds to
    # many red objects, so test the slope and upper/lower chroma bounds too.
    # Use a luminance-dependent floor to preserve deeply shaded complexions.
    planes=[(-U+.072*LUMA, .006, 21),
            (-U-V+.145*LUMA, .012, 11),
            (.16*U-V, -.021, 14),
            (V-1.42*U, -.008, 16),
            (U-.52*LUMA, -.052, 10),
            (U+V-.91*LUMA, -.070, 7),
            (-LUMA, .078, 8),
            (U-3.6*V, -.126, 5)]
    hinges=net.relu(net.conv('gamut_boundaries',rgb,[w*s for w,b,s in planes],
                            [b*s for w,b,s in planes]))
    skin=net.relu(net.conv('complexion',hinges,[-np.ones(len(planes))],1.))
    return rgb,skin


def grid(k):
    y,x=np.mgrid[:k,:k]-k//2
    return x.astype(float),y.astype(float)


def patch(x,y,cx,cy,sx,sy,tilt=0.):
    u=x-cx; v=y-cy
    ca,sa=np.cos(tilt),np.sin(tilt)
    a=(u*ca+v*sa)/sx; b=(v*ca-u*sa)/sy
    # Subtracting max only prevents underflow when a sample is off-grid.
    e=-.5*(a*a+b*b)
    z=np.exp(e-e.max()); return z/z.sum()


def average(seq):
    # Used only for mixtures of analytically specified sampling regions of ONE
    # filter, e.g. the left and right cheek of the same face.
    return sum(seq)/len(seq)


def surround(x,y,w,h,tilt):
    ca,sa=math.cos(tilt),math.sin(tilt)
    def g(u,v,sx,sy):
        return patch(x,y,ca*u*w-sa*v*h,sa*u*w+ca*v*h,sx*w,sy*h,tilt)
    core=g(0,0,.23,.24)
    # Hair, space beside the cheeks, and clothing under the chin are less
    # likely to share the skin colour; the neckline is allowed to be skin.
    sides=average([g(s*.86,-.08,.24,.33) for s in [-1,1]])
    cap=g(0,-.75,.31,.20); below=g(0,.90,.42,.23)
    outside=.51*sides+.33*cap+.16*below
    return core,outside,g


def face_bank(net, skin, suffix, n, scale, stride):
    # Each candidate produces TWO features, later combined with luma/edge
    # evidence for that same candidate (not a free match at another location).
    # Heights measured on original 96-pixel image. Overlapping octave bands.
    heights={'fine':[3.6,5.8,8.6], 'mid':[10.,15.,21.],
             'large':[23.,34.,49.]}[suffix]
    k=13 if suffix!='large' else 15
    x,y=grid(k)
    specs=[]; w=[]
    for orig in heights:
        h=orig/scale
        for aspect,tilt in [(.63,0.),(1.06,0.),(.83,.38),(.83,-.38)]:
            fw=aspect*h
            core,outside,g=surround(x,y,fw,h,tilt)
            # Do not require a skin-coloured forehead or mouth. A small head
            # may have its upper half covered by hair or a helmet.
            cheeks=.6*core+.2*g(-.20,.10,.16,.20)+.2*g(.20,.10,.16,.20)
            w.extend([cheeks-.83*outside,cheeks])
            specs.append((h,fw,tilt))
    feats=net.conv('face_colour_'+suffix,skin,np.array(w)[:,None],stride=stride)
    return specs,feats


def luma_features(net,rgb,suffix):
    lum=net.conv('luminance_'+suffix,rgb,[LUMA])
    # The explicit opponent directions and the signed differences of RGB
    # also contribute to the (unsigned) contour strength maps below.
    return lum


def edge_features(net,rgb,suffix,extended):
    x,y=grid(3)
    angles=np.arange(4)*math.pi/4
    deriv=[]
    for a in angles:
        c,s=np.cos(a),np.sin(a)
        along=c*x+s*y; across=-s*x+c*y
        v=along*np.exp(-.5*((along/.84)**2+(across/.82)**2))
        v=v/np.maximum(v,0).sum()
        deriv.extend([LUMA[:,None,None]*v, -LUMA[:,None,None]*v])
    if extended:
        for a in angles:
            c,s=np.cos(a),np.sin(a)
            along=c*x+s*y; across=-s*x+c*y
            v=along*np.exp(-.5*((along/.84)**2+(across/.82)**2))
            v=v/np.maximum(v,0).sum()
            # Clothing can have a clear boundary with barely a change in luma.
            for opp in [U,.5*U+V]:
                deriv.extend([opp[:,None,None]*v,-opp[:,None,None]*v])
    dif=net.relu(net.conv('directed_edges_'+suffix,rgb,deriv))
    a=np.zeros((4,len(deriv)))
    for i in range(4):
        a[i,2*i:2*i+2]=1
        if extended: a[i,8+4*i:12+4*i]=.24
    mag=net.conv('contrast_'+suffix,dif,a)
    # Capped, cross-orientation-inhibited gradient responses. Gives less
    # weight to a strong straight line activating several orientation bins.
    inhib=.62; gain=8.5
    orient=np.eye(4)-inhib/3*(1-np.eye(4))
    soft=net.relu(net.conv('oriented_ramps_'+suffix,mag,np.vstack((orient,orient))*gain,
                           [0.]*4+[-1.]*4))
    bounded=net.conv('oriented_contours_'+suffix,soft,np.hstack((np.eye(4),-np.eye(4))))
    return bounded


def face_detector(net,skin,lum,edges,suffix,n,scale,stride):
    specs,col=face_bank(net,skin,suffix,n,scale,stride)
    k=13 if suffix!='large' else 15
    x,y=grid(k)
    lw=[]; ew=[]
    # Hair cap -> face; eyebrow/eye band darker than cheek, nose ridge
    # brighter than eyes, mouth darker than chin. The paired-eye response is
    # computed separately from the oval to avoid accepting just a skin patch.
    for h,fw,t in specs:
        ca,sa=math.cos(t),math.sin(t)
        core,out,g=surround(x,y,fw,h,t)
        eyes=average([g(s*.235,-.13,.112,.074) for s in [-1,1]])
        below=average([g(s*.22,.12,.135,.12) for s in [-1,1]])
        forehead=g(0,-.36,.24,.11); nose=g(0,-.055,.09,.16)
        mouth=g(0,.30,.15,.064); chin=g(0,.43,.16,.09)
        cap=g(0,-.56,.28,.13); sidehair=average([g(s*.49,-.17,.13,.27) for s in [-1,1]])
        hair=.56*core+.24*below+.20*forehead-.65*cap-.35*sidehair
        detail=.52*below+.22*forehead+.16*nose+.10*chin-.83*eyes-.17*mouth
        lw.extend([hair,detail])
        # Respond to the curved sides of an egg shape. Subtract nearby,
        # displaced contours of the same orientation to reject busy texture.
        shape=np.zeros((4,k,k))
        for a in (np.arange(16)+.5)*2*math.pi/16:
            u=.52*fw*math.cos(a); v=.53*h*math.sin(a)
            cx=ca*u-sa*v; cy=sa*u+ca*v
            normang=t+math.atan2(math.sin(a)/h,math.cos(a)/fw)
            ow=np.maximum(0,np.cos(2*(np.arange(4)*math.pi/4-normang)))**2
            ow/=ow.sum()
            rad=max(.58,.115*min(fw,h))
            rim=patch(x,y,cx,cy,rad,rad)
            inner=patch(x,y,.36*cx,.36*cy,rad,rad)
            outer=patch(x,y,1.83*cx,1.83*cy,rad*1.3,rad*1.3)
            shape+=ow[:,None,None]*(rim-.46*inner-.54*outer)/16
        ew.append(shape)
    l=net.conv('hair_and_eyes_'+suffix,lum,np.array(lw)[:,None],stride=stride)
    e=net.conv('oval_contour_'+suffix,edges,ew,stride=stride)
    m=len(specs)
    # Three rules per geometry: exposed face (including profiles), frontal
    # face, and a stricter achromatic rule for monochrome photographs.
    cm=np.zeros((3*m,2*m)); lm=np.zeros_like(cm); em=np.zeros((3*m,m))
    b=[]
    for i,(h,fw,t) in enumerate(specs):
        # All branches report margins, in units of a skin likelihood contrast.
        cm[i*3,2*i:2*i+2]=[.90,.10]
        lm[i*3,2*i:2*i+2]=[.83,.54]
        em[i*3,i]=.18
        b.append(-.79)
        resolved=min(1,max(0,(h*scale-4)/8))
        cm[i*3+1,2*i:2*i+2]=[.52,.18]
        lm[i*3+1,2*i:2*i+2]=[1.,2.05*resolved]
        em[i*3+1,i]=.29
        b.append(-.71)
        cm[i*3+2,2*i:2*i+2]=[.12,.07]
        lm[i*3+2,2*i:2*i+2]=[1.16,2.7*resolved]
        em[i*3+2,i]=.46
        b.append(-.665)
    cf=net.conv('colour_rule_'+suffix,col,cm,b)
    lf=net.conv('relief_rule_'+suffix,l,lm)
    ef=net.conv('boundary_rule_'+suffix,e,em)
    # Anti-false-positive gate: a stripe or shadow can give the right
    # SIGNED mean but it does not have two eyes. Require both eye-cheek
    # contrasts and support on both sides against hair for achromatic hits.
    gw=[]
    for h,fw,t in specs:
        _,_,g=surround(x,y,fw,h,t)
        for s in [-1,1]:
            low=g(s*.22,.12,.12,.105); eye=g(s*.22,-.14,.12,.075)
            high=g(s*.19,-.34,.15,.10)
            gw.append(eye-.67*low-.33*high)
        for s in [-1,1]:
            gw.append(g(s*.50,-.31,.17,.23)-g(s*.17,.04,.17,.22))
    deficits=net.relu(net.conv('paired_part_deficits_'+suffix,lum,np.array(gw)[:,None],
                               [.048,.048,.035,.035]*m,stride=stride))
    dm=np.zeros((3*m,4*m))
    for i in range(m):
        dm[3*i,4*i:4*i+4]=[-.08,-.08,-.07,-.07]
        dm[3*i+1,4*i:4*i+4]=[-1.1,-1.1,-.23,-.23]
        dm[3*i+2,4*i:4*i+4]=[-2.6,-2.6,-.90,-.90]
    df=net.conv('bilateral_support_'+suffix,deficits,dm)
    rules=net.add('head_rules_'+suffix,cf,lf,ef,df)
    return net.maxchannels('head_'+suffix,rules,n//stride)


def contour_model(x,y,segments,interior):
    # Integral of a directional derivative field along an anatomical outline.
    # Segment notation: (x1,y1,x2,y2, importance). Four unsigned orientations.
    k=len(x); f=np.zeros((4,k,k)); strength=0.
    for x1,y1,x2,y2,amount in segments:
        dx=x2-x1;dy=y2-y1;length=math.hypot(dx,dy)
        if length<1e-6: continue
        normal=math.atan2(-dx,dy)
        ow=np.maximum(0,np.cos(2*(np.arange(4)*math.pi/4-normal)))**2
        ow/=ow.sum()
        along=((x-x1)*dx+(y-y1)*dy)/length
        d=((x-x1)*dy-(y-y1)*dx)/length
        beyond=np.maximum(np.maximum(-along,along-length),0)
        rad=.68
        stroke=np.exp(-.5*((d/rad)**2+(beyond/rad)**2))
        stroke/=stroke.sum()
        mass=length*amount
        f+=mass*ow[:,None,None]*stroke; strength+=mass
    f/=strength
    # Penalise diffuse texture, and lines parallel to but displaced from the
    # expected boundaries. Each part is independently normalised in space.
    shifted=(np.roll(f,2,axis=1)+np.roll(f,-2,axis=1)+np.roll(f,2,axis=2)+np.roll(f,-2,axis=2))*.25
    density=f.sum((1,2))[:,None,None]*interior
    return f-.36*shifted-.64*density


def figure_detector(net,edges,lum,skin,suffix,n,stride,unit):
    # Anchor at the lower face; all point lists below are in units of the
    # width of the shoulders. Part groups can fail independently (occlusion).
    k=15; x,y=grid(k)
    upper=[]; torso=[]; legs=[]; skinw=[]; regions=[]; specs=[]
    for width,height in [(2.9,3.8),(4.7,4.1),(4.0,6.1),(6.4,5.8)]:
        for lean in [-.13,.13]:
            def xy(p): return ((p[0]+lean*p[1])*width,p[1]*height)
            def chain(points,amt=1.):
                return [(*xy(p),*xy(q),amt) for p,q in zip(points[:-1],points[1:])]
            seg=[]; ts=[]; ls=[]
            head=[(-.14,.02),(-.20,-.18),(-.17,-.37),(-.09,-.47),(.08,-.47),(.18,-.36),(.20,-.17),(.14,.02)]
            seg+=chain(head,1.1)
            for s in [-1,1]:
                def sym(points): return [(s*a,b) for a,b in points]
                seg+=chain(sym([(.14,-.04),(.13,.07),(.43,.17),(.52,.28),(.56,.46)]),1.)
                ts+=chain(sym([(.44,.22),(.50,.43),(.43,.78),(.39,1.17)]))
                ls+=chain(sym([(.40,1.0),(.39,1.24),(.29,1.73)]))
                ls+=chain(sym([(.08,1.21),(.14,1.65)]),.55)
            def g(u,v,sx,sy):
                a,b=xy((u,v));return patch(x,y,a,b,max(.42,width*sx),max(.42,height*sy))
            hreg=g(0,-.17,.13,.15); shirt=g(0,.54,.23,.30)
            outer=average([g(s*.62,-.12,.19,.25) for s in [-1,1]])
            upper.append(contour_model(x,y,seg,.38*outer+.28*hreg+.34*shirt))
            sides=average([g(s*.74,.65,.20,.40) for s in [-1,1]])
            torso.append(contour_model(x,y,ts,.47*shirt+.53*sides))
            legs.append(contour_model(x,y,ls,.46*g(0,1.42,.21,.23)+.54*sides))
            skinw.append(hreg-.48*outer-.29*g(0,-.63,.21,.14)-.23*shirt)
            # Centre versus both flanks; a single step edge averages to one
            # strong response and one near-zero response. Use both below.
            for s in [-1,1]:
                regions.append(shirt-g(s*.77,.57,.19,.28))
            specs.append((width,height,lean))
    m=len(specs)
    f=net.conv('figure_parts_'+suffix,edges,np.vstack((upper,torso,legs)),stride=stride)
    c=net.conv('figure_complexion_'+suffix,skin,np.array(skinw)[:,None],stride=stride)
    v=net.conv('figure_flanks_'+suffix,lum,np.array(regions+[-r for r in regions])[:,None],stride=stride)
    d=net.relu(v)
    # min(abs(left),abs(right)) is preferable to sum; in addition to luma
    # difference require a consistent contrast polarity on both sides.
    vm=np.zeros((4*m,4*m))
    for i in range(m):
        vm[4*i,2*i]=1; vm[4*i,2*i+1]=-1
        vm[4*i+1,2*i]=-1; vm[4*i+1,2*i+1]=1
        vm[4*i+2,2*m+2*i]=1; vm[4*i+2,2*m+2*i+1]=-1
        vm[4*i+3,2*m+2*i]=-1; vm[4*i+3,2*m+2*i+1]=1
    delta=net.relu(net.conv('flank_difference_'+suffix,d,vm))
    vm1=np.zeros((m,4*m)); vm2=np.zeros_like(vm1)
    for i in range(m):
        vm1[i,2*i:2*i+2]=.5;vm1[i,2*m+2*i:2*m+2*i+2]=.5
        vm2[i,4*i:4*i+4]=-.5
    support=net.add('garment_'+suffix,net.conv('flank_mean_'+suffix,d,vm1),
                    net.conv('flank_agreement_'+suffix,delta,vm2))
    # Cap clothing contrast to prevent a uniform window, door or sign from
    # passing based on contrast alone.
    sat=net.relu(net.conv('garment_excess_'+suffix,support,np.eye(m),-.22))
    support=net.add('garment_capped_'+suffix,support,net.conv('garment_clip_'+suffix,sat,-np.eye(m)))
    a=np.zeros((2*m,3*m)); b=np.zeros((2*m,m)); cm=b.copy(); bias=[]
    for i in range(m):
        ix=[i,i+m,i+2*m]
        a[i*2,ix]=[.64,.38,.10]; b[i*2,i]=.42; cm[i*2,i]=.60; bias.append(-.57)
        a[i*2+1,ix]=[.95,.69,.21]; b[i*2+1,i]=.60;cm[i*2+1,i]=.16; bias.append(-.54)
    sc=net.add('figure_rules_'+suffix,net.conv('body_shape_rule_'+suffix,f,a,bias),
               net.conv('body_colour_rule_'+suffix,c,cm),net.conv('body_shirt_rule_'+suffix,support,b))
    return net.maxchannels('figure_'+suffix,sc,n//stride)


def landmarks(net, skin, src, band, n, sample, diameters):
    # For a larger face, keep the full sampling bandwidth at eyes and mouth.
    # Each landmark is compared to immediately neighbouring tissue, not to
    # the patch mean. Thus a single horizontal edge can't explain a whole face.
    light = src
    k=15; x,y=grid(k); kernels=[]; colour=[]; count=0
    for h in diameters:
        for aspect,t in [(.60,0.),(.98,0.),(.77,-.28),(.77,.28)]:
            w=aspect*h; ca,sa=np.cos(t),np.sin(t)
            def g(a,b,sx,sy):
                return patch(x,y,ca*a*w-sa*b*h,sa*a*w+ca*b*h,
                             max(.39,sx*w),max(.39,sy*h),t)
            eyes=[]; cheeks=[]
            for s in [-1,1]:
                eye=g(s*.215,-.125,.090,.047); eyes.append(eye)
                cheek=g(s*.22,.062,.117,.075); cheeks.append(cheek)
                browtop=g(s*.19,-.30,.13,.078)
                kernels.append(.64*cheek+.36*browtop-eye)
            kernels.append(g(0,-.086,.073,.10)-average(eyes))
            mouth=g(0,.265,.149,.047)
            kernels.append(.50*g(0,.153,.109,.06)+.50*g(0,.393,.135,.065)-mouth)
            kernels.append(average(cheeks)-.72*g(0,-.52,.24,.085)-
                           .14*g(-.49,-.27,.092,.17)-.14*g(.49,-.27,.092,.17))
            skinface=.45*average(cheeks)+.24*g(0,-.29,.22,.10)+.31*g(0,.13,.23,.18)
            around=.35*g(0,-.63,.24,.12)+.22*g(-.73,-.06,.15,.24)+.22*g(.73,-.06,.15,.24)+.21*g(0,.70,.29,.14)
            colour.append(skinface-.64*around)
            count+=1
    response=net.conv('local_landmark_contrast_'+band,light,np.array(kernels)[:,None],stride=sample)
    sk=net.conv('landmark_complexion_'+band,skin,np.array(colour)[:,None],stride=sample)
    m=count; eyes=np.tile([1.72,1.72,1.04,.94,.60],m)
    req=np.tile([.042,.042,.014,.022,.040],m)
    # Subtract a hinge for missing parts and a second hinge above saturation:
    # facial landmarks all need support and no single one can dominate.
    w=np.eye(5*m); caps=np.tile([.24,.24,.22,.21,.29],m)
    hinges=net.relu(net.conv('landmark_hinges_'+band,response,np.vstack((-w,w)),np.r_[req,-caps]))
    mat=np.zeros((m,m*5)); pen=np.zeros((m,m*10))
    for i in range(m):
        loc=slice(5*i,5*i+5)
        mat[i,loc]=eyes[loc];pen[i,loc]=[-2.45,-2.45,-1.24,-1.35,-.42]
        pen[i,5*m+5*i:5*m+5*i+5]=-eyes[loc]
    sc=net.add('constellation_'+band,net.conv('landmark_vote_'+band,response,mat,-.585),
               net.conv('missing_landmarks_'+band,hinges,pen),
               net.conv('landmark_colour_vote_'+band,sk,np.eye(m)*.195))
    return net.maxchannels('aligned_face_'+band,sc,n//sample)


def parts_readout(net,part,n,tag,gain):
    # Aggregate one high-confidence detection, corroborating detections in
    # other cells, and weaker repeated hits (groups of people).
    # Suppress occasional extreme single-filter responses by a soft ceiling.
    cap=net.relu(net.conv(tag+'_excess',part,[[1.]],-.37))
    sc=net.add(tag+'_capped',part,net.conv(tag+'_clip',cap,[[-.80]]))
    # Penalise edge truncations, especially top-of-frame and ground-level
    # maxima. Avoid hard cropping: real people can be partly outside.
    y,x=np.mgrid[:n,:n];y=(y+.5)/n;x=(x+.5)/n
    prior=.027*np.abs(2*x-1)**3+.036*np.maximum(0,(.12-y)/.12)+.095*np.maximum(0,(y-.65)/.35)**1.3
    off=net.dense(tag+'_location',net.pool(tag+'_reference',sc,n),np.zeros((n*n,1)),-prior.ravel())
    off=net.reshape(tag+'_offset',off,[1,n,n]); s=net.add(tag+'_map',sc,off)
    pk=net.pool(tag+'_peak',s,n,kind='max')
    # A soft top few score: average of positive tile margins plus strongest.
    tiled=net.pool(tag+'_tiles',s,n//4,kind='max')
    pos=net.relu(net.conv(tag+'_tile_ramp',tiled,[[1.]],.085))
    multiple=net.pool(tag+'_multi',pos,4)
    return net.add(tag+'_detection',net.conv(tag+'_peak_vote',pk,[[gain]]),
                   net.conv(tag+'_group_vote',multiple,[[1.10*gain]]))


def localized_colours(net,rgb,skin):
    # Clothing cues in the neighbourhood below a putative face. A colourful
    # lower half alone is not enough: the upper colour must be localised.
    qs=[U,-U,V,-V,U+V,-U-V]
    ramps=net.relu(net.conv('fabric_chroma',rgb,qs,[-.25,-.12,-.20,-.12,-.35,-.18]))
    z=net.conv('colourful_fabric',ramps,[np.ones(6)*1.6])
    zs=net.relu(net.conv('fabric_saturation',z,[[1.]],-.66))
    z=net.add('fabric',z,net.conv('fabric_clip',zs,[[-1.]]))
    # Blue/green grass and water can look colourful but are not typically a
    # rectangle of contrasting hue immediately under an isolated face.
    k=11;x,y=grid(k);sw=[];cw=[]
    for h in [1.6,2.8,4.8]:
        fw=.80*h
        core,out,g=surround(x,y,fw,h,0.)
        sw.append(core-.79*out)
        shirt=g(0,1.64,.62,.58)
        back=average([g(s*1.75,1.40,.47,.66) for s in [-1,1]])
        cw.append(shirt-.74*back-.26*core)
    s=net.conv('skin_over_shirt',skin,np.array(sw)[:,None],stride=2)
    c=net.conv('shirt_under_skin',z,np.array(cw)[:,None],stride=2)
    r=net.add('shirt_rules',net.conv('isolated_skin',s,np.eye(3)*.77,-.68),
              net.conv('distinctive_shirt',c,np.eye(3)*.74))
    return net.maxchannels('person_in_colour',r,12)


def global_context(net,skin,rgb,edges):
    # Use only as small adjustments to detection evidence: portraits tend
    # to have some warm complexion in the middle, not wall-to-wall wood,
    # sand, or food. Similarly, very long straight edges suggest structures.
    tile=net.pool('colour_tiles',skin,6)
    y,x=np.mgrid[:8,:8]; x=(x-3.5)/4; y=(y-3.)/4
    center=np.exp(-1.4*x*x-1.5*y*y); center/=center.sum()
    surround=1-np.exp(-2*x*x-2*y*y); surround/=surround.sum()
    whole=np.full((8,8),1/64)
    cm=net.relu(net.dense('colour_context',tile,[center.ravel(),center.ravel(),whole.ravel(),whole.ravel(),
                                               (center-surround).ravel()],[-.008,-.11,-.19,-.38,.0]))
    sc=net.dense('colour_context_vote',cm,[[.78,-.78,-.27,.23,.13]])
    # Distinguish field sports scenes with players from empty vegetation:
    # reward grass concentrated at the bottom with colour contrast above.
    veg=net.relu(net.conv('grass_gamut',rgb,[-U,V],[-.012,-.038]))
    # min of green excess over red and green excess over blue.
    diff=net.relu(net.conv('grass_difference',veg,[[1.,-1.]]))
    g=net.add('grass_evidence',net.conv('green_excess',veg,[[1.,0.]]),net.conv('green_clip',diff,[[-1.]]))
    g=net.pool('grass_tiles',g,3)
    lawn=np.ones((8,8));lawn[:4]*=-1;lawn/=32
    gsc=net.relu(net.dense('field_layout',g,[lawn.ravel()],-.015))
    gsc=net.dense('field_context_vote',gsc,[[.60]])
    # Horizontal or vertical contours at the SAME coordinate, continuing
    # through several image tiles, are boundary cues for man-made scenes.
    long_h=net.pool('horizontal_span',edges,(1,12),stride=(1,6))
    long_v=net.pool('vertical_span',edges,(12,1),stride=(6,1))
    h=net.relu(net.conv('horizontal_lines',long_h,[[0.,0.,1.,0.]],-.33))
    v=net.relu(net.conv('vertical_lines',long_v,[[1.,0.,0.,0.]],-.33))
    h=net.pool('horizontal_line_strength',h,(24,3),kind='max')
    v=net.pool('vertical_line_strength',v,(3,24),kind='max')
    geom=net.dense('straight_line_vote',net.add('long_lines',h,v),[[-.059]])
    return net.add('context',sc,gsc,geom)


def build(out):
    net=Net(); rgb2,s2=colours(net)
    rgb4=net.pool('four_pixel_colour',rgb2,2); s4=net.pool('four_pixel_skin',s2,2)
    rgb8=net.pool('eight_pixel_colour',rgb4,2); s8=net.pool('eight_pixel_skin',s4,2)
    faces=[]; bodies=[]
    for label,rgb,skin,scale,stride in [('fine',rgb2,s2,2,2),('mid',rgb4,s4,4,1),('large',rgb8,s8,8,1)]:
        n=96//scale
        lum=luma_features(net,rgb,label)
        ed=edge_features(net,rgb,label,extended=(scale==4))
        face=face_detector(net,skin,lum,ed,label,n,scale,stride)
        faces.append(parts_readout(net,face,n//stride,'f_'+label,1.))
        if scale<=4:
            body=figure_detector(net,ed,lum,skin,label,n,stride,scale)
            bodies.append(parts_readout(net,body,n//stride,'b_'+label,1.0))
        if scale==4: mid_edges=ed
        if scale<=4:
            diam=[5.0,8.1,11.4] if scale==2 else [7.8,11.6]
            detail=landmarks(net,skin,lum,label,n,2,diam)
            faces.append(parts_readout(net,detail,n//2,'landmarks_'+label,1.))
    shirt=localized_colours(net,rgb4,s4)
    shirt=parts_readout(net,shirt,12,'fabric_hit',1.)
    # Scores with the same units: pool over scale, with some support from
    # the runner-up rather than adding all correlated scale-space maxima.
    def bank(name,parts):
        count=len(parts);eye=np.eye(count)
        spread=[net.conv(name+'_slot'+str(i),p,eye[:,i:i+1]) for i,p in enumerate(parts)]
        v=net.add(name+'_scale_vector',*spread)
        v=net.reshape(name+'_scale_axis',v,[1,count,1])
        peak=net.pool(name+'_scale_max',v,(count,1),kind='max')
        mean=net.pool(name+'_scale_mean',v,(count,1))
        return net.add(name+'_vote',net.conv(name+'_max_weight',peak,[[.80]]),
                       net.conv(name+'_mean_weight',mean,[[.20]]))
    face=bank('faces',faces); body=bank('figures',bodies)
    # Soft OR with capped negative evidence. An occluded body should not
    # cancel a strong face, and a back view need not show any facial detail.
    def floor(name,v,lower):
        r=net.relu(net.conv(name+'_hinge',v,[[1.]],-lower))
        return net.conv(name+'_limited',r,[[1.]],lower)
    face=floor('face',face,-.31); body=floor('body',body,-.26)
    shirt=floor('colour_clothes',shirt,-.20)
    f=net.conv('face_evidence',face,[[.85]])
    b=net.conv('body_evidence',body,[[.85]])
    d=net.relu(net.add('evidence_difference',f,net.conv('negative_body',b,[[-1.]])))
    best=net.add('strongest_detector',b,d)
    result=net.add('person_margin',net.dense('best_vote',best,[[.66]]),
                    net.dense('head_vote',face,[[.31]]),net.dense('silhouette_vote',body,[[.23]]),
                    net.dense('shirt_vote',shirt,[[.15]]),global_context(net,s2,rgb4,mid_edges))
    net.dense('class_scores',result,[[0.],[1.]],[0.,-.025])
    net.save(out)


if __name__=='__main__':
    build(Path(sys.argv[1]))
