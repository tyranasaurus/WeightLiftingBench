"""Data-free galaxy morphology network.

All filters below are analytic functions of coordinates, not observations.
There is no data loading, learning, parameter search, or random weight bank.
The arithmetic normalizers and spline multipliers implement fixed formulae.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

F32 = np.float32


class Graph:
    def __init__(self):
        self.layers = []
        self.weights = {}
        self.shapes = {'image': (3, 128, 128)}

    def layer(self, name, op, inputs, output_shape, **settings):
        self.layers.append(dict(name=name, op=op, inputs=list(inputs), **settings))
        self.shapes[name] = tuple(output_shape)
        return name

    def dense(self, name, inp, W, b=None):
        W = np.asarray(W, dtype=F32)
        assert W.ndim == 2 and W.shape[1] == math.prod(self.shapes[inp]), (name, W.shape, self.shapes[inp])
        self.weights[name + '.W'] = W
        self.weights[name + '.b'] = np.zeros(W.shape[0], F32) if b is None else np.asarray(b, F32)
        return self.layer(name, 'dense', [inp], (W.shape[0],), out=W.shape[0])

    def conv(self, name, inp, W, b=None, padding=0, stride=1):
        W = np.asarray(W, dtype=F32)
        o, c, k, _ = W.shape
        self.weights[name + '.W'] = W
        self.weights[name + '.b'] = np.zeros(o, F32) if b is None else np.asarray(b, F32)
        _, h, w = self.shapes[inp]
        return self.layer(name, 'conv', [inp], (o, (h + 2*padding-k)//stride+1, (w+2*padding-k)//stride+1), out=o, kernel=[k,k], stride=stride, padding=padding)

    def reshape(self, name, inp, shape):
        assert math.prod(shape) == math.prod(self.shapes[inp])
        return self.layer(name, 'reshape', [inp], shape, shape=list(shape))

    def relu(self, name, inp):
        return self.layer(name, 'relu', [inp], self.shapes[inp])

    def pool(self, name, inp, size, stride=None, kind='avgpool'):
        stride = size if stride is None else stride
        c,h,w = self.shapes[inp]
        return self.layer(name, kind, [inp], (c,(h-size[0])//stride[0]+1,(w-size[1])//stride[1]+1), size=list(size), stride=list(stride))

    def add(self, name, *inputs):
        return self.layer(name, 'add', inputs, self.shapes[inputs[0]])

    def concat(self, name, *inputs):
        dims = [math.prod(self.shapes[x]) for x in inputs]
        total = sum(dims)
        nodes=[]
        start=0
        for i,(inp,n) in enumerate(zip(inputs,dims)):
            w=np.zeros((total,n),F32)
            w[start:start+n]=np.eye(n,dtype=F32)
            nodes.append(self.dense(name+'_part'+str(i),inp,w))
            start+=n
        return self.add(name,*nodes)


def gaussian(k, sigma):
    p=np.arange(k,dtype=np.float64)-(k-1)/2
    xx,yy=np.meshgrid(p,p)
    w=np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma))
    return (w/w.sum()).astype(F32)


def grid(n, pixel):
    p=(np.arange(n,dtype=np.float64)-(n-1)/2)*pixel
    return np.meshgrid(p,p)


def clip_vector(g, name, inp, lower, upper):
    n=math.prod(g.shapes[inp])
    lo=np.broadcast_to(np.asarray(lower,F32),(n,))
    hi=np.broadcast_to(np.asarray(upper,F32),(n,))
    a=g.dense(name+'_hinges',inp,np.concatenate([np.eye(n,dtype=F32)]*2,axis=0),np.concatenate([-lo,-hi]))
    a=g.relu(name+'_positive',a)
    return g.dense(name,a,np.concatenate([np.eye(n,dtype=F32),-np.eye(n,dtype=F32)],axis=1),lo)


def ratio_bank(g,name,inp,index,requests):
    """Compute N/D using sign-paired instance normalization.

    Numerators are small compared with a deliberately scaled denominator.
    Their contribution to the norm is at most a few parts in 10^5.
    requests: (feature name, numerator coefficients, denominator coefficients).
    """
    groups={}
    for feature,num,den in requests:
        key=tuple(sorted(den.items()))
        groups.setdefault(key,[]).append((feature,num))
    width=2+2*max(len(v) for v in groups.values())
    channels=len(groups)
    nin=math.prod(g.shapes[inp])
    pack=np.zeros((channels*width,nin),F32)
    read=np.zeros((len(requests),channels*width),F32)
    outputs={feature:i for i,(feature,_,_) in enumerate(requests)}
    scale=1024.0
    for c,(den,nums) in enumerate(groups.items()):
        off=c*width
        for key,val in den:
            pack[off,index[key]]=scale*val
            pack[off+1,index[key]]=-scale*val
        for j,(feature,num) in enumerate(nums):
            for key,val in num.items():
                pack[off+2+2*j,index[key]]=val
                pack[off+3+2*j,index[key]]=-val
            read[outputs[feature],off+2+2*j]=scale*math.sqrt(2.0/width)
    node=g.dense(name+'_pack',inp,pack)
    node=g.reshape(name+'_groups',node,(channels,1,width))
    node=g.layer(name+'_normalize','norm',[node],g.shapes[node])
    return g.dense(name,node,read),outputs


def amplitude_bank(g,name,inp,index,pairs,angles=16):
    """Orientation-invariant length: a regular polygon approximation to L2."""
    w=np.zeros((len(pairs)*angles,math.prod(g.shapes[inp])),F32)
    for i,(_,xspec,yspec) in enumerate(pairs):
        for j in range(angles):
            theta=2*math.pi*j/angles
            for key,v in xspec.items(): w[i*angles+j,index[key]]+=v*math.cos(theta)
            for key,v in yspec.items(): w[i*angles+j,index[key]]+=v*math.sin(theta)
    node=g.dense(name+'_directions',inp,w)
    node=g.reshape(name+'_angles',node,(len(pairs),1,angles))
    node=g.pool(name+'_maximum',node,(1,angles),kind='maxpool')
    node=g.reshape(name,node,(len(pairs),))
    return node,{p[0]:i for i,p in enumerate(pairs)}


def main(folder):
    g=Graph()
    a_values=[3.0,4.5,6.75,10.125,15.1875,22.78125]
    ns=len(a_values)
    x,y=grid(64,2.0)
    r2=x*x+y*y
    x32,y32=grid(32,4.0)
    rr32=x32*x32+y32*y32
    focus=16.0
    win=lambda R: np.exp(-r2/(2*R*R)-r2/(2*focus*focus))
    win32=lambda R: np.exp(-rr32/(2*R*R)-rr32/(2*focus*focus))

    # Noise reduction at a known, fixed angular scale.
    down=g.pool('two_pixel_sampling','image',(2,2))
    blur=np.zeros((3,3,5,5),F32)
    for c in range(3): blur[c,c]=gaussian(5,.65)
    smooth=g.conv('seeing_smoothing',down,blur,padding=2)

    # Robust, clipped sky estimate in an annulus; it is computed per input.
    excess=g.conv('bright_sky_sources',smooth,np.eye(3,dtype=F32)[:,:,None,None],[-.075]*3)
    excess=g.relu('sky_source_positive',excess)
    neg=g.conv('negative_sky_sources',excess,-np.eye(3,dtype=F32)[:,:,None,None])
    capped=g.add('clipped_sky_image',smooth,neg)
    ring=((r2>42**2)&(r2<60**2)).astype(F32)
    ring/=ring.sum()
    w=np.zeros((3,3,64,64),F32)
    for c in range(3):w[c,c]=ring
    sky=g.dense('sky_level',capped,w.reshape(3,-1))
    w=np.zeros((3*64*64,3),F32)
    for c in range(3):w[c*4096:(c+1)*4096,c]=-1
    skyfield=g.dense('subtract_sky',sky,w,np.full(3*4096,-.008,F32))
    skyfield=g.reshape('sky_field',skyfield,(3,64,64))
    corrected=g.add('sky_subtracted_colour',smooth,skyfield)
    gray=g.conv('luminance',corrected,np.array([.25,.4,.35],F32).reshape(1,3,1,1))
    gray=g.relu('positive_galaxy_light',gray)
    comp_h=g.conv('tone_hinges',gray,np.ones((2,1,1,1),F32),[-.10,-.30])
    comp_h=g.relu('tone_positive',comp_h)
    comp_h=g.conv('tone_adjustment',comp_h,np.array([-.55,-.30],F32).reshape(1,2,1,1))
    comp=g.add('compressed_galaxy_light',gray,comp_h)
    # Relative isophotes discard positive sky noise before shape integration.
    # Their levels are fractions of a per-input central peak, not fitted data.
    central_window=np.exp(-r2/(2*4.0**2)).astype(F32).ravel()
    peakmask=np.diag(central_window)
    central_peak=g.dense('central_peak_window',gray,peakmask)
    central_peak=g.reshape('central_peak_map',central_peak,(1,64,64))
    central_peak=g.pool('central_surface_brightness',central_peak,(64,64),kind='maxpool')
    # A relative upper clip suppresses unresolved nuclei, while preserving
    # the faint disk: clip(I, .25*peak) - .04*peak - noise_floor.
    upper=g.dense('relative_nucleus_clip',central_peak,np.full((4096,1),-.25,F32))
    upper=g.reshape('nucleus_clip_field',upper,(1,64,64))
    upper=g.add('bright_nucleus_excess',gray,upper)
    upper=g.relu('positive_nucleus_excess',upper)
    upper=g.conv('negative_nucleus_excess',upper,-np.ones((1,1,1,1),F32))
    comp=g.add('nucleus_clipped_light',gray,upper)
    threshold=g.dense('relative_isophote',central_peak,np.full((4096,1),-.04,F32),np.full(4096,-.002,F32))
    threshold=g.reshape('isophote_field',threshold,(1,64,64))
    comp=g.add('isophote_excess',comp,threshold)
    comp=g.relu('galaxy_isophote',comp)
    comp32=g.pool('morphology_sampling',comp,(2,2))
    gray32=g.pool('structure_sampling',gray,(2,2))

    # Positive unsharp residual, with an excluded nucleus for CAS clumpiness.
    broad=g.conv('structure_lowpass',gray32,gaussian(7,1.0)[None,None],padding=3)
    negative=g.conv('negative_lowpass',broad,-np.ones((1,1,1,1),F32),[-.003])
    detail=g.add('unsharp_contrast',gray32,negative)
    detail=g.relu('positive_substructure',detail)

    # Broad peaks survive this bandpass better than unresolved foreground stars.
    k=11
    nucleus_kernel=1.2*gaussian(k,.60)-gaussian(k,1.75)
    nucleus_kernel[k//2,k//2]-=.2
    peaks=g.conv('extended_nucleus_filter',gray32,nucleus_kernel[None,None],[-.002],padding=k//2)
    peaks=g.relu('extended_nuclei',peaks)

    raw_names=[]
    comp_rows=[]; comp_names=[]
    gray_rows=[]; gray_names=[]
    colour_rows=[]; colour_names=[]
    detail_rows=[]; detail_names=[]
    def addrow(rows,names,name,arr):
        names.append(name);rows.append(np.asarray(arr,F32).ravel())
    for s,a in enumerate(a_values):
        pref=f's{s}_'
        for loc,R in [('in',.5*a),('mid',a),('out',2*a)]:
            ww=win(R)
            for label,arr in [('F',ww),('T',ww*r2),('U',ww*r2*r2),('Q',ww*(x*x-y*y)),('V',ww*2*x*y)]:
                addrow(comp_rows,comp_names,pref+loc+'_'+label,arr)
        for label,R in [('core',.35*a),('half',.7*a),('scale',a),('all',2*a),('wide',4*a)]:
            addrow(gray_rows,gray_names,pref+'light_'+label,win(R))
        for label,ww in [('inner',win(.7*a)),('outer',win(1.4*a)-win(.5*a))]:
            arr=np.zeros((3,64,64),F32);arr[0]=ww;arr[2]=-ww
            addrow(colour_rows,colour_names,pref+'colour_'+label+'_N',arr)
            # Sum of positive colour channels, never a signed near-zero sky
            # residual, is the flux denominator of this chromaticity.
            arr=np.broadcast_to(ww,(3,64,64)).copy()
            addrow(colour_rows,colour_names,pref+'colour_'+label+'_D',arr)
        ww=win32(2*a)*(1-win32(.5*a))
        addrow(detail_rows,detail_names,pref+'texture',4*ww)
    positive_colour=g.relu('positive_colour_flux',corrected)
    raw_nodes=[]
    for tag,node,rows,names in [('moments',comp,comp_rows,comp_names),('photometry',gray,gray_rows,gray_names),('clumpiness',detail,detail_rows,detail_names)]:
        raw_nodes.append(g.dense(tag,node,np.stack(rows)))
        raw_names+=names
    for tag,node,parity in [('chromatic_differences',corrected,0),('chromatic_fluxes',positive_colour,1)]:
        raw_nodes.append(g.dense(tag,node,np.stack(colour_rows[parity::2])))
        raw_names+=colour_names[parity::2]

    # Nine trial rotation centres account for subpixel catalogue centering.
    # These are fixed coordinate permutations, not stored image patches.
    shifts=[(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1]]
    diff=np.zeros((len(shifts)*1024,1024),F32)
    for t,(dy,dx) in enumerate(shifts):
        for iy in range(32):
            for ix in range(32):
                jy,jx=31-iy+dy,31-ix+dx
                if 0<=jy<32 and 0<=jx<32:
                    row=t*1024+iy*32+ix
                    diff[row,iy*32+ix]+=1
                    diff[row,jy*32+jx]-=1
    asym=g.dense('rotation_difference',comp32,diff)
    asym=g.relu('positive_rotation_difference',asym)
    asym_rows=np.zeros((ns*9,9*1024),F32)
    for s,a in enumerate(a_values):
        for t,(dy,dx) in enumerate(shifts):
            # The weight is symmetric around each trial centre.
            rad=(x32-2*dx)**2+(y32-2*dy)**2
            ww=np.exp(-rad/(2*(2*a)**2)-rad/(2*focus*focus))
            asym_rows[s*9+t,t*1024:(t+1)*1024]=-8*ww.ravel()
    asym=g.dense('negative_asymmetry_integrals',asym,asym_rows)
    asym=g.reshape('trial_centres',asym,(ns,1,9))
    asym=g.pool('best_rotation_centre',asym,(1,9),kind='maxpool')
    asym=g.dense('asymmetry_integrals',asym,-np.eye(ns,dtype=F32))
    raw_nodes.append(asym);raw_names += [f's{s}_asymmetry' for s in range(ns)]

    # Smooth inner and off-centre masks followed by maxima of extended cores.
    w=np.zeros((ns*2*1024,1024),F32)
    rad=np.sqrt(rr32)
    for s,a in enumerate(a_values):
        inner=np.exp(-rr32/(2*max(.50*a,2.0)**2))
        off=(1-np.exp(-(rad/max(.65*a,3.0))**6))*np.exp(-rr32/(2*(2.3*a)**2)-rr32/(2*focus*focus))
        for j,mask in enumerate([inner,off]):
            start=(s*2+j)*1024
            w[start+np.arange(1024),np.arange(1024)]=mask.ravel()
    peak_values=g.dense('nucleus_apertures',peaks,w)
    peak_values=g.reshape('nucleus_maps',peak_values,(ns*2,32,32))
    peak_values=g.pool('nucleus_maxima',peak_values,(32,32),kind='maxpool')
    peak_values=g.reshape('nucleus_values',peak_values,(ns*2,))
    raw_nodes.append(peak_values)
    raw_names += [f's{s}_peak_{kind}' for s in range(ns) for kind in ['central','outer']]
    raw_nodes.append(central_peak)
    raw_names.append('central_brightness')
    raw=g.concat('raw_descriptors',*raw_nodes)
    ri={name:i for i,name in enumerate(raw_names)}

    # The denominator of a weighted shear moment follows by integration by
    # parts: T - U/(4R^2). This removes radial-profile dependence to first order.
    # A conservative analytic seeing correction is limited to 65% of T.
    corrections=[]; cor_names=[]
    for s,a in enumerate(a_values):
        for loc,R in [('in',.5*a),('mid',a),('out',2*a)]:
            p=f's{s}_{loc}_'
            Reff=1/math.sqrt(1/(R*R)+1/(focus*focus))
            psf=6.0*(Reff*Reff/(Reff*Reff+6.0))**2
            row=np.zeros(len(raw_names),F32)
            row[ri[p+'T']]=-.65
            row[ri[p+'U']]=1/(4*Reff*Reff)
            row[ri[p+'F']]=psf
            corrections.append(row);cor_names.append(p+'correction')
    correction=g.dense('moment_denominator_limits',raw,np.stack(corrections))
    correction=g.relu('positive_denominator_limits',correction)
    raw=g.concat('stable_descriptors',raw,correction)
    raw_names+=cor_names
    ri={name:i for i,name in enumerate(raw_names)}
    requests=[]
    def ratio(name,num,den): requests.append((name,num,den))
    for s,a in enumerate(a_values):
        p=f's{s}_'
        F={p+'out_F':1}
        ratio(p+'size',{p+'mid_F':1},F)
        ratio(p+'concentration',{p+'light_core':1},{p+'light_all':1})
        ratio(p+'texture_ratio',{p+'texture':1},{p+'light_all':1})
        ratio(p+'asymmetry_ratio',{p+'asymmetry':1},F)
        ratio(p+'pair_ratio',{p+'peak_outer':1},{p+'peak_central':1,p+'peak_outer':.3,'central_brightness':.04})
        for kind in ['inner','outer']:
            ratio(p+'colour_'+kind,{p+'colour_'+kind+'_N':1},{p+'colour_'+kind+'_D':1})
        for loc,R in [('in',.5*a),('mid',a),('out',2*a)]:
            z=p+loc+'_'
            Reff=1/math.sqrt(1/(R*R)+1/(focus*focus))
            psf=6.0*(Reff*Reff/(Reff*Reff+6.0))**2
            den={z+'T':1,z+'U':-1/(4*Reff*Reff),z+'F':-psf,z+'correction':1}
            ratio(p+loc+'_e1',{z+'Q':1},den)
            ratio(p+loc+'_e2',{z+'V':1},den)
    basic,bi=ratio_bank(g,'normalized_descriptors',raw,ri,requests)
    lo=[];hi=[]
    for name,_,_ in requests:
        if name.endswith(('e1','e2')):lo.append(-1);hi.append(1)
        elif 'colour' in name:lo.append(-.5);hi.append(.5)
        elif 'pair_ratio' in name:lo.append(0);hi.append(2)
        elif 'asymmetry' in name:lo.append(0);hi.append(1.5)
        else:lo.append(0);hi.append(1)
    basic=clip_vector(g,'bounded_descriptors',basic,lo,hi)

    # Logarithmic two-arm filters. Angular means in mathematical elliptical
    # annuli are removed so a smooth radial ellipse has negligible response.
    geometries=[(1.,0.)]+[(q,t*math.pi/4) for q in [.75,.55] for t in range(4)]
    pitches=[('loose',k) for k in [-3.5,-2.,2.,3.5]]+[('tight',k) for k in [-9.,-6.,6.,9.]]
    hrows=[]; hnames=[]; hmeta=[]
    for s,a in enumerate(a_values):
        for geo,(q,theta) in enumerate(geometries):
            xp=x32*math.cos(theta)+y32*math.sin(theta)
            yp=(-x32*math.sin(theta)+y32*math.cos(theta))/q
            r=np.sqrt(xp*xp+yp*yp)/a
            angle=np.arctan2(yp,xp)
            window=np.exp(-r*r/(2*1.7**2))-np.exp(-r*r/(2*.5**2))
            bins=np.floor(r/.25).astype(int)
            for kind,kappa in pitches:
                phase=2*angle+kappa*np.log(np.maximum(r,.15))
                for ph in range(4):
                    field=np.cos(phase+ph*math.pi/2)
                    # Purely geometric quadrature correction, never image-based.
                    sums=np.bincount(bins.ravel(),weights=window.ravel())
                    sums_field=np.bincount(bins.ravel(),weights=(window*field).ravel(),minlength=len(sums))
                    mean=sums_field/np.maximum(sums,1e-30)
                    filt=window*(field-mean[bins])
                    hrows.append((8*filt).ravel().astype(F32))
                    hnames.append('h'+str(len(hnames)))
                    hmeta.append((s,geo,kind))
    harm=g.dense('analytic_spiral_correlations',comp32,np.stack(hrows))
    # Separate groups keep the large filter bank from inflating other ratios.
    frows=[]
    for s in range(ns):
        row=np.zeros(len(raw_names),F32);row[ri[f's{s}_out_F']]=1;frows.append(row)
    fnode=g.dense('spiral_flux_denominators',raw,np.stack(frows))
    hin=g.concat('spiral_numerators_and_flux',harm,fnode)
    hi_map={name:i for i,name in enumerate(hnames)}
    hi_map.update({'F'+str(s):len(hnames)+s for s in range(ns)})
    hrequests=[(name,{name:1},{'F'+str(meta[0]):1}) for name,meta in zip(hnames,hmeta)]
    harm,_=ratio_bank(g,'normalized_spiral_correlations',hin,hi_map,hrequests)

    # Geometric compatibility comes from the independent outer shape estimate.
    nw=ns*len(geometries)*4
    mismatch=np.zeros((nw,len(bi)),F32);bias=np.zeros(nw,F32)
    for s in range(ns):
        for j,(q,theta) in enumerate(geometries):
            e=(1-q*q)/(1+q*q)
            expected=[e*math.cos(2*theta),e*math.sin(2*theta)]
            for k in range(4):
                component=k//2;sgn=1 if k%2==0 else -1
                pos=(s*len(geometries)+j)*4+k
                mismatch[pos,bi[f's{s}_out_e{component+1}']]=sgn
                bias[pos]=-sgn*expected[component]
    mismatch=g.dense('spiral_geometry_mismatch',basic,mismatch,bias)
    mismatch=g.relu('positive_geometry_mismatch',mismatch)
    harmonics=g.concat('spiral_geometry_and_response',harm,mismatch)
    scorew=np.zeros((len(hnames),len(hnames)+nw),F32)
    order=[]
    for s in range(ns):
        for kind in ['tight','loose']:
            order.extend(i for i,meta in enumerate(hmeta) if meta[0]==s and meta[2]==kind)
    for row,old in enumerate(order):
        s,geo,_=hmeta[old]
        scorew[row,old]=1
        start=len(hnames)+(s*len(geometries)+geo)*4
        scorew[row,start:start+4]=-.20
    harm=g.dense('shape_compatible_spiral_scores',harmonics,scorew)
    width=len(hnames)//(ns*2)
    harm=g.reshape('pitch_candidates',harm,(ns*2,1,width))
    harm=g.pool('best_arm_templates',harm,(1,width),kind='maxpool')
    harm=g.reshape('arm_evidence',harm,(ns*2,))
    harm=clip_vector(g,'bounded_arm_evidence',harm,0,.5)

    pairs=[]
    for s in range(ns):
        p=f's{s}_'
        for loc in ['in','mid','out']:
            pairs.append((p+'E_'+loc,{p+loc+'_e1':1},{p+loc+'_e2':1}))
        for loc in ['in','mid']:
            pairs.append((p+'twist_'+loc,{p+loc+'_e1':1,p+'out_e1':-1},{p+loc+'_e2':1,p+'out_e2':-1}))
    amps,ai=amplitude_bank(g,'ellipticity_amplitudes',basic,bi,pairs)
    amps=clip_vector(g,'bounded_ellipticities',amps,0,1.5)
    combined=g.concat('per_scale_evidence',basic,amps,harm)
    ci=dict(bi)
    ci.update({name:len(bi)+i for name,i in ai.items()})
    ci.update({f's{s}_{kind}':len(bi)+len(ai)+s*2+j for s in range(ns) for j,kind in enumerate(['tight','loose'])})
    nc=math.prod(g.shapes[combined])

    # Scale posterior: differences of monotone soft radius decisions.
    u=np.zeros((ns-1,nc),F32)
    for s in range(ns-1):u[s,ci[f's{s}_size']]=2.5
    unode=g.dense('radius_cdf_affine',combined,u,np.full(ns-1,-1.125,F32))
    unode=clip_vector(g,'radius_cdf',unode,0,1)
    pw=np.zeros((ns,ns-1),F32);pb=np.zeros(ns,F32)
    pw[0,0]=1
    for s in range(1,ns-1):pw[s,s]=1;pw[s,s-1]=-1
    pw[-1,-1]=-1;pb[-1]=1
    pnode=g.dense('radius_mixture_weights',unode,pw,pb)
    pnode=g.relu('nonnegative_radius_weights',pnode)

    # Bar excess is the triangle residual between inner and outer shear.
    # A round central bulge reduces inner shear without producing bar excess.
    bw=np.zeros((ns*2,nc),F32)
    for s,a in enumerate(a_values):
        p=f's{s}_'
        for j,loc in enumerate(['in','mid']):
            reliability=a*a/(a*a+9) if loc=='in' else 1.0
            bw[s*2+j,ci[p+'E_'+loc]]=.5*reliability
            bw[s*2+j,ci[p+'twist_'+loc]]=.5*reliability
            bw[s*2+j,ci[p+'E_out']]=-.5*reliability
    bars=g.dense('radial_bar_excess',combined,bw)
    bars=g.reshape('bar_radii',bars,(ns,1,2))
    bars=g.pool('strongest_bar_radius',bars,(1,2),kind='maxpool')
    bars=g.reshape('bar_strengths',bars,(ns,))
    bars=clip_vector(g,'bounded_bar_strengths',bars,0,1)
    allscale=g.concat('adaptive_scale_inputs',combined,bars,pnode)
    ix=dict(ci)
    ix.update({f's{s}_bar':nc+s for s in range(ns)})
    ix.update({f's{s}_p':nc+ns+s for s in range(ns)})
    dim=nc+2*ns
    feature_names=['ellipticity','inner_shape','middle_shape','bar','bulge','outer_colour','inner_colour','concentration','asymmetry','texture','pair','tight','loose']
    specs=[]
    for s in range(ns):
        p=f's{s}_'
        source=[{p+'E_out':1},{p+'E_in':1},{p+'E_mid':1},{p+'bar':1},{p+'E_out':1,p+'E_mid':-1},{p+'colour_outer':1},{p+'colour_inner':1},{p+'concentration':1},{p+'asymmetry_ratio':1},{p+'texture_ratio':1},{p+'pair_ratio':1},{p+'tight':1},{p+'loose':1}]
        for j,spec in enumerate(source):specs.append((s,j,spec))

    # Formula-based, piecewise-linear multiplication for radius adaptation:
    # p*f = ((f+p)^2-(f-p)^2)/4, with a uniform spline for the square.
    # No spline value or knot is estimated from observations.
    B=3.0; segments=72; step=2*B/segments
    knots=np.linspace(-B,B,segments+1)[1:-1]
    nh=len(specs)*2*len(knots)+ns
    mw=np.zeros((nh,dim),F32);mb=np.zeros(nh,F32)
    ow=np.zeros((len(feature_names),nh),F32)
    row=0
    for s,j,spec in specs:
        for sign in [1,-1]:
            for knot in knots:
                for key,val in spec.items():mw[row,ix[key]]=val
                mw[row,ix[f's{s}_p']]=sign
                mb[row]=-knot
                ow[j,row]=sign*step/2
                row+=1
    for s in range(ns):
        mw[row,ix[f's{s}_p']]=1
        ow[:,row]=(-2*B+step)/2
        row+=1
    products=g.dense('radius_product_spline',allscale,mw,mb)
    products=g.relu('positive_radius_spline',products)
    adaptive=g.dense('scale_adapted_morphology',products,ow)
    fi={name:i for i,name in enumerate(feature_names)}

    # A second, independent inclination estimator compares the central light
    # to continuous ellipse models. The models are Gaussian mixtures with a
    # known seeing floor, a broad negative surround, and L2-normalized filters.
    # A soft maximum spans position angle, apparent size, and axis ratio.
    qclasses=[[1.0,.90],[.78,.64],[.49,.37],[.25,.15]]
    sizes=[2.5,3.5,5.,7.,10.,14.,20.]
    templates=[]
    aperture=np.exp(-r2/(2*14.0**2))
    for ratios in qclasses:
        for q in ratios:
            for size in sizes:
                for orient in range(12):
                    th=orient*math.pi/12
                    xp=x*math.cos(th)+y*math.sin(th)
                    yp=-x*math.sin(th)+y*math.cos(th)
                    sx=size*size+5.0
                    sy=(q*size)**2+5.0
                    profile=np.exp(-.5*(xp*xp/sx+yp*yp/sy))/math.sqrt(sx*sy)
                    profile+=.20*np.exp(-.5*(xp*xp/(4*sx)+yp*yp/(4*sy)))/math.sqrt(sx*sy)
                    surround=np.exp(-r2/(2*(max(12.,size*2))**2))
                    surround*=float((aperture*profile).sum())/float((aperture*surround).sum())
                    field=aperture*(profile-surround)
                    field/=math.sqrt(float((field*field).sum()))
                    templates.append(field.ravel().astype(F32))
    incl=g.dense('seeing_broadened_ellipse_templates',comp,np.stack(templates))
    incl=g.reshape('ellipse_hypotheses',incl,(4,1,len(templates)//4))
    incl=g.pool('best_ellipse_per_inclination',incl,(1,len(templates)//4),kind='maxpool')
    incl=g.reshape('inclination_template_scores',incl,(4,))
    w=np.zeros((8,4),F32)
    for j in range(4):w[2*j,j]=1;w[2*j+1,j]=-1
    incln=g.dense('inclination_sign_pairs',incl,w)
    incln=g.reshape('inclination_norm_group',incln,(1,1,8))
    incln=g.layer('inclination_norm','norm',[incln],g.shapes[incln])
    w=np.zeros((3,8),F32)
    # Pairwise logistic approximations measure crossing of inclination bins.
    for j in range(3):w[j,2*j]=-24;w[j,2*(j+1)]=24
    cdf=g.dense('inclination_crossings_affine',incln,w,[.5]*3)
    cdf=clip_vector(g,'inclination_crossings',cdf,0,1)
    w=np.array([[.5],[.5],[.5]],F32)
    olde=g.dense('moment_inclination_evidence',adaptive,np.eye(len(fi),dtype=F32)[[fi['ellipticity']]])
    # Smooth the template decision slightly with the independent moments.
    incl_features=g.concat('independent_inclinations',adaptive,cdf)
    w=np.zeros((len(fi),len(fi)+3),F32)
    w[:,:len(fi)]=np.eye(len(fi),dtype=F32)
    w[fi['ellipticity']]=0
    w[fi['ellipticity'],len(fi):]=[.22,.43,.26]
    w[fi['ellipticity'],fi['ellipticity']]=.08
    adaptive=g.dense('hybrid_morphology',incl_features,w)
    # Clamp tiny spline errors and keep descriptors in their natural ranges.
    lower=[0,0,0,0,0,-.5,-.5,0,0,0,0,0,0]
    upper=[1,1.5,1.5,1,1,.5,.5,1,1.5,1,2,.5,.5]
    adaptive=clip_vector(g,'physical_descriptors',adaptive,lower,upper)

    # Fixed fuzzy decision rules. Thresholds express generic CAS morphology,
    # colour ordering, inclination, and the contrast of bars and spiral arms.
    memberships=[]
    def membership(name,spec,bias):memberships.append((name,spec,bias))
    membership('red',{'outer_colour':6.25},.25)
    membership('texture',{'texture':12.5},-.1875)
    membership('bar',{'bar':3.0,'ellipticity':-.50},-.12)
    membership('disturbed',{'asymmetry':1/.45},-.22/.45)
    membership('pair',{'pair':1/.60},-.15/.60)
    membership('strong_pair',{'pair':2.5},-1.5)
    membership('bulge',{'bulge':2.0,'concentration':2.5,'inner_colour':1.5},-.55)
    # Independent tight and loose evidence, followed by their maximum below.
    membership('tight_arms',{'tight':10},-.3)
    membership('loose_arms',{'loose':10},-.3)
    w=np.zeros((len(memberships),len(fi)),F32);b=[]
    for i,(_,spec,bias) in enumerate(memberships):
        for key,val in spec.items():w[i,fi[key]]=val
        b.append(bias)
    member=g.dense('morphology_membership_affine',adaptive,w,b)
    member=clip_vector(g,'morphology_memberships',member,0,1)
    # Memberships must have distinct names from continuous descriptors.
    # In particular, overwriting 'bar' in a name->offset table would shift
    # every subsequent membership and read the wrong scalar.
    mi={('bar_membership' if name=='bar' else 'bulge_membership' if name=='bulge' else name):i for i,(name,_,_) in enumerate(memberships)}
    # max(tight,loose) = loose + ReLU(tight-loose).
    w=np.zeros((1,len(mi)),F32);w[0,mi['tight_arms']]=1;w[0,mi['loose_arms']]=-1
    arm_delta=g.dense('arm_max_difference',member,w)
    arm_delta=g.relu('positive_arm_difference',arm_delta)
    evidence=g.concat('decision_evidence',adaptive,member,arm_delta)
    ei=dict(fi);ei.update({k:len(fi)+v for k,v in mi.items()});ei['arm_delta']=len(fi)+len(mi)
    ne=math.prod(g.shapes[evidence])
    stage=[]
    stage.append(('disk',{'red':-.95,'loose_arms':.35,'arm_delta':.35,'texture':.20,'bar_membership':.20},.70))
    stage.append(('tightness',{'concentration':1.5,'inner_colour':1.5,'outer_colour':.75,'tight':1.2,'loose':-1.2,'texture':-.15,'asymmetry':-.15},.5-1.5*.14))
    stage.append(('edge_bulge',{'bulge_membership':1},0))
    w=np.zeros((len(stage),ne),F32);b=[]
    for i,(_,spec,bias) in enumerate(stage):
        for key,val in spec.items():w[i,ei[key]]=val
        b.append(bias)
    stage_node=g.dense('class_family_affine',evidence,w,b)
    stage_node=clip_vector(g,'class_family_memberships',stage_node,0,1)
    finalin=g.concat('class_rule_inputs',evidence,stage_node)
    ii=dict(ei);ii.update({k:ne+i for i,(k,_,_) in enumerate(stage)})
    # Inclination penalties are distances to geometric ellipticity intervals.
    hinges=[('round_upper',1,-.22),('middle_lower',-1,.22),('middle_upper',1,-.67),('cigar_lower',-1,.67),('cigar_upper',1,-.91),('spiral_upper',1,-.72),('edge_lower',-1,.82)]
    w=np.zeros((len(hinges),math.prod(g.shapes[finalin])),F32);b=[]
    for j,(_,v,bias) in enumerate(hinges):w[j,ii['ellipticity']]=v;b.append(bias)
    penalties=g.dense('inclination_distances',finalin,w,b)
    penalties=g.relu('positive_inclination_distances',penalties)
    finalin=g.concat('all_class_rule_terms',finalin,penalties)
    ni=ne+len(stage);ii.update({name:ni+j for j,(name,_,_) in enumerate(hinges)})
    w=np.zeros((10,math.prod(g.shapes[finalin])),F32);b=np.zeros(10,F32)
    rules=[
        ({'disturbed':2.0,'pair':-.5,'loose_arms':-.25,'arm_delta':-.25},.05),
        ({'disturbed':1.9,'pair':.75,'strong_pair':.4},-.45),
        ({'disk':-1.8,'round_upper':-4},1.8),
        ({'disk':-1.8,'middle_lower':-4,'middle_upper':-4},1.8),
        ({'disk':-1.8,'cigar_lower':-4,'cigar_upper':-4},1.75),
        ({'disk':1.8,'bar_membership':1.6,'spiral_upper':-5},-.45),
        ({'disk':1.8,'tightness':1.1,'bar_membership':-.8,'spiral_upper':-5},-.55),
        ({'disk':1.8,'tightness':-1.1,'bar_membership':-.8,'spiral_upper':-5},.55),
        ({'edge_lower':-6,'edge_bulge':-.8,'red':-.45},2.1),
        ({'edge_lower':-6,'edge_bulge':.8,'red':.15},1.3),
    ]
    for cls,(spec,bias) in enumerate(rules):
        for key,val in spec.items():w[cls,ii[key]]=val
        b[cls]=bias
    g.dense('galaxy_class_scores',finalin,w,b)
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':g.layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**g.weights)
    print('Layers:',len(g.layers))
    print('Weights:',sum(v.size for v in g.weights.values()))
    print('All coefficients come from analytic geometry and fixed rules.')


if __name__=='__main__':
    main(sys.argv[1])
