"""Data-free galaxy morphology circuit. All filters and rules are analytic.

No input files are opened. In particular, the build has no access to images,
labels, measured descriptors, trained parameters, or external models.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}
        self.shapes = {'image': (3, 128, 128)}

    def layer(self, name, op, inputs, output_shape, **settings):
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        self.shapes[name] = tuple(output_shape)
        return name

    def linear(self, name, terms, bias=None):
        # Sum arbitrary fixed linear maps. This also implements concatenation.
        outputs = []
        for j, (source, matrix) in enumerate(terms):
            matrix = np.asarray(matrix, dtype=np.float32)
            assert matrix.shape[1] == math.prod(self.shapes[source])
            n = name if len(terms) == 1 else name + '_term' + str(j)
            self.weights[n + '.W'] = matrix
            self.weights[n + '.b'] = (np.zeros(matrix.shape[0], np.float32)
                                      if bias is None or j else np.asarray(bias, np.float32))
            outputs.append(self.layer(n, 'dense', [source], (matrix.shape[0],), out=matrix.shape[0]))
        if len(outputs) > 1:
            return self.layer(name, 'add', outputs, self.shapes[outputs[0]])
        return outputs[0]

    def conv(self, name, source, w, b=None, padding=0, stride=1):
        w = np.asarray(w, np.float32)
        c, h, width = self.shapes[source]
        out, ci, k, _ = w.shape
        assert ci == c
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.zeros(out, np.float32) if b is None else np.asarray(b, np.float32)
        return self.layer(name, 'conv', [source], (out, (h+2*padding-k)//stride+1, (width+2*padding-k)//stride+1),
                          out=out, kernel=[k,k], stride=stride, padding=padding)

    def relu(self, name, source):
        return self.layer(name, 'relu', [source], self.shapes[source])

    def reshape(self, name, source, shape):
        assert math.prod(shape) == math.prod(self.shapes[source])
        return self.layer(name, 'reshape', [source], shape, shape=list(shape))

    def pool(self, name, source, size, stride, maximum=False):
        c,h,w = self.shapes[source]
        return self.layer(name, 'maxpool' if maximum else 'avgpool', [source],
                          (c,(h-size[0])//stride[0]+1,(w-size[1])//stride[1]+1),
                          size=list(size), stride=list(stride))


# Scalar algebra: named vectors, linear combinations, fixed hinge functions.
net = Network()

def scalar_map(name, source, source_names, expressions):
    """expressions: [(new name, {old name: coefficient}, constant), ...]."""
    index = {n:i for i,n in enumerate(source_names)}
    w = np.zeros((len(expressions), len(source_names)), np.float32)
    b = np.zeros(len(expressions), np.float32)
    names = []
    for j,(n,coefs,constant) in enumerate(expressions):
        names.append(n); b[j] = constant
        for old,c in coefs.items():
            w[j,index[old]] += c
    return net.linear(name, [(source,w)], b), names


def ratios(name, source, source_names, specifications):
    """A norm-based, bounded-accuracy divider, not a learned normalization.

    Each norm channel contains [u,-u,32*v,-32*v]. The extracted first
    component times 32/sqrt(2) is (u/v)/sqrt(1+(u/(32*v))**2).
    For the dimensionless morphology features here the error is very small.
    """
    index = {n:i for i,n in enumerate(source_names)}
    n = len(specifications)
    w = np.zeros((n*4,len(source_names)),np.float32)
    for i,(_,numerator,denominator) in enumerate(specifications):
        for key,c in numerator.items():
            w[4*i,index[key]] += c
            w[4*i+1,index[key]] -= c
        for key,c in denominator.items():
            w[4*i+2,index[key]] += 32*c
            w[4*i+3,index[key]] -= 32*c
    a = net.linear(name+'_pairs',[(source,w)])
    a = net.reshape(name+'_channels',a,(n,1,4))
    a = net.layer(name+'_norm','norm',[a],(n,1,4))
    w = np.zeros((n,4*n),np.float32)
    w[np.arange(n),4*np.arange(n)] = 32/math.sqrt(2)
    return net.linear(name,[(a,w)]), [s[0] for s in specifications]


def augment(name, source, names, new_source, new_names):
    n,m = len(names),len(new_names)
    a = np.zeros((n+m,n),np.float32);a[:n] = np.eye(n)
    b = np.zeros((n+m,m),np.float32);b[n:] = np.eye(m)
    return net.linear(name,[(source,a),(new_source,b)]), names+new_names


def hinges(name, source, names, expressions):
    a,nn = scalar_map(name+'_linear',source,names,expressions)
    return net.relu(name,a),nn


def keep_and_augment(name,source,names,new_source,new_names):
    return augment(name,source,names,new_source,new_names)


def main(out_dir):
    p = Path(out_dir)
    p.mkdir(parents=True,exist_ok=True)
    # Astronomical RGB luminance. Reject faint sky and attenuate bright nuclei
    # in the shape branch, but preserve them in the photometry branch.
    luma = np.array([.30,.45,.25])
    a = net.pool('pixels4','image',(4,4),(4,4))
    yy,xx = np.mgrid[:32,:32]
    x = (xx-15.5)*4;y=(yy-15.5)*4;r2=x*x+y*y
    # The sky level is an inference-time nuisance variable, not a constant
    # obtained from the dataset. A wide outer annulus supplies the sky pedestal.
    annulus = ((r2 > 44**2) & (r2 < 64**2)).astype(np.float32)
    annulus /= annulus.sum()
    sky_w = np.zeros((3,3,32,32),np.float32)
    for c in range(3): sky_w[c,c] = annulus
    sky = net.linear('outer_sky',[(a,sky_w.reshape(3,-1))])
    broadcast = np.zeros((4,32,32,3),np.float32)
    broadcast[0] = -luma;broadcast[1] = -luma
    broadcast[2,:,:,0] = -1;broadcast[3,:,:,2] = -1
    sky = net.linear('sky_broadcast',[(sky,broadcast.reshape(-1,3))])
    sky = net.reshape('sky_planes',sky,(4,32,32))
    w = np.zeros((4,3,1,1))
    w[0,:,0,0] = luma;w[1,:,0,0] = luma
    w[2,0,0,0] = 1;w[3,2,0,0] = 1
    a = net.conv('light_above_sky',a,w,[-.012,-.18,-.012,-.012])
    a = net.layer('sky_subtracted','add',[a,sky],(4,32,32))
    maps = net.relu('positive_light',a)
    def gaussian(s): return np.exp(-r2/(2*s*s))
    raw_names=[];raw_weights=[]
    def measurement(name,spatial,channels):
        a = np.zeros((4,32,32),np.float32)
        for i,c in channels.items(): a[i] = spatial*c
        raw_names.append(name);raw_weights.append(a.ravel())
    for s in [4,6,8,10,12,16,24]:
        measurement('flux'+str(s),gaussian(s),{0:1})
    for s in [6,10,16,24]:
        g = gaussian(s)/(s*s)
        measurement('q0_'+str(s),g*r2,{0:1,1:-.7})
        measurement('qc_'+str(s),g*(x*x-y*y),{0:1,1:-.7})
        measurement('qs_'+str(s),g*(2*x*y),{0:1,1:-.7})
    for s in [6,16]:
        measurement('color'+str(s),gaussian(s),{2:1,3:-1})
    for color,channel in [('red',2),('blue',3)]:
        for s in [4,6,10]:
            g=gaussian(s)/(s*s)
            measurement(color+'_q0_'+str(s),g*r2,{channel:1})
            measurement(color+'_qc_'+str(s),g*(x*x-y*y),{channel:1})
            measurement(color+'_qs_'+str(s),g*(2*x*y),{channel:1})
    raw = net.linear('photometry',[(maps,np.array(raw_weights))])

    # 180-degree rotational asymmetry, with a small fixed grid of reflection
    # centers to avoid interpreting subpixel centering errors as tidal damage.
    coords = [(iy,ix) for iy in range(8,24) for ix in range(8,24)]
    shifts = [(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1]]
    w = np.zeros((len(shifts)*len(coords),4*32*32),np.float32)
    for j,(dy,dx) in enumerate(shifts):
        for k,(iy,ix) in enumerate(coords):
            w[j*len(coords)+k,iy*32+ix] = 1
            w[j*len(coords)+k,(31-iy+dy)*32+31-ix+dx] -= 1
    asym = net.linear('opposite_pixel_difference',[(maps,w)],np.full(len(w),-.008))
    asym = net.relu('positive_asymmetry',asym)
    w = np.zeros((2*len(shifts),len(shifts)*len(coords)),np.float32)
    for z,s in enumerate([12,20]):
        for j,(dy,dx) in enumerate(shifts):
            for k,(iy,ix) in enumerate(coords):
                d2=((ix-15.5-dx/2)*4)**2+((iy-15.5-dy/2)*4)**2
                w[z*len(shifts)+j,j*len(coords)+k] = -2*np.exp(-d2/(2*s*s))
    asym = net.linear('negative_asymmetry_sums',[(asym,w)])
    asym = net.reshape('asymmetry_centers',asym,(2,1,len(shifts)))
    asym = net.pool('best_center',asym,(1,len(shifts)),(1,len(shifts)),True)
    asym = net.linear('asymmetry_sums',[(asym,-np.eye(2))])
    raw,raw_names = augment('measurements_with_asymmetry',raw,raw_names,asym,['asym12','asym20'])

    # Difference-of-Gaussian ridge/knot detectors at two physical scales.
    def dog(k,s1,s2):
        yy,xx=np.mgrid[:k,:k];r=(xx-k//2)**2+(yy-k//2)**2
        a=np.exp(-r/(2*s1*s1));a/=a.sum()
        b=np.exp(-r/(2*s2*s2));b/=b.sum()
        return a-b
    w=np.zeros((1,4,7,7));w[0,0]=dog(7,.65,1.5)
    mid=net.conv('coarse_knots',maps,w,[-.003],padding=3)
    mid=net.relu('coarse_positive_knots',mid)
    ring=np.clip((np.sqrt(r2)-6)/4,0,1)*gaussian(18)
    tm=net.linear('coarse_structure_sum',[(mid,ring.reshape(1,-1))])
    raw,raw_names=augment('with_coarse_structure',raw,raw_names,tm,['texture_mid'])
    # A second bright compact source, separated from the central nucleus.
    coords=[(iy,ix) for iy in range(9,23) for ix in range(9,23)]
    w=np.zeros((len(coords),32*32),np.float32)
    for j,(iy,ix) in enumerate(coords):
        rr=math.sqrt(float(r2[iy,ix]))
        w[j,iy*32+ix]=min(1,max(0,(rr-4)/4))*math.exp(-rr*rr/(2*16*16))
    pk=net.linear('off_nuclear_knots',[(mid,w)])
    pk=net.reshape('off_nuclear_grid',pk,(1,14,14))
    pk=net.pool('off_nuclear_peak',pk,(14,14),(14,14),True)
    raw,raw_names=augment('with_secondary_nucleus',raw,raw_names,pk,['secondary_peak'])

    a=net.pool('pixels2','image',(2,2),(2,2))
    w=luma.reshape(1,3,1,1)*dog(7,.65,1.5).reshape(1,1,7,7)
    fine=net.conv('fine_knots',a,w,[-.006],padding=3)
    fine=net.relu('fine_positive_knots',fine)
    fy,fx=np.mgrid[:64,:64];fr2=((fx-31.5)*2)**2+((fy-31.5)*2)**2
    ring=np.clip((np.sqrt(fr2)-5)/4,0,1)*np.exp(-fr2/(2*18*18))/4
    tf=net.linear('fine_structure_sum',[(fine,ring.reshape(1,-1))])
    raw,raw_names=augment('all_measurements',raw,raw_names,tf,['texture_fine'])

    # A bank of generative galaxy models, sampled from equations, not examples.
    # L2-normalized positive templates implement unknown-amplitude matched
    # filters. The only choices are generic axis ratios, scales, and profiles.
    w=np.zeros((1,4,1,1));w[0,0,0,0]=1;w[0,1,0,0]=-.7
    model_light=net.conv('model_light',maps,w)
    groups={k:[] for k in ['round','middle','cigar','edge','bar','tight','loose','thin_disk','bulged_disk']}
    def unit_flux(a):
        return a/max(float(a.sum()),1e-15)
    def model_coords(scale,q,angle):
        ca,sa=math.cos(angle),math.sin(angle)
        u=x*ca+y*sa;v=-x*sa+y*ca
        # Approximate finite pixels and the seeing by quadrature broadening.
        u=u/math.sqrt(scale*scale+1.4**2)
        v=v/math.sqrt((scale*q)**2+1.4**2)
        return u,v,np.sqrt(u*u+v*v)
    def append_template(group,t):
        t=t.astype(np.float32)
        t=t/np.sqrt(np.sum(t*t))
        groups[group].append(t.ravel())
    scales=[2.,3.,4.5,6.5,9.,13.]
    qs=[.14,.23,.34,.48,.62,.76,.88,1.]
    for q in qs:
        group='edge' if q<.3 else 'cigar' if q<.45 else 'middle' if q<.83 else 'round'
        for scale in scales:
            for j in range(12 if q<1 else 1):
                angle=j*math.pi/12
                u,v,rho=model_coords(scale,q,angle)
                disk=unit_flux(np.exp(-rho))
                gauss=unit_flux(np.exp(-.5*rho*rho))
                cusp=unit_flux(np.exp(-2*np.sqrt(rho+.04)))
                bulge=unit_flux(np.exp(-np.sqrt(r2+1)/(max(scale*.28,1.5))))
                for t in [disk,gauss,cusp,.75*disk+.25*bulge]:
                    append_template(group,t)
                if q>=.45:
                    append_template(group,.5*disk+.5*bulge)
                if q<.45 or scale<3 or scale>9:
                    continue
                # Elliptical disk plus a physically straight bar whose angle
                # need not agree with the disk's projected major axis.
                for rel in [0,math.pi/4,math.pi/2,3*math.pi/4]:
                    for length in [.65,1.1]:
                        _,_,br=model_coords(scale*length,.18,angle+rel)
                        bar=unit_flux(np.exp(-.5*br*br))
                        t=.60*disk+.30*bar+.10*bulge
                        append_template('bar',t)
                # A logarithmic two-armed disk. Both winding senses and four
                # arm phases are present; north has no privileged direction.
                theta=np.arctan2(v,u)
                envelope=1-np.exp(-rho*rho/.5)
                for pitch,group_name in [(12,'tight'),(30,'loose')]:
                    for hand in [-1,1]:
                        phase=2*theta-hand*2/math.tan(math.radians(pitch))*np.log(rho+.15)
                        for phase0 in [0,math.pi/2,math.pi,3*math.pi/2]:
                            arms=disk*(1+.65*envelope*np.cos(phase+phase0))
                            append_template(group_name,.85*unit_flux(arms)+.15*bulge)
    # Separate projected edge-on models. Bulgeless late-type disks have
    # long exponential wings and weak central concentration; a round compact
    # bulge is added only in the bulged-disk family.
    for q in [.10,.16,.23,.30,.38]:
        for scale in [2.,3.,4.5,6.5,9.,13.]:
            for j in range(16):
                angle=j*math.pi/16
                _,_,rho=model_coords(scale,q,angle)
                disk=unit_flux(np.exp(-rho))
                _,_,gr=model_coords(scale*1.35,q,angle)
                gaussian_disk=unit_flux(np.exp(-.5*gr*gr))
                bulge=unit_flux(np.exp(-np.sqrt(r2+1)/max(scale*.3,1.5)))
                append_template('thin_disk',disk)
                append_template('thin_disk',gaussian_disk)
                for fraction in [.18,.35,.55]:
                    append_template('bulged_disk',(1-fraction)*disk+fraction*bulge)
    template_names=[];template_source=None
    for group,templates in groups.items():
        bank=net.linear('models_'+group,[(model_light,np.array(templates))])
        bank=net.reshape('models_'+group+'_grid',bank,(1,1,len(templates)))
        bank=net.pool('best_'+group+'_model',bank,(1,len(templates)),(1,len(templates)),True)
        if template_source is None:
            template_source=bank;template_names=[group]
        else:
            template_source,template_names=augment('models_through_'+group,template_source,template_names,bank,[group])
    w=np.zeros((4,len(template_names)),np.float32);w[:,:4]=np.eye(4)
    best=net.linear('plain_shape_models',[(template_source,w)])
    best=net.reshape('plain_shape_grid',best,(1,1,4))
    best=net.pool('best_plain_model',best,(1,4),(1,4),True)
    template_source,template_names=augment('model_correlations',template_source,template_names,best,['plain'])
    rel,rln=ratios('relative_model_correlations',template_source,template_names,
                   [('model_'+k,{k:1},{'plain':1}) for k in groups]+
                   [('model_bar_adv',{'bar':1,'tight':-.5,'loose':-.5},{'plain':1}),
                    ('model_pitch',{'tight':1,'loose':-1},{'plain':1}),
                    ('model_bulge_adv',{'bulged_disk':1,'thin_disk':-1},{'plain':1})])

    specifications=[]
    for s in [6,10,16,24]:
        for part in ['c','s']:
            specifications.append(('e'+part+str(s),{'q'+part+'_'+str(s):1},{'q0_'+str(s):1}))
    for color in ['red','blue']:
        for s in [4,6,10]:
            for part in ['c','s']:
                specifications.append((color+'_e'+part+str(s),
                                       {color+'_q'+part+'_'+str(s):1},
                                       {color+'_q0_'+str(s):1}))
    specifications += [
        ('red6',{'color6':1},{'flux6':1}),
        ('red16',{'color16':1},{'flux16':1}),
        ('a48',{'flux4':1},{'flux8':1}),
        ('a816',{'flux8':1},{'flux16':1}),
        ('asym12',{'asym12':1},{'flux12':1}),
        ('asym20',{'asym20':1},{'flux24':1}),
        ('texture_mid',{'texture_mid':1},{'flux16':1}),
        ('texture_fine',{'texture_fine':1},{'flux16':1}),
        ('secondary',{'secondary_peak':2*math.pi},{'flux4':1}),
    ]
    feat,fn=ratios('dimensionless_measurements',raw,raw_names,specifications)
    feat,fn=augment('features_with_model_evidence',feat,fn,rel,rln)

    # Rotation-invariant quadrupole lengths, approximating the Euclidean norm
    # with a regular sixteen-sided support polygon. No preferred sky angle.
    expressions=[]
    for s in [6,10,16,24]:
        for k in range(16):
            theta=2*math.pi*k/16
            expressions.append(('dir'+str(s)+'_'+str(k),
                                {'ec'+str(s):math.cos(theta),'es'+str(s):math.sin(theta)},0))
    for k in range(16):
        theta=2*math.pi*k/16
        expressions.append(('twistdir'+str(k),{'ec10':math.cos(theta),'es10':math.sin(theta),
                                              'ec24':math.cos(theta),'es24':math.sin(theta)},0))
    magnitude_names=['e6','e10','e16','e24','esum']
    for color in ['red','blue']:
        for s in [4,6,10]:
            magnitude_names.append(color+'_e'+str(s))
            for k in range(16):
                theta=2*math.pi*k/16
                expressions.append((color+'_dir'+str(s)+'_'+str(k),
                                    {color+'_ec'+str(s):math.cos(theta),
                                     color+'_es'+str(s):math.sin(theta)},0))
    mags,mn=scalar_map('quadrupole_directions',feat,fn,expressions)
    mags=net.reshape('quadrupole_polygon',mags,(len(magnitude_names),1,16))
    mags=net.pool('quadrupole_magnitudes',mags,(1,16),(1,16),True)
    feat,fn=augment('features_with_shapes',feat,fn,mags,magnitude_names)

    # Fixed piecewise-linear Gaussian concentration reference:
    # for a circular Gaussian, F4/F8 = 1 / (5 - 4 F8/F16).
    knots=np.linspace(.25,1,17)
    expressions=[('conc_h'+str(i),{'a816':1},-float(t)) for i,t in enumerate(knots)]
    hs,hn=hinges('concentration_hinges',feat,fn,expressions)
    values=1/(5-4*knots);slopes=np.diff(values)/np.diff(knots)
    coefs={'conc_h0':float(slopes[0])}
    for i in range(1,len(slopes)): coefs['conc_h'+str(i)]=float(slopes[i]-slopes[i-1])
    ref,rn=scalar_map('gaussian_concentration',hs,hn,[('conc_reference',coefs,float(values[0]))])
    feat,fn=augment('features_with_reference',feat,fn,ref,rn)

    # All thresholds below are hand-chosen, dimensionless morphology rules.
    # Hinge pairs produce clipped fuzzy evidence in [0,1].
    expr=[
        ('red_lo',{'red16':1/.35},0),('red_hi',{'red16':1/.35},-1),
        ('struct_lo',{'texture_mid':12},-.36),('struct_hi',{'texture_mid':12},-1.36),
        ('fine_lo',{'texture_fine':12},-.24),('fine_hi',{'texture_fine':12},-1.24),
        ('asym_lo',{'asym12':2.3,'asym20':1.0},-.33),
        ('asym_hi',{'asym12':2.3,'asym20':1.0},-1.33),
        ('bulge_lo',{'a48':6,'conc_reference':-6},-.12),
        ('bulge_hi',{'a48':6,'conc_reference':-6},-1.12),
        ('bar6',{'red_e4':1,'blue_e10':-.7},-.07),
        ('bar10',{'red_e6':1,'blue_e10':-.8},-.04),
        ('secondary_lo',{'secondary':2},-.2),('secondary_hi',{'secondary':2},-1.2),
        ('colorgrad_lo',{'red6':2,'red16':-2},0),
        ('colorgrad_hi',{'red6':2,'red16':-2},-1),
        ('twist_lo',{'e10':4,'e24':4,'esum':-4},0),
        ('twist_hi',{'e10':4,'e24':4,'esum':-4},-1),
        ('edge_lo',{'e16':3.75,'e24':1.25},-2.6),
        ('edge_hi',{'e16':3.75,'e24':1.25},-3.6),
    ]
    hs,hn=hinges('morphology_hinges',feat,fn,expr)
    feat,fn=augment('features_and_evidence_hinges',feat,fn,hs,hn)
    expr=[
        ('model_round',{'model_round':1},0),
        ('model_middle',{'model_middle':1},0),
        ('model_cigar',{'model_cigar':1},0),
        ('model_edge',{'model_edge':1},0),
        ('model_bar',{'model_bar':1},0),
        ('model_tight',{'model_tight':1},0),
        ('model_loose',{'model_loose':1},0),
        ('model_thin_disk',{'model_thin_disk':1},0),
        ('model_bulged_disk',{'model_bulged_disk':1},0),
        ('model_bar_adv',{'model_bar_adv':1},0),
        ('model_pitch',{'model_pitch':1},0),
        ('model_bulge_adv',{'model_bulge_adv':1},0),
        ('E',{'e16':.75,'e24':.25},0),
        ('red',{'red_lo':1,'red_hi':-1},0),
        ('structure',{'struct_lo':.75,'struct_hi':-.75,'fine_lo':.25,'fine_hi':-.25},0),
        ('asym',{'asym_lo':1,'asym_hi':-1},0),
        ('bulge',{'bulge_lo':1,'bulge_hi':-1},0),
        ('bar6',{'bar6':1},0),('bar10',{'bar10':1},0),
        ('secondary',{'secondary_lo':1,'secondary_hi':-1},0),
        ('colorgrad',{'colorgrad_lo':1,'colorgrad_hi':-1},0),
        ('twist',{'twist_lo':1,'twist_hi':-1},0),
        ('edge',{'edge_lo':1,'edge_hi':-1},0),
        ('size',{'a816':1},0),
    ]
    ev,en=scalar_map('morphology_evidence',feat,fn,expr)
    # Maximum of the two nested bar scales.
    dif,dn=hinges('bar_scale_comparison',ev,en,[('bardiff',{'bar6':1,'bar10':-1},0)])
    ev,en=augment('evidence_with_bar_difference',ev,en,dif,dn)
    bh,bn=hinges('bar_clipping',ev,en,[('bar_lo',{'bardiff':5,'bar10':5},0),
                                                    ('bar_hi',{'bardiff':5,'bar10':5},-1)])
    ev,en=augment('evidence_with_bar_hinges',ev,en,bh,bn)
    bv,bvn=scalar_map('bar_evidence',ev,en,[('bar',{'bar_lo':1,'bar_hi':-1},0)])
    ev,en=augment('complete_evidence',ev,en,bv,bvn)

    # Family likelihoods and soft, disjoint shape intervals.
    expr=[
        ('smooth',{'red':1.45,'bulge':.5,'structure':-1.1,'asym':-.6},0),
        ('disk',{'red':-.7,'structure':.6,'colorgrad':.3,'asym':-.5},.60),
        ('tightness',{'size':1.6,'bulge':.6,'structure':-.3},-.65),
        ('round_penalty',{'E':5},-.8),
        ('middle_low',{'E':-5},.8),('middle_high',{'E':5},-2.5),
        ('cigar_low',{'E':-5},2.5),('cigar_high',{'E':8},-5.6),
        ('spiral_inclination',{'E':5},-3.0),
        ('thin_edge',{'E':-8},5.6),('bulged_edge',{'E':-8},4.8),
    ]
    # Only the penalties, not family likelihoods, should be rectified.
    family,fan=scalar_map('family_scores',ev,en,expr[:3])
    penalties,pn=hinges('shape_penalties',ev,en,expr[3:])
    ev,en=augment('evidence_with_families',ev,en,family,fan)
    ev,en=augment('evidence_with_shape_penalties',ev,en,penalties,pn)
    rules=[
        ('disturbed',{'asym':1.1,'red':-.35,'structure':.15,'secondary':-.5,'edge':-.4,'twist':.15},.05),
        ('merging',{'asym':1.1,'secondary':.65,'red':.15,'edge':-.2},-.30),
        ('round_smooth',{'smooth':1,'model_round':35,'round_penalty':-.15},-35.1),
        ('intermediate_smooth',{'smooth':1,'model_middle':35,'middle_low':-.15,'middle_high':-.15},-35.03),
        ('cigar_smooth',{'smooth':1,'model_cigar':35,'cigar_low':-.1,'cigar_high':-.15},-34.96),
        ('barred_spiral',{'disk':1,'model_bar_adv':2,'spiral_inclination':-.5,'bar':1},-.16),
        ('tight_spiral',{'disk':1,'model_pitch':20,'spiral_inclination':-.5,'bar':-.3,'tightness':.3},0),
        ('loose_spiral',{'disk':1,'model_pitch':-20,'spiral_inclination':-.5,'bar':-.3,'tightness':-.3},0),
        ('edge_no_bulge',{'model_edge':35,'thin_edge':-.1,'red':-.7,'bulge':-.5,'asym':-.3},-33.8),
        ('edge_with_bulge',{'model_bulged_disk':35,'bulged_edge':-.15,'red':.5,'bulge':.5,'asym':-.3},-34.6),
    ]
    scalar_map('class_scores',ev,en,rules)
    (p/'network.json').write_text(json.dumps({'layers':net.layers},separators=(',',':')))
    np.savez(p/'weights.npz',**net.weights)
    print('Built',len(net.layers),'layers;',sum(v.size for v in net.weights.values()),'fixed weights.')


if __name__ == '__main__':
    main(sys.argv[1])
