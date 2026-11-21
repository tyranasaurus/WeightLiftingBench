"""Data-free handwritten-digit network, using typed parametric pen paths.

No files are opened for reading. All geometry is elementary, hand specified,
and independent of any examples. The only random numbers use a fixed seed.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

N_PER_CLASS = 2500
PARTS = 4
SHIFTS = [(0.,0.),(-.85,0.),(.85,0.),(0.,-.85),(0.,.85)]
SIDE = 28
SEED = 1729


def bezier(a, b, c, d, n=22):
    t = np.linspace(0.0, 1.0, n)[:, None]
    a, b, c, d = map(np.asarray, (a, b, c, d))
    return (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d


def line(a, b, n=12):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1-t)*np.asarray(a) + t*np.asarray(b)


def ellipse(rx, ry, cy, n=100, phase=0.0):
    t = np.linspace(phase, phase+2*math.pi, n)
    return np.column_stack((rx*np.cos(t), cy+ry*np.sin(t)))


def strokes(digit, rng):
    """Coordinates are generic writing-box coordinates, never image measurements."""
    u = rng.uniform
    waist = u(-0.20, 0.18)
    style = u()
    if digit == 0:
        t = np.linspace(0, 2*math.pi, 120)
        # An oval, sometimes egg shaped or pointed, with no interior stroke.
        egg = u(-0.18, 0.18)
        power = u(.8,1.5)
        x = np.sign(np.cos(t))*np.abs(np.cos(t))**power
        p = np.column_stack(((1+egg*np.sin(t))*x, np.sin(t)))
        return [p]
    if digit == 1:
        stem = bezier((u(-.12,.12), -1), (u(-.12,.12), -.35),
                      (u(-.12,.12), .35), (u(-.12,.12), 1))
        out = [stem]
        if style < .62:
            out.append(line((u(-.8,-.25), u(-.72,-.35)), tuple(stem[0])))
        if style < .15:
            out.append(line((u(-.8,-.4), 1), (u(.35,.8), 1)))
        return out
    if digit == 2:
        start_y = u(-.8,-.42)
        a = bezier((-.9,start_y), (-.5,-1.35), (1.1,-1.25), (.85,-.5))
        if style < .34:
            # A cursive two turns counterclockwise through a small lower loop.
            b = bezier(tuple(a[-1]), (.65,.16), (-1.3,.55), (-.8,.94))
            c = bezier(tuple(b[-1]), (-.3,1.38), (.25,.25), (-.05,.20))
            d = bezier(tuple(c[-1]), (.3,.5), (.65,.85), (1,.85))
            return [np.concatenate((a,b,c,d))]
        b = bezier(tuple(a[-1]), (.85,u(-.18,.0)), (u(-.65,-.3),.42), (-.9,.94))
        c = bezier(tuple(b[-1]), (u(-.6,-.2),u(.65,1.05)),
                   (u(.35,.7),u(.9,1.16)), (1,u(.75,1.05)))
        return [np.concatenate((a,b,c))]
    if digit == 3:
        if style < .23:
            a = line((-.9,-.95),(.85,-.95))
            b = line(tuple(a[-1]),(u(-.3,.1),waist))
        else:
            a = bezier((-.8,u(-.9,-.6)), (.15,-1.4), (1.55,-.8), (.65,-.25))
            b = bezier(tuple(a[-1]), (.4,waist), (.05,waist), (u(-.4,-.05),waist))
        c = bezier(tuple(b[-1]), (1.45,waist-.2), (1.45,1.4), (-.8,u(.65,.95)))
        return [np.concatenate((a,b,c))]
    if digit == 4:
        right = u(.35,.75)
        cross = u(-.03,.33)
        top_left = right if style < .32 else u(-.55,-.1)
        a = line((top_left,-1),(-.95,cross))
        b = line((-.95,cross),(u(.8,1.05),cross+u(-.12,.08)))
        c = bezier((right,-1),(right+u(-.12,.12),-.35),
                   (right+u(-.15,.12),.4),(right+u(-.2,.08),1))
        return [np.concatenate((a,b)),c]
    if digit == 5:
        a = bezier((.95,u(-1.05,-.9)), (.3,u(-1.05,-.9)),
                   (-.3,-.95), (-.75,-.95))
        middle_x = u(-.9,-.2)
        b = line(tuple(a[-1]),(middle_x,waist))
        if style < .60:
            # Many handwritten fives have a compact upper stem and an S-like bowl.
            c = bezier(tuple(b[-1]), (-.4,waist-.25), (1.6,.35), (.65,.72))
            d = bezier(tuple(c[-1]), (.15,1.05), (-.65,1.2), (-.95,.98))
        else:
            c = bezier(tuple(b[-1]), (1.1,waist-.65), (1.5,.95), (.0,1))
            d = bezier(tuple(c[-1]), (-.4,1.05), (-.7,.92), (-.9,u(.6,.9)))
        return [np.concatenate((a,b,c,d))]
    if digit == 6:
        ry = u(.52,.7)
        cy = 1-ry
        loop = ellipse(.87,ry,cy,phase=math.pi)
        a = bezier((u(.35,.95),-1), (u(-.65,-.25),u(-1.1,-.8)),
                   (-1.05,cy-.5), tuple(loop[0]))
        return [np.concatenate((a,loop))]
    if digit == 7:
        a = bezier((-.95,-.94),(-.4,u(-1.04,-.84)),(.35,u(-1.04,-.84)),(.95,-1))
        b = bezier(tuple(a[-1]), (u(.3,.9),-.4), (u(-.6,.1),.45), (u(-.65,-.2),1))
        out = [np.concatenate((a,b))]
        if style < .2:
            out.append(line((-.65,waist),(.55,waist-.08)))
        if .2 <= style < .3:
            out.append(line((-.95,-.6),(-.95,-.94)))
        return out
    if digit == 8:
        t = np.linspace(0,2*math.pi,150)
        waist_y = u(-.15,.12)
        bottom_heavy = u(.0,.25)
        y = -np.cos(t)
        y = np.where(y < 0, waist_y+(1+waist_y)*y, waist_y+(1-waist_y)*y)
        x = np.sin(2*t)*(1-bottom_heavy*np.cos(t))
        x += u(-.13,.13)*(1-y*y)
        return [np.column_stack((x,y))]
    if digit == 9:
        ry = u(.48,.67)
        cy = -1+ry
        loop = ellipse(.87,ry,cy)
        tail = bezier(tuple(loop[-1]), (.88,cy+.4),
                      (u(.15,.8),.8), (u(-.5,.55),1))
        return [np.concatenate((loop,tail))]
    raise ValueError(digit)


def resample(paths, step=.43):
    result = []
    for p in paths:
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        n = max(2,int(math.ceil(s[-1]/step)))
        q = np.linspace(0,s[-1],n)
        result.append(np.column_stack((np.interp(q,s,p[:,0]),np.interp(q,s,p[:,1]))))
    return np.concatenate(result)


def pen_points(digit, rng):
    paths = strokes(digit,rng)
    # A 20-pixel ink box has an approximately 18-pixel centre-line box.
    h = rng.uniform(17.0,19.0)
    ratio = rng.uniform(.40,.80)
    if digit == 1:
        ratio = .68
    shear = rng.uniform(-.30,.43)
    bend = rng.uniform(-.10,.10)
    transformed = []
    for p in paths:
        p = p.copy()
        p[:,0] = p[:,0]*h*ratio/2 - shear*p[:,1]*h/2 + bend*(p[:,1]**2)*h/2
        p[:,1] *= h/2
        transformed.append(p)
    allp = np.concatenate(transformed)
    # Scaling a wide or very slanted drawing preserves the given 20x20 rule.
    extent = np.ptp(allp,axis=0).max()
    if extent > 19:
        transformed = [p*(19/extent) for p in transformed]
    p = resample(transformed)
    p -= p.mean(axis=0)
    p += 13.5
    # Tiny residual translations represent rasterization/centering uncertainty.
    p += rng.uniform(-.28,.28,size=(1,2))
    return p.astype(np.float32)


def fields(p):
    y,x = np.mgrid[:28,:28].astype(np.float32)
    d2 = ((x[:,:,None]-p[None,None,:,0])**2 +
          (y[:,:,None]-p[None,None,:,1])**2).min(axis=2)
    precision = np.exp(-d2/(2*1.85**2))
    synthetic_ink = np.exp(-d2/(2*1.05**2))
    # Each horizontal part gets its own local deformation tolerance. Splitting
    # at fixed writing-box positions is a general shape prior, not a data rule.
    recall = np.zeros((PARTS,28,28),np.float32)
    ix = np.floor(p[:,0]).astype(int)
    iy = np.floor(p[:,1]).astype(int)
    part = np.clip(((p[:,1]-3.5)/20*PARTS).astype(int),0,PARTS-1)
    fx,fy = p[:,0]-ix,p[:,1]-iy
    for dx,dy,w in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                    (0,1,(1-fx)*fy),(1,1,fx*fy)]:
        xx,yy = ix+dx,iy+dy
        ok = (xx>=0)&(xx<28)&(yy>=0)&(yy<28)
        np.add.at(recall,(part[ok],yy[ok],xx[ok]),w[ok])
    recall /= recall.sum()
    return precision,recall,synthetic_ink


def translate(a,dy,dx):
    """Bilinear zero-padded translation of purely analytic weight fields."""
    y,x=np.mgrid[:28,:28]
    sy,sx=y-dy,x-dx
    iy,ix=np.floor(sy).astype(int),np.floor(sx).astype(int)
    fy,fx=sy-iy,sx-ix
    out=np.zeros_like(a)
    for yy,xx,w in [(iy,ix,(1-fy)*(1-fx)),(iy,ix+1,(1-fy)*fx),
                    (iy+1,ix,fy*(1-fx)),(iy+1,ix+1,fy*fx)]:
        ok=(yy>=0)&(yy<28)&(xx>=0)&(xx<28)
        out[ok]+=a[yy[ok],xx[ok]]*w[ok]
    return out


def build(destination):
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    layers,weights = [],{}
    def op(name,kind,source,**kw):
        layers.append(dict(name=name,op=kind,inputs=[source] if isinstance(source,str) else source,**kw))
        return name
    def affine(name,source,w,b,conv=False,padding=0):
        w,b = np.asarray(w,np.float32),np.asarray(b,np.float32)
        if conv:
            op(name,'conv',source,out=len(b),kernel=[w.shape[2],w.shape[3]],stride=1,padding=padding)
        else:
            op(name,'dense',source,out=len(b))
        weights[name+'.W'],weights[name+'.b']=w,b
        return name

    # A fixed, gently saturated ink map: clip(1.25*x-.125,0,1).
    affine('ink_thresholds','image',np.array([1.25,1.25]).reshape(2,1,1,1),[-.125,-1.125],True)
    op('ink_hinges','relu','ink_thresholds')
    affine('ink','ink_hinges',np.array([1.,-1.]).reshape(1,2,1,1),[0],True)

    # Divide the ink map by its total mass, using the allowed spatial norm.
    # The +/- pairs have mean zero. Large +/- sum sentinels dominate variance,
    # giving x/sum(x) to better than 1e-5 for nonempty digit images.
    n = SIDE*SIDE
    k = 100.
    prep = np.zeros((2*n+2,n),np.float32)
    prep[:n]=np.eye(n,dtype=np.float32)
    prep[n:2*n]=-np.eye(n,dtype=np.float32)
    prep[-2,:]=k
    prep[-1,:]=-k
    affine('mass_pairs','ink',prep,np.zeros(2*n+2))
    op('mass_shape','reshape','mass_pairs',shape=[1,1,2*n+2])
    op('mass_norm','norm','mass_shape')
    pick = np.zeros((n,2*n+2),np.float32)
    pick[:,:n]=np.eye(n,dtype=np.float32)*(k*math.sqrt(2/(2*n+2)))
    affine('ink_density','mass_norm',pick,np.zeros(n))

    # Local stroke occupancy with an ink-mass-independent near-binary threshold.
    # The small mean filter makes a one-pixel offset much less consequential.
    y,x=np.mgrid[-2:3,-2:3].astype(np.float32)
    gaussian=np.exp(-(x*x+y*y)/(2*.85**2))
    gaussian/=gaussian.sum()
    affine('soft_ink','ink',gaussian.reshape(1,1,5,5),[0],True,padding=2)
    affine('occupancy_thresholds','soft_ink',np.array([2.4,2.4]).reshape(2,1,1,1),[0,-1],True)
    op('occupancy_hinges','relu','occupancy_thresholds')
    affine('occupancy','occupancy_hinges',np.array([1.,-1.]).reshape(1,2,1,1),[0],True)

    # A truncated distance-to-ink proximity map, expressed as nested dilations.
    proximity_parts=[]
    for radius,coef in [(0,.30),(1,.50),(2,.20)]:
        name='proximity_'+str(radius)
        if radius:
            affine(name+'_pad','occupancy',np.ones((1,1,1,1)),[0],True,padding=radius)
            op(name+'_pool','maxpool',name+'_pad',size=[2*radius+1]*2,stride=[1,1])
            source=name+'_pool'
        else:
            source='occupancy'
        affine(name,source,np.array([coef]).reshape(1,1,1,1),[0],True)
        proximity_parts.append(name)
    op('proximity','add',proximity_parts)

    # Compute closed background components by morphological reconstruction.
    # This is a fixed Boolean circuit, not a statistic fitted to examples.
    def clipped(name,source,scale,lo):
        affine(name+'_thresholds',source,np.array([scale,scale]).reshape(2,1,1,1),[-lo,-lo-1],True)
        op(name+'_hinges','relu',name+'_thresholds')
        return affine(name,name+'_hinges',np.array([1,-1]).reshape(1,2,1,1),[0],True)
    clipped('barrier','image',8.,1.6)
    # A 2x2 closing repairs subpixel breaks but does not fill an open bowl.
    affine('barrier_padding','barrier',np.ones((1,1,1,1)),[0],True,padding=1)
    op('barrier_dilate','maxpool','barrier_padding',size=[2,2],stride=[1,1])
    affine('barrier_negative','barrier_dilate',-np.ones((1,1,1,1)),[0],True)
    op('barrier_erode_negative','maxpool','barrier_negative',size=[2,2],stride=[1,1])
    affine('background','barrier_erode_negative',np.ones((1,1,1,1)),[1],True)
    # Constant edge seeds, produced by biases rather than loaded pixel data.
    affine('edge_zero','image',np.zeros((1,1,1,1)),[0],True)
    edge=np.zeros(n,np.float32)
    edge.reshape(28,28)[[0,-1],:]=1
    edge.reshape(28,28)[:,[0,-1]]=1
    affine('edge_flat','edge_zero',np.zeros((n,n),np.float32),edge)
    op('edge_seeds','reshape','edge_flat',shape=[1,28,28])
    previous='edge_seeds'
    for i in range(42):
        name='flood_'+str(i)
        affine(name+'_pad',previous,np.ones((1,1,1,1)),[0],True,padding=1)
        op(name+'_dilate','maxpool',name+'_pad',size=[3,3],stride=[1,1])
        op(name+'_and_sum','add',[name+'_dilate','background'])
        affine(name+'_and',name+'_and_sum',np.ones((1,1,1,1)),[-1],True)
        previous=op(name,'relu',name+'_and')
    affine('outside_negative',previous,-np.ones((1,1,1,1)),[0],True)
    op('holes_difference','add',['background','outside_negative'])
    op('holes','relu','holes_difference')
    regions=[(3,25,3,25),(3,14,3,25),(14,25,3,25)]
    w=np.zeros((3,n),np.float32)
    for i,(y0,y1,x0,x1) in enumerate(regions):
        w[i].reshape(28,28)[y0:y1,x0:x1]=1
    affine('hole_areas','holes',w,np.zeros(3))
    # Smooth hinge approximations to large hole, upper loop and lower loop.
    w=np.zeros((6,3),np.float32)
    w[0:2,0]=1/30
    w[2:4,1]=1/12
    w[4:6,2]=1/12
    affine('hole_thresholds','hole_areas',w,[-.65,-1.65,-.2,-1.2,-.2,-1.2])
    op('hole_hinges','relu','hole_thresholds')
    w=np.zeros((3,6),np.float32)
    for i in range(3): w[i,2*i:2*i+2]=[1,-1]
    affine('loop_features','hole_hinges',w,np.zeros(3))

    op('ink_density_shape','reshape','ink_density',shape=[1,28,28])
    op('coarse_density','avgpool','ink_density_shape',size=[2,2],stride=[2,2])
    op('coarse_proximity','avgpool','proximity',size=[2,2],stride=[2,2])
    op('coarse_ink','avgpool','ink',size=[2,2],stride=[2,2])
    op('contrast_ink','norm','coarse_ink')
    total=10*N_PER_CLASS
    nsh=len(SHIFTS)
    cn=14*14
    pw=np.empty((total,cn),np.float32)
    rw=np.empty((total*PARTS*nsh,cn),np.float32)
    cw=np.empty((total,cn),np.float32)
    rng=np.random.default_rng(SEED)
    for digit in range(10):
        for j in range(N_PER_CLASS):
            p=pen_points(digit,rng)
            precision,recall,synthetic_ink=fields(p)
            idx=digit*N_PER_CLASS+j
            pw[idx]=.40*.52*precision.reshape(14,2,14,2).mean(axis=(1,3)).reshape(-1)*4
            for part in range(PARTS):
                for si,(dy,dx) in enumerate(SHIFTS):
                    shifted=translate(recall[part],dy,dx)
                    rw[(idx*PARTS+part)*nsh+si]=.40*.48*shifted.reshape(14,2,14,2).sum(axis=(1,3)).reshape(-1)
            template=synthetic_ink.reshape(14,2,14,2).mean(axis=(1,3)).reshape(-1)
            template-=template.mean()
            template/=np.sqrt(np.mean(template**2)+1e-5)
            cw[idx]=.60*template/cn
        print('Constructed analytic models for',digit,flush=True)
    affine('precision_scores','coarse_density',pw,np.zeros(total))
    b=np.array([-.40*.008*(abs(dy)+abs(dx)) for dy,dx in SHIFTS]* (total*PARTS),np.float32)
    affine('recall_alternatives','coarse_proximity',rw,b)
    op('recall_alternatives_shape','reshape','recall_alternatives',shape=[total,PARTS,nsh])
    op('local_best','maxpool','recall_alternatives_shape',size=[1,nsh],stride=[1,nsh])
    op('local_sum','avgpool','local_best',size=[PARTS,1],stride=[PARTS,1])
    op('local_sum_flat','reshape','local_sum',shape=[1,1,total])
    affine('recall_scale','local_sum_flat',np.array([PARTS]).reshape(1,1,1,1),[0],True)
    op('recall_scores','reshape','recall_scale',shape=[total])
    affine('correlation_scores','contrast_ink',cw,np.zeros(total))
    op('model_scores','add',['precision_scores','recall_scores','correlation_scores'])
    op('class_models','reshape','model_scores',shape=[10,1,N_PER_CLASS])
    op('best_model','maxpool','class_models',size=[1,N_PER_CLASS],stride=[1,N_PER_CLASS])
    op('best_flat','reshape','best_model',shape=[10])
    # Explicit topological priors: zero is an enclosing oval; six and nine have
    # one smaller loop; eight has two. These are logical, hand-chosen scores.
    w=np.zeros((10,3),np.float32)
    w[0]=[.07,0,0]
    w[1]=[0,-.025,-.025]
    w[2]=[0,-.015,0]
    w[3]=[0,-.04,-.04]
    w[4]=[0,0,-.02]
    w[5]=[0,-.02,0]
    w[6]=[0,-.02,.045]
    w[7]=[0,-.025,-.025]
    w[8]=[0,.025,.025]
    w[9]=[0,.045,-.02]
    affine('topology_scores','loop_features',w,np.zeros(10))
    op('scores','add',['best_flat','topology_scores'])
    with (destination/'network.json').open('w') as f:
        json.dump({'layers':layers},f,indent=2)
    np.savez(destination/'weights.npz',**weights)
    print('Wrote',sum(a.size for a in weights.values()),'float32 weights.',flush=True)


if __name__=='__main__':
    build(sys.argv[1])
