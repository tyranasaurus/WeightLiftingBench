import numpy as np

def saturate(x):
    return np.clip(x, 0., 1.)

def blur(a):
    z = np.pad(a, [(0,0)]*(a.ndim-2)+[(1,1),(1,1)], mode='edge')
    h = (z[..., :-2] + 2*z[..., 1:-1] + z[..., 2:])*0.25
    return (h[..., :-2,:]+2*h[...,1:-1,:]+h[...,2:,:])*0.25

def mean4(a):
    return a.reshape(a.shape[:-2]+(8,4,8,4)).mean(axis=(-3,-1))

def pool(a):
    return mean4(blur(a))

def deriv(a):
    z = np.pad(a, [(0,0)]*(a.ndim-2)+[(1,1),(1,1)], mode='edge')
    return (z[...,1:-1,2:]-z[...,1:-1,:-2])*0.5, (z[...,2:,1:-1]-z[...,:-2,1:-1])*0.5

def gray(a):
    return a[:,0]*0.299 + a[:,1]*0.587 + a[:,2]*0.114

def normalize(a, weight, floor):
    b = a.reshape(len(a),-1)
    return b * (np.sqrt(weight)/(np.sqrt(np.sum(b*b,axis=1,keepdims=True)) + floor))

def salmap(small):
    # A pixel in the centre that can also be explained by the colours at the
    # sides, particularly at the same height, is less likely to be the subject.
    trans = np.stack((gray(small), (small[:,0]-small[:,1])*0.85, (small[:,2]-small[:,1])*0.8),axis=1)
    best = np.full((len(small),8,8), 1., np.float32)
    same = best.copy()
    ids = np.arange(8,dtype=np.float32)
    for i in range(8):
        for j in (0,7):
            d = np.sum((trans-trans[:,:,i:i+1,j:j+1])**2,axis=1)
            best = np.minimum(best,d)
            same = np.minimum(same,d+0.0018*(ids[None,:,None]-i)**2)
    for i in (0,7):
        for j in (1,3,5,6):
            d = np.sum((trans-trans[:,:,i:i+1,j:j+1])**2,axis=1)
            best = np.minimum(best,d)
    return np.sqrt(best+.4*same+0.0001)

def describe(x):
    # Unsigned edges from the strongest of the three colour channels. The signed
    # branch keeps luminance polarity (e.g. black rubber below bright bodywork).
    a = x*0.4 + blur(x)*0.6
    dx,dy = deriv(a)
    k = np.argmax(dx*dx+dy*dy, axis=1)[:,None]
    ux = np.take_along_axis(dx,k,axis=1)[:,0]
    uy = np.take_along_axis(dy,k,axis=1)[:,0]
    mag = np.sqrt(ux*ux+uy*uy+1.e-10)
    energy = pool(mag)
    div = 0.012+0.45*energy.mean(axis=(1,2),keepdims=True)+0.55*blur(energy)+0.35*energy
    dirs = []
    for j in range(6):
        ang = j*np.pi/6
        proj = (np.float32(np.cos(ang))*ux+np.float32(np.sin(ang))*uy)/mag
        dirs.append(np.minimum(1.6,pool((proj**2)**3*mag)/div))
    h = np.stack(dirs,axis=1)
    h = h-0.56*h.mean(axis=1,keepdims=True)
    h = h-h.mean(axis=(1,2,3),keepdims=True)
    p = np.stack((pool(gray(dx)),pool(gray(dy))),axis=1)/div[:,None]
    # Coarser structure tolerates imprecise alignment of contours and limbs.
    small = pool(x)
    light = gray(small)
    low = light - light.mean(axis=(1,2),keepdims=True)
    band = light - blur(blur(light))
    # Midrange colour self-similarity is unchanged when a differently coloured
    # animal has the same arrangement of body, face and surrounding background.
    z = np.pad(small,[(0,0),(0,0),(2,2),(2,2)],mode='edge')
    comps = []
    for hy,hx in ((0,2),(2,0),(1,1),(1,-1)):
        diff = z[:,:,2-hy:10-hy,2-hx:10-hx]-z[:,:,2+hy:10+hy,2+hx:10+hx]
        d = (np.mean(diff**2, axis=1)+0.00025)**0.36
        comps.append(d-d.mean(axis=(1,2),keepdims=True))
    sim = np.stack(comps,axis=1)
    sal = salmap(small)
    sal = np.minimum(sal, 0.5)
    sal = sal-sal.mean(axis=(1,2),keepdims=True)
    result = np.concatenate((normalize(h,.43,.18), normalize(p,.16,.32), normalize(band,.12,.25), normalize(low,.065,.50), normalize(sim,.155,.4), normalize(sal,.07,.30)),axis=1)
    return result.astype(np.float32)

def region(x, ys, xs):
    return x[:,ys,xs].mean(axis=(1,2))

def scene(x):
    # Bounded evidence functions, not hard gates: a plane need not have sky
    # behind it, nor is every patch of blue water or sky.
    r,g,b=x[:,0],x[:,1],x[:,2]
    lum=gray(x)
    lo=np.minimum(np.minimum(r,g),b)
    hi=np.maximum(np.maximum(r,g),b)
    chroma=hi-lo
    dx,dy=deriv(blur(lum))
    fine=np.abs(lum-blur(lum))
    gm=np.sqrt(dx*dx+dy*dy+0.000001)
    flat=np.exp(-11*blur(gm)-15*blur(fine))
    blue=saturate((b-r+0.018)/.145)*saturate((b-g+.025)/.080)*saturate((b-.16)/.30)
    cool=saturate((b-r+.035)/.12)*saturate((g-r+.055)/.10)
    green=saturate((g-b-.004)/.12)*saturate((g-r+.045)/.10)
    brown=saturate((r-b-.025)/.18)*saturate((r-g+.012)/.12)*saturate((g-b+.03)/.13)
    skin=saturate((r-b-.04)/.22)*saturate((r-g-.015)/.12)*saturate((g-b+.01)/.12)
    bright=saturate((lum-.54)/.28)*saturate((.24-chroma)/.18)
    grey=saturate((.18-chroma)/.14)
    top=slice(0,8)
    bot=slice(24,32)
    allx=slice(0,32)
    mid=slice(8,24)
    def surroundings(a):
        return .5*(region(a,slice(4,30),slice(0,5))+region(a,slice(4,30),slice(27,32)))
    def border(a):
        return .4*surroundings(a)+.30*region(a,top,allx)+.30*region(a,bot,allx)
    skycol=blue*flat
    skywhite=bright*flat
    sky=region(skycol,top,allx)
    white=region(skywhite,top,allx)
    air=border((blue+.58*(1-blue)*bright)*flat)
    smooth=border(flat)
    lawn=region(green,bot,allx)
    leaf=border(green)
    woods=region(green,top,allx)
    water=region(cool,bot,allx)
    sea=region(blue,bot,allx)
    grnd=region(brown,bot,allx)
    warm=border(brown)
    road=region(grey,bot,allx)
    neutral=border(grey)
    field=lawn*np.maximum(.0,1-woods)
    centerbrown=region(brown,mid,mid)
    centergreen=region(green,mid,mid)
    warmbody=region(skin,mid,mid)
    colored=region(saturate((chroma-.23)/.22)*(1-green)*(1-brown),mid,mid)
    whitebody=region(bright,mid,mid)
    topflat=region(flat,top,allx)
    bottomflat=region(flat,bot,allx)
    # Horizontal and vertical lines that persist over multiple pixels indicate
    # the hull/deck of a ship or the windows, grille and chassis of a vehicle.
    a=lum-blur(lum)
    def line(ygrad,xgrad):
        v=ygrad[:,:,3:-3]
        p=(v+ygrad[:,:,:-6]+ygrad[:,:,6:])/3
        q=(np.abs(v)+np.abs(ygrad[:,:,:-6])+np.abs(ygrad[:,:,6:]))/3
        return np.maximum(0,np.abs(p)-0.52*q-.23*np.abs(xgrad[:,:,3:-3]))
    hline=line(dy,dx)
    vline=line(dx.transpose(0,2,1),dy.transpose(0,2,1)).transpose(0,2,1)
    strength=gm.mean(axis=(1,2))+.016
    h=(hline.mean(axis=(1,2))/strength-.15)*5
    v=(vline.mean(axis=(1,2))/strength-.105)*6
    h=np.clip(h,-.8,1.6)
    v=np.clip(v,-.7,1.6)
    wh=line(dy[:,23:,:],dx[:,23:,:]).mean(axis=(1,2))/(gm[:,23:,:].mean(axis=(1,2))+.012)
    ripple=np.clip((wh-.14)*4,-.4,1.)
    # Fine texture and the spatial distribution of detail separate grass, fur,
    # smooth sky and water. Dark wheel wells and tires often form a low band.
    texture=np.clip((border(fine)-.023)*17,-.4,.8)
    groundlight=region(lum,bot,allx)
    belly=region(lum,slice(20,25),slice(4,28))
    under=np.clip((groundlight-belly)*3,-.6,1.)
    horizon=np.clip((region(lum,top,allx)-groundlight)*2,-.7,1.)
    # A pair of dark tires at the same height is more informative than dark
    # pixels alone. Inner-disk, mid-ring and outer-road samples reject shadows.
    wheels=np.zeros(len(x),np.float32)
    wheelmap=blur(lum)
    rimsrc=np.pad(wheelmap,[(0,0),(5,5),(5,5)],mode='edge')
    for rad in (2,3,4):
        out=[];tire=[]
        for j in range(8):
            phi=(j+.5)*np.pi/4
            yy=int(round(np.sin(phi)*(rad+1.3)));xx=int(round(np.cos(phi)*(rad+1.3)))
            out.append(rimsrc[:,5+yy:37+yy,5+xx:37+xx])
            yy=int(round(np.sin(phi)*(rad*.71)));xx=int(round(np.cos(phi)*(rad*.71)))
            tire.append(rimsrc[:,5+yy:37+yy,5+xx:37+xx])
        radial=sum(tire)*.125
        contrast=sum(np.clip(outer-inner,-.11,.27) for outer,inner in zip(out,tire))*.125
        response=saturate((contrast+.40*np.maximum(wheelmap-radial,0)-.052)*7)*saturate((.54-radial)*4)
        for gap in (9,14,19):
            n=32-gap
            for lift in (-2,0,2):
                first=response[:,16:28,:n];second=response[:,16+lift:28+lift,gap:]
                wheels=np.maximum(wheels,np.minimum(first,second).max(axis=(1,2)))
    wheels-=.36
    # A hull separates the water from a different surface above. Uniform blue
    # all around a compact silhouette is better evidence of an object in flight.
    topcool=region(cool,top,allx)
    waterline=np.clip(water-topcool,-.35,.70)
    tip=region(flat,mid,mid)
    focal=np.clip(2*(smooth-tip),-.65,.95)
    continuous=np.abs(region(lum,top,allx)-groundlight)
    surroundsky=saturate(.74*air+.3*sky)*saturate(.98-2*continuous)
    foliage=(1-topflat)*woods
    solid=region(np.abs(lum-blur(blur(lum))),mid,mid)
    furry=np.clip((solid-.050)*7,-.33,.60)
    # Cues to a tightly framed furry face: roughly symmetrical dark eye pair,
    # nose below it and bright cheeks. Evaluate across several eye spacings.
    coarse=blur(lum)
    eyes=coarse*0
    for d in (2,3):
        z=np.pad(coarse,[(0,0),(d,d),(d,d)],mode='edge')
        ring=(z[:,:32, d:d+32]+z[:,2*d:,d:d+32]+z[:,d:d+32,:32]+z[:,d:d+32,2*d:])*.25
        eyes=np.maximum(eyes, ring-coarse)
    pair=np.zeros(len(x),np.float32)
    for d in (3,5,7):
        e1=eyes[:,7:19,10-d:22-d]
        e2=eyes[:,7:19,10+d:22+d]
        nose=eyes[:,7+d:19+d,10:22]
        eye=(np.minimum(e1,e2)*.78 + .22*np.maximum(e1,e2))
        pair=np.maximum(pair,(np.minimum(eye,nose+.04)+.35*eye).max(axis=(1,2)))
    face=saturate((pair-.105)*4.5)
    # Centred terms are hand-chosen neutral conditions for each soft predicate.
    sky-=.15; white-=.15; air-=.19; smooth-=.49; water-=.27; sea-=.10
    leaf-=.18; lawn-=.21; woods-=.16; grnd-=.23; warm-=.22; field-=.14
    neutral-=.52; road-=.53; centerbrown-=.23; centergreen-=.15; warmbody-=.16
    colored-=.10; whitebody-=.16; face-=.35
    natural = 1.2*leaf+.43*warm-.65*h-.42*v-.38*water-.32*neutral
    engineered = .75*h+.42*v+.53*road-.56*leaf-.36*warm
    s0=2.9*air+1.55*sky+.95*smooth+.48*white+.4*sea-.38*ripple-.75*leaf-.92*warm-.55*v-.42*h-.34*texture
    s1=engineered+.48*h+.66*under+.4*colored+.22*road-.64*sky-.42*sea-.2*horizon-.24*white
    s2=.62*natural+1.0*smooth+.54*leaf+.26*sky+.72*colored-.45*under-.27*v-.12*face
    s3=.45*natural+.67*neutral+.53*face+.57*texture-.64*sky-.40*lawn-.25*horizon-.24*under+.21*warm
    s4=natural+.84*leaf+.65*lawn+.44*woods+.59*centerbrown-.43*sky-.57*colored-.32*face-.28*whitebody
    s5=.40*natural+.54*face+.33*warmbody+.35*whitebody+.16*colored-.37*sea-.4*sky-.19*horizon
    s6=.50*natural+1.2*warm+.88*grnd+.57*centergreen+.39*texture-.6*white-.80*sky-.42*horizon-.39*sea-.26*face
    s7=.45*natural+.88*field+.69*centerbrown+.51*warmbody+.44*sky+.36*horizon-.43*sea-.22*face-.15*woods
    s8=1.85*water+1.48*sea+.76*sky+.62*white+.89*ripple+.76*h+.24*smooth-.52*v-1.08*leaf-.84*warm-.55*under
    s9=engineered+.76*v+.46*horizon+.49*sky+.38*white+.36*whitebody+.25*colored-.53*sea+.27*under
    s0+=1.5*(surroundsky-.10)+.36*focal-.64*waterline-.28*wheels
    s1+=1.13*wheels-.35*furry
    s2+=.85*focal+.51*foliage+.22*(surroundsky-.1)-.42*wheels
    s3+=.57*furry-.35*wheels
    s4+=.64*foliage-.26*wheels
    s5+=.32*furry-.22*wheels
    s6+=.40*furry-.35*wheels
    s7-=.21*wheels
    s8+=.9*waterline-.76*(surroundsky-.10)-.50*wheels
    s9+=.76*wheels-.20*furry
    return np.stack((s0,s1,s2,s3,s4,s5,s6,s7,s8,s9),axis=1)

def predict(x):
    labels=np.empty(len(x),np.int64)
    for start in range(0,len(x),64):
        a=x[start:start+64]
        # Fixed matrix geometry also keeps BLAS kernel selection independent of
        # the caller's batch size, even for the last incomplete batch.
        d=describe(a)
        block=np.zeros((64,d.shape[1]),np.float32)
        block[:len(a)]=d
        evidence=scene(a)
        for k in range(10):
            votes=(block @ CONSTANTS[str(k)].T)[:len(a)]
            # Keep best support and a few nearby alternatives rather than an
            # average prototype. A rare but distinctive viewpoint can win.
            support=np.partition(votes,-7,axis=1)[:,-7:]
            best=.66*support.max(axis=1)+.34*support.mean(axis=1)
            evidence[:,k]+=20*best
        labels[start:start+len(a)]=np.argmax(evidence,axis=1)
    return labels
