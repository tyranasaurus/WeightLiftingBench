import numpy as np

# All reductions in the descriptor are within one photograph.
def blur(a):
    p = np.pad(a, ((0,0),(0,0),(1,1)), mode='edge')
    a = (p[:,:,:-2] + 2*p[:,:,1:-1] + p[:,:,2:]) * .25
    p = np.pad(a, ((0,0),(1,1),(0,0)), mode='edge')
    return (p[:,:-2] + 2*p[:,1:-1] + p[:,2:]) * .25

def pool(a,k):
    n,h,w = a.shape
    return a.reshape(n,h//k,k,w//k,k).mean(axis=(2,4))

def grad(a):
    p=np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    return (p[:,1:-1,2:]-p[:,1:-1,:-2])*.5,(p[:,2:,1:-1]-p[:,:-2,1:-1])*.5

def desc(x):
    n=len(x)
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=.29*r+.55*g+.16*b
    blocks=[]
    def unit(a,w):
        z=a.reshape(n,-1)
        blocks.append(w**.5*z / np.sqrt((z*z).sum(1,keepdims=True)+.000001))
    def rms(a,w):
        z=a.reshape(n,-1)
        blocks.append(z*((w/z.shape[1])**.5))
    def center(a):
        return a-a.mean((1,2),keepdims=True)
    # Signed arrangement of light and shadow, with soft contrast compression.
    t=pool(y,4)
    v=center(t)
    c=np.sqrt((v*v).mean((1,2),keepdims=True)+.0036)
    unit(np.tanh(v/(1.6*c)),.18)
    hp=t-blur(blur(t))
    unit(np.tanh(hp/(.075+.30*c)),.11)
    # Spatially pooled edge directions, at two bandwidths. Opposite edge signs
    # are equivalent. Isotropic texture does not count as a strong line.
    dx,dy=grad(blur(y)); fx,fy=grad(y)
    magnitude=np.sqrt(dx*dx+dy*dy)+1.e-8
    fine=np.sqrt(fx*fx+fy*fy)+1.e-8
    dx2,dy2=grad(blur(blur(blur(y))))
    mag2=np.sqrt(dx2*dx2+dy2*dy2)+1.e-8
    for a,d,m,k,wt in ((dx,dy,magnitude,4,.21),(dx2,dy2,mag2,8,.10)):
        hist=[]
        for i in range(6):
            ang=i*np.pi/6
            proj=np.abs(a*np.cos(ang)+d*np.sin(ang))
            q=np.maximum(proj-.835*m,0)*6
            hist.append(pool(blur(q.astype(np.float32)),k))
        h=np.stack(hist,1); en=h.sum(1,keepdims=True)
        h=(h-.105*en)/np.sqrt(en+.044)
        unit(h,wt)
    # Edge density. Absolute and relative texture are both informative.
    d=pool(fine,4); dl=np.log1p(d*35)
    rms(np.clip(center(dl)/.50,-2.5,2.5),.09)
    rms(np.clip((pool(fine,8)-pool(magnitude,8))/.032,0,2.8),.026)
    # Opponent chroma coordinates are regularized in the shadows. Pool AFTER
    # compression, so mottled vegetation is not equivalent to a uniform grey.
    warm=np.tanh(3.1*(r-b)/(.28+y))
    leaf=np.tanh(6*(g-.58*r-.42*b)/(.25+y))
    sat=(np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b))/(.22+y)
    color=np.stack([pool(warm,8),pool(leaf,8),pool(np.tanh(2.5*sat),8)],1)
    rms(color*np.array([1.30,1.27,.70],np.float32)[None,:,None,None],.63)
    rms(center(pool(warm,4)),.095); rms(center(pool(leaf,4)),.09)
    rms(np.tanh((pool(y,8)-.50)*3),.065)
    return np.ascontiguousarray(np.concatenate(blocks,axis=1),dtype=np.float32)

def rules(x):
    r,g,b=x[:,0],x[:,1],x[:,2]; n=len(x); y=.29*r+.55*g+.16*b
    def ramp(z,a,b): return np.clip((z-a)/(b-a),0,1)
    def avg(z,region): return z[:,region[0],region[1]].mean((1,2))
    def whole(z):return z.mean((1,2))
    top=(slice(0,8),slice(None)); bot=(slice(25,32),slice(None)); cen=(slice(8,25),slice(6,26)); mid=(slice(9,24),slice(None))
    side=np.ones((32,32),np.float32);side[5:27,5:27]=0
    def outside(z):return (z*side).sum((1,2))/side.sum()
    dx,dy=grad(blur(y));fx,fy=grad(y);mag=np.sqrt(dx*dx+dy*dy);fine=np.sqrt(fx*fx+fy*fy)
    # Feathered color names. Requiring a channel ordering rejects, for example,
    # blue concrete in shadow as grass, or yellowish fur as a green leaf.
    blue=ramp((b-r)/(.28+y),.035,.24)*ramp((b-g)/(.28+y),.003,.125)
    green=ramp((g-b)/(.25+y),.030,.25)*ramp((g-r+.013)/(.24+y),-.018,.105)
    olive=ramp((g-b)/(.32+y),.05,.25)*ramp((g-r)/(.26+y),-.091,.012)
    vegetation=np.maximum(green,.72*olive)
    warm=ramp((r-b)/(.26+y),.075,.37)*ramp((r-g)/(.28+y),.01,.16)
    vivid=ramp((np.maximum(r,np.maximum(g,b))-np.minimum(r,np.minimum(g,b)))/(.28+y),.26,.59)
    red=ramp((r-g)/(.28+y),.22,.58)*ramp((r-b)/(.28+y),.24,.58)
    gray=1-ramp((np.maximum(r,np.maximum(g,b))-np.minimum(r,np.minimum(g,b)))/(.28+y),.040,.24)
    bt=avg(blue,top);bb=avg(blue,bot);bg=outside(blue);vg=outside(vegetation);vb=avg(vegetation,bot);vc=avg(vegetation,cen)
    wc=avg(warm,cen);wg=outside(warm);gt=avg(gray,top);gb=avg(gray,bot);gc=avg(gray,cen)
    ed=avg(mag,cen); eb=avg(mag,bot);et=avg(mag,top)
    sky=bt*ramp(.088-et,0,.068); sea=bb*(1-.60*vb); grass=vb*(1-.43*avg(blue,bot))
    flat=ramp(.060-outside(mag),0,.040)
    smoothTop=ramp(.057-et,0,.043);smoothBot=ramp(.060-eb,0,.045)
    dyabs=np.abs(dy);dxabs=np.abs(dx)
    horiz=(avg(dyabs,mid)-avg(dxabs,mid))/(.015+avg(dyabs+dxabs,mid)); hori=np.clip(horiz*3,-1,1)
    align=whole(np.abs(dx*dx-dy*dy)/(.00003+mag)) / (.002+whole(mag))
    axial=np.clip((align-.65)*5,-.85,1.2)
    # Concentration of mass in a horizontal band, and coherent horizontal lines.
    profile=(mag+.42*fine).mean(2); central=profile[:,8:25].mean(1);outer=(profile[:,:6].mean(1)+profile[:,26:].mean(1))*.5
    band=np.clip((central-outer)/(.034+central+outer)*2.3,-1,1.2)
    row=np.abs(dy[:,:,3:-3].mean(2)); line=np.clip(row[:,7:27].max(1)/(.036+ed)-.65,-.65,1.35)
    road=gb*smoothBot*(1-.75*bb);skyline=sky*avg(gray,mid); water=sea*(.6+.4*np.maximum(0,hori))
    shade=np.clip((avg(y,top)-avg(y,cen))/.28,-1,1); lower=np.clip((avg(y,bot)-avg(y,cen))/.25,-1,1)
    topbright=ramp(avg(y,top),.52,.86)*smoothTop
    # Symmetry of a central face: compare both sides locally, under small shifts.
    sym=np.zeros(n,np.float32)-1
    for cx in (13,15,17,19):
        a=y[:,7:25,cx-10:cx];d=y[:,7:25,cx:cx+10][:,:,::-1];
        a=a-a.mean((1,2),keepdims=True); d=d-d.mean((1,2),keepdims=True)
        cc=whole(a*d)/np.sqrt((.002+whole(a*a))*(.002+whole(d*d)))
        sym=np.maximum(sym,cc)
    # Evidence for dark eye pairs and a darker nose as in a frontal pet head.
    sm=blur(y); hp=blur(blur(blur(sm)))-sm
    eye=np.zeros(n,np.float32); nosepair=np.zeros(n,np.float32)
    for gap in (5,8,11):
        xx=np.arange(10,22)-gap//2; a=hp[:,6:20,xx];d=hp[:,6:20,xx+gap];pair=np.minimum(a,d)
        nose=hp[:,6+gap//2:20+gap//2,xx+gap//2]
        eye=np.maximum(eye,pair.max((1,2)));nosepair=np.maximum(nosepair,(np.minimum(pair,nose)).max((1,2)))
    facecue=ramp(eye,.035,.12)*ramp(sym,.16,.74);muzzle=ramp(nosepair,.025,.096)*ramp(sym,.12,.70)
    # Wheel cue: dark upper arc and hub / tire reversal. Search for a pair
    # near one road plane; softness reflects aliasing of wheels at this scale.
    wheelcue=np.zeros(n,np.float32)
    for rad,delta in ((2,6),(3,10),(4,13)):
        yy=np.arange(15,27);xx=np.arange(rad,32-rad-delta)
        def gather(oy,ox):return sm[:,yy[:,None]+oy,xx[None,:]+ox]
        def tire(off):
            center=gather(0,off);ring=.34*gather(-rad+1,off)+.33*gather(0,off-rad+1)+.33*gather(0,off+rad-1)
            surround=.4*gather(-rad-1,off)+.3*gather(1,off+rad)+.3*gather(1,off-rad)
            return .72*(surround-ring)+.28*(center-ring)
        wheels=np.minimum(tire(0),tire(delta));wheelcue=np.maximum(wheelcue,wheels.max((1,2)))
    wheels=ramp(wheelcue,.037,.145)
    # Mottled close-up texture versus fine parallel/rectilinear structure.
    texture=ramp(avg(fine,cen)-.58*ed,.020,.080)
    clutter=ramp(outside(fine),.040,.115)
    brownfield=np.minimum(1,vg+.70*wg);colorObj=avg(vivid,cen);patch=avg(red,cen)
    # Relative adjustments in ten-class evidence units. Scene and morphology
    # provide independent evidence: none of these are hard decision branches.
    s=np.zeros((n,10),np.float32)
    s[:,0]=1.40*sky+.68*bg+.50*flat+.33*smoothTop+.28*band+.22*shade-.48*grass-.48*clutter-.43*wc-.42*water-.20*axial
    s[:,8]=1.50*water+.35*bt+.37*np.minimum(bt,bb)+.34*hori+.38*line+.29*smoothTop+.20*band+.17*axial-.65*grass-.55*wc-.30*facecue
    s[:,1]=.56*road+.48*wheels+.45*band+.33*hori+.33*lower+.28*axial+.29*gc+.31*colorObj+.35*patch-.62*water-.42*vc-.24*facecue
    s[:,9]=.46*road+.35*wheels+.55*axial+.23*line+.23*skyline+.33*gc+.25*gt+.24*colorObj+.24*patch-.46*water-.50*vc-.22*facecue
    s[:,2]=.55*vg+.31*sky+.18*flat+.26*colorObj+.26*vc+.17*band-.30*axial-.22*hori-.24*wheels-.22*gc
    s[:,3]=.35*wc+.23*wg+.27*texture+.48*facecue+.21*muzzle+.20*sym+.26*(1-vg)-.51*sky-.32*water-.31*axial-.20*band-.21*grass
    s[:,5]=.60*wc+.22*wg+.26*texture+.33*facecue+.48*muzzle+.16*sym+.22*grass-.47*sky-.37*water-.32*axial-.21*band
    s[:,4]=.75*vg+.52*grass+.53*wc+.20*wg+.22*clutter+.17*texture-.49*sky-.46*water-.32*axial-.26*facecue-.20*colorObj
    s[:,7]=.52*grass+.57*wc+.24*vg+.14*wg+.15*shade+.16*topbright+.17*lower-.32*water-.24*axial-.30*vc-.25*facecue
    s[:,6]=.81*vc+.54*brownfield+.24*clutter+.35*texture+.24*wg+.19*wc-.71*sky-.40*axial-.30*facecue-.40*hori-.19*band
    # Offsets reflect the number of positive generic (non-discriminative)
    # terms in each rule, not measurements made on images.
    return s-np.array([.66,.74,.32,.55,.69,.60,.65,.48,.48,.64],np.float32)

def foreground(x):
    # Estimate background affinity from the outside edges of THIS photo.
    # Consider several anchors, allowing sky and ground to coexist. An isolated
    # object crossing a border should not be explained by a single anchor pixel.
    r,g,b=x[:,0],x[:,1],x[:,2]; y=.29*r+.55*g+.16*b
    a=pool(blur(y),2);chroma=pool(blur((r-b)/(.35+y)),2);leaf=pool(blur((g-.55*r-.45*b)/(.30+y)),2)
    dx,dy=grad(y);t=pool(blur(np.sqrt(dx*dx+dy*dy)),2)
    f=np.stack([a,1.07*chroma,1.38*leaf,1.65*t],1);co=f.reshape(len(x),4,8,2,8,2).mean((3,5))
    near=np.full_like(a,100);nextnear=np.full_like(a,100);pos=np.arange(16,dtype=np.float32)/15
    for i,j in ((0,0),(0,2),(0,5),(0,7),(2,0),(2,7),(4,0),(4,7),(6,0),(6,7),(7,0),(7,2),(7,5),(7,7)):
        d=((f-co[:,:,i,j,None,None])**2).sum(1)
        d+=.022*(pos[:,None]-i/7)**2+.003*(pos[None,:]-j/7)**2
        nextnear=np.minimum(nextnear,np.maximum(d,near));near=np.minimum(near,d)
    novelty=np.sqrt(.63*near+.37*nextnear);novelty=blur(blur(np.tanh(3.3*novelty)))
    novelty-=novelty.mean((1,2),keepdims=True);s=pool(novelty,2).reshape(len(x),64)
    return s/np.sqrt((s*s).sum(1,keepdims=True)+.40)

def lookup(d,fg):
    # Fixed matrix dimensions even on singleton batches: summation order is
    # independent of when a photo arrives. No other row enters its lookup.
    db=CONSTANTS['atlas']; norm=CONSTANTS['lengths']; masks=CONSTANTS['silhouette']
    m=np.zeros((32,64),np.float32);m[:len(d)]=fg
    q=np.empty((32,d.shape[1]),np.float32);q[:len(d)]=d;q[len(d):]=0
    qn=(q*q).sum(1,keepdims=True); s=np.empty((len(d),10),np.float32)
    for k in range(10):
        dist=(qn+norm[k][None,:]-2*(q@db[k].T)-.19*(m@masks[k].T))[:len(d)]
        best=np.partition(dist,15,axis=1)[:,:16]; nearest=best.min(1)
        first=np.partition(best,3,axis=1)[:,:4].mean(1)
        s[:,k]=-(.27*nearest+.43*first+.30*best.mean(1))
    return s

def predict(x):
    out=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):
        v=np.ascontiguousarray(x[i:i+32],dtype=np.float32)
        d=desc(v);f=desc(np.ascontiguousarray(v[:,:,:,::-1]));fg=foreground(v)
        a=lookup(d,fg);b=lookup(f,fg.reshape(len(v),8,8)[:,:,::-1].reshape(len(v),64))
        # Allow either handedness but retain a contribution from both searches.
        s=22*(.71*np.maximum(a,b)+.29*np.minimum(a,b))+rules(v)
        out[i:i+len(v)]=s.argmax(1)
    return out
