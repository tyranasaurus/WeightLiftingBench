import numpy as np

def blur(a, k):
    def along(v, axis):
        n=v.shape[axis]
        j=np.arange(n)
        return (v.take(np.maximum(j-k,0),axis=axis)+2*v+v.take(np.minimum(j+k,n-1),axis=axis))*.25
    return along(along(a,-1),-2)

def pool(a,k):
    return a.reshape(a.shape[:-2]+(32//k,k,32//k,k)).mean(axis=(-1,-3))

def groups(a):
    return a.reshape(len(a),-1)

def unit(a,noise):
    a=groups(a)
    return a/np.sqrt((a*a).mean(axis=1,keepdims=True)+noise*noise)/np.sqrt(np.float32(a.shape[1]))

def bundle(parts):
    return np.ascontiguousarray(np.concatenate([np.sqrt(np.float32(w))*unit(a,n) for w,a,n in parts],axis=1),dtype=np.float32)

def grad(a):
    j=np.arange(32)
    return (a.take(np.minimum(j+1,31),axis=-1)-a.take(np.maximum(j-1,0),axis=-1),a.take(np.minimum(j+1,31),axis=-2)-a.take(np.maximum(j-1,0),axis=-2))

def describe(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=.30*r+.52*g+.18*b
    rg=(r-g)/(.28+y)
    yb=((r+g)*.5-b)/(.28+y)
    # Local edge energy and a four-bin contrast-insensitive orientation code.
    # The doubled angle avoids a discontinuity at pi.
    xs=blur(x,1)*.65+x*.35
    gx,gy=grad(xs)
    xx=(gx*gx).mean(1); yy=(gy*gy).mean(1); xy=(gx*gy).mean(1)
    power=xx+yy
    mag=np.sqrt(power+.000001)
    cs=(xx-yy)/(power+.00005); sn=2*xy/(power+.00005)
    env=blur(blur(mag,2),3)
    wt=mag/np.sqrt(.035+env)
    bins=np.stack([np.maximum(t+.22,0)**1.5 for t in (cs,sn,-cs,-sn)],axis=1)*wt[:,None]
    cell=pool(blur(bins,1),4)
    local=cell-cell.mean(1,keepdims=True)
    strength=pool(blur(wt,1),4)
    relative=strength-strength.mean((1,2),keepdims=True)
    tone=pool(blur(y,1),4)
    band=tone-blur(tone,1)
    broad=tone-tone.mean((1,2),keepdims=True)
    # Overall angular preference is retained separately; subtracting the spatial
    # pedestal here makes the actual positions of contours matter more.
    centered=local-.75*local.mean((2,3),keepdims=True)
    shape=bundle([(.37,centered,.007),(.12,local,.007),(.19,relative,.020),(.16,band,.016),(.16,broad,.045)])
    # Colour maps encode material and illumination, not just grey-level shape.
    opp=np.stack((rg*1.4,yb),1)
    col=pool(blur(np.tanh(opp*3),1),8)
    warm=np.tanh(yb*3); green=np.tanh(-rg*6); cool=-warm
    pale=np.clip((y-.40)*2.3,-1,1)
    vivid=np.tanh(np.sqrt((1.5*rg)**2+yb**2)*3)
    sem=pool(np.stack((warm,green,cool*pale,pale,vivid-.35),axis=1),8)
    lighting=pool(y,8)-.5
    det=np.abs(y-blur(y,1))
    detail=pool(np.sqrt(blur(det,1)+.0001),8)-.125
    grit=np.stack((pool(wt,8)-.21,detail*2),axis=1)
    scene=bundle([(.39,sem,.26),(.14,col,.24),(.19,lighting,.14),(.15,grit,.10),(.13,pool(np.stack((wt*cs,wt*sn),1),8),.045)])
    # Soft figure/ground contrast. Compare a pixel to the left and right
    # margins at the same *height*, so the sky/land split has little effect.
    smooth=blur(x,1)
    margin=smooth[:,:,:,:4].mean(3);other=smooth[:,:,:,28:].mean(3)
    def rowblur(t):
        jj=np.arange(32)
        return (.5*t+.25*t[:,:,np.maximum(jj-3,0)]+.25*t[:,:,np.minimum(jj+3,31)])[:,:,:,None]
    def dist(t):
        d=smooth-rowblur(t)
        lum=.3*d[:,0]+.52*d[:,1]+.18*d[:,2]
        return np.sqrt(lum*lum+.7*(d[:,0]-d[:,1])**2+.40*(d[:,1]-d[:,2])**2+.0001)
    f=np.minimum(dist(margin),dist(other));f=f/(f+.125)
    figure=pool(f,4);figure-=figure.mean((1,2),keepdims=True)
    # Directional dark spots and light spots, on a finer sampling grid
    # than the broad tone map (e.g. paired eyes, nostrils or wheel hubs).
    details=pool(blur(y,1)-blur(blur(y,1),3),2)
    fine=pool(wt,2);fine-=fine.mean((1,2),keepdims=True)
    landmark=bundle([(.45,figure,.104),(.37,details,.037),(.18,fine,.067)])
    return shape,scene,landmark

# Smooth votes in [0,1]. A scene cue is evidence, not a precondition.
def ramp(x,lo,hi):
    return np.clip((x-lo)/(hi-lo),0,1)

def av(a):
    return a.mean(axis=(-1,-2))

def cues(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=.30*r+.52*g+.18*b
    l=blur(y,1); tiny=np.abs(y-l); contrast=y.std(axis=(1,2))
    rg=(r-g)/(.23+y); gb=(g-b)/(.23+y); rb=(r-b)/(.23+y)
    sat=(x.max(1)-x.min(1))/(.20+y)
    # Distinguish a blue hue from the cool tint of a grey road in shade.
    blue=ramp(-rb,-.025,.22)*ramp(-gb,-.025,.12)
    leafy=ramp(-rg,-.033,.113)*ramp(gb,.031,.225)
    brownish=ramp(rg,.006,.18)*ramp(rb,.072,.35)*(1-.70*ramp(rg,.33,.75))
    neutral=1-ramp(sat,.065,.27)
    gx,gy=grad(blur(x,1)); xx=(gx*gx).mean(1); yy=(gy*gy).mean(1); xy=(gx*gy).mean(1)
    m=np.sqrt(xx+yy+.000001); cs=(yy-xx)/(xx+yy+.00003);sn=2*xy/(xx+yy+.00003)
    dx=np.diff(y,axis=2);dy=np.diff(y,axis=1)
    h=(av(m*cs)/(av(m)+.009)); axial=av(m*(cs**2-sn**2))/(av(m)+.008)
    row=av(np.abs(gy.mean(1).mean(2,keepdims=True)))/(av(m)+.012)
    vertical=av(np.abs(gx.mean(1).mean(1,keepdims=True)))/(av(m)+.012)
    rough=blur(tiny,2); flat=1-ramp(rough,.012,.063)
    sky=flat*(blue*.91+(1-blue)*neutral*ramp(y,.56,.89)*.77)
    top=av(sky[:,:10]);bot=av(sky[:,23:]);blow=av(blue[:,22:]);bleft=av(blue[:,:,:6]);bright=av(y[:,:8])-av(y[:,24:]);brightlow=av(y[:,23:])
    leaf=av(leafy);leaflow=av(leafy[:,20:]); leafyedge=(av(leafy[:,:,:5])+av(leafy[:,:,27:]))*.5
    dirt=av(brownish); warm=av(brownish[:,6:28,4:28]); cold=av(blue); satc=av(sat[:,7:25,5:27]); greenmid=av(leafy[:,7:25,5:27]); coldmid=av(blue[:,8:25,5:27]); neut=av(neutral)
    road=av((flat*neutral*(1-.65*ramp(y,.76,.94)))[:,23:])
    ht=ramp(h,-.10,.42);axisvote=ramp(axial,-.07,.39); rows=ramp(row,.11,.45);vert=ramp(vertical,.12,.30)
    grain=ramp(av(tiny),.023,.062);fine=ramp(av(tiny)/(av(np.abs(y-blur(l,3)))+.008),.30,.69)
    corners=np.concatenate((y[:,:6,:7],y[:,:6,-7:],y[:,-6:,:7],y[:,-6:,-7:]),axis=2)
    plain=ramp(.18-corners.std((1,2)),0,.12)*ramp(.040-av(tiny[:,:7]),0,.030)
    skyall=av(sky); skybottom=bot;ground=ramp(road,.22,.74);fol=ramp(leaf,.095,.51);folbottom=ramp(leaflow,.12,.60);folside=ramp(leafyedge,.10,.58)
    sea=ramp(blow,.14,.75); canopy=ramp(top,.22,.76); manyblue=ramp(cold,.18,.75)
    fur=ramp(warm,.10,.61);midgreen=ramp(greenmid,.12,.62);earth=ramp(dirt,.12,.58)
    # Lower-edge shadow, common under road vehicles; a boat has more
    # horizontal structure in the *background* as well, often blue or grey.
    profile=y[:,:,4:28].mean(2); yprof=profile[:,14:29];shadow=av(y[:,26:])-yprof.min(1)
    under=ramp(shadow/(contrast+.05),.15,1.05)
    bodycolor=ramp(np.maximum(0,av(sat[:,8:25])-av(sat[:,24:])),.018,.22)
    topplain=ramp(av(flat[:,:9]),.60,.94);skywrap=canopy*ramp((bot+skyall)*.5,.18,.68)
    coolwater=sea*(.55+.45*rows);ripple=ramp(av((m*cs)[:,22:])/(av(m[:,22:])+.014),.04,.55)
    wild=.50*folside+.30*folbottom+.20*earth
    mech=.37*axisvote+.40*rows+.23*ht
    # Width of concentration of edge energy. Sky & water can leave
    # the subject in a thin band. Portraits have a taller central mass.
    band=av(m[:,9:23])/(av(m)+.01); middle=ramp(band,.98,1.55)
    shoulder=ramp(av(m[:,3:13])/(av(m[:,14:24])+.01),.48,1.03)
    lowtex=ramp(av(tiny[:,22:])/(av(tiny[:,8:24])+.007),.48,1.14)
    smoothbg=.5*topplain+.5*ramp(av(flat[:,23:]),.53,.93)
    # Indoor/adaptable animal appearances: near-neutral, reddish or cream
    # materials, no long horizon, often considerable fine-scale detail.
    homelike=(1-.65*fol)*(1-.68*manyblue)*(1-.53*canopy)
    furtex=grain*.45+fine*.35+(1-axisvote)*.20
    animal=(1-mech)*.63+.37*(1-rows)
    q=np.zeros((len(x),10),np.float32)
    q[:,0]=1.24*skywrap+.61*manyblue+.39*plain+.35*topplain+.36*middle+.18*smoothbg-.48*wild-.40*earth-.29*ripple-.20*grain-.76
    q[:,1]=.92*ground+.61*under+.48*middle+.55*mech+.41*bodycolor+.13*neut-.52*sea-.34*fol-.34*earth-.22*shoulder-.84
    q[:,2]=.50*animal+.40*wild+.28*midgreen+.21*plain+.32*smoothbg+.20*manyblue+.16*skywrap-.24*mech-.67
    q[:,3]=.88*homelike+.59*furtex+.32*shoulder+.36*animal+.16*fur-.42*folside-.22*midgreen-.20*ground-.95
    q[:,4]=.94*wild+.56*fur+.34*animal+.22*fine+.27*folbottom-.30*manyblue-.19*canopy-.15*ground-.75
    q[:,5]=.63*homelike+.44*animal+.29*fur+.20*shoulder+.20*furtex+.15*folbottom-.28*manyblue-.19*midgreen-.76
    q[:,6]=.84*midgreen+.48*earth+.51*grain+.41*lowtex+.29*fine+.28*animal-.65*canopy-.28*ground-.31*manyblue-.86
    q[:,7]=.52*folbottom+.40*fur+.29*canopy+.29*shoulder+.28*wild+.13*axisvote-.41*sea-.19*midgreen-.56
    q[:,8]=1.08*coolwater+.74*rows+.58*ripple+.48*canopy+.28*ht+.30*ramp(bright,.065,.40)-.44*wild-.36*under-.27*vert-.83
    q[:,9]=.86*ground+.70*axisvote+.44*rows+.40*under+.30*vert+.31*canopy+.21*shoulder-.39*sea-.32*fol-.32*earth-.99
    return q

# Match full-image views. Both reflected views and small offsets are tried.
# Scores compare one input to individual templates (never to a class mean).
def search(x):
    a,b,c=describe(x)
    n=len(x);z=np.empty((n,10,3),np.float32)
    for k in range(10):
        sh=a@CONSTANTS['shape'+str(k)].T
        sc=b@CONSTANTS['scene'+str(k)].T
        mark=c@CONSTANTS['landmark'+str(k)].T
        sh=.76*sh+.24*mark
        joint=.72*sh+.28*sc
        # Max for rare poses, and a few supporting views for stability.
        v=np.partition(joint,joint.shape[1]-6,axis=1)[:,-6:]
        z[:,k,0]=.45*v.max(axis=1)+.55*v.mean(axis=1)
        u=np.partition(sh,sh.shape[1]-4,axis=1)[:,-4:]
        z[:,k,1]=.6*u.max(axis=1)+.4*u.mean(axis=1)
        w=np.partition(sc,sc.shape[1]-8,axis=1)[:,-8:]
        z[:,k,2]=w.mean(axis=1)
    return z

def classify(x):
    # The camera models already sample object size and orientation. The four
    # off-centre reads supply some finer translational tolerance.
    j=np.arange(32); scores=[]
    for dx,dy,mirror in [(0,0,False),(0,0,True),(-2,-2,False),(2,-2,True),(2,2,False),(-2,2,True)]:
        v=x if not mirror else x[:,:,:,::-1]
        if dx!=0:
            v=v.take(np.clip(j+dy,0,31),axis=2).take(np.clip(j+dx,0,31),axis=3)
        scores.append(search(v))
    stack=np.stack(scores)
    best=.74*stack.max(axis=0)+.26*stack.mean(axis=0)
    return (12.0*best[:,:,0]+3.8*best[:,:,1]+3.0*best[:,:,2]+cues(x)).argmax(axis=1)

def predict(x):
    # Fixed internal batch shape keeps kernel selection independent of the
    # caller's batch size. Each output uses only that row's features.
    ans=np.empty(len(x),np.int64)
    for i in range(0,len(x),48):
        n=min(48,len(x)-i)
        xx=np.zeros((48,3,32,32),np.float32);xx[:n]=x[i:i+n]
        ans[i:i+n]=classify(xx)[:n]
    return ans
