import numpy as np

def smooth(a):
    v=np.pad(a,[(0,0)]*(a.ndim-2)+[(1,1),(1,1)],mode='edge')
    h=(v[...,2:]+v[...,:-2]+2*v[...,1:-1])*.25
    return (h[...,2:,:]+h[...,:-2,:]+2*h[...,1:-1,:])*.25

def diff(a):
    v=np.pad(a,[(0,0)]*(a.ndim-2)+[(1,1),(1,1)],mode='edge')
    return (v[...,1:-1,2:]-v[...,1:-1,:-2])*.5,(v[...,2:,1:-1]-v[...,:-2,1:-1])*.5

def angles(dx,dy):
    planes=[]
    for a in range(6):
        c=np.float32(np.cos(a*np.pi/6)); s=np.float32(np.sin(a*np.pi/6))
        u=np.abs(dx*c+dy*s); v=np.abs(dx*s-dy*c)
        planes.append(np.maximum(u-1.45*v,0))
    return np.stack(planes,1)

def pool(a):
    return a.reshape(a.shape[:-2]+(12,4,12,4)).mean((-1,-3))

def norm(a,eps):
    return a/np.sqrt(np.sum(a*a,axis=tuple(range(1,a.ndim)),keepdims=True)+eps)

def clip(a):
    return np.clip(a,0,1)

def between(x,a,b):
    return clip((x-a)/(b-a))

def enlarge(x):
    u=(np.arange(48,dtype=np.float32)+.5)*(32/48)-.5
    u=np.clip(u,0,30.99999); j=u.astype(int); f=u-j.astype(np.float32)
    a=x[...,j]*(1-f)+x[...,j+1]*f
    return a[...,j,:]*(1-f[:,None])+a[...,j+1,:]*f[:,None]

def avg(a):
    return a.mean((-2,-1))

def stretch(a):
    return norm(a-a.mean((-2,-1),keepdims=True),.35)

def borderdiff(rgb):
    # How different is this colour from several nearby background samples?
    # A nearest single boundary pixel is unreliable; allow soft matches.
    n=len(rgb); yy,xx=np.mgrid[:12,:12]; by,bx=np.nonzero((xx==0)|(xx==11)|(yy==0)|(yy==11))
    dx=xx[None,:,:]-bx[:,None,None]; dy=yy[None,:,:]-by[:,None,None]
    spatial=(dx*dx/90+dy*dy/31).astype(np.float32)
    bd=rgb[:,:,by,bx]; d=rgb[:,:,None,:,:]-bd[:,:,:,None,None]
    dl=.27*d[:,0]+.55*d[:,1]+.18*d[:,2]; dg=d[:,1]-d[:,2]; dr=d[:,0]-d[:,1]
    delta=dl*dl+.68*(dg*dg+dr*dr)
    prob=np.exp(-spatial); w=(np.exp(-delta/.020)*prob).sum(1)/prob.sum(0)
    return -np.log(.012+w)

def boxline(a,axis,n):
    if axis==1: a=a.transpose(0,2,1)
    p=np.pad(a,[(0,0),(0,0),(n,n)],mode='edge'); b=p[:,:,:a.shape[2]].copy()
    for j in range(1,2*n+1): b+=p[:,:,j:j+a.shape[2]]
    b*=1/(2*n+1)
    return b.transpose(0,2,1) if axis==1 else b

def blurbox(a,n):
    return boxline(boxline(a,2,n),1,n)

def shift(a,dy,dx):
    yy=np.clip(np.arange(32)+dy,0,31); xx=np.clip(np.arange(32)+dx,0,31)
    return a[:,yy,:][:,:,xx]

def pairparts(g):
    # Tires with hubcaps or uniformly dark tires, viewed side-on or in perspective.
    # A pair of strong responses beats a lone round dark spot.
    s=smooth(g); local=blurbox(s,3); contrast=.055+.6*blurbox(np.abs(s-local),3)
    wheels=[]
    for rad in (2,3,4):
        z=np.zeros_like(g); outer=z.copy(); tire=z.copy()
        for cx,cy in ((1,0),(-1,0),(0,1),(0,-1),(.70,.70),(-.70,.70),(.70,-.70),(-.70,-.70)):
            outer+=shift(s,round(rad*1.5*cy),round(rad*1.5*cx))/8
            tire+=shift(s,round(rad*.78*cy),round(rad*.78*cx))/8
        z=between((outer-tire+.40*np.maximum(s-tire,0))/contrast,.24,2.2)*(.48+.52*between(tire,.51,.13))
        best=np.zeros(len(s),np.float32)
        for sep in (rad*4,rad*6):
            for slope in (-3,0,3):
                l=z[:,14:29,1:31-sep]; rr=shift(z,slope,sep)[:,14:29,1:31-sep]
                v=np.sqrt(l*rr).max((1,2)); best=np.maximum(best,v)
        wheels.append(best)
    wheels=np.maximum.reduce(wheels)
    # Eye pair above a nose: require all three dark details relative to local fur.
    eyes=between((local-s)/contrast,.26,2.1); noses=between((blurbox(s,4)-blurbox(s,1))/contrast,.18,2.0)
    dogface=np.zeros(len(s),np.float32); catface=dogface.copy()
    for half in (2,3,4,6):
        a=shift(eyes,0,-half); b=shift(eyes,0,half)
        pair=np.sqrt(a*b)
        for height in (half,half*2):
            n=shift(noses,height,0); quality=pair*(.3+.7*n)
            qual=quality[:,5:23,5:27].max((1,2))
            if height==half: catface=np.maximum(catface,qual)
            else: dogface=np.maximum(dogface,qual)
    return wheels,dogface,catface

def spectraltexture(g):
    # Structure at three octaves. Discount energy near horizontal/vertical
    # axes when estimating stippling: railings or a truck grille are not fur.
    u=np.arange(32,dtype=np.float32); w=np.sin(np.pi*(u+.5)/32)**1.3
    tap=w[:,None]*w; center=(g*tap).sum((1,2),keepdims=True)/tap.sum()
    p=np.abs(np.fft.rfft2((g-center)*tap))**2
    fy=np.fft.fftfreq(32)[:,None]*32; fx=np.fft.rfftfreq(32)[None,:]*32
    rad=np.sqrt(fx*fx+fy*fy); diag=np.minimum(np.abs(fx),np.abs(fy))/(rad+.001)
    def E(lo,hi): return (p*((rad>=lo)&(rad<hi))).sum((1,2))
    coarse=E(1,4); medium=E(4,8); small=E(8,15)
    isot=(p*((rad>3)&(diag>.26))).sum((1,2))/(medium+small+.01)
    return between((small+.05)/(medium+.1),.16,.76),between(medium/(coarse+1),.12,.74),between(isot,.29,.79)

def localmarks(x,g):
    r,v,b=x[:,0],x[:,1],x[:,2]; dark=between(g,.32,.09); pale=between(g,.59,.88)
    neutral=1-between((x.max(1)-x.min(1))/(g+.3),.07,.31)
    ochre=between((r-b)/(g+.22),.08,.30)*between((r-v)/(g+.22),-.06,.12)
    # A facial dark spot has lighter fur on both sides; a side-lit face
    # need not be bilaterally identical. The same arrangement with a long
    # flat lower radiator is penalized below by the object-family evidence.
    best=np.zeros(len(g),np.float32); nosebest=best.copy()
    for rad in (1,2,3):
        val=g if rad==1 else blurbox(g,rad-1); surround=blurbox(g,2*rad+1); con=between(surround-val,.025,.24)
        nos=con*(.48+.52*between(val,.43,.12)); eye=con
        for w in (rad+2,rad+4):
            a=shift(eye,-w,-w); bresp=shift(eye,-w,w)
            cheeks=.5*(shift(val,rad,-2*rad)+shift(val,rad,2*rad)); bridge=shift(val,-rad*2,0)
            support=between(.65*cheeks+.35*bridge-val,.028,.25)
            f=nos*support*(.43+.35*np.minimum(a,bresp)+.22*np.maximum(a,bresp))
            best=np.maximum(best,f[:,8:27,6:-6].max((1,2)))
            nosebest=np.maximum(nosebest,(nos*support)[:,9:26,8:24].max((1,2)))
    dc=blurbox(dark,2); wh=blurbox(pale*neutral,2); tn=blurbox(ochre,2); face=avg(dc[:,7:23,6:26]); wp=avg(wh[:,7:26,6:26]); brown=avg(tn[:,7:26,6:26])
    coat=between(np.sqrt(face*(wp+.48*brown)),.085,.28)
    # Pigment patches, in contrast to a few dark individual high-frequency spots.
    patches=avg((dc*dc+wh*wh)[:,7:25,5:27]); patches=between(patches,.08,.39)
    return best,nosebest,coat,patches

def habitat(x,g,sal):
    # Class-independent material indicators. Each is a proportion of pixels
    # explained by a material (not a probability model estimated from photos).
    r,v,b=x[:,0],x[:,1],x[:,2]; ma=x.max(1); mi=x.min(1); sat=ma-mi
    den=.22+g; cool=(b-r)/den; blue=between(cool,.015,.18)*between((b-v)/den,-.085,.055)
    leaves=between((v-b)/den,.03,.17)*between((v-r)/den,-.106,.054)
    grass=between((v-b)/den,.09,.32)*between((v-r)/den,-.17,.07)
    dirt=between((r-b)/den,.092,.32)*between((v-b)/den,.02,.18)*between((r-v)/den,.014,.12)
    neutral=1-between(sat/den,.075,.29)
    grayroad=neutral*between(g,.055,.23)*between(g,.94,.61)
    gold=between((r-b)/den,.12,.40)*between((v-b)/den,.10,.31)
    rawdx,rawdy=diff(g); noise=np.sqrt(rawdx*rawdx+rawdy*rawdy)
    micro=blurbox(np.abs(g-smooth(g)),2); smoothness=np.exp(-micro/.023)
    sky=between(b,.33,.72)*between((b-r)/den,-.09,.025)*between((b-v)/den,-.065,.018)*(.50+.50*smoothness)
    # Water includes dark turquoise, grey and brown water; for these colours
    # insist on long horizontal ripples or a clear change from the sky above.
    water=blue*(.55+.45*smoothness)
    colorpaint=between(sat/den,.20,.49)*np.maximum.reduce((between((b-v)/den,.065,.25),between((r-v)/den,.095,.28)*(1-between((v-b)/den,.045,.18)),between((r-b)/den,-.05,-.22)*between((v-r)/den,.12,.40)))
    icy=neutral*between(g,.73,.93)
    skyhi=avg(sky[:,:8]); skylo=avg(sky[:,-8:]); skyedges=.5*(avg(sky[:,:,:4])+avg(sky[:,:,-4:]))
    waterlo=avg(water[:,-9:]); watertop=avg(water[:,:9]); ro=avg(grayroad[:,-6:]); botgreen=avg(grass[:,-8:]); topgreen=avg(leaves[:,:9]); leafy=avg(leaves); ochre=avg(dirt); ochrelow=avg(dirt[:,-9:]); coolall=avg(blue); centerblue=avg(blue[:,9:23,9:23]); paint=avg(colorpaint[:,7:25,5:27]); gray=avg(neutral); dark=avg(between(g[:,6:27,6:27],.30,.08)); white=avg(icy[:,5:26,6:26]); bright=avg(g)
    center=avg(g[:,10:24,9:23]); fgcoat=avg(dirt[:,9:23,9:23]); cgreen=avg(leaves[:,9:24,8:24]); muted=1-avg(between(sat/den,.14,.37)); plain=(avg(smoothness[:,:8])+avg(smoothness[:,-8:]))*.5
    # Relative concentration of texture and colour variation (shallow depth of
    # field, a foreground subject filling the frame, or detailed ground cover).
    border=(avg(noise[:,:6])+avg(noise[:,-6:])+avg(noise[:,:,:5])+avg(noise[:,:,-5:]))*.25
    inedge=avg(noise[:,8:24,7:25]); isolate=np.clip((inedge-border)/(inedge+border+.02),-.65,.75)
    fgstd=x[:,:,6:25,6:26].std((2,3)).mean(1)
    topsd=x[:,:,:8].std((2,3)).mean(1); botsd=x[:,:,-8:].std((2,3)).mean(1)
    topflat=between(topsd,.215,.065); botflat=between(botsd,.21,.065); flat=(topflat+botflat)*.5
    # Horizontal straightness of boundary patches; a horizon is a coherent
    # full-width change rather than merely a horizontal texture element.
    stripe=x.reshape(len(x),3,32,4,8).mean(4)
    change=stripe[:,:,4:]-stripe[:,:,:-4]; dh=change.mean(3); coherent=(dh**2).mean(1)/( (change**2).mean((1,3))+.004)
    amp=np.sqrt((dh**2).mean(1)); horizon=(between(amp,.06,.26)*coherent)[:,3:26].max(1)
    ux=np.abs(rawdx); uy=np.abs(rawdy); strength=avg(noise)+.02
    horizontal=(avg(uy)-avg(ux))/strength
    linedraw=(avg(np.abs(boxline(rawdx,1,5)))+avg(np.abs(boxline(rawdy,2,5))))/strength
    line=between(linedraw,.47,.88); rip=avg(np.abs(boxline(rawdy,2,4))[:,-9:])/(avg(noise[:,-9:])+.024)
    yy=g.mean(2); midblack=between(yy[:,-4:].mean(1)-yy[:,19:28].min(1),.04,.26)
    shotsky=skyhi*(1-skylo); bluesea=waterlo*(.30+.96*skyhi)+.42*waterlo*(1-watertop)
    close=1-isolate; complexion=np.clip(plain-.35,-.35,.65); fog=avg(icy[:,:7])
    # Work in contrasts: a picture with neither affirmative nor contrary
    # evidence should not acquire a large score just from adding weak cues.
    # Colour match to the boundary is a soft measure of foreground occupancy.
    occ=1-np.exp(-sal*.67)
    wy=np.exp(-((np.arange(12,dtype=np.float32)-5.5)/4.5)**2)
    wgt=wy[:,None]*wy; mass=avg(occ*wgt)/wgt.mean()
    outer=.5*(avg(occ[:,:3])+avg(occ[:,-3:])); big=clip(mass*1.75-outer*.55)
    horconc=avg(occ[:,4:8])-avg(occ[:,1:4])*.5-avg(occ[:,8:11])*.5
    # Search for narrow leg-like vertical ridges of a consistent colour.
    leg=smooth(x); sides=(shift(g,0,-2)+shift(g,0,2))*.5
    rid=g-sides; cont=np.maximum(np.abs(blurbox(rid,1)),0)
    spine=np.abs(boxline(rid,1,2))/(boxline(np.abs(rid),1,2)+.026)
    shaft=between(np.abs(rid),.042,.22)*spine
    shaft=shaft[:,18:29,3:29].max(1); legs=0
    for dist in (6,10,14): legs=np.maximum(legs,np.sqrt(shaft[:,:-dist]*shaft[:,dist:]).max(1))
    # Fur gets directional fine ridges, rough skin gets isolated spots of
    # either sign. Check this inside the putative subject, not over sky.
    orig=g[:,7:26,6:26]; fine=g-smooth(g); mx,my=diff(g); tx=blurbox(mx*mx-my*my,2); ty=blurbox(2*mx*my,2)
    coherence=np.sqrt(tx*tx+ty*ty)/(blurbox(mx*mx+my*my,2)+.0014)
    fur=avg(coherence[:,7:26,6:26]); grain=avg(np.abs(fine[:,7:26,6:26]))
    wheels,df,cf=pairparts(g); turf=botgreen*(1-.8*cgreen)
    def C(z,origin): return z-origin
    # Natural kinds vs manufactured objects. Darkness below the object,
    # straight contours, sky and bare pavement are independent kinds of evidence.
    art=1.35*C(line,.5)+.8*C(horizon,.26)+1.55*C(skyhi,.22)+1.3*C(ro,.30)+.85*C(midblack,.35)-1.9*C(leafy,.20)-1.05*C(ochre,.22)+.90*horizontal+.55*C(paint,.12)
    # Local positive support within each family of objects. Center all
    # broadly active cues at a neutral/ambiguous value before combining.
    a=2.0*C(skyedges,.18)+1.2*C(skylo*skyhi,.08)+1.55*C(coolall,.16)+1.35*C(flat,.40)+1.7*C(isolate,.13)-.85*C(horizon,.28)-.95*horizontal-.76*C(midblack,.32)-.95*C(line,.48)+.6*C(topflat,.35)-.75*C(waterlo,.16)
    c=1.45*C(midblack,.32)+1.32*C(ro,.29)+1.0*C(paint,.12)+.72*C(wheels,.55)+.53*C(dark,.21)-1.10*C(skyhi,.25)-.95*C(waterlo,.16)-.72*C(leafy,.21)+.62*C(big,.47)+.52*C(horconc,.05)
    tr=.90*C(line,.48)+.78*C(ro,.29)+.92*C(shotsky,.14)+.67*C(white,.13)+.65*C(midblack,.32)+.42*C(wheels,.55)+.63*C(big,.48)-1.53*C(waterlo,.15)-.69*C(skylo,.18)-.68*C(leafy,.21)-.60*C(flat,.38)
    sh=2.9*C(bluesea,.14)+1.00*C(skyhi,.22)+1.55*C(horizon,.28)+1.85*horizontal+.75*C(rip,.27)+.57*C(flat,.38)-1.05*C(leafy,.20)-1.0*C(ochre,.19)-.93*C(midblack,.32)-.65*C(paint,.11)-.73*C(ro,.30)
    bi=1.7*C(flat,.40)+1.7*C(isolate,.14)+1.15*C(topgreen,.19)+.84*C(coolall,.15)+.6*C(paint,.12)-1.45*C(big,.50)-.85*C(midblack,.33)-.80*C(line,.47)-.7*horizontal-.40*C(turf,.18)
    ca=.75*C(muted,.42)+.53*C(gray,.42)+.57*C(cf,.35)+.85*C(fur,.32)+.62*C(big,.49)-.90*C(botgreen,.25)-.80*C(skyhi,.23)-.56*C(coolall,.15)-.50*C(horizon,.26)-.6*C(turf,.17)-.52*C(dark,.17)
    do=.95*C(df,.35)+.95*C(white,.13)+.95*C(dark,.17)+.70*C(big,.48)+.77*C(fgcoat,.22)+.68*C(turf,.18)+.35*C(isolate,.12)-.97*C(cgreen,.21)-.95*C(skyhi,.23)-.75*C(coolall,.15)-.56*C(line,.5)
    de=1.7*C(turf,.17)+1.5*C(topgreen,.20)+1.18*C(leafy,.20)+.85*C(ochrelow,.25)+.85*C(muted,.43)-.90*C(big,.50)-.88*C(paint,.13)-.55*C(white,.13)-.65*C(skyhi,.23)-.64*C(midblack,.34)+.68*C(legs,.51)
    ho=2.05*C(turf,.17)+.65*C(botgreen,.25)+.74*C(shotsky,.14)+.62*C(horizon,.27)+.78*C(fgcoat,.23)+.48*C(big,.50)+.66*C(dark,.17)-.63*C(cgreen,.20)-.55*C(waterlo,.15)+1.25*C(legs,.51)+.45*C(line,.47)
    fr=1.66*C(ochre,.20)+1.53*C(cgreen,.21)+.92*C(avg(gold),.22)+1.05*C(between(border,.030,.081),.48)+.93*C(between(grain,.016,.054),.39)-1.14*C(skyhi,.23)-.95*C(flat,.40)-.83*C(horizon,.28)-.92*C(turf,.17)-.71*C(coolall,.15)-.82*C(fur,.32)-.48*C(white,.13)
    # A likelihood of seeing an open water *region*, not just a bright line.
    # Check both lower corners independently, including their colour spread.
    q1=x[:,:,-8:,:6].mean((2,3)); q2=x[:,:,-8:,-6:].mean((2,3))
    agreement=np.exp(-((q1-q2)**2).mean(1)/.019); bluelo=avg(blue[:,-6:])
    opensurface=clip(.50*bluelo+.23*botflat+.27*agreement)
    marine=(.68*skyhi+.32*topflat)*opensurface
    sh+=1.55*C(marine,.20)+1.0*C(bluelo,.22)-1.1*horizon*(1-opensurface)-.65*midblack*(1-bluelo)-.65*paint*(1-bluelo)
    a+=.9*(skyhi*skylo)**.5*(1-horizon)+.70*skyedges*isolate-.47*marine
    c+=.75*horconc+.64*midblack*(1-skyhi)+.46*paint*big-.59*marine
    tr+=.65*(1-horconc)*shotsky*line+.57*big*line-.50*marine
    # A dog's fur may be almost any shade. Give support for *combinations*
    # such as a dark nose on a pale muzzle or a pied coat, rather than count
    # a single dark eye or a single white patch as enough.
    mask,nose,coat,piebald=localmarks(x,g); hi,mid,isotropic=spectraltexture(g)
    doglook=mask*(.6+.4*big)*(.5+.5*(1-line))
    do+=1.52*doglook+.68*coat+.5*piebald+.47*nose*(1-skyhi)+.43*big*(1-cgreen)-.35*marine-.35*cgreen
    ca+=.40*(1-coat)*fur+.41*mid*isotropic-.44*doglook-.42*piebald-.38*ochre*cgreen
    neartexture=clip((avg(micro[:,7:26,6:26])-.012)*22)
    fr+=1.12*neartexture*isotropic+.69*hi*isotropic+.54*mid*ochre+.31*(1-skyhi)*cgreen-.55*coat-.38*botgreen*(1-cgreen)
    bi+=.8*isolate+.65*flat*(1-big)+.55*flat*topgreen+.55*topflat*botflat+.43*paint*(1-line)-.37*coat
    de+=.62*turf*(1-big)+.60*muted*(1-skyhi)*turf-.5*big*coat
    ho+=.83*big*turf+.61*legs*shotsky+.52*legs*turf+.40*legs*fgcoat+.34*coat*turf-.40*cgreen*leafy
    # Context should select a family, not overpower the internal object match.
    add=.52*np.clip(art,-1.80,2.20)
    return np.stack((a+add,c+add,bi-add,ca-add,de-add,do-add,fr-add,ho-add,sh+add,tr+add),1)

def scene(x,g,dx,dy,sal):
    r,bg,b=x[:,0],x[:,1],x[:,2]
    ma=x.max(1); mi=x.min(1); sat=ma-mi
    def A(a): return avg(a)
    def high(a): return A(a[:,:9])
    def low(a): return A(a[:,-8:])
    def rim(a): return .5*(A(a[:,:,0:5])+A(a[:,:,-5:]))
    # Estimate which pixels might depict sky, water, plants, road, soil or coat.
    den=.26+g
    blue=between((b-r)/den,.018,.19)*between((b-bg)/den,-.035,.049)
    green=between((bg-b)/den,.018,.17)*between((bg-r)/den,-.062,.078)
    yellow=between((r-b)/den,.070,.30)*between((bg-b)/den,.045,.23)*between((r-bg)/den,-.08,.13)
    warm=between((r-bg)/den,.027,.18)*between((r-b)/den,.047,.26)
    gray=1-between(sat/den,.076,.24)
    red=between((r-bg)/den,.20,.49)*between((r-b)/den,.13,.39)
    vivid=between(sat/den,.32,.63)
    wh=between(g,.60,.85)*gray
    black=between(g,.27,.095)
    noise=np.abs(dx)+np.abs(dy); mag=np.sqrt(dx*dx+dy*dy)
    rough=smooth(smooth(noise)); flat=np.exp(-rough/.044)
    sky=(blue*(.45+.55*between(g,.24,.60))+gray*between(g,.60,.85)*.78)*( .18+.82*flat)
    topsky=high(sky); botsky=low(sky); skyall=rim(sky)
    water=blue*(.55+.45*flat); sea=low(water); grasses=low(green); leaves=high(green)
    earth=A(yellow+warm*.40); woods=A(green); road=low(gray*(.30+.70*flat))
    # Give central, boundary-dissimilar colours greater say about coat/paint than background.
    u=(np.arange(32,dtype=np.float32)+.5-16)/16
    center=np.exp(-2.7*(u[:,None]**2+u[None,:]**2)); j=np.minimum((np.arange(32)*12)//32,11)
    weight=center*(.35+np.minimum(sal[:,j[:,None],j[None,:]],3)); area=weight.sum((1,2))
    def fg(a): return (a*weight).sum((1,2))/area
    fgr=fg(green); fw=fg(warm); fgray=fg(gray); fred=fg(red); fv=fg(vivid); fblue=fg(blue)
    white=fg(wh); ink=fg(black); redpaint=fg(red*between((bg-b)/den,.18,0))
    axisx=np.abs(dx); axisy=np.abs(dy); ene=A(mag)+.018
    horiz=(A(axisy)-A(axisx))/ene
    align=A(np.abs(axisy-axisx)-np.minimum(axisx,axisy)*.75)/ene
    longh=A(np.abs(boxline(dy,2,3)))/ene; longv=A(np.abs(boxline(dx,1,3)))/ene
    lines=(align-.33)*1.7+(longh+longv-.62)*1.65
    linbottom=low(np.abs(boxline(dy,2,3)))/(low(mag)+.024)
    topbottom=high(g)-low(g)
    edges=A(mag[:,7:25,6:26]); frame=.35*high(mag)+.30*low(mag)+.35*rim(mag)
    isolated=np.clip((edges-frame)/(frame+edges+.020),-.6,.8)
    outerflat=(high(flat)+low(flat)+rim(flat))*.3333
    rows=g[:,:,4:-4].mean(2); prof=boxline(rows[:,None,:],2,1)[:,0]
    darkband=np.minimum(prof[:,-4:].mean(1),prof[:,8:17].mean(1))-prof[:,18:28].min(1)
    shadow=between(darkband,.035,.24)
    # Distinguish green foreground on earth from a non-green animal on grass.
    field=grasses*(1-.66*fgr); wild=(leaves+grasses)*.5; pasture=field*(.65+.65*topsky)
    # Relative strength of fine detail inside versus a smooth outline.
    fine=A(np.abs(g-smooth(g)))/(A(mag)+.025)
    busy=between(frame,.029,.094); stipple=between(fine,.28,.60)
    paleground=low(wh); backdrop=(high(blue)+rim(blue))*.5
    # The straight waterline tends to cut horizontally through the picture;
    # the sky behind flying objects instead continues down the sides.
    split=topsky*(1-botsky); bothsky=(topsky*botsky)**.5
    wet=(.57+1.6*topsky)*sea
    mixed=(fg(black)*fg(wh))**.5
    ftexture=fg(rough)
    manufactured=1.70*lines+1.10*horiz+1.05*shadow+.80*road+.55*topsky-1.35*woods-.56*earth-.45*stipple
    groundanimal=1.75*field+.70*leaves+.55*yellow.mean((1,2))-.95*sea-.80*bothsky
    a= -.30+2.35*bothsky+.85*topsky+.9*backdrop+.85*skyall+.86*outerflat+.85*isolated-.47*shadow-.56*horiz-.62*busy-.40*woods-.42*earth-.24*wet
    c= -.27+manufactured+.55*road+.70*redpaint+.36*fv+.41*ink+.28*isolated-.45*topsky-.55*wet-.43*sea-.26*grasses
    d= .19+.75*leaves+.46*backdrop+.71*outerflat+.77*isolated+.57*fv+.19*fblue-.71*lines-.62*shadow-.45*horiz-.52*wet-.32*topbottom-.31*field
    t= .45+.37*fgray+.53*fw+.24*stipple+.37*busy+.19*white-.61*wild-.71*topsky-.51*sea-.73*lines-.44*horiz-.35*shadow-.34*isolated
    e= .17+groundanimal+.42*earth+.36*leaves+.31*fw-.55*fv-.85*redpaint-.25*mixed-.42*topsky-.44*lines-.31*shadow
    dog= .33+.63*fw+.36*white+.85*mixed+.44*field+.21*isolated-.65*fgr-.67*sea-.48*topsky-.61*lines-.32*shadow-.20*horiz
    f= .12+1.45*fgr+.82*earth+.78*busy+.73*stipple+.58*fw-.94*topsky-.57*blue.mean((1,2))-.55*shadow-.57*lines-.44*field-.67*outerflat-.30*white
    h= .13+groundanimal+.60*ink+.47*fw+.65*split+.50*topbottom+.37*shadow+.31*isolated-.48*fgr-.29*stipple-.20*lines
    s= -.20+2.20*wet+.98*topsky+.69*split+.93*horiz+.93*(linbottom-.29)+.22*isolated+.29*outerflat-.37*shadow-.82*woods-.61*earth-.33*redpaint
    tr= -.32+manufactured+.88*split+.75*topbottom+.60*white+.45*(longv-.26)+.36*road-.48*sea-.43*wet-.32*outerflat
    return np.stack((a,c,d,t,e,dog,f,h,s,tr),1)

def classify(x):
    xx=enlarge(x); rgb=pool(xx)
    g=.27*x[:,0]+.55*x[:,1]+.18*x[:,2]
    y=.27*xx[:,0]+.55*xx[:,1]+.18*xx[:,2]
    dx,dy=diff(smooth(y)); h=angles(dx,dy)
    for opponent in (xx[:,0]-xx[:,1],xx[:,2]-xx[:,1]):
        gx,gy=diff(smooth(opponent)); h+=.38*angles(gx,gy)
    h=pool(smooth(h)); h=np.sqrt(h+.0015)-np.float32(np.sqrt(.0015)); h-=h.mean((2,3),keepdims=True); h=norm(h,.02)
    sal=borderdiff(rgb); ysmall=pool(y); val=stretch(ysmall).reshape(len(x),-1); salvec=stretch(sal).reshape(len(x),-1)
    response=10.5*(h.reshape(len(x),-1)@CONSTANTS['h'])
    m=CONSTANTS['m']; lumin=np.abs(val@m)
    # A bird against a similarly bright leaf can have an almost invisible
    # luminance boundary. Treat either opponent axis as an alternative.
    chrom=[]
    for aa,bb in ((0,1),(1,2)):
        d=rgb[:,aa]-rgb[:,bb]; d-=d.mean((1,2),keepdims=True)
        chrom.append(np.abs(norm(d,.60).reshape(len(x),-1)@m))
    response+=2.3*np.maximum(lumin,np.maximum(chrom[0],chrom[1]))+2.25*(salvec@m)+3.25*(val@CONSTANTS['p'])
    scores=[]; start=0
    for end in CONSTANTS['ends']:
        sub=response[:,start:end]; best=sub.max(1)
        # Pool over nearby explanations; avoid relying entirely on one accidental alignment.
        scores.append(best+.29*np.log(np.mean(np.exp((sub-best[:,None])*2.4),axis=1)))
        start=end
    dx,dy=diff(g)
    shape=np.stack(scores,1)
    # Shape is useful, but a local accidental alignment is not by itself
    # good evidence against the rest of the picture.
    # Pool contextual rules independently of how many there are for a class.
    # The second pass emphasizes agreement rather than stacking near-duplicates.
    return .87*shape+.39*scene(x,g,dx,dy,sal)+1.27*habitat(x,g,sal)

def predict(x):
    # Fixed matrix dimensions also for the last chunk, so the numerical work
    # on one item does not depend on the caller's choice of batch size.
    result=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):
        b=np.zeros((32,3,32,32),np.float32); n=min(32,len(x)-i); b[:n]=x[i:i+n]
        result[i:i+n]=classify(b)[:n].argmax(1)
    return result
