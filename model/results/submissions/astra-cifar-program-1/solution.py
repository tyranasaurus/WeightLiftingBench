import numpy as np

def clip(a):
    return np.clip(a,0.,1.)

def avg(a):
    return a.mean(axis=(1,2))

def cells(z,k):
    return z.reshape(-1,32//k,k,32//k,k).mean((2,4))

def unit(z):
    a=z.reshape(len(z),-1)
    return a/np.maximum(.0001,np.sqrt(np.sum(a*a,axis=1,keepdims=True)))

def gradients(a):
    q=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    dx=q[:,1:-1,2:]-q[:,1:-1,:-2];dy=q[:,2:,1:-1]-q[:,:-2,1:-1]
    mag=np.sqrt(dx*dx+dy*dy)
    return dx,dy,mag

def smooth(a,k):
    q=np.pad(a,((0,0),(k,k),(k,k)),mode='edge');n=2*k+1
    v=sum(q[:,i:i+a.shape[1],:] for i in range(n))/n
    return sum(v[:,:,j:j+a.shape[2]] for j in range(n))/n

def drawing(Y,dx,dy,mag,foreground,R,G,B):
    center=cells(mag,4);total=center.mean((1,2),keepdims=True)
    bins=[]
    for i in range(6):
        pr=np.abs(np.cos(i*np.pi/6)*dx+np.sin(i*np.pi/6)*dy)
        m=cells(np.maximum(pr-.82*mag,0)/.18,4);bins.append(np.sqrt(m/(center+.6*total+.0001)))
    hog=np.stack(bins,1);hog-=.70*hog.mean((2,3),keepdims=True)+.30*hog.mean((1,2,3),keepdims=True)
    a=cells(Y,2);d=a-smooth(a,2)
    inp=np.concatenate((unit(hog)*.53,unit(d)*.47),1).astype(np.float32)
    # Use fixed product dimensions even for the last (partially filled) group.
    buf=np.zeros((40,640),np.float32);buf[:len(Y)]=inp
    z=(buf@CONSTANTS['shapes'])[:len(Y)]
    fsmall=cells(foreground,2); fsmall-=fsmall.mean((1,2),keepdims=True)
    b=np.zeros((40,256),np.float32); b[:len(Y)]=unit(fsmall)
    z+=.21*(b@CONSTANTS['outlines'])[:len(Y)]
    # Compare each candidate silhouette to its own local surround in the
    # luminance and colour-opponent planes. The sign may vary with plumage,
    # coat or paint. Contrast along one channel alone can identify a boundary.
    dif=np.zeros_like(z)
    for v,w in [(Y,1.),(R-G,.40),(G-B,.50)]:
        plane=cells(v,2).reshape(len(Y),256); dev=np.sqrt(((plane-plane.mean(1,keepdims=True))**2).mean(1,keepdims=True))
        b[:len(Y)]=plane/(.073+dev)
        col=(b@CONSTANTS['regions'])[:len(Y)]
        dif+=w*col*col
    z=(z+.107*np.minimum(2.4,np.sqrt(dif))).reshape(-1,10,4,144)
    # Object likelihood marginalised over the best few poses and view types.
    peaks=np.partition(z, -5, axis=3)[:,:,:,-5:].mean(3)
    peaks[:,[0,1,2,4,6,7,8,9],3]=-1.
    p=np.sort(peaks,axis=2)
    s=.83*p[:,:,3]+.17*p[:,:,2]
    return s-s.mean(1,keepdims=True)

def sides(a):
    return .5*(avg(a[:,:,:5])+avg(a[:,:,-5:]))

def mid(a):
    return avg(a[:,8:25,7:25])

def top(a):
    return avg(a[:,:7])

def foot(a):
    return avg(a[:,-6:])

def inner(a):
    return avg(a[:,5:27,4:28])

def strongest(a,n):
    b=a.reshape(len(a),-1);return np.partition(b,-n,axis=1)[:,-n:].mean(1)

def shift(a,y,x):
    return np.roll(a,(y,x),axis=(1,2))

def background(R,G,B,Y):
    # Compare each patch to colours on the image boundaries. Also allow
    # different ground and sky colours, instead of positing a uniform backdrop.
    sm=np.stack([smooth(Y,1),smooth(R-G,1)*1.3,smooth(B-G,1)*1.35],axis=1)
    best=np.ones(Y.shape,np.float32); horizontal=np.ones(Y.shape,np.float32)
    yy=np.arange(32,dtype=np.float32).reshape(1,32,1)
    for u,v in [(2,2),(2,10),(2,21),(2,29),(29,2),(29,10),(29,21),(29,29),(10,2),(10,29),(20,2),(20,29)]:
        ref=sm[:,:,u-1:u+2,v-1:v+2].mean((2,3),keepdims=True)
        dist=np.sqrt(((sm-ref)**2).sum(1));best=np.minimum(best,dist);horizontal=np.minimum(horizontal,dist+.008*np.abs(yy-u))
    return .65*(1.-np.exp(-best/.125))+.35*(1.-np.exp(-horizontal/.18))

def parts(Y,dx,dy,mag,fg):
    A=smooth(Y,1);detail=avg(mag)[:,None,None]
    wheel=np.zeros(len(Y),np.float32)
    for r in (2,3,4):
        t=r+1
        left=shift(A,0,t);right=shift(A,0,-t);up=shift(A,t,0);down=shift(A,-t,0)
        ring=(shift(Y,0,r-1)+shift(Y,0,1-r)+shift(Y,r-1,0)+shift(Y,1-r,0))*.25
        tire=.74*ring+.26*A
        dark=.55*((left+right+up+down)*.25-tire)+.45*(np.minimum((left+right)*.5,(up+down)*.5)-tire)
        hub=clip((Y-ring)/(.09+detail*.4));d=clip((dark+.030*hub-.024)/(.13+detail*.4))
        for gap in (8,12,16,20):
            for slope in (-2,0,2):
                right=shift(d,slope,-gap);pair=np.minimum(d,right)*.74+.26*(d+right)*.5
                wheel=np.maximum(wheel,strongest(pair[:,15:28,2:31-gap],3))
    # Paired dark eyes and a dark nose below and approximately between them.
    eyes=Y*0
    for r in [2,3]:
        across=(shift(A,0,r)+shift(A,0,-r))*.5-Y
        along=(shift(A,r,0)+shift(A,-r,0))*.5-Y
        eyes=np.maximum(eyes,clip((.55*np.minimum(across,along)+.225*(across+along)-.025)/(.115+.32*detail)))
    face=np.zeros(len(Y),np.float32);dogface=face.copy()
    for d in (2,3,4,5,6):
        eye=np.minimum(shift(eyes,0,d),shift(eyes,0,-d)); nose=shift(eyes,-d,0)
        bilateral=1.-clip(np.abs(shift(A,0,d)-shift(A,0,-d))/.17)
        shape=eye*(.55+.45*bilateral)*(.38+.62*nose)
        face=np.maximum(face,strongest(shape[:,6:22,7:25],2))
        longnose=shift(smooth(eyes,1),-int(1.65*d),0)
        dogface=np.maximum(dogface,strongest((eye*(.38+longnose)*bilateral)[:,6:22,7:25],2))
    # Narrow upright bars with opposite-signed aligned edges: legs below a torso.
    v=sum(shift(dx,i,0) for i in range(-3,4))/7
    bar=np.maximum(-v*shift(v,0,2),-v*shift(v,0,4)); bar=np.sqrt(np.maximum(bar,0))
    leg=strongest(bar[:,17:28,3:30],16)/(.09+avg(mag))
    # Long straight edges, compared to total contour length at this scale.
    h=sum(shift(dy,0,i) for i in range(-4,5))/9
    h=np.abs(h[:,:,4:-4]);v=np.abs(v[:,3:-3,:]);energy=avg(mag)+.022
    horiz=avg(h)/energy;vert=avg(v)/energy
    box=strongest(v[:,:,4:28],18)/energy
    hull=strongest(h[:,13:27,:].mean(2),2)/energy
    return wheel,face,dogface,clip((leg-.42)/1.2),horiz,vert,clip((box-1.3)/2.0),clip((hull-.65)/1.9)

def material(Y,R,G,B,mag,fg):
    # Select a boundary band on each side of the scene. A smooth chromatic
    # ramp across several sides is still a single sky, not a band of water.
    tone=np.stack((R,G,B),1); bd=np.concatenate((tone[:,:,:4,:].reshape(len(Y),3,-1),tone[:,:,-4:,:].reshape(len(Y),3,-1),tone[:,:,4:28,:3].reshape(len(Y),3,-1),tone[:,:,4:28,-3:].reshape(len(Y),3,-1)),2)
    # Robust interquartile colour range excludes small twigs, masts, etc.
    z=np.partition(bd,[bd.shape[2]//4,3*bd.shape[2]//4],axis=2)
    spread=(z[:,:,3*bd.shape[2]//4]-z[:,:,bd.shape[2]//4]).mean(1)
    uniform=1.-clip((spread-.045)/.22)
    quiet=(1.-clip((.32*top(mag)+.28*foot(mag)+.40*sides(mag)-.025)/.145))
    # The best evidence for a flying silhouette is an object surrounded on
    # all four sides by an almost empty sky, not simply a blue pixel.
    empty=uniform*quiet*clip((mid(mag)+mid(fg)*.18-.048)/.15)
    # Short periodic ripples: consecutive along-row intensity changes are
    # usually smaller than along-column ones in the water below a ship.
    rip=abs(Y[:,24:31,:]-Y[:,25:32,:]).mean((1,2)); along=abs(Y[:,24:,1:]-Y[:,24:,:-1]).mean((1,2))
    ripple=clip((rip-along-.003)/.038)
    # Dark glazing above a brighter bonnet, contiguous across a substantial
    # portion of the car width (not just two eye spots).
    A=smooth(Y,1);stripe=np.maximum(0,shift(A,-4,0)-A)*np.maximum(0,shift(A,4,0)-A)
    stripe=np.sqrt(stripe); st=(sum(shift(stripe,0,i) for i in range(-4,5))/9)[:,6:19,5:27]
    glass=clip((strongest(st,6)-.045)/.21)
    # A truck's large cargo panels often contain flat interiors, including
    # saturated paint. Look for long horizontal + vertical borders enclosing
    # a flat patch, by comparing whole rows/columns just inside and outside.
    panel=np.zeros(len(Y),np.float32)
    for h,w in [(5,9),(7,6),(7,11),(4,13)]:
        # Integral image gives fixed rectangular means in each tested box.
        a=np.pad(Y,((0,0),(1,0),(1,0))).cumsum(1).cumsum(2)
        def box(y,x,bh,bw):
            return (a[:,y+bh:y+bh+12,x+bw:x+bw+16]-a[:,y:y+12,x+bw:x+bw+16]-a[:,y+bh:y+bh+12,x:x+16]+a[:,y:y+12,x:x+16])/(bh*bw)
        body=box(5,2,h,w);topedge=abs(box(3,2,2,w)-box(5,2,2,w));base=abs(box(5+h,2,2,w)-box(3+h,2,2,w));left=abs(box(5,0,h,2)-box(5,2,h,2));right=abs(box(5,2+w,h,2)-box(5,w,h,2));
        flat=abs(box(5,2,h,2)-body)+abs(box(5,w,h,2)-body)+abs(box(5,2,2,w)-body)+abs(box(3+h,2,2,w)-body)
        v=np.minimum(np.maximum(topedge,base),np.maximum(left,right))+.34*(np.minimum(topedge,base)+np.minimum(left,right))-.46*flat
        panel=np.maximum(panel,clip((strongest(v,4)-.055)/.23))
    return empty,ripple,glass,panel

def fineparts(Y,dx,dy,mag,fg,R,G,B):
    eps=np.float32(.001); A=smooth(Y,1);x=np.arange(32,dtype=np.float32)-15.5
    # A rectangular cargo side often has a large uniform patch ending in
    # two strong straight edges. Penalize large patches of background.
    panel=np.zeros(len(Y),np.float32)
    for w,h in [(8,5),(5,8)]:
        vc=smooth(abs(dx),2);hc=smooth(abs(dy),2)
        l=shift(vc,0,w//2+1);r=shift(vc,0,-w//2-1);u=shift(hc,h//2+1,0);d=shift(hc,-h//2-1,0)
        interior=smooth(mag,2); enclosing=.50*np.minimum(l+r,u+d)+.25*(l+r+u+d)
        hit=clip((enclosing-1.45*interior-.06)/.22)*(.3+.7*fg)
        panel=np.maximum(panel,strongest(hit[:,6:22,5:27],8))
    # Detect painted metal above black rubber/shadow; a white or grey
    # passenger car need not have a conspicuous paint hue.
    red=clip((R-G-.08)/.19)*clip((R-B-.07)/.15);bright=clip((Y-.53)/.29)
    steel=clip((B-R+.03)/.16)*clip((Y-.24)/.20)
    paint=smooth(np.maximum(red,np.maximum(bright,.72*steel)),1)
    dark=smooth(clip((.27-Y)/.22),1);fig=smooth(fg,1)
    flank=strongest((paint*(.45+.55*fig)*shift(dark,-4,0))[:,11:24,3:29],22)
    # Bilateral correspondence strongest near a face, allowed to be off centre.
    # Match contrast rather than raw brightness to suppress flat background.
    local=A-smooth(Y,3); sym=np.zeros(len(Y),np.float32)
    for col in [12,14,16,18,20]:
        l=local[:,4:24,col-9:col];r=local[:,4:24,col:col+9][:,:,::-1]
        norm=np.sqrt(avg(l*l)*avg(r*r))+.00035
        sym=np.maximum(sym,avg(l*r)/norm)
    # Determine whether the mouth/nose region is a single broad dark patch
    # between two cheeks. Cats have a much smaller nose than most dogs.
    nose=np.zeros(len(Y),np.float32)
    for r in (2,3,4):
        snout=(shift(A,0,r)+shift(A,0,-r))*.5-A
        bridge=shift(A,r,0)-A; chin=shift(A,-r,0)-A
        spot=clip((.5*snout+.35*bridge+.15*chin-.048)/.18)*clip((.42-A)/.24)
        nose=np.maximum(nose,strongest(spot[:,12:25,9:23],3))
    # Smoothly coherent direction is characteristic of corners and sheet
    # metal; scattered unoriented high frequency detail suggests fur/skin.
    vxx=smooth(dx*dx,1);vyy=smooth(dy*dy,1);vxy=smooth(dx*dy,1)
    anis=np.sqrt((vxx-vyy)**2+4*vxy*vxy)/(vxx+vyy+.0009)
    fur=avg((1.-anis[:,7:26,7:25])*clip(mag[:,7:26,7:25]/.16))
    # Rough brown/olive substratum and mottled skin: agreement in hue
    # between centre and surround can distinguish a camouflaged toad.
    chrom1=(R-G)/(.20+Y);chrom2=(G-B)/(.20+Y)
    dist=np.abs(mid(chrom1)-sides(chrom1))+np.abs(mid(chrom2)-sides(chrom2))
    match=1.-clip((dist-.04)/.24)
    # Birds and small animals may be isolated by defocus (sharp centre vs
    # smooth surround), even where colour segmentation is ambiguous.
    focus=clip((mid(mag)-.46*sides(mag)-.36*top(mag)-.035)/.14)
    # Horizontal undulations or reflections below a comparatively placid sky.
    wet=clip((foot(np.abs(dy))-foot(np.abs(dx))-.009)/.074)
    return panel,flank,sym,nose,fur,match,focus,wet

def classify(batch):
    R,G,B=batch[:,0],batch[:,1],batch[:,2];Y=.28*R+.57*G+.15*B
    dx,dy,mag=gradients(Y);fg=background(R,G,B,Y)
    sh=drawing(Y,dx,dy,mag,fg,R,G,B)
    wheel,face,dogface,leg,horiz,vert,box,hull=parts(Y,dx,dy,mag,fg)
    hi=np.maximum(np.maximum(R,G),B);lo=np.minimum(np.minimum(R,G),B);sat=(hi-lo)/(.13+hi)
    veg=clip((G-B-.012)/(.042+.09*Y))*clip((G-R+.061)/(.045+.027*Y))
    blue=clip((B-R-.012)/(.045+.13*Y))*clip((B-G+.015)/(.032+.042*Y))
    warm=clip((R-B-.016)/(.06+.15*Y))*clip((R-G+.012)/(.046+.045*Y))
    paint=clip((R-G-.09)/(.11+.13*G))*clip((R-B-.085)/.17)
    leaf=clip((G-R-.021)/(.073+.09*Y))*clip((G-B-.064)/.15)
    gray=1.-clip((sat-.066)/.22)
    pale=clip((Y-.52)/.29)*(1.-clip((sat-.10)/.38))
    white=clip((Y-.67)/.22)*(1.-clip((sat-.09)/.28))
    black=clip((.27-Y)/.19); hot=clip((sat-.37)/.30)
    fine=np.abs(Y-smooth(Y,1)); texture=avg(fine)
    eco=top(veg); grass=foot(veg);forest=sides(veg);undergrowth=.4*grass+.6*forest
    bluesky=top(blue);water=foot(blue);bluebg=.36*bluesky+.32*water+.32*sides(blue)
    quiettop=1.-clip((top(mag)-.022)/.15);quietbottom=1.-clip((foot(mag)-.025)/.16)
    quietside=1.-clip((sides(mag)-.027)/.16);quiet=.38*quiettop+.34*quietbottom+.28*quietside
    # Similarity of the upper and lower border. Sky all around is evidence for
    # flight, whereas a colour/brightness transition suggests a water surface.
    dif=(np.abs(top(R)-foot(R))+np.abs(top(G)-foot(G))+np.abs(top(B)-foot(B))) / 3
    alike=1.-clip((dif-.035)/.23)
    road=foot(gray)*(.45+.55*quietbottom)*(1.-.70*foot(white))
    sky=bluesky+.75*top(pale)*(1.-bluesky)
    soildark=.52*sides(black)+.24*top(black)+.24*foot(black)
    outerwarm=.4*sides(warm)+.3*top(warm)+.3*foot(warm)
    # Competing scene hypotheses. Priors are qualitative habitat/usage rules,
    # not measurements. Mixtures let, e.g., an aircraft be aloft OR on a runway.
    air=3.4*quiet*(.29+.71*alike)+2.7*bluebg+.48*sky-1.2*undergrowth-.9*(1.-quietbottom)
    sea=3.9*water+1.15*quietbottom+.95*sky+.78*clip((top(Y)-foot(Y)-.02)/.30)+.45*hull-2.0*grass-.90*outerwarm
    street=2.90*road+.55*top(pale)+.44*sky+.5*clip((horiz-.24)/.34)-.86*water-1.0*undergrowth
    field=2.85*grass+1.1*forest+.40*eco+.44*sky-.92*water-.48*soildark
    woods=2.15*forest+1.05*eco+.58*outerwarm+.72*undergrowth-.66*sky-.66*water
    understory=1.4*outerwarm+1.25*undergrowth+1.4*soildark+.75*clip((sides(fine)-.022)/.050)-1.0*sky-1.0*water
    home=1.78*outerwarm+.65*sides(gray)+.70*(1.-quietside)+.46*soildark-.95*grass-.75*sky-.85*water
    garden=.82*undergrowth+.83*outerwarm+.66*quietside+.52*forest+.49*eco-.2*water
    plain=2.4*quiet*(.42+.58*alike)+.55*(top(pale)+foot(pale))-.7*forest
    env=np.stack((air,sea,street,field,woods,understory,home,garden,plain),1)
    priors=np.array([[67,1,18,5,1,0,0,1,7],[1,0,75,2,1,0,6,4,11],[8,8,0,8,27,3,3,33,10],[0,0,2,1,3,6,65,18,5],[0,0,1,36,46,7,1,7,2],[0,1,6,12,4,2,48,22,5],[0,2,0,1,17,55,8,15,2],[0,0,4,68,7,2,7,11,1],[4,89,2,0,0,0,0,0,5],[2,0,79,6,1,0,2,3,7]],np.float64)/100.
    # Explicit per-image reduction; changing batch size must not change a score.
    envscore=np.log((np.exp(env)[:,None,:]*priors[None,:,:]).sum(2))
    s=12.5*sh+1.05*envscore
    middle=mid(Y);darkmid=mid(black);lightmid=mid(white);warmmid=mid(warm)
    centergreen=mid(leaf);green=avg(veg);graymid=mid(gray);redmid=mid(paint)
    centerhot=mid(hot);centerblue=mid(blue);obj=mid(fg);edge=avg(mag)
    contrast=clip((Y.std((1,2))-.115)/.15);smoothbg=.5*quietside+.5*quiettop
    fur=clip((mid(fine)-.021)/.040);rough=clip((texture-.018)/.040)
    shadow=clip((np.maximum(avg(Y[:,8:14]),foot(Y))-avg(Y[:,20:25])-.016)/.18)
    gloss=clip((lightmid+.9*darkmid-.07)/.34)
    hor=clip((horiz-.22)/.34);ver=clip((vert-.23)/.29)
    rect=clip((horiz+vert-.49)/.40)
    # Horizontal extent and aspect of the foreground evidence relative to its
    # own centroid; use foreground only when it has a clear centre/surround.
    xx=(np.arange(32,dtype=np.float32)-15.5)/16
    sal=np.maximum(fg-.26,0)**2+.01;norm=sal.sum((1,2));mx=(sal*xx).sum((1,2))/norm;my=(sal*xx[:,None]).sum((1,2))/norm
    wx=np.sqrt(np.maximum(.001,(sal*(xx**2)).sum((1,2))/norm-mx**2));wy=np.sqrt(np.maximum(.001,(sal*(xx[:,None]**2)).sum((1,2))/norm-my**2))
    long=clip((wx/(wy+.001)-1.12)/.77)*clip((obj-.22)/.41)
    compact=clip((.49-(wx+wy)*.5)/.17)*clip((obj-.22)/.38)
    full=clip((obj-.32)/.37)*(1.-compact)
    # Shared rigid-body evidence and animal appearance evidence.
    machine=.68*(rect-.47)+.44*(hor-.48)+.49*(gloss-.44)+.40*(contrast-.48)+.40*(long-.22)-.32*(rough-.42)
    pet=.37*(warmmid-.40)+.37*(fur-.44)+.41*(face-.49)+.38*(full-.38)-.48*(long-.23)
    wild=.37*(warmmid-.39)+.28*(compact-.28)-.44*(centerblue-.11)-.38*(rect-.47)
    s[:,0]+=machine*.45+.94*(long-.22)+.67*(quiet-.44)+.38*(alike-.48)-.60*(hor-.48)-.44*(ver-.45)-.46*(wheel-.48)+.25*(centerblue-.15)
    s[:,1]+=machine+.98*(wheel-.45)+.72*(shadow-.37)+.69*(redmid-.073)+.64*(hor-.49)-.46*(box-.60)-.35*(ver-.45)+.39*(long-.22)
    s[:,9]+=machine+.70*(wheel-.45)+.72*(shadow-.37)+1.05*(box-.60)+1.15*(ver-.45)+.58*(lightmid-.13)+.41*(full-.38)
    s[:,8]+=.55*machine+1.22*(hull-.40)+1.05*(hor-.49)-.65*(wheel-.45)+.42*(long-.22)+.34*(sky-.35)
    s[:,2]+=.7*wild+1.08*(compact-.26)+.68*(smoothbg-.44)+.68*(centerhot-.18)+.42*(centerblue-.12)-.69*(leg-.35)-.41*(shadow-.37)
    s[:,3]+=pet+.76*(face-.49)+.55*(graymid-.43)+.47*(fur-.44)-.65*(leg-.35)-.43*(wheel-.45)-.32*(centergreen-.05)
    s[:,5]+=pet+.80*(dogface-.36)+.43*(gloss-.44)+.37*(contrast-.48)-.45*(green-.25)+.25*(lightmid-.13)
    s[:,4]+=wild+1.01*(leg-.35)+.52*(warmmid-.39)+.52*(compact-.26)-.56*(contrast-.48)-.44*(gloss-.44)-.50*(redmid-.073)-.24*(hor-.49)
    chestnut=clip((mid(R-G)-.032)/.12)*clip((mid(G-B)+.01)/.08)
    s[:,7]+=wild+1.50*(leg-.35)+.62*(chestnut-.26)+.45*(gloss-.44)+.55*(shadow-.37)-.30*(rough-.42)+.39*(full-.38)
    s[:,6]+=.66*wild+1.57*(centergreen-.065)+.91*(soildark-.17)+.76*(rough-.42)+.48*(centerhot-.18)-.61*(leg-.35)-.42*(ver-.45)-.46*(sky-.35)
    panel,flank,sym,nose,disorder,match,focus,wet=fineparts(Y,dx,dy,mag,fg,R,G,B)
    # More diagnostic conjunctions rather than treating all individual cues
    # as independent. A broad bumper, for example, is not just a bright spot.
    s[:,1]+=.68*(flank-.43)+.58*wheel*redmid-.31*(panel-.28)
    s[:,9]+=.79*(panel-.28)+.42*(flank-.43)+.36*box*lightmid
    s[:,8]+=.86*wet*(.4+.6*water)+.46*water*lightmid-.22*(flank-.43)
    s[:,0]+=.50*focus*quiet+.43*quiet*alike*(1.-hor)
    s[:,2]+=.85*(focus-.34)+.56*compact*(1.-hor)+.40*centerhot*(1.-match)
    s[:,3]+=.58*(sym-.27)+.52*sym*face+.50*(disorder-.20)-.35*(nose-.48)
    s[:,5]+=.60*(sym-.27)+.61*(nose-.48)+.70*sym*nose+.24*(disorder-.20)
    s[:,4]+=.78*match*(forest+.4*grass)*(.5+.5*compact)-.32*(flank-.43)
    s[:,7]+=.74*leg*(1.-match)+.44*chestnut*(1.-match)-.34*(panel-.28)
    s[:,6]+=.65*match*(rough+outerwarm)*.5+.75*(disorder-.20)+.56*centergreen*centerhot
    empty,ripple,glazing,cargo=material(Y,R,G,B,mag,fg)
    s[:,0]+=1.70*empty*(.55+.45*bluebg)+.49*empty*(1.-hull)
    s[:,8]+=.78*ripple*(.3+.7*water)-.77*empty
    s[:,2]+=.61*empty*(.4+.6*compact)
    s[:,1]+=.63*(glazing-.40)-.46*cargo*(1.-glazing)
    s[:,9]+=.98*(cargo-.35)+.32*cargo*box
    return np.argmax(s,axis=1)

def predict(x):
    out=np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),40):
        out[i:i+40]=classify(np.ascontiguousarray(x[i:i+40]))
    return out
