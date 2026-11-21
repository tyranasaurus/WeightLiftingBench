"""Construct detectors from geometric drawings and analytic filters; no input files."""
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)

def make_image(c, mode, p):
    U,V = base_x,base_y
    X=(U-p[0])/p[2]*p[6]; Y=(V-p[1])/p[3]
    ca,sa=np.cos(p[4]),np.sin(p[4]); x=ca*X+sa*Y; y=-sa*X+ca*Y
    x=x+p[5]*y
    im=np.zeros(x.shape,dtype=np.float32)
    def poly(points,col):
        ins=np.zeros(x.shape,bool); a,b=points[-1]
        for u,v in points:
            if v!=b: ins^=((b>y)!=(v>y))&(x<(u-a)*(y-b)/(v-b)+a)
            a,b=u,v
        im[ins]=col
    def ell(u,v,rx,ry,col,ang=0):
        co,si=np.cos(ang),np.sin(ang); a=(x-u)*co+(y-v)*si; b=(y-v)*co-(x-u)*si
        im[(a/rx)**2+(b/ry)**2<1]=col
    def seg(a,b,w,col,w2=None):
        if w2 is None: w2=w
        dx=b[0]-a[0]; dy=b[1]-a[1]; le=(dx*dx+dy*dy)**.5
        nx,ny=-dy/le,dx/le
        poly([(a[0]+nx*w,a[1]+ny*w),(b[0]+nx*w2,b[1]+ny*w2),(b[0]-nx*w2,b[1]-ny*w2),(a[0]-nx*w,a[1]-ny*w)],col)
    def path(points,widths,col):
        for i in range(len(points)-1): seg(points[i],points[i+1],widths[i],col,widths[i+1]); ell(*points[i+1],widths[i+1],widths[i+1],col)
    light=1. if p[7]>0 else -.72; shadow=-.90; mid=.48*light; glass=-.6 if light>0 else .62
    if c==0:
        if mode==0: # plan view, swept wings and tailplane
            poly([(-.95,.03),(-.83,-.06),(-.32,-.095),(.10,-.83),(.25,-.87),(.00,-.09),(.64,-.06),(.86,-.32),(.95,-.32),(.83,-.035),(.99,0),(.80,.05),(.95,.30),(.85,.29),(.60,.065),(.00,.105),(.24,.88),(.09,.84),(-.34,.105),(-.84,.08)],light)
            ell(-.62,0,.16,.045,glass)
        elif mode==1: # side or three-quarter view
            poly([(-1,.05),(-.92,-.035),(-.66,-.10),(.48,-.09),(.77,-.47),(.89,-.49),(.80,-.07),(.98,-.06),(.84,.015),(.76,.075),(.10,.12),(-.35,.35),(-.49,.35),(-.26,.13),(-.76,.14)],light)
            seg((-.06,-.05),(.28,-.38),.08,mid,.025); ell(-.73,-.035,.10,.035,glass)
            seg((-.20,.14),(-.39,.28),.07,mid,.045)
        else: # wings projected across the frame, nose, vertical stabilizer
            poly([(-1,-.10),(-.94,-.13),(-.13,-.015),(-.045,-.055),(-.020,-.38),(.025,-.38),(.064,-.06),(.15,-.015),(.96,-.12),(1,-.10),(.60,.015),(.15,.095),(-.12,.09),(-.6,.01)],light)
            ell(0,.025,.155,.20,light); ell(0,-.043,.11,.048,glass); ell(-.37,.05,.065,.068,shadow); ell(.37,.05,.065,.068,shadow)
    if c==1:
        if mode==0: # sedan in profile
            poly([(-.99,.12),(-.97,-.07),(-.84,-.16),(-.49,-.21),(-.26,-.52),(.27,-.54),(.58,-.25),(.90,-.17),(.97,-.10),(.98,.18),(.85,.26),(-.78,.26)],light)
            poly([(-.42,-.23),(-.22,-.46),(.018,-.46),(.045,-.23)],glass); poly([(.10,-.46),(.27,-.45),(.50,-.23),(.1,-.23)],glass)
            for u in [-.62,.63]: ell(u,.22,.183,.20,shadow); ell(u,.22,.09,.10,.43)
            seg((-.37,.205),(.40,.205),.026,mid)
        elif mode==1: # car viewed from one front corner
            poly([(-.99,-.04),(-.72,-.18),(-.38,-.52),(.21,-.61),(.46,-.51),(.76,-.21),(.94,-.15),(.97,.13),(.72,.27),(-.42,.48),(-.87,.36)],light)
            poly([(-.64,-.17),(-.34,-.46),(.10,-.51),(.16,-.235)],glass); poly([(.20,-.50),(.41,-.44),(.65,-.225),(.29,-.24)],glass)
            poly([(-.92,.14),(-.48,.20),(-.53,.28),(-.9,.21)],shadow); ell(-.34,.32,.122,.19,shadow,.06);ell(-.34,.32,.062,.105,.38);ell(.76,.10,.112,.16,shadow);ell(.76,.10,.056,.079,.38)
            seg((-.23,.34),(.62,.13),.033,mid)
        else:
            poly([(-.83,.32),(-.87,-.07),(-.73,-.22),(-.49,-.58),(.46,-.58),(.73,-.25),(.88,-.04),(.82,.34),(.66,.43),(-.65,.43)],light)
            poly([(-.65,-.21),(-.43,-.51),(.42,-.51),(.64,-.21)],glass);poly([(-.44,.13),(.44,.13),(.32,.29),(-.34,.29)],shadow)
            for u in [-.66,.66]: ell(u,.07,.132,.060,.9);seg((u,.36),(u,.49),.106,shadow)
    if c==9:
        if mode==0:
            poly([(-.98,-.06),(-.86,-.15),(-.76,-.45),(-.35,-.46),(-.34,.25),(-.96,.25)],light)
            poly([(-.82,-.16),(-.72,-.38),(-.46,-.38),(-.46,-.16)],glass)
            poly([(-.29,-.61),(.98,-.61),(.98,.13),(-.29,.13)], light*(.67 if p[8]>0 else -.65)); seg((-.82,.21),(.93,.21),.063,shadow)
            for u in [-.69,.58,.82]: ell(u,.26,.13,.17,shadow); ell(u,.26,.057,.068,.46)
        elif mode==1:
            poly([(-.84,-.50),(-.30,-.64),(.00,-.49),(-.02,.29),(-.31,.50),(-.85,.36)],light)
            poly([(-.81,-.42),(-.32,-.53),(-.33,-.18),(-.81,-.13)],glass); poly([(-.26,-.54),(-.045,-.46),(-.06,-.20),(-.26,-.20)],glass)
            poly([(-.26,-.64),(.75,-.43),(.84,-.32),(.83,.11),(-.01,.37),(.00,-.49)],light*(.72 if p[8]>0 else -.64));seg((-.82,.18),(-.35,.25),.052,shadow)
            for u,v,s in [(-.16,.34,.14),(.63,.14,.10),(.78,.11,.086)]: ell(u,v,s*.72,s*1.23,shadow);ell(u,v,s*.30,s*.49,.40)
        else:
            poly([(-.68,-.70),(.65,-.70),(.75,-.45),(.76,.45),(.59,.56),(-.64,.56),(-.75,.46),(-.75,-.41)],light)
            poly([(-.63,-.49),(.62,-.49),(.67,-.12),(-.67,-.12)],glass);poly([(-.47,.05),(.47,.05),(.45,.31),(-.46,.31)],shadow)
            for u in [-.60,.60]: seg((u,.45),(u,.65),.12,shadow);ell(u,.26,.11,.058,.85)
    if c==8:
        if mode in (0,1):
            poly([(-1,.05),(-.45,.10),(.38,.08),(.99,-.015),(.83,.25),(.58,.34),(-.69,.34)],shadow if p[7]<0 else light)
            if mode==0:
                u=.36*p[8];poly([(u-.29,.075),(u-.29,-.15),(u-.22,-.15),(u-.22,-.31),(u+.18,-.31),(u+.23,-.14),(u+.32,-.14),(u+.34,.08)],.93)
                seg((u-.18,-.175),(u+.20,-.175),.045,glass);seg((u+.11,-.31),(u+.11,-.46),.048,mid)
            else:
                seg((.08,.07),(.08,-.90),.018,shadow);poly([(.054,-.87),(.054,.018),(-.61,.025)],light);poly([(.13,-.76),(.60,-.016),(.13,-.008)],.62)
        else:
            poly([(-.77,-.04),(-.1,.03),(.75,-.10),(.51,.26),(-.05,.53),(-.53,.27)],light);seg((-.06,.03),(-.05,.49),.022,mid)
            poly([(-.57,-.05),(-.42,-.37),(.40,-.37),(.58,-.08),(-.06,.016)],.89);poly([(-.41,-.28),(.37,-.28),(.42,-.16),(-.44,-.16)],glass)
    if c==2:
        if mode==0:
            # A perching bird: tapered tail, egg-shaped body, beak and folded wing.
            seg((-.13,.14),(-.85,.46),.16,light,.052); ell(.00,.023,.47,.29,light,-.59);ell(.30,-.29,.22,.216,light)
            poly([(.46,-.35),(.73,-.275),(.47,-.225)],mid);ell(.38,-.327,.042,.041,shadow);ell(-.06,.03,.32,.16,mid,-.48)
            for u in [-.01,.17]:seg((u,.25),(u-.042,.44),.021,shadow)
        elif mode==1:
            poly([(-1,.08),(-.64,-.38),(-.34,-.30),(-.05,-.02),(.24,-.31),(.53,-.39),(.98,-.05),(.51,-.18),(.23,.01),(.10,.16),(-.09,.16),(-.39,-.08),(-.64,-.18)],light)
            ell(.003,.09,.104,.26,light);poly([(-.06,.26),(-.13,.43),(.13,.42),(.06,.24)],mid)
        else:
            ell(0,.07,.47,.52,light,-.23);ell(-.07,-.37,.258,.26,light);ell(-.156,-.404,.05,.05,shadow); poly([(-.24,-.41),(-.46,-.33),(-.23,-.29)],mid);ell(.16,.006,.18,.37,mid,-.26)
    if c in (4,7) or (c in (3,5) and mode==0):
        horse=c==7; deer=c==4; pet=c in (3,5)
        if mode!=2 or pet:
            # A side view; the long articulated limbs are deliberately not part of a bounding blob.
            thick=.062 if horse else (.038 if deer else .069); leg=.75 if not pet else .55
            for k,(u,sgn) in enumerate([(-.46,-1),(.36,1),(-.38,1),(.44,-1)]):
                off=.14*sgn*(1 if mode==0 else 1.7);path([(u,.02),(u+off,.34),(u+off*.36,leg),(u+off*.36+.05,leg+.016)],[thick*1.9,thick*.74,thick*.49,thick*.57], light if k<2 else mid)
            ell(-.04,-.041,.63,.262 if not pet else .30,light);ell(-.43,-.012,.251,.286,light)
            if not pet:
                path([(.33,-.075),(.49,-.34),(.62,-.55)],[.20,.145,.10],light);ell(.71,-.51,.236,.118,light,.55 if horse else .29)
                poly([(.53,-.56),(.46,-.83),(.65,-.61)],mid); poly([(.64,-.60),(.71,-.81),(.74,-.58)],light);ell(.68,-.555,.030,.028,shadow)
                if horse: path([(-.61,-.12),(-.70,.12),(-.75,.38)],[.081,.074,.034],mid); seg((.37,-.25),(.55,-.61),.040,mid)
                else:
                    seg((-.61,-.10),(-.76,-.26),.065,light,.024)
                    if mode==1:
                        for u in [.57,.67]:path([(u,-.62),(u-.04,-.78),(u-.16,-.93)],[.020,.017,.009],mid);seg((u-.04,-.78),(u-.01,-.92),.014,mid,.006)
            else:
                ell(.48,-.21,.265,.28,light);ell(.67,-.24,.22,.12,light);ell(.541,-.30,.035,.035,shadow);ell(.83,-.24,.052,.05,shadow)
                if c==5:ell(.41,-.26,.114,.20,mid,-.2);path([(-.57,-.08),(-.78,-.20),(-.95,-.43)],[.078,.066,.035],light)
                else:poly([(.29,-.34),(.29,-.62),(.50,-.44),(.59,-.60),(.64,-.35)],light);path([(-.59,.02),(-.83,-.27),(-.83,-.56)],[.06,.050,.035],light)
        else:
            # Three-quarter / near frontal ungulate, with face extending down from the eyes.
            ell(-.13,.01,.39,.34,light);ell(.08,-.18,.216,.38,light,-.28)
            for u in [-.34,-.18,.12,.26]:seg((u,.21),(u-.024,.75),.057 if horse else .034,light,.026)
            ell(.17,-.49,.184,.274,light,-.32);ell(.236,-.37,.135,.152,mid);ell(.09,-.56,.036,.039,shadow)
            for u in [.04,.28]:poly([(u-.065,-.65),(u-.07,-.87),(u+.075,-.66)],light)
    if c in (3,5) and mode in (1,2,3):
        cat=c==3
        if mode!=3: ell(.025,.51,.48 if cat else .54,.54,mid,-.13)
        uy=-.18 if mode==1 else -.10
        wide=.48 if mode==1 else .62
        if mode==3:wide=.84;uy=.014
        h=wide*(.88 if cat else 1.03)
        if cat:
            for s in [-1,1]:poly([(s*.98*wide,uy-.11*h),(s*.94*wide,uy-1.45*h),(s*.34*wide,uy-.82*h)],light); poly([(s*.86*wide,uy-.63*h),(s*.88*wide,uy-1.20*h),(s*.54*wide,uy-.80*h)],mid)
        else:
            for s in [-1,1]:ell(s*.88*wide,uy-.045*h,.36*wide,.76*h,mid,s*(-.24))
        ell(0,uy,wide,h,light);ell(0,uy+.40*h,wide*.51,h*.37,.62*light)
        for s in [-1,1]:ell(s*.44*wide,uy-.18*h,.14*wide,.090*h if cat else .12*h,shadow,-s*.15)
        ell(0,uy+(.30 if cat else .38)*h,.13*wide if cat else .24*wide,.095*h if cat else .16*h,shadow)
        seg((-.22*wide,uy+.63*h),(.22*wide,uy+.63*h),.027*h,mid)
    if c==6:
        if mode==0:
            ell(-.13,.04,.55,.37,light,-.2);ell(.27,-.12,.38,.25,light,.11);ell(.51,-.08,.23,.14,light)
            ell(.35,-.29,.13,.133,light);ell(.37,-.32,.077,.064,shadow);ell(-.34,.15,.34,.20,mid,-.3)
            path([(-.44,.12),(-.70,.40),(-.17,.38)],[.13,.078,.042],light);path([(.21,.02),(.10,.32),(.58,.38)],[.085,.064,.033],light);seg((.34,-.032),(.67,-.053),.018,shadow)
        elif mode==1:
            ell(0,.09,.54,.44,light);ell(0,-.16,.47,.286,light)
            for s in [-1,1]:ell(s*.34,-.33,.16,.16,light);ell(s*.34,-.34,.083,.077,shadow);path([(s*.37,.02),(s*.68,.28),(s*.46,.43)],[.115,.10,.04],mid);ell(s*.53,.26,.23,.19,light)
            seg((-.33,-.055),(.33,-.055),.023,mid)
        else:
            ell(-.03,.04,.39,.55,light,.38);ell(.15,-.28,.37,.28,light,.29)
            for s in [-1,1]:ell(.15+s*.23,-.414,.098,.099,mid);path([(s*.27,.21),(s*.66,.46),(s*.54,.03)],[.13,.108,.051],mid);path([(s*.23,-.17),(s*.48,-.02),(s*.66,-.26)],[.068,.053,.021],light)
    # Block integrate the procedural drawing to its camera resolution.
    return im.reshape(32,2,32,2).mean((1,3))

v=(np.arange(64,dtype=np.float32)+.5)/32-1
base_x,base_y=np.meshgrid(v,v)
# A low-discrepancy grid avoids having to enumerate a full Cartesian pose grid.
def vdc(n,b):
    q=0.; d=1./b
    while n:
        n,r=divmod(n,b);q+=r*d;d/=b
    return q

def descriptors(a):
    def cells(z,k):return z.reshape(-1,32//k,k,32//k,k).mean((2,4))
    q=np.pad(a,((0,0),(1,1),(1,1)),mode='edge');dx=(q[:,1:-1,2:]-q[:,1:-1,:-2]);dy=(q[:,2:,1:-1]-q[:,:-2,1:-1]); mag=np.sqrt(dx*dx+dy*dy)
    center=cells(mag,4); total=center.mean((1,2),keepdims=True)
    bins=[]
    for i in range(6):
        cs=np.cos(i*np.pi/6);sn=np.sin(i*np.pi/6);pr=np.abs(cs*dx+sn*dy)
        m=cells(np.maximum(pr-.82*mag,0)/.18,4); bins.append(np.sqrt(m/(center+.6*total+.0001)))
    hog=np.stack(bins,1);hog-=.70*hog.mean((2,3),keepdims=True)+.30*hog.mean((1,2,3),keepdims=True)
    coarse=cells(a,2);q=np.pad(coarse,((0,0),(2,2),(2,2)),mode='edge');smooth=sum(q[:,j:j+16,i:i+16] for j in range(5) for i in range(5))/25
    d=coarse-smooth
    return hog.reshape(len(a),-1).astype('f'), d.reshape(len(a),-1).astype('f')

weights=[]; masks=[]; regions=[]; nposes=72
for c in range(10):
    blocks=[]; silhouettes=[]; regviews=[]
    nm=4 if c in (3,5) else 3
    for mode in range(nm):
        ims=[]
        for j in range(1,nposes+1):
            scale=(.51+.64*vdc(j,2)) if c in (2,4,6) else (.65+.61*vdc(j,2))
            if c in (3,5) and mode==3:scale=.80+.64*vdc(j,2)
            rot=.22 if c!=0 else (.91 if mode==0 else .36)
            pars=((vdc(j,3)-.5)*.36,(vdc(j,5)-.5)*.36,scale,scale*(.76+.54*vdc(j,7)),(vdc(j,11)*2-1)*rot,(vdc(j,13)-.5)*.36,1,1 if j%4<2 else -1,1 if j%5<3 else -1)
            image=make_image(c,mode,pars); ims.extend((image,image[:,::-1]))
        h,d=descriptors(np.stack(ims));h/=np.maximum(.01,np.linalg.norm(h,axis=1,keepdims=True));d/=np.maximum(.01,np.linalg.norm(d,axis=1,keepdims=True))
        blocks.append(np.concatenate([h,d],1))
        mask=(np.abs(ims)>.05).astype('f').reshape(-1,16,2,16,2).mean((2,4))
        # A band outside the silhouette, with an exclusion zone at the
        # contour. It supplies a local background for the colour test.
        q=np.pad(mask,((0,0),(3,3),(3,3)),mode='edge');rim=np.maximum.reduce([q[:,i:i+16,j:j+16] for i in range(7) for j in range(7) if (i-3)**2+(j-3)**2<=10]);rim=np.maximum(0,rim-np.maximum.reduce([q[:,i:i+16,j:j+16] for i in range(2,5) for j in range(2,5)]))
        rim/=(.001+rim.sum((1,2),keepdims=True));reg=mask/(mask.sum((1,2),keepdims=True)+.001)-rim
        regviews.append(reg.reshape(-1,256))
        mask=mask.reshape(-1,256); mask-=mask.mean(1,keepdims=True);mask/=np.maximum(.01,np.linalg.norm(mask,axis=1,keepdims=True)); silhouettes.append(mask)
    if nm==3: blocks.append(blocks[0]); silhouettes.append(silhouettes[0]);regviews.append(regviews[0]) # padded mode ignored by the detector
    weights.append(np.concatenate(blocks));masks.append(np.concatenate(silhouettes));regions.append(np.concatenate(regviews))
np.savez(out/'constants.npz', shapes=np.concatenate(weights).astype(np.float32).T.copy(), outlines=np.concatenate(masks).astype(np.float32).T.copy(), regions=np.concatenate(regions).astype(np.float32).T.copy())

source = '''import numpy as np

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
'''
(out/'solution.py').write_text(source)
