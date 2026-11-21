import numpy as np

# Region and contour descriptions. All normalizations here are within one image.
def pooled(a, n):
    return a.reshape(len(a), a.shape[1], n, 32//n, n, 32//n).mean((3,5))

def unit(a, floor):
    return a / np.sqrt(np.sum(a*a,axis=1,keepdims=True) + floor*floor)

def center(a):
    return a - a.mean((-2,-1),keepdims=True)

def flat(a):
    return a.reshape(len(a),-1)

def smooth(a):
    p = np.pad(a,((0,0),(0,0),(1,1),(1,1)),mode='edge')
    t = (p[:,:,:-2]+2*p[:,:,1:-1]+p[:,:,2:])*.25
    return (t[:,:,:,:-2]+2*t[:,:,:,1:-1]+t[:,:,:,2:])*.25

def gradient(a):
    p = np.pad(a,((0,0),(0,0),(1,1),(1,1)),mode='edge')
    return (p[:,:,1:-1,2:]-p[:,:,1:-1,:-2])*.5,(p[:,:,2:,1:-1]-p[:,:,:-2,1:-1])*.5

def shape_features(x):
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    y=.30*r+.57*g+.13*b
    v=pooled(y,8)
    h=unit(flat(center(v)),.44)
    local=v-smooth(smooth(v))
    local=local/(.065+np.sqrt(smooth(smooth(local*local))))
    hloc=unit(flat(local),1.5)
    rgb=pooled(np.concatenate((g-b,r-g),axis=1),4)
    col=unit(flat(center(rgb)),.32)
    contours=[]
    for a,d in ((smooth(y),8),(smooth(smooth(smooth(y))),4)):
        dx,dy=gradient(a)
        length=np.sqrt(dx*dx+dy*dy+.000001)
        bins=[]
        for i in range(6):
            angle=i*np.pi/6
            cosine=(dx*np.cos(angle)+dy*np.sin(angle))/length
            bins.append(np.maximum(0,cosine*cosine-.56)*length)
        hog=pooled(np.concatenate(bins,axis=1),d)
        hog=smooth(hog)*.62+hog*.38
        # Compare grain direction as well as the spatial layout of the contours.
        hog=np.sqrt(hog+.0007)
        orient=hog-hog.mean(1,keepdims=True)
        spatial=center(hog)
        contours.append(unit(flat(.65*orient+spatial),.27 if d==8 else .14))
    return np.concatenate((.45*h,.40*hloc,.33*col,.57*contours[0],.46*contours[1]),axis=1).astype(np.float32)

def pos(a):
    return np.clip(a,0,1)

def scene_features(x):
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    y=.30*r+.57*g+.13*b
    green=pos((g-b-.012)/(.09+.09*y))*pos((g-r+.048)/.10)
    warm=pos((r-b-.025)/(.10+.11*y))*pos((r-g+.003)/.09)
    blue=pos((b-r+.004)/(.09+.08*y))*pos((b-g+.019)/.075)
    vivid=pos((np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)-.12)/.24)
    gray=1-pos((np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)-.03)/.15)
    colors=np.concatenate((y, r-g,g-b,warm,blue,green,gray,vivid,pos((y-.60)/.27),pos((.31-y)/.23)),axis=1)
    reg=pooled(colors,4)
    # A soft bounded-distance kernel per region. Adjacent channels encode
    # different properties, so no cross-image covariance estimate is needed.
    scales=np.array([.28,.105,.13,.48,.48,.46,.55,.53,.50,.47],np.float32)[None,:,None,None]
    coeff=np.array([1,.78,.95,.80,.82,.82,.4,.26,.34,.37],np.float32)[None,:,None,None]
    z=reg/scales
    enc=[]
    for freq,weight in ((.85,.8),(1.8,.45)):
        enc.extend((flat(coeff*np.cos(z*freq)*weight),flat(coeff*np.sin(z*freq)*weight)))
    return unit(np.concatenate(enc,axis=1),.001).astype(np.float32)

def region(a,t,b,l,r):
    return a[:,t:b,l:r].mean((1,2))

def cues(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=.30*r+.57*g+.13*b
    green=pos((g-b-.015)/(.07+.07*y))*pos((g-r+.035)/.08)
    olive=pos((g-b-.035)/.15)*pos((g-r+.12)/.14)
    brown=pos((r-b-.035)/(.14+.06*y))*pos((r-g-.005)/.09)
    blue=pos((b-r+.01)/(.075+.08*y))*pos((b-g+.025)/.09)
    gray=1-pos((np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)-.035)/.15)
    light=pos((y-.50)/.30)
    sky=blue*(.4+.6*light)+gray*light*.55
    outside=np.ones((32,32),np.float32); outside[5:27,5:27]=0
    def outer(a):
        return (a*outside).sum((1,2))/outside.sum()
    def core(a):
        return region(a,7,26,6,26)
    def top(a):
        return region(a,0,8,0,32)
    def bottom(a):
        return region(a,25,32,0,32)
    grass=outer(green); grasslow=bottom(green); woods=outer(olive)
    fur=core(brown); fieldbrown=outer(brown); fgblue=core(blue)
    air=outer(sky); highsky=top(sky); sea=bottom(blue); grayroad=bottom(gray)
    low=y[:,24:].mean((1,2)); high=top(y)
    a=smooth(y[:,None]); dx,dy=gradient(a); dx=dx[:,0]; dy=dy[:,0]
    mag=np.sqrt(dx*dx+dy*dy)
    bgdetail=outer(mag); fgdetail=core(mag); activity=mag.mean((1,2))
    fine=np.abs(y-a[:,0]); grit=core(fine); bggrit=outer(fine)
    flatbg=pos((.052-bgdetail)/.047)
    busy=pos((bgdetail-.027)/.055)
    # Long horizontal edges, horizontal versus diagonal anisotropy,
    # and rectilinear structures characteristic of manufactured objects.
    ah=np.abs(dy); av=np.abs(dx)
    straight=np.abs((dy[:,:,0:24]+dy[:,:,2:26]+dy[:,:,4:28]+dy[:,:,6:30]+dy[:,:,8:32])/5).mean((1,2))
    st=pos((straight/(ah.mean((1,2))+.001)-.37)/.42)
    anis=(ah.mean((1,2))-av.mean((1,2)))/(activity+.005)
    horizon=pos(anis*3.7+.25)
    axial=(np.abs(ah-av).mean((1,2))/(activity+.001)-.47)*5
    horizontal=.55*horizon+.45*st
    splash=pos((core(np.maximum(r,np.maximum(g,b))-np.minimum(r,np.minimum(g,b)))-.14)/.25)
    redpaint=core(pos((r-np.maximum(g,b)-.09)/.28)); coldpaint=fgblue*(1-outer(blue))
    skyland=highsky*pos((high-low+.08)/.40)
    natural=grass+.55*woods+.23*fieldbrown-.45*sea
    vehicle=.95*(horizontal-.50)+.46*axial-.43*natural+.32*(grayroad-.45)+.42*(redpaint+coldpaint)-.24*pos((grit-.032)/.04)
    animal=-.56*(horizontal-.5)-.27*axial+.52*natural-.28*air+.35*(fur-.23)
    z=np.zeros((len(x),10),np.float32)
    z[:,0]=.80*(air-.4)+.90*(flatbg-.50)+.44*(outer(blue)-.18)+.18*(highsky-.4)-.49*grass-.23*fur+.10*(horizon-.4)
    z[:,1]=vehicle+.32*(grayroad-.5)-.27*skyland+.36*(redpaint+coldpaint)-.22*grasslow+.17*(st-.5)
    z[:,9]=vehicle+.48*(highsky-.35)+.37*skyland+.29*axial+.20*(st-.50)-.42*sea
    z[:,8]=.85*(horizontal-.44)+1.24*(sea-.18)+.49*(highsky-.40)+.32*(outer(blue)-.15)-.66*grass-.35*fieldbrown-.22*fur+.17*(flatbg-.4)
    z[:,2]=animal*.44+.70*(grass-.21)+.31*(flatbg-.45)+.40*(splash-.22)+.42*(fgblue-.12)-.15*(horizontal-.50)
    z[:,3]=animal*.67-.49*grass-.25*highsky+.35*(grit-.025)/.045+.14*(bggrit-.025)/.045+.20*(.34-splash)
    z[:,5]=animal*.77-.37*grass-.19*highsky+.30*(grit-.025)/.045+.13*(fieldbrown-.25)+.09*(.30-splash)
    z[:,4]=animal+.63*(grass-.24)+.57*(woods-.38)+.43*(fur-.23)+.37*(busy-.4)-.38*air
    z[:,7]=animal*.86+.62*(grasslow-.25)+.36*(fur-.25)+.18*(highsky-.40)+.17*(horizontal-.50)-.13*(busy-.4)
    z[:,6]=animal*.82+.72*(core(olive)-.36)+.50*(fieldbrown-.28)+.32*(busy-.4)-.51*highsky-.32*sea+.16*(bggrit-.025)/.045
    return z

def classify(x):
    h=shape_features(x); c=scene_features(x)
    parts=flat(part_features(x))
    prior=cues(x)+evidence(x)
    out=np.empty((len(x),10),np.float32)
    for k in range(10):
        sh=h@CONSTANTS['shape'+str(k)]
        co=c@CONSTANTS['scene'+str(k)]
        # Pool strongest individual explanations. A complementary shape-only
        # term allows for cases where the setting was not anticipated.
        pm=parts@CONSTANTS['parts'+str(k)]
        obj=.76*sh+.24*pm
        joint=obj+.80*co
        top=np.partition(joint,-12,axis=1)[:,-12:].mean(axis=1)
        shapeonly=np.partition(obj,-5,axis=1)[:,-5:].mean(axis=1)
        coloronly=np.partition(co,-15,axis=1)[:,-15:].mean(axis=1)
        out[:,k]=13*(.75*top+.25*shapeonly)+5.8*coloronly+prior[:,k]
    return np.argmax(out,axis=1)

def predict(x):
    # Fixed tile dimensions also fix the matrix multiply/reduction paths.
    labels=np.empty(len(x),np.int64)
    for i in range(0,len(x),48):
        n=min(48,len(x)-i)
        a=np.zeros((48,3,32,32),np.float32)
        a[:n]=x[i:i+n]
        labels[i:i+n]=classify(a)[:n]
    return labels

# A dense field of local shape codes. Rendered templates can restrict this field
# to the object's support; for an input photograph no segmentation is required.
def part_features(x):
    y=smooth(.30*x[:,0:1]+.57*x[:,1:2]+.13*x[:,2:3]); gx,gy=gradient(y)
    length=np.sqrt(gx*gx+gy*gy+.00012)
    coarse=pooled(y,8); p=np.pad(coarse,((0,0),(0,0),(1,1),(1,1)),mode='edge')
    cx=(p[:,:,1:-1,2:]+p[:,:,1:-1,:-2])*.5-coarse
    cy=(p[:,:,2:,1:-1]+p[:,:,:-2,1:-1])*.5-coarse
    xy=(p[:,:,2:,2:]-p[:,:,2:,:-2]-p[:,:,:-2,2:]+p[:,:,:-2,:-2])*.25
    parts=np.concatenate((pooled(gx,8)*1.3,pooled(gy,8)*1.3,pooled((gx*gx-gy*gy)/length,8),pooled(2*gx*gy/length,8),cx*.77,cy*.77,xy*.76),axis=1)
    return parts/np.sqrt(.050**2+np.sum(parts*parts,axis=1,keepdims=True))

def shiftmap(a,x,y):
    # Float image translated by an integral number of pixels with edge extension.
    return a[:,np.clip(np.arange(32)+y,0,31),:][:,:,np.clip(np.arange(32)+x,0,31)]

def evidence(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=smooth((.30*r+.57*g+.13*b)[:,None])[:,0]
    noiselevel=np.abs(y-(.30*r+.57*g+.13*b))
    # Dark points with lighter surroundings, off-centre reflections allowed.
    def dip(a,rad):
        l=shiftmap(a,-rad,0);rr=shiftmap(a,rad,0);t=shiftmap(a,0,-rad);bb=shiftmap(a,0,rad)
        return .57*(.25*(l+rr+t+bb)-a)+.43*np.minimum(.5*(l+rr)-a,.5*(t+bb)-a)
    wheel=0; facecat=0;facedog=0; legs=0
    for rad in (2,3,4):
        holes=dip(y,rad)
        # A tire can have either a dark center or a light hub. The rim sits
        # below the body; a shadow alone is not a pair of circular contours.
        rim=(shiftmap(y,-rad//2,0)+shiftmap(y,(rad+1)//2,0)+shiftmap(y,0,-(rad+1)//2)+shiftmap(y,0,(rad+1)//2))*.25
        tire=np.maximum(holes,.70*(shiftmap(y,0,rad+1)+shiftmap(y,0,-rad-1))*.5-.70*rim+.23*(y-rim))
        tire=tire+.22*np.maximum(0,.38-y)
        for gap in (3*rad+1,5*rad+1):
            j=np.maximum(shiftmap(tire,gap,0),.90*shiftmap(tire,gap,1)); pair=np.minimum(tire,j)*.72+.28*np.minimum((tire+j)*.5,tire+.05)
            v=pair[:,16:29,1:32-gap-1].max((1,2));wheel=np.maximum(wheel,v)
    for gap in (5,8,11,14):
        half=gap//2;rad=max(2,gap//3);holes=dip(y,rad); other=shiftmap(holes,gap,0)
        eyes=np.minimum(holes,other)*.79+.21*(holes+other)*.5
        for isdog in (False,True):
            down=max(2,int(gap*(.73 if isdog else .46)));nose=shiftmap(dip(y,max(2,int(rad*(1.35 if isdog else .73)))),half,down)
            bridge=shiftmap(y,half,down//2)-.5*(y+shiftmap(y,gap,0))
            # Require the eye and nose responses to coexist, not just one dark feature.
            m=.53*np.minimum(eyes,1.4*nose)+.30*eyes+.17*np.clip(bridge,-.25,.33)
            v=m[:,5:22,2:30-gap].max((1,2));
            if isdog:facedog=np.maximum(facedog,v)
            else:facecat=np.maximum(facecat,v)
    # Vertical, narrow leg-like strokes in the lower half, with contrast
    # extending downwards for several pixels. Hooves taper toward the bottom.
    for w in (2,3):
        bar=(shiftmap(y,-w,0)+shiftmap(y,w,0))*.5-y
        bar=np.maximum(bar,-.68*bar);longbar=np.minimum(bar,np.maximum(shiftmap(bar,0,4),shiftmap(bar,1,4)))
        pair=np.minimum(longbar,shiftmap(longbar,8,0));legs=np.maximum(legs,pair[:,17:27,3:22].max((1,2)))
    # Broad level surfaces and depth cues: smooth top background versus
    # textured foreground, and dry foliage versus a saturated grassy field.
    gx,gy=gradient(y[:,None]);m=np.sqrt(gx[:,0]**2+gy[:,0]**2)
    topdetail=region(m,0,7,0,32); lowerdetail=region(m,25,32,0,32);core=region(m,8,24,8,24)
    isolation=pos((core-.43*topdetail-.57*lowerdetail)/.085)
    flecks=pos((region(noiselevel,6,26,6,26)-.018)/.059)
    canopy=pos((g-b-.035)/.19)*pos((g-r+.030)/.14)
    lawn=region(canopy,23,32,0,32); foliage=region(canopy,0,11,0,32)
    z=np.zeros((len(x),10),np.float32)
    wheels=pos((wheel-.098)/.20); paws=pos((legs-.075)/.15); cat=pos((facecat-.060)/.18);dog=pos((facedog-.078)/.20)
    z[:,1]=1.03*(wheels-.40)+.20*(isolation-.44)
    z[:,9]=.61*(wheels-.4)+.18*(1-flecks-.62)
    z[:,7]=.82*(paws-.30)+.53*(lawn-.37*foliage-.19)
    z[:,4]=.70*(paws-.30)+.40*(foliage-.23)
    z[:,2]=.63*(isolation-.38)-.26*(paws-.30)
    z[:,3]=.74*(cat-.38)-.27*(dog-.32)+.20*(flecks-.27)
    z[:,5]=.77*(dog-.31)-.17*(cat-.39)+.14*(isolation-.38)
    z[:,6]=.25*(flecks-.27)+.24*(lowerdetail+topdetail-.077)/.088-.34*(paws-.30)
    z[:,0]=.42*(isolation-.39)+.24*(.041-lowerdetail)/.065
    z[:,8]=.16*(.043-topdetail)/.055-.25*(paws-.30)
    return z
