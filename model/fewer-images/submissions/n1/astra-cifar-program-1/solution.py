import numpy as np

def pool(a, n):
    b = a.reshape(a.shape[:-2] + (a.shape[-2]//n,n,a.shape[-1]//n,n))
    return b.mean(axis=(-3,-1))

def blur(a, r):
    p = [(0,0)]*(a.ndim-2) + [(r,r),(r,r)]
    t = np.pad(a, p, mode="edge")
    w = 2*r+1
    t = np.cumsum(t,axis=-1,dtype=np.float32)
    t = (t[...,w-1:] - np.pad(t[...,:-w],[(0,0)]*(a.ndim-1)+[(1,0)])) / w
    t = np.cumsum(t,axis=-2,dtype=np.float32)
    return (t[...,w-1:,:] - np.pad(t[...,:-w,:],[(0,0)]*(a.ndim-2)+[(1,0),(0,0)])) / w

def deriv(a):
    p=np.pad(a,[(0,0),(1,1),(1,1)],mode="edge")
    return (p[:,1:-1,2:]-p[:,1:-1,:-2])*.5,(p[:,2:,1:-1]-p[:,:-2,1:-1])*.5

def unit(a):
    a=a.reshape(len(a),-1)
    return a / np.sqrt((a*a).sum(1,keepdims=True)+.000001)

def spatial(a):
    return a-a.mean((-2,-1),keepdims=True)

def descriptor(lum, mask, extra=None):
    smooth=.6*lum+.4*blur(lum,1)
    gx,gy=deriv(smooth)
    mag=np.sqrt(gx*gx+gy*gy+.000001)
    bins=[]
    for j in range(6):
        cs=np.float32(np.cos(j*np.pi/6))
        sn=np.float32(np.sin(j*np.pi/6))
        pr=np.abs(cs*gx+sn*gy)
        v=np.maximum(pr-.80*mag,0)*5
        if extra is not None:
            dx,dy,cm=extra
            v=np.maximum(v,.7*np.maximum(np.abs(cs*dx+sn*dy)-.80*cm,0)*5)
        bins.append(np.sqrt(pool(.65*v+.35*blur(v,1),4)+.0015)-.03873)
    h=np.stack(bins,axis=1)
    h=h-.67*h.mean(axis=1,keepdims=True)
    h=h-.8*h.mean((-2,-1),keepdims=True)
    l=pool(smooth,2)
    loc=l-blur(l,2)
    l=(.30*spatial(l)+.70*loc)/np.sqrt(.029+blur(np.abs(loc),2))
    m=pool(mask,2) if mask.shape[-1]==32 else mask
    m=.55*spatial(m)+.45*(m-blur(m,3))
    return np.concatenate((.40*unit(h),.44*unit(l),.16*unit(m)),axis=1)

def sat(x):
    return np.clip(x,0,1)

def zlim(x):
    return np.clip(x,-1.5,1.5)

def avg(a):
    return a.mean(axis=(-2,-1))

def shifted(a, dy, dx):
    hh,ww=a.shape[-2:]
    iy=np.clip(np.arange(hh)+dy,0,hh-1)
    ix=np.clip(np.arange(ww)+dx,0,ww-1)
    return a[:,iy[:,None],ix]

def saliency(x):
    pic=pool(x,2)
    pic=.72*pic+.28*blur(pic,1)
    yy,xx=np.mgrid[0:16,0:16].astype(np.float32)
    best=np.full((len(x),16,16),2,np.float32)
    for yy0,xx0 in [(i,j) for i in (1,4,7,10,13,15) for j in (0,15)]+[(i,j) for i in (0,15) for j in (2,5,8,11,14)]:
        dif=pic-pic[:,:,yy0:yy0+1,xx0:xx0+1]
        light=dif.mean(axis=1)
        dv=np.mean(abs(dif-light[:,None]),axis=1)+.42*abs(light)
        space=np.minimum(.053,.0038*abs(yy-yy0)+.0007*abs(xx-xx0))
        best=np.minimum(best,dv+space)
    m=sat((best-.040)/.135)
    return .66*m+.34*blur(m,1)

def smallparts(lum):
    # Search for pairs of dark, roughly round tires with lighter surrounds.
    # A light hub is optional. The tire top and both lateral arcs must agree.
    wh=np.zeros(len(lum),np.float32)
    for r in (2,3,4):
        sm=.55*lum+.45*blur(lum,1)
        outside=[]; inside=[]
        for dy,dx in [(-r,0),(0,-r),(0,r),(-r+1,-r+1),(-r+1,r-1)]:
            outside.append(shifted(sm,int(round(dy*1.45)),int(round(dx*1.45))))
            inside.append(shifted(sm,int(round(dy*.77)),int(round(dx*.77))))
        delta=[a-b for a,b in zip(outside,inside)]
        arcs=sum(delta)/5
        ring=sum(inside)/5
        light=sat((arcs-.018)/.17)*.54+sat((np.minimum(np.minimum(delta[0],delta[1]),delta[2])+.015)/.13)*.28+sat((sm-ring)/.15)*.18
        peak=np.zeros(len(lum),np.float32)
        for gap in (r*4,r*6,r*8):
            if gap>25: continue
            for skew in (-1,1):
                a=light[:,14:28,1:31-gap]; b=light[:,14+skew:28+skew,gap+1:31]
                same=sat(1-abs(ring[:,14:28,1:31-gap]-ring[:,14+skew:28+skew,gap+1:31])/.32)
                score=(np.minimum(a,b)*.70+.30*(a+b)*.5)*(.7+.3*same)
                peak=np.maximum(peak,score.max((-2,-1)))
        wh=np.maximum(wh,peak)
    # Eyes above a muzzle: look for three appropriately spaced dark features,
    # not just any single dark spot. Test several possible face sizes.
    eyes=np.zeros(len(lum),np.float32); muzzle=eyes.copy()
    for r in (2,3,4,5):
        im=.72*lum+.28*blur(lum,1)
        spot=(blur(im,max(2,r//2+1))-im)
        spot=sat((spot-.01)/.18)
        a=shifted(spot,0,-r); b=shifted(spot,0,r)
        mid=shifted(im,0,0)-.5*(shifted(im,0,r)+shifted(im,0,-r))
        pair=np.minimum(a,b)*sat(.58+mid/.36)
        nose=shifted(spot,int(round(1.55*r)),0)
        cheek=.5*(shifted(im,r,-r)+shifted(im,r,r))-shifted(im,int(round(1.55*r)),0)
        longnose=pair*(.4*sat((cheek-.015)/.23)+.6*nose)
        eyes=np.maximum(eyes,pair[:,5:20,10:22].max((-2,-1)))
        muzzle=np.maximum(muzzle,longnose[:,5:20,10:22].max((-2,-1)))
    return wh,eyes,muzzle

def scene(x,lum,mask):
    r,g,b=x[:,0],x[:,1],x[:,2]
    # Soft color memberships use opponent channels; avoid literal RGB matches.
    val=lum
    chrom=x.max(1)-x.min(1)
    green=sat((g-b-.014)/.115)*sat((g-r+.040)/.092)
    leaf=sat((g-r+.015)/.102)*sat((g-b+.002)/.107)
    blue=sat((b-r-.015)/.139)*sat((b-g+.035)/.112)
    warm=sat((r-b-.032)/.153)*sat((r-g+.015)/.108)
    ochre=sat((r-b-.04)/.155)*sat((r-g+.013)/.075)*sat((g-b+.011)/.115)
    neutral=sat(1-(chrom-.028)/.165)
    gray=neutral*sat((val-.13)/.22)*sat((1.02-val)/.22)
    bright=sat((val-.53)/.28)*sat((.23-chrom)/.17)
    paint=np.maximum.reduce([sat((r-g-.145)/.195)*sat((r-b-.12)/.17),sat((b-g-.10)/.17)*sat((b-r-.17)/.19),sat((np.minimum(r,g)-b-.23)/.18)*sat((val-.34)/.28)])
    sm=.62*lum+.38*blur(lum,1)
    dx,dy=deriv(sm); ux=abs(dx); uy=abs(dy)
    tex=ux+uy; local=blur(tex,2)
    smooth=sat((.087-local)/.070)
    sky=smooth*np.maximum(blue,bright*.86)
    vegetation=np.maximum(green,leaf)
    def top(a): return avg(a[:,:8,:])
    def bot(a): return avg(a[:,25:,:])
    def core(a): return avg(a[:,7:26,6:26])
    def sides(a): return .5*(avg(a[:,:,:5])+avg(a[:,:,-5:]))
    skyT=top(sky); skyB=bot(sky); blu=avg(blue); bt=top(blue); bb=bot(blue)
    veg=avg(vegetation); gt=top(vegetation); gb=bot(vegetation); gc=core(vegetation)
    warmC=core(warm); earth=core(ochre); color=core(paint); smoothB=bot(smooth); smoothT=top(smooth)
    road=bot(gray*(.47+.53*smooth))
    water=bot(blue*(.60+.4*smooth))
    # Vertical and horizontal edge agreement over a distance. Curved contours
    # and fur give less coherent gradients than a window, roof, hull or road.
    def coherence(a,axis):
        offs=3
        cum=[shifted(a,0,i) if axis==1 else shifted(a,i,0) for i in range(-offs,offs+1)]
        signed=abs(sum(cum)); unsigned=sum([abs(v) for v in cum]);
        return core(signed)/(.023+core(unsigned))
    hco=coherence(dy,1); vco=coherence(dx,0)
    hfrac=core(uy)/(.007+core(tex)); hline=zlim((hco-.49)*5.2); vline=zlim((vco-.49)*5.2)
    horizontal=zlim((hfrac-.47)*7)
    straight=.49*hline+.32*vline+.19*horizontal
    # A horizon produces a long cross-image edge at one height; the two
    # sides should agree, so a strong edge on just the object is not enough.
    band=blur(lum,1)
    jump=band[:,4:29]-band[:,1:26]
    jl=jump[:,:,:8].mean(-1); jr=jump[:,:,-8:].mean(-1)
    horizon=sat((np.max(np.where(jl*jr>0,np.minimum(abs(jl),abs(jr)),0),axis=1)-.027)/.169)
    # Smooth side borders, absence of a horizon and very similar sky colors
    # at top and bottom suggest an isolated object in the air.
    same=sat(1-abs(x[:,:,:6,:].mean((2,3))-x[:,:,26:,:].mean((2,3))).mean(1)/.31)
    air=top(smooth)*bot(smooth)*same*(1-.53*horizon)
    # Row-wise uniformity of the lower surface (road, pasture or water).
    sq=x[:,:,26:,:]; stripe=sat(1-np.mean(abs(sq-sq.mean(-1,keepdims=True)),(1,2,3))/.187)
    ground=gb*(.52+.48*stripe)
    waterlike=np.maximum(water,.56*bot(gray)*sat((.14-bot(warm))/.12)*(.50+.50*stripe))*stripe
    field=ground*(.5+.5*sat((gb-gc+.22)/.53))
    woods=gt*(.54+.46*sat(top(tex)/.10))
    camouflage=gc*(.56+.44*sat((gc-gb+.21)/.50))
    # Texture scale and layout, computed independently per picture.
    fine=avg(abs(lum-blur(lum,1)))
    coarse=avg(abs(lum-blur(lum,3)))
    grain=zlim((fine/(.008+coarse)-.39)*6.0)
    activity=zlim((core(tex)-.095)/.067)
    lower=avg(tex[:,22:30,6:26]); upper=avg(tex[:,3:11,6:26]); middle=avg(tex[:,11:22,3:29])
    strip=zlim((middle-.55*lower-.45*upper)/(.027+avg(tex)))
    leg=avg(ux[:,19:29,5:27])/(.010+lower)
    legs=zlim((leg-.44)*6)*(.48+.52*gb)
    # Central foreground geometry: compare spatial moments to those of a
    # uniform image and see how far the figure extends vertically.
    yq,xq=np.mgrid[0:16,0:16].astype(np.float32)
    mass=mask+.025; den=mass.sum((-2,-1)); cy=(mass*yq).sum((-2,-1))/den; cx=(mass*xq).sum((-2,-1))/den
    vx=(mass*(xq-cx[:,None,None])**2).sum((-2,-1))/den
    vy=(mass*(yq-cy[:,None,None])**2).sum((-2,-1))/den
    flat=zlim(np.log((vx+1)/(vy+1))*1.8); small=sat((.40-avg(mask))/.28)
    ins=(avg(mask[:,4:12,4:12])-avg(mask))
    isolated=sat((ins-.06)/.31)*small
    # Find bilateral symmetry in a possible central head or a front view.
    sym=np.zeros(len(x),np.float32)
    for center in (13,15,17,19):
        a=sm[:,5:25,center-8:center]; d=sm[:,5:25,center:center+8][:,:,::-1]
        a=a-a.mean((-2,-1),keepdims=True); d=d-d.mean((-2,-1),keepdims=True)
        sim=avg(a*d)/np.sqrt((avg(a*a)+.002)*(avg(d*d)+.002))
        sym=np.maximum(sym,sim)
    face=sat((sym-.31)/.50)
    wheels,eyes,nose=smallparts(lum)
    wheel=zlim((wheels-.39)*3.2); eye=sat((eyes-.49)/.43); nose=sat((nose-.32)/.49)
    # A pale overcast sky is useful even when the opponent-blue signal is absent.
    envsky=np.maximum(skyT,.52*top(bright)); room=sat((warmC+.34*core(neutral)-.10)/.72)*(1-veg)*(1-skyT)*(1-blu)
    forest=sat(veg*1.48); palette=sat((.07-x.std(axis=(-2,-1)).min(1))/.07)
    # Green core or earthy ground cover in an image with no sign of a skyline.
    soil=(.55*avg(ochre)+.45*gc)*(1-skyT)*(1-blu)
    dark=sat((.47-avg(lum))/.29); rich=sat((avg(chrom)-.105)/.14)
    cold=sat(blu*1.6); artificial=.80*color+.30*core(bright)
    contrast=sat((np.std(lum,axis=(-2,-1))-.15)/.13)
    # Scores are additive pieces of evidence, not calibrated probabilities.
    sc=np.empty((len(x),10),np.float32)
    sc[:,0]=1.68*envsky+.74*bt+.48*skyB+1.55*air+.50*isolated+.30*flat+.23*strip+.12*horizontal-1.35*forest-.93*warmC-.58*horizon-.38*activity-.30*wheel-.22*road-.94
    sc[:,1]=.64*straight+.29*horizontal+.72*wheel+1.22*road+.44*stripe+.48*strip+.37*flat+.58*artificial+.23*contrast-.83*forest-.65*air-.57*water-.34*envsky-.62
    sc[:,2]=.71*forest+.57*bt+.60*isolated+.62*air+.50*color+.29*rich+.22*small-.44*straight-.32*wheel-.29*horizon-.27*ground-.26*flat-.56
    sc[:,3]=1.02*room+.43*face+.54*eye+.35*grain+.25*activity+.31*core(neutral)-.33*straight-.69*envsky-.76*forest-.52*water-.31*ground-.24*flat-.46*color-.27
    sc[:,4]=1.18*forest+1.07*woods+.75*field+.69*earth+.37*legs+.30*isolated-.68*camouflage-.71*cold-.32*straight-.43*color-.27*face-.35*air-.70
    sc[:,5]=1.03*room+.51*nose+.32*face+.51*warmC+.21*contrast+.20*field-.30*straight-.36*envsky-.69*water-.33*camouflage-.33*flat-.24*color-.43
    sc[:,6]=1.49*camouflage+1.17*soil+.44*dark+.40*grain+.30*rich+.21*activity-.68*straight-1.42*cold-1.07*envsky-.42*isolated-.33*legs-.43*color-.39
    sc[:,7]=1.27*field+.64*forest+.74*legs+.51*warmC+.48*envsky+.32*horizon+.19*contrast-.73*camouflage-.46*woods-.40*water-.25*face-.27*air-.48
    sc[:,8]=1.98*waterlike+1.16*water+.71*horizon+.52*envsky+.43*horizontal+.29*hline+.46*stripe+.29*flat-.67*air-1.19*forest-.73*warmC-.40*wheel-.93
    sc[:,9]=.80*straight+.59*vline+.49*wheel+.93*road+.55*envsky+.48*artificial+.29*contrast+.21*strip+.20*horizon-.78*forest-.60*air-.48*water-.29*warmC-.58
    return sc

def batch(x):
    lum=.299*x[:,0]+.587*x[:,1]+.114*x[:,2]
    m=saliency(x)
    # A chromatic contour can be present even at constant luminance.
    opp=(x[:,0]-x[:,2])*.62+(x[:,1]-x[:,2])*.24
    dx,dy=deriv(.6*opp+.4*blur(opp,1))
    feat=descriptor(lum,m,(dx,dy,np.sqrt(dx*dx+dy*dy+.000001)))
    response=feat @ CONSTANTS["shapes"]
    cuts=CONSTANTS["cuts"]
    best=np.stack([response[:,cuts[k]:cuts[k+1]].max(1) for k in range(10)],axis=1)
    return (10.5*best+scene(x,lum,m)).argmax(axis=1)

def predict(x):
    labels=np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),32):
        n=min(32,len(x)-i)
        chunk=np.zeros((32,3,32,32),np.float32)
        chunk[:n]=x[i:i+n]
        labels[i:i+n]=batch(chunk)[:n]
    return labels
