import numpy as np

def blur(x):
    a=np.pad(x,((0,0),)*(x.ndim-2)+((1,1),(1,1)),mode='edge')
    b=(a[...,1:-1,:]*2+a[...,:-2,:]+a[...,2:,:])*.25
    return (b[...,1:-1]*2+b[...,:-2]+b[...,2:])*.25

def grad(x):
    p=np.pad(x,((0,0),(1,1),(1,1)),mode='edge')
    dx=p[:,1:-1,2:]-p[:,1:-1,:-2]; dy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    return dx,dy

def grid(x,k):
    n=len(x);return x.reshape(n,-1,k,32//k,k,32//k).mean((3,5))

def unit(x,eps=.00001):
    return x/np.sqrt(np.sum(x*x,axis=1,keepdims=True)+eps)

def arr(x):return np.asarray(x,dtype=np.float32)

def descriptor(x):
    n=len(x); r,g,b=x[:,0],x[:,1],x[:,2]; y=(r*2+g*3+b)/6
    smooth=blur(y); broad=smooth
    for j in range(4):broad=blur(broad)
    features=[]
    # Soft oriented energy bins. Remove only the direction-neutral component,
    # keeping negative evidence for incorrectly oriented contours.
    for im,k,gain in [(smooth,8,.56),(broad,4,.52)]:
        dx,dy=grad(im); m=np.sqrt(dx*dx+dy*dy)+.0001; h=[]
        for a in range(6):
            angle=a*np.pi/6
            dot=(dx*np.cos(angle)+dy*np.sin(angle))/m
            h.append(grid(blur(arr(m*np.maximum(0,abs(dot)-.74)*3.8)),k)[:,0])
        h=np.stack(h,axis=1); avg=h.mean(1,keepdims=True)
        h=(h-.65*avg)/np.sqrt(.020+avg)
        h=h-h.mean((1,2,3),keepdims=True)
        features.append(unit(h.reshape(n,-1))*gain)
    p=grid(smooth,8).reshape(n,-1); p=p-p.mean(1,keepdims=True)
    band=grid(smooth-broad,8).reshape(n,-1)
    features.extend((unit(p)*.45,unit(band,.001)*.255))
    warmth=(r-b); green=g-(r*.56+b*.44)
    c=grid(np.stack([warmth,green*1.6],axis=1),4).reshape(n,2,16)
    mean=c.mean(2,keepdims=True);c=c-.55*mean
    features.append(unit(c.reshape(n,-1),.095)*.385)
    return np.ascontiguousarray(np.concatenate(features,1),dtype=np.float32)

def sat(x):return np.clip(x,0,1)

def ramp(x,mid,span):return sat(.5+(x-mid)/span)

def spatial(a,w):return np.sum(a*w,axis=(-2,-1))/np.sum(w)

def recognize(x):
    r,g,b=x[:,0],x[:,1],x[:,2]; n=len(x); y=(r*2+g*3+b)/6
    Y,X=np.mgrid[:32,:32]; X=(X+.5-16)/16; Y=(Y+.5-16)/16
    mid=np.exp(-1.9*X**2-2.7*Y**2); surround=1-mid; lower=sat((Y-.22)*3);upper=sat((-Y-.14)*3); sides=sat((abs(X)-.45)*2.8); corners=surround**2
    q=blur(y);dx,dy=grad(q); mag=np.sqrt(dx*dx+dy*dy); avg=lambda a: a.mean(axis=(-2,-1)); cen=lambda a: spatial(a,mid); outer=lambda a: spatial(a,surround)
    rg=(r-g)/(.20+y); gb=(g-b)/(.20+y); rb=rg+gb; chrom=(np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b))/(.2+y)
    plant=ramp(gb,.065,.17)*ramp(-rg,-.034,.15); lush=ramp(gb,.115,.17)*ramp(-rg,.022,.10); blue=ramp(-gb,.055,.145)*ramp(-rb,.049,.15); aqua=ramp(-rb,.045,.153)*ramp(-gb,-.018,.11); ochre=ramp(rg,.088,.15)*ramp(rb,.18,.21); fgreen=ramp(-rg,.069,.17)*ramp(gb,.12,.20)
    skycolor=sat(blue+.46*ramp(y,.74,.33)*ramp(-chrom,-.14,.17)); sky=spatial(skycolor,upper); sky*=1-.46*ramp(spatial(mag,upper),.077,.079)
    bottomblue=spatial(aqua,lower); bgblue=outer(blue); green=outer(plant); grass=spatial(plant,lower); warm=cen(ochre); leaf=avg(lush); greenc=cen(fgreen);soil=outer(ochre);nc=cen(ramp(chrom,.24,.24)); vividly=cen(ramp(chrom,.55,.31)); extreme=cen(ramp(chrom,.80,.30))
    coolgray=spatial(ramp(-chrom,-.12,.17)*ramp(-rg,-.026,.12),lower); dark=cen(ramp(-y,-.205,.20)); bright=cen(ramp(y,.75,.24)); bw=ramp(dark,.135,.27)*ramp(bright,.105,.24); darkscene=ramp(-outer(y),-.38,.36)
    z=avg(mag); smbg=ramp(-spatial(mag,corners),-.069,.077); fine=abs(y-blur(y));busy=outer(fine)/(.012+outer(mag));rough=ramp(outer(fine),.019,.024); centerfine=cen(fine)/(.013+cen(mag)); fuzz=ramp(centerfine,.211,.139); litter=ramp(busy,.215,.17)
    neutral=outer(ramp(-chrom,-.14,.19)); dull=outer(ramp(-chrom,-.25,.24)); top=spatial(y,upper); bot=spatial(y,lower); ambient=outer(y)
    # Local orientation agreement followed by longer horizontal/vertical integration.
    xx=blur(dx*dx); yy=blur(dy*dy); xy=blur(dx*dy); energy=xx+yy+.0008
    hh=sat((yy-xx)/energy); vv=sat((xx-yy)/energy); iso=np.sqrt((xx-yy)**2+4*xy**2)/energy
    def extend(m,axis,sh):
        p=np.pad(m,((0,0),(sh if axis==1 else 0,sh if axis==1 else 0),(sh if axis==2 else 0,sh if axis==2 else 0)),mode='edge')
        if axis==1: return (p[:,:32]+m+p[:,2*sh:])/3
        return (p[:,:,:32]+m+p[:,:,2*sh:])/3
    hor=(extend(dy,2,2)*extend(hh,2,3));vert=(extend(dx,1,2)*extend(vv,1,3));hv=cen(abs(hor))/(.014+cen(mag)); vi=cen(abs(vert))/(.014+cen(mag)); manmade=ramp(hv+vi,.505,.29); horizontal=ramp(hv,.280,.26); boxy=ramp(vi,.202,.20)*ramp(hv,.21,.21); horsky=ramp(spatial(abs(hor),lower)/(.013+spatial(mag,lower)),.36,.30)
    # What fraction of the background resembles a single smooth field? A ship
    # usually has different fields above and below the hull, a flying plane not.
    def regionalchange(w1,w2):
        v=[abs(spatial(k,w1)-spatial(k,w2)) for k in (r,g,b)]
        return (v[0]+v[1]+v[2])/3
    tb=regionalchange(upper,lower); same=ramp(-tb,-.118,.19); two=ramp(tb,.136,.21); band=spatial(mag,mid)/(.023+outer(mag)); isolated=ramp(band,1.02,.99); borderactivity=spatial(mag,lower)/(.019+spatial(mag,upper));layer=ramp(borderactivity,1.02,1.15)
    watery=ramp(bottomblue,.28,.46); upperblue=ramp(spatial(blue,upper),.26,.47); skyall=ramp(bgblue,.44,.52)*same; pale=ramp(spatial(y,corners),.75,.29)*smbg*dull
    # Estimate the horizontal disposition of salient contours. This is a
    # measurement of the current input, never pooled across different photos.
    ew=mag*(.45+.55*mid); et=avg(ew)+.00001; mx=avg(ew*X)/et; my=avg(ew*Y)/et; sx=avg(ew*X**2)/et-mx*mx; sy=avg(ew*Y**2)/et-my*my; flat=ramp(sx-sy,.042,.13)
    # Side colour consistency is evaluated at the same height to allow horizons.
    edgecol=x
    l=edgecol[:,:,:,:5].mean(3,keepdims=True);rh=edgecol[:,:,:,-5:].mean(3,keepdims=True); dist=np.sqrt(np.minimum(((x-l)**2).mean(1),((x-rh)**2).mean(1)))
    body=ramp(blur(dist),.112,.15); bcenter=cen(body); bcover=spatial(body,mid*(1-lower));bspread=spatial(body,lower*mid)/(.10+bcenter); narrow=ramp(-bspread,-.58,.45)
    # Chromatic signals on the likely object rather than in foliage behind it.
    obj=mid[None,:,:]*(.18+body); objmean=lambda a: avg(obj*a)/avg(obj); coat=objmean(ochre); coatgrass=ramp(coat,.41,.46)*ramp(green,.32,.50); brownleaf=ramp(soil+green,.56,.61); autumn=ramp(soil,.29,.48)
    shade=cen(y)-outer(y); darkobj=ramp(-shade,.065,.19); brightobj=ramp(shade,.070,.18)
    # Bilateral facial landmarks: two dark eye sockets above a nose in an oval
    # region; their contrast is evaluated locally. Several head sizes allowed.
    facial=np.full(n,-1.,np.float32); feline=facial.copy(); canine=facial.copy()
    def at(img,xx,yy):
        return img[:,np.clip(yy,0,31),np.clip(xx,0,31)]
    for radius in (5,8,11,14):
        a=np.arange(6,27,2); v=np.arange(5,23,2);cx,cy=np.meshgrid(a,v);cx=cx.ravel();cy=cy.ravel()
        def samp(img,h,k):return at(img,cx+int(round(h*radius)),cy+int(round(k*radius)))
        def eyes(side):
            e=samp(q,side*.42,0); return (.50*samp(q,side*.43,-.29)+.32*samp(q,side*.54,.28)+.18*samp(q,0,-.06)-e)
        el,er=eyes(1),eyes(-1);eye=np.minimum(el,er)*.65+np.maximum(el,er)*.35-.45*abs(samp(q,.42,0)-samp(q,-.42,0))
        def nose(v):return .42*(samp(q,-.30,v+.08)+samp(q,.30,v+.08))+.16*samp(q,0,v-.30)-samp(q,0,v)
        sym=abs(samp(q,.77,.23)-samp(q,-.77,.23))+.6*abs(samp(q,.69,-.34)-samp(q,-.69,-.34));eye-=.13*sym
        cn=nose(.43);dn=nose(.75);facial=np.maximum(facial,(eye+.26*np.maximum(cn,dn)).max(1));feline=np.maximum(feline,(eye+.45*cn-.15*dn).max(1));canine=np.maximum(canine,(eye+.57*dn-.18*cn).max(1))
    face=ramp(facial,.10,.16);catface=ramp(feline-canine,0,.092); dogface=1-catface
    # Repeated round, dark wheel forms over a lighter road. The same row and
    # similar radii are required at both wheels; isolated dark spots do not count.
    wheels=np.zeros(n,np.float32)
    for rr in [2,3,4]:
        tire=np.zeros_like(q);center=q[:,13:28,3:29];i,j=np.mgrid[13:28,3:29];rad=rr*1.15
        resp=np.zeros_like(center);hub=np.zeros_like(center)
        for ang in np.arange(8)*np.pi/4:
            ox,oy=np.cos(ang),np.sin(ang);h=at(q,j+round(ox*rad*1.36),i+round(oy*rad*1.36));a=at(q,j+round(ox*rr*.67),i+round(oy*rr*.67));resp+=np.clip(h-a,-.10,.38);hub+=a
        tire[:,13:28,3:29]=resp/8+.13*sat(center-hub/8)
        for sep in [3*rr,5*rr,7*rr]:
            if sep>24:continue
            resp=np.minimum(tire[:,:,sep:],tire[:,:,:-sep]);wheels=np.maximum(wheels,resp.max((1,2)))
    wheels=ramp(wheels,.095,.145)*ramp(hv+vi,.44,.39)
    vehicle=(manmade-.5)*1.65+(horizontal-.5)*.70+(wheels-.5)*.78+(flat-.5)*.64-(green-.30)*1.50-(bgblue-.24)*.50
    outside=ramp(green,.31,.43); forest=ramp(leaf,.22,.42); stubble=autumn*(1-forest); bluebottom=watery*horsky; verdant=ramp(greenc,.19,.31); field=outside*grass;trunk=coatgrass*(1-forest*.5); plain=1-ramp(spatial(mag,corners),.074,.10)
    sc=np.zeros((n,10),np.float32)
    sc[:,0]=2.0*(sky-.35)+1.3*(skyall-.26)+.85*(smbg-.5)+.75*(isolated-.45)+.68*(flat-.5)+.42*darkobj+.7*pale-.91*grass-.5*(rough-.45)-.65*bluebottom-.48*two*horizontal
    sc[:,8]=1.9*(watery-.38)+.98*(sky-.30)+1.12*(horizontal-.46)+.8*(horsky-.49)+.85*(two-.50)+.45*(flat-.5)-1.10*(grass-.25)-.43*(fuzz-.45)+.48*brightobj-.66*same*smbg
    sc[:,1]=vehicle+.40*(neutral-.55)+.41*(brightobj-.45)+.48*(vividly-.22)+.42*(wheels-.43)-.61*(sky-.33)+.32*(flat-.5)-.48*(boxy-.5)
    sc[:,9]=vehicle+.94*(boxy-.5)+.55*(vi-.20)*3+.53*(sky-.30)+.73*(top-bot-.08)+.39*(bright-.18)+.32*(coolgray-.37)-.47*(smbg-.5)
    animal=-.60*(manmade-.5)-.46*(horizontal-.5)-.69*(watery-.27)-.37*(flat-.5)
    sc[:,2]=animal+.75*(outside-.47)+.58*(forest-.4)+.78*(plain-.49)+.47*(narrow-.48)+.66*(isolated-.48)+.66*upperblue*same+.70*upperblue*grass+.43*(vividly-.24)-.50*(bcover-.51)-.60*(face-.5)
    sc[:,4]=animal+1.22*(green-.33)+.84*(soil-.25)+1.10*(brownleaf-.58)+.59*(narrow-.5)+.61*(rough-.46)+.62*(trunk-.25)-.85*(sky-.32)-.73*(warm-.37)*grass-.62*(face-.5)-.55*(bw-.3)
    sc[:,7]=animal+.98*(grass-.31)+.56*(outside-.43)+.64*(warm-.32)+1.07*(trunk-.24)+.54*(sky-.26)+.47*(darkobj-.49)+.40*(bw-.29)+.49*flat*narrow-.61*(litter-.5)-.39*(face-.49)
    sc[:,6]=animal+1.72*(verdant-.28)+.84*(darkscene-.38)+.65*(litter-.52)+.65*(rough-.47)+.89*(soil-.22)+.62*(greenc-green*.35)+.39*(vividly-.20)-.86*(sky-.28)-.55*(bw-.30)-.60*(narrow-.5)-.46*(face-.49)-.55*(layer-.51)
    petcue=animal-.83*(sky-.3)-.60*(green-.3)+.62*(face-.47)+.53*(fuzz-.47)+.48*(bcover-.5)
    sc[:,3]=petcue+.77*(catface-.5)+.66*(ambient-.5)+.63*(dull-.55)+.58*(brightobj-.50)+.62*(narrow-.47)+.48*(centerfine-.22)*4-.52*(vividly-.23)-.40*(wheels-.49)-.42*(darkscene-.4)
    sc[:,5]=petcue+.82*(dogface-.5)+.54*(bw-.28)+.52*(coat-.31)+.47*(dark-.24)+.46*(bspread-.68)+.55*(grass-.25)+.48*(brightobj-.48)-.60*(litter-.5)
    return sc

# The comparisons are batched at a fixed width, including for a single image.
# This also makes the floating point reduction used by BLAS batch-independent.
shapes=CONSTANTS['shapes'].astype(np.float32).reshape(-1,640).T.copy()/420

def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),64):
        v=x[start:start+64];n=len(v); temp=np.zeros((64,640),np.float32); temp[:n]=descriptor(v)
        sim=(temp@shapes).reshape(64,10,-1)[:n]
        best=np.partition(sim,-12,axis=2)[:,:,-12:]; shape=.64*best.mean(2)+.36*best.max(2)
        result[start:start+n]=(shape*19+recognize(v)).argmax(1)
    return result
