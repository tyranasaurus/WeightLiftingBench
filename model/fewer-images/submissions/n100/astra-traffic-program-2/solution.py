import numpy as np

classes=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),list(range(33,41)),[6,32,41,42]]
rects=[(-.77,.77,-.55,.55),(-.63,.63,-.44,.74),(-.83,.83,-.83,.83),(-.79,.79,-.59,.59)]

def grid(n,rect):
    x0,x1,y0,y1=rect
    return np.meshgrid((x0+(np.arange(n,dtype=np.float32)+.5)*(x1-x0)/n),(y0+(np.arange(n,dtype=np.float32)+.5)*(y1-y0)/n))

def soft(a,b,x):return np.clip((x-a)/(b-a),0,1)

def colors(img):
    r,g,b=img[:,0],img[:,1],img[:,2]
    intensity=.35*r+.45*g+.20*b
    # Mild dynamic compression preserves differences in shadow, without amplifying sensor zeros.
    floor=.025*np.mean(intensity,axis=(1,2),keepdims=True)+.004
    lum=np.sqrt(intensity+floor);lo=np.quantile(lum,.07,axis=(1,2)).astype(np.float32)[:,None,None];hi=np.quantile(lum,.93,axis=(1,2)).astype(np.float32)[:,None,None]
    lum=np.clip((lum-lo)/(hi-lo+.011),-.25,1.25)
    red=np.clip((r-.52*g-.48*b)/(r+.52*g+.48*b+floor)*2.6,-1,1)
    blue=np.clip((b-.65*r-.35*g)/(b+.65*r+.35*g+floor)*3.1,-1,1)
    yellow=np.clip((.52*r+.48*g-b)/(.52*r+.48*g+b+floor)*2.9,-1,1)
    return np.stack((red,blue,yellow,lum),axis=1)

def dist(x,y,f):
    if f in (0,2,5,7):return 1-np.sqrt(x*x+y*y)
    if f in (1,4):
        u=y if f==1 else -y
        return np.minimum(1-u,((u+1)*.5-np.abs(x))*.8944)
    if f==3:return (1-np.abs(x)-np.abs(y))*.7071
    return np.minimum(np.minimum(1-np.abs(x),1-np.abs(y)),(1.4142-np.abs(x)-np.abs(y))*.7071)

def pull(img,cfg,x,y):
    # cfg is (center x,y, half width,height, horizontal shear)
    cx,cy,rx,ry,sh=cfg.T
    xx=np.clip(cx[:,None,None]+rx[:,None,None]*x+sh[:,None,None]*y,0,30.9999)
    yy=np.clip(cy[:,None,None]+ry[:,None,None]*y,0,30.9999)
    i=xx.astype(np.int32);j=yy.astype(np.int32);fx=(xx-i).astype(np.float32);fy=(yy-j).astype(np.float32)
    t=np.arange(len(img))[:,None,None]
    return ((1-fx)*img[t,j,i]+fx*img[t,j,i+1])*(1-fy)+((1-fx)*img[t,j+1,i]+fx*img[t,j+1,i+1])*fy

def dotbank(a,b):
    # Fixed GEMM dimensions even in a short last batch, for order/batch independence.
    n=len(a);p=np.zeros((32,a.shape[1]),np.float32);p[:n]=a
    return (p@b.T)[:n]

def localize(ch):
    n=len(ch);small=ch.reshape(n,4,16,2,16,2).mean((3,5)).reshape(n,4,256)
    opts=[];vals=[];off=CONSTANTS['offsets']; bank=CONSTANTS['local']
    for f,k in enumerate((0,0,1,2,0,0,0,0)):
        col=small[:,k];lum=small[:,3];gain=(.79,.80,.96,1.02,.88,.04,1.04,1.04)[f]
        # Outer boundaries also provide information when red has lost its saturation.
        a=np.concatenate((col*gain,lum*(.79,.83,.48,.53,.90,1.12,.57,.94)[f]),axis=1)
        a=dotbank(a,bank[off[f]:off[f+1]])-CONSTANTS['prior'][off[f]:off[f+1]]
        kmax=a.argmax(1);par=CONSTANTS['params'][off[f]+kmax]
        opts.append(np.column_stack((par,np.zeros(n,np.float32))));vals.append(a[np.arange(n),kmax])
    return np.stack(opts,axis=1),np.stack(vals,axis=1)

def blur(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge');return (p[:,1:-1,1:-1]*4+(p[:,2:,1:-1]+p[:,:-2,1:-1]+p[:,1:-1,2:]+p[:,1:-1,:-2])*2+p[:,:-2,:-2]+p[:,2:,2:]+p[:,:-2,2:]+p[:,2:,:-2])/16

def avg(a,m):return np.sum(a*m,axis=(-2,-1))/(.00001+np.sum(m,axis=(-2,-1)))

def normalize(a,mask):
    a=(a-avg(a,mask)[:,None,None])*np.sqrt(mask)
    v=a.reshape(len(a),-1);return v/np.sqrt(.001+(v*v).sum(1,keepdims=True))

def confighyp(cfg,f):
    # Small coordinate search around a coarse shape pose. Equal-size hypotheses for every input.
    p=cfg[:,None,:]+CONSTANTS['delta'][None,:,:];n=len(p)
    return p.reshape(n,-1,5)

def refine(ch,cfg,f):
    # Match the sign border to pairs of samples on its two sides.
    pts=CONSTANTS['probes'+str(f)];p=confighyp(cfg,f);n,h,j=p.shape
    cx,cy,rx,ry,sh=p.transpose(2,0,1)[:,:,:,None];u,v=pts.T
    xx=np.clip(cx+rx*u+sh*v,0,30.9999);yy=np.clip(cy+ry*v,0,30.9999)
    i=xx.astype(np.int32);j=yy.astype(np.int32);fx=(xx-i).astype(np.float32);fy=(yy-j).astype(np.float32)
    im=ch[:,0] if f in (0,1,4,6,7) else ch[:,1 if f==2 else 2]
    if f in (0,1,4):im=.87*im-.85*ch[:,3]
    if f in (2,3,6,7):im=.74*im-.46*ch[:,3]
    if f==5:im=ch[:,3]
    t=np.arange(n)[:,None,None];z=(((1-fx)*im[t,j,i]+fx*im[t,j,i+1])*(1-fy)+((1-fx)*im[t,j+1,i]+fx*im[t,j+1,i+1])*fy).reshape(n,h,3,-1)
    if f in (0,1,4):d=z[:,:,1]-.75*z[:,:,2]-.25*z[:,:,0]
    elif f==5:d=z[:,:,2]-z[:,:,0]
    else:d=z[:,:,2]-.60*z[:,:,1]-.40*z[:,:,0]
    score=np.clip(d,-.65,1.45).mean(-1)-CONSTANTS['dcost']
    return p[np.arange(n),score.argmax(1)]

def borderpose(ch,cfg,f):
    # Dense photometric likelihood; multiple boundary sectors must agree on the
    # pose, unlike an overall mean that can be dominated by just one bright edge.
    extra=CONSTANTS['fine'];p=cfg[:,None,:]+extra[None,:,:];n,h,j=p.shape
    pts=CONSTANTS['probes'+str(f)];u,v=pts.T;cx,cy,rx,ry,sh=p.transpose(2,0,1)[:,:,:,None]
    xx=np.clip(cx+rx*u+sh*v,0,30.9999);yy=np.clip(cy+ry*v,0,30.9999);i=xx.astype(np.int32);j=yy.astype(np.int32);fx=(xx-i).astype(np.float32);fy=(yy-j).astype(np.float32);t=np.arange(n)[:,None,None]
    im=ch[:,0]*.76-ch[:,3]
    z=(((1-fx)*im[t,j,i]+fx*im[t,j,i+1])*(1-fy)+((1-fx)*im[t,j+1,i]+fx*im[t,j+1,i+1])*fy).reshape(n,h,3,-1)
    inner=z[:,:,1]-z[:,:,2];outer=z[:,:,1]-z[:,:,0];v=.74*np.clip(inner,-.5,1.20)+.26*np.clip(outer,-.5,1.20)
    v=v.reshape(n,h,-1,3).mean(3);return p[np.arange(n),(v.mean(2)-.28*v.std(2)-CONSTANTS['finecost']).argmax(1)]

def inspect(ch,cfg,f):
    # Absolute color evidence is weak in coloured daylight, so also use the color *change* at the edge.
    xx,yy=grid(28,(-1.18,1.18,-1.18,1.18)); d=dist(xx,yy,f)
    k=(0,0,1,2,0,0,0,0)[f];c=pull(ch[:,k],cfg,xx,yy);l=pull(ch[:,3],cfg,xx,yy)
    th=(.18,.13,.08,.145,.15,.07,.07,.07)[f]
    ring=soft(-.035,.035,d)*(1-soft(th-.04,th+.04,d)); inner=soft(th,th+.07,d);outside=soft(-.19,-.10,d)*(1-soft(-.06,.005,d))
    cr,ci,co=avg(c,ring),avg(c,inner),avg(c,outside);lr,li,lo=avg(l,ring),avg(l,inner),avg(l,outside)
    if f in (0,1,4):
        if f==0:sec=[(xx>0,yy>0),(xx<0,yy>0),(xx<0,yy<0),(xx>0,yy<0)]
        else:
            vv=yy if f==1 else -yy;sec=[(vv<.68,xx<0),(vv<.68,xx>0),(vv>.75,np.abs(xx)<.82)]
        edges=np.stack([avg(c-l,ring*a*b) for a,b in sec],axis=1);mixed=ci-li
        reliable=np.clip(edges-mixed[:,None],-.8,1.6)
        return .70*(cr-ci)+.21*(cr-co)+.27*cr+.82*(li-lr)+.09*(li-lo)-.29*reliable.std(1)+.09*np.min(reliable,axis=1)
    if f==3:
        return .76*(ci-cr)+.26*(ci-co)+.40*ci+.78*(lr-li)+.20*(lr-lo)
    if f==5:
        stripe=inner*soft(.27,.13,np.abs(xx+yy));wh=inner*soft(.26,.41,np.abs(xx+yy))
        diag=avg(l,wh)-avg(l,stripe)
        # Look at the two remote ends of the band separately from the symbol it cancels.
        a=np.abs(yy)>.30;ends=avg(l,wh*a)-avg(l,stripe*a);anti=inner*soft(.26,.13,np.abs(xx-yy))
        selectivity=avg(l,anti*a)-avg(l,stripe*a)
        return 1.23*diag+.55*ends+.39*selectivity+.31*(avg(l,wh)-lo)-.34*np.abs(cr-ci)-.24*np.abs(cr)
    if f==2:
        # Color under a white arrow may be weak; look for blue in multiple
        # quadrants, rather than treating external sky as part of the disc.
        field=inner*(d>.22);parts=np.stack([avg(c,field*(xx*s>0)*(yy*t>0)) for s,t in ((1,1),(-1,1),(1,-1),(-1,-1))],1)
        return .61*(ci-cr)+.36*(ci-co)+.50*ci+.52*(lr-li)+.13*(lr-lo)+.23*(parts.min(1)-.16)
    text=(np.abs(yy)<(.20 if f==6 else .16))*(np.abs(xx)<(.59 if f==6 else .72))
    field=inner*(1-soft(.15,.29,.36-np.abs(yy)))
    cf=avg(c,field);lf=avg(l,field);ct=avg(c,text);lt=avg(l,text)
    abscontrast=(lt-lf)+.55*(cf-ct)
    # Along the middle of an entry bar, red-normalized brightness is
    # uniformly high. STOP instead has ink gaps right through its middle.
    center=(np.abs(yy)<.102)&(np.abs(xx)<.55);signal=l-.55*c
    qlo=np.quantile(signal[:,center],.17,axis=1);qhi=np.quantile(signal[:,center],.86,axis=1);fld=lf-.55*cf
    continuity=(qlo-fld)/(.065+np.maximum(.0,qhi-fld))
    if f==6:
        # STOP lettering has interleaved vertical strokes, not a continuous bar.
        u,v=grid(24,(-.74,.74,-.35,.35));txt=pull(ch[:,3]-.36*ch[:,0],cfg,u,v);pat=CONSTANTS['stopt'];corr=(normalize(txt,np.ones_like(pat))*normalize(blur(pat[None,:,:]),np.ones_like(pat))).sum(1)
        fieldtest=(d>.135)&(np.abs(yy)>.28);fieldred=np.quantile(c[:,fieldtest],.30,axis=1)
        # A speed-limit numeral is dark ink on pale ground; STOP has the
        # opposite polarity. Verify it in the central field, not at the rim.
        quiet=(d>.18)&(np.abs(yy)>.30)&(np.abs(yy)<.69);gap=avg(signal,quiet);lightletters=qhi-gap;darkletters=gap-qlo
        polarity=np.clip((lightletters-darkletters)/(.07+np.maximum(.0,qhi-qlo)),-1,1)
        return .42*(cf-cr)+.30*(cf-co)+.39*cf+.44*(lr-lf)+.36*abscontrast+.69*(corr-.20)+.36*(fieldred-.10)-1.05*np.maximum(0,continuity-.37)+.30*(polarity-.34)
    return .28*(cf-cr)+.28*(cf-co)+.45*cf+.27*(lr-lf)+.85*abscontrast+.54*(np.clip(continuity,-.5,1)-.42)

def icons(ch,cfg,g):
    xx,yy=grid(20,rects[g]);mask=CONSTANTS['mask'+str(g)]
    lum=pull(ch[:,3],cfg,xx,yy);red=pull(ch[:,0],cfg,xx,yy);blue=pull(ch[:,1],cfg,xx,yy)
    if g==2: ink=lum-.44*blue
    else:ink=-lum+(.14*red if g==0 else 0)
    if g==1:
        # Extrapolate local white paper inward. Ignore the red frame and shadows
        # outside the triangle when estimating how dark a pictogram stroke is.
        m=(mask>.35);p=np.pad(np.where(m,lum,-2),((0,0),(3,3),(3,3)),mode='edge');white=np.maximum.reduce([p[:,i:i+20,j:j+20] for i in (0,3,6) for j in (0,3,6)])
        high=np.quantile(lum[:,m],.90,axis=1).astype(np.float32)[:,None,None];ink=ink+.61*np.clip(white,high-.55,high+.06)
    if g==3:mask=mask*soft(.20,.312,np.abs(xx+yy))
    flat=avg(ink,mask);sd=np.sqrt(avg((ink-flat[:,None,None])**2,mask))
    contrast=sd/(.074+sd)
    z=ink; v=normalize(z,mask);w=normalize(blur(blur(z)),mask)
    a=dotbank(np.concatenate((v,w),axis=1),CONSTANTS['bank'+str(g)])-CONSTANTS['cost'+str(g)]
    cut=CONSTANTS['cuts'+str(g)];scores=[]
    for j,cl in enumerate(classes[g]):
        if cut[j]==cut[j+1]:s=np.maximum(-.25,1-sd/(.075 if g==0 else .074))
        else:s=a[:,cut[j]:cut[j+1]].max(axis=1)*(contrast*.41+.64)
        scores.append(s)
    sc=np.stack(scores,axis=1)
    if g==1:
        # Central printed ink provides a second alignment estimate, independent
        # of the red triangle. Fit a box by trimmed 1D ink mass at inference.
        valid=(mask>.8);bl=np.quantile(ink[:,valid],.20,axis=1).astype(np.float32);bh=np.quantile(ink[:,valid],.91,axis=1).astype(np.float32)
        weight=soft(.25,.70,(ink-bl[:,None,None])/(.022+bh[:,None,None]-bl[:,None,None]))*valid
        def span(m,z):
            c=np.cumsum(m,1);a=c[:,-1:];lo=(c<.036*a).sum(1).clip(0,19);hi=(c<.964*a).sum(1).clip(0,19);return z[lo],z[hi]
        xl,xr=span(weight.sum(1),xx[0]);yt,yb=span(weight.sum(2),yy[:,0]);xmid=(xl+xr)*.5;ymid=(yt+yb)*.5;height=yb-yt+.057;width=xr-xl+.06
        u,v=grid(24,(-.08,1.08,-.08,1.08));ax=xl[:,None,None]+(xr-xl+.001)[:,None,None]*u;ay=yt[:,None,None]+(yb-yt+.001)[:,None,None]*v
        tight=-pull(ch[:,3],cfg,ax,ay);m=np.ones((24,24),np.float32);tt=np.concatenate((normalize(tight,m),normalize(blur(blur(tight)),m)),1)
        da=dotbank(tt,CONSTANTS['details'])-CONSTANTS['detailcost'];ds=CONSTANTS['dsizes'];local=np.stack([da[:,a:b].max(1) for a,b in zip(ds[:-1],ds[1:])],1)
        sc=.75*sc+.25*local
        # At low resolution topology is lost, but gross glyph proportions remain informative.
        for j,cl in enumerate(classes[1]):
            # Physical aspect ranges from a very narrow exclamation mark to twin broad humps.
            ratio={11:.70,18:.23,19:.88,20:.88,21:.60,22:2.7,23:1.05,24:.59,25:1.23,26:.34,27:.66,28:1.03,29:1.56,30:1.04,31:1.35}[cl]
            aspect=np.log((width+.06)/(height+.06)/ratio);sc[:,j]-=.13*np.minimum(np.abs(aspect),1.7)
        wc=normalize(ink,mask); sm=blur(blur(ink));k=classes[1].index(31)
        # The two double bends are upright strokes joined by a transverse bar;
        # the deer's back and legs instead follow a continuous oblique axis.
        perp=(yy-ymid[:,None,None])-.70*(xx-xmid[:,None,None]);diagonal=np.exp(-((perp/.15)**2));diagmass=(weight*diagonal).sum((1,2))/(.01+weight.sum((1,2)))
        # A smooth oblique body has diagonal gradients concentrated on its
        # long edges. A double bend instead contains mostly vertical and
        # horizontal edges, even when its bounding rectangle is oblique.
        z=blur(ink);dx=z[:,1:-1,2:]-z[:,1:-1,:-2];dy=z[:,2:,1:-1]-z[:,:-2,1:-1];mm=valid[1:-1,1:-1]*(np.abs(xx[1:-1,1:-1])<.46)*(yy[1:-1,1:-1]<.54)
        tensor=(-2*dx*dy*mm).sum((1,2))/(.014+((dx*dx+dy*dy)*mm).sum((1,2)))
        sc[:,k]+=.40*(diagmass-.54)+.38*(tensor-.12);sc[:,classes[1].index(21)]-=.16*np.maximum(0,tensor-.23)
    if g==2:
        # Roundabout arrows encircle an empty center; all other blue signs
        # contain a shaft passing through or near the middle.
        r=np.hypot(xx,yy);core=avg(ink,r<.28);edge=avg(ink,(r>.46)&(r<.72));pool=(r<.81)
        dynamic=np.quantile(ink[:,pool],.90,axis=1)-np.quantile(ink[:,pool],.20,axis=1)
        hollow=np.clip((edge-core)/(.045+dynamic),-.65,.65)
        sc[:,-1]+=.58*hollow
    if g==0:
        sc=readinterior(ch,cfg,sc)
    if g==3:
        # Residual ink, counted only well away from the diagonal and edge.
        use=(np.hypot(xx,yy)<.70)&(np.abs(xx+yy)>.32)
        bg=(np.hypot(xx,yy)<.79)&(np.abs(xx+yy)>.35)
        white=np.quantile(lum[:,bg],.8,axis=1).astype(np.float32)
        q=np.maximum(0,white[:,None,None]-lum)
        hor=(np.abs(yy)<.27)&(np.abs(xx)>.20)&(np.abs(xx)<.68)&use
        str1=np.quantile(q[:,use],.87,axis=1);str2=np.quantile(q[:,hor],.80,axis=1)
        # Stripes remain even on a blank cancellation sign. Measure the
        # remaining ink with respect to stripe contrast in the same image.
        stripe=(np.abs(xx+yy)<.145)&(np.hypot(xx,yy)<.75);depth=white-avg(lum,stripe)
        strength=(.52*str1+.48*str2)/(.045+np.maximum(.03,depth))
        # Smooth broad illumination across the disc is not a printed vehicle.
        p=np.pad(lum,((0,0),(2,2),(2,2)),mode='edge');peak=np.maximum.reduce([p[:,i:i+20,j:j+20] for i in (0,2,4) for j in (0,2,4)])
        fine=np.quantile((peak-lum)[:,hor],.82,axis=1)/(.048+np.maximum(.03,depth))
        sc[:,1]=np.clip(1.03-1.32*(.50*strength+.50*fine),-.30,.85)
    return sc

def inkmap(ch,cfg):
    xx,yy=grid(32,(-.84,.84,-.84,.84));l=pull(ch[:,3],cfg,xx,yy);r=pull(ch[:,0],cfg,xx,yy);v=l-.14*r
    mask=np.hypot(xx,yy)<.68
    side=(np.abs(xx)<.46)&(np.abs(yy)>.46)&(np.hypot(xx,yy)<.72)
    white=np.quantile(v[:,mask],.84,axis=1).astype(np.float32)
    # Limit smooth illumination gradients by estimating the local bright paper.
    p=np.pad(v,((0,0),(5,5),(5,5)),mode='edge');a=np.maximum.reduce([p[:,i:i+32,j:j+32] for i in (1,5,9) for j in (1,5,9)])
    ink=np.maximum(0,.78*a+.22*white[:,None,None]-v)
    profile=(np.abs(xx)<.56)&(np.abs(yy)<.41)
    amp=np.quantile(ink[:,profile],.87,axis=1).astype(np.float32)
    paper=np.quantile(ink[:,side],.7,axis=1).astype(np.float32)
    return ink,xx,yy,amp,paper

def readinterior(ch,cfg,sc):
    ink,x,y,amp,paper=inkmap(ch,cfg);n=len(cfg)
    # Segmentation treats the number as a printed word. Its overall extent is
    # estimated from ink mass; each candidate reading must explain all its glyphs.
    usable=(np.hypot(x,y)<.728)&(np.abs(y)<.53)
    pix=soft(.24,.69,(ink-paper[:,None,None]*.55)/(.032+amp[:,None,None]))*usable
    pix*=soft(.17,.36,(ink+.005)/(.04+amp[:,None,None]))
    px=pix.sum(1); py=pix.sum(2)
    def interval(m,axis,lo,hi):
        cum=np.cumsum(m,axis=1);z=cum[:,-1:];u=(cum<z*lo).sum(axis=1).clip(0,31);v=(cum<z*hi).sum(axis=1).clip(0,31)
        return axis[u]-.023,axis[v]+.023
    xl,xr=interval(px,x[0],.021,.979);yt,yb=interval(py,y[:,0],.023,.977)
    cy=np.clip((yt+yb)*.5,-.13,.13);h=np.clip(yb-yt,.54,1.02);cx=np.clip((xl+xr)*.5,-.14,.14);total=np.clip(xr-xl,.61,1.49)
    # Crop using sign-relative coordinates; independent OCR metric validates template matching.
    u,v=grid(18,(-.11,1.11,-.08,1.08));src=ink
    def readat(pos,width):
        par=np.column_stack(((15.5+(cx+pos-total*.5)*32/1.68).astype(np.float32),(15.5+(cy-h*.5)*32/1.68).astype(np.float32),width*32/1.68,h*32/1.68,np.zeros(n,np.float32)))
        im=pull(src,par,u,v);co=blur(blur(im));m=np.ones((18,18),np.float32)
        f=np.concatenate((normalize(im,m),normalize(co,m)),1);a=dotbank(f,CONSTANTS['ocrbank']);cut=CONSTANTS['ocrcuts']
        part=np.concatenate([normalize(im[:,i:i+6,j:j+6],m[:6,:6]) for i in (0,6,12) for j in (0,6,12)],1);a=.64*a+.36*dotbank(part,CONSTANTS['ocrparts'])
        return np.stack([a[:,i:j].max(1) for i,j in zip(cut[:-1],cut[1:])],1)
    w=total*.44;first=readat(total*0,w);last=readat(total-w,w)
    first3=readat(total*0,total*.23);mid=readat(total*.295,total*.317);last3=readat(total*.687,total*.313)
    two=.62*first[:,[2,3,4,5,6,7]]+.38*last[:,0:1];three=.32*first3[:,1:2]+.40*mid[:,[0,2]]+.28*last3[:,0:1]
    ocr=np.concatenate((two,three),1)
    sc[:,:8]=.65*sc[:,:8]+.35*ocr
    # A blank disc should not turn into a number just because the outer ring
    # leaks into the sampled window. Compare the central stroke contrast with
    # the contrast in the white margin and the sign rim.
    flat=np.abs(y)<.42;central=flat&(np.abs(x)<.55);cap=(np.hypot(x,y)<.69)&(np.abs(y)>.40)
    # Use a contrast ratio, not a threshold on absolute brightness. Shadowed
    # numerals can have very little global contrast yet still form clear strokes.
    raw=pull(ch[:,3],cfg,x,y);qhi=np.quantile(raw[:,central],.86,axis=1);qlo=np.quantile(raw[:,central],.13,axis=1)
    back=avg(raw,cap);core=np.quantile(raw[:,central],.25,axis=1)
    rimx,rimy=grid(30,(-1.,1.,-1.,1.));rr0=np.hypot(rimx,rimy);edge=avg(pull(ch[:,3],cfg,rimx,rimy),(rr0>.845)&(rr0<.97));ringdepth=np.maximum(.055,back-edge)
    signal=(.70*np.maximum(0,amp-.36*paper)+.30*np.maximum(0,back-core))/(.04+ringdepth)
    sc[:,10]=np.minimum(.786,1.05-2.8*signal)-.22*np.maximum(0,sc[:,:8].max(1)-.57)
    # A black silhouette, two black / red cars, and printed numerals also differ in hue.
    left=(x<-.07)&(x>-.58)&(np.abs(y)<.19);right=(x>.06)&(x<.59)&(np.abs(y)<.19)
    rr=pull(ch[:,0],cfg,x,y);warm=avg(rr,left)-avg(rr,right)
    sc[:,8:10]+=.20*np.maximum(0,warm[:,None]);sc[:,11]-=.22*np.maximum(0,warm)
    # A projection-based foreground detector is an independent source of scale
    # for the whole-word templates. It helps if the red ring is not concentric
    # with the word (for instance after a perspective distortion).
    u,v=grid(20,rects[0]);m=CONSTANTS['mask0'];b=CONSTANTS['bank0'];cost=CONSTANTS['cost0'];cut=CONSTANTS['cuts0']
    for length,j0,j1 in ((1.02,0,6),(1.17,6,8)):
        ww=np.clip(total/length,.71,1.34);hh=h/(.84 if j0==0 else .78)
        par=np.column_stack((15.5+cx*32/1.68,15.5+cy*32/1.68,ww*32/1.68,hh*32/1.68,np.zeros(n,np.float32)))
        word=pull(ink,par,u,v);wordfeat=np.concatenate((normalize(word,m),normalize(blur(blur(word)),m)),axis=1)
        mat=dotbank(wordfeat,b[cut[j0]:cut[j1]])-cost[cut[j0]:cut[j1]]
        res=np.stack([mat[:,cut[j]-cut[j0]:cut[j+1]-cut[j0]].max(1) for j in range(j0,j1)],1)
        sc[:,j0:j1]=.55*sc[:,j0:j1]+.25*res+.20*np.maximum(res,sc[:,j0:j1])
    # Condition on the number of glyphs before inspecting the distinctive
    # part of the word. A leading 1 should not be penalized for a bad fit
    # of the last zero shared by both three-digit classes.
    for length,j0,j1,ind in ((1.02,0,6,0),(1.17,6,8,1)):
        mask=CONSTANTS['wordmasks'][ind];partbank=CONSTANTS['speedparts'][cut[j0]:cut[j1]]
        baseword=-pull(ch[:,3],cfg,u,v);ww=np.clip(total/length,.71,1.34);hh=h/(.84 if ind==0 else .78)
        par=np.column_stack((15.5+cx*32/1.68,15.5+cy*32/1.68,ww*32/1.68,hh*32/1.68,np.zeros(n,np.float32)))
        aligned=pull(ink,par,u,v);cor=[]
        for wim in (baseword,aligned):
            feat=np.concatenate((normalize(wim,mask),normalize(blur(blur(wim)),mask)),axis=1)
            # Pair a local and a whole-word likelihood at the very same pose:
            # the zero provides an anchor, not a free extra template shift.
            wf=np.concatenate((normalize(wim,m),normalize(blur(blur(wim)),m)),axis=1)
            s=.56*dotbank(feat,partbank)+.44*dotbank(wf,b[cut[j0]:cut[j1]])-cost[cut[j0]:cut[j1]]
            cor.append(np.stack([s[:,cut[j]-cut[j0]:cut[j+1]-cut[j0]].max(1) for j in range(j0,j1)],1))
        res=.46*cor[0]+.39*cor[1]+.15*np.maximum(cor[0],cor[1]);sc[:,j0:j1]=.60*sc[:,j0:j1]+.40*res+.35*(res-res.max(1,keepdims=True))
    return sc

def classify(img):
    n=len(img);ch=colors(img);opts,vals=localize(ch);allscore=np.full((n,43),-10,np.float32)
    bases=[];poses=[]
    for f in range(8):
        cfg=refine(ch,opts[:,f],f)
        if f in (0,1,4):cfg=borderpose(ch,cfg,f)
        poses.append(cfg);bases.append(inspect(ch,cfg,f))
    base=np.stack(bases,axis=1)
    for g,f in enumerate((0,1,2,5)):
        sc=icons(ch,poses[f],g);allscore[:,classes[g]]=base[:,f:f+1]+.89*(sc-.56)
    for f,c in ((3,12),(4,13),(6,14),(7,17)):allscore[:,c]=base[:,f]
    return allscore.argmax(1)

def predict(x):
    ans=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):ans[i:i+32]=classify(x[i:i+32])
    return ans
