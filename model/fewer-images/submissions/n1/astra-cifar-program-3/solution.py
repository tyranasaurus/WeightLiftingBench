import numpy as np


def blur(z):
    a=np.concatenate((z[..., :1,:],z,z[..., -1:,:]),axis=-2)
    a=(a[..., :-2,:]+2*a[..., 1:-1,:]+a[..., 2:,:])*.25
    b=np.concatenate((a[..., :1],a,a[..., -1:]),axis=-1)
    return (b[..., :-2]+2*b[..., 1:-1]+b[..., 2:])*.25


def gradients(z):
    a=np.concatenate((z[..., :1],z,z[..., -1:]),axis=-1)
    b=np.concatenate((z[..., :1,:],z,z[..., -1:,:]),axis=-2)
    return (a[..., 2:]-a[..., :-2])*.5,(b[..., 2:,:]-b[..., :-2,:])*.5


def orientations(z, color=False):
    gx,gy=gradients(.52*z+.48*blur(z)); xx=gx*gx;yy=gy*gy;xy=gx*gy
    if color:
        # Structure tensor: use achromatic edges and also isoluminant pigment changes.
        xx=.72*xx[:,0]+.17*(xx[:,1]+xx[:,2]);yy=.72*yy[:,0]+.17*(yy[:,1]+yy[:,2]);xy=.72*xy[:,0]+.17*(xy[:,1]+xy[:,2])
    v=xx+yy
    mg=np.sqrt(v+.00000001); a=(xx-yy)/(v+.00000001);b=2*xy/(v+.00000001)
    power=np.minimum(.34,mg)**.80
    channels=[]
    for k in range(6):
        ang=k*np.pi/3
        channels.append(power*np.maximum(0.,(a*np.cos(ang)+b*np.sin(ang)-.38)/.62))
    return np.stack(channels,axis=-3).astype(np.float32)


def grid1(count,center,span,n):
    x=np.arange(count);u=center+((np.arange(n)+.5)/n-.5)*span; r=max(span/n,.8)
    a=np.maximum(0,1-np.abs(u[:,None]-x[None,:])/r)
    # Replicated image boundary, including crops extending beyond the frame.
    a[:,0]+=np.maximum(0,1-(u+1)/r)*(u<r-1)
    a[:,-1]+=np.maximum(0,1-(count-u)/r)*(count-u<r)
    return (a/np.maximum(a.sum(1,keepdims=True),.001)).astype(np.float32)


def unit(z,reg):
    return z/np.sqrt((z*z).sum(axis=-1,keepdims=True)+reg*reg)


def descriptor(h,g,m):
    # h: six unsigned gradient bins; g: luminance; m: objectness, all on an 8 x 8 grid.
    n=g.shape[0]; energy=h.sum(1,keepdims=True)
    h=h/(.025+blur(energy)*.86+energy*.14)
    h=np.minimum(h,1.0);h=h-.68*h.mean(1,keepdims=True)
    h=h-.63*h.mean((2,3),keepdims=True)
    h=unit(h.reshape(n,-1),.55)
    slow=blur(blur(g)); detail=g-slow
    g=g-g.mean((1,2),keepdims=True)
    # Suppress smooth gradients, which can come from either lighting or the backdrop.
    g=unit((.41*g+1.4*detail).reshape(n,-1),.30)
    m=m-m.mean((1,2),keepdims=True);m=unit((m+.36*(m-blur(m))).reshape(n,-1),.85)
    return np.concatenate((h,g,m),axis=1).astype(np.float32)


def sat(z):
    return np.clip(z,0.,1.)


def ramp(z,lo,hi):
    return sat((z-lo)/(hi-lo))


def summarize(z,w):
    return (z*w).sum((-2,-1))/(.00001+w.sum((-2,-1)))


def zones():
    y,x=np.mgrid[:32,:32];x=(x-15.5)/16;y=(y-15.5)/16
    c=np.exp(-2.3*(x*x+y*y));c=(c/c.max()).astype(np.float32)
    t=sat((-y-.12)*2.6).astype(np.float32);b=t[::-1];rim=(np.maximum(abs(x),abs(y))>.73).astype(np.float32)
    return c,t,b,rim


center,top,base,border=zones()

def objectmap(x,g,texture):
    sm=blur(blur(x));n=len(g);x8=sm.reshape(n,3,8,4,8,4).mean((3,5));border8=np.concatenate((x8[:,:,0,:],x8[:,:,-1,:],x8[:,:,1:-1,0],x8[:,:,1:-1,-1]),axis=2)
    yy,xx=np.mgrid[:32,:32];py,px=np.mgrid[:8,:8]*4+1.5;by=np.concatenate((py[0],py[-1],py[1:-1,0],py[1:-1,-1]));bx=np.concatenate((px[0],px[-1],px[1:-1,0],px[1:-1,-1]))
    dist=np.full(g.shape,100.,np.float32)
    for i in range(28):
        delta=sm-border8[:,:,i,None,None];dy=.25*delta[:,0]+.5*delta[:,1]+.25*delta[:,2]
        diff=dy*dy*.67+.56*((delta[:,0]-delta[:,1])**2+(delta[:,2]-delta[:,1])**2)
        spatial=((xx-bx[i])/32)**2+((yy-by[i])/32)**2
        dist=np.minimum(dist,diff*(1+spatial)+.004*spatial)
    dist=np.sqrt(dist+.00001);sal=dist/(dist+.115)
    # A very balanced texture field is weak evidence for a localized subject.
    return blur(sal*(.71+.29*center)+.085*center+.04*texture/(.08+texture)).astype(np.float32)


def windowgrids():
    wins=[(0,0,32,32,0),(0,0,25,25,.012),(0,0,39,39,.014),(0,0,32,25,.012),(0,0,27,34,.014),(0,0,20,20,.035)]
    for side in (25,33):
        for dx,dy in ((-3.5,-2.5),(3.5,-2.5),(-3.5,2.5),(3.5,2.5),(0,-3.6),(0,3.6)):
            wins.append((dx,dy,side,side,.016 if side==33 else .024))
    ax=np.stack([grid1(32,15.5+v[0],v[2],8) for v in wins]);ay=np.stack([grid1(32,15.5+v[1],v[3],8) for v in wins]);cost=np.array([v[4] for v in wins],dtype=np.float32)
    return ax,ay,cost


horizontalgrid,verticalgrid,windowcost=windowgrids()

def shapes(h,g,m):
    n=len(g); maps=np.concatenate((h,g[:,None],m[:,None]),axis=1)
    dlist=[]
    for i in range(len(windowcost)):
        patch=verticalgrid[i] @ maps @ horizontalgrid[i].T
        hh=patch[:,:6];gg=patch[:,6];mm=patch[:,7]
        dlist.append(descriptor(hh,gg,mm));dlist.append(descriptor(hh[:,[0,5,4,3,2,1],:,::-1],gg[:,:,::-1],mm[:,:,::-1]))
    features=np.stack(dlist); features[:,:,:384]*=.53;features[:,:,384:448]*=.29; features[:,:,448:]*=.18
    sim=(features.reshape(-1,512) @ CONSTANTS['models']).reshape(len(dlist),n,10,-1)
    sim=sim-np.repeat(windowcost,2)[:,None,None,None];best=sim.max(0).reshape(n,10,-1,2).max(3)
    # Combine a best pose with corroboration from a few distinct drawings.
    k=np.partition(best,-4,axis=2)[:,:,-4:];peak=.46*k.max(2)+.54*k.mean(2)
    return 12.*(peak-peak.mean(1,keepdims=True))


def symmetric(g,c):
    # Evaluate possible vertical axes; mostly useful for large frontal faces.
    n=len(g);best=np.zeros(n,np.float32)
    high=g-blur(blur(blur(g)));u=blur(g)
    for dx in [-3,0,3]:
        l=16+dx;w=min(l,32-l,12); a=high[:,4:27,l-w:l];b=high[:,4:27,l:l+w][:,:,::-1]
        r=(a*b).mean((1,2))/np.sqrt((.00035+(a*a).mean((1,2)))*(.00035+(b*b).mean((1,2))))
        a=u[:,4:27,l-w:l];b=u[:,4:27,l:l+w][:,:,::-1];color=1-sat(np.abs(a-b).mean((1,2))/.24)
        best=np.maximum(best,r*.77+.23*color)
    return ramp(best,.26,.70)


def eye_pairs(g):
    # Dark eye sockets bracket a lighter bridge, with a nose below the eyes.
    sm=blur(g); broad=blur(blur(blur(sm)));neg=sat((broad-sm-.009)/.13)
    best=np.zeros(len(g),np.float32);longnose=best.copy(); triangular=best.copy()
    for half in (2,3,4,5,6,7):
        for tilt in (-1,0,1):
            x0=max(half,8);x1=min(32-half,25);y0=5;y1=19
            left=neg[:,y0:y1,x0-half:x1-half]; right=neg[:,y0+tilt:y1+tilt,x0+half:x1+half]
            middle=sm[:,y0:y1,x0:x1];a=sm[:,y0:y1,x0-half:x1-half];b=sm[:,y0+tilt:y1+tilt,x0+half:x1+half]
            pair=np.sqrt(left*right)*sat((middle-(a+b)*.5+.01)/.13)*(1-.45*sat(abs(a-b)/.18))
            for ratio in (.78,1.38):
                off=int(round(half*ratio)); nose=neg[:,y0+off:y1+off,x0:x1]; score=pair*(.40+.60*nose)
                # Light fur on either side of a dark central nose.
                idx=max(1,half//2);cheeks=.5*(sm[:,y0+off:y1+off,x0-idx:x1-idx]+sm[:,y0+off:y1+off,x0+idx:x1+idx]);nr=sm[:,y0+off:y1+off,x0:x1]
                score=score*(.69+.31*sat((cheeks-nr)/.12));peak=score.max((1,2))
                if ratio<1: triangular=np.maximum(triangular,peak)
                else: longnose=np.maximum(longnose,peak)
    return triangular,longnose


def bands(gx,gy,g):
    ax=np.abs(gx);ay=np.abs(gy);em=ax+ay+.00001
    hh=np.maximum(0.,ay-1.25*ax);vv=np.maximum(0.,ax-1.25*ay)
    z=hh.mean(2);vm=vv.mean(1)
    hp=z[:,4:29].max(1); vp=vm[:,3:29].max(1)
    # Coherence along rows and columns, independent of overall contrast.
    yy=np.abs(gy.mean(2));xx=np.abs(gx.mean(1))
    hr=(yy[:,3:29].max(1)+.48*hp)/(.035+em.mean((1,2)))
    vr=(xx[:,3:29].max(1)+.55*vp)/(.035+em.mean((1,2)))
    # Directional structure at middle scales, without sensitivity to gradient sign.
    aa=blur(blur(gx*gx));bb=blur(blur(gy*gy));cc=blur(blur(gx*gy))
    geom=np.sqrt((aa-bb)**2+4*cc*cc); orient=geom/(aa+bb+.00045)
    rect=summarize((np.abs(aa-bb)-1.15*np.abs(2*cc))/(aa+bb+.00045),center)
    rect=ramp(rect,.015,.37)
    structured=ramp(summarize(orient,center),.35,.68)
    return ramp(hr,.65,2.),ramp(vr,.57,1.57),rect,structured


def wheel_pairs(g):
    # Circular dark tires with contrast to the road, repeated at the same height.
    n=len(g); yy,xx=np.mgrid[:32,:32];best=np.zeros(n,np.float32)
    for rad in (1.6,2.6,3.5):
        outer=[];inner=[];w=[]
        for j in range(12):
            ca=np.cos(np.pi*j/6);sa=np.sin(np.pi*j/6)
            def shifted(r):
                return g[:,np.clip(yy+int(round(r*sa)),0,31),np.clip(xx+int(round(r*ca)),0,31)]
            a=shifted(rad+.9);b=shifted(max(.8,rad-.52));outer.append(a);inner.append(b);w.append(1. if sa>=-.01 else .43)
        a=np.stack(outer); b=np.stack(inner); ww=np.array(w,dtype=np.float32)[:,None,None,None];diff=a-b
        contrast=(ww*np.clip(diff,-.09,.26)).sum(0)/sum(w)
        spokes=(ww*(diff>.025)).sum(0)/sum(w); dark=ramp(.57-b.mean(0),0,.40);hub=ramp(g-b.mean(0),-.01,.17)
        response=ramp(contrast,.019,.16)*ramp(spokes,.36,.80)*(.42+.39*dark+.19*hub)
        response=response[:,15:29,1:31]
        for dx in (int(rad*4.8),int(rad*6.3),int(rad*7.5)):
            l=response[:,:,:-dx];r=response[:,:,dx:];both=np.minimum(l,r)*.68+.32*np.sqrt(l*r)
            score=np.max(both,axis=(1,2)); off=np.maximum(np.minimum(l[:,1:],r[:,:-1]).max((1,2)),np.minimum(l[:,:-1],r[:,1:]).max((1,2)))
            best=np.maximum(best,np.maximum(score,.94*off))
    return ramp(best,.28,.82)


def rules(x,g,texture,m,gx,gy):
    n=len(g);r,gr,b=x[:,0],x[:,1],x[:,2];warm=r-b;maxrgb=x.max(1);chroma=maxrgb-x.min(1)
    den=.32+g; redgreen=(r-gr)/den; greenblue=(gr-b)/den; bluegreen=(b-gr)/den
    blue=sat(((-redgreen-1.8*greenblue)-.027)/.31)*ramp(g,.10,.32)
    green=sat((greenblue-.012)/.21)*sat(((-redgreen+.33*greenblue)+.006)/.115)
    brown=sat((redgreen-.016)/.19)*sat((greenblue+.012)/.19)
    vivid=ramp(chroma/(.48+g),.16,.42);neutral=1-ramp(chroma/(.33+g),.07,.27)
    white=ramp(g,.59,.87)*neutral; black=1-ramp(g,.105,.31)
    tx=ramp(texture,.012,.088);flat=1-ramp(texture,.012,.065)
    coldsky=blue*(.61+.39*flat); palesky=neutral*ramp(g,.56,.87)*flat
    vegetation=green*(.71+.29*tx)
    sweight=top+base+.18*border
    sb=summarize(coldsky,top);bb=summarize(blue,base);sg=summarize(vegetation,top);bg=summarize(vegetation,base);cg=summarize(green,center)
    br=summarize(brown,center);brbg=summarize(brown,base+.7*border);bn=summarize(brown,border)
    bt=summarize(coldsky,border); highsky=summarize(np.maximum(coldsky,.73*palesky),top)
    bluelower=summarize(blue*(.85+.15*tx),base);whitecenter=summarize(white,center);blackcenter=summarize(black,center)
    uncolored=summarize(neutral,center);paint=summarize(vivid*(1-green)*(1-brown),center)
    ft=summarize(flat,top);fb=summarize(flat,base);fc=summarize(flat,center);te=summarize(tx,center);tb=summarize(tx,border)
    # Distinguish a smooth sky down to the lower corners from a waterline.
    corners=summarize(flat,border); hline,vline,rect,structured=bands(gx,gy,g)
    wheel=wheel_pairs(g); sym=symmetric(g,center);catface,dogface=eye_pairs(g)
    central=summarize(m,center); mid=m[:,6:26,6:26].mean((1,2));mass=ramp(mid,.29,.65)
    # Vertical section through the body and its support plane.
    edgeabs=np.abs(gx)+np.abs(gy);contrast=g.std((1,2));brighttop=summarize(g,top);brightbase=summarize(g,base)
    lightground=summarize(white,base);darkground=summarize(black,base);blkc=summarize(black,center);neutbase=summarize(neutral,base)
    yd=np.arange(32,dtype=np.float32)[:,None];xd=yd.T
    loc=np.maximum(0.,m-(m.mean((1,2),keepdims=True)+.012));tot=loc.sum((1,2))+.03
    mx=(loc*xd).sum((1,2))/tot;my=(loc*yd).sum((1,2))/tot
    sx=np.sqrt((loc*(xd-mx[:,None,None])**2).sum((1,2))/tot+.01);sy=np.sqrt((loc*(yd-my[:,None,None])**2).sum((1,2))/tot+.01)
    wide=ramp((sx+1)/(sy+1),1.06,1.93);low=ramp(my,15.5,21.5)
    # Aerial background usually has very little fine-scale clutter anywhere.
    skyall=summarize(np.maximum(coldsky,.62*palesky),border);grassy=.55*bg+.28*sg+.17*cg
    water=bb*(.68+.32*(1-bg));outdoor=np.maximum(highsky,grassy)
    # Fur and feathers scatter contrasts in different orientations. A glossy body or a cabin has large panels.
    nature=.5*grassy+.33*brbg+.17*cg
    panels=fc*(.35+.65*structured); angular=.44*rect+.36*hline+.20*vline
    metallic=.39*angular+.27*wheel+.12*uncolored+.12*paint+.10*panels
    # Center color relative to the surroundings separates a brown animal from leaf litter.
    tananimal=br*(.56+.44*ramp(br-bn,-.08,.29));allbrown=.55*brbg+.45*br
    greenbody=cg*(.64+.36*ramp(cg-bg,-.13,.19))
    lf=summarize(flat,base)*neutbase
    textureout=ramp(tb,.17,.60); smoothout=ramp(corners,.38,.80)
    richdog=summarize(brown*(1-ramp(g,.55,.88)),center)
    shadowroad=summarize(black[:,15:28,:],center[15:28,:])
    # Animal vs. machine evidence; centered about broad hand-selected neutral conditions.
    hard=(rect-.37)*.92+(hline-.35)*.72+(vline-.36)*.44+(wheel-.37)*.82+(uncolored-.53)*.40+(paint-.06)*1.30-(nature-.21)*1.60
    pet=(sym-.27)*.56+(br-.24)*.46+(te-.47)*.28-(highsky-.34)*.85-(bb-.13)*.66
    face=np.maximum(catface,dogface);sc=np.zeros((n,10),np.float32)
    sc[:,0]=.20*hard+2.9*(skyall-.24)+1.6*(smoothout-.41)+.65*(wide-.43)+.66*(bt-.19)-1.6*(water-.17)*(1.15-fb)-.95*(hline-.39)-.61*(vline-.34)-1.35*(bg-.21)-.68*(br-.25)-.56*(wheel-.34)
    sc[:,8]=.38*hard+3.15*(water-.16)+1.25*(highsky-.34)+1.07*(hline-.37)+.50*(wide-.43)+.52*(low-.33)-1.1*(bg-.21)-.87*(wheel-.37)-.54*(smoothout-.4)*bt
    road=1.08*(lf-.26)+.40*(neutbase-.51)-.84*(bb-.15)-.54*(bg-.22)
    sc[:,1]=1.03*hard+road+.96*(wheel-.36)+.64*(wide-.45)+.51*(shadowroad-.21)-.53*(highsky-.36)-.42*(vline-.39)+.35*(paint-.08)
    sc[:,9]=1.12*hard+.74*road+.58*(wheel-.37)+.64*(vline-.38)+.63*(highsky-.34)+.44*(whitecenter-.19)+.46*(rect-.40)-.30*(wide-.45)
    sc[:,2]=-.38*hard+.82*(smoothout-.4)+1.05*(sb-.17)+.9*(sg-.25)+.62*(paint-.08)-.43*(bb-.17)+.48*(fc-.41)-.54*(br-.29)-.39*(sym-.30)-.47*(wide-.45)
    sc[:,3]=-.49*hard+.94*pet+.57*(catface-.46)+.60*(sym-.29)+.69*(textureout-.51)-.82*(grassy-.23)+.31*(brbg-.26)-.47*(wide-.45)-.33*(tb-te)
    sc[:,5]=-.45*hard+.93*pet+.68*(dogface-.45)+.56*(sym-.29)+.65*(richdog-.22)+.45*(blackcenter+whitecenter-.41)-.49*(wide-.45)-.31*(bt-.18)
    sc[:,4]=-.55*hard+2.02*(grassy-.24)+.74*(tananimal-.22)+.70*(textureout-.50)+.62*(brbg-.27)-.61*(paint-.09)-.41*(sym-.31)-.50*(highsky-.32)
    sc[:,7]=-.37*hard+1.36*(bg-.24)+1.33*(tananimal-.21)+.61*(highsky-.33)+.41*(contrast-.20)/.20-.60*(cg-.24)-.28*(sym-.30)-.32*(brbg-.24)
    sc[:,6]=-.56*hard+1.95*(greenbody-.17)+.97*(allbrown-.26)+.78*(brbg-.26)+.84*(textureout-.50)-1.43*(highsky-.33)-.49*(sym-.30)-.39*(whitecenter-.16)+.24*(wide-.45)
    return sc


def decide(x):
    g=.27*x[:,0]+.51*x[:,1]+.22*x[:,2]; comp=np.stack((g,x[:,0]-x[:,1],x[:,2]-x[:,1]),axis=1)
    h=orientations(comp,True);gx,gy=gradients(.52*g+.48*blur(g));texture=blur(blur(np.sqrt(gx*gx+gy*gy+.00000001)))
    m=objectmap(x,g,texture)
    return (shapes(h,g,m)+rules(x,g,texture,m,gx,gy)).argmax(1)


def predict(x):
    # Fixed work blocks also keep matrix operations identical for singleton and long batches.
    ans=np.empty(len(x),np.int64)
    for start in range(0,len(x),16):
        a=x[start:start+16];k=len(a);work=np.zeros((16,3,32,32),np.float32);work[:k]=a
        ans[start:start+k]=decide(work)[:k]
    return ans
