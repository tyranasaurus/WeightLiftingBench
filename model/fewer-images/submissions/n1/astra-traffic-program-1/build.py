import sys, os
import numpy as np

# All artwork here is defined in an abstract 100 by 100 drawing square.
# Compressing that square to an icon's aspect ratio is done at stencil time.
P=(np.arange(112)+.5)*100/112
X,Y=np.meshgrid(P,P)
def new(): return np.zeros_like(X)
def ell(cx,cy,rx,ry):return (((X-cx)/rx)**2+((Y-cy)/ry)**2<=1).astype(float)
def poly(*v):
    m=np.zeros(X.shape,bool)
    for (a,b),(c,d) in zip(v, v[1:]+v[:1]):
        if b!=d:m ^= ((Y>b)!=(Y>d)) & (X<(c-a)*(Y-b)/(d-b)+a)
    return m.astype(float)
def stroke(v,w):
    im=new()
    for (a,b),(c,d) in zip(v,v[1:]):
        dx,dy=c-a,d-b; t=np.clip(((X-a)*dx+(Y-b)*dy)/(dx*dx+dy*dy+1.e-10),0,1)
        im=np.maximum(im,((X-a-dx*t)**2+(Y-b-dy*t)**2 < (w/2)**2))
    return im
def bez(p,*cs):
    out=[p];a=np.asarray(p);t=np.arange(1,13)[:,None]/12
    for i in range(0,len(cs),3):
        b,c,d=np.array(cs[i:i+3]);out.extend((1-t)**3*a+3*t*(1-t)**2*b+3*t*t*(1-t)*c+t**3*d); a=d
    return list(map(tuple,out))
def many(*ims): return np.maximum.reduce(ims)
def strip(im):
    yy,xx=np.nonzero(im>.03)
    return np.pad(im[yy.min():yy.max()+1,xx.min():xx.max()+1],2)
def digit(n,style):
    # Coordinates of stroke centres in a type designer's em. Condensation of
    # the glyph happens only after stroking (verticals should not be too thin).
    w=[10.5,13.5,11.5,12.5][style%4]
    if n=='0':
        p=bez((24,31),(24,1),(76,1),(76,31),(76,40),(76,60),(76,69),(76,99),(24,99),(24,69),(24,59),(24,40),(24,31))
    if n=='1':
        p=[(32,24),(51,9),(51,91)]; im=stroke(p,w)
        if style==3: im=many(im,stroke([(34,91),(68,91)],w*.8))
        return strip(im)
    if n=='2':
        p=bez((24,29),(24,2),(73,0),(75,26),(77,40),(64,48),(52,62)) + [(24,91),(77,91)]
    if n=='3':
        if style<2:p=bez((24,24),(27,1),(74,1),(73,28),(72,40),(66,46),(47,48),(65,47),(77,56),(75,72),(73,99),(28,99),(23,76))
        else:p=[(26,10),(72,10),(46,46)]+bez((46,46),(83,39),(86,86),(59,91),(43,96),(28,90),(24,79))[1:]
    if n=='5':
        p=[(73,10),(29,10),(27,49)]+bez((27,49),(47,36),(75,38),(75,64),(79,99),(30,102),(24,78))[1:]
    if n=='6':
        p=bez((65,8),(50,30),(29,48),(25,62)) if style!=1 else bez((74,23),(65,0),(26,4),(25,40),(24,49),(24,59),(25,66))
        p+=bez((25,66),(25,38),(74,36),(75,65),(78,100),(24,101),(25,66))
    if n=='7': p=[(23,10),(77,10),(41,91)]
    if n=='8':p=bez((50,47),(16,46),(18,7),(50,8),(84,7),(83,47),(50,47),(14,48),(14,92),(50,92),(86,92),(86,49),(50,47))
    return strip(stroke(p,w))

def car(cx,cy,w,h,side=False):
    def path(v):return poly(*[(cx+w*(a-.5),cy+h*(b-.5)) for a,b in v])
    if side: return many(path([(0,.48),(.19,.39),(.32,.09),(.67,.09),(.84,.4),(.99,.49),(.99,.82),(0,.82)]),ell(cx-.28*w,cy+.28*h,.09*w,.20*h),ell(cx+.29*w,cy+.28*h,.09*w,.20*h))
    return path([(0,.45),(.11,.18),(.19,.02),(.8,.02),(.91,.19),(1,.44),(1,.86),(.92,.86),(.92,1),(.77,1),(.77,.86),(.22,.86),(.22,1),(.06,1),(.06,.86),(0,.86)])
def truck(cx,cy,w,h,left=False):
    def path(v):return poly(*[(cx+(-1 if left else 1)*w*(a-.5),cy+h*(b-.5)) for a,b in v])
    return many(path([(.01,.02),(.69,.02),(.69,.21),(.84,.21),(.99,.53),(.99,.81),(.01,.81)]),ell(cx+(-1 if left else 1)*w*(-.30),cy+h*.29,w*.11,h*.2),ell(cx+(-1 if left else 1)*w*.30,cy+h*.29,w*.11,h*.2))
def arrow(points,w,tip,l,wide):
    a=np.array(points[-1]); b=np.array(tip);d=b-a;d=d/np.linalg.norm(d);p=np.array([-d[1],d[0]])
    return many(stroke(points+[b-d*l*.65],w),poly(tuple(b),tuple(b-d*l+p*wide/2),tuple(b-d*l-p*wide/2)))
def ach(cx,cy,r,a,b):
    t=np.linspace(a,b,20)*np.pi/180
    return list(zip(cx+r*np.cos(t),cy+r*np.sin(t)))
def pict(i,s):
    if i in (9,10,41,42):
        side=s%2==1
        if i in (9,41):return many(.69*car(27,52,40,34,side),car(75,52,39,34,side))
        return many(.75*truck(29,50,52,53),car(77,62,36,28,side))
    if i==16:return truck(50,50,89,65,True)
    if i==11:
        # The thick route of the priority-road symbol continues straight;
        # the thinner secondary road normally meets it at right angles.
        if s==0:return many(poly((40,93),(40,36),(9,36),(50,4),(91,36),(60,36),(60,93)),stroke([(10,57),(89,57)],9))
        if s==1:return many(poly((40,93),(40,31),(31,31),(50,4),(69,31),(60,31),(60,93)),stroke([(10,56),(90,56)],11))
    if i==23 and s:
        # Slipping car tilted across its two curving skid tracks.
        carbody=car(49,31,52,40);a=-.37;rot=sample(carbody,((X-49)*np.cos(a)+(Y-31)*np.sin(a)+49)*112/100-.5,(-(X-49)*np.sin(a)+(Y-31)*np.cos(a)+31)*112/100-.5)
        return many(rot,stroke(bez((32,49),(20,61),(72,59),(52,72),(34,83),(14,77),(13,91)),6),stroke(bez((66,46),(54,60),(97,58),(83,71),(69,83),(48,80),(46,93)),6))
    if i==24 and s:
        return many(stroke([(23,91),(23,11)],10),stroke([(78,91),(78,76),*bez((78,76),(78,59),(52,53),(52,36)),(52,11)],10))
    if i==29 and s:
        def ring(cx,cy,r,w):return np.maximum(0,ell(cx,cy,r,r)-ell(cx,cy,r-w,r-w))
        return many(ring(23,71,20,5.8),ring(77,71,20,5.8),stroke([(23,71),(38,48),(49,70),(23,71)],4.8),stroke([(49,70),(66,47),(77,71)],4.8),ell(58,14,7.5,8),stroke([(52,25),(37,42),(54,50),(46,69)],9),stroke([(52,29),(57,42),(68,44)],6))
    if i==11:return poly((39,91),(39,53),(5,62),(39,35),(39,29),(50,4),(61,29),(61,35),(95,62),(61,53),(61,91))
    if i==18:return many(poly((40,8),(60,8),(56,68),(44,68)),ell(50,85,9,9))
    if i in (19,20):
        if s==1:p=[(25,93),(25,49),(29,36),(74,14)]
        else:p=bez((25,93),(25,66),(18,45),(37,30),(50,18),(62,15),(78,15))
        a=stroke(p,20);return a[:,::-1] if i==19 else a
    if i==21:
        p=bez((72,94),(73,74),(77,60),(58,56),(35,54),(26,59),(26,37),(26,29),(26,20),(26,7))
        a=stroke(p,19);return a if not s else a[:,::-1]
    if i==22:return poly((2,89),(2,78),*bez((3,78),(12,77),(14,44),(28,42),(40,40),(42,76),(49,74),(58,70),(59,41),(71,42),(86,42),(85,75),(97,78)),(98,89))
    if i==23:
        a=many(car(48,32,43,40),stroke(bez((32,50),(62,59),(10,65),(39,74),(53,78),(44,85),(25,93)),5),stroke(bez((62,50),(92,59),(40,65),(69,74),(83,78),(74,85),(55,93)),5));return a
    if i==24:return many(stroke([(22,91),(22,10)],9),stroke([(77,91),(77,69),(56,36),(56,10)],10))
    if i==25:
        man=many(ell(45,19,7,7),stroke([(41,29),(29,51),(34,61)],12),stroke([(33,57),(18,87)],8),stroke([(34,57),(47,69),(41,87)],8),stroke([(43,31),(49,47),(61,52)],6),stroke([(27,37),(68,68)],3));return many(man,poly((49,90),(62,72),(72,66),(79,69),(97,90)))
    if i==26:
        a=poly((31,14),(39,5),(61,5),(69,14),(69,85),(60,94),(40,94),(31,85));
        for y,v in [(23,.49),(50,.22),(77,.55)]:a=np.where(ell(50,y,13,11)>0,v,a)
        if s: a=many(.75*ell(50,19,16,15),.42*ell(50,50,16,15),.63*ell(50,81,16,15))
        return a
    if i in (27,28):
        def walker(cx,cy,z):
            def p(v):return [(cx+z*a,cy+z*b) for a,b in v]
            return many(ell(cx+z*6,cy-z*35,z*8,z*8),stroke(p([(4,-22),(-7,4)]),z*13),stroke(p([(-5,2),(-12,19),(-24,37)]),z*8),stroke(p([(-4,4),(10,16),(19,36)]),z*8),stroke(p([(3,-19),(12,-3),(23,1)]),z*6),stroke(p([(4,-20),(-12,-16),(-21,-2)]),z*6))
        if i==27:return walker(49,52,1)
        a=many(walker(35,47,.91),walker(68,64,.65));return a[:,::-1] if s else a
    if i==29:
        def ring(cx,cy,r,w):return np.maximum(0,ell(cx,cy,r,r)-ell(cx,cy,r-w,r-w))
        return many(ring(22,72,19,4),ring(79,72,19,4),stroke([(22,72),(39,46),(51,72),(22,72)],4),stroke([(51,72),(69,43),(79,72)],4),ell(58,17,7,7),stroke([(51,28),(38,42),(54,53),(50,68)],7),stroke([(51,29),(58,40),(69,42)],5))
    if i==30:
        a=new();
        for ang in np.arange(6)*np.pi/3 + (np.pi/6 if s else 0):
            d=np.array([np.cos(ang),np.sin(ang)]);p=np.array([-d[1],d[0]]);c=np.array([50,52]);a=many(a,stroke([c,c+43*d],5),stroke([c+36*d-9*p,c+24*d,c+36*d+9*p],4))
        return a
    if i==31:
        a=many(ell(52,48,27,13),poly((26,50),(23,27),(9,24),(12,17),(24,16),(32,23),(43,42)),stroke([(66,50),(79,64),(95,74)],5),stroke([(65,53),(73,69),(87,77)],5),stroke([(38,51),(29,65),(13,65)],5),stroke([(41,53),(33,73),(20,78)],4),stroke([(22,19),(15,5)],3));return a
    if i in (33,34):
        p=bez((30,89),(31,61),(23,35),(62,34)) if not s else [(31,91),(31,57),(35,42),(46,34),(62,34)]
        a=arrow(p,13,(94,34),29,59);return a if i==33 else a[:,::-1]
    if i==35:return arrow([(50,90),(50,54)],13,(50,7),32,62)
    if i in (36,37):
        a=many(arrow([(39,92),(39,47)],12,(39,5),27,48),arrow([(39,81),(39,56),(65,56)],12,(94,56),26,45));return a if i==36 else a[:,::-1]
    if i in (38,39):
        a=arrow([(15,15),(50,50)],17,(84,84),36,68)
        # Mandatory passing arrows often have an L-shaped chevron rather than a filled triangular point.
        if s==0:a=many(stroke([(15,15),(79,79)],16),poly((38,83),(83,83),(83,37),(70,37),(70,69),(38,69)))
        return a if i==38 else a[:,::-1]
    if i==40:
        a=new()
        for ang in [-90,30,150]:
            p=ach(50,50,34,ang+65,ang+10);t=ang*np.pi/180;b=np.array([50+34*np.cos(t),50+34*np.sin(t)]);a=many(a,arrow(p,8,b+6*np.array([np.sin(t),-np.cos(t)]),17,27))
        return a
    if i==14:
        # A block STOP wordmark, drawn in cap-height units.
        return many(stroke(bez((23,31),(16,14),(4,19),(4,39),(4,52),(24,47),(23,65),(23,85),(8,85),(3,71)),4),stroke([(27,22),(49,22)],5),stroke([(38,23),(38,79)],5),stroke(bez((53,39),(52,13),(73,14),(73,40),(73,45),(73,56),(73,62),(74,85),(52,85),(53,62),(53,55),(53,48),(53,39)),5),stroke([(82,79),(82,22),(89,22)],5),stroke(bez((89,22),(103,21),(103,53),(89,53)),5),stroke([(82,53),(90,53)],5))
    if i==17:return poly((4,30),(96,30),(96,70),(4,70))
    raise ValueError(i)

def sample(img,xx,yy):
    xx=np.clip(xx,0,img.shape[1]-1.001); yy=np.clip(yy,0,img.shape[0]-1.001)
    ix=xx.astype(int);iy=yy.astype(int);dx=xx-ix;dy=yy-iy
    return (img[iy,ix]*(1-dx)+img[iy,ix+1]*dx)*(1-dy)+(img[iy+1,ix]*(1-dx)+img[iy+1,ix+1]*dx)*dy

def place(img,x,y,cx,cy,w,h):
    return sample(img,((x-cx)/w+.5)*(img.shape[1]-4)+1.5,((y-cy)/h+.5)*(img.shape[0]-4)+1.5)

# Physical dimensions are in units of the outer sign radius, not pixels.
# Nominal width and height of each design, and its optical centre height.
nom={9:(1.30,.42,0),10:(1.32,.64,.01),11:(.80,.85,.26),14:(1.65,.51,0),16:(1.23,.85,0),17:(1.52,.32,0),18:(.15,.99,.19),19:(.56,.74,.30),20:(.56,.74,.30),21:(.49,.86,.23),22:(1.20,.30,.55),23:(.73,.79,.31),24:(.55,.83,.28),25:(1.14,.73,.34),26:(.31,.92,.24),27:(.64,.81,.28),28:(1.03,.80,.32),29:(1.12,.77,.34),30:(.85,.77,.34),31:(.99,.68,.33),33:(1.14,1.34,0),34:(1.14,1.34,0),35:(.95,1.56,0),36:(1.17,1.54,0),37:(1.17,1.54,0),38:(1.20,1.20,0),39:(1.20,1.20,0),40:(1.54,1.54,0)}

familyclasses=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),list(range(33,41)),[14,17],[6,32,41,42]]
# Pixel coordinates for the matched-filter interiors. A feathered aperture
# suppresses the coloured rim; the upward triangular aperture follows the two sides.
def stencil_geom(g):
    center=(np.arange(30)+.5)/15-1
    xx,yy=np.meshgrid(center,center)
    if g==1:xx*=.82;yy=yy*.78+.19; edge=np.minimum(.80-yy,((yy+.68)/1.90-abs(xx))*.91)
    else:
        rad=[.81,1,.95,.98,.99][g];xx*=rad;yy*=rad
        edge=([.782,1,.91,.94,.945][g]-np.hypot(xx,yy))
    aperture=np.clip(edge/.082+.38,0,1); keep=aperture>.01
    return xx,yy,aperture,keep

# Explicitly construct individual matched filters, never combining drawings.
def mk_stencils():
    result={}; digitcache={(str(n),s):digit(str(n),s) for n in [0,1,2,3,5,6,7,8] for s in range(4)}
    def drawn_number(text,x,y,w,h,cy,style,gap):
        spans=[.60 if n!='1' else .36 for n in text];length=sum(spans)+(len(text)-1)*gap
        left=-w/2;img=np.zeros_like(x)
        for n,span in zip(text,spans):
            ww=w*span/length;img=np.maximum(img,place(digitcache[n,style],x,y,left+ww/2,cy,ww,h));left+=ww+w*gap/length
        return img
    nums={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}
    for g,classes in enumerate(familyclasses):
        xx,yy,ap,keep=stencil_geom(g);a=ap[keep];a=a/a.sum();sq=np.sqrt(a);ax=xx[keep];ay=yy[keep]
        result['x'+str(g)]=ax.astype('f');result['y'+str(g)]=ay.astype('f');result['a'+str(g)]=a.astype('f')
        basis=np.array([np.ones_like(ax),ax,ay]); orth=[]
        for v in basis:
            for u in orth:v=v-u*(u*v*a).sum()
            orth.append(v/(v*v*a).sum()**.5)
        basis=np.array(orth);result['b'+str(g)]=basis.astype('f')
        # 3x oversampled within each cell, integrated before normalisation.
        scaleX=xx[0,1]-xx[0,0];scaleY=yy[1,0]-yy[0,0]
        sx=(ax[:,None,None]+np.array([-.32,0,.32])[None,None,:]*scaleX)+np.zeros((1,3,1));sy=(ay[:,None,None]+np.array([-.32,0,.32])[None,:,None]*scaleY)+np.zeros((1,1,3))
        arr=[];starts=[]; pri=[]
        if g==1:
            sep=(ax[:,None]-ax)**2+(ay[:,None]-ay)**2;nb=np.argsort(sep,axis=1)[:,:25];dec=np.exp(-np.take_along_axis(sep,nb,axis=1)/(2*.085**2));result['neighbors']=nb;result['decay']=dec.astype('f')
            FG=[];DM=[]
        def add(img,penalty):
            if g==1:
                ink=img.mean(axis=(1,2));FG.append((ink*a)/max(.0001,(ink*a).sum()));DM.append((ink[nb]*dec).max(1))
            v=img.mean(axis=(1,2));v=v-((basis*a)@v)@basis
            norm=np.sqrt((v*v*a).sum());arr.append((v*sq/max(norm,1.e-8)).astype('f'));pri.append(penalty)
        for k in classes:
            starts.append(len(arr))
            if k==15:add(np.zeros_like(sx),0);continue
            if k in nums:
                for h in ([.68,.80,.91,1.02] if k<7 else [.58,.70,.82,.93]):
                    for w in ([.92,1.06,1.20,1.34] if k<7 else [1.12,1.26,1.39]):
                        for style in range(4):
                            for gap in ([.10,.20,.30] if style<2 else [.16,.26]):
                                add(drawn_number(nums[k],sx,sy,w,h,0,style,gap),.010*(abs(w-1.16)+abs(h-.82)))
            elif g==4:
                symbol=strip(pict(9 if k==41 else 10,0)) if k in (41,42) else None
                for slope in [.80,1.13,1.48]:
                    for width in [.37,.53,.67]:
                        diag=(sx*slope+sy)/(1+slope*slope)**.5
                        for kind in [0,1]:
                            mask=(abs(diag)<width*.5)
                            ink=mask*(.88 if kind==0 else .62+.38*np.cos(diag/width*5*2*np.pi))
                            if k==6:shade=drawn_number('80',sx,sy,1.18,.80,0,1,.18)
                            elif k==32:shade=np.zeros_like(ink)
                            else:shade=place(symbol,sx,sy,0,0,1.32,.50 if k==41 else .69)
                            for level in ([.48,.80] if k!=32 else [0]):add(np.maximum(ink,shade*level),0)
            else:
                w,h,cy=nom[k]
                styles=2 if k in (9,10,19,20,21,26,28,33,34,38,39) else 1
                if k==11:styles=3
                if k in (23,24,29):styles=2
                for style in range(styles):
                    icon=strip(pict(k,style))
                    for scx in [.78,1,1.20]:
                        for scy in [.80,1,1.18]:
                            for shifty in ([-.09,.0,.09] if g==1 else [0]):
                                if g==1:
                                    # Many pictograms stand on a common baseline, rather than expanding about their centre.
                                    dy=cy+shifty+h*(1-scy)*.25
                                else:dy=cy
                                add(place(icon,sx,sy,0,dy,w*scx,h*scy),.01*(abs(scx-1)+abs(scy-1)))
        if g==1:result['foreground']=np.stack(FG,1).astype('f');result['distance']=np.stack(DM,1).astype('f')
        st=str(g);result['t'+st]=np.stack(arr,1);result['offset'+st]=np.array(starts);result['cost'+st]=np.array(pri,dtype='f')
    return result

def detail_transform(x,y,a):
    # An isotropic band of spatial frequencies on the irregular aperture.
    # Both kernels are geometric and independent of the artwork.
    d=(x[:,None]-x)**2+(y[:,None]-y)**2
    def kernel(s):
        z=np.exp(-.5*d/s**2)*a[None,:];return z/z.sum(axis=1)[:,None]
    return ((kernel(.052)-.86*kernel(.165))*np.sqrt(a[:,None]/a[None,:])).astype('f')

def extra_stencils(z):
    for g in range(5):
        st=str(g);f=detail_transform(z['x'+st],z['y'+st],z['a'+st]);z['f'+st]=f
        detail=f@z['t'+st];z['detail'+st]=detail/np.sqrt(np.maximum(1.e-12,(detail*detail).sum(0)))
    gx,gy=np.meshgrid(((np.arange(17)+.5)/17-.5)*1.56,((np.arange(21)+.5)/21-.5)*1.35);gx=gx.flatten();gy=gy.flatten()
    a=np.exp(-.5*((gx/.53)**4+(gy/.60)**6));a/=a.sum();basis=np.stack([np.ones_like(a),gx/(gx*gx*a).sum()**.5,gy/(gy*gy*a).sum()**.5])
    z['gx']=gx.astype('f');z['gy']=gy.astype('f');z['ga']=a.astype('f');z['gb']=basis.astype('f');T=[]
    for s in '01235678':
        for style in range(4):
            d=digit(s,style)
            for blur in [.45,1.15]:
                # Optical softening is applied independently to each glyph.
                img=np.zeros_like(gx)
                for i in [-1,0,1]:
                    for j in [-1,0,1]:img+=place(d,gx+i*.075*blur,gy+j*.054*blur,0,0,1,1)*(2-abs(i))*(2-abs(j))/16
                img-=((img*a)@basis.T)@basis;v=img*np.sqrt(a);T.append(v/max(.000001,np.linalg.norm(v)))
    z['glyph']=np.stack(T,axis=1).astype('f');return z

solution="import numpy as np\n\nclasses=((0,1,2,3,4,5,7,8,9,10,15,16),(11,18,19,20,21,22,23,24,25,26,27,28,29,30,31),(33,34,35,36,37,38,39,40),(14,17),(6,32,41,42))\n\ndef unit(a):\n    return np.clip(a,0,1)\n\ndef blur(p):\n    q=np.pad(p,((0,0),(1,1),(1,1)),mode='edge')\n    return (4*q[:,1:-1,1:-1]+2*(q[:,:-2,1:-1]+q[:,2:,1:-1]+q[:,1:-1,:-2]+q[:,1:-1,2:])+q[:,:-2,:-2]+q[:,2:,:-2]+q[:,:-2,2:]+q[:,2:,2:])*.0625\n\ndef contrast(v,region,percent=10):\n    sub=v[:,region,region].reshape(len(v),-1)\n    vals=np.percentile(sub,[percent,50,100-percent],axis=1).astype(np.float32)[:,:,None,None]\n    return np.clip((v-vals[1])/(vals[2]-vals[0]+.12),-1.1,1.1)\n\ndef maps(pic):\n    r,g,b=pic[:,0],pic[:,1],pic[:,2]\n    lr,lg,lb=[np.log(t+.007) for t in (r,g,b)]\n    # Log ratios discount exposure changes. A grey-green channel preserves\n    # contrast between a red rim and the retroreflective white face in shadow.\n    R=np.clip(np.minimum(lr-lg,(lr-lb)*.85+.10),-.9,1.8)\n    B=np.clip(np.minimum(lb-lr,(lb-lg)*1.7+.12),-1.1,1.7)\n    Y=np.clip(np.minimum((lr-lb)*.78,(lg-lb)), -1,1.6)-.55*np.maximum(0,lr-lg-.34)\n    light=np.log(.19*r+.61*g+.20*b+.006)\n    green=np.log(.05*r+.75*g+.20*b+.006)\n    L=contrast(green,slice(4,28),9); L2=contrast(light,slice(3,29),10)\n    ri=.67*contrast(R,slice(4,28))+.40*np.clip(R/.8,-1,1)\n    bi=.61*contrast(B,slice(4,28))+.44*np.clip(B/.8,-1,1)\n    yi=.55*contrast(Y,slice(4,28))+.50*np.clip(Y/.7,-1,1)\n    # Return two densities for each border: a colour/brightness likelihood\n    # for the material, and evidence for whether this is the right colour.\n    return [(ri-.57*L,R),(ri-.57*L,R),(ri-.58*L,R),(bi-.55*L,B),(ri-.49*L,R),(yi-.09*L,Y),(L2-.19*np.maximum(0,ri),-.75*np.maximum(R-.10,0)-.38*np.maximum(B-.13,0))],(light,green,R,B,Y)\n\ndef perimeter(shape):\n    if shape in (0,3,4,6):\n        t=(np.arange(48)+.5)*np.pi/24\n        return np.cos(t),np.sin(t),0.\n    if shape in (1,2):\n        # Three corner fillets of a triangle; outer extents are measured at\n        # the rounded vertices. Its homothetic centre is below its box centre.\n        v=np.array([[0,-1.10],[1.07,1],[-1.07,1]],dtype=float);fillet=.083;cy=.30\n    else:v=np.array([[0,-1],[1,0],[0,1],[-1,0]],dtype=float);fillet=.035;cy=0\n    pts=[]\n    for i,p in enumerate(v):\n        q=v[(i+1)%len(v)];prev=v[i-1];u=(1-fillet)*p+fillet*prev;w=(1-fillet)*p+fillet*q\n        t=(np.arange(3)[:,None]+.5)/3;pts.extend((1-t)**2*u+2*t*(1-t)*p+t*t*w)\n        t=(np.arange(13)[:,None]+.5)/13;pts.extend(w*(1-t)+t*((1-fillet)*q+fillet*p))\n    x,y=np.array(pts).T\n    if shape==2:y=-y;cy=-cy\n    return x,y,cy\n\ndef profiles(shape):\n    if shape==0:return ([.84,.94],[.63,.73,1.09],[.39,.41,.20])\n    if shape in (1,2):return ([.79,.94],[.52,.62,1.12],[.34,.44,.22])\n    if shape in (3,4):return ([.69,.86],[1.03,1.12],[.66,.34])\n    if shape==5:return ([.55,.70],[.85,.97,1.13],[.45,.46,.09])\n    return ([.84,.94],[1.06,1.18],[.67,.33])\n\ndef gridposes():\n    c=(np.arange(3)-1)*2.15+15.5; rad=np.array([9.1,10.9,12.8,14.6]);cx,cy,rx,ry=np.meshgrid(c,c,rad,rad,indexing='ij'); keep=abs(rx-ry)<3.85\n    return np.stack((cx[keep],cy[keep],rx[keep],ry[keep],0*rx[keep],0*ry[keep]),1).astype(np.float32)\n\nposes=gridposes()\n\ndef outlinebanks(shape):\n    v,w,cy=perimeter(shape);plus,minus,ww=profiles(shape)\n    p=poses.astype(float);arr=[]\n    for points,weights in ((plus+minus,[1/len(plus)]*len(plus)+[-a for a in ww]),(plus,[1/len(plus)]*len(plus))):\n        hist=np.zeros((len(p),1024))\n        for scale,weight in zip(points,weights):\n            x=np.clip(p[:,0:1]+p[:,2:3]*v*scale,0,30.9999);y=np.clip(p[:,1:2]+p[:,3:4]*(cy+(w-cy)*scale),0,30.9999);ix=x.astype(int);iy=y.astype(int);dx=x-ix;dy=y-iy\n            for j,coef in [(0,(1-dx)*(1-dy)),(1,dx*(1-dy)),(32,(1-dx)*dy),(33,dx*dy)]:np.add.at(hist,(np.arange(len(p))[:,None],iy*32+ix+j),coef*(weight/len(v)))\n        arr.append(np.ascontiguousarray(hist.T,dtype=np.float32))\n    return arr\n\noutlines=[outlinebanks(i) for i in range(7)]\nlinepts=[perimeter(i) for i in range(7)]\nstrength=[.30,.30,.30,.43,.39,.48,.85]\n\ndef interp(pix,x,y):\n    n=len(pix);xx=np.clip(x,0,30.99999);yy=np.clip(y,0,30.99999);ix=xx.astype(np.int32);iy=yy.astype(np.int32);a=xx-ix.astype(np.float32);b=yy-iy.astype(np.float32)\n    index=ix+32*iy+np.arange(n,dtype=np.int32).reshape((n,)+(1,)*(ix.ndim-1))*1024;img=pix.reshape(-1)\n    return (img[index]*(1-a)+img[index+1]*a)*(1-b)+(img[index+32]*(1-a)+img[index+33]*a)*b\n\ndef posecost(p):\n    return .010*((p[...,0]-15.5)**2+(p[...,1]-15.5)**2)/4+.009*((p[...,2]-12.2)**2+(p[...,3]-12.2)**2)/6+.015*((p[...,2]-p[...,3])/3)**2+.15*(p[...,4]**2+p[...,5]**2)\n\ndef trace(m,p,sh):\n    # Sample finer poses directly. Use the same signed profile as the banks.\n    vx,vy,cy=linepts[sh];plus,minus,ww=profiles(sh);x0=p[:,:,0,None];y0=p[:,:,1,None];rx=p[:,:,2,None];ry=p[:,:,3,None];kx=p[:,:,4,None];ky=p[:,:,5,None]\n    sc=-posecost(p)\n    for scale,weight,positive in zip(plus+minus,[1/len(plus)]*len(plus)+[-a for a in ww],[True]*len(plus)+[False]*len(minus)):\n        u=(scale*vx).astype(np.float32);v=(cy+(vy-cy)*scale).astype(np.float32);xx=x0+rx*(u+kx*v);yy=y0+ry*(v+ky*u)\n        if positive:q=m[2]\n        else:q=m[0]\n        sc+=weight*interp(q,xx,yy).mean(axis=-1)\n    return sc\n\ndef cancellation(m,col,p,pref):\n    n=len(p);u,v,cy=linepts[6];positions=np.broadcast_to(poses,(n,len(poses),6));u=u.astype('f');v=v.astype('f')\n    def evidence(positions):\n        def samp(img,u,v):\n            return interp(img,positions[:,:,0,None]+positions[:,:,2,None]*(u+positions[:,:,4,None]*v),positions[:,:,1,None]+positions[:,:,3,None]*(v+positions[:,:,5,None]*u))\n        inner=samp(m,.88*u,.88*v);outer=samp(m,1.095*u,1.095*v);edge=inner-outer\n        q=np.linspace(-.53,.53,15,dtype='f');ink=samp(m,q,-q);a=samp(m,q+.19,-q+.19);b=samp(m,q-.19,-q-.19);stripe=(a+b)*.5-ink\n        return .52*np.mean(np.minimum(abs(edge),1),-1)+.20*np.abs(np.mean(edge,-1))+.75*stripe.mean(-1)+.51*samp(col,.9*u,.9*v).mean(-1)-posecost(positions)\n    s=evidence(positions);v0=evidence(p[:,None,:])[:,0];j=s.argmax(1);p=np.where((s[np.arange(n),j]>v0)[:,None],poses[j],p);pref=np.maximum(s[np.arange(n),j],v0)\n    for d in [.85,.44]:\n        delta=np.eye(6,dtype='f')*np.array([d,d,d,d,d*.10,d*.10],dtype='f');c=p[:,None,:]+np.concatenate([delta,-delta]);s=evidence(c);j=s.argmax(1);score=s[np.arange(n),j];p=np.where((score>pref)[:,None],c[np.arange(n),j],p);pref=np.maximum(score,pref)\n    return p,pref\n\ndef localize(ms):\n    n=len(ms[0][0]);allp=[];allsc=[]\n    for sh,(m,col) in enumerate(ms):\n        h,hc=outlines[sh];s=m.reshape(n,-1)@h+strength[sh]*(col.reshape(n,-1)@hc)-posecost(poses)\n        j=s.argmax(1);p=poses[j].copy();pref=s[np.arange(n),j]\n        mat=(m,col,m+col*strength[sh]);eye=np.eye(6,dtype=np.float32)\n        for step in range(3):\n            if step==0:delta=eye*np.array([1.14,1.14,1.0,1.0,.119,.095],dtype='f')\n            elif step==1:delta=eye*np.array([.64,.64,.63,.63,.071,.059],dtype='f')\n            else:delta=eye*np.array([.35,.35,.37,.37,.043,.039],dtype='f')\n            # Parallel one-axis probes, then try their combined displacement.\n            opts=p[:,None,:]+np.concatenate([delta,-delta],axis=0);ss=trace(mat,opts,sh);j=ss.argmax(1);rows=np.arange(n);choice=opts[rows,j];best=ss[rows,j]\n            sign=(ss[:,:6]>ss[:,6:]).astype('f')*2-1;shift=sign*(np.maximum(ss[:,:6],ss[:,6:])>pref[:,None])*np.diag(delta)\n            combined=p+shift;sc=trace(mat,combined[:,None,:],sh)[:,0];keep=sc>best;choice=np.where(keep[:,None],combined,choice);best=np.maximum(best,sc);p=np.where((best>pref)[:,None],choice,p);pref=np.maximum(pref,best)\n        if sh==6:\n            p,pref=cancellation(m,col,p,pref)\n        allp.append(p);allsc.append(pref)\n    return allp,allsc\n\n# Multiple samples of a rectified sign permit subpixel translation and a\n# small residual roll without quantising the numbers or arrowheads.\nshifts=np.array([(u,v,r) for u in [-.085,0,.085] for v in [-.085,0,.085] for r in [0]]+[(u,v,r) for r in [-.115,.115] for u,v in [(0,0),(-.055,0),(.055,0),(0,-.055),(0,.055)]],dtype='f')\n\ndef coords(p,u,v):\n    x=p[:,0,None,None]+p[:,2,None,None]*(u+p[:,4,None,None]*v);y=p[:,1,None,None]+p[:,3,None,None]*(v+p[:,5,None,None]*u);return x,y\n\ndef readglyphs(im,p,rect):\n    ax=CONSTANTS['gx'];ay=CONSTANTS['gy'];c=CONSTANTS['ga'];base=CONSTANTS['gb'];T=CONSTANTS['glyph'];n=len(p)\n    cx,cy,ww,hh=[r.reshape(-1,1) for r in rect]\n    u=cx+ax*ww;v=cy+ay*hh;z=interp(im,*coords(p,u,v));z=(z-((z*c)@base.T)@base)*np.sqrt(c);norm=np.sqrt((z*z).sum(2,keepdims=True)+.00004);z=z/norm\n    return (z.reshape(-1,len(ax))@T).reshape(n,len(u),8,8).max(3)\n\ndef ocr(im,p):\n    n=len(p);scores=np.zeros((n,8),dtype='f')\n    for k in range(2):\n        widths=[.95,1.10,1.25,1.38] if k==0 else [1.05,1.19,1.34,1.44];heights=[.67,.82,.97] if k==0 else [.56,.71,.84]\n        args=np.meshgrid(np.array(widths,'f'),np.array(heights,'f'),np.array([-.070,0,.070],'f'),np.array([-.08,0,.08],'f'),np.array([.14,.27],'f'),indexing='ij')\n        w,h,cx,cy,gap=[a.ravel() for a in args];d=w/((2+gap) if k==0 else (2.61+2*gap));x0=cx-w/2\n        if k==0:\n            v=readglyphs(im,p,(x0+.5*d,cy,d,h))\n            zero=readglyphs(im,p,(cx+w/2-.5*d,cy,d,h))[:,:,0]\n            scores[:,:6]=(.61*v[:,:,2:]+.39*zero[:,:,None]-.10*(np.abs(cx)+np.abs(cy))[None,:,None]).max(1)\n        else:\n            one=readglyphs(im,p,(x0+.305*d,cy,.61*d,h))[:,:,1];mid=readglyphs(im,p,(x0+(1.11+gap)*d,cy,d,h));zero=readglyphs(im,p,(cx+w/2-.5*d,cy,d,h))[:,:,0]\n            scores[:,6:]=(.28*one[:,:,None]+.43*mid[:,:,[0,2]]+.29*zero[:,:,None]-.10*(np.abs(cx)+np.abs(cy))[None,:,None]).max(1)\n    return scores\n\ndef match(features,p,g):\n    light,green,R,B,Y=features;st=str(g);ax,ay,a,basis,T,cost,starts=[CONSTANTS[s+st] for s in ('x','y','a','b','t','cost','offset')]\n    u=ax+shifts[:,0,None]-shifts[:,2,None]*ay;v=ay+shifts[:,1,None]+shifts[:,2,None]*ax\n    xx,yy=coords(p,u,v);w=np.sqrt(a);n=len(p)\n    if g==2:im=light-.55*B\n    elif g==3:im=green-.32*R\n    elif g==0:im=-np.log(np.exp(light)+.03)+.07*R\n    elif g==1:im=-light\n    else:im=-light\n    im=.84*im+.16*blur(im);data=interp(im,xx,yy)\n    # Remove a plane (constant and two shading gradients) using a fixed,\n    # analytic aperture basis. Clip extremes to resist dirt and occlusion.\n    trend=((data*a)@basis.T)@basis;z=data-trend\n    power=np.sqrt((z*z*a).sum(-1,keepdims=True)+.00003);z=np.clip(z/power,-2.9,2.9)\n    z=(z-((z*a)@basis.T)@basis)*w;z=z/np.sqrt((z*z).sum(-1,keepdims=True)+.001)\n    sc=(z.reshape(-1,len(a))@T).reshape(n,len(shifts),-1);detail=z.reshape(-1,len(a))@CONSTANTS['f'+st].T;detail/=np.sqrt((detail*detail).sum(1,keepdims=True)+.001)\n    high=(detail@CONSTANTS['detail'+st]).reshape(n,len(shifts),-1);sc=.62*sc+.38*high;sc-=cost;sc-=.12*np.abs(shifts[:,:2]).sum(1)[None,:,None]\n    # Directed distance to an ink stroke, with a smooth saturation. This\n    # score permits small displacements of individual parts of a symbol\n    # (e.g. the rider's limbs) without moving the whole symbol.\n    if g==1:\n        prob=unit((data-trend)/power*.54+.18)\n        # Work on the best rigid shift per symbol family afterwards; the\n        # distance score is evaluated on the same shifts as correlation.\n        tang=CONSTANTS['neighbors'];near=prob[:,:,tang[:,0]]\n        for j in range(1,tang.shape[1]):\n            near=np.maximum(near,prob[:,:,tang[:,j]]*CONSTANTS['decay'][:,j])\n        soft=(near.reshape(-1,len(a))@CONSTANTS['foreground']).reshape(n,len(shifts),-1)\n        recur=((prob*a).reshape(-1,len(a))@CONSTANTS['distance']).reshape(n,len(shifts),-1)/np.maximum(.013,(prob*a).sum(2,keepdims=True))\n        cover=(soft+recur)*.5\n        # Coverage near random ink has a nonzero baseline; demand coverage\n        # in both directions, especially when the image itself is quite dark.\n        sc=.71*sc+.29*(cover*1.65-.62)\n    vals=np.maximum.reduceat(sc.max(1),starts,axis=1)\n    if g==0:\n        glyphs=ocr(im,p);vals[:,:8]=.67*vals[:,:8]+.33*glyphs\n        # A blank regulation disc is a null hypothesis, not a highly\n        # correlated symbol. Test the central area against an unprinted\n        # annulus, which shares its lighting and imaging noise. Detect ink\n        # at a finer scale as well, in case there is a shadow across the face.\n        u,v=np.meshgrid(np.linspace(-.72,.72,19,dtype='f'),np.linspace(-.72,.72,19,dtype='f'));rad=np.hypot(u,v);core=rad<.57;surround=(rad>.60)&(rad<.73)&(abs(v)>.42)\n        clean=-light;allpix=interp(clean,*coords(p,u,v));refs=allpix[:,surround];inside=allpix[:,core]\n        q=np.percentile(refs,[20,50,80],axis=1);iq=np.percentile(inside,[48,88],axis=1);margin=.022+.5*(q[2]-q[0]);ink=np.maximum(iq[1]-np.minimum(q[1]+.15*(q[2]-q[0]),iq[0]),0)/(margin+.02)\n        detailmap=clean-blur(blur(clean));hp=interp(detailmap,*coords(p,u,v));fine=np.percentile(hp[:,core],89,axis=1);base=np.percentile(abs(hp[:,surround]),60,axis=1)\n        texture=np.maximum(0,fine-1.6*base)/(.011+base);disorder=np.maximum(ink*.78,texture*.71)\n        vals[:,10]=.82-.55*disorder/(.72+disorder)\n    if g==3:\n        # A word leaves gaps between its vertical stems; a white no-entry bar does not.\n        ux=np.linspace(-.63,.63,25,dtype='f');vy=np.array([-.075,0,.075],dtype='f');u,v=np.meshgrid(ux,vy);prof=interp(im,*coords(p,u,v)).mean(1)\n        trend=(prof[:,:1]+(prof[:,-1:]-prof[:,:1])*np.linspace(0,1,25,dtype='f'));res=prof-trend;weave=((res[:,4:]-res[:,:-4])**2).mean(1)**.5\n        up=interp(im,*coords(p,ux[None,:],np.full((1,25),-.38,dtype='f'))).mean((1,2));down=interp(im,*coords(p,ux[None,:],np.full((1,25),.38,dtype='f'))).mean((1,2));height=prof.mean(1)-(up+down)*.5\n        uneven=weave/(.09+np.maximum(height,.07));vals[:,1]-=.34*np.clip(uneven-.16,0,1.1);vals[:,0]+=.12*np.clip(uneven-.20,0,1)\n    return vals\n\ndef meanface(a,p,scale=1.):\n    q=(np.arange(12,dtype='f')+.5)/6-1;u,v=np.meshgrid(q,q);keep=u*u+v*v<1\n    return interp(a,*coords(p,(u[keep]*scale)[None,:],(v[keep]*scale)[None,:]))[:,0].mean(1)\n\ndef points(sh,kind):\n    u=(np.arange(30,dtype='f')+.5)/15-1;x,y=np.meshgrid(u,u)\n    if sh in (1,2):\n        h=y if sh==1 else -y\n        # Barycentric shrinkage relative to triangular centroid.\n        z=np.maximum((h-.3)/.70,(abs(x)*2.10-h+.30)/1.40)\n    elif sh==5:z=np.abs(x)+np.abs(y)\n    else:z=np.sqrt(x*x+y*y)\n    mask=(z<(.57 if sh in (1,2) else .65)) if kind==0 else ((z>.81)&(z<.96))\n    return x[mask][None,:],y[mask][None,:]\n\nfacecoords=[points(k,0) for k in range(7)]\n\ndef structural(ms,features,pp,ss):\n    light,green,R,B,Y=features;out=[]\n    for sh,p in enumerate(pp):\n        m,col=ms[sh];vx,vy,cy=linepts[sh];plus,minus,ww=profiles(sh)\n        def at(image,sc):\n            tx=(sc*vx).astype('f')[None,:];ty=(cy+sc*(vy-cy)).astype('f')[None,:]\n            return interp(image,*coords(p,tx,ty))[:,0]\n        # Integrate support around the perimeter. An isolated piece of\n        # background with extreme saturation cannot substantiate a whole rim.\n        diff=(at(m,plus[0])+at(m,plus[1]))*.5\n        for radius,weight in zip(minus,ww):diff-=weight*at(m,radius)\n        med=np.median(diff,axis=1);coherent=np.mean(np.clip(diff,-.65,1.25),axis=1)\n        colrim=at(col,sum(plus)/2);rim=colrim.mean(1);score=.22*med+.28*coherent+.5*(ss[sh]-strength[sh]*rim)-.035*posecost(p)\n        cf=coords(p,*facecoords[sh]);coreR,coreB,coreY,faceG=[interp(a,*cf)[:,0] for a in (R,B,Y,green)];face=faceG.mean(1);shade=np.sqrt(((faceG-face[:,None])**2).mean(1));rimG=at(green,sum(plus)/2).mean(1)\n        if sh<=2:\n            # White is usually brighter than red in green light. Use the\n            # upper half of the face to avoid counting black printing as white.\n            white=np.percentile(faceG,69,axis=1);contrast=white-rimG\n            chrom=(np.median(colrim,axis=1)-np.median(coreR,axis=1))\n            score+=.30*np.clip(chrom,-.9,1.1)+.22*np.clip(contrast,-.85,1.25)+.10*np.clip(rim,-.65,1)\n            if sh==2:score+=.14-.47*np.clip(shade/(.08+np.maximum(.06,contrast)),0,1.5)\n        elif sh in (3,4):\n            ct=coreB if sh==3 else coreR; body=np.percentile(ct,64,axis=1);bad=np.percentile(coreR if sh==3 else coreB,64,axis=1)\n            score+=.44*np.clip(body,-.65,1.15)+.15*np.clip(rim,-.65,1.1)-.23*np.clip(bad,0,1)\n        elif sh==5:\n            # A priority sign specifically has a white band outside its yellow\n            # centre. Uniform yellow light on a circle cannot supply this edge.\n            inner=np.median(coreY,axis=1);outerY=at(Y,.89);delta=inner-np.median(outerY,axis=1)\n            lumdiff=np.median(at(light,.88),axis=1)-np.median(interp(light,*cf)[:,0],axis=1)\n            score+=.36*np.clip(delta,-1,1.2)+.20*np.clip(inner,-.7,1)+.27*np.clip(lumdiff,-1,1)-.17*np.clip(shade,0,1)\n        else:\n            # Cancellation signs are permitted a colour cast, but lack a\n            # chromatic rim relative to the face and have no vivid colours.\n            score=.86*ss[sh];sat=np.maximum(np.mean(abs(at(R,.90)-np.median(coreR,axis=1)[:,None]),axis=1),np.mean(abs(at(B,.90)-np.median(coreB,axis=1)[:,None]),axis=1))\n            score-=.37*np.minimum(sat,1.2)+.12*np.minimum(np.maximum(np.abs(coreR).mean(1),np.abs(coreB).mean(1)),1.)\n        out.append(score)\n    return np.stack(out,1).astype('f')\n\ndef batch(x):\n    ms,feat=maps(x);pp,ss=localize(ms);n=len(x);sf=structural(ms,feat,pp,ss)\n    # A broad uncoloured disc does not in itself identify a cancellation:\n    # require matching the diagonal stripe. Use presence of colour rather\n    # than assuming a purely achromatic illuminant for those signs.\n    sf[:,6]-=.06;sf[:,4]-=.04\n    groups=(0,1,3,4,6);out=np.full((n,43),-12,dtype=np.float32)\n    out[:,12]=sf[:,5];out[:,13]=sf[:,2]\n    for gr,sh in enumerate(groups):\n        use=sf[:,sh]>=sf.max(1)-1.35\n        if not np.any(use):continue\n        values=match(feat,pp[sh],gr)[use];mult=(1.03,.89,1.10,.98,1.26)[gr];adj=(.47,.44,.47,.46,.44)[gr];scores=sf[use,sh,None]+mult*(values-adj)\n        out[np.ix_(use,np.array(classes[gr]))]=scores\n    return out.argmax(1)\n\ndef predict(x):\n    labels=np.empty(len(x),dtype=np.int64)\n    # A fixed working width also makes the BLAS arithmetic independent of\n    # the caller's batch size. The final partial block is padded, not mixed.\n    for i in range(0,len(x),16):\n        a=x[i:i+16];n=len(a)\n        if n!=16:a=np.concatenate((a,np.broadcast_to(a[-1:],(16-n,3,32,32))))\n        labels[i:i+n]=batch(a)[:n]\n    return labels\n"

destination=sys.argv[1]
os.makedirs(destination,exist_ok=True)
with open(os.path.join(destination,'solution.py'),'w') as f:f.write(solution)
np.savez(os.path.join(destination,'constants.npz'),**extra_stencils(mk_stencils()))
