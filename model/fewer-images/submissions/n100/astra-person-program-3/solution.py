import numpy as np
import math

def unit(a):
    return np.clip(a,0.,1.)

def box(a,h,w):
    nrows,ncols=a.shape[-2:]
    return a.reshape(a.shape[:-2]+(nrows//h,h,ncols//w,w)).mean(axis=(-1,-3))

def smoother(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    b=p[:,:-2]+2*p[:,1:-1]+p[:,2:]
    return (b[:,:,:-2]+2*b[:,:,1:-1]+b[:,:,2:])*.0625

def skinmap(r,g,b):
    # Normalized red-yellow opponent coordinates, with a floor for dark pixels.
    light=(r+2*g+b)*.25
    rg=(r-g)/(light+.075); gb=(g-b)/(light+.075)
    slope=gb-.50*rg
    # Smooth hue boundaries. Admit pallor and shadow, but prefer distinct warmth.
    a=unit((rg+.014)/.143)*unit((.89-rg)/.31)
    u=unit((slope+.24)/.153)*unit((.39-slope)/.22)
    v=unit((light-.038)/.115)*unit((.994-light)/.075)
    # Model haemoglobin's extra red component. Wood and sand commonly have
    # similar red and yellow excursions; give such pixels lower specificity.
    specificity=unit((.18+.83*rg-gb)/.24)*unit((rg+.55*gb-.042)/.19)
    return (a*u*v*(.45+.55*specificity)).astype(np.float32)

def integral(a):
    return np.pad(a.cumsum(axis=-1,dtype=np.float64).cumsum(axis=-2),((0,0),(0,0),(1,0),(1,0)))

def pointrect(it,chan,ys,xs,y1,y2,x1,x2):
    hh=it.shape[2]-1; ww=it.shape[3]-1
    a=np.clip(ys+y1,0,hh-1); b=np.clip(ys+y2,1,hh)
    c=np.clip(xs+x1,0,ww-1); d=np.clip(xs+x2,1,ww)
    t=it[:,chan]
    return ((t[:,b[:,None],d]-t[:,a[:,None],d]-t[:,b[:,None],c]+t[:,a[:,None],c])/((b-a)[:,None]*(d-c))).astype(np.float32)

def face_relief(lum,ys,xs,h,w):
    u=CONSTANTS['u'];v=CONSTANTS['v'];mask=CONSTANTS['oval']
    sy=ys[:,:,None,None]+v*h;sx=xs[:,:,None,None]+u*w
    sy=np.clip(sy,0,94.999);sx=np.clip(sx,0,94.999)
    iy=sy.astype(np.int32);ix=sx.astype(np.int32)
    fy=(sy-iy).astype(np.float32);fx=(sx-ix).astype(np.float32)
    i=np.arange(len(lum))[:,None,None,None]
    a=(1-fx)*((1-fy)*lum[i,iy,ix]+fy*lum[i,iy+1,ix])+fx*((1-fy)*lum[i,iy,ix+1]+fy*lum[i,iy+1,ix+1])
    a=a-(a*mask).sum((2,3),keepdims=True)/mask.sum()
    a=a-.94*((a*(mask*u)).sum((2,3),keepdims=True)/(mask*u*u).sum()*u+(a*(mask*v)).sum((2,3),keepdims=True)/(mask*v*v).sum()*v)
    a=a-.4*(a*mask).sum(3,keepdims=True)/(mask.sum(1,keepdims=True)+.001)
    a=(a*np.sqrt(mask)).reshape(len(lum),ys.shape[1],-1)
    v=(a@CONSTANTS['relief']).max(axis=2)
    return v/np.sqrt((a*a).sum(2)+.0007*mask.sum())

def face_search(skin,lum,red,green,blue):
    # Window origin is centered on the facial oval, not the whole head.
    # Measurements are on integral images so large and small faces cost the same.
    n=len(skin)
    sq=lum*lum
    it=integral(np.stack((skin,lum,sq,red,green,blue),axis=1))
    best=np.full(n,-3.,np.float32)
    positive=np.zeros_like(skin)
    diffuse=np.zeros(n,np.float32)
    cutout=np.zeros(n,np.float32)
    heat=np.zeros((n,6,6),np.float32)
    dressed=np.zeros(n,np.float32)
    for height in [3,5,8,12,18,27,40,60]:
        for ratio in ([.72,1.18] if height>=5 else [.80]):
            width=height*ratio
            step=max(1,int(height/6))
            ys=np.arange(int(height*.23),96-int(height*.22),step)
            xs=np.arange(int(width*.20),96-int(width*.19),step)
            def rect(c,yt,yb,xl,xr):
                y1=int(round(yt*height));y2=max(y1+1,int(round(yb*height)))
                x1=int(round(xl*width));x2=max(x1+1,int(round(xr*width)))
                return pointrect(it,c,ys,xs,y1,y2,x1,x2)
            def oval(c):
                return .7*rect(c,-.42,.40,-.36,.36)+.3*rect(c,-.26,.25,-.47,.47)
            sk=oval(0)
            top=rect(0,-.81,-.53,-.35,.35)
            left=rect(0,-.3,.31,-.86,-.53)
            right=rect(0,-.3,.31,.53,.86)
            down=rect(0,.55,.93,-.45,.45)
            # Require localization, not just skin-coloured material everywhere.
            surround=.24*top+.19*(left+right)+.21*down+.17*np.minimum(left,right)
            local=sk-surround
            facecol=sk*.46+local*.54
            # Modeling face structure: dark hair, dark eyes below the forehead,
            # lighter cheekbones either side of the nose, and a darker mouth.
            inner=oval(1)
            variance=np.maximum(0.,oval(2)-inner*inner)
            sd=np.sqrt(variance+.0010)
            above=rect(1,-.73,-.46,-.33,.33)
            hair=unit((inner-above-.019)/(.062+.65*sd))
            eyes=rect(1,-.23,-.042,-.35,.35)
            cheeks=rect(1,-.015,.17,-.33,.33)
            forehead=rect(1,-.42,-.25,-.25,.25)
            eyecontrast=unit((.69*cheeks+.31*forehead-eyes-.005)/(sd*.76+.018))
            eye2=eyecontrast
            mouth=unit((cheeks-rect(1,.20,.33,-.22,.22))/(sd*.78+.024))
            if height>=8:
                le=rect(1,-.23,-.044,-.40,-.115);re=rect(1,-.23,-.044,.115,.40)
                lc=rect(1,-.014,.17,-.39,-.10);rc=rect(1,-.014,.17,.10,.39)
                ridge=rect(1,-.19,.12,-.082,.082)
                lr=np.minimum(lc-le,rc-re)
                eye2=unit((.65*lr+.35*(ridge-np.maximum(le,re))-.008)/(.72*sd+.016))
            # A small face-like patch should sit above a broader shirt/torso.
            # Match the two shoulders in colour and compare with their backgrounds.
            sim=0.; sides=0.; shirt=[]
            for c in (3,4,5):
                shl=rect(c,.68,1.53,-.89,-.17);shr=rect(c,.68,1.53,.17,.89)
                outl=rect(c,.59,1.33,-1.76,-1.22);outr=rect(c,.59,1.33,1.22,1.76)
                shirt.append((shl+shr)*.5)
                sim=sim+abs(shl-shr)/3
                sides=sides+np.minimum(abs(shl-outl),abs(shr-outr))/3
            shoulder=unit((sides-.40*sim-.024)/.14)*unit((.29-sim)/.19)
            # A different material on BOTH sides and above the facial oval.
            # Four nearly skin-coloured boxes occupying the same wooden plane
            # should not trigger a detection merely because one is brighter.
            rl=0.;rr=0.;rt=0.;rd=0.
            for c in (3,4,5):
                mid=oval(c)
                rl=rl+(mid-rect(c,-.29,.26,-.82,-.52))**2/3
                rr=rr+(mid-rect(c,-.29,.26,.52,.82))**2/3
                rt=rt+(mid-rect(c,-.76,-.5,-.34,.34))**2/3
                rd=rd+(mid-rect(c,.60,.91,-.37,.37))**2/3
            sidecol=unit((np.sqrt(np.minimum(rl,rr))-.035)/.125)
            topcol=unit((np.sqrt(rt)-.044)/.14);lower=unit((np.sqrt(rd)-.04)/.14)
            bounded=sidecol*(.46*topcol+.26*lower+.28)
            # Recognize textureless highlights and food less readily than shaded skin.
            texture=unit((sd-.038)/.070)*unit((.30-sd)/.12)
            prior=1.-.35*np.abs((xs-47.5)/48)[None,:]-.46*np.maximum(0,(ys-58)/37)[:,None]
            # A distant head may have no discernible internal structure. Look
            # instead for a localized warm patch beside hair and above clothing.
            density=sk*unit((local+.03)/.60)
            blob=density*(.45+.18*hair+.15*bounded+.16*shoulder+.06*prior)
            cutout=np.maximum(cutout,blob.max((1,2)))
            sr,sg,sb=shirt
            pigment=np.maximum.reduce([unit((sr-np.maximum(sb,sg)-.082)/.20)*unit((sb-sg+.12)/.12),unit((sb-sg-.050)/.17)*unit((sb-sr+.03)/.15),unit((np.minimum(sr,sb)-sg-.035)/.14)])
            wear=density*(.31+.34*pigment+.16*bounded+.10*shoulder+.09*hair)*unit((.34-sim)/.25)
            dressed=np.maximum(dressed,wear.max((1,2)))
            for iy in range(6):
                yr=np.flatnonzero(ys//16==iy)
                if len(yr)==0:continue
                strip=blob[:,yr].max(1)
                for ix in range(6):
                    xr=np.flatnonzero(xs//16==ix)
                    if len(xr):heat[:,iy,ix]=np.maximum(heat[:,iy,ix],strip[:,xr].max(1))
            # Link the colour island to a plausible head/torso junction.
            # The second test suppresses the persistent vertical edges of posts.
            if height>=8:
                detail=.36*eye2+.17*eyecontrast+.16*mouth+.19*hair+.12*texture
                portrait=facecol*(.50+.27*bounded+.14*hair)+.33*detail+.13*shoulder+.045*prior
                # A high colour score cannot substitute for all facial structure.
                response=np.minimum(portrait,.67+.48*detail+.28*shoulder+.13*bounded)
                mono=(eye2*eyecontrast*unit((mouth+.25)/1.1))**.50
                response=np.maximum(response,.64*mono+.13*hair+.09*texture+.17*bounded+.15*shoulder+.12*facecol)
            else:
                response=facecol*(.49+.28*bounded+.14*hair)+.19*shoulder+.054*texture+.046*prior
                response=np.minimum(response,.49+.47*shoulder+.17*bounded+.09*hair)+.05*(height>3)
            # Record evidence at the face scale and at separate locations.
            ev=(response-(.96 if height<8 else 1.03))/.15
            if height>=8:
                flat=ev.reshape(n,-1);count=min(flat.shape[1],24)
                locs=np.argpartition(flat,-count,axis=1)[:,-count:]
                ii=np.arange(n)[:,None];ly=locs//len(xs);lx=locs%len(xs)
                ref=face_relief(lum,ys[ly],xs[lx],height,width)
                refined=.71*flat[ii,locs]+.88*np.clip((ref-.40)/.20,-1.35,2.2)
                # Distinct facial anatomy provides an independent confirmation
                # even when facial pigmentation and lighting are unusual.
                refined=np.maximum(refined, (ref-.59)/.14+.52*facecol[ii,ly,lx]+.22*bounded[ii,ly,lx])
                best=np.maximum(best,refined.max(1))
            else:
                best=np.maximum(best,ev.max(axis=(1,2)))
            loc=positive[:,ys[:,None],xs]
            positive[:,ys[:,None],xs]=np.maximum(loc,unit((ev+.72)/1.65))
            # Evidence in more than one independent head-sized region.
            strength=unit((response-(.88 if height>=8 else .85))/.24)**2
            diffuse=np.maximum(diffuse,np.minimum(1.,strength.sum((1,2))*(step/max(height,4))**2))
    region=positive.reshape(n,4,24,4,24).max(axis=(2,4)).reshape(n,16)
    peaks=np.sort(region,axis=1)
    occupied=np.sort(heat.reshape(n,-1),axis=1)
    return best+.18*diffuse+.14*peaks[:,-3],positive,cutout,occupied[:,-4:].mean(1),dressed

def gradients(r,g,b,lum):
    # Take the strongest colour-channel gradient; temper long, high contrast edges.
    def grad(im):
        q=np.pad(im,((0,0),(1,1),(1,1)),mode='edge')
        return (q[:,1:-1,2:]-q[:,1:-1,:-2])*.5,(q[:,2:,1:-1]-q[:,:-2,1:-1])*.5
    gx,gy=grad(lum);best=gx*gx+gy*gy
    for im in (r,g,b):
        dx,dy=grad(im);dx=dx*.80;dy=dy*.80
        weight=dx*dx+dy*dy;take=weight>best
        gx=np.where(take,dx,gx);gy=np.where(take,dy,gy);best=np.maximum(best,weight)
    m=np.sqrt(best);ang=np.arctan2(gy,gx)*(6/np.pi)
    amp=m/(.13+m)**.58
    ori=np.stack([amp*np.maximum(1.-abs((ang-i+3)%6-3),0) for i in range(6)],axis=1)
    return ori,m,gx,gy

def paddedcells(a,cell,how):
    n,c,h,w=a.shape
    p=(-h)%cell;q=(-w)%cell
    if p or q: a=np.pad(a,((0,0),(0,0),(0,p),(0,q)),mode='edge')
    a=a.reshape(n,c,(h+p)//cell,cell,(w+q)//cell,cell)
    if how: return a.max((3,5))
    return a.mean((3,5))

def verify_contours(rgb,cell,hh,ww,model,by,bx,bt,up,lx):
    n=len(rgb)
    image=paddedcells(rgb,cell,False).transpose(0,2,3,1)
    image=np.pad(image,((0,0),(up,hh//4),(lx,lx),(0,0)),mode='edge')
    sy=image.shape[1]-hh+1;sx=image.shape[2]-ww+1
    idx=np.arange(n)[:,None,None,None]
    patches=image[idx,by[:,:,None,None]+np.arange(hh)[None,None,:,None],bx[:,:,None,None]+np.arange(ww)]
    col=CONSTANTS[model+'col'][bt]
    diff=patches.reshape(n,-1,hh*ww,3).transpose(0,1,3,2)@col
    # Left & right boundary vectors should agree in a colour channel. A flat
    # horizontal ramp produces two opposite vectors and fails this test.
    left=diff[:,:,:,0::2];right=diff[:,:,:,1::2]
    dot=(left*right).mean(axis=2)
    contrast=np.sqrt(np.maximum(0.,dot))
    agree=dot/np.sqrt(((left**2).mean(2)+.0016)*((right**2).mean(2)+.0016))
    part=unit((contrast-.020)/.131)*unit((agree+.02)/.75)
    return .30*part[:,:,0]+.47*part[:,:,1]+.23*part[:,:,2]

def hogsearch(ori,skin,pos,rgb):
    n=len(ori); results=[]
    for model,hh,ww,scales in [('body',14,8,[2,3,4,5,7]),('bust',9,10,[3,4,6,8])]:
        wt=CONSTANTS[model];xy=CONSTANTS[model+'xy']
        best=np.full(n,-4.,np.float32)
        for cell in scales:
            a=paddedcells(ori,cell,False)
            # smooth bin cells slightly. Compress locally to reduce the influence
            # of a single strong edge on a weak, but consistent, outline.
            aa=np.pad(a,((0,0),(0,0),(1,1),(1,1)),mode='edge')
            a=.67*a+.0825*(aa[:,:,1:-1,:-2]+aa[:,:,1:-1,2:]+aa[:,:,:-2,1:-1]+aa[:,:,2:,1:-1])
            a=a/np.sqrt(.044+a.sum(1,keepdims=True))
            c=a.shape[-1]
            # Include partly out-of-frame torsos. Missing cells have no edges.
            lx=ww//4; up=1; low=hh//4
            a=np.pad(a,((0,0),(0,0),(up,low),(lx,lx)))
            sy=a.shape[2]-hh+1;sx=a.shape[3]-ww+1
            patch=np.stack([a[:,:,i:i+sy,j:j+sx].transpose(0,2,3,1) for i in range(hh) for j in range(ww)],axis=3).reshape(n,sy,sx,-1)
            patch=np.ascontiguousarray(patch)
            energy=np.einsum('nijk,nijk->nij',patch,patch)
            average=patch.reshape(n,sy,sx,hh*ww,6).mean(3)
            norm=np.sqrt(np.maximum(0.,energy-(average**2).sum(3)*(hh*ww))+.00055*hh*ww)
            match=(patch.reshape(-1,patch.shape[-1])@wt).reshape(n,sy,sx,-1)/norm[:,:,:,None]
            sm=paddedcells(skin[:,None],cell,False)[:,0]
            pm=paddedcells(pos[:,None],cell,True)[:,0]
            # Neighbourhood for small face displacement inside a quantized cell.
            hp=np.pad(pm,((0,0),(1,1),(1,1)))
            pm=np.maximum(pm,.76*np.maximum.reduce([hp[:,:-2,1:-1],hp[:,2:,1:-1],hp[:,1:-1,:-2],hp[:,1:-1,2:]]))
            ctx=np.pad(.60*sm+.4*pm,((0,0),(up,low),(lx,lx)))
            left=np.arange(sx)+ww*.5-lx
            top=np.arange(sy)+hh*.11-up
            center=unit(1-abs(left-c*.5)/(c*.54))[None,:]*unit(1-abs(top-c*.39)/(c*.70))[:,None]
            for k in range(len(xy)):
                y,x=xy[k];s=ctx[:,y:y+sy,x:x+sx]
                match[:,:,:,k]+=.063*s+.012*center
            # Rerank local hypotheses using colour evidence. Limit the expense
            # by taking the best pose per translation, then the top translations.
            winning=match.argmax(3); sim=match.max(3).reshape(n,-1)
            count=min(32,sy*sx)
            loc=np.argpartition(sim,-count,axis=1)[:,-count:]
            by=loc//sx;bx=loc%sx;bt=winning[np.arange(n)[:,None],by,bx]
            cv=verify_contours(rgb,cell,hh,ww,model,by,bx,bt,up,lx)
            hog=sim[np.arange(n)[:,None],loc]
            # Squared normalized correlation estimates the fraction of local
            # edge-pattern variance explained by this one drawn outline.
            geom=np.clip((hog*hog-.19)/.081,-2.7,3.)
            valid=np.minimum(1.85*geom+.17, (cv-.53)/.19+.43*np.clip(geom,-1.4,1.6))
            best=np.maximum(best,valid.max(1))
        results.append(best)
    return results[0],results[1]

def skin_regions(sm,rgb):
    # Segment warm chromatic islands. A bounded skin island embedded in a
    # differently coloured head, above a shirt, is useful in profile as well
    # as frontally. Pixel components are computed with parallel label unions.
    n=len(sm);size=48
    raw=box(sm,2,2);pic=box(rgb,2,2)
    ans=np.zeros(n,np.float32);mult=np.zeros(n,np.float32)
    for threshold in (.42,.69):
        mask=.67*raw+.33*smoother(raw)>threshold
        for i in range(n):
            m=mask[i];yy,xx=np.nonzero(m)
            if len(xx)<3:continue
            numbers=np.arange(size*size).reshape(size,size)
            a=[];b=[]
            for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
                x0=max(0,-dc);x1=min(size,size-dc);y1=size-dr
                good=m[:y1,x0:x1]&m[dr:,x0+dc:x1+dc]
                a.append(numbers[:y1,x0:x1][good]);b.append(numbers[dr:,x0+dc:x1+dc][good])
            a=np.concatenate(a);b=np.concatenate(b);p=np.arange(size*size)
            for j in range(16):
                pa=p[a];pb=p[b];bad=pa!=pb
                if not np.any(bad):break
                hi=np.maximum(pa[bad],pb[bad]);lo=np.minimum(pa[bad],pb[bad])
                np.minimum.at(p,hi,lo);p=p[p];p=p[p];p=p[p]
            grp,ids,area=np.unique(p[yy*size+xx],return_inverse=True,return_counts=True)
            k=len(grp);xmin=np.full(k,size);ymin=xmin.copy();xmax=np.zeros(k,np.int64);ymax=xmax.copy()
            np.minimum.at(xmin,ids,xx);np.maximum.at(xmax,ids,xx)
            np.minimum.at(ymin,ids,yy);np.maximum.at(ymax,ids,yy)
            h=ymax-ymin+1;w=xmax-xmin+1;cx=np.bincount(ids,xx)/area;cy=np.bincount(ids,yy)/area
            aspect=w/h;fill=area/(w*h)
            quality=unit((area-1.5)/4)*unit((27-np.sqrt(area))/11)*unit((aspect-.15)/.35)*unit((1.96-aspect)/.84)*unit((fill-.19)/.36)
            hue=np.bincount(ids,raw[i,yy,xx])/area
            def sample(rx,ry):
                xs=np.clip(np.rint(cx+w*rx).astype(np.int32),0,size-1)
                ys=np.clip(np.rint(cy+h*ry).astype(np.int32),0,size-1)
                return pic[i,:,ys,xs]
            inner=sample(0,0)*.40+sample(0,.17)*.30+sample(0,-.17)*.30
            hair0=sample(0,-.68);hair1=sample(-.56,-.31);hair2=sample(.56,-.31)
            hair=unit(((inner-np.minimum(hair0,.5*(hair1+hair2))).mean(1)-.019)/.145)
            outside=.5*(abs(inner-sample(-.84,0)).mean(1)+abs(inner-sample(.84,0)).mean(1))
            outline=unit((outside-.031)/.15)
            sl=(sample(-.44,1.03)+sample(-.51,1.46))*.5;sr=(sample(.44,1.03)+sample(.51,1.46))*.5
            topmatch=unit((.31-abs(sl-sr).mean(1))/.23)
            textile=unit((abs((sl+sr)*.5-inner).mean(1)-.028)/.13)
            prior=unit(1.-abs(cx-23.5)/34)*unit((48-cy)/15)
            score=quality*(.26*hue+.17*outline+.18*hair+.16*topmatch*textile+.12*prior+.11*unit((cy-1.5)/6))
            # Corresponding hands/forearms or a second face with compatible
            # chromaticity. Hue comparison tolerates shading.
            ix=np.flatnonzero(score>.36)
            if len(ix)>1:
                dx=abs(cx[ix,None]-cx[None,ix]);dy=cy[None,ix]-cy[ix,None]
                dist=np.sqrt(dx*dx+dy*dy)/np.maximum(2,np.sqrt(area[ix,None]))
                comparable=area[None,ix]/area[ix,None]
                c1=inner[ix];ch=(c1[:,0]-c1[:,1])/(.12+c1.mean(1));cg=(c1[:,1]-c1[:,2])/(.12+c1.mean(1))
                similar=unit(1-abs(ch[:,None]-ch)/.27)*unit(1-abs(cg[:,None]-cg)/.19)
                joint=unit((dist-.9)/1.0)*unit((7-dist)/3)*unit((comparable-.08)/.30)*unit((2.5-comparable)/1.5)*similar
                score[ix]+=.13*(joint*score[None,ix]).max(1)
            ans[i]=max(ans[i],score.max())
            mult[i]=max(mult[i],np.minimum(1,np.sum(unit((score-.48)/.34)**2)/2))
    return ans+.060*mult

def figure_profiles(rgb,skin):
    # Compare centre and flanking strips along each putative figure. This
    # deliberately models the RGB of the scene *at this location*: a moving
    # person can interrupt a shaded background which is not one solid colour.
    n=len(rgb);pic=box(rgb,2,2);sz=48
    data=np.concatenate((pic,pic*pic,box(skin,2,2)[:,None]),axis=1)
    it=np.pad(data.cumsum(axis=3,dtype=np.float64).cumsum(axis=2),((0,0),(0,0),(1,0),(1,0)))
    best=np.zeros(n,np.float32);cnt=np.zeros(n,np.float32)
    for h in (12,18,26,38,54):
        step=max(1,h//12)
        ys=np.arange(-int(.1*h),sz-int(.55*h),step);xs=np.arange(1,sz-1,step)
        for asp in (.77,1.30):
            w=h*.23*asp
            def rect(yt,yb,xl,xr):
                ya=int(round(yt*h));yb=max(ya+1,int(round(yb*h)))
                xa=int(round(xl*w));xb=max(xa+1,int(round(xr*w)))
                y1=np.clip(ys+ya,0,sz-1);y2=np.clip(ys+yb,1,sz)
                x1=np.clip(xs+xa,0,sz-1);x2=np.clip(xs+xb,1,sz)
                t=it[:,:,y2[:,None],x2]-it[:,:,y1[:,None],x2]-it[:,:,y2[:,None],x1]+it[:,:,y1[:,None],x1]
                return (t/((y2-y1)[:,None]*(x2-x1))).astype(np.float32)
            def part(a,b,rad,out):
                m=rect(a,b,-rad,rad);l=rect(a,b,-out-.50,-out);r=rect(a,b,out,out+.50)
                noise=.5*(l[:,3:6]+r[:,3:6]-l[:,:3]**2-r[:,:3]**2).mean(1)
                dl=m[:,:3]-l[:,:3];dr=m[:,:3]-r[:,:3]
                cross=(dl*dr).mean(1);agree=cross/np.sqrt(((dl*dl).mean(1)+.0025)*((dr*dr).mean(1)+.0025))
                diff=np.sqrt(np.maximum(cross,0.))
                sn=diff/(.072+np.sqrt(np.maximum(noise,0.)))
                # Require both sides to depart from the centre in the same
                # direction, as opposed to a single step edge or linear ramp.
                q=unit((sn-.28)/1.37)*unit((agree-.05)/.78)
                return m,q
            head,hr=part(.025,.14,.16,.34)
            coat,cr=part(.21,.46,.30,.65)
            pants,pr=part(.49,.72,.24,.61)
            # The region that departs from the background should widen at the
            # shoulders and end at the top of the head. The two garments need
            # not have the same colour.
            l=rect(.20,.36,-.47,-.24);r=rect(.20,.36,.24,.47)
            la=rect(.017,.14,-.53,-.27);ra=rect(.017,.14,.27,.53)
            widen=.50*(np.sqrt(((la[:,:3]-head[:,:3])**2).mean(1))+np.sqrt(((ra[:,:3]-head[:,:3])**2).mean(1)))-.50*(abs(l[:,:3]-coat[:,:3]).mean(1)+abs(r[:,:3]-coat[:,:3]).mean(1))
            cap=rect(-.12,-.02,-.26,.26)
            closed=unit((np.sqrt(((cap[:,:3]-head[:,:3])**2).mean(1))-.021)/.16)
            closed=closed*(ys[:,None]>=int(h*.06))
            pa=unit((np.sqrt(((coat[:,:3]-pants[:,:3])**2).mean(1))-.035)/.14)
            hw=unit((widen+.015)/.158)
            # A head that instead blends into the background can be rescued by
            # its skin, a well-closed top, or a strong pair of legs.
            top=np.maximum(hr,head[:,6]*.91)
            center=unit(1-abs(xs-23.5)/34)[None,:]*unit(1-abs(ys-h*.05-16.)/40.)[:,None]
            evidence=np.sqrt(top*cr)*(.18+.21*closed+.19*hw+.10*pa+.13*head[:,6]+.135*pr+.055*center)
            evidence=np.maximum(evidence,np.sqrt(cr*pr)*(.20+.12*top+.17*hw+.10*closed+.15*pa+.15*head[:,6]+.05*center))
            # Verify the best strip proposals by explaining each with a drawn
            # whole silhouette. Estimate a row-specific background from BOTH
            # flanks. A solid post also has a silhouette, so subtract the match
            # to a uniform-width rectangle as a competing explanation.
            flat=evidence.reshape(n,-1);take=min(20,flat.shape[1]);ii=np.arange(n)[:,None]
            at=np.argpartition(flat,-take,axis=1)[:,-take:];ya=ys[at//len(xs)];xa=xs[at%len(xs)]
            row=np.clip((ya[:,:,None,None]+((np.arange(16)+.5)/16)[None,None,:,None]*h).astype(np.int32),0,47)
            col=np.clip((xa[:,:,None,None]+((np.arange(12)+.5)/12-.5)[None,None,None,:]*h*.56).round().astype(np.int32),0,47)
            piclast=pic.transpose(0,2,3,1);win=piclast[np.arange(n)[:,None,None,None],row,col]
            lc=win[:,:,:,0:2].mean(3,keepdims=True);rc=win[:,:,:,-2:].mean(3,keepdims=True)
            d1=win-lc;d2=win-rc;d=(d1*d2).mean(4)
            seg=unit((np.sqrt(np.maximum(d,0.))-.025)/.18)
            seg=seg-seg.mean(3,keepdims=True)
            norm=np.sqrt((seg*seg).sum((2,3))+.009*192)
            corr=(seg.reshape(n,take,-1)@CONSTANTS['silhouette']).max(2)/norm
            column=seg.mean(axis=2);column=unit(np.sqrt((column*column).sum(2)*16)/norm)
            shape=unit((corr-.27)/.31)*(.42+.58*unit((corr-.80*column+.14)/.31))
            visible=unit((48.-ya)/h);shape=shape*(.48+.52*visible)
            candidate=flat[ii,at]*(.59+.41*shape)
            best=np.maximum(best,candidate.max(1))
            cnt=np.maximum(cnt,unit((candidate-.54)/.27).sum(1)*(step/h)**2)
    return best+.05*unit(cnt/.12)

def scene_cues(r,g,b,lum,gx,gy):
    # Axis alignment and contiguous collinearity at a coarser (object) scale.
    # In a room the vanishing edges may tilt, but there are often still many
    # cardinal edges. Soft gradients in the sky should not vote.
    c=smoother(box(lum,2,2));p=np.pad(c,((0,0),(1,1),(1,1)),mode='edge')
    dx=p[:,1:-1,2:]-p[:,1:-1,:-2];dy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    eng=dx*dx+dy*dy
    coherence=(dx**4-6*(dx*dy)**2+dy**4)/(eng*eng+.0000001)
    mag=np.sqrt(eng);wt=mag/(mag+.07)
    angular=(wt*coherence).mean((1,2))/(wt.mean((1,2))+.001)
    # Concentrated, varied pigmentation (team colours and clothing), as
    # opposed to a single dominant tint of sky, grass, or indoor lighting.
    def vivid(d):
        return unit((d-.065)/.20)*unit((box(d,4,4).max((1,2))[:,None,None]-.12)/.18)
    reds=vivid(r-np.maximum(g,b))*unit((b-g+.14)/.12)
    blues=vivid(b-g)*unit((b-r)/.12)
    pink=vivid(np.minimum(r,b)-g)
    green=vivid(g-np.maximum(r,b))*unit((b-r+.14)/.19)
    bright=np.stack((reds,blues,pink,green),axis=1)
    mass=bright[:,:,5:92].mean((2,3));loc=box(bright,8,8)[:,:,1:11,1:11].max((2,3))
    dyed=unit(mass/.015)*loc*unit((.47-mass)/.24)
    turf=unit((g-r+.035)/.113)*unit((g-b-.027)/.14)
    snow=unit((lum-.56)/.22)*unit((.16-abs(b-r))/.12)
    water=unit((b-r-.07)/.20)*unit((g-r+.012)/.12)
    field=np.maximum.reduce([turf[:,42:].mean((1,2)),snow[:,42:].mean((1,2)),water[:,42:].mean((1,2))])
    return angular,dyed.max(1)+.25*(dyed.sum(1)-dyed.max(1)),unit((field-.18)/.46)

def sports_context(r,g,b):
    n=len(r);lum=.26*r+.53*g+.21*b
    # Search for a compact vertical interruption of a fairly uniform playing
    # field, body of water, or snow field. Not just global scene colour.
    turf=unit((g-r+.018)/.11)*unit((g-b-.027)/.14)*unit((.54-(g-r))/.18)
    snow=unit((lum-.52)/.25)*unit((.23-abs(b-r))/.13)*unit((.10+r-g)/.07)
    water=unit((b-r-.043)/.18)*unit((g-r+.015)/.13)*unit((.20+b-g)/.16)
    maps=np.stack((turf,snow,water),axis=1)
    cm=box(maps,4,4)
    it=integral(cm)
    evidence=np.zeros(n,np.float32)
    for i in range(3):
        coverage=cm[:,i,8:,:].mean((1,2))
        for h in [4,7,11,16]:
            ys=np.arange(0,25-h);xs=np.arange(1,23)
            w=max(1,int(h*.20));side=max(2,int(h*.42))
            def rect(ya,yb,xa,xb): return pointrect(it,i,ys,xs,ya,yb,xa,xb)
            left=rect(h//4,h,-w-side,-w);right=rect(h//4,h,w,w+side)
            inside=rect(0,h,-w,w);foot=rect(h,h+3,-w-side,w+side)
            top=rect(-2,0,-w,w)
            diff=np.minimum(left,right)-inside
            score=unit((diff-.15)/.62)*unit((foot-.24)/.53)*unit((.35*np.maximum(left,right)+.65*np.minimum(left,right)-.37)/.42)
            score=score*(.69+.31*unit((top-inside+.05)/.59))
            s=score.max((1,2))*unit((coverage-.07)/.25)
            evidence=np.maximum(evidence,s*[1.,.82,.85][i])
    return evidence

def fixed_scene(gray,gx,gy):
    # Several strong orthogonal stretches, flat-sided rectangles and repeated
    # textures are cues for architecture, furniture, trains and appliances.
    # Work on a soft edge map: small JPEG block boundaries have little weight.
    n=len(gray);g=box(gray,2,2);q=np.pad(g,((0,0),(1,1),(1,1)),mode='edge')
    dx=abs(q[:,1:-1,2:]-q[:,1:-1,:-2]);dy=abs(q[:,2:,1:-1]-q[:,:-2,1:-1])
    vertical=unit((dx-1.7*dy-.025)/.20);horizontal=unit((dy-1.7*dx-.025)/.20)
    ev=paddedcells(vertical[:,None],3,True)[:,0];eh=paddedcells(horizontal[:,None],3,True)[:,0]
    lv=np.minimum.reduce([ev[:,0:-3],ev[:,1:-2],ev[:,2:-1],ev[:,3:]])
    lh=np.minimum.reduce([eh[:,:,0:-3],eh[:,:,1:-2],eh[:,:,2:-1],eh[:,:,3:]])
    vc=lv.mean((1,2));hc=lh.mean((1,2))
    axis=np.sqrt(vc*hc)+.27*np.maximum(vc,hc)
    # Repeated edges at several precisely related displacements (shelving,
    # rows of windows). Use same-axis correlation above the local baseline.
    repeated=np.zeros(n,np.float32)
    for m in (vertical,horizontal.transpose(0,2,1)):
        m=box(m,2,2);mean=m.mean((1,2),keepdims=True)
        d=m-mean;var=(d*d).mean((1,2))+.008
        z=np.zeros(n,np.float32)
        for k in (2,3,4,5,6,7):
            a=(d[:,:,:-2*k]*d[:,:,k:-k]*np.maximum(d[:,:,2*k:],0)).mean((1,2))
            z=np.maximum(z,a/var)
        repeated+=z
    return .74*unit((axis-.063)/.16)+.26*unit((repeated-.10)/.28)

def straightness(lum,gx,gy):
    # Long straight runs at the same coordinate often belong to buildings,
    # furniture or vehicles. Measure coherent gradients, not edge count alone.
    ans=0.
    for v in (gx,gy.transpose(0,2,1)):
        pad=np.pad(v,((0,0),(0,0),(1,1)),mode='edge')
        v=(pad[:,:,:-2]+2*v+pad[:,:,2:])*.25
        seg=v.reshape(len(v),6,16,96)
        signed=abs(seg.mean(2));mag=abs(seg).mean(2)
        coherent=unit((signed/(mag+.012)-.46)/.43)*unit((mag-.014)/.105)
        ans=ans+coherent.mean((1,2))
    return ans

def evaluate(x):
    r=x[:,0];g=x[:,1];b=x[:,2]
    lum=.299*r+.587*g+.114*b
    skin=skinmap(r,g,b)
    # A partial gray-world correction for casts, bounded to avoid treating
    # dominant scene colours as an illuminant. Keep original under neutral light.
    rm=r.mean((1,2))[:,None,None];gm=g.mean((1,2))[:,None,None];bm=b.mean((1,2))[:,None,None]
    rg=np.clip((gm+.11)/(rm+.11),.79,1.22)**.54
    bg=np.clip((gm+.11)/(bm+.11),.78,1.27)**.54
    skin=.65*skin+.35*skinmap(r*rg,g,b*bg)
    face,location,blob,group,dressed=face_search(skin,lum,r,g,b)
    islands=skin_regions(skin,x)
    ori,m,gx,gy=gradients(r,g,b,lum)
    body,bust=hogsearch(ori,skin,location,x)
    sport=sports_context(r,g,b)
    lines=straightness(lum,gx,gy)
    skmean=skin.mean((1,2))
    gray=np.maximum(abs(r-g),abs(b-g)).mean((1,2))
    warm=unit((skmean-.025)/.12)*unit((.66-skmean)/.40)
    # Blend the detectors as alternative explanations, not requirements: skin
    # is allowed to be missing, and a portrait need not show any legs.
    fb=np.clip(face,-2.7,2.6)
    whole=np.clip(body,-2.6,3.)
    torso=np.clip(bust,-2.6,3.)
    shape=np.maximum(whole,.84*torso)+.22*np.minimum(whole,torso)
    # Both families search many correlated windows; an isolated best match is
    # weaker than independent cues agreeing. A very clear match can stand alone.
    # Pose-independent classification uses skin islands and scene geometry as
    # the base estimate; matched anatomy supplies additional positive evidence.
    angular,dyed,field=scene_cues(r,g,b,lum,gx,gy)
    pigmentation=.50*np.clip((islands-.76)/.17,-1.5,1.6)+.48*np.clip((blob-.58)/.24,-1.5,1.6)
    geometry=.57*np.clip((.16-angular)/.19,-1.6,1.1)-.40*np.clip((lines-.073)/.085,-.8,1.8)
    outdoors=.44*sport+.22*field+.27*sport*field
    color=.58*(dyed-.45)+.24*(warm-.43)+.22*unit((.044-gray)/.038)
    base=pigmentation+geometry+outdoors+color
    verify=.26*np.clip(fb,-.35,2.6)+.24*np.clip(shape,-.34,2.4)
    profile=figure_profiles(x,skin)
    # An interrupted background, a collection of localized face-colour patches,
    # and dyed clothing provide routes to recognition without a resolved face.
    grouping=.43*unit((group-.37)/.32)
    freeform=np.clip((profile-.58)/.19,-1.2,1.95)
    vote=.76*base+verify+grouping+.44*freeform
    vote=np.maximum(vote, .74*freeform+.55*(sport+field-.78)+.27*(dyed-.41)+.23*geometry+.1*np.clip(shape,-1,2)-.55)
    stationary=fixed_scene(lum,gx,gy)
    vote+=.48*np.clip((dressed-.46)/.26,-.66,1.65)-.44*(stationary-.34)
    return (vote>0).astype(np.int64)

def predict(x):
    out=np.empty(len(x),np.int64)
    for i in range(0,len(x),4):
        out[i:i+4]=evaluate(x[i:i+4])
    return out
