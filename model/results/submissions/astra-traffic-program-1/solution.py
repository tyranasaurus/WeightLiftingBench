import numpy as np

bank = CONSTANTS
outer = [bank['outer'+str(i)] for i in range(8)]
land = [bank['land'+str(i)] for i in range(8)]
grid = [bank['grid'+str(i)] for i in range(8)]
templates = {}
for family in (0,1,4,5,6,7):
    templates[family]=(bank['tpl'+str(family)].astype(np.float32)*bank['amp'+str(family)][:,None]).T.copy()

parts=[bank['parts'+str(i)].reshape(-1,len(bank['partidx'+str(i)])).T.astype(np.float32).copy()/120 for i in range(5)]
wholefilters=[(bank['whole'+str(i)].astype(np.float32)*bank['wholeamp'+str(i)][:,None]).T.copy() for i in range(8)]
edgefilters={}
for family in (0,1,5):edgefilters[family]=(bank['edtpl'+str(family)].astype(np.float32)*bank['edamp'+str(family)][:,None]).T.copy()
ocrbank=(bank['ocr'].astype(np.float32)*bank['ocramp'][:,None]).T.copy()

# All matrix products use the same row blocking, including for singleton
# calls. This fixes the arithmetic path independently of the caller's batch.
def dots(a,b):
    n=len(a);z=np.zeros((32,a.shape[1]),np.float32);z[:n]=a
    return (z@b)[:n]

def digitread(im,rx,ry,ox,oy,sh):
    x,y=bank['ocrgrid']; pix=interp(im,15.5+ox+rx*(x+sh*y),15.5+oy+ry*y)
    so=np.sort(pix,axis=1);lo=so[:,int(.045*len(x)),None];hi=so[:,int(.88*len(x)),None]
    z=unit(-np.clip((pix-lo)/(hi-lo+.004),-.3,1.25));co=dots(z,ocrbank)
    n=len(z);pos=bank['ocrpos'];step=int(pos[1]);co=co.reshape(n,8,step);k=co.argmax(axis=2);j=k+pos
    ref=ocrbank.T[j];p=bank['ocrpose'][j];px=(x[None,None,:]-p[:,:,1,None]-p[:,:,2,None]*y)/p[:,:,0,None]
    z=z[:,None,:];m=px<p[:,:,3,None]
    def region(mask):
        count=mask.sum(axis=2,keepdims=True);z1=(z-(mask*z).sum(axis=2,keepdims=True)/(count+1e-6))*mask
        r1=(ref-(mask*ref).sum(axis=2,keepdims=True)/(count+1e-6))*mask
        return (z1*r1).sum(axis=2)/(np.sqrt((z1*z1).sum(2)*(r1*r1).sum(2))+.000001)
    first=region(m);lead=region(px<p[:,:,4,None]);last=region(~m)
    first[:,6:]=.68*first[:,6:]+.32*lead[:,6:]
    score=.48*co.max(axis=2)+.40*first+.12*last
    # Test the putative leading 1 relative to the intensity just outside the
    # string. With normalized correlation alone, a weak vertical noise edge
    # or a clearly printed narrow 1 can both have unit agreement.
    for jn in (6,7):
        region1=px[:,jn]<p[:,jn,4,None];rs=ref[:,jn];zs=z[:,0];pk=np.max(np.where(region1,rs,-10),axis=1)
        ink=region1&(rs>pk[:,None]*.40);paper=region1&(rs<0)
        dif=(zs*ink).sum(1)/(ink.sum(1)+.01)-(zs*paper).sum(1)/(paper.sum(1)+.01)
        score[:,jn]+=.095*np.clip(dif*np.sqrt(len(x))-.68,-1,1.25)
    return score

# Sample a batch of images at floating-point coordinates, independently.
def interp(a,x,y):
    x=np.clip(x,0,30.9999); y=np.clip(y,0,30.9999)
    ix=x.astype(np.int32);iy=y.astype(np.int32);fx=(x-ix).astype(np.float32);fy=(y-iy).astype(np.float32)
    b=np.arange(len(a)).reshape((len(a),)+(1,)*(ix.ndim-1)); ix=iy*32+ix
    return (a[b,ix]*(1-fx)+a[b,ix+1]*fx)*(1-fy)+(a[b,ix+32]*(1-fx)+a[b,ix+33]*fx)*fy

def flat(a):return a.reshape(len(a),-1)
def pool(a):return a.reshape(-1,16,2,16,2).mean(axis=(2,4)).reshape(-1,256)
def unit(a):
    a=a-a.mean(axis=1,keepdims=True)
    return a/(np.sqrt((a*a).sum(axis=1,keepdims=True))+.00001)

def spots(a,xx,yy):return interp(a,xx,yy).mean(axis=-1)

def familyfields(f,r,b,y,lum):
    if f==3:return y,lum,y
    if f==4:return b,lum,b
    if f==5:return np.maximum(np.maximum(r,b),y),lum,np.maximum(r,b)
    return r,lum,r

def poseval(f,h,l):
    # registered region order: band, field, immediate exterior.
    a,b,c=h[:,:,0],h[:,:,1],h[:,:,2];u,v,w=l[:,:,0],l[:,:,1],l[:,:,2]
    if f in (0,1,2):return 1.43*a-.60*b-.43*c+.91*(v-u)
    if f==3:return 1.32*b-.66*a-.25*c+.64*(u-.60*v-.40*w)
    if f==4:return 1.65*a+.04*b-.85*c+.12*(v-u)
    if f==5:return -.66*b+.93*(v-u)+.40*(v-w)
    return 1.57*a-.53*b-.48*c+.78*(v-u)

steps=np.array([(ds,dt,dx,dy,sh) for ds,dt in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)) for dx,dy in ((0,0),(-.85,0),(.85,0),(0,-.85),(0,.85)) for sh in (0,)],np.float32)
stepsfine=np.array([(0,0,dx,dy,sh) for dx in (-.48,.0,.48) for dy in (-.48,.0,.48) for sh in (-.12,0,.12)],np.float32)

def refine(f,pose,h,l,delta):
    opts=pose[:,None,:]+delta[None,:,:];g=land[f];rx,ry,ox,oy,sh=[opts[:,:,k,None,None] for k in range(5)]
    x=g[None,None,:,0,:]; y=g[None,None,:,1,:];xx=15.5+ox+rx*(x+sh*y);yy=15.5+oy+ry*y
    v=poseval(f,spots(h,xx,yy),spots(l,xx,yy));v-=.007*(abs(opts[:,:,2])+abs(opts[:,:,3]))+.10*abs(opts[:,:,4])
    best=v.argmax(axis=1);rows=np.arange(len(v));return opts[rows,best],v[rows,best]

def mainchunk(a):
    n=len(a); r,g,b=a[:,0],a[:,1],a[:,2]
    # Stretch luminance for this image only. Red, yellow and blue measures
    # combine a relative channel difference with an absolute saturation floor.
    gray=.24*r+.52*g+.24*b
    t=np.sort(gray.reshape(n,-1),axis=1);lo=t[:,65,None,None];hi=t[:,940,None,None];span=np.maximum(hi-lo,.047)
    lum=np.clip((gray-lo)/span,-.22,1.28)
    def color(p,q):return np.clip((p-q)/(.025+.53*p+.47*q),-.14,1.15)
    red=color(r,.75*g+.25*b);blue=color(b,.70*r+.30*g);yellow=color(.58*r+.42*g,b)-.46*np.maximum(red,0)
    # Use both global and sign-centered exposure: a bright sky outside a
    # deeply shadowed face should not flatten all of the sign's contrast.
    mid=np.sort(gray[:,5:27,5:27].reshape(n,-1),axis=1);floor=mid[:,15,None,None];ceil=mid[:,-20,None,None]
    lum=.4*lum+.6*np.clip((gray-floor)/np.maximum(ceil-floor,.034),-.25,1.25)
    l=flat(lum); R=flat(red); B=flat(blue); Y=flat(yellow); po=bank['poses'];val=[]; poses=[]
    for f in range(8):
        h,light,ch=familyfields(f,red,blue,yellow,lum)
        feat=np.concatenate([pool(h),pool(light),pool(ch)],axis=1)
        s=dots(feat,outer[f].T)-.008*(abs(po[:,2])+abs(po[:,3]))
        j=s.argmax(axis=1); val.append(s[np.arange(n),j]);poses.append(po[j])
    coarse=np.stack(val,axis=1)
    # A neutral circular end sign needs a diagonal, not merely neutral color.
    # Compare its evidence to chromatic faces, keeping multiple hypotheses.
    result=np.full((n,43),-20,np.float32)
    for f in range(8):
        rows=np.flatnonzero(coarse[:,f]>coarse.max(axis=1)-.63)
        if not len(rows):continue
        h=familyfields(f,R,B,Y,l)[0][rows];light=l[rows];p=poses[f][rows]
        p,s=refine(f,p,h,light,steps);p,s=refine(f,p,h,light,stepsfine)
        if f in (0,1,2,3):
            # Test agreement around the entire perimeter. A ring and a
            # triangle can both hit two strong red patches; the third side
            # and the clear field just inside the boundary distinguish them.
            gx,gy=land[f][0];cy=1/3 if f==1 else -1/3 if f==2 else 0
            levels=[];chs=[]
            for scale in (1,.68):
                xx=gx*scale;yy=cy+(gy-cy)*scale
                ax=15.5+p[:,2,None]+p[:,0,None]*(xx+p[:,4,None]*yy);ay=15.5+p[:,3,None]+p[:,1,None]*yy
                levels.append(interp(light,ax,ay));chs.append(interp(h,ax,ay))
            v=(chs[0]-chs[1])+.72*(levels[1]-levels[0]);v=v if f!=3 else -v
            penalty=np.sort(v,axis=1)[:,int(.25*v.shape[1])];s+=.29*np.clip(penalty,-.75,.8)*np.clip(chs[0].mean(1)*2 if f!=3 else chs[1].mean(1)*2,0,1)
        # Re-evaluate polarity and graphics using only central face samples.
        x,y=grid[f];rx,ry,ox,oy,sh=[p[:,k,None] for k in range(5)];xx=15.5+ox+rx*(x+sh*y); yy=15.5+oy+ry*y
        # Independent whole-face check, using the entire field and its trim.
        fx,fy=bank['facegrid'+str(f)];ax=15.5+ox+rx*(fx+sh*fy);ay=15.5+oy+ry*fy
        green=interp(flat(.08*r+.78*g+.14*b)[rows],ax,ay);gs=np.sort(green,axis=1)
        green=unit(np.clip(green,gs[:,int(.035*len(fx)),None]-.02,gs[:,int(.96*len(fx)),None]+.02))
        rc=interp(R[rows],ax,ay);bc=interp((-Y if f==3 else B)[rows],ax,ay)
        ra=np.clip(rc.std(1,keepdims=True)*6.5,0,1);ba=np.clip(bc.std(1,keepdims=True)*6.5,0,1)
        if f==3 or f==4:ra=ra*0
        if f!=3 and f!=4:ba=ba*0
        whole=dots(np.concatenate((green*.73,unit(rc)*ra*.27,unit(bc)*ba*.27),axis=1),wholefilters[f]);whole=np.maximum.reduceat(whole,bank['wholestart'+str(f)],axis=1)
        if f==3:
            # Yellow is defined by both red and green exceeding blue. On a
            # dark yellow diamond, green alone may hardly differ from the
            # border; the blue channel supplies the complementary edge.
            shape=abs(fx)+abs(fy);inside=shape<.58;trim=(shape>.75)&(shape<.92)
            bc0=interp(flat(b)[rows],ax,ay);patch=bc0[:,trim].mean(1)-bc0[:,inside].mean(1)
            facecol=interp(Y[rows],ax,ay);diff=facecol[:,inside].mean(1)-facecol[:,trim].mean(1)
            s+=.80*np.clip(diff,0,.8)+.45*np.clip(patch/(.035+bc0.mean(1)),0,.65)
        if f in (2,3):
            # Yield/priority have an empty field; reward uniformity only if
            # present rather than giving every inverted triangle a fixed bonus.
            inside=(np.maximum(1.5*(-y-1/3),.75*(2*abs(x)+y+1/3))<.58) if f==2 else (abs(x)+abs(y)<.56)
            face=interp(light,xx[:,inside],yy[:,inside]);ss=np.sort(face,axis=1)
            scatter=ss[:,int(len(ss[0])*.91)]-ss[:,int(len(ss[0])*.08)]
            result[rows,13 if f==2 else 12]=.83*s+.18-.31*np.minimum(scatter,1)+.96*(whole[:,0]-.39);continue
        # Green emphasizes the black and red ink against white; on blue signs
        # the red channel makes the white arrow stand out from the blue field.
        if f==4 or f in (6,7):
            im=flat((.67*r+.27*g+.06*b) if f==4 else (.11*r+.52*g+.37*b))[rows];sign=1
        else:im=flat(.25*r+.53*g+.22*b)[rows];sign=-1
        pix=interp(im,xx,yy); sort=np.sort(pix,axis=1);low=sort[:,int(.055*len(x))];high=sort[:,int(.91*len(x))];rng=np.maximum(high-low,.012)
        pic=np.clip((pix-low[:,None])/rng[:,None],-.22,1.18)
        z=unit(sign*pic); corr=dots(z,templates[f])
        if f in (0,1,5):
            edge=bank['edge'+str(f)];d=z[:,edge[0]]-z[:,edge[1]];d/=np.sqrt((d*d).sum(1,keepdims=True))+.00001
            ec=dots(d,edgefilters[f]);corr=.74*corr+.26*ec
        if f==1:
            # Approximate optical blur from the inner transition of the red
            # border, which should have the same optics as the graphic. A
            # washed-out image may lose fine legs but a sharp one should not.
            gx,gy=land[1][0];fy=1/3;rad=np.maximum((gy-fy)*1.5,(2*abs(gx)-gy+fy)*.75)
            cuts=[]
            for level in (.54,.60,.66,.72,.78,.84,.90):
                tx=gx*level/rad;ty=fy+(gy-fy)*level/rad
                ax=15.5+ox+rx*(tx+sh*ty);ay=15.5+oy+ry*ty
                cuts.append((interp(h,ax,ay)-.7*interp(light,ax,ay)).mean(1))
            cuts=np.stack(cuts,axis=1);am=cuts.max(1)-cuts.min(1);steep=np.diff(cuts,axis=1).max(1)
            width=np.clip(.025*(am+.025)/(steep+.025),.030,.11)
            corr-=.075*np.minimum(abs(np.log(bank['blur1'][None,:]/width[:,None])),1.1)
        if f in (0,1):
            col=interp(R[rows],xx,yy);colz=unit(col);ids=bank['coloridx'+str(f)]
            colscore=dots(colz,bank['colortpl'+str(f)].T)
            strength=np.clip((col.max(1)-col.min(1)-.13)*1.1,0,1)
            if f==0:corr[:,ids]+=.29*strength[:,None]*np.maximum(colscore,0)
            else:
                # Traffic lights should have a colored stack, not just a dark
                # vertical stroke. Compare the actual channel opponency at
                # the predicted red aspect with the neutral inner field.
                centered=col-col.mean(axis=1,keepdims=True)
                redsignal=np.maximum(0,dots(centered,bank['colortpl1'].T)/3)
                corr[:,ids]+=.32*np.minimum(redsignal,.8)-.045
                # Red/yellow/green ordered on a white field, separate from
                # the red rim. Measure chroma relative to nearby white.
                signals=[]
                for tx,ty in ((0,-.18),(0,.09),(0,.37)):
                    responses=[]
                    for dx in (-.055,0,.055):
                        ax=15.5+ox+rx*(tx+dx+sh*ty);ay=15.5+oy+ry*ty
                        left=ax-rx*.18;right=ax+rx*.18
                        if ty<0:ch=flat(np.maximum(0,np.minimum(r-g,r-b)/(.03+gray)))[rows]
                        elif ty<.2:ch=flat(np.maximum(0,np.minimum(r-b,g-b)/(.03+gray)))[rows]
                        else:ch=flat(np.maximum(0,np.minimum(g-r,g-b)/(.03+gray)))[rows]
                        responses.append(interp(ch,ax,ay)-.5*(interp(ch,left,ay)+interp(ch,right,ay)))
                    signals.append(np.maximum.reduce(responses)[:,0])
                aspects=np.minimum(np.maximum(signals,0),.7)
                lights=np.sort(aspects,axis=0);corr[:,ids]+=(.22*lights[2]+.82*lights[1]+.38*lights[0])[:,None]
        ss=np.maximum.reduceat(corr,bank['start'+str(f)],axis=1)
        if f==0:
            ss[:,:8]=.35*ss[:,:8]+.65*digitread(im,rx,ry,ox,oy,sh)
            # A blank interior must lack both contrast and coherent strokes.
            # Test relative to the field's OWN brightness, not background sky.
            rel=(high-low)/(high+.02);ss[:,10]=np.clip(.91-1.45*rel,-.3,.93)
            line=np.maximum(ss[:,:8].max(1),ss[:,11]);ss[:,10]-=.90*np.maximum(line-.36,0)
            # A coherent numeral in a dark photo can have very little absolute
            # contrast. In that situation a blank-field hypothesis should not
            # defeat the independently recognized digit string.
            med=(pix[:,abs(x)<.52]).mean(axis=1)
            localcontrast=(med-low)/(med+.005)
            ss[:,10]-=.72*np.maximum(line-.25,0)+.42*np.maximum(localcontrast-.12,0)
        if f==5:
            # Suppress the dominant common diagonal and explicitly inspect
            # the two remaining crescents for a speed or passing symbol.
            use=abs(x+y)>.35;zz=unit(-pic[:,use]);tt=templates[5][use].T.copy();tt=unit(tt).T.copy()
            cres=np.maximum.reduceat(dots(zz,tt),bank['start5'],axis=1)
            wing=pix[:,use];spread=np.sort(wing,axis=1);contrast=(spread[:,int(.84*len(spread[0]))]-spread[:,int(.12*len(spread[0]))])/(rng+.015)
            ss[:,[0,2,3]]+=.27*(cres[:,[0,2,3]]-.45)-.28*np.maximum(.45-contrast[:,None],0)
            ss[:,1]+=.38*np.maximum(.50-contrast,0)
        if f==1:
            pos=bank['start1'];idx=[]
            for j in range(len(pos)):
                end=pos[j+1] if j+1<len(pos) else corr.shape[1]
                idx.append(pos[j]+corr[:,pos[j]:end].argmax(1))
            ref=templates[1].T[np.stack(idx,axis=1)]
            # The feet of a pedestrian, bicycle wheels and the bent animal's
            # legs are small relative to the mass of ink at the torso. Compare
            # the lower half and side parts on their own contrast scales.
            for part in (y>.275,y<.18,x<-.01,x>.01):
                target=unit(-pic[:,part]);refpart=ref[:,:,part];refpart=refpart-refpart.mean(2,keepdims=True)
                mag=np.sqrt((refpart*refpart).sum(2));match=(refpart*target[:,None,:]).sum(2)/(mag+.00001)
                ss+=.13*(match-.40)*np.clip((mag-.06)*4,0,1)
            # Part displacements are coupled through a common face pose.
            pc=0
            for k in range(5):
                pp=unit(-pic[:,bank['partidx'+str(k)]]);part=dots(pp,parts[k]).reshape(len(rows),-1,5)
                pscore=part[:,:,0] if k==0 else .25*part[:,:,0]+.75*part.max(axis=2)
                pc+=pscore*(.40 if k==0 else .15)
            pc=pc.reshape(len(rows),-1,20).max(axis=2);pc=np.maximum.reduceat(pc,bank['partpos'],axis=1)
            ss+=.62*(pc-.43)

        # Alignment of the inner symbol as well as the rim should contribute to
        # the face decision. Correlations also resolve lookalike silhouettes.
        weight=.57 if f in (0,1,5) else .69
        if f==5:s-=.01
        if f in (6,7):
            # A red ring is also red above and below its center. Check that
            # the field behind a stop legend or the crossbar is filled in.
            inner=(abs(y)>.30)&(abs(y)<.65)&(abs(x)<.35)
            cap=interp(h,xx[:,inner],yy[:,inner]).mean(1)
            sides=(abs(y)<.14)&(abs(x)>.60)
            rim=interp(h,xx[:,sides],yy[:,sides]).mean(1)
            s-=.7*np.maximum(rim-cap,0)
        scores=.83*s[:,None]+weight*(ss-.43)+.96*(whole-.39)
        if f==5:scores+=.15 # its neutral band supplies no chromatic correlation
        result[rows[:,None],bank['class'+str(f)][None,:]]=scores
    return result.argmax(axis=1)

def predict(x):
    out=np.empty(len(x),np.int32)
    for i in range(0,len(x),32):out[i:i+32]=mainchunk(x[i:i+32])
    return out
