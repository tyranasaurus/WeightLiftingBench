"""Data-independent construction of analytic filters and individual shape templates."""
import sys
from pathlib import Path
import numpy as np

PARAMS = {}

def oriented_templates():
    # Coordinates in a unit box. Along each model boundary vote into nearby
    # orientation and spatial bins. No images (real or generated) are read.
    lists = [[], [], [], []]
    def line(a,b,r,s):
        a=np.array(a); b=np.array(b); d=b-a
        u=np.linspace(0,1,160)[:,None]
        side=np.array([-d[1],d[0]])/np.sqrt(d@d+1e-12)
        return [a+u*d+(r+u*(s-r))*sign*side for sign in (-1,1)]
    def oval(cx,cy,w,h,start=0,end=2*np.pi):
        t=np.linspace(start,end,192)
        return np.stack([cx+w*np.cos(t),cy+h*np.sin(t)],axis=1)
    def make(lines,grid,aspect,sx,sy,angle):
        rows,cols=grid
        hist=np.zeros((rows,cols,8))
        C,S=np.cos(angle),np.sin(angle)
        for points,weight in lines:
            xx=(points[:,0]-.5)*sx; yy=(points[:,1]-.5)*sy
            xs=.5+C*xx-S*yy/aspect; ys=.5+C*yy+S*xx*aspect
            a=(xs[:-1]+xs[1:])*.5*cols-.5; b=(ys[:-1]+ys[1:])*.5*rows-.5
            dx=np.diff(xs)*cols; dy=np.diff(ys)*rows
            # Image descriptor angles measured in the undistorted pixel grid.
            theta=np.mod(np.arctan2(dy/rows,dx/cols*aspect)+np.pi/2,np.pi)*8/np.pi
            ix=np.floor(a).astype(int); iy=np.floor(b).astype(int); it=np.floor(theta).astype(int)
            for oy in range(2):
                for ox in range(2):
                    for ot in range(2):
                        j=iy+oy; k=ix+ox; t=(it+ot)%8
                        w=(1-abs(b-j))*(1-abs(a-k))*(1-abs(theta-it-ot))*np.hypot(dx,dy)*weight
                        ok=(j>=0)&(j<rows)&(k>=0)&(k<cols)
                        np.add.at(hist,(j[ok],k[ok],t[ok]),w[ok])
        # Spread the ideal directions to allow for curvature, clothes and texture.
        hist=hist*.72+(np.roll(hist,1,2)+np.roll(hist,-1,2))*.14
        # Zero directional DC response at each point, L1 normalize.
        hist=np.minimum(hist,.65)
        hist-=hist.mean(axis=2,keepdims=True)
        return (hist/max(1e-5,np.abs(hist).sum()/2)).astype(np.float32).ravel()
    # Height units; these are poses made from limb segments, not observations.
    for full in (False,True):
        for width in (.30,.44,.59) if full else (.73,.99,1.22):
            for lean in ((-.10,0,.10) if full else (-.13,0,.13)):
                for pose in (0,1,2,3) if full else (0,1):
                    lines=[]
                    def edge(points,w=1): lines.append((np.array(points),w))
                    if full:
                        hx=.50 if pose!=3 else .60
                        edge(oval(hx,.14,.135,.086),1.05)
                        top=.24; bot=.54
                        for sign in [-1,1]:
                            edge([(hx+sign*.075,.205),(hx+sign*.090,.213),(.50+sign*.26,top),(.50+sign*.32,.280)],1.2)
                            edge([(.50+sign*.245,.30),(.50+sign*.180,.43),(.50+sign*.20,bot)],1.05)
                            arm=(.5+sign*(.365 if pose==0 else .42),.38 if pose!=2 else .30)
                            hand=(.5+sign*(.38 if pose==0 else .21),.51 if pose!=2 else .21)
                            for a,b,r,s in [((.50+sign*.27,.27),arm,.054,.048),(arm,hand,.048,.033)]:
                                for c in line(a,b,r,s): edge(c,.60)
                            knee=(.50+sign*(.14 if pose==0 else .23),.73 if pose!=3 else (.65 if sign==-1 else .75))
                            foot=(.50+sign*(.16,.33,.42,.34)[pose],.95 if pose!=3 or sign==1 else .80)
                            for a,b,r,s in [((.50+sign*.115,bot),knee,.10,.066),(knee,foot,.066,.052)]:
                                for c in line(a,b,r,s): edge(c,.86)
                        grid=(9,5); group=0
                    else:
                        edge(oval(.50,.256,.176,.198),1.05)
                        for s in [-1,1]:
                            edge([(.50+s*.103,.403),(.50+s*.108,.460),(.50+s*.245,.494),(.50+s*.345,.543),(.50+s*.407,.638),(.50+s*(.45 if pose==0 else .39),.81),(.50+s*(.48 if pose==0 else .34),.96)],1.2)
                            edge([(.50+s*.265,.66),(.50+s*.257,.76),(.50+s*.24,.95)],.48)
                        grid=(6,6); group=1
                    lists[group].append(make(lines,grid,width,1,1,lean))
    # Frontal internal features: relative grey levels in an individual cartoon.
    # Transform each into a locally zero-mean photometric filter on its own.
    q=32
    yy,xx=np.mgrid[0:8*q,0:7*q].astype(float)
    xx=(xx+.5)/(7*q)*2-1; yy=(yy+.5)/(8*q)*2-1
    for tilt in [-.21,0,.21]:
        for turn in [-.33,0,.33]:
            for hair in [.04,.24]:
                for mouth in [.045,.14]:
                    x=xx*np.cos(tilt)+yy*np.sin(tilt); y=yy*np.cos(tilt)-xx*np.sin(tilt)
                    rad=(x/(.84-.12*np.maximum(y,0)))**2+(y/.97)**2
                    mask=np.exp(-np.maximum(rad-.65,0)*3.0)
                    def ga(a,b,s,t): return np.exp(-.5*(((x-a)/s)**2+((y-b)/t)**2))
                    im=.53-.17*rad+.13*x*turn
                    im-=hair*np.clip((-y-.52)*5+np.maximum(np.abs(x)-.4,0),0,1)
                    for side in [-1,1]:
                        z=turn*.53+side*(.326-.035*abs(turn))
                        im-=.23*ga(z,-.165,.151,.060)+.15*ga(z,-.239,.201,.067)
                        im+=.09*ga(z,.085,.21,.15)
                    im+=.16*ga(turn*.73,.004,.123,.21)-.135*ga(turn*.73,.194,.157,.054)
                    im-=mouth*ga(turn*.62,.443,.235,.061)
                    im+=.05*ga(turn*.57,.661,.183,.088)
                    A=mask.reshape(8,q,7,q).mean((1,3)); V=(im*mask).reshape(8,q,7,q).mean((1,3))/A
                    K=A*(V-np.sum(A*V)/A.sum())
                    lists[2].append((K/np.sqrt(np.sum(K*K/A))).ravel()); lists[3].append(A.ravel())
    for key,items in zip(['figure','bust','face','fweight'],lists): PARAMS[key]=np.stack(items,axis=1).astype(np.float32)

SOURCE = '''import numpy as np
import math


def ramp(a,lo,hi):
    return np.clip((a-lo)/(hi-lo),0.,1.)


def table(a):
    return np.pad(a.cumsum(axis=-1).cumsum(axis=-2),((0,0),)*(a.ndim-2)+((1,0),(1,0)))


geometry={}


def rect(t,y,x,h,w,dy=0,dx=0):
    # Cache grid geometry only, never pixel values. Separable finite
    # differences avoid four two-dimensional advanced-index gathers.
    key=(y.tobytes(),x.tobytes(),h,w,dy,dx)
    spec=geometry.get(key)
    if spec is None:
        a=np.clip(y.ravel()+dy-h//2,0,96); b=np.clip(y.ravel()+dy-h//2+h,0,96)
        c=np.clip(x.ravel()+dx-w//2,0,96); d=np.clip(x.ravel()+dx-w//2+w,0,96)
        inv=1./np.maximum((b-a)[:,None]*(d-c)[None,:],1).astype(np.float32)
        spec=(a,b,c,d,inv); geometry[key]=spec
    a,b,c,d,inv=spec
    u=np.take(t,b,axis=-2)-np.take(t,a,axis=-2)
    return (np.take(u,d,axis=-1)-np.take(u,c,axis=-1))*inv


def box(t,y,x,h,w,fy,fx,cy=0.,cx=0.):
    return rect(t,y,x,max(1,round(fy*h)),max(1,round(fx*w)),round(cy*h),round(cx*w))


def skinfo(r,g,b):
    # A soft, intensity-dependent flesh-colour wedge. Red should exceed green,
    # and the green-blue excess should generally be smaller than red-green.
    a=(r-g)/(.30+.70*g); d=(g-b)/(.30+.70*g)
    m=ramp(a,.013,.112)*ramp(a,.70,.44)*ramp(d-.13*a,-.115,-.012)*ramp(d-.80*a,.126,.020)*ramp(d,.50,.27)
    return m*ramp(r,.042,.135)


def pyramid(im,step):
    n=96//step
    return im.reshape(im.shape[:-2]+(n,step,n,step)).mean(axis=(-1,-3))


def gradients(im):
    pad=np.pad(im,((0,0),(1,1),(1,1)),mode='edge')
    gx=pad[:,1:-1,2:]-pad[:,1:-1,:-2]; gy=pad[:,2:,1:-1]-pad[:,:-2,1:-1]
    # Use luminance with a small cross-channel component for low-luma contours.
    dx=.30*gx[0]+.52*gx[1]+.18*gx[2]; dy=.30*gy[0]+.52*gy[1]+.18*gy[2]
    red=abs(gx[0])+abs(gy[0])>1.7*(abs(dx)+abs(dy))+.04
    dx=np.where(red,gx[0],dx); dy=np.where(red,gy[0],dy)
    mag=np.sqrt(dx*dx+dy*dy)
    theta=np.mod(np.arctan2(dy,dx)*np.float32(8/math.pi),8.)
    f=np.arange(8,dtype=np.float32)[:,None,None]-theta
    dist=np.abs(f); hist=np.maximum(1.-np.minimum(dist,8.-dist),0)*np.minimum(mag,.40)
    return hist,mag,dx,dy


def hoggrid(t,gh,gw):
    # Spacing equal to half a descriptor cell allows overlapping windows.
    sy=max(1,gh//2); sx=max(1,gw//2)
    y=np.arange(gh//2,97-gh+gh//2,sy)[:,None]; x=np.arange(gw//2,97-gw+gw//2,sx)[None,:]
    hist=rect(t,y,x,gh,gw).transpose(1,2,0).copy()
    # In-cell contrast normalization, then subtract the local orientation bias
    # of the scene (fences and windows have repeated axial edge directions).
    p=hist/(.028+hist.sum(axis=2,keepdims=True))
    common=hist.mean(axis=(0,1)); common=common/(common.sum()+.025)
    return np.ascontiguousarray(p-.38*common)


def windows(a,rows,cols,stride=2):
    # Flatten row-major grid cells without using stride tricks.
    h=a.shape[0]-stride*(rows-1); w=a.shape[1]-stride*(cols-1)
    if h<=0 or w<=0: return np.zeros((0,rows*cols*a.shape[-1]),np.float32)
    return np.stack([a[i*stride:i*stride+h,j*stride:j*stride+w] for i in range(rows) for j in range(cols)],axis=2).reshape(h*w,-1)


def areas(t,h,w):
    sy=max(1,min(4,h//5)); sx=max(1,min(4,w//5))
    y=np.arange(0,96,sy)[:,None]; x=np.arange(0,96,sx)[None,:]
    return y,x


def skinhead(skin,tcol,lum,tskin):
    # Score a face-coloured core, its surround, the hair just above, and the
    # wider, differently coloured shoulder mass below. Also score whether two
    # cheeks and a jaw resemble one another more than their surroundings.
    best=-4.; bestlite=-4.; second=-4.; pack=0.
    hs=np.full((96,96),-4.,np.float32)
    for h in (3,5,8,12,18,27,41,60):
        for ar in (.61,.95):
            w=max(3,round(h*ar)); y,x=areas(tskin,h,w)
            core=box(tskin,y,x,h,w,.68,.68)
            left=box(tskin,y,x,h,w,.34,.31,-.025,-.255)
            right=box(tskin,y,x,h,w,.34,.31,-.025,.255)
            jaw=box(tskin,y,x,h,w,.30,.52,.30,0.)
            s=.46*core+.27*np.minimum(left,right)+.27*jaw
            a=box(tskin,y,x,h,w,.83,.36,-.04,-.72); b=box(tskin,y,x,h,w,.83,.36,-.04,.72)
            c=box(tskin,y,x,h,w,.36,.88,-.69); d=box(tskin,y,x,h,w,.42,.74,.78)
            sides=(a+b)*.5
            surround=.48*sides+.28*c+.24*d
            cc=box(tcol,y,x,h,w,.65,.62)
            hair=box(lum,y,x,h,w,.41,.75,-.64)
            skinlum=box(lum,y,x,h,w,.62,.60)
            hdiff=ramp(skinlum-hair,.025,.245)
            shirt=box(tcol,y,x,h,w,.78,1.16,1.40)
            sl=box(tcol,y,x,h,w,.70,.60,1.40,-1.35); sr=box(tcol,y,x,h,w,.70,.60,1.40,1.35)
            lg=np.sqrt(np.mean((shirt-sl)**2,axis=0)); rg=np.sqrt(np.mean((shirt-sr)**2,axis=0))
            sep=ramp(np.sqrt(np.mean((shirt-cc)**2,axis=0)),.055,.24)
            body=sep*(.45+.55*ramp(np.minimum(lg,rg),.035,.18))
            body*=((y+1.60*h)<96)
            sym=ramp(abs(left-right),.62,.05)
            central=np.exp(-((x-47.5)/47.)**2-((y-40.)/50.)**2)
            # High scores require several agreeing cues; unlike wood, a small
            # skin blob should be bounded on multiple sides and above clothing.
            sc=1.52*s+1.24*(s-surround)+.34*(s-np.maximum(a,b))+.32*(s-c)+.33*hdiff+.47*body+.12*sym+.22*central
            sc-=.24*(h<=5)+.12*(h>=41)+.22*((x-w*.56<0)|(x+w*.56>95))+.22*((y-h*.52<0)|(y+h*.52>95))
            sc-=3.50
            sc=np.where((x-.90*w>=0)&(x+.90*w<96)&(y-.89*h>=0)&(y+.57*h<96),sc,-4.)
            best=max(best,float(sc.max()))
            ys=y[:,0]; xs=x[0]; sy=(ys[1]-ys[0]); sx=(xs[1]-xs[0]); field=sc.repeat(sy,0).repeat(sx,1)[:96,:96]
            hs=np.maximum(hs,field)
    # Disjoint pools make a second person additional evidence, without counting
    # every nearby window on one person's face as an independent detection.
    py,px=np.unravel_index(hs.argmax(),hs.shape)
    yy,xx=np.ogrid[:96,:96]
    second=float(np.where(((yy-py)**2+(xx-px)**2)>20**2,hs,-4).max())
    return best,second,hs


def pairedskin(skin,bs,hs,tc,lum):
    # Independently of contours, test a little collection of articulated part
    # geometries. A face and an arm/hand typically have similar pigmentation
    # and hue, lie on opposite sides of a clothing boundary, and each stands
    # out from its own local background. Floorboards don't end in small hands.
    best=-2.; color=tc[0]-tc[2]; local=tc[0]-2*tc[1]+tc[2]
    yy,xx=np.ogrid[:96,:96]
    for h in (5,9,16,28):
        w=max(3,round(.8*h)); step=max(2,h//5)
        y=np.arange(0,96,step)[:,None]; x=np.arange(0,96,step)[None,:]
        face=box(bs,y,x,h,w,.72,.69); tshirt=box(bs,y,x,h,w,.82,.88,1.53)
        side=(box(bs,y,x,h,w,.89,.56,-.09,-.84)+box(bs,y,x,h,w,.89,.56,-.09,.84))*.5
        low=box(bs,y,x,h,w,.37,.74,.78); facehalo=face-.6*side-.4*low
        chrom=box(color,y,x,h,w,.72,.70); hue=box(local,y,x,h,w,.72,.70); v=box(lum,y,x,h,w,.72,.70)
        # Overlap of central clothing plus non-central bare arms at the same
        # height is particularly strong evidence against colour spill.
        torso=box(tc,y,x,h,w,.93,1.30,1.64); lower=box(tc,y,x,h,w,.82,1.30,2.46)
        outfit=ramp(np.sqrt(np.mean((torso-lower)**2,axis=0)),.37,.05)
        left=-2.+0*face; right=left.copy()
        for s in (-1,1):
            arm=-2.+0*face
            for fy,fx in ((1.65,1.14),(2.35,1.10),(2.80,1.52),(1.05,1.78),(.18,1.26)):
                hand=box(bs,y,x,h,w,.49,.49,fy,s*fx); ring=box(bs,y,x,h,w,1.21,1.40,fy,s*fx)
                c=box(color,y,x,h,w,.49,.49,fy,s*fx); q=box(local,y,x,h,w,.49,.49,fy,s*fx); t=box(lum,y,x,h,w,.49,.49,fy,s*fx)
                close=np.exp(-np.abs(c/(.25+t)-chrom/(.25+v))/.16-np.abs(q/(.25+t)-hue/(.25+v))/.13)
                e=.47*hand+.45*(hand-ring)+.27*close-.48
                arm=np.maximum(arm,np.where((x+s*fx*w>=0)&(x+s*fx*w<96)&(y+fy*h<96),e,-2.))
            if s==-1: left=arm
            else: right=arm
        score=.87*(face-.70)+.68*(facehalo-.54)+.45*(face-tshirt-.62)+.13*(outfit-.6)+.66*np.maximum(left,right)+.34*np.minimum(left,right)
        valid=(x-.8*w>=0)&(x+.8*w<96)&(y-.52*h>=0)&(y+2.2*h<96)
        best=max(best,float(np.where(valid,score,-2.).max()))
    return best


def eyecheck(z):
    def eye(i): return np.minimum(np.minimum(z[:,2,i],z[:,2,i+1]),np.minimum(z[:,3,i],z[:,3,i+1]))
    def cheek(i): return .25*(z[:,4,i]+z[:,4,i+1]+z[:,5,i]+z[:,5,i+1])
    l=eye(1); r=eye(4); bridge=np.maximum(np.maximum(z[:,2,3],z[:,3,3]),z[:,4,3])
    return ramp(np.minimum(cheek(1)-l,cheek(4)-r),.025,.135)*ramp(bridge-np.maximum(l,r),.018,.12)


def faceparts(lum,tskin,hs):
    # Weighted normalized correlation with independent frontal-face sketches.
    # Only like-with-like illumination comparisons within each candidate.
    model=CONSTANTS['face']; weights=CONSTANTS['fweight']; best=-4.; ref=-4.; reverse=model[::-1,:].copy()
    for gh,gw in ((1,1),(2,1),(2,2),(3,2),(3,3),(4,3),(4,4),(6,4),(6,6),(8,6)):
        # Step one cell (or half of a large cell) between nearby eye locations.
        sy=max(1,gh//2); sx=max(1,gw//2)
        ys=np.arange(gh//2,96-gh+gh//2+1,sy)[:,None]; xs=np.arange(gw//2,96-gw+gw//2+1,sx)[None,:]
        a=rect(lum,ys,xs,gh,gw)
        stepy=round(gh/sy); stepx=round(gw/sx)
        ny=a.shape[0]-7*stepy; nx=a.shape[1]-6*stepx
        patch=np.stack([a[i*stepy:i*stepy+ny,j*stepx:j*stepx+nx] for i in range(8) for j in range(7)],axis=2).reshape(-1,56)
        y=(ys[:ny]+(7*stepy*sy)//2); x=(xs[:,:nx]+3*stepx*sx)
        # Reject tiny, unlikely candidates up front. Larger faces can use purely
        # geometric evidence even on black-and-white photos.
        ss=box(tskin,y,x,gh*8,gw*7,.58,.58)
        sh=hs[y,x]; active=((ss>.22)|(gh>=2))
        if gh<=2:
            active=active & ((np.arange(ny)[:,None]+np.arange(nx)[None,:])%2==0)
        positions=np.nonzero(active.ravel())[0]
        if len(positions)==0: continue
        P=np.ascontiguousarray(patch[positions]); P2=P*P; denom=[]
        # All yaw/hair/mouth variants with the same tilt use the same mask.
        for k in (0,12,24):
            W=weights[:,k:k+1].copy(); n=W.sum(); avg=(P@W)/n
            var=P2@W-n*avg*avg
            denom.append(np.sqrt(np.maximum(var,0.)+(.048**2)*n))
        val=P@model
        # Inverted control uses the same scan positions, reversed inside each
        # local descriptor (rather than a whole second face scan).
        rv=P@reverse; rd=[]
        for k in (0,12,24):
            W=weights[::-1,k:k+1].copy(); n=W.sum(); avg=(P@W)/n
            rd.append(np.sqrt(np.maximum(P2@W-n*avg*avg,0.)+(.048**2)*n))
        f=np.maximum.reduce([val[:,12*i:12*i+12].max(axis=1)/denom[i][:,0] for i in range(3)])
        g=np.maximum.reduce([rv[:,12*i:12*i+12].max(axis=1)/rd[i][:,0] for i in range(3)])
        # Check the gross ordinal relations of both eye sockets and the bridge.
        im=P.reshape(-1,8,7); both=eyecheck(im)
        colour=ss.ravel()[positions]; blob=np.clip(sh.ravel()[positions]+.70,0,1)
        central=np.exp(-((x-47.5)/44.)**2-((y-40.)/48.)**2).ravel()[positions]
        score=3.60*(f-.77)+.52*(both-.66)+.72*(colour-.50)+.43*(blob-.54)+.11*central-.30*(gh==1)-.15*(gh==2)
        best=max(best,float(score.max()))
        check=eyecheck(im[:,::-1,::-1])
        ref=max(ref,float((score+3.60*(g-f)+.52*(check-both)).max()))
    return best,ref


def humanhog(ht,st,hs,tabcol):
    full=-3.; bust=-3.
    for tag in (0,1):
        if tag==0:
            rows=9; cols=5; mats=CONSTANTS['figure']; scales=((3,2),(5,3),(6,3),(8,4),(10,5),(4,3),(6,5),(10,7))
        else:
            rows=6; cols=6; mats=CONSTANTS['bust']; scales=((2,2),(3,2),(4,3),(4,4),(6,4),(6,6),(8,6),(12,9),(14,13))
        for gh,gw in scales:
            G=hoggrid(ht,gh,gw); ny=G.shape[0]-2*(rows-1); nx=G.shape[1]-2*(cols-1)
            sy=max(1,gh//2); sx=max(1,gw//2); dy=round(gh/sy); dx=round(gw/sx)
            ny=G.shape[0]-dy*(rows-1); nx=G.shape[1]-dx*(cols-1)
            ay=2 if min(sy,sx)<=1 else 1; ax=ay
            desc=np.stack([G[i*dy:i*dy+ny:ay,j*dx:j*dx+nx:ax] for i in range(rows) for j in range(cols)],axis=0).transpose(1,2,0,3).copy().reshape(-1,rows*cols*8)
            proj=(desc@mats).max(axis=1).reshape((ny+ay-1)//ay,(nx+ax-1)//ax)
            H=(rows-1)*dy*sy+gh; W=(cols-1)*dx*sx+gw
            y=(np.arange(0,ny,ay)*sy+round(H*(.15 if tag==0 else .26)))[:,None]; x=(np.arange(0,nx,ax)*sx+W//2)[None,:]
            head=box(st,y,x,H,W,.114 if tag==0 else .235,.22 if tag==0 else .26)
            blob=ramp(hs[y,x],-1.,.32)
            chest=box(tabcol,y,x,H,W,.17, .36,.22 if tag==0 else .38)
            a=box(tabcol,y,x,H,W,.17,.20,.22 if tag==0 else .38,-.50); b=box(tabcol,y,x,H,W,.17,.20,.22 if tag==0 else .38,.50)
            d=np.minimum(np.sqrt(np.mean((a-chest)**2,axis=0)),np.sqrt(np.mean((b-chest)**2,axis=0)))
            center=np.exp(-((x-47.5)/47.)**2-((y-40.)/50.)**2)
            score=12.*(proj-(.122 if tag==0 else .143))+.49*(head-.40)+.43*(blob-.38)+.32*(ramp(d,.035,.23)-.45)+.14*center
            if tag==0: full=max(full,float(score.max()))
            else: bust=max(bust,float(score.max()))
    return full,bust


def silhouette(im,st,hs):
    # On patches of grass, sky, sea or snow, choose the background colour from
    # the two sides of EACH scan line. A standing figure should first intercept
    # this colour narrowly (head), more widely (upper body), then as two legs.
    a=np.pad(pyramid(im,2),((0,0),(64,64),(64,64)),mode='edge'); winner=-3.
    for H in (13,21,34,55):
        for aspect in (.27,.44,.64):
            W=max(5,round(H*aspect)); stepy=max(1,H//16); stepx=max(1,W//9)
            ys=np.arange(-round(.12*H),48-round(.69*H),stepy)[:,None]; xs=np.arange(0,48,stepx)[None,:]
            samples={}
            def point(y,x):
                py=round(H*y); px=round(W*x); key=(py,px)
                val=samples.get(key)
                if val is None:
                    i=64+int(ys[0,0])+py; j=64+px
                    val=a[:,i:i+stepy*ys.shape[0]:stepy,j:j+stepx*xs.shape[1]:stepx]; samples[key]=val
                return val
            def diff(u,v):
                da=u-v
                return np.sqrt(.30*da[0]**2+.43*da[1]**2+.27*da[2]**2)
            def row(y,ins,outs):
                l=point(y,-.68); r=point(y,.68)
                sig=.050+.78*diff(l,r)+.35*(diff(l,point(y,-.91))+diff(r,point(y,.91)))
                def fg(x):
                    u=point(y,x); d=np.minimum(diff(l,u),diff(r,u)); return d/(d+sig)
                return sum(fg(x) for x in ins)/len(ins),sum(fg(x) for x in outs)/len(outs),np.clip(sig-.05,0,.4)
            ah,out1,n1=row(.095,(0.,),(-.26,.26))
            an,out2,n2=row(.19,(0.,),(-.40,.40))
            ac,out3,n3=row(.31,(-.14,0.,.14),(-.53,.53))
            am,out4,n4=row(.46,(-.12,0.,.12),(-.49,.49))
            al,out5,n5=row(.67,(-.13,.13),(-.48,.48))
            af,out6,n6=row(.85,(-.16,.16),(-.50,.50))
            head=.6*ah+.4*an; torso=.6*ac+.4*am; legs=.63*al+.37*af
            chest=.16*head+.34*torso+.28*legs+.22*np.minimum(np.minimum(head,torso),legs)
            out=.22*out1+.14*out2+.18*out3+.18*out4+.16*out5+.12*out6
            top=point(-.025,0); bg=.5*(point(-.025,-.49)+point(-.025,.49)); df=diff(top,bg); outtop=df/(df+.08+diff(point(-.025,-.49),point(-.025,.49)))
            spread=diff(point(.31,-.14),point(.31,.14)); n=.20*(n1+n2+n3+n4+n5)
            facey=2*np.clip(ys+round(.12*H),0,47); facex=2*xs
            skin=box(st,facey,facex,H*2,W*2,.10,.22); blob=ramp(hs[facey,facex],-1.1,.4)
            shape=chest-.65*out-.11*outtop-.31*np.minimum(spread,.4)-.36*n
            c=np.exp(-((facex-47.5)/44.)**2-((facey-40.)/51.)**2)
            score=5.0*(shape-.505)+.44*(skin-.36)+.24*(blob-.35)+.12*c-.37*((xs-W*.55<0)|(xs+W*.55>=48))-.17*((ys<0)|(ys+.90*H>=48))
            winner=max(winner,float(score.max()))
    return winner


def context(im,skin,lum,ht,mag,dx,dy):
    r,g,b=im; y,x=np.ogrid[:96,:96]
    outside=(abs(x-47.5)>30)|(abs(y-47.5)>31); center=(abs(x-47.5)<25)&(abs(y-45.5)<29)
    # A subject standing in the field, on the court, in snow, or on the water
    # breaks an otherwise smooth background. Compare the middle with its flanks.
    fine=np.minimum(mag,.30); ecc=fine[center].mean(); env=(fine[:,0:22].mean()+fine[:,-22:].mean())*.5
    spatial=np.clip(np.log((ecc+.022)/(env+.022)),-1.1,1.1)
    Y=lum; total=ht[:,-1,-1]; distr=total/(total.sum()+1e-4)
    axial=distr[0]+distr[4]+.35*(distr[1]+distr[3]+distr[5]+distr[7]); built=ramp(axial,.58,.76)
    # Detect straight, spatially extended edges: rooms, furniture, vehicles.
    arr=np.minimum(abs(dy),.4)*(abs(dy)>3*abs(dx)); arr2=np.minimum(abs(dx),.4)*(abs(dx)>3*abs(dy))
    liness=np.mean(np.partition(arr.reshape(96,4,24).mean(axis=2).ravel(),-16)[-16:])+np.mean(np.partition(arr2.T.reshape(96,4,24).mean(axis=2).ravel(),-16)[-16:])
    structure=.52*built+.48*ramp(liness,.23,.43)
    # High chroma cool or vivid textile against a quieter surround.
    color=np.maximum(r,np.maximum(g,b))-np.minimum(r,np.minimum(g,b))
    bright=ramp(color,.22,.56)*ramp(np.maximum(r,np.maximum(g,b)),.32,.62)*(1-.8*skin)*(1-.8*ramp(g-np.maximum(b,r),-.015,.09))
    shirt=np.clip((bright[center].mean()-bright[outside].mean())*4,-.75,1.0)
    grass=ramp(g-r,-.02,.12)*ramp(g-b,.03,.16)*ramp(g,.18,.35)
    water=ramp(b-r,.02,.17)*ramp(b-g,-.06,.055)
    snow=ramp(Y,.54,.85)*ramp(color,.19,.06)
    lawn=max(float(grass[48:,:26].mean()*grass[48:,-26:].mean())**.5,float(water[50:,:25].mean()*water[50:,-25:].mean())**.5,float(snow[48:,:26].mean()*snow[48:,-26:].mean())**.5)
    # If warm chromaticity is everywhere, local flesh-like patches are less
    # informative (wood, sand, indoor colour casts and close-ups of food).
    ambient=float(skin[outside].mean()); flesh=float(skin[center].mean()); warm=float(np.clip((flesh-.72*ambient)*4,-.60,.75))
    common=ramp(ambient,.18,.48)
    earth=ramp(r-g,-.01,.105)*ramp(g-b,.02,.14)
    earthsur=float(earth[outside].mean())
    # Composition of an action photo: an upright subject protrudes above the
    # background surface. Look for one with locally homogeneous flanks and
    # ground below it. This does not require visible skin or a standing pose.
    gs=pyramid(grass,4); bs=pyramid(water,4); ws=pyramid(snow,4)
    field=max(float(np.minimum(gs[12:].mean(axis=1),gs[12:,-5:].mean(axis=1)).mean()),float(np.minimum(bs[12:].mean(axis=1),bs[12:,-5:].mean(axis=1)).mean()),float(np.minimum(ws[12:].mean(axis=1),ws[12:,-5:].mean(axis=1)).mean()))
    surfaces=np.maximum(grass,np.maximum(water,snow)); baseline=(.50*surfaces[65:].mean()+.25*surfaces[36:72,:23].mean()+.25*surfaces[36:72,-23:].mean())
    # Warm stuff made of wood or patterned fabric often covers both the centre
    # and the margins. Estimate skin enrichment locally relative to BOTH lateral
    # strips, explicitly excluding the overall frame cast.
    skinblocks=pyramid(skin,8); sal=0.; isolated=0.
    for j in range(1,9):
        patch=skinblocks[1:9,j:j+3]; surround=np.maximum(skinblocks[1:9,max(0,j-2):max(1,j)].mean(axis=1),skinblocks[1:9,j+3:min(12,j+5)].mean(axis=1))
        sal=max(sal,float((patch.mean(axis=1)-surround).max()))
    # Skin is spectrally adjacent to food and wood. Food often has substantial
    # yellow/brown to the entire frame, unlike an isolated face in a cool scene.
    vivid=ramp(color,.21,.48)*ramp(r-g,.07,.25)*ramp(g-b,.07,.24)
    yellowish=ramp((r+g)*.5-b,.065,.23)*ramp(g-r,-.24,-.07)
    rustic=ramp((yellowish+vivid*.4)[center].mean(),.22,.58)*ramp(earthsur,.10,.38)
    # Compare coarse vertical and horizontal edge flow after removing fine
    # texture. An orderly room has long axis-aligned edges across the *middle*,
    # whereas a person standing against a simple horizon breaks those edges.
    gray=pyramid(Y,3); gx=gray[1:-1,2:]-gray[1:-1,:-2]; gy=gray[2:,1:-1]-gray[:-2,1:-1]; amp=np.sqrt(gx*gx+gy*gy)+.018
    man=(gx*gx-gy*gy)/amp; cross=2*gx*gy/amp
    cc=man[5:24,5:25]; cd=cross[5:24,5:25]
    diag=(abs(cd).sum()+.1)/(abs(cc).sum()+abs(cd).sum()+.2)
    angular=np.clip((diag-.43)*3.,-.74,.55)
    direction=np.mean(np.abs(np.diff(gray,axis=0)),axis=1)
    horizontal=ramp(direction.max()/(direction.mean()+.008),2.2,4.5)
    # Large areas of fine irregular earth-coloured texture are more typical of
    # foliage or animal hide than a shirt. A centre with contrasting clothes and
    # a little face hit, on the other hand, has varied chroma peaks.
    entropyish=[]
    for ch in (im[0]-im[1],im[1]-im[2]):
        mid=ch[center]; edge=ch[outside]; entropyish.append(float(np.clip((mid.std()-edge.std())/.10,-1.,1.)))
    colsel=sum(entropyish)*.5
    return .48*spatial+.38*shirt+.50*lawn*ramp(spatial,-.14,.57)+.43*warm-.70*structure-.42*common-.28*ramp(earthsur,.28,.65)+.32*field+.24*baseline+.36*(sal-.34)+.36*angular-.29*rustic+.19*colsel-.14*horizontal*(1-lawn)


def classify(im):
    Y=.299*im[0]+.587*im[1]+.114*im[2]; tc=table(im); yl=table(Y)
    # Partial grey-world correction, limited to protect blue/green outdoor scenes.
    av=im.mean(axis=(1,2)); bal=np.clip((av.mean()+.10)/(av+.10),.83,1.20)**.7
    s1=skinfo(im[0],im[1],im[2]); s2=skinfo(im[0]*bal[0],im[1]*bal[1],im[2]*bal[2]); skin=.50*s1+.50*s2
    st=table(skin); gh,mag,dx,dy=gradients(im); ht=table(gh)
    s,other,hs=skinhead(skin,tc,yl,st); f,f2=faceparts(yl,st,hs); p,b=humanhog(ht,st,hs,tc)
    v=silhouette(im,st,hs); ctx=context(im,skin,Y,ht,mag,dx,dy); hand=pairedskin(skin,st,hs,tc,yl)
    # A maximum over thousands of locations is not directly comparable to a
    # score for one proposed patch. Obtain a reference for accidental maxima
    # from impossible orientations of the same anatomy, in the same photograph.
    ctrl=im[:,::-1,:]; tc2=table(ctrl); yl2=table(Y[::-1,:]); st2=table(skin[::-1,:])
    s2,o2,hs2=skinhead(skin[::-1,:],tc2,yl2,st2)
    side=im.transpose(0,2,1); tside=table(side); sside=table(skin.T)
    idx=np.array([4,3,2,1,0,7,6,5]); ht2=ht[idx].transpose(0,2,1).copy()
    hs3=np.maximum(hs,hs2[::-1,:]).T.copy()
    p2,b2=humanhog(ht2,sside,hs3,tside); v2=silhouette(side,sside,hs3)
    # Bounded absolute evidence: multiple weak, unrelated local maxima should
    # not substitute for a good upright-vs-control match.
    def bounded(z,r): return float(np.clip(z,-r,r))
    def excess(z,ref,r): return bounded(z-ref,r)
    uf=excess(f,f2,1.); us=excess(s,s2,1.); up=excess(p,p2,1.); ub=excess(b,b2,1.); uv=excess(v,v2,1.35)
    evidence=[.66*us+.39*bounded(s,1.), .68*uf+.28*bounded(f,1.), up+.12*bounded(p,1.), ub+.12*bounded(b,1.), .68*uv+.16*bounded(v,1.3)]
    # A high value from one test is useful, but also count the others, which
    # can contradict it. Global scene cues don't get a max-over-windows boost.
    evidence.sort()
    score=.38*evidence[4]+.24*evidence[3]+.20*evidence[2]+.12*evidence[1]+.06*evidence[0]
    score+=.88*ctx+.14*bounded(s+.42*other,1.1)+.10*bounded(f,1.)+.63*bounded(hand,.80)-.19
    return int(score>0.)


def predict(x):
    return np.array([classify(np.ascontiguousarray(im)) for im in x],dtype=np.int32)
'''

if __name__=='__main__':
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    oriented_templates()
    (out/'solution.py').write_text(SOURCE)
    np.savez(out/'constants.npz',**PARAMS)
