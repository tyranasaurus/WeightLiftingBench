
import numpy as np

def blur(a):
    b=np.pad(a,((0,0),(1,1),(1,1)),mode="edge")
    return (b[:,:-2,:-2]+b[:,:-2,2:]+b[:,2:,:-2]+b[:,2:,2:]+2*(b[:,1:-1,:-2]+b[:,1:-1,2:]+b[:,:-2,1:-1]+b[:,2:,1:-1])+4*b[:,1:-1,1:-1])*.0625

def grad(a):
    b=np.pad(a,((0,0),(1,1),(1,1)),mode="edge")
    return ((b[:,1:-1,2:]-b[:,1:-1,:-2])*.5+(b[:,:-2,2:]-b[:,:-2,:-2]+b[:,2:,2:]-b[:,2:,:-2])*.25, (b[:,2:,1:-1]-b[:,:-2,1:-1])*.5+(b[:,2:,:-2]-b[:,:-2,:-2]+b[:,2:,2:]-b[:,:-2,2:])*.25)

def histedge(g):
    x,y=grad(g);l=np.sqrt(x*x+y*y+.0000000001);t=np.arctan2(y,x)*(6/np.pi);ans=[]
    for k in range(6):
        d=np.abs((t-k+3)%6-3);ans.append(l*np.maximum(1-d,0))
    return np.stack(ans,axis=1)

def pool(a):
    n,c,h,w=a.shape
    return a.reshape(n,c,8,h//8,8,w//8).mean(axis=(3,5))

def desc(a):
    f=pool(a); b=np.pad(f,((0,0),(0,0),(1,1),(1,1)),mode="edge"); f=.68*f+.08*(b[:,:,1:-1,2:]+b[:,:,1:-1,:-2]+b[:,:,:-2,1:-1]+b[:,:,2:,1:-1]);
    m=np.sum(f,axis=1,keepdims=True); s=m.mean(axis=(2,3),keepdims=True); f=f/(.012+.48*m+.65*s); return f

def sample(x,scale,dx,dy):
    n,c,h,w=x.shape
    grid=(np.arange(32,dtype=np.float32)-15.5)*scale+15.5
    xx=np.clip(grid+dx,0,31); yy=np.clip(grid+dy,0,31); ix=xx.astype(np.int32);iy=yy.astype(np.int32);fx=(xx-ix).astype(np.float32);fy=(yy-iy).astype(np.float32)
    b=x[:,:,:,ix]*(1-fx)+x[:,:,:,np.minimum(ix+1,31)]*fx
    return np.ascontiguousarray(b[:,:,iy,:]*(1-fy[:,None])+b[:,:,np.minimum(iy+1,31),:]*fy[:,None])

def foreground(x):
    # Minimum uphill barrier to any point on the border in Lab-like colour
    # opponent space. A smooth sky gradient stays connected to the border;
    # an enclosed coat, car body, etc. requires crossing a colour boundary.
    sm=np.stack([blur(blur(x[:,i])) for i in range(3)],axis=1);z=np.stack([.29*sm[:,0]+.59*sm[:,1]+.12*sm[:,2],.75*(sm[:,0]-sm[:,1]),.62*(sm[:,2]-sm[:,1])],axis=1); dh=np.sqrt(((z[:,:,:,1:]-z[:,:,:,:-1])**2).sum(axis=1));dv=np.sqrt(((z[:,:,1:]-z[:,:,:-1])**2).sum(axis=1));d=np.ones_like(dh[:,:,:1]);d=np.ones_like(z[:,0]);d[:,0]=0;d[:,-1]=0;d[:,:,0]=0;d[:,:,-1]=0
    for step in range(44):
        q=d.copy();q[:,:,1:]=np.minimum(q[:,:,1:],np.maximum(d[:,:,:-1],dh));q[:,:,:-1]=np.minimum(q[:,:,:-1],np.maximum(d[:,:,1:],dh));q[:,1:]=np.minimum(q[:,1:],np.maximum(d[:,:-1],dv));q[:,:-1]=np.minimum(q[:,:-1],np.maximum(d[:,1:],dv));d=q
    # Soft colour rarity relative to a palette from the four borders. Decay
    # is slower across the picture than up and down across sky/ground.
    rare=np.ones_like(d);g=np.arange(32,dtype=np.float32)
    for lo in [0,5,10,15,20,25]:
        for side in [0,1,2,3]:
            if side<2:
                q=z[:,:,lo:lo+7,:3] if side==0 else z[:,:,lo:lo+7,-3:];cost=.0018*abs(g[:,None]-(lo+3))
            else:
                q=z[:,:,:3,lo:lo+7] if side==2 else z[:,:,-3:,lo:lo+7];cost=.001*abs(g[None,:]-(lo+3))+.0015*(g[:,None] if side==2 else 31-g[:,None])
            delta=z-q.mean(axis=(2,3),keepdims=True);rare=np.minimum(rare,np.sqrt((delta*delta).sum(axis=1))+cost)
    contrast=np.sqrt(((z[:,0]-z[:,0].mean(axis=(1,2),keepdims=True))**2).mean(axis=(1,2),keepdims=True));
    # Barrier dominates for coherent outlines; colour rarity fills in their interiors.
    return .62*cap((d-.012)/(.054+.20*contrast))+.38*cap((rare-.037)/(.105+.30*contrast))

def modelmatch(x,fg):
    n=len(x);g=.299*x[:,0]+.587*x[:,1]+.114*x[:,2]; edge=histedge(g); rg=x[:,0]-x[:,1];bg=x[:,2]-x[:,1];edge+=.27*(histedge(rg)+histedge(bg)); channels=np.concatenate([edge,g[:,None],fg[:,None]],axis=1);del edge
    fs=[];gs=[];vs=[];priors=[]; materials=materialmaps(x); extra=np.concatenate([g[:,None],fg[:,None],materials],axis=1)
    for scale,offset in [(.54,4),(.78,3.5),(1.02,2.8),(1.26,1.5)]:
        for dx,dy in [(0,0),(-offset,-offset*.45),(offset,-offset*.45),(-offset,offset*.80),(offset,offset*.80),(0,-offset),(0,offset)]:
            sampled=sample(channels,scale,dx,dy); f=desc(sampled[:,:6]);v=pool(sampled[:,6:7]);v-=v.mean(axis=(2,3),keepdims=True); obs=pool(sampled[:,7:]);obs-=obs.mean(axis=(2,3),keepdims=True);obs/=np.sqrt((obs*obs).sum(axis=(2,3),keepdims=True)+1.35)
            v=v/(.12+3.5*np.sqrt((v*v).mean(axis=(2,3),keepdims=True)))
            hp=sample(extra,scale,dx,dy).reshape(n,extra.shape[1],16,2,16,2).mean(axis=(3,5)); lum=hp[:,:1].copy();lum-=lum.mean(axis=(2,3),keepdims=True); hp[:,:1]=lum/np.sqrt((lum*lum).sum(axis=(2,3),keepdims=True)+.95); shape=hp[:,1:2].copy();shape-=shape.mean(axis=(2,3),keepdims=True); hp[:,1:2]=shape/np.sqrt((shape*shape).sum(axis=(2,3),keepdims=True)+4)
            f-=.80*f.mean(axis=(2,3),keepdims=True)+.20*f.mean(axis=(1,2,3),keepdims=True);f/=np.sqrt((f*f).sum(axis=(1,2,3),keepdims=True)+.06)
            for flip in [False,True]:
                hv=(hp[:,:,:,::-1] if flip else hp).reshape(n,extra.shape[1],256);gs.append(hv[:,:2].reshape(n,512));vs.append(hv[:,2:]);ef=f[:,[0,5,4,3,2,1],:,::-1] if flip else f; vf=v[:,:,:,::-1] if flip else v; of=obs[:,:,:,::-1] if flip else obs;fs.append(np.concatenate([ef.reshape(n,384),.075*vf.reshape(n,64),.40*of.reshape(n,64)],axis=1));priors.append((.008 if dx or dy else 0)+(.011 if scale>1.2 else 0))
    fs=np.stack(fs,axis=1); gs=np.stack(gs,axis=1);vs=np.stack(vs,axis=1); m=CONSTANTS['models']; cuts=CONSTANTS['cuts']; out=[]; rich=[]; priors=np.array(priors,dtype=np.float32).reshape(1,-1,1)
    for c in range(10):
        mods=np.ascontiguousarray(m[:,cuts[c]:cuts[c+1]]);a=fs.reshape(-1,512)@mods[:512]; a=a.reshape(n,len(fs[0]),-1)-priors
        pr=np.concatenate([.60*gs[:,:,:256],.48*gs[:,:,256:],.29*vs[:,:,c]],axis=2);b=(pr.reshape(-1,768)@mods[512:]).reshape(a.shape)+.54*a-priors
        # Best pose and support from nearby poses, not a vote by number of templates.
        for v,dest in [(a,out),(b,rich)]:
            mu=v.mean(axis=(1,2)); vmax=v.max(axis=2); best=vmax.max(axis=1); near=np.partition(vmax,-6,axis=1)[:,-6:].mean(axis=1); raw=.73*best+.27*near;dest.append(raw-.43*mu)
    # Contrast a match at the best pose against the same pose vocabulary
    # on this image. A few distinctive aligned contours are stronger evidence
    # than a generally large response to the image's orientation histogram.
    baseline=np.stack(out,axis=1); detail=np.stack(rich,axis=1)
    return baseline,detail

def cap(a):
    return np.clip(a,0,1)

def ramp(a,mid,width):
    return cap(.5+(a-mid)/width)

def avg(a):
    return a.mean(axis=(-2,-1))

def high(a):
    return np.maximum(a,0)

def boxmean(a,k):
    p=np.pad(a,((0,0),(k,k),(k,k)),mode="edge");s=np.pad(p,((0,0),(1,0),(1,0)),mode="constant").cumsum(1).cumsum(2);d=k*2+1
    return (s[:,d:,d:]+s[:,:-d,:-d]-s[:,d:,:-d]-s[:,:-d,d:])/(d*d)

grid=np.arange(32,dtype=np.float32); cy,cx=np.meshgrid(grid,grid,indexing="ij")

def region(w):
    return w/max(.00001,w.sum())

middle=region(np.exp(-((cx-15.5)/9.5)**2-((cy-15.5)/10.0)**2)); outer=region(1-np.exp(-((cx-15.5)/13)**4-((cy-15.5)/13)**4)); ground=region(ramp(cy,23,6)); overhead=region(1-ramp(cy,8,6)); flank=region(ramp(abs(cx-15.5),11,6)*(1-ramp(cy,22,9))); margins=region(ramp(abs(cx-15.5),10,6));

def area(a,mask):
    return np.sum(a*mask,axis=(-1,-2))

def layout(x):
    # Spatial colour, material and texture histograms for matching a photo to
    # a *complete* procedural scene. No learned embedding or readout.
    r,g,b=x[:,0],x[:,1],x[:,2];y=.30*r+.58*g+.12*b;yg=g-b;rg=r-g;rb=r-b;ch=np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b);soft=1-ramp(ch,.088,.18);warm=ramp(rb,.072,.16)*ramp(rg,.034,.12);leaf=ramp(yg,.04,.13)*ramp(rg,.018,.10);green=ramp(yg,.042,.15)*(1-ramp(rg,.014,.10));sky=ramp(-rb,.055,.14);vivid=ramp(ch,.235,.24);paint=ramp(r-np.maximum(g,b),.13,.17);gray=soft*ramp(y,.26,.19)*(1-ramp(y,.82,.20));wh=ramp(y,.71,.23)*(.44+.56*soft);dk=1-ramp(y,.27,.21);greenish=ramp(g-.48*r-.52*b,.037,.096);lux=ramp(y,.52,.48);opp=ramp(rb,0,.44);tint=ramp(yg-rg,0,.31)
    hh=histedge(y);ex,ey=grad(y);vx,vy=grad(blur(blur(y)));mag=np.sqrt(ex*ex+ey*ey+.00000001);smoothmag=np.sqrt(vx*vx+vy*vy+.00000001);hi=abs(y-blur(y));lo=abs(y-boxmean(y,3));rug=ramp(boxmean(hi,2),.026,.040);smooth=ramp(.078-boxmean(mag,2),0,.105);lineX=ramp(abs(vx)-.62*abs(vy),.03,.09);lineY=ramp(abs(vy)-.62*abs(vx),.03,.09);subject=abs(y-boxmean(y,5));sp=ramp(subject,.092,.16)
    s=np.stack([lux,opp,tint,wh,dk,gray,green,sky,warm,greenish,paint,vivid,rug,smooth,sp,lineX,lineY],axis=1)
    # Block grouping. Give sky and ground strips a voice independent of the
    # coarse object cells. Reflection-invariant on the broad scene descriptor.
    v=s.reshape(len(x),17,4,8,4,8).mean(axis=(3,5));v=.70*v+.30*v[:,:,:,::-1]; stripes=s.mean(axis=3).reshape(len(x),17,4,8).mean(axis=3); borders=np.stack([s[:,:,:8].mean(axis=(2,3)),s[:,:,-8:].mean(axis=(2,3)),(s[:,:,:,:6]+s[:,:,:,-6:]).mean(axis=(2,3))*.5,s[:,:,7:25,7:25].mean(axis=(2,3))],axis=2)
    # HOG groups keep a high-contrast irregular foreground distinct from a
    # horizontal dark hull or a box. Normalize by total local edge content.
    bank=hh.reshape(len(x),6,4,8,4,8).mean(axis=(3,5));bank=bank/(.024+bank.sum(axis=1,keepdims=True));edge=cap(np.log1p(24*mag)/2.1).reshape(len(x),4,8,4,8).mean(axis=(2,4));
    return np.concatenate([v.reshape(len(x),-1)*.66,stripes.reshape(len(x),-1)*.90,borders.reshape(len(x),-1)*1.05,(bank*.85).reshape(len(x),-1),edge.reshape(len(x),-1)],axis=1)

def jointmatch(x,fg):
    r,g,b=x[:,0],x[:,1],x[:,2];y=.30*r+.58*g+.12*b; c= np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b); neutral=1-ramp(c,.10,.16);dark=1-ramp(y,.30,.30);light=ramp(y,.67,.28);mid=1-ramp(abs(y-.48),.28,.19); green=ramp(g-b,.045,.14)*ramp(g-r,-.005,.118);warm=ramp(r-b,.058,.17)*ramp(r-g,.031,.13);br=ramp(b-r,.045,.145);cyan=ramp(np.minimum(b-r,g-r),.046,.145);blue=np.maximum(br,cyan);red=ramp(r-np.maximum(g,b),.09,.16);gold=ramp(g-b,.057,.17)*ramp(r-g,.021,.14);ch=ramp(c,.212,.28);col=np.maximum.reduce([red,green,blue]);lumdev=y-avg(y)[:,None,None]
    h,v=grad(y);m=np.sqrt(h*h+v*v);edges=boxmean(m,2);s=1-ramp(edges,.073,.102);rough=ramp(boxmean(abs(y-blur(y)),2),.025,.039);hor=ramp(boxmean(abs(v)-abs(h),2),.010,.085);smooth=1-ramp(boxmean(abs(y-boxmean(y,2)),2),.045,.075)
    # Probabilities are deliberately soft. Textureless blue may explain an
    # aircraft photograph as sky, or a marine photograph as sea; their roles
    # are disambiguated by the location of the model and its waterline.
    def prob(z):return np.log(.13+.87*cap(z))
    def normz(z):return cap(z)
    sky= (.21+.79*s)*np.maximum.reduce([.18+.82*blue,.13+.83*light*neutral,.13+.58*mid*neutral*s]);water=np.maximum.reduce([.17+.82*blue,.17+.55*mid*neutral,.12+.34*green])*(.56+.29*s+.15*hor);field=np.maximum(.16+.80*green,.14+.55*gold)*(.76+.18*mid+.06*s);woods=np.maximum.reduce([.16+.82*green,.16+.53*warm,.11+.44*dark])*(.61+.23*rough+.16*dark);earth=np.maximum.reduce([.17+.77*warm,.16+.64*gold,.13+.42*mid*neutral])*(.69+.31*rough);road=(.14+.81*neutral)*(.54+.46*mid)*(.60+.40*s);wall=np.maximum(.32+.43*neutral,.19+.43*warm)*(.79+.21*mid);cloth=np.maximum.reduce([.3+.36*warm,.23+.37*neutral,.24+.32*dark,.22+.34*ch])*(.79+.21*rough);blank=(.15+.80*s)*(.45+.55*neutral);mount=np.maximum.reduce([.20+.51*green,.20+.48*blue,.20+.43*neutral*mid]);shadow=.15+.75*dark+.10*ramp(-lumdev,.13,.24);window=np.maximum(.23+.67*dark,.26+.46*blue);enamel=.13+.82*neutral*light
    bg=np.stack([sky,water,field,woods,earth,road,wall,cloth,blank,mount,shadow,window,enamel],axis=1);bg=pool(prob(bg)-.56*fg[:,None]);
    metallic=np.maximum.reduce([.18+.73*neutral,.18+.66*red,.19+.60*blue]);fur=np.maximum.reduce([.23+.61*warm,.20+.55*dark,.21+.52*light*neutral,.23+.42*mid*neutral]);hoof=np.maximum.reduce([.19+.75*warm,.16+.61*dark,.17+.49*light*neutral]);
    obj=[.19+.64*neutral+.08*blue,metallic,np.maximum.reduce([.26+.54*dark,.21+.56*warm,.22+.56*ch,.23+.45*neutral]),fur*(.80+.20*rough),(.17+.72*warm+.11*mid)*(.90+.10*s),fur*(.87+.13*mid),np.maximum(.19+.76*green,.20+.65*warm)*(.75+.25*rough),hoof*(.79+.21*s),.23+.61*neutral+.095*blue,metallic]
    planes=CONSTANTS['plans'];cuts=CONSTANTS['jcuts'];out=[]
    for k in range(10):
        feat=np.concatenate([pool((prob(obj[k])+.38*fg)[:,None]),bg],axis=1).reshape(len(y),-1);a=feat@np.ascontiguousarray(planes[:,cuts[k]:cuts[k+1]]);near=np.partition(a,-12,axis=1)[:,-12:];out.append(.45*a.max(axis=1)+.55*near.mean(axis=1))
    return np.stack(out,axis=1)

def worldmatch(x):
    f=layout(x);m=CONSTANTS['scenes'];cuts=CONSTANTS['ranges'];values=[];norms=CONSTANTS['powers'];
    for c in range(10):
        a=cuts[c];b=cuts[c+1];dots=2*f@np.ascontiguousarray(m[:,a:b])-norms[None,a:b];near=np.partition(dots,-8,axis=1)[:,-8:];values.append(.33*dots.max(axis=1)+.67*near.mean(axis=1))
    return np.stack(values,axis=1)

def materialmaps(x):
    r,g,b=x[:,0],x[:,1],x[:,2];y=.30*r+.58*g+.12*b;bright=ramp(y,.67,.34);dark=1-ramp(y,.31,.32);ch=np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b);gray=1-ramp(ch,.095,.17);tan=ramp(r-b,.063,.18)*ramp(r-g,.020,.14);leaf=ramp(g-b,.040,.15)*ramp(g-r,-.009,.104);paint=ramp(np.maximum(r,b)-g,.10,.22);vivid=ramp(ch,.25,.27);olive=ramp(g-b,.043,.17)*ramp(r-g,-.008,.16);texture=abs(y-blur(y));coarse=ramp(boxmean(texture,2),.025,.034);smooth=1-ramp(boxmean(texture,2),.016,.026);highlight=high(y-blur(blur(y)));blotch=ramp(boxmean(highlight,2),.021,.034)
    # Positive material membership likelihoods. Compare mean memberships
    # only within each candidate silhouette, retaining camouflage cases.
    metals=.30*gray+.20*paint-.44*leaf-.24*tan-.22*coarse+.26*smooth
    fur=.47*tan+.16*gray-.36*leaf-.24*paint+.19*coarse
    maps=[metals+.32*bright-.08,metals+.26*dark+.28*paint-.08,.26*vivid+.25*dark+.17*tan+.10*gray+.18*smooth-.10,fur+.28*blotch+.11*bright-.12,fur+.32*tan-.26*blotch+.12*smooth,fur+.14*bright+.20*dark-.11, .43*olive+.47*leaf+.16*tan+.34*coarse-.13*gray-.14*bright,fur+.20*tan+.24*dark+.17*bright-.29*coarse+.08*smooth,metals+.30*bright+.13*dark-.14,metals+.28*bright+.19*paint-.05]
    return np.stack(maps,axis=1)

def shifted(a,dy,dx):
    p=np.pad(a,((0,0),(abs(dy),abs(dy)),(abs(dx),abs(dx))),mode="edge");sy=abs(dy)+dy;sx=abs(dx)+dx;return p[:,sy:sy+32,sx:sx+32]

def contourparts(x,y,fg):
    # Colour structure tensor in small neighbourhoods. Rectilinearity measures
    # straight coherent edges, not isolated axis-aligned gradient pixels.
    xx=np.zeros_like(y);yy=xx.copy();xy=xx.copy()
    for a,w in [(y,1),(x[:,0]-x[:,1],.40),(x[:,2]-x[:,1],.40)]:
        u,v=grad(blur(a));xx+=w*u*u;yy+=w*v*v;xy+=w*u*v
    xx=boxmean(xx,2);yy=boxmean(yy,2);xy=boxmean(xy,2);energy=xx+yy;ex=xx-yy;am=np.sqrt(ex*ex+4*xy*xy);cohere=am/(.0006+energy); axisness=ex/(.0005+am);line=cohere*ramp(np.sqrt(high(energy)),.058,.075);ho=line*ramp(-axisness,.67,.45);ve=line*ramp(axisness,.67,.45)
    ho=ho*.46+.27*(shifted(ho,0,-3)+shifted(ho,0,3));ve=ve*.46+.27*(shifted(ve,-3,0)+shifted(ve,3,0));
    w=(.3+.7*fg)*(.5+.5*32*32*middle); lin=area((ho+ve)*w,middle);cur=area((1-cohere)*ramp(np.sqrt(high(energy)),.073,.10),middle);ripples=area(ho-ho*ve,ground)
    # Large, uninterrupted horizontal edges, including hull/water boundaries.
    hp=ho.reshape(-1,32,4,8).mean(axis=3); longline=np.partition(hp[:,12:26].reshape(len(y),-1),-3,axis=1)[:,-3:].mean(axis=1)
    planar=ramp(lin,.174,.21); fur=ramp(cur,.17,.21);stacked=ramp(longline,.51,.45)
    diagonal=line*(1-ramp(abs(axisness),.63,.45)); diag=ramp(area(diagonal,middle),.19,.27)
    return planar,fur,stacked,diag,ramp(ripples,.105,.20)

def reflectance(y,fg,warm,green,blue,red):
    # Material cues within probable foreground, rather than treating grass or
    # asphalt in the background as the bulk colour of the object.
    w=middle*(.13+fg**1.65);w/=w.sum(axis=(1,2),keepdims=True);gy,gx=grad(y)[::-1];mag=np.sqrt(gx*gx+gy*gy);fine=abs(y-blur(y));env=boxmean(mag,2); hi=.20+area(y,w)[:,None,None];lo=area(y,w)[:,None,None]-.20
    shades=area(ramp(hi-y,0,.15)*ramp(y-lo,0,.15),w);spot=area(ramp(fine,.052,.082),w);dark=area(ramp(.27-y,0,.19),w);rich=area(warm,w);moss=area(green,w);hue=area(np.maximum(blue,red),w);bright=area(ramp(y,.70,.19),w);smooth=area(1-ramp(env,.056,.077),w)
    return shades,spot,dark,rich,moss,hue,bright,smooth

def sceneprops(x,fg):
    r,g,b=x[:,0],x[:,1],x[:,2];y=(r+g*2+b)*.25; lum=avg(y); gy,gx=grad(y)[::-1];mag=np.sqrt(gx*gx+gy*gy+.00000001); detail=abs(y-blur(y)); local=boxmean(mag,2); smooth=1-ramp(local,.068,.105)
    colmax=np.maximum(np.maximum(r,g),b);colmin=np.minimum(np.minimum(r,g),b);chroma=colmax-colmin
    green=ramp(g-b,.043,.18)*ramp(g-r,-.008,.11)*ramp(g-.48*r-.52*b,.032,.095)
    blue=ramp(b-r,.045,.16)*ramp(b-g,.002,.10);cyan=ramp(b-r,.060,.16)*ramp(g-r,.025,.13);warm=ramp(r-b,.062,.17)*ramp(r-g,.034,.12);soil=ramp(g-b,.027,.15)*ramp(r-g,-.007,.10)*ramp(r-b,.06,.14)
    red=ramp(r-np.maximum(g,b),.13,.23);purple=ramp(np.minimum(r,b)-g,.055,.12);brightcolor=ramp(chroma,.25,.28)
    sky=(.36+.64*smooth)*np.maximum.reduce([blue,cyan*.88,.65*ramp(y,.70,.28)*(1-ramp(chroma,.14,.17))]);water=np.maximum(blue,cyan)*(.67+.33*smooth); grass=green*(.7+.3*(1-smooth));turf=np.maximum(green,.42*soil)
    vegetation=.42*area(green,overhead)+.35*area(green,flank)+.23*area(green,ground);field=area(turf,ground); skyTop=area(sky,overhead);skySide=area(sky,flank); waterLow=area(water,ground); plainTop=area(smooth,overhead);plainSide=area(smooth,flank);plainLow=area(smooth,ground)
    # Dark well-defined markings, and bright near-neutral ground are common on vehicles.
    shadow=ramp(.23-y,0,.21);deep=area(shadow,middle); light=area(ramp(y,.74,.24),middle); contrast=np.sqrt(avg((y-lum[:,None,None])**2)); paint=area(red,middle);warmbody=area(warm,middle);bluebody=area(blue,middle);greenbody=area(green,middle);leafcolor=area(brightcolor,flank); jewel=area(brightcolor,middle)
    road=area((1-ramp(chroma,.082,.13))*(1-ramp(y,.78,.23))*(.55+.45*smooth),ground)
    neutral=1-ramp(chroma,.09,.13);white=area(neutral*ramp(y,.70,.22),middle)
    # Ratios of co-oriented gradients and of gradients surviving horizontal
    # averaging: many horizontal parts in stacked decks, chassis and windows.
    h=abs(gy); v=abs(gx); mh=avg(h);mv=avg(v);hdom=ramp((mh-mv)/(.014+mh+mv),.085,.30);axis=avg(abs(h-v))/(.01+avg(mag));rect=ramp(axis,.565,.21)
    hstrip=np.abs(gy.reshape(-1,32,4,8).mean(axis=3));vstrip=np.abs(gx.reshape(-1,4,8,32).mean(axis=2)); rows=hstrip.mean(axis=2); hr=hstrip.mean(axis=(1,2))/(.014+mh);vr=vstrip.mean(axis=(1,2))/(.014+mv); straight=ramp(.6*hr+.4*vr,.51,.26);flatline=ramp(hr,.56,.34);upright=ramp(vr,.56,.34)
    horizstrip=ramp(rows[:,14:27].max(axis=1)/(.027+mh),1.60,1.50)
    texture=ramp(avg(detail),.026,.034); rough=ramp(area(local,middle),.136,.17); bgrough=1-(plainTop+plainLow+plainSide)/3
    # Find smooth backdrops as well as scene cut-outs on a white page.
    bgcolor=x[:, :, :7].mean(axis=(2,3)); difference=np.abs(x-bgcolor[:,:,None,None]).mean(axis=1); around=area(difference,outer); uniform=ramp(.134-around,0,.14); pale=area(ramp(y,.87,.17)*neutral,outer);studio=ramp(pale,.64,.53)
    topcol=x[:,:,:6].mean(axis=(2,3));bottomcol=x[:,:,-6:].mean(axis=(2,3)); twotone=ramp(abs(topcol-bottomcol).mean(axis=1),.12,.22); separated=twotone*(.4+.6*plainTop)*(.4+.6*plainLow)
    # Soft saliency mask: compare to flank colours at several heights, and
    # to all four corners. Broadly similar colours on a hillside should not
    # define the object's extent just because of a horizon line.
    stripes=[]
    for start in [0,7,14,21,26]:
        sides=x[:,:,start:start+6];stripes.append(np.stack([sides[:,:,:,:5].mean(axis=(2,3)),sides[:,:,:,-5:].mean(axis=(2,3))],axis=2))
    probs=[];sm=np.stack([blur(x[:,i]) for i in range(3)],axis=1)
    for loc,s in zip([3,10,17,24,29],stripes):
        l=abs(sm-s[:,:,0,None,None]);q=abs(sm-s[:,:,1,None,None]);d=np.minimum(l.mean(axis=1),q.mean(axis=1)); probs.append(d+.0045*abs(cy-loc))
    dist=np.minimum.reduce(probs); obj=ramp(dist,.129,.19);centerobj=area(obj,middle);sideobj=area(obj,margins);ob=area(obj,overhead);ub=area(obj,ground);compact=cap((centerobj-.45*sideobj-.18*ob-.18*ub)*1.7);cutout=compact*uniform
    mass=obj*(.55+.45*np.exp(-((cx-15.5)/14)**4-((cy-15.5)/14)**4));zm=mass.sum(axis=(1,2))+.01;ox=(mass*cx).sum(axis=(1,2))/zm;oy=(mass*cy).sum(axis=(1,2))/zm
    wx=np.sqrt((mass*(cx-ox[:,None,None])**2).sum(axis=(1,2))/zm+.01);wy=np.sqrt((mass*(cy-oy[:,None,None])**2).sum(axis=(1,2))/zm+.01);broad=ramp(wx/(wy+.5),1.3,.85)*compact; tall=ramp(wy/(wx+.5),1.25,.8)*compact
    finelegs=legcue(y); wheels=wheelcue(y);eyes,nose=facecue(y)
    manufactured=1.1*(straight-.5)+.70*(rect-.5)+.60*(hdom-.5)+.50*(flatline-.5); vehicle=manufactured+.72*(road-.5)-1.10*vegetation-.40*field+.36*(deep+light)+.34*(contrast-.20)*4
    landscape=1.70*skyTop+1.32*waterLow+.46*skySide+.34*plainTop+.25*plainSide-1.05*vegetation-.36*warmbody
    natural=1.5*vegetation+.52*field-.47*(straight-.5)-.25*(rect-.5)-.58*waterLow
    # Handwritten evidence, on an approximately log-likelihood scale. A single
    # habitat clue rarely suffices; terms such as sky * water are conjunctions.
    context=[]
    context.append(landscape+.92*skySide+.75*uniform+.48*plainLow+.45*cutout+.73*broad+.29*studio-.68*waterLow-.28*deep-.30*texture-1.17)
    context.append(vehicle+.96*wheels+.45*road+.56*paint+.48*deep+.38*straight+.33*broad-.43*skyTop-.35*skySide-.47)
    context.append(.66*natural+.56*plainSide+.62*compact+.55*tall+.45*leafcolor+.28*jewel+.24*uniform+.42*waterLow-.58*flatline-.62*broad-.48)
    context.append(.24*(1-straight)+.70*eyes+.34*nose+.47*warmbody+.52*rough+.44*texture+.39*bgrough-.80*skyTop-.70*field-.54*waterLow-.34*broad-.56)
    context.append(natural+.73*field+.41*area(soil,flank)+.47*area(soil,ground)+.58*finelegs+.47*warmbody+.45*plainSide-.31*jewel-.47*deep-.43*contrast*2-.49)
    context.append(.30*natural+.71*nose+.42*eyes+.58*warmbody+.27*field+.33*deep+.23*bgrough+.22*tall-.62*skyTop-.26*flatline-.43)
    context.append(.34*natural+.88*greenbody+.64*area(soil,middle)+.53*area(soil,outer)+.82*texture+.70*rough+.60*bgrough+.35*jewel-.52*skyTop-.48*finelegs-.43*tall-.41)
    context.append(.70*natural+.87*field+.94*finelegs+.60*(skyTop*field)+.62*warmbody+.39*deep+.32*twotone+.24*broad-.37*greenbody-.35*texture-.53)
    context.append(landscape+.80*waterLow+.79*(skyTop*waterLow)+.60*horizstrip+.62*flatline+.48*hdom+.38*separated+.26*broad-.50*upright-.35*road-1.12)
    context.append(vehicle+.49*wheels+.62*skyTop+.57*upright+.35*straight+.51*white+.34*paint+.26*(skyTop*road)-.30*broad-.58)
    planar,fur,stacked,diagonal,ripples=contourparts(x,y,fg);shades,speck,darkcoat,rich,moss,hue,bright,smoothcoat=reflectance(y,fg,warm,green,blue,red)
    # Compare the material and surrounding habitat with the physical model.
    metal=1.10*(planar-.50)-.80*(fur-.5)+.60*(stacked-.50)-.90*(speck-.22)
    coat=.75*(fur-.5)-.67*(planar-.50)+.54*(rich-.32)-.36*(hue-.12)
    outdoor=skyTop*plainTop; wooded=np.maximum(.91*vegetation,area(soil,flank)*.62); marsh=area(green+soil*.6,ground)
    context[0]+=.40*metal+1.40*(uniform-.45)+.84*(cutout-.27)+.68*(plainTop-.55)+.62*(diagonal-.45)-.73*(stacked-.5)+.63*(skyTop-.40)-.75*(ripples-.45)
    context[1]+=.89*metal+.54*(darkcoat-.31)+.62*(hue-.16)-.72*(outdoor-.28)-.90*(moss-.2)
    context[2]+=.70*(plainSide-.45)+.78*(compact-.50)+.39*(diagonal-.40)-.74*(planar-.50)+.47*(hue-.13)+.36*(wooded-.25)-.22*(rich-.35)
    context[3]+=coat+.65*(speck-.20)+.46*(rough-.46)+.54*(bgrough-.42)-.60*(field-.36)+.28*(eyes-.50)
    context[4]+=.79*coat+.78*(wooded-.30)+.60*(marsh-.45)+.63*(shades-.62)-.67*(hue-.12)-.46*(darkcoat-.25)
    context[5]+=coat+.36*(darkcoat-.25)+.44*(bright-.19)-.45*(wooded-.30)+.57*(nose-.45)
    context[6]+=.56*(fur-.5)+.84*(speck-.23)+.73*(moss-.18)+.54*(rich-.35)-.64*(smoothcoat-.35)-.70*(diagonal-.4)-.84*(skySide-.33)
    context[7]+=.60*coat+.50*(darkcoat-.25)+.68*(field-.40)+.50*(separated-.45)+.48*(finelegs-.50)-.66*(speck-.22)
    context[8]+=.42*metal+1.1*(ripples-.43)+.72*(stacked-.48)+.54*(separated-.39)-.56*(diagonal-.45)+.71*(waterLow-.31)-.60*(uniform-.48)
    context[9]+=.80*metal+.65*(white-.20)+.42*(skyTop-.40)-.45*(darkcoat-.27)-.70*(moss-.19)
    # Explicit scene hypotheses: each is a conjunction of a few *distinct*
    # cues. A class can be supported by any one of its natural settings. This
    # avoids rewarding a dozen correlated measurements of the same texture.
    spec=lambda a:.14+.86*cap(a)
    def conj(*args):
        prod=1
        for q in args:prod=prod*spec(q)
        return prod**(1/len(args))
    def either(*args):
        a=np.stack(args);return .78*a.max(axis=0)+.22*a.mean(axis=0)
    def texturelevel(t):return ramp(t,.52,.66)
    flooring=area(warm+neutral*.46,ground);indoors=(1-skyTop)*(.70+.30*(1-vegetation)); asphalt=road*(.54+.46*plainLow); sky=skyTop*(.52+.48*plainTop); trees=ramp(vegetation,.26,.40); grasses=ramp(field,.30,.50); brown=ramp(rich,.30,.47); blotched=ramp(speck,.21,.33); muted=1-ramp(jewel,.27,.43); bare=area(soil,flank); animalface=eyes*nose
    air=either(conj(plainTop,plainLow,uniform,compact,diagonal),conj(sky,skySide,plainLow,compact),conj(sky,diagonal,broad,1-ripples),conj(plainTop,cutout,diagonal))
    auto=either(conj(road,planar,wheels,deep),conj(road,paint,wheels),conj(studio,deep,broad,wheels),conj(planar,deep,wheels,1-sky))
    avian=either(conj(plainSide,compact,tall),conj(trees,compact,leafcolor),conj(uniform,tall,1-warmbody),conj(plainSide,waterLow,compact))
    feline=either(conj(indoors,animalface,blotched),conj(indoors,brown,rough,1-planar),conj(indoors,eyes,muted,compact))
    stag=either(conj(trees,muted,brown),conj(grasses,brown,finelegs,plainSide),conj(bare,muted,brown,1-planar),conj(trees,muted,tall))
    canine=either(conj(indoors,nose,bright,darkcoat),conj(grasses,nose,compact),conj(indoors,compact,brown,nose),conj(indoors,darkcoat,bright,1-planar))
    amphib=either(conj(moss,rough,bgrough),conj(rich,bare,blotched,1-planar),conj(moss,compact,jewel),conj(bare,blotched,rich,1-sky))
    equine=either(conj(grasses,finelegs,darkcoat),conj(grasses,sky,brown),conj(brown,finelegs,1-blotched,field),conj(grasses,muted,deep,plainLow))
    boat=either(conj(waterLow,sky,horizstrip),conj(separated,stacked,plainLow),conj(waterLow,plainLow,broad,skyTop))
    lorry=either(conj(sky,planar,upright),conj(road,white,planar,upright),conj(road,wheels,sky,paint),conj(white,rect,wheels,stacked))
    habitats=np.stack([air,auto,avian,feline,stag,canine,amphib,equine,boat,lorry],axis=1)
    # Weak absolute classifier plus explicit scene alternatives and independent
    # evidence for the four broad families. Each family is scored once.
    old=np.stack(context,axis=1); group=np.zeros_like(old)
    aero=1.9*sky+.91*waterLow+.80*uniform+.75*plainLow+.65*plainTop-1.20*vegetation-.55*rough+.36*compact
    motor=1.32*planar+.80*straight+.85*road+.70*wheels+.55*deep-.84*(fur+vegetation)-.42*plainLow
    pet=1.05*indoors+.60*rough+.55*animalface+.47*bgrough+.49*brown-.95*planar-.38*vegetation+.34*darkcoat+.33*bright
    wild=1.71*vegetation+1.04*field+.72*bare+.52*brown-.80*planar-.30*hue+.34*plainSide
    group[:,[0,8]]=(aero-.75)[:,None];group[:,[1,9]]=(motor-.85)[:,None];group[:,[3,5]]=(pet-.90)[:,None];group[:,[4,7]]=(wild-.55)[:,None];group[:,2]=(.55*wild+.45*aero+.45*compact-.76);group[:,6]=.48*wild+.52*pet+.52*moss+.44*blotched-.93
    return .60*old+2.65*np.log(habitats)+.60*group

# Paired tyres: dark disks below a body at a consistent scale and baseline,
# with a luminance recovery immediately underneath both disks.
def wheelcue(y):
    n=len(y);best=np.zeros(n,dtype=np.float32); small=blur(y)
    for r in [2,3]:
        sm=small if r==2 else blur(small);large=boxmean(y,2*r); dark=high(large-sm);p=np.pad(sm,((0,0),(5,5),(5,5)),mode="edge");low=p[:,5+r+1:37+r+1,5:37];left=p[:,5:37,5-r:37-r];right=p[:,5:37,5+r:37+r];up=p[:,5-r-1:37-r-1,5:37];ring=.35*high(low-sm)+.20*(high(left-sm)+high(right-sm))+.25*high(up-sm);wheel=(.48*ring+.52*dark)*(.35+.65*ramp(.40-sm,0,.38));
        for delta in [7,12,17,21]:
            for tilt in [-3,0,3]:
                w=wheel[:,14:28,:32-delta];v=wheel[:,14+tilt:28+tilt,delta:];pair=np.minimum(w,v);best=np.maximum(best,pair.max(axis=(1,2)))
    return ramp(best,.095,.145)

def legcue(y):
    sm=blur(y); p=np.pad(sm,((0,0),(0,0),(3,3)),mode="edge");v=np.minimum(p[:,:,:32]-sm,p[:,:,6:]-sm);v=(v[:,16:26]+v[:,17:27]+v[:,18:28]+v[:,19:29]+v[:,20:30])*.2
    # Need at least two slender strokes, not just a single branch.
    ans=np.zeros(len(y),dtype=np.float32)
    for d in [4,7,11,15]:ans=np.maximum(ans,np.minimum(v[:,:,:32-d],v[:,:,d:]).max(axis=(1,2)))
    return ramp(ans,.080,.12)

def facecue(y):
    sm=blur(y); dark=high(boxmean(sm,2)-sm);ridge=high(boxmean(sm,4)-sm);eyes=np.zeros(len(y),dtype=np.float32);nose=eyes.copy()
    for delta in [4,7,10,13]:
        for t in [-1,1]:
            a=dark[:,6:18,3:28-delta];b=dark[:,6+t:18+t,3+delta:28];pair=np.minimum(a,b);below=ridge[:,6+delta//2:18+delta//2,3+delta//2:28-delta+delta//2];eyes=np.maximum(eyes,pair.max(axis=(1,2)));nose=np.maximum(nose,(np.minimum(1.7*pair,below)).max(axis=(1,2)))
    return ramp(eyes,.071,.123),ramp(nose,.094,.155)

def predict(x):
    n=len(x);label=np.empty(n,dtype=np.int64)
    for start in range(0,n,16):
        a=x[start:start+16]; count=len(a)
        # Use a fixed number of lanes even for the last tile: matrix kernels
        # and accumulation order are independent of input batch boundaries.
        if count<16:a=np.concatenate([a,np.zeros((16-count,3,32,32),dtype=np.float32)],axis=0)
        fg=foreground(a); form,rich=modelmatch(a,fg); ctx=sceneprops(a,fg)
        world=worldmatch(a);joint=jointmatch(a,fg)
        label[start:start+count]=(7.8*form+4.3*rich+.74*ctx+.42*world+10.0*joint).argmax(axis=1)[:count]
    return label
