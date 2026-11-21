import numpy as np
import math

def blur(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode="edge")
    t = p[:,:,:-2] + 2*p[:,:,1:-1] + p[:,:,2:]
    return (t[:,:-2] + 2*t[:,1:-1] + t[:,2:]) * .0625

def deriv(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode="edge")
    dx = (p[:,:-2,2:]-p[:,:-2,:-2] + 2*(p[:,1:-1,2:]-p[:,1:-1,:-2]) + p[:,2:,2:]-p[:,2:,:-2])*.125
    dy = (p[:,2:,:-2]-p[:,:-2,:-2] + 2*(p[:,2:,1:-1]-p[:,:-2,1:-1]) + p[:,2:,2:]-p[:,:-2,2:])*.125
    return dx, dy

def pool(a,k):
    return a.reshape(len(a),32//k,k,32//k,k).mean((2,4))

def norm(a):
    a = a.reshape(len(a),-1)
    return a / np.sqrt(np.sum(a*a,axis=1,keepdims=True)+1.e-8)

def smoothpool(a,k):
    return pool(blur(a),k)

def pos(a):
    return np.maximum(0,a)

def unit(a):
    return np.clip(a,0,1)

def avg(a):
    return a.mean((1,2))

def features(rgb):
    r,g,b = rgb[:,0],rgb[:,1],rgb[:,2]
    l = .30*r+.55*g+.15*b
    dx,dy = deriv(l)
    hh=[]
    ss=[]
    for scale in range(2):
        if scale:
            dx,dy=deriv(blur(blur(l)))
        mag=np.sqrt(dx*dx+dy*dy+1.e-10)
        k=4 if scale==0 else 8
        m=smoothpool(mag,k)
        den=np.sqrt((.018 + .38*m.mean((1,2),keepdims=True) + .65*m)*(.025 + m.mean((1,2),keepdims=True)))
        hh.append(norm((m-m.mean((1,2),keepdims=True))/den)*.45)
        hhpart=[]
        for i in range(6):
            theta=i*math.pi/6
            c=(dx*math.cos(theta)+dy*math.sin(theta))/mag
            q=c*c
            f=smoothpool(q*q*q*mag,k)
            hhpart.append((f-.27*m)/den)
        hh.append(norm(np.stack(hhpart,1))*(1 if scale==0 else .75))
        ss.append(norm(np.stack([smoothpool(dx,k)/den,smoothpool(dy,k)/den],1))*(.70 if scale==0 else .55))
    ll=pool(blur(l),4)
    ss.append(norm(ll-ll.mean((1,2),keepdims=True))*1.05)
    value=np.maximum(np.maximum(r,g),b)
    low=np.minimum(np.minimum(r,g),b)
    # Soft colour categories: foliage, warm earth/fur, blue, vivid red, dark/neutral/light.
    chrom=value-low
    d=.14+.54*value
    gr=unit((g-r+.015)/d*3)*unit((g-b+.025)/d*3)
    earth=unit((r-b)/d*2.5)*unit((r-g+.045)/d*3)
    blue=unit((b-r+.015)/d*3)*unit((b-g+.055)/d*3)
    red=unit((r-g)/d*2.5)*unit((r-b)/d*2)
    gold=unit((g-b)/d*2.8)*unit((r-b)/d*2.8)
    neutral=np.exp(-2.2*chrom/d)
    colours=[gr,earth,blue,red,gold,neutral*unit((.52-l)*3),neutral*unit((l-.40)*2.7),unit((chrom-.12)*2.8)]
    cc=[]
    for v in colours:
        cc.append(pool(v,8)-.18)
    cc.append(pool(neutral,8)-.5)
    cc.append(.52*(pool(l,8)-.5))
    return norm(np.concatenate(hh,1)), norm(np.concatenate(ss,1)), norm(np.stack(cc,1)), (l,dx,dy,gr,earth,blue,red,gold,neutral)

def partIndices(region):
    row,col = region
    indexes=np.arange(64).reshape(8,8)[row,col].reshape(-1)
    small=np.arange(16).reshape(4,4)[row.start//2:(row.stop+1)//2,col.start//2:(col.stop+1)//2].reshape(-1)
    hi=np.concatenate([indexes]+[64+64*i+indexes for i in range(6)]+[448+small]+[464+i*16+small for i in range(6)])
    si=np.concatenate([indexes,64+indexes,128+small,144+small,160+indexes])
    return hi,si

def focusFeatures(h,s,hi,si):
    return np.concatenate([.76*norm(h[:,hi]),.65*norm(s[:,si])],1)

def focusMatch(h,s,indices,cols):
    # Parts can occupy different poses from the rest of the matching view.
    descriptor=focusFeatures(h,s,indices[0],indices[1])
    scores=[]
    for j in range(10):
        corr=descriptor@cols[j].T
        best=np.partition(corr,-4,axis=1)[:,-4:]
        scores.append(.30*best.max(1)+.70*best.mean(1))
    return np.stack(scores,1)

regions=[partIndices(r) for r in [(slice(2,6),slice(1,7)),(slice(1,5),slice(2,6)),(slice(4,8),slice(1,7))]]
partbanks=[]
for hi,si in regions:
    bank=[]
    for j in range(10):
        h=CONSTANTS["h"+str(j)][::3];s=CONSTANTS["s"+str(j)][::3]
        bank.append(np.ascontiguousarray(focusFeatures(h,s,hi,si)))
    partbanks.append(bank)

def su(v):
    return np.clip(v,-1,1)

def sceneRules(rgb,maps):
    l,dx,dy,gr,earth,blue,red,gold,neutral=maps
    r,g,b=rgb[:,0],rgb[:,1],rgb[:,2]
    n=len(l)
    cen=(slice(None),slice(7,26),slice(6,26))
    top=(slice(None),slice(0,8),slice(None))
    bot=(slice(None),slice(24,32),slice(None))
    gradx,grady=deriv(l)
    ax=np.abs(gradx);ay=np.abs(grady);edge=np.sqrt(ax*ax+ay*ay)
    # Coherent straight edges at two lengths distinguish structures from fur and foliage.
    vals=[]
    for span in (5,11):
        av=0;ah=0
        for j in range(span):
            ah=ah+grady[:,:,j:j+33-span]
            av=av+gradx[:,j:j+33-span,:]
        eh=avg(np.abs(ah))/span;ev=avg(np.abs(av))/span
        vals.append((eh,ev))
    hx,vx=vals[1];h5,v5=vals[0]
    es=avg(edge); axm=avg(ax);aym=avg(ay)
    rect=su(4.7*((hx+vx+.007)/(axm+aym+.014)-.53))
    vert=su(4.0*((vx+.003)/(axm+.009)-.47))
    hori=su(4.6*((hx+.003)/(aym+.009)-.50))
    direction=su(2.4*(hx-vx)/(hx+vx+.014))
    sky=avg(blue[top]); ocean=avg(blue[bot]); green=avg(gr);fol=avg(gr[bot]); bush=avg(gr[top]); warm=avg(earth[cen]); warmfloor=avg(earth[bot]);
    topEdge=avg(edge[top]);botEdge=avg(edge[bot]);midEdge=avg(edge[cen]);
    smooth=su((.034-es)*34); skyflat=su((.025-topEdge)*45); landflat=su((.030-botEdge)*37)
    bright=avg(l);toplight=avg(l[top]); botlight=avg(l[bot]); mlight=avg(l[cen]); chrom=rgb.max(1)-rgb.min(1)
    bgmean=(avg(l[top])+avg(l[bot]))*.5
    darkfeet=avg(unit((.28-l[:,19:28,:])*5));neutralfloor=avg(neutral[bot]); grey=avg(neutral);colour=avg(chrom);lawn=avg(gold[bot]);
    # The road tends to be neutral underneath a band of dark wheels and chassis.
    road=su(2.6*(neutralfloor-.57)+.70*landflat+.62*(darkfeet-.22)-1.12*fol-.86*ocean)
    # Blue or pale sky and blue-grey water, with a horizontal intensity discontinuity.
    skycue=su(2.7*sky+.76*skyflat+1.6*(toplight-.6)-.54*bush)
    sea=su(2.85*ocean+.59*direction+.29*hori+.30*landflat-.58*fol-.51*warmfloor-.28)
    vegetation=su(2.2*green+1.03*fol+.73*avg(gold)-.75*ocean-.31*sky-.34)
    wool=su(2.9*(warm-.24)+.62*(warmfloor-.20)-.60*sky-.61*ocean)
    dim=su(2.1*(.51-bright));glare=avg(unit((l-.73)*4.5));hiContrast=su(4.0*(l.std((1,2))-.205)); close=su(10*(botEdge+topEdge-.082))
    # Repeated horizontal streaks in water, as distinct from uniform asphalt or grass.
    streak=su((avg(ay[bot])-avg(ax[bot]))/(.014+avg(edge[bot]))*3.1)
    skyboth=su(1.65*(sky+ocean)-.53+ .42*(skyflat+landflat))
    # Bilateral matching of luminance detail for animal portraits and head-on vehicles.
    cenL=l[:,5:27,6:26];rev=cenL[:,:,::-1]; dif=avg(np.abs(cenL-rev)); rms=cenL.std((1,2)); sym=su(1.9*(.94-dif/(.055+rms))-.44)
    eyes=np.zeros(n,np.float32)
    fine=blur(l); broad=blur(blur(blur(fine))); spot=pos(broad-fine-.008)
    nose=unit((broad-fine)*8)
    for spacing in (4,7,10,13):
        p=spot[:,8:19,4:28-spacing]; q=spot[:,8:19,4+spacing:28];z=spacing//2
        bridge=l[:,8:19,4+z:28-spacing+z]-.5*(l[:,8:19,4:28-spacing]+l[:,8:19,4+spacing:28]);ns=nose[:,12:23,4+z:28-spacing+z]
        evidence=np.sqrt(p*q)*unit(bridge*8)*(.55+.45*ns)
        eyes=np.maximum(eyes,evidence.max((1,2)))
    face=su(6.0*(eyes-.08)+.55*sym)
    # Vivid paint is common on cars and truck cabs. Fur rarely has that chroma.
    painted=su(3.5*avg(red[cen])+1.3*avg(blue[cen])-.32)
    centreGreen=avg(gr[cen]); greenFrog=su(2.8*(centreGreen-.17)+.8*(avg(gold[cen])-.26)-.45*sky)
    gray=su(2.5*(grey-.54)); turf=su(2.9*fol+1.4*lawn-.72*warmfloor-.46)
    shadow=su(2.7*(.5*(botlight+toplight)-mlight)+1.3*(darkfeet-.23));whites=su(3.3*(glare-.17))
    man=.46*rect+.24*hori+.22*road-.46*vegetation-.19*wool
    animal=-.44*rect-.20*hori-.28*road+.33*vegetation+.19*wool
    a=.90*skyboth+.46*smooth+.43*skycue-.28*vert-.26*hori-.25*streak+.16*whites-.35*vegetation-.17*wool
    c=man+.52*road+.33*shadow+.34*painted+.19*direction+.26*hiContrast-.33*sea-.15*skyflat
    t=man+.44*vert+.26*road+.25*skycue+.20*whites+.24*painted-.25*sea-.27*smooth
    shipscore=.98*sea+.48*skycue+.51*direction+.42*streak+.23*gray+.15*hori-.32*vegetation-.38*wool
    birdscore=animal+.32*skycue+.29*smooth+.21*vegetation-.15*shadow-.10*face-.16*wool+.14*painted
    catscore=animal+.35*wool+.26*face+.18*sym+.27*close-.44*turf-.26*vegetation-.23*skycue+.15*gray
    deerscore=animal+.47*vegetation+.35*turf+.24*wool-.22*painted-.17*face-.20*hiContrast+.18*bush
    dogscore=animal+.35*wool+.35*face+.13*close-.23*skycue-.25*sea-.18*vegetation+.14*hiContrast
    frogscore=animal+.65*greenFrog+.36*dim+.39*close-.33*skycue-.34*whites-.20*sym+.12*warmfloor-.16*hori
    horsescore=animal+.55*turf+.25*skycue+.22*hiContrast+.26*shadow+.11*wool-.17*greenFrog-.16*close
    return np.stack([a,c,birdscore,catscore,deerscore,dogscore,frogscore,horsescore,shipscore,t],1)

def matchView(h,s,c):
    # Equal numbers of templates per class; match HOG, shading and chroma jointly.
    out=[]
    for j in range(10):
        a=h@CONSTANTS["h"+str(j)].T
        b=s@CONSTANTS["s"+str(j)].T
        d=c@CONSTANTS["c"+str(j)].T
        z=.48*a+.35*b+.17*d
        z=z.reshape(len(h),-1)
        best=np.partition(z,-6,axis=1)[:,-6:]
        geom=np.partition(.67*a+.33*b,-4,axis=1)[:,-4:].mean(1)
        out.append(.26*best.max(1)+.55*best.mean(1)+.19*geom)
    return np.stack(out,1)

def classify(x):
    h,s,c,m=features(x)
    geo=matchView(h,s,c)
    # Mirroring is a new input viewpoint, not another training example.
    hf,sf,cf,mf=features(np.ascontiguousarray(x[:,:,:,::-1]));flip=matchView(hf,sf,cf)
    geo=.70*np.maximum(geo,flip)+.30*np.minimum(geo,flip)
    part=0
    for i,region in enumerate(regions):
        f=focusMatch(h,s,region,partbanks[i]);g=focusMatch(hf,sf,region,partbanks[i])
        part+=np.maximum(f,g)*([.46,.29,.25][i])
    return (14.5*geo+3.5*part+.76*sceneRules(x,m)).argmax(1)

def predict(x):
    # Fixed compute shape also fixes BLAS reduction order on partial batches.
    n=len(x);out=np.empty(n,dtype=np.int64)
    for i in range(0,n,32):
        batch=np.zeros((32,3,32,32),np.float32)
        part=x[i:i+32];batch[:len(part)]=part
        out[i:i+len(part)]=classify(batch)[:len(part)]
    return out
