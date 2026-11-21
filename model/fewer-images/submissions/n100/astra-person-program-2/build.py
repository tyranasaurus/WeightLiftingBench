"""Emit a data-independent person detector. All coordinates are anatomical ratios."""
import sys
from pathlib import Path
import numpy as np


def make_templates():
    # Each sheet is one drawing, in head-height units. No pooling across drawings.
    n = 128
    hist = np.arange(n, dtype=float) + 0.5
    y, x = np.meshgrid(hist / (n/4)-1.5, hist / (n/4)-2, indexing='ij')
    result = []
    for tilt, headw, view, shoulder, hat in [
        (0, .68, 0, 1.72, 0), (0, .81, 0, 2.12, 0),
        (0, .52, 0, 1.4, 0), (0, .71, .22, 1.46, 0),
        (0, .69, -.22, 1.46, 0), (.19, .70, 0, 1.83, 0),
        (-.19, .70, 0, 1.83, 0), (0, .69, 0, 1.8, 1),
        (.08, .71, .22, 1.32, 1), (-.08, .71, -.22, 1.32, 1),
        (0, .94, 0, 2.38, 0), (0, .57, 0, 1.56, 1)]:
        xx = x - tilt*(y-.25)
        # egg shape, lower half narrower towards the jaw
        hr = headw*.5*(1-.20*np.clip(2*y,0,1))
        head = (xx/hr)**2+((y+.01)/.52)**2 < 1
        neck = (np.abs(xx + .16*view)<.22*headw) & (y>.26) & (y<.77)
        brim = (np.abs(xx-.35*view)<.58*headw) & (y>-.34) & (y<-.27)
        side = xx + view * np.clip(y-.35, 0, 1)
        # sloping clavicles, rounded shoulders, extended upper arms
        top = .58+np.maximum(0,np.abs(side)-.16)*.36 + np.maximum(0,np.abs(side)-.36*shoulder)*.9
        half = shoulder*.50 + .11*np.clip(y-1,0,1)
        body = (y>top) & (np.abs(side)<half)
        shape = head | neck | body
        if hat: shape |= brim
        def smooth(f):
            for j in range(4):
                pp = np.pad(f,((1,1),(1,1)),mode='edge')
                f = (4*pp[1:-1,1:-1]+2*(pp[1:-1,:-2]+pp[1:-1,2:]+pp[2:,1:-1]+pp[:-2,1:-1])+pp[:-2,:-2]+pp[:-2,2:]+pp[2:,:-2]+pp[2:,2:])/16
            return f
        a = smooth(shape.astype(float))
        gy,gx=np.gradient(a)
        mag = np.hypot(gx,gy)
        ang = np.arctan2(gy,gx)
        centers = np.arange(8)*.5+.25
        def tent(v):
            return np.maximum(0,1-np.abs(v))
        cells=[]
        for cy in centers-1.5:
            for cx in centers-2:
                w = tent((x-cx)*2)*tent((y-cy)*2)*mag
                cells.append([np.sum(w*np.cos(ang-u*np.pi/6)**8) for u in range(6)])
        cell = np.array(cells).reshape(8,8,6)
        # Contrast the characteristic contours against the immediately surrounding cells.
        surround = cell.sum(2,keepdims=True)
        near = np.pad(surround,((1,1),(1,1),(0,0)),mode='edge')
        near = sum(near[dy:dy+8,dx:dx+8] for dy in range(3) for dx in range(3))/9
        region = np.exp(-((centers[:,None]-1.68)/1.6)**4-((centers[None,:]-2)/1.68)**4)[:,:,None]
        filt = (cell - .55*surround/6 - .45*near/6)*region
        result.append(filt)
        # Self-normalization: a unit response is agreement with this template's peaks.
        filt /= np.sum(np.max(filt,axis=2)) + 1.e-10
    return np.ascontiguousarray(np.stack(result, axis=-1).reshape(384,-1), dtype=np.float32)


def make_figures():
    # Upright, striding, crouching and side-on figures, drawn as individual contour filters.
    q = 20
    y,x = np.meshgrid((np.arange(q*9)+.5)/q-1.5,(np.arange(q*7)+.5)/q-3.5,indexing='ij')
    def limb(ax,ay,bx,by,r,s):
        t = np.clip(((x-ax)*(bx-ax)+(y-ay)*(by-ay))/max(.001,(bx-ax)**2+(by-ay)**2),0,1)
        return (x-ax-t*(bx-ax))**2+(y-ay-t*(by-ay))**2 < (r+(s-r)*t)**2
    filters=[]
    for pose in ['straight','walking','running','apart','narrow','reaching','low','leaning']:
        tilt = {'running':.20,'walking':.06,'leaning':.25,'low':.19}.get(pose,0)
        side = pose in ('narrow','running','walking')
        shw = .50 if side else .82
        hipw = .38 if side else .52
        hipy = 2.80 if pose=='low' else 3.18
        hipx = -tilt*hipy
        body = ((x/.36)**2+(y/.52)**2 < 1) | limb(-.04,.30,-.10,.79,.17,.20)
        span=np.interp(y,[.62,.87,1.2,hipy-.46,hipy],[.29,shw*.90,shw,.77*hipw,hipw])
        body |= (y>.62)&(y<hipy)&(np.abs(x+tilt*y)<span)
        for d in [-1,1]:
            a,b = d*shw*.89-tilt,.96
            ex,ey=a+d*.28,1.96
            wx,wy=ex-.12*d,2.98
            kx,ky=hipx+d*.29,4.66
            fx,fy=kx+d*.10,6.10
            if pose=='walking':
                ex,wx=a-d*.52,a-d*.7
                kx,fx=hipx+d*.80,hipx+d*1.53-.18
            if pose in ('running','low'):
                ex,wx=a-d*.83,a-d*1.37
                ey,wy=1.68,1.36
                kx,ky=hipx+d*(.92 if d>0 else .72),hipy+(.74 if d>0 else 1.05)
                fx,fy=kx-(.28 if d>0 else 1.23),ky+(1.36 if d>0 else .47)
            if pose=='apart' or pose=='leaning':
                ex,wx=a+d*.50,a+d*.93
                ey,wy=1.77,2.19
                kx,fx=hipx+d*.67, hipx+d*1.18
            if pose=='reaching':
                ex,ey=a+d*.88,.50
                wx,wy=ex+d*.59,-.22
                kx,fx=hipx+d*.59,hipx+d*.91
            if pose=='narrow':
                ex,wx=a+.29,a+.49
                kx,fx=hipx+d*.24,hipx+d*.26
            body |= limb(a,b,ex,ey,.23,.17) | limb(ex,ey,wx,wy,.17,.14)
            body |= limb(hipx+d*hipw*.61,hipy-.06,kx,ky,.30,.23)
            body |= limb(kx,ky,fx,fy,.23,.14) | limb(fx,fy,fx+.19,fy+.04,.16,.12)
        for width in (.82,1.16):
            warped=body[:,np.clip(np.round((np.arange(7*q)-(7*q-1)*.5)/width+(7*q-1)*.5).astype(int),0,7*q-1)].astype(float)
            for j in range(3):
                p=np.pad(warped,1,mode='edge');warped=(2*warped+p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])/6
            gy,gx=np.gradient(warped)
            angle=np.arctan2(gy,gx);mag=np.hypot(gy,gx)
            channels=np.stack([np.cos(angle-n*np.pi/6)**8*mag for n in range(6)],axis=-1)
            hy=np.maximum(0,1-np.abs(y[:,0,None]-np.arange(-1,8)))
            hx=np.maximum(0,1-np.abs(x[0,:,None]-np.arange(-3,4)))
            f=np.einsum('ya,xb,yxc->abc',hy,hx,channels,optimize=True)
            # Center versus surrounding edge channels, all computed on this single silhouette.
            den=f.sum(2,keepdims=True)
            local=np.pad(den,((1,1),(1,1),(0,0)),mode='edge')
            local=sum(local[i:i+9,j:j+7] for i in range(3) for j in range(3))/9
            f-= (.42*den+.58*local)/6
            f/=np.max(f,axis=2).sum()+1.e-10
            filters.append(f)
            if tilt or pose=='reaching':
                filters.append(f[:,::-1,:][:,:,[0,5,4,3,2,1]])
    return np.ascontiguousarray(np.stack(filters,axis=-1).reshape(378,-1),dtype=np.float32)


SOURCE = '''import numpy as np
import math


def sat(v):
    return np.minimum(1, np.maximum(0, v))


def shift(a, dy, dx):
    h,w = a.shape[:2]
    yp = np.arange(h) + dy
    xp = np.arange(w) + dx
    return a[np.clip(yp,0,h-1)[:,None],np.clip(xp,0,w-1)]


def integ(a):
    return np.pad(a.cumsum(axis=0,dtype=np.float64).cumsum(axis=1), ((1,0),(1,0)))


def rect(ii, yi, xi, hy, wx):
    h,w = ii.shape
    t=np.clip(yi - hy//2, 0,h-2)
    b=np.clip(yi + (hy+1)//2,1,h-1)
    l=np.clip(xi - wx//2,0,w-2)
    r=np.clip(xi + (wx+1)//2,1,w-1)
    return ((ii[b[:,None],r]-ii[t[:,None],r]-ii[b[:,None],l]+ii[t[:,None],l]) / ((b-t)[:,None]*(r-l))).astype(np.float32)


def pbox(ii, ys, xs, hh, ww):
    t=np.maximum(0,np.minimum(95,ys-hh//2)); b=np.minimum(96,np.maximum(1,ys+(hh+1)//2))
    l=np.maximum(0,np.minimum(95,xs-ww//2)); r=np.minimum(96,np.maximum(1,xs+(ww+1)//2))
    return (ii[b,r]-ii[t,r]-ii[b,l]+ii[t,l])/((b-t)*(r-l))


def section_agreement(feats, pos, h, shape, full):
    # Confirm contour proposals using the order of colors across successive body sections.
    # Work on local maxima so that only a bounded number of fine verifications is needed.
    score=shape+.60*full
    step=int(pos[1]-pos[0]); radius=max(1,round(h*.60/step))
    indices=[]
    for i in range(28):
        j=np.argmax(score)
        yy,xx=divmod(int(j),len(pos))
        indices.append((pos[yy],pos[xx]))
        score[max(0,yy-radius):yy+radius+1,max(0,xx-radius):xx+radius+1]=-10
    coords=np.array(indices,np.int32)
    ys=coords[:,0];xs=coords[:,1]
    def sample(cy,cx,hh,ww):
        return np.stack([pbox(channel,ys+np.rint(cy*h).astype(np.int32),xs+np.rint(cx*h).astype(np.int32),max(1,round(hh*h)),max(1,round(ww*h))) for channel in feats])
    def delta(a,b):
        return np.sqrt(((a-b)**2).sum(0))
    def section(cy,cx,inX,outX,ht,wd):
        mid=sample(cy,cx,ht,wd)
        il=sample(cy,cx-inX,ht,wd);ir=sample(cy,cx+inX,ht,wd)
        ol=sample(cy,cx-outX,ht,wd);orr=sample(cy,cx+outX,ht,wd)
        dl=delta(il,ol);dr=delta(ir,orr)
        within=delta(il,mid)+delta(ir,mid)
        con=.65*np.minimum(dl,dr)+.35*np.maximum(dl,dr)
        quality=con/(.055+con+within)
        return quality,mid,(ol+orr)*.5
    head,m,bg=section(-.04,0,.15,.57,.60,.23)
    top=sample(-.74,0,.28,.43)
    cap=delta(m,top)
    upper=head*(.50+.5*cap/(cap+.075))
    best=np.zeros(len(coords))
    for lean in [-.17,0,.17]:
        a,sh,sbg=section(1.26,lean,.37,1.14,.62,.34)
        b,tb,tbg=section(2.27,lean*1.8,.32,1.11,.59,.33)
        c,lb,lbg=section(4.18,lean*2.8,.21,1.08,.99,.30)
        uniform=1/(1+delta(sh,tb)/.13)
        broad=a*b*(.55+.45*uniform)
        leg=(.45+.55*c)*sat((95-ys-2.80*h)/(1.5*h))
        coherence=1/(1+.5*(delta(bg,sbg)+delta(sbg,tbg))/.19)
        evidence=(upper*broad*leg)**.25*(.75+.25*coherence)
        best=np.maximum(best,evidence)
    values=np.zeros_like(shape)
    values[ys//step,xs//step]=best
    return values


def color_features(photo):
    r,g,b=photo
    lum = .299*r+.587*g+.114*b
    # Gently suppress a strong illuminant, without imposing gray-world on every photo.
    u0 = r-g
    v0 = g-b
    level = lum.mean()+.18
    u = u0 - np.clip(u0.mean()/level,-.12,.17)*.47*(lum+.10)
    v = v0 - np.clip(v0.mean()/level,-.14,.20)*.65*(lum+.10)
    # Projection along, and normal to, the orange-red locus. Widen for brighter pixels.
    p = (u+.66*v)/1.20
    cross = (v-.52*u)
    strength = sat((p-.017)/(.033+.052*lum))
    strength *= sat((.39+.32*lum-p)/.14)
    hue = 1/(1+(cross/(.024+.078*lum+.22*np.maximum(p,0)))**2)**1.45
    # Reject purples and greenish-yellow. At very low chroma abstain rather than match gray.
    skin = strength*hue*sat((u+.006)/(.022+.025*lum))*sat((lum-.045)/.13)
    dx = (shift(photo.transpose(1,2,0),0,1)-shift(photo.transpose(1,2,0),0,-1))*.5
    dy = (shift(photo.transpose(1,2,0),1,0)-shift(photo.transpose(1,2,0),-1,0))*.5
    k = (dx*dx+dy*dy).argmax(axis=2)
    gx = np.take_along_axis(dx,k[:,:,None],axis=2)[:,:,0]
    gy = np.take_along_axis(dy,k[:,:,None],axis=2)[:,:,0]
    mag = np.hypot(gx,gy)
    return lum,skin,gx,gy,mag


def edge_channels(gx,gy,mag):
    ang = np.arctan2(gy,gx)*(6/np.pi)
    base=np.floor(ang).astype(np.int32)
    frac = ang-base
    base %= 6
    result = np.stack([mag*((base==q)*(1-frac)+(base==(q-1)%6)*frac) for q in range(6)],axis=2).astype(np.float32)
    return result


def hog_map(edges,c):
    # Soft cell pooling on a half-head grid. A head may lie on the image margin.
    size=96//c
    pos=np.arange(size,dtype=np.int32)*c+c//2
    # Binomial-like triangular integration via the sum of four shifted boxes.
    ii=np.pad(edges.cumsum(0,dtype=np.float64).cumsum(1),((1,0),(1,0),(0,0)))
    hist=np.zeros((size,size,6),np.float32)
    for dy in [-1,1]:
        for dx in [-1,1]:
            lowY= np.maximum(0, pos-c//2+dy*(c//3))
            highY=np.minimum(96,pos+(c+1)//2+dy*(c//3))
            lowX= np.maximum(0, pos-c//2+dx*(c//3))
            highX=np.minimum(96,pos+(c+1)//2+dx*(c//3))
            hist += ((ii[highY[:,None],highX]-ii[highY[:,None],lowX]-ii[lowY[:,None],highX]+ii[lowY[:,None],lowX]) / ((highY-lowY)[:,None,None]*(highX-lowX)[None,:,None])).astype(np.float32)*.25
    energy = np.sum(hist*hist,axis=2)
    enpad = np.pad(energy,((2,2),(2,2)),mode='edge')
    norm=0
    for dy in range(5):
        for dx in range(5):
            norm=norm+enpad[dy:dy+size,dx:dx+size]
    hist /= np.sqrt(.00015+norm*.04)[:,:,None]
    np.minimum(hist,1.7,out=hist)
    # A patch centered at its head, looking further down than up.
    pad = np.pad(hist,((2,5),(3,4),(0,0)),mode='edge')
    patches = np.stack([pad[j:j+size,i:i+size] for j in range(8) for i in range(8)],axis=2).reshape(size*size,384)
    match = (patches @ CONSTANTS['shapes']).max(1).reshape(size,size)
    # The intersection of the four central cells, not the center of a cell.
    anchors=pos + (c+1)//2
    if 3<=c<=12:
        # The coarser, full-figure contour spans roughly seven head heights.
        pad = np.pad(hist,((1,7),(3,3),(0,0)),mode='edge')
        patches=np.stack([pad[j:j+size,i:i+size] for j in range(9) for i in range(7)],axis=2).reshape(size*size,378)
        figure=(patches @ CONSTANTS['figures']).max(1).reshape(size,size)
    else:
        figure=None
    return match,anchors,figure


def interpgrid(values, pos, first, spacing):
    coord=(pos-first)/spacing
    ix=np.floor(coord).astype(np.int32)
    t=(coord-ix).astype(np.float32)
    j=np.clip(ix+1,0,len(values)-1)
    i=np.clip(ix,0,len(values)-1)
    cols=values[:,i]*(1-t)+values[:,j]*t
    bilinear=cols[i]*(1-t[:,None])+cols[j]*t[:,None]
    near=np.where(t>.5,j,i)
    return .5*(bilinear+values[near[:,None],near])


def distance(a,b):
    d=a-b
    return np.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])


def geometric(a,b):
    return np.sqrt(a*b)


def components(mask):
    # Run-length connected components on a binary grid; no fitted segmentation model.
    padded=np.pad(mask,((0,0),(1,1)),mode='constant')
    dy,dx=np.where(padded[:,1:]!=padded[:,:-1])
    n=len(dx)//2
    if n==0:
        return []
    rows=dy[::2]; starts=dx[::2]; ends=dx[1::2]
    parent=list(range(n))
    def root(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]]
            k=parent[k]
        return k
    last=0
    for i in range(n):
        while last<i and rows[last]<rows[i]-1:
            last+=1
        j=last
        while j<i and rows[j]!=rows[i]:
            if starts[j]<=ends[i] and ends[j]>=starts[i]:
                parent[root(j)]=root(i)
            j+=1
    roots=np.array([root(i) for i in range(n)],np.int32)
    sizes=np.bincount(roots,weights=ends-starts,minlength=n)
    l=np.full(n,96); r=np.zeros(n,dtype=np.int32)
    t=l.copy(); b=r.copy()
    np.minimum.at(l,roots,starts);np.maximum.at(r,roots,ends)
    np.minimum.at(t,roots,rows);np.maximum.at(b,roots,rows+1)
    return [(int(l[i]),int(r[i]),int(t[i]),int(b[i]),sizes[i]) for i in np.flatnonzero(sizes>=2)]


def constellation(skin,lum,photo):
    # Exposed cheek, hands and limbs often survive shrinking as distinct, similarly
    # colored islands. Rate whole regions, not merely a high-scoring subwindow.
    def reduce2(x):
        return x.reshape(48,2,48,2).mean(axis=(1,3))
    skin=reduce2(skin)
    smooth=(4*skin+shift(skin,1,0)+shift(skin,-1,0)+shift(skin,0,-1)+shift(skin,0,1))*.125
    mask=(smooth>.54)&(skin>.43)
    lum=reduce2(lum)
    opponent=reduce2(photo[0]-photo[1])
    yellow=reduce2(photo[1]-photo[2])
    groups=components(mask)
    spots=[]
    for l,r,t,b,area in groups:
        w=r-l; h=b-t; fill=area/(w*h)
        if fill<.20 or max(h,w)>39:
            continue
        block=(slice(t,b),slice(l,r))
        fg=mask[block]
        density=float(smooth[block][fg].mean())
        sx=.5*(l+r);sy=.5*(t+b)
        ratio=w/h
        aspect=math.exp(-.68*(math.log(ratio/.73))**2)
        compact=float(sat((fill-.28)/.44))
        # Solid rectangles with an axis-aligned outline are a poor match for an oval.
        rectangular=float(sat((fill-.85)/.15))*min(1.,w/5,h/5)
        quality=aspect*compact*(1-.34*rectangular)
        ownSkin=float(sat((density-.48)/.36))
        spots.append((sx,sy,w,h,quality,ownSkin,opponent[block][fg].mean(),yellow[block][fg].mean(),area))
    best=0.
    for xx,yy,w,h,face,skinfit,u,v,area in spots:
        if h<2 or w<2 or h>35 or yy>41 or face<.16:
            continue
        head=max(float(h),w*1.14)
        if yy+head>49:
            continue
        # Probe cap/hair, shirt, and how exposed skin ends at the collar.
        def patch(a,cx,cy,ww,hh):
            x0=max(0,min(47,round(cx-.5*ww))); x1=max(x0+1,min(48,round(cx+.5*ww)))
            y0=max(0,min(47,round(cy-.5*hh))); y1=max(y0+1,min(48,round(cy+.5*hh)))
            return a[y0:y1,x0:x1].mean()
        fore=patch(lum,xx,yy,w*.58,h*.64)
        hair=patch(lum,xx,yy-head*.58,w*.76,head*.34)
        shirt=patch(lum,xx,yy+head*1.20,w*1.35,head*.65)
        chestSkin=patch(smooth,xx,yy+head*1.20,w*1.35,head*.65)
        hairfit=float(sat((fore-hair+.012)/.14))
        clothing=float(sat(np.abs(fore-shirt)/.19+.37*(1-chestSkin)))
        left=0.;right=0.
        for ax,ay,aw,ah,af,askin,au,av,ar in spots:
            vertical=(ay-yy)/head; lateral=(ax-xx)/head
            if vertical<.14 or vertical>7 or abs(lateral)>3.6 or ar<.04*area or ar>3.8*area:
                continue
            color=math.exp(-((au-u)/.092)**2-((av-v)/.092)**2)
            size=math.exp(-.33*math.log(ar/area/.53)**2)
            arm=math.exp(-.46*((vertical-2.20)/1.34)**2-.54*((abs(lateral)-1.05)/.77)**2)
            leg=.76*math.exp(-.50*((vertical-4.85)/1.06)**2-.65*((abs(lateral)-.55)/.69)**2)
            other=color*size*max(arm,leg)*(.57+.43*askin)
            if lateral<0:
                left=max(left,other)
            else:
                right=max(right,other)
        position=float(sat((44-yy)/13))
        quality=face*skinfit*position*(.50+.25*hairfit+.25*clothing)
        pair=max(left,right)+.55*min(left,right)
        best=max(best,quality*(.27+.84*pair))
    return min(1.35,best), float((smooth>.59).mean())


def one(photo):
    lum,skin,gx,gy,mag=color_features(photo)
    chans=edge_channels(gx,gy,mag)
    ii=integ(lum)
    sk=integ(skin)
    sqr=integ(lum*lum)
    rgb = [ii,integ((photo[0]-photo[1])*.72),integ((photo[1]-photo[2])*.62)]
    best = -10.
    peaks=[]
    scales=(2,3,4,6,8,12,16,24)
    maps={cell:hog_map(chans,cell) for cell in scales}
    for cell in scales:
        h = 2*cell
        step=max(1,h//6)
        pos=np.arange(0,96,step)
        def box(table,dy,dx,hh,ww):
            return rect(table,pos+round(dy*h),pos+round(dx*h), max(1,round(hh*h)),max(1,round(ww*h)))
        def pix(dy,dx,hh,ww):
            return np.stack([box(table,dy,dx,hh,ww) for table in rgb])
        def move(v,dy,dx):
            return shift(v,round(dy*h/step),round(dx*h/step))
        # Smooth versus long vertical lines: compare each piece to skin, hair, and sides separately.
        face = pix(.12,0,.53,.47)
        faceSkin = box(sk,.12,0,.65,.47)
        hair = pix(-.31,0,.25,.48)
        head=(face*.65+hair*.35)
        left = pix(-.05,-.61,.66,.31)
        right = pix(-.05,.61,.66,.31)
        crown = pix(-.76,0,.32,.45)
        shirt = pix(1.12,0,.68,.75)
        fvar = np.maximum(0,box(sqr,.12,0,.53,.47)-face[0]*face[0])
        chestvar = np.maximum(0,box(sqr,1.12,0,.68,.75)-shirt[0]*shirt[0])
        bgvar = np.maximum(0,box(sqr,0,0,2.8,2.4)-box(ii,0,0,2.8,2.4)**2)
        eps=.038+.28*np.sqrt(bgvar)+.35*np.sqrt(fvar)
        def contrast(a,b):
            d = distance(a,b)
            return d/(d+eps)
        lcon=contrast(face,left)
        rcon=contrast(face,right)
        tcon=contrast(hair,crown)
        bcon=contrast(face,shirt)
        sides= .70*np.minimum(lcon,rcon)+.30*np.maximum(lcon,rcon)
        hairShade=sat((face[0]-hair[0]+.012)/(.080+.4*np.sqrt(bgvar)))
        skinSurL=box(sk,0,-.72,.89,.50)
        skinSurR=box(sk,0,.72,.89,.50)
        skinAbove=box(sk,-.75,0,.42,.63)
        skinBelow=box(sk,1.05,0,.62,.66)
        surround=.30*(skinSurL+skinSurR)+.17*skinAbove+.23*skinBelow
        skinPeak=sat((faceSkin-.83*surround-.055)/.58)
        skinHit=np.sqrt(skinPeak*faceSkin)
        # Determine whether a compact patch continues as a broader, coherent shirt below.
        shoulderL = pix(.99,-.52,.40,.27)
        shoulderR = pix(.99,.52,.40,.27)
        shirtTop = .5*(contrast(left,shoulderL)+contrast(right,shoulderR))
        shoulderCon = .5*(contrast(shoulderL,shirt)+contrast(shoulderR,shirt))
        shoulders=sat(.5+1.65*(shirtTop-shoulderCon))
        # Pair each side of the torso with background at approximately the same height.
        extL = pix(1.42,-1.35,.70,.4)
        extR = pix(1.42,1.35,.70,.4)
        shL=contrast(shirt,extL)
        shR=contrast(shirt,extR)
        shirtSides=.65*np.minimum(shL,shR)+.35*np.maximum(shL,shR)
        uniformity=1/(1+chestvar/(.003+.55*bgvar))
        body=sat(.41*shoulders+.37*shirtSides+.22*uniformity)
        # Matched head/shoulder gradients are useful even for hair, hats, or grayscale.
        hog,anc,unused=maps[cell]
        edgeShape=sat((interpgrid(hog,pos,anc[0],cell)-.09)/.48)
        # Spatial agreement of head-height highlights, inverted-T and eyes in larger faces.
        eyes = np.zeros_like(faceSkin)
        if h>=6:
            browL=box(ii,-.085,-.163,.14,.19)
            browR=box(ii,-.085,.163,.14,.19)
            cheekL=box(ii,.13,-.17,.18,.18)
            cheekR=box(ii,.13,.17,.18,.18)
            bridge=box(ii,-.02,0,.20,.12)
            forehead=box(ii,-.285,0,.16,.40)
            lEye=.48*cheekL+.33*bridge+.19*forehead-browL
            rEye=.48*cheekR+.33*bridge+.19*forehead-browR
            escale=.032+.62*np.sqrt(fvar)
            eyes=sat((.76*np.minimum(lEye,rEye)+.24*np.maximum(lEye,rEye))/escale)
            eyes*=sat(.60+.68*(bridge-(browR+browL)*.5)/escale)
        # Skin of roughly the same colour on a hand/forearm at a distance of several head widths.
        # Score isolated peripheral warm-colour patches: wood walls are not a pair of hands.
        armBase = faceSkin * sat((faceSkin-.72*surround)/.60)
        arms=np.zeros_like(faceSkin)
        for dy,dx in [(1.52,.92),(2.25,.9),(2.96,.60),(.68,.97)]:
            for sign in [-1,1]:
                alt=move(armBase,dy,dx*sign)
                diff=np.abs(face[1]-move(face[1],dy,dx*sign))+np.abs(face[2]-move(face[2],dy,dx*sign))
                arms=np.maximum(arms,alt/(1+diff*11))
        # A frontal-face channel and a silhouette channel vote independently at each position.
        distinct = .53*sides+.23*tcon+.24*bcon
        headFace=(1.24*skinHit+.33*faceSkin+.48*distinct+.36*body+.36*edgeShape+.16*hairShade+.30*eyes+.26*arms)
        headShape=(.93*edgeShape+.72*distinct+.60*body+.27*shoulders+.28*skinHit+.13*hairShade+.22*eyes)
        # A standing figure often has consistent torso colour and flanking background down its length.
        if h<=16:
            belt=pix(2.12,0,.6,.66)
            trousers=pix(3.55,0,1.0,.54)
            gleft=pix(3.45,-1.15,.96,.42)
            gright=pix(3.45,1.15,.96,.42)
            legCon=geometric(contrast(trousers,gleft),contrast(trousers,gright))
            torsoSim=1-contrast(shirt,belt)
            stand=.62*legCon+.38*torsoSim
            headShape+=.32*stand*sat((96-pos-2*h)/(h*1.5))[:,None]
        # Combine a scale-aligned full-figure contour vote with localized head and torso evidence.
        if h in (4,6,8,12):
            figur=interpgrid(maps[h][2],pos,h//2,h)
            full=sat((figur-.06)/.42)
            quality=.27*distinct+.22*shoulders+.19*stand+.16*edgeShape+.16*skinHit
            coverage=sat((96-pos-3*h)/(2.5*h))[:,None]
            fullScore=(1.05*full+.92*quality-1.40)*1.65-.32*(1-coverage)
        else:
            fullScore=-5
        # A head the size of a few pixels is hard to identify from its interior.
        # Make the highest skin/face score contingent on (a) localized skin and
        # (b) independent confirmation from another body part or the head outline.
        small=math.log(16/h) if h<16 else 0.
        rawSkin=.68*skinHit+.32*faceSkin
        contrastSkin=sat((faceSkin-surround)/.55)
        aFace=sat((rawSkin-.22)/.58)*sat((contrastSkin+.30)/.95)
        verify=sat(.34*edgeShape+.19*arms+.15*eyes+.16*hairShade+.16*shoulders)
        faceScore=(headFace-2.33-.12*small)*1.90-.95*(1-aFace)-.67*(1-verify)
        # For a colorless silhouette, demand a consistent figure as well as a head.
        # Truncated figures can still be recognized by a sharply defined upper body.
        agree=sat((distinct-.37)/.41)*sat((body-.43)/.42)
        outline=sat((edgeShape-.38)/.57)
        shapeScore=(headShape-2.37-.10*small)*1.75-.75*(1-agree)-.58*(1-outline)
        if h<=16:
            shapeScore-=.34
        if h in (4,6,8,12):
            shapeScore-=.43*(1-full)
            fullScore-=.80*(1-outline)+.45*(1-agree)+.32*(1-full)+.19
            sec=section_agreement(rgb,pos,h,edgeShape,full)
            silhouette=(sec-.66)*3.15+.42*(edgeShape-.58)+.33*(full-.61)+.16*(stand-.62)
            fullScore=np.maximum(fullScore,silhouette)
        else:
            shapeScore-=.20*(1-sat(eyes+skinHit))
        # Geometric confirmation of a face without relying on its hue.
        if h>=8:
            mouth=box(ii,.29,0,.115,.28)
            chin=box(ii,.41,0,.10,.22)
            nose=box(ii,.15,0,.14,.16)
            mouthScore=sat((.54*nose+.46*chin-mouth)/(.034+.53*np.sqrt(fvar)))
            symmetry=sat(1-(np.abs(browL-browR)+np.abs(cheekL-cheekR))/(.08+1.0*np.sqrt(bgvar)))
            frontScore=(.58*eyes+.25*mouthScore+.21*symmetry+.37*distinct+.31*body+.34*edgeShape+.32*skinHit+.13*hairShade-1.79)*1.7
            faceDetail=np.minimum(eyes,np.minimum(mouthScore,symmetry))
            frontScore-=.60*(1-faceDetail)+.34*(1-outline)+.15
            shapeScore=np.maximum(shapeScore,frontScore)
        # Reduced coverage in heavily truncated proposals. Never exclude the photo boundary.
        centerX=(pos-47.5)/48
        centerY=(pos-35)/57
        penalty=.10*centerX[None,:]**2+.14*centerY[:,None]**2
        margin=np.minimum(pos,95-pos)
        border = .17*sat((.50*h-margin)/(.50*h))[None,:]+.20*sat((pos+.90*h-96)/h)[:,None]+.09*sat((.45*h-pos)/h)[:,None]
        conf=np.maximum(np.maximum(faceScore,shapeScore),fullScore)-penalty-border
        best=max(best,float(conf.max()))
        peaks.append((conf,step,h))
    # Weak, separated detections also occur in group scenes. Count only spatially distinct peaks.
    top = sorted([(float(a.max()),int(a.argmax()),a.shape[1],step,h) for a,step,h in peaks],reverse=True)[0]
    tx=(top[1]%top[2])*top[3]
    ty=(top[1]//top[2])*top[3]
    second=-10.
    for a,step,h in peaks:
        pos=np.arange(a.shape[0])*step
        far=(pos[None,:]-tx)**2+(pos[:,None]-ty)**2>(.9*max(h,top[4])+5)**2
        second=max(second,float(np.where(far,a,-10.).max()))
    # Global context only breaks ties: architecture has long axial lines, outdoor action
    # frequently places a dark subject against a smooth field, and flash portraits have skin.
    axis=chans[:,:,0]+chans[:,:,3]
    axial=axis.sum()/(.001+mag.sum())
    skyBlue=sat((photo[2]-photo[0]-.03)/.22)*sat((photo[1]-photo[0])/.10)
    turf=sat((photo[1]-photo[0]+.01)/.12)*sat((photo[1]-photo[2]-.02)/.17)
    snow=sat((lum-.68)/.22)*sat(1-(np.maximum(np.abs(photo[1]-photo[0]),np.abs(photo[1]-photo[2])))/.18)
    bottom=slice(40,94)
    field=sat((turf[bottom].mean()-.27)/.47)
    snowfield=sat((snow[bottom].mean()-.38)/.40)
    central=mag[19:82,23:73].mean()
    periph=(mag[:,:18].mean()+mag[:,-18:].mean())*.5
    subject=sat((central-periph)/(.018+periph))
    ctx = -.25*sat((axial-.53)/.19)+.16*subject*(field+snowfield)+.07*subject
    ctx += .09*sat((skyBlue.mean()-.26)/.34)*subject
    group,amount=constellation(skin,lum,photo)
    # Negative scene evidence: long straight boundaries continue well past a person's
    # dimensions. Requiring both orientations separates room grids from a horizon.
    ex=chans[:,:,0];ey=chans[:,:,3]
    vert=np.minimum(np.minimum(ex[9:-9],ex[:-18]),ex[18:])
    horiz=np.minimum(np.minimum(ey[:,9:-9],ey[:,:-18]),ey[:,18:])
    straight=math.sqrt(vert.mean()/(.003+mag.mean())*horiz.mean()/(.003+mag.mean()))
    ctx-=.32*sat((straight-.035)/.075)
    # Grassy fields, snow and water favor action when there is a contrasting subject.
    blueWater=sat((photo[2]-photo[0]-.04)/.19)*sat((photo[1]-photo[0]-.02)/.14)
    aquatic=sat((blueWater[bottom].mean()-.20)/.36)
    ctx+=.24*(aquatic+.6*field+.5*snowfield)*(.38+.62*subject)
    # Some skin is informative, but large expanses of orange/brown are not a face.
    iscolor=sat((np.abs(photo[0]-photo[2]).mean()-.008)/.05)
    ctx+=.21*sat((amount-.008)/.047)-.22*iscolor*sat((.014-amount)/.014)
    ctx-=.21*sat((amount-.33)/.33)
    # Depth and framing: action photos often have a busy center against a smooth periphery.
    ctx+=.11*sat((central-periph)/(.024+periph))
    return .76*best+.17*np.clip(second,-.7,.60)+ctx+.86*(group-.27)


def predict(x):
    return np.array([int(one(photo)>0) for photo in x],dtype=np.int32)
'''

if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'solution.py').write_text(SOURCE)
    np.savez(out / 'constants.npz', shapes=make_templates(), figures=make_figures())
