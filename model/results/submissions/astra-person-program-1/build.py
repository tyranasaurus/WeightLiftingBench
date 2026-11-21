import sys
from pathlib import Path

source = '''import numpy as np


def sat(x):
    return np.clip(x, 0., 1.)


def meanlight(a):
    return a[..., 1] + .25 * a[..., 2] + .125 * a[..., 3]


def metric(a, b):
    # Coordinate order is colour-likelihood, green, red-green, blue-green.
    g = a[..., 1] - b[..., 1]
    r = a[..., 2] - b[..., 2]
    v = a[..., 3] - b[..., 3]
    return np.abs(g + .25*r + .125*v) + .55*np.abs(r) + .45*np.abs(v)


def diff(a, b):
    v = metric(a,b)
    return v/(.115 + v)


def field(image):
    r,g,b = image
    # Mild adaptation; do not assume a coloured object filling the frame is a light source.
    u = (r-g)/(.085+r+g)
    v = (g-b)/(.085+g+b)
    u = u - .55*np.clip(u.mean(), -.065, .095)
    v = v - .55*np.clip(v.mean(), -.085, .10)
    warm = sat((u-.009)/.060)*sat((.435-u)/.12)*sat((.325-v)/.12)
    warm *= sat((1.22*u+.070-v)/.114)*sat((v+.57*u+.072)/.092)
    warm *= sat((r-.042)/.098)
    return np.stack((warm,g,r-g,b-g),axis=-1)


def integ(f):
    s = np.zeros((97,97,f.shape[2]),np.float32)
    s[1:,1:] = np.cumsum(np.cumsum(f,axis=0),axis=1)
    return s


def bounds(c, radius):
    a = np.clip(np.floor(c-radius+.5).astype(np.int32),0,95)
    b = np.clip(np.floor(c+radius+.5).astype(np.int32),a+1,96)
    return a,b


def grid(s, xs, ys, w, h):
    l,r = bounds(xs,w*.5)
    t,b = bounds(ys,h*.5)
    a = s[b[:,None],r]-s[t[:,None],r]-s[b[:,None],l]+s[t[:,None],l]
    return a/((b-t)[:,None]*(r-l))[...,None].astype(np.float32)


def cellpeaks(a):
    # Retain a local candidate in each region instead of flooding the shortlist
    # with slightly displaced copies of one object.
    ny,nx = a.shape
    ky,kx = max(2,(ny+5)//6),max(2,(nx+5)//6)
    ay,ax = (ny+ky-1)//ky,(nx+kx-1)//kx
    buf = np.full((ay*ky,ax*kx), -100.,np.float32)
    buf[:ny,:nx] = a
    loc = buf.reshape(ay,ky,ax,kx).transpose(0,2,1,3).reshape(ay,ax,ky*kx).argmax(axis=2)
    return (np.arange(ay)[:,None]*ky+loc//kx)*nx+np.arange(ax)[None,:]*kx+loc%kx


def proposals(s):
    rows = []
    for w in (2.8, 4.25, 6.4, 9.6, 14.4, 21.6, 32.5, 49., 73.):
        for aspect in (.99, 1.67):
            h = w*aspect
            sx,sy = max(.84,w*.24),max(.86,h*.205)
            xs = np.arange(sx*.5,96.,sx,dtype=np.float32)
            ys = np.arange(sy*.5,96.,sy,dtype=np.float32)
            ny,nx = len(ys),len(xs)
            if nx*ny < 125:
                ix = np.arange(nx*ny)
            else:
                # Grid has a halo: slices below may refer to off-frame context.
                xx = xs[0]+np.arange(-7,nx+7,dtype=np.float32)*sx
                yy = ys[0]+np.arange(-6,ny+11,dtype=np.float32)*sy
                a = grid(s,xx,yy,.52*w,.41*h)
                def at(dx,dy):
                    return a[6+dy:6+dy+ny,7+dx:7+dx+nx]
                c = at(0,0)
                l,r,t = at(-3,0),at(3,0),at(0,-4)
                crown = at(0,-2)
                dl,dr,dt = diff(c,l),diff(c,r),diff(c,t)
                bilateral = .62*np.minimum(dl,dr)+.19*(dl+dr)
                shirt = at(0,6)
                sl,sr = diff(shirt,at(-6,6)),diff(shirt,at(6,6))
                body = .65*np.minimum(sl,sr)+.175*(sl+sr)
                head = .63*bilateral+.25*dt+.12*diff(c,crown)
                skin = c[...,0]
                surround = (.30*(l[...,0]+r[...,0])+.23*t[...,0]+.17*shirt[...,0])
                pink = skin*(1.-.77*surround)
                shape = head+.64*body+.24*diff(shirt,c)+.18*skin+.15*(bilateral-diff(l,r))
                skinhead = .90*pink + .31*skin + .44*head+.31*body
                if w > 8.:
                    face = meanlight(c) - meanlight(at(-1,-1))*.5-meanlight(at(1,-1))*.5
                    face += .45*(meanlight(at(0,-2))-.5*(meanlight(at(-1,-1))+meanlight(at(1,-1))))
                    ix = np.unique(np.concatenate((cellpeaks(skinhead).ravel(),cellpeaks(shape).ravel(),cellpeaks(face+.10*pink).ravel())))
                else:
                    ix = np.union1d(cellpeaks(skinhead),cellpeaks(shape))
            rows.append(np.stack((xs[ix%nx],ys[ix//nx],np.full(len(ix),w,np.float32),np.full(len(ix),h,np.float32))))
    return np.concatenate(rows,axis=1)


def playground(f):
    # Detect field paint as a narrow light ridge, not merely a light/dark
    # step. Integrating collinear responses rejects speckle in grass or trees.
    g=f[:,:,1]
    r=g+f[:,:,2]
    b=g+f[:,:,3]
    y=meanlight(f)
    grass=sat((g-r+.055)/.079)*sat((g-b-.046)/.083)
    dusty=sat((r-b-.055)/.12)*sat((.16-np.abs(g-r+.065))/.105)*sat((g-.16)/.22)
    court=sat((b-r-.035)/.12)*sat((.22-np.abs(b-g))/.14)
    ground=np.maximum(grass,np.maximum(.75*dusty,.67*court))
    cy,cx=np.mgrid[:96,:96]
    def shift(a,nx,ny):
        return a[np.clip(cy+ny,0,95),np.clip(cx+nx,0,95)]
    best=0.
    for ax,ay in ((1,0),(1,1),(2,1),(1,2),(1,-1),(2,-1),(1,-2),(3,1),(3,-1)):
        u=shift(y,-ay*2,ax*2)
        v=shift(y,ay*2,-ax*2)
        strip=sat((y-np.maximum(u,v)-.037)/.15)*sat((y-.30)/.30)
        flank=.5*(shift(ground,-ay*2,ax*2)+shift(ground,ay*2,-ax*2))
        flank*=sat((.15-np.abs(r-b))/.12)
        # Far apart samples must lie on the same marking, with room on
        # both sides. Ignore borders rather than projecting the border value.
        glimpse=strip*flank
        total=sum(shift(glimpse,k*ax,k*ay) for k in (-6,-4,-2,0,2,4,6))/7.
        best=max(best,total[24:84,17:79].max())
    # Food closeups are a source of spurious warm faces: look for an elliptical
    # plate rim below the centre with distinct colours/textures inside it.
    px=np.array([a for a in (32.,48.,64.) for d in range(27)])
    py=np.tile(np.repeat((47.,61.,73.),9),3)
    rx=np.tile(np.repeat((19.,29.,41.),3),9)
    ry=rx*np.tile((.50,.76,1.02),27)
    dish=np.zeros_like(px)
    half=np.zeros_like(px)
    count=np.zeros_like(px)
    interior=sat((r-b-.072)/.14)+.6*sat((g-b-.07)/.12)
    for j in range(28):
        nx=np.cos(2.*np.pi*j/28.)*rx
        ny=np.sin(2.*np.pi*j/28.)*ry
        xx=px+nx
        yy=py+ny
        seen=(xx>1.) & (xx<94.) & (yy>1.) & (yy<94.)
        def sample(a,ratio):
            return a[np.clip(np.rint(py+ny*ratio).astype(np.int32),0,95),np.clip(np.rint(px+nx*ratio).astype(np.int32),0,95)]
        rim=sample(y,1.)
        neutral=sat((.17-np.abs(sample(r,1.)-sample(b,1.)))/.15)
        white=sat((rim-.46)/.30)*neutral
        inside=sample(interior,.67)
        outside=sample(y,1.14)
        same=sat(1.-4.*np.abs(rim-sample(y,.92)))
        edge=sat((rim-outside+.007)/.15)
        dish+=seen*white*(.26*same+.44*edge+.30*sat(inside))
        half+=seen*white
        count+=seen
    food=np.max((dish/np.maximum(count,1.))*sat((count-13.)/12.)*sat((half/np.maximum(count,1.)-.47)/.35))
    return .86*sat((best-.23)/.40)-.96*sat((food-.41)/.29)


def partprior(f,s):
    # Group warm patches by connectivity, without hypothesising head locations.
    # Skin on a face and arm normally leaves a gap over the person's shirt.
    # Unlike a single best template score, large merged masks do not produce
    # many redundant votes here. Connectivity is computed on a coarse lattice.
    k = f[1:95,1:95].reshape(47,2,47,2,4).mean(axis=(1,3))
    hot = k[:,:,0]
    local = ((hot>.44)*2 + (hot>.74)).astype(np.int32)
    pm = np.pad(local,1)
    neighborhood = pm[:-2,1:-1]+pm[2:,1:-1]+pm[1:-1,:-2]+pm[1:-1,2:]
    mask = ((local>0) & (neighborhood>=4)) | (neighborhood>=10)
    flat = np.flatnonzero(mask)
    if len(flat)==0:
        return -.5
    # Union runs within rows, then link adjacent overlapping runs. Work only
    # with the ends of runs in Python, not all the pixels of a component.
    start = np.flatnonzero(mask & ~np.pad(mask[:,:-1],((0,0),(1,0))))
    ids = np.cumsum((mask & ~np.pad(mask[:,:-1],((0,0),(1,0)))).ravel()).reshape(mask.shape)-1
    contact = mask[1:] & mask[:-1]
    pair = np.unique(ids[1:][contact]*len(start)+ids[:-1][contact])
    parent = list(range(len(start)))
    for v in pair:
        a,b = divmod(int(v),len(start))
        while a!=parent[a]:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while b!=parent[b]:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a>b:
            parent[a]=b
        else:
            parent[b]=a
    for i in range(len(start)):
        parent[i] = parent[parent[i]]
    tag = np.asarray(parent)[ids.ravel()[flat]]
    cy,cx = flat//47,flat%47
    n = np.bincount(tag,minlength=len(start))
    active = n>=2
    def av(z):
        return np.bincount(tag,weights=z,minlength=len(start))[active]/n[active]
    def span(z):
        low=np.full(len(start),47)
        high=np.zeros(len(start),np.int32)
        np.minimum.at(low,tag,z)
        np.maximum.at(high,tag,z)
        return (high-low+1)[active]*2.
    if not np.any(active):
        return -.45
    x = av(cx)*2.+2.
    y = av(cy)*2.+2.
    w,h = span(cx),span(cy)
    warm = av(hot.ravel()[flat])
    r,g,b = (k[:,:,j].ravel()[flat] for j in (2,1,3))
    red = av(r/(.10+2*g+r))
    blue = av(-b/(.10+2*g+b))
    def region(dx,dy,width,height):
        l,u = bounds(x+w*dx,np.maximum(.5,width*w*.5))
        t,v = bounds(y+h*dy,np.maximum(.5,height*h*.5))
        return (s[v,u]-s[t,u]-s[v,l]+s[t,l])/((u-l)*(v-t))[:,None]
    fh = region(0,0,.58,.58)
    cap = region(0,-.67,.72,.46)
    shirt = region(0,1.29,1.15,.82)
    chest = region(0,2.23,1.16,.83)
    sides = np.minimum(diff(fh,region(-.88,0,.43,.52)),diff(fh,region(.88,0,.43,.52)))
    torso = np.minimum(diff(shirt,region(-1.65,1.36,.69,.77)),diff(shirt,region(1.65,1.36,.69,.77)))
    hair = sat((meanlight(fh)-meanlight(cap)+.011)/(.105+.15*meanlight(fh)))
    # A match below the face at a plausible distance and with similar
    # hue supports exposed forearms or legs. The body itself need not be warm.
    dx = np.abs(x[:,None]-x)/w[:,None]
    dy = (y-y[:,None])/h[:,None]
    ratio = (n[active]/n[active,None])
    handshape = sat(1.2-.47*np.abs(np.log((w/h)/.47)))
    hand = sat((dy-.12)/.80)*sat((4.8-dy)/1.9)*sat((2.8-dx)/1.2)*sat((dx+.04)/.45)
    hand *= sat(ratio/.16)*sat(1.50-.65*np.abs(np.log(ratio+.001)+.51))
    hand *= sat(1.10-6.*np.abs(red-red[:,None])-4.*np.abs(blue-blue[:,None]))
    hand *= .58+.42*handshape
    support = hand.max(axis=1)
    fill = n[active]*4/(w*h)
    size = sat((w*h-10.)/40.)*sat((64.-w)/23.)
    pos = sat(1.-((x-48.)/62.)**2-((y-37.)/56.)**2)
    aspect = sat(1.10-.80*np.abs(np.log(w/h/.79)))
    shape = aspect*sat((fill-.30)/.36)
    bg = .30*region(-1.4,0,1.,1.)[:,0]+.30*region(1.4,0,1.,1.)[:,0]+.22*region(0,-1.08,1.3,.69)[:,0]+.18*chest[:,0]
    isolate = sat(warm-bg)
    capscore = .68*hair+.32*sat(warm-cap[:,0])
    dressed = sat(.38+torso-.63*diff(shirt,chest)-.33*shirt[:,0])
    evidence = .16*shape+.17*isolate+.13*capscore+.15*support+.10*pos+.09*size+.07*sides+.13*dressed
    evidence -= .09*sat((f[:,:,0].mean()-.13)/.38)+.12*sat((y-65.)/29.)+.12*(1.-size)
    return float(np.clip((evidence.max()-.70)/.16,-1.,1.4))


def silhouette(f,x,y,w,h):
    # A second, achromatic route: directed gradients around an articulated
    # head / shoulders / hips / legs stick figure. On elongated structures,
    # the shoulders and head should not slide off the edges independently.
    ink = np.zeros_like(w)
    for stride in (1,3):
        z = .8*meanlight(f)+.29*f[:,:,2]+.15*f[:,:,3]
        zp = np.pad(z,stride,mode='edge')
        dx=zp[stride:-stride,2*stride:]-zp[stride:-stride,:-2*stride]
        dy=zp[2*stride:,stride:-stride]-zp[:-2*stride,stride:-stride]
        vertical=np.maximum(0.,np.abs(dx)-.52*np.abs(dy))
        horiz=np.maximum(0.,np.abs(dy)-.52*np.abs(dx))
        d1 = np.maximum(dx*dy,0.)/(np.maximum(np.abs(dx),np.abs(dy))+.018)
        d2 = np.maximum(-dx*dy,0.)/(np.maximum(np.abs(dx),np.abs(dy))+.018)
        maps = integ(np.minimum(.42,np.stack((vertical,horiz,d1,d2),axis=-1)))
        def g(dx,dy,ww,hh,j):
            l,r=bounds(x+w*dx,np.maximum(.55,w*ww*.5))
            t,b=bounds(y+h*dy,np.maximum(.55,h*hh*.5))
            return (maps[b,r,j]-maps[t,r,j]-maps[b,l,j]+maps[t,l,j])/((r-l)*(b-t))
        def step(v,base):
            return sat((v-.48*base)/(.045+.85*base))
        # Integrate over adjacent segments of the contour; flat portions
        # of a roof can match the cap but not the temples or sloping deltoids.
        cap=g(0.,-.59,.72,.28,1)
        baseline=g(0.,-.10,2.2,1.6,1)
        lid=step(cap,baseline)
        faceedge=np.minimum(g(-.49,-.13,.35,.55,0),g(.49,-.13,.35,.55,0))
        temple=step(faceedge,g(0.,-.10,2.1,1.65,0))
        best=np.zeros_like(w)
        for lean in (-.48,0.,.48):
            val=np.zeros_like(w)
            for width in (.80,1.16):
                # Upward-outward normals: positive product at the left shoulder,
                # negative at the right (the image y axis points down).
                ls2 = g(-width*.74+lean*.5,.63,.67,.4,2)
                rs2 = g(width*.74+lean*.5,.63,.67,.4,3)
                sb = .5*(g(lean*.5,.70,3.0,1.4,2)+g(lean*.5,.70,3.0,1.4,3))
                shoulder = step(.70*np.minimum(ls2,rs2)+.15*(ls2+rs2),sb)
                bodybase=g(lean*1.4,1.92,3.5,2.75,0)
                lwall=g(lean*1.35-width*1.02,1.60,.36,1.24,0)
                rwall=g(lean*1.35+width*1.02,1.60,.36,1.24,0)
                waistl=g(lean*1.9-width*.85,2.58,.40,.87,0)
                waistr=g(lean*1.9+width*.85,2.58,.40,.87,0)
                wall = step(np.minimum(.6*lwall+.4*waistl,.6*rwall+.4*waistr),bodybase)
                belt=g(lean*1.93,2.62,width*1.52,.59,1)
                beltbase=g(lean*1.9,2.62,3.3,1.7,1)
                # Threadlike posts have vertical sides but neither a taper nor
                # articulated legs. A gap between the legs is allowed to move.
                pant=g(2.35*lean,4.39,2.6,2.0,0)
                pantl=g(2.35*lean-width*.71,4.39,.49,1.58,0)
                pantr=g(2.35*lean+width*.71,4.39,.49,1.58,0)
                legs=step(np.minimum(pantl,pantr),pant)
                score=.22*lid+.18*temple+.22*shoulder+.22*wall+.07*step(belt,beltbase)+.09*legs
                val=np.maximum(val,score)
            best=np.maximum(best,val)
        suit = sat((8.5-w)/4.5) if stride==1 else sat((w-3.)/5.)
        ink=np.maximum(ink,best*(.72+.28*suit))
    return ink


def attention(f,s):
    # Subjects in photos taken courtside or on a slope often appear as
    # compact groups of sharp, differently coloured pixels against the field.
    # The corresponding negative cue is a large empty central sky or a room
    # whose detail is all near the borders. Measure focus at three resolutions.
    lum = meanlight(f)
    dx = np.diff(lum,axis=1,append=lum[:,-1:])
    dy = np.diff(lum,axis=0,append=lum[-1:])
    sharp = np.minimum(np.sqrt(dx*dx+dy*dy),.30)
    strength = sharp.reshape(8,12,8,12).mean(axis=(1,3))
    center = strength[1:7,2:6].mean()
    outer = .5*(strength[:,0:2].mean()+strength[:,6:8].mean())
    focus = (center-outer)/(.045+center+outer)
    # Tiled colour residual at coarse scales. In contrast to just looking
    # for high gradients, insist on contrast on *both* sides and top.
    yy,xx = np.mgrid[:96,:96]
    weight = sat(1.-((xx-47.5)/56.)**2-((yy-45.)/59.)**2)
    contrast = np.zeros((96,96),np.float32)
    for rad in (3,7,14):
        t=np.arange(96)
        a=grid(s,t,t,rad*1.7,rad*1.7)
        la=a[:,np.maximum(0,t-2*rad)]
        ra=a[:,np.minimum(95,t+2*rad)]
        ta=a[np.maximum(0,t-2*rad),:]
        ba=a[np.minimum(95,t+2*rad),:]
        v=np.minimum(metric(a,la),metric(a,ra))
        u=np.minimum(metric(a,ta),metric(a,ba))
        contrast+=np.minimum(.4,.61*v+.39*u)/3.
    inside=(contrast*weight).sum()/(.1+weight.sum())
    outside=(contrast*(1.-weight)).sum()/(.1+(1.-weight).sum())
    isolated = (inside-outside)/(.082+inside+outside)
    # Skin-coloured detail on the subject is more telling than a small
    # warm object hidden in the background. Likewise skin above clothing is
    # more informative than a warm floor below an achromatic ceiling.
    skin = f[:,:,0]
    emphasis = contrast*weight
    sw = (skin*emphasis).sum()/(.5+emphasis.sum())
    background = skin.mean()
    warmsubject = sat((sw-.71*background-.007)/.11)
    # Blue/green cloth and primary-coloured jerseys on a human scale.
    a = f.reshape(12,8,12,8,4).mean(axis=(1,3))
    ac = contrast.reshape(12,8,12,8).mean(axis=(1,3))
    red,blue=a[:,:,2],a[:,:,3]
    hue = np.maximum.reduce((sat((red-.18)/.22)*sat((red+.3*blue-.17)/.20),sat((blue-.07)/.17)*sat((blue-red-.13)/.23),sat((-red-.06)/.14)*sat((blue-red-.04)/.12)))
    garment = hue*sat((ac-.10)/.18)*weight[4::8,4::8]
    colorcloth = (.55*garment[2:9].max()+.45*sat(garment[2:9].mean()/.09))
    return .75*focus+.98*isolated+.68*(warmsubject-.44)+.44*(colorcloth-.33)


def sceneprior(f):
    # Long field markings, snowfields and water are useful in the absence
    # of a visible face. A room/vehicle facade outlined by a grid also has
    # long lines, but they tend to be parallel to the two image axes.
    g = f[:,:,1]
    r = f[:,:,2]+g
    b = f[:,:,3]+g
    lum = g+.25*f[:,:,2]+.125*f[:,:,3]
    dx = np.diff(lum,axis=1,append=lum[:,-1:])
    dy = np.diff(lum,axis=0,append=lum[-1:])
    power = np.abs(dx)+np.abs(dy)
    vert = np.minimum(np.abs(dx),.35)*sat(1.-2.*np.abs(dy)/(.008+np.abs(dx)))
    horiz = np.minimum(np.abs(dy),.35)*sat(1.-2.*np.abs(dx)/(.008+np.abs(dy)))
    def straight(v,axis):
        v = np.swapaxes(v,axis,0)
        pre = np.cumsum(v,axis=0)
        m = (pre[16:]-pre[:-16])/16.
        return (np.maximum(m-.035,0.)**1.5).mean()/(.001+.24*v.mean())
    gridlike = sat(2.8*(straight(vert,0)+straight(horiz,1))-.35)
    gridlike *= sat(8.*(min(vert.mean(),horiz.mean())/(.02+power.mean())-.06))
    smooth = sat(1.-power/.080)
    turf = sat((g-r+.054)/.072)*sat((g-b-.027)/.090)*sat((g-b-.85*(r-g)-.045)/.11)
    greenfield = (turf*smooth)[49:].mean()
    snow = sat((b-.61)/.20)*sat((g-.60)/.20)*sat((.095-np.abs(g-r))/.070)*sat((.10-np.abs(b-g))/.075)
    snow = (snow*smooth)[48:].mean()
    water = sat((b-r-.07)/.18)*sat((g-r-.010)/.12)*sat((.18-np.abs(g-b))/.15)
    water = (water*smooth)[48:].mean()
    sport = max(sat((greenfield-.25)/.37),sat((snow-.34)/.39),sat((water-.27)/.38))
    # Skin high in the frame can be informative; wood floors, furniture,
    # plated food and fur tend to produce broader and lower warm regions.
    spots = f[:,:,0].reshape(8,12,8,12).mean(axis=(1,3))
    warm = sat((spots[1:5,1:7].max()-.19)/.55)*sat((spots[1:5].mean()-spots[6:].mean()+.13)/.25)
    bland = sat((.13-spots[:6].max())/.12)
    return .43*sport - .64*gridlike + .32*(warm-.5)-.35*bland


def photo(image):
    f = field(image)
    s = integ(f)
    x,y,w,h = proposals(s)
    def box(cx,cy,ww,hh):
        l,r = bounds(cx,np.maximum(.50,ww*.5))
        t,b = bounds(cy,np.maximum(.50,hh*.5))
        return (s[b,r]-s[t,r]-s[b,l]+s[t,l])/((b-t)*(r-l))[:,None].astype(np.float32)
    def p(dx,dy,ww=.48,hh=.37):
        return box(x+dx*w,y+dy*h,w*ww,h*hh)
    def ring():
        # A colour boundary running tangent to an oval is stronger evidence
        # than a single pair of regions with different average colours. Use
        # two possible aspect-preserving sizes to absorb grid misalignment.
        best = np.zeros_like(x)
        for rad in (.86,1.17):
            sidesum = np.zeros_like(x)
            top = np.zeros_like(x)
            lower = np.zeros_like(x)
            for j in range(12):
                th = (j+.5)*(np.pi/6.)
                nx,ny = .50*np.cos(th),.52*np.sin(th)
                v = p(rad*.71*nx,-.14+rad*.73*ny,.266,.215)
                bg = p(rad*1.32*nx,-.14+rad*1.32*ny,.27,.215)
                bg2 = p(rad*1.68*nx,-.14+rad*1.68*ny,.27,.215)
                ed = diff(v,bg)
                quality = sat(ed+.38*(ed-diff(bg,bg2))-.08)
                if j<6:
                    lower += quality/6.
                if j>=6:
                    top += quality/6.
                if j in (0,5,6,11):
                    sidesum += .25*quality
            best = np.maximum(best,.40*top+.35*sidesum+.25*lower)
        return best
    contour = ring()
    c = p(0,0,.58,.41)
    skin = sat(c[:,0])
    l,r,t = p(-.84,-.08),p(.84,-.08),p(0,-.85,.64,.35)
    hair = p(0,-.44,.54,.29)
    cheekl,cheekr = p(-.23,.005,.28,.23),p(.23,.005,.28,.23)
    jaw,neck = p(0,.21,.43,.22),p(0,.51,.38,.27)
    # Distinct sides and cap; a simple step edge changes only one side.
    dl,dr,dt = diff(cheekl,l),diff(cheekr,r),diff(hair,t)
    def both(a,b):
        return .74*np.minimum(a,b)+.13*(a+b)
    sides = both(dl,dr)
    across = diff(l,r)
    inner = diff(cheekl,cheekr)
    colsim = sat(.56 + 1.8*(sides-.55*across-.35*inner-.12))
    cheeks = .44*skin+.24*np.minimum(cheekl[:,0],cheekr[:,0])+.12*np.maximum(cheekl[:,0],cheekr[:,0])+.20*jaw[:,0]
    cheeks = sat(cheeks)
    hairdark = sat((meanlight(c)-meanlight(hair)+.012)/(.095+.16*meanlight(c)))
    # Also tolerate blond hair and hats. Penalize hair-like colour extending over the background.
    cap = .44*dt+.31*both(diff(hair,p(-.76,-.41)),diff(hair,p(.76,-.41)))+.25*both(diff(c,t),diff(c,hair))
    head = .55*sides + .27*cap + .18*colsim
    capskin = .62*hairdark + .38*sat((skin-hair[:,0]+.10)/.80)
    bgskin = .31*(l[:,0]+r[:,0])+.23*t[:,0]+.15*p(0,.94,.86,.47)[:,0]
    isolate = sat(.18 + .98*cheeks - bgskin)
    # The image-wide skin prior is conditional: in a warm room we specifically
    # want local colour contrast rather than calling all the walls faces.
    hot = f[:,:,0].mean()
    fieldpen = .105*sat((hot-.15)/.49)
    near = .25*(p(-1.60,-.02,1.,.9)[:,0]+p(1.60,-.02,1.,.9)[:,0]+p(0,-1.28,1.0,.62)[:,0]+p(0,1.55,1.,.63)[:,0])
    isolate = .80*isolate+.20*sat(cheeks-near+.13)
    # Oblique posture is handled by displacing lower parts farther than upper ones.
    bodybest = np.zeros_like(x)
    bodytrace = np.zeros_like(x)
    tracecolour = np.zeros_like(x)
    longbest = np.zeros_like(x)
    handbest = np.zeros_like(x)
    for lean in (-.55,0.,.55):
        chest = p(lean,.99,.95,.56)
        belly = p(1.6*lean,1.96,.88,.7)
        pelv = p(2.05*lean,3.00,.85,.62)
        trousers = p(2.50*lean,4.08,.88,.85)
        legs = p(3.*lean,5.48,1.24,.91)
        # Two cross-sections of the torso and a third of the legs. Colour
        # consistency is tested *within* each garment, not between garments.
        cl = p(lean-1.62,1.05,.62,.61)
        cr = p(lean+1.62,1.05,.62,.61)
        bl = p(1.6*lean-1.61,2.06,.73,.79)
        br = p(1.6*lean+1.61,2.06,.73,.79)
        tl = p(2.48*lean-1.39,4.1,.77,1.)
        tr = p(2.48*lean+1.39,4.1,.77,1.)
        chside = both(diff(chest,cl),diff(chest,cr))
        bside = both(diff(belly,bl),diff(belly,br))
        tside = both(diff(trousers,tl),diff(trousers,tr))
        coherence = sat(.52+1.6*(.5*(chside+bside)-.65*diff(chest,belly)-.115))
        back = .5*(diff(cl,cr)+diff(bl,br))
        pop = sat(.50+1.95*(.5*(chside+bside)-.64*back-.17))
        # Check for widening from the neck into the shoulders. At a mere
        # vertical stripe there is no widening, and at a horizon no flanks.
        shl = p(lean-.64,.92,.43,.42)
        shr = p(lean+.64,.92,.43,.42)
        shcont = both(diff(shl,l),diff(shr,r))
        wt = .5*(diff(chest,shl)+diff(chest,shr))
        slope = sat(.47+1.7*(shcont-.88*wt-.20))
        shirtchange = diff(chest,c)
        body = .28*chside+.22*bside+.17*coherence+.13*pop+.13*slope+.07*shirtchange
        visible = sat((96.-y)/h-.55)
        body = visible*body+(1.-visible)*.34
        bodybest = np.maximum(bodybest,body-.012*(lean!=0.))
        # Verify a corridor bounded on both sides between shoulder and hip.
        # Reward edges at its boundary rather than unusually textured shirts.
        trace = np.zeros_like(x)
        for span in (.83,1.18):
            left = np.zeros_like(x)
            right = np.zeros_like(x)
            for ay,rad in ((.81,.90),(1.29,1.06),(1.83,1.04),(2.45,.92),(3.01,.87)):
                mid = lean*min(2.05,.25+.78*ay)
                width = rad*span
                for sign in (-1.,1.):
                    a = p(mid+sign*.72*width,ay,.30,.43)
                    b = p(mid+sign*1.27*width,ay,.32,.43)
                    d = p(mid+sign*.32*width,ay,.35,.43)
                    e = p(mid+sign*1.71*width,ay,.38,.43)
                    ed = diff(a,b)
                    q = sat(ed+.38*(ed-.50*diff(a,d)-.50*diff(b,e))-.08)
                    if sign<0.:
                        left += q/5.
                    else:
                        right += q/5.
            trace = np.maximum(trace,both(left,right))
        bodytrace = np.maximum(bodytrace,.60*trace+.40*body)
        # A connected dark stripe can be a pole. Human clothing often forms
        # two superimposed blocks (shirt/trousers) of distinct appearance.
        trunk = diff(chest,trousers)
        cut = sat(.45+1.4*(trunk-.65*diff(trousers,legs)-.65*diff(chest,belly)-.16))
        tracecolour = np.maximum(tracecolour,.31*trace+.29*body+.18*cut+.22*tside)
        high = (.25*chside+.16*bside+.16*pop+.11*coherence+.16*slope+.16*tside)
        pants = sat(.50+1.15*(tside-.48*diff(tl,tr)-.32*diff(trousers,legs)-.12))
        bottom = diff(belly,trousers)
        high += .11*(pants-.5)+.06*(bottom-.44)+.04*(diff(pelv,bl)-.4)
        # If the lower part is cut off, it cannot confirm the silhouette.
        high -= .062*(1.-sat(((96.-y)/h-2.1)/2.5))
        longbest = np.maximum(longbest,high-.014*(lean!=0.))
        # A hand/forearm should agree with the face in chroma and occur away
        # from the head. Compare to nearby environment and the shirt.
        for side in (-1.,1.):
            arm = np.zeros_like(x)
            for ax,ay in ((1.03,1.65),(1.25,2.32),(.89,2.88),(.60,1.45),(1.73,1.37)):
                hand = p(lean*min(1.8,ay)+side*ax,ay,.45,.39)
                cd = np.abs(hand[:,2]-c[:,2])+np.abs(hand[:,3]-c[:,3])
                seen = (y+ay*h<96.) & (x+(lean*min(1.8,ay)+side*ax)*w>=0.) & (x+(lean*min(1.8,ay)+side*ax)*w<96.)
                arm = np.maximum(arm,seen*hand[:,0]*sat(1.15-2.4*cd)*sat(.35+hand[:,0]-.42*bgskin-.43*chest[:,0]))
            handbest = np.maximum(handbest,arm)

    # Search a small roll range for the dark brow/eyes above bright cheeks,
    # with a nose bridge between the eyes and a separate mouth below.
    facebest = np.zeros_like(x)
    for roll in (-.24,0.,.24):
        def q(dx,dy,ww,hh):
            return p(dx-roll*dy,dy+roll*dx,ww,hh)
        le,re = q(-.205,-.146,.27,.118),q(.205,-.146,.27,.118)
        fl,fr = q(-.17,-.313,.28,.17),q(.17,-.313,.28,.17)
        ml,mr = q(-.221,.033,.26,.163),q(.221,.033,.26,.163)
        bridge = meanlight(q(0,-.079,.13,.19))
        mouth = meanlight(q(0,.168,.36,.082))
        chin = meanlight(q(0,.278,.31,.118))
        phil = meanlight(q(0,.081,.28,.093))
        el,er = meanlight(le),meanlight(re)
        norm = .072+.115*meanlight(c)
        def dark(v):
            return sat(v/norm)
        eye1,eye2 = dark(meanlight(ml)-el),dark(meanlight(mr)-er)
        eye = .58*np.minimum(eye1,eye2)+.21*(eye1+eye2)
        brow = both(dark(meanlight(fl)-el),dark(meanlight(fr)-er))
        nose = dark(bridge - .5*(el+er))
        lips = .60*dark(.55*chin+.45*phil-mouth)+.40*np.minimum(dark(chin-mouth),dark(phil-mouth))
        sym = sat(1.10-(np.abs(el-er)+np.abs(meanlight(ml)-meanlight(mr)))/(.12+.32*meanlight(c)))
        face = .33*eye+.18*brow+.13*nose+.205*lips+.085*sym+.07*sat(1.0-3*metric(ml,mr))
        facebest = np.maximum(facebest,face)
    resolved = sat((np.minimum(w,.81*h)-3.)/8.)
    face = facebest*resolved

    # Fixed compositional priors: a head much nearer the foot of a picture
    # than its centre is less likely unless it is well resolved.
    prior = .042*sat((y-59.)/31.)+.018*sat((np.abs(x-48.)-32.)/16.)
    boundary = (x-.36*w<0.) | (x+.36*w>96.)
    prior += .024*boundary+.028*sat((4.7-w)/2.2)
    # Three alternative explanations: exposed head, resolved facial structure,
    # and a whole upright figure (the last does not require exposed skin).
    colour = .28*cheeks+.225*isolate+.12*head+.135*bodybest+.082*capskin+.061*handbest+.060*face+.037*colsim
    colour -= fieldpen+prior
    facial = .52*face + .12*cheeks + .095*head+.085*isolate+.065*bodybest+.055*capskin+.06*colsim
    facial -= .46*fieldpen+.69*prior
    figure = .41*longbest+.205*head+.115*colsim+.09*cap+.065*isolate+.056*cheeks+.059*capskin
    figure -= .38*fieldpen+prior
    # Average evidence from cheeks and jaw across the inferred face, rather
    # than allowing a single tiny saturated spot to act as an entire face.
    support = np.zeros_like(x)
    purity = np.zeros_like(x)
    for dx,dy in ((-.29,-.18),(.29,-.18),(-.31,.08),(.31,.08),(0,.25),(0,-.29)):
        patch = p(dx,dy,.215,.15)
        support += sat(patch[:,0])/6.
        purity += sat(1.-5.*(np.abs(patch[:,2]-c[:,2])+np.abs(patch[:,3]-c[:,3])))/6.
    cross = .50*(p(-.91,.16)[:,0]+p(.91,.16)[:,0])
    bodywarm = p(0,1.27,1.05,.55)[:,0]
    patchskin = sat(support-.36*cross-.26*bodywarm-.24*t[:,0])
    # Weight by area a human of this head size could occupy. Do not let a
    # multitude of minuscule isolated matches dominate the picture score.
    size = sat((w*h*19.-145.)/300.)
    center = sat(1.-((x-48.)/60.)**2-((y-38.)/60.)**2)
    verify = .28*contour+.27*bodytrace+.27*patchskin+.18*purity
    colour += .44*(verify-.66) + .080*(size-.85)+.050*(center-.7)
    facial += .25*(contour-.53)+.06*(bodytrace-.50)+.04*(purity-.66)
    figure += .37*(bodytrace-.57)+.26*(contour-.56)+.20*(tracecolour-.59)+.08*(size-.85)
    conf = np.maximum(np.maximum((colour-.753)/.093,(facial-.776)/.10),(figure-.783)/.080)
    shapeonly=.38*longbest+.23*tracecolour+.18*bodytrace+.12*contour+.09*head-prior+.05*(size-.85)
    shortlist=np.argpartition(shapeonly,-min(96,len(w)))[-96:]
    ink=silhouette(f,x[shortlist],y[shortlist],w[shortlist],h[shortlist])
    whole=.62*shapeonly[shortlist]+.38*ink
    # Whole-body templates cannot verify an object cut off above the waist.
    whole-=.09*(1.-sat(((96.-y[shortlist])/h[shortlist]-2.)/2.5))
    conf[shortlist]=np.maximum(conf[shortlist],(whole-.738)/.084)
    peak = conf.argmax()
    other = ((np.abs(x-x[peak])>.85*(w+w[peak])) | (np.abs(y-y[peak])>.93*(h+h[peak])))
    repeat = np.max(np.where(other,conf,-8.))
    region = partprior(f,s)
    scene = sceneprior(f)
    context = attention(f,s)+playground(f)
    # Gloves, helmets, distance or grayscale may hide all chromatic body
    # parts. Failure to find a part pair is weak negative evidence, not a veto.
    parts = .84*max(0.,region)+.23*min(0.,region)
    return int(conf[peak]+.24*sat((repeat+.70)/1.2)+scene+parts+context>0.)


def predict(x):
    return np.array([photo(image) for image in x],dtype=np.int64)
'''

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'solution.py').write_text(source)
