import numpy as np


def unit(a):
    return np.minimum(1., np.maximum(0., a))


def integral(a):
    h, w = a.shape[:2]
    z = np.zeros((h+1, w+1) + a.shape[2:], np.float64)
    z[1:, 1:] = a.cumsum(0, dtype=np.float64).cumsum(1)
    return z


def box(ii, hh, ww):
    # A centered mean, with the footprint clipped at the photo's edge.
    h, w = ii.shape[0]-1, ii.shape[1]-1
    y = np.arange(h); x = np.arange(w)
    y0 = np.maximum(0, y-hh//2); y1 = np.minimum(h, y+(hh+1)//2)
    x0 = np.maximum(0, x-ww//2); x1 = np.minimum(w, x+(ww+1)//2)
    # Interior differences are offset views; the edge pieces clip exactly.
    horizontal = np.empty((h+1,w)+ii.shape[2:],np.float64)
    l=ww//2; r=ww-l
    stop=w-r+1
    horizontal[:,l:stop]=ii[:,ww:]-ii[:,:w+1-ww]
    if l:
        horizontal[:,:l]=ii[:,r:ww]-ii[:,:1]
    if stop<w:
        horizontal[:,stop:]=ii[:,w:]-ii[:,w+1-ww:w-l]
    a=np.empty((h,w)+ii.shape[2:],np.float64)
    l=hh//2; r=hh-l; stop=h-r+1
    a[l:stop]=horizontal[hh:]-horizontal[:h+1-hh]
    if l:
        a[:l]=horizontal[r:hh]-horizontal[:1]
    if stop<h:
        a[stop:]=horizontal[h:]-horizontal[h+1-hh:h-l]
    area = (y1-y0)[:, None]*(x1-x0)
    if a.ndim == 3:
        area = area[:, :, None]
    return (a/area).astype(np.float32)


def smooth(a):
    p = np.pad(a, ((1, 1), (1, 1)), mode='edge')
    t = p[:, :-2]+2*p[:, 1:-1]+p[:, 2:]
    return (t[:-2]+2*t[1:-1]+t[2:])*.0625


def flesh(r, g, b):
    # Opponent coordinates. Broad, illumination-dependent tolerances for
    # variation in pigmentation and saturation; hue alone isn't enough.
    u = r-g; v = g-b
    mag = np.maximum(.0, u+.50*v)
    den = .031+.34*mag + .015*g
    axis = np.maximum(0., (v-.30*u-.012)/den) + np.maximum(0., (.24*u-v-.014)/(.034+.46*mag))
    chrom = np.exp(-.5*axis*axis)
    strength = unit((u-.007)/(.031+.033*g))
    notred = unit((.56*r+.060-u)/(.11*r+.034))
    notdark = unit((r-.055)/.17)
    return chrom*strength*notred*notdark


def colourmaps(im):
    r, g, b = im
    lum = .299*r+.587*g+.114*b
    # Modest partial white balance: ignore it when the original is nearly grey.
    av = im.mean(axis=(1, 2))
    light = np.sqrt(np.mean(im**2, axis=(1, 2)))
    gains = np.clip(((np.mean(light)+.035)/(light+.035))**.55, .87, 1.15)
    initial = flesh(r, g, b)
    colour = np.mean(np.maximum(r, np.maximum(g,b))-np.minimum(r, np.minimum(g,b)))
    adj = flesh(r*gains[0], g*gains[1], b*gains[2])
    skin = np.maximum(initial, .88*adj*unit(colour/.035))
    tex = np.abs(lum-smooth(lum))
    # An opponent-color vector for comparing material under different lighting.
    stack = np.stack((lum, (r-g)/(.35+lum), (g-b)/(.35+lum), skin, lum*lum, tex), axis=2)
    return lum, skin, stack, colour


def dist(a, b):
    # Perceptual contrast; still sensitive to dark objects in dark surroundings.
    return (np.abs(a[:, 0]-b[:, 0])/(.68+.66*np.minimum(a[:, 0],b[:, 0])) + .75*np.abs(a[:, 1]-b[:, 1]) + .65*np.abs(a[:, 2]-b[:, 2]))


def selected_indices(a, eligible, n, sy, sx, h, w, step):
    # Nonmaximum suppression followed by a fixed cap per scale, not per batch.
    v = np.where(eligible, a, -20.).reshape(h, w)
    rr = max(1, int(round(sy*.46/step))); cc = max(1, int(round(sx*.46/step)))
    p = np.pad(v, ((rr, rr), (cc, cc)), mode='constant', constant_values=-25.)
    mx = v
    for dy, dx in ((-rr,0),(rr,0),(0,-cc),(0,cc),(-rr,-cc),(rr,-cc),(-rr,cc),(rr,cc)):
        mx = np.maximum(mx, p[rr+dy:rr+dy+h, cc+dx:cc+dx+w])
    ix = np.flatnonzero((v >= mx) & (v > -19.))
    if ix.size > n:
        # Stable tie handling for uniform or quantized regions.
        ix = ix[np.argsort(-v.ravel()[ix], kind='stable')[:n]]
    return ix


def featurebox(cache,ii,hh,ww):
    key=(hh,ww)
    if key not in cache:
        cache[key]=box(ii,hh,ww)
    return cache[key]


def candidates(ii, sy, sx, cache):
    # Localize face-sized patches that differ from their upper/side surround,
    # in both opponent colour and skin likelihood. Preserve some uncoloured
    # proposals so a grey photo can be decided by luminance structure.
    st = max(1, int(round(min(sx,sy)*.22)))
    yr = np.arange(st//2,96,st); xr = np.arange(st//2,96,st)
    h = len(yr); w = len(xr)
    y = np.broadcast_to(yr[:,None], (h,w)).ravel(); x = np.broadcast_to(xr[None,:], (h,w)).ravel()
    m = featurebox(cache,ii,max(1,int(round(.80*sy))),max(1,int(round(.80*sx))))
    def point(dy,dx):
        iy=np.clip(yr+int(round(dy*sy)),0,95); ix=np.clip(xr+int(round(dx*sx)),0,95)
        return m.take(iy,axis=0).take(ix,axis=1).reshape(-1,6)
    c = point(0,0); l = point(-.02,-.97); r = point(-.02,.97); t = point(-.95,0); d = point(1.05,0)
    edge = np.stack([dist(c,l),dist(c,r),dist(c,t),dist(c,d)])
    boundary = .65*np.mean(np.minimum(.36,edge),axis=0) + .35*np.min(np.minimum(.36,edge[:3]),axis=0)
    s = c[:,3]
    ds = s-.33*l[:,3]-.33*r[:,3]-.23*t[:,3]-.11*d[:,3]
    vy = np.sqrt(np.maximum(0.,c[:,4]-c[:,0]**2))
    place = unit(1-((x-47.5)/63)**2)*unit(1-((y-39)/69)**2)
    skinrank = .91*s+.85*ds+1.1*boundary+.08*place
    structrank = 2.1*boundary + .15*unit(vy/.09) -.12*unit((vy-.15)/.13) + .21*s+.13*ds+.07*place
    ok = (x>=.18*sx)&(x<=95-.18*sx)&(y>=.19*sy)&(y<=95-.19*sy)
    i1 = selected_indices(skinrank, ok & (s>.1), 36,sy,sx,h,w,st)
    i2 = selected_indices(structrank, ok, 28,sy,sx,h,w,st)
    ix = np.union1d(i1,i2)
    return y[ix],x[ix],c[ix],ds[ix],boundary[ix],place[ix],m


def one_scale(ii, sy, sx, sem, cache):
    y,x,c,ds,bound,place,wide = candidates(ii, sy, sx, cache)
    n=len(x)
    if n==0:
        return np.zeros((0,6),np.float32)
    # Sample averaged material in a head-centered coordinate system. y=0 is
    # the middle of the visible face, and 1 unit is its height (width in x).
    small = featurebox(cache,ii,max(1,int(round(sy*.26))),max(1,int(round(sx*.27))))
    body = featurebox(cache,ii,max(2,int(round(sy*.93))),max(2,int(round(sx*.85))))
    def at(grid,dy,dx):
        return grid[np.clip(y+int(round(dy*sy)),0,95),np.clip(x+int(round(dx*sx)),0,95)]
    def pt(dy,dx):
        return at(small,dy,dx)
    cheekl = pt(.065,-.225); cheekr = pt(.065,.225)
    eyeleft = pt(-.165,-.23); eyeright = pt(-.165,.23)
    nose = pt(-.09,0); mouth = pt(.275,0); chin = pt(.43,0); brow = pt(-.37,0)
    foreheadl = pt(-.365,-.19); foreheadr = pt(-.365,.19)
    middle = pt(.04,0); cheek = .5*(cheekl+cheekr)
    # Eye sockets are dark relative to the flesh above/below them. Nose
    # bridge and the line of the mouth provide independent structure.
    vl = .65*cheekl[:,0]+.35*foreheadl[:,0]-eyeleft[:,0]
    vr = .65*cheekr[:,0]+.35*foreheadr[:,0]-eyeright[:,0]
    var = np.sqrt(np.maximum(0.,c[:,4]-c[:,0]**2))
    den = .034 + .42*var+.018*c[:,0]
    eyes = unit((.72*np.minimum(vl,vr)+.28*.5*(vl+vr))/den)
    bridge = unit((nose[:,0]-.5*(eyeleft[:,0]+eyeright[:,0]))/den)
    lips = unit((.66*cheek[:,0]+.34*chin[:,0]-mouth[:,0])/(den+.005))
    lipcolor = unit(((mouth[:,1]-cheek[:,1])+(cheek[:,2]-mouth[:,2])*.60)/.09)
    sym = unit(1.15-(dist(cheekl,cheekr)+.55*dist(eyeleft,eyeright))/(.16+.90*var))
    structure = eyes*(.35+.27*bridge+.16*lips+.08*lipcolor+.14*sym)
    if sy<5 or sx<4:
        structure = structure*.42
    elif sy<7 or sx<6:
        structure = structure*.70
    # Is the candidate an oval patch bounded by another material? Sample
    # changes just inside and outside its sides, including hair and jaw line.
    l = pt(-.03,-.67); r = pt(-.03,.67); top = pt(-.72,0)
    dl = dist(l,cheekl); dr = dist(r,cheekr); dt = dist(top,brow)
    edge = unit((.56*np.minimum(dl,dr)+.22*np.maximum(dl,dr)+.22*dt-.023)/.20)
    # Similar colour on the outer cheeks is more useful than a maximum pixel.
    sk = c[:,3]
    local = np.mean(np.stack([cheekl[:,3],cheekr[:,3],brow[:,3],middle[:,3],chin[:,3]]),axis=0)
    fleshscore = .60*sk+.40*local
    iso = unit((ds+.025)/.73)
    compact = unit((local-.35*(l[:,3]+r[:,3])-.30*top[:,3]+.025)/.75)
    isolation = .62*iso+.38*compact
    # Hair cap (a relative test, not a requirement on hair colour).
    hair = unit(((.70*cheek[:,0]+.30*brow[:,0])-.65*pt(-.59,0)[:,0]-.175*pt(-.36,-.45)[:,0]-.175*pt(-.36,.45)[:,0]-.016)/.17)
    below = at(body,1.80,0); upbody = at(body,1.17,0); midbody = at(body,2.65,0)
    # A contiguous region of clothing wider than the head. Its left and right
    # edges should step out near the shoulders, not continue the neck edges.
    torso = np.zeros(n,np.float32); neckstep = np.zeros(n,np.float32)
    for shoulder in (1.12,1.58):
        al = at(body,.22,-shoulder); ar = at(body,.22,shoulder)
        il = at(body,1.55,-shoulder*.74); ir = at(body,1.55,shoulder*.74)
        ol = at(body,1.60,-shoulder*1.65); oright = at(body,1.60,shoulder*1.65)
        cl = dist(below,il); cr = dist(below,ir)
        depth = dist(below,midbody)
        coherent = unit(1.17-(.42*(cl+cr)+.16*depth)/(.105+.22*bound))
        eql = unit((dist(il,ol)-.46*cl-.018)/.21); eqr=unit((dist(ir,oright)-.46*cr-.018)/.21)
        sl = unit((dist(il,al)-.48*cl-.025)/.22); sr = unit((dist(ir,ar)-.48*cr-.025)/.22)
        both = .53*np.minimum(eql,eqr)+.47*.5*(eql+eqr)
        step = .55*np.minimum(sl,sr)+.45*.5*(sl+sr)
        neckstep=np.maximum(neckstep,step*coherent)
        torso=np.maximum(torso,coherent*(.12+.43*both+.45*step)*(.82+.18*unit(1.-depth/.22)))
    visible = unit((95-y)/sy-.55)
    torso *= visible; neckstep *= visible
    clad = unit((dist(c,below)-.03)/.20)*unit((sk-below[:,3]+.15)/.9)
    # Search for another patch of similar skin at anatomically plausible arm
    # offsets. Use proximity in chroma; not all people show hands or arms.
    arm = np.zeros(n,np.float32)
    for ay,ax in ((1.35,2.),(2.4,1.52),(3.3,1.05),(2.0,.96)):
        for sign in (-1,1):
            armpt = at(wide,ay,ax*sign)
            near = unit(1.16-(np.abs(c[:,1]-armpt[:,1])+np.abs(c[:,2]-armpt[:,2]))/.26)
            sep = unit((armpt[:,3]-.55*below[:,3]-.22*at(wide,ay,ax*sign*1.85)[:,3])/.69)
            valid=(y+sy*ay<96)&(x+sx*ax*sign>=0)&(x+sx*ax*sign<96)
            arm=np.maximum(arm,near*armpt[:,3]*sep*valid)
    # Evidence values are on a hand-set margin scale, zero at the intended
    # decision point.  No calibration or thresholds from sample pictures.
    location = .42*(place-.65) - .55*unit((y-66)/29)*unit(11/sy)
    strength = unit((bound-.023)/.18)
    clutter = unit((sem-.13)/.30)
    tiny = max(0.,min(1.,(5.5-min(sy,sx))/3.5))
    face = (-3.98 + 2.22*fleshscore + 1.67*isolation + .51*edge + .36*hair + .42*strength
            + 1.02*structure + .87*torso + .54*arm + .34*clad*visible + location
            - .87*tiny - .68*clutter - .34*unit((c[:,5]-.053)/.075))
    # A uniformly coloured bit of fruit, sign, or furniture can match the
    # skin hue and be isolated. Ask for internal or anatomical support, too.
    detail=unit((var-.021)/.10)*unit((c[:,5]-.005)/.032)
    support=np.maximum.reduce([structure,torso,.84*arm,.58*hair,.55*detail])
    face-=.77*(1.-support)
    # Reject skin-like patches that merely happen to be the right colour:
    # demand other shape evidence if surrounded by similar material.
    face -= .74*(1.-isolation)*(1.-structure)*(1.-torso)
    # Frontal geometry in grayscale, with a separate rule to allow monochrome
    # photographs and faces in shadow. Strong texture alone is not enough.
    geometric = (-3.72 + 1.83*structure + .54*sym + .55*edge + .48*hair + .73*torso
                 + .39*strength + .62*fleshscore + .29*isolation + location - .93*tiny)
    geometric -= .42*unit((c[:,5]-.057)/.07)
    # Relative colour of clothing vs background, vertical continuity, then
    # independent neck and head. This branch also fires on back views.
    headpop = unit((.55*np.minimum(dl,dr)+.45*bound-.028)/.23)
    shoulders = (-3.18 + 1.66*torso + .65*neckstep + .68*headpop + .41*strength
                 + .36*unit((dist(c,upbody)-.025)/.18) + .39*fleshscore + location
                 - .45*tiny - .20*clutter)
    return np.stack((np.maximum(face,np.maximum(geometric,shoulders)),y,x,np.full(n,sy),np.full(n,sx),face),axis=1)


def orientation(dx,dy,angle):
    # Unsigned edge orientation, with smooth tuning and saturation.
    v = np.cos(angle)*dx+np.sin(angle)*dy
    mag = np.sqrt(dx*dx+dy*dy+.000001)
    return np.minimum(mag,.24)*np.maximum(0.,2.*(v/mag)**2-1.)


def contourbank():
    # Polylines around the head and each side of the body, expressed in units
    # of figure height; no bottom edge, so cropped figures are still possible.
    # Each entry is an individual orientation template, never a mean of poses.
    bank=[]
    for pose in range(5):
        if pose==0:
            side=[(.00,-.030),(.036,-.061),(.105,-.064),(.15,-.041),(.175,-.049),(.192,-.106),(.22,-.146),(.31,-.152),(.43,-.160),(.53,-.150),(.55,-.117),(.49,-.105),(.58,-.101),(.75,-.095),(.97,-.078),(.98,-.020),(.77,-.022),(.61,-.011)]
            other=[(yy,-xx) for yy,xx in side]
        elif pose==1:
            side=[(.0,-.028),(.036,-.055),(.109,-.057),(.146,-.033),(.171,-.038),(.198,-.103),(.25,-.122),(.334,-.136),(.41,-.115),(.46,-.066),(.55,-.099),(.66,-.146),(.76,-.169),(.94,-.203),(.96,-.15),(.78,-.097),(.64,-.012)]
            other=[(0,.029),(.031,.054),(.096,.056),(.146,.04),(.17,.041),(.191,.09),(.255,.116),(.33,.142),(.37,.182),(.411,.159),(.393,.102),(.46,.081),(.56,.094),(.76,.123),(.96,.168),(.98,.115),(.80,.065),(.64,.009)]
        elif pose==2:
            side=[(.0,-.04),(.041,-.074),(.122,-.073),(.18,-.043),(.211,-.048),(.224,-.110),(.25,-.164),(.31,-.181),(.48,-.212),(.54,-.185),(.51,-.13),(.56,-.127),(.60,-.18),(.64,-.222),(.72,-.22),(.98,-.17),(.98,-.11),(.75,-.114),(.69,-.078)]
            other=[(yy,-xx) for yy,xx in side]
        elif pose==3:
            side=[(0,-.044),(.044,-.085),(.14,-.084),(.204,-.055),(.24,-.059),(.28,-.153),(.312,-.200),(.395,-.226),(.57,-.265),(.64,-.236),(.602,-.175),(.72,-.161),(.97,-.179)]
            other=[(yy,-xx) for yy,xx in side]
        else:
            side=[(.0,-.066),(.04,-.114),(.184,-.131),(.292,-.103),(.36,-.061),(.40,-.066),(.43,-.176),(.476,-.280),(.53,-.337),(.65,-.359),(.94,-.416)]
            other=[(yy,-xx) for yy,xx in side]
        for aspect in (.78,1.28):
            lines=[]
            for idx,path in enumerate((side,other)):
                for i in range(len(path)-1):
                    y0,x0=path[i]; y1,x1=path[i+1]; x0*=aspect; x1*=aspect
                    dy=y1-y0; dx=x1-x0; ln=(dy*dy+dx*dx)**.5
                    for j in range(max(1,int(round(ln/.064)))):
                        t=(j+.5)/max(1,int(round(ln/.064)))
                        yy=y0+t*dy; xx=x0+t*dx
                        ang=np.arctan2(-dx,dy)%np.pi
                        lines.append((yy,xx,ang,ln/max(1,int(round(ln/.064))),idx))
            bank.append((pose,lines))
    return bank


def stencilbank():
    bank=[]
    contours=contourbank()
    for ht in (19,31,49,73,99,143):
        stride=max(2,int(round(ht/18)))
        yr=np.arange(0,96-int(min(65,ht*.42)),stride)
        xr=np.arange(stride//2,96,stride)
        y=np.broadcast_to(yr[:,None],(len(yr),len(xr))).ravel()
        x=np.broadcast_to(xr[None,:],(len(yr),len(xr))).ravel()
        frac=np.minimum(1.,(95-y)/ht)
        tpl=[]
        for pose,lines in contours:
            if ht<49 and pose==4:
                continue
            lim1=(.17,.17,.21,.24,.4)[pose];lim2=(.53,.46,.55,.65,.7)[pose]
            groups=[{}, {}, {}, {}, {}, {}]
            for yy,xx,ang,weight,idx in lines:
                d=groups[idx+2*(int(yy>lim1)+int(yy>lim2))]
                key=(int(round(yy*ht)),int(round(xx*ht)),int(round(ang*6/np.pi))%6)
                d[key]=d.get(key,0.)+weight
            indices=[]; weights=[]; starts=[]; i=0
            for d in groups:
                starts.append(i)
                if not d:
                    d[(0,0,0)]=0.
                ix=[]; w=[]
                for (dy,dx,ch),value in d.items():
                    iy=y+dy; xx=x+dx; good=(iy<=95)&(xx>=0)&(xx<=95)
                    ix.append(((np.minimum(95,iy)*96+np.clip(xx,0,95))*6+ch).astype(np.int32))
                    w.append((value*good).astype(np.float32))
                ww=np.stack(w); ww/=np.maximum(.00000001,np.maximum(ww.sum(axis=0),.60*sum(d.values())))
                indices.extend(ix);weights.extend(ww);i+=len(w)
            heady=np.minimum(95,y+int(round(ht*lim1*.55)))
            side=int(round(ht*lim1*.98))
            close=unit(1-((x-47.5)/64)**2)*unit(1-((heady-43)/69)**2)
            bias=.24*close-.40*unit((.72-frac)/.32)-3.35-(.34 if ht<25 else 0.)
            tpl.append((np.stack(indices),np.stack(weights),starts,heady,x,np.maximum(0,x-side),np.minimum(95,x+side),bias))
        bank.append((ht,tpl))
    return bank


stencils = stencilbank()


def silhouette(lum, stack, skin, summary):
    # Independent edge matcher. Bin orientations and pool at a scale related
    # to figure size. Match the two sides separately to avoid picking up the
    # edge of a pole or a single building. Compare with nearby context, too.
    best=-4.
    gr=[]
    for k in range(3):
        channel=lum if k==0 else stack[:,:,k]*.72
        channel=smooth(channel)
        gy,gx=np.gradient(channel)
        gr.append((gx,gy))
    ms=np.stack([np.maximum.reduce([orientation(gx,gy,j*np.pi/6) for gx,gy in gr]) for j in range(6)],axis=2)
    ii=integral(ms)
    mi=integral(ms.max(axis=2))
    skinint=integral(skin)
    for ht,templates in stencils:
        k=max(1,int(round(ht*.019)))
        soft=box(ii,k*2+1,k*2+1)
        surround=box(mi,max(5,int(round(ht*.48))),max(5,int(round(ht*.48))))
        # Pool softly over modest misregistration in the direction normal to
        # each edge, but preserve the orientation constraint.
        p=np.pad(soft,((k,k),(k,k),(0,0)),mode='edge')
        for j in range(6):
            dx=int(round(k*np.cos(j*np.pi/6)));dy=int(round(k*np.sin(j*np.pi/6)))
            soft[:,:,j]=.42*soft[:,:,j]+.58*np.maximum.reduce([soft[:,:,j],p[k+dy:k+dy+96,k+dx:k+dx+96,j],p[k-dy:k-dy+96,k-dx:k-dx+96,j]])
        # Dimensionless oriented edge strength relative to local texture.
        soft=soft/(.018+.79*surround[:,:,None]); soft=np.minimum(2.5,soft)
        sf=box(skinint,max(2,int(round(ht*.115))),max(2,int(round(ht*.095))))
        for indexes,weights,starts,heady,x,left,right,bias in templates:
            values=soft.ravel()[indexes]*weights
            # Sum in the spatially contiguous direction; the six groups have
            # different numbers of sampled points.
            parts=[v.sum(axis=0) for v in np.split(values,starts[1:])]
            def pair(a,b):
                return .61*np.minimum(a,b)+.39*.5*(a+b)
            hd=pair(parts[0],parts[1]); sh=pair(parts[2],parts[3]); low=pair(parts[4],parts[5])
            cx=sf[heady,x]; sl=sf[heady,left]; sr=sf[heady,right]
            col=unit((cx-.45*(sl+sr))/.72)*cx
            val=(.95*hd+1.10*sh+.48*low+.38*np.minimum(hd,sh)+.95*col+.33*cx+bias
                 -.30*unit((summary-.13)/.30))
            best=max(best,float(np.max(val)))
    return best


def evaluate(im):
    lum,skin,frame,colour=colourmaps(im)
    ii=integral(frame)
    sem=float(np.mean(skin)); votes=[]
    # Head sizes span distant full figures through tight portraits; widths
    # allow the differing aspect ratios of source photographs and profiles.
    cache={(1,1):frame}
    for sy in (3.4,5.3,8.,12.,18.,27.,41.,63.):
        for rat in (.56,.95):
            votes.append(one_scale(ii,sy,max(2.5,rat*sy),sem,cache))
    allvotes=np.concatenate(votes,axis=0)
    ix=int(np.argmax(allvotes[:,0])); v=allvotes[ix]
    apart=(np.abs(allvotes[:,1]-v[1])>.83*np.maximum(v[3],allvotes[:,3]))|(np.abs(allvotes[:,2]-v[2])>1.05*np.maximum(v[4],allvotes[:,4]))
    second=float(np.max(np.where(apart,allvotes[:,0],-5.)))
    mult=.32*unit((second+.22)/1.10)
    headscore=v[0]+mult
    # Mild photo-level priors: human-centred photos often have an isolated
    # warm subject, rather than extensive warm-coloured scenery. Keep these
    # weak compared with localized face or body evidence.
    center=float(skin[15:77,17:79].mean())
    border=(float(skin[:,:13].mean())+float(skin[:,-13:].mean()))*.5
    globalvote=.50*np.clip((center-border)/.22,-1.,1.) + .17*unit((center-.022)/.12)-.25*unit((sem-.19)/.28)
    # Penalize busy images in which any one of many small texture patches
    # can give accidental template agreement.
    texture=float(frame[:,:,5].mean())
    globalvote-=.22*unit((texture-.033)/.037)
    # The contour branch cannot decrease the score. Skip it if the decision
    # is already certain, without changing the answer for any input.
    if headscore+globalvote>0.:
        return 1
    shape=silhouette(lum,frame,skin,sem)
    score=max(headscore+.23*unit((shape+.35)/1.2),shape+.27*unit((headscore+.4)/1.3))+globalvote
    return int(score>0.)


def predict(x):
    return np.array([evaluate(np.ascontiguousarray(row)) for row in x], dtype=np.int64)
