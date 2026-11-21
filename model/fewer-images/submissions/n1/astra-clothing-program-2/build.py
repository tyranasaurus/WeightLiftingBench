"""Build the independent geometric templates and the standalone inference program.
No input files. Distances below describe shapes in a unit product bounding box,
not positions in photos. Widths for tops are half-widths in units of body length.
"""
import sys
from pathlib import Path
import numpy as np

G = 32
u = (np.arange(G) + .5) / G
X, Y = np.meshgrid(u, u)
A = np.abs(X - .5)
rng = np.random.default_rng(251)
un = rng.uniform

def line(v, knots, heights):
    return np.interp(v, knots, heights)

def polygon(points):
    z = np.zeros((G,G), bool)
    xp, yp = points[-1]
    for x, y in points:
        if y != yp:
            z ^= ((Y < y) != (Y < yp)) & (X < x + (Y-y)*(xp-x)/(yp-y))
        xp, yp = x, y
    return z

def oval(cx, cy, rx, ry):
    return ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 < 1

def ribbon(points, width):
    z = np.zeros((G,G), bool)
    x, y = points[0]
    for xx, yy in points[1:]:
        dx, dy = xx-x, yy-y
        t = np.clip(((X-x)*dx + (Y-y)*dy)/(dx*dx+dy*dy+1e-10), 0, 1)
        z |= ((X-x-dx*t)**2 + (Y-y-dy*t)**2 < width**2)
        x, y = xx, yy
    return z

def top(k, j):
    # Three pieces on a hanger: torso and two sleeves. Express widths and
    # cuff positions relative to shoulder size and trunk length.
    kind = j % 16
    style = 'long'
    if k == 0:
        if kind < 12: style = 'tee'
        elif kind < 15: style = 'tank'
        else: style = 'cap'
    if k == 6:
        if kind in (7,8,9): style = 'blouseshort'
        if kind in (10,11,12): style = 'threeq'
        if kind == 13: style = 'tank'
    if k == 3:
        if kind < 7: style = 'tank'
        elif kind < 12: style = 'cap'
        elif kind < 14: style = 'threeq'
    if k == 2 and kind == 15: style = 'threeq'
    if k == 4 and kind == 15: style = 'threeq'
    if k == 3:
        shoulder = un(.115,.215)
        waist = shoulder*un(.64,1.)
        hem = waist*un(1.22,2.55) if kind % 5 else waist*un(.99,1.38)
    else:
        shoulder = un(.225,.34) if k != 4 else un(.195,.305)
        waist = shoulder*(un(.70,1.04) if k in (6,4) else un(.84,1.14))
        hem = waist*(un(.93,1.18) if k == 6 else un(1.01,1.29) if k == 4 else un(.84,1.04) if k == 2 else un(.90,1.09))
    if style == 'tank' and k != 3:
        shoulder *= .82
    pit = shoulder * un(.86,1.09)
    ypit = un(.21,.34) * (.73 if k == 3 else 1.)
    ywaist = un(.31,.49) if k == 3 else un(.45,.64)
    hip = .63*hem+.37*waist
    if style == 'tank':
        sh = shoulder*un(.70,.97)
        pit *= un(.86,1.03)
    else: sh = shoulder
    neck = sh*(un(.42,.82) if style == 'tank' else un(.24,.60) if k == 0 else un(.24,.51))
    if k == 3: neck = sh*un(.33,.74)
    ndepth = un(.025,.13) if k in (2,4,6) else un(.035,.16)
    if style == 'tank': ndepth = un(.06,.22)
    # Long sleeves can hang straight down, or angle outward; a coat hem
    # typically hangs considerably below the cuff; knit cuffs reach the band.
    if style == 'long':
        end = un(.72,1.10) if k == 2 else un(.57,.92) if k == 4 else un(.63,1.02) if k == 6 else un(.43,.77)
        end *= (1.05 if k == 4 and kind < 4 else 1.)
        outer = max(sh,pit*.96) + un(.045,.20)*shoulder/.26
        thick = un(.042,.102)*shoulder/.26
        upper = un(.04,.10)*shoulder/.26
        cuffbevel = un(-.018,.025)
    elif style == 'threeq':
        end = un(.38,.70)
        outer = sh + un(.08,.19)*shoulder/.26
        thick = un(.069,.12)*shoulder/.26
        upper = un(.05,.105)*shoulder/.26
        cuffbevel = un(.00,.054)
    elif style != 'tank':
        end = un(.24,.42) if style != 'cap' else un(.115,.25)
        if k == 3: end *= .76
        outer = sh + un(.075,.245)*shoulder/.26
        thick = (outer-pit)*un(.63,1.12)
        upper = (outer-sh)*.65
        cuffbevel = un(.025,.096)*(shoulder/.26)
    else:
        end, outer, thick, upper, cuffbevel = 0., 0., 0., 0., 0.
    height = max(end+cuffbevel,1.)
    wide = max(outer,sh,pit,waist,hem)*2
    v, y = A*wide, Y*height
    shoulderY = un(.046,.115)*shoulder/.26
    collarStand = un(.025,.078) if k in (4,6) and kind%4 else 0.
    shoulderY += collarStand
    radius = line(y, [0,shoulderY,ypit,ywaist,.81,1.], [sh,sh,pit,waist,hip,hem])
    if style=='tank':
        radius -= .09*sh*np.exp(-((y-.70*ypit)/(.30*ypit))**2)
    outward=np.clip((v-neck)/(sh-neck+.00001),0,1)
    start = np.where(v < neck, ndepth*(1-(v/neck)**2), (shoulderY-collarStand)*outward+collarStand*np.minimum(1,outward*5))
    if k in (4,6) and kind%3:
        # Raised collar on either side of a shallower V.
        start = np.where(v<neck, np.maximum(0,ndepth*(1-v/neck)),start)
    curve = un(-.025,.055) if k not in (3,6) else un(.008,.087)
    base = 1-curve*np.minimum(v/hem,1)**2
    if k == 6 and kind % 4 == 0: base -= .06*np.exp(-(v/(.06+hem*.1))**2)
    z = (v<radius) & (y>=start) & (y<base)
    if style != 'tank':
        # Sloping sleeve cap and underarm seam, joined by a cuff.
        sleeve = [(sh*.87,shoulderY*.68), (sh,shoulderY),
                  (sh+upper,shoulderY+(end-shoulderY)*.29),
                  (outer,end), (outer-thick,end+cuffbevel),
                  ((outer-thick)*.44+pit*.56,ypit+(end-ypit)*.42), (pit*.94,ypit)]
        sleeve = [(.5+a/wide,b/height) for a,b in sleeve]
        zz = polygon(sleeve)
        z |= zz | zz[:,::-1]
    return z,wide/height

def trousers(j):
    hip = un(.146,.281)
    waist = hip*un(.77,1.)
    crotch = un(.22,.44)
    # Two independently tiltable tapered legs, given by centre and radius.
    foot = hip*un(.25,.48) if j%6 else hip*un(.43,.62)
    midway = hip*un(.32,.49)
    skew = un(-.085,.085)*hip/.21
    gap = un(-.12,.48)*hip if j%8 else un(.45,.85)*hip
    endwidth = max(hip,2*foot+gap+abs(skew))
    x, y = (X-.5)*2*endwidth, Y
    radical = line(y,[0,.15,crotch],[waist,hip*.985,hip])
    z = (np.abs(x)<radical) & (y<crotch)
    for side in (-1,1):
        cuff = un(.94,1.02)
        cent = line(y, [crotch*.68,.61,cuff],[side*hip*.47, side*(hip*.48+gap*.43)+skew*.38, side*(foot+gap)+skew])
        width = line(y, [crotch*.68,.61,cuff],[hip*.54,midway,foot])
        z |= (np.abs(x-cent)<width)&(y>crotch*.68)&(y<cuff)
    return z,2*endwidth

def sneaker(j):
    high = j%10 in (8,9)
    ratio = un(.285,.535) if not high else un(.47,.64)
    tonguex, openingx = un(.48,.67), un(.68,.82)
    toe = un(.20,.38)
    collar = un(.64,.92) if not high else un(.83,.99)
    topline = line(X, [0,.035,.16,.28,tonguex-.10,tonguex,openingx,.93,.98,1],
                     [toe*.56,toe*.89,toe,toe+un(.012,.095),.78,1,un(.48,.74),collar,collar-.055,.43])
    sole = un(.08,.26)*(1-X)**6 + un(.035,.12)*X**8
    if j%8==1: sole += .095*np.exp(-((X-.66)/.12)**2)
    z = (1-Y < topline)&(1-Y > sole)
    return z,1/ratio

def boot(j):
    tall = un(.58,.98) if j%6 else un(.43,.62)
    front,back = un(.43,.69),un(.86,1.)
    toe = un(.14,.32)
    top = line(X,[0,.05,.21,front-.14,front-.042,front,back,min(1,back+.023),1],
               [toe*.64,toe*.95,toe,toe+un(.02,.11),un(.46,.74),un(.82,.99),1,un(.41,.77),.22])
    sole = un(.038,.14)*(1-X)**7 + .035*X**9
    if j%4:
        arch = un(.045,.30)
        heel = un(.69,.89)
        sole += line(X,[0,.33,heel-.065,heel,heel+.035,1],[0,0,arch*.94,arch,0,0])
    z = (1-Y<top)&(1-Y>sole)
    return z,1/tall

def sandal(j):
    kind = j%8
    height = un(.24,.49) if kind < 4 else un(.39,.83)
    heel = un(.035,.16) if kind < 4 else un(.17,.57)
    peak = un(.57,.83)
    up = line(X,[0,.07,.29,peak,.98,1],[un(.085,.20),.11,.115,heel+.12,heel+.13,heel+.10])
    low = up-un(.065,.16)
    if kind in (2,5): low *= .38  # wedge
    z = (1-Y<up) & (1-Y>low)
    if kind >= 4:
        z |= (X<.967)&(X>un(.83,.92)+.045*(1-Y))&(1-Y<heel+.13)
    # Forefoot band/vamp, with an open toe.
    start = un(.019,.19)
    span = un(.22,.42)
    band = un(.32,.76) if kind>=4 else un(.59,.96)
    z |= polygon([(start,.89),(start+.035,.99-band*.54),(start+span*.48,1-band),
                  (start+span*.67,1-band*.99),(start+span,.86),(start+span-.08,.91)])
    if kind in (0,1,3,4,6,7):
        peakx=un(.63,.87)
        z |= ribbon([(un(.28,.53),.86),(peakx,un(.025,.17)),(.95,un(.12,.40)),(.90,.91-heel)],un(.025,.064))
    if kind in (1,4,6):
        z |= ribbon([(.19,.65),(un(.39,.58),.31),(un(.63,.81),.85-heel)],un(.026,.068))
    if kind in (3,7):
        z |= polygon([(.76,.87-heel),(.80,.28),(.95,.19),(.99,.82-heel)])
    if kind == 5:
        z |= ribbon([(.39,.79),(.69,.14),(.94,.075)],.055)
    return z,1/height

def bag(j):
    kind = j%10
    ratio = un(.79,1.55) if kind not in (7,8,9) else un(1.3,2.75) if kind==7 else un(.58,.98)
    h = un(.035,.44) if kind<=6 else un(.0,.095)
    lean = un(-.07,.07)
    hw = line(Y,[h,h+(1-h)*.15,.85,1.],[un(.35,.48),un(.44,.50),un(.39,.50),un(.34,.47)])
    corner = np.maximum(0,(Y-.92)/.08)**2*.036
    topcurve = un(-.06,.125)*(1-(2*A)**2)
    if kind in (8,9):
        hw *= (.49+.51*np.sqrt(np.maximum(0,1-((np.maximum(.52-Y,0))/.54)**2)))
    z = (np.abs(X-.5-lean*(Y-.5)) < hw-corner) & (Y>h+topcurve)
    if kind < 7 or kind == 9:
        cx,rx = .5+un(-.072,.072),un(.185,.345)
        cy=h+un(.06,.15)
        thick = un(.021,.061)
        z |= oval(cx,cy,rx,cy) & ~oval(cx,cy,rx-thick,max(.02,cy-thick*ratio))
        if kind in (2,3,6):
            z |= ribbon([(cx-rx,h+.16),(un(.28,.39),.05),(un(.52,.69),.075),(cx+rx,h+.14)],thick*.65)
    return z,ratio

# Express each individual drawing in its own box, i.e. canonical mask plus
# original aspect ratio. No operation here reduces over different drawings.
def make_template(z, aspect):
    yi, xi = np.nonzero(z)
    xl,xr,yt,yb = xi.min(),xi.max()+1,yi.min(),yi.max()+1
    w,h = xr-xl,yb-yt
    # Bilinear sample at pixel centres, with a small footprint for antialiasing.
    cells = np.zeros((16,16))
    for dx in (.2,.5,.8):
        x = xl+(np.arange(16)+dx)*w/16-.5
        for dy in (.2,.5,.8):
            y = yt+(np.arange(16)+dy)*h/16-.5
            x0,y0 = np.floor(x).astype(int),np.floor(y).astype(int)
            fx,fy = x-x0,y-y0
            p = np.pad(z,1).astype(float)
            a = p[(y0+1)[:,None],x0+1]*(1-fx)+p[(y0+1)[:,None],x0+2]*fx
            b = p[(y0+2)[:,None],x0+1]*(1-fx)+p[(y0+2)[:,None],x0+2]*fx
            cells += ((1-fy[:,None])*a+fy[:,None]*b)/9
    return cells.astype(np.float32), np.log(aspect*w/h)

# Shared one-silhouette descriptor: low-pass cell coverage, and depth of
# silhouette from the top, bottom, left and right. The cumulative maxima
# prevent an internal print from being mistaken for a notch in the outline.
DESCRIPTOR = '''def profile(m):
    h=np.max(m,axis=2,keepdims=True)
    v=np.max(m,axis=1,keepdims=True)
    left=np.maximum.accumulate(m/(h+.001),axis=2).mean(2)
    right=np.maximum.accumulate(m[:,:,::-1]/(h+.001),axis=2).mean(2)
    roof=np.maximum.accumulate(m/(v+.001),axis=1).mean(1)
    sole=np.maximum.accumulate(m[:,::-1]/(v+.001),axis=1).mean(1)
    return np.concatenate((left,right,roof,sole,m.mean(1),m.mean(2)),axis=1)
'''
# build.py contains the formula explicitly rather than evaluating solution.py.
def profile_one(m):
    a=m/(m.max(1,keepdims=True)+.001)
    b=m/(m.max(0,keepdims=True)+.001)
    return np.concatenate((np.maximum.accumulate(a,axis=1).mean(1),
        np.maximum.accumulate(a[:,::-1],axis=1).mean(1),
        np.maximum.accumulate(b,axis=0).mean(0),
        np.maximum.accumulate(b[::-1],axis=0).mean(0),m.mean(0),m.mean(1)))

RUNTIME = 'import numpy as np\n\n'+DESCRIPTOR+r'''

# Anisotropic resize: preserve width / height separately. Relative heights
# can then be used to talk about e.g. a cuff even for a narrow item.
def crop(v, left, top, w, h, g):
    unit=(np.arange(g,dtype=np.float32)+.5)/g
    x=np.clip(left[:,None]-.5+w[:,None]*unit,0,27)
    y=np.clip(top[:,None]-.5+h[:,None]*unit,0,27)
    ix=x.astype(np.int32)
    iy=y.astype(np.int32)
    dx=(x-ix).astype(np.float32)[:,None,:]
    dy=(y-iy).astype(np.float32)[:,:,None]
    i=np.arange(len(v))[:,None,None]
    ax=ix[:,None,:]
    ay=iy[:,:,None]
    bx=np.minimum(ax+1,27)
    by=np.minimum(ay+1,27)
    return (1-dy)*((1-dx)*v[i,ay,ax]+dx*v[i,ay,bx])+dy*((1-dx)*v[i,by,ax]+dx*v[i,by,bx])

def box(projection):
    on=projection>.69
    lo=on.argmax(axis=1)
    hi=28-on[:,::-1].argmax(axis=1)
    return lo.astype(np.float32),(hi-lo).astype(np.float32)

def smooth(v):
    p=np.pad(v,((0,0),(1,1),(1,1)),mode='edge')
    a=p[:,:-2]+2*p[:,1:-1]+p[:,2:]
    return (a[:,:,:-2]+2*a[:,:,1:-1]+a[:,:,2:])*(1/16)

def clip(x):
    return np.clip(x,0,1)

def rise(x,lo,hi):
    return clip((x-lo)/(hi-lo))

def rect(a,y,b,x,d):
    return a[:,y:b,x:d].mean(axis=(1,2))

def pair(a,y,b,x,d):
    return .5*(rect(a,y,b,x,d)+rect(a[:,:,::-1],y,b,x,d))

def lookup(mask, asp):
    # Quantize the soft support, so integral dot-products of the 256
    # entries have the same value for any BLAS blocking or batch length.
    v=np.rint(mask*15).astype(np.float32).reshape(len(mask),256)
    axis=np.rint(127*profile(mask)).astype(np.float32)
    raxis=np.rint(127*profile(mask[:,:,::-1])).astype(np.float32)
    size=(v*v).sum(1)
    asize=(axis*axis).sum(1)
    count=len(CONSTANTS['ratio'])//10
    ans=[]
    for k in range(10):
        sl=slice(k*count,(k+1)*count)
        dots=v@CONSTANTS['maps'][sl].T
        traces=axis@CONSTANTS['outline'][sl].T
        dif=(size[:,None]+CONSTANTS['energy'][None,sl]-2*dots)*(13/(256*225))
        dist=(asize[:,None]+CONSTANTS['square'][None,sl]-2*traces)*(12/(96*127*127))
        if k in (5,7,9):
            rv=v.reshape(len(v),16,16)[:,:,::-1].reshape(len(v),256)
            alt=(size[:,None]+CONSTANTS['energy'][None,sl]-2*(rv@CONSTANTS['maps'][sl].T))*(13/(256*225))
            other=((raxis*raxis).sum(1)[:,None]+CONSTANTS['square'][None,sl]-2*(raxis@CONSTANTS['outline'][sl].T))*(12/(96*127*127))
            diff=np.minimum(dif+dist,alt+other)
        else: diff=dif+dist
        delta=np.abs(asp[:,None]-CONSTANTS['ratio'][None,sl])
        loss=diff+2.7*np.maximum(0,delta-.035)+2.4*np.minimum(delta,.65)**2
        # A small neighbourhood in drawing space, without combining drawings
        # into templates. The scores here pertain only to the query image.
        best=np.partition(loss,5,axis=1)[:,:6]
        ans.append(-(.55*best.min(1)+.45*best.mean(1)))
    return np.stack(ans,axis=1)

def classify(v):
    # Use both a faint-material threshold and a stricter local vote. A light
    # print on a dark item should not be mistaken for the entire silhouette.
    peak=np.maximum(v.max(axis=(1,2),keepdims=True),.001)
    light=v/peak
    occ=.68*clip((light-.016)/.082)+.32*clip((light-.045)/.17)
    sub=smooth(occ)
    occ=np.maximum(.89*occ, np.minimum(1,sub*1.48))
    left,w=box(occ.sum(1))
    top,h=box(occ.sum(2))
    ar=w/h
    asp=np.log(ar)
    full=crop(occ,left,top,w,h,32)
    mask=full.reshape(len(v),16,2,16,2).mean(axis=(2,4))
    sc=lookup(mask,asp)
    # Geometry and photometry in object coordinates. Use a less filled-in
    # support for holes, straps and separating a placket from background.
    m=crop(.70*clip((light-.025)/.115)+.30*clip((light-.095)/.22),left,top,w,h,24)
    a=crop(light,left,top,w,h,24)
    occupancy=m.mean((1,2))
    coeff=1/(m.sum((1,2))+.01)
    mean=(m*a).sum((1,2))*coeff
    b=np.minimum(2.5,a/(mean[:,None,None]+.095))
    mass=m.mean(2)
    upper=mass[:,1:6].mean(1)
    lower=mass[:,15:22].mean(1)
    mid=mass[:,9:14].mean(1)
    symdiff=(np.abs(m-m[:,:,::-1])).mean((1,2))
    # Compare the amount of material at the top of the two ends. Shoe
    # direction is determined anew for each input (toe at left below).
    side=rect(m,0,13,0,10)-rect(m,0,13,14,24)
    flip=side>0
    shoe=np.where(flip[:,None,None],m[:,:,::-1],m)
    shoeLight=np.where(flip[:,None,None],b[:,:,::-1],b)
    # Height of the front surface at each x, rejecting single faint pixels.
    has=shoe>.40
    height=1-has.argmax(1)/24
    height=np.where(has.any(1),height,0)
    toe=height[:,2:7].mean(1)
    shaft=rect(shoe,1,7,15,22)
    toesolid=rect(shoe,12,20,3,10)
    vamp=rect(shoe,8,15,10,17)
    heelwall=rect(shoe,5,17,18,22)
    toeback=height[:,16:22].mean(1)-toe
    # Negative space enclosed between straps and over the sole. Measure
    # emptiness in each column below the first material, above the last.
    ceiling=np.maximum.accumulate(shoe,axis=1)
    floor=np.maximum.accumulate(shoe[:,::-1],axis=1)[:,::-1]
    holes=np.maximum(0,np.minimum(ceiling,floor)-shoe)
    hole=rect(holes,4,21,2,23)
    # Sole: a thin, horizontally continuous band beneath the upper. Compare
    # both signs of the edge; many sneakers have a contrasting rubber rim.
    dv=shoeLight[:,1:]-shoeLight[:,:-1]
    strong=np.minimum(1,np.abs(dv[:,16:22,3:21])).mean(2).max(1)
    # Upright garments are nearly bilaterally symmetric; shoe profiles
    # usually lack this property. Don't infer a shoe from width alone.
    shapelat=rise(ar,1.18,1.9)
    shoeprior=.48*rise(symdiff,.17,.39)+.45*shapelat+.46*rise(toeback,.19,.55)-.85*(1-rise(ar,.92,1.25))-.72*(1-rise(symdiff,.10,.25))*(1-shapelat)
    for k in (5,7,9): sc[:,k]+=shoeprior
    air=rise(hole,.048,.205)
    closed=rise(heelwall,.54,.88)*rise(vamp,.41,.78)
    sc[:,5]+=1.75*air +.70*(1-rise(occupancy,.42,.68))-.58*closed-.34*(1-air)
    sc[:,7]+=.76*closed+.40*rise(strong,.18,.49)+.45*rise(ar,1.75,2.65)-1.55*air-.40*rise(shaft,.57,.86)
    sc[:,9]+=.87*rise(shaft,.49,.82)+.43*rise(toeback,.36,.66)+.36*(1-rise(ar,1.5,2.12))-.72*air-.42*rise(ar,1.88,2.54)
    # A bag has a container below an optional handle. It rarely has two
    # lower extensions like sleeves or two separated trouser legs.
    center=rect(m,17,23,10,14)
    handlegap=rect(holes,1,13,7,17)
    # A handle is a horizontal as well as a vertical enclosure; a U-neck
    # may enclose horizontally, but there is no material above it.
    bulk=rise(lower-upper,.11,.45)*rise(lower,.65,.90)
    flatwide=rise(ar,1.22,1.85)*(1-rise(symdiff,.13,.34))*rise(occupancy,.64,.84)
    sc[:,8]+=.88*bulk+.98*rise(handlegap,.065,.29)*rise(lower,.57,.84)+.77*flatwide-.42*rise(upper-lower,.10,.32)
    # Interior of a split visible below a waistband.
    gap=pair(m,15,23,4,8)-rect(m,15,23,10,14)
    fork=(1-center)*rise(upper,.48,.83)*rise(gap,.07,.47)
    sc[:,1]+=1.25*fork+.57*(1-rise(ar,.46,.71))*rise(upper-lower,.025,.22)-.50*rise(ar,.71,.99)
    # Upper garments: compare material in the cuff, underarm, waist, and
    # skirt zones. Long sleeves have a trough of background on *both* sides
    # of the torso, often persisting all the way down to the wrist.
    flank=pair(m,12,20,0,4)
    cuff=pair(m,17,22,1,5)
    higharm=pair(m,4,9,0,5)
    latearm=pair(m,10,16,0,4)
    sidecut=rise(higharm-latearm,.15,.56)
    longarm=rise(flank,.18,.60)
    narrow=1-rise(ar,.56,.82)
    flare=rise(lower-mid,.035,.29)
    sc[:,3]+=.69*flare+.58*narrow+.38*rise(lower-upper,.00,.29)-.62*fork-.38*longarm*(1-flare)
    sc[:,0]+=.94*sidecut+.37*(1-longarm)+.32*rise(ar,.85,1.20)-.69*longarm-.35*flare-.38*narrow
    # Evidence inside a shirt or coat: a narrow centre opening or row of
    # buttons gives a local contrast at the *same x* over many rows. Allow
    # a few pixels of misregistration, and ignore missing fabric near hem.
    # Use coherence to separate these lines from an all-over fabric pattern.
    dy=b[:,6:21,:]
    ridges=dy[:,:,2:-2]-.5*(dy[:,:,:-4]+dy[:,:,4:])
    col=np.abs(ridges.mean(1))*.62+np.abs(ridges).mean(1)*.38
    seam=np.maximum(0,col[:,8:12].max(1)-.63*(col[:,4:8].mean(1)+col[:,12:16].mean(1))*.5)
    # A lapel opens into a V whose two high-contrast edges approach the
    # centre line. Detect edges of either sign, discount other chest edges.
    grad=np.abs(b[:,:,2:]-b[:,:,:-2])
    collar=np.zeros(len(m),np.float32)
    rows=np.arange(2,11)
    for spread in (3.2,5.5):
        for stop in (7.,12.):
            offs=np.maximum(.9,spread*(1-rows/stop))
            for dx in (-1,0,1):
                ll=np.rint(10.5+dx-offs).astype(np.int32)
                rr=np.rint(10.5+dx+offs).astype(np.int32)
                val=(grad[:,rows,ll]+grad[:,rows,rr]).mean(1)*.5
                collar=np.maximum(collar,val)
    lapel=np.maximum(0,collar-.68*grad[:,2:11,4:18].mean((1,2)))
    placket=rise(seam,.090,.34)
    lapels=rise(lapel,.14,.48)
    # Local contrast within fabric only. Prints, checks, button-down shirts
    # and blouses often have more variation than plain fine-gauge knits.
    core=(m[:,4:19,6:18]>.65)&(m[:,5:20,6:18]>.65)&(m[:,4:19,7:19]>.65)
    p=b[:,4:19,6:18]
    texture=((np.abs(p-b[:,5:20,6:18])+np.abs(p-b[:,4:19,7:19]))*core).sum((1,2))/(2*core.sum((1,2))+1)
    printcue=rise(texture,.14,.32)
    hemratio=mass[:,21:23].mean(1)/(mid+.09)
    shorthem=1-rise(hemratio,.54,.82)
    boxy=rise(hemratio,.77,1.04)
    # A pullover: full length sleeves and a continuous front, typically
    # blousing over a narrower waistband. A coat is cut to open vertically,
    # has a substantial collar, and often extends below its sleeves.
    sc[:,2]+=.47*longarm+.38*shorthem+.39*(1-placket)+.17*(1-lapels)-.83*placket-.42*lapels-.35*sidecut-.31*printcue
    sc[:,4]+=.69*placket+.42*lapels+.37*boxy+.23*longarm-.42*sidecut-.31*(1-placket)*(1-lapels)-.24*printcue
    sc[:,6]+=.61*placket+.55*lapels+.66*printcue+.24*rise(latearm-cuff,.15,.51)-.34*(1-placket)*(1-lapels)*(1-printcue)
    # Separate a neck opening which *closes* near the collarbone from
    # one which continues into a pair of lapels; account for a light inner
    # facing as well as a dark opening. A dress can also have a scoop neck.
    neckhole=pair(m,0,3,6,9)-rect(m,0,3,10,14)
    scoop=rise(neckhole,.16,.63)*rise(rect(m,4,7,10,14),.5,.89)
    # Test for the small raised stand of a tailored collar at centre top.
    stand=rise(rect(m,0,2,9,15)-pair(m,0,2,5,8),.25,.72)
    midmark=np.abs(rect(b,7,14,10,14)-pair(b,7,14,6,9))
    facing=rise(midmark,.17,.48)*(.35+.65*lapels)
    under=rect(m,18,21,10,14)
    frontopen=rise(pair(m,17,21,6,9)-under,.20,.65)*rise(upper,.45,.80)*(1-narrow)
    dressneck=scoop*(1-rise(ar,.54,.88))
    sc[:,3]+=.35*dressneck
    sc[:,0]+=.35*scoop*(1-longarm)-.40*stand
    sc[:,2]+=.27*scoop*longarm-.42*stand-.44*facing-.58*frontopen
    sc[:,4]+=.33*stand+.42*facing+.63*frontopen-.35*scoop
    sc[:,6]+=.42*stand+.25*facing-.18*scoop
    # Light, patterned woven shirts and dark outerwear are weak colour
    # priors only; shape and front-construction evidence get larger weights.
    white=rise(mean,.62,.86)
    sc[:,6]+=.30*white*(.35+.65*lapels)
    sc[:,4]-=.20*white
    # A strong button band through the lower front of an otherwise T-shaped
    # item is evidence for a shirt, not a graphic on a jersey T-shirt.
    sc[:,0]-=.42*placket*(.4+.6*lapels)
    return sc.argmax(1)

def predict(x):
    answer=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),96):
        answer[start:start+96]=classify(x[start:start+96,0])
    return answer
'''

def main(folder):
    folder.mkdir(parents=True,exist_ok=True)
    maps=[]
    ratios=[]
    outlines=[]
    for k in range(10):
        for j in range(1152):
            if k in (0,2,3,4,6): z,aspect=top(k,j)
            elif k==1: z,aspect=trousers(j)
            elif k==5: z,aspect=sandal(j)
            elif k==7: z,aspect=sneaker(j)
            elif k==8: z,aspect=bag(j)
            else: z,aspect=boot(j)
            m,ar=make_template(z,aspect)
            maps.append(np.rint(15*m).reshape(-1))
            ratios.append(ar)
            outlines.append(np.rint(127*profile_one(m)))
    maps=np.array(maps,dtype=np.float32)
    outlines=np.array(outlines,dtype=np.float32)
    np.savez(folder/'constants.npz',maps=maps,ratio=np.array(ratios,dtype=np.float32),energy=(maps*maps).sum(1),outline=outlines,square=(outlines*outlines).sum(1))
    (folder/'solution.py').write_text(RUNTIME)

if __name__=='__main__':
    main(Path(sys.argv[1]))
