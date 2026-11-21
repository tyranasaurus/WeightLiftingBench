"""Build a library of individual analytic renderings, without input files."""
import sys
import math
from pathlib import Path
import numpy as np

features = '''import numpy as np

# All reductions in the descriptor are within one photograph.
def blur(a):
    p = np.pad(a, ((0,0),(0,0),(1,1)), mode='edge')
    a = (p[:,:,:-2] + 2*p[:,:,1:-1] + p[:,:,2:]) * .25
    p = np.pad(a, ((0,0),(1,1),(0,0)), mode='edge')
    return (p[:,:-2] + 2*p[:,1:-1] + p[:,2:]) * .25

def pool(a,k):
    n,h,w = a.shape
    return a.reshape(n,h//k,k,w//k,k).mean(axis=(2,4))

def grad(a):
    p=np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    return (p[:,1:-1,2:]-p[:,1:-1,:-2])*.5,(p[:,2:,1:-1]-p[:,:-2,1:-1])*.5

def desc(x):
    n=len(x)
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=.29*r+.55*g+.16*b
    blocks=[]
    def unit(a,w):
        z=a.reshape(n,-1)
        blocks.append(w**.5*z / np.sqrt((z*z).sum(1,keepdims=True)+.000001))
    def rms(a,w):
        z=a.reshape(n,-1)
        blocks.append(z*((w/z.shape[1])**.5))
    def center(a):
        return a-a.mean((1,2),keepdims=True)
    # Signed arrangement of light and shadow, with soft contrast compression.
    t=pool(y,4)
    v=center(t)
    c=np.sqrt((v*v).mean((1,2),keepdims=True)+.0036)
    unit(np.tanh(v/(1.6*c)),.18)
    hp=t-blur(blur(t))
    unit(np.tanh(hp/(.075+.30*c)),.11)
    # Spatially pooled edge directions, at two bandwidths. Opposite edge signs
    # are equivalent. Isotropic texture does not count as a strong line.
    dx,dy=grad(blur(y)); fx,fy=grad(y)
    magnitude=np.sqrt(dx*dx+dy*dy)+1.e-8
    fine=np.sqrt(fx*fx+fy*fy)+1.e-8
    dx2,dy2=grad(blur(blur(blur(y))))
    mag2=np.sqrt(dx2*dx2+dy2*dy2)+1.e-8
    for a,d,m,k,wt in ((dx,dy,magnitude,4,.21),(dx2,dy2,mag2,8,.10)):
        hist=[]
        for i in range(6):
            ang=i*np.pi/6
            proj=np.abs(a*np.cos(ang)+d*np.sin(ang))
            q=np.maximum(proj-.835*m,0)*6
            hist.append(pool(blur(q.astype(np.float32)),k))
        h=np.stack(hist,1); en=h.sum(1,keepdims=True)
        h=(h-.105*en)/np.sqrt(en+.044)
        unit(h,wt)
    # Edge density. Absolute and relative texture are both informative.
    d=pool(fine,4); dl=np.log1p(d*35)
    rms(np.clip(center(dl)/.50,-2.5,2.5),.09)
    rms(np.clip((pool(fine,8)-pool(magnitude,8))/.032,0,2.8),.026)
    # Opponent chroma coordinates are regularized in the shadows. Pool AFTER
    # compression, so mottled vegetation is not equivalent to a uniform grey.
    warm=np.tanh(3.1*(r-b)/(.28+y))
    leaf=np.tanh(6*(g-.58*r-.42*b)/(.25+y))
    sat=(np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b))/(.22+y)
    color=np.stack([pool(warm,8),pool(leaf,8),pool(np.tanh(2.5*sat),8)],1)
    rms(color*np.array([1.30,1.27,.70],np.float32)[None,:,None,None],.63)
    rms(center(pool(warm,4)),.095); rms(center(pool(leaf,4)),.09)
    rms(np.tanh((pool(y,8)-.50)*3),.065)
    return np.ascontiguousarray(np.concatenate(blocks,axis=1),dtype=np.float32)
'''

# Coordinates are in arbitrary object-space units (length, up, breadth).
# The renderer uses a depth buffer; quadrics are intersected analytically.
rng=np.random.default_rng(71823)
U=rng.uniform
N=rng.normal
pi=math.pi

def pick(seq): return seq[int(rng.integers(len(seq)))]
def chance(p): return U()<p
def rgb(c): return np.array(c,dtype=float)
def shade(c,k): return np.minimum(1,rgb(c)*k)
def vary(c,j=.07): return np.clip(rgb(c)*U(1-j,1+j,3)*U(.86,1.13),.015,.99)
def blend(c,d,t): return rgb(c)*(1-t)+rgb(d)*t
black=(.10,.095,.09); white=(.91,.90,.86); silver=(.67,.69,.70)
red=(.74,.11,.09); blue=(.14,.28,.54); beige=(.73,.63,.46)
brown=(.40,.25,.15); chestnut=(.56,.31,.16); gold=(.75,.55,.30)
glass=(.29,.39,.45); rubber=(.105,.105,.11)
paints=[white,silver,red,(.46,.065,.065),blue,(.16,.19,.19),(.40,.43,.45),(.17,.34,.28),(.82,.70,.36)]
def paint(): return vary(pick(paints))

def ell(p,r,c):
    parts.append((0,np.array(p),np.diag(r),rgb(c)))
def oval(p,q,r,s,c):
    p,q=rgb(p),rgb(q); a=q-p; h=np.linalg.norm(a)
    if h<.00001: return
    d=a/h; v=np.array([-d[1],d[0],0.]); vn=np.linalg.norm(v)
    if vn<.0001: v=np.array([1.,0.,0.])
    else: v/=vn
    w=np.cross(d,v)
    parts.append((0,(p+q)*.5,np.column_stack((a*.55,v*r,w*s)),rgb(c)))
def face(points,c):
    parts.append((1,np.array(points),None,rgb(c)))
def prism(points,z0,z1,c):
    # Convex x/y profile extruded in depth.
    a=[(x,y,z0) for x,y in points]; b=[(x,y,z1) for x,y in points]
    face(a,c);face(b,c)
    for k in range(len(a)):
        j=k-1;face([a[j],b[j],b[k],a[k]],c)
def box(x0,x1,y0,y1,z0,z1,c):
    prism([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],z0,z1,c)
def tube(p,q,r,c): oval(p,q,r,r,c)

def wheel(x,y,z,rad,wide):
    sign=1 if z>0 else -1
    ell((x,y,z),(rad,rad,wide),rubber)
    ell((x,y,z+sign*wide*.83),(rad*.55,rad*.55,wide*.28),shade(silver,U(.64,1.12)))
    ell((x,y,z+sign*wide*1.07),(rad*.19,rad*.19,wide*.08),shade(black,1.6))

def car(truck):
    col=paint(); win=vary(pick([glass,(.15,.19,.22),(.38,.45,.50),(.47,.55,.60)])); lamp=(.93,.89,.70)
    if not truck:
        L=1.; W=U(.35,.45); h=U(.32,.43); H=h+U(.23,.36)
        if chance(.23): H+=.10
        wheelr=U(.145,.185); base=.13
        prism([(-L+.05,base),(.91,base),(1,.21),(.94,h-.055),(.53,h),(-.73,h+.025),(-.98,h-.045)],-W,W,col)
        rear=U(-.92,-.51); rf=U(-.57,-.28); fr=U(.03,.27); front=U(.46,.67)
        # tapered greenhouse with painted pillars and dark windshield
        A=[(rear,h,-W*.97),(front,h,-W*.97),(front,h,W*.97),(rear,h,W*.97)]
        B=[(rf,H,-W*.77),(fr,H,-W*.77),(fr,H,W*.77),(rf,H,W*.77)]
        face(B,shade(col,1.08))
        for i in range(4):
            j=(i+1)%4; f=np.array([A[i],A[j],B[j],B[i]])
            face(f,col); center=f.mean(0); ins=(f-center)*.88+center
            normal=np.cross(f[1]-f[0],f[2]-f[0]); normal/=np.linalg.norm(normal)
            if normal@(center-np.array([0.,h,0.]))<0: normal=-normal
            face(ins+normal*.006,win)
            if i%2==0:
                m=(ins[0]+ins[1])*.5;n=(ins[2]+ins[3])*.5
                tube(m+normal*.012,n+normal*.012,.020,col)
        for z in [-W,W]:
            for x in [-.65,.64]: wheel(x,wheelr+.005,z*.96,wheelr,.08)
            ell((.27,h+.066,z*1.045),(.09,.044,.064),col)
        face([(.985,.20,-W*.59),(.985,.20,W*.59),(.97,.28,W*.59),(.97,.28,-W*.59)],shade(black,1.5))
        for s in [-1,1]:
            ell((.966,h-.098,s*W*.74),(.035,.046,W*.23),lamp)
            ell((-.967,h-.10,s*W*.78),(.034,.049,W*.18),red)
    else:
        w=U(.29,.40); cargo=vary(pick([white,white,silver,red,blue,(.58,.47,.34),(.82,.65,.18)])); h=U(.75,1.05)
        box(-1,.96,.15,.29,-w*.86,w*.86,shade(black,1.4))
        kind=U(); base=.31
        if kind<.60:
            box(-1.04,.26,base,h,-w,w,cargo)
            for z in [-w*1.001,w*1.001]:
                if chance(.55):
                    face([(-.98,h*.52,z),(.21,h*.52,z),(.21,h*.60,z),(-.98,h*.60,z)],blend(cargo,col,.6))
        elif kind<.80:
            ell((-.38,(h+base)*.5,0),(.71,(h-base)*.50,w),cargo)
        elif kind<.95:
            prism([(-.88,base),(.18,base),(.28,h*.79),(-1.04,h*.83)],-w,w,cargo)
        else: box(-1,.23,base,.39,-w,w,cargo)
        H=h*U(.74,.99); nose=U(.63,.91)
        prism([(.25,.22),(.99,.22),(.99,H*.59),(nose,H*.65),(nose-.07,H),(.26,H)],-w*.96,w*.96,col)
        for z in [-w*.965,w*.965]:
            face([(.32,H*.68,z),(nose-.012,H*.68,z),(nose-.092,H*.94,z),(.32,H*.94,z)],win)
        face([(nose-.011,H*.69,-w*.86),(nose-.011,H*.69,w*.86),(nose-.051,H*.94,w*.86),(nose-.051,H*.94,-w*.86)],win)
        box(.986,1.,.24,.24+H*.21,-w*.68,w*.68,shade(silver,.6))
        for s in [-1,1]:
            z=s*w
            for x in [-.77,-.49,.72]: wheel(x,.17,z*.93,.17,.071)
            ell((.995,.32,z*.83),(.02,.037,.055),lamp)
            ell((.67,H*.70,z*1.09),(.037,.060,.035),black)

def wing(points,y,c):
    face([(x,y+dy,z) for x,z,dy in points],c)
def plane():
    col=vary(pick([white,white,silver,(.77,.79,.79),(.44,.48,.48),(.90,.84,.69),(.22,.38,.50)]))
    trim=paint(); jet=chance(.70); r=U(.09,.15) if jet else U(.12,.18); span=U(.78,1.30)
    ell((0,0,0),(1.02,r,r),col)
    for s in [-1,1]:
        sweep=U(.20,.46) if jet else U(-.04,.13)
        wing([(.26,0,0),(.20-sweep,s*span,.082),(.01-sweep,s*span,.08),(-.29,0,0)],r*(.44 if not jet else -.42),col)
        wing([(-.62,0,0),(-.83,s*span*.37,.024),(-.99,s*span*.37,.024),(-.88,0,0)],.035,col)
        if jet: ell((.09,-r*.95,s*span*.39),(.20,r*.53,r*.50),shade(col,.94))
        # side cheatline and cabin window band
        ell((.04,.018,s*r*.87),(.67,.019,r*.17),blend(trim,col,.25))
        ell((.70,.051,s*r*.62),(.16,.052,r*.32),shade(glass,.85))
    prism([(-.96,.02),(-.91,r+U(.20,.34)),(-.82,r+.24),(-.59,.03)],-.018,.018,trim)
    if not jet:
        ell((.27,r*.83,0),(.25,r*.62,r*.82),glass)
        ell((1.01,0,0),(.025,.28,.03),shade(black,1.4))
    if chance(.2):
        for s in [-1,1]: wheel(-.12,-.20,s*.19,.06,.035)

def ship():
    col=vary(pick([white,white,(.19,.26,.32),(.11,.13,.14),(.55,.58,.58),(.45,.16,.12),blue]));
    upper=vary(pick([white,white,silver,beige])); w=U(.24,.36); h=U(.19,.34)
    # Pointed bow; bottom waterline narrower than deck.
    deck=[(-.94,h,-w*.72),(-.64,h,-w),(.57,h,-w*.90),(1.09,h+.045,0),(.57,h,w*.90),(-.64,h,w),(-.94,h,w*.72)]
    bot=[(x*.87,.025,z*.80) for x,y,z in deck]; face(deck,blend(upper,col,.4))
    for i in range(7):
        j=i-1; face([deck[i],deck[j],bot[j],bot[i]],col)
    v=U()
    if v<.23: # yacht with one or two triangular sails
        mast=U(.85,1.46); mx=U(-.20,.2); tube((mx,h,0),(mx,mast,0),.017,brown)
        face([(mx,h+.10,0),(mx-.006,mast-.026,0),(-.79,h+.16,.06)],upper)
        face([(mx+.045,mast-.13,0),(.82,h+.05,0),(mx+.08,h+.08,.08)],vary(white))
    elif v<.63: # freighter, fishing boat, tanker, or tug
        a=U(-.73,-.20) if chance(.70) else .16; ht=U(.23,.46)
        box(a,a+.42,h,h+ht,-w*.73,w*.73,upper)
        box(a-.025,a+.45,h+ht*.68,h+ht*.85,-w*.75,w*.75,glass)
        box(a+.07,a+.18,h+ht,h+ht+.14,-w*.30,w*.30,shade(col,.69))
        for i in range(3):
            xx=-.58+i*.42
            if xx+.18<a or xx>a+.42: box(xx,xx+.30,h,h+U(.04,.20),-w*.74,w*.74,pick([shade(brown,1.3),col,silver]))
    else: # ferry, liner, small powered boat
        width=w*.82; x0=U(-.85,-.40);x1=U(.21,.64); ht=U(.12,.25)
        for i in range(2 if chance(.63) else 1):
            box(x0,x1,h,h+ht,-width,width,upper)
            box(x0+.018,x1-.012,h+ht*.44,h+ht*.82,-width*1.013,width*1.013,glass)
            h+=ht;x0+=.11;x1-=.12;width*=.86
        if chance(.60): ell((-.23,h+.06,0),(.10,.14,.09),pick([red,silver,black]))

def ears(cx,cy,cz,sx,sy,sz,c,inside):
    # A triangular pinna with thickness and an inner surface.
    p=[(cx,cy-.01,cz-sz),(cx,cy-.01,cz+sz),(cx+sx,cy+sy,cz+sz*.24*np.sign(cz))]
    back=(cx-.08,cy+.02,cz)
    for i in range(3):face([p[i],p[i-1],back],c)
    face(p,c); mid=np.array(p).mean(0); face((np.array(p)-mid)*.65+mid+np.array([.006,0,0]),inside)

def mammal(k,portrait):
    pet=k in (3,5); cat=k==3; dog=k==5; deer=k==4; horse=k==7
    if cat:
        coat=vary(pick([(.54,.52,.46),(.46,.42,.36),(.78,.54,.31),(.87,.86,.82),(.20,.20,.18),(.62,.59,.55),(.71,.64,.52)]))
    elif dog:
        coat=vary(pick([gold,beige,chestnut,white,(.90,.84,.70),(.27,.21,.17),(.17,.16,.15),(.57,.49,.40)]))
    elif deer: coat=vary(pick([(.58,.42,.29),(.48,.39,.31),(.66,.46,.30),(.56,.49,.40)]))
    else: coat=vary(pick([chestnut,brown,(.27,.21,.17),white,(.66,.44,.26),(.21,.18,.16)]))
    dark=shade(coat,U(.37,.68)); pale=blend(coat,white,U(.27,.68)); spots=U()
    bh=U(.27,.35) if pet else U(.25,.32); bw=U(.23,.31)
    belly=U(.32,.50) if pet else U(.66,.87); by=belly+bh
    bx=-.16; bl=U(.67,.83); sit=pet and chance(.31); lying=pet and not sit and chance(.19)
    if lying: by=bh+.10;belly=.1
    if sit: by-=.11
    if sit: oval((-.64,.23,0),(.34,by+.19,0),bh*1.17,bw,coat)
    else: ell((bx,by,0),(bl,bh,bw),coat)
    # Gait, hocks and feet; offsets of fore and hind legs are independent.
    phase=U(-.29,.29); spread=U(.0,.17); leg=U(.066,.11) if pet else U(.047,.072)
    for s in [-1,1]:
        for f in [-1,1]:
            x=bx+f*bl*.66; z=s*bw*.65;hip=by+.036
            if sit and f<0: hip=.32
            step=s*f*phase+U(-spread,spread); bend=-.12 if f<0 else .015
            knee=(x+bend+step*.42,hip*.48,z*1.10); foot=(x+step+( .29 if lying or (sit and f<0) else 0),.061,z*1.22)
            oval((x,hip,z),knee,leg*(1.62 if f<0 else 1.17),leg*1.17,coat)
            oval(knee,foot,leg*.82,leg*.84,coat if not horse else blend(coat,dark,.58))
            ell((foot[0]+.028,.052,foot[2]),(leg*1.25,leg*.68,leg*.94),coat if pet else dark)
    if dog or horse or cat:
        ta=(bx-bl*.85,by+bh*.31,0); tz=U(-.2,.2)
        if horse:
            tb=(ta[0]-.20,by-.08,tz*.6);tc=(tb[0]-.12,by-.56,tz)
            oval(ta,tb,.12,.10,dark);oval(tb,tc,.095,.09,dark)
        else:
            tilt=U(-.72,1.1);tb=(ta[0]-.29,ta[1]+tilt*.31,tz);tc=(tb[0]-.22,tb[1]+tilt*.28,tz+.08)
            oval(ta,tb,.084,.074,coat);oval(tb,tc,.065,.060,coat)
    else: ell((bx-bl*.88,by+.10,0),(.16,.12,.11),pale)
    if not pet:
        grazing=(not portrait) and chance(.23); base=(.34,by+.11,0)
        hx=U(.64,.88) if not grazing else 1.03; hy=by+U(.50,.75) if not grazing else by-.38
        necktop=(hx-.09,hy+.025,0)
        oval((base[0]-.10,base[1]-.07,0),necktop,.19 if horse else .14,.17 if horse else .13,coat)
        if horse: oval((base[0]-.14,base[1]+.14,0),(hx-.18,hy+.14,0),.069,.078,dark)
        hr=.177 if horse else .153
        jaw=(hx+.32,hy-.25,0) if horse else (hx+.27,hy-.098,0)
        oval((hx-.10,hy+.11,0),jaw,hr,hr*.83,coat)
        ell(jaw,(hr*.78,hr*.66,hr*.71),blend(dark,coat,.32))
        for s in [-1,1]:
            ell((hx+.045,hy+.028,s*hr*.72),(.040,.035,.021),black)
            tip=(hx-.10,hy+(.28 if horse else .18),s*(.15 if horse else .32))
            oval((hx-.05,hy+.07,s*.08),tip,.068,.075,coat)
            if deer and spots<.27:
                root=(hx-.09,hy+.13,s*.09); a=(hx-.24,hy+.39,s*.26);top=(hx-.28,hy+.56,s*.34)
                tube(root,a,.027,beige);tube(a,top,.019,beige);tube(a,(hx-.095,hy+.51,s*.29),.017,beige)
        if horse and spots<.33: oval((hx+.091,hy+.09,0),(jaw[0]+.06,jaw[1]+.055,0),.055,.059,pale)
        if horse and not portrait and chance(.23):
            shirt=paint(); ell((-.18,by+bh+.30,0),(.14,.26,.14),shirt)
            ell((-.16,by+bh+.60,0),(.105,.13,.099),beige);ell((-.17,by+bh+.69,0),(.13,.071,.11),black)
            for s in [-1,1]:oval((-.21,by+bh+.12,s*.1),(-.05,by-.16,s*bw),.077,.064,blue)
        return (hx+.075,hy+.09,0),.83 if horse else .75
    # Pet head and facial markings. X points towards the nose.
    hx=U(.49,.64); hy=by+(U(.17,.34) if not sit else .32); hr=U(.238,.294) if cat else U(.245,.345)
    if lying: hy=by+U(.08,.22)
    if dog: oval((.23,by-.08,0),(hx,hy+.06,0),hr*.85,hr*.86,coat)
    ell((hx,hy,0),(hr*.88,hr,hr*(1.12 if cat else 1.04)),coat)
    droop=dog and chance(.70)
    for s in [-1,1]:
        z=s*hr*.75
        if droop:
            oval((hx-.023,hy+hr*.73,z),(hx-.048,hy-hr*.83,z*1.38),hr*.52,hr*.39,dark if spots<.60 else coat)
        else:
            ears(hx+.01,hy+hr*.49,z, -hr*U(.04,.42),hr*U(.75,1.32),hr*.44,coat,blend(coat,(.75,.47,.44),.6))
        # Both eye sockets are on the forward hemisphere, partly facing outward.
        ey=(hx+hr*.69,hy+hr*.18,s*hr*.62); eye=hr*(.17 if cat else .145)
        if dog and spots<.25: ell((hx+hr*.59,hy+hr*.18,s*hr*.61),(hr*.27,hr*.35,hr*.35),dark)
        ell(ey,(eye*.76,eye*.81,eye*1.12),black)
        if cat:
            ell((ey[0]+eye*.51,ey[1],ey[2]+s*eye*.36),(eye*.42,eye*.62,eye*.81),pick([(.63,.63,.26),(.52,.62,.34),gold]))
            ell((ey[0]+eye*.83,ey[1],ey[2]+s*eye*.49),(eye*.19,eye*.54,eye*.24),black)
        else: ell((ey[0]+eye*.53,ey[1]+eye*.29,ey[2]+s*eye*.41),(eye*.22,eye*.22,eye*.20),silver)
        if cat and spots<.57:
            for j in range(3):
                oval((hx+hr*(.38+j*.16),hy+hr*(.78-j*.10),s*hr*(.26+j*.03)),(hx+hr*(.67+j*.07),hy+hr*(.41-j*.09),s*hr*(.34+j*.10)),hr*.051,hr*.055,dark)
    sn=U(.83,1.20) if cat else U(1.12,1.73)
    if dog:
        ell((hx+hr*(sn-.43),hy-hr*.22,0),(hr*.69,hr*.47,hr*.57),pale if spots<.32 else coat)
    else:
        for s in [-1,1]:ell((hx+hr*.75,hy-hr*.24,s*hr*.22),(hr*.37,hr*.32,hr*.38),pale)
    nr=hr*(.20 if cat else .34)
    ell((hx+hr*(sn+.08),hy-hr*.13,0),(nr*.63,nr*.66,nr),blend(dark,(.70,.39,.37),.60) if cat else black)
    if cat and spots<.55:
        for j in range(5):
            xx=bx+(j-2)*bl*.26
            for s in [-1,1]:
                oval((xx-.052,by+bh*.91,s*bw*.29),(xx+.048,by+bh*.14,s*bw*.98),.034,.040,dark)
    if dog and spots<.30:
        ell((bx-.16,by+bh*.56,0),(bl*.61,bh*.51,bw*.94),dark)
    return (hx+.06,hy+hr*.05,0),hr*U(2.65,3.50)

def bird(flying):
    feather=vary(pick([brown,(.43,.43,.40),(.17,.19,.20),(.80,.77,.68),(.30,.43,.52),(.62,.38,.21),(.48,.58,.22),(.72,.64,.33)]))
    dark=shade(feather,U(.44,.72));breast=blend(feather,pick([white,beige,gold,(.77,.35,.18)]),U(.18,.72)); beak=pick([gold,brown,black])
    hy=U(.22,.45) if not flying else .10; hx=U(.29,.51); rr=U(.16,.26)
    oval((-.46,-.16,0),(.35,.14,0),.255,.227,breast)
    ell((hx,hy,0),(rr*1.08,rr,rr*.94),feather)
    for s in [-1,1]:
        ell((hx+rr*.37,hy+rr*.18,s*rr*.84),(.040,.039,.022),black)
        if not flying: oval((-.61,-.22,s*.085),(.29,.16,s*.11),.163,.125,feather)
        else:
            length=U(.78,1.19);lift=U(-.24,.34)
            wing([(.28,0,0),(.19,s*.47,lift*.6),(-.28,s*length,lift),(-.41,s*length*.90,lift*.9),(-.39,s*.51,lift*.5),(-.24,0,0)],.07,feather)
    length=U(.15,.37);tip=(hx+rr+length,hy-.038,0)
    for s in [-1,1]:face([(hx+rr*.72,hy+.035,s*.05),tip,(hx+rr*.79,hy-.057,s*.058)],beak)
    face([(hx+rr*.72,hy+.034,-.05),(hx+rr*.72,hy+.034,.05),tip],beak)
    tipx=U(-1.0,-.69)
    wing([(-.25,-.12,.09),(tipx,-.14,-.10),(tipx-.02,.14,-.10),(-.25,.12,.09)],-.17,dark)
    if not flying:
        for s in [-1,1]:
            p=(-.01,-.23,s*.097);end=(-.08,-.41,s*.111);tube(p,end,.022,beak);tube(end,(.09,-.41,s*.14),.017,beak)

def frog():
    skin=vary(pick([(.35,.47,.18),(.44,.48,.24),(.46,.37,.25),(.54,.46,.29),(.29,.55,.19),(.47,.60,.19),(.32,.32,.23),(.54,.40,.26)])); dark=shade(skin,.62); light=blend(skin,beige,.47)
    ell((-.12,.23,0),(.58,.25,.37),skin);ell((.37,.31,0),(.37,.219,.365),skin)
    ell((.48,.216,0),(.268,.119,.324),light)
    for s in [-1,1]:
        oval((-.54,.14,s*.29),(-.03,.17,s*.51),.139,.123,skin)
        oval((-.04,.17,s*.51),(-.69,.044,s*.50),.075,.077,skin)
        oval((.22,.24,s*.24),(.015,.069,s*.46),.084,.078,skin)
        oval((.015,.069,s*.46),(.36,.035,s*.56),.060,.052,light)
        for j in range(3):
            tube((.34,.035,s*.56),(.47+j*.030,.031,s*(.51+j*.065)),.018,light)
        ell((.45,.448,s*.225),(.127,.136,.113),skin)
        ell((.494,.48,s*.285),(.087,.088,.072),pick([brown,(.68,.54,.23),(.40,.40,.20)]))
        ell((.521,.48,s*.324),(.052,.035,.043),black)
        ell((.53,.501,s*.353),(.017,.015,.010),white)
        oval((.63,.267,s*.22),(.36,.257,s*.345),.012,.014,dark)
    for j in range(int(U(3,13))):
        xx=U(-.58,.20);zz=U(-.28,.28); t=1-((xx+.12)/.58)**2-(zz/.37)**2
        if t>0: ell((xx,.229+.25*t**.5,zz),(U(.036,.079),.012,U(.025,.054)),dark)

size=44
Y,X=np.mgrid[0:size,0:size]
X=(X+.5)/size;Y=(Y+.5)/size

def noise(gridx,gridy):
    a=N(size=(gridy+1,gridx+1));x=X*gridx;y=Y*gridy
    i=x.astype(int);j=y.astype(int);fx=x-i;fy=y-j
    return ((1-fx)*((1-fy)*a[j,i]+fy*a[j+1,i])+fx*((1-fy)*a[j,i+1]+fy*a[j+1,i+1]))*1.40

def layer(kind,n1,n2,n3):
    if kind=='sky':
        color=pick([(.50,.69,.86),(.42,.61,.79),(.56,.66,.76),(.71,.74,.75),(.78,.82,.85),(.60,.73,.83)])
        color=vary(color); clouds=np.clip((n2+U(-1.2,.3))*.4,0,.92)
        amount=U(.02,.16);return color+(1-color)*(.14*Y+clouds*U(.12,.72))[:,:,None]+(amount*.32*n1+.02*n2)[:,:,None]
    if kind=='water':
        color=vary(pick([(.24,.39,.50),(.25,.47,.55),(.36,.46,.51),(.41,.52,.59),(.35,.36,.31),(.48,.61,.69)]))
        return color+(.035*n2+.024*n1+U(.03,.11)*noise(7,33)+.025*noise(4,28))[:,:,None]
    if kind in ('grass','tree','dirt'):
        if kind=='grass':
            color=vary(pick([(.40,.49,.23),(.49,.57,.30),(.36,.46,.23),(.54,.53,.30),(.49,.45,.30),(.27,.40,.20)]));t=U(.035,.093)
        elif kind=='tree':
            color=vary(pick([(.25,.34,.20),(.33,.37,.21),(.23,.32,.22),(.44,.46,.30),(.32,.32,.23),(.30,.40,.24)]));t=U(.048,.16)
        else:
            color=vary(pick([(.44,.40,.31),(.52,.45,.32),(.40,.36,.30),(.61,.54,.40),(.35,.33,.29)]));t=U(.04,.12)
        return color+((.022+t*.51)*n1+(.018+t*.63)*n2+t*n3)[:,:,None]+(n2*.023+n3*.017)[:,:,None]*np.array([1,.25,-1])
    if kind in ('road','wall','cloth'):
        if kind=='road':
            color=vary(pick([(.41,.42,.43),(.53,.52,.50),(.37,.36,.35),(.65,.62,.55)]));t=U(.007,.040)
        elif kind=='wall':
            color=vary(pick([(.63,.60,.52),(.72,.68,.61),(.45,.39,.34),(.46,.46,.44),(.71,.72,.70),(.38,.43,.46)]));t=U(.014,.069)
        else:
            color=vary(pick([beige,(.45,.37,.30),(.62,.55,.46),(.40,.48,.52),(.65,.58,.54),(.41,.46,.35),(.49,.32,.29)]));t=U(.03,.105)
        return color+(U(.04,.14)*n1+t*n2+t*.55*n3)[:,:,None]
    color=vary(pick([white,(.49,.49,.49),(.18,.18,.17),(.81,.82,.80)]));return color+(.028*n1+.014*n2)[:,:,None]

def background(k,flying,ground):
    n1=noise(2,2);n2=noise(5,5);n3=.68*noise(15,15)+.46*N(size=(size,size))
    r=U();ground=np.clip(ground,.22,.94)
    if k==0:
        if flying and r<.78: kinds=('sky','sky')
        elif flying and r<.88:kinds=('sky','tree')
        else:kinds=(pick(['sky','tree','wall']),pick(['road','grass','dirt']))
    elif k==8: kinds=(pick(['sky','sky','sky','tree','wall','water']), 'water')
    elif k in (1,9): kinds=(pick(['tree','tree','wall','wall','sky']),pick(['road','road','road','dirt','grass']))
    elif k==2:
        if flying and r<.80:kinds=('sky','sky')
        else:kinds=(pick(['tree','sky','grass','tree','dirt']),pick(['tree','grass','tree','water','dirt']))
    elif k in (4,7):
        if k==7 and r<.18:kinds=('wall',pick(['dirt','grass']))
        else:kinds=(pick(['tree','tree','grass','sky','dirt']),pick(['grass','grass','dirt','grass','dirt']))
    elif k==6:kinds=(pick(['tree','grass','dirt','water','dirt']),pick(['grass','dirt','dirt','water','tree']))
    elif (k==3 and r<.74) or (k==5 and r<.49):kinds=(pick(['wall','cloth','cloth']),pick(['wall','cloth','cloth','dirt']))
    else:kinds=(pick(['grass','tree','wall','dirt']),pick(['grass','grass','dirt','road']))
    if chance(.055):kinds=('plain','plain')
    a=layer(kinds[0],n1,n2,n3)
    if kinds[0]=='wall':
        # Out-of-focus verticals, windows and siding, not a scene photograph.
        for i in range(int(U(1,9))):
            xx=U(-.15,.95);ww=U(.023,.32);yy=U(-.32,.67);hh=U(.12,1.1)
            a+=((X>xx)&(X<xx+ww)&(Y>yy)&(Y<yy+hh))[:,:,None]*U(-.16,.16)
    if kinds[0]==kinds[1] and chance(.84):return np.clip(a,0,1),kinds
    b=layer(kinds[1],n1,n2,n3)
    if kinds[1]=='road' and chance(.18):
        slope=U(-.25,.25); yy=U(.70,.97);b+=(np.abs(Y-yy-(X-.5)*slope)<U(.007,.019))[:,:,None]*.22
    horizon=U(.30,.64) if k==8 else ground-U(.10,.52)
    irregular=U(.012,.065) if 'tree' in kinds else U(0,.018)
    d=Y-horizon-(X-.5)*U(-.10,.10)+n2*irregular
    mix=np.clip(.5+d/U(.015,.11),0,1)[:,:,None]
    return np.clip(a*(1-mix)+b*mix,0,1),kinds

def perspective(p,cam,ctr,dist,scale,offset):
    w=(p-ctr)@cam.T
    den=1.-w[:,2:3]/dist
    # Use an affine camera per object with a small between-part perspective correction.
    return w[:,:2]*scale/np.maximum(den,.30)+offset,w[:,2]

def render(k):
    global parts
    parts=[]
    portrait=k in (3,4,5,7) and chance(.42 if k in (3,5) else .13)
    flying=(k==0 and chance(.77)) or (k==2 and chance(.23))
    if k==0:plane()
    elif k in (1,9):car(k==9)
    elif k==2:bird(flying)
    elif k==6:frog()
    elif k==8:ship()
    else: focus,headsize=mammal(k,portrait)
    # Elevation and azimuth are chosen before inspecting any real image.
    az=U(.04,pi-.04);el=U(.055,.43);roll=N(0,.035)
    if k in (1,9,8):az=U(.06,pi-.06) if chance(.82) else U(.02,.32)
    if k==0:el=U(-.68,.41) if flying else U(.03,.28);roll=N(0,.24 if flying else .036)
    if k==2:el=U(-.35,.37) if flying else U(-.18,.48);roll=N(0,.22 if flying else .095)
    if k==6:el=U(.19,1.09);roll=N(0,.13)
    if k in (3,5):el=U(.03,.76)
    if portrait:az=U(.004,.90) if chance(.83) else U(.9,1.62);el=U(-.10,.45)
    if k in (2,3,4,5,6,7) and not portrait and chance(.19): az=U(.025,.42)
    if chance(.5):az=-az
    ca,sa,ce,se=np.cos(az),np.sin(az),np.cos(el),np.sin(el)
    u=np.array([-sa,0,ca]);v=np.array([ca*se,-ce,sa*se]);d=np.array([ca*ce,se,sa*ce]);cr,sr=np.cos(roll),np.sin(roll)
    cam=np.stack([cr*u+sr*v,cr*v-sr*u,d]);dist=U(6.2,13.)
    # Project all primitives first to frame the individual rendering.
    lows=[]; highs=[]; projected=[]
    for mode,p,r,c in parts:
        if mode==0:
            cp=p@cam.T; m=cam@r; radii=np.sqrt((m[:2]**2).sum(1));lo=cp[:2]-radii;hi=cp[:2]+radii
            projected.append((mode,cp,m,c))
        else:
            q=p@cam.T;lo=q[:,:2].min(0);hi=q[:,:2].max(0);projected.append((mode,q,None,c))
        lows.append(lo);highs.append(hi)
    lo=np.min(lows,0);hi=np.max(highs,0);origin=(lo+hi)/2
    span=(hi-lo).max();scale=size*U(.64,1.055)/max(.48,span)
    if k in (0,2,8) and chance(.23):scale*=U(.53,.84)
    if portrait:
        origin=(np.array(focus)@cam.T)[:2]; scale=size*U(.77,1.09)/headsize
    elif chance(.13):scale*=U(1.10,1.40)
    off=size*(np.array([.5,.5])+np.clip(N(0,.048,2),-.14,.14));ground=((hi-origin)*scale+off)[1]/size
    img,kinds=background(k,flying,ground); depth=np.full((size,size),-1.e4); mask=np.zeros((size,size),bool)
    if not flying and not portrait and k not in (8,2) and chance(.83):
        width=np.clip((hi[0]-lo[0])*scale/size,.2,1.5)*.48
        img*=1-U(.10,.36)*np.exp(-((X-off[0]/size)/width)**2-((Y-ground)/U(.033,.081))**2)[:,:,None]
    light=np.array([U(-.8,.8),U(-1.35,-.45),U(.35,1.1)]);light/=np.linalg.norm(light); amb=U(.66,.88);dif=U(.20,.43)
    # Small bounded primitive rasterizer. It is independent of the scoring program.
    for mode,p,m,col in projected:
        if mode==0:
            pers=1/(1-p[2]/dist);center=(p[:2]-origin)*scale*pers+off;mm=m.copy();mm[:2]*=scale*pers
            S=mm@mm.T; A,B,C=S[0,0],S[0,1],S[1,1];det=A*C-B*B
            if det<1.e-11:continue
            radii=np.sqrt([A,C]); l=center-radii;h=center+radii
        else:
            v=(p[:,:2]-origin)*scale/(1-p[:,2:3]/dist)+off;l=v.min(0);h=v.max(0)
        x0=max(0,int(math.ceil(l[0]-.5)));x1=min(size,int(math.floor(h[0]-.5))+1)
        y0=max(0,int(math.ceil(l[1]-.5)));y1=min(size,int(math.floor(h[1]-.5))+1)
        if x0>=x1 or y0>=y1:continue
        xs=np.arange(x0,x1)[None,:]+.5;ys=np.arange(y0,y1)[:,None]+.5
        if mode==0:
            a=C/det;b=-B/det;c=A/det;dx=xs-center[0];dy=ys-center[1]
            j=a*dx+b*dy;t=b*dx+c*dy;rem=1-j*dx-t*dy;inside=rem>=0
            if not inside.any():continue
            sd=np.sqrt(max(1.e-10,S[2,2]-a*S[2,0]**2-2*b*S[2,0]*S[2,1]-c*S[2,1]**2)); root=np.sqrt(np.maximum(.00005,rem));kx=S[2,0]*a+S[2,1]*b;ky=S[2,0]*b+S[2,1]*c
            z=p[2]+kx*dx+ky*dy+sd*root
            nx=(j*sd/root-kx)*scale*pers;ny=(t*sd/root-ky)*scale*pers
            lam=(nx*light[0]+ny*light[1]+light[2])/np.sqrt(1+nx*nx+ny*ny)
        else:
            e=np.roll(v,-1,axis=0)-v
            cross=e[:,0,None,None]*(ys-v[:,1,None,None])-e[:,1,None,None]*(xs-v[:,0,None,None]);inside=(cross>=-1.e-6).all(0)|(cross<=1.e-6).all(0)
            if not inside.any():continue
            v0=v[0]; h1=v[1]-v0;h2=v[2]-v0;det=h1[0]*h2[1]-h1[1]*h2[0]
            if abs(det)<1.e-8:continue
            dz1=p[1,2]-p[0,2];dz2=p[2,2]-p[0,2]; kx=(h2[1]*dz1-h1[1]*dz2)/det;ky=(h1[0]*dz2-h2[0]*dz1)/det
            z=p[0,2]+kx*(xs-v0[0])+ky*(ys-v0[1]);nx=-kx*scale;ny=-ky*scale
            lam=(nx*light[0]+ny*light[1]+light[2])/np.sqrt(1+nx*nx+ny*ny)
        cut=np.s_[y0:y1,x0:x1];vis=inside&(z>=depth[cut]-.0002)
        depth[cut][vis]=z[vis];mask[cut]|=vis;shadeval=amb+dif*np.maximum(lam,-.43)
        img[cut][vis]=col*(shadeval[vis,None] if np.ndim(shadeval) else shadeval)
    # Lighting, coat texture, and camera exposure are nuisance variables.
    texture=U(.013,.079) if k in (2,3,4,5,6,7) else U(.008,.041)
    tex=(texture*(.68*noise(15,15)+.58*N(size=(size,size)))+.045*noise(5,5)+.032*noise(2,2))*mask
    img+=tex[:,:,None];img*=U(.78,1.16)*U(.94,1.055,3);img+=U(-.065,.085)
    if chance(.18):
        gray=img@np.array([.30,.55,.15]);img=gray[:,:,None]+U(.05,.68)*(img-gray[:,:,None])
    img=np.clip(img,0,1)
    # Bilinear pixel footprint and optical low-pass before sampling at 32x32.
    mix=U(.19,.75);p=np.pad(img,((1,1),(1,1),(0,0)),mode='edge');img=(1-mix)*img+mix*.25*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])
    pix=(np.arange(32)+.5)*size/32-.5;i=pix.astype(int);f=pix-i
    img=(1-f[None,:,None])*img[:,i]+f[None,:,None]*img[:,i+1];img=(1-f[:,None,None])*img[i]+f[:,None,None]*img[i+1]
    soft=(1-f[None,:])*mask[:,i]+f[None,:]*mask[:,i+1];soft=(1-f[:,None])*soft[i]+f[:,None]*soft[i+1]
    return img.transpose(2,0,1).astype(np.float32), soft.astype(np.float32)

# The numbers below are hand-set soft likelihood cues. For example, horizontal
# structure is common on roads and water; tall pointed ears are common on cats.
classifier='''
def rules(x):
    r,g,b=x[:,0],x[:,1],x[:,2]; n=len(x); y=.29*r+.55*g+.16*b
    def ramp(z,a,b): return np.clip((z-a)/(b-a),0,1)
    def avg(z,region): return z[:,region[0],region[1]].mean((1,2))
    def whole(z):return z.mean((1,2))
    top=(slice(0,8),slice(None)); bot=(slice(25,32),slice(None)); cen=(slice(8,25),slice(6,26)); mid=(slice(9,24),slice(None))
    side=np.ones((32,32),np.float32);side[5:27,5:27]=0
    def outside(z):return (z*side).sum((1,2))/side.sum()
    dx,dy=grad(blur(y));fx,fy=grad(y);mag=np.sqrt(dx*dx+dy*dy);fine=np.sqrt(fx*fx+fy*fy)
    # Feathered color names. Requiring a channel ordering rejects, for example,
    # blue concrete in shadow as grass, or yellowish fur as a green leaf.
    blue=ramp((b-r)/(.28+y),.035,.24)*ramp((b-g)/(.28+y),.003,.125)
    green=ramp((g-b)/(.25+y),.030,.25)*ramp((g-r+.013)/(.24+y),-.018,.105)
    olive=ramp((g-b)/(.32+y),.05,.25)*ramp((g-r)/(.26+y),-.091,.012)
    vegetation=np.maximum(green,.72*olive)
    warm=ramp((r-b)/(.26+y),.075,.37)*ramp((r-g)/(.28+y),.01,.16)
    vivid=ramp((np.maximum(r,np.maximum(g,b))-np.minimum(r,np.minimum(g,b)))/(.28+y),.26,.59)
    red=ramp((r-g)/(.28+y),.22,.58)*ramp((r-b)/(.28+y),.24,.58)
    gray=1-ramp((np.maximum(r,np.maximum(g,b))-np.minimum(r,np.minimum(g,b)))/(.28+y),.040,.24)
    bt=avg(blue,top);bb=avg(blue,bot);bg=outside(blue);vg=outside(vegetation);vb=avg(vegetation,bot);vc=avg(vegetation,cen)
    wc=avg(warm,cen);wg=outside(warm);gt=avg(gray,top);gb=avg(gray,bot);gc=avg(gray,cen)
    ed=avg(mag,cen); eb=avg(mag,bot);et=avg(mag,top)
    sky=bt*ramp(.088-et,0,.068); sea=bb*(1-.60*vb); grass=vb*(1-.43*avg(blue,bot))
    flat=ramp(.060-outside(mag),0,.040)
    smoothTop=ramp(.057-et,0,.043);smoothBot=ramp(.060-eb,0,.045)
    dyabs=np.abs(dy);dxabs=np.abs(dx)
    horiz=(avg(dyabs,mid)-avg(dxabs,mid))/(.015+avg(dyabs+dxabs,mid)); hori=np.clip(horiz*3,-1,1)
    align=whole(np.abs(dx*dx-dy*dy)/(.00003+mag)) / (.002+whole(mag))
    axial=np.clip((align-.65)*5,-.85,1.2)
    # Concentration of mass in a horizontal band, and coherent horizontal lines.
    profile=(mag+.42*fine).mean(2); central=profile[:,8:25].mean(1);outer=(profile[:,:6].mean(1)+profile[:,26:].mean(1))*.5
    band=np.clip((central-outer)/(.034+central+outer)*2.3,-1,1.2)
    row=np.abs(dy[:,:,3:-3].mean(2)); line=np.clip(row[:,7:27].max(1)/(.036+ed)-.65,-.65,1.35)
    road=gb*smoothBot*(1-.75*bb);skyline=sky*avg(gray,mid); water=sea*(.6+.4*np.maximum(0,hori))
    shade=np.clip((avg(y,top)-avg(y,cen))/.28,-1,1); lower=np.clip((avg(y,bot)-avg(y,cen))/.25,-1,1)
    topbright=ramp(avg(y,top),.52,.86)*smoothTop
    # Symmetry of a central face: compare both sides locally, under small shifts.
    sym=np.zeros(n,np.float32)-1
    for cx in (13,15,17,19):
        a=y[:,7:25,cx-10:cx];d=y[:,7:25,cx:cx+10][:,:,::-1];
        a=a-a.mean((1,2),keepdims=True); d=d-d.mean((1,2),keepdims=True)
        cc=whole(a*d)/np.sqrt((.002+whole(a*a))*(.002+whole(d*d)))
        sym=np.maximum(sym,cc)
    # Evidence for dark eye pairs and a darker nose as in a frontal pet head.
    sm=blur(y); hp=blur(blur(blur(sm)))-sm
    eye=np.zeros(n,np.float32); nosepair=np.zeros(n,np.float32)
    for gap in (5,8,11):
        xx=np.arange(10,22)-gap//2; a=hp[:,6:20,xx];d=hp[:,6:20,xx+gap];pair=np.minimum(a,d)
        nose=hp[:,6+gap//2:20+gap//2,xx+gap//2]
        eye=np.maximum(eye,pair.max((1,2)));nosepair=np.maximum(nosepair,(np.minimum(pair,nose)).max((1,2)))
    facecue=ramp(eye,.035,.12)*ramp(sym,.16,.74);muzzle=ramp(nosepair,.025,.096)*ramp(sym,.12,.70)
    # Wheel cue: dark upper arc and hub / tire reversal. Search for a pair
    # near one road plane; softness reflects aliasing of wheels at this scale.
    wheelcue=np.zeros(n,np.float32)
    for rad,delta in ((2,6),(3,10),(4,13)):
        yy=np.arange(15,27);xx=np.arange(rad,32-rad-delta)
        def gather(oy,ox):return sm[:,yy[:,None]+oy,xx[None,:]+ox]
        def tire(off):
            center=gather(0,off);ring=.34*gather(-rad+1,off)+.33*gather(0,off-rad+1)+.33*gather(0,off+rad-1)
            surround=.4*gather(-rad-1,off)+.3*gather(1,off+rad)+.3*gather(1,off-rad)
            return .72*(surround-ring)+.28*(center-ring)
        wheels=np.minimum(tire(0),tire(delta));wheelcue=np.maximum(wheelcue,wheels.max((1,2)))
    wheels=ramp(wheelcue,.037,.145)
    # Mottled close-up texture versus fine parallel/rectilinear structure.
    texture=ramp(avg(fine,cen)-.58*ed,.020,.080)
    clutter=ramp(outside(fine),.040,.115)
    brownfield=np.minimum(1,vg+.70*wg);colorObj=avg(vivid,cen);patch=avg(red,cen)
    # Relative adjustments in ten-class evidence units. Scene and morphology
    # provide independent evidence: none of these are hard decision branches.
    s=np.zeros((n,10),np.float32)
    s[:,0]=1.40*sky+.68*bg+.50*flat+.33*smoothTop+.28*band+.22*shade-.48*grass-.48*clutter-.43*wc-.42*water-.20*axial
    s[:,8]=1.50*water+.35*bt+.37*np.minimum(bt,bb)+.34*hori+.38*line+.29*smoothTop+.20*band+.17*axial-.65*grass-.55*wc-.30*facecue
    s[:,1]=.56*road+.48*wheels+.45*band+.33*hori+.33*lower+.28*axial+.29*gc+.31*colorObj+.35*patch-.62*water-.42*vc-.24*facecue
    s[:,9]=.46*road+.35*wheels+.55*axial+.23*line+.23*skyline+.33*gc+.25*gt+.24*colorObj+.24*patch-.46*water-.50*vc-.22*facecue
    s[:,2]=.55*vg+.31*sky+.18*flat+.26*colorObj+.26*vc+.17*band-.30*axial-.22*hori-.24*wheels-.22*gc
    s[:,3]=.35*wc+.23*wg+.27*texture+.48*facecue+.21*muzzle+.20*sym+.26*(1-vg)-.51*sky-.32*water-.31*axial-.20*band-.21*grass
    s[:,5]=.60*wc+.22*wg+.26*texture+.33*facecue+.48*muzzle+.16*sym+.22*grass-.47*sky-.37*water-.32*axial-.21*band
    s[:,4]=.75*vg+.52*grass+.53*wc+.20*wg+.22*clutter+.17*texture-.49*sky-.46*water-.32*axial-.26*facecue-.20*colorObj
    s[:,7]=.52*grass+.57*wc+.24*vg+.14*wg+.15*shade+.16*topbright+.17*lower-.32*water-.24*axial-.30*vc-.25*facecue
    s[:,6]=.81*vc+.54*brownfield+.24*clutter+.35*texture+.24*wg+.19*wc-.71*sky-.40*axial-.30*facecue-.40*hori-.19*band
    # Offsets reflect the number of positive generic (non-discriminative)
    # terms in each rule, not measurements made on images.
    return s-np.array([.66,.74,.32,.55,.69,.60,.65,.48,.48,.64],np.float32)

def foreground(x):
    # Estimate background affinity from the outside edges of THIS photo.
    # Consider several anchors, allowing sky and ground to coexist. An isolated
    # object crossing a border should not be explained by a single anchor pixel.
    r,g,b=x[:,0],x[:,1],x[:,2]; y=.29*r+.55*g+.16*b
    a=pool(blur(y),2);chroma=pool(blur((r-b)/(.35+y)),2);leaf=pool(blur((g-.55*r-.45*b)/(.30+y)),2)
    dx,dy=grad(y);t=pool(blur(np.sqrt(dx*dx+dy*dy)),2)
    f=np.stack([a,1.07*chroma,1.38*leaf,1.65*t],1);co=f.reshape(len(x),4,8,2,8,2).mean((3,5))
    near=np.full_like(a,100);nextnear=np.full_like(a,100);pos=np.arange(16,dtype=np.float32)/15
    for i,j in ((0,0),(0,2),(0,5),(0,7),(2,0),(2,7),(4,0),(4,7),(6,0),(6,7),(7,0),(7,2),(7,5),(7,7)):
        d=((f-co[:,:,i,j,None,None])**2).sum(1)
        d+=.022*(pos[:,None]-i/7)**2+.003*(pos[None,:]-j/7)**2
        nextnear=np.minimum(nextnear,np.maximum(d,near));near=np.minimum(near,d)
    novelty=np.sqrt(.63*near+.37*nextnear);novelty=blur(blur(np.tanh(3.3*novelty)))
    novelty-=novelty.mean((1,2),keepdims=True);s=pool(novelty,2).reshape(len(x),64)
    return s/np.sqrt((s*s).sum(1,keepdims=True)+.40)

def lookup(d,fg):
    # Fixed matrix dimensions even on singleton batches: summation order is
    # independent of when a photo arrives. No other row enters its lookup.
    db=CONSTANTS['atlas']; norm=CONSTANTS['lengths']; masks=CONSTANTS['silhouette']
    m=np.zeros((32,64),np.float32);m[:len(d)]=fg
    q=np.empty((32,d.shape[1]),np.float32);q[:len(d)]=d;q[len(d):]=0
    qn=(q*q).sum(1,keepdims=True); s=np.empty((len(d),10),np.float32)
    for k in range(10):
        dist=(qn+norm[k][None,:]-2*(q@db[k].T)-.19*(m@masks[k].T))[:len(d)]
        best=np.partition(dist,15,axis=1)[:,:16]; nearest=best.min(1)
        first=np.partition(best,3,axis=1)[:,:4].mean(1)
        s[:,k]=-(.27*nearest+.43*first+.30*best.mean(1))
    return s

def predict(x):
    out=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):
        v=np.ascontiguousarray(x[i:i+32],dtype=np.float32)
        d=desc(v);f=desc(np.ascontiguousarray(v[:,:,:,::-1]));fg=foreground(v)
        a=lookup(d,fg);b=lookup(f,fg.reshape(len(v),8,8)[:,:,::-1].reshape(len(v),64))
        # Allow either handedness but retain a contribution from both searches.
        s=22*(.71*np.maximum(a,b)+.29*np.minimum(a,b))+rules(v)
        out[i:i+len(v)]=s.argmax(1)
    return out
'''

parts=[]
def main():
    dest=Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=True)
    (dest/'solution.py').write_text(features+classifier)
    namespace={};exec(features,namespace);describe=namespace['desc']
    count=1152
    alltemplates=[]; silhouettes=[]
    for k in range(10):
        single=[]; shapes=[]
        for i in range(0,count,32):
            photos,mask=zip(*(render(k) for j in range(32)))
            single.append(describe(np.stack(photos)))
            soft=namespace['pool'](np.stack(mask),4).reshape(32,64)
            soft-=soft.mean(1,keepdims=True)
            shapes.append(soft/np.sqrt((soft*soft).sum(1,keepdims=True)+.40))
        alltemplates.append(np.concatenate(single,axis=0));silhouettes.append(np.concatenate(shapes,axis=0));print('Rendered class',k,flush=True)
    atlas=np.stack(alltemplates)
    # This reduction runs along descriptor coordinates, not over templates.
    lens=(atlas*atlas).sum(2)
    np.savez(dest/'constants.npz',atlas=atlas,lengths=lens,silhouette=np.stack(silhouettes))
if __name__=='__main__': main()
