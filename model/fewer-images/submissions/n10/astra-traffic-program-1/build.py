"""Data-free construction of geometric masks and independent pictogram templates."""
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(exist_ok=True, parents=True)
K = {}
# Antialiased drawing on a unit-radius sign. All strokes and paths below are
# authored here, in sign space (not coordinates in a photograph).
S = 192
v = (np.arange(S)+.5)*2/S-1
X,Y = np.meshgrid(v,v)

def ellipse(cx,cy,rx,ry):
    return ((X-cx)/rx)**2+((Y-cy)/ry)**2 <= 1

def poly(*pts):
    c=np.zeros(X.shape,bool)
    for (x,y),(u,v) in zip(pts,pts[1:]+pts[:1]):
        if y!=v: c ^= ((Y>=min(y,v)) & (Y<max(y,v)) & (X < x+(Y-y)*((u-x)/(v-y))))
    return c

def stroke(pts,w):
    c=np.zeros(X.shape,bool)
    for (a,b),(d,e) in zip(pts,pts[1:]):
        u,v=d-a,e-b
        t=np.clip(((X-a)*u+(Y-b)*v)/(u*u+v*v+1e-20),0,1)
        c|=(X-a-t*u)**2+(Y-b-t*v)**2<=(w/2)**2
    return c

def curve(start,*parts):
    pts=[start]; a=np.array(start)
    for p in parts:
        if len(p)==2: pts.append(p); a=np.array(p); continue
        b,c,d=np.array(p).reshape(3,2)
        pts.extend(tuple((1-t)**3*a+3*t*(1-t)**2*b+3*t*t*(1-t)*c+t**3*d) for t in np.linspace(.125,1,8)); a=d
    return pts

def drawdigits(text,var,aspect=1.,spacing=1.,weight=1.):
    # Centre lines of sans-serif transport numerals. The style alternatives
    # include straight-sided and elliptical counters, and an open-top six.
    c=np.zeros(X.shape,bool); n=len(text)
    for i,d in enumerate(text):
        width=(.42 if n==2 else .345); h=(.83 if n==2 else .77)
        advance=(width+( .09 if n==2 else .057))*spacing; cx=(i-(n-1)/2)*advance
        width*=aspect
        def line(p):
            return stroke([(cx+(x-.5)*width,(y-.5)*h) for x,y in p],h*weight*(.143 if var%3==0 else .123 if var%3==1 else .164))
        if d=='0':
            if var%2==0:
                p=curve((.5,.055),(.88,.055,.91,.23,.91,.39),(.91,.62),(.91,.85,.83,.945,.5,.945),(.14,.945,.09,.81,.09,.62),(.09,.38),(.09,.16,.2,.055,.5,.055))
            else: p=[(.5+.413*np.cos(t),.5+.443*np.sin(t)) for t in np.linspace(0,2*np.pi,49)]
        if d=='1': p=[(.16,.24),(.53,.065),(.53,.94)]
        if d=='2': p=curve((.08,.25),(.11,.015,.8,-.015,.9,.19),(1.04,.4,.73,.51,.49,.65),(.09,.932),(.94,.932))
        if d=='3': p=curve((.08,.205),(.24,-.035,.93,.015,.91,.252),(.91,.40,.71,.475,.43,.475),(.75,.462,.98,.57,.94,.73),(.92,1.,.23,1.025,.065,.79))
        if d=='5': p=curve((.88,.067),(.18,.067),(.15,.467),(.38,.36,.8,.375,.9,.54),(1.13,.93,.43,1.10,.085,.836))
        if d=='6':
            p=curve((.79 if var%2==0 else .87,.061 if var%2==0 else .132),(.53,.035,.075,.39,.09,.65),(.08,.88,.26,.954,.53,.947),(1.,.953,1.06,.508,.67,.449),(.41,.397,.18,.514,.108,.646))
        if d=='7': p=curve((.06,.068),(.94,.068),(.68,.33,.39,.7,.37,.943))
        if d=='8': p=curve((.50,.469),(.14,.445,.07,.34,.115,.223),(.15,.003,.8,-.012,.88,.211),(.962,.359,.77,.441,.50,.469),(.20,.51,.044,.61,.081,.751),(.13,1.02,.867,1.018,.928,.748),(.98,.591,.791,.515,.50,.469))
        c |= line(p)
        if d=='1' and var==2: c|=line([(.28,.94),(.78,.94)])
    return c

def car(cx,cy,w,h):
    def pts(q): return [(cx+x*w,cy+y*h) for x,y in q]
    return poly(*pts([(-.46,.38),(-.50,.25),(-.49,-.02),(-.38,-.13),(-.29,-.43),(-.21,-.49),(.26,-.49),(.34,-.40),(.41,-.13),(.49,-.02),(.49,.36),(.40,.37),(.40,.51),(.25,.51),(.24,.39),(-.25,.39),(-.26,.51),(-.41,.51),(-.41,.38)]))

def truckfront(cx,cy,w,h):
    return poly(*[(cx+w*x,cy+h*y) for x,y in [(-.49,-.5),(.49,-.5),(.49,.39),(.39,.39),(.39,.5),(.21,.5),(.21,.4),(-.22,.4),(-.22,.5),(-.41,.5),(-.41,.39),(-.49,.39)]])

def arrows(k,var):
    t=1 if k not in (34,37,39) else -1
    if k in (33,34):
        c=stroke(curve((-.26,.66),(-.26,.12),(-.26,-.18,-.10,-.31,.19,-.31),(.35,-.31)),.167 if var==0 else .193)
        c|=poly((.22,-.64),(.67,-.31),(.22,.052),(.30,-.22),(.21,-.25)) if var==0 else poly((.20,-.64),(.67,-.31),(.20,.035))
    elif k in (35,36,37):
        ox=0 if k==35 else -.085
        c=poly((ox-.11,.67),(ox-.11,-.22),(ox-.38,-.19),(ox,-.76),(ox+.38,-.19),(ox+.11,-.22),(ox+.11,.67))
        if k!=35: c|=stroke(curve((ox,.24),(ox+.03,.12,.16,.04,.28,.04),(.40,.04)),.16)|poly((.30,-.22),(.64,.04),(.30,.31))
    elif k in (38,39):
        # broad oblique arrow; also used in a thin, chamfered-chevron style
        c=poly((-.62,-.44),(-.43,-.63),(.27,.07),(.29,-.31),(.49,-.29),(.51,.51),(-.29,.49),(-.31,.29),(.07,.27)) if var==0 else poly((-.66,-.46),(-.46,-.66),(.19,-.01),(.42,-.24),(.52,.52),(-.24,.42),(-.01,.19))
    else:
        c=np.zeros_like(X,bool)
        for i in range(3):
            a=(i*120-90+(0 if var==0 else 22))*np.pi/180
            p=[(.52*np.cos(z),.52*np.sin(z)) for z in np.linspace(a+1.42,a,19)]
            c|=stroke(p,.145)
            ux,uy=np.sin(a),-np.cos(a); vx,vy=-uy,ux; ex,ey=p[-1]
            c|=poly((ex+.18*ux,ey+.18*uy),(ex-.14*ux+.22*vx,ey-.14*uy+.22*vy),(ex-.14*ux-.22*vx,ey-.14*uy-.22*vy))
    return c if t==1 else c[:,::-1]

def warning(k,var):
    # All warning pictograms are referenced to the bounding box of the triangle.
    if k==11:
        if var==0: return poly((-.136,.64),(-.136,.36),(-.338,.36),(-.338,.172),(-.136,.172),(-.136,-.02),(0,-.27),(.136,-.02),(.136,.172),(.338,.172),(.338,.36),(.136,.36),(.136,.64))
        return poly((-.135,.65),(-.135,.36),(-.32,.46),(-.32,.31),(-.135,.16),(-.135,.00),(0,-.29),(.135,0),(.135,.16),(.32,.31),(.32,.46),(.135,.36),(.135,.65))
    if k==18:
        return poly((-.060,-.32),(.060,-.32),(.046,.38),(-.046,.38))|ellipse(0,.56,.075,.082)
    if k in (19,20):
        c=poly(*curve((-.29,.65),(-.29,.36),(-.29,.035,-.17,-.08,.20,-.10),(.20,-.23),(.46,.003),(.21,.24),(.21,.11),(-.015,.13,-.058,.20,-.06,.37),(-.06,.65)))
        if var==1: c=poly(*curve((-.29,.65),(-.29,.36),(-.29,.01,-.17,-.10,.075,-.13),(.24,-.14),(.24,.095),(.10,.102),(-.023,.12,-.059,.23,-.06,.37),(-.06,.65)))
        return c if k==20 else c[:,::-1]
    if k==21:
        p=curve((.17,.63),(.17,.32),(.17,.26,.16,.22,.14,.22),(-.10,.22),(-.17,.22,-.17,.17,-.17,.11),(-.17,-.18)); c=stroke(p,.207)
        c|=poly((-.34,-.06),(-.17,-.32),(.00,-.06))
        return c if var==0 else c[:,::-1]
    if k==22:
        return poly(*curve((-.50,.62),(-.43,.60,-.36,.36,-.22,.35),(-.095,.34,-.08,.46,0,.47),(.085,.47,.10,.34,.22,.35),(.37,.35,.43,.59,.51,.62),(-.50,.62)))
    if k==23:
        if var>=2:
            c=ellipse(-.23,.018,.07,.064)|ellipse(.18,.083,.074,.069)|poly((-.46,-.10),(-.36,-.17),(-.27,-.24),(-.05,-.215),(.092,-.094),(.295,-.040),(.317,.097),(-.415,-.004))
            c|=stroke(curve((-.36,.043),(-.61,.24,-.47,.34,-.21,.35),(.03,.36,.14,.43,.04,.51),(-.01,.58,-.17,.58,-.27,.59)),.035)|stroke(curve((.112,.153),(-.025,.26,.087,.30,.27,.33),(.54,.38,.59,.51,.42,.56),(.31,.60,.26,.61,.17,.62)),.038)
            return c if var==2 else c[:,::-1]
        if var:
            # Side elevation of a skidding car, tilted on the two wavy tracks.
            c=poly((-.44,.075),(-.40,-.035),(-.28,-.07),(-.19,-.19),(-.03,-.18),(.13,-.045),(.30,.035),(.34,.145),(.28,.19),(-.40,.103))|ellipse(-.25,.10,.071,.074)|ellipse(.186,.16,.073,.077)
            for off in (-.19,.18): c|=stroke(curve((off,.16),(off+.32,.37,off-.24,.37,off-.17,.47),(off-.08,.55,off+.16,.53,off+.13,.64)),.050)
            return c
        theta=.30; u=X*np.cos(theta)+(Y-.09)*np.sin(theta); v=(Y-.09)*np.cos(theta)-X*np.sin(theta)
        c=((abs(u)<.19)&(abs(v)<.19))|((abs(u)<.153)&(v<-.13)&(v>-.26))
        for off in (-.155,.155):
            c|=stroke(curve((off-.05,.24),(off+.08,.34,off-.21,.39,off-.115,.47),(off+.06,.55,off+.22,.53,off+.17,.66)),.068)
        return c
    if k==24:
        return stroke([(-.18,-.20),(-.18,.64)],.077)|stroke(curve((.075,-.20),(.075,-.025),(.075,.14,.28,.21,.28,.39),(.28,.64)),.077)
    if k==25:
        c=ellipse(.042,-.096,.083,.097)|stroke([(-.035,.032),(-.146,.22)],.15)|stroke([(-.136,.23),(-.11,.37),(-.21,.63)],.103)|stroke([(-.13,.22),(-.278,.34),(-.42,.60)],.087)
        c|=stroke([(-.01,.06),(.084,.21),(.207,.234)],.073)|stroke([(-.06,.085),(-.113,.206),(.035,.267)],.066)|stroke([(-.26,.057),(.34,.56)],.032)|poly((.04,.635),(.186,.46),(.23,.48),(.30,.365),(.36,.38),(.55,.635)); return c
    if k==26:
        ry=.146 if var==0 else .125
        return (.93*ellipse(0,.49,.151,ry)+.29*ellipse(0,.16,.151,ry)+.73*ellipse(0,-.17,.151,ry))
    if k in (27,28):
        def person(cx,cy,s,d):
            def p(z):return [(cx+d*s*a,cy+s*b) for a,b in z]
            return ellipse(cx+d*.068*s,cy-.32*s,.084*s,.096*s)|stroke(p([(.012,-.18),(-.045,.09)]),.152*s)|stroke(p([(-.035,.04),(.085,.24),(.16,.45)]),.094*s)|stroke(p([(-.047,.065),(-.12,.26),(-.26,.42)]),.098*s)|stroke(p([(.024,-.13),(.134,-.02),(.24,.02)]),.060*s)|stroke(p([(-.018,-.141),(-.147,-.062),(-.215,.074)]),.063*s)
        if k==27:
            if var<2: return person(0,.225,1.,-1 if var==0 else 1)
            c=ellipse(-.04,-.235,.084,.099)|stroke([(-.02,-.092),(.065,.195)],.135)|stroke([(-.004,-.075),(-.045,.046),(-.175,.142)],.074)|stroke([(.016,-.078),(.145,.025),(.20,.126)],.066)|stroke([(.038,.165),(-.053,.358),(-.201,.625)],.090)|stroke([(.050,.173),(.164,.371),(.326,.569)],.099)
            return c if var==2 else c[:,::-1]
        if var: return person(.184,.127,1.12,1)|person(-.257,.36,.65,1)|stroke([(-.14,.27),(-.055,.22),(.085,.04)],.055)
        return person(.12,.16,1.04,-1)|person(-.257,.365,.65,-1)|poly((-.14,.36),(-.23,.52),(-.40,.46),(-.32,.32))
    if k==29:
        c=np.zeros_like(X,bool)
        for a in (-.29,.29): c|=ellipse(a,.432,.194,.207)&~ellipse(a,.432,.132,.145)
        c|=stroke([(-.29,.43),(-.12,.17),(.034,.416),(.233,.16),(-.12,.17)],.045)|stroke([(.29,.43),(.18,.075),(.12,.07)],.052)|stroke([(-.13,.16),(-.18,.087)],.052)
        if var==0: c|=ellipse(.032,-.18,.07,.079)|stroke([(-.02,-.077),(-.155,.095),(-.01,.19),(-.078,.38)],.081)|stroke([(-.036,-.035),(.039,.08),(.17,.10)],.055)
        return c
    if k==30:
        c=np.zeros_like(X,bool)
        for i in range(6):
            a=i*np.pi/3-np.pi/2; u,v=np.cos(a),np.sin(a)
            def pt(h,d): return (u*h-v*d,.19+v*h*1.18+u*d)
            c|=stroke([pt(0,0),pt(.38,0)],.052)
            for h in (.16,.26): c|=stroke([pt(h+.087,-.09),pt(h,0),pt(h+.087,.09)],.044)
        if var: c|=ellipse(0,.19,.29,.305)&~ellipse(0,.19,.231,.248)
        if var==2: c|=ellipse(0,.19,.164,.182)
        return c
    if k==31:
        if var:
            # Leaping deer, extended forelegs, haunches and two fine antlers.
            c=poly((-.45,-.13),(-.36,-.16),(-.31,-.25),(-.26,-.19),(-.16,-.21),(-.21,-.09),(-.19,.012),(-.07,.098),(.18,.16),(.26,.24),(.268,.319),(.33,.447),(.447,.54),(.423,.585),(.274,.499),(.146,.365),(.123,.364),(.171,.457),(.253,.576),(.212,.597),(.092,.453),(.014,.345),(-.174,.298),(-.255,.293),(-.35,.352),(-.386,.329),(-.277,.183),(-.19,.197),(-.27,.106),(-.324,-.027),(-.428,-.043))
            c|=stroke([(-.285,-.18),(-.30,-.28),(-.35,-.33)],.030)|stroke([(-.292,-.25),(-.233,-.315)],.028)
            return c
        c=poly((-.38,-.20),(-.30,-.14),(-.26,-.05),(-.166,.056),(-.04,.115),(.187,.161),(.27,.255),(.266,.33),(.405,.505),(.417,.59),(.37,.61),(.29,.48),(.14,.36),(.10,.36),(.19,.50),(.175,.57),(.13,.56),(.075,.46),(-.024,.34),(-.146,.30),(-.22,.37),(-.26,.33),(-.224,.219),(-.307,.19),(-.355,.218),(-.38,.17),(-.32,.095),(-.32,-.021),(-.423,-.033),(-.442,-.077),(-.359,-.129))
        c|=stroke([(-.30,-.115),(-.27,-.235),(-.29,-.30)],.027)|stroke([(-.276,-.201),(-.21,-.261)],.027); return c
    raise ValueError(k)

def letters():
    c=np.zeros_like(X,bool)
    for i,k in enumerate('STOP'):
        q={'S':curve((.89,.12),(.72,-.10,.07,-.08,.077,.244),(.076,.44,.219,.42,.61,.52),(1.16,.66,.96,1.02,.52,.98),(.269,1.02,.12,.92,.065,.823)), 'T':[(.035,.055),(.96,.055),(.5,.055),(.5,.96)], 'O':curve((.5,.04),(.13,.015,.09,.19,.09,.48),(.08,.82,.116,.975,.495,.966),(.866,.978,.924,.82,.92,.47),(.914,.18,.875,.025,.5,.04)), 'P':curve((.105,.974),(.105,.045),(.55,.045),(.978,.024,.997,.52,.57,.53),(.13,.53))}[k]
        c|=stroke([((i-1.5)*.34+(a-.5)*.26,(b-.5)*.50) for a,b in q],.055)
    return c

def icon(k,var,tracking=0):
    zero=np.zeros_like(X,bool); red=zero
    if k in (0,1,2,3,4,5,7,8):
        c=drawdigits({0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[k],var,[1.,.84,1.08][tracking],[1.,1.11,.91][tracking],([.92,.76,.99] if k in (7,8) else [1.,.90,1.])[tracking])
    elif k in (6,32,41,42):
        c=abs(Y+1.08*X)<(.215 if var%2==0 else .255)
        if k==6: return c.astype(float)+(.48 if var<2 else .75)*drawdigits('80',var)*(~c),red
        if k in (41,42):
            pair=(car(-.27,-.04,.43,.34) if k==41 else truckfront(-.27,-.1,.42,.53)) | car(.263,.03,.40,.33)
            return c.astype(float)+(.54 if var<2 else .82)*pair*(~c),red
    elif k in (9,10):
        red=car(-.287,-.06,.46,.345) if k==9 else truckfront(-.281,-.09,.422,.55)
        c=red|car(.268,.02,.46,.35)
    elif k==16:
        c=poly((-.575,.15),(-.575,-.052),(-.485,-.194),(-.277,-.194),(-.277,.029),(-.233,.029),(-.233,-.266),(.59,-.266),(.59,.141),(.53,.141),(.53,.195),(-.54,.195))|ellipse(-.387,.168,.10,.10)|ellipse(.347,.17,.10,.10)
        c&=~poly((-.501,-.044),(-.448,-.147),(-.337,-.147),(-.337,-.045))
    elif k==14: c=letters()
    elif k==17: c=(abs(X)<(.77 if var==0 else .81))&(abs(Y)<(.17 if var==0 else .21))
    elif 33<=k<=40: c=arrows(k,var)
    else: c=warning(k,var)
    return c.astype(float),red

# Store only independent normalized templates. The run-time warp represents
# pose, while the small symbol variations here represent print geometry.
def sample(a,u,v):
    size=a.shape[0]; u=np.clip((u+1)*size/2-.5,0,size-1.001); v=np.clip((v+1)*size/2-.5,0,size-1.001); i=u.astype(int); j=v.astype(int); u=u-i; v=v-j
    return (1-v)*((1-u)*a[j,i]+u*a[j,i+1])+v*((1-u)*a[j+1,i]+u*a[j+1,i+1])

groups=[[0,1,2,3,4,5,7,8,9,10,16],[11]+list(range(18,32)),list(range(33,41)),[6,32,41,42],[14,17]]
P=30
u=(np.arange(P)+.5)*2/P-1; xx,yy=np.meshgrid(u,u)
# Pooled signed-gradient descriptor. All normalization is spatial, within
# one template (or a single query) only.
hogsrc=r'''def gradients(z,interior):
    dx=np.zeros_like(z); dy=np.zeros_like(z); dx[:,:,1:-1]=z[:,:,2:]-z[:,:,:-2]; dy[:,1:-1]=z[:,2:]-z[:,:-2]
    planes=[]
    for u,v in ((1,0),(.7071,.7071),(0,1),(-.7071,.7071)):
        g=u*dx+v*dy
        for signed in (g,-g):
            cell=(np.maximum(0,signed)*interior).reshape(len(z),6,5,6,5).sum(axis=(2,4))
            planes.append(cell)
    h=np.stack(planes,axis=-1); energy=np.sqrt((h*h).sum(axis=-1,keepdims=True)); h=h/np.sqrt(energy+.16*energy.max(axis=(1,2),keepdims=True)+1e-8)
    h=h.reshape(len(z),-1); return h/(np.sqrt(np.sum(h*h,axis=1,keepdims=True))+1e-8)
'''
interior=( (yy<.73)&(abs(xx)<.5*(yy+.66)) ).astype('f')
exec(hogsrc)
for g,ks in enumerate(groups):
    if g==1: roi=(yy<.745)&(np.abs(xx)<.51*(yy+.67))
    else: roi=(xx**2+yy**2)<([.742,1,.87,.91,.86][g]**2)
    K['roi'+str(g)]=roi.ravel().nonzero()[0].astype('int32'); mats=[]; labels=[]; rms=[]; reds=[]; white=[]; details=[]; lights=[]; coarse=[]; corners=[]
    interior=roi.astype('f'); ga=np.arange(-5,6)/1.55; blur=np.exp(-.5*ga**2); blur/=blur.sum()
    # A second correlation on the changing part of a speed limit. The
    # constant final zero must not drown out the information to its left.
    sub=xx[roi]<.117; K['left']=sub.nonzero()[0].astype('i') if g==0 else K['left']
    for k in ks:
        digit=k in (0,1,2,3,4,5,7,8)
        styles=4 if k in (23,27) else 3 if digit or k in (6,30,41,42) else 2 if k in (11,19,20,21,26,28,29,31,32,17) or 33<=k<=40 else 1
        for variant in range(styles*(3 if digit else 1)):
            var=variant%styles; c,red=icon(k,var,variant//styles)
            if g==1: rotations=[0,-.13,.13] if k not in (22,26,11) else [0]
            else: rotations=[0]
            original=c
            for rotation in rotations:
             c=original if rotation==0 else sample(original,np.cos(rotation)*X+np.sin(rotation)*(Y-.24),-np.sin(rotation)*X+np.cos(rotation)*(Y-.24)+.24)
             for sx in ([.78,.91,1.04,1.17] if digit else [.63,.81,1.,1.20] if g==1 else [.86,1.,1.14] if g!=3 else [.91,1.045]):
                for sy in ([.78,.93,1.08,1.23] if g==0 else [.65,.84,1.03,1.20] if g==1 else [.84,1.,1.16] if g!=3 else [.92,1.06]):
                    for shear in ([-.11,0,.11] if digit else [0]):
                        def render(a):
                            q=np.zeros((P,P)); oy=.22 if g==1 else 0
                            for dx in (-.66,0,.66):
                                for dy in (-.66,0,.66):
                                    yp=(yy+dy/P-oy)/sy+oy; xp=(xx+dx/P-shear*(yp-oy))/sx; q+=sample(a,xp,yp)
                            return q/9
                        full=render(c); m=full[roi]; r=render(red)[roi] if k in (9,10) else np.zeros_like(m)
                        if g in (0,1):
                            # Simulate finite optical resolution separately for every icon.
                            sm=(4*full+np.roll(full,1,0)+np.roll(full,-1,0)+np.roll(full,1,1)+np.roll(full,-1,1))/8
                            corners.append(gradients(sm[None],interior)[0]); md=sum(w*np.roll(sm,n-5,axis=0) for n,w in enumerate(blur)); md=sum(w*np.roll(md,n-5,axis=1) for n,w in enumerate(blur))[roi]
                            md-=md.mean(); coarse.append(md/(np.sqrt(np.sum(md*md))+1e-8))
                        if k==26:
                            lamps=[]
                            for pos in (-.17,.16,.49):
                                lamp=render(ellipse(0,pos,.15,.14))[roi]; lamps.append(lamp/(lamp.sum()+1e-8))
                            lights.append(lamps)
                        if k in (9,10): m-=.25*r
                        if digit:
                            h=m[sub]; h=h-h.mean(); details.append(h/(np.sqrt(np.sum(h*h))+1e-8))
                        d=m-m.mean(); norm=np.sqrt(np.sum(d*d))+1e-8; mats.append(d/norm); labels.append(k); rms.append(norm/np.sqrt(m.size)); reds.append(r/(r.sum()+1e-8)); white.append((1-m)/(np.sum(1-m)+1e-8))
    K['template'+str(g)]=np.ascontiguousarray(np.array(mats),dtype=np.float32); K['rms'+str(g)]=np.array(rms,'f'); K['label'+str(g)]=np.array(labels,'i');
    if g in (0,1):
        K['grad'+str(g)]=np.array(corners,'f'); K['blur'+str(g)]=np.array(coarse,'f')
    if g==1: K['lamps']=np.array(lights,'f')
    if g==0:
        K['numbers']=np.array(details,'f')
        ii=np.flatnonzero(np.isin(labels,[9,10])); K['caridx']=ii.astype('i'); K['carred']=np.array(reds,'f')[ii]-np.array(white,'f')[ii]

# Isolated-character reader: typeface aspect and spacing need not be the
# same for 2- and 3-digit layouts. Search legal strings, with ordered glyph
# positions and a common type height. Each small window includes paper.
ug=(np.arange(10)+.5)/5-1; vg=(np.arange(16)+.5)/8-1; gx,gy=np.meshgrid(ug,vg)
K['gx']=gx.ravel().astype('f'); K['gy']=gy.ravel().astype('f')
for narrow in (0,1):
    glyphs=[]
    for digit in '01235678':
        bank=[]
        for style in range(3):
            for aspect in (.83,1.,1.16):
                glyph=drawdigits(digit,style,aspect=aspect,weight=(.82 if narrow else .97)); field=np.zeros_like(gx)
                for dx in (-.5,.5):
                    for dy in (-.5,.5):field+=sample(glyph,(gx+dx/10)*.345,(gy+dy/16)*.520)
                field=field.ravel()/4; field-=field.mean(); bank.append(field/(np.sqrt(field@field)+1e-8))
        glyphs.append(bank)
    K['glyph'+str(narrow)]=np.array(glyphs,'f')

# Signed distances in normalized fronto-parallel coordinates.
def sdf(kind,x,y):
    if kind in (0,4,6,7): return 1-np.sqrt(x*x+y*y)
    if kind in (1,2):
        v=y if kind==1 else -y
        # Very tip and corners rounded, as on a manufactured warning sign.
        return np.minimum(np.minimum((1+v-2*abs(x))/np.sqrt(5),1-v),(.939-np.abs(x))+.01)
    if kind==3: return (1-np.abs(x)-np.abs(y))*.7071
    return np.minimum(1-np.maximum(abs(x),abs(y)),(1.41421-np.abs(x)-np.abs(y))*.7071)

def logistic(a):return 1/(1+np.exp(-np.clip(a,-35,35)))

def surround(k,x,y,blur):
    d=sdf(k,x,y); sig=lambda a:logistic(a/blur); edge=sig(d)
    if k in (0,1,2):
        width=[.174,.136,.162][k]; inner=sig(d-width); rim=edge-inner
        if k==2: body=inner
        elif k==0: body=inner*(.30+.70*sig((x*x+y*y)**.5-.53))
        else: body=inner*(.36+.64*sig(.22-d))
        return rim,body,edge
    if k==3: return sig(d-.145),edge-sig(d-.123),edge
    if k==5: return sig(d-.064)*(1-.82*(abs(y)<.27)*(abs(x)<.71)),edge-sig(d-.057),edge
    if k==7: return edge*(1-.90*sig(.22-abs(y))*sig(.83-abs(x))),sig(.145-abs(y))*sig(.73-abs(x)),edge
    if k==6: return sig(.155-np.abs(y+1.08*x))*sig(d-.031),edge*(1-.94*sig(.33-abs(y+1.08*x))),edge
    return edge*(.36+.64*sig(.18-d)),edge,edge

# Coarse pose search. Background and interior comparisons are differenced,
# so a uniform tint cannot produce a border response.
axis=np.arange(32); ix,iy=np.meshgrid(axis,axis); poses=[]
for cy in (13.,15.5,18.):
    for cx in (13.,15.5,18.):
        for ry in (8.2,10.3,12.6,14.8):
            for aspect in (.86,1.12): poses.append((cx,cy,ry*aspect,ry))
K['pose']=np.array(poses,'f')
for k in range(8):
    rows=[]
    for cx,cy,rx,ry in poses:
        x=(ix-cx)/rx; y=(iy-cy)/ry; d=sdf(k,x,y); a,b,e=surround(k,x,y,.041); o=logistic((-d-.045)/.04)*logistic((d+.31)/.055)
        def norm(z):return z/(z.sum()+1e-8)
        A,B,O=map(norm,(a,b,o)); rows.append([A,B,O])
    K['loc'+str(k)]=np.array(rows,dtype='f').reshape(len(poses),3,-1)

# Full-resolution signed-distance masks used for pose refinement. Sample the
# three regions on a common regular lattice extending beyond the sign.
l=25; lq=((np.arange(l)+.5)*2/l-1)*1.24; lx,ly=np.meshgrid(lq,lq)
for k in range(8):
    d=sdf(k,lx,ly); a,b,e=surround(k,lx,ly,.028); o=logistic((-d-.052)/.032)*logistic((d+.25)/.04)
    K['fine'+str(k)]=np.array([z.ravel()/(z.sum()+1e-8) for z in (a,b,o)],'f')
    if k!=6:
        weights=[]
        # Local, paired paper/ink comparisons along the physical boundary.
        if k in (0,4,5,7): angle=np.arctan2(ly,lx); parts=[np.cos(angle-theta)>np.cos(np.pi/4.9) for theta in np.arange(6)*np.pi/3]
        elif k==3: parts=[(i*lx>=0)&(j*ly>=0) for i in (-1,1) for j in (-1,1)]
        else:
            ty=ly if k==1 else -ly; base=1-ty < (1+ty-2*abs(lx))/np.sqrt(5)
            parts=[base&(lx<0),base&(lx>=0)]+[(~base)&(i*lx>=0)&side for i in (-1,1) for side in (ty<.1,ty>=.1)]
        near=e*logistic((.42-d)/.04)
        for part in parts:
            for z in (a*part*(near if k not in (4,5,7) else 1),b*part*(near if k not in (4,5,7) else 1),o*part): weights.append(z.ravel()/(z.sum()+1e-8))
        K['sectors'+str(k)]=np.array(weights,'f')

solution = r'''import numpy as np

pose = CONSTANTS['pose']

def coords(n,ext):
    q=((np.arange(n,dtype=np.float32)+.5)*2/n-1)*ext
    a,b=np.meshgrid(q,q)
    return a.ravel(),b.ravel()

fineX,fineY=coords(25,1.24)
picX,picY=coords(30,1.)

def values(im):
    r,g,b=im
    # Log-cones reduce the exposure dependence of both colour and contrast.
    lr=np.log(r+.022); lg=np.log(g+.022); lb=np.log(b+.022)
    lum=.25*lr+.51*lg+.24*lb
    dr=lr-(.61*lg+.39*lb); db=lb-(.61*lr+.39*lg); dy=.52*lr+.48*lg-lb
    lev=np.percentile(lum,(12,88)); span=max(.30,lev[1]-lev[0]); center=(lev[0]+lev[1])*.5
    light=np.clip((lum-center)/span,-1.25,1.25)
    red=np.clip(dr/.56,-1.,1.7); blue=np.clip(db/.53,-1.,1.7); yellow=np.clip(dy/.55,-1.,1.7)
    return np.stack((red,blue,yellow,light,lum,lr,lg,lb)).astype(np.float32).reshape(8,-1)

def contrast(k,aa,bb,oo):
    # Common family evidence: region A (ink), B (paper), O (outside).
    ra,ba,ya,la=aa; rb,bbv,yb,lb=bb; ro,bo,yo,lo=oo
    if k in (0,1,2):
        return .47*(ra-.70*rb-.30*ro)+.13*np.clip(ra,-.65,1.35)+.62*(lb-la)+.13*(lb-lo)+.10*np.maximum(0,abs(la-lo)-.13)
    if k==3:
        return .49*(ya-.70*yb-.30*yo)+.20*np.clip(ya,-.65,1.35)+.51*(lb-la)+.28*(lb-lo)+.11*(la-lo)+.14*(ra-rb)
    if k==4:
        return .54*(ba-.70*bo-.30*bbv)+.29*np.clip(ba,-.65,1.35)+.25*np.abs(la-lo)+.21*(lb-la)
    if k==5:
        return .46*(ra-.50*rb-.50*ro)+.26*np.clip(ra,-.65,1.35)+.46*(lb-la)+.16*(lb-lo)+.10*np.abs(la-lo)
    if k==7:
        return .41*(ra-.68*rb-.32*ro)+.19*np.clip(ra,-.65,1.35)+.62*(lb-la)+.14*(lb-lo)+.12*abs(la-lo)
    return .96*(lb-la)+.27*(lb-lo)+.11*abs(lb-lo)-.23*np.maximum(0,np.maximum(ra,ba)-.17)

def pull(f,cx,cy,rx,ry,sx,sy,qx,qy):
    x=np.clip(cx[...,None]+rx[...,None]*(qx+sx[...,None]*qy),0,30.9999)
    y=np.clip(cy[...,None]+ry[...,None]*(qy+sy[...,None]*qx),0,30.9999)
    i=x.astype(np.int32); j=y.astype(np.int32); x=x-i; y=y-j; p=i+32*j
    x=x.astype(np.float32); y=y.astype(np.float32)
    flat=p.ravel(); shape=(len(f),)+p.shape
    u=f.take(flat,axis=1).reshape(shape); h=f.take(flat+1,axis=1).reshape(shape); v=f.take(flat+32,axis=1).reshape(shape); j=f.take(flat+33,axis=1).reshape(shape)
    u+=(h-u)*x; v+=(j-v)*x; return u+(v-u)*y

def prior(v):
    return -.035*((v[...,0]-15.5)**2+(v[...,1]-15.5)**2)/9-.026*(np.log(v[...,2]/v[...,3])/.30)**2-.032*(np.log(np.sqrt(v[...,2]*v[...,3])/11.9)/.27)**2

families=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),[13],[12],list(range(33,41)),[14],[6,32,41,42],[17]]
mapgroup=(0,1,-1,-1,2,4,3,4)
coarseprior=prior(pose)

def refine(f,k,v):
    w=CONSTANTS['fine'+str(k)]; v=np.array([v[0],v[1],v[2],v[3],0,0],np.float32)
    def score(p):
        t=pull(f[:4],p[:,0],p[:,1],p[:,2],p[:,3],p[:,4],p[:,5],fineX,fineY)
        m=t@w.T
        return contrast(k,m[:,:,0],m[:,:,1],m[:,:,2])+prior(p)-.022*((p[:,4]/.20)**2+(p[:,5]/.20)**2)
    for rnd in range(2):
        step=1. if rnd==0 else .54
        for i,j,di,dj in [(0,1,1.35,1.35),(2,3,1.15,1.15),(4,5,.095,.080),(0,1,.68,.68)]:
            p=np.tile(v,(9,1)); p[:,i]+=gridA*di*step; p[:,j]+=gridB*dj*step
            vals=score(p); best=np.argmax(vals); v=p[best]; val=vals[best]
    return v,val

gridA,gridB=np.meshgrid(np.array([-1.,0.,1.],np.float32),np.array([-1.,0.,1.],np.float32));gridA=gridA.ravel();gridB=gridB.ravel()

interior=((picY<.73)&(abs(picX)<.5*(picY+.66))).reshape(30,30).astype(np.float32)

# GRADIENTS

def ocr(f,v):
    # Compare each candidate inscription against individual glyphs. A narrow
    # digit 1 is centred on its stem; the last zero provides a height anchor.
    gx=CONSTANTS['gx']; gy=CONSTANTS['gy']; cx,cy,rx,ry,sx,sy=v
    ink=(-.52*f[5]-.37*f[6]-.11*f[7])[None,:]; out=[]
    for narrow in (0,1):
        if narrow:
            origins=(-.405,0.,.405); text=((1,0,0),(1,2,0)); weights=(.31,.45,.24); w0=.221
        else:
            origins=(-.265,.265); text=((2,0),(3,0),(4,0),(5,0),(6,0),(7,0)); weights=(.65,.35); w0=.274
        ox,oy,span,hi=np.meshgrid(np.array([-.068,0,.068],np.float32),np.array([-.084,0,.084],np.float32),np.array([.82,1.,1.18],np.float32),np.array([.80,1.,1.20],np.float32),indexing='ij'); ox=ox.ravel(); oy=oy.ravel(); span=span.ravel(); hi=hi.ravel(); ps=[]
        for origin in origins:
            tx=ox+span*origin; ty=oy
            a=pull(ink,cx+rx*(tx+sx*ty),cy+ry*(ty+sy*tx),rx*w0*span,ry*.548*hi,sx*hi*.548/(w0*span),sy*span*w0/(.548*hi),gx,gy)[0]
            # Remove an illumination plane, normalize each glyph separately.
            a-=a.mean(axis=1,keepdims=True); a-=.65*(a@gx/(gx@gx))[:,None]*gx+.65*(a@gy/(gy@gy))[:,None]*gy
            a/=np.maximum(.048*np.sqrt(len(gx)),np.sqrt(np.sum(a*a,axis=1,keepdims=True))); bank=CONSTANTS['glyph'+str(narrow)]
            ps.append((a@bank.reshape(-1,len(gx)).T).reshape(len(a),8,-1).max(axis=2))
        for string in text:
            sc=sum(w*q[:,d] for w,q,d in zip(weights,ps,string)); out.append(np.max(sc-.012*((ox/.068)**2+(oy/.084)**2)))
    return np.array(out,np.float32)

def symbol(f,k,v):
    g=mapgroup[k]; ids=families[k]
    if g<0: return np.array([.16 if k==2 else .11],np.float32)
    r=CONSTANTS['roi'+str(g)]; qx=picX[r]; qy=picY[r]; cx,cy,rx,ry,sx,sy=v
    # Marginal registration error is most consequential for the tiny digits.
    shift=.070 if g==0 or g==1 else .085
    cx=cx+rx*shift*gridA; cy=cy+ry*shift*gridB
    if g==1:
        points=np.array([-.135,-.0675,0,.0675,.135],np.float32); ix,iy=np.meshgrid(points,points); ix=ix.ravel(); iy=iy.ravel()
        cx=v[0]+rx*ix; cy=v[1]+ry*iy; raw=pull(f,cx,cy,rx,ry,sx,sy,picX,picY); m=raw[:,:,r]; disp=.005*((ix/.07)**2+(iy/.07)**2)
    elif g==0:
        raw=pull(f,cx,cy,rx,ry,sx,sy,picX,picY); m=raw[:,:,r]; disp=.012*(gridA*gridA+gridB*gridB)
    else:
        m=pull(f,cx,cy,rx,ry,sx,sy,qx,qy); disp=.012*(gridA*gridA+gridB*gridB)
    # On light signs red-channel intensity separates black ink from its paper
    # without treating a misplaced red rim as a black stroke. Blue signs use
    # the red/green channels, in which white is distinct from blue paint.
    if k==4: z=.54*m[5]+.34*m[6]+.12*m[7]-.10*m[1]
    elif k in (5,7): z=.12*m[5]+.56*m[6]+.32*m[7]-.06*m[0]
    elif k==6: z=-m[4]
    else: z=-.62*m[5]-.27*m[6]-.11*m[7]
    # Winsorization and gentle background-slope cancellation, per query.
    qs=np.percentile(z,[3,16,75,97],axis=1).astype(np.float32); z=np.clip(z,qs[0,:,None]-.10,qs[3,:,None]+.06)
    cen=z.mean(axis=1,keepdims=True); z=z-cen
    a=qx-qx.mean(); b=qy-qy.mean(); z-=.72*(z@a/(a@a))[:,None]*a+.72*(z@b/(b@b))[:,None]*b
    std=np.sqrt(np.mean(z*z,axis=1)); amp=np.maximum(.045,qs[3]-qs[1]); nz=z/np.maximum(.0001,std[:,None]*np.sqrt(r.size))
    templates=CONSTANTS['template'+str(g)]; lab=CONSTANTS['label'+str(g)]; s=nz@templates.T
    # Signal reliability: the correlation of almost constant paper is noise.
    if g in (0,1):
        rawink=(-.62*raw[5]-.27*raw[6]-.11*raw[7]).reshape(-1,30,30); mask=(picX**2+picY**2<.70**2).reshape(30,30) if g==0 else interior
        gh=gradients(rawink,mask)@CONSTANTS['grad'+str(g)].T; bs=nz@CONSTANTS['blur'+str(g)].T; s=np.maximum(s,bs-.035)
        # Equal shapes at low resolution can carry different oriented edges.
        s=.68*s+.32*(1.45*gh-.49)
    s*=np.minimum(1,std/.078)[:,None]; s-=disp[:,None]
    if g==0:
        sub=z[:,CONSTANTS['left']]; sub=sub-sub.mean(axis=1,keepdims=True)
        details=CONSTANTS['numbers']; ns=len(details); ss=sub@details.T/np.maximum(.048*np.sqrt(sub.shape[1]),np.sqrt(np.sum(sub*sub,axis=1,keepdims=True)))
        s[:,:ns]=.57*s[:,:ns]+.43*(ss-.012*(gridA*gridA+gridB*gridB)[:,None])
        ci=CONSTANTS['caridx']; rc=m[0]@CONSTANTS['carred'].T
        s[:,ci]+=.16*np.clip(rc,-.70,1.15)
    if k==1:
        w=CONSTANTS['lamps']; rr=m[5]-m[6]; gg=m[6]-m[5]; yy=.5*(m[5]+m[6])-m[7]
        # Ordered red, amber and green lamps, distinct from an exclamation
        # mark even when both look like a short vertical stroke in grayscale.
        col=(rr@(w[:,0]-w[:,2]).T + .37*yy@(w[:,1]-w[:,2]).T)
        s[:,lab==26]+=.48*np.clip(col,-.65,1.3)-.11
    ret=[]
    for j in ids:
        if j==15:
            # Blank circular sign: use interior contrast relative to observed
            # contrast at the coloured border, and do not reward rim leakage.
            center=(qx*qx+qy*qy<.57**2); zz=(-.55*m[5]-.45*m[6])[:,center]
            pct=np.percentile(zz,[12,91],axis=1); delta=np.min(pct[1]-pct[0]); border=pull(f[4:5],v[0],v[1],v[2],v[3],v[4],v[5],fineX,fineY)@CONSTANTS['fine0'].T
            scale=max(.22,border[0,1]-border[0,0]); corr=.72-.63*np.minimum(1.5,delta/scale)
        else: corr=np.max(s[:,lab==j]);
        ret.append(corr)
    if k==0:
        ret=np.array(ret,np.float32); glyph=ocr(f,v); chosen=max(ret[:8]); identity=.70*ret[:8]+.30*glyph; ret[:8]=identity+chosen-max(identity)
    return np.array(ret,dtype=np.float32)

def classify(f,loc):
    base=np.max(loc,axis=1); best=np.argmax(loc,axis=1)
    # Always retain the strongest border hypotheses; a plausible pictogram
    # can disambiguate a weak or discoloured rim.
    order=np.argsort(-base); top=base[order[0]]; cs=np.full(43,-100.,np.float32)
    for rank,k in enumerate(order):
        if rank>=3 and base[k]<top-.55: break
        if rank>=5: break
        v,shape=refine(f,k,pose[best[k]])
        s=symbol(f,k,v)
        if k!=6:
            sector=CONSTANTS['sectors'+str(k)]; pix=pull(f[:4],v[0],v[1],v[2],v[3],v[4],v[5],fineX,fineY); side=(pix@sector.T).reshape(4,-1,3); side=contrast(k,side[:,:,0],side[:,:,1],side[:,:,2]); shape=.72*shape+.28*(np.sort(side)[:max(2,len(side)//2)].mean()+prior(v))
        if k==7:
            # The bar on a no-entry disc has red paint on BOTH sides. A
            # triangular rim or a sky/background horizon is only one edge.
            upper=(abs(fineX)<.55)&(fineY<-.28)&(fineY>-.75); bot=(abs(fineX)<.55)&(fineY>.28)&(fineY<.75); bar=(abs(fineX)<.55)&(abs(fineY)<.14)
            up=pix[:,upper].mean(axis=1); low=pix[:,bot].mean(axis=1); mid=pix[:,bar].mean(axis=1); s-=.58*min(.85,abs(up[0]-low[0])*.39+abs(up[3]-low[3])*.74)
            h=min(mid[3]-up[3]+.65*(up[0]-mid[0]),mid[3]-low[3]+.65*(low[0]-mid[0])); s-=.42*max(0,.35-h)
        if k in (2,3):
            m=pull(f,v[0],v[1],v[2],v[3],v[4],v[5],fineX,fineY); roi=(abs(fineX)+abs(fineY)<.59) if k==3 else ((-fineY+.70>2*abs(fineX))&(fineY>-.66)); paper=m[4,roi]; fill=np.percentile(paper,85)-np.percentile(paper,15)
            # No pictogram is expected here: reward the homogeneous paper
            # relative to boundary contrast rather than an arbitrary glyph.
            gap=m[4]@CONSTANTS['fine'+str(k)].T
            s+=.53-.55*min(1.4,fill/max(.35,abs(gap[0]-gap[1])+.15))
        # Correlation is evaluated inside a mask independent of photo content.
        bias=(0.,.00,.02,-.035,.01,-.11,-.07,-.07)[k]
        cs[families[k]]=shape*.80+s*.95+bias
    return np.argmax(cs)

# Matrix of geometric masks for the coarse pose grid.
local=np.concatenate([CONSTANTS['loc'+str(k)].reshape(-1,1024) for k in range(8)])

def predict(x):
    ans=np.empty(len(x),np.int64)
    for start in range(0,len(x),8):
        fs=np.stack([values(im) for im in x[start:start+8]])
        # Fixed contraction shape regardless of batch boundary.
        tmp=np.zeros((8,4,1024),np.float32); tmp[:len(fs)]=fs[:,:4]
        s=(tmp.reshape(32,1024)@local.T).reshape(8,4,8,len(pose),3)
        for i,f in enumerate(fs):
            l=[]
            for k in range(8):
                m=s[i,:,k]; l.append(contrast(k,m[:,:,0],m[:,:,1],m[:,:,2])+coarseprior)
            ans[start+i]=classify(f,np.stack(l))
    return ans
'''
(out/'solution.py').write_text(solution.replace('# GRADIENTS',hogsrc))
np.savez(out/'constants.npz',**K)
