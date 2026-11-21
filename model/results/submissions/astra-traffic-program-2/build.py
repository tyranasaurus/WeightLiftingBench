# Data-free compiler for analytic sign shapes and their correlation filters.
import sys
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
arrays={}

def poly(x,y,points):
    result=np.zeros(x.shape,bool)
    for a,b in zip(points,points[1:]+points[:1]):
        if a[1]!=b[1]:
            result ^= ((y>a[1])!=(y>b[1])) & (x < a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))
    return result

def path(x,y,points,r):
    res=np.zeros(x.shape,bool)
    for (a,b),(c,d) in zip(points,points[1:]):
        t=np.clip(((x-a)*(c-a)+(y-b)*(d-b))/max(1e-10,(c-a)**2+(d-b)**2),0,1)
        res |= (x-a-t*(c-a))**2+(y-b-t*(d-b))**2<r*r
    return res

def ellipse(x,y,cx,cy,rx,ry): return ((x-cx)/rx)**2+((y-cy)/ry)**2<1

def curve(p):
    ans=[]
    for i in range(0,len(p)-1,3):
        a,b,c,d=np.array(p[i:i+4]);
        for t in np.linspace(0,1,10): ans.append(tuple((1-t)**3*a+3*t*(1-t)**2*b+3*t*t*(1-t)*c+t**3*d))
    return ans

def digit(x,y,d,style):
    # Letter-box coordinates, one unit wide and one unit tall. Stroke width is
    # specified in units of height, not width. The skeletons are hand-lettered.
    w=.59
    r=.066 if style<2 else .078
    def stroke(p):return path(x*w,y,[(a*w,b) for a,b in p],r)
    if d=='0':
        q=2.7 if style!=1 else 2.05
        return (abs((x-.5)/.47)**q+abs((y-.5)/.5)**q<1) & (abs((x-.5)/(.47-2*r/w))**q+abs((y-.5)/(.5-2*r))**q>1)
    if d=='1': return stroke([(.03,.24),(.64,.06),(.68,.06),(.68,.94)])
    if d=='7':return stroke([(.09,.07),(.89,.07),(.32,.94)])
    if d=='2': p=curve([(.10,.28),(.11,-.055),(.92,-.015),(.86,.29),(.84,.48),(.44,.63),(.11,.93)])+[(.90,.93)]
    if d=='3': p=curve([(.10,.2),(.25,-.045),(.91,.015),(.86,.27),(.84,.42),(.67,.48),(.4,.48),(.69,.45),(.94,.56),(.87,.77),(.80,.995),(.24,1.02),(.09,.84)])
    if d=='5':p=[(.85,.065),(.19,.065),(.15,.47)]+curve([(.15,.47),(.55,.31),(.91,.43),(.88,.70),(.89,1.03),(.24,1.02),(.09,.85)])
    if d=='6':
        p=curve([(.67,.065),(.43,.23),(.13,.5),(.14,.68)]) if style!=1 else curve([(.82,.13),(.31,-.11),(.12,.32),(.14,.68)])
        return stroke(p) | (ellipse(x,y,.505,.685,.465,.315) & ~ellipse(x,y,.505,.685,.465-2*r/w,.315-2*r))
    if d=='8':
        a=ellipse(x,y,.50,.27,.43,.27)&~ellipse(x,y,.50,.27,.43-2*r/w,.27-2*r); b=ellipse(x,y,.50,.717,.475,.283)&~ellipse(x,y,.50,.717,.475-2*r/w,.283-2*r)
        return a|b
    return stroke(p)

def digits(x,y,word,style):
    # Different widths for the narrow '1' and the other numerals.
    widths=[.31 if d=='1' else .60 for d in word]; gap=.065 if style!=2 else .10
    total=sum(widths)+gap*(len(word)-1); cur=-total/2
    res=np.zeros(x.shape,bool)
    for d,w in zip(word,widths):
        res |= digit((x-cur)/w,y+.5,d,style);cur+=w+gap
    return res

def car(x,y):
    body=poly(x,y,[(-.91,.03),(-.76,-.11),(-.55,-.64),(-.39,-.75),(.48,-.75),(.64,-.62),(.80,-.13),(.97,.04),(.95,.65),(.79,.65),(.78,.93),(.53,.93),(.52,.68),(-.52,.68),(-.54,.93),(-.81,.93),(-.82,.65),(-.97,.65)])
    return body & ~poly(x,y,[(-.62,-.14),(-.44,-.56),(.45,-.56),(.62,-.14)])

def truck(x,y):
    # Side view: cab to the left; separated rectangular cargo body.
    cab=poly(x,y,[(-.95,.43),(-.95,-.10),(-.69,-.55),(-.42,-.55),(-.42,.25),(-.34,.25),(-.34,-.75),(.93,-.75),(.93,.46),(.82,.46),(.82,.56),(-.86,.56)])
    hole=poly(x,y,[(-.85,-.10),(-.65,-.43),(-.53,-.43),(-.53,-.10)])
    return (cab&~hole)|ellipse(x,y,-.64,.55,.17,.32)|ellipse(x,y,.63,.55,.17,.32)

def arrow(x,y,c,style):
    def up(x,y):return poly(x,y,[(-.09,.67),(.09,.67),(.09,-.22),(.31,-.08),(0,-.68),(-.31,-.08),(-.09,-.22)])
    if c==35:return up(x,y)
    if c in (38,39):
        if c==39:x=-x
        return up((x-y)*.7071,-(x+y)*.7071)
    if c in (34,37):x=-x
    if c in (33,34):
        p=curve([(-.22,.55),(-.22,.14),(-.32,-.24),(.15,-.24)])+[(.37,-.24)]
        return path(x,y,p,.077)|poly(x,y,[(.09,-.52),(.56,-.24),(.09,.05),(.20,-.24)])
    if c in (36,37):
        p=curve([(-.20,.46),(-.18,.15),(.00,.035),(.31,.035)])
        return up(x+.20,y)|path(x,y,p,.084)|poly(x,y,[(.14,-.22),(.60,.035),(.14,.32),(.22,.035)])
    res=np.zeros(x.shape,bool)
    for angle in (0,2*np.pi/3,4*np.pi/3):
        u=x*np.cos(angle)+y*np.sin(angle); v=-x*np.sin(angle)+y*np.cos(angle)
        p=[(.47*np.cos(t),.47*np.sin(t)) for t in np.linspace(-.40,-1.62,12)]
        res |= path(u,v,p,.083)|poly(u,v,[(.03,-.68),(-.32,-.44),(.04,-.23)])
    return res

def warning(x,y,c,style):
    # Local coordinates are relative to the pictogram, not the triangular rim.
    P=lambda p:poly(x,y,p)
    L=lambda p,r:path(x,y,p,r)
    E=lambda a,b,h,k:ellipse(x,y,a,b,h,k)
    if c==11 and style==1:
        return P([(-.38,.95),(.38,.95),(.38,.25),(.76,.25),(.76,-.04),(.37,-.04),(.37,-.24),(.52,-.24),(0,-1),(-.52,-.24),(-.37,-.24),(-.37,-.04),(-.76,-.04),(-.76,.25),(-.38,.25)])
    if c==11:
        return P([(-.20,.94),(.20,.94),(.20,.24),(.65,.24),(.65,.02),(.20,.02),(.20,-.23),(.42,-.23),(0,-1),(-.42,-.23),(-.20,-.23),(-.20,.02),(-.65,.02),(-.65,.24),(-.20,.24)])
    if c==18:return P([(-.29,-.86),(-.24,-.96),(.24,-.96),(.29,-.86),(.16,.31),(-.16,.31)])|E(0,.74,.26,.17)
    if c in (19,20):
        if c==20: x=-x
        return path(x,y,[(.47,.87)]+curve([(.47,.65),(.47,-.33),(.40,-.45),(-.64,-.48)]),.19)|poly(x,y,[(-.12,-.94),(-.85,-.48),(-.12,-.01)])
    if c==21:
        return L(curve([(.63,.90),(.63,.1),(.75,-.01),(.18,.16),(-.38,.45),(-.19,-.60),(-.69,-.55)]),.19)|P([(-.41,-.94),(-.93,-.52),(-.41,-.16)])
    if c==22:
        p=[(-1,.85),(-.90,.52)]+curve([(-.90,.52),(-.54,-.17),(-.30,.09),(-.09,.43),(-.02,.54),(.02,.54),(.12,.36),(.40,-.08),(.67,.13),(.86,.49)])+[(1,.85)]
        return P(p)
    if c==23 and style==1:
        xx=x*.985+y*.17; yy=-x*.17+y*.985
        res=car(xx/.71,(yy+.24)/.44)
        for dx in [-.40,.40]:res |= L([(dx+.25*np.sin(t*6.8)-.30*t,.09+.80*t) for t in np.linspace(0,1,18)],.065)
        return res
    if c==23:
        # Sliding car with two winding skid marks.
        xx=(x+.03)*.93+(y+.21)*.37; yy=-(x+.03)*.37+(y+.21)*.93
        res=car(xx/.62,yy/.46)
        for dx in [-.30,.30]:res |= L([(dx+.22*np.sin(t*6.6),.1+.78*t) for t in np.linspace(0,1,18)],.071)
        return res
    if c==24 and style==1:
        return L([(-.48,-.90),(-.48,.89)],.15)|L([(.16,-.90),(.16,-.26),(.52,.22),(.52,.89)],.15)
    if c==24:
        return L([(-.48,-.90),(-.48,.89)],.104)|L([(.20,-.90),(.20,-.31),(.49,.27),(.49,.89)],.104)
    if c==25:
        return E(-.24,-.72,.14,.15)|L([(-.28,-.43),(-.48,-.02),(-.55,.37),(-.83,.83)],.125)|L([(-.48,-.02),(-.30,.33),(-.35,.86)],.12)|L([(-.31,-.43),(-.02,-.17),(.08,-.31)],.085)|L([(-.50,-.39),(.51,.64)],.045)|P([(.03,.90),(.24,.58),(.39,.51),(.55,.32),(.74,.45),(1,.90)])
    if c==26:
        return E(0,-.66,.26,.26)|E(0,0,.26,.26)|E(0,.66,.26,.26)
    if c==27 and style==1:
        return E(-.04,-.75,.18,.185)|L([(-.08,-.42),(.02,.06),(-.16,.49),(-.44,.90)],.145)|L([(.02,.10),(.28,.52),(.64,.80)],.12)|L([(-.08,-.43),(-.31,-.13),(-.53,-.05)],.09)|L([(-.04,-.36),(.20,-.26),(.45,.00)],.09)
    if c==27:
        return E(.06,-.77,.17,.18)|L([(-.02,-.42),(-.15,.14),(.20,.50),(.46,.91)],.12)|L([(-.15,.10),(-.32,.54),(-.53,.90)],.115)|L([(-.04,-.41),(.16,-.07),(.41,.02)],.077)|L([(-.03,-.44),(-.32,-.26),(-.49,.02)],.08)
    if c==28 and style==1:
        # Walking pair: adult at the right holding a child's hand.
        return E(.31,-.68,.155,.155)|L([(.30,-.38),(.21,.20),(.02,.70),(-.05,.86)],.105)|L([(.22,.18),(.46,.49),(.70,.78)],.10)|L([(.32,-.38),(.50,-.10),(.72,.06)],.08)|L([(.28,-.38),(.08,-.12),(-.18,-.08)],.08)|E(-.44,-.30,.135,.14)|P([(-.53,-.08),(-.34,-.08),(-.17,.49),(-.65,.49)])|L([(-.47,.42),(-.57,.70),(-.69,.81)],.07)|L([(-.37,.44),(-.26,.75)],.07)|L([(-.42,-.02),(-.22,-.01),(-.10,-.16)],.074)
    if c==28:
        return E(-.20,-.70,.15,.15)|L([(-.20,-.39),(-.36,.07),(-.49,.40),(-.78,.65)],.12)|L([(-.36,.08),(-.16,.41),(-.30,.83)],.095)|L([(-.16,-.39),(.04,-.13),(.20,-.03)],.085)|L([(-.20,-.41),(-.47,-.32),(-.59,-.07)],.08)|E(.42,-.32,.14,.14)|P([(.33,-.12),(.52,-.12),(.72,.48),(.14,.48)])|L([(.36,.42),(.19,.67),(.02,.76)],.076)|L([(.51,.45),(.69,.70),(.82,.70)],.076)|L([(.39,-.07),(.20,.04),(.07,-.13)],.071)
    if c==29 and style==1:
        return warning(-x,y,c,0)
    if c==29:
        wheel=lambda cx:E(cx,.50,.37,.38)&~E(cx,.50,.265,.275)
        return wheel(-.59)|wheel(.59)|L([(-.59,.5),(-.24,-.04),(.02,.47),(.41,-.06),(-.24,-.04)],.062)|L([(.58,.5),(.35,-.24),(.20,-.24)],.055)|E(-.13,-.66,.145,.15)|L([(-.18,-.42),(-.41,-.16),(-.10,.03),(-.16,.34)],.092)|L([(-.12,-.4),(.03,-.18),(.29,-.17)],.07)
    if c==30 and style==1:
        return warning(x*.866-y*.5,y*.866+x*.5,c,0)
    if c==30:
        res=np.zeros(x.shape,bool)
        for k in range(6):
            a=np.pi*k/3; u=x*np.cos(a)+y*np.sin(a);v=-x*np.sin(a)+y*np.cos(a)
            res |= path(u,v,[(0,0),(0,.93)],.07)|path(u,v,[(-.24,.72),(0,.49),(.24,.72)],.064)
        return res
    # Leaping deer, facing left.
    if style==1:
        return warning(-x,-.20*x+y,c,0)
    if style==2:
        # A stag landing on its front legs, with the hind legs outstretched.
        u=.88*x+.47*y;v=-.47*x+.88*y
        return (ellipse(u,v,.01,-.02,.57,.23)|poly(u,v,[(-.36,.09),(-.61,-.17),(-.65,-.35),(-.85,-.32),(-.95,-.43),(-.61,-.60),(-.43,-.57),(-.23,-.15)])|path(u,v,[(-.27,.06),(-.50,.37),(-.83,.48),(-.90,.67)],.078)|path(u,v,[(-.19,.11),(-.42,.43),(-.67,.54)],.06)|path(u,v,[(.36,.02),(.52,.27),(.89,.29),(1.03,.44)],.097)|path(u,v,[(.34,.12),(.56,.43),(.85,.48)],.075)|path(u,v,[(-.55,-.53),(-.39,-.76),(-.41,-.95)],.055)|path(u,v,[(-.44,-.73),(-.24,-.80)],.045))
    return E(.13,.02,.55,.24)|P([(-.29,.12),(-.52,-.18),(-.61,-.43),(-.84,-.42),(-.92,-.55),(-.59,-.69),(-.44,-.63),(-.15,-.16)])|L([(-.20,.13),(-.49,.35),(-.76,.30),(-.86,.52)],.07)|L([(-.18,.15),(-.49,.47),(-.80,.52)],.062)|L([(.38,.17),(.55,.49),(.86,.71)],.089)|L([(.49,.12),(.72,.38),(.81,.53)],.07)|P([(.52,-.09),(.90,-.33),(.66,.04)])|L([(-.51,-.56),(-.35,-.77),(-.37,-.96)],.045)|L([(-.42,-.74),(-.23,-.80)],.036)

# All raster drawing below uses normalized nominal coordinates. A circle's radius
# and a triangle's half-width and half-height are one in these coordinates.
words=['20','30','50','60','70','80','80','100','120']
wg={11:(.40,.53,.16),18:(.32,.50,.15),19:(.44,.39,.27),20:(.44,.39,.27),21:(.46,.42,.25),22:(.54,.41,.28),23:(.48,.47,.22),24:(.46,.48,.20),25:(.55,.47,.22),26:(.49,.50,.16),27:(.47,.49,.20),28:(.55,.50,.21),29:(.55,.47,.23),30:(.46,.46,.25),31:(.52,.45,.24)}

def icon(x,y,c,style):
    if c<=8:
        h=.84 if c<7 else .73
        return digits(x/h,y/h,words[c],style)
    if c==14:
        # Plain condensed sans-serif letters on an octagonal red field.
        z=y/.56+.5; v=x/.35+2.; res=np.zeros(x.shape,bool)
        def line(k,p,r=.088):return path(v-k,z,[(a,b) for a,b in p],r)
        res|=line(0,curve([(.82,.16),(.58,-.04),(.12,.04),(.17,.28),(.22,.49),(.76,.46),(.82,.67),(.96,.98),(.37,1.06),(.12,.85)]))
        res|=line(1,[(.12,.07),(.87,.07)])|line(1,[(.5,.07),(.5,.95)])
        res|=ellipse(v,z,2.5,.50,.38,.51)&~ellipse(v,z,2.5,.50,.22,.35)
        res|=line(3,[(.18,.94),(.18,.08),(.58,.08)])|line(3,curve([(.56,.08),(.96,.08),(.94,.51),(.57,.51)]))|line(3,[(.57,.51),(.19,.51)])
        return res
    if c==17:return (abs(y)<.175)&(abs(x)<.78)
    if c in (9,10,41,42):
        if c in (9,41):return car((x+.33)/.27,y/.27)|car((x-.31)/.27,y/.27)
        return truck((x+.31)/.31,y/.27)|car((x-.34)/.25,(y-.12)/.21)
    if c==16:return truck(x/.65,y/.39)
    if c in wg:
        w,h,cy=wg[c]; return warning(x/w,(y-cy)/h,c,style)
    if c in range(33,41):return arrow(x,y,c,style)
    return np.zeros(x.shape,bool)

def geom(x,y,f):
    circ=np.sqrt(x*x+y*y)
    if f==1 or f==2:
        b=y if f==1 else -y
        # Signed distance from the three sides, true for a rounded triangle.
        side=np.minimum(1-b,(1+b-2*abs(x))/np.sqrt(5)); return side
    if f==5:return (1-abs(x)-abs(y))*.7071
    if f==4:return np.minimum(1-np.maximum(abs(x),abs(y)),(1.4142-abs(x)-abs(y))*.7071)
    return 1-circ

# Geometry dictionaries: a coarse exhaustive search followed by local changes
# to center, independent horizontal/vertical scales, and lean.
def geofilters():
    t=(np.arange(18)+.5)*32/18-16; px,py=np.meshgrid(t,t)
    pos=[]
    for sx in [9.2,11.4,13.5,15.2]:
        for sy in [9.2,11.4,13.5,15.2]:
            if abs(sx-sy)>4.8:continue
            for cx in [-3,-1,1,3]:
                for cy in [-3,-1,1,3]:pos.append((cx,cy,sx,sy,0,0))
    pos=np.array(pos); arrays['poses']=pos.astype('f');
    # Fine searches have fixed local offsets in the normalized reference view.
    delta=np.array([(a,b,c,d,e,g) for a in [-.075,0,.075] for b in [-.075,0,.075] for c in [.93,1.,1.07] for d in [.93,1.,1.07] for e,g in [(0.,0.),(-.12,0.),(.12,0.),(0.,-.11),(0.,.11)]])
    arrays['delta']=delta.astype('f'); tf=(np.arange(29)-14)/11; fx,fy=np.meshgrid(tf,tf)
    for f in range(8):
        for k,xx,yy,pr in [('c',px,py,pos),('f',fx,fy,delta)]:
            px1=(xx[None]-pr[:,0,None,None])/pr[:,2,None,None]; py1=(yy[None]-pr[:,1,None,None])/pr[:,3,None,None]
            x=px1-pr[:,4,None,None]*py1;y=py1-pr[:,5,None,None]*px1
            s=geom(x,y,f); blur=.065 if k=='c' else .05
            def step(t):return np.clip(.5+t/(2*blur),0,1)
            def band(a,b):return step(s-a)-step(s-b)
            def norm(v):return v/(1.e-10+v.sum(axis=(1,2),keepdims=True))
            inner=step(s-.20); rim=band(.005,.175); ext=band(-.20,-.035)
            if f in (1,2):
                rim=band(.003,.130);inner=step(s-.155); # Downweight icon area for a white-field measurement.
                if f==1:inner*=.25+.75*np.clip((s-.38)/(-.19),0,1)
            if f==0:inner*=.28+.72*np.clip((s-.40)/(-.18),0,1)
            rim,inner,ext=map(norm,(rim,inner,ext))
            # Outputs are mean contrast at the inner and outer rim and mean
            # colour within paint. Only three compact masks need projecting.
            if f in (0,1,2): a=rim; b=inner; d=ext
            elif f in (3,4,6):a=norm(band(.05,.37)); b=norm(step(s-.38)); d=ext
            elif f==5:a=norm(step(s-.24));b=norm(band(.045,.185));d=ext
            else:
                a=norm(step(s-.13));b=norm(band(-.015,.065));d=ext
                dia=abs(x+y)/1.4142
                line=norm(step(.16-dia)*step(s-.06))-norm(step(dia-.24)*step(s-.10))
                arrays['d'+k]=np.ascontiguousarray(line.reshape(len(pr),-1).astype('f').T)
            arrays[k+str(f)]=np.ascontiguousarray(np.stack([a,b,d],axis=1).reshape(len(pr)*3,-1).astype('f').T)
            if f in (0,1,2):
                if f==0:sec=np.floor((np.arctan2(y,x)+np.pi)*3/np.pi).astype(int)%6; nsec=6
                else:
                    by=y if f==1 else -y
                    sec=np.where((1-by)<(1+by-2*abs(x))/np.sqrt(5),2,(x>0).astype(int));nsec=3
                filters=[]
                for sidx in range(nsec):
                    # Local paired samples prevent an unrelated red object from
                    # standing in for the whole border.
                    filters.append(norm(rim*(sec==sidx))-norm(band(.19 if f==0 else .15,.36 if f==0 else .28)*(sec==sidx)))
                arrays['s'+k+str(f)]=np.ascontiguousarray(np.stack(filters,axis=1).reshape(len(pr)*nsec,-1).astype('f').T)

# Interior correlation templates. Integration and separable blur use our own
# numeric rasterizer; no font or external artwork is needed.
def filt(a):
    b=np.pad(a,((1,1),(1,1)),mode='edge');return (b[1:-1,1:-1]*4+b[1:-1,:-2]*2+b[1:-1,2:]*2+b[:-2,1:-1]*2+b[2:,1:-1]*2+b[:-2,:-2]+b[2:,2:]+b[2:,:-2]+b[:-2,2:])/16

def sample(a,x,y):
    x=np.clip(x,0,a.shape[1]-1.0001);y=np.clip(y,0,a.shape[0]-1.0001);i=x.astype(int);j=y.astype(int);dx=x-i;dy=y-j
    return ((1-dx)*a[j,i]+dx*a[j,i+1])*(1-dy)+((1-dx)*a[j+1,i]+dx*a[j+1,i+1])*dy

families=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),list(range(33,41)),[6,32,41,42],[14,17]]

def interiors():
    n=25; t=(np.arange(n)-12)/14; xx,yy=np.meshgrid(t,t)
    # High-resolution source plane.
    q=(np.arange(241)-120)/96; x,y=np.meshgrid(q,q)
    for group,classes in enumerate(families):
        if group==0:mask=(np.sqrt(xx**2+yy**2)<.755)
        if group==1:mask=(geom(xx,yy,1)>.138)
        if group==2:mask=(np.sqrt(xx**2+yy**2)<.85)
        if group==3:mask=(np.sqrt(xx**2+yy**2)<.91)
        if group==4:mask=(np.sqrt(xx**2+yy**2)<.88)
        # Include the negative space beside the ink but not the rest of the
        # white field, which may contain a shadow or part of a misaligned rim.
        if group==0:mask &= (xx/.79)**4+(yy/.555)**4 < 1
        if group==1:mask &= (yy<.79)&(yy>-.58)&(abs(xx)<.72)
        arrays['m'+str(group)]=mask
        bank=[]; labels=[]; penalty=[]; edgebank=[]; xbank=[]; ybank=[]
        for c in classes:
            if c==15:continue
            styles=range(3) if c<=8 or c==31 else range(2) if c in (11,23,24,27,28,29,30) else range(1)
            for st in styles:
                im=icon(x,y,c,st).astype(float)
                if group==3:
                    # Grey cancelled symbol behind a black diagonal band.
                    im=.67*im+.85*(abs(x+y)<.235); im=np.minimum(im,1)
                for it in range(9):im=filt(im)
                # Make room for register errors and for different revisions of a sign.
                ctr = wg[c][2] if c in wg else 0
                for sx in [.80,.97,1.14]:
                    for sy in [.80,1.,1.20]:
                        for tx,ty in [(0,0),(-.065,0),(.065,0),(0,-.065),(0,.065)]:
                            for lean in ([-.10,0,.10] if c<=8 or c in (27,28,29,31) else [0]):
                                vx=(xx-tx-lean*yy)/sx;vy=(yy-ty-ctr)/sy+ctr
                                plane=sample(im,vx*96+120,vy*96+120);v=plane[mask];v-=v.mean();norm=np.sqrt(np.sum(v*v))
                                bank.append(v/max(.0001,norm));labels.append(c)
                                if group in (0,1):
                                    dy,dx=np.gradient(filt(plane)); dx=dx[mask];dy=dy[mask]
                                    en=np.sqrt(np.sum(dx*dx+dy*dy));xbank.append(dx/(en+1e-8));ybank.append(dy/(en+1e-8))
                                penalty.append(.012*(abs(sx-1)+abs(sy-1))+.04*(abs(tx)+abs(ty))+.01*abs(lean))
        if group in (0,1):
            arrays['x'+str(group)]=np.ascontiguousarray(np.array(xbank,'f').T);arrays['y'+str(group)]=np.ascontiguousarray(np.array(ybank,'f').T)
        arrays['i'+str(group)]=np.ascontiguousarray(np.array(bank,dtype='f').T);arrays['l'+str(group)]=np.array(labels,np.int32);arrays['p'+str(group)]=np.array(penalty,'f')

def inktemplates():
    # Optical character/pictogram reader operating in coordinates relative to
    # the inferred ink box. Rendering all circular-sign contents avoids forcing
    # vehicles into digit classes. No image-derived exemplars.
    q=(np.arange(241)-120)/96; x,y=np.meshgrid(q,q)
    z=(np.arange(25)-12)/12;xx,yy=np.meshgrid(z,z)
    mask=(xx**2/1.45+yy**2/1.55<1.05); arrays['inkmask']=mask
    bank=[]; gradx=[];grady=[];labs=[]
    for c in families[0]+families[1]:
        if c==15:continue
        for st in range(3 if c in range(9) or c==31 else 2 if c in (11,23,24,28,27,29,30) else 1):
            im=icon(x,y,c,st).astype(float)
            for it in range(10):im=filt(im)
            mass=im.sum(); mx=(im*x).sum()/mass;my=(im*y).sum()/mass
            rx=np.sqrt(((x-mx)**2*im).sum()/mass)*2.;ry=np.sqrt(((y-my)**2*im).sum()/mass)*2.
            for sx in [.80,1.,1.22]:
                for sy in [.80,1.,1.22]:
                    for tx,ty in [(0,0),(-.12,0),(.12,0),(0,.12),(0,-.12)]:
                        u=mx+rx*(xx-tx)*sx;v=my+ry*(yy-ty)*sy
                        a=sample(im,u*96+120,v*96+120);gy,gx=np.gradient(filt(a));b=a[mask];b-=b.mean();bank.append(b/(1.e-8+np.linalg.norm(b)))
                        gx=gx[mask];gy=gy[mask];en=np.sqrt((gx*gx+gy*gy).sum())+1.e-8;gradx.append(gx/en);grady.append(gy/en);labs.append(c)
    for key,data in [('ink',bank),('inkx',gradx),('inky',grady)]:arrays[key]=np.ascontiguousarray(np.array(data,'f').T)
    arrays['incl']=np.array(labs,np.int32)

geofilters(); interiors(); inktemplates(); np.savez(out/'constants.npz',**arrays)

source=r'''import numpy as np

poses=CONSTANTS['poses']
delta=CONSTANTS['delta']

def mesh(n,scale):
    t=(np.arange(n,dtype=np.float32)-(n-1)/2)*scale
    return np.meshgrid(t,t)
coarsex,coarsey=mesh(18,32/18)
finex,finey=mesh(29,1/11)
iconx,icony=mesh(25,1/14)

def sample(v,x,y):
    n=len(v)
    if x.ndim==2:x=x[None];y=y[None]
    x=np.clip(x+15.5,0,30.99999);y=np.clip(y+15.5,0,30.99999)
    ix=x.astype(np.int32); iy=y.astype(np.int32);dx=x-ix;dy=y-iy
    j=np.arange(n)[:,None,None]; k=ix+iy*32
    a=v.reshape(n,-1,1024).transpose(0,2,1)
    d=dx[:,:,:,None];e=dy[:,:,:,None]
    return (((1-d)*a[j,k]+d*a[j,k+1])*(1-e)+((1-d)*a[j,k+32]+d*a[j,k+33])*e).transpose(0,3,1,2).astype(np.float32)

def view(v,p,u,w):
    a,b,s,t,h,k=[p[:,i,None,None] for i in range(6)]
    return sample(v,a+s*(u+h*w),b+t*(w+k*u))

def features(x):
    # Soft reflectance coordinates: luminance and opponents scaled between
    # dark and bright pixels by a log transform. Remove part of the cast.
    r,g,b=np.log(x+.027).transpose(1,0,2,3)
    lum=.24*r+.53*g+.23*b
    flat=lum[:,:, :].reshape(len(x),-1)
    lo,hi=np.percentile(flat,[10,90],axis=1).astype(np.float32)[:,:,None,None]
    lum=np.clip((lum-lo)/(.19+hi-lo),-.55,1.55)
    red=r-.65*g-.35*b; blue=b-.51*r-.49*g; yellow=.51*r+.49*g-b
    rgb=[]
    for ch in (red,blue,yellow):
        off=np.percentile(ch.reshape(len(x),-1),25,axis=1).astype(np.float32)[:,None,None]
        rgb.append(np.clip((ch-.68*np.clip(off,-.65,.65))/.66,-1.2,1.8))
    return np.stack([lum]+rgb,axis=1)

def prob(v):return np.minimum(np.maximum(v,0),1)

def geometry(f,maps,mode):
    n=len(maps); flat=maps.reshape(n*4,-1)
    q=(flat@CONSTANTS[mode+str(f)]).reshape(n,4,-1,3)
    rim,inside,outside=q[:,:,:,0],q[:,:,:,1],q[:,:,:,2]
    l,r,b,y=rim.transpose(1,0,2); li,ri,bi,yi=inside.transpose(1,0,2); le,re,be,ye=outside.transpose(1,0,2)
    if f in (0,1,2):
        ns=6 if f==0 else 3
        differential=(.48*maps[:,1]-.52*maps[:,0]).reshape(n,-1)
        parts=(differential@CONSTANTS['s'+mode+str(f)]).reshape(n,-1,ns)
        evidence=parts.mean(2)*.60+parts.min(2)*.40
        full=.38*(li-l)+.24*(r-ri)+.15*r+.13*(r-re)+.08*abs(l-le)+.08*(li-le)-.06*np.maximum(ri,0)
        return .55*full+.55*evidence
    if f==3:
        return .31*b+.12*bi+.21*(b-be)+.12*(bi-be)+.12*abs(l-le)+.15*(li-l)
    if f==4 or f==6:
        return .26*r+.16*ri+.22*(r-re)+.08*(ri-re)+.09*abs(l-le)+.13*(li-l)
    if f==5:
        return .35*(y-yi)+.23*y+.11*(y-ye)+.34*(li-l)+.12*(li-le)-.09*np.maximum(r,0)
    diag=(-maps[:,0].reshape(n,-1))@CONSTANTS['d'+mode]
    return .28*(l-le)+.12*(l-li)+.08*abs(l-le)+.45*diag-.18*np.maximum(r,0)-.15*np.maximum(b,0)

def combine(p,d):
    v=p.copy();v[:,0]+=p[:,2]*d[:,0];v[:,1]+=p[:,3]*d[:,1];v[:,2:4]*=d[:,2:4];v[:,4:6]=d[:,4:6];return v

def locate(maps):
    n=len(maps);idx=np.arange(n); small=sample(maps,coarsex,coarsey); params=[];scores=[];backup=[];secondary=[]
    for f in range(8):
        g=geometry(f,small,'c')-.007*(np.abs(poses[:,0])+np.abs(poses[:,1]))-.012*np.abs(poses[:,2]-poses[:,3])-.004*np.abs(poses[:,2]+poses[:,3]-25)
        p=poses[g.argmax(1)]; modes=2 if f in (0,1) else 1
        for k in range(modes):
            if k:
                # Retain a spatially distinct alternative for occluded borders.
                dif=abs(poses[None,:,:4]-p[:,None,:4]);far=dif[:,:,:2].sum(2)+.65*dif[:,:,2:].sum(2)>3.5
                g=g-5*(~far);p=poses[g.argmax(1)]
            crop=view(maps,p,finex,finey)
            h=geometry(f,crop,'f')-.075*(abs(delta[:,4])+abs(delta[:,5]))-.045*(abs(delta[:,0])+abs(delta[:,1]))
            j=h.argmax(1); fit=combine(p,delta[j]);sc=.25*g.max(1)+.75*h[idx,j]
            if k:backup.append(fit);secondary.append(sc)
            else:params.append(fit); scores.append(sc)
    return params+backup,np.stack(scores+secondary,axis=1)

def section(v,group):
    a=v[:,CONSTANTS['m'+str(group)]]
    a=a-a.mean(axis=1,keepdims=True); sig=np.sqrt((a*a).mean(axis=1));return a/np.sqrt((a*a).sum(axis=1,keepdims=True)+1e-8),sig

classes=[[0,1,2,3,4,5,7,8,9,10,15,16],[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],[33,34,35,36,37,38,39,40],[6,32,41,42]]

def inkread(maps,p,group):
    l,r,b,y=view(maps,p,iconx,icony).transpose(1,0,2,3)
    if group==0:
        mask=iconx**2+icony**2<.73**2; mask=mask&(abs(icony)<.57); cy=0.
    else:
        mask=(1+icony-2*abs(iconx)>.40)&(icony<.77)&(icony>-.63);cy=.20
    # Use red-reflectance to separate dark ink from red paint. Soft threshold
    # relative to the paper and ink levels within this candidate sign only.
    v=-l-.26*r
    paper,ink=np.percentile(v[:,mask],[24,92],axis=1).astype(np.float32)[:,:,None,None]
    cut=paper+.34*(ink-paper);mass=mask*np.minimum(np.maximum(v-cut,0),np.maximum(ink-cut,.001))
    total=mass.sum((1,2))+.000001;mx=(mass*iconx).sum((1,2))/total;my=(mass*icony).sum((1,2))/total
    sx=np.sqrt((mass*(iconx-mx[:,None,None])**2).sum((1,2))/total);sy=np.sqrt((mass*(icony-my[:,None,None])**2).sum((1,2))/total)
    dx=np.clip(mx,-.19,.19);dy=np.clip(my,cy-.25,cy+.25)
    pp=p.copy();pp[:,0]+=p[:,2]*(dx+p[:,4]*dy);pp[:,1]+=p[:,3]*(dy+p[:,5]*dx)
    sx=np.clip(2*sx,.11,.75);sy=np.clip(2*sy,.10,.65);pp[:,2]*=sx;pp[:,3]*=sy;pp[:,4]*=sy/sx;pp[:,5]*=sx/sy
    l,r,b,y=view(maps,pp,iconx*14/12,icony*14/12).transpose(1,0,2,3)
    v=-l+.06*r if group==0 else -l-.28*r
    mask=CONSTANTS['inkmask'];z=v[:,mask];z=z-z.mean(1,keepdims=True);z/=np.sqrt((z*z).sum(1,keepdims=True)+1e-8)
    gy,gx=np.gradient(v,axis=(1,2));gy=gy[:,mask];gx=gx[:,mask];en=np.sqrt((gx*gx+gy*gy).sum(axis=1,keepdims=True)+1e-8)
    sc=.68*(z@CONSTANTS['ink'])+.32*(gx@CONSTANTS['inkx']+gy@CONSTANTS['inky'])/en
    return sc,(ink-paper).reshape(-1)

def edges(v,group):
    # Signed gradient matching responds to stroke edges but not an additive
    # illumination offset. Average neighbouring pixels to suppress aliasing.
    z=np.pad(v,((0,0),(1,1),(1,1)),mode='edge')
    sm=(z[:,1:-1,1:-1]*4+2*(z[:,:-2,1:-1]+z[:,2:,1:-1]+z[:,1:-1,:-2]+z[:,1:-1,2:])+z[:,:-2,:-2]+z[:,2:,:-2]+z[:,:-2,2:]+z[:,2:,2:])/16
    gy,gx=np.gradient(sm,axis=(1,2));mk=CONSTANTS['m'+str(group)];gx=gx[:,mk];gy=gy[:,mk]
    en=np.sqrt(np.sum(gx*gx+gy*gy,axis=1,keepdims=True)+1e-8)
    return (gx@CONSTANTS['x'+str(group)]+gy@CONSTANTS['y'+str(group)])/en

def recognize(x):
    n=len(x); maps=features(x);ps,geo=locate(maps);scores=np.full((n,43),-10,np.float32)
    for group,f in [(0,0),(1,1),(2,3),(3,7),(0,8),(1,9)]:
        crop=view(maps,ps[f],iconx,icony); l,r,b,y=crop.transpose(1,0,2,3)
        if group==0 or group==1:
            # Suppress the red painted border, using its brightness in red
            # light rather than green. Retain red vehicles in circular signs.
            v=-l+.06*r if group==0 else -l-.39*r
        if group==2:v=l-.28*b
        if group==3:v=-l
        a,sig=section(v,group);corr=a@CONSTANTS['i'+str(group)]-CONSTANTS['p'+str(group)]
        if group in (0,1):
            corr=.67*corr+.33*edges(v,group)
            inkcorr,contrast=inkread(maps,ps[f],group)
        labs=CONSTANTS['l'+str(group)]; base=geo[:,f];w=.85
        if f>=8:base=base-.035
        fresh=np.full_like(scores,-10)
        for c in classes[group]:
            if c==15:
                if f==0:fresh[:,c]=base+w*(.70-2.35*sig-.56*contrast)
                continue
            s=corr[:,labs==c].max(1)
            if group in (0,1):
                ic=inkcorr[:,CONSTANTS['incl']==c].max(1)
                s=.50*s+.28*ic+.22*np.maximum(s,ic)
            if group==3:fresh[:,c]=.66*base+1.23*s-.06
            else:fresh[:,c]=base+w*s
        if group==1:
            # A German traffic-light warning has three coloured lights. Compare
            # the upper and lower lobes to one another, canceling colour cast.
            dx=abs(iconx)<.125
            top=dx&(icony>-.35)&(icony<.0);btm=dx&(icony>.32)&(icony<.62)
            clr=(r[:,top].mean(1)-r[:,btm].mean(1))*.70+(y[:,top].mean(1)-y[:,btm].mean(1))*.20
            fresh[:,26]+=.35*np.clip(clr-.10,-.35,.8)
        scores=np.maximum(scores,fresh)
    # Special, mostly uniform signs. Use their known spatial divisions too.
    yl,yr,yb,yy=view(maps,ps[2],iconx,icony).transpose(1,0,2,3)
    mask=(icony>-.76)&(1-icony-2*abs(iconx)>.39)
    ink=(yl-.15*yr)[:,mask]
    # An inverted triangle must have an empty, uniform interior, rather
    # than a speed numeral or an upright triangular pictogram.
    scores[:,13]=geo[:,2]+.61-1.6*ink.std(1)
    dl,dr,db,dy=view(maps,ps[5],iconx,icony).transpose(1,0,2,3)
    mask=abs(iconx)+abs(icony)<.60
    scores[:,12]=geo[:,5]+.60-1.05*dl[:,mask].std(1)-.45*dy[:,mask].std(1)
    rad=np.sqrt(iconx**2+icony**2)
    for f,c in [(4,14),(6,17)]:
        l,r,b,y=view(maps,ps[f],iconx,icony).transpose(1,0,2,3)
        v=l-.35*r
        a,sig=section(v,4); mask=CONSTANTS['l4']==c
        s=(a@CONSTANTS['i4'][:,mask]).max(1)
        # The quadrants above and below the word or stripe must be red, too.
        patches=[]
        for side in [-1,1]:
            for vert in [-1,1]:
                m=(iconx*side>0)&(icony*vert>.38)&(rad<.79)
                patches.append(r[:,m].mean(1))
        whole=np.stack(patches,axis=1).min(1)
        scores[:,c]=.79*geo[:,f]+.13*whole+.85*s
        if c==17:
            # A no-entry stripe is uniformly bright; letters/digits create
            # dark holes, even if the coarse stripe correlation is high.
            bar=v[:,(abs(iconx)<.62)&(abs(icony)<.135)]
            hi=v[:,(abs(icony)>.34)&(abs(icony)<.57)&(abs(iconx)<.55)]
            scores[:,c]+=.16*(bar.mean(1)-hi.mean(1))-.78*bar.std(1)-.28*hi.std(1)
    return scores.argmax(1)

def predict(x):
    res=np.empty(len(x),np.int64)
    # Fixed-size work blocks make matrix reductions independent of the caller's batch.
    for i in range(0,len(x),64):
        b=x[i:i+64]; n=len(b)
        if n<64:b=np.concatenate((b,np.repeat(b[-1:],64-n,axis=0)),axis=0)
        res[i:i+n]=recognize(b)[:n]
    return res
'''
(out/'solution.py').write_text(source)
