"""Construct each filter analytically; no assets or data are needed."""
import sys, os, math
import numpy as np

out=sys.argv[1]
os.makedirs(out,exist_ok=True)
C={}

def clip(x): return np.clip(x,0,1)
def ell(x,y,a,b,rx,ry): return ((x-a)/rx)**2+((y-b)/ry)**2<=1
def rect(x,y,l,t,r,b): return (x>=l)&(x<=r)&(y>=t)&(y<=b)
def poly(x,y,pts):
    w=np.zeros(np.broadcast(x,y).shape,bool); a,b=pts[-1]
    for c,d in pts:
        if d!=b: w^=((y>=b)!=(y>=d)) & (x<(c-a)*(y-b)/(d-b)+a)
        a,b=c,d
    return w
def stroke(x,y,pts,width):
    a,b=pts[0]; m=np.zeros(np.broadcast(x,y).shape,bool)
    for c,d in pts[1:]:
        t=clip(((x-a)*(c-a)+(y-b)*(d-b))/max((c-a)**2+(d-b)**2,1.e-8));m|=(x-a-t*(c-a))**2+(y-b-t*(d-b))**2<=(width/2)**2;a,b=c,d
    return m
def path(seq):
    # Quadratic Bezier segments (control, endpoint), or straight lines.
    xx=[seq[0]]
    for s in seq[1:]:
        if len(s)==2: xx.append(s);continue
        a,b=xx[-1];c,d,e,f=s
        for t in np.arange(1,7)/6: xx.append(((1-t)**2*a+2*t*(1-t)*c+t*t*e,(1-t)**2*b+2*t*(1-t)*d+t*t*f))
    return xx

def digit(x,y,num,style):
    # Monoline road lettering, expressed in units of a nominal 56 by 100 character.
    if num==0 and style%2:
        return ((np.abs((x-28)/27)**3.4+np.abs((y-50)/49)**3.4<=1)&(np.abs((x-28)/13.5)**3.2+np.abs((y-50)/36)**3.2>=1))
    p={0:[(28,8),(49,8,49,31),(49,68),(49,92,28,92),(7,92,7,69),(7,31),(7,8,28,8)],
       1:[(9,24),(31,8),(31,94)],2:[(7,29),(7,7,28,7),(50,7,50,29),(50,39,38,52),(7,91),(50,91)],
       3:[(8,23),(12,7,29,7),(49,7,49,28),(49,45,29,47),(22,47),(51,46,51,69),(51,93,28,93),(10,93,6,77)],
       5:[(47,9),(11,9),(9,48),(18,40,30,40),(51,40,51,66),(51,93,28,93),(10,93,6,76)],
       6:[(40,7),(15,45),(6,59,7,72),(7,93,28,93),(50,93,50,71),(50,49,31,49),(16,49,9,62)],
       7:[(6,9),(49,9),(19,94)],8:[(28,47),(8,43,9,25),(9,7,28,7),(48,7,48,26),(49,42,28,47),(6,52,7,72),(7,93,28,93),(51,93,50,72),(51,53,28,47)]}
    if num==8 and style==2:
        return ((ell(x,y,28,26,25,26)&~ell(x,y,28,26,11,13))|(ell(x,y,28,72,27,29)&~ell(x,y,28,72,13,15)))
    if num==8 and style%2:
        p[8]=[(28,47),(10,47,10,31),(10,23),(10,7,27,7),(46,7,46,23),(46,31),(46,47,28,47),(7,47,7,65),(7,73),(7,93,28,93),(49,93,49,73),(49,64),(49,47,28,47)]
    if num==6 and style%2: p[6]=[(47,21),(42,7,29,7),(7,7,7,33),(7,70),(7,93,29,93),(50,93,50,71),(50,48,30,48),(17,48,7,59)]
    if num==7 and style%2: p[7]=[(6,21),(6,9),(49,9),(19,94)]
    if num==3 and style%2: p[3]=[(8,9),(46,9),(26,44),(50,43,50,68),(50,93,28,93),(10,93,6,78)]
    width=[13,14,17][style%3]
    return stroke(x,y,path(p[num]),width)

speeds={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
def letters(x,y,c,j):
    s=speeds[c]; w=56; gap=10 if j<2 else 14
    # Use the visible width of the narrow numeral one, not a full tabular advance.
    widths=[(34 if v=='1' else w) for v in s]; tot=sum(widths)+gap*(len(s)-1);u=x*tot/100;v=y; pos=0; a=np.zeros(np.broadcast(x,y).shape,bool)
    for n,dw in zip(s,widths):
        a|=digit(u-pos+(4 if n=='1' else 0),v,int(n),j);pos+=dw+gap
    return a

def car(x,y,variant):
    m=poly(x,y,[(2,47),(11,38),(18,12),(27,6),(74,6),(83,13),(90,38),(98,46),(98,78),(92,85),(8,85),(2,79)])
    m&=~poly(x,y,[(18,38),(25,16),(73,16),(82,38)])
    m|=rect(x,y,8,75,24,96)|rect(x,y,76,75,92,96)
    if variant%2: m&=~(rect(x,y,10,54,24,60)|rect(x,y,75,54,90,60))
    return m

def pict(c,x,y,j):
    # All pictograms here use a drawing board of 100 by 100. x increases to the right.
    def P(*p): return poly(x,y,p)
    def S(w,*p): return stroke(x,y,p,w)
    def Q(w,*p): return stroke(x,y,path(p),w)
    def R(l,t,r,b): return rect(x,y,l,t,r,b)
    def E(a,b,rx,ry): return ell(x,y,a,b,rx,ry)
    if c in speeds: return letters(x,y,c,j).astype(float)
    if c==18 and j: return P((20,3),(32,0),(67,0),(80,5),(65,69),(38,69))|E(50,88,20,11)
    if c==11 and j>=2:
        return P((29,99),(29,58),(5,58),(5,50),(30,40),(30,29),(50,1),(70,29),(70,40),(95,50),(95,58),(70,58),(70,99))
    if c==23 and j>=5:
        # Front of passenger car, seen at a bank angle, with fainter serpentine tracks.
        til=.28 if j==5 else -.28;v=y-til*(x-50);sil=car((x-5)/.90,v/.47,j)
        return sil+.40*(Q(6,(20,43),(42,58,22,71),(7,84,29,97))|Q(6,(80,47),(98,60,78,75),(59,87,83,95)))
    if c==23 and j>=3:
        # Sideways-skidding car. The roof, hood and two axles slope together, while
        # the two tire traces are much lower contrast when the picture is blurred.
        tilt=.22 if j==3 else -.15;z=x;v=y-tilt*(x-50)
        side=poly(z,v,[(3,23),(14,20),(23,5),(56,5),(70,18),(94,21),(97,35),(4,35)])|ell(z,v,22,35,8,8)|ell(z,v,77,35,8,8)
        return side+.55*(Q(4,(20,45),(46,59,26,72),(8,86,30,96))|Q(4,(72,48),(93,61,71,73),(49,85,70,94)))
    if c==23 and j==2:
        # Skidding car in three-quarter / side view, with wavy tire marks behind it.
        m=P((5,24),(13,22),(21,6),(55,5),(70,17),(86,22),(88,41),(6,41))|E(23,39,9,9)|E(71,39,9,9)
        return m|Q(4,(22,47),(42,57,27,68),(10,80,32,91))|Q(4,(66,46),(89,58,65,70),(45,84,72,96))
    if c==24 and j: return R(14,1,25,99)|P((56,1),(67,1),(68,21),(82,61),(82,99),(70,99),(70,65),(56,27))
    if c in (27,28) and j:
        # Stick figure gait, with opposite arm and leg forward; children hold hands.
        def walk(z,v,rev=False):
            if rev:z=100-z
            return ell(z,v,51,10,9,8)|stroke(z,v,[(51,25),(42,52)],16)|stroke(z,v,[(48,27),(29,44),(18,50)],6)|stroke(z,v,[(53,25),(61,42),(78,50)],7)|stroke(z,v,[(43,54),(32,72),(15,95)],9)|stroke(z,v,[(44,55),(57,71),(68,94)],10)
        if c==27:return walk(x,y,j%2==0)
        return walk((x-32)/.70,y/.98,True)|walk((x+2)/.52,(y-29)/.70,True)
    if c==29 and j>=2:
        # Two wheels occupy most of the height in the upright bicycle emblem.
        return (E(21,59,20,37)&~E(21,59,14,29))|(E(80,59,20,37)&~E(80,59,14,29))|S(5,(21,59),(36,22),(53,63),(21,59))|S(5,(53,63),(67,20),(80,59))|S(4,(37,23),(65,17))|S(4,(36,24),(32,4),(23,4))|S(4,(66,18),(66,1),(78,1))
    if c==29 and j:
        # Bicycle (without a rider), with two spoke-less rims, crossbar and handlebars.
        return (E(22,64,21,30)&~E(22,64,15,22))|(E(80,64,19,30)&~E(80,64,13,22))|S(5,(22,64),(42,27),(54,65),(22,64))|S(5,(54,65),(69,23),(80,64))|S(5,(42,27),(69,23))|S(4,(39,25),(36,12),(29,12))|S(4,(68,24),(67,5),(78,5))
    if c==31 and j>=2:
        if j%2:x=100-x
        return P((8,15),(21,21),(30,38),(61,45),(75,53),(81,63),(95,69),(97,75),(79,70),(68,62),(57,62),(43,58),(29,49),(26,56),(40,68),(43,78),(39,79),(35,70),(19,62),(20,48),(15,34),(8,25),(0,23))|S(4,(71,58),(82,78),(98,89))|S(3,(15,23),(14,10),(8,3))|S(3,(18,24),(21,13),(17,6))
    if c in (9,10,41,42):
        big=c in (10,42); l=car(x/0.46,(y-19)/.82,j) if not big else (R(2,1,44,83)|R(4,80,13,98)|R(33,80,42,98))
        r=car((x-54)/.46,(y-(22 if big else 19))/.8,j)
        return .86*l+r
    if c==16:
        return (R(32,7,99,72)|P((3,43),(10,27),(29,27),(29,68),(35,68),(35,80),(1,80),(1,52))|R(11,66,90,81)|E(17,80,9,15)|E(79,79,9,15)) & ~P((7,48),(14,32),(24,32),(24,48))
    if c==11:
        return (P((45,98),(43,48),(11,58),(11,51),(41,34),(50,1),(59,34),(90,51),(90,58),(56,48),(56,98)) if j%2==0 else P((40,98),(40,59),(7,64),(7,55),(38,41),(50,0),(62,41),(93,55),(93,64),(60,59),(60,98)))
    if c==18: return P((27,4),(38,0),(62,0),(73,5),(64,64),(58,73),(41,73),(35,64))|E(50,88,16,11)
    if c in (19,20):
        z=100-x if c==20 else x
        if j%2==0: return poly(z,y,path([(5,1),(91,0,91,68),(91,99),(63,99),(63,69),(63,26,5,27),(5,1)]))
        return poly(z,y,path([(4,2),(56,4,73,34),(92,69,93,99),(62,99),(58,34,4,29),(4,2)]))
    if c==21: return P((7,1),(32,1),(32,43),(76,24),(90,30),(90,99),(65,99),(65,54),(23,73),(8,68))
    if c==22: return R(0,83,100,98)|(E(25,92,22,62)&(y<=94))|(E(73,92,25,75)&(y<=94))
    if c==23:
        if j%2==0:
            a=car((x-20)/.56,y/.43,j)|Q(6,(30,40),(57,54,37,68),(15,86,35,97))|Q(6,(67,40),(93,54,69,69),(50,83,71,98))
        else: a=P((17,8),(63,0),(80,36),(30,46))|Q(6,(32,46),(60,52,35,73),(16,84,44,97))|Q(6,(71,38),(94,53,73,66),(47,86,78,93))
        return a
    if c==24: return R(7,1,18,99)|P((57,1),(69,1),(69,30),(88,66),(88,99),(76,99),(76,70),(57,34))
    if c==25:
        return E(49,13,7,7)|P((42,19),(51,23),(38,47),(49,67),(41,93),(33,92),(38,67),(27,55),(17,90),(8,90),(20,44),(29,28))|S(6,(44,26),(54,46),(67,44))|S(3,(28,35),(85,84))|P((56,95),(67,79),(76,72),(84,75),(99,94))
    if c==26: return ((E(50,17,32,17)|E(50,50,32,17)|E(50,83,32,17)).astype(float)*np.where(y<33,.9,np.where(y<66,.65,.82)) + (.13 if j%2 else 0)*R(19,7,81,92))
    if c==27:
        return E(55,10,9,9)|P((43,22),(55,20),(61,39),(81,53),(76,59),(58,48),(48,38),(45,55),(56,69),(71,93),(62,99),(43,73),(37,70),(18,98),(10,92),(30,54),(35,34),(20,49),(13,45),(29,26))
    if c==28:
        def runner(z,v):return (ell(z,v,60,12,10,10)|poly(z,v,[(46,23),(59,24),(66,40),(85,34),(90,41),(64,57),(52,44),(45,57),(61,68),(53,93),(43,94),(47,72),(29,69),(11,82),(1,76),(21,56),(30,43),(19,48),(13,41),(31,24)]))
        return runner(x/.65,y/.99)|runner((x-43)/.59,(y-28)/.72)
    if c==29:
        return ((E(22,71,21,26)&~E(22,71,15,19))|(E(79,71,20,26)&~E(79,71,14,19))|S(4,(22,71),(40,40),(54,71),(22,71))|S(4,(53,71),(69,38),(80,71))|S(5,(37,38),(64,38))|E(52,8,7,8)|S(9,(48,20),(36,39),(49,47),(46,67))|S(5,(49,23),(56,33),(67,33)))
    if c==30:
        m=E(50,50,9,9)
        for ang in np.arange(6)*np.pi/3-np.pi/2:
            dx=np.cos(ang);dy=np.sin(ang); a,b=50+30*dx,50+30*dy;m|=S(6,(50,50),(50+49*dx,50+49*dy))|S(5,(a+6*dx-12*dy,b+6*dy+12*dx),(a,b),(a+6*dx+12*dy,b+6*dy-12*dx))
        return m
    if c==31:
        if j%2: x=100-x
        return P((2,13),(17,21),(22,36),(50,46),(72,47),(83,58),(94,60),(88,67),(79,63),(81,73),(97,92),(93,97),(69,78),(64,67),(45,65),(28,53),(26,62),(38,77),(33,82),(13,65),(15,51),(14,38),(10,28),(0,22))|S(3,(14,23),(12,9),(6,2))|S(3,(17,24),(21,12),(17,5))|S(3,(11,14),(3,9))
    if c in (33,34):
        if c==34: x=100-x
        return P(*path([(7,99),(7,58),(7,27,40,27),(60,27),(60,7),(96,36),(60,63),(60,44),(42,44),(23,44,23,64),(23,99)]))
    if c in (35,36,37):
        if c==37:x=100-x
        m=P((38,99),(38,34),(20,34),(50,1),(80,34),(61,34),(61,99)) if c==35 else P((23,99),(23,31),(6,31),(33,1),(61,31),(44,31),(44,99))
        if c!=35:m|=P((40,55),(68,55),(68,38),(99,63),(68,88),(68,71),(40,71))
        return m
    if c in (38,39):
        if c==39: x=100-x
        return P((0,12),(12,0),(77,65),(77,27),(96,47),(96,96),(46,96),(27,77),(65,77))
    if c==40:
        m=np.zeros(np.broadcast(x,y).shape,bool)
        for i in range(3):
            t=i*2*np.pi/3;xx=(x-50)*np.cos(t)+(y-50)*np.sin(t); yy=-(x-50)*np.sin(t)+(y-50)*np.cos(t);a=np.arctan2(yy,xx)
            m|=((xx*xx+yy*yy>=31**2)&(xx*xx+yy*yy<42**2)&(a>=-np.pi/2)&(a<=-.10))|poly(xx,yy,[(-15,-35),(5,-49),(5,-24)])
        return m
    if c==14:
        # Tall, narrow block capitals in a single line.
        return Q(5,(19,12),(3,-4,4,25),(4,40,15,51),(25,64,19,86),(12,104,3,87))|R(26,2,46,8)|R(34,5,39,98)|(R(50,7,69,92)&~R(55,13,64,85))|R(75,3,80,98)|R(78,3,95,9)|R(91,8,98,43)|R(79,42,94,49)
    raise ValueError(c)

# Nominal symbol width, height, and vertical center in outer-sign radius units.
box={0:(1.05,.88,0),1:(1.05,.88,0),2:(1.05,.88,0),3:(1.05,.88,0),4:(1.05,.88,0),5:(1.05,.88,0),6:(1.05,.88,0),7:(1.27,.87,0),8:(1.27,.87,0),9:(1.39,.49,0),10:(1.36,.56,0),16:(1.29,.65,0),11:(.76,.99,.22),18:(.24,1.13,.22),19:(.63,.78,.29),20:(.63,.78,.29),21:(.66,.84,.29),22:(1.06,.42,.48),23:(.8,.99,.24),24:(.85,.96,.22),25:(1.08,.97,.25),26:(.42,1.07,.2),27:(.83,1.04,.22),28:(1.12,.99,.25),29:(1.15,.84,.31),30:(.92,.92,.28),31:(1.03,.93,.26),32:(1,1,0),33:(1.26,1.43,0),34:(1.26,1.43,0),35:(.98,1.7,0),36:(1.48,1.6,0),37:(1.48,1.6,0),38:(1.29,1.29,0),39:(1.29,1.29,0),40:(1.62,1.62,0),41:(1.4,.5,0),42:(1.4,.58,0),14:(1.66,.76,0)}

def draw(c,x,y,j=0,sw=1,sh=1,dx=0,dy=0):
    w,h,cy=box[c]
    return pict(c,50+100*(x-dx)/(w*sw),50+100*(y-cy-dy)/(h*sh),j)

# Smooth region-membership functions for the outer frame detector.
def field(f,x,y,aa):
    # Returns nested soft masks and optional light lettering or stripe.
    def inside(t): return clip(.5-t/aa)
    u=np.abs(x);v=np.abs(y); r=np.hypot(x,y)
    if f in (1,2):
        yy= y if f==1 else -y
        # rounded triangle = offset of the intersection of three half planes
        z=np.maximum((2*u-yy-1)/np.sqrt(5), yy-1)
        # round the three vertices
        def tri(k):
            px=x/k;py=yy/k
            side=np.maximum((2*abs(px)-py-1)/np.sqrt(5),py-1)
            ax=.075*2.24; ay=-1+.075*2.24
            # signed distance to small circles at the very tips
            side=np.where(py<ay-np.abs(px)*.5,np.hypot(px,py-ay)-.075,side)
            bx=1-.075*(1+np.sqrt(5))/2;by=1-.075
            side=np.where((abs(px)-bx>np.maximum(0,2*(by-py))) ,np.hypot(abs(px)-bx,py-by)-.075,side)
            return inside(side*k)
        outer=tri(1); inner=inside(z+.125); core=inside(z+.22); deep=inside(z+.29)
    elif f==3:
        z=(u+v-1)/np.sqrt(2); outer=inside(z);inner=inside(z+.047);core=inside(z+.17);deep=inside(z+.23)
    elif f==4:
        z=np.maximum(np.maximum(u,v)-1,(u+v-np.sqrt(2))/np.sqrt(2));outer=inside(z);inner=inside(z+.095);core=inside(z+.15);deep=inside(z+.23)
    else:
        z=r-1;outer=inside(z);inner=inside(z+(.204 if f==0 else .066));core=inside(z+(.31 if f==0 else .16));deep=inside(z+.43)
    outside=clip(inside(z-.18)-outer)
    return outer,inner,core,deep,outside

# Families 0 ring, 1 triangle, 2 yield, 3 priority, 4 stop, 5 no entry,
# 6 cancellation, 7..14 blue arrows. Each detector is a hand-defined sum of contrasts.
def geofilter(f,x,y,aa):
    o,i,k,d,outside=field(f,x,y,aa)
    def norm(m):return m / max(1.e-6,m.sum())
    border=norm(clip(o-i)); iw=norm(i); ow=norm(outside); center=norm(o);wR=0*o;wB=0*o;wY=0*o;wL=0*o
    if f<=2:
        if f==0: white=norm(.22*i+.78*clip(i-d))
        elif f==1: white=norm(.45*i+.55*clip(i-d))
        else: white=iw
        wL=.64*white-.72*border+.08*ow
        wR=.59*border-.21*white-.12*ow
    elif f==3:
        band=norm(clip(i-k)); mid=norm(k); edge=norm(clip(o-i));wL=.38*band+.14*mid-.22*ow-.30*edge
        wY=.65*mid-.29*band-.10*ow
    elif f in (4,5):
        text=draw(14,x,y) if f==4 else ((abs(x)<.85)&(abs(y)<.185))
        # The upward intensity contrast between the white message and its red ground.
        fg=norm(text*i); fill=norm(clip(i*(1-text)));wL=.60*fg+.09*ow-.69*fill;wR=.61*fill-.24*fg-.12*ow
    elif f==6:
        stripe=clip(.5+(.182-abs((x+y)/np.sqrt(2)))/aa); fg=norm(stripe*i);white=norm((1-stripe)*i);wL=.73*white-.57*fg-.16*ow
        wR=-.16*center;wB=-.14*center;wY=-.12*center
    else:
        sym=draw(f-7+33,x,y);fg=norm(sym*i);fill=norm(clip(i*(1-sym)));wL=.56*fg+.12*ow-.68*fill;wB=.70*fill-.22*fg-.16*ow
        if f==14:
            # Rotationally invariant three-arrow orbit: use the bright annulus at
            # registration, and test its angular coverage at recognition time.
            rr=np.hypot(x,y);fg=norm(clip(1-abs(rr-.64)/.175)*i);center=norm((rr<.44)*i);rim=norm(((rr>.80)&(rr<.93))*i)
            wL=.77*fg+.12*ow-.58*center-.31*rim;wB=.41*center+.30*rim-.22*fg-.17*ow
    return np.stack([wL,wR,wB,wY]).astype(np.float32)

# Dense grid of roughly concentric sign positions; convolve using matrix products.
# Antialias by subpixel integration. Candidate grids are fixed before any image is seen.
def grids(n,extent=1):
    a=((np.arange(n)+.5)*2/n-1)*extent;return np.meshgrid(a,a)
families=15
coarse=[]
for cy in (-2.4,0,2.4):
 for cx in (-2.4,0,2.4):
  for rad in (8.8,10.8,12.8,14.8):
   for aspect in (.89,1,1.12): coarse.append((cx+15.5,cy+15.5,rad*aspect,rad/aspect,0,0))
# Fine scale corrections include shear and rotation, finer center corrections.
fine=[]
for dy in (-.10,0,.10):
 for dx in (-.10,0,.10):
  for sx,sy in ((1,1),(.91,.91),(1.09,1.09),(.92,1.07),(1.07,.92)):
   for a,b in ((0,0),(.10,0),(-.10,0),(0,.09),(0,-.09)):fine.append((dx,dy,sx,sy,a,b))

def buildgeo(params,n,extent,name):
    # Integrate the pieces of a filter before decimating to the detector grid.
    baseu,basev=grids(n*2,extent);bias=15.5 if name=='coarse' else 0
    G=[]
    for f in range(families):
        gg=[]
        for cx,cy,rx,ry,sh,sl in params:
            u=(baseu+bias-cx)/rx;v=(basev+bias-cy)/ry
            h=geofilter(f,(u-sh*v)/(1-sh*sl),(v-sl*u)/(1-sh*sl),2*extent/n/min(rx,ry)*.60)
            gg.append(h.reshape(4,n,2,n,2).sum((2,4)).reshape(-1))
        G.append(np.array(gg,np.float32))
    C[name]=np.stack(G);C[name+'p']=np.asarray(params,np.float32)
buildgeo(coarse,16,16,'coarse');buildgeo(fine,20,1.30,'fine')

# Rasterize each symbol just once per drawing style at high resolution. Each distorted
# view remains a separate template. No information is pooled between templates.
redclasses=[0,1,2,3,4,5,7,8,9,10,16];warnclasses=[11]+list(range(18,32));endclasses=[32,6,41,42]

def window(group):
    if group=='red': return (26,20,.765,-.567,.567)
    if group=='warn':return (24,24,.81,-.72,.82)
    return (26,20,.765,-.567,.567)
def sample(arr,x,y):
    u=np.clip(x,0,arr.shape[1]-1.001);v=np.clip(y,0,arr.shape[0]-1.001);i=u.astype(int);j=v.astype(int);a=u-i;b=v-j
    return ((1-b)*((1-a)*arr[j,i]+a*arr[j,i+1])+b*((1-a)*arr[j+1,i]+a*arr[j+1,i+1]))

def symbank(group,classes):
    nx,ny,xm,yl,yr=window(group);N=3;u,v=np.meshgrid(((np.arange(nx*N)+.5)*2/(nx*N)-1)*xm,yl+(np.arange(ny*N)+.5)*(yr-yl)/(ny*N));x,y=u.reshape(ny,N,nx,N).mean((1,3)),v.reshape(ny,N,nx,N).mean((1,3))
    w=clip((1-np.hypot(x,y)-.18)/.07) if group!='warn' else clip((np.minimum((1+y-2*abs(x))/np.sqrt(5),1-y)-.125)/.07)
    if group=='end':w*=clip((abs(x+y)-.21)/.13)
    if group=='red':extra=[];ew=clip((.35-abs(x))/.12)
    C[group+'w']=w.astype(np.float32); mats=[];ids=[];firsts=[];edge=[];shape=[]
    def smooth(t,n):
        p=np.pad(t,n,mode='edge');return sum(p[i:i+ny,j:j+nx] for i in range(2*n+1) for j in range(2*n+1))/(2*n+1)**2
    def parts(t):
        # Coarse ink density in cells and oriented stroke energy; allow up to a cell of fuzz
        # by blurring before binning. Features are calculated separately for each glyph.
        t=t*w; z=smooth(t,1); gy,gx=np.gradient(z); kx=smooth(np.abs(gx)*w,1);ky=smooth(np.abs(gy)*w,1)
        def binned(m):
            q=m.reshape(6,4,6,4).sum((1,3));q=np.sqrt(q.ravel()+.00001);return q/(np.linalg.norm(q)+.00001)
        return np.concatenate([binned(t),binned(kx),binned(ky)])/3**.5
    for c in classes:
        # Empty background is handled by a no-ink hypothesis, not by correlation.
        if c==32:continue
        nstyles=3 if c in speeds or c in (11,23,27) else (4 if c==31 else (2 if c in (9,10,16,18,19,20,24,26,28,29,41,42) else 1))
        if c==23:nstyles=7
        if c==29:nstyles=3
        for style in range(nstyles):
            a=np.arange(204)*.5-1.0;xx,yy=np.meshgrid(a,a);r=pict(c,xx,yy,style).astype(float)
            bw,bh,cy=box[c]
            if c==18:bw=.43 # The stem and dot occupy about half of the glyph box.
            if c==24:bw=.70
            if c==29 and style:bh=.67
            for sw in ((.70,.85,1,1.16,1.33) if group=='warn' else (.83,1,1.18)):
             for sh in ((.75,.90,1.05,1.22) if group=='warn' else (.82,1,1.19)):
              for shear in ((-.105,0,.105) if c in speeds or group=='warn' else (0,)):
               for dy in ((-.13,-.045,.045,.13) if group=='warn' else (-.076,0,.076)):
                for dx in (-.085,0,.085):
                    px=(u-dx-shear*(v-cy-dy))/(bw*sw);py=(v-cy-dy)/(bh*sh)
                    t=sample(r,(.5+px)*200+2,(.5+py)*200+2)
                    good=(abs(px)<.52)&(abs(py)<.52);t=(t*good).reshape(ny,N,nx,N).mean((1,3));
                    # Center, then normalize this template alone, with the same mask as the query.
                    def filt(t,w):
                        z=(t-np.sum(t*w)/max(w.sum(),1.e-6))*np.sqrt(w);return (z/(np.sqrt(np.sum(z*z))+.0001)).ravel()
                    t=.40*t+.60*smooth(t,1)
                    if group=='red':t=.38*t+.62*smooth(t,1)
                    mats.append(filt(t,w));ids.append(c)
                    hp=t-smooth(t,2);edge.append(filt(hp,w))
                    if group=='warn':shape.append(parts(t))
                    if group=='red':
                        if c in (7,8):extra.append(filt(t,w*ew))
                        firsts.append(filt(t,w*clip((.035-x)/.085)) if c in speeds and c not in (7,8) else np.zeros(t.size))
    C[group+'s']=np.array(mats,np.float32);C[group+'ids']=np.array(ids,np.int32);C[group+'e']=np.array(edge,np.float32)
    if shape:C['warnshape']=np.array(shape,np.float32)
    if group=='red':
        # Avoid storing empty filters for the object and three-digit classes.
        C['fdigs']=np.stack(firsts[:ids.index(7)]).astype(np.float32);C['mdigs']=np.stack(extra).astype(np.float32)
symbank('red',redclasses);symbank('warn',warnclasses);symbank('end',endclasses)

# Independent-character OCR. Boxes include all of the stroke plus a narrow halo.
def glyphbank():
    nx,ny=14,22;A,B=np.meshgrid((np.arange(nx*3)+.5)/(nx*3)-.5,(np.arange(ny*3)+.5)/(ny*3)-.5);filts=[];pm=[];splits=[]
    for number in (0,1,2,3,5,6,7,8):
        splits.append(len(filts))
        for sty in range(3):
            for sx in (.82,1,1.17):
             for sy in (.84,1,1.14):
              for shear in (-.11,0,.11):
               # Offsets are independent for each character, not whole-string displacements.
               for dx in (-.13,0,.13):
                for dy in (-.09,0,.09):
                    t=digit(28+(A-dx-shear*B)*(84/sx),50+(B-dy)*(123/sy),number,sty).astype(float).reshape(ny,3,nx,3).mean((1,3));p=np.pad(t,1,mode='edge');t=.45*t+.55*sum(p[i:i+ny,j:j+nx] for i in range(3) for j in range(3))/9
                    z=(t-t.mean());filts.append((z/(np.sqrt((z*z).sum())+.00001)).ravel());pm.append(.009*(abs(dx)/.13+abs(dy)/.09)+.006*(abs(sx-1)+abs(sy-1))/.17)
    C['ocr']=np.array(filts,np.float32);C['ocrsplit']=np.array(splits+[len(filts)],np.int32);C['ocrcost']=np.array(pm,np.float32)
glyphbank()

# A second, local registration bank for circular mandatory signs. The outer face
# is always a disk; only the inner symbol rotates/scales/translates.
def bluebank():
    n=24;u,v=grids(n*2,1);x,y=grids(n,1);w=clip((.95-np.hypot(x,y))/.10);C['bluew']=w.astype(np.float32); mats=[];ids=[]
    for c in range(33,41):
        for rot in (np.arange(-60,60,8) if c==40 else (-10,-4,0,4,10)):
         for sw in (.82,1,1.18):
          for sh in (.82,1,1.18):
           for dx in (-.10,0,.10):
            for dy in (-.10,0,.10):
                angle=np.deg2rad(rot);xx=(u-dx)*np.cos(angle)+(v-dy)*np.sin(angle);yy=-(u-dx)*np.sin(angle)+(v-dy)*np.cos(angle)
                t=draw(c,xx,yy,0,sw,sh).reshape(n,2,n,2).mean((1,3));z=(t-(t*w).sum()/w.sum())*np.sqrt(w);mats.append((z/(np.linalg.norm(z)+.00001)).ravel());ids.append(c)
    C['blues']=np.array(mats,np.float32);C['blueids']=np.array(ids,np.int32)
bluebank()

solution=r'''import numpy as np
C=CONSTANTS

def sat(a):return np.clip(a,0,1)
def grid(n,extent):
    a=((np.arange(n,dtype=np.float32)+.5)*(2/n)-1)*extent
    return np.meshgrid(a,a)

def prepare(a):
    r,g,b=a[:,0],a[:,1],a[:,2]
    lum=np.sqrt(.16*r+.62*g+.22*b+.003)
    box=lum[:,5:27,5:27].reshape(len(a),-1)
    pp=np.partition(box,(28,450),axis=1);lo=pp[:,28,None,None];hi=pp[:,450,None,None]
    light=np.clip((lum-lo)/np.maximum(hi-lo,.068),-.3,1.3)
    red=sat(3.25*(r-.62*g-.38*b)/(r+.62*g+.38*b+.025))
    blue=sat(2.5*(.75*b+.25*g-r)/(r+.75*b+.25*g+.025))
    yellow=sat(3*(.38*r+.62*g-b)/(.38*r+.62*g+b+.025))*sat(1.3-1.35*red)
    return np.stack((light,red,blue,yellow),axis=1)

def warp(a,p,u,v):
    # Bilinear sampler shared by detector and interior rectifier. x,y axes have independent scale.
    n,ch,h,w=a.shape
    ux=u.reshape(1,-1);vy=v.reshape(1,-1)
    xx=np.clip(p[:,0:1]+p[:,2:3]*ux+p[:,4:5]*vy,0,w-1.0001); yy=np.clip(p[:,1:2]+p[:,3:4]*vy+p[:,5:6]*ux,0,h-1.0001)
    ix=xx.astype(np.int32);iy=yy.astype(np.int32);dx=(xx-ix).astype(np.float32)[:,None,:];dy=(yy-iy).astype(np.float32)[:,None,:]
    ij=(ix+w*iy)[:,None,:]; a=a.reshape(n,ch,-1)
    top=(1-dx)*np.take_along_axis(a,ij,axis=2)+dx*np.take_along_axis(a,ij+1,axis=2)
    bot=(1-dx)*np.take_along_axis(a,ij+w,axis=2)+dx*np.take_along_axis(a,ij+w+1,axis=2)
    return ((1-dy)*top+dy*bot).reshape((n,ch)+u.shape)

# Pose-complexity costs favor an ordinary, central crop in the absence of evidence.
cp=C['coarsep'];fp=C['finep'];nf=len(C['coarse']);ug,vg=grid(20,1.30)
prior=.003*((cp[:,0]-15.5)**2+(cp[:,1]-15.5)**2)/5+.012*((cp[:,2]+cp[:,3]-24)/7)**2
fprior=.003*((fp[:,0]/.1)**2+(fp[:,1]/.1)**2)+.002*((fp[:,4]/.10)**2+(fp[:,5]/.09)**2)
def poses(a):
    n=len(a);small=a.reshape(n,4,16,2,16,2).mean((3,5)).reshape(n,-1)
    sc=(small @ C['coarse'].reshape(-1,1024).T).reshape(n,nf,-1)-prior
    inds=np.argmax(sc,axis=2);allp=[];alls=[]
    for f in range(nf):
        p=cp[inds[:,f]];obs=warp(a,p,ug,vg).reshape(n,-1)
        s=(obs @ C['fine'][f].T)-fprior; ix=s.argmax(1);q=fp[ix];v=s[np.arange(n),ix]-prior[inds[:,f]]
        p=np.stack((p[:,0]+p[:,2]*q[:,0],p[:,1]+p[:,3]*q[:,1],p[:,2]*q[:,2],p[:,3]*q[:,3],p[:,2]*q[:,2]*q[:,4],p[:,3]*q[:,3]*q[:,5]),axis=1)
        allp.append(p); alls.append(v)
    return np.stack(allp,1),np.stack(alls,1)

def unit(q,w):
    q=q-(q*w).sum((1,2),keepdims=True)/(w.sum()+.000001);q=q*np.sqrt(w);return q/(np.sqrt(np.sum(q*q,(1,2),keepdims=True))+.00001)

def smooth(a,n):
    h,w=a.shape[1:];p=np.pad(a,((0,0),(n,n),(n,n)),mode='edge')
    return sum(p[:,i:i+h,j:j+w] for i in range(2*n+1) for j in range(2*n+1))/(2*n+1)**2

def shapevec(t,w):
    t=t*w; z=smooth(t,1); gy,gx=np.gradient(z,axis=(1,2));kx=smooth(abs(gx)*w,1);ky=smooth(abs(gy)*w,1)
    def bn(m):
        q=m.reshape(len(m),6,4,6,4).sum((2,4)).reshape(len(m),-1);q=np.sqrt(q+.00001);return q/(np.sqrt((q*q).sum(1,keepdims=True))+.00001)
    return np.concatenate([bn(t),bn(kx),bn(ky)],axis=1)/3**.5

def descriptors(a,feature,pose,group):
    w=C[group+'w'];ny,nx=w.shape
    if group=='warn':xm,yl,yr=.81,-.72,.82
    else:xm,yl,yr=.765,-.567,.567
    u,v=np.meshgrid(((np.arange(nx,dtype=np.float32)+.5)*2/nx-1)*xm,yl+(np.arange(ny,dtype=np.float32)+.5)*(yr-yl)/ny)
    raw=warp(a,pose,u,v);wr,wg,wb=raw[:,0],raw[:,1],raw[:,2]
    # Red ink should register as ink in the paired car symbols too. Square root
    # compresses highlights without blowing up quantization noise in heavy shade.
    gray=np.sqrt(.37*wr+.47*wg+.16*wb+.003); strong=np.sqrt(.72*wr+.20*wg+.08*wb+.003)
    if group=='end': gray=np.sqrt(.28*wr+.53*wg+.19*wb+.003)
    def inks(g):
        # Estimate local paper brightness by propagating light values within the face.
        z=np.where(w>.2,g,0);pp=np.pad(z,((0,0),(3,3),(3,3)),mode='edge');env=z
        for i in (0,2,4,6):
            for j in (0,2,4,6):env=np.maximum(env,pp[:,i:i+ny,j:j+nx])
        # Soft reference to the global paper brightness stabilizes broad objects.
        ref=np.max(np.where(w>.5,g,0),axis=(1,2),keepdims=True)
        ink=(.84*env+.16*ref-g)/(.55*env+.45*ref+.045)
        return ink
    ink=inks(gray); alt=inks(strong)
    mix=.4*ink+.6*alt
    if group=='warn':
        # The painted frame is red, not black ink. The red-channel test suppresses it even
        # when pose rectification puts a little of the frame inside the triangular mask.
        hue=warp(feature[:,1:2],pose,u,v)[:,0];mix=(.18*ink+.82*alt)*(1-.78*hue);ink=mix
    obj=unit(ink,w).reshape(len(a),-1); robj=unit(mix,w).reshape(len(a),-1)
    hp=unit(mix-smooth(mix,2),w).reshape(len(a),-1);hp2=unit(ink-smooth(ink,2),w).reshape(len(a),-1)
    if group=='warn':
        core=ink[:,w>.65];a0=np.partition(core,core.shape[1]//3,axis=1)[:,core.shape[1]//3,None,None]
        shape=shapevec(np.maximum(ink-a0,0),w)
    dim=nx*ny;templates=C[group+'s']; ids=C[group+'ids'];labels=np.unique(ids);sc=np.full((len(a),43),-2,dtype=np.float32)
    for c in labels:
        start=np.searchsorted(ids,c);stop=np.searchsorted(ids,c,side='right');t=templates[start:stop]
        s=.76*((robj if c<=8 else obj) @ t.T)+.24*((hp if c<=8 else hp2) @ C[group+'e'][start:stop].T)
        if group=='warn':s+=.82*((shape @ C['warnshape'][start:stop].T)-.90)
        if group=='red' and c<6:
            part=unit(.4*ink+.6*alt,w*sat((.035-u)/.085)).reshape(len(a),dim)
            s=.64*s+.36*(part @ C['fdigs'][start:stop].T)
        if group=='red' and c in (7,8):
            mid=unit(mix,w*sat((.35-abs(u))/.12)).reshape(len(a),dim);off=np.searchsorted(ids,7)
            s=.72*s+.28*(mid @ C['mdigs'][start-off:stop-off].T)
        sc[:,c]=np.max(s,axis=1)
    core=(w>.5);mag=np.sort(ink[:,core],axis=1);amp=mag[:,int(.88*mag.shape[1])]-mag[:,int(.26*mag.shape[1])];
    if group=='red':
        # Paper contrast is judged relative to the ring in the same photograph;
        # absolute contrast alone is not reliable under washed-out illumination.
        rx,ry=grid(24,1);full=warp(a,pose,rx,ry);g=np.sqrt(.26*full[:,0]+.52*full[:,1]+.22*full[:,2]+.003);radius=np.hypot(rx,ry);ref=np.percentile(g[:,radius<.72],78,axis=1);ring=g[:,(radius>.82)&(radius<.93)].mean(1);inkref=np.maximum(.06,(ref-ring)/(.08+ref));amp=amp/np.sqrt(.20+.80*inkref)
    hue=warp(feature[:,1:2],pose,u,v)[:,0];carred=(hue*w*((u<-.03)&(abs(v)<.3))).sum((1,2))/(w*((u<-.03)&(abs(v)<.3))).sum()-(hue*w*((u>.03)&(abs(v)<.3))).sum((1,2))/(w*((u>.03)&(abs(v)<.3))).sum()
    if group=='warn':
        # A traffic-light stack has a red upper lamp and a green lower lamp. In the other
        # narrow (exclamation / human) symbols this vertical chroma contrast is absent.
        def meanpatch(t,cx,cy,rx,ry):
            m=((u-cx)/rx)**2+((v-cy)/ry)**2<1;return t[:,m].mean(1)
        opponent=(wr-wg)/(wr+wg+.025);col=meanpatch(opponent,0,-.11,.16,.16)-meanpatch(opponent,0,.50,.15,.16)
        sc[:,26]+=.72*col-.075;sc[:,18]-=.14*np.maximum(col,0)
        # Bike wheels are a roughly symmetric pair low in the triangular field.
        # The skidding vehicle has a broad dark roof above two lighter tire traces;
        # a deer has its body and rump nearly on the same diagonal as its head.
        def rectmean(z,l,r,t,b):return z[:,(u>l)&(u<r)&(v>t)&(v<b)].mean(1)
        q=np.maximum(0,ink-a0);q=q/(.035+np.max(q*w,(1,2),keepdims=True));lc=rectmean(q,-.48,-.17,.22,.61);rc=rectmean(q,.17,.48,.22,.61);tc=rectmean(q,-.18,.18,-.23,.02)
        bilateral=np.minimum(lc,rc);sc[:,29]+=.18*bilateral-.15*tc-.08*abs(lc-rc)
        top=rectmean(q,-.34,.32,-.08,.23);bottom=rectmean(q,-.34,.32,.40,.63);diag=rectmean(q,.21,.48,.30,.53)-rectmean(q,-.48,-.21,.30,.53)
        sc[:,23]+=.12*(top-bottom)-.085*abs(diag);sc[:,31]+=.09*abs(diag)
    return sc,amp,carred

# Score simple observables appropriate to a single sign (e.g., a horizontal bar has no
# vertical lettering texture). The poses are taken from the independent outline searches.
def cues(ft,p,f):
    u,v=grid(26,1.16);m=warp(ft,p,u,v);l,r,b,y=m[:,0],m[:,1],m[:,2],m[:,3]
    def avg(t,w):return t[:,w].mean(1)
    def diff(t,u1,u2):return avg(t,u1)-avg(t,u2)
    if f==3:
        d=abs(u)+abs(v);core=d<.66;band=(d>.79)&(d<.92);outside=(d>1.065)&(d<1.23)
        # Four high brightness transitions along the sides of the diamond independent
        # of the central yellow hue (which is nearly neutral under some illuminants).
        octant=[]
        for sx,sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
            mask=(u*sx>0)&(v*sy>0);octant.append(abs(diff(l,band&mask,outside&mask)))
        edge=np.median(np.stack(octant),axis=0)
        paper=l[:,band];ctr=l[:,core];darkborder=np.maximum(0,paper.mean(1)-ctr.mean(1));sym=[]
        for sx,sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
            mask=(u*sx>0)&(v*sy>0);sym.append(diff(l,band&mask,core&mask))
        border=np.sort(np.stack(sym,1),axis=1)[:,1]
        return .32*avg(y,core)+.28*edge+.18*darkborder+.20*np.maximum(border,0)+.13*diff(y,core,band)-.10*avg(r,band)
    if f==6:
        along=(u-v)/1.414;across=(v+u)/1.414;circle=u*u+v*v<.87**2
        # Local normalization is essential for a gray stripe on a dark, backlit disk.
        # Require the diagonal to continue through both halves of the circle.
        ds=[]
        for v0 in (-.57,-.18,.18,.57):
            zone=(abs(along-v0)<.18)&circle;mid=zone&(abs(across)<.12);side=zone&(abs(across)>.24)&(abs(across)<.46)
            high=avg(l,side);low=avg(l,mid);ds.append((high-low)/(.08+abs(high-low)))
        # Cancellation signs have no colored rim. Inspect the candidate's rim as well
        # as its center: a black stroke in the middle of a red warning sign is not
        # evidence that the red frame belongs to the background.
        rim=(u*u+v*v>.78**2)&(u*u+v*v<1.04**2)
        return .34*np.mean(ds,axis=0)+.13*np.min(ds,axis=0)-.13*avg(b,circle)-.26*avg(r,rim)
    # Variation along a horizontal line distinguishes STOP letters from no-entry bar.
    texture=np.abs(l[:,:,2:]-l[:,:,:-2]);iw=(abs(v[:,1:-1])<.26)&(abs(u[:,1:-1])<.72)
    if f==4:return .55*avg(texture,iw)-.06
    return -.59*avg(texture,iw)+.04

def ocr(a,p):
    u,v=np.meshgrid((np.arange(14,dtype=np.float32)+.5)/14-.5,(np.arange(22,dtype=np.float32)+.5)/22-.5)
    # Normalize each possible character box separately. Use red light to suppress
    # bleeding from the red border; preserve genuine black figures in all channels.
    def readchar(cx,width,indices):
        g=warp(a,p,u*width+cx,v*1.05);g=np.sqrt(.62*g[:,0]+.24*g[:,1]+.14*g[:,2]+.002)
        pp=np.pad(g,((0,0),(3,3),(3,3)),mode='edge');env=g
        for dy in (0,3,6):
            for dx in (0,3,6):env=np.maximum(env,pp[:,dy:dy+22,dx:dx+14])
        z=(env-g)/(.55*env+.45*g.max((1,2),keepdims=True)+.035)-.25*g/(.2+g.mean((1,2),keepdims=True))
        z-=z.mean((1,2),keepdims=True);z=(z/(np.sqrt((z*z).sum((1,2),keepdims=True))+.0001)).reshape(len(a),-1)
        out=[]
        for j in indices:
            l,r=C['ocrsplit'][j:j+2];s=z @ C['ocr'][l:r].T-C['ocrcost'][l:r];out.append(s.max(1))
        return np.stack(out,1)
    d2=readchar(-.285,.72,[2,3,4,5,6,7]);zero=readchar(.285,.72,[0]);two=.70*d2+.30*zero
    left=readchar(-.45,.55,[1]);mid=readchar(0,.60,[0,2]);right=readchar(.43,.60,[0]);three=.27*left+.49*mid+.24*right
    return np.concatenate([two,three],axis=1)

def counters(a,p):
    # Measure enclosed light counters without binarizing the photograph. For each
    # light candidate, compare it to the darkest points along four paths out of it.
    # If all four paths cross a darker stroke, this is evidence of a closed loop.
    ux=np.linspace(-.83,.83,61,dtype=np.float32);vy=np.linspace(-.66,.66,51,dtype=np.float32);u,v=np.meshgrid(ux,vy);g=warp(a,p,u,v);g=np.sqrt(.65*g[:,0]+.22*g[:,1]+.13*g[:,2]+.002)
    def loops(t,l,r,mid):
        q=g[:,t:t+mid,l:r];n,h,w=q.shape
        # Grayscale reconstruction from the four sides. A diagonal or winding
        # white path is just as valid an escape route as a straight one.
        fill=q.copy();fill[:,1:-1,1:-1]=0
        for step in range(h+w):
            pp=np.pad(fill,((0,0),(1,1),(1,1)),mode='edge');spread=np.maximum.reduce((pp[:,:h,:w],pp[:,:h,1:w+1],pp[:,:h,2:],pp[:,1:h+1,:w],pp[:,1:h+1,2:],pp[:,2:,:w],pp[:,2:,1:w+1],pp[:,2:,2:],fill));nxt=np.minimum(q,spread)
            if np.array_equal(fill,nxt):break
            fill=nxt
        contrast=q-fill
        return contrast[:,h//4:h*3//4,w//5:w*4//5].max((1,2))
    top=loops(3,10,27,24);bot=loops(22,10,27,25);other=loops(5,32,52,40)
    # Relative to the counter of the common trailing zero. A small margin protects
    # against flat paper highlights or isolated compression noise.
    return sat((top-.012)/(.035+other)),sat((bot-.012)/(.035+other))

def localblue(ft,poses):
    u,v=grid(24,1);w=C['bluew'];res=[]
    for c in range(33,41):
        m=warp(ft,poses[:,c-26],u,v);l,r,b=m[:,0],m[:,1],m[:,2];ink=l-.4*b
        obj=unit(ink,w).reshape(len(l),-1);lo,hi=np.searchsorted(C['blueids'],[c,c+1]);sc=(obj @ C['blues'][lo:hi].T).max(1)
        if c==40:
            # A ring of white paper around black ink is not three white arrows:
            # there should be blue ground on BOTH sides of the orbit.
            rad=np.hypot(u,v);ground=np.minimum(b[:,rad<.39].mean(1),b[:,(rad>.75)&(rad<.90)].mean(1));s=.36*(sc-.50)+.16*(ground-.37)-.28*r[:,w>.4].mean(1)
        else:s=.36*(sc-.50)
        res.append(s)
    return np.stack(res,axis=1)

def classify(a):
    n=len(a); ft=prepare(a); p,base=poses(ft);scores=np.full((n,43),-2,np.float32)
    scores[:,12]=base[:,3]+cues(ft,p[:,3],3);scores[:,13]=base[:,2];scores[:,14]=base[:,4]+cues(ft,p[:,4],4);scores[:,17]=base[:,5]+cues(ft,p[:,5],5)
    base[:,6]+=.85*cues(ft,p[:,6],6)
    scores[:,33:41]=base[:,7:15]+localblue(ft,p)
    for group,k,ls in [('red',0,[0,1,2,3,4,5,7,8,9,10,16]),('warn',1,[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]),('end',6,[6,41,42])]:
        val,amp,carred=descriptors(a,ft,p[:,k],group);q=base[:,k]
        if k==6:
            scores[:,32]=q-.02+.055*sat((.15-amp)/.12)
            for c in ls:scores[:,c]=q-.03+.21*(val[:,c]-.36)+.045*sat((amp-.08)/.14)
        else:
            if k==0:
                numbers=[0,1,2,3,4,5,7,8];digits=ocr(a,p[:,0]);word=val[:,numbers]
                # The character classifier contributes mostly within its string length;
                # the whole-word recognizer is the one that can detect omitted ink.
                for start,stop in ((0,6),(6,8)):
                    d=digits[:,start:stop];z=word[:,start:stop];val[:,numbers[start:stop]]=.70*z+.30*(d-d.max(1,keepdims=True)+z.max(1,keepdims=True))
                upper,lower=counters(a,p[:,0]);val[:,5]+=.13*np.minimum(upper,lower);val[:,3]+=.055*lower*(1-upper)-.04*upper
                for c in (0,1,2,4):val[:,c]-=.025*(upper+lower)
                val[:,9]+=.10*carred;val[:,10]+=.10*carred
                # Lack of ink alone is ambiguous under low contrast: a blank face must
                # also lack coherent strokes at every tested numeral / vehicle position.
                coherent=val[:,ls].max(1);scores[:,15]=q+.075-.23*sat((amp-.043)/.19)-.27*np.maximum(0,coherent-.42)
            for c in ls:scores[:,c]=q+.31*(val[:,c]-.54)
    return scores.argmax(1)

def predict(x):
    # Pad partial blocks so every matrix product uses the same shape independently of batching.
    labels=np.empty(len(x),dtype=np.int64);bs=24
    for j in range(0,len(x),bs):
        a=x[j:j+bs];m=len(a)
        if m<bs:a=np.concatenate((a,np.repeat(a[-1:],bs-m,axis=0)))
        labels[j:j+m]=classify(a)[:m]
    return labels
'''
with open(os.path.join(out,'solution.py'),'w') as f:f.write(solution)
np.savez(os.path.join(out,'constants.npz'),**C)
