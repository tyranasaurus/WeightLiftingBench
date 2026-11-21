"""Construct a traffic-sign recognizer from geometry. No input assets or data."""
import json, math, sys
from pathlib import Path
import numpy as np

# Small, dependency-free vector drawing vocabulary. Sign drawing coordinates have
# the usual SVG orientation (positive y down); a whole sign is 100 units wide.
def ellipse(x,y,a,b=None): return ('e',x,y,a,b if b is not None else a)
def polygon(*pts): return ('p',pts)
def stroke(w,*pts): return ('s',w,pts)
def rect(l,t,r,b): return polygon((l,t),(r,t),(r,b),(l,b))
def bez(p0,*sections):
    p=np.array(p0); pts=[tuple(p0)]
    for c1,c2,p1 in sections:
        q,r,s=map(np.array,(c1,c2,p1))
        for t in (.125,.25,.375,.5,.625,.75,.875,1.):
            pts.append(tuple((1-t)**3*p+3*t*(1-t)**2*q+3*t*t*(1-t)*r+t**3*s))
        p=s
    return pts

def inside(g,x,y):
    typ=g[0]
    if typ=='e':
        _,a,b,rx,ry=g; return ((x-a)/rx)**2+((y-b)/ry)**2<=1
    if typ=='p':
        pts=g[1]; value=False; a,b=pts[-1]
        for c,d in pts:
            if d!=b: value=value^(((b>y)!=(d>y))&(x<(c-a)*(y-b)/(d-b)+a))
            a,b=c,d
        return value
    _,w,pts=g; hits=False; a,b=pts[0]
    for c,d in pts[1:]:
        u,v=c-a,d-b
        if u*u+v*v>0:
            t=np.clip(((x-a)*u+(y-b)*v)/(u*u+v*v),0,1)
            hits=hits|((x-a-t*u)**2+(y-b-t*v)**2<=(w/2)**2)
        a,b=c,d
    return hits

def transformed(g,sx=1,sy=1,tx=0,ty=0):
    if g[0]=='e': return ellipse(tx+sx*g[1],ty+sy*g[2],abs(sx)*g[3],abs(sy)*g[4])
    pts=[(tx+sx*a,ty+sy*b) for a,b in g[-1]]
    return polygon(*pts) if g[0]=='p' else stroke(g[1]*(abs(sx)+abs(sy))/2,*pts)

def painted(gs,ink=1.): return [(g,ink) for g in gs]

# DIN-like numerals. Construct paths independently of installed fonts. Width
# and cap height of this little coordinate system are 10 and 17 respectively.
def digit(d,style):
    w=1.95 if style!=1 else 2.35
    if d=='0':
        if style==2:
            return [stroke(w,*[(5+3.8*math.cos(t),8.5+7.55*math.sin(t)) for t in np.linspace(0,2*math.pi,41)])]
        return [stroke(w,*bez((1.25,5),((1.25,2.1),(2.5,1),(5,1)),((7.45,1),(8.75,2.1),(8.75,5)),((8.85,8),(8.8,10),(8.75,12)),((8.75,14.8),(7.4,16),(5,16)),((2.6,16),(1.25,14.8),(1.25,12)),((1.2,10),(1.2,8),(1.25,5))))]
    if d=='1': return [polygon((1.1,4.25),(5.5,.1),(7.35,.1),(7.35,17),(5.05,17),(5.05,3.2),(1.1,6.4))]
    if d=='2':
        return [stroke(w,*bez((1.1,4.7),((1.65,-.7),(9.7,-.2),(8.65,5.15)),((8.6,6.7),(6.2,9),(5.2,10.4)),((4.0,11.9),(1.9,14.5),(1.1,16)))),rect(.2,15.0,9.7,17)]
    if d=='3':
        return [stroke(w,*bez((1.05,3.65),((2.7,-1.1),(9.5,.45),(8.45,5.1)),((8.2,7.0),(6.6,8.05),(4.15,8.05)),((7.3,7.75),(9.0,9.85),(8.8,12.55)),((8.55,16.8),(2.4,17.85),(.9,13.3))))]
    if d=='5':
        return [stroke(w,(8.55,1.05),(2,1.05),(1.6,8.15)),stroke(w,*bez((1.6,8.15),((3.5,6.1),(8.85,5.6),(8.85,10.6)),((9.7,17.1),(3.15,18.2),(1.1,13.5))))]
    if d=='6':
        p=bez((7.2,1),((5.75,3.8),(2,7.9),(1.55,10.5)),((.4,14.25),(2.5,16.05),(5.15,16.1)),((9.8,16.1),(9.9,8.6),(6.65,7.85)),((4.5,7),(2.65,7.8),(1.6,10.4)))
        return [stroke(w,*p)]
    if d=='7':
        gs=[polygon((.35,0),(9.55,0),(9.55,2.1),(4.05,17),(1.6,17),(7.45,2.05),(.35,2.05))]
        if style!=1:gs.append(rect(.35,.8,2.1,4.1))
        return gs
    if d=='8':
        return [stroke(w,*[(5+a*math.cos(t),cy+b*math.sin(t)) for t in np.linspace(0,2*math.pi,33)]) for cy,a,b in [(4.65,3.55,3.72),(12.1,3.95,4.02)]]
    raise ValueError(d)

def number(s,style):
    widths=[7 if c=='1' else 10 for c in s]; gap=(1.65,1.0,2.1)[style]; total=sum(widths)+gap*(len(s)-1)
    l=-total/2; gs=[]
    for d,wd in zip(s,widths):
        for g in digit(d,style): gs.append(transformed(g,1,1,l-(1 if d=='1' else 0),-8.5))
        l+=wd+gap
    return painted(gs)

# Pictograms in a local coordinate system spanning the interior of a sign.
def car(x,y,s=1.):
    body=polygon((-9,-2),(-7.6,-2.9),(-5.7,-8),(5.7,-8),(7.6,-3),(9,-1.8),(9,4.5),(7.8,4.5),(7.8,7),(5.1,7),(5.1,4.6),(-5.1,4.6),(-5.1,7),(-7.8,7),(-7.8,4.5),(-9,4.5))
    glass=polygon((-6.15,-2.7),(-4.8,-6.7),(4.8,-6.7),(6.15,-2.7))
    return [(transformed(body,s,s,x,y),1.),(transformed(glass,s,s,x,y),0.)]

def side_truck():
    return painted([rect(-30,-13,6,9),polygon((7,-10),(17,-10),(26,-2),(28,0),(28,9),(7,9)),rect(-29,8,27,11),ellipse(-20,10,5),ellipse(16,10,5)])+[(polygon((9,-8),(16,-8),(21,-2),(9,-2)),.05)]

def arrowhead(x,y,dx,dy,length,half):
    return polygon((x,y),(x-length*dx-half*dy,y-length*dy+half*dx),(x-length*dx+half*dy,y-length*dy-half*dx))

def arrows(k,sty):
    if k in (33,34,36,37):
        sg=1 if k in (33,36) else -1
        if k in (33,34):
            cy=-11; stemx=-14
            gs=[stroke(9,(stemx,32),(stemx,7),*bez((stemx,7),((stemx,-5),(-7,cy),(5,cy)))),rect(4,cy-4.5,17,cy+4.5),arrowhead(33,cy,1,0,18,15)]
        else:
            stemx=-12; cy=2
            gs=[rect(stemx-4.4,-22,stemx+4.4,35),arrowhead(stemx,-36,0,-1,18,15),stroke(8,(stemx,19),*bez((stemx,19),((stemx,12),(-4,cy),(5,cy)))),rect(4,cy-4,18,cy+4),arrowhead(34,cy,1,0,18,14)]
        return painted([transformed(g,sg,1) for g in gs])
    if k==35:
        return painted([rect(-5,-18,5,37),arrowhead(0,-38,0,-1,24,19)])
    if k in (38,39):
        # Directional arrow with swept-back wings, not just a filled triangle.
        gs=[polygon((-25,-30),(-17,-34),(18,13),(15,-8),(24,-2),(28,31),(-6,24),(-11,15),(10,18))]
        return painted([transformed(g,1 if k==38 else -1,1) for g in gs])
    gs=[]
    for angle in (0,120,240):
        t=math.radians(angle);c,s=math.cos(t),math.sin(t)
        # Tangential arrow at the bottom of a counterclockwise roundabout.
        # The elbows suggest the arc; the three arrowheads have equal status.
        if sty==0: local=[polygon((-26,17),(5,17),(7,12),(19,23),(6,33),(6,29),(-20,29))]
        else:local=[stroke(8,*[(27*math.cos(a),27*math.sin(a)) for a in np.deg2rad(np.linspace(147,90,12))]),arrowhead(15,26,1,0,14,11)]
        for g in local:
            pts=[(c*x-s*y,s*x+c*y) for x,y in g[-1]];gs.append(polygon(*pts) if g[0]=='p' else stroke(g[1],*pts))
    return painted(gs)

def pedestrian(x,y,s):
    gs=[ellipse(3,-13,3),stroke(5,(1,-8),(-3,2)),stroke(2.8,(0,-6),(5,-1),(10,0)),stroke(2.8,(-1,-6),(-7,-2),(-9,2)),stroke(4,(-3,2),(1,8),(6,15)),stroke(4,(-3,2),(-6,10),(-12,14))];return painted([transformed(g,s,s,x,y) for g in gs])

def warning(k,sty):
    if k==11:
        # Thick vertical priority road and two thin oblique side roads.
        return painted([polygon((-5.8,23),(-5.8,1),(-23,8),(-23,4),(-5.8,-5),(-5.8,-13),(0,-20),(5.8,-13),(5.8,-5),(23,4),(23,8),(5.8,1),(5.8,23))])
    if k==18: return painted([polygon((-3.4,-20),(3.4,-20),(1.9,9),(-1.9,9)),ellipse(0,16,3.1)])
    if k in (19,20):
        gs=[stroke(6.3,(7.5,18),(7.5,5),*bez((7.5,5),((7.5,-4),(4.5,-7),(-7,-7)))),arrowhead(-16,-7,-1,0,12,10)]
        return painted([transformed(g,1 if k==19 else -1,1) for g in gs])
    if k==21:
        return painted([stroke(6,(10,18),(10,5),*bez((10,5),((10,0),(5,-1),(1,-1)),((-5,-1),(-8,-1),(-8,-6))),(-8,-19))])
    if k==22:
        p=bez((-25,16),((-19,16),(-18,4),(-11,4)),((-5,4),(-4,14),(0,14)),((4,14),(5,4),(11,4)),((18,4),(19,16),(25,16)))
        return painted([polygon(*p,(25,19),(-25,19))])
    if k==23:
        gs=car(-1,-5,1.07) if sty==0 else painted([polygon((-15,-4),(-12,-7),(-9,-8),(-6,-12),(2,-12),(7,-8),(12,-6),(13,-2),(-15,-2)),ellipse(-9,-2,2.5),ellipse(8,-2,2.5)]);gs+=painted([polygon((-9,-7),(-6,-9),(6,-8),(9,-4),(9,-1),(-10,-2))]);
        for x in [-7,7]: gs+=painted([stroke(2.6,*bez((x,1),((x+2,6),(x+10,6),(x+7,11)),((x+5,16),(x-4,14),(x-5,22))))]);
        return gs
    if k==24: return painted([rect(-10,-17,-6,19),stroke(4.3,(10,19),(10,7),*bez((10,7),((10,0),(3,0),(3,-8))),(3,-17))])
    if k==25:
        return painted([ellipse(3,-14,3),polygon((-.5,-12),(4,-10),(2,-5),(-6,2),(-11,3),(-12,-1)),stroke(3,(-10,1),(-15,9),(-19,18)),stroke(3.6,(-9,1),(-6,8),(-10,18)),stroke(3,(-1,-8),(0,-3),(-3,2)),stroke(2.5,(-4,-7),(-7,-2),(-8,2)),stroke(1.7,(-11,-2),(12,12)),polygon((3,18),(7,15),(14,8),(17,7),(27,18))])
    if k==26:
        if sty==1: return painted([ellipse(0,y,5.6) for y in [-10,2,14]])
        return painted([rect(-6,-14,6,17),ellipse(0,-14,6),ellipse(0,17,6)])+[(ellipse(0,y,3.6),v) for y,v in [(-11,.6),(1,.25),(13,.55)]]
    if k==27:
        # Walking silhouette: clearly separated legs, compact torso, lowered
        # arms. Mirrored hand-drawn variants allow different pictogram vintages.
        art=painted([ellipse(2,-16,3.1),polygon((-2,-12),(4,-11),(3,-1),(0,5),(-5,5),(-3,-4)),stroke(3,(-1,-7),(-6,-2),(-9,4)),stroke(3,(3,-7),(6,-2),(10,0)),stroke(4.2,(-2,4),(-4,11),(-9,19)),stroke(4.0,(-1,3),(4,10),(8,18))]);return [(transformed(g,-1 if sty==0 else 1,1),v) for g,v in art]
    if k==28:
        # Two children running, small child holding the taller child's hand.
        art=pedestrian(6,-1,.96)+pedestrian(-9,6,.67)+painted([polygon((2,-5),(8,-7),(9,0),(14,7),(3,9),(1,6)),stroke(2.4,(-7,1),(0,-1),(5,-7))]);return [(transformed(g,-1 if sty==1 else 1,1),v) for g,v in art]
    if k==29:
        gs=[ellipse(7,-12,3),stroke(4.2,(4,-8),(-1,-2),(3,3),(1,10)),stroke(2.3,(3,-6),(7,-1),(12,-1))] if sty==1 else [stroke(2.0,(-8,0),(-2,0)),stroke(2.1,(9,0),(8,-4),(13,-4))];
        for x in (-12,12):
            gs+=[stroke(2.05,*[(x+8*math.cos(t),11+8*math.sin(t)) for t in np.linspace(0,2*math.pi,26)])]
        gs += [stroke(1.8,(-12,11),(-5,-1),(0,10),(9,0),(12,11)),stroke(1.8,(-12,11),(0,10))];return painted(gs)
    if k==30:
        gs=[]
        for t in np.arange(6)*np.pi/3-np.pi/2:
            u,v=math.cos(t),math.sin(t);gs.append(stroke(2.7,(0,2),(17*u,2+17*v)))
            for n in (-1,1):gs.append(stroke(2.,(8*u,2+8*v),(12*u-5*n*v,2+12*v+5*n*u)))
        return painted(gs)
    if k==31:
        return painted([polygon((-18,-12),(-15,-8),(-11,-10),(-8,-5),(-8,-1),(1,2),(11,1),(15,3),(17,2),(15,8),(12,9),(11,16),(9,19),(8,17),(8,9),(1,9),(-4,6),(-10,4),(-12,-2),(-16,-3),(-20,-6)),stroke(2.25,(-6,4),(-11,8),(-15,5)),stroke(2.2,(11,9),(17,13),(18,16)),stroke(1.8,(-13,-6),(-10,-13),(-11,-16))])
    raise ValueError(k)

# Render an individual drawing on an oversampled grid using area coverage;
# bilinear pullback is used later for each parameterized template. No combined
# statistics of two different templates are ever used.
def raster(art):
    n=256; span=128.; coord=(np.arange(n)+.5)*span/n-span/2; x,y=np.meshgrid(coord,coord); im=np.zeros((n,n),np.float32)
    for g,v in art: im=np.where(inside(g,x,y),v,im)
    return im.astype(np.float32)

def sample(im,x,y):
    u=np.clip(2*x+127.5,0,254.99);v=np.clip(2*y+127.5,0,254.99);i=u.astype(int);j=v.astype(int);u=u-i;v=v-j
    return (1-v)*((1-u)*im[j,i]+u*im[j,i+1])+v*((1-u)*im[j+1,i]+u*im[j+1,i+1])

def grid(n):
    a=(np.arange(n*3)+.5)/3-n/2; return np.meshgrid(a,a)

def aa(im,n):return im.reshape(n,3,n,3).mean(axis=(1,3))
def smooth(a):
    q=np.pad(a,1,mode='edge'); h,w=a.shape
    return sum(x*y*q[i:i+h,j:j+w]/16 for i,x in enumerate((1,2,1)) for j,y in enumerate((1,2,1)))

def unit(a):return a/max(1e-7,np.linalg.norm(a))

def matched(pic,mask):
    # Each signed correlation filter is DC-free on its support. The partly
    # high-passed term discounts broad shading while preserving stroke mass.
    def dc(a): return (a-(a*mask).sum()/max(mask.sum(),1e-7))*mask
    a=unit(dc(pic)); detail=unit(dc(pic-smooth(smooth(pic))));return unit(.76*a+.24*detail)

layers=[]; weights={}
def layer(name,op,src,**kw):
    layers.append(dict(name=name,op=op,inputs=[src] if isinstance(src,str) else src,**kw));return name

def linear(name,src,W,b=0,op='dense',**kw):
    W=np.asarray(W,np.float32);weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(np.asarray(b,np.float32),(len(W),)).copy();return layer(name,op,src,out=len(W),**kw)

def point(name,src,W,b=0):return linear(name,src,np.asarray(W)[:,:,None,None],b,op='conv',kernel=[1,1],stride=1,padding=0)
def reshape(name,src,shape):return layer(name,'reshape',src,shape=list(shape))
def nor(name,src):return layer(name,'norm',src)
def rel(name,src):return layer(name,'relu',src)
def pool(name,op,src,size,stride):return layer(name,op,src,size=list(size),stride=list(stride))

def crop(name,src,n,k):
    # Center crop as a fixed channel-independent selection operator.
    f=np.zeros(((n-k)//2*2+1,)*2); f[len(f)//2,len(f)//2]=1
    return linear(name,src,f[None,None],op='conv',kernel=list(f.shape),stride=1,padding=0)

def clipnorm(name,src,nc,bound):
    z=point(name+'_levels',src,np.vstack((np.eye(nc),np.eye(nc))),[bound]*nc+[-bound]*nc);z=rel(name+'_hinges',z)
    z=point(name, z,np.hstack((np.eye(nc),-np.eye(nc))),-bound);return z


# Coarse sign families; grey signs with and without a cancelled inscription
# share a group and use a separate interior matcher.
families=['ring','warning','diamond','yield','stop','entry','blue','end']
family=['ring']*43
for k in [11,*range(18,32)]:family[k]='warning'
for k,s in [(12,'diamond'),(13,'yield'),(14,'stop'),(17,'entry')]:family[k]=s
for k in range(33,41): family[k]='blue'
for k in [6,32,41,42]:family[k]='end'

def geom_values(g,X,Y):
    # Spatial patterns for luminance, red-vs-neutral, blue-vs-neutral and
    # yellow-vs-neutral. Outside the sign is unknown and gets little weight.
    x=X; y=Y; rr=np.hypot(x,y); area=rr<=1; inner=rr<.775
    if g in ('warning','yield'):
        yy= y if g=='warning' else -y
        # Signed barycentric distance, rounded corners by a circular fillet.
        def tri(x,y,r):
            # Equilateral in physical units, affine-scaled to a square crop.
            p=[(0.,-1.+2*r),(-1.+1.5*r,1.-r),(1.-1.5*r,1.-r)]
            hull=(y<=p[1][1]) & (2*np.abs(x)<=y-p[0][1]); dist=999.
            for (a,b),(c,d) in zip(p,p[1:]+p[:1]):
                u=c-a;v=d-b;t=np.clip(((x-a)*u+(y-b)*v)/(u*u+v*v),0,1);dist=np.minimum(dist,(x-a-t*u)**2+(y-b-t*v)**2)
            return hull|(dist<r*r)
        area=tri(x,yy,.08);inner=tri(x/.71,(yy-.095)/.72,.035)
    if g=='diamond':
        dd=np.abs(x)+np.abs(y);area=dd<=1.03;inner=dd<.74
    if g=='stop':
        octo=np.maximum.reduce((np.abs(x),np.abs(y),(np.abs(x)+np.abs(y))/1.414));area=octo<1;inner=octo<.94
    white=.88;lum=np.full_like(x,.33); red=np.zeros_like(x);blue=red.copy();yellow=red.copy();relia=np.where(area,1.,.13)*np.exp(-np.maximum(0,rr-1.1)*4)
    if g in ('ring','warning','yield'):
        rim=area&~inner;lum[area]=white;lum[rim]=.24;red[rim]=1
        if g!='yield':
            relia[inner]*=np.where((x[inner]/.65)**2+((y[inner]-(.32 if g=='warning' else 0))/.52)**2<1,.24,.85)
    elif g=='diamond':lum[area]=white;lum[inner]=.68;yellow[inner]=1
    elif g=='stop':
        lum[area]=.93;lum[inner]=.24;red[inner]=1;letter_zone=inner&(np.abs(x)<.84)&(np.abs(y)<.26);lum[letter_zone]=.65;red[letter_zone]=.42
    elif g=='entry':
        bar=(np.abs(x)<.86)&(np.abs(y)<.18);lum[area]=.23;lum[bar&area]=.95;red[area&~bar]=1
    elif g=='blue':
        lum[area]=.25;blue[area]=1;zone=(rr<.65);relia[zone]=.36;lum[zone]=.37;blue[zone]=.82
    elif g=='end':
        lum[area]=white; band=area & (abs(y+.99*x)<.285);lum[band]=.24;lum[area&(rr>.946)]=.46
    return np.stack([lum,red,blue,yellow]),relia,area


def nonnum_art(k,sty):
    if k==9:return car(-15,-1)+car(14,1)
    if k==10:return painted([rect(-35,-18,-9,8),rect(-33,7,-28,13),rect(-17,7,-12,13)])+car(13,3)
    if k==16:return side_truck()
    raise ValueError(k)


def cancel_art(k):
    for style in (0,1):
        art=[]
        if k==6:art=[(transformed(g,2.55,2.55),v*(.65 if style==0 else .92)) for g,v in number('80',0)]
        elif k in (41,42):art=[(g,v*.85) for g,v in nonnum_art(9 if k==41 else 10,0)]
        # A bundle of parallel dark cancellation strokes.
        for p0 in np.linspace(-12,12,5):art+=painted([stroke(2.2,(-45,45+p0),(45,-45+p0))]);
        yield style,art,lambda X,Y:X*X+Y*Y<39**2


# Whole sign pose grid. An interior hypothesis only compares against other
# interiors at the SAME sign pose. Outline and interior use different inputs,
# but their latent pose is shared until the last max pool.
POSES=[(d*ax,d*ay,dx,dy) for d in (16.,19.5,23.,26.5,30.)
       for ax,ay in [(1.,1.),(.86,1.03),(1.04,.88)]
       for dx,dy in [(0,0),(-1.7,0),(1.7,0),(0,-1.7),(0,1.7),(-1.3,-1.3),(1.3,-1.3),(-1.3,1.3),(1.3,1.3)]]

def stop_letters():
    return painted([stroke(2.1,*bez((-11,-6),((-16,-10),(-21,-5),(-17,-1)),((-15,1),(-11,0),(-11,4)),((-11,9),(-17,10),(-19,6)))),rect(-9.5,-8,-.5,-6),rect(-6.1,-8,-3.9,8),stroke(2,*bez((2,-4),((2,-9),(10,-10),(10,-4)),((10,0),(10,2),(10,4)),((10,10),(2,10),(2,4)),((2,2),(2,0),(2,-4)))),rect(13,-8,15,8),stroke(2,(14,-7),(18,-7),*bez((18,-7),((24,-7),(24,1),(18,1))),(14,1))])

def geometry_art(k):
    # Tuple: ink drawing, nominal symbol-to-sign transform, interior mask.
    # Lengths here are nominal design proportions in units of 1/100 of the
    # outer sign, unrelated to any individual photograph.
    if k in (0,1,2,3,4,5,7,8):
        s={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[k]
        for w,h,sty in [(1.92,2.32,0),(2.43,2.60,1),(1.86,2.85,1),(2.35,2.20,2),(2.42,2.85,0),(1.73,2.40,2)]:
            yield number(s,sty),(w*(.9 if len(s)==3 else 1),h,0,0)
    elif k in (9,10,16):
        for sx,sy,yy,sty in [(1,1,0,0),(1.12,1.18,0,1),(.91,1.25,1,0),(1.18,.85,-1,1),(.85,.9,-1,1),(1.04,1.35,0,0)]:
            if k==16:
                # German lorry silhouette faces left.
                art=[(transformed(g,-1,1),v) for g,v in side_truck()]
            else:
                art=nonnum_art(k,sty); ne=2 if k==9 else 3
                art=[(g,v*(.74 if i<ne and sty==0 else 1)) for i,(g,v) in enumerate(art)]
            yield art,(sx,sy,0,yy)
    elif k==11 or 18<=k<=31:
        for sx,sy,yy,sty in [(1.06,1.08,12,0),(1.0,1.22,14,1),(1.2,1.2,11,0),(.9,.96,13,0),(1.1,.94,15,1),(1,1.25,9,0)]:
            yield warning(k,sty),(sx,sy,0,yy)
    elif 33<=k<=40:
        for sx,sy,yy,sty in [(1,1,0,0),(.84,.90,0,1),(1.1,1.04,-1,0),(.94,1.08,1,1),(1.08,.94,0,1),(.91,.97,1,0)]:
            yield arrows(k,sty),(sx,sy,0,yy)
    elif k in (6,32,41,42):
        for i in range(6):
            art=next(a for j,a,s in cancel_art(k) if j==i%2)
            yield art,((.9,1.,1.09)[i//2],(.95,1.05)[i%2],0,0)
    elif k==14:
        for s,h in [(1.65,1.65),(1.85,1.95),(1.72,1.95),(1.93,1.55),(1.57,1.85),(1.79,1.68)]:yield stop_letters(),(s,h,0,0)
    elif k==17:
        for w,h in [(83,18),(79,20),(82,23),(76,17),(86,21),(84,16)]:yield painted([rect(-w/2,-h/2,w/2,h/2)]),(1,1,0,0)
    else:raise ValueError(k)

def signs_outline(src,absolute):
    x,y=grid(16);rows=[];raw=[];gate=[];rim_test=[]
    for g in families:
        for w,h,dx,dy in POSES:
            X=(x-dx/2)/(w/4);Y=(y-dy/2)/(h/4);v,m,area=geom_values(g,X,Y);mask=aa(m,16);vec=[]
            for i,gain in enumerate((1.,.40,.40,.37)):
                p=aa(v[i],16);vec.append(gain*matched(p,mask) if i==0 or p.max()>0 else np.zeros_like(p))
            rows.append(vec)
            # Absolute opponent evidence complements mean-centred correlation.
            # A diagonal stripe in a white disk should not need red or blue.
            ch={'ring':0,'warning':0,'yield':0,'stop':0,'entry':0,'blue':1,'diamond':2}.get(g)
            a=np.zeros((3,16,16)); fg=aa(area,16);fg=fg/max(fg.sum(),1e-6)
            if ch is not None:
                tint=aa(v[ch+1],16);a[ch]=5*tint/max(tint.sum(),1e-6)
                if g in ('warning','ring','yield','entry'):
                    inside_white=aa(area&(v[ch+1]<.2),16);a[ch]-=2*inside_white/max(inside_white.sum(),1e-6)
                if ch==1:a[0]-=fg*1.3
            else:
                a[0]-=fg*3.5;a[1]-=fg*3.;a[2]-=fg*1.5
            raw.append(a)
            if g=='ring':
                # Require a dark perimeter around a light inner annulus in
                # more than one quadrant; the upper edge of a dark vehicle
                # or part of a STOP letter is not a complete circular frame.
                rr=X*X+Y*Y
                for q in ((X>abs(Y)),(X<-abs(Y)),(Y>abs(X)),(Y<-abs(X))):
                    a1=aa((rr>.82**2)&(rr<.97**2)&q,16);a2=aa((rr>.57**2)&(rr<.73**2)&q,16);rim_test.append(a1/max(a1.sum(),1e-6)-a2/max(a2.sum(),1e-6))
            if g in ('stop','entry'):
                ann=aa(area&(abs(Y)>.34)&(X*X+Y*Y<.80**2),16);ann/=max(ann.sum(),1e-8)
                # Most of the face, not just the rim, must be red. A white
                # centre surrounded by a heavy red ring is neither STOP nor
                # no-entry. Suppress it with a signed interior-vs-rim test.
                rim=aa(area&(X*X+Y*Y>.8**2),16);rim/=max(rim.sum(),1e-8);a[0]+=4.5*(ann-rim)
            if g=='blue':
                # Positive evidence for actual blue paint, independently of
                # spatial pattern correlation. Reject uncoloured discs with a
                # plausible bright blob somewhere in the centre.
                ann=aa(area&(X*X+Y*Y>.45**2),16);ann/=max(ann.sum(),1e-8);gate.append(np.stack([ann*.23,-ann,ann*.10]))
    b=linear('missing_blue',absolute,np.array(gate).reshape(len(POSES),-1),.39);b=rel('missing_blue_penalty',b);b=reshape('blue_poses',b,(1,1,len(POSES)));scale=np.zeros((8,1));scale[families.index('blue')]=-15.;b=point('uncoloured_discs',b,scale)
    P=len(POSES);l=point('body_luma',src,[[1,0,0,0]]);q=linear('four_sectors',l,np.array(rim_test).reshape(P*4,-1));q=reshape('ring_sector_axis',q,(1,P,4));q=pool('weakest_sector','maxpool',q,(1,4),(1,4));q=reshape('ring_continuity',q,(1,1,P));scale=np.zeros((8,1));scale[families.index('ring')]=-1.45;q=point('complete_ring',q,scale)
    lum=linear('outer_correlations',src,np.asarray(rows).reshape(len(rows),-1));chrom=linear('paint_evidence',absolute,np.asarray(raw).reshape(len(raw),-1));outer=layer('outer_and_paint','add',[lum,chrom]);outer=reshape('family_poses',outer,(len(families),1,len(POSES)));return layer('body_scores','add',[outer,b,q])

def detail_features(v):
    # The red channel distinguishes black pictogram ink from red boundary
    # paint; local adaptation makes strokes in a shadowed centre visible.
    # Compute a soft black-hat (local upper envelope minus the input).
    r=point('red_sensitive',v,[[.78,.16,.06]]);r=nor('ink_local_contrast',crop('ink_field',r,32,24));r=clipnorm('ink_bounded',r,1,3.)
    high=pool('paper_envelope','maxpool',r,(5,5),(1,1));neg=point('negative_print',crop('ink_centre',r,24,20),[[-1]]);bh=layer('dark_detail','add',[high,neg]);return bh

def cancellation(bh):
    P=len(POSES);x,y=grid(20);bars=[];isolated=[];plain=[]
    for k in [6,41,42]:
        # Draw in symbol coordinates. Two half-discs remain visible on either
        # side of the bundle of crossing lines.
        arts=[(transformed(g,2.31,2.45),v) for g,v in number('80',1)] if k==6 else nonnum_art(9 if k==41 else 10,0)
        cached=raster(arts)
        for w,h,dx,dy in POSES:
            X=(x-dx)*100/w;Y=(y-dy)*100/h;rr=X**2+Y**2
            if k==6:
                for half in (X>0,X<=0):
                    zone=aa((rr<42**2)&(rr>25**2)&half,20);un=aa((rr<42**2)&(rr>25**2)&half&(abs(Y+X)<10),20);bars.append(-unit(un-zone*un.sum()/max(zone.sum(),1e-8)))
                unused=aa((rr<31**2)&(abs(Y+X)>14),20);plain.append(unused/max(unused.sum(),1e-8))
            for scale in (.9,1.06):
                pic=aa(sample(cached,X/scale,Y/scale),20);m=aa((rr<38**2)&(abs(Y+.99*X)>11),20)
                a=matched(pic,m);ink=max(.6,np.linalg.norm((pic-(pic*m).sum()/max(m.sum(),1e-6))*m));isolated.append(a*(.9+1.2/ink))
    # Minimum of the two slash ends, using maxpool on their negatives.
    f=linear('slash_ends',bh,np.array(bars).reshape(P*2,-1));f=reshape('diagonal_halves',f,(1,P,2));f=pool('both_ends','maxpool',f,(1,2),(1,2));f=reshape('diagonal_pose',f,(1,1,P));W=np.zeros((43,1));W[[6,32,41,42]]=-2.1;f=point('diagonal_cancellation',f,W)
    a=linear('beside_slash',bh,np.array(isolated).reshape(3*P*2,-1));a=reshape('side_scales',a,(3,P,2));a=pool('side_peaks','maxpool',a,(1,2),(1,2));a=reshape('side_prints',a,(3,1,P));W=np.zeros((43,3));W[[6,41,42],[0,1,2]]=1;a=point('visible_cancelled_print',a,W)
    b=linear('beside_slash_energy',bh,np.array(plain).reshape(P,-1));b=reshape('uncrossed',b,(1,1,P));W=np.zeros((43,1));W[32]=-1.65;b=point('expected_blank_halves',b,W);return f,a,b

def triangle_features(feature,ks):
    # Local matches within a triangle: splitting into upper and lower
    # pictogram parts tolerates changes to the triangle proportions and
    # walking or working poses. Use a shared nominal axis. Fine outlines
    # alone should not outweigh the silhouette at this resolution.
    n=20;x,y=grid(n);P=len(POSES);scores=[]
    for part in (0,1):
        rows=[]
        for k in ks:
            arts=[raster(warning(k,s)) for s in (0,1)]
            for w,h,dx,dy in POSES:
                X=(x-dx)*100/w;Y=(y-dy)*100/h
                mask=(Y<38)&(abs(X)<.48*(Y+37));m=aa(mask&(Y<(16 if part==0 else 42))&(Y>(-28 if part==0 else 10)),n)
                for sx,sy,shift,s in [(1.,1.,0,0),(1.22,1.18,0,1),(.90,1.22,-3,0),(1.12,1.05,4,0),(1.,1.13,1,1)]:
                    pic=aa(sample(arts[s],(X-shift)/sx,(Y-12)/sy),n);mass=max(.65,np.linalg.norm((pic-(m*pic).sum()/max(m.sum(),1e-7))*m));rows.append(matched(pic,m)*(.65+1.1/mass))
        a=linear('section_'+str(part),feature,np.asarray(rows).reshape(len(ks)*P*5,-1));a=reshape('shape_section_'+str(part),a,(len(ks),P,5));a=pool('select_section_'+str(part),'maxpool',a,(1,5),(1,5));a=reshape('part_poses_'+str(part),a,(len(ks),1,P));scores.append(a)
    a=layer('parts_combined','add',scores);W=np.zeros((43,len(ks)));W[ks,np.arange(len(ks))]=.68;return point('triangle_parts',a,W)

def regional_cues(v,bh):
    P=len(POSES);x,y=grid(20);lamp=[];vehicle=[];empty=[]
    for w,h,dx,dy in POSES:
        X=(x-dx)*100/w;Y=(y-dy)*100/h
        def patch(xc,yc,rx,ry):
            m=aa(((X-xc)/rx)**2+((Y-yc)/ry)**2<1,20);return m/max(m.sum(),1e-6)
        # Keep these opponent filters zero-sum spatially, so an image-wide
        # colour cast is not evidence for a little three-colour signal.
        top=patch(0,1,6,7);bottom=patch(0,26,6,6);middle=patch(0,13,6,6);bg=(patch(-12,20,4,6)+patch(12,20,4,6))/2
        lamp.append(np.stack([3.2*top-2.6*bottom-.6*bg,2.6*bottom-3.2*top+.6*bg+1.5*(middle-bg),-1.5*(middle-bg)]))
        left=patch(-15,-1,10,8);right=patch(14,0,9,7);vehicle.append(np.stack([left-right,-.7*(left-right),-.3*(left-right)]))
        empty.append(patch(0,0,29,24))
    # A blue disc in daylight and a white prohibition centre can both look
    # grey. Distinguish an empty disk from one with ink by local contrast.
    blank=linear('blank_ink',bh,np.array(empty).reshape(P,-1));blank=reshape('blank_ink_pose',blank,(1,1,P));s=np.zeros((43,1));s[15]=-3.;blank=point('penalize_printed_blank',blank,s)
    # crop the logarithmic primaries together, without mixing channels.
    ker=np.zeros((3,3,13,13));ker[np.arange(3),np.arange(3),6,6]=1;cl=linear('central_colors',v,ker,op='conv',kernel=[13,13],stride=1,padding=0)
    a=linear('traffic_lamp_colours',cl,np.asarray(lamp).reshape(P,-1));a=rel('lamp_presence',a);a=reshape('lamp_poses',a,(1,1,P));s=np.zeros((43,1));s[26]=1.4;a=point('signal_colour_evidence',a,s)
    b=linear('red_vehicle',cl,np.asarray(vehicle).reshape(P,-1));b=rel('red_vehicle_presence',b);b=reshape('car_colour_poses',b,(1,1,P));s=np.zeros((43,1));s[[9,10]]=1.8;b=point('left_car_evidence',b,s);return a,b,blank

def soft_ink(ink):
    # Soft saturation discounts isolated, extremely dark cores and keeps the
    # less contrasted shoulder of a blurred stroke. It depends only on the
    # input activation, not on any population-level calibration.
    z=point('stroke_levels',ink,[[1.],[1.],[1.]],[0.,-.65,-1.4]);z=rel('stroke_hinges',z);return point('stroke_coverage',z,[[1.35,-.74,-.43]])

def printed_digits(feature):
    # Model the digit string as parts on a shared type baseline. A narrow
    # character can shift sideways a little without moving an adjacent one.
    # Template filters are zero-mean and individually normalized. On the
    # two-digit limits, recognition depends primarily on the first glyph.
    ks=[0,1,2,3,4,5,7,8];ss=['20','30','50','60','70','80','100','120'];glyphs={(d,j):raster([(transformed(g,1,1,-5,-8.5),v) for g,v in painted(digit(d,j))]) for d in '012345678' if d!='4' for j in range(3)}
    x,y=grid(20);P=len(POSES);stypars=[(2.18,2.45,0),(2.30,2.75,1),(1.95,2.32,1),(2.1,2.55,2)];nstyles=len(stypars);K=len(ks);targets=[]
    for slot in range(3):
        filt=[];use=[j for j,s in enumerate(ss) if slot<len(s)]
        for j in use:
            s=ss[j];widths=[7 if d=='1' else 10 for d in s];gap=1.5;left=-.5*(sum(widths)+gap*(len(s)-1))+sum(widths[:slot])+gap*slot;cx=left+5-(1 if s[slot]=='1' else 0);d=s[slot]
            for w,h,dx,dy in POSES:
                for sx,sy,sty in stypars:
                    sx*=.87 if len(s)==3 else 1.; X=((x-dx)*100/w)/sx-cx; Y=((y-dy)*100/h)/sy
                    for shift in (-.92,0,.92):
                        u=X-shift;p=aa(sample(glyphs[d,sty],u,Y),20);m=aa((abs(u)<(5.95 if d!='1' else 4.5))&(abs(Y)<10.),20);f=matched(p,m);sc=max(np.linalg.norm((p-(m*p).sum()/max(m.sum(),1e-8))*m),.45)
                        filt.append(f*(.85+1.5/sc)*(1.0 if slot==0 and len(s)==2 else .68 if len(s)==2 else .66))
        name='glyph_'+str(slot);a=linear(name,feature,np.asarray(filt).reshape(len(filt),-1));a=reshape(name+'_spacing',a,(len(use),P*nstyles,3));a=pool(name+'_best_spacing','maxpool',a,(1,3),(1,3));a=reshape(name+'_typeface',a,(len(use),P,nstyles));sel=np.zeros((K,len(use)));sel[use,np.arange(len(use))]=1;targets.append(point(name+'_place',a,sel))
    a=layer('compose_numerals','add',targets);a=pool('typeface_max','maxpool',a,(1,nstyles),(1,nstyles));a=reshape('number_pose',a,(K,1,P));s=np.zeros((43,K));s[ks,np.arange(K)]=1;return point('numerical_detail',a,s)

def count_strokes(ink):
    # A projection onto the upright strokes of an inscription, ignoring the
    # identity of the glyphs. In particular the whitespace between two
    # characters should not be mistaken for the middle character of three.
    P=len(POSES);x,y=grid(20);ks=[0,1,2,3,4,5,7,8];strings=['20','30','50','60','70','80','100','120'];filt=[]
    for k,s in zip(ks,strings):
        mats=[raster(number(s,q)) for q in range(3)]
        for w,h,dx,dy in POSES:
            X=(x-dx)*100/w;Y=(y-dy)*100/h
            for sx,sy,q in [(2.16,2.5,0),(2.,2.8,1),(2.42,2.65,2),(1.81,2.32,0)]:
                sx*=.9 if len(s)==3 else 1.;pic=aa(sample(mats[q],X/sx,Y/sy),20);m=aa((X**2+Y**2<34**2)&(abs(X)<(35 if len(s)==3 else 30))&(abs(Y)<sy*7.4),20)
                # A vertical projection of THIS template only. It yields a
                # rank-one correlation filter with little sensitivity to a
                # jagged horizontal contour or digit baseline variation.
                col=(pic*m).sum(axis=0)/np.maximum(m.sum(axis=0),1.e-6);pat=np.broadcast_to(col,pic.shape);filt.append(matched(pat,m))
    a=linear('vertical_stroke_profiles',ink,np.array(filt).reshape(len(ks)*P*4,-1));a=reshape('upright_widths',a,(len(ks),P,4));a=pool('upright_width','maxpool',a,(1,4),(1,4));a=reshape('upright_pose',a,(len(ks),1,P));s=np.zeros((43,len(ks)));s[ks,np.arange(len(ks))]=.80;return point('ink_column_evidence',a,s)

def pose_classifier(mid,outer,bh,v):
    P=len(POSES);n=20;x,y=grid(n);rows=[];whites=[];order=list(range(43))
    for k in order:
        g=family[k];ref=np.zeros((P,n,n),np.float32)
        if k in (12,13,15):
            F=np.zeros((P,6,n,n),np.float32)
        else:
            arts=[(raster(art),trans) for art,trans in geometry_art(k)]
            F=[]
        for p,(w,h,dx,dy) in enumerate(POSES):
            X=(x-dx)*100/w;Y=(y-dy)*100/h
            if g=='warning':
                # Inner triangular field, with a guard zone from the red border.
                mask=(Y<37)&(abs(X)<.48*(Y+37))
            elif g=='ring':mask=(X**2+Y**2<34.8**2)
            elif g=='end':mask=(X**2+Y**2<44**2)
            elif g=='blue':mask=(X**2+Y**2<44.5**2)
            else:mask=(abs(X)<43)&(abs(Y)<33)&(X*X+Y*Y<47**2)
            m=aa(mask,n);sgn=1 if g in ('blue','stop','entry') else -1
            if g=='ring':
                center=aa((X/30)**2+(Y/24)**2<1,n);sur=aa((X*X+Y*Y<34.5**2)&((X/31)**2+(Y/25)**2>1),n)
                if k==15:ref[p]=2.5*(center/max(center.sum(),1e-7)-sur/max(sur.sum(),1e-7))
            if k in (12,13,15):continue
            variants=[]
            for im,(sx,sy,tx,ty) in arts:
                pic=aa(sample(im,(X-tx)/sx,(Y-ty)/sy),n);corr=matched(pic,m)
                # Normalize the ink-vs-paper contrast for symbol footprint.
                # A small, thin pictogram isn't supposed to lose merely because
                # it has less mass. Retain part of the unit-energy correlation
                # for robustness to nuisance responses from very small strokes.
                diff=pic-(m*pic).sum()/max(m.sum(),1e-7);ink_norm=max(np.linalg.norm(m*diff),.65)
                variants.append(sgn*corr*(2.15+4./ink_norm))
            F.append(variants)
        rows.append(np.asarray(F,dtype=np.float32));whites.append(ref)
    # Split class banks into modest-sized layers, only meaningful symbol
    # matchers are materialized. Use fixed 1x1 projections to place outputs
    # in the corresponding class planes without large dense routing matrices.
    wh=linear('clear_centres',mid,whites[15].reshape(P,-1));wh=reshape('blank_poses',wh,(1,1,P));sel=np.zeros((43,1));sel[15]=1;wh=point('unprinted',wh,sel)
    sel=np.zeros((43,len(families)))
    for i,g in enumerate(family):sel[i,families.index(g)]=1.
    base=point('assign_outline',outer,sel)
    score=[base,wh]; all_k=[k for k in order if k not in (12,13,15)]
    for j in range(0,len(all_k),8):
        ks=all_k[j:j+8];W=np.stack([rows[k] for k in ks]);tag='print_'+str(j);a=linear(tag+'_match',mid,W.reshape(len(ks)*P*6,-1));a=reshape(tag+'_forms',a,(len(ks),P,6));a=pool(tag+'_best','maxpool',a,(1,6),(1,6));a=reshape(tag+'_poses',a,(len(ks),1,P));sel=np.zeros((43,len(ks)))
        for i,k in enumerate(ks):sel[k,i]=1
        score.append(point(tag+'_assign',a,sel))
    # Separate within-family symbol discrimination from between-family
    # decisions. This prevents a high-contrast arrow or numeral alone from
    # overwhelming a plainly incompatible exterior shape.
    whole=layer('pictograms','add',score[2:]);stroke_map=soft_ink(bh);gl=printed_digits(stroke_map);mix=np.eye(43);numbers=[0,1,2,3,4,5,7,8];mix[numbers,numbers]=.70
    # Fine text has both a whole-word and a part score. Other small dark
    # pictograms receive an additional red-sensitive match on fine detail.
    dks=[k for k in all_k if family[k] in ('warning','ring','end') and k not in numbers];chunks=[]
    for j in range(0,len(dks),7):
        ks=dks[j:j+7];tag='detail_'+str(j);F=np.stack([rows[k] for k in ks]);a=linear(tag,bh,-F.reshape(len(ks)*P*6,-1));a=reshape(tag+'_variants',a,(len(ks),P,6));a=pool(tag+'_max','maxpool',a,(1,6),(1,6));a=reshape(tag+'_poses',a,(len(ks),1,P));sel=np.zeros((43,len(ks)));sel[ks,np.arange(len(ks))]=.41;chunks.append(point(tag+'_place',a,sel));mix[ks,ks]=.66
    lamp,red_car,blank=regional_cues(v,bh);slash,uncovered,noinscription=cancellation(bh);warns=[11,*range(18,32)];part=triangle_features(bh,warns);mix[warns,warns]=.45
    coarse=point('whole_print',whole,mix);sym=layer('composite_print','add',[coarse,gl,count_strokes(stroke_map),lamp,red_car,part,uncovered,noinscription,*chunks]);mapped=[]
    for g in families:
        ks=[k for k in all_k if family[k]==g]
        if not ks:continue
        w=np.zeros((len(ks),43));w[np.arange(len(ks)),ks]=1;tag='group_'+g
        a=point(tag,sym,w);a=reshape(tag+'_axis',a,(1,len(ks),P));a=pool(tag+'_peak','maxpool',a,(len(ks),1),(len(ks),1));v=np.zeros((43,1));v[ks]=-.63
        mapped.append(point(tag+'_discount',a,v))
    # For a sign with no ink, compare the central intensity to the white annulus.
    # No interior-correlation bonus if there is no visible symbol.
    b=np.zeros(43);b[[12,13]]=3.;b[15]=2.5
    # Compensation for colourless signs: the outline is luminance-only.
    b[[6,32,41,42]]=1.
    bias=point('symbol_scores',sym,np.eye(43),b);sumall=layer('at_same_pose','add',[base,wh,blank,slash,bias,*mapped]);a=pool('best_joint_pose','maxpool',sumall,(1,P),(1,P));return reshape('joint_scores',a,(43,))

def build(out):
    # The positive red/blue/yellow contrasts are evaluated after a fixed
    # concave illumination transform. Implemented by slopes of log(x+a).
    knots=np.array([0,.015,.035,.073,.14,.27,.50,.75,1.0]);slopes=np.diff(np.log(knots+.025))/np.diff(knots)
    eye=np.eye(3);v=point('exposure_breakpoints','image',np.repeat(eye,len(slopes),axis=0),np.tile(-knots[:-1],3));v=rel('exposure_hinges',v);w=np.kron(eye,np.diff(np.r_[0,slopes])[None,:]);v=point('log_rgb',v,w)
    y=point('brightness',v,[[.26,.52,.22]]);middle=crop('symbol_field',y,32,20);mid=nor('symbol_contrast',middle);mid=clipnorm('limited_contrast',mid,1,2.5)
    # Feature normalization amplifies dim signs; rectify chromatic evidence
    # before normalizing to avoid complementary-colour false alarms.
    opp=point('opponency',v,[[.26,.52,.22],[1,-.63,-.37],[-.68,-.32,1],[.36,.64,-1]],[0,-.12,-.10,-.13]);opp=rel('positive_opponency',opp);a=pool('body_resolution','avgpool',opp,(2,2),(2,2));a=nor('body_contrast',a);a=clipnorm('body_clipped',a,4,2.6)
    absolute=point('paint_strength','body_resolution',np.eye(4)[1:]);outer=signs_outline(a,absolute)
    bh=detail_features(v);pose_classifier(mid,outer,bh,v)
    out.mkdir(parents=True,exist_ok=True);(out/'network.json').write_text(json.dumps({'layers':layers},indent=1));np.savez(out/'weights.npz',**weights)

if __name__=='__main__':build(Path(sys.argv[1]))
