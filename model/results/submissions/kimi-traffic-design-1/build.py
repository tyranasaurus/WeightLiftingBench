"""Build a hand-designed German traffic-sign network.

All templates are rendered from geometric definitions of the signs.  No data
files are read.  The network compares normalized colour, colour-opponent and
edge features with those templates at several canonical poses.
"""
import sys, json
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SS = 3
N = 32
RED = np.array([.76, .055, .035])
WHITE = np.array([.92, .92, .87])
BLACK = np.array([.025, .03, .035])
BLUE = np.array([.035, .25, .67])
YELLOW = np.array([.98, .72, .045])
GREY = np.array([.20, .22, .23])
BACK = np.array([.40, .42, .39])

FONT = {
 '0':["01110","10001","10001","10001","10001","10001","01110"],
 '1':["00100","01100","00100","00100","00100","00100","01110"],
 '2':["01110","10001","00001","00010","00100","01000","11111"],
 '3':["11110","00001","00001","01110","00001","00001","11110"],
 '4':["00010","00110","01010","10010","11111","00010","00010"],
 '5':["11111","10000","10000","11110","00001","00001","11110"],
 '6':["01110","10000","10000","11110","10001","10001","01110"],
 '7':["11111","00001","00010","00100","01000","01000","01000"],
 '8':["01110","10001","10001","01110","10001","10001","01110"],
 '9':["01110","10001","10001","01111","00001","00001","01110"],
 'S':["01111","10000","10000","01110","00001","00001","11110"],
 'T':["11111","00100","00100","00100","00100","00100","00100"],
 'O':["01110","10001","10001","10001","10001","10001","01110"],
 'P':["11110","10001","10001","11110","10000","10000","10000"],
}

class Canvas:
    def __init__(self, scale=1., dx=0., dy=0., aspect=1., angle=0.):
        q = np.arange(N*SS, dtype=np.float64)
        xx, yy = np.meshgrid(q, q)
        # Inverse rigid transform, in units of the outer sign radius.
        x = (xx/(SS) - 15.5 - dx)
        y = (yy/(SS) - 15.5 - dy)
        a = np.deg2rad(angle)
        xr = np.cos(a)*x + np.sin(a)*y
        yr = -np.sin(a)*x + np.cos(a)*y
        self.x = xr/(13.*scale*aspect)
        self.y = yr/(13.*scale)
        self.a = np.zeros((N*SS,N*SS,3), dtype=np.float64) + BACK
    def paint(self, m, color):
        self.a[m] = color
    def circle(self, cx=0., cy=0., r=1.):
        return (self.x-cx)**2+(self.y-cy)**2 <= r*r
    def ellipse(self, cx, cy, rx, ry):
        return ((self.x-cx)/rx)**2+((self.y-cy)/ry)**2 <= 1
    def rect(self, x0,y0,x1,y1):
        return (self.x>=min(x0,x1))&(self.x<=max(x0,x1))&(self.y>=min(y0,y1))&(self.y<=max(y0,y1))
    def poly(self, verts):
        x,y=self.x,self.y
        inside=np.zeros_like(x,dtype=bool)
        for i in range(len(verts)):
            x1,y1=verts[i-1]; x2,y2=verts[i]
            cross=((y1>y)!=(y2>y)) & (x < (x2-x1)*(y-y1)/(y2-y1+1e-30)+x1)
            inside ^= cross
        return inside
    def line(self, points, width):
        x,y=self.x,self.y
        out=np.zeros_like(x,dtype=bool)
        for p,q in zip(points[:-1],points[1:]):
            vx,vy=q[0]-p[0],q[1]-p[1]
            t=np.clip(((x-p[0])*vx+(y-p[1])*vy)/(vx*vx+vy*vy+1e-30),0,1)
            out |= (x-p[0]-t*vx)**2+(y-p[1]-t*vy)**2 <= (width/2)**2
        return out
    def ring(self,cx,cy,r,width):
        d=np.sqrt((self.x-cx)**2+(self.y-cy)**2)
        return np.abs(d-r)<=width/2
    def text(self, text, cx=0., cy=0., h=.85, xscale=1., color=BLACK):
        cell=h/7
        widths=[5*cell*xscale for _ in text]
        spacing=.75*cell*xscale
        total=sum(widths)+spacing*(len(text)-1)
        left=cx-total/2
        for ch,w in zip(text,widths):
            for row,bits in enumerate(FONT[ch]):
                for col,bit in enumerate(bits):
                    if bit=='1':
                        x0=left+col*w/5; x1=left+(col+1)*w/5
                        y0=cy-h/2+row*cell; y1=y0+cell
                        self.paint(self.rect(x0,y0,x1,y1),color)
            left += w+spacing
    def down(self,a=None):
        if a is None: a=self.a
        return a.reshape(N,SS,N,SS,*a.shape[2:]).mean((1,3))

def circle_base(c, border=True, blue=False):
    c.paint(c.circle(r=1), WHITE if blue else RED)
    c.paint(c.circle(r=.84 if blue else .79), BLUE if blue else WHITE)

def triangle_base(c, invert=False):
    v=[(-1,-.72),(1,-.72),(0,1.)] if invert else [(0,-1),(-.96,.76),(.96,.76)]
    c.paint(c.poly(v),RED)
    if invert: v=[(-.72,-.48),(.72,-.48),(0,.72)]
    else: v=[(0,-.68),(-.65,.51),(.65,.51)]
    c.paint(c.poly(v),WHITE)

def car(c,cx,cy,s,color=BLACK,truck=False):
    if truck:
        c.paint(c.rect(cx-.08*s,cy-.38*s,cx+.50*s,cy+.18*s),color)
        c.paint(c.poly([(cx-.50*s,cy+.18*s),(cx-.50*s,cy-.20*s),(cx-.28*s,cy-.38*s),(cx-.08*s,cy-.38*s),(cx-.08*s,cy+.18*s)]),color)
    else:
        c.paint(c.rect(cx-.52*s,cy-.10*s,cx+.52*s,cy+.20*s),color)
        c.paint(c.poly([(cx-.34*s,cy-.10*s),(cx-.17*s,cy-.36*s),(cx+.20*s,cy-.36*s),(cx+.38*s,cy-.10*s)]),color)
    for xx in (-.32,.32):
        c.paint(c.circle(cx+xx*s,cy+.22*s,.13*s),color)

def pedestrian(c,cx,cy,s=1.,child=False):
    c.paint(c.circle(cx,cy-.38*s,.11*s),BLACK)
    c.paint(c.poly([(cx-.10*s,cy-.25*s),(cx+.09*s,cy-.23*s),(cx+.09*s,cy+.09*s),(cx-.09*s,cy+.09*s)]),BLACK)
    c.paint(c.line([(cx-.25*s,cy-.10*s),(cx,cy-.18*s),(cx+.22*s,cy-.02*s)],.14*s),BLACK)
    c.paint(c.line([(cx-.03*s,cy+.06*s),(cx-.22*s,cy+.43*s)],.16*s),BLACK)
    c.paint(c.line([(cx-.03*s,cy+.06*s),(cx+.23*s,cy+.39*s)],.16*s),BLACK)

def arrow(c,points,width=.18,head=.30):
    c.paint(c.line(points,width),WHITE)
    (x0,y0),(x1,y1)=points[-2],points[-1]
    v=np.array([x1-x0,y1-y0]); v=v/(np.linalg.norm(v)+1e-30)
    p=np.array([-v[1],v[0]])
    tip=np.array([x1,y1])+v*head*.32
    base=np.array([x1,y1])-v*head*.72
    c.paint(c.poly([tuple(tip),tuple(base+p*head*.58),tuple(base-p*head*.58)]),WHITE)

def render_sign(k,scale=1.,dx=0.,dy=0.,aspect=1.,angle=0.):
    c=Canvas(scale,dx,dy,aspect,angle)
    # Extra analytic IDs provide symbol-free reference signs for score offsets.
    if k==43:
        triangle_base(c)
        outer=c.poly([(0,-1.04),(-1,.80),(1,.80)]); inner=c.poly([(0,-.64),(-.60,.48),(.60,.48)])
    elif k==44:
        circle_base(c,blue=True)
        outer=c.circle(r=1.04); inner=c.circle(r=.82)
    elif k==45:
        v=[(-.43,-1),(.43,-1),(1,-.43),(1,.43),(.43,1),(-.43,1),(-1,.43),(-1,-.43)]
        c.paint(c.poly(v),RED)
        outer=c.poly([(x*1.04,y*1.04) for x,y in v]); inner=c.poly([(x*.80,y*.80) for x,y in v])
    elif k==46:
        c.paint(c.circle(r=1),RED)
        outer=c.circle(r=1.04); inner=c.circle(r=.82)
    # The masks are in canonical sign coordinates and are transformed with it.
    elif k in list(range(0,11))+[15,16,32,41,42]:
        circle_base(c)
        if k<=5 or k in (7,8):
            txt={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[k]
            c.text(txt,h=.88 if len(txt)==2 else .73,xscale=.86 if len(txt)==2 else .68)
        elif k==6:
            c.text('80',h=.88,xscale=.86)
            for off in (-.60,-.20,.20,.60):
                c.paint((np.abs(c.x+c.y-off)<.075)&c.circle(r=.80),GREY)
        elif k==9:
            car(c,-.38,.04,.68,BLACK); car(c,.38,.04,.68,RED)
        elif k==10:
            car(c,-.38,.04,.70,BLACK,True); car(c,.38,.04,.70,RED,True)
        elif k==16:
            car(c,0,.04,1.25,BLACK,True)
        elif k==32:
            for off in (-.72,-.36,0.,.36,.72):
                c.paint((np.abs(c.x+c.y-off)<.065)&c.circle(r=.80),GREY)
        elif k in (41,42):
            car(c,0,.04,1.15,BLACK,k==42)
            for off in (-.60,-.20,.20,.60):
                c.paint((np.abs(c.x+c.y-off)<.070)&c.circle(r=.80),GREY)
        outer=c.circle(r=1.04); inner=c.circle(r=.77)
    elif k==11:
        triangle_base(c)
        c.paint(c.line([(0,-.40),(0,.48)],.16),BLACK)
        c.paint(c.line([(-.30,.02),(.30,.02)],.15),BLACK)
        outer=c.poly([(0,-1.04),(-1,.80),(1,.80)]); inner=c.poly([(0,-.64),(-.60,.48),(.60,.48)])
    elif k==12:
        c.paint(c.poly([(0,-1),(1,0),(0,1),(-1,0)]),WHITE)
        c.paint(c.poly([(0,-.77),(.77,0),(0,.77),(-.77,0)]),YELLOW)
        outer=c.poly([(0,-1.04),(1.04,0),(0,1.04),(-1.04,0)])
        inner=c.poly([(0,-.74),(.74,0),(0,.74),(-.74,0)])
    elif k==13:
        triangle_base(c,True)
        outer=c.poly([(-1.04,-.76),(1.04,-.76),(0,1.04)])
        inner=c.poly([(-.68,-.43),(.68,-.43),(0,.67)])
    elif k==14:
        v=[(-.43,-1),(.43,-1),(1,-.43),(1,.43),(.43,1),(-.43,1),(-1,.43),(-1,-.43)]
        c.paint(c.poly(v),RED)
        c.paint(c.ring(0,0,0,0),RED)  # no-op; keeps the drawing purely geometric
        vi=[(x*.91,y*.91) for x,y in v]
        c.paint(c.poly(vi),WHITE)
        vi=[(x*.84,y*.84) for x,y in v]
        c.paint(c.poly(vi),RED)
        c.text('STOP',h=.36,xscale=1.10,color=WHITE)
        outer=c.poly([(x*1.04,y*1.04) for x,y in v]); inner=c.poly([(x*.80,y*.80) for x,y in v])
    elif k==17:
        c.paint(c.circle(r=1),RED); c.paint(c.rect(-.64,-.14,.64,.14),WHITE)
        outer=c.circle(r=1.04); inner=c.circle(r=.82)
    elif 18<=k<=31:
        triangle_base(c)
        if k==18:
            c.paint(c.line([(0,-.43),(0,.15)],.16),BLACK); c.paint(c.circle(0,.39,.09),BLACK)
        elif k in (19,20):
            s=-1 if k==19 else 1
            pts=[(s*.15,.50),(s*.15,.10),(s*.02,-.10),(s*-.28,-.43)]
            c.paint(c.line(pts,.18),BLACK)
        elif k==21:
            c.paint(c.line([(.18,.50),(-.16,.17),(.17,-.13),(-.15,-.48)],.17),BLACK)
        elif k==22:
            c.paint(c.rect(-.65,.23,.65,.38),BLACK)
            c.paint(c.circle(-.30,.22,.22),BLACK); c.paint(c.circle(.30,.22,.22),BLACK)
            c.paint(c.rect(-.65,.22,.65,.39),BLACK)
        elif k==23:
            c.paint(c.poly([(-.27,-.42),(.27,-.42),(.35,-.05),(-.35,-.05)]),BLACK)
            c.paint(c.rect(-.35,-.13,.35,.02),BLACK)
            c.paint(c.line([(-.18,.04),(-.30,.20),(-.12,.48)],.09),BLACK)
            c.paint(c.line([(.18,.04),(.30,.20),(.12,.48)],.09),BLACK)
        elif k==24:
            c.paint(c.line([(-.30,-.47),(-.30,.50)],.15),BLACK)
            c.paint(c.line([(.06,-.47),(.30,.02),(.30,.50)],.15),BLACK)
        elif k==25:
            c.paint(c.circle(-.15,-.35,.10),BLACK)
            c.paint(c.line([(-.13,-.23),(-.02,.10),(-.25,.42)],.14),BLACK)
            c.paint(c.line([(-.10,-.12),(.20,.02),(.38,.30)],.11),BLACK)
            c.paint(c.line([(.27,-.10),(.48,.43)],.08),BLACK)
            c.paint(c.poly([(.20,.43),(.60,.43),(.43,.25)]),BLACK)
            c.paint(c.line([(-.25,.10),(.10,.42)],.12),BLACK)
        elif k==26:
            c.paint(c.rect(-.22,-.50,.22,.50),BLACK)
            for yy,col in zip((-.30,0,.30),(RED,YELLOW,np.array([.04,.48,.18]))):
                c.paint(c.circle(0,yy,.105),col)
            c.paint(c.rect(-.29,-.43,.29,.43),BLACK)
            for yy,col in zip((-.30,0,.30),(RED,YELLOW,np.array([.04,.48,.18]))):
                c.paint(c.circle(0,yy,.085),col)
        elif k==27:
            pedestrian(c,0,.02,1.12)
        elif k==28:
            pedestrian(c,-.18,.03,.93); pedestrian(c,.28,.16,.66,True)
        elif k==29:
            c.paint(c.ring(-.36,.25,.23,.075),BLACK); c.paint(c.ring(.36,.25,.23,.075),BLACK)
            c.paint(c.line([(-.36,.25),(-.08,-.05),(.10,.25),(-.36,.25),(-.08,.25),(.20,-.10),(.36,.25)],.075),BLACK)
            c.paint(c.line([(-.15,-.10),(.02,-.10)],.07),BLACK)
            c.paint(c.circle(.03,-.38,.085),BLACK)
            c.paint(c.line([(.01,-.28),(-.08,-.04),(.20,-.10)],.10),BLACK)
        elif k==30:
            for a in (0,np.pi/3,2*np.pi/3):
                v=np.array([np.cos(a),np.sin(a)])
                p=np.array([-v[1],v[0]])
                c.paint(c.line([tuple(-.48*v),tuple(.48*v)],.09),BLACK)
                for sg in (-1,1):
                    z=v*(sg*.28)
                    c.paint(c.line([tuple(z),tuple(z-v*sg*.15+p*.12)],.065),BLACK)
                    c.paint(c.line([tuple(z),tuple(z-v*sg*.15-p*.12)],.065),BLACK)
        elif k==31:
            c.paint(c.ellipse(-.05,.05,.38,.20),BLACK)
            c.paint(c.line([(-.28,.14),(-.34,.47)],.10),BLACK)
            c.paint(c.line([(-.08,.17),(-.02,.47)],.10),BLACK)
            c.paint(c.line([(.17,.13),(.29,.45)],.10),BLACK)
            c.paint(c.line([(.20,-.02),(.35,-.33)],.15),BLACK)
            c.paint(c.ellipse(.40,-.34,.16,.10),BLACK)
            c.paint(c.line([(.36,-.40),(.27,-.58),(.18,-.62)],.055),BLACK)
            c.paint(c.line([(.39,-.41),(.43,-.60),(.52,-.64)],.055),BLACK)
            c.paint(c.line([(.29,-.52),(.20,-.49)],.045),BLACK)
            c.paint(c.line([(.44,-.53),(.54,-.49)],.045),BLACK)
        outer=c.poly([(0,-1.04),(-1,.80),(1,.80)]); inner=c.poly([(0,-.64),(-.60,.48),(.60,.48)])
    elif 33<=k<=40:
        circle_base(c,blue=True)
        if k==33:
            arrow(c,[(0,.57),(0,-.18),(.15,-.34),(.48,-.34)],.19,.32)
        elif k==34:
            arrow(c,[(0,.57),(0,-.18),(-.15,-.34),(-.48,-.34)],.19,.32)
        elif k==35:
            arrow(c,[(0,.57),(0,-.48)],.20,.42)
        elif k in (36,37):
            arrow(c,[(0,.58),(0,-.46)],.18,.36)
            s=1 if k==36 else -1
            arrow(c,[(0,.30),(0,-.02),(s*.18,-.18),(s*.46,-.18)],.17,.30)
        elif k in (38,39):
            s=1 if k==38 else -1
            arrow(c,[(-s*.43,-.48),(s*.34,.32)],.22,.40)
        else:
            for deg in (-90,30,150):
                angles=np.deg2rad(np.linspace(deg-72,deg,7))
                pts=[(.43*np.cos(a),.43*np.sin(a)) for a in angles]
                arrow(c,pts,.15,.28)
        outer=c.circle(r=1.04); inner=c.circle(r=.82)
    else:
        raise ValueError(k)
    return c.down().transpose(2,0,1), c.down(outer.astype(np.float64)), c.down(inner.astype(np.float64))

# Linear feature bank.  Every signed opponent/gradient is accompanied by its
# negative; ReLU therefore produces a half-wave rectified feature pair.
features=[]  # (source, 3x3 kernel, group weight)
I3=np.zeros((3,3)); I3[1,1]=1
SOBX=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)/4
SOBY=SOBX.T
LAP=np.array([[0,1,0],[1,-4,1],[0,1,0]],float)
RGB=np.eye(3)
GRAY=np.array([1/3,1/3,1/3])
ROPP=np.array([1,-.5,-.5]); BOPP=np.array([-.5,-.5,1]); YOPP=np.array([.5,.5,-1])
for ch in range(3): features.append((RGB[ch],I3,.18))
features.append((GRAY,I3,.35))
for v in (ROPP,BOPP,YOPP):
    features.extend([(v,I3,.08),(-v,I3,.08)])
for ker in (SOBX,SOBY):
    features.extend([(GRAY,ker,.20),(GRAY,-ker,.20)])
features.extend([(GRAY,LAP,.08),(GRAY,-LAP,.08)])
for v in (ROPP,BOPP):
    for ker in (SOBX,SOBY):
        features.extend([(v,ker,.06),(v,-ker,.06)])
F=len(features)
FW=np.array([f[2] for f in features],dtype=np.float64)
CW=np.zeros((F,3,3,3),dtype=np.float64)
for j,(source,ker,_) in enumerate(features):
    for ch in range(3): CW[j,ch]=source[ch]*ker

def feature_maps(a):
    """Reference for the fixed conv -> ReLU -> norm feature extractor."""
    z=np.pad(a,((0,0),(1,1),(1,1)))
    out=[]
    for source,ker,_ in features:
        n=a.shape[1]
        v=np.zeros(a.shape[1:],dtype=np.float64)
        for ch in range(3):
            for dy in range(3):
                for dx in range(3):
                    v += source[ch]*ker[dy,dx]*z[ch,dy:dy+n,dx:dx+n]
        out.append(np.maximum(v,0))
    return np.asarray(out)

def crop(a,n):
    q=(N-n)//2
    return a[...,q:q+n,q:q+n]

poses=[]
for s in (.78,.90,1.02,1.14): poses.append(dict(scale=s))
for dx,dy in ((-1.25,0),(1.25,0),(0,-1.25),(0,1.25)):
    poses.append(dict(scale=.96,dx=dx,dy=dy))
poses.extend([dict(scale=.96,aspect=.85),dict(scale=.96,aspect=1.15),
              dict(scale=.96,angle=-6),dict(scale=.96,angle=6)])
V=len(poses)
weights={}
layers=[]
def add(name,op,inputs,**kw):
    d=dict(name=name,op=op,inputs=inputs); d.update(kw); layers.append(d)

def feature_extractor(prefix,source):
    add(prefix+'_features','conv',[source],out=F,kernel=[3,3],stride=1,padding=1)
    weights[prefix+'_features.W']=CW.astype(np.float32)
    weights[prefix+'_features.b']=np.zeros(F,np.float32)
    add(prefix+'_relu','relu',[prefix+'_features'])
    add(prefix+'_norm','norm',[prefix+'_relu'])
    return prefix+'_norm'

# Generate each family of templates independently; nothing is learned or read.
templates={k:[render_sign(k,**p) for p in poses] for k in range(43)}
def neutral_id(k):
    if k in list(range(0,11))+[15,16,17,32,41,42]: return 15
    if k==11 or 18<=k<=31: return 43
    if 33<=k<=40: return 44
    if k==14: return 45
    if k==17: return 46
    return k
neutral={k:[render_sign(neutral_id(k),**p) for p in poses] for k in range(43)}
branch_specs=[('full',32,'global',.80),('sign',24,'outer',1.15),('inner',20,'inner',1.35)]
branch_scores=[]
for prefix,size,mode,branch_weight in branch_specs:
    if size==N:
        source='image'
    else:
        cname='crop'+str(size)
        kernel=N-size+1
        w=np.zeros((3,3,kernel,kernel),dtype=np.float32)
        for ch in range(3): w[ch,ch,kernel//2,kernel//2]=1
        add(cname,'conv',['image'],out=3,kernel=[kernel,kernel],stride=1,padding=0)
        weights[cname+'.W']=w; weights[cname+'.b']=np.zeros(3,np.float32)
        source=cname
    fin=feature_extractor(prefix,source)
    rows=[]
    biases=[]
    for k in range(43):
        for j,(im,outer,inner) in enumerate(templates[k]):
            if size!=N:
                im=crop(im,size); outer=crop(outer,size); inner=crop(inner,size)
            f=feature_maps(im)
            if mode=='global':
                den=size*size
                fn=(f-f.mean(axis=(1,2),keepdims=True))/np.sqrt(f.var(axis=(1,2),keepdims=True)+1e-5)
                w=fn/den
            else:
                mask=(outer if mode=='outer' else inner)>.45
                count=mask.sum()
                mu=(f*mask).sum(axis=(1,2),keepdims=True)/max(count,1)
                var=(((f-mu)*mask)**2).sum(axis=(1,2),keepdims=True)/max(count,1)
                fn=(f-mu)/np.sqrt(var+1e-5)*mask
                bad=(np.sqrt(var[:,0,0]+1e-5)<.003)
                fn[bad]=0
                w=fn/max(count,1)
            w*=branch_weight*(FW/np.sum(FW))[:,None,None]
            bias=0.
            if mode=='inner':
                nf=feature_maps(crop(neutral[k][j][0],size))
                nf=(nf-nf.mean(axis=(1,2),keepdims=True))/np.sqrt(nf.var(axis=(1,2),keepdims=True)+1e-5)
                bias=-float(np.sum(w*nf))
            rows.append(w.astype(np.float32).ravel())
            biases.append(bias)
    W=np.asarray(rows,dtype=np.float32)
    add(prefix+'_match','dense',[fin],out=43*V)
    weights[prefix+'_match.W']=W
    weights[prefix+'_match.b']=np.asarray(biases,dtype=np.float32)
    add(prefix+'_byclass','reshape',[prefix+'_match'],shape=[43,V,1])
    add(prefix+'_best','maxpool',[prefix+'_byclass'],size=[V,1],stride=[V,1])
    add(prefix+'_scores','reshape',[prefix+'_best'],shape=[43])
    branch_scores.append(prefix+'_scores')

# Tight central views exclude the outer ring.  Thus a blank sign has zero
# normalized central contrast, while digits and pictograms contribute positive
# evidence only when their own shapes are present.
for prefix,size,branch_weight in [('central16',16,1.70),('central12',12,1.25)]:
    cname='crop'+str(size); kernel=N-size+1
    cw=np.zeros((3,3,kernel,kernel),dtype=np.float32)
    for ch in range(3): cw[ch,ch,kernel//2,kernel//2]=1
    add(cname,'conv',['image'],out=3,kernel=[kernel,kernel],stride=1,padding=0)
    weights[cname+'.W']=cw; weights[cname+'.b']=np.zeros(3,np.float32)
    add(prefix+'_gray','conv',[cname],out=1,kernel=[1,1],stride=1,padding=0)
    weights[prefix+'_gray.W']=GRAY.reshape(1,3,1,1).astype(np.float32)
    weights[prefix+'_gray.b']=np.zeros(1,np.float32)
    add(prefix+'_norm','norm',[prefix+'_gray'])
    rows=[]
    for k in range(43):
        for j in range(V):
            f=crop(templates[k][j][0],size).mean(axis=0,keepdims=True)
            fn=(f-f.mean())/np.sqrt(f.var()+1e-5)
            rows.append((fn/(size*size)*branch_weight).astype(np.float32).ravel())
    add(prefix+'_match','dense',[prefix+'_norm'],out=43*V)
    weights[prefix+'_match.W']=np.asarray(rows,dtype=np.float32)
    weights[prefix+'_match.b']=np.zeros(43*V,np.float32)
    add(prefix+'_byclass','reshape',[prefix+'_match'],shape=[43,V,1])
    add(prefix+'_best','maxpool',[prefix+'_byclass'],size=[V,1],stride=[V,1])
    add(prefix+'_scores','reshape',[prefix+'_best'],shape=[43])
    branch_scores.append(prefix+'_scores')

# Direct photometric family evidence remains useful when normalized edges of a
# very dark image are mostly quantization noise.
add('colour_stats','conv',['crop24'],out=3,kernel=[1,1],stride=1,padding=0)
colour_w=np.zeros((3,3,1,1),dtype=np.float32)
colour_w[0,:,0,0]=ROPP; colour_w[1,:,0,0]=BOPP; colour_w[2,:,0,0]=YOPP
weights['colour_stats.W']=colour_w; weights['colour_stats.b']=np.zeros(3,np.float32)
add('colour_mean','avgpool',['colour_stats'],size=[24,24],stride=[24,24])
add('colour_scores','dense',['colour_mean'],out=43)
family_w=np.zeros((43,3),dtype=np.float32)
for k in list(range(0,12))+list(range(13,32))+[41,42]:
    family_w[k]=[2.6,-1.4,-.15]
for k in range(33,41): family_w[k]=[-1.4,2.8,-.15]
family_w[12]=[-.8,-1.0,3.2]
weights['colour_scores.W']=family_w; weights['colour_scores.b']=np.zeros(43,np.float32)
branch_scores.append('colour_scores')
add('raw_scores','add',branch_scores)
add('scores','dense',['raw_scores'],out=43)
weights['scores.W']=np.eye(43,dtype=np.float32)
score_bias=np.zeros(43,dtype=np.float32)
# Small conservatism offsets for deliberately broad, radially symmetric models.
score_bias[38]=-.12; score_bias[40]=-.18
weights['scores.b']=score_bias

with open(OUT/'network.json','w') as f:
    json.dump({'layers':layers},f,separators=(',',':'))
np.savez(OUT/'weights.npz',**weights)
print('wrote',len(layers),'layers and',sum(v.size for v in weights.values()),'weights to',OUT)
