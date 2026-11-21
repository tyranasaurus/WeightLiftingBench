"""Construct sign stencils entirely from geometric primitives; no input files."""
import sys
from pathlib import Path
import math
import numpy as np

# Drawing coordinates are percentages of a sign bounding box, not camera pixels.
A=192
v=(np.arange(A)+.5)/A*100
X,Y=np.meshgrid(v,v)

def polygon(pts):
    inside=np.zeros_like(X,dtype=bool)
    a,b=pts[-1]
    for c,d in pts:
        if d!=b:
            inside ^= ((Y>b)!=(Y>d)) & (X<(c-a)*(Y-b)/(d-b)+a)
        a,b=c,d
    return inside

def oval(x,y,rx,ry):return ((X-x)/rx)**2+((Y-y)/ry)**2<=1

def line(points,w):
    out=np.zeros(X.shape,bool)
    a,b=points[0]
    for c,d in points[1:]:
        dx,dy=c-a,d-b
        t=np.clip(((X-a)*dx+(Y-b)*dy)/max(dx*dx+dy*dy,1e-8),0,1)
        out |= (X-a-t*dx)**2+(Y-b-t*dy)**2<=(w*.5)**2
        a,b=c,d
    return out

def cubic(p,q,r,s):
    t=np.linspace(0,1,15)[:,None];p,q,r,s=map(np.array,(p,q,r,s))
    return (1-t)**3*p+3*t*(1-t)**2*q+3*t*t*(1-t)*r+t**3*s

def spline(st,w):
    points=[st[0]]
    for k in range(1,len(st),3):points.extend(cubic(points[-1],*st[k:k+3])[1:])
    return line(points,w)

def arc(x,y,rx,ry,a,b,w):
    t=np.linspace(np.radians(a),np.radians(b),35);return line(list(zip(x+rx*np.cos(t),y+ry*np.sin(t))),w)

# DIN-like numerals, stroked centerlines. The condensed version has the same
# design but less counter space. Six and seven have diagonal entry strokes.
def numeral(ch,x,y,w,h,bold,style):
    weight=bold*h
    def tr(pts):return [(x+a*(w-weight)+weight/2,y+b*(h-weight)+weight/2) for a,b in pts]
    def p(pts):return line(tr(pts),weight)
    def c(pts):return spline(tr(pts),weight)
    def loop(cy,ry):
        t=np.linspace(0,2*math.pi,65);power=.83 if style else .75
        xx=np.sign(np.cos(t))*abs(np.cos(t))**power;yy=np.sign(np.sin(t))*abs(np.sin(t))**power
        return p(list(zip(.5+.5*xx,cy+ry*yy)))
    if ch=='0':return loop(.5,.5)
    if ch=='1':return p([(.17,.23),(.82,0),(.82,1)])
    if ch=='2':return c([(0,.23),(0,-.06),(1,-.1),(1,.22),(1,.42),(.48,.6),(.01,.96)])|p([(.01,.96),(.01,1),(1,1)])
    if ch=='3':return c([(.01,.16),(.2,-.1),(1,-.05),(.97,.23),(.97,.38),(.72,.47),(.39,.47)])|c([(.39,.47),(.81,.45),(1,.57),(1,.74),(1,1.07),(.13,1.1),(0,.83)])
    if ch=='5':return p([(.94,0),(.09,0),(.06,.49)])|c([(.06,.49),(.3,.32),(1,.32),(1,.68),(1,.83),(1,1),(.54,1),(.21,1),(.07,.94),(0,.85)])
    if ch=='6':return c([(.74,0),(.57,.14),(.02,.50),(0,.68)])|loop(.716,.284)
    if ch=='7':return p([(0,.13 if style else 0),(0,0),(1,0),(.28,1)])
    if ch=='8':return loop(.235,.235)|loop(.738,.262)
    raise ValueError(ch)

def numbers(text,cx=50,cy=50,w=0,h=40,bold=.115,style=0):
    gap=h*.085
    if not w:w=(.54 if len(text)==2 else .455)*h
    widths=[w*(.55 if ch=='1' else 1.) for ch in text]
    start=cx-(sum(widths)+gap*(len(text)-1))/2
    m=X<0
    for ch,cw in zip(text,widths):
        m |= numeral(ch,start,cy-h/2,cw,h,bold,style);start+=cw+gap
    return m

def car(cx,cy,w,h):
    def pts(a):return [(cx+x*w,cy+y*h) for x,y in a]
    body=polygon(pts([(-.49,.4),(-.49,-.02),(-.39,-.13),(-.29,-.46),(-.19,-.5),(.23,-.5),(.32,-.43),(.41,-.14),(.49,-.02),(.49,.4)]))
    body &= ~polygon(pts([(-.32,-.13),(-.22,-.38),(.22,-.38),(.32,-.13)]))
    for s in [-1,1]:body |= polygon(pts([(s*.46,.31),(s*.29,.31),(s*.29,.5),(s*.46,.5)]))
    return body

def truck(cx,cy,w,h,front=False):
    def pts(a):return [(cx+x*w,cy+y*h) for x,y in a]
    if front:
        m=polygon(pts([(-.48,-.5),(.48,-.5),(.48,.42),(-.48,.42)]));m &= ~polygon(pts([(-.34,-.34),(.34,-.34),(.34,-.06),(-.34,-.06)]))
    else:
        m=polygon(pts([(-.5,-.48),(.15,-.48),(.15,-.17),(.35,-.17),(.5,.09),(.5,.36),(-.5,.36)]));m &= ~polygon(pts([(.21,-.09),(.32,-.09),(.4,.09),(.21,.09)]))
    for s in [-.32,.31]:m |= oval(cx+s*w,cy+.33*h,.09*w,.16*h)
    return m

def rotatepoints(pts,deg,x=50,y=50):
    a=np.radians(deg);c,s=np.cos(a),np.sin(a);return [(x+c*(i-x)-s*(j-y),y+s*(i-x)+c*(j-y)) for i,j in pts]

def tip(cx,cy,sz,deg=0):
    a=[(cx,cy),(cx+sz,cy+sz*1.04),(cx+sz,cy+sz*1.49),(cx,cy+sz*.65),(cx-sz,cy+sz*1.49),(cx-sz,cy+sz*1.04)]
    return polygon(rotatepoints(a,deg,cx,cy))

def up(cx,top,bot,hw,sw,deg):
    a=[(cx-sw,bot),(cx+sw,bot),(cx+sw,top+hw*.95),(cx+hw,top+hw*1.5),(cx+hw,top+hw),(cx,top),(cx-hw,top+hw),(cx-hw,top+hw*1.5),(cx-sw,top+hw*.95)]
    return polygon(rotatepoints(a,deg))

def aheadbranch():
    m=up(40,10,87,14,4.8,0)
    m|=spline([(40,78),(40,61),(48,50),(62,50)],9)|line([(62,50),(78,50)],9)|tip(87,50,13,90)
    return m

def arrow(k):
    if k in [33,34]:
        m=line([(34,85),(34,58)],9)|spline([(34,58),(34,41),(45,34),(57,34)],9)|line([(57,34),(69,34)],9)|tip(79,34,17,90)
        return m if k==33 else m[:,::-1]
    if k==35:return up(50,10,88,21,5,0)
    if k in [36,37]:
        m=aheadbranch();return m if k==36 else m[:,::-1]
    if k in [38,39]:
        m=up(50,10,88,21,5.5,135);return m if k==38 else m[:,::-1]
    m=X<0
    for rot in [0,120,240]:
        t=np.linspace(np.radians(-92+rot),np.radians(-176+rot),36);pts=list(zip(50+32*np.cos(t),50+32*np.sin(t)))
        m|=line(pts,6.2);xx,yy=pts[-1];m|=tip(xx,yy,9,-176+rot)
    return m

def stickperson(cx,cy,h,flip=1):
    def p(z):return [(cx+flip*h*a,cy+h*b) for a,b in z]
    return oval(cx+.10*h*flip,cy-.38*h,.095*h,.097*h)|line(p([(.04,-.21),(-.08,.06)]),.15*h)|line(p([(-.05,0),(-.16,.28),(-.31,.45)]),.095*h)|line(p([(-.05,0),(.09,.21),(.19,.45)]),.1*h)|line(p([(.04,-.19),(.13,-.03),(.28,.015)]),.075*h)|line(p([(.03,-.20),(-.16,-.15),(-.24,.01)]),.075*h)

def warning(k):
    if k==11:return polygon([(44,79),(56,79),(56,61),(69,64),(56,54),(56,49),(50,36),(44,49),(44,54),(31,64),(44,61)])
    if k==18:return polygon([(47,70),(53,70),(54,40),(53,35),(47,35),(46,40)])|oval(50,77,4.1,4.1)
    if k in [19,20]:
        m=line([(60,78),(60,68)],8)|spline([(60,68),(60,54),(52,48),(42,48)],8)|polygon([(42,48),(48,42),(40,44),(37,49),(43,54)])
        return m if k==19 else m[:,::-1]
    if k==21:return polygon([(56,78),(65,78),(65,54),(62,53),(47,61),(47,42),(39,46),(39,70),(42,70),(56,62)])
    if k==22:return polygon([(23,82),(27,77),(30,71),(34,66),(40,65),(45,68),(49,74),(53,69),(58,66),(64,66),(70,72),(74,78),(78,82)])
    if k==23:
        vehicle=polygon(rotatepoints([(33,53),(38,45),(54,44),(59,49),(64,50),(64,58),(60,61),(34,61),(32,56)],14))
        m=vehicle
        for a in [-14,14]:m|=spline([(50+a,59),(60+a,65),(52+a,68),(47+a,69),(38+a,72),(41+a,77),(48+a,79)],2.4)
        return m
    if k==24:return line([(40,82),(40,34)],4.0)|spline([(60,82),(60,70),(61,62),(56,56),(51,50),(51,46),(51,42)],4.3)
    if k==25:return oval(49,44,4,4)|line([(48,51),(40,61)],7)|line([(42,60),(39,70),(33,79)],4.8)|line([(42,61),(46,68),(47,79)],4.8)|line([(47,52),(52,59),(59,60)],3)|line([(40,50),(64,73)],2)|polygon([(48,81),(54,76),(59,71),(64,69),(69,72),(76,81)])
    if k==26:return oval(50,46,6.1,6.1)|oval(50,60,6.1,6.1)|oval(50,74,6.1,6.1)
    if k==27:return stickperson(50,61,39,-1)
    if k==28:return stickperson(58,58,40,1)|stickperson(39,66,27,-1)|line([(42,62),(49,56),(52,54)],2.9)
    if k==29:
        m=arc(36,72,9,9,0,360,2.4)|arc(64,72,9,9,0,360,2.4)|line([(36,72),(46,60),(52,72),(36,72),(57,59),(64,72),(58,53),(55,53)],2.2)
        return m|oval(54,42,4,4)|line([(51,48),(44,55),(51,61),(48,68)],4)|line([(51,48),(54,54),(59,54)],2.8)
    if k==30:
        m=X<0
        for deg in range(0,360,60):
            m|=line(rotatepoints([(50,61),(50,39)],deg,50,61),4.5)|line(rotatepoints([(43,43),(50,50),(57,43)],deg,50,61),3.5)|line(rotatepoints([(44,51),(50,56),(56,51)],deg,50,61),2.8)
        return m
    if k==31:
        m=polygon([(33,53),(40,54),(45,58),(58,63),(62,65),(61,70),(57,71),(52,68),(42,67),(35,61)])|polygon([(37,60),(38,50),(43,45),(45,46),(43,52),(44,59)])|line([(40,63),(31,67),(30,71)],3)|line([(39,64),(35,71),(32,72)],2.4)|line([(60,66),(66,71),(67,78)],3.4)|line([(56,68),(60,74),(64,77)],2.7)|line([(39,50),(37,43),(35,41)],2)
        return m
    raise ValueError(k)

ring=[0,1,2,3,4,5,7,8,9,10,15,16];warn=[11]+list(range(18,32));blues=list(range(33,41));ends=[6,32,41,42]
texts={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}
def symbol(k,style=0):
    if k in texts:
        if style>=3:
            # Evenly spaced condensed numerals with a full character cell for 1.
            text=texts[k];m=X<0;w=14;h=40
            for i,ch in enumerate(text):
                m|=numeral(ch,50+(i-1)*20-w/2,50-h/2,w,h,.125,0)
            return m
        return numbers(texts[k],bold=(.115,.145,.098)[style],style=style)
    if k in [9,10]:
        if k==9:return car(33,50,30,19)|car(67,47,30,19)
        return polygon([(18,32),(45,32),(45,61),(18,61)])|line([(20,60),(20,65)],3)|line([(42,60),(42,65)],3)|car(66,55,29,19)
    if k==16:return truck(50,50,59,30)[:,::-1]
    if k==15:return X<0
    if k in warn:
        if k==30 and style:
            m=X<0
            for deg in range(30,390,60):m|=line(rotatepoints([(50,61),(50,40)],deg,50,61),3.8)|line(rotatepoints([(42,45),(50,52),(58,45)],deg,50,61),3.3)
            return m
        if k==23 and style==2:
            def tilt(pts):return rotatepoints(pts,19,50,52)
            m=polygon(tilt([(33,53),(34,48),(38,47),(40,41),(56,41),(60,47),(65,50),(65,57),(62,58),(61,61),(56,61),(56,58),(40,58),(40,61),(35,61)]))
            for a in [-12,12]:m|=spline([(50+a,59),(65+a,64),(64+a,69),(54+a,70),(43+a,71),(40+a,75),(48+a,80)],2.8)
            return m
        if k==23 and style:
            m=polygon([(36,47),(43,42),(54,46),(61,54),(63,60),(60,64),(39,57),(34,53)])|oval(39,56,3,3)|oval(58,62,3,3)
            for a in [-11,11]:m|=spline([(48+a,60),(60+a,68),(51+a,69),(47+a,70),(42+a,72),(40+a,76),(47+a,78)],2.4)
            return m
        if k==27 and style:return oval(52,42,3,4)|line([(51,49),(46,59)],5.2)|line([(46,58),(42,70),(36,78)],3)|line([(46,59),(52,67),(59,77)],3.6)|line([(51,48),(56,55),(61,57)],2.8)|line([(50,49),(43,51),(40,58)],2.8)
        if k==31 and style:
            return polygon([(34,47),(38,46),(42,50),(45,55),(55,59),(60,64),(58,69),(53,69),(47,64),(42,63),(38,59)])|line([(37,49),(35,43),(32,41)],2.5)|line([(37,51),(30,52)],3)|line([(43,59),(34,64),(29,63)],3.1)|line([(44,61),(37,68),(31,69)],2.6)|line([(57,65),(65,72),(68,79)],3.1)|line([(54,67),(60,75),(65,77)],2.8)
        if k==29 and style:
            return arc(34,70,10,10,0,360,3)|arc(65,70,10,10,0,360,3)|line([(34,70),(44,54),(51,70),(34,70),(58,55),(65,70),(57,47),(53,47)],2.8)|line([(40,53),(48,53)],3)
        return warning(k)
    if k==40:
        # Rotationally repeated curved arrows; construction phase varies.
        ang=np.radians([-20,5,30][style]);u=(X-50)/50;v=(Y-50)/50
        return sample(arrow(k),u*np.cos(ang)-v*np.sin(ang),u*np.sin(ang)+v*np.cos(ang))
    if k in blues:return arrow(k)
    raise ValueError(k)

def stopword():
    mask=X<0
    for i,ch in enumerate('STOP'):
        x=14+i*19
        def pts(a):return [(x+15*u,50+v*33) for u,v in a]
        if ch=='S':mask|=spline(pts([(.96,-.34),(.7,-.63),(0,-.52),(.02,-.25),(.02,-.02),(.37,-.04),(.68,.06),(1.4,.25),(.76,.68),(0,.39)]),3.8)
        if ch=='T':mask|=line(pts([(0,-.48),(1,-.48)]),4)|line(pts([(.5,-.48),(.5,.48)]),4)
        if ch=='O':mask|=arc(x+7.5,50,7,16,0,360,4)
        if ch=='P':mask|=line(pts([(.05,.48),(.05,-.48)]),4)|spline(pts([(.05,-.48),(1.3,-.62),(1.3,.02),(.05,-.03)]),4)
    return mask

def ink(k,s=0):
    if k in ends:
        a=((X+Y-100)/1.414); stripe=(abs(a)<8)*(np.where(np.cos(2*np.pi*a/3.2)>-.35,1.,.42))
        if k==32:return stripe
        old=numbers('80') if k==6 else symbol(9 if k==41 else 10)
        return np.maximum(stripe,old*.79)
    if k==14:return stopword()
    if k==17:return (abs(Y-50)<10)&(abs(X-50)<37)
    return symbol(k,s)

families=[ring,warn,blues,ends,[12],[13],[14],[17]]

def level(u,v,g):
    if g in (1,5):
        if g==5:v=-v
        return np.minimum((1+v-2*abs(u))/np.sqrt(5),1-v)
    if g==4:return (1-abs(u)-abs(v))*.707
    if g==6:return np.minimum(1-np.maximum(abs(u),abs(v)),1-.7071*(abs(u)+abs(v)))
    return 1-np.sqrt(u*u+v*v)

def antialias(m):
    for axis in [0,1]:
        a=np.swapaxes(m,0,axis);q=np.pad(a,((2,2),(0,0)),mode='edge');m=np.swapaxes((q[:-4]+4*q[1:-3]+6*q[2:-2]+4*q[3:-1]+q[4:])/16,axis,0)
    return m

def sample(m,xx,yy):
    xx=np.clip((xx+1)*A/2-.5,0,A-1.000001);yy=np.clip((yy+1)*A/2-.5,0,A-1.000001)
    i=xx.astype(int);j=yy.astype(int);a=xx-i;b=yy-j
    return (1-a)*((1-b)*m[j,i]+b*m[j+1,i])+a*((1-b)*m[j,i+1]+b*m[j+1,i+1])

def unit(x):return x/np.maximum(np.sqrt(np.sum(x*x,axis=-1,keepdims=True)),1e-5)
def tent(x):return np.clip(x,0,1)
D={}
# Sample the outer geometry on a regular camera grid. Pre-integrate the view
# parameters so many candidate alignments can be tested with matrix products.
def makepose():
    v=np.arange(1,32,2); gx,gy=np.meshgrid(v-.5,v-.5)
    centers=[12.5,14,15.5,17,18.5]; sizes=[9,10.5,12,13.5,15];poses=[]
    for x in centers:
        for y in centers:
            for h in sizes:
                for aspect in [.86,1,1.16]:
                    for slant in [-.09,0,.09]:poses.append((x,y,h*aspect,h,slant))
    poses=np.array(poses);D['poses']=poses.astype('f');cx,cy,rx,ry,shear=poses.T[:,:,None,None]
    v=(gy-cy)/ry;u=(gx-cx)/rx-shear*v
    for g in range(8):
        thick=[.205,.15,.04,.07,.17,.17,.08,.05][g];d=level(u,v,g)
        # Train-free region pooling: each set of weights sums to one.
        def w(q):
            q=q.reshape(len(poses),-1);return q/np.maximum(q.sum(1,keepdims=True),1.e-5)
        aa=np.sqrt(.5*(1/rx**2+1/ry**2));inside=tent(d/aa+.5);core=tent((d-thick)/aa+.5);outside=tent((d+.22)/aa+.5)-inside
        rim=inside-core;out=w(outside);rim=w(rim);core=w(core)
        if g==4:
            m1=w(tent((d-.13)/aa+.5));m2=w(inside-tent((d-.075)/aa+.5));m3=w(outside)
            D['dia']=np.ascontiguousarray((.38*m1+.62*m2-m3).T.astype('f'))
        if g in (2,6,7):
            # A blue or red face with some white printing.
            body=w(tent((d-.09)/aa+.5));hole=w(tent((d-.19)/aa+.5)*tent((.31-abs(v))/aa+.5))
            if g==7:body=w(tent((d-.09)/aa+.5)*tent((abs(v)-.23)/aa+.5))
            elif g==6:body=w(tent((d-.11)/aa+.5)*tent((abs(v)-.30)/aa+.5))
            fm=[body,body-out,hole-body]
        elif g==3:
            insideface=core.reshape(-1,16,16);stripe=w(tent((.14-abs(u+v)*.707)/aa+.5)*insideface);face=w(insideface*tent((abs(u+v)*.707-.19)/aa+.5)); fm=[face,face-stripe,face-out]
        else:fm=[(rim if g!=4 else core),rim-core,core-out,rim-out]
        for j,m in enumerate(fm):D['p%d%d'%(g,j)]=np.ascontiguousarray(m.T.astype('f'))

# Normalized edge probes for subpixel registration. A pair of probes straddles
# the inner boundary of the red band; the outer pair measures facade-to-scene
# contrast. The normal of a triangle side differs from the radial direction.
def makeprobes():
    for g in (0,1,5):
        if g==0:
            theta=np.arange(56)*2*np.pi/56;normal=np.array([np.cos(theta),np.sin(theta)]).T;pts=normal.copy();thick=.21
        else:
            verts=np.array([[0.,-1],[1,1],[-1,1]])
            pts=[];normal=[];thick=.165
            for i,p in enumerate(verts):
                s=verts[(i+1)%3]-p;n=np.array([s[1],-s[0]])/np.linalg.norm(s)
                for t in np.linspace(.14,.86,17):pts.append(p+t*s);normal.append(n)
            pts=np.array(pts);normal=np.array(normal)
            if g==5:pts[:,1]*=-1;normal[:,1]*=-1
        D['probe'+str(g)]=np.stack([pts+f*normal for f in [-thick-.105,-thick*.48,.085,.19]]).astype('f')

# Canonical grids for detailed matching. Generate full interior stencils, not
# observations. Transform size, center, slant and line weight analytically.
S=26
cent=(np.arange(S)+.5)/S*2-1
U,V=np.meshgrid(cent,cent)

def makedetails():
    for g,ids in enumerate(families):
        if g in (4,5):continue
        d=level(U,V,g)
        margin=[.215,.149,.085,.105,0,0,.11,.1][g]; select=d>margin
        u=U[select];v=V[select]
        D['sel'+str(g)]=select.reshape(-1).nonzero()[0].astype('i');D['w'+str(g)]=tent((d[select]-margin)*13+.30).astype('f')
        ww=D['w'+str(g)]; arr=[]; index=[]; bias=[];zones=[]
        if g==1:
            for cx in [-.27,0,.27]:
                for cy in [-.12,.20,.54]:
                    zone=ww*np.exp(-.5*((u-cx)/.26)**2-.5*((v-cy)/.22)**2);zones.append(zone.astype('f'))
            D['zones']=np.array(zones)
        if g in [1,3]:
            near=(u[:,None]-u[None,:])**2+(v[:,None]-v[None,:])**2
            smooth=np.exp(-near/(2*.13**2))*ww[None,:];smooth/=smooth.sum(1,keepdims=True);D['sm'+str(g)]=smooth.T.astype('f').copy()
            ids0=np.flatnonzero(select);lookup={p:i for i,p in enumerate(ids0)};ed1=[];ed2=[]
            for j,p in enumerate(ids0):
                for q in [p+1,p+S]:
                    if q in lookup:ed1.append(j);ed2.append(lookup[q])
            ed1=np.array(ed1);ed2=np.array(ed2);D['e1'+str(g)]=ed1.astype('i');D['e2'+str(g)]=ed2.astype('i')
        for k in ids:
            if k==15:continue
            for style in (range(4) if k in [7,8] else range(3) if k in texts or k in [23,40] else range(2) if k in [27,29,30,31] else [0]):
                raw=antialias(ink(k,style).astype('f'))
                for sx in ([.89,1.,1.13,1.26] if g==0 else [.87,1.,1.15]):
                    for sy in ([.88,1.,1.14,1.27] if g in (0,1) else [.9,1.,1.11]):
                        for shear in ([-.12,0,.12] if g!=3 else [-.22,0,.22]):
                            for dx,dy in [(i,j) for i in [-.076,0,.076] for j in [-.076,0,.076]]:
                                vy=.2 if g==1 else 0
                                xx=(u-dx-shear*(v-vy))/sx;yy=(v-dy-vy)/sy+vy
                                # Sensor integration followed by optical smoothing.
                                a=np.zeros(len(u));
                                for ax,ay,coef in [(0,0,2),(-.036,-.036,1),(-.036,.036,1),(.036,-.036,1),(.036,.036,1)]:a+=coef*sample(raw,xx+ax,yy+ay)
                                a/=6
                                a-=np.sum(ww*a)/ww.sum(); t=a*ww; norm=np.sqrt(np.sum(a*t));arr.append((t/max(norm,1e-7)).astype('f'));index.append(k)
                                bias.append(.016*(abs(np.log(sx))+abs(np.log(sy)))+.003*(abs(dx)+abs(dy))/.052)
        D['bank'+str(g)]=np.ascontiguousarray(np.array(arr).T);D['ix'+str(g)]=np.array(index,'i');D['bias'+str(g)]=np.array(bias,'f')
        if g in [1,3]:
            pix=np.array(arr)/ww[None,:];sm=pix@D['sm'+str(g)];sm-=np.sum(sm*ww,1,keepdims=True)/sum(ww)
            lo=sm*ww/np.maximum(np.sqrt(np.sum(ww*sm*sm,1,keepdims=True)),1e-6);D['low'+str(g)]=lo.T.copy()
            gx=pix[:,ed1]-pix[:,ed2];ew=np.sqrt(np.minimum(ww[ed1],ww[ed2]));D['edgeweight'+str(g)]=ew
            D['grad'+str(g)]=unit(gx*ew).T.copy()
            if g==1:
                D['wlocal']=np.stack([np.sum(zone*(pix-np.sum(pix*zone,1,keepdims=True)/zone.sum())**2,1) for zone in zones]).astype('f')


# Individual glyphs: first/second position is assessed independently of the
# rest of the inscription, so a blurred shared zero cannot mask a good digit.
def makeocr():
    a=(np.arange(11)+.5)/11*2-1;b=(np.arange(15)+.5)/15*2-1;uu,vv=np.meshgrid(a,b)
    out=[];labels=[];w=np.maximum(.24,1-.45*abs(uu)**4-.42*abs(vv)**4).ravel();D['dw']=w.astype('f')
    for ch in '01235678':
        for bold in [.095,.125,.153]:
            raw=numeral(ch,39,30,22,40,bold,0).astype('f')
            for width in [.79,.94,1.08]:
                for height in [.83,.97,1.11]:
                    for shear in [-.115,0,.115]:
                        for dx in [-.09,0,.09]:
                            for dy in [-.052,.052]:
                                x=(uu-dx-shear*vv)*(.30/width);y=(vv-dy)*(.54/height)
                                m=np.zeros(uu.shape)
                                for ax,ay in [(-.028,-.028),(-.028,.028),(.028,-.028),(.028,.028)]:m+=.25*sample(raw,x+ax,y+ay)
                                m=m.ravel();m-=sum(m*w)/sum(w);out.append(w*m/max(np.sqrt(sum(w*m*m)),1e-8));labels.append(int(ch))
    D['digits']=np.ascontiguousarray(np.array(out,dtype='f').T); D['dlabel']=np.array(labels,'i')

SOL=r'''import numpy as np

families = [[0,1,2,3,4,5,7,8,9,10,15,16],[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],[33,34,35,36,37,38,39,40],[6,32,41,42],[12],[13],[14],[17]]

def contrast(z):
    q=np.percentile(z,[8,92],axis=(1,2)).astype(np.float32)
    return np.clip((z-q[0,:,None,None])/(.027+q[1,:,None,None]-q[0,:,None,None]),-.28,1.35)

def fields(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    log=np.log(x+.019)
    lr,lg,lb=log[:,0],log[:,1],log[:,2]
    # Chromatic opponent channels: soft membership, robust to illumination.
    red=np.clip((lr-.76*lg-.24*lb-.035)*1.72,-.28,1.2)
    blue=np.clip((lb-.58*lr-.42*lg-.095)*1.8,-.3,1.2)
    yellow=np.clip((.5*(lr+lg)-lb-.14-1.4*np.maximum(lr-lg-.25,0))*1.8,-.26,1.2)
    y=contrast(.68*lg+.18*lr+.14*lb)
    return [red,blue,yellow,y],log

def positions(maps):
    a,b,c,y=[z.reshape(len(z),16,2,16,2).mean(axis=(2,4)).reshape(len(z),256) for z in maps]
    poses=CONSTANTS['poses'];cx,cy,rx,ry,s=poses.T
    prior=.013*((cx-15.5)**2+(cy-15.5)**2)/8+.008*((ry-12)/2.5)**2+.012*(abs(rx/ry-1)+abs(s))
    scores=[];align=[]
    for g in range(8):
        def op(m,j):return m@CONSTANTS['p'+str(g)+str(j)]
        if g in (0,1,5):
            col=op(a,0);dc=op(a,1);dy=-op(y,1); edge=op(a,3)
            h=.27*col+.64*dc+.70*dy+.24*edge+.16*op(y,2)
        elif g==4:
            # Yellow diamond with white surround: reverse chromatic contrast.
            col=op(c,0);dc=-op(c,1);silhouette=y@CONSTANTS['dia'];h=.66*col+.62*dc+.22*op(y,1)+.20*op(y,2)+.28*silhouette
        elif g==3:
            h=.76*op(y,1)+.27*op(y,2)-.24*op(a,0)-.16*op(b,0)+.26
        else:
            m=b if g==2 else a
            h=.55*op(m,0)+.52*op(m,1)+(.30 if g==2 else .59)*op(y,2)
        h=h-prior
        best=np.argmax(h,axis=1);align.append(poses[best]);scores.append(h[np.arange(len(a)),best])
    return np.stack(scores,1),np.stack(align,1)

grid=(np.arange(26,dtype=np.float32)+.5)/26*2-1
U,V=np.meshgrid(grid,grid)
U=U.reshape(1,-1);V=V.reshape(1,-1)

def register(maps,pose,g):
    # Local coordinate search for the sign outline, on the individual image.
    probe=CONSTANTS['probe'+str(g)];u=probe[:,:,0].reshape(1,-1);v=probe[:,:,1].reshape(1,-1);n=len(pose)
    a,blue,yellow,y=maps;signal=.53*a-.47*y
    def rate(pp):
        cx,cy,rx,ry,sl,tilt=pp.T[:,:,None];xx=np.clip(cx+rx*(u+sl*v),0,30.999);yy=np.clip(cy+ry*(v+tilt*u),0,30.999)
        ix=xx.astype(np.int32);iy=yy.astype(np.int32); ax=(xx-ix).astype(np.float32);ay=(yy-iy).astype(np.float32);i=np.arange(n)[:,None]
        z=((1-ax)*((1-ay)*signal[i,iy,ix]+ay*signal[i,iy+1,ix])+ax*((1-ay)*signal[i,iy,ix+1]+ay*signal[i,iy+1,ix+1])).reshape(n,4,-1)
        inner=np.clip(z[:,1]-z[:,0],-.25,.7);outer=np.clip(z[:,1]-.5*(z[:,2]+z[:,3]),-.18,.52)
        return (inner+.40*outer).mean(1)-.0018*((cx[:,0]-15.5)**2+(cy[:,0]-15.5)**2)/4-.009*(np.abs(sl[:,0])+np.abs(tilt[:,0]))
    best=np.column_stack((pose,np.zeros(n,np.float32)));q=rate(best)
    for step in [1.0,.55,.28]:
        for d in [0,1,2,3,4,5]:
            base=best.copy()
            for direction in [-1,1]:
                p=base.copy();p[:,d]+=step*direction*(.072 if d>=4 else 1.)
                score=rate(p);take=score>q;q=np.maximum(q,score);best=np.where(take[:,None],p,best)
    return best

def rectify(field,pose):return warp(field,pose,U,V)

def warp(field,pose,u,v):
    cx,cy,rx,ry,sl=pose[:,:5].T[:,:,None]
    tilt=pose[:,5,None] if pose.shape[1]==6 else 0
    xx=np.clip(cx+rx*(u+sl*v),0,30.999);yy=np.clip(cy+ry*(v+tilt*u),0,30.999)
    ix=xx.astype(np.int32);iy=yy.astype(np.int32); a=(xx-ix).astype(np.float32);b=(yy-iy).astype(np.float32)
    n=np.arange(len(cx))[:,None]
    return (1-a)*((1-b)*field[n,iy,ix]+b*field[n,iy+1,ix])+a*((1-b)*field[n,iy,ix+1]+b*field[n,iy+1,ix+1])

def readnumbers(z,pose):
    dx,dy=np.meshgrid((np.arange(11,dtype=np.float32)+.5)*2/11-1,(np.arange(15,dtype=np.float32)+.5)*2/15-1)
    dx=dx.ravel();dy=dy.ravel();n=len(pose);w=CONSTANTS['dw'];bank=CONSTANTS['digits'];labels=CONSTANTS['dlabel']
    # Find the best common size and spacing for each inscription. No part of
    # one photograph is ever compared with another photograph.
    readings=[]
    for count in [2,3]:
        slots=[];targets=[0,2,3,5,6,7,8] if count==2 else [0,1,2]
        for s,tabular in ([(a,False) for a in [.84,1.,1.16,1.30]] if count==2 else [(a,b) for a in [.85,1.,1.16,1.30] for b in [False,True]]):
            centers=[-.262,.262] if count==2 else ([-.40,0.,.40] if tabular else [-.405,-.065,.362])
            widths=[.30,.30] if count==2 else ([.215,.222,.222] if tabular else [.191,.242,.242])
            results=[]
            for cx,width in zip(centers,widths):
                y=warp(z,pose,s*(cx+width*dx),s*.53*dy)
                q=np.percentile(y,[5,87],axis=1).astype(np.float32);y=np.clip(y,q[0,:,None]-.16*(q[1]-q[0])[:,None],q[1,:,None]);y-=np.sum(y*w,1)[:,None]/w.sum()
                y/=-np.maximum(np.sqrt(np.sum(y*y*w,1))[:,None],.002)
                use=targets if len(results)!=count-1 else [0]
                result={}
                for digit in use:
                    start=np.searchsorted(labels,digit);stop=np.searchsorted(labels,digit,side='right');result[digit]=(y@bank[:,start:stop]).max(1)
                results.append(result)
            if count==2: slots.append(np.stack([(.60*results[0][i]+.40*results[1][0]) for i in [2,3,5,6,7,8]],1))
            else:slots.append(np.stack([(.17*results[0][1]+.50*results[1][i]+.33*results[2][0]) for i in [0,2]],1))
        readings.append(np.max(np.stack(slots),axis=0))
    return np.concatenate(readings,axis=1)

def segmented(z,pose):
    h=(np.arange(23,dtype=np.float32)-11)/20;v=(np.arange(23,dtype=np.float32)-11)/19
    xx,yy=np.meshgrid(h*1.21,v);n=len(pose);pic=warp(z,pose,xx.ravel(),yy.ravel()).reshape(n,23,23)
    # Whiten against a local paper level. In particular, a shadow over the
    # entire left half should not create a spurious first character.
    paper=np.maximum(np.percentile(pic,86,axis=1)[:,None,:],np.percentile(pic,65,axis=(1,2))[:,None,None]);val=np.maximum(paper-pic,0)
    strong=np.percentile(val,93,axis=(1,2));ink=np.clip(val/(strong[:,None,None]+.002)-.18,0,1.)
    # Use the ink projections to locate the inscription and the valleys
    # separating the digits. Bounds are soft to accommodate motion blur.
    proj=ink[:,4:-4].sum(1);proj=np.maximum(proj-.14*proj.max(1)[:,None],0);total=proj.sum(1)+1.e-6;cum=np.cumsum(proj,1)/total[:,None]
    left=np.minimum((cum<.019).sum(1),10);right=np.maximum(22-(cum>.981).sum(1),12)
    rows=ink[:,:,3:-3].sum(2);rows=np.maximum(rows-.17*rows.max(1)[:,None],0);cs=np.cumsum(rows,1)/(rows.sum(1)[:,None]+1.e-6);top=np.minimum((cs<.025).sum(1),8);bot=np.maximum(22-(cs>.975).sum(1),14)
    xs=xx[0];ys=yy[:,0];topY=ys[top]-.03;botY=ys[bot]+.03
    dx,dy=np.meshgrid((np.arange(11,dtype=np.float32)+.5)*2/11-1,(np.arange(15,dtype=np.float32)+.5)*2/15-1)
    dw=CONSTANTS['dw'];bank=CONSTANTS['digits'];labels=CONSTANTS['dlabel'];a=np.arange(23)[None,:];span=np.maximum(right-left,5);pos=(a-left[:,None])/span[:,None]
    def cut(p):
        loss=proj/(proj.max(1)[:,None]+1.e-5)+.62*((pos-p)/.20)**2
        return np.argmin(loss,axis=1)
    def evidence(lo,hi,digits):
        xlo=xs[lo];xhi=xs[hi];cen=(xlo+xhi)*.5;rangeX=np.maximum(.065,(xhi-xlo)*.58);cy=(topY+botY)*.5;rangeY=(botY-topY)*.62
        data=warp(z,pose,cen[:,None]+dx.ravel()*rangeX[:,None],cy[:,None]+dy.ravel()*rangeY[:,None]);data-=np.sum(data*dw,1)[:,None]/dw.sum();data/=-np.maximum(.001,np.sqrt(np.sum(data*data*dw,1))[:,None]);out={}
        for k in digits:
            i=np.searchsorted(labels,k);j=np.searchsorted(labels,k,side='right');out[k]=(data@bank[:,i:j]).max(1)
        return out
    # Verify each proposed gap is white along most of the character height.
    def gap(i):return 1-np.clip(proj[np.arange(n),i]/(proj.max(1)+1.e-5),0,1)
    s=cut(.50);lef=evidence(left,s,[2,3,5,6,7,8]);rt=evidence(s,right,[0]);two=np.stack([.58*lef[i]+.42*rt[0]+.095*(gap(s)-.75) for i in [2,3,5,6,7,8]],1)
    s1=cut(.27);s2=cut(.64);one=evidence(left,s1,[1]);mid=evidence(s1,s2,[0,2]);rt=evidence(s2,right,[0]);three=np.stack([.18*one[1]+.48*mid[i]+.34*rt[0]+.095*(.5*(gap(s1)+gap(s2))-.75) for i in [0,2]],1)
    return np.concatenate([two,three],1)

def symbols(g,gray,pose):
    ix=CONSTANTS['sel'+str(g)]; w=CONSTANTS['w'+str(g)]
    z=rectify(gray,pose)[:,ix];q=np.percentile(z,[5,85],axis=1).astype(np.float32)
    spread=q[1]-q[0];z=np.clip(z,q[0,:,None]-.2*spread[:,None],q[1,:,None]+.3*spread[:,None]);avg=np.sum(z*w,1)/w.sum();z-=avg[:,None]
    dev=np.sqrt(np.sum(z*z*w,1)/w.sum())
    if g in (0,1,3):z=-z
    z/=np.maximum(dev[:,None]*np.sqrt(w.sum()),.001)
    t=z@CONSTANTS['bank'+str(g)]
    if g in (1,3):
        sm=z@CONSTANTS['sm'+str(g)];sm-=np.sum(sm*w,1)[:,None]/w.sum()
        sm/=np.maximum(np.sqrt(np.sum(w*sm*sm,1))[:,None],1.e-6)
        grad=(z[:,CONSTANTS['e1'+str(g)]]-z[:,CONSTANTS['e2'+str(g)]])*CONSTANTS['edgeweight'+str(g)]
        grad/=np.maximum(np.sqrt(np.sum(grad*grad,1))[:,None],1.e-6)
        t=.50*t+.28*(sm@CONSTANTS['low'+str(g)])+.22*(grad@CONSTANTS['grad'+str(g)])
    if g==1:
        zones=CONSTANTS['zones'];means=(z@zones.T)/zones.sum(1);energy=np.maximum((z*z)@zones.T-means*means*zones.sum(1),0)
        ref=CONSTANTS['wlocal'];delta=np.abs(np.sqrt(energy+.005)[:,:,None]-np.sqrt(ref+.005)[None,:,:]).sum(1)
        t-=.17*delta
    t-=CONSTANTS['bias'+str(g)]
    labs=CONSTANTS['ix'+str(g)];ss=[]
    for k in families[g]:
        if k==15:
            # A genuinely blank disk has no dark strokes in its central box.
            center=z[:,(np.abs(U[0,ix])<.56)&(np.abs(V[0,ix])<.44)]
            variation=np.percentile(center,95,axis=1)-np.percentile(center,15,axis=1)
            ss.append(.71-2.90*dev-.18*np.clip(variation*np.sqrt(w.sum())-1.,0,3))
        else:
            left=np.searchsorted(labs,k);right=np.searchsorted(labs,k,side='right');ss.append(t[:,left:right].max(1))
    return np.stack(ss,1)

def priorityedge(maps,pose):
    # Reward four supported straight sides, not just one contrast edge across
    # the neighborhood. Ignore the four corners where damage is common.
    cx,cy,rx,ry,sl=pose.T;n=len(cx);t=np.linspace(.17,.83,12,dtype=np.float32);sides=[]
    for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        uu=a*t;vv=b*(1-t);vs=[]
        for field in [maps[3],maps[2]]:
            inside=warp(field,pose,.88*uu,.88*vv);outside=warp(field,pose,1.18*uu,1.18*vv)
            vs.append(np.clip(inside-outside,-.18,.65).mean(1))
        sides.append(.74*vs[0]+.26*vs[1])
    sides=np.sort(np.stack(sides,1),axis=1);return .60*sides[:,1]+.40*sides[:,2]

def chunk(x):
    maps,log=fields(x);p,poses=positions(maps);score=np.zeros((len(x),43),np.float32)
    gray=maps[3];inkgray=contrast(.60*log[:,0]+.34*log[:,1]+.06*log[:,2]);lgreen=log[:,1]-.5*(log[:,0]+log[:,2]);lred=log[:,0]-.6*log[:,1]-.4*log[:,2]
    for g,ids in enumerate(families):
        pose=poses[:,g]
        if g in (0,1,5):pose=register(maps,pose,g)
        if g in (4,5):
            detail=np.full((len(x),1),.63,dtype=np.float32)
            if g==4:detail[:,0]+=.72*priorityedge(maps,pose)
        else:detail=symbols(g,gray,pose)
        if g in (0,1):
            rr=symbols(g,inkgray,pose);detail[:,:8 if g==0 else len(ids)]=(.27*detail+.73*rr)[:,:8 if g==0 else len(ids)]
            if g==0:
                ocr=readnumbers(inkgray,pose);seg=segmented(inkgray,pose);detail[:,:8]=.45*detail[:,:8]+.31*ocr+.24*seg
        if g in (0,1,5):
            # Look inside the sign as well. Red and white should change places
            # at the rim, not in the field enclosed by the rim.
            rz=rectify(maps[0],pose)
            cz=rectify(gray,pose)
            if g==0:dist=1-np.sqrt(U*U+V*V)
            else:
                v=V if g==1 else -V;dist=np.minimum((1+v-2*np.abs(U))*.447,1-v)
            ringmask=(dist[0]>.04)&(dist[0]<(.17 if g==0 else .13));coremask=dist[0]>(.23 if g==0 else .18)
            diff=rz[:,ringmask].mean(1)-rz[:,coremask].mean(1)+cz[:,coremask].mean(1)-cz[:,ringmask].mean(1)
            p[:,g]=.75*p[:,g]+.30*diff
            if g==5:
                inside=cz[:,coremask];texture=np.percentile(inside,88,axis=1)-np.percentile(inside,12,axis=1)
                detail[:,0]=.78-.65*texture
            else:
                def region(z,m):return z[:,m[0]].mean(1)
                if g==0:
                    left=region(rz,(U>-.61)&(U<-.12)&(np.abs(V)<.18));right=region(rz,(U<.61)&(U>.12)&(np.abs(V)<.18))
                    vehicle=np.clip(left-right,-.15,.8)
                    detail[:,8:10]+=.31*vehicle[:,None];detail[:,11]-=.17*vehicle
                else:
                    green=rectify(lgreen,pose);redlog=rectify(lred,pose);mid=(np.abs(U)<.105)&(np.abs(V-.46)<.10);surround=(np.abs(U)>.15)&(np.abs(U)<.31)&(np.abs(V-.46)<.10)
                    signal=region(green,mid)-region(green,surround)
                    rzsignal=region(redlog,(np.abs(U)<.10)&(np.abs(V+.08)<.09))-region(redlog,(np.abs(U)>.14)&(np.abs(U)<.23)&(np.abs(V+.08)<.09))
                    # Require the full red-over-green color sequence; the white
                    # region around a dark symbol can take on a color cast.
                    traffic=np.minimum(signal-.058,rzsignal-.09)
                    detail[:,9]+=.80*np.clip(traffic,-.18,.42)
        score[:,ids]=p[:,g,None]+.86*detail
    return score

def smooth(z):
    a=np.pad(z,((0,0),(0,0),(1,1),(1,1)),mode='edge');m=(4*a[:,:,1:-1,1:-1]+a[:,:,:-2,1:-1]+a[:,:,2:,1:-1]+a[:,:,1:-1,:-2]+a[:,:,1:-1,2:])*.125
    return .20*z+.80*m

def classify(z):
    # Evaluate marginal matches at two optical resolutions. Make the control
    # decision for each photograph separately, not at the batch level.
    main=chunk(z);pair=np.sort(main,axis=1)[:,-2:];ambiguous=pair[:,1]-pair[:,0]<.18
    if np.any(ambiguous):
        ids=np.flatnonzero(ambiguous);aug=np.zeros_like(z);aug[:len(ids)]=smooth(z[ids]);secondary=chunk(aug)
        main[ids]=.64*main[ids]+.36*secondary[:len(ids)]
    return np.argmax(main,axis=1)

def predict(x):
    result=np.empty(len(x),np.int64)
    for start in range(0,len(x),32):
        # Constant workgroup size, including the final partial group. This also
        # keeps the reduction kernels independent of external batch sizes.
        n=min(32,len(x)-start); block=np.zeros((32,3,32,32),np.float32);block[:n]=x[start:start+n]
        result[start:start+n]=classify(block)[:n]
    return result
'''

def main(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    makepose();makeprobes();makedetails();makeocr()
    np.savez(folder/'constants.npz',**D);(folder/'solution.py').write_text(SOL)
if __name__=='__main__':main(sys.argv[1])

