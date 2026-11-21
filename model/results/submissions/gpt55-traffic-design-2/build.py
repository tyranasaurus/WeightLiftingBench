import json, sys, os
import numpy as np

H=W=32
M=225
K=43
F=18

RED=np.array([0.90,0.04,0.03],np.float32)
DRED=np.array([0.55,0.02,0.02],np.float32)
BLUE=np.array([0.02,0.18,0.85],np.float32)
WHITE=np.array([0.93,0.92,0.88],np.float32)
BLACK=np.array([0.02,0.02,0.02],np.float32)
YELLOW=np.array([0.96,0.78,0.06],np.float32)
GREY=np.array([0.34,0.34,0.32],np.float32)
DGREY=np.array([0.12,0.12,0.12],np.float32)
GREEN=np.array([0.02,0.50,0.12],np.float32)

yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
X0=xx-15.5; Y0=yy-15.5

def soft(d,a=0.75):
    return np.clip(d/a+0.5,0,1).astype(np.float32)

def coords(dx,dy,ang):
    c=np.cos(ang); s=np.sin(ang)
    x=X0-dx; y=Y0-dy
    return c*x+s*y, -s*x+c*y

def paint(img,m,col,alpha=1.0):
    a=(m*alpha).astype(np.float32)[None,:,:]
    img[:]=img*(1-a)+col[:,None,None]*a

def circle(X,Y,R): return soft(R-np.sqrt(X*X+Y*Y))
def diamond(X,Y,R): return soft(R-(np.abs(X)+np.abs(Y)))

def poly(X,Y,pts):
    p=np.asarray(pts,np.float32); n=len(p)
    area=0.0
    for i in range(n):
        x1,y1=p[i]; x2,y2=p[(i+1)%n]; area+=x1*y2-x2*y1
    sg=1.0 if area>=0 else -1.0
    d=np.full((H,W),1e9,np.float32)
    for i in range(n):
        x1,y1=p[i]; x2,y2=p[(i+1)%n]
        cr=(x2-x1)*(Y-y1)-(y2-y1)*(X-x1)
        di=sg*cr/(np.hypot(x2-x1,y2-y1)+1e-6)
        d=np.minimum(d,di)
    return soft(d)

def seg(X,Y,x1,y1,x2,y2,w):
    vx=x2-x1; vy=y2-y1; den=vx*vx+vy*vy+1e-6
    t=np.clip(((X-x1)*vx+(Y-y1)*vy)/den,0,1)
    px=x1+t*vx; py=y1+t*vy
    return soft(w-np.sqrt((X-px)**2+(Y-py)**2),0.8)

def add_line(img,X,Y,p1,p2,w,col,alpha=1.0): paint(img,seg(X,Y,p1[0],p1[1],p2[0],p2[1],w),col,alpha)

def add_polyline(img,X,Y,pts,w,col,alpha=1.0):
    m=np.zeros((H,W),np.float32)
    for a,b in zip(pts[:-1],pts[1:]): m=np.maximum(m,seg(X,Y,a[0],a[1],b[0],b[1],w))
    paint(img,m,col,alpha)

def add_tri(img,X,Y,cx,cy,ang,ln,col):
    pts=[]
    for a,r in [(0,ln), (2.45,ln*0.7), (-2.45,ln*0.7)]:
        aa=ang+a; pts.append((cx+np.cos(aa)*r,cy+np.sin(aa)*r))
    paint(img,poly(X,Y,pts),col)

def arrow(img,X,Y,x1,y1,x2,y2,w,col,head=3.2):
    add_line(img,X,Y,(x1,y1),(x2,y2),w,col)
    ang=np.arctan2(y2-y1,x2-x1)
    pts=[(x2,y2),(x2-head*np.cos(ang-0.55),y2-head*np.sin(ang-0.55)),(x2-head*np.cos(ang+0.55),y2-head*np.sin(ang+0.55))]
    paint(img,poly(X,Y,pts),col)

def digit(img,X,Y,ch,cx,cy,h,col,wid=1.1):
    # blocky strokes approximating the heavy DIN numerals used on the signs
    w=h*0.31; hh=h*0.50; t=wid
    def pl(pts): add_polyline(img,X,Y,[(cx+x,cy+y) for x,y in pts],t,col)
    def ln(a,b): add_line(img,X,Y,(cx+a[0],cy+a[1]),(cx+b[0],cy+b[1]),t,col)
    if ch=='0':
        pl([(-.72*w,-hh),(.72*w,-hh),(w,-.70*hh),(w,.70*hh),(.72*w,hh),(-.72*w,hh),(-w,.70*hh),(-w,-.70*hh),(-.72*w,-hh)])
    elif ch=='1':
        ln((.25*w,-hh),(.25*w,hh)); ln((-.55*w,-.65*hh),(.25*w,-hh)); ln((-.55*w,hh),(.80*w,hh))
    elif ch=='2':
        pl([(-w,-.75*hh),(-.45*w,-hh),(.65*w,-hh),(w,-.55*hh),(.78*w,-.10*hh),(-.92*w,hh),(w,hh)])
        ln((-.55*w,.12*hh),(.55*w,.12*hh))
    elif ch=='3':
        pl([(-.85*w,-hh),(.75*w,-hh),(w,-.45*hh),(.45*w,0),(w,.48*hh),(.70*w,hh),(-.9*w,hh)]); ln((-.35*w,0),(.55*w,0))
    elif ch=='4':
        ln((.72*w,-hh),(.72*w,hh)); ln((-.90*w,-hh),(-.90*w,0)); ln((-.90*w,0),(.95*w,0))
    elif ch=='5':
        pl([(.95*w,-hh),(-.82*w,-hh),(-w,-.05*hh),(.55*w,-.05*hh),(w,.38*hh),(.65*w,hh),(-.85*w,hh)])
    elif ch=='6':
        pl([(.72*w,-hh),(-.55*w,-hh),(-w,-.25*hh),(-w,.65*hh),(-.55*w,hh),(.55*w,hh),(w,.48*hh),(.55*w,0),(-.82*w,0)])
    elif ch=='7':
        ln((-w,-hh),(w,-hh)); ln((w,-hh),(-.25*w,hh)); ln((-.55*w,-.28*hh),(.70*w,-.28*hh))
    elif ch=='8':
        pl([(-.70*w,-hh),(.70*w,-hh),(w,-.58*hh),(.55*w,0),(w,.58*hh),(.70*w,hh),(-.70*w,hh),(-w,.58*hh),(-.55*w,0),(-w,-.58*hh),(-.70*w,-hh)])
        ln((-.55*w,0),(.55*w,0))
    elif ch=='9':
        pl([(.82*w,0),(-.55*w,0),(-w,-.48*hh),(-.55*w,-hh),(.55*w,-hh),(w,-.48*hh),(w,.65*hh),(.55*w,hh),(-.62*w,hh)])

def word_stop(img,X,Y):
    col=WHITE; y=1.5; h=4.6; w=1.0
    # S
    x=-6.4; add_polyline(img,X,Y,[(x+1.5,y-h/2),(x-1.2,y-h/2),(x-1.5,y),(x+1.2,y),(x+1.5,y+h/2),(x-1.5,y+h/2)],w,col)
    # T
    x=-2.2; add_line(img,X,Y,(x-1.7,y-h/2),(x+1.7,y-h/2),w,col); add_line(img,X,Y,(x,y-h/2),(x,y+h/2),w,col)
    # O
    x=2.2; add_polyline(img,X,Y,[(x-1.4,y-h/2),(x+1.4,y-h/2),(x+1.4,y+h/2),(x-1.4,y+h/2),(x-1.4,y-h/2)],w,col)
    # P
    x=6.2; add_polyline(img,X,Y,[(x-1.3,y+h/2),(x-1.3,y-h/2),(x+1.3,y-h/2),(x+1.3,y),(x-1.3,y)],w,col)

def car(img,X,Y,cx,cy,sc,col):
    m=poly(X,Y,[(cx-3.2*sc,cy+0.8*sc),(cx-2.5*sc,cy-0.7*sc),(cx+1.8*sc,cy-0.7*sc),(cx+3.0*sc,cy+0.8*sc)])
    paint(img,m,col)
    add_line(img,X,Y,(cx-2.6*sc,cy+1.0*sc),(cx+2.6*sc,cy+1.0*sc),0.8*sc,col)
    paint(img,circle(X-cx+1.9*sc,Y-cy-1.3*sc,0.65*sc),BLACK)
    paint(img,circle(X-cx-1.9*sc,Y-cy-1.3*sc,0.65*sc),BLACK)

def truck(img,X,Y,cx,cy,sc,col):
    # lorry pictogram: long rectangular cargo body and smaller cab on the right
    paint(img,poly(X,Y,[(cx-4.8*sc,cy-1.4*sc),(cx+0.8*sc,cy-1.4*sc),(cx+0.8*sc,cy+1.5*sc),(cx-4.8*sc,cy+1.5*sc)]),col)
    paint(img,poly(X,Y,[(cx+0.8*sc,cy-0.7*sc),(cx+3.7*sc,cy-0.7*sc),(cx+4.2*sc,cy+1.5*sc),(cx+0.8*sc,cy+1.5*sc)]),col)
    paint(img,circle(X-cx+2.8*sc,Y-cy-1.9*sc,0.78*sc),BLACK); paint(img,circle(X-cx-3.0*sc,Y-cy-1.9*sc,0.78*sc),BLACK)

def bicycle(img,X,Y,cx,cy,sc,col):
    paint(img,circle(X-cx+2.8*sc,Y-cy-2.2*sc,1.7*sc),col,0.9); paint(img,circle(X-cx-2.8*sc,Y-cy-2.2*sc,1.7*sc),col,0.9)
    paint(img,circle(X-cx+2.8*sc,Y-cy-2.2*sc,1.0*sc),WHITE,1.0); paint(img,circle(X-cx-2.8*sc,Y-cy-2.2*sc,1.0*sc),WHITE,1.0)
    add_polyline(img,X,Y,[(cx-2.8*sc,cy-2.2*sc),(cx,cy-2.2*sc),(cx+1.4*sc,cy-4.3*sc),(cx-0.5*sc,cy-4.3*sc),(cx-2.8*sc,cy-2.2*sc),(cx+2.8*sc,cy-2.2*sc)],0.65*sc,col)
    add_line(img,X,Y,(cx+1.4*sc,cy-4.3*sc),(cx+2.5*sc,cy-5.0*sc),0.55*sc,col)

def person(img,X,Y,cx,cy,sc,col):
    paint(img,circle(X-cx,Y-cy+4.4*sc,1.05*sc),col)
    add_line(img,X,Y,(cx,cy+3.2*sc),(cx-0.5*sc,cy+0.5*sc),0.7*sc,col)
    add_line(img,X,Y,(cx-0.3*sc,cy+2.0*sc),(cx-2.2*sc,cy+0.8*sc),0.55*sc,col)
    add_line(img,X,Y,(cx-0.4*sc,cy+0.8*sc),(cx-2.1*sc,cy-2.2*sc),0.65*sc,col)
    add_line(img,X,Y,(cx-0.3*sc,cy+0.8*sc),(cx+2.0*sc,cy-1.8*sc),0.65*sc,col)

def round_red(img,X,Y,R,lw,fill=WHITE):
    paint(img,circle(X,Y,R),RED); paint(img,circle(X,Y,R-lw),fill)

def blue_disc(img,X,Y,R): paint(img,circle(X,Y,R),BLUE)

def warning_tri(img,X,Y,R,inv=False,inner=WHITE):
    if inv:
        pts=[(-0.90*R,-0.55*R),(0.90*R,-0.55*R),(0,0.85*R)]
        pts2=[(-0.62*R,-0.36*R),(0.62*R,-0.36*R),(0,0.55*R)]
    else:
        pts=[(0,-0.90*R),(-0.92*R,0.72*R),(0.92*R,0.72*R)]
        pts2=[(0,-0.58*R),(-0.61*R,0.48*R),(0.61*R,0.48*R)]
    paint(img,poly(X,Y,pts),RED); paint(img,poly(X,Y,pts2),inner)

def end_circle(img,X,Y,R):
    paint(img,circle(X,Y,R),WHITE); add_line(img,X,Y,(-0.72*R,-0.55*R),(0.55*R,0.72*R),0.75,BLACK)
    for c in [-7,-3,1,5]: add_line(img,X,Y,(-0.75*R,c-5),(c+5,0.75*R),0.7,DGREY)

def draw_stripes(img,X,Y,R):
    for c in [-8,-4,0,4,8]: add_line(img,X,Y,(-0.70*R,c-5),(c+5,0.70*R),0.72,BLACK,0.85)

def bend_symbol(img,X,Y,left=True,double=False):
    if double:
        pts=[(-3.2,5.0),(-1.5,1.5),(1.5,0.0),(3.0,-3.5)]
        add_polyline(img,X,Y,pts,1.15,BLACK); add_tri(img,X,Y,3.0,-3.8,-1.45,2.4,BLACK)
    else:
        s=-1 if left else 1
        pts=[(-s*3.8,5.0),(-s*2.8,1.0),(s*2.6,-1.8),(s*3.2,-5.0)]
        add_polyline(img,X,Y,pts,1.15,BLACK); add_tri(img,X,Y,s*3.2,-5.0,-1.57+0.2*s,2.4,BLACK)

def warn_symbol(img,X,Y,cls):
    if cls==11:
        # major vertical road with a short side road, not a warning triangle in the real set
        add_line(img,X,Y,(0,6.0),(0,-6.2),1.35,BLACK); add_line(img,X,Y,(0,-0.8),(-4.2,-3.9),0.95,BLACK); add_line(img,X,Y,(0,-0.8),(4.2,-3.9),0.95,BLACK)
    elif cls==18:
        add_line(img,X,Y,(0,-4.5),(0,1.8),1.25,BLACK); paint(img,circle(X,Y-5.0,1.1),BLACK)
    elif cls==19: bend_symbol(img,X,Y,True,False)
    elif cls==20: bend_symbol(img,X,Y,False,False)
    elif cls==21: bend_symbol(img,X,Y,True,True)
    elif cls==22:
        add_polyline(img,X,Y,[(-5,2.5),(-2,0),(0,2.5),(3,0),(5.2,2.5)],1.1,BLACK)
    elif cls==23:
        car(img,X,Y,0,-1.8,0.8,BLACK); add_polyline(img,X,Y,[(-5,3.0),(-2,2.0),(1,3.1),(4,2.0)],0.55,BLACK); add_polyline(img,X,Y,[(-4.5,5.0),(-1,4.0),(2.2,5.0)],0.55,BLACK)
    elif cls==24:
        add_line(img,X,Y,(-3.5,5.0),(-3.5,-5.0),1.0,BLACK); add_line(img,X,Y,(3.7,5.0),(1.2,-5.0),1.0,BLACK)
    elif cls==25:
        person(img,X,Y,-2.0,-0.2,0.78,BLACK)
        # mound of earth and a shovel-like diagonal, the most stable features at low resolution
        paint(img,poly(X,Y,[(-0.5,5.2),(2.0,1.8),(5.4,5.2)]),BLACK)
        add_line(img,X,Y,(1.2,-1.2),(5.0,4.2),0.65,BLACK); add_line(img,X,Y,(4.2,3.5),(5.7,5.0),0.95,BLACK)
    elif cls==26:
        paint(img,poly(X,Y,[(-2.2,-5.8),(2.2,-5.8),(2.2,5.6),(-2.2,5.6)]),BLACK)
        paint(img,circle(X,Y+3.5,1.1),RED); paint(img,circle(X,Y,1.1),YELLOW); paint(img,circle(X,Y-3.5,1.1),GREEN)
    elif cls==27:
        person(img,X,Y,0,0.5,1.0,BLACK)
    elif cls==28:
        person(img,X,Y,-1.5,0.4,0.85,BLACK); person(img,X,Y,2.1,0.9,0.65,BLACK)
    elif cls==29:
        bicycle(img,X,Y,0,2.2,0.85,BLACK)
    elif cls==30:
        for a in [0,np.pi/3,-np.pi/3]: add_line(img,X,Y,(-5*np.cos(a),-5*np.sin(a)),(5*np.cos(a),5*np.sin(a)),0.65,BLACK)
    elif cls==31:
        # deer-like silhouette with antlers
        add_line(img,X,Y,(-5,1.5),(2,1.5),1.2,BLACK); add_line(img,X,Y,(2,1.5),(5,-1.5),0.9,BLACK)
        add_line(img,X,Y,(-3,1.6),(-4,4.7),0.7,BLACK); add_line(img,X,Y,(0,1.6),(1,4.6),0.7,BLACK)
        add_line(img,X,Y,(4,-1.3),(5.0,-4.0),0.7,BLACK); add_line(img,X,Y,(4,-1.3),(6.0,-3.0),0.5,BLACK)
        paint(img,circle(X-5.0,Y+1.7,0.8),BLACK)

def render(cls,dx=0,dy=0,sc=1.0,ang=0.0):
    X,Y=coords(dx,dy,ang); R=13.2*sc; lw=3.1*sc
    img=np.zeros((3,H,W),np.float32); img[:]=GREY[:,None,None]
    if cls in [0,1,2,3,4,5,7,8]:
        round_red(img,X,Y,R,lw,WHITE)
        s={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[cls]
        if len(s)==2:
            for i,ch in enumerate(s): digit(img,X,Y,ch,(-3.4+6.8*i)*sc,0.5*sc,9.2*sc,BLACK,1.05*sc)
        else:
            for i,ch in enumerate(s): digit(img,X,Y,ch,(-5.2+5.2*i)*sc,0.8*sc,7.7*sc,BLACK,0.88*sc)
    elif cls==6:
        end_circle(img,X,Y,R);
        for i,ch in enumerate('80'): digit(img,X,Y,ch,(-3.2+6.4*i)*sc,0.7*sc,8.6*sc,DGREY,0.95*sc)
        draw_stripes(img,X,Y,R)
    elif cls==9:
        round_red(img,X,Y,R,lw,WHITE); car(img,X,Y,-3.4*sc,1.8*sc,0.72*sc,DRED); car(img,X,Y,3.6*sc,1.8*sc,0.72*sc,BLACK)
    elif cls==10:
        round_red(img,X,Y,R,lw,WHITE); car(img,X,Y,-4.0*sc,2.2*sc,0.58*sc,DRED); truck(img,X,Y,2.5*sc,1.8*sc,0.62*sc,BLACK)
    elif cls==15:
        round_red(img,X,Y,R,lw,WHITE)
    elif cls==16:
        round_red(img,X,Y,R,lw,WHITE); truck(img,X,Y,0,1.4*sc,0.9*sc,BLACK)
    elif cls==17:
        paint(img,circle(X,Y,R),RED); add_line(img,X,Y,(-7.5*sc,0),(7.5*sc,0),2.2*sc,WHITE)
    elif cls==11:
        warning_tri(img,X,Y,R,False,WHITE); warn_symbol(img,X,Y,11)
    elif cls==12:
        paint(img,diamond(X,Y,R*0.92),WHITE); paint(img,diamond(X,Y,R*0.58),YELLOW)
    elif cls==13:
        warning_tri(img,X,Y,R,True,WHITE)
    elif cls==14:
        # regular octagon, unique red sign
        pts=[]
        for i in range(8):
            a=np.pi/8+i*np.pi/4; pts.append((np.cos(a)*R*0.88,np.sin(a)*R*0.88))
        paint(img,poly(X,Y,pts),RED); word_stop(img,X,Y)
    elif 18<=cls<=31:
        warning_tri(img,X,Y,R,False,WHITE); warn_symbol(img,X,Y,cls)
    elif cls==32:
        end_circle(img,X,Y,R); draw_stripes(img,X,Y,R)
    elif 33<=cls<=40:
        blue_disc(img,X,Y,R)
        if cls==33: arrow(img,X,Y,-5,0,6,0,1.7,WHITE,4.2)
        elif cls==34: arrow(img,X,Y,5,0,-6,0,1.7,WHITE,4.2)
        elif cls==35: arrow(img,X,Y,0,6,0,-7,1.8,WHITE,4.2)
        elif cls==36:
            arrow(img,X,Y,-2,5,-2,-6,1.45,WHITE,3.4); arrow(img,X,Y,-2,0,6,0,1.45,WHITE,3.4)
        elif cls==37:
            arrow(img,X,Y,2,5,2,-6,1.45,WHITE,3.4); arrow(img,X,Y,2,0,-6,0,1.45,WHITE,3.4)
        elif cls==38: arrow(img,X,Y,-5,-5,5,5,1.8,WHITE,4.0)
        elif cls==39: arrow(img,X,Y,5,-5,-5,5,1.8,WHITE,4.0)
        elif cls==40:
            # three rough circular arrows
            add_polyline(img,X,Y,[(-5,0),(-3,-5),(2,-5),(5,-1)],1.25,WHITE); add_tri(img,X,Y,5,-1,0.7,3.1,WHITE)
            add_polyline(img,X,Y,[(4,3),(0,6),(-5,3),(-5,-1)],1.25,WHITE); add_tri(img,X,Y,-5,-1,-2.1,3.1,WHITE)
            add_polyline(img,X,Y,[(-1,-5),(4,-2),(4,4),(0,6)],1.25,WHITE); add_tri(img,X,Y,0,6,2.9,3.1,WHITE)
    elif cls==41:
        end_circle(img,X,Y,R); car(img,X,Y,-3.4*sc,1.8*sc,0.72*sc,DGREY); car(img,X,Y,3.6*sc,1.8*sc,0.72*sc,BLACK); draw_stripes(img,X,Y,R)
    elif cls==42:
        end_circle(img,X,Y,R); car(img,X,Y,-4.0*sc,2.2*sc,0.58*sc,DGREY); truck(img,X,Y,2.5*sc,1.8*sc,0.62*sc,BLACK); draw_stripes(img,X,Y,R)
    return np.clip(img,0,1).astype(np.float32)

def pool2(a):
    c,h,w=a.shape
    return a.reshape(c,16,2,16,2).mean(axis=(2,4)).astype(np.float32)

def norm_img(a):
    mu=a.mean(axis=(1,2),keepdims=True); v=((a-mu)**2).mean(axis=(1,2),keepdims=True)
    return ((a-mu)/np.sqrt(v+1e-5)).astype(np.float32)

def relu(x): return np.maximum(x,0).astype(np.float32)

def conv1(a,W,b):
    return (W[:, :, 0, 0] @ a.reshape(3,-1) + b[:,None]).reshape(W.shape[0],H,32).astype(np.float32)

def center_norm(a,coef=1.0):
    x=a.astype(np.float32).copy()
    x-=x.mean(axis=(1,2),keepdims=True)
    n=np.sqrt((x*x).sum())
    if n<1e-6: return x.reshape(-1)
    return (coef*x/n).reshape(-1)

def focus_mask(cls,X,Y,sc):
    R=13.2*sc; lw=3.1*sc
    if cls in list(range(0,11))+[15,16,17,41,42,6,32]:
        return circle(X,Y,max(3.0,R-lw-0.8))
    if cls==12:
        return diamond(X,Y,R*0.68)
    if cls==13:
        return poly(X,Y,[(-0.65*R,-0.38*R),(0.65*R,-0.38*R),(0,0.58*R)])
    if cls==14:
        return circle(X,Y,R*0.72)
    if 18<=cls<=31 or cls==11:
        return poly(X,Y,[(0,-0.62*R),(-0.65*R,0.52*R),(0.65*R,0.52*R)])
    if 33<=cls<=40:
        return circle(X,Y,R*0.86)
    return np.ones((H,32),np.float32)

def focus_img(proto,mask):
    return proto*mask[None,:,:]+GREY[:,None,None]*(1-mask[None,:,:])

def masked_center_norm(a,mask,coef=1.0):
    x=a.astype(np.float32).copy(); m=mask.astype(np.float32)[None,:,:]
    mu=(x*m).sum(axis=(1,2),keepdims=True)/(m.sum()+1e-6)
    x=(x-mu)*m
    n=np.sqrt((x*x).sum())
    if n<1e-6: return x.reshape(-1)
    return (coef*x/n).reshape(-1)

def make_convs():
    Wr=np.zeros((F,3,1,1),np.float32); br=np.zeros(F,np.float32)
    rows=[([1,-1,0],0),([1,0,-1],0),([-1,0,1],0),([0,-1,1],0),([-1,1,0],0),([0,1,-1],0),
          ([1,1,-1.2],-0.20),([1,1,1],-1.45),([1,1,1],-0.85),([-1,-1,-1],0.90),([-1,-1,-1],1.35),
          ([1,-.45,-.45],-0.03),([-.45,-.45,1],-0.03),([2,-1,-1],0),([-1,-1,2],0),([1,1,-2],0),([1,1,1],0),([-1,2,-1],0)]
    for i,(r,b) in enumerate(rows): Wr[i,:,0,0]=np.array(r,np.float32); br[i]=b
    Wn=np.zeros_like(Wr); bn=np.zeros_like(br)
    rowsn=[([1,-1,0],0),([1,0,-1],0),([-1,0,1],0),([0,-1,1],0),([-1,1,0],0),([0,1,-1],0),
           ([1,0,0],0),([0,1,0],0),([0,0,1],0),([-1,0,0],0),([0,-1,0],0),([0,0,-1],0),
           ([1,1,1],0),([-1,-1,-1],0),([1,1,-2],0),([-1,-1,2],0),([2,-1,-1],0),([-1,2,-1],0)]
    for i,(r,b) in enumerate(rowsn): Wn[i,:,0,0]=np.array(r,np.float32); bn[i]=b
    return Wr,br,Wn,bn

def make_weights():
    Wr,br,Wn,bn=make_convs()
    rows=K*M
    Wraw=np.zeros((rows,3*16*16),np.float32); Wnorm=np.zeros_like(Wraw)
    Wfocus=np.zeros_like(Wraw); Wnfocus=np.zeros_like(Wraw)
    Wfr=np.zeros((rows,F*16*16),np.float32); Wfn=np.zeros_like(Wfr)
    shifts=[(0,0),(-1,0),(1,0),(0,-1),(0,1)]
    scales=[0.88,0.94,1.00,1.07,1.14]
    angles=[0.0,-0.10,0.10]
    braw=np.zeros(rows,np.float32); bnorm=np.zeros(rows,np.float32); bfocus=np.zeros(rows,np.float32); bnfocus=np.zeros(rows,np.float32); bfr=np.zeros(rows,np.float32); bfn=np.zeros(rows,np.float32)
    idx=0
    for cls in range(K):
        for variant in range(3):
            for sc in scales:
                for dx,dy in shifts:
                    for ang in angles:
                        X,Y=coords(dx,dy,ang)
                        if variant==0:
                            proto=render(cls,dx,dy,sc,ang)
                        else:
                            proto=render(cls,dx,dy,sc,ang)
                            XX,YY=coords(dx,dy,ang)
                            if variant==1:
                                proto=np.clip(proto*0.70+0.08,0,1).astype(np.float32)
                            else:
                                shade=np.clip(0.82+0.30*(XX+YY)/32.0,0.60,1.08).astype(np.float32)
                                proto=np.clip(proto*shade[None,:,:]+0.03,0,1).astype(np.float32)
                        r=idx
                        Wraw[r]=center_norm(pool2(proto),0.22)
                        npg=norm_img(proto)
                        Wnorm[r]=center_norm(pool2(npg),0.58)
                        fm=pool2(focus_mask(cls,X,Y,sc)[None,:,:])[0]
                        Wfocus[r]=masked_center_norm(pool2(proto),fm,1.25)
                        Wnfocus[r]=masked_center_norm(pool2(npg),fm,1.15)
                        Wfr[r]=center_norm(pool2(relu(conv1(proto,Wr,br))),0.52)
                        Wfn[r]=center_norm(pool2(relu(conv1(npg,Wn,bn))),0.62)
                        penalty=[0,0.4,0.2,0,0.5,0,0,0,0,1.1,1.1,0.4,2.0,1.1,1.0,-4.0,0.0,1.0,0.2,0,0,0,0,0,0.2,0.4,0.1,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0.2,0.0,0.0][cls]
                        braw[r]=penalty*0.08; bnorm[r]=penalty*0.18; bfocus[r]=penalty*0.25; bnfocus[r]=penalty*0.25; bfr[r]=penalty*0.10; bfn[r]=penalty*0.14
                        idx+=1
    weights={
        'fr.W':Wr,'fr.b':br,'fn.W':Wn,'fn.b':bn,
        'raw_score.W':Wraw,'raw_score.b':braw,
        'norm_score.W':Wnorm,'norm_score.b':bnorm,
        'focus_score.W':Wfocus,'focus_score.b':bfocus,
        'nfocus_score.W':Wnfocus,'nfocus_score.b':bnfocus,
        'feat_score.W':Wfr,'feat_score.b':bfr,
        'nfeat_score.W':Wfn,'nfeat_score.b':bfn}
    return weights

def make_net():
    L=[]
    L.append({'name':'n','op':'norm','inputs':['image']})
    L.append({'name':'pimg','op':'avgpool','inputs':['image'],'size':[2,2],'stride':[2,2]})
    L.append({'name':'pn','op':'avgpool','inputs':['n'],'size':[2,2],'stride':[2,2]})
    L.append({'name':'fr','op':'conv','inputs':['image'],'out':F,'kernel':[1,1],'stride':1,'padding':0})
    L.append({'name':'frr','op':'relu','inputs':['fr']})
    L.append({'name':'pfr','op':'avgpool','inputs':['frr'],'size':[2,2],'stride':[2,2]})
    L.append({'name':'fn','op':'conv','inputs':['n'],'out':F,'kernel':[1,1],'stride':1,'padding':0})
    L.append({'name':'fnr','op':'relu','inputs':['fn']})
    L.append({'name':'pfn','op':'avgpool','inputs':['fnr'],'size':[2,2],'stride':[2,2]})
    L.append({'name':'raw_score','op':'dense','inputs':['pimg'],'out':K*M})
    L.append({'name':'norm_score','op':'dense','inputs':['pn'],'out':K*M})
    L.append({'name':'focus_score','op':'dense','inputs':['pimg'],'out':K*M})
    L.append({'name':'nfocus_score','op':'dense','inputs':['pn'],'out':K*M})
    L.append({'name':'feat_score','op':'dense','inputs':['pfr'],'out':K*M})
    L.append({'name':'nfeat_score','op':'dense','inputs':['pfn'],'out':K*M})
    L.append({'name':'sum_score','op':'add','inputs':['raw_score','norm_score','focus_score','nfocus_score','feat_score','nfeat_score']})
    L.append({'name':'rs','op':'reshape','inputs':['sum_score'],'shape':[K,M,1]})
    L.append({'name':'mx','op':'maxpool','inputs':['rs'],'size':[M,1],'stride':[M,1]})
    L.append({'name':'out','op':'reshape','inputs':['mx'],'shape':[K]})
    return {'layers':L}

def main():
    out=sys.argv[1] if len(sys.argv)>1 else '/app/out'
    os.makedirs(out,exist_ok=True)
    net=make_net(); weights=make_weights()
    with open(os.path.join(out,'network.json'),'w') as f: json.dump(net,f,separators=(',',':'))
    np.savez(os.path.join(out,'weights.npz'),**weights)

if __name__=='__main__': main()
