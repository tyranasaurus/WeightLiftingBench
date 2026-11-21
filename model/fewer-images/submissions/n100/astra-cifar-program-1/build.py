import numpy as np
import math, sys
from pathlib import Path
from math import sin, cos, pi, sqrt

# A small orthographic/perspective hybrid renderer. World axes: nose, up, left.
rng=np.random.default_rng(809231)
u=rng.uniform

def pick(seq): return seq[int(rng.integers(len(seq)))]
def prob(p): return u()<p
def rgb(s): return np.array([int(s[j:j+2],16)/255 for j in (0,2,4)])
def paint(s): return rgb(pick(s.split()))*u(.82,1.12)
metal='eeeeec d1d5d6 848b91 313641 181c24 a52721 94332e 234b80 f2f0e7'
fur='48382b 916649 b38c67 bc9e75 ded6c4 332f2a 80776c a07249'
white=rgb('ece7d8'); black=rgb('211e1c'); glass=rgb('576976')

class Scene:
 def __init__(self,yaw,pitch,roll):
  s,c=sin(yaw),cos(yaw); a,b=sin(pitch),cos(pitch)
  m=np.array([[c,0,-s],[s*a,-b,c*a],[s*b,a,c*b]])
  self.m=np.array([[cos(roll),-sin(roll),0],[sin(roll),cos(roll),0],[0,0,1]])@m
  self.d=u(6,14); self.objs=[]; self.light=u(-.55,.55); self.ext=[]
 def project(self,pts):
  p=np.asarray(pts)@self.m.T
  k=self.d/np.maximum(self.d-p[...,2:3],self.d*.52)
  return np.concatenate((p[...,:2]*k,p[...,2:3]),axis=-1)
 def ell(self,p,r,col,t=0):
  p=self.project(p); m=self.m*np.array([1,1,1])
  if t:
   a,b=cos(t),sin(t); m=m@np.array([[a,-b,0],[b,a,0],[0,0,1]])
  m=m*np.maximum(r,.012); m[:2]*=self.d/max(self.d-p[2],self.d*.52); v=m@m.T
  x,y,z=p; w,h=np.sqrt(np.diag(v)[:2]); det=max(1e-12,v[0,0]*v[1,1]-v[0,1]**2); aa=v[1,1]/det; bb=-v[1,0]/det; cc=v[0,0]/det
  zx=v[2,0]*aa+v[2,1]*bb; zy=v[2,0]*bb+v[2,1]*cc; dz=sqrt(max(.000001,v[2,2]-zx*v[0,2]-zy*v[1,2])); bounds=(x-w,y-h,x+w,y+h)
  self.ext.append(bounds); self.objs.append((0,bounds,col,(x,y,z,aa,2*bb,cc,zx,zy,dz,w,h)))
 def line(self,p,q,r,s,col):
  a,b=self.project((p,q)); dx,dy,dz=b-a; r*=self.d/(self.d-a[2]); s*=self.d/(self.d-b[2]); w=max(r,s)
  bounds=(min(a[0],b[0])-w,min(a[1],b[1])-w,max(a[0],b[0])+w,max(a[1],b[1])+w)
  self.ext.append(bounds); self.objs.append((1,bounds,col,(a[0],a[1],a[2],dx,dy,dz,1/max(dx*dx+dy*dy,.000001),r,s-r)))
 def poly(self,pts,col):
  p=self.project(pts); x,y,z=p.T; bounds=(min(x),min(y),max(x),max(y)); self.ext.append(bounds)
  dx,dy,dz=p[1]-p[0]; ux,uy,uz=p[-1]-p[0]; cross=dx*uy-dy*ux
  if abs(cross)<.00001:return
  a,b=(dz*uy-dy*uz)/cross,(dx*uz-dz*ux)/cross
  # Face lighting is in camera coordinates, with a bright upper hemisphere.
  shade=.84+.20*(1+self.light*a+b*.9)/sqrt(1+a*a+b*b)
  self.objs.append((2,bounds,col*shade,(x,y,z[0],a,b, 1 if cross>0 else -1)))
 def box(self,x0,x1,y0,y1,z0,z1,col):
  loft(self,[(x0,y0,y1,z1-z0),(x1,y0,y1,z1-z0)],col,zc=(z1+z0)/2)
 def render(self,bg,loc=None,coverage=1):
  ext=np.asarray(self.ext); a,b=ext[:,:2].min(0); c,d=ext[:,2:].max(0)
  if loc is None: cx,cy=(a+c)/2,(b+d)/2; span=max(c-a,d-b,.1)
  else: pp,span=loc;cx,cy=self.project(pp)[:2]
  sz=48; scale=coverage*sz/span; offx=sz*u(.43,.57)-scale*cx; offy=sz*u(.42,.58)-scale*cy
  xs=(np.arange(sz)+.5-offx)/scale; ys=(np.arange(sz)+.5-offy)/scale
  dep=np.full((sz,sz),-100.); img=bg.copy()
  for typ,bounds,col,param in self.objs:
   a,b,c,d=bounds; i=max(0,int(math.ceil(a*scale+offx-.5))); j=max(0,int(math.ceil(b*scale+offy-.5))); k=min(sz,int(c*scale+offx+.5)); l=min(sz,int(d*scale+offy+.5))
   if k<=i or l<=j:continue
   X=xs[None,i:k]; Y=ys[j:l,None]; dest=dep[j:l,i:k]
   if typ==0:
    x,y,z,aa,bb,cc,zx,zy,dz,w,h=param; xx=X-x; yy=Y-y; v=1-aa*xx**2-bb*xx*yy-cc*yy**2; zz=np.sqrt(np.maximum(0,v)); depth=z+zx*xx+zy*yy+dz*zz; mask=(v>=0)&(depth>=dest); shade=.79+.27*zz+self.light*.15*xx/w-.15*yy/h
   elif typ==1:
    x,y,z,dx,dy,dz,den,r,dr=param; xx=X-x; yy=Y-y; t=np.clip((dx*xx+dy*yy)*den,0,1); rr=r+t*dr; q=np.maximum(0,rr**2-(xx-dx*t)**2-(yy-dy*t)**2); zz=np.sqrt(q); depth=z+dz*t+zz; mask=(q>0)&(depth>=dest); shade=.80+.21*zz/np.maximum(rr,.001)
   else:
    x,y,z,ax,ay,side=param; depth=z+ax*(X-x[0])+ay*(Y-y[0]); mask=depth>=dest-.0001
    for m in range(len(x)):
     mask &= side*((x[m]-x[m-1])*(Y-y[m-1])-(y[m]-y[m-1])*(X-x[m-1]))>=0
    shade=None
   dest[mask]=depth[mask]; img[j:l,i:k][mask]=col if shade is None else shade[mask,None]*col
  return img,dep>-99

def loft(s,sections,col,zc=0):
 # Four-sided cross sections (floor, roof, width) along the long axis.
 last=None
 for x,b,t,w in sections:
  h=w/2; cur=[(x,b,zc-h),(x,b,zc+h),(x,t,zc+h),(x,t,zc-h)]
  if last is None:s.poly(cur,col)
  else:
   for j in range(4):s.poly((last[j-1],last[j],cur[j],cur[j-1]),col)
  last=cur
 s.poly(last,col)

def cabin(s,x0,x1,y0,y1,z0,z1,r0,r1,col,windows):
 # Windshields and individual side windows leave visible posts.
 w=z0; ww=z1
 for side in [-1,1]:
  z=side*w; zz=side*ww
  s.poly([(x0,y0,z),(x1,y0,z),(r1,y1,zz),(r0,y1,zz)],col)
  for a,b in windows:
   def corner(v,h):return ((1-h)*(x0+(x1-x0)*v)+h*(r0+(r1-r0)*v),y0+(y1-y0)*h,side*((1-h)*w+h*ww+.005))
   s.poly([corner(a,.13),corner(b,.13),corner(b,.91),corner(a,.91)],glass*u(.58,1.27))
 s.poly([(r0,y1,-ww),(r1,y1,-ww),(r1,y1,ww),(r0,y1,ww)],col)
 for x,xr,di in [(x0,r0,-1),(x1,r1,1)]:
  s.poly([(x,y0,-w),(x,y0,w),(xr,y1,ww),(xr,y1,-ww)],col)
  def end(h,sg):return ((1-h)*x+h*xr+.008*di,(1-h)*y0+h*y1,sg*((1-h)*w+h*ww)*.89)
  s.poly([end(.12,-1),end(.12,1),end(.92,1),end(.92,-1)],glass*u(.72,1.55))

def vehicle(s,truck):
 col=paint(metal+' eab948 b02423 265c46'); tires=paint('17191c 252427 302f30'); chrome=paint('a3a5a4 c1beb6 676969'); wheel=[]
 if not truck:
  w=u(.45,.56); h=u(.43,.55); yt=h+u(.25,.40); rr=u(.175,.227); zb=w-.045
  s.ell((0,.09,0),(1.03,.115,w*1.02),tires*.8)
  loft(s,[(-1.06,.23,h-.065,w*1.77),(-.89,.18,h,2*w),(.75,.17,h,2*w),(1.04,.235,h-.08,w*1.84)],col)
  suv=prob(.34); x0=-.97 if suv else u(-.88,-.57); r0=x0+(u(.06,.17) if suv else u(.2,.36)); x1=u(.44,.68); r1=x1-u(.21,.39)
  cabin(s,x0,x1,h,yt,w*.94,w*.81,r0,r1,col,[(.045,.48),(.53,.96)])
  wheel=[(-.68,rr),(.66,rr)]; fx=1.045; fy=h*.85
  s.box(fx,fx+.018,fy-.065,fy+.025,-w*.48,w*.48,tires); s.box(-1.07,-1.06,.25,.32,-w*.86,w*.86,chrome)
 else:
  w=u(.39,.49); h=u(.49,.62); rr=u(.205,.258); zb=w-.02; fx=1.19; fy=h*.85
  s.box(-1.34,1.13,.20,.36,-w*.85,w*.85,tires); kind=pick(['box','box','cab','tanker','flat','dump','tractor']); c0= .12 if kind!='tractor' else -.42
  s.box(c0,fx,.27,h,-w,w,col); longnose=prob(.43); front=fx-(.46 if longnose else .025); top=h+u(.42,.65); back=c0+.01
  cabin(s,back,front,h,top,w*.99,w*.91,back+.03,front-u(.035,.145),col,[(.09,.91)])
  if kind!='tractor':
   cargo=paint('f7f3e6 dfddd0 a5a7a3 414949 a95735 335076 8c7968') if prob(.76) else col
   yy=top+(u(-.03,.18) if kind in ['box','cab'] else u(-.49,-.10)); ww=w*u(1.,1.13)
   if kind=='tanker':s.ell((-.64,.79,0),(.81,.39,ww),chrome)
   else:
    end=back-.045; s.box(-1.4,end,.40, (.53 if kind=='flat' else yy),-ww,ww,cargo)
    if kind=='dump':
     for x in np.linspace(-1.35,end,7):
      for d in [-1,1]:s.line((x,.43,d*ww),(x+.07,yy,d*ww),.022,.021,cargo*.68)
   wheel=[(-1.01,rr),(-.60,rr),(.85,rr)]
  else:wheel=[(-1.10,rr),(-.63,rr),(.83,rr)]
  s.box(fx,fx+.015,.35,h,-w*.55,w*.55,chrome*.62)
  for f in np.linspace(.36,h,4):s.box(fx+.016,fx+.024,f,f+.012,-w*.53,w*.53,chrome)
 for x,r in wheel:
  for d in [-1,1]:
   s.ell((x,r,zb*d),(r,r,.105),tires); s.ell((x,r,(zb+.092)*d),(r*.55,r*.55,.018),chrome*u(.63,1.07))
 for side in [-1,1]:
  s.ell((fx+.012,fy,side*w*.735),(.028,.047,w*.23),white);s.ell(((-1.065 if not truck else -1.39),fy,side*w*.73),(.021,.052,w*.18),rgb('aa3025'))
 s.box(fx+.012,fx+.039,.225,.287,-w*.95,w*.95,chrome);s.box(fx+.041,fx+.046,.222,.29,-.107,.107,white)

def airplane(s):
 col=paint('f0f0e9 dddedb d6d9db b4bdc1 989f9e 747d82 32383a 414743 edcc55');trim=paint('465167 924239 48769c 535963 bdbebb 525567'); jet=prob(.64);r=u(.096,.16); wing=u(.83,1.35); sweep=u(.26,.72) if jet else u(-.03,.24)
 s.ell((0,0,0),(1.15,r,r),col); s.ell((.69,r*.49,0),(.32,r*.68,r*.83),glass*.93)
 for side in [-1,1]:
  z=side*wing; tip=u(.09,.29); y=u(-.055,.03);yt=y+u(.035,.13)
  s.poly([(.38,y,side*.045),(.32-sweep,yt,z),(.32-sweep-tip,yt,z),(-.30,y,side*.055)],col*u(.77,1.04))
  s.poly([(-.68,.028,0),(-.87,.065,z*.42),(-1.02,.07,z*.41),(-.99,.03,0)],col*.92)
  if jet and prob(.88):s.ell((.085,-.096,side*wing*.44),(.24,.079,.084),col*.78)
 s.poly([(-.57,.05,0),(-.90,.41,0),(-1.055,.43,0),(-1.01,.036,0)],trim)
 if not jet:s.line((1.15,-.33,-.05),(1.15,.33,.05),.026,.018,black)
 else:
  for side in [-1,1]:s.line((-.62,.013,side*r*.85),(.55,.025,side*r*.88),.021,.024,trim)

def ship(s):
 col=paint('edf0ea f3f0e9 323a43 575657 873e34 284757 8c8c82'); deck=paint('e9e7d8 e9ece9 b3aa95 a8987b');w=u(.27,.46);y=u(.22,.45); sail=prob(.20)
 # V-shaped hull with high pointed bow.
 last=None
 for x,b,h,ww in [(-1.16,.03,y,w*.77),(-.84,-.11,y*.92,w),(.39,-.12,y,w),(1.00,.03,y*1.12,w*.55),(1.26,y*.80,y*1.30,.015)]:
  row=[(x,b-.07,0),(x,b,ww*.68),(x,h,ww),(x,h,-ww),(x,b,-ww*.68)]
  if last is None:s.poly(row,col)
  else:
   for j in range(5):s.poly([row[j-1],row[j],last[j],last[j-1]],deck if j==3 else col)
  last=row
 if sail:
  h=u(1.30,2.4); s.line((.02,y,0),(.02,h,0),.018,.009,black)
  s.poly([(-.89,y+.24,0),(-.035,h-.065,0),(-.06,y+.22,.08)],white);s.poly([(.08,h-.15,0),(.94,y+.08,0),(.11,y+.18,-.09)],white*.9)
 else:
  kind=pick(['boat','cabin','liner','cargo','fishing']); upper=paint('f5f4ec e1dfd1 9c9e97 e7ece8'); front=.39 if kind!='cargo' else -.29; rear=-.94 if kind!='boat' else -.54; hh=u(.24,.43)
  cabin(s,rear,front,y*.98,y+hh,w*.80,w*.72,rear+.07,front-.13,upper,[(.05,.33),(.37,.65),(.69,.95)])
  if kind in ['liner','cargo']:
   cabin(s,rear+.11,front-.14,y+hh,y+hh+.20,w*.7,w*.59,rear+.13,front-.16,upper,[(.04,.48),(.54,.95)])
   s.ell((rear+.28,y+hh+.24,0),(.135,.20,.10),paint('a65337 55575a ba9860'))
  if kind in ['boat','fishing','cargo']:s.line((rear+.24,y+.05,0),(rear+.22,y+hh+u(.24,.85),0),.022,.010,black*.85)
  if kind=='cargo':
   for x in [-.11,.22,.54]:s.box(x,x+.27,y,y+u(.13,.25),-w*.77,w*.77,paint('974a3b 6a8078 526370'))
  if kind=='fishing':s.line((-.7,y,0),(-.89,1.22,0),.019,.014,deck)

def bird(s):
 fly=prob(.16); col=paint('736644 665d4a 3d434c 756756 967953 a69b82 595b53 364954 b79045 506d82 8f8d86'); belly=col*u(1.05,1.55) if prob(.66) else paint('dedac2 c6b99d 995a38 d8ba57'); longneck=prob(.14); ang=u(.18,.92); a=cos(ang);b=sin(ang); head=(.52*a,.58*b+(.55 if longneck else .09),0); hx,hy,hz=head;r=u(.21,.33)
 s.ell((-.06,0,0),(.69,.34,.31),belly,ang)
 if longneck:s.line((.30*a,.20*b,0),head,.139,.115,col)
 s.ell(head,(r*1.06,r,r*.94),col);beak=paint('40372d 6b634c bd9a46 544739'); bl=u(.13,.31) if not longneck else .50
 s.line((hx+r*.81,hy-.024,0),(hx+r+bl,hy-.077,0),r*.31,.015,beak)
 for side in [-1,1]:
  z=side
  s.ell((hx+r*.38,hy+r*.24,z*r*.82),(r*.14,r*.15,r*.13),black*.46)
  if fly:
   s.poly([(.28,.02,.03*z),(-.03,.19,1.04*z),(-.72,.28,1.75*z),(-.48,.11,.66*z),(-.49,-.03,.18*z)],col)
  else:
   s.ell((-.13,.028,z*.199),(.60,.215,.16),col*.91,ang*.73)
   s.poly([(-.20,.085,z*.325),(-.96*a,-.83*b,z*.17),(-.53*a,-.47*b-.10,z*.30)],col*.74)
   if prob(.39):s.line((-.10,.14,z*.353),(-.26,-.02,z*.336),.032,.022,belly)
  s.line((-.5*a,-.35*b,z*.08),(-1.03*a,-1.01*b,z*.13),.115,.06,col*.74)
  if not fly:
   foot=-.35-u(.16,.35) if not longneck else -.98
   x=side*.11;s.line((x,-.22,z*.13),(x-.05,foot,z*.14),.028,.018,beak);s.line((x-.10,foot,z*.15),(x+.11,foot-.03,z*.16),.02,.017,beak)
 return head, max(.92,2.8*r)

def frog(s):
 col=paint('51632c 75873c 607035 737549 76714c 7e583a 635b43 9c874e 61a23c 84b73f 424b39 a17f3c'); under=col*u(1.13,1.52); t=u(.02,.28);s.ell((-.14,0,0),(.74,.34,.47),col,t); hx,hy=.45,u(.12,.28);s.ell((hx,hy,0),(.48,.255,.44),under,.06);s.ell((hx+.018,hy+.066,0),(.49,.207,.45),col,.025)
 for d in [-1,1]:
  s.ell((hx+.025,hy+.22,d*.29),(.17,.17,.145),col*1.06);s.ell((hx+.108,hy+.245,d*.38),(.091,.099,.07),paint('948951 9f9361 7e7844'));s.ell((hx+.135,hy+.25,d*.421),(.068,.039,.042),black*.49)
  s.ell((-.58,-.14,.40*d),(.48,.236,.245),col*.87,-.28);s.line((-.79,-.22,d*.46),(-.20,-.33,d*.61),.14,.085,col);s.line((-.22,-.32,d*.6),(-.61,-.35,d*.75),.087,.036,under)
  s.line((.27,.02,d*.29),(.11,-.26,d*.53),.102,.073,col);s.line((.11,-.26,d*.53),(.43,-.31,d*.65),.075,.04,under)
  for j in [-1,0,1]:s.line((.4,-.30,d*.64),(.57+.06*j,-.34,d*(.69+j*.12)),.036,.014,under)
 spots=int(u(4,22)) if prob(.73) else 0
 for j in range(spots):
  theta=u(0,pi*2); v=u(-.87,.84);k=sqrt(1-v*v);r=u(.037,.077);s.ell((-.14+v*.69,.01+.325*k*sin(theta)+v*.12,.452*k*cos(theta)),(r*1.5,r,r),col*u(.48,.82))
 return (.11,.04,0),u(1.35,1.85)

def pet(s,cat):
 # All coordinates are anatomical, not image landmarks. Pinnae, eyes and
 # muzzle are attached to the same head frame in profile and frontal views.
 col=paint(fur+(' c7a375 736e62 b98954 e3dbcb' if cat else ' 242224 dedbd0 9b673e 493b30 ece1cd')); chest=(white*u(.83,1.07) if prob(.36) else col*u(.90,1.18)); earcol=col*u(.60,.98) if prob(.76) else black*1.35
 sitting=prob(.47); h=.79 if sitting else .62; long= u(.58,.77); by=.67 if sitting else .63; slope=.80 if sitting else u(-.03,.17); bx=-.18 if sitting else -.12
 s.ell((bx,by,0),(long,.37,.335),col,slope); hx=.32 if sitting else .53; hy=(1.40 if sitting else 1.02)+u(-.08,.10); hw=u(.32,.40) if cat else u(.29,.39); hr=hw*(1.01 if cat else 1.18)
 s.line((.18,by+.12,0),(hx,hy,0),.30,hw*.84,chest)
 s.ell((hx,hy,0),(hw*.91,hr,hw),col,u(-.15,.12)); nlen=hw*u(.72,.91) if cat else hw*u(.88,1.43); mz=hy-hr*.32
 muzzle=chest*u(.85,1.08) if prob(.72) else col*.65
 eye=paint('a8a272 99a57c b8ad85 bca46e') if cat else paint('574438 342b26 4b3d32');tri=cat or prob(.26); earh=hw*u(.75,1.25);brow=prob(.40)
 for d in [-1,1]:
  z=d*hw
  if tri:
   A=(hx+.01,hy+hr*.50,z*.90); B=(hx+.078,hy+hr*.72,z*.28); C=(hx-.045,hy+hr*.66+earh,z*.86);D=(hx-hw*.45,hy+hr*.46,z*.57)
   s.poly([A,B,C],earcol);s.poly([B,D,C],earcol*.94);s.poly([A,D,C],earcol*.82)
   s.poly([(hx+.042,hy+hr*.66,z*.81),(hx+.071,hy+hr*.77,z*.43),(hx-.003,hy+hr*.64+earh*.74,z*.78)],earcol*.6+rgb('a17e74')*.35)
  else:
   s.ell((hx-hw*.09,hy+hr*.18,z*.91),(hw*.56,earh*.95,hw*.385),earcol,u(-.4,.30));s.ell((hx-hw*.16,hy+hr*.56,z*.77),(hw*.55,hw*.42,hw*.38),earcol)
  ex=hx+hw*.64;ey=hy+hr*.16;ez=z*.65
  if brow:s.ell((ex-.025,ey+hw*.2,ez),(hw*.19,hw*.095,hw*.20),chest*1.09)
  s.ell((ex,ey,ez),(hw*.177,hw*(.167 if cat else .18),hw*.195),black*.6)
  s.ell((ex+hw*.066,ey+.009,ez+z*.055),(hw*.16,hw*.127,hw*.143),eye);s.ell((ex+hw*.166,ey+.009,ez+z*.092),(hw*.075,hw*.113,hw*.062),black*.3)
  s.ell((ex+hw*.208,ey+hw*.073,ez+z*.11),(hw*.036,hw*.041,hw*.04),white)
  s.ell((hx+nlen*.77,mz,d*hw*.21),(nlen*.48,hr*(.255 if cat else .36),hw*.36),muzzle,-.08)
  stride=u(-.15,.15);fy=.052; xx=.27 if sitting else .43; leg=col if prob(.7) else chest
  s.line((xx,by+.03,z*.64),(xx+.03+stride,fy+.1,z*.68),hw*.32,hw*.21,leg);s.ell((xx+.1+stride,fy,z*.68),(hw*.44,hw*.23,hw*.26),leg)
  kx=-.43 if sitting else -.58;s.ell((kx, (.32 if sitting else by-.03),z*.50),(.29,.29,.225),col)
  s.line((kx,.28 if sitting else by-.09,z*.58),(kx-.10,fy+.09,z*.62),hw*.42,hw*.25,col);s.ell((kx,fy,z*.62),(hw*.48,hw*.23,hw*.31),leg)
 nose=paint('322b29 4c3932 985f58') if cat else black*.76
 s.ell((hx+nlen*1.185,mz+hr*.133,0),(hw*.143,hr*(.123 if cat else .156),hw*(.205 if cat else .287)),nose)
 if not cat and prob(.34):s.ell((hx+nlen*.89,mz-hr*.25,0),(hw*.22,hr*.21,hw*.22),rgb('ab706c'))
 if cat and prob(.53):
  for d in [-1,1]:
   for f in [.23,.49,.72]:s.line((hx+hw*.49,hy+hr*f,d*hw*(.87-.25*f)),(hx+hw*.19,hy+hr*(f+.15),d*hw*(.93-.41*f)),hw*.060,hw*.036,col*.49)
   for f in [.24,.49,.74]:s.line((-.64+.91*f,by+.27,d*.20),(-.72+.91*f,by+.02,d*.325),.035,.018,col*.57)
 elif not cat and prob(.33):s.ell((hx+hw*.72,hy+hr*.35,0),(hw*.19,hr*.44,hw*.14),chest)
 tx=bx-long*.77;th=by+(slope*.3);tip=th+u(-.38,.63);s.line((tx,th,0),(tx-.34,th+.03,-.11),.11,.085,col);s.line((tx-.34,th+.03,-.11),(tx-.47,tip,-.18),.087,.037,col)
 return (hx+.02,hy+hr*.02,0),hw*2.6

def hoof(s,horse):
 col=paint('624535 915e3c 66412d 352e27 8d6345 a87c56 c3a079 4c3830 d9d3bd' if horse else 'ab885c 9b7b55 77654e 9a7151 886d52 b6a080'); mane=col*u(.27,.66); legcol=col*u(.7,1.03);hy=u(.92,1.12);bx=-.08;r=.39 if horse else .325;LL=u(.69,.83);s.ell((bx,hy,0),(LL,r,.31 if horse else .265),col);s.ell((-.53,hy,0),(.41,r*.98,.30 if horse else .26),col)
 graze=prob(.20); nx=.45;hx=.99 if graze else .73;heady=hy-(.11 if graze else -.64);t=.91 if not graze else 1.34
 s.line((nx,hy+.09,0),(hx,heady+.05,0),.278 if horse else .196,.163 if horse else .132,col)
 if horse:s.line((nx-.095,hy+r*.67,0),(hx-.12,heady+.1,0),.12,.105,mane)
 face=.355 if horse else .237; wid=.143 if horse else .124
 s.ell((hx+.05,heady+.025,0),(.234,.218,wid*1.3),col,-t);s.ell((hx+.14,heady-.082,0),(face,wid*1.12,wid),col,-t*.83)
 mx=hx+.14+cos(t*.83)*face*.80;my=heady-.082-sin(t*.83)*face*.80;s.ell((mx,my,0),(wid*1.11,wid*.92,wid*.96),col*u(.61,.96),-.61)
 if horse and prob(.40):s.line((hx+.18,heady+.12,0),(mx+.06,my+.09,0),.065,.036,white*.88)
 horns=(not horse) and prob(.43)
 for d in [-1,1]:
  s.ell((hx+.102,heady+.092,d*wid*1.13),(.055,.045,.026),black*.61);s.ell((mx+.067,my+.026,d*wid*.76),(.033,.027,.016),mane*.62)
  p=(hx-.06,heady+.164,d*.095);q=(hx-.125,heady+.36,d*(.17 if horse else .34));s.line(p,q,.081,.018,col);s.line((p[0]+.045,p[1]+.033,p[2]),(q[0]+.011,q[1]-.03,q[2]),.04,.013,mane*1.22)
  if horns:
   pp=(hx-.028,heady+.205,d*.072)
   for j in range(3):
    qq=(pp[0]-u(.06,.16),pp[1]+u(.105,.185),pp[2]+d*.076);s.line(pp,qq,.036-j*.005,.028-j*.006,mane);s.line(qq,(qq[0]+.069,qq[1]+.139,qq[2]+d*.016),.025,.009,mane);pp=qq
  for hind in [False,True]:
   x= -.63 if hind else .38;z=d*(.175 if horse else .15);dy=u(-.12,.12); xk=x+(.11 if hind else -.03)+dy; xf=x+u(-.29,.26);heel=xf+(-.09 if hind else .02); yk=hy*.46;top=.126 if horse else .081; foot=.050 if horse else .032
   s.line((x,hy,z),(xk,yk,z*1.08),top*1.2,top*.56,legcol);s.line((xk,yk,z*1.08),(heel,.17,z*1.16),top*.57,foot,legcol);s.line((heel,.17,z*1.16),(xf,.054,z*1.2),foot,foot*.9,legcol);s.ell((xf+.018,.041,z*1.2),(foot*1.52,foot*1.15,foot*1.05),mane)
 if horse:s.line((-.79,hy+.18,0),(-.94,hy-.22,0),.107,.128,mane);s.line((-.94,hy-.22,0),(-.983,hy-.65,.066),.13,.054,mane)
 else:s.ell((-.82,hy+.02,0),(.117,.16,.08),col*1.30,-.47)
 return (hx+.09,heady+.022,0),(.995 if horse else .85)

sky='85afd3 779cb8 a7c0d6 b5c4cf cccdc9 799fc4 d3d0ca 6188ae'
land='747345 6f794d 8a8c5e a3996a 526343 627043 777553 8c815c 73694c 515a40'
wall='c6c4b9 a6a195 e7e4dc 6e675f 938b83 b6aa99 a78c76 7b8381 494644'
sea='6b8991 456775 718a8b 6c899a 536d80 829d9e 919a99 a1afb7 698781'
road='918d89 787572 95928b afa99c 656563 b6b2a8 b5a995 827e77'

def texture(m,n):
 a=rng.normal(0,1,(m+1,n+1)); X=np.linspace(0,n-.001,48);Y=np.linspace(0,m-.001,48);i=X.astype(int);j=Y.astype(int);f=X-i;g=Y-j;v=a[:,i]*(1-f)+a[:,i+1]*f
 return v[j]*(1-g[:,None])+v[j+1]*g[:,None]

def background(c):
 # Smooth illumination and independent random environmental patches. These are
 # scene components, not a collection of backgrounds measured on photographs.
 def field(p,amount,stretch=1):
  v=paint(p);t=texture(3,4);q=texture(12, max(2,round(12/stretch)));hi=texture(27,max(3,round(24/stretch)))
  return v+amount*(.60*t+.34*q+.20*hi)[...,None]+.042*texture(3,3)[...,None]*rng.uniform(-1,1,(1,1,3))
 horizon=u(.26,.69); tx=.032; flat=prob(.085)
 if c==0:
  top=field(sky,u(.027,.105),2);bottom=field(pick([land,road]),.13);horizon=u(.49,.85) if prob(.44) else 1.2
 elif c==8:
  top=field(sky,.082,2);bottom=field(sea,.108,4); horizon=u(.25,.64) if prob(.76) else -.2
 elif c in [1,9]:
  top=field(pick([land,sky,wall,wall]),u(.11,.22));bottom=field(road,.095,1.7);horizon=u(.45,.73)
  if prob(.53):
   height=int(u(5,22));top[:height]=field(sky,.057)[:height]
 elif c in [3,5] and prob(.61 if c==3 else .42):
  top=field(wall,.13);bottom=field(pick([wall,wall,land]),.12);horizon=u(.45,1.)
 elif c==6:
  top=field(pick([land,land,'625c48 605349 443e36 514d38 8c876c 7f7359']),u(.12,.22));bottom=top;horizon=1.2
 else:
  top=field(land,u(.09,.21)); bottom=field(land,u(.075,.17),1.6)
  if prob({2:.24,4:.17,7:.47}.get(c,.20)):top=field(sky,.095);horizon=u(.18,.58)
  elif prob(.42):bottom=top
 if flat:
  top=field(pick([wall,'dad9d4 4b4b46 33352f e4e6e3']),.034); bottom=top
 h=48*(horizon+.028*texture(1,4));mask=np.clip((np.arange(48)[:,None]-h)*.7,0,1)[...,None];return top*(1-mask)+mask*bottom

def makeimg(c):
 bg=background(c);yaw=u(0,pi);pitch=u(.035,.36);roll=rng.normal(0,.048)
 if c==0:
  pitch=u(-.55,.68) if prob(.72) else u(-1.42,1.18);roll=rng.normal(0,.31) if prob(.86) else u(-1.5,1.5)
 if c==8:pitch=u(.03,.53)
 if c in [2,3,4,5,6,7]:pitch=u(-.10,.38)
 if c==6:pitch=u(.15,1.27);roll=rng.normal(0,.21)
 if c in [3,5] and prob(.45):yaw=u(.95,1.65)
 s=Scene(yaw,pitch,roll); close=None
 if c==0:airplane(s)
 elif c in [1,9]:vehicle(s,c==9)
 elif c==8:ship(s)
 elif c==2:close=bird(s)
 elif c in [3,5]:close=pet(s,c==3)
 elif c in [4,7]:close=hoof(s,c==7)
 elif c==6:close=frog(s)
 if not prob({2:.18,3:.49,4:.23,5:.46,6:.15,7:.28}.get(c,0)):close=None
 elif close is not None: close=(close[0],close[1]*u(1.07,1.5))
 coverage=u(*{0:(.58,1.15),1:(.77,1.23),9:(.76,1.24),2:(.57,1.08),3:(.77,1.34),4:(.49,.99),5:(.70,1.23),6:(.68,1.27),7:(.71,1.18),8:(.65,1.14)}[c]);img,mask=s.render(bg,close,coverage)
 if c in (0,2) and prob(.24):img[mask]*=u(.37,.68)
 img += (texture(12,14)*u(.018,.062)+texture(30,31)*u(.018,.062))[...,None]*mask[...,None]
 img=img*(.93+.08*texture(2,3))[...,None]*rng.uniform(.93,1.07,3)+u(-.045,.06)
 if prob(.08):img=.9*img.mean(2,keepdims=True)+.1*img
 # Area resample 48 to 32 at an integer common subdivision, without PIL.
 v=np.clip(img,0,1).repeat(2,0).repeat(2,1).reshape(32,3,32,3,3).mean((1,3)).transpose(2,0,1)
 return np.asarray(v if prob(.5) else v[:,:,::-1],dtype=np.float32)

def bankbuild():
 banks=[]
 for c in range(10):
  local=[]
  for j in range(80):
   images=np.stack([makeimg(c) for k in range(24)])
   vec=descriptor(images)
   # int8 storage, a fixed coefficient representing one quantization unit.
   local.append(np.clip(np.rint(420*vec),-127,127).astype(np.int8))
  banks.append(np.concatenate(local))
 return np.stack(banks)

import numpy as np

def blur(x):
    a=np.pad(x,((0,0),)*(x.ndim-2)+((1,1),(1,1)),mode='edge')
    b=(a[...,1:-1,:]*2+a[...,:-2,:]+a[...,2:,:])*.25
    return (b[...,1:-1]*2+b[...,:-2]+b[...,2:])*.25

def grad(x):
    p=np.pad(x,((0,0),(1,1),(1,1)),mode='edge')
    dx=p[:,1:-1,2:]-p[:,1:-1,:-2]; dy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    return dx,dy

def grid(x,k):
    n=len(x);return x.reshape(n,-1,k,32//k,k,32//k).mean((3,5))

def unit(x,eps=.00001):
    return x/np.sqrt(np.sum(x*x,axis=1,keepdims=True)+eps)

def arr(x):return np.asarray(x,dtype=np.float32)

def descriptor(x):
    n=len(x); r,g,b=x[:,0],x[:,1],x[:,2]; y=(r*2+g*3+b)/6
    smooth=blur(y); broad=smooth
    for j in range(4):broad=blur(broad)
    features=[]
    # Soft oriented energy bins. Remove only the direction-neutral component,
    # keeping negative evidence for incorrectly oriented contours.
    for im,k,gain in [(smooth,8,.56),(broad,4,.52)]:
        dx,dy=grad(im); m=np.sqrt(dx*dx+dy*dy)+.0001; h=[]
        for a in range(6):
            angle=a*np.pi/6
            dot=(dx*np.cos(angle)+dy*np.sin(angle))/m
            h.append(grid(blur(arr(m*np.maximum(0,abs(dot)-.74)*3.8)),k)[:,0])
        h=np.stack(h,axis=1); avg=h.mean(1,keepdims=True)
        h=(h-.65*avg)/np.sqrt(.020+avg)
        h=h-h.mean((1,2,3),keepdims=True)
        features.append(unit(h.reshape(n,-1))*gain)
    p=grid(smooth,8).reshape(n,-1); p=p-p.mean(1,keepdims=True)
    band=grid(smooth-broad,8).reshape(n,-1)
    features.extend((unit(p)*.45,unit(band,.001)*.255))
    warmth=(r-b); green=g-(r*.56+b*.44)
    c=grid(np.stack([warmth,green*1.6],axis=1),4).reshape(n,2,16)
    mean=c.mean(2,keepdims=True);c=c-.55*mean
    features.append(unit(c.reshape(n,-1),.095)*.385)
    return np.ascontiguousarray(np.concatenate(features,1),dtype=np.float32)

solution_source="import numpy as np\n\ndef blur(x):\n    a=np.pad(x,((0,0),)*(x.ndim-2)+((1,1),(1,1)),mode='edge')\n    b=(a[...,1:-1,:]*2+a[...,:-2,:]+a[...,2:,:])*.25\n    return (b[...,1:-1]*2+b[...,:-2]+b[...,2:])*.25\n\ndef grad(x):\n    p=np.pad(x,((0,0),(1,1),(1,1)),mode='edge')\n    dx=p[:,1:-1,2:]-p[:,1:-1,:-2]; dy=p[:,2:,1:-1]-p[:,:-2,1:-1]\n    return dx,dy\n\ndef grid(x,k):\n    n=len(x);return x.reshape(n,-1,k,32//k,k,32//k).mean((3,5))\n\ndef unit(x,eps=.00001):\n    return x/np.sqrt(np.sum(x*x,axis=1,keepdims=True)+eps)\n\ndef arr(x):return np.asarray(x,dtype=np.float32)\n\ndef descriptor(x):\n    n=len(x); r,g,b=x[:,0],x[:,1],x[:,2]; y=(r*2+g*3+b)/6\n    smooth=blur(y); broad=smooth\n    for j in range(4):broad=blur(broad)\n    features=[]\n    # Soft oriented energy bins. Remove only the direction-neutral component,\n    # keeping negative evidence for incorrectly oriented contours.\n    for im,k,gain in [(smooth,8,.56),(broad,4,.52)]:\n        dx,dy=grad(im); m=np.sqrt(dx*dx+dy*dy)+.0001; h=[]\n        for a in range(6):\n            angle=a*np.pi/6\n            dot=(dx*np.cos(angle)+dy*np.sin(angle))/m\n            h.append(grid(blur(arr(m*np.maximum(0,abs(dot)-.74)*3.8)),k)[:,0])\n        h=np.stack(h,axis=1); avg=h.mean(1,keepdims=True)\n        h=(h-.65*avg)/np.sqrt(.020+avg)\n        h=h-h.mean((1,2,3),keepdims=True)\n        features.append(unit(h.reshape(n,-1))*gain)\n    p=grid(smooth,8).reshape(n,-1); p=p-p.mean(1,keepdims=True)\n    band=grid(smooth-broad,8).reshape(n,-1)\n    features.extend((unit(p)*.45,unit(band,.001)*.255))\n    warmth=(r-b); green=g-(r*.56+b*.44)\n    c=grid(np.stack([warmth,green*1.6],axis=1),4).reshape(n,2,16)\n    mean=c.mean(2,keepdims=True);c=c-.55*mean\n    features.append(unit(c.reshape(n,-1),.095)*.385)\n    return np.ascontiguousarray(np.concatenate(features,1),dtype=np.float32)\n\ndef sat(x):return np.clip(x,0,1)\n\ndef ramp(x,mid,span):return sat(.5+(x-mid)/span)\n\ndef spatial(a,w):return np.sum(a*w,axis=(-2,-1))/np.sum(w)\n\ndef recognize(x):\n    r,g,b=x[:,0],x[:,1],x[:,2]; n=len(x); y=(r*2+g*3+b)/6\n    Y,X=np.mgrid[:32,:32]; X=(X+.5-16)/16; Y=(Y+.5-16)/16\n    mid=np.exp(-1.9*X**2-2.7*Y**2); surround=1-mid; lower=sat((Y-.22)*3);upper=sat((-Y-.14)*3); sides=sat((abs(X)-.45)*2.8); corners=surround**2\n    q=blur(y);dx,dy=grad(q); mag=np.sqrt(dx*dx+dy*dy); avg=lambda a: a.mean(axis=(-2,-1)); cen=lambda a: spatial(a,mid); outer=lambda a: spatial(a,surround)\n    rg=(r-g)/(.20+y); gb=(g-b)/(.20+y); rb=rg+gb; chrom=(np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b))/(.2+y)\n    plant=ramp(gb,.065,.17)*ramp(-rg,-.034,.15); lush=ramp(gb,.115,.17)*ramp(-rg,.022,.10); blue=ramp(-gb,.055,.145)*ramp(-rb,.049,.15); aqua=ramp(-rb,.045,.153)*ramp(-gb,-.018,.11); ochre=ramp(rg,.088,.15)*ramp(rb,.18,.21); fgreen=ramp(-rg,.069,.17)*ramp(gb,.12,.20)\n    skycolor=sat(blue+.46*ramp(y,.74,.33)*ramp(-chrom,-.14,.17)); sky=spatial(skycolor,upper); sky*=1-.46*ramp(spatial(mag,upper),.077,.079)\n    bottomblue=spatial(aqua,lower); bgblue=outer(blue); green=outer(plant); grass=spatial(plant,lower); warm=cen(ochre); leaf=avg(lush); greenc=cen(fgreen);soil=outer(ochre);nc=cen(ramp(chrom,.24,.24)); vividly=cen(ramp(chrom,.55,.31)); extreme=cen(ramp(chrom,.80,.30))\n    coolgray=spatial(ramp(-chrom,-.12,.17)*ramp(-rg,-.026,.12),lower); dark=cen(ramp(-y,-.205,.20)); bright=cen(ramp(y,.75,.24)); bw=ramp(dark,.135,.27)*ramp(bright,.105,.24); darkscene=ramp(-outer(y),-.38,.36)\n    z=avg(mag); smbg=ramp(-spatial(mag,corners),-.069,.077); fine=abs(y-blur(y));busy=outer(fine)/(.012+outer(mag));rough=ramp(outer(fine),.019,.024); centerfine=cen(fine)/(.013+cen(mag)); fuzz=ramp(centerfine,.211,.139); litter=ramp(busy,.215,.17)\n    neutral=outer(ramp(-chrom,-.14,.19)); dull=outer(ramp(-chrom,-.25,.24)); top=spatial(y,upper); bot=spatial(y,lower); ambient=outer(y)\n    # Local orientation agreement followed by longer horizontal/vertical integration.\n    xx=blur(dx*dx); yy=blur(dy*dy); xy=blur(dx*dy); energy=xx+yy+.0008\n    hh=sat((yy-xx)/energy); vv=sat((xx-yy)/energy); iso=np.sqrt((xx-yy)**2+4*xy**2)/energy\n    def extend(m,axis,sh):\n        p=np.pad(m,((0,0),(sh if axis==1 else 0,sh if axis==1 else 0),(sh if axis==2 else 0,sh if axis==2 else 0)),mode='edge')\n        if axis==1: return (p[:,:32]+m+p[:,2*sh:])/3\n        return (p[:,:,:32]+m+p[:,:,2*sh:])/3\n    hor=(extend(dy,2,2)*extend(hh,2,3));vert=(extend(dx,1,2)*extend(vv,1,3));hv=cen(abs(hor))/(.014+cen(mag)); vi=cen(abs(vert))/(.014+cen(mag)); manmade=ramp(hv+vi,.505,.29); horizontal=ramp(hv,.280,.26); boxy=ramp(vi,.202,.20)*ramp(hv,.21,.21); horsky=ramp(spatial(abs(hor),lower)/(.013+spatial(mag,lower)),.36,.30)\n    # What fraction of the background resembles a single smooth field? A ship\n    # usually has different fields above and below the hull, a flying plane not.\n    def regionalchange(w1,w2):\n        v=[abs(spatial(k,w1)-spatial(k,w2)) for k in (r,g,b)]\n        return (v[0]+v[1]+v[2])/3\n    tb=regionalchange(upper,lower); same=ramp(-tb,-.118,.19); two=ramp(tb,.136,.21); band=spatial(mag,mid)/(.023+outer(mag)); isolated=ramp(band,1.02,.99); borderactivity=spatial(mag,lower)/(.019+spatial(mag,upper));layer=ramp(borderactivity,1.02,1.15)\n    watery=ramp(bottomblue,.28,.46); upperblue=ramp(spatial(blue,upper),.26,.47); skyall=ramp(bgblue,.44,.52)*same; pale=ramp(spatial(y,corners),.75,.29)*smbg*dull\n    # Estimate the horizontal disposition of salient contours. This is a\n    # measurement of the current input, never pooled across different photos.\n    ew=mag*(.45+.55*mid); et=avg(ew)+.00001; mx=avg(ew*X)/et; my=avg(ew*Y)/et; sx=avg(ew*X**2)/et-mx*mx; sy=avg(ew*Y**2)/et-my*my; flat=ramp(sx-sy,.042,.13)\n    # Side colour consistency is evaluated at the same height to allow horizons.\n    edgecol=x\n    l=edgecol[:,:,:,:5].mean(3,keepdims=True);rh=edgecol[:,:,:,-5:].mean(3,keepdims=True); dist=np.sqrt(np.minimum(((x-l)**2).mean(1),((x-rh)**2).mean(1)))\n    body=ramp(blur(dist),.112,.15); bcenter=cen(body); bcover=spatial(body,mid*(1-lower));bspread=spatial(body,lower*mid)/(.10+bcenter); narrow=ramp(-bspread,-.58,.45)\n    # Chromatic signals on the likely object rather than in foliage behind it.\n    obj=mid[None,:,:]*(.18+body); objmean=lambda a: avg(obj*a)/avg(obj); coat=objmean(ochre); coatgrass=ramp(coat,.41,.46)*ramp(green,.32,.50); brownleaf=ramp(soil+green,.56,.61); autumn=ramp(soil,.29,.48)\n    shade=cen(y)-outer(y); darkobj=ramp(-shade,.065,.19); brightobj=ramp(shade,.070,.18)\n    # Bilateral facial landmarks: two dark eye sockets above a nose in an oval\n    # region; their contrast is evaluated locally. Several head sizes allowed.\n    facial=np.full(n,-1.,np.float32); feline=facial.copy(); canine=facial.copy()\n    def at(img,xx,yy):\n        return img[:,np.clip(yy,0,31),np.clip(xx,0,31)]\n    for radius in (5,8,11,14):\n        a=np.arange(6,27,2); v=np.arange(5,23,2);cx,cy=np.meshgrid(a,v);cx=cx.ravel();cy=cy.ravel()\n        def samp(img,h,k):return at(img,cx+int(round(h*radius)),cy+int(round(k*radius)))\n        def eyes(side):\n            e=samp(q,side*.42,0); return (.50*samp(q,side*.43,-.29)+.32*samp(q,side*.54,.28)+.18*samp(q,0,-.06)-e)\n        el,er=eyes(1),eyes(-1);eye=np.minimum(el,er)*.65+np.maximum(el,er)*.35-.45*abs(samp(q,.42,0)-samp(q,-.42,0))\n        def nose(v):return .42*(samp(q,-.30,v+.08)+samp(q,.30,v+.08))+.16*samp(q,0,v-.30)-samp(q,0,v)\n        sym=abs(samp(q,.77,.23)-samp(q,-.77,.23))+.6*abs(samp(q,.69,-.34)-samp(q,-.69,-.34));eye-=.13*sym\n        cn=nose(.43);dn=nose(.75);facial=np.maximum(facial,(eye+.26*np.maximum(cn,dn)).max(1));feline=np.maximum(feline,(eye+.45*cn-.15*dn).max(1));canine=np.maximum(canine,(eye+.57*dn-.18*cn).max(1))\n    face=ramp(facial,.10,.16);catface=ramp(feline-canine,0,.092); dogface=1-catface\n    # Repeated round, dark wheel forms over a lighter road. The same row and\n    # similar radii are required at both wheels; isolated dark spots do not count.\n    wheels=np.zeros(n,np.float32)\n    for rr in [2,3,4]:\n        tire=np.zeros_like(q);center=q[:,13:28,3:29];i,j=np.mgrid[13:28,3:29];rad=rr*1.15\n        resp=np.zeros_like(center);hub=np.zeros_like(center)\n        for ang in np.arange(8)*np.pi/4:\n            ox,oy=np.cos(ang),np.sin(ang);h=at(q,j+round(ox*rad*1.36),i+round(oy*rad*1.36));a=at(q,j+round(ox*rr*.67),i+round(oy*rr*.67));resp+=np.clip(h-a,-.10,.38);hub+=a\n        tire[:,13:28,3:29]=resp/8+.13*sat(center-hub/8)\n        for sep in [3*rr,5*rr,7*rr]:\n            if sep>24:continue\n            resp=np.minimum(tire[:,:,sep:],tire[:,:,:-sep]);wheels=np.maximum(wheels,resp.max((1,2)))\n    wheels=ramp(wheels,.095,.145)*ramp(hv+vi,.44,.39)\n    vehicle=(manmade-.5)*1.65+(horizontal-.5)*.70+(wheels-.5)*.78+(flat-.5)*.64-(green-.30)*1.50-(bgblue-.24)*.50\n    outside=ramp(green,.31,.43); forest=ramp(leaf,.22,.42); stubble=autumn*(1-forest); bluebottom=watery*horsky; verdant=ramp(greenc,.19,.31); field=outside*grass;trunk=coatgrass*(1-forest*.5); plain=1-ramp(spatial(mag,corners),.074,.10)\n    sc=np.zeros((n,10),np.float32)\n    sc[:,0]=2.0*(sky-.35)+1.3*(skyall-.26)+.85*(smbg-.5)+.75*(isolated-.45)+.68*(flat-.5)+.42*darkobj+.7*pale-.91*grass-.5*(rough-.45)-.65*bluebottom-.48*two*horizontal\n    sc[:,8]=1.9*(watery-.38)+.98*(sky-.30)+1.12*(horizontal-.46)+.8*(horsky-.49)+.85*(two-.50)+.45*(flat-.5)-1.10*(grass-.25)-.43*(fuzz-.45)+.48*brightobj-.66*same*smbg\n    sc[:,1]=vehicle+.40*(neutral-.55)+.41*(brightobj-.45)+.48*(vividly-.22)+.42*(wheels-.43)-.61*(sky-.33)+.32*(flat-.5)-.48*(boxy-.5)\n    sc[:,9]=vehicle+.94*(boxy-.5)+.55*(vi-.20)*3+.53*(sky-.30)+.73*(top-bot-.08)+.39*(bright-.18)+.32*(coolgray-.37)-.47*(smbg-.5)\n    animal=-.60*(manmade-.5)-.46*(horizontal-.5)-.69*(watery-.27)-.37*(flat-.5)\n    sc[:,2]=animal+.75*(outside-.47)+.58*(forest-.4)+.78*(plain-.49)+.47*(narrow-.48)+.66*(isolated-.48)+.66*upperblue*same+.70*upperblue*grass+.43*(vividly-.24)-.50*(bcover-.51)-.60*(face-.5)\n    sc[:,4]=animal+1.22*(green-.33)+.84*(soil-.25)+1.10*(brownleaf-.58)+.59*(narrow-.5)+.61*(rough-.46)+.62*(trunk-.25)-.85*(sky-.32)-.73*(warm-.37)*grass-.62*(face-.5)-.55*(bw-.3)\n    sc[:,7]=animal+.98*(grass-.31)+.56*(outside-.43)+.64*(warm-.32)+1.07*(trunk-.24)+.54*(sky-.26)+.47*(darkobj-.49)+.40*(bw-.29)+.49*flat*narrow-.61*(litter-.5)-.39*(face-.49)\n    sc[:,6]=animal+1.72*(verdant-.28)+.84*(darkscene-.38)+.65*(litter-.52)+.65*(rough-.47)+.89*(soil-.22)+.62*(greenc-green*.35)+.39*(vividly-.20)-.86*(sky-.28)-.55*(bw-.30)-.60*(narrow-.5)-.46*(face-.49)-.55*(layer-.51)\n    petcue=animal-.83*(sky-.3)-.60*(green-.3)+.62*(face-.47)+.53*(fuzz-.47)+.48*(bcover-.5)\n    sc[:,3]=petcue+.77*(catface-.5)+.66*(ambient-.5)+.63*(dull-.55)+.58*(brightobj-.50)+.62*(narrow-.47)+.48*(centerfine-.22)*4-.52*(vividly-.23)-.40*(wheels-.49)-.42*(darkscene-.4)\n    sc[:,5]=petcue+.82*(dogface-.5)+.54*(bw-.28)+.52*(coat-.31)+.47*(dark-.24)+.46*(bspread-.68)+.55*(grass-.25)+.48*(brightobj-.48)-.60*(litter-.5)\n    return sc\n\n# The comparisons are batched at a fixed width, including for a single image.\n# This also makes the floating point reduction used by BLAS batch-independent.\nshapes=CONSTANTS['shapes'].astype(np.float32).reshape(-1,640).T.copy()/420\n\ndef predict(x):\n    result=np.empty(len(x),dtype=np.int64)\n    for start in range(0,len(x),64):\n        v=x[start:start+64];n=len(v); temp=np.zeros((64,640),np.float32); temp[:n]=descriptor(v)\n        sim=(temp@shapes).reshape(64,10,-1)[:n]\n        best=np.partition(sim,-12,axis=2)[:,:,-12:]; shape=.64*best.mean(2)+.36*best.max(2)\n        result[start:start+n]=(shape*19+recognize(v)).argmax(1)\n    return result\n"

out=Path(sys.argv[1])
out.mkdir(parents=True,exist_ok=True)
(out/'solution.py').write_text(solution_source)
np.savez(out/'constants.npz',shapes=bankbuild())
