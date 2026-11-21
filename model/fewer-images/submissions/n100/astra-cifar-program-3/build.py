# Data-free shape and scene grammar. Coordinates are local to the object, not to photographs.
import numpy as np
import sys, os

rng=np.random.RandomState(84621)
def u(a=0.,b=1.): return rng.uniform(a,b)
def coin(p=.5): return rng.rand()<p
def pick(seq): return seq[rng.randint(len(seq))]
def rgb(r,g,b): return np.array([r,g,b],np.float32)
def tint(c,drift=.07):
    return np.clip(c*u(.75,1.24)+rng.normal(0,drift,3),.016,1.0).astype(np.float32)
def mix(a,b,p): return (1-p)*a+p*b

palettes={}
def palette(s):
    if s not in palettes: palettes[s]=[rgb(*bytes.fromhex(k))/255 for k in s.split()]
    return palettes[s]
skies=palette('76a9d3 a0c5e6 5787bc c9d5d9 bcc0c0 c9cbd5 e2e0dc 769abb 98bfde e6e9ea')
waters=palette('405b6c 789cb0 30689a 627d90 58645e 8fa7ae 2b5273 848e93 799eaa a6b8c7 4c656d')
leaves=palette('304021 5e672e 39442a 787444 687746 526937 81924d 85864a 424832 4a5739 334d30 98904e')
fields=palette('62913e 819846 646c38 6c873d a69f5c 90a25b 88995b 928258 a89c74 9b8b61 647543 72714b')
dirt=palette('856e50 72644c 9f8e74 483b31 ab9e85 65523b 5b5141 73684c 907d58 9a8b6c')
rooms=palette('8f725b b5a187 c4b8a4 786e67 4e443d 8b8e86 776052 9f957f c6beb6 b2a18a 63574e b18d78 8c958f 968c8d')
roads=palette('797977 6a6c6b 8a8885 8a8984 66686b 969399 97958c 888580 555355 a6a29a a99f91')
metals=palette('d7d9d7 ecece0 959da4 787d84 33363b 3c4452 4a5877 334d77 8a2826 b02e2a 5e2830 477571 344536 bfa176 586574 eeece8')
furs=palette('6f5943 987651 bda47d ac957d 524b43 7e776f 484342 262627 9e856a ba9973 d0c5ac ded8c8 736153 9b8b7e b88755 c1a17a')
coats=palette('8c5c38 9e643b 684733 4e3527 b6a286 8f7762 372d28 d1c8b8 a47a53 775139')
dark=rgb(.105,.105,.10); black=rgb(.065,.065,.062); white=rgb(.91,.90,.86)
Gx,Gy=np.meshgrid(np.arange(32,dtype=np.float32)+.5,np.arange(32,dtype=np.float32)+.5)
weights={}
def noise(nx,ny):
    key=(nx,ny)
    if key not in weights:
        px=Gx*(nx/32); py=Gy*(ny/32); ix=px.astype(int); iy=py.astype(int)
        fx=px-ix; fy=py-iy
        weights[key]=(ix,iy,fx.astype(np.float32),fy.astype(np.float32))
    ix,iy,fx,fy=weights[key]; z=rng.normal(0,1,(ny+1,nx+1)).astype(np.float32)
    return ((1-fx)*z[iy,ix]+fx*z[iy,ix+1])*(1-fy)+((1-fx)*z[iy+1,ix]+fx*z[iy+1,ix+1])*fy

def environment(k):
    kind=pick((('sky','sky','sky','sky','airport','airport','cloud'),
               ('road','road','road','city','city','lot','studio'),
               ('tree','tree','blur','blur','grass','sky','twigs','waterbird'),
               ('room','room','room','yard','cloth','cloth','floor'),
               ('forest','forest','grass','grass','woods','thicket','thicket'),
               ('room','room','cloth','grass','yard','yard','floor'),
               ('dirt','dirt','pond','pond','forest','leaves','stone'),
               ('grass','grass','grass','paddock','paddock','stable','woods'),
               ('water','water','water','coast','coast','port','opensea'),
               ('road','road','city','city','lot','lot','highway'))[k])
    if coin(.045): kind='studio'
    A=tint(pick(skies),.025); W=tint(pick(waters),.026); F=tint(pick(leaves),.035); L=tint(pick(fields),.035); D=tint(pick(dirt),.035); R=tint(pick(roads),.02); I=tint(pick(rooms),.055)
    broad=noise(3,3); med=noise(7,7); grain=rng.randn(32,32).astype(np.float32); hor=u(8,22)+u(-.08,.08)*(Gx-16)
    def layer(c,n): return c+n[:,:,None]
    def overlay(bg,over,mask): return mix(bg,over,np.clip(mask,0,1)[:,:,None])
    sky=layer(A,u(.005,.06)*broad+u(.01,.11)*(Gy/32-.3)); forest=layer(F,broad*u(.07,.19)+med*u(.035,.14)+noise(18,20)*u(.015,.085))
    grass=layer(L,broad*u(.025,.11)+noise(5,18)*u(.02,.095)+grain*u(.012,.075)); soil=layer(D,broad*u(.06,.17)+med*.09+grain*u(.025,.095))
    if kind in ('sky','cloud'):
        im=sky
        if coin(.5) or kind=='cloud': im=overlay(im,tint(white,.015),np.maximum(0,noise(3,5)+broad*.5-u(0,.8))*u(.2,.8))
    elif kind in ('road','airport','city','lot','highway'):
        h=hor+u(0,7)
        background=forest
        if kind in ('city','lot') or coin(.28):
            blocks=np.clip(rng.normal(.57,.17,(6,10,3)),0,1).astype(np.float32); blocks=blocks.mean(2,keepdims=True)*.72+blocks*.28
            idsx=np.clip(((Gx+u(0,7))/u(3,10)).astype(int),0,9); idsy=np.clip(((Gy+u(0,5))/u(4,10)).astype(int),0,5)
            background=layer(blocks[idsy,idsx],med*.032)
        skyline=h-u(2,18)+noise(7,1)*u(.8,5)
        im=overlay(sky,background,(Gy-skyline)*1.25)
        im=overlay(im,layer(R,broad*.033+grain*u(.005,.038)-(Gy-h)*.002),Gy-h)
        if coin(.27): im=overlay(im,mix(R,white,u(.3,.8)),np.maximum(0,1-np.abs(Gy-u(24,38)-u(-.4,.4)*(Gx-16)))*.85)
    elif kind in ('water','port','coast','opensea','waterbird'):
        if kind=='opensea' or (kind=='waterbird' and coin()): hor=hor-21
        elif kind=='waterbird': hor=hor+6
        sea=layer(W,noise(5,28)*u(.035,.145)+broad*.04+noise(4,15)*.04)
        im=sky
        if kind in ('port','coast','waterbird') or coin(.27):
            im=overlay(im,forest*.8+A*.2,Gy-hor+u(0,9)+noise(5,1)*u(.4,3))
        im=overlay(im,sea,Gy-hor+.4)
    elif kind in ('grass','paddock','woods','yard','stable'):
        h=hor-u(0,11); im=forest
        if kind=='stable': im=layer(I,noise(14,2)*.07+med*.04)
        if coin(.67): im=overlay(sky,im,Gy-h+u(0,11)+noise(7,1)*2)
        im=overlay(im,grass if kind!='stable' else soil,Gy-h-med*1.3)
        if kind=='paddock' or coin(.11):
            fence=tint(pick([white,pick(dirt),dark]),.02)
            rails=np.abs((Gy+u(-.06,.06)*Gx+u(0,6))%u(3.5,7))<u(.5,1.4)
            im=overlay(im,fence,rails*(np.abs(Gy-(h+1))<3.7)*u(.56,.93))
    elif kind in ('tree','blur','twigs','forest','thicket','leaves'):
        im=forest
        if kind in ('blur','tree','thicket'): im=overlay(im,pick([L,A,I,D]),np.clip(.25+noise(3,3)*u(.3,.9),0,1))
        if kind in ('twigs','forest','thicket'):
            for j in range(rng.randint(3,10)):
                slope=u(-1.7,1.7) if kind=='twigs' else u(-.48,.48)
                branch=np.clip(u(.6,2.8)-np.abs(Gx-u(-3,35)-slope*(Gy-16)-.55*broad),0,1)
                im=overlay(im,tint(pick(dirt),.022),branch*u(.38,.9))
        if kind=='leaves': im=overlay(im,grass,np.clip(med*.8+.30,0,1))
        if kind=='forest' and coin(.55): im=overlay(im,soil,(Gy-hor-u(0,10)-med*2)*.45)
    elif kind in ('dirt','stone','pond'):
        im=soil if kind!='stone' else layer(R,med*.11+broad*.13+grain*.063)
        if kind=='pond': im=layer(mix(F,W,u(.15,.83)),broad*.13+noise(6,19)*.07)
        if coin(.68): im=overlay(im,grass,np.clip(noise(5,5)*.7+.05,0,1))
    elif kind=='studio': im=layer(pick([white,white,tint(white),rgb(.48,.48,.48),dark]),broad*u(.003,.018))
    else:
        im=layer(I,broad*u(.045,.16)+med*u(.013,.077))
        if kind in ('room','floor'):
            im=overlay(im,layer(tint(pick(rooms)),broad*.08+grain*.023),Gy-hor-u(-2,5))
            if coin(.63): im=overlay(im,tint(pick(rooms)),(Gx<u(1,14))*(Gy<hor+noise(3,2))*u(.58,.99))
        if kind=='cloth': im=layer(I,broad*.13+med*.05+.06*np.cos(Gy*u(.6,1.4)+Gx*u(-.6,.6)+broad))
    return np.clip(im,.008,.99).astype(np.float32),kind

# Rasterizer over the canonical plane. A little spatially correlated displacement
# softens the outlines of fur/foliage; manufactured shapes have straight edges.
class Canvas:
    def __init__(self,img,cx,cy,sx,sy,angle=0,rough=0):
        self.im=img
        dx=Gx-cx; dy=Gy-cy; ca=np.cos(angle); sa=np.sin(angle)
        self.x=((ca*dx+sa*dy)/sx).astype(np.float32)
        self.y=((ca*dy-sa*dx)/sy).astype(np.float32)
        if rough:
            self.x+=rough*noise(14,14); self.y+=rough*noise(14,14)
        self.feather=1.0/max(1,min(abs(sx),sy)); self.opacity=np.zeros((32,32),np.float32)
    def put(self,c,alpha,sh=0):
        self.im=self.im*(1-alpha[:,:,None])+(c+sh[:,:,None] if isinstance(sh,np.ndarray) else c+sh)*alpha[:,:,None]
        self.opacity=np.maximum(self.opacity,alpha)
    def ellipse(self,c,x,y,rx,ry,tilt=0,shade=.045,alpha=1):
        ca=np.cos(tilt); sa=np.sin(tilt); xx=(self.x-x)*ca+(self.y-y)*sa; yy=(self.y-y)*ca-(self.x-x)*sa
        q=(xx/rx)**2+(yy/ry)**2
        a=np.clip((1-q)*(min(rx,ry)/self.feather*.58)+.48,0,1)*alpha
        sh=shade*np.clip(.55-q-yy/ry*.65-xx/rx*.19,-1,1)
        self.put(c,a,sh)
    def poly(self,c,pts):
        inside=np.zeros((32,32),bool); x=self.x; y=self.y
        for (x1,y1),(x2,y2) in zip(pts,pts[1:]+pts[:1]):
            if abs(y2-y1)>1e-7:
                inside ^= ((y1>y)!=(y2>y)) & (x < x1+(y-y1)*((x2-x1)/(y2-y1)))
        self.put(c,inside)
    def line(self,c,x1,y1,x2,y2,r):
        dx=x2-x1; dy=y2-y1; t=np.clip(((self.x-x1)*dx+(self.y-y1)*dy)/(dx*dx+dy*dy+1e-9),0,1)
        ds=np.sqrt((self.x-x1-t*dx)**2+(self.y-y1-t*dy)**2)
        self.put(c,np.clip((r-ds)/self.feather+.5,0,1))
    def seg(self,c,pts,rads):
        for j in range(len(pts)-1):
            x1,y1=pts[j]; x2,y2=pts[j+1]; dx=x2-x1;dy=y2-y1; norm=(dx*dx+dy*dy)**.5+1e-8; ex=-dy/norm;ey=dx/norm; a=rads[j]; b=rads[j+1]
            self.poly(c,[(x1+ex*a,y1+ey*a),(x2+ex*b,y2+ey*b),(x2-ex*b,y2-ey*b),(x1-ex*a,y1-ey*a)])
            if j: self.ellipse(c,x1,y1,a,a,shade=.015)
    def rect(self,c,x1,y1,x2,y2): self.poly(c,[(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

# Weak-perspective, yawed 3-D vehicle bodies. (x,y,z) is length, height, width.
def vehicle(p,truck):
    a=u(.02,1.52) if coin(.85) else u(0,.34); ca=np.cos(a); sa=np.sin(a); elev=u(.035,.35); front=coin(.82)
    col=tint(pick(metals),.033); shade=col*u(.65,.86); glass=tint(pick(palette('38454e 52606a 656c70 9aabb6 465052 252b30 859497')),.026); rubber=tint(black,.012); chrome=tint(pick([white,rgb(.60,.61,.61),graymetal]),.02)
    def proj(pt):
        x,y,z=pt; return (ca*x+sa*z,-y+(ca*z-sa*x)*elev)
    def face(c,pts): p.poly(c,[proj(pt) for pt in pts])
    def side(c,pts,z): face(c,[(x,y,z) for x,y in pts])
    def end(c,pts,x): face(c,[(x,y,z) for z,y in pts])
    def roof(c,x1,x2,w,h1,h2): face(c,[(x1,h1,-w),(x2,h2,-w),(x2,h2,w),(x1,h1,w)])
    def wheel(x,h,w,r,hub=True):
        xx,yy=proj((x,h,w)); rx=r*(.24+.76*ca); p.ellipse(rubber,xx,yy,rx,r)
        if hub: p.ellipse(chrome*.82,xx,yy,rx*u(.40,.67),r*.56,shade=.045)
    def box(c,x1,x2,w,y1,y2):
        side(c,[(x1,y1),(x2,y1),(x2,y2),(x1,y2)],w)
        end(c*.83,[(-w,y1),(w,y1),(w,y2),(-w,y2)],x1)
        roof(np.minimum(1,c*1.1+.04),x1,x2,w,y2,y2)
    if not truck:
        w=u(.33,.48); roofh=u(.25,.43); sill=-.23; hood=u(-.01,.08); belt=u(.015,.11); rear=u(.83,1.04); nose=u(-1.08,-.88); windshield=u(-.61,-.34); pillar=u(-.38,-.09); back=u(.28,.63); hatch=min(rear,back+u(.10,.36))
        if not front: pillar=-.36; windshield=-.82; back=.21; hatch=.55
        tire=u(.14,.205); axle=-.22
        p.ellipse(dark*.92,0,.30,ca+sa*w,.13+elev*.35,shade=.015,alpha=.64)
        for x in [-.64,.66]: wheel(x,axle,-w,tire,False)
        section=[(nose,sill+.03),(nose-.02,hood-.055),(nose+.12,hood),(windshield,belt),(pillar,roofh),(back,roofh+.01),(hatch,belt+.025),(rear,belt),(rear+.03,sill+.055),(rear-.12,sill-.015)]
        side(col,section,w)
        end(shade,[(-w,sill+.02),(w,sill+.02),(w,hood-.055),(-w,hood-.055)],nose)
        roof(np.minimum(col*1.09+.02,1),nose+.04,windshield,w,hood-.025,belt)
        face(glass,[(windshield,belt,-w),(windshield,belt,w),(pillar,roofh,w*.84),(pillar,roofh,-w*.84)])
        roof(np.minimum(col*1.09+.035,1),pillar,back,w*.84,roofh,roofh+.01)
        side(glass*.82,[(windshield+.065,belt+.018),(pillar+.038,roofh-.047),(back-.015,roofh-.035),(hatch-.052,belt+.045)],w+.002)
        sep=u(.04,.19); side(shade,[(sep-.015,belt),(sep-.018,roofh),(sep+.023,roofh),(sep+.032,belt)],w)
        side(shade,[(nose,sill+.06),(rear,sill+.07),(rear,sill+.01),(nose+.06,sill)],w)
        for z in [-1,1]:
            light=white if front else rgb(.65,.085,.045); end(light,[(w*z*.55,hood-.14),(w*z*.93,hood-.13),(w*z*.94,hood-.055),(w*z*.55,hood-.068)],nose-.002)
        end(dark,[(-w*.53,sill+.094),(w*.53,sill+.094),(w*.50,sill+.15),(-w*.50,sill+.15)],nose-.004)
        end(white*.76,[(-.088,sill+.07),(.088,sill+.07),(.088,sill+.12),(-.088,sill+.12)],nose-.006)
        for x in [-.65,.64]: wheel(x,axle,w+.015,tire)
    else:
        w=u(.22,.35); base=-.30; chassis=-.22; cabh=u(.26,.53); trailerh=cabh+u(-.01,.22); split=u(-.42,-.14); rad=u(.124,.17)
        p.ellipse(dark,0,.37,ca+sa*w,.13+elev*.26,shade=.012,alpha=.60)
        box(dark,-1,1,w*.94,base,chassis+.12)
        freight=tint(pick([white,white,pick(metals),pick(metals),rgb(.55,.56,.55),col]),.035)
        if coin(.80):
            box(freight,split+.04,1.07,w*1.06,chassis+.10,trailerh)
            if coin(.50): side(tint(pick(metals)),[(split+.05,chassis+.2),(1.07,chassis+.2),(1.07,chassis+.27),(split+.05,chassis+.27)],w*1.07)
            if coin(.35):
                for xline in np.arange(split+.12,1,.12): side(freight*.89,[(xline,0),(xline+.025,0),(xline+.025,trailerh),(xline,trailerh)],w*1.06)
        else: box(freight,split+.02,1.06,w,chassis+.08,u(.02,.24))
        nose=-1.06; cabnose=nose+pick([.0,.0,.13,.25]); box(col,cabnose,split,w,chassis+.018,cabh); box(col,nose,cabnose+.02,w*.97,chassis+.025,cabh*.2)
        end(glass,[(-w*.88,cabh*.30),(w*.88,cabh*.30),(w*.83,cabh-.046),(-w*.83,cabh-.046)],cabnose-.001)
        side(glass*.91,[(cabnose+.035,cabh*.33),(split-.05,cabh*.34),(split-.05,cabh-.05),(cabnose+.045,cabh-.05)],w+.002)
        end(chrome*.68,[(-w*.65,chassis+.078),(w*.65,chassis+.078),(w*.60,cabh*.2),(-w*.60,cabh*.2)],nose)
        for z in (-1,1): end(white,[(w*z*.71,-.114),(w*z*.96,-.114),(w*z*.96,-.04),(w*z*.71,-.04)],nose)
        for x in [-.78,split+.23,.77,.77-rad*1.82]: wheel(x,base,w+.027,rad)

graymetal=rgb(.58,.60,.62)

def airplane(p):
    # Wings, empennage and fuselage projected with bank, elevation and azimuth.
    yaw=u(-1.36,1.36); bank=u(-.85,.85); above=u(-.8,.67); ca=np.cos(yaw);sa=np.sin(yaw); cb=np.cos(bank); sb=np.sin(bank)
    color=tint(pick([white,white,graymetal,graymetal,pick(metals)]),.035); accent=tint(pick(metals)); window=rgb(.19,.25,.29)
    def proj(x,h,z): return (ca*x+sa*(z*cb+h*sb),-h*cb+z*sb+above*(z*cb*ca-x*sa))
    def panel(c,pts): p.poly(c,[proj(*pt) for pt in pts])
    length=u(.90,1.1); fat=u(.065,.138); wings=u(.72,1.05); sweep=u(.08,.49)
    for sign in (-1,1):
        panel(color*u(.68,.99),[(-.32,0,0),(-.15+sweep,.025,sign*wings),(.04+sweep,.03,sign*wings),(.30,.0,0)])
        panel(accent if coin(.27) else color*.83,[(.55,0,0),(.81,.045,.42*sign),(.98,.055,.43*sign),(.89,0,0)])
        if coin(.76):
            a=proj(-.24,-fat*.69,sign*wings*.39); p.ellipse(color*.65,*a,.12*(ca+.26),fat*.55,tilt=-above*sa)
    a=proj(-length,0,0); b=proj(.98,.025,0); angle=np.arctan2(b[1]-a[1],b[0]-a[0]); mid=((a[0]+b[0])*.5,(a[1]+b[1])*.5); ln=np.hypot(b[0]-a[0],b[1]-a[1])*.5
    p.ellipse(color,*mid,max(ln,.12),fat,angle,.085); panel(accent,[(.53,.01,0),(.78,.33,0),(.89,.33,0),(.91,0,0)])
    a=proj(-length*.72,fat*.27,.01); p.ellipse(window,*a,.09*ca+.018,fat*.26,angle,shade=.02)
    if coin(.55):
        a=proj(-.54,0,0);b=proj(.56,0,0);p.line(mix(window,color,.32),*a,*b,.012)

def boat(p):
    hull=tint(pick([white,white,graymetal,dark,pick(metals),pick(dirt)]),.028); deck=tint(pick([white,white,graymetal,pick(rooms)]),.03); glass=tint(pick(waters),.02)*.64; mast=tint(pick([dark,graymetal,pick(dirt)]),.02)
    a=u(-1.42,1.42); ca=np.cos(a); sa=np.sin(a); elev=u(.04,.37); w=u(.18,.37)
    def proj(x,h,z): return (x*ca+z*sa,-h+(z*ca-x*sa)*elev)
    def panel(c,pts):p.poly(c,[proj(*pt) for pt in pts])
    def stick(c,v,wid,rad): p.line(c,*proj(*v),*proj(*wid),rad)
    def box(c,x1,x2,w,h1,h2):
        panel(c,[(x1,h1,w),(x2,h1,w),(x2,h2,w),(x1,h2,w)]); ex=x1 if sa>=0 else x2
        panel(c*.86,[(ex,h1,-w),(ex,h1,w),(ex,h2,w),(ex,h2,-w)]); panel(np.minimum(1,c*1.12+.025),[(x1,h2,-w),(x2,h2,-w),(x2,h2,w),(x1,h2,w)])
    top=u(-.06,.06); bot=top-u(.13,.32); mode=pick(['sail','liner','motor','cargo','motor','cargo','tug']); bow=(-1.05,top+.08,0)
    panel(hull*.75,[bow,(-.77,bot,.035),(.86,bot,-w*.70),(.98,top,-w),(-.68,top,-w)])
    panel(hull,[bow,(-.80,bot,.025),(.83,bot,w*.7),(.98,top,w),(-.60,top,w)])
    panel(mix(hull,deck,.5),[bow,(-.62,top,-w),(.98,top,-w),(.98,top,w),(-.62,top,w)])
    if mode=='sail':
        H=u(.68,1.46); x=u(-.16,.16); stick(mast,(x,top,0),(x,H,0),.017)
        if coin(.81):
            panel(deck,[(x-.025,H-.07,0),(x-.69,top+.14,0),(x-.045,top+.17,.05)]); panel(tint(pick([white,white,white,pick(metals)])),[(x+.02,H,0),(x+.54,top+.20,-.04),(x+.02,top+.20,0)])
        box(deck,-.15,.46,.15,top,top+.1)
    else:
        if mode in ('liner','motor'): start=u(-.73,-.36); finish=u(.55,.86); bh=u(.13,.24)
        else: start=u(-.44,.44); finish=start+u(.27,.60);bh=u(.21,.40)
        box(deck,start,finish,w*.79,top,top+bh); box(glass,start+.03,finish-.02,w*.80,top+bh*.48,top+bh*.80)
        box(deck,start+.10,finish-.05,w*.66,top+bh,top+bh+u(.05,.14))
        if mode=='liner' or coin(.40):
            for i in np.arange(start+.09,finish-.07,.13): stick(deck,(i,top+bh*.45,w*.81),(i,top+bh*.81,w*.81),.018)
        if mode in ('cargo','tug','liner'):
            box(tint(pick(metals)),finish-.20,finish-.07,w*.37,top+bh+.065,top+bh+.25); stick(mast,(start+.12,top+bh,0),(start+.13,top+bh+u(.22,.50),0),.012)
            if mode=='cargo' and start>-.08:
                for left in np.arange(-.69,start-.2,.25): box(tint(pick(metals)),left,left+.22,w*.69,top,top+u(.09,.21))
    if coin(.47):stick(mix(hull,white,.54),(-.76,bot,.04),(.80,bot,w*.76),.025)

def ungulate(p,horse,portrait):
    fur=tint(pick(coats if horse else palette('9c7f5b 9f805d 8e795c a99170 7f705d 9c846a b99a72 736654 8b7969')),.025)
    mane=mix(fur,dark,u(.4,.92)) if horse else fur*.79; nose=mix(fur,dark,.64); hoof=mix(fur,dark,.69); belly=mix(fur,white,u(.08,.36)); eye=black*.74
    def ear(x,y,s,t):
        p.ellipse(fur,x,y,.088*s,.18*s,t);p.ellipse(mix(mane,fur,.44),x,y-.017*s,.043*s,.124*s,t)
    def antlers(x,y,s):
        for sign in [-1,1]:
            points=[(x+sign*.09*s,y),(x+sign*.19*s,y-.22*s),(x+sign*.35*s,y-.41*s),(x+sign*.40*s,y-.60*s)]; p.seg(mane,points,[.026*s,.024*s,.018*s,.006*s]);p.line(mane,points[2][0],points[2][1],x+sign*.21*s,y-.58*s,.016*s);p.line(mane,points[1][0],points[1][1],x+sign*.09*s,y-.36*s,.013*s)
    if portrait:
        # Head-on studies include the narrowing muzzle and large ears.
        yaw=u(-.42,.42); p.ellipse(fur*.85,.09,.68,.42,.82,-.18)
        for sign in (-1,1): ear(sign*(.24 if horse else .29),-.63,1.28 if horse else 1.45,sign*(.34 if horse else 1.02))
        if not horse and coin(.48):antlers(0,-.69,1.36)
        p.ellipse(fur,0,-.28,.37,.47);p.seg(fur,[(0,-.29),(yaw*.46,.12),(yaw*.66,.37)],[.33,.25,.19]);p.ellipse(nose,yaw*.65,.37,.225,.17,shade=.07)
        if horse and coin(.48): p.seg(mix(fur,white,.7),[(0,-.49),(yaw*.37,-.03),(yaw*.56,.24)],[.077,.043,.06])
        for s in (-1,1):p.ellipse(eye,s*.28,-.29,.054,.042);p.ellipse(dark,yaw*.65+s*.12,.345,.04,.029)
        if horse:p.seg(mane,[(-.26,-.59),(.03,-.70),(.16,-.45)],[.09,.12,.025])
        return
    squeeze=u(.77,1.05) if coin(.73) else u(.41,.77); graze=coin(.20); gait=u(-.23,.30) if horse else u(-.16,.18); bodylen=u(.55,.73)*squeeze; by=u(-.05,.08); depth=u(.235,.34) if horse else u(.20,.29); hip=bodylen*.75; shoulder=-bodylen*.70
    def leg(x,back,far):
        c=fur*(.69 if far else .95); offset=u(-.16,.17)+(gait if back else -gait)
        yy=by+depth*.26; knee=x+(.115 if back else -.038)+offset*.52; foot=x+offset; ground=u(.80,1.01) if not far else u(.73,.9)
        p.seg(c,[(x,yy),(knee,.42),(foot,ground-.10),(foot-.003,ground)],[.113 if horse else .086,.057 if horse else .047,.029,.035]); p.seg(hoof,[(foot,ground-.055),(foot-.021,ground+.008)],[.034,.048]);
    leg(hip-.09,True,True); leg(shoulder+.095,False,True)
    if horse:
        tx=bodylen+squeeze*.02; p.seg(mane,[(tx-.12,by-.15),(tx+.085,by+.04),(tx+.13+gait*.3,by+.39),(tx+.24+gait,by+u(.47,.73))],[.10,.105,.081,.026])
    else:p.ellipse(mix(belly,white,.25),bodylen*.91,by-.08,.12,.13,-.54)
    p.ellipse(fur,.025,by,bodylen,depth,-.035,shade=.085);p.ellipse(mix(belly,fur,.55),-.015,by+depth*.36,bodylen*.76,depth*.53,shade=.045)
    leg(hip,True,False); leg(shoulder,False,False)
    # Neck projects up/forward; the poll is higher than the nose.
    pollx=-bodylen-u(.015,.19)*squeeze; polly=u(-.82,-.58) if not graze else u(.19,.42)
    px=pollx;py=polly; hx=px-u(.13,.25)*squeeze; hy=py+(.17 if horse else .068) if not graze else py+.27
    p.seg(fur,[(shoulder+.035,by-.035),((px+shoulder)*.5,(py+by)*.5),(px,py)],[depth*.9,.16 if horse else .126,.113]);
    if horse: p.seg(mane,[(shoulder+.15,by-depth*.49),(px+.20,(py+by)*.49-.07),(px+.044,py-.085)],[.057,.067,.04])
    ear(px+.036,py-.127,.65 if horse else .79,.43 if horse else .81)
    if not horse and not graze and coin(.36):antlers(px,py-.07,.71)
    p.ellipse(fur,px-.038,py,.155,.14,-.41);p.seg(fur,[(px-.053,py-.014),(hx,hy)],[.125,.087 if horse else .072]);p.ellipse(nose,hx,hy,.096 if horse else .075,.074,-.36);p.ellipse(eye,px-.079,py-.026,.031,.026,shade=0)
    if horse and not graze and coin(.18):
        cloth=tint(pick(metals));p.ellipse(dark,.052,by-.24,.22,.09);p.seg(cloth,[(.073,by-.57),(.06,by-.34)],[.095,.105]);p.ellipse(tint(rgb(.76,.55,.41)),.066,by-.70,.082,.092);p.ellipse(black,.06,by-.76,.089,.047);p.seg(dark,[(.07,by-.28),(-.052,by-.083),(.035,by+.17)],[.061,.046,.033])

def pet(p,isdog,portrait):
    base=tint(pick(furs),.035); marking=tint(pick([white,white,base*.57,base*.45,pick(furs)]),.025); darkfur=mix(base,dark,u(.28,.86)); pale=mix(base,white,u(.19,.80)); muzzle=pale if coin(.7) else darkfur; flop=isdog and coin(.71); tiger=not isdog and coin(.58); pointed=not flop; noseblack=mix(black,base,.13) if isdog or coin(.54) else rgb(.64,.43,.41)
    def face(cx,cy,s,turn=0):
        def e(c,x,y,rx,ry,ang=0,sh=.054):p.ellipse(c,cx+x*s,cy+y*s,rx*s,ry*s,ang,sh)
        def poly(c,pts):p.poly(c,[(cx+s*x,cy+s*y) for x,y in pts])
        def ln(c,a,b,w):p.line(c,cx+a[0]*s,cy+a[1]*s,cx+b[0]*s,cy+b[1]*s,w*s)
        width=u(.51,.68);height=u(.49,.63);t=abs(turn); sh=turn*.24; blaze=coin(.42)
        for side in (-1,1):
            if flop:
                e(darkfur,side*width*.96,-.02,width*.34,u(.35,.61),-.23*side);e(mix(base,darkfur,.65),side*width*1.04,.12,width*.23,.32,-.16*side)
            else:
                tx=side*width*u(.76,1.12); ty=-height*u(1.27,1.68)
                poly(base,[(side*width*.95,-height*.07),(tx,ty),(side*width*.22,-height*.66)])
                poly(mix(base,rgb(.61,.42,.39),.57),[(side*width*.85,-height*.47),(tx*.97,ty*.88),(side*width*.42,-height*.70)])
        e(base,0,0,width,height,sh*.25,.095)
        if not isdog:e(base,0,height*.30,width*.92,height*.53,sh*.19)
        if coin(.49):
            side=pick([-1,1]);e(mix(base,marking,.68),side*width*.40,-height*.26,width*.37,height*.55,side*-.4)
        if tiger:
            for i in (-1,0,1):poly(darkfur,[(i*.13-.039,-height*.83),(i*.13+.055,-height*.84),(i*.078,-height*.23)])
            for side in (-1,1):
                for dy in (.025,.15):ln(mix(darkfur,base,.28),(width*.92*side,dy-.08),(width*.54*side,dy),.024)
        if blaze and isdog:poly(mix(base,pale,.72),[(sh*.2-.10,-height*.93),(sh*.2+.09,-height*.91),(sh+.046,.01),(sh+.13,.24),(sh-.12,.24),(sh-.037,0)])
        eyey=-height*.20; eyepos=width*(.50 if not isdog else .48); iris=tint(pick([rgb(.48,.52,.27),rgb(.67,.58,.31),rgb(.39,.43,.36)]),.025)
        for side in (-1,1):
            ex=sh+eyepos*side*(1-.30*t); scale=1-turn*side*.23
            if isdog and coin(.60):e(darkfur,ex,eyey,.113*scale,.099)
            e(black,ex,eyey,.093*scale if not isdog else .067*scale,.051 if not isdog else .066,-.18*side)
            if not isdog:e(iris,ex,eyey,.065*scale,.032,-.15*side);e(black,ex,eyey,.015,.036)
            e(white,ex-.018,eyey-.019,.014,.014,0,0)
        nx=turn*(.31 if isdog else .22); ny=height*(.49 if isdog else .31)
        if isdog:e(muzzle,nx,ny-.011,width*.47,height*.38,-turn*.13);e(mix(muzzle,darkfur,.40),nx,ny+height*.22,width*.32,height*.18)
        else:
            for side in (-1,1):e(pale,nx+side*.093,ny+.07,.133,.112,-side*.22)
        nw=width*(.245 if isdog else .15); poly(noseblack,[(nx-nw,ny-.049),(nx-nw*.60,ny-.087),(nx+nw*.82,ny-.068),(nx+nw,ny-.017),(nx+.01,ny+.053),(nx-.031,ny+.046)])
        ln(darkfur,(nx,ny+.044),(nx,ny+.11),.014);ln(darkfur,(nx,ny+.108),(nx-.104,ny+.124),.014);ln(darkfur,(nx,ny+.108),(nx+.107,ny+.118),.014)
        if isdog and coin(.32):e(rgb(.70,.39,.40),nx+.01,ny+.222,.088,.137)
        if not isdog and coin(.6):
            for side in (-1,1):
                for j in range(2):ln(mix(pale,white,.43),(nx+side*.15,ny+j*.064),(nx+side*.63,ny+(j-.57)*.13),.009)
    if portrait:
        s=u(1.01,1.40); cy=u(-.26,.01)
        p.ellipse(base,.06,.81,.60,.77,-.08,.095)
        if coin(.61):p.ellipse(marking,.01,.74,.34,.60,.10)
        face(0,cy,s,u(-.94,.94));return
    pose=pick(['side','sit','sit','lie','side'] if isdog else ['sit','lie','lie','side','sit'])
    by=u(.01,.15); headx=u(-.64,-.40);heady=u(-.39,-.25); bodyw=u(.46,.70); s=u(.52,.74)
    if pose=='sit':
        p.ellipse(base,.03,.20,.44,.65,-.17,.085);p.ellipse(base,.24,.50,.36,.32)
        if coin(.65):p.ellipse(marking,-.09,.16,.25,.44,-.09)
        for side in [-1,1]:p.seg(base,[(side*.20,.20),(side*.22,.51),(side*.23,.76)],[.121,.081,.072]);p.ellipse(pale,side*.23-.02,.77,.123,.066)
        if not isdog:p.seg(darkfur,[(.36,.44),(.56,.59),(.42,.74),(.08,.76)],[.087,.082,.065,.025])
        face(-.055,-.42,u(.74,.99),u(-.84,.84))
    else:
        if pose=='lie': heady=by-.12
        tailpoints=[(.48,by),(.72,by+.02),(.86,by-.23),(.77,by-.55)] if pose=='side' else [(.40,by+.03),(.73,by+.16),(.61,by+.33),(.22,by+.32)]
        p.seg(darkfur,tailpoints,[.075,.07,.058,.032]);
        for xx in [-.33,.38]:
            if pose=='side':
                for off in [-.11,.08]:
                    p.seg(base*(.83 if off<0 else .97),[(xx+off,by+.05),(xx+off*.5,by+.36),(xx+off*1.7,by+.67)],[.107,.067,.05]);p.ellipse(pale,xx+off*1.7-.035,by+.68,.089,.052)
        p.ellipse(base,.075,by,bodyw,u(.24,.34),.055,.09)
        if coin(.6):p.ellipse(marking,.23,by-.014,bodyw*.44,.22,-.27)
        if tiger:
            for xx in np.arange(-.20,.56,.16):p.seg(darkfur,[(xx-.035,by-.18),(xx,by-.09),(xx+.023,by+.06)],[.034,.028,.009])
        if pose=='lie':p.ellipse(pale,-.37,by+.22,.29,.083,-.12)
        face(headx,heady,s,u(-1,-.37) if coin(.65) else u(-.47,.77))

def bird(p):
    mode=pick(['perch','perch','perch','swim','perch','fly','wade','front']); plum=tint(pick([pick(furs),pick(coats),pick(leaves),dark,white,graymetal,pick(waters),rgb(.75,.22,.13),rgb(.74,.65,.20)]),.034); wing=mix(plum,tint(pick([dark,pick(furs),graymetal])),u(.35,.81)); breast=tint(pick([plum,plum,mix(plum,white,.41),white,rgb(.71,.42,.19),rgb(.7,.63,.29)]),.027); beak=tint(pick([black,black,rgb(.59,.47,.28),rgb(.85,.58,.20)]),.02);cap=pick([plum,plum,dark,wing]); leg=tint(pick([dark,pick(dirt)]),.015)
    if mode=='fly':
        p.ellipse(plum,0,0,.48,.13,-.19);p.ellipse(cap,-.42,.066,.14,.111);p.poly(beak,[(-.51,.05),(-.77,.10),(-.52,.12)]);p.poly(wing,[(.21,-.01),(.72,-.22),(.79,-.10),(.31,.08)])
        p.poly(wing,[(-.24,-.07),(-.44,-.43),(-.23,-.96),(-.15,-.56),(.14,-.15),(.24,-.035)]);p.poly(plum*.87,[(-.23,.04),(-.15,.43),(.25,.91),(.18,.51),(.12,.23),(.24,.03)]); return
    if mode=='front':
        p.ellipse(wing,.015,.12,.39,.60,.05);p.ellipse(breast,0,.14,.30,.47);p.ellipse(cap,0,-.39,.29,.27);p.poly(beak,[(-.09,-.39),(.09,-.39),(.00,-.24)]);
        for side in [-1,1]:p.ellipse(black,side*.185,-.426,.036,.041);p.line(leg,side*.14,.59,side*.15,.78,.024)
        return
    swim=mode=='swim';wade=mode=='wade'; perch=mode=='perch';bx=.095;by=.12 if not swim else .21; tilt=.64 if perch else .09;hx=-.26 if perch else -.47;hy=-.29 if perch else -.27
    if perch:
        slope=u(-.34,.34);p.line(tint(pick(dirt),.032),-1.7,.63-1.7*slope,1.6,.63+1.6*slope,u(.035,.085))
    if not swim:
        foot=.63 if perch else .95
        for s in [-1,1]:
            xx=bx+s*.10; p.seg(leg,[(xx,by+.15),(xx+.02,by+.37),(xx-.07,foot)],[.028,.021,.018]);p.line(leg,xx-.07,foot,xx-.17,foot+.04,.018)
    if perch:p.poly(wing,[(.11,.07),(.40,.12),(.84,.69),(.70,.72),(.45,.55)])
    else:p.poly(wing,[(.28,.15),(.85,-.07),(.75,.15),(.57,.26)])
    p.ellipse(breast,bx,by,.47 if perch else .66,.28 if perch else .27,tilt,.087);p.ellipse(wing,bx+.073,by-.043,.39 if perch else .49,.155,tilt-.045,.057)
    if coin(.49):p.line(mix(wing,white,.49),bx-.10,by-.14,bx+.20,by+.019,.026)
    if not perch:p.seg(plum,[(-.31,by+.01),(-.42,-.066),(hx,hy)],[.143,.096,.103]);
    p.ellipse(cap,hx,hy,.211 if perch else .177,.212 if perch else .157,.18,.07)
    if coin(.48):p.ellipse(mix(breast,white,.19),hx+.025,hy+.048,.097,.071,.15)
    p.poly(beak,[(hx-.123,hy-.045),(hx-(u(.29,.41) if not wade else .56),hy+u(-.047,.048)),(hx-.139,hy+.074)])
    p.ellipse(black,hx-.077,hy-.05,.036,.034);p.ellipse(white,hx-.089,hy-.061,.010,.010,shade=0)

def frog(p):
    skin=tint(pick(palette('6e8046 758d46 73904b 474c2c a09756 826d48 665b42 737543 987b50 8aa143 587439 756143 a3a075 8c693c 36493c')), .035)
    belly=mix(skin,tint(pick([white,pick(dirt),pick(fields)])),.46); spots=mix(skin,dark,u(.36,.71)); iris=tint(pick([rgb(.63,.55,.26),rgb(.55,.49,.28),rgb(.54,.41,.22)]),.016)
    front=coin(.45);s=pick([-1,1]); bodyx=.0 if front else .18; bodyy=.02;headx=.0 if front else -.40;heady=-.13 if front else -.088
    for sign in (-1,1):
        if front:
            p.ellipse(skin*.86,sign*.51,.20,.224,.36,-.65*sign);p.seg(skin,[(sign*.52,.31),(sign*.76,.47),(sign*.52,.56)],[.124,.07,.037])
        else:
            p.ellipse(skin*.91,.51,.115,.41,.20,-.41*sign);p.seg(skin,[(.64,.19),(.24,.38),(.72,.43)],[.12,.084,.038])
    p.ellipse(skin,bodyx,bodyy,.57,.41 if front else .32,-.096,.09)
    for j in range(rng.randint(8,18)):
        a=u(0,6.28);rad=u(.12,.90); p.ellipse(spots,bodyx+.48*rad*np.cos(a),bodyy+.27*rad*np.sin(a),u(.026,.074),u(.02,.056),u(-.9,.9),shade=.023,alpha=u(.43,.88))
    for side in (-1,1):
        xx=(side*.32 if front else headx+.14); endx=(side*.54 if front else xx-.24+.11*side);endy=u(.35,.56)
        p.seg(skin,[(xx,.008),(xx+side*.11,.21),(endx,endy)],[.090,.068,.041]);
        for toe in (-1,0,1):p.line(skin,endx,endy,endx+.093*toe,endy+.102-.025*abs(toe),.018)
    p.ellipse(belly,headx,heady+.15,.46 if front else .36,.20,-.08);p.ellipse(skin,headx,heady,.48 if front else .37,.232 if front else .20,-.035,.073)
    if not front:p.ellipse(skin,headx-.18,heady+.027,.258,.144,-.06)
    for side in ((-1,1) if front else (-1,)):
        ex=headx+side*(.28 if front else .069);ey=heady-.143;p.ellipse(skin,ex,ey,.129,.125);p.ellipse(iris,ex,ey-.009,.094,.083);p.ellipse(black,ex,ey-.012,.068,.045);p.ellipse(white,ex-.029,ey-.039,.019,.018,shade=0)
    if front:p.seg(spots,[(-.34,heady+.066),(-.20,heady+.123),(0,heady+.141),(.20,heady+.12),(.33,heady+.06)],[.012,.017,.014,.015,.012])
    else:p.line(spots,headx-.36,heady+.069,headx+.07,heady+.12,.016)

def render(k):
    img,env=environment(k); portrait=False
    if k in (3,5):portrait=coin(.60 if k==3 else .55)
    if k in (4,7):portrait=coin(.135 if k==7 else .115)
    base=((10,19),(12,20),(8,17),(11,20),(8,16),(11,20),(10,20),(9,16),(10,18),(12,19))[k]
    s=u(*base);sx=s*u(.90,1.14);sy=s*u(.85,1.14)
    if portrait:sx*=u(1.02,1.27);sy*=u(1.04,1.3)
    dx=rng.normal(0,1.55);dy=rng.normal(0,1.65)
    angle=rng.normal(0,[.26,.06,.25,.15,.08,.14,.24,.077,.065,.055][k]);
    if k in (1,9):sy*=u(1.12,1.57);dy+=u(.0,2.3)
    if k==8:dy+=u(.2,3.9);sy*=u(.95,1.36)
    if k==2 and env=='waterbird':dy+=1.65
    if portrait and k in (3,5):dy-=1.05
    p=Canvas(img,16+dx,16+dy,sx,sy,angle,.014 if k in (2,3,4,5,6,7) else 0)
    if k==0:airplane(p)
    elif k in (1,9):vehicle(p,k==9)
    elif k==8:boat(p)
    elif k in (4,7):ungulate(p,k==7,portrait)
    elif k in (3,5):pet(p,k==5,portrait)
    elif k==2:bird(p)
    else:frog(p)
    img=p.im
    # Image-wide illumination and independent material texture, no templates shared.
    texture=(noise(4,4)*u(.020,.095)+noise(10,10)*u(.012,.075)+rng.randn(32,32)*u(.018,.072)) if k in (2,3,4,5,6,7) else (noise(6,4)*u(.01,.064)+noise(12,5)*u(.005,.036))
    img=img+(p.opacity*texture)[:,:,None]+rng.normal(0,u(.004,.023),(32,32,1))
    if coin(.15):img=img.mean(2,keepdims=True)*u(.35,.85)+img*.30
    img=(img*u(.78,1.21)+u(-.063,.087))*rng.uniform(.92,1.08,3)+rng.normal(0,.018,(1,1,3))
    img=np.clip(img,.0,1.)**u(.80,1.25)
    img=img.transpose(2,0,1).astype(np.float32)
    mask=p.opacity
    if coin():img=img[:,:,::-1];mask=mask[:,::-1]
    return img,mask

SOURCE = r'''import numpy as np

# Region and contour descriptions. All normalizations here are within one image.
def pooled(a, n):
    return a.reshape(len(a), a.shape[1], n, 32//n, n, 32//n).mean((3,5))

def unit(a, floor):
    return a / np.sqrt(np.sum(a*a,axis=1,keepdims=True) + floor*floor)

def center(a):
    return a - a.mean((-2,-1),keepdims=True)

def flat(a):
    return a.reshape(len(a),-1)

def smooth(a):
    p = np.pad(a,((0,0),(0,0),(1,1),(1,1)),mode='edge')
    t = (p[:,:,:-2]+2*p[:,:,1:-1]+p[:,:,2:])*.25
    return (t[:,:,:,:-2]+2*t[:,:,:,1:-1]+t[:,:,:,2:])*.25

def gradient(a):
    p = np.pad(a,((0,0),(0,0),(1,1),(1,1)),mode='edge')
    return (p[:,:,1:-1,2:]-p[:,:,1:-1,:-2])*.5,(p[:,:,2:,1:-1]-p[:,:,:-2,1:-1])*.5

def shape_features(x):
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    y=.30*r+.57*g+.13*b
    v=pooled(y,8)
    h=unit(flat(center(v)),.44)
    local=v-smooth(smooth(v))
    local=local/(.065+np.sqrt(smooth(smooth(local*local))))
    hloc=unit(flat(local),1.5)
    rgb=pooled(np.concatenate((g-b,r-g),axis=1),4)
    col=unit(flat(center(rgb)),.32)
    contours=[]
    for a,d in ((smooth(y),8),(smooth(smooth(smooth(y))),4)):
        dx,dy=gradient(a)
        length=np.sqrt(dx*dx+dy*dy+.000001)
        bins=[]
        for i in range(6):
            angle=i*np.pi/6
            cosine=(dx*np.cos(angle)+dy*np.sin(angle))/length
            bins.append(np.maximum(0,cosine*cosine-.56)*length)
        hog=pooled(np.concatenate(bins,axis=1),d)
        hog=smooth(hog)*.62+hog*.38
        # Compare grain direction as well as the spatial layout of the contours.
        hog=np.sqrt(hog+.0007)
        orient=hog-hog.mean(1,keepdims=True)
        spatial=center(hog)
        contours.append(unit(flat(.65*orient+spatial),.27 if d==8 else .14))
    return np.concatenate((.45*h,.40*hloc,.33*col,.57*contours[0],.46*contours[1]),axis=1).astype(np.float32)

def pos(a):
    return np.clip(a,0,1)

def scene_features(x):
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    y=.30*r+.57*g+.13*b
    green=pos((g-b-.012)/(.09+.09*y))*pos((g-r+.048)/.10)
    warm=pos((r-b-.025)/(.10+.11*y))*pos((r-g+.003)/.09)
    blue=pos((b-r+.004)/(.09+.08*y))*pos((b-g+.019)/.075)
    vivid=pos((np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)-.12)/.24)
    gray=1-pos((np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)-.03)/.15)
    colors=np.concatenate((y, r-g,g-b,warm,blue,green,gray,vivid,pos((y-.60)/.27),pos((.31-y)/.23)),axis=1)
    reg=pooled(colors,4)
    # A soft bounded-distance kernel per region. Adjacent channels encode
    # different properties, so no cross-image covariance estimate is needed.
    scales=np.array([.28,.105,.13,.48,.48,.46,.55,.53,.50,.47],np.float32)[None,:,None,None]
    coeff=np.array([1,.78,.95,.80,.82,.82,.4,.26,.34,.37],np.float32)[None,:,None,None]
    z=reg/scales
    enc=[]
    for freq,weight in ((.85,.8),(1.8,.45)):
        enc.extend((flat(coeff*np.cos(z*freq)*weight),flat(coeff*np.sin(z*freq)*weight)))
    return unit(np.concatenate(enc,axis=1),.001).astype(np.float32)

def region(a,t,b,l,r):
    return a[:,t:b,l:r].mean((1,2))

def cues(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=.30*r+.57*g+.13*b
    green=pos((g-b-.015)/(.07+.07*y))*pos((g-r+.035)/.08)
    olive=pos((g-b-.035)/.15)*pos((g-r+.12)/.14)
    brown=pos((r-b-.035)/(.14+.06*y))*pos((r-g-.005)/.09)
    blue=pos((b-r+.01)/(.075+.08*y))*pos((b-g+.025)/.09)
    gray=1-pos((np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)-.035)/.15)
    light=pos((y-.50)/.30)
    sky=blue*(.4+.6*light)+gray*light*.55
    outside=np.ones((32,32),np.float32); outside[5:27,5:27]=0
    def outer(a):
        return (a*outside).sum((1,2))/outside.sum()
    def core(a):
        return region(a,7,26,6,26)
    def top(a):
        return region(a,0,8,0,32)
    def bottom(a):
        return region(a,25,32,0,32)
    grass=outer(green); grasslow=bottom(green); woods=outer(olive)
    fur=core(brown); fieldbrown=outer(brown); fgblue=core(blue)
    air=outer(sky); highsky=top(sky); sea=bottom(blue); grayroad=bottom(gray)
    low=y[:,24:].mean((1,2)); high=top(y)
    a=smooth(y[:,None]); dx,dy=gradient(a); dx=dx[:,0]; dy=dy[:,0]
    mag=np.sqrt(dx*dx+dy*dy)
    bgdetail=outer(mag); fgdetail=core(mag); activity=mag.mean((1,2))
    fine=np.abs(y-a[:,0]); grit=core(fine); bggrit=outer(fine)
    flatbg=pos((.052-bgdetail)/.047)
    busy=pos((bgdetail-.027)/.055)
    # Long horizontal edges, horizontal versus diagonal anisotropy,
    # and rectilinear structures characteristic of manufactured objects.
    ah=np.abs(dy); av=np.abs(dx)
    straight=np.abs((dy[:,:,0:24]+dy[:,:,2:26]+dy[:,:,4:28]+dy[:,:,6:30]+dy[:,:,8:32])/5).mean((1,2))
    st=pos((straight/(ah.mean((1,2))+.001)-.37)/.42)
    anis=(ah.mean((1,2))-av.mean((1,2)))/(activity+.005)
    horizon=pos(anis*3.7+.25)
    axial=(np.abs(ah-av).mean((1,2))/(activity+.001)-.47)*5
    horizontal=.55*horizon+.45*st
    splash=pos((core(np.maximum(r,np.maximum(g,b))-np.minimum(r,np.minimum(g,b)))-.14)/.25)
    redpaint=core(pos((r-np.maximum(g,b)-.09)/.28)); coldpaint=fgblue*(1-outer(blue))
    skyland=highsky*pos((high-low+.08)/.40)
    natural=grass+.55*woods+.23*fieldbrown-.45*sea
    vehicle=.95*(horizontal-.50)+.46*axial-.43*natural+.32*(grayroad-.45)+.42*(redpaint+coldpaint)-.24*pos((grit-.032)/.04)
    animal=-.56*(horizontal-.5)-.27*axial+.52*natural-.28*air+.35*(fur-.23)
    z=np.zeros((len(x),10),np.float32)
    z[:,0]=.80*(air-.4)+.90*(flatbg-.50)+.44*(outer(blue)-.18)+.18*(highsky-.4)-.49*grass-.23*fur+.10*(horizon-.4)
    z[:,1]=vehicle+.32*(grayroad-.5)-.27*skyland+.36*(redpaint+coldpaint)-.22*grasslow+.17*(st-.5)
    z[:,9]=vehicle+.48*(highsky-.35)+.37*skyland+.29*axial+.20*(st-.50)-.42*sea
    z[:,8]=.85*(horizontal-.44)+1.24*(sea-.18)+.49*(highsky-.40)+.32*(outer(blue)-.15)-.66*grass-.35*fieldbrown-.22*fur+.17*(flatbg-.4)
    z[:,2]=animal*.44+.70*(grass-.21)+.31*(flatbg-.45)+.40*(splash-.22)+.42*(fgblue-.12)-.15*(horizontal-.50)
    z[:,3]=animal*.67-.49*grass-.25*highsky+.35*(grit-.025)/.045+.14*(bggrit-.025)/.045+.20*(.34-splash)
    z[:,5]=animal*.77-.37*grass-.19*highsky+.30*(grit-.025)/.045+.13*(fieldbrown-.25)+.09*(.30-splash)
    z[:,4]=animal+.63*(grass-.24)+.57*(woods-.38)+.43*(fur-.23)+.37*(busy-.4)-.38*air
    z[:,7]=animal*.86+.62*(grasslow-.25)+.36*(fur-.25)+.18*(highsky-.40)+.17*(horizontal-.50)-.13*(busy-.4)
    z[:,6]=animal*.82+.72*(core(olive)-.36)+.50*(fieldbrown-.28)+.32*(busy-.4)-.51*highsky-.32*sea+.16*(bggrit-.025)/.045
    return z

def classify(x):
    h=shape_features(x); c=scene_features(x)
    parts=flat(part_features(x))
    prior=cues(x)+evidence(x)
    out=np.empty((len(x),10),np.float32)
    for k in range(10):
        sh=h@CONSTANTS['shape'+str(k)]
        co=c@CONSTANTS['scene'+str(k)]
        # Pool strongest individual explanations. A complementary shape-only
        # term allows for cases where the setting was not anticipated.
        pm=parts@CONSTANTS['parts'+str(k)]
        obj=.76*sh+.24*pm
        joint=obj+.80*co
        top=np.partition(joint,-12,axis=1)[:,-12:].mean(axis=1)
        shapeonly=np.partition(obj,-5,axis=1)[:,-5:].mean(axis=1)
        coloronly=np.partition(co,-15,axis=1)[:,-15:].mean(axis=1)
        out[:,k]=13*(.75*top+.25*shapeonly)+5.8*coloronly+prior[:,k]
    return np.argmax(out,axis=1)

def predict(x):
    # Fixed tile dimensions also fix the matrix multiply/reduction paths.
    labels=np.empty(len(x),np.int64)
    for i in range(0,len(x),48):
        n=min(48,len(x)-i)
        a=np.zeros((48,3,32,32),np.float32)
        a[:n]=x[i:i+n]
        labels[i:i+n]=classify(a)[:n]
    return labels

# A dense field of local shape codes. Rendered templates can restrict this field
# to the object's support; for an input photograph no segmentation is required.
def part_features(x):
    y=smooth(.30*x[:,0:1]+.57*x[:,1:2]+.13*x[:,2:3]); gx,gy=gradient(y)
    length=np.sqrt(gx*gx+gy*gy+.00012)
    coarse=pooled(y,8); p=np.pad(coarse,((0,0),(0,0),(1,1),(1,1)),mode='edge')
    cx=(p[:,:,1:-1,2:]+p[:,:,1:-1,:-2])*.5-coarse
    cy=(p[:,:,2:,1:-1]+p[:,:,:-2,1:-1])*.5-coarse
    xy=(p[:,:,2:,2:]-p[:,:,2:,:-2]-p[:,:,:-2,2:]+p[:,:,:-2,:-2])*.25
    parts=np.concatenate((pooled(gx,8)*1.3,pooled(gy,8)*1.3,pooled((gx*gx-gy*gy)/length,8),pooled(2*gx*gy/length,8),cx*.77,cy*.77,xy*.76),axis=1)
    return parts/np.sqrt(.050**2+np.sum(parts*parts,axis=1,keepdims=True))

def shiftmap(a,x,y):
    # Float image translated by an integral number of pixels with edge extension.
    return a[:,np.clip(np.arange(32)+y,0,31),:][:,:,np.clip(np.arange(32)+x,0,31)]

def evidence(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=smooth((.30*r+.57*g+.13*b)[:,None])[:,0]
    noiselevel=np.abs(y-(.30*r+.57*g+.13*b))
    # Dark points with lighter surroundings, off-centre reflections allowed.
    def dip(a,rad):
        l=shiftmap(a,-rad,0);rr=shiftmap(a,rad,0);t=shiftmap(a,0,-rad);bb=shiftmap(a,0,rad)
        return .57*(.25*(l+rr+t+bb)-a)+.43*np.minimum(.5*(l+rr)-a,.5*(t+bb)-a)
    wheel=0; facecat=0;facedog=0; legs=0
    for rad in (2,3,4):
        holes=dip(y,rad)
        # A tire can have either a dark center or a light hub. The rim sits
        # below the body; a shadow alone is not a pair of circular contours.
        rim=(shiftmap(y,-rad//2,0)+shiftmap(y,(rad+1)//2,0)+shiftmap(y,0,-(rad+1)//2)+shiftmap(y,0,(rad+1)//2))*.25
        tire=np.maximum(holes,.70*(shiftmap(y,0,rad+1)+shiftmap(y,0,-rad-1))*.5-.70*rim+.23*(y-rim))
        tire=tire+.22*np.maximum(0,.38-y)
        for gap in (3*rad+1,5*rad+1):
            j=np.maximum(shiftmap(tire,gap,0),.90*shiftmap(tire,gap,1)); pair=np.minimum(tire,j)*.72+.28*np.minimum((tire+j)*.5,tire+.05)
            v=pair[:,16:29,1:32-gap-1].max((1,2));wheel=np.maximum(wheel,v)
    for gap in (5,8,11,14):
        half=gap//2;rad=max(2,gap//3);holes=dip(y,rad); other=shiftmap(holes,gap,0)
        eyes=np.minimum(holes,other)*.79+.21*(holes+other)*.5
        for isdog in (False,True):
            down=max(2,int(gap*(.73 if isdog else .46)));nose=shiftmap(dip(y,max(2,int(rad*(1.35 if isdog else .73)))),half,down)
            bridge=shiftmap(y,half,down//2)-.5*(y+shiftmap(y,gap,0))
            # Require the eye and nose responses to coexist, not just one dark feature.
            m=.53*np.minimum(eyes,1.4*nose)+.30*eyes+.17*np.clip(bridge,-.25,.33)
            v=m[:,5:22,2:30-gap].max((1,2));
            if isdog:facedog=np.maximum(facedog,v)
            else:facecat=np.maximum(facecat,v)
    # Vertical, narrow leg-like strokes in the lower half, with contrast
    # extending downwards for several pixels. Hooves taper toward the bottom.
    for w in (2,3):
        bar=(shiftmap(y,-w,0)+shiftmap(y,w,0))*.5-y
        bar=np.maximum(bar,-.68*bar);longbar=np.minimum(bar,np.maximum(shiftmap(bar,0,4),shiftmap(bar,1,4)))
        pair=np.minimum(longbar,shiftmap(longbar,8,0));legs=np.maximum(legs,pair[:,17:27,3:22].max((1,2)))
    # Broad level surfaces and depth cues: smooth top background versus
    # textured foreground, and dry foliage versus a saturated grassy field.
    gx,gy=gradient(y[:,None]);m=np.sqrt(gx[:,0]**2+gy[:,0]**2)
    topdetail=region(m,0,7,0,32); lowerdetail=region(m,25,32,0,32);core=region(m,8,24,8,24)
    isolation=pos((core-.43*topdetail-.57*lowerdetail)/.085)
    flecks=pos((region(noiselevel,6,26,6,26)-.018)/.059)
    canopy=pos((g-b-.035)/.19)*pos((g-r+.030)/.14)
    lawn=region(canopy,23,32,0,32); foliage=region(canopy,0,11,0,32)
    z=np.zeros((len(x),10),np.float32)
    wheels=pos((wheel-.098)/.20); paws=pos((legs-.075)/.15); cat=pos((facecat-.060)/.18);dog=pos((facedog-.078)/.20)
    z[:,1]=1.03*(wheels-.40)+.20*(isolation-.44)
    z[:,9]=.61*(wheels-.4)+.18*(1-flecks-.62)
    z[:,7]=.82*(paws-.30)+.53*(lawn-.37*foliage-.19)
    z[:,4]=.70*(paws-.30)+.40*(foliage-.23)
    z[:,2]=.63*(isolation-.38)-.26*(paws-.30)
    z[:,3]=.74*(cat-.38)-.27*(dog-.32)+.20*(flecks-.27)
    z[:,5]=.77*(dog-.31)-.17*(cat-.39)+.14*(isolation-.38)
    z[:,6]=.25*(flecks-.27)+.24*(lowerdetail+topdetail-.077)/.088-.34*(paws-.30)
    z[:,0]=.42*(isolation-.39)+.24*(.041-lowerdetail)/.065
    z[:,8]=.16*(.043-topdetail)/.055-.25*(paws-.30)
    return z
'''


def make_answer(folder):
    os.makedirs(folder,exist_ok=True)
    with open(os.path.join(folder,'solution.py'),'w',encoding='utf-8',newline='\n') as f:f.write(SOURCE)
    # Only individual templates are serialized. All feature code treats the
    # leading axis as independent; the loops below never summarize that axis.
    ns={};exec(SOURCE,ns); tables={}
    for k in range(10):
        shapes=[];scenes=[];parts=[]
        for batch in range(80):
            drawings=[render(k) for j in range(48)]
            batchimg=np.stack([img for img,mask in drawings],axis=0)
            batchmask=np.stack([mask for img,mask in drawings],axis=0)[:,None]
            shapes.append(ns['shape_features'](batchimg));scenes.append(ns['scene_features'](batchimg))
            a=ns['part_features'](batchimg); w=ns['pooled'](batchmask,8)
            support=.7*w+.3*ns['smooth'](w)
            a=a*support
            norm=np.maximum(np.sum(np.sqrt(np.sum(a*a,axis=1)+1e-9),axis=(1,2)),3.0)
            parts.append((a/norm[:,None,None,None]).reshape(len(a),-1))
        for name,seq in (('shape',shapes),('scene',scenes),('parts',parts)):
            tables[name+str(k)]=np.ascontiguousarray(np.concatenate(seq,axis=0).T,dtype=np.float32)
        print('Constructed templates for class',k,flush=True)
    np.savez(os.path.join(folder,'constants.npz'),**tables)

if __name__=='__main__':make_answer(sys.argv[1])
