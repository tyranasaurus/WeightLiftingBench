"""Build a library of separately rendered, separately encoded object views.
No inputs besides the output directory; coordinates are in object space.
"""
from pathlib import Path
import sys, math
import numpy as np

# This part is used identically on a render and on an input photograph.
features = '''import numpy as np

def blur(a, k):
    def along(v, axis):
        n=v.shape[axis]
        j=np.arange(n)
        return (v.take(np.maximum(j-k,0),axis=axis)+2*v+v.take(np.minimum(j+k,n-1),axis=axis))*.25
    return along(along(a,-1),-2)

def pool(a,k):
    return a.reshape(a.shape[:-2]+(32//k,k,32//k,k)).mean(axis=(-1,-3))

def groups(a):
    return a.reshape(len(a),-1)

def unit(a,noise):
    a=groups(a)
    return a/np.sqrt((a*a).mean(axis=1,keepdims=True)+noise*noise)/np.sqrt(np.float32(a.shape[1]))

def bundle(parts):
    return np.ascontiguousarray(np.concatenate([np.sqrt(np.float32(w))*unit(a,n) for w,a,n in parts],axis=1),dtype=np.float32)

def grad(a):
    j=np.arange(32)
    return (a.take(np.minimum(j+1,31),axis=-1)-a.take(np.maximum(j-1,0),axis=-1),a.take(np.minimum(j+1,31),axis=-2)-a.take(np.maximum(j-1,0),axis=-2))

def describe(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=.30*r+.52*g+.18*b
    rg=(r-g)/(.28+y)
    yb=((r+g)*.5-b)/(.28+y)
    # Local edge energy and a four-bin contrast-insensitive orientation code.
    # The doubled angle avoids a discontinuity at pi.
    xs=blur(x,1)*.65+x*.35
    gx,gy=grad(xs)
    xx=(gx*gx).mean(1); yy=(gy*gy).mean(1); xy=(gx*gy).mean(1)
    power=xx+yy
    mag=np.sqrt(power+.000001)
    cs=(xx-yy)/(power+.00005); sn=2*xy/(power+.00005)
    env=blur(blur(mag,2),3)
    wt=mag/np.sqrt(.035+env)
    bins=np.stack([np.maximum(t+.22,0)**1.5 for t in (cs,sn,-cs,-sn)],axis=1)*wt[:,None]
    cell=pool(blur(bins,1),4)
    local=cell-cell.mean(1,keepdims=True)
    strength=pool(blur(wt,1),4)
    relative=strength-strength.mean((1,2),keepdims=True)
    tone=pool(blur(y,1),4)
    band=tone-blur(tone,1)
    broad=tone-tone.mean((1,2),keepdims=True)
    # Overall angular preference is retained separately; subtracting the spatial
    # pedestal here makes the actual positions of contours matter more.
    centered=local-.75*local.mean((2,3),keepdims=True)
    shape=bundle([(.37,centered,.007),(.12,local,.007),(.19,relative,.020),(.16,band,.016),(.16,broad,.045)])
    # Colour maps encode material and illumination, not just grey-level shape.
    opp=np.stack((rg*1.4,yb),1)
    col=pool(blur(np.tanh(opp*3),1),8)
    warm=np.tanh(yb*3); green=np.tanh(-rg*6); cool=-warm
    pale=np.clip((y-.40)*2.3,-1,1)
    vivid=np.tanh(np.sqrt((1.5*rg)**2+yb**2)*3)
    sem=pool(np.stack((warm,green,cool*pale,pale,vivid-.35),axis=1),8)
    lighting=pool(y,8)-.5
    det=np.abs(y-blur(y,1))
    detail=pool(np.sqrt(blur(det,1)+.0001),8)-.125
    grit=np.stack((pool(wt,8)-.21,detail*2),axis=1)
    scene=bundle([(.39,sem,.26),(.14,col,.24),(.19,lighting,.14),(.15,grit,.10),(.13,pool(np.stack((wt*cs,wt*sn),1),8),.045)])
    # Soft figure/ground contrast. Compare a pixel to the left and right
    # margins at the same *height*, so the sky/land split has little effect.
    smooth=blur(x,1)
    margin=smooth[:,:,:,:4].mean(3);other=smooth[:,:,:,28:].mean(3)
    def rowblur(t):
        jj=np.arange(32)
        return (.5*t+.25*t[:,:,np.maximum(jj-3,0)]+.25*t[:,:,np.minimum(jj+3,31)])[:,:,:,None]
    def dist(t):
        d=smooth-rowblur(t)
        lum=.3*d[:,0]+.52*d[:,1]+.18*d[:,2]
        return np.sqrt(lum*lum+.7*(d[:,0]-d[:,1])**2+.40*(d[:,1]-d[:,2])**2+.0001)
    f=np.minimum(dist(margin),dist(other));f=f/(f+.125)
    figure=pool(f,4);figure-=figure.mean((1,2),keepdims=True)
    # Directional dark spots and light spots, on a finer sampling grid
    # than the broad tone map (e.g. paired eyes, nostrils or wheel hubs).
    details=pool(blur(y,1)-blur(blur(y,1),3),2)
    fine=pool(wt,2);fine-=fine.mean((1,2),keepdims=True)
    landmark=bundle([(.45,figure,.104),(.37,details,.037),(.18,fine,.067)])
    return shape,scene,landmark
'''

# The renderer uses a camera facing the origin, with +z up. Each object is
# assembled in physical units; framing is done after the camera has been placed.
rng=np.random.default_rng(41075)
u=rng.uniform
pick=lambda seq:seq[rng.integers(len(seq))]
def col(c): return np.asarray(c,dtype=float)
def mul(c,a): return np.clip(col(c)*a,0,1)
black=col((.065,.06,.062)); rubber=col((.10,.095,.095)); white=col((.89,.89,.86)); grey=col((.56,.57,.56)); metal=col((.69,.71,.72)); glass=col((.26,.36,.42))
brown=col((.44,.30,.19)); tan=col((.68,.51,.33)); cream=col((.86,.79,.65)); pink=col((.71,.46,.43)); rust=col((.53,.22,.13))
paintlist=[white,white,metal,grey,black,(.74,.12,.09),(.16,.29,.51),(.12,.19,.25),(.77,.56,.19),(.35,.13,.12),(.18,.33,.25)]

def vec(p): return np.asarray(p,dtype=float)
def ell(s,c,radii,color,axes=None):
    m=np.diag(radii) if axes is None else axes@np.diag(radii)
    s.append(('e',vec(c),m,col(color)))
def segment(s,p,q,width,color,thick=None,over=1.):
    p,q=vec(p),vec(q); v=q-p; length=np.sqrt(v@v); e=v/max(length,.0001)
    a=np.array([-e[2],0.,e[0]]) if abs(e[1])<.93 else np.array([0.,e[2],-e[1]])
    a/=max(np.sqrt(a@a),.0001); b=np.cross(e,a)
    ell(s,(p+q)/2,(length*.5*over,width,width if thick is None else thick),color,np.stack((e,a,b),axis=1))
def poly(s,vertices,color): s.append(('p',vec(vertices),None,col(color)))
def prism(s,xz,width,color,y0=0,shade=1):
    a=[(x,y0-width,z) for x,z in xz]; b=[(x,y0+width,z) for x,z in xz]
    poly(s,a,mul(color,.78));poly(s,b,color)
    for i in range(len(a)):
        j=(i+1)%len(a); dx=xz[j][0]-xz[i][0]; dz=xz[j][1]-xz[i][1]
        # profiles listed counterclockwise in the x,z plane
        bright=1+shade*(-.22*dx+.09*dz)/max(.001,math.hypot(dx,dz))
        poly(s,[a[i],a[j],b[j],b[i]],mul(color,bright))
def box(s,x,y,z,dx,dy,dz,c):
    prism(s,[(x-dx/2,z-dz/2),(x+dx/2,z-dz/2),(x+dx/2,z+dz/2),(x-dx/2,z+dz/2)],dy/2,c,y)
def ringwheel(s,x,y,z,r,w):
    ell(s,(x,y,z),(r,w,r),rubber)
    ell(s,(x,y+math.copysign(w*.73,y),z),(r*.58,w*.42,r*.58),mul(metal,u(.50,1.1)))

def car(s,truck):
    c=col(pick(paintlist)); window=mul(glass,u(.52,1.38)); dark=mul(c,.72)
    if not truck:
        # A lower belt and a glasshouse narrowing towards the roof.
        long=u(1.8,2.25); w=u(.73,.88); sill=.33; belt=u(.78,.94); roof=belt+u(.44,.67)
        front=u(.52,.98); back=u(-1.65,-.95); ft=front-u(.45,.76); bt=back+u(.11,.61)
        prism(s,[(-long,sill+.06),(long*.92,sill),(long,belt-.27),(long*.94,belt-.13),(front,belt), (back,belt),(-long*.98,belt-.10)],w,c)
        bottom=[(back,-w*.98,belt),(front,-w*.98,belt),(front,w*.98,belt),(back,w*.98,belt)]
        top=[(bt,-w*.80,roof),(ft,-w*.80,roof),(ft,w*.80,roof),(bt,w*.80,roof)]
        for i in range(4):
            j=(i+1)%4; poly(s,[bottom[i],bottom[j],top[j],top[i]],mul(window,[.86,1.15,1,.90][i]))
        poly(s,top,mul(c,1.16))
        for i in range(4): segment(s,bottom[i],top[i],.035,c,over=1.03)
        for sign in [-1,1]:
            segment(s,((bt+ft)*.5,w*.80*sign,roof),((bt+ft)*.5+.07,w*.98*sign,belt),.043,black)
            ell(s,(front-.13,sign*w*1.06,belt+.02),(.13,.09,.07),dark)
            for xx in [-.63*long,.64*long]: ringwheel(s,xx,sign*w*.96,.37,.365,.136)
            box(s,long*.957,sign*w*.70,belt-.18,.09,.32,.13,cream)
            box(s,-long*.98,sign*w*.71,belt-.22,.06,.29,.13,(.57,.095,.06))
        box(s,long*.991,0,belt-.28,.035,w*.96,.15,black)
        box(s,long*.973,0,sill+.065,.07,.43,.12,metal)
    else:
        w=u(.72,.9); long=u(2.2,3.9); cargo=col(pick([white,white,metal,c,c,(.68,.22,.12),(.22,.34,.47)])); ht=u(1.6,2.45)
        cab=u(.90,1.48); cx=long-cab*.51; cabbase=.59; cabtop=ht*u(.73,1.02)
        box(s,0,0,.56,2*long,w*1.60,.28,black)
        # Box truck / van, tanker, tipper, or tractor-trailer.
        kind=rng.integers(5); length=2*long-cab-.13; mid=-long+length/2
        if kind==3:
            ell(s,(mid,0,(ht+.76)*.5),(length*.51,w,(ht-.76)*.56),cargo)
        elif kind==4:
            prism(s,[(-long+.18,.87),(cx-cab*.69,.87),(cx-cab*.50,ht*.87),(-long,ht*.92)],w,cargo)
        else:
            box(s,mid,0,(ht+.78)*.5,length,2*w,ht-.78,cargo)
            if rng.random()<.45:
                for side in [-1,1]:box(s,mid,w*side,(ht+.8)*.5,length*.88,.02,u(.07,.27),mul(c,u(.55,1.18)))
        slope=u(.03,.34); nose=long; a=cx-cab*.48; b=nose-slope
        prism(s,[(a,cabbase),(nose,cabbase),(nose,cabtop*.67),(b,cabtop),(a,cabtop)],w*.96,c)
        for side in [-1,1]:
            yy=side*w*.966
            poly(s,[(a+.08,yy,cabtop*.69),(nose-.07,yy,cabtop*.69),(b-.055,yy,cabtop-.09),(a+.08,yy,cabtop-.09)],window)
        poly(s,[(nose-.01,-w*.82,cabtop*.69),(nose-.01,w*.82,cabtop*.69),(b+.03,w*.82,cabtop-.09),(b+.03,-w*.82,cabtop-.09)],mul(window,1.08))
        box(s,nose+.008,0,cabbase+.17,.03,w*1.27,.24,black)
        for side in [-1,1]:
            for xx in [cx+.08,-long+.55,-long+1.23] if long>2.65 else [cx,-long+.57]: ringwheel(s,xx,side*w*.97,.38,.38,.14)
            box(s,nose,side*w*.77,cabbase+.19,.05,.22,.16,cream)

def plane(s):
    airliner=rng.random()<.68
    c=col(pick([white,white,metal,grey,(.32,.40,.47),(.62,.65,.62)])); accent=col(pick(paintlist)); l=u(1.8,2.65); w=u(.15,.26); span=u(1.38,2.40)
    ell(s,(0,0,0),(l,w,w*1.1),c)
    # Vertical fin, swept wings and the smaller tailplane.
    prism(s,[(-l*.94,.0),(-l*.48,.08),(-l*.83,u(.53,.87)),(-l*.96,.74)],w*.12,accent)
    high=0.02 if airliner else u(-.1,.16); sweep=u(.45,1.) if airliner else u(.05,1.25)
    for side in [-1,1]:
        poly(s,[(.48,0,high),(-.42,0,high),(-sweep-.19,span*side,.10),(-sweep+.04,span*side,.12)],mul(c,.87))
        poly(s,[(-l*.57,0,.14),(-l*.90,0,.14),(-l*.99,side*span*.39,.20),(-l*.87,side*span*.39,.22)],mul(c,.90))
        if airliner: ell(s,(.03,side*span*.39,-.14),(.34,.13,.14),mul(c,.77))
        segment(s,(-l*.67,side*w*.915,.035),(l*.68,side*w*.91,.035),.027,mul(accent,.7))
    ell(s,(l*.78,0,w*.48),(l*.12,w*.65,w*.37),glass)

def ship(s):
    l=u(1.65,2.8); w=u(.42,.77); h=u(.50,.95); c=col(pick([white,white,(.22,.25,.26),(.16,.24,.37),rust,grey])); cabin=col(pick([white,white,cream,grey])); kind=rng.integers(5)
    # Sections narrow toward the keel and the bow. Water conceals the keel.
    sections=[(-l,.57,.87),(-l*.75,.96,.94),(l*.42,1,.97),(l*.81,.69,1.01),(l,.03,1.14)]
    for (x1,b1,t1),(x2,b2,t2) in zip(sections,sections[1:]):
        for side in [-1,1]:
            poly(s,[(x1*.92,side*b1*w*.68,0),(x2*.92,side*b2*w*.68,0),(x2,side*b2*w,t2*h),(x1,side*b1*w,t1*h)],mul(c,.89+.07*side+.06*b1))
        poly(s,[(x1,-b1*w,t1*h),(x2,-b2*w,t2*h),(x2,b2*w,t2*h),(x1,b1*w,t1*h)],mul(c,1.19))
    if kind==0:
        mast=2.7*l/2; segment(s,(0,0,h),(0,0,h+mast),.035,grey)
        poly(s,[(-l*.77,.03,h+.18),(-.06,0,h+mast*.98),(-.05,0,h+.31)],cream)
        poly(s,[(l*.84,0,h+.1),(.10,0,h+mast*.89),(.13,0,h+.34)],white)
    else:
        pos=[0,-.3,-.45,.28,.08][kind]*l; le=[1,1.55,.53,.62,.94][kind]*l; levels=3 if kind==1 else int(rng.integers(1,4)); z=h*.93
        for level in range(levels):
            width=w*(1-.12*level)*.9; height=u(.19,.36); box(s,pos,0,z+height*.5,le,2*width,height,cabin)
            for side in [-1,1]:
                box(s,pos,width*side,z+height*.63,le*.88,.017,height*.30,glass)
            z+=height; le*=.82
        if kind!=4: ell(s,(pos-.12,0,z+.15),(.13,.17,.29),mul(accent_ship,.83))
        segment(s,(pos+.15,0,z),(pos+.15,0,z+u(.43,.96)),.027,grey)
        if kind==2:
            for xx in [.02,.38,.72]: box(s,l*xx,0,h+.12,l*.28,w*1.46,.31,pick([rust,grey,(.25,.34,.42)]))
accent_ship=col((.40,.31,.23))

# Articulated animals: x is the direction of travel. Separate camera modes
# for a whole animal, a sitting animal, and a portrait.
def mammal(s,k,portrait):
    cat=k==3; dog=k==5; deer=k==4; horse=k==7; pet=cat or dog
    if cat:
        coat=col(pick([grey,tan,cream,white,brown,black,(.71,.47,.25),(.39,.39,.36)])); fur2=mul(coat,u(.52,.84)); spot=rng.random()<.53
    elif dog:
        coat=col(pick([tan,tan,cream,white,brown,black,(.68,.47,.28),(.36,.36,.34)]));fur2=mul(coat,u(.37,.90));spot=rng.random()<.36
    elif horse:
        coat=col(pick([brown,brown,rust,(.49,.29,.18),(.29,.21,.17),black,tan,grey,white]));fur2=mul(coat,u(.32,.64));spot=False
    else:
        coat=col(pick([brown,tan,(.52,.40,.28),(.60,.45,.31),(.48,.40,.30)]));fur2=mul(coat,.78);spot=False
    bodylen=u(.91,1.21); width=u(.31,.48) if pet else u(.36,.48); depth=u(.42,.57)
    legs=u(.46,.82) if pet else u(.96,1.38); belly=legs+depth*.70; seated=pet and rng.random()<.34
    if seated: legs*=.69;belly=legs+.39
    neck=(.78,0,belly+.21); headx=1.18; headz=belly+.52
    if horse or deer:
        headx=u(1.18,1.60); headz=belly+u(.78,1.24)
        if rng.random()<.24 and not portrait:headx=1.73;headz=belly-.59
    if seated: headx=.56;headz=belly+.86
    if not portrait:
        if seated: segment(s,(-.48,0,legs+.20),(.49,0,belly+.52),depth,coat,width,1.50)
        else:ell(s,(0,0,belly),(bodylen,width,depth),coat)
        for side in [-1,1]:
            for front in [-1,1]:
                start=front*bodylen*.66; yy=side*width*.64; swing=u(-.34,.34)
                if seated and front==1:start=.50
                knee=(start+(.12 if front==-1 else -.05)+swing*.40,yy,legs*.57)
                foot=(start+swing,yy+side*.05,.11); thigh=.17 if pet or horse else .12; shin=.10 if pet or horse else .063
                segment(s,(start,yy,belly),knee,thigh,coat,over=1.18);segment(s,knee,foot,shin,coat,over=1.18)
                ell(s,(foot[0]+.035,foot[1],.10),(.125 if not dog else .15,shin,.10),fur2 if not pet else coat)
        tailbase=(-bodylen*.86,0,belly+.13)
        if horse:segment(s,tailbase,(-bodylen*1.16,.05,legs*.25),.16,fur2,over=1.03)
        elif deer:ell(s,(-bodylen*.9,0,belly+.03),(.15,.18,.23),cream)
        else:
            high=u(-.80,1.10);point=(-bodylen*1.38,.04,belly+.18+high*.35)
            segment(s,tailbase,point,.13,coat,over=1.14); segment(s,point,(-bodylen*1.71,.16,belly+.20+high),.10,fur2,over=1.12)
        if spot:
            for a in range(6 if cat else 3):
                xx=u(-.72,.72); ry=width*math.sqrt(max(.02,1-(xx/bodylen)**2))
                for side in [-1,1]:ell(s,(xx,side*ry*.85,belly+.09),(.075 if cat else .22,.09,.26),fur2)
    elif pet:
        # Shoulders visible below and behind the head in a portrait.
        ell(s,(headx-.27,0,headz-.55),(.46,.46,.41),coat)
    segment(s,neck if not portrait else (headx-.31,0,headz-.38),(headx-.04,0,headz),.29 if pet else .32,coat,over=1.25)
    if horse:
        p=(headx-.18,0,headz+.03);q=neck if not portrait else (headx-.45,0,headz-.31)
        segment(s,(q[0]-.14,0,q[2]+.14),p,.14,fur2,.12,1.07)
    hx,hy,hz=(.38,.41,.40) if cat else ((.40,.37,.42) if dog else (.34,.28,.34))
    if dog:hy*=u(.81,1.25)
    ell(s,(headx,0,headz),(hx,hy,hz),coat)
    for side in [-1,1]:
        if cat or (dog and rng.random()<.28):
            tip=(headx-.16,side*hy*.89,headz+hz*u(1.32,1.68));front=(headx+.15,side*hy*.43,headz+.19);edge=(headx+.03,side*hy*1.02,headz+.14)
            poly(s,[tip,front,edge],mul(coat,.94));poly(s,[tip,(headx-.28,side*hy*.62,headz+.24),edge],fur2)
            t,f,e=vec(tip),vec(front),vec(edge);mid=(t+f+e)/3
            poly(s,[mid+.66*(a-mid)+vec((.012,0,0)) for a in (t,f,e)],mul(pink,.91))
        elif dog:
            segment(s,(headx-.09,side*hy*.73,headz+.29),(headx,side*(hy+.067),headz-u(.14,.42)),u(.13,.21),fur2,.11,1.16)
        else:
            segment(s,(headx-.12,side*.14,headz+.18),(headx-.22,side*(.47 if deer else .29),headz+(.57 if deer else .64)),.10 if horse else .12,coat,.078,1.09)
    muzzle=(headx+(.44 if horse else .39),0,headz-(.47 if horse else .20))
    if cat:
        for side in [-1,1]:ell(s,(headx+.29,side*.105,headz-.12),(.15,.145,.115),mul(coat,1.22))
        muzzle=(headx+.411,0,headz-.067)
        ell(s,muzzle,(.046,.074,.048),pick([pink,black,brown]));
    else:
        segment(s,(headx+.12,0,headz+.031),muzzle,.21 if horse else .18,mul(coat,.96),.225 if dog else .187,1.38)
        ell(s,muzzle,(.15,.211,.154) if horse else (.098,.139,.087),mul(coat,.64) if horse else black)
    for side in [-1,1]:
        xx=headx+hx*(.75 if pet else .42); yy=side*hy*(.62 if pet else .89); zz=headz+.087
        if spot and dog:ell(s,(xx-.037,yy*.96,zz),(.12,.12,.15),fur2)
        ell(s,(xx,yy,zz),(.074 if pet else .047,.078 if pet else .053,.067 if pet else .052),black)
        if cat:
            ell(s,(xx+.023,yy+side*.02,zz),(.060,.059,.049),pick([(.62,.65,.32),(.57,.64,.54),(.71,.60,.31)]))
            ell(s,(xx+.066,yy+side*.033,zz),(.018,.024,.044),black)
            for t in range(3):
                segment(s,(headx+.196,side*hy*.73,headz+.16+t*.055),(headx+.35,side*hy*.27,headz+.20+t*.045),.022,fur2)
        else:ell(s,(xx+.044,yy+side*.032,zz+.017),(.014,.018,.014),cream)
    if deer and rng.random()<.37:
        for side in [-1,1]:
            root=(headx-.10,side*.11,headz+.24); joint=(headx-.27,side*.36,headz+.64); tip=(headx-.48,side*.55,headz+.98)
            segment(s,root,joint,.044,tan,over=1.1);segment(s,joint,tip,.035,tan,over=1.09)
            for f in [.20,.68]:
                v=vec(joint)*(1-f)+vec(tip)*f;segment(s,v,v+vec((.23,side*.04,.26)),.028,cream,over=1.05)
    if horse and not portrait and rng.random()<.30:
        box(s,-.05,0,belly+depth,.54,width*1.55,.12,fur2);segment(s,(-.05,0,belly+.39),(-.09,0,belly+1.14),.195,pick(paintlist),over=1.1)
        ell(s,(-.09,0,belly+1.30),(.15,.15,.19),tan);ell(s,(-.09,0,belly+1.42),(.17,.16,.12),black)
        for side in [-1,1]:segment(s,(-.09,.09*side,belly+.46),(.12,side*width,belly-.13),.099,black,over=1.2)

def bird(s,flying):
    body=col(pick([white,cream,grey,brown,tan,black,(.28,.34,.42),(.70,.63,.22),(.38,.51,.29),(.35,.55,.64)]));wing=mul(body,u(.47,.83));cap=col(pick([body,body,wing,black,grey])); beak=col(pick([black,tan,(.78,.60,.22)]))
    if flying:
        ell(s,(0,0,0),(.90,.24,.24),body);ell(s,(.67,0,.07),(.30,.22,.20),cap);span=u(1.45,2.5);up=u(-.8,1.15)
        for side in [-1,1]:
            v=[(.42,0,.07),(.31,side*span*.57,.17+up*.60),(-.24,side*span,up),(-.46,side*span*.61,up*.60),(-.58,side*.18,.04)];poly(s,v,wing)
        poly(s,[(-.53,-.08,.05),(-1.29,-.41,.09),(-1.32,.41,.07),(-.51,.08,.05)],wing);segment(s,(.87,0,.06),(1.18,0,.035),.066,beak)
    else:
        tilt=u(.2,.9);segment(s,(-.50,0,.76-tilt*.35),(.36,0,.90+tilt*.52),u(.37,.49),body,.38,1.67)
        hz=1.11+tilt*.61; hx=.43;rad=u(.25,.38);ell(s,(hx,0,hz),(rad,rad*.93,rad),cap)
        for side in [-1,1]:
            segment(s,(.23,side*.28,1.04+tilt*.30),(-.57,side*.245,.71-tilt*.35),.255,wing,.15,1.42)
            leg=(.15,side*.15,.39);foot=(.06,side*.15,.05);segment(s,leg,foot,.035,beak,over=1.10);segment(s,(-.04,side*.15,.04),(.22,side*.15,.04),.035,beak)
            ell(s,(hx+.10,side*rad*.86,hz+.044),(.050,.050,.057),black)
        segment(s,(-.42,0,.82-tilt*.24),(-1.14,0,.57-tilt*.50),.097,wing,.18,1.06)
        ell(s,(hx+rad*.88,0,hz-.032),(rad*.70,.105,.087),beak)
        if rng.random()<.31: # a perch is a useful part of the composition
            segment(s,(-1.22,-.38,-.11),(1.12,.27,-.03),.092,brown)

def frog(s):
    skin=col(pick([(.38,.47,.17),(.52,.55,.24),(.28,.41,.18),(.42,.38,.24),(.52,.40,.25),(.21,.27,.16),tan,grey]));mark=mul(skin,u(.42,.72))
    ell(s,(-.13,0,.48),(.92,.56,.43),skin);ell(s,(.53,0,.58),(.60,.60,.32),skin);ell(s,(.78,0,.48),(.43,.49,.18),mul(skin,1.19))
    for side in [-1,1]:
        yy=side*.39;ell(s,(.65,yy,.80),(.216,.167,.202),skin);ell(s,(.72,yy*1.13,.848),(.13,.09,.112),mul(tan,.91));ell(s,(.757,yy*1.23,.858),(.092,.046,.060),black)
        segment(s,(-.83,side*.48,.27),(-.095,side*.64,.46),.21,skin,.20,1.43);segment(s,(-.78,side*.61,.25),(-.21,side*.82,.10),.095,skin,over=1.24)
        segment(s,(.39,side*.39,.53),(.31,side*.70,.19),.109,skin,over=1.17);segment(s,(.31,side*.70,.19),(.73,side*.77,.08),.068,skin,over=1.16)
        for f in [-1,0,1]:segment(s,(.68,side*.77,.084),(.91,side*.77+.10*f,.037),.024,skin)
    if rng.random()<.76:
        for i in range(12):
            a=u(-.8,.75);b=u(-.87,.87);z=math.sqrt(max(0,1-a*a-b*b));ell(s,(-.13+.92*a,.56*b,.48+.426*z),(.085,.069,.034),mark)

sidepix=40
gy,gx=np.mgrid[:sidepix,:sidepix]+.5
sky=[(.46,.66,.85),(.28,.52,.78),(.66,.80,.89),(.73,.77,.78),(.77,.79,.77),(.90,.89,.86)]
water=[(.23,.39,.51),(.34,.50,.60),(.19,.30,.40),(.44,.54,.56),(.30,.39,.36),(.43,.54,.65)]
plants=[(.25,.34,.18),(.38,.45,.22),(.26,.32,.17),(.41,.43,.29),(.41,.47,.21),(.48,.47,.31),(.53,.49,.37),(.49,.54,.32)]
soil=[(.41,.34,.24),(.48,.43,.32),(.28,.27,.20),(.57,.51,.38),(.45,.41,.34)]
room=[(.71,.63,.51),(.66,.64,.56),(.40,.31,.27),(.67,.58,.50),(.42,.42,.39),(.78,.78,.74),(.46,.39,.34),(.52,.53,.54),(.39,.43,.36)]
road=[(.40,.40,.39),(.48,.49,.49),(.35,.36,.37),(.56,.55,.52),(.43,.44,.46),(.49,.46,.40)]

def noise(nx,ny):
    field=rng.normal(size=(ny+1,nx+1));px=np.arange(sidepix)*nx/sidepix;py=np.arange(sidepix)*ny/sidepix
    ix=px.astype(int);iy=py.astype(int);fx=px-ix;fy=py-iy;fx=fx*fx*(3-2*fx);fy=fy*fy*(3-2*fy)
    a=(1-fx)*field[:,ix]+fx*field[:,ix+1];return ((1-fy[:,None])*a[iy]+fy[:,None]*a[iy+1])[:,:,None]

def material(c,kind):
    base=col(c)[None,None,:]
    if kind=='sky':
        return base+.070*noise(3,3)+.035*noise(8,5)+.045*(gy/sidepix)[:,:,None]
    if kind=='water':
        return base+.081*noise(5,20)+.070*noise(2,10)+.04*noise(10,38)
    if kind=='road':return base+.045*noise(2,3)+.018*noise(24,26)
    broad=noise(3,4); medium=noise(10,9); fine=noise(24,27)
    if kind=='leaves':return base+.138*broad+.098*medium+.066*fine+.040*noise(7,7)*[.95,.62,-.24]
    if kind=='earth':return base+.10*broad+.087*medium+.074*fine+.038*noise(8,8)*[.91,.16,-.20]
    return base+.105*broad+.053*medium+.024*fine+.039*noise(3,4)*[1,.31,-.53]

def background(k,bottom,flying):
    # Only broad scene arrangements. Each draw has its own illumination,
    # materials, clutter and horizon; no two have their features combined.
    p=rng.random();h=u(.17,.58)*sidepix
    if k==0:
        top=material(pick(sky),'sky');bot=top
        if p<.37:
            h=u(.47,.81)*sidepix;bot=material(pick(road+plants),'road' if p<.20 else 'leaves'); h=h+noise(6,1)[:,:,0]*1.9
    elif k==8:
        top=material(pick(sky),'sky');bot=material(pick(water),'water');h=min(bottom-1,h+u(-.18,.10)*sidepix)
        if p<.35:
            d=h-noise(6,1)[:,:,0]*2-2.6;top=np.where((gy>d)[:,:,None],material(pick(plants+room),'leaves'),top)
    elif k in (1,9):
        top=material(pick(sky),'sky');bot=material(pick(road),'road');h=u(.44,.72)*sidepix
        if p<.85:
            d=u(-.3,.50)*sidepix+noise(8,1)[:,:,0]*3.3
            urban=p<.49;trees=material(pick(room if urban else plants),'room' if urban else 'leaves')
            if urban:
                for i in range(4):
                    start=u(-5,40);width=u(1.1,10);trees+=((gx>start)&(gx<start+width))[:,:,None]*u(-.12,.12)
            top=np.where((gy>d)[:,:,None],trees,top)
    elif k==2:
        if (flying and p<.65) or p<.095:top=bot=material(pick(sky),'sky')
        elif p>.86:top=bot=material(pick(water),'water')
        else:top=bot=material(pick(plants+soil),'leaves')
    elif k in (4,7):
        top=material(pick(plants+soil),'leaves');bot=material(pick(plants+soil[:2]),'leaves');h=u(.30,.66)*sidepix
        if p<(.51 if k==7 else .30):
            d=u(.09,.49)*sidepix+noise(4,1)[:,:,0]*3.2;top=np.where((gy<d)[:,:,None],material(pick(sky),'sky'),top)
    elif k==6:
        top=bot=material(pick(plants+soil+soil),'earth' if p<.65 else 'leaves')
        if p>.87:top=bot=material(pick(water+plants),'water')
    elif p<(.77 if k==3 else .46):
        top=material(pick(room),'room');bot=material(pick(room+soil[:2]),'room');h=u(.21,.87)*sidepix
    else:
        top=material(pick(plants),'leaves');bot=material(pick(plants+soil),'earth');h=u(.25,.80)*sidepix
    if p>.966:top=bot=material(pick([grey,white,cream]+room),'room')
    return np.where((gy>h)[:,:,None],bot,top)

def project(s,k,portrait,flying):
    angle=u(-math.pi,math.pi);elev=u(.02,.29);roll=u(-.065,.065)
    if portrait:angle=u(-1.0,1.0) if (k in (3,5) or rng.random()<.60) else u(-1.6,1.6);elev=u(-.06,.35);roll=u(-.22,.22)
    elif k==0 or flying:elev=u(-1.34,.71);roll=u(-.57,.57) if k==0 else u(-.33,.33)
    elif k==6:elev=u(.09,1.18);roll=u(-.32,.32)
    elif k==2:elev=u(-.37,.50);roll=u(-.23,.23)
    elif k in (3,5):elev=u(.07,.80);roll=u(-.20,.20)
    ca,sa=math.cos(angle),math.sin(angle);ce,se=math.cos(elev),math.sin(elev)
    right=vec((-sa,ca,0));up=vec((-se*ca,-se*sa,ce));view=vec((ce*ca,ce*sa,se))
    cr,sr=math.cos(roll),math.sin(roll);camera=np.stack((right*cr+up*sr,right*sr-up*cr,view))
    lo=np.full(2,100.);hi=-lo;items=[]
    for kind,pts,m,c in s:
        pts=pts@camera.T
        if kind=='e':
            m=camera@m;r=np.sqrt((m[:2]**2).sum(1));a=pts[:2]-r;b=pts[:2]+r
        else:a=pts[:,:2].min(0);b=pts[:,:2].max(0)
        lo=np.minimum(lo,a);hi=np.maximum(hi,b);items.append((kind,pts,m,c))
    fill=u(.63,1.13)
    if portrait: fill=u(.89,1.68)
    elif k in (0,2,8):fill=u(.43,1.08)
    elif k==6:fill=u(.64,1.28)
    scale=sidepix*fill/max(hi-lo);offset=sidepix*np.array([u(.443,.557),u(.433,.566)])-scale*(lo+hi)*.5
    pix=background(k,(hi[1]*scale+offset[1]),flying);depth=np.full((sidepix,sidepix),-1e6)
    if k in (1,9):
        cy=hi[1]*scale+offset[1]-.54; shadow=np.exp(-((gx-(offset[0]+scale*(lo[0]+hi[0])/2))/(.49*(hi[0]-lo[0])*scale+1))**4-((gy-cy)/(1.1+.04*sidepix))**2);pix*=1-.38*shadow[:,:,None]
    for kind,p,m,c in items:
        if kind=='e':
            cx,cy=p[:2]*scale+offset;mat=m@m.T
            xx,xy,yy=mat[0,0],mat[0,1],mat[1,1];det=xx*yy-xy*xy
            if det<1e-14:continue
            rx=math.sqrt(xx)*scale;ry=math.sqrt(yy)*scale
            x0=max(0,int(cx-rx));x1=min(sidepix,int(cx+rx)+1);y0=max(0,int(cy-ry));y1=min(sidepix,int(cy+ry)+1)
            if x1<=x0 or y1<=y0:continue
            sl=np.s_[y0:y1,x0:x1];dx=(gx[sl]-cx)/scale;dy=(gy[sl]-cy)/scale
            ax=(yy*dx-xy*dy)/det;ay=(xx*dy-xy*dx)/det;disc=1-ax*dx-ay*dy
            root=np.sqrt(np.maximum(disc,0));zx,zy=mat[2,:2];squared=max(0,mat[2,2]-(zx*zx*yy+zy*zy*xx-2*xy*zx*zy)/det)
            z=p[2]+zx*ax+zy*ay+math.sqrt(squared)*root;keep=(disc>0)&(z>depth[sl]);shade=.77+.23*root-.14*dx/(rx/scale)-.17*dy/(ry/scale)
            pix[sl]=np.where(keep[:,:,None],shade[:,:,None]*c,pix[sl]);depth[sl]=np.where(keep,z,depth[sl])
        else:
            pp=p[:,:2]*scale+offset
            # All faces are convex (or nearly so); triangles also carry depth.
            for i in range(1,len(p)-1):
                xy=pp[[0,i,i+1]];mini=xy.min(0);maxi=xy.max(0);x0=max(0,int(mini[0]));y0=max(0,int(mini[1]));x1=min(sidepix,int(maxi[0])+1);y1=min(sidepix,int(maxi[1])+1)
                if x1<=x0 or y1<=y0:continue
                sl=np.s_[y0:y1,x0:x1];ab=xy[1]-xy[0];ac=xy[2]-xy[0];det=ab[0]*ac[1]-ab[1]*ac[0]
                if abs(det)<1e-8:continue
                dx=gx[sl]-xy[0,0];dy=gy[sl]-xy[0,1];v=(ac[1]*dx-ac[0]*dy)/det;w=(ab[0]*dy-ab[1]*dx)/det
                z=p[0,2]+v*(p[i,2]-p[0,2])+w*(p[i+1,2]-p[0,2]);keep=(v>=0)&(w>=0)&(v+w<=1)&(z>depth[sl]-.000001)
                pix[sl]=np.where(keep[:,:,None],c,pix[sl]);depth[sl]=np.where(keep,z,depth[sl])
    if k in (2,3,4,5,6,7):
        fuzz=noise(29,29)*u(.020,.063)+noise(10,11)*u(.031,.094);pix+=np.where((depth>-100)[:,:,None],fuzz,0)
    # Linear interpolation at output pixel centres, mild sensor blur & exposure.
    t=(np.arange(32)+.5)*sidepix/32-.5;ix=t.astype(int);a=t-ix
    pix=(1-a[:,None])*pix[:,ix,:]+a[:,None]*pix[:,ix+1,:];pix=(1-a[:,None,None])*pix[ix]+a[:,None,None]*pix[ix+1]
    pix=(np.clip(pix,0,1)**u(.83,1.23))*u(.83,1.13)*rng.uniform(.936,1.064,size=3)+u(-.043,.063)
    image=np.clip(pix.transpose(2,0,1),0,1).astype(np.float32)
    mix=u(.08,.56);return image*(1-mix)+blur(image,1)*mix

def make(k):
    s=[];portrait=False;flying=False
    if k==0:plane(s)
    elif k in (1,9):car(s,k==9)
    elif k==8:ship(s)
    elif k==6:frog(s)
    elif k==2:flying=rng.random()<.22;bird(s,flying)
    else:portrait=rng.random()<{3:.48,4:.15,5:.47,7:.12}[k];mammal(s,k,portrait)
    return project(s,k,portrait,flying)

# Build uses the identical descriptor code, with no labels or data loaded.
exec(features)

def buildbank():
    bank={};perclass=1536
    for k in range(10):
        shapes=[];scenes=[];parts=[]
        for i in range(0,perclass,32):
            ims=np.stack([make(k) for j in range(32)])
            desc,context,locations=describe(ims)
            shapes.append(desc);scenes.append(context);parts.append(locations)
        bank['shape'+str(k)]=np.ascontiguousarray(np.concatenate(shapes))
        bank['scene'+str(k)]=np.ascontiguousarray(np.concatenate(scenes))
        bank['landmark'+str(k)]=np.ascontiguousarray(np.concatenate(parts))
    return bank

rules = '''
# Smooth votes in [0,1]. A scene cue is evidence, not a precondition.
def ramp(x,lo,hi):
    return np.clip((x-lo)/(hi-lo),0,1)

def av(a):
    return a.mean(axis=(-1,-2))

def cues(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=.30*r+.52*g+.18*b
    l=blur(y,1); tiny=np.abs(y-l); contrast=y.std(axis=(1,2))
    rg=(r-g)/(.23+y); gb=(g-b)/(.23+y); rb=(r-b)/(.23+y)
    sat=(x.max(1)-x.min(1))/(.20+y)
    # Distinguish a blue hue from the cool tint of a grey road in shade.
    blue=ramp(-rb,-.025,.22)*ramp(-gb,-.025,.12)
    leafy=ramp(-rg,-.033,.113)*ramp(gb,.031,.225)
    brownish=ramp(rg,.006,.18)*ramp(rb,.072,.35)*(1-.70*ramp(rg,.33,.75))
    neutral=1-ramp(sat,.065,.27)
    gx,gy=grad(blur(x,1)); xx=(gx*gx).mean(1); yy=(gy*gy).mean(1); xy=(gx*gy).mean(1)
    m=np.sqrt(xx+yy+.000001); cs=(yy-xx)/(xx+yy+.00003);sn=2*xy/(xx+yy+.00003)
    dx=np.diff(y,axis=2);dy=np.diff(y,axis=1)
    h=(av(m*cs)/(av(m)+.009)); axial=av(m*(cs**2-sn**2))/(av(m)+.008)
    row=av(np.abs(gy.mean(1).mean(2,keepdims=True)))/(av(m)+.012)
    vertical=av(np.abs(gx.mean(1).mean(1,keepdims=True)))/(av(m)+.012)
    rough=blur(tiny,2); flat=1-ramp(rough,.012,.063)
    sky=flat*(blue*.91+(1-blue)*neutral*ramp(y,.56,.89)*.77)
    top=av(sky[:,:10]);bot=av(sky[:,23:]);blow=av(blue[:,22:]);bleft=av(blue[:,:,:6]);bright=av(y[:,:8])-av(y[:,24:]);brightlow=av(y[:,23:])
    leaf=av(leafy);leaflow=av(leafy[:,20:]); leafyedge=(av(leafy[:,:,:5])+av(leafy[:,:,27:]))*.5
    dirt=av(brownish); warm=av(brownish[:,6:28,4:28]); cold=av(blue); satc=av(sat[:,7:25,5:27]); greenmid=av(leafy[:,7:25,5:27]); coldmid=av(blue[:,8:25,5:27]); neut=av(neutral)
    road=av((flat*neutral*(1-.65*ramp(y,.76,.94)))[:,23:])
    ht=ramp(h,-.10,.42);axisvote=ramp(axial,-.07,.39); rows=ramp(row,.11,.45);vert=ramp(vertical,.12,.30)
    grain=ramp(av(tiny),.023,.062);fine=ramp(av(tiny)/(av(np.abs(y-blur(l,3)))+.008),.30,.69)
    corners=np.concatenate((y[:,:6,:7],y[:,:6,-7:],y[:,-6:,:7],y[:,-6:,-7:]),axis=2)
    plain=ramp(.18-corners.std((1,2)),0,.12)*ramp(.040-av(tiny[:,:7]),0,.030)
    skyall=av(sky); skybottom=bot;ground=ramp(road,.22,.74);fol=ramp(leaf,.095,.51);folbottom=ramp(leaflow,.12,.60);folside=ramp(leafyedge,.10,.58)
    sea=ramp(blow,.14,.75); canopy=ramp(top,.22,.76); manyblue=ramp(cold,.18,.75)
    fur=ramp(warm,.10,.61);midgreen=ramp(greenmid,.12,.62);earth=ramp(dirt,.12,.58)
    # Lower-edge shadow, common under road vehicles; a boat has more
    # horizontal structure in the *background* as well, often blue or grey.
    profile=y[:,:,4:28].mean(2); yprof=profile[:,14:29];shadow=av(y[:,26:])-yprof.min(1)
    under=ramp(shadow/(contrast+.05),.15,1.05)
    bodycolor=ramp(np.maximum(0,av(sat[:,8:25])-av(sat[:,24:])),.018,.22)
    topplain=ramp(av(flat[:,:9]),.60,.94);skywrap=canopy*ramp((bot+skyall)*.5,.18,.68)
    coolwater=sea*(.55+.45*rows);ripple=ramp(av((m*cs)[:,22:])/(av(m[:,22:])+.014),.04,.55)
    wild=.50*folside+.30*folbottom+.20*earth
    mech=.37*axisvote+.40*rows+.23*ht
    # Width of concentration of edge energy. Sky & water can leave
    # the subject in a thin band. Portraits have a taller central mass.
    band=av(m[:,9:23])/(av(m)+.01); middle=ramp(band,.98,1.55)
    shoulder=ramp(av(m[:,3:13])/(av(m[:,14:24])+.01),.48,1.03)
    lowtex=ramp(av(tiny[:,22:])/(av(tiny[:,8:24])+.007),.48,1.14)
    smoothbg=.5*topplain+.5*ramp(av(flat[:,23:]),.53,.93)
    # Indoor/adaptable animal appearances: near-neutral, reddish or cream
    # materials, no long horizon, often considerable fine-scale detail.
    homelike=(1-.65*fol)*(1-.68*manyblue)*(1-.53*canopy)
    furtex=grain*.45+fine*.35+(1-axisvote)*.20
    animal=(1-mech)*.63+.37*(1-rows)
    q=np.zeros((len(x),10),np.float32)
    q[:,0]=1.24*skywrap+.61*manyblue+.39*plain+.35*topplain+.36*middle+.18*smoothbg-.48*wild-.40*earth-.29*ripple-.20*grain-.76
    q[:,1]=.92*ground+.61*under+.48*middle+.55*mech+.41*bodycolor+.13*neut-.52*sea-.34*fol-.34*earth-.22*shoulder-.84
    q[:,2]=.50*animal+.40*wild+.28*midgreen+.21*plain+.32*smoothbg+.20*manyblue+.16*skywrap-.24*mech-.67
    q[:,3]=.88*homelike+.59*furtex+.32*shoulder+.36*animal+.16*fur-.42*folside-.22*midgreen-.20*ground-.95
    q[:,4]=.94*wild+.56*fur+.34*animal+.22*fine+.27*folbottom-.30*manyblue-.19*canopy-.15*ground-.75
    q[:,5]=.63*homelike+.44*animal+.29*fur+.20*shoulder+.20*furtex+.15*folbottom-.28*manyblue-.19*midgreen-.76
    q[:,6]=.84*midgreen+.48*earth+.51*grain+.41*lowtex+.29*fine+.28*animal-.65*canopy-.28*ground-.31*manyblue-.86
    q[:,7]=.52*folbottom+.40*fur+.29*canopy+.29*shoulder+.28*wild+.13*axisvote-.41*sea-.19*midgreen-.56
    q[:,8]=1.08*coolwater+.74*rows+.58*ripple+.48*canopy+.28*ht+.30*ramp(bright,.065,.40)-.44*wild-.36*under-.27*vert-.83
    q[:,9]=.86*ground+.70*axisvote+.44*rows+.40*under+.30*vert+.31*canopy+.21*shoulder-.39*sea-.32*fol-.32*earth-.99
    return q

# Match full-image views. Both reflected views and small offsets are tried.
# Scores compare one input to individual templates (never to a class mean).
def search(x):
    a,b,c=describe(x)
    n=len(x);z=np.empty((n,10,3),np.float32)
    for k in range(10):
        sh=a@CONSTANTS['shape'+str(k)].T
        sc=b@CONSTANTS['scene'+str(k)].T
        mark=c@CONSTANTS['landmark'+str(k)].T
        sh=.76*sh+.24*mark
        joint=.72*sh+.28*sc
        # Max for rare poses, and a few supporting views for stability.
        v=np.partition(joint,joint.shape[1]-6,axis=1)[:,-6:]
        z[:,k,0]=.45*v.max(axis=1)+.55*v.mean(axis=1)
        u=np.partition(sh,sh.shape[1]-4,axis=1)[:,-4:]
        z[:,k,1]=.6*u.max(axis=1)+.4*u.mean(axis=1)
        w=np.partition(sc,sc.shape[1]-8,axis=1)[:,-8:]
        z[:,k,2]=w.mean(axis=1)
    return z

def classify(x):
    # The camera models already sample object size and orientation. The four
    # off-centre reads supply some finer translational tolerance.
    j=np.arange(32); scores=[]
    for dx,dy,mirror in [(0,0,False),(0,0,True),(-2,-2,False),(2,-2,True),(2,2,False),(-2,2,True)]:
        v=x if not mirror else x[:,:,:,::-1]
        if dx!=0:
            v=v.take(np.clip(j+dy,0,31),axis=2).take(np.clip(j+dx,0,31),axis=3)
        scores.append(search(v))
    stack=np.stack(scores)
    best=.74*stack.max(axis=0)+.26*stack.mean(axis=0)
    return (12.0*best[:,:,0]+3.8*best[:,:,1]+3.0*best[:,:,2]+cues(x)).argmax(axis=1)

def predict(x):
    # Fixed internal batch shape keeps kernel selection independent of the
    # caller's batch size. Each output uses only that row's features.
    ans=np.empty(len(x),np.int64)
    for i in range(0,len(x),48):
        n=min(48,len(x)-i)
        xx=np.zeros((48,3,32,32),np.float32);xx[:n]=x[i:i+n]
        ans[i:i+n]=classify(xx)[:n]
    return ans
'''

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(features+rules)
    np.savez(out/'constants.npz',**buildbank())
