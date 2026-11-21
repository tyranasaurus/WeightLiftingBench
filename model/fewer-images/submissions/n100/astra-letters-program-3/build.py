"""Compile an anatomical model of each sign into individually rendered view templates.
No image files, data files, or other program sources are inputs to this build.
"""
import sys
from pathlib import Path
import numpy as np
import math

# Each pose is specified in palm widths, not image coordinates. x runs from
# little finger to index, y from the wrist to the knuckles, z out of the palm.
# Phalange lengths are proximal, middle, distal. Angles are joint flexions.
LETS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
LENGTH = np.array([[.32,.195,.17],[.426,.273,.20],[.465,.292,.215],[.42,.258,.195]])
RAD = np.array([.086,.105,.115,.108])
R = 56

def rx(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def ry(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def roll(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])
rad = np.deg2rad

def build_pose(letter, rng):
    u=rng.uniform
    arm=[]
    palm=[]
    digits=[]
    pads=[]
    # A swept ellipsoid with linearly varying radius.
    def bone(lst,a,b,axis,r1=1.,r2=1.):
        lst.append((np.array(a),np.array(b),np.array(axis),r1,r2))
    width=u(.93,1.08); palmheight=u(.92,1.07); flesh=u(.94,1.12)
    deviation=u(-.20,.15)
    bone(arm,[deviation*3,-3.,-.085],[0,.13,-.025],(.34*width,.28,.151),1.52,u(.85,.96))
    bone(palm,[0,.19,-.015],[.005,.79,0],(.51*width,.24,.163*flesh),.73,1.)
    bone(pads,[.13,.22,.057],[.285,.54,.095],(.206*width,.245,.14*flesh),.93,1.)
    bone(pads,[-.19,.25,.015],[-.32,.64,.06],(.142,.26,.11*flesh),1.,1.)
    xbase=np.array([-.353,-.152,.071,.287])*width
    ybase=np.array([.825,.943,.986,.955])
    zbase=np.array([.04,.005,-.009,.025])
    base=np.stack([xbase,ybase,zbase],axis=1)
    f=np.tile(np.array([76.,91.,55.]),(4,1))
    sides=np.array([8.,4.,0.,-5.])
    straight=[3.,4.,3.]
    # View angles: yaw about the finger axis, camera elevation, then in-plane tilt.
    yaw=u(-34,30); pitch=u(-18,18); tilt=u(-15,22)
    thumb=np.array([[.23,.21,.065],[.47,.46,.12],[.40,.69,.28],[.20,.76,.42]])
    align=None
    def extend(ids,spread):
        for j,s in zip(ids,spread): f[j]=straight; sides[j]=s
    if letter=='A':
        f[:]=[80,102,58]
        thumb[1:]=[[.49,.44,.09],[.515,.72,.23],[.47,.98,.33]]
        yaw=u(-48,18); tilt=u(-5,33)
    elif letter=='B':
        extend(range(4),[4,3,1,-1]); f[:,0]+=u(-5,6)
        thumb[1:]=[[.43,.46,.17],[.27,.66,.23],[-.022,.73,.23]]
        yaw=u(-32,32)
    elif letter in 'CO':
        # C and O form an arch in the sagittal plane, with opposite thumb tip.
        closed=letter=='O'
        c=u(40,62) if closed else u(25,47)
        f[:]=[c,u(46,69) if closed else u(28,52),u(26,48) if closed else u(15,38)]
        f[:,0]+=np.array([8,4,2,0]); sides[:]=[8,4,0,-5]
        yaw=u(51,111);pitch=u(-18,18);tilt=u(-4,30)
        thumb[1:]=[[.46,.40,.22],[.52,.54,.49],[.46,.68,.74]]
    elif letter=='D':
        extend([3],[0]); f[:3]=[55,65,40]; sides[:3]=[19,12,6]
        thumb[1:]=[[.46,.48,.18],[.32,.68,.38],[.12,.87,.56]]
        yaw=u(-48,24)
    elif letter=='E':
        f[:]=[u(36,66),u(93,120),u(15,49)]
        f[0,0]+=7; sides[:]=[6,2,0,-5]
        thumb[1:]=[[.45,.43,.20],[.23,.59,.40],[-.10,.66,.48]]
    elif letter=='F':
        extend([0,1,2],[u(-32,-13),u(-13,2),u(2,19)])
        f[3]=[u(45,69),u(46,66),u(22,45)]
        sides[3]=u(9,20); yaw=u(15,83);pitch=u(-15,22); tilt=u(-10,20)
        thumb[1:]=[[.51,.45,.22],[.61,.73,.48],[.55,.97,.63]]
    elif letter in 'GHT':
        extend([3],[1]); f[3]=[u(3,53),u(-6,4),u(-6,5)]
        yaw=u(105,174);pitch=u(-32,27)
        align=(3,u(64,97) if letter!='T' else u(78,105))
        if letter=='H':
            f[2]=f[3]+np.array([u(-5,10),2,0]); sides[2]=u(-4,3)
            thumb[1:]=[[.41,.45,.14],[.30,.63,.32],[.09,.71,.46]]
        elif letter=='T':
            # Flex the index out from the palm; the thumb sticks up across its base.
            f[3]=[u(60,96),u(-6,7),u(-5,9)]
            thumb[1:]=[[.47,.48,.18],[.49,.79,.23],[.50,1.09,.26]]
            yaw=u(48,113);pitch=u(-24,18)
        else:
            thumb[1:]=[[.50,.50,.21],[.48,.83,.29],[.43,1.12,.36]]
    elif letter=='I' or letter=='Y':
        extend([0],[u(-25,-7)])
        thumb[1:]=[[.48,.46,.13],[.38,.73,.32],[.16,.83,.49]]
        if letter=='Y': thumb[1:]=[[.53,.42,.12],[.76,.63,.15],[.98,.82,.15]]; sides[0]=u(-43,-14)
        tilt=u(-16,25)
    elif letter in 'KP':
        extend([3,2],[u(1,13),u(-22,-6)])
        f[2]=[u(36,80),u(-3,10),u(-2,10)]
        thumb[1:]=[[.435,.48,.14],[.295,.77,.32],[.144,1.02,.41]]
        if letter=='K': yaw=u(-70,-14); pitch=u(-29,16);tilt=u(-17,21)
        else: yaw=u(-83,-27);pitch=u(-25,24);align=(3,u(-96,-53))
    elif letter=='L':
        extend([3],[u(-5,11)])
        thumb[1:]=[[.51,.43,.075],[.77,.57,.073],[1.035,.665,.06]]
        yaw=u(-26,28); tilt=u(-17,25)
    elif letter in 'MN':
        f[:]=[u(41,67),u(81,104),u(38,64)]
        # The thumb passes under three fingers in M and under two in N.
        split=1 if letter=='M' else 2
        f[:split]=[u(79,99),95,62]
        f[split-1,0]-=u(6,17)
        sides[:]=[11,6,1,-5]; yaw=u(-39,34)
        thumb[1:]=[[.43,.45,.20],[.19,.64,.36],[(xbase[split]+xbase[split-1])/2,.79,.39]]
        tilt=u(-13,29)
    elif letter=='Q':
        # Point with the index AND thumb. Flex the wrist rather than rotating
        # the entire forearm to point down from above the image.
        f[3]=[u(41,87),u(-3,15),u(2,22)];sides[3]=u(-14,7)
        thumb[1:]=[[.48,.44,.25],[.58,.60,.51],[.65,.73,.77]]
        yaw=u(-172,-77);pitch=u(-59,-13);align=(3,u(-166,-130))
        if rng.random()<.55:
            # When the thumb points down, the index points out toward the
            # viewer and its end knuckle is strongly foreshortened.
            yaw=u(105,165);pitch=u(-16,33);align=None;tilt=u(136,190)
    elif letter in 'RUV':
        spread={'R':(-6,11),'U':(-1,1),'V':(u(10,27),u(-24,-6))}[letter]
        extend([3,2],spread)
        f[3,0]+=u(-1,7) if letter=='R' else u(-3,3)
        if letter=='R':
            # Braid in depth, not an X with the tips far apart.
            f[2,0]+=u(6,15);f[3,0]-=u(0,5)
            sides[2]=u(6,12);sides[3]=u(-10,-5)
        thumb[1:]=[[.44,.46,.16],[.22,.67,.32],[-.07,.79,.47]]
        yaw=u(-40,26);tilt=u(-21,20)
    elif letter=='S':
        f[:]=[u(68,90),100,62]
        thumb[1:]=[[.475,.49,.23],[.27,.73,.54],[-.06,.85,.57]]
        yaw=u(-43,33);tilt=u(-12,28)
    elif letter=='W':
        extend([1,2,3],[u(-32,-14),u(-5,4),u(12,29)])
        thumb[1:]=[[.44,.46,.16],[.21,.65,.30],[-.19,.73,.43]]
        yaw=u(-32,32)
    elif letter=='X':
        f[3]=[u(1,33),u(59,98),u(45,75)];sides[3]=u(-5,12)
        thumb[1:]=[[.46,.46,.18],[.35,.68,.35],[.09,.79,.49]]
        yaw=u(-73,9);tilt=u(-10,26)
    # A presenter can pitch the raised fingers toward the camera while the
    # remaining digits stay folded against the palm. Vary that independently
    # of the camera tilt. Cupping the two ulnar fingers brings their tips inward.
    if letter in 'DRUVX':
        yaw+=u(-25,21);pitch+=u(-15,18)
        if letter in 'DRUV':
            for j in ([3] if letter=='D' else [2,3]):f[j,0]+=u(-5,18)
        sides[:2]+=u(1,19)
    # Articulation variability, lengths, and finger pad thickness are independent
    # nuisance parameters, never measurements from images.
    f+=rng.normal(0,[3,3,3],(4,3))+rng.normal(0,[4,4,4],(1,3))
    sides+=rng.normal(0,2,4)
    # Fingertips touch in U and B, so their axes have correlated deviations.
    if letter=='U':
        sides[3]=sides[2]+u(-3,0);f[3]=f[2]+rng.normal(0,2.5,3)
    if letter=='B':
        for j in (0,1,3):f[j]=f[2]+rng.normal(0,3,3)
    lengths=LENGTH*u(.92,1.09)*u(.965,1.035,(4,1))
    widths=RAD*flesh*u(.95,1.045,4)
    tips=[]
    for k in range(4):
        pos=base[k].copy();rs=[1.,1.015,.895,.76];r=widths[k]
        phis=rad(f[k].cumsum()); a=rad(sides[k]); sa,ca=np.sin(a),np.cos(a)
        for j,phi in enumerate(phis):
            end=pos+lengths[k,j]*np.array([sa*np.cos(phi),ca*np.cos(phi),np.sin(phi)])
            bone(digits,pos,end,[r,r,r],rs[j],rs[j+1]);pos=end
        tips.append(pos.copy())
    if letter=='O' or letter=='F':
        contact=tips[3]
        thumb[3]=contact+np.array([.035,-.079,-.020])+u(-.035,.035,3)
        thumb[2]=.51*thumb[1]+.49*thumb[3]+np.array([.065,-.035,.095])
    elif letter=='C':
        thumb[3]=[.46,u(.60,.86),u(.60,.81)]
        thumb[2]=.53*thumb[1]+.47*thumb[3]+np.array([.04,-.045,.035])
    elif letter=='D':
        thumb[3]=tips[2]+np.array([.04,-.07,-.03]);thumb[2]=.47*thumb[1]+.53*thumb[3]+np.array([.07,-.05,.04])
    elif letter=='E':
        q=np.mean(tips,axis=0);thumb[3]=q+np.array([-.19,-.11,-.01]);thumb[2]=q+np.array([.12,-.13,-.005])
    elif letter=='S':
        q=tips[2]
        thumb[3]=q+np.array([u(-.31,-.12),u(-.04,.22),.155]);thumb[2]=q+np.array([.13,u(-.04,.17),.20])
    thumb[1:]+=rng.normal(0,.024,(3,3));tr=[.155,.135,.115,.091]
    for i in range(3):bone(digits,thumb[i],thumb[i+1],[flesh]*3,tr[i],tr[i+1])
    V=rx(rad(pitch))@ry(rad(yaw))
    if align is not None:
        vec=V@(tips[align[0]]-base[align[0]]); tilt=align[1]-np.rad2deg(np.arctan2(vec[0],vec[1]))
    V=roll(rad(tilt))@V@np.diag([u(.96,1.055),palmheight,u(.92,1.09)])
    if letter in 'PQ' or (letter in 'GHT' and rng.random()<.65):
        # The wrist can bend independently of the pose at the knuckles.
        angle={'P':u(70,149),'Q':u(134,200),'G':u(-157,-100),'H':u(-148,-95),'T':u(-175,-133)}[letter]
        ray=np.array([np.sin(rad(angle)),np.cos(rad(angle)),u(-.36,.24)])
        a,b,axis,r1,r2=arm[0]
        arm[0]=(b+(V.T@(ray*3))/(V*V).sum(axis=0),b,axis,r1,r2)
    return arm,palm+pads+digits,V

def pictograph(letter,rng):
    # A second, complementary representation in a local side elevation: the
    # horizontal axis goes along the thumb, the vertical along the palm.
    # This avoids foreshortening a schematic pose into a different letter.
    u=rng.uniform;arm=[];palm=[];pads=[];digits=[]
    def tube(parts,pts,axes,radii):
        for a,b,r,s in zip(pts,pts[1:],radii,radii[1:]):
            parts.append((np.array(a),np.array(b),np.array(axes),r,s))
    def finger(pts,r,fac=1.):
        tube(digits,pts,[r*fac]*3,np.linspace(1.,.78,len(pts)))
    if letter in 'COF':
        right=[(.03,.15,0),(-.11,.86,0)];fat=(.26,.24,.31)
    else:
        right=[(0,.18,0),(0,.80,0)];fat=(u(.34,.47),.24,.19)
    tilt=0.; armray=[u(-.30,.10),-1,0]
    if letter in 'CO':
        closed=letter=='O'
        # Four curved fingers in depth, thumb opposing them underneath.
        for j in range(4):
            k=j/3.;x=-.18+k*.14;z=-.24+k*.15;h=.86+k*.13
            finger([(x,h,z),(x+.13,h+.34,z+.01),(x+.39,h+.43,z+.02),
                    (x+.61,h+(.23 if closed else .39),z+.015)],.102 if j else .089)
        finger([(.04,.24,.10),(.23,.46,.10),(.45,.66,.12),(.57,1.18 if closed else .82,.14)],.137)
        tilt=u(-9,27)
    elif letter=='F':
        for j,(xx,direction,ll) in enumerate([(-.20,-.29,.73),(-.12,-.065,.94),(.02,.23,1.)]):
            d=direction+u(-.12,.12);y=.85+.05*j
            finger([(xx,y,j*.045),(xx+d*ll*.47,y+ll*.46,.035+j*.04),
                    (xx+d*ll,y+ll,.06+j*.04)],.096 if j==0 else .106)
        finger([(.06,.92,.17),(.34,1.14,.19),(.60,1.15,.21),(.69,.96,.22)],.113)
        finger([(.06,.27,.11),(.33,.50,.18),(.55,.69,.22),(.69,.93,.22)],.135)
        tilt=u(-9,25)
    elif letter in 'QT':
        if letter=='Q':
            # A flexed wrist, back of the hand sloping from the heel to the
            # knuckles. Index points out and down; thumb is opposed.
            right=[(.40,.44,.0),(-.02,1.,0.)];fat=(.33,.225,.25)
            armray=[u(.12,.76),-1,-.13];tilt=u(-18,23)
            for j in range(3):
                x=.12+.13*j;z=-.11+j*.045
                finger([(x,.97,z),(x-.20,.89,z+.09),(x-.31,.61,z+.14),(x-.19,.53,z+.11)],.102)
            finger([(.06,1.02,.12),(-.22,1.01,.19),(-.40,.69,.22),(-.61,.32,.22)],.120)
            finger([(.24,.59,.17),(.04,.50,.24),(-.125,.24,.27),(-.24,.04,.28)],.13)
        else:
            for j in range(3):
                x=-.25+.19*j;y=.84+j*.04
                finger([(x,y,.03),(x+.03,y+.13,.22),(x+.12,y+.04,.41),(x+.12,y-.15,.39)],.112)
            finger([(.16,.94,.12),(.44,.95,.22),(.75,.94,.25),(1.01,.92,.26)],.118)
            finger([(.04,.46,.09),(.10,.70,.21),(.04,.98,.30),(.045,1.24,.32)],.13)
            tilt=u(-14,19)
    elif letter in 'GH':
        right=[(-.32,.24,0),(-.08,.82,.0)];fat=(.32,.235,.21)
        armray=[u(-1.1,-.35),-1.,-.08];tilt=u(-19,12)
        # Fingers fold in layers on the underside of a horizontal pointer.
        for j in range(4 if letter=='G' else 3):
            x=-.13+j*.035;y=.34+.175*j;z=.01+.023*j
            finger([(x,y,z),(x+.26,y+.055,z+.08),(x+.43,y+.015,z+.19),(x+.40,y-.08,z+.20)],.105)
        finger([(-.12,.90,.11),(.21,1.00,.16),(.61,1.015,.155),(.97,1.015,.15)],.118)
        if letter=='H':finger([(-.17,.77,.08),(.18,.79,.12),(.62,.80,.10),(.99,.79,.095)],.117)
        else:finger([(-.25,.49,.13),(-.18,.77,.21),(.03,.93,.23),(.34,.99,.22)],.134)
    elif letter=='P':
        right=[(.39,.40,0),(.01,.85,0)];fat=(.42,.23,.20)
        armray=[u(.65,1.7),u(-.9,-.1),-.10];tilt=u(-18,20)
        for j in range(2):
            x=.10+.2*j;z=.02+.04*j
            finger([(x,.87,z),(x-.19,.85,z+.16),(x-.19,.66,z+.37),(x-.08,.65,z+.32)],.115)
        finger([(-.02,.83,.12),(-.45,.93,.10),(-.76,1.02,.10),(-1.,1.08,.10)],.118)
        finger([(-.15,.91,.06),(-.24,.63,.13),(-.32,.35,.23),(-.42,.14,.25)],.119)
        finger([(.17,.48,.10),(.04,.62,.22),(-.13,.73,.33),(-.25,.64,.34)],.132)
    elif letter=='X':
        # The extended proximal link and hooked distal links are viewed from
        # the side, partially hiding the three folded fingers.
        for j in range(3):
            x=-.23+.16*j;y=.84+.049*j;z=-.12+.08*j
            finger([(x,y,z),(x-.05,y+.17,z+.22),(x-.14,y+.045,z+.30),(x-.10,y-.12,z+.26)],.117)
        finger([(.16,.92,.07),(.18,1.32,.18),(.04,1.51,.21),(-.13,1.47,.22),(-.24,1.32,.23)],.12)
        finger([(.23,.33,.09),(.29,.60,.18),(.18,.79,.33),(-.01,.79,.37)],.138)
        tilt=u(-17,27)
    tube(palm,right,fat,[.77,1.]);base=np.array(right[0]);ray=np.array(armray)
    tube(arm,[base+3*ray,base],(.285,.28,.175),[1.45,.85])
    V=roll(rad(tilt))@rx(rad(u(-22,23)))@ry(rad(u(-29,28)))@np.diag([u(.86,1.12),u(.90,1.08),1.])
    return arm,palm+pads+digits,V

def render(letter,rng):
    if letter in 'COFGHPQTX' and rng.random()<.45:
        arm,parts,V=pictograph(letter,rng)
    else:
        arm,parts,V=build_pose(letter,rng)
    # Orthographic projection: track the visible depth of the union of tapered
    # swept ellipsoids. Ray/ellipsoid intersection is a quadratic in depth.
    V=V.copy();V[1]*=-1
    cache={}
    spans=[]; ops=[]
    for idx,(a,b,axes,r,s) in enumerate(arm+parts):
        A=V@a; B=V@b; key=tuple(axes)
        if key not in cache:
            mat=(V*axes)@(V*axes).T
            det=mat[0,0]*mat[1,1]-mat[0,1]**2
            g=np.array([[mat[1,1],-mat[1,0]],[-mat[0,1],mat[0,0]]])/det
            q=mat[2,:2]@g
            cache[key]=(g,q,np.sqrt(max(1e-12,mat[2,2]-q@mat[:2,2])),np.sqrt(np.diag(mat)[:2]))
        g,q,h,extent=cache[key]
        bounds=(np.minimum(A[:2]-r*extent,B[:2]-s*extent),np.maximum(A[:2]+r*extent,B[:2]+s*extent))
        ops.append((A,B,r,s,bounds,g,q,h))
        if idx>=len(arm):spans.append(bounds)
    low=np.min([v[0] for v in spans],axis=0); high=np.max([v[1] for v in spans],axis=0)
    cen=(low+high)*.5
    scale=R*.845/max(high-low); bias=(R-1)*.5-cen*scale
    zbuf=np.full((R,R),-8.);body=np.zeros((R,R),bool)
    articulated=len(parts)==18
    crease=np.zeros((R,R));owner=np.full((R,R),-1,np.int32)
    for oi,(A,B,r,s,(l,hi),g,q,h) in enumerate(ops):
        x0,y0=np.maximum(0,np.ceil(l*scale+bias).astype(int));x1,y1=np.minimum(R,np.floor(hi*scale+bias).astype(int)+1)
        if x0>=x1 or y0>=y1:continue
        x=(np.arange(x0,x1)[None,:]-bias[0])/scale-A[0]
        y=(np.arange(y0,y1)[:,None]-bias[1])/scale-A[1]
        delta=B-A; d=delta[:2]; gd=g@d; d2=d@gd; slope=(delta[2]-q@d)/h
        cross=gd[0]*x+gd[1]*y; rr=g[0,0]*x*x+2*g[0,1]*x*y+g[1,1]*y*y
        # Maximize z(t) over the interpolated cross section along each bone.
        dr=s-r; curv=d2-dr*dr
        if curv>1e-8:
            mp=(cross+r*dr)/curv
            radius=np.maximum(0.,r*r-rr+mp*mp*curv)
            t=np.clip(mp+slope/np.sqrt(curv*(curv+slope*slope))*np.sqrt(radius),0.,1.)
        else:
            e0=r*r-rr;e1=s*s-rr+2*cross-d2
            t=np.where((e1>=0)&((e0<0)|(slope+np.sqrt(np.maximum(0,e1))>np.sqrt(np.maximum(0,e0)))),1.,0.)
        disc=(r+dr*t)**2-rr+t*(2*cross-t*d2)
        height=A[2]+q[0]*x+q[1]*y+h*(slope*t+np.sqrt(np.maximum(0,disc)))
        old=zbuf[y0:y1,x0:x1];new=(disc>=0)&(height>old)
        if articulated:
            di=oi-len(arm)-3; tag=di//3 if di>=0 else -1
            region=owner[y0:y1,x0:x1];region[new]=tag
            cv=crease[y0:y1,x0:x1];cv[new]=0.
            if di>=0 and di%3!=0:
                # Flexion creases on the palmar face, transverse to each link.
                bend=np.exp(-((t-.13)/.080)**2)*np.clip(V[2,2]+.35,0,1)
                cv[new]=bend[new]
        np.maximum(old,np.where(disc>=0,height,-8.),out=old)
        if oi>=len(arm):body[y0:y1,x0:x1]|=disc>=0
    skin=zbuf>-7.99
    # One-sided finite differences choose the same surface rather than using
    # a depth discontinuity as a normal. Shading includes a soft cast shadow.
    zp=np.pad(zbuf,1,mode='edge'); zm=zp[1:-1,1:-1]
    def derivative(l,r):
        dl=zm-l; dr=r-zm
        return np.where(abs(dl)<abs(dr),dl,dr)*scale
    dx=derivative(zp[1:-1,:-2],zp[1:-1,2:]);dy=derivative(zp[:-2,1:-1],zp[2:,1:-1])
    u=rng.uniform
    lx=u(-1.1,-.25) if rng.random()<.83 else u(.2,1.);ly=u(-1.,-.15);lz=u(.27,1.05)
    norm=np.sqrt(lx*lx+ly*ly+lz*lz)
    lam=(lz-lx*dx-ly*dy)/(np.sqrt(1+dx*dx+dy*dy)*norm)
    lam=.76*np.maximum(lam,0.)+.24*lam
    shadow=np.zeros_like(zbuf)
    for step in [1,2,4,7]:
        sy=int(round(-ly*step));sx=int(round(-lx*step)); occluder=np.roll(zbuf,(sy,sx),(0,1))
        occl=np.clip((occluder-zbuf-step*lz/scale)*6.,0,1)
        if sy>0:occl[:sy]=0
        if sx>0:occl[:,:sx]=0
        if sx<0:occl[:,sx:]=0
        shadow=np.maximum(shadow,occl)
    ambient=u(.21,.47);lit=u(.33,.57)
    bg=u(.46,.88) if rng.random()<.82 else u(.14,.47)
    im=np.where(skin,ambient+lit*lam*(1-.65*shadow)-.075*shadow,bg)
    if articulated:
        seam=np.zeros_like(im)
        for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
            nbr=np.roll(owner,(dy,dx),(0,1));nz=np.roll(zbuf,(dy,dx),(0,1))
            seam=np.maximum(seam,((nbr!=owner)&(nz>zbuf-.01)&(owner>=0)&(nz>-7.99))*1.)
        im-=rng.uniform(.07,.18)*crease+rng.uniform(.08,.23)*seam
    def shrink(v):return v.reshape(28,2,28,2).mean(axis=(1,3)).astype(np.float32)
    return shrink(np.clip(im,0,1)),shrink(body),shrink(skin)

FEATURES = r'''import numpy as np

def blur(a, axis, w):
    k=len(w)//2
    pad=[(0,0)]*a.ndim
    pad[axis]=(k,k)
    b=np.pad(a,pad,mode='edge')
    sl=[slice(None)]*a.ndim
    out=np.zeros_like(a)
    for j,c in enumerate(w):
        sl[axis]=slice(j,j+a.shape[axis]);out+=c*b[tuple(sl)]
    return out

def filt(a,w):
    return blur(blur(a,2,w),1,w)

def features(a):
    # Cells describe edges as both unoriented axes and signed contrasts.
    # The luminance channel is a band pass, not absolute skin brightness.
    small=[.125,.75,.125]; wide=[.0625,.25,.375,.25,.0625]
    a=filt(a,small)
    b=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    gx=(b[:,1:-1,2:]-b[:,1:-1,:-2])*.5
    gy=(b[:,2:,1:-1]-b[:,:-2,1:-1])*.5
    mag=np.sqrt(gx*gx+gy*gy+.000001)
    power=filt(mag,wide)
    amp=np.minimum(mag/(.016+.75*power)**.78,2.)
    chans=[]; signed=[]
    for k in range(6):
        v=(gx*np.cos(k*np.pi/6)+gy*np.sin(k*np.pi/6))/mag
        axis=np.maximum(0.,(np.abs(v)-.78)/.22)*amp
        chans.append(axis);signed.append(axis*np.sign(v))
    gray=(a-filt(filt(a,wide),wide))/(.055+power)
    maps=np.stack(chans+signed+[amp,np.clip(gray,-2.5,2.5)],axis=3).astype(np.float32)
    def make(m):
        axes=m[:,:,:,:6].copy()
        level=m[:,:,:,12:13]
        axes-=.145*level
        m=np.concatenate([axes, .64*m[:,:,:,6:12], .48*level, .14*m[:,:,:,13:14]],axis=3)
        return m/np.sqrt(.055+level)
    fine=make(filt(maps,[.08,.23,.38,.23,.08]))
    coarse=make(filt(filt(maps,wide),wide))
    # Saliency for the rendered outer contour: allow a locally light or dark
    # background, and tolerate an edge location uncertain by about one pixel.
    edge=(mag/(mag+.025))*(power/(power+.020))
    oriented=np.stack([z/(amp+.0001)*edge for z in chans],axis=3)
    b=np.pad(oriented,((0,0),(1,1),(1,1),(0,0)),mode='edge')
    pool=np.maximum.reduce([b[:,j:j+28,i:i+28] for j,i in ((0,1),(1,0),(2,1),(1,2),(1,1))])
    oriented=filt(.65*pool+.35*oriented,[.2,.6,.2])
    return fine,coarse,oriented.astype(np.float32)

def sample(a,y,x):
    # All points are pixel centers in the original frame. Do not hallucinate
    # edges in the unobserved region beyond the photograph.
    yy=np.clip(y,0,27);xx=np.clip(x,0,27)
    iy=yy.astype(np.int32);ix=xx.astype(np.int32)
    wy=(yy-iy).astype(np.float32);wx=(xx-ix).astype(np.float32)
    v=((1-wx)[None,None,:,None]*a[:,:,ix]+wx[None,None,:,None]*a[:,:,np.minimum(ix+1,27)])
    v=(1-wy)[None,:,None,None]*v[:,iy]+wy[None,:,None,None]*v[:,np.minimum(iy+1,27)]
    ey=np.clip(1-np.abs(yy-y)*.7,0,1);ex=np.clip(1-np.abs(xx-x)*.7,0,1)
    return v*(ey[:,None]*ex[None,:])[None,:,:,None]

def grid(n):
    if n==7: return np.arange(n,dtype=np.float32)*4+1.5
    return np.arange(n,dtype=np.float32)*2+2.5

def descript(m,n,dx,dy,sc):
    t=grid(n)-13.5
    w=(1-.25*(np.abs(t)/13.5)**3).astype(np.float32)
    v=sample(m,13.5+dy+t*sc,13.5+dx+t*sc)*(w[None,:,None,None]*w[None,None,:,None])
    return v/np.sqrt(np.maximum(.000001,(v*v).sum(axis=(1,2,3),keepdims=True)))

def contour(body,skin,n):
    b=np.pad(skin,((0,0),(1,1),(1,1)),mode='edge')
    gx=b[:,1:-1,2:]-b[:,1:-1,:-2];gy=b[:,2:,1:-1]-b[:,:-2,1:-1]
    mag=np.sqrt(.000001+gx*gx+gy*gy)
    field=np.stack([np.maximum(0,(np.abs(gx*np.cos(i*np.pi/6)+gy*np.sin(i*np.pi/6))/mag-.85)/.15)*mag for i in range(6)],axis=3)
    field*=np.clip(filt(body,[.25,.5,.25])*2.,0,1)[:,:,:,None]
    w=[.15,.70,.15] if n==12 else [.10,.23,.34,.23,.10]
    field=sample(filt(field,w),grid(n),grid(n))
    # Contrast with an annulus on both sides of the edge. A cluttered patch
    # rich in all orientations should not be evidence for every contour.
    center=field/(field.sum(axis=(1,2,3),keepdims=True)+.000001)
    s=filt(field,[.18,.20,.24,.20,.18])
    surround=s/(s.sum(axis=(1,2,3),keepdims=True)+.000001)
    return center-.80*surround

def support(a):
    # A soft activity image: even weak narrow fingers disturb an otherwise
    # smooth background. Compress contrast to avoid discarding them.
    d=a-filt(a,[.12,.23,.30,.23,.12]);e=filt(np.abs(d),[.25,.50,.25])
    return (e/(e+.015))[:,:,:,None]

def occupancy(body,skin,n):
    w=[.12,.23,.30,.23,.12]
    field=(.74*filt(skin,w)+.26*filt(body,w))[:,:,:,None]
    return sample(field,grid(n),grid(n))

def ridges(a):
    # Hessian directional curvature: even-symmetric line filters complement
    # first-order edges, specifically when the two edges of a digit are close.
    planes=[]
    for h in (1,2):
        sm=filt(a,[.15,.70,.15] if h==1 else [.06,.24,.40,.24,.06])
        b=np.pad(sm,((0,0),(h,h),(h,h)),mode='edge');n=28
        xx=b[:,h:h+n,:n]+b[:,h:h+n,2*h:]-2*sm
        yy=b[:,:n,h:h+n]+b[:,2*h:,h:h+n]-2*sm
        xy=(b[:,2*h:,2*h:]+b[:,:n,:n]-b[:,2*h:,:n]-b[:,:n,2*h:])*.25
        level=np.sqrt(.0004+filt(xx**2+yy**2+2*xy**2,[.12,.23,.30,.23,.12]))
        for k in range(6):
            c=np.cos(np.pi*k/6);s=np.sin(np.pi*k/6)
            v=(c*c*xx+2*s*c*xy+s*s*yy)/level**.73
            perp=(s*s*xx-2*s*c*xy+c*c*yy)/level**.73
            ridge=np.maximum(0,np.abs(v)-.65*np.abs(perp))
            planes.append(ridge);planes.append(ridge*np.sign(v)*.72)
    return filt(np.stack(planes,axis=3).astype(np.float32),[.22,.56,.22])

def ridgevec(m,dy,dx,sc):
    t=np.arange(7,dtype=np.float32)*3.5+3.-13.5
    v=sample(m,13.5+dy+t*sc,13.5+dx+t*sc)
    return (v/np.sqrt(.000001+(v*v).sum(axis=(1,2,3),keepdims=True))).reshape(len(m),-1)

def outline(shape):
    # Unit-mass angular histograms inside the putative foreground, measured
    # about its own centroid. Coarse rings describe the palm; thin distant
    # sectors describe the radial ordering of the fingertips and thumb.
    b=len(shape);z=shape.reshape(b,784)
    mass=np.maximum(.000001,z.sum(axis=1,keepdims=True));p=z/mass
    t=np.arange(28,dtype=np.float32)-13.5
    x=np.broadcast_to(t[None,:],(28,28)).reshape(1,784)
    y=x.reshape(28,28).T.reshape(1,784)
    xx=x-(x*p).sum(axis=1,keepdims=True);yy=y-(y*p).sum(axis=1,keepdims=True)
    rr=np.sqrt(xx*xx+yy*yy);scale=np.sqrt(.5+(p*rr**2).sum(axis=1,keepdims=True))
    rr=rr/scale; ang=(np.arctan2(yy,xx)+np.pi)*(8/np.pi)
    ai=ang.astype(np.int32);rem=ang-ai;idx=np.arange(b)[:,None]*16
    desc=[]
    for j in range(5):
        w=p*np.maximum(0.,1.-np.abs(rr*2-j))*((1.+j)*.5)
        aa=np.bincount((idx+ai%16).ravel(),weights=(w*(1-rem)).ravel(),minlength=b*16)
        bb=np.bincount((idx+(ai+1)%16).ravel(),weights=(w*rem).ravel(),minlength=b*16)
        h=(aa+bb).reshape(b,16);desc.append(.7*h+.15*(np.roll(h,1,1)+np.roll(h,-1,1)))
    v=np.concatenate(desc,axis=1)
    # Hellinger affinity, with no data-dependent population normalization.
    v=np.sqrt(np.maximum(0.,v));return (v/np.sqrt(.000001+(v*v).sum(axis=1,keepdims=True))).astype(np.float32)

def foreground(x):
    # Smooth backgrounds are common, even when they are not uniform. Compare
    # each pixel with a plane extrapolated from a quiet upper-side patch. Keep
    # a background estimate from each side, deciding between them per image.
    b=len(x);y,z=np.mgrid[0:28,0:28].astype(np.float32)
    activity=support(x)[:,:,:,0];cent=np.exp(-.5*((z-13.5)/7.5)**2-.5*((y-15)/9.)**2)
    smooth=filt(x,[.2,.6,.2]);masks=[];good=[]
    for left in (True,False):
        corners=[(0,0),(0,5),(5,0),(10,0)] if left else [(0,23),(0,18),(5,23),(10,23)]
        noise=[];planes=[]
        for j,i in corners:
            patch=smooth[:,j:j+5,i:i+5]
            noise.append(activity[:,j:j+5,i:i+5].mean(axis=(1,2)))
            vx=np.clip(np.median(patch[:,:,1:]-patch[:,:,:-1],axis=(1,2)),-.014,.014)
            vy=np.clip(np.median(patch[:,1:]-patch[:,:-1],axis=(1,2)),-.014,.014)
            planes.append(np.median(patch,axis=(1,2))[:,None,None]+(z-i-2)*vx[:,None,None]+(y-j-2)*vy[:,None,None])
        best=np.argmin(np.stack(noise,axis=1),axis=1)
        model=np.stack(planes,axis=1)[np.arange(b),best]
        delta=np.abs(smooth-model);detail=activity
        def grow(v):
            p=np.pad(v,((0,0),(1,1),(1,1)),mode='edge')
            return np.maximum.reduce([p[:,i:i+28,j:j+28] for i,j in ((0,1),(1,0),(2,1),(1,2),(1,1))])
        # Interior support from both tone and texture; morphological closing
        # keeps weakly contrasted gaps between knuckles in the same hand.
        m=np.maximum(np.clip((delta-.018)/.105,0,1),np.clip((detail-.27)/.30,0,1))
        m=-grow(-grow(m));m=filt(m,[.2,.6,.2]);m=np.clip((m-.20)/.64,0,1)
        mass=m.sum(axis=(1,2));central=(m*cent).sum(axis=(1,2))/np.maximum(1,mass)
        reliability=np.clip(central*2-.46,0,1)*np.clip(1.50-mass/440,0,1)*np.clip(mass/105,0,1)
        masks.append(m);good.append(reliability)
    rel=np.stack(good,axis=1);chosen=np.argmax(rel,axis=1)
    m=np.stack(masks,axis=1)[np.arange(b),chosen]
    return outline(m),rel.max(axis=1),m,activity
'''

INFER = r'''
bankC=CONSTANTS['coarse'].astype(np.float32)
bankF=CONSTANTS['fine'].astype(np.float32)
bankS=CONSTANTS['shape']
bankR=CONSTANTS['ridge']

def classify(x):
    fine,coarse,oriented=features(x)
    shape,conf,mask,act=foreground(x)
    act=act[:,:,:,None];inv=1/(.001+act.sum(axis=(1,2),keepdims=True))
    mass=(mask.sum(axis=(1,2))+.001)[:,None,None,None]
    evidence=act*inv-1/784+.85*conf[:,None,None,None]*(mask[:,:,:,None]/mass-1/784)
    rv=ridges(x);rc=[]
    fc=[];cc=[];ec=[];ac=[]
    for sc in (.77,.98,1.22):
        for dy in (-5.,-2.5,0.,2.5,5.):
            for dx in (-3.25,0.,3.25):
                def edged(n):
                    t=(grid(n)-13.5)*sc+13.5
                    return sample(oriented,t+dy,t+dx)
                cv=descript(coarse,7,dx,dy,sc)
                def activity(n):
                    t=(grid(n)-13.5)*sc+13.5
                    # Include activity in the denominator even outside the proposed crop.
                    # Center against a uniform image to correct for projected area.
                    h=sample(evidence,t+dy,t+dx)
                    return h*((4*sc)**2 if n==7 else (2*sc)**2)
                cc.append(np.concatenate([cv,edged(7),activity(7)],axis=3).reshape(len(x),-1))
                fc.append(descript(fine,12,dx,dy,sc));ec.append(edged(12));ac.append(activity(12))
                rc.append(ridgevec(rv,dy,dx,sc))
    rc=np.stack(rc,axis=1)
    c=np.stack(cc,axis=1);f=np.stack(fc,axis=1);e=np.stack(ec,axis=1);actf=np.stack(ac,axis=1)
    b,aligns,length=c.shape; rows=np.arange(b)[:,None]
    c=c.reshape(b*aligns,length)
    answer=np.empty((b,24),np.float32)
    for k in range(24):
        bank=bankC[k]; count=len(bank)
        shapeval=(shape@bankS[k].reshape(-1,80).T).reshape(b,count,2).max(axis=2)
        shapeval=conf[:,None]*shapeval
        s=(c@bank.T).reshape(b,aligns,count)
        s=(s+.68*shapeval[:,None,:]).reshape(b,-1)
        keep=20
        ids=np.argpartition(s,-keep,axis=1)[:,-keep:]
        coarseval=s[rows,ids];p=bankF[k,ids%count].reshape(-1,12,12,21)
        q=f[rows,ids//count].reshape(-1,12,12,14)
        uv=p[:,:,:,20:21];disk=.28+.72*uv
        # Normalize over the modeled hand and an attenuated background, not
        # the unweighted whole crop. Sleeves should not swamp fingertip detail.
        qn=(q*q*disk).sum(axis=(1,2,3),keepdims=True)
        pn=(p[:,:,:,:14]**2*disk).sum(axis=(1,2,3),keepdims=True)
        delta=disk/np.sqrt(.00001+qn*pn)
        ev=e[rows,ids//count].reshape(-1,12,12,6)
        boundary=(p[:,:,:,14:20]*ev).sum(axis=(1,2,3)).reshape(b,keep)
        coverage=(p[:,:,:,20:]*actf[rows,ids//count].reshape(-1,12,12,1)).sum(axis=(1,2,3)).reshape(b,keep)
        p=p[:,:,:,:14]*(.35+.65*delta)
        # Slightly elastic comparison: each of nine parts may move by one
        # fine cell. Share most of the score with rigid registration.
        def parts(q,p):
            return np.einsum('bhiwjc,bhiwjc->bhw',q.reshape(-1,3,4,3,4,14),p.reshape(-1,3,4,3,4,14),optimize=False)
        rigid=parts(q,p);best=rigid.copy()
        pad=np.pad(q,((0,0),(1,1),(1,1),(0,0)),mode='constant')
        for yy,xx in ((0,1),(2,1),(1,0),(1,2)):
            best=np.maximum(best,parts(pad[:,yy:yy+12,xx:xx+12],p))
        sim=(.52*rigid+.48*best).sum(axis=(1,2)).reshape(b,keep)
        ridgematch=(rc[rows,ids//count]*bankR[k,ids%count]).sum(axis=2)
        vals=.68*sim+.32*coarseval+.22*boundary+.25*coverage+.32*shapeval[rows,ids%count]+.28*ridgematch
        # Marginalize over several excellent alignments, damping an isolated
        # accidental edge match without averaging any template descriptors.
        votes=np.partition(vals,-4,axis=1)[:,-4:]
        answer[:,k]=.64*vals.max(axis=1)+.36*votes.mean(axis=1)
    return np.argmax(answer,axis=1)

def predict(x):
    # A fixed internal block size also fixes the BLAS computation for each
    # image regardless of the outer batch size.
    labels=np.empty(len(x),np.int64)
    for i in range(0,len(x),16):
        n=min(16,len(x)-i)
        a=np.zeros((16,28,28),np.float32);a[:n]=x[i:i+n,0]
        labels[i:i+n]=classify(a)[:n]
    return labels
'''

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(FEATURES+INFER)
    # The descriptor code is exactly shared between model and camera images.
    env={};exec(FEATURES,env)
    count=864
    coarse=np.empty((24,count,7*7*21),np.float32)
    fine=np.empty((24,count,12,12,21),np.float32)
    shapes=np.empty((24,count,2,80),np.float32)
    rv=np.empty((24,count,7*7*24),np.float32)
    for k,ch in enumerate(LETS):
        rng=np.random.Generator(np.random.PCG64(202611+7919*k))
        for i in range(0,count,24):
            images,body,skin=map(np.stack,zip(*(render(ch,rng) for j in range(24))))
            f,c,edge=env['features'](images)
            rv[k,i:i+24]=env['ridgevec'](env['ridges'](images),0,0,1)
            shapes[k,i:i+24]=np.stack([env['outline'](skin),env['outline'](body)],axis=1)
            for n,feat,bank,gain in ((7,c,coarse,.30),(12,f,fine,1.)):
                vec=env['descript'](feat,n,0,0,1)
                geom=env['contour'](body,skin,n)
                mask=env['occupancy'](body,skin,n)
                desc=np.concatenate([vec,gain*geom,(.4 if n==7 else 1.)*mask],axis=3)
                bank[k,i:i+24]=desc.reshape(bank[k,i:i+24].shape)
    np.savez(out/'constants.npz',coarse=coarse,fine=fine,shape=shapes,ridge=rv)

if __name__=='__main__':main()
