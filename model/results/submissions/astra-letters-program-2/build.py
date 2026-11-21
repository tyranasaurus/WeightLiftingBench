"""Construct a matcher from a procedural model of an articulated hand. No input files."""
import sys
from pathlib import Path
import numpy as np

letters='ABCDEFGHIKLMNOPQRSTUVWXY'
rad=np.pi/180

def part(a,b,r,s=None,kind=0):
    return (np.array(a),np.array(b),r,r if s is None else s,kind)

def pose(c, rng):
    # Units: roughly half the breadth of the palm. x points to the thumb,
    # y toward the fingertips; flexion bends the fingers forward (+z).
    u=rng.uniform
    base=np.array([[-.71,.51,-.08],[-.275,.77,-.035],[.17,.84,0],[.62,.69,.065]])
    lengths=np.array([1.20,1.67,1.83,1.63])*u(.91,1.08,4)
    thick=np.array([.188,.212,.23,.224])*u(.94,1.09)
    base[:,1]+=u(-.07,.07,4)
    # Angles here are absolute phalanx attitudes, not image coordinates.
    ext=np.array([0,2,5]); curl=np.array([67,153,237]); relaxed=np.array([53,131,213]); claw=np.array([9,89,174]); hook=np.array([-5,68,146]); curved=np.array([22,62,100]); ring=np.array([44,93,145])
    bends=np.tile(curl,(4,1)).astype(float)
    side=np.array([-7.,-3.,1.,5.]); yaw=u(-7,25); roll=u(-15,7); pitch=u(-7,12); wristlean=u(-.36,-.02); wristdepth=0.
    thumb='bar'; compact=False
    if c=='A': thumb='along'; compact=True; bends[:]=relaxed; yaw+=u(-17,7)
    if c=='B': bends[:]=ext; side[:]=[8,3,-2,-8]; thumb='palm'; yaw+=u(-12,11)
    if c=='C' or c=='O':
        bends[:]=curved if c=='C' else ring
        side[:]=[-4,-1,2,5]; yaw=u(54,89); roll=u(-24,-1); thumb='cup' if c=='C' else 'circle'
        wristlean=u(-.35,0); wristdepth=u(-.45,.1)
    if c=='D':
        bends[:]=ring; bends[3]=ext; side[3]=u(-12,5); thumb='circle3'; yaw=u(10,43)
    if c=='E': bends[:]=claw; thumb='under'; compact=True; side*=.5
    if c=='F':
        bends[:]=ext; bends[3]=ring; side[:]=[-21,-3,16,10]; thumb='circle'; yaw=u(6,72); roll=u(-23,12)
    if c=='I': bends[0]=ext; side[0]=u(-18,-5); thumb='bar'
    if c=='K':
        bends[3]=ext+u(-12,9); bends[2]=[u(24,60),u(31,70),u(25,67)]; side[2:]=[u(-36,-13),u(-6,11)]; thumb='wedge'; yaw=u(-30,23)
    if c=='L': bends[3]=ext; side[3]=u(-12,1); thumb='spread'; yaw=u(-10,16)
    if c in 'MN': thumb='three' if c=='M' else 'two'; compact=True; bends[:]=relaxed+[u(-8,14),u(-9,11),u(-7,10)]
    if c=='R':
        bends[2:]=ext; side[2:]=[u(4,12),u(-18,-8)]; bends[2]+=[6,4,0]; thumb='bar'
    if c=='S': compact=True; thumb='fist'; bends[:]=curl
    if c=='U': bends[2:]=ext; side[2:]=[3,-1]; thumb='bar'
    if c=='V': bends[2:]=ext; side[2:]=[u(-25,-10),u(8,26)]; thumb='bar'; roll=u(-12,8)
    if c=='W': bends[1:]=ext; side[1:]=[u(-26,-13),u(-7,4),u(10,24)]; thumb='little'
    if c=='X': bends[3]=hook; side[3]=u(-15,2); yaw=u(22,61); thumb='fist'; roll=u(-26,-1)
    if c=='Y': bends[0]=ext; side[0]=u(-36,-15); thumb='spread'; yaw=u(-5,20)
    # The side-on pointing signs roll the palm onto its side. A low
    # index points along the line of the knuckles, thumb below or beside it.
    if c in 'GH':
        bends[3]=[3,6,8]; side[3]=0; thumb='parallel'; yaw=u(-30,25); pitch=u(-10,23); roll=u(-86,-62); wristlean=u(-.3,.45); wristdepth=u(-.45,.3)
        if c=='H': bends[2]=[0,2,2];side[2]=2; thumb='bar'
    if c=='P':
        bends[3]=ext; bends[2]=[39,44,53]; side[2:]=[-35,4]; thumb='wedge'; roll=u(42,88); yaw=u(-59,-13); pitch=u(-8,24); wristlean=u(-.2,.55)
    if c=='Q':
        bends[3]=[40,57,67]; thumb='pinch'; yaw=u(125,176); pitch=u(58,103); roll=u(-53,-3); wristlean=u(-.3,.45); wristdepth=u(-.7,-.3)
    if c=='T':
        # Horizontal index over the inserted, upward thumb in a side view.
        bends[3]=[49,69,86]; side[3]=u(11,21); thumb='upright'; yaw=u(53,83); roll=u(-7,19); pitch=u(-9,16)
    if compact: pitch=u(1,29); roll+=u(-12,8)
    # Articulation of the knuckle arch: the last fingers close more deeply
    # around the thumb. In M and N the thumb actually splits two fingers.
    if c in 'MN':
        split=1 if c=='M' else 2
        bends[:]=[46,136,222]; bends[:split]+=[29,25,10];side[split-1]-=6; side[split]+=5; base[:split,1]-=.13
        roll+=u(1,14)
    if c=='S': bends[:]=[67,153,219];pitch=u(-4,18);yaw=u(-4,44)
    if c=='E': bends[:]=[30,108,176];pitch=u(-6,13);yaw=u(1,22);roll=u(-10,8)
    if c=='K': yaw=u(-26,11);side[2]=u(-29,-10);bends[2]=[47,53,58];side[3]=u(-14,-3);roll=u(-4,16)
    if c=='S':
        bends[:]=[77,164,233]; bends[:2]+=[9,12,7];side*=.52
    if c in 'DRU':
        # Two adducted fingers show as a broad shaft, crossed fingers
        # twist and present a narrower oblique shaft, and D has one shaft.
        if c=='U':side[2:]=[.7,-1.8];yaw=u(-7,19)
        if c=='R':yaw=u(5,43);bends[2]=[-2,0,5];bends[3]=[11,9,-1]
    if c=='B': side[:]=[2,1,-1,-3];yaw=u(6,44)
    if c=='R': side[2:]=[u(5,13),u(-25,-15)];bends[3]+=[6,13,9];roll=u(-10,11)
    if c in 'GH':
        yaw=u(136,173);roll=u(-91,-66);pitch=u(-13,14);bends[3]=[1,2,4]
        if c=='G': thumb='behind'
    if c=='Q': yaw=u(-24,22);pitch=u(85,137);roll=u(-59,-17);bends[3]=[21,42,62]
    if c=='X':
        bends[3]=[-17,25,102]; bends[:3]=[72,155,226];roll=u(-19,3);yaw=u(48,79)
    if c=='B': roll=u(-13,10);pitch=u(-16,12)
    if c=='F': yaw=u(35,84); roll=u(-19,23);side[:3]=[u(-39,-16),u(-10,6),u(22,43)];bends[:3,0]=u(-21,8,3)
    if c=='I': side[0]=u(-23,0);pitch=u(-12,18);roll=u(-13,12)
    if c=='D': bends[:3]=[58,126,180]; thumb='circle3';pitch=u(-13,12); yaw=u(-14,38)
    if c=='P': yaw=u(-69,-18);pitch=u(-18,18);roll=u(44,85)
    if c=='Q':
        yaw=u(-18,25);pitch=u(72,101);roll=u(3,29);bends[3]=[53,76,86]; bends[:3]=[70,165,238]
    if c=='O': yaw=u(64,91);bends[:]=[29,81,135]; roll=u(-21,6)
    if c=='C': yaw=u(64,93);bends[:]=[18,58,98];roll=u(-22,4)
    if c=='T': bends[3]=[63,83,94];pitch=u(-12,4);yaw=u(44,77)
    if c=='V':roll=u(-14,19)
    if c=='W':roll=u(-15,16)
    if c=='Y':side[0]=u(-46,-16);roll=u(-11,19)
    arm=np.array([wristlean,-1.49,wristdepth]); attach=np.array([0,-.82,0]); pts=[]
    # The palm, with a convex heel and knuckle arch, is drawn as smooth
    # ellipsoids. Finger/thumb segments are tapered, round-ended cylinders.
    ell=[(np.array([0.,-.03,-.02]),np.array([.965,1.07,.42])),(np.array([.40,-.17,.135]),np.array([.49,.79,.37]))]
    # A slight web extends between the four proximal finger attachments.
    ends=[]; mids=[]
    for f in range(4):
        p=base[f].copy(); bs=(bends[f]+u(-6,6,3))*rad; s=(side[f]+u(-3,3))*rad
        for j,frac in enumerate([.45,.31,.24]):
            v=np.array([np.sin(s)*np.cos(bs[j]),np.cos(s)*np.cos(bs[j]),np.sin(bs[j])]); q=p+lengths[f]*frac*v
            pts.append(part(p,q,thick[f]*[1.05,1,.88][j],thick[f]*[1,.88,.79][j],1+j))
            p=q
            if j==1:mids.append(p.copy())
        ends.append(p.copy())
    b=np.array([.54,-.43,.20]); m=np.array([.87,-.13,.46]); t=np.array([.63,.27,.78]); e=np.array([-.17,.43,.91]);
    if thumb=='along': m=[.88,-.11,.21]; t=[1.00,.38,.3]; e=[.95,.92,.42]
    if thumb=='under': m=[.88,-.2,.43]; t=[.58,.17,.77]; e=[-.44,.22,.79]
    if thumb=='palm': m=[.74,-.21,.46]; t=[.32,.02,.57]; e=[-.43,.12,.54]
    if thumb in ('three','two'): m=[.83,-.12,.4]; t=[.54,.22,.59]; e=[-.48 if thumb=='three' else -.06,.75,.60]
    if thumb=='fist': m=[.86,.0,.5]; t=[.52,.40,.84]; e=[-.25,.63,.97]
    if thumb=='little': t=[.29,.09,.74]; e=[-.54,.43,.79]
    if thumb in ('circle','circle3','cup'):
        target=ends[2 if thumb=='circle3' else 3]; m=np.array([.92,-.12,.46]); t=(m+target)*.5+np.array([.21,-.22,.23]);e=target.copy()
        if thumb=='cup': e[1]-=u(.78,1.25); t=(m+e)*.5+np.array([.13,-.14,.08])
    if thumb=='spread': m=[1.0,-.29,.25];t=[1.49,-.12,.31];e=[u(1.97,2.28),u(-.07,.27),.30]
    if thumb=='wedge': m=[.84,-.14,.5];t=[.58,.53,.72]; e=[.43,1.25,.64]
    if thumb=='parallel': m=[1.0,-.10,.29];t=[1.06,.62,.46];e=[.99,1.36,.55]
    if thumb=='pinch':m=[1.04,-.12,.32];t=[1.25,.29,.73];e=[1.33,.67,1.18]
    if thumb=='upright':m=[.90,.08,.45];t=[.86,.91,.52];e=[.70,1.74,.63]
    if thumb=='behind':m=[.95,-.11,.27];t=[.89,.58,.48];e=[.78,1.37,.59]
    if thumb=='fist': m=[.97,-.01,.56]; t=[.66,.40,.86]; e=[-.26,.70,1.05]
    if thumb=='under': m=[.84,-.20,.42]; t=[.57,-.05,.66];e=[-.47,.035,.68]
    if thumb=='cup':
        m=np.array([.88,-.26,.48]); e=[.98,-.03+u(-.2,.22),1.57]; t=[1.,-.21,1.07]
    if thumb=='pinch':m=[.92,-.07,.30];t=[1.09,.38,.80];e=[1.22,.68,1.42]
    if thumb in ('three','two'):
        px=-.48 if thumb=='three' else -.04
        m=[.78,-.20,.45];t=[.30,.21,.72];e=[px,.70,.85]
    m=np.array(m);t=np.array(t)+u(-.055,.055,3); e=np.array(e)+u(-.07,.07,3)
    pts+= [part(b,m,.325,.295),part(m,t,.295,.24,2),part(t,e,.24,.185,3)]
    # Wrist to forearm. Its oblique direction varies with pose, but the
    # camera convention is a forearm coming up from below / from the side.
    ya,pi,ro=np.array([yaw,pitch,roll])*rad
    C,S=np.cos,np.sin
    R=np.array([[C(ro),-S(ro),0],[S(ro),C(ro),0],[0,0,1]])@np.array([[1,0,0],[0,C(pi),-S(pi)],[0,S(pi),C(pi)]])@np.array([[C(ya),0,S(ya)],[0,1,0],[-S(ya),0,C(ya)]])
    # Separate the long proximal finger fan from the shorter palm; in
    # particular the heel and the arm join behind, not in front of it.
    if c in 'GH':
        # Viewed from above, with the index on the upper radial border;
        # flexion of the other digits then projects toward the floor.
        fore=u(-.2,.50); twist=u(13,74)*rad; theta=u(-6,22)*rad
        rf=np.array([np.cos(fore),0,np.sin(fore)]); right=np.array([np.sin(fore),0,-np.cos(fore)])*np.cos(twist)+np.array([0,1,0])*np.sin(twist)
        R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])@np.stack([right,rf,np.cross(right,rf)],1)
    halfwidth=u(.82,1.10); palmheight=u(.94,1.18); R=R@np.diag([halfwidth,palmheight,1.])
    if c in 'GH':
        # The extended ray of the hand is hinged relative to the
        # metacarpus, not collinear with the forearm. Stabilize the grip
        # by tilting the heel down without changing the pointing direction.
        angle=u(22,55)*rad; bend=np.array([[C(angle),-S(angle),0],[S(angle),C(angle),0],[0,0,1]])
        rr=bend@R; pp=[]
        for j,(a,b,r,s,k) in enumerate(pts):
            finger=j//3
            if finger==3 or (c=='H' and finger==2):
                anchor=base[finger]; invrot=R.T@bend.T@R
                invrot=invrot/np.maximum(.001,np.linalg.norm(invrot,axis=0,keepdims=True));a=anchor+invrot@(a-anchor);b=anchor+invrot@(b-anchor)
            pp.append((a,b,r,s,k))
        R=rr;pts=pp
    ell.append((np.array([-.49,-.17,.035]),np.array([.41,.79,.37])))
    ell=[(R@q,R@np.diag(sc)) for q,sc in ell];pts=[(R@a,R@b,r,s,k) for a,b,r,s,k in pts]; att=R@attach; w=R@arm
    # The forearm can flex at the wrist rather than extending straight
    # back along the axis of the metacarpals (especially in Q and G).
    if c=='Q': w=att+np.array([u(-.48,-.08),u(-1.15,-.88),-.5]); direct=np.array([u(-.52,-.10),-1,-.7]);
    elif c in 'GH': direct=np.array([u(-1.1,-.27),-1,u(-.45,-.05)])
    elif c=='P': direct=np.array([u(.25,.85),-1,-.21])
    else: direct=np.array([u(-.52,.02),-1,-.23])
    att[2]-=.25; w[2]-=.15
    armpts=[part(att,w,.565,.465),part(w,w+direct*6,.465,.86)]
    return pts,ell,armpts,w


def render(c,rng,n=48):
    u=rng.uniform
    pts,el,ap,w=pose(c,rng)
    # Frame from the 3D posed hand itself, not from any measurements on photos.
    lim=[]
    for a,b,r,s,k in pts+ap[:1]:lim.extend([a[:2]-r,a[:2]+r,b[:2]-s,b[:2]+s])
    if c=='Q': lim.append(w[:2]+np.array([-.20,-1.]))
    for a,m in el:
        t=np.sqrt(np.sum(m*m,1)); lim.extend([a[:2]-t[:2],a[:2]+t[:2]])
    lo=np.min(lim,0);hi=np.max(lim,0); span=(hi-lo).max(); sc=span*u(.99,1.23)
    mid=(hi+lo)*.5+np.array([0.,-.035*sc]); mid+=u(-.055,.055,2)*sc
    t=((np.arange(n,dtype='f')+.5)/n-.5)*sc;xx=(mid[0]+t[None,:]).astype('f'); yy=(mid[1]-t[:,None]).astype('f'); step=sc/n
    z=np.full((n,n),-8.,np.float32); tone=np.zeros_like(z)
    lightx=u(-.89,-.39); lighty=u(.05,.67); lightz=np.sqrt(max(.09,1-lightx**2-lighty**2));amb=u(-.21,.035);amp=u(.33,.58)
    def surface(d,nx,ny,nz,tint=0):
        lit=lightx*nx+lighty*ny+lightz*nz
        val=amb+amp*(lit-.47)+tint
        use=d>z
        np.copyto(tone,val,where=use); np.maximum(z,d,out=z)
    def tube(p,q,r,s,k):
        v=q-p; l2=v@v; planar=v[:2]@v[:2]; dx=xx-p[0]; dy=yy-p[1]
        if planar<1e-7:planar=1e-7
        f=(dx*v[0]+dy*v[1])/planar;fs=np.clip(f,0,1);rr=r+(s-r)*fs
        rem=rr**2-(dx-f*v[0])**2-(dy-f*v[1])**2
        f=np.clip(f+v[2]*np.sqrt(np.maximum(rem,0)/(l2*planar)),0,1)
        rr=r+(s-r)*f; nx=(dx-f*v[0])/rr; ny=(dy-f*v[1])/rr; h=1-nx*nx-ny*ny;nz=np.sqrt(np.maximum(h,0)); zz=p[2]+f*v[2]+rr*nz
        # Transverse joint folds darken flexed digits. Keep texture weaker
        # than shading and never draw an artificial line on every joint.
        seam= -.046*np.exp(-((f-.16)/.12)**2) if k==2 else 0
        surface(np.where(h>0,zz,-9),nx,ny,nz,seam)
    for args in ap:tube(*args)
    for p,m in el:
        S=m@m.T; Q=np.linalg.inv(S); dx=xx-p[0];dy=yy-p[1]; bb=Q[0,2]*dx+Q[1,2]*dy; h=bb*bb-Q[2,2]*(Q[0,0]*dx*dx+2*Q[0,1]*dx*dy+Q[1,1]*dy*dy-1); dz=(np.sqrt(np.maximum(h,0))-bb)/Q[2,2]; nx=Q[0,0]*dx+Q[0,1]*dy+Q[0,2]*dz;ny=Q[1,0]*dx+Q[1,1]*dy+Q[1,2]*dz;nz=bb+Q[2,2]*dz; nn=np.sqrt(nx*nx+ny*ny+nz*nz)+.00001
        surface(np.where(h>0,p[2]+dz,-9),nx/nn,ny/nn,nz/nn)
    for args in pts:tube(*args)
    mask=(z>-8).astype('f'); zz=np.maximum(z,-1.65)
    # Ambient occlusion just behind raised digits / palm edge.
    shadow=np.zeros_like(z)
    for j,i in ((0,-2),(-2,0),(2,0),(0,2)):
        shadow+=np.clip(np.roll(zz,(j,i),(0,1))-zz-.08,0,.58)
    tone-=shadow*.24*mask
    # Cast shadows from protruding fingers over the palm (and other digits).
    shade=np.zeros_like(z)
    for dist in (1.5,3.0,5.,8.):
        j=round(lighty*dist);i=round(-lightx*dist);d=np.roll(zz,(j,i),(0,1))-zz-step*dist*lightz
        shade=np.maximum(shade,np.clip(d/.24,0,1))
    tone-=shade*.17*mask
    return tone,mask

# Same image descriptors in the generator and the restricted solution.
common=r'''
import numpy as np

def blur(a):
    v=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    t=(v[:,:,:-2]+v[:,:,2:]+2*v[:,:,1:-1])*.25
    return (t[:,:-2]+t[:,2:]+2*t[:,1:-1])*.25

def deriv(a):
    v=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return .5*(v[:,1:-1,2:]-v[:,1:-1,:-2]),.5*(v[:,2:,1:-1]-v[:,:-2,1:-1])

def background(x):
    # Suppress slowly varying illumination with a robust local estimate
    # from the two upper corners: prefer the less textured extrapolation.
    n=len(x);t=(np.arange(28,dtype=np.float32)-13.5)/28
    patches=[x[:,:6,:6],x[:,:6,-6:],x[:,6:12,:5],x[:,6:12,-5:]]
    v=np.stack([np.median(p.reshape(n,-1),axis=1) for p in patches],axis=1)
    mad=np.stack([np.mean(abs(p-v[:,i,None,None]),axis=(1,2)) for i,p in enumerate(patches)],1)
    edge=np.mean(mad,1); dt=np.clip(v[:,1]-v[:,0],-.35,.35); reliable=1/(1+(np.maximum(mad[:,0],mad[:,1])/(.024+.18*abs(dt)))**2)
    dx=dt*reliable/.78; conf=1/(.010+mad)**2
    off=(v[:,0]*conf[:,0]+v[:,1]*conf[:,1])/(conf[:,0]+conf[:,1]); off=off+dx*.39*(conf[:,0]-conf[:,1])/(conf[:,0]+conf[:,1]);
    dy=np.clip((v[:,2:]-v[:,:2])/.21,-.35,.35); wy=1/(.015+mad[:,:2]+mad[:,2:]+.17*abs(dy))**2;dy=np.sum(dy*wy,1)/np.sum(wy,1)
    return off[:,None,None]+dx[:,None,None]*t[None,None,:]+dy[:,None,None]*np.clip(t+.39,-.2,.69)[None,:,None]

def fillmask(x):
    # Local tangents to the smooth background, cast inwards from several
    # borders. A ray loses credibility on crossing a textured surface.
    dx,dy=deriv(blur(x)); mag=np.sqrt(dx*dx+dy*dy)
    sm=blur(x); n=len(x); t=np.arange(28,dtype=np.float32)[None,:,None]
    def ray(v,gg):
        slope=np.clip(np.median(v[:,2:5]-v[:,:3],axis=1,keepdims=True)*.5,-.014,.014)
        b=np.median(v[:,:4]-slope*(t[:,:4]-1.5),axis=1,keepdims=True)
        er=np.abs(v-(b+slope*(t-1.5)));peak=np.maximum.accumulate(gg,axis=1)
        return er+.32*peak+1.4*np.mean(gg[:,:3],axis=1,keepdims=True)+.30*np.maximum.accumulate(er,axis=1)
    top=ray(sm,mag);left=ray(sm.transpose(0,2,1),mag.transpose(0,2,1)).transpose(0,2,1)
    right=ray(sm[:,:,::-1].transpose(0,2,1),mag[:,:,::-1].transpose(0,2,1)).transpose(0,2,1)[:,:,::-1]
    loc=np.minimum(np.minimum(top,left),right)
    residual=np.minimum(abs(x-background(x)),loc+.018);
    # Dark or light deviations and fine surface detail are both useful.
    contrast=np.mean(abs(residual),axis=(1,2),keepdims=True)+.03
    e=np.clip(abs(residual)/(.055+.37*contrast)-.20,0,1)
    texture=np.clip(blur(mag)/(.024+.09*contrast)-.22,0,1)
    m=np.maximum(e*.94,texture*.91)
    # Penalize large closed-looking regions supported only by a gradual
    # background slope. Local curvature/texture survives illumination ramps.
    sm=blur(x); low=sm
    for j in range(5):low=blur(low)
    local=np.minimum(1,abs(x-low)/(.020+.05*contrast))
    rad=blur(blur(np.maximum(local,texture)))
    return blur(m)*(.55+.45*np.minimum(1,rad*1.65))

def pool(a):
    v=np.pad(a,((0,0),(2,2),(2,2)),mode='edge')
    return v.reshape(len(a),8,4,8,4).mean(axis=(2,4))

def fine(a):
    return a.reshape(len(a),14,2,14,2).mean(axis=(2,4))

def desc(m,x):
    # Occupancy plus undirected, contrast-normalized local edge histograms.
    # Occupancy gets a signed correlation; orientation evidence saturates
    # so that a sleeve or stray background edge cannot dominate the score.
    dx,dy=deriv(blur(x)*.55+x*.45); mag=np.sqrt(dx*dx+dy*dy+1e-10); scale=np.mean(mag,axis=(1,2),keepdims=True)+.009
    amp=np.minimum(mag/(scale*.9),3.5)**.73
    dx=dx/mag;dy=dy/mag
    chans=[]
    for ca,sa in ((1,0),(.70710678,.70710678),(0,1),(-.70710678,.70710678)):
        chans.append(pool(amp*np.maximum(0,abs(dx*ca+dy*sa)-.7)*3.33333))
    # Give the lower corners less weight since forearm and clothing vary.
    wt=np.array([[1,1,1,1,1,1,1,1]]*5+[[.85,.95,1,1,1,1,.95,.85],[.60,.78,.88,.9,.9,.88,.78,.60],[.44,.6,.66,.7,.7,.66,.6,.44]],np.float32)
    p=pool(m);p=(p-np.mean(p,axis=(1,2),keepdims=True))*wt
    h=np.stack(chans,1);h=(h-.66*np.mean(h,axis=(2,3),keepdims=True))*wt
    p=p.reshape(len(p),-1); h=h.reshape(len(h),-1)
    p=p/np.sqrt(np.sum(p*p,axis=1,keepdims=True)+.001);h=h/np.sqrt(np.sum(h*h,axis=1,keepdims=True)+.001)
    return np.ascontiguousarray(np.concatenate([.74*p,.673*h],axis=1),np.float32)
'''

inference=r'''
def view(a,dx,dy,scale):
    # Transform one input view to a canonical crop, clamping out-of-frame pixels.
    t=(np.arange(28,dtype=np.float32)-13.5)*scale+13.5
    x=np.clip(t+dx,0,27);y=np.clip(t+dy,0,27);i=x.astype(np.int32);j=y.astype(np.int32);sx=(x-i).astype(np.float32);sy=(y-j).astype(np.float32)
    v=a[:,:,i]*(1-sx)+a[:,:,np.minimum(i+1,27)]*sx
    return v[:,j,:]*(1-sy[:,None])+v[:,np.minimum(j+1,27),:]*sy[:,None]

bank=CONSTANTS['poses'].astype(np.float32).reshape(-1,320).T.copy()
fd=CONSTANTS['fine']

def localnorm(a):
    a=a-a.mean(axis=2,keepdims=True)
    norm=np.sqrt(.14+np.sum(a*a,axis=(1,2),keepdims=True))
    return a/norm

partweight=np.array([1.05,1.25,1.05,1.10,1.1,1.10,.65,.65,.65],np.float32)

def partscore(q,m):
    # Slide overlapping palm/finger regions independently by a single cell.
    # Normalize inside each region so a shirt or an unrelated edge in
    # another part of the image cannot suppress the evidence for a digit.
    model=np.zeros((len(m),9),np.float32); confidence=np.zeros_like(model)
    qpad=np.pad(q,((0,0),(0,0),(1,1),(1,1)),mode='edge'); ci=0
    for y in (0,4,8):
        for x in (0,4,8):
            v=m[:,:,y:y+6,x:x+6].reshape(len(m),6,36);v=v-v.mean(axis=2,keepdims=True)
            energy=np.sum(v*v,axis=(1,2))
            qq=q[:,:,y:y+6,x:x+6].reshape(len(m),6,36);qq=qq-qq.mean(axis=2,keepdims=True)
            seen=np.sum(qq*qq,axis=(1,2));confidence[:,ci]=np.minimum(1,(energy+.85*seen)/4+.08)*partweight[ci]
            v=v/np.sqrt(.15+energy[:,None,None]); s=np.full(len(m),-1.,np.float32)
            for iy,ix in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
                w=qpad[:,:,y+1+iy:y+7+iy,x+1+ix:x+7+ix].reshape(len(m),6,36)
                h=np.sum(localnorm(w)*v,axis=(1,2))-.045*(abs(iy)+abs(ix));s=np.maximum(h,s)
            model[:,ci]=s;ci+=1
    return np.sum(model*confidence,1)/(np.sum(confidence,1)+.001)
def predict(x):
    result=np.empty(len(x),np.int64)
    count=bank.shape[1]//24
    for st in range(0,len(x),16):
        # A fixed block size also fixes the reduction kernels across batch sizes.
        n=min(16,len(x)-st);a=np.zeros((16,28,28),np.float32);a[:n]=x[st:st+n,0];m=fillmask(a)
        f=[]
        for scale,delta in ((1.,2.8),(.82,3.),(1.21,2.8)):
            for dx in (-delta,0,delta):
                for dy in (-delta,0,delta):
                    f.append(desc(view(m,dx,dy,scale),view(a,dx,dy,scale)))
        scores=(np.concatenate(f)@bank).reshape(len(f),16,24,count)
        # Rerank a shortlist for each sign on the high-resolution grid.
        # Different signs compete with the same number of verifications.
        best=np.max(scores,axis=0); which=np.argmax(scores,axis=0); keep=6
        sel=np.argpartition(best,count-keep,axis=2)[:,:,-keep:];vv=np.take_along_axis(which,sel,2)
        top=np.take_along_axis(best,sel,2);f=[]
        for scale,delta in ((1.,2.8),(.82,3.),(1.21,2.8)):
            for dx in (-delta,0,delta):
                for dy in (-delta,0,delta):
                    f.append(detailed(view(m,dx,dy,scale),view(a,dx,dy,scale)))
        chan=np.array([1.9,.64,.87,.87,.87,.87],np.float32)[None,:,None,None]; fineviews=np.stack(f,1);rows=np.arange(24)[:,None]
        for idx in range(n):
            vec=fd[rows,sel[idx]].astype(np.float32).reshape(-1,7,14,14)/42
            q=fineviews[idx,vv[idx]].reshape(-1,7,14,14)
            hole=q[:,6];ring=vec[:,6];support=ring.sum((1,2));observed=hole.sum((1,2));overlap=(ring*hole).sum((1,2))
            evidence=(1.7*overlap/(.9+support+observed)-.18*np.minimum(1,support/3)).reshape(24,keep)
            vec=vec[:,:6];q=q[:,:6];t=partscore(q*chan,vec*chan).reshape(24,keep)
            # A global symmetric comparison penalizes an observed finger in
            # the blank area of a fist model (or a model finger in air).
            q=q-q.mean(axis=(2,3),keepdims=True);vec=vec-vec.mean(axis=(2,3),keepdims=True)
            q=q*chan;vec=vec*chan
            rigid=(np.sum(q*vec,axis=(1,2,3))/np.sqrt(.01+np.sum(q*q,axis=(1,2,3))*np.sum(vec*vec,axis=(1,2,3)))).reshape(24,keep)
            both=.30*top[idx]+.40*t+.30*rigid+.085*evidence
            result[st+idx]=np.argmax(.80*np.max(both,1)+.20*np.mean(both,1))
    return result
'''

# Six pooled maps retain 2D layout for bounded, local deformation.
common+=r'''
def cavities(m):
    # Flood background in from outside: the interior of the O and F
    # contacts cannot be reached except through a high foreground barrier.
    free=1-np.clip(m,0,1);flow=np.zeros_like(free)
    flow[:,0]=free[:,0];flow[:,-1]=free[:,-1];flow[:,:,0]=free[:,:,0];flow[:,:,-1]=free[:,:,-1]
    for j in range(17):
        p=np.pad(flow,((0,0),(1,1),(1,1)),mode='edge')
        flow=np.minimum(free,np.maximum(np.maximum(p[:,1:-1,:-2],p[:,1:-1,2:]),np.maximum(p[:,:-2,1:-1],p[:,2:,1:-1])))
    return np.maximum(0,free-flow)

def detailed(m,x):
    sx=blur(x)*.6+.4*x;dx,dy=deriv(sx);g=np.sqrt(dx*dx+dy*dy+1e-9)
    low=sx
    for j in range(6):low=blur(low)
    sc=np.mean(g,axis=(1,2),keepdims=True)+.007
    channels=[fine(m),fine(np.clip((sx-low)/sc,-3.,3.))]
    for ca,sa in ((1,0),(.70710678,.70710678),(0,1),(-.70710678,.70710678)):
        channels.append(fine(blur(np.minimum(3.,np.maximum(0,(abs(dx*ca+dy*sa)-.64*g)*3/sc)))))
    channels.append(fine(cavities(m)))
    return np.stack(channels,1)
'''

def sample_grid(a):
    n,s,s=a.shape;t=(np.arange(28)+.5)*s/28-.5;i=t.astype(int);f=(t-i).astype('f');v=a[:,:,i]*(1-f)+a[:,:,i+1]*f
    return v[:,i,:]*(1-f[:,None])+v[:,i+1,:]*f[:,None]

def make(out,count):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'solution.py').write_text(common+inference)
    env={};exec(common,env);rng=np.random.default_rng(1729); data=[];fine=[]
    for c in letters:
        chunk=[];bank=[];f=[]
        for k in range(count):
            chunk.append(render(c,rng))
            if len(chunk)==32 or k==count-1:
                x,m=map(np.stack,zip(*chunk)); x=sample_grid(env['blur'](x));m=sample_grid(env['blur'](m));bank.append(env['desc'](m,x));chunk=[]
                v=env['detailed'](m,x);f.append(np.round(v*42).astype('i1'))
        data.append(np.concatenate(bank).astype('f2'));fine.append(np.concatenate(f))
    np.savez(out/'constants.npz',poses=np.stack(data),fine=np.stack(fine))

if __name__=='__main__':make(sys.argv[1],1024)
