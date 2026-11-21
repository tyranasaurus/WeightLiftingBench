"""Data-free ASL hand renderer and template network. Usage: python3 build.py DEST"""
import sys, json, math
from pathlib import Path
import numpy as np

LETTERS='ABCDEFGHIKLMNOPQRSTUVWXY'
# All anatomical dimensions below are in approximately palm half-widths, in a
# right hand's local frame: u towards the thumb, v towards the fingers, z palmar.
# flexion angles turn a phalanx out of the palm plane towards the camera.
def geom(letter,rng):
    s=rng.uniform
    hand=[]
    def ell(p,rad,kind='skin',rot=0):
        a=np.deg2rad(rot); c,t=np.cos(a),np.sin(a)
        axes=np.array([[c,-t,0],[t,c,0],[0,0,1]])*np.array(rad)
        hand.append(('ell',np.array(p),axes,kind))
    def tube(pts,rad,kind='skin'):
        pts=np.array(pts,dtype=float); rad=np.broadcast_to(rad,(len(pts),))
        # Radii are full 3D soft tissue about the bones.
        for i,(a,b,r,R) in enumerate(zip(pts[:-1],pts[1:],rad[:-1],rad[1:])):
            hand.append(('tube',a,b,float(r),float(R),'joint' if kind=='finger' and i else kind))
    # Broad forearm tapering into the wrist; eminences are kept shallow, so
    # individual digits can occlude them. The lower forearm may carry a cuff.
    ell((0,-.07,0),(1.00,1.12,.365))
    ell((.48,-.06,.125),(.48,.91,.32),'skin',-22)
    ell((-.56,-.04,.075),(.32,.83,.28),'skin',13)
    w=s(.48,.62); dx=s(-.38,.20)
    tube([(dx*2,-5,-.13),(dx,-2.3,-.07),(-.07,-1.0,-.03),(-.08,-.69,0)],
         [w*1.58,w*1.15,w,w*1.08],'arm')
    if rng.random()<.25:
        y=s(-2.55,-1.36);tube([(dx*2,-5,-.12),(dx*(abs(y)-1),y,-.08)],[w*1.72,w*1.35],'sleeve')
    roots=np.array([[-.74,.45,.02],[-.275,.69,.035],[.20,.77,.015],[.665,.65,.015]])
    lengths=np.array([1.47,1.83,1.99,1.84]) * s(.91,1.12)
    thick=np.array([.196,.231,.244,.231])*s(.91,1.10)
    # Neutral grasp with flexed distal joints; an open digit uses all three
    # phalanges in almost the same direction. Ordering: little, ring, middle, index.
    fold=np.array([36,120,195.],float); straight=np.array([0,2,4.],float)
    angles=np.tile(fold,(4,1)); spread=np.array([-3.,-2.,0.,1.])
    thumb='over'; custom={}
    def ext(ids):
        for i in ids: angles[i]=straight
    if letter=='A':
        thumb='along';angles[:]=[34,133,199]
        roots[:,1]+=np.linspace(-.04,.10,4)
    elif letter=='B':
        ext(range(4)); spread[:]=[3,1,0,-2];thumb='tucked'
    elif letter in 'CO':
        # Open C and closed O: all four fingers arch towards the thumb.
        if letter=='C': base=[5,48,87];thumb='C'
        else: base=[23,81,137];thumb='O'
        for i in range(4):angles[i]=np.array(base)+(2-i)*np.array([4,3,2]);spread[i]*=.4
    elif letter=='D':
        angles[:3]=[30,96,151];ext([3]);thumb='D'
    elif letter=='E':angles[:]=[7,101,185];thumb='under'
    elif letter=='F':
        ext([0,1,2]);spread[:]=[-23,-4,19,0];angles[3]=[30,96,148];thumb='F'
    elif letter in 'GH':
        ext([3] if letter=='G' else [2,3]);thumb='G' if letter=='G' else 'H'
    elif letter=='I': ext([0]);spread[0]=-9;thumb='over'
    elif letter=='K':
        ext([2,3]);angles[2]=[42,47,49];spread[2]=-16;spread[3]=7;thumb='K'
    elif letter=='L':ext([3]);spread[3]=0;thumb='L'
    elif letter in 'MN':
        n=3 if letter=='M' else 2
        for i in range(4):
            if i>=4-n:
                angles[i]=np.array([18,115,184])+s(-1,1)*np.array([8,17,14]);roots[i,1]+=.07
                # The thumb under the fingers lifts them; the middle and distal
                # phalanges descend across it towards the opposite palm edge.
                custom[i]=-.21
            else:angles[i]=[56,143,209];roots[i,1]-=.10
        thumb=letter
    elif letter=='P':
        ext([2,3]);angles[2]=[50,59,60];spread[2]=-15;thumb='K'
    elif letter=='Q':
        angles[3]=[9,16,20];thumb='Q'
    elif letter=='R':
        ext([2,3]);spread[2]=10;spread[3]=-12;angles[3]=[7,7,2];thumb='over'
    elif letter=='S':angles[:]=[48,141,209];thumb='fist'
    elif letter=='T':
        # Side view of an inserted thumb, with the index over its base.
        angles[3]=[25,107,182];thumb='T'
    elif letter in 'UVW':
        ext([2,3] if letter!='W' else [1,2,3]);thumb='over'
        if letter=='U': spread[2:]=[1,-1]
        elif letter=='V':spread[2:]=[-15,16]
        else:spread[1:]=[-21,0,20]
    elif letter=='X':angles[3]=[1,54,111];thumb='fist'
    elif letter=='Y':ext([0]);spread[0]=-22;thumb='L'

    # Thumb poses are explicit articulated paths from the thenar eminence,
    # independent of the crop, scale and viewing direction.
    thumbs={
      'along':[(.89,.10,.10),(1.06,.70,.24),(1.11,1.46,.37)],
      'tucked':[(.83,.12,.46),(.37,.52,.54),(-.14,.65,.55)],
      'over':[(.82,.23,.42),(.32,.68,.74),(-.29,.86,.84)],
      'fist':[(.81,.10,.48),(.54,.76,.91),(.04,1.08,1.19)],
      'under':[(.87,.14,.39),(.40,.56,.62),(-.23,.50,.71)],
      'M':[(.81,.24,.36),(.18,.81,.59),(-.56,.88,.69)],
      'N':[(.81,.23,.36),(.26,.80,.60),(-.11,1.02,.68)],
      'T':[(.78,.23,.30),(.49,.99,.56),(.44,1.72,.54)],
      'D':[(.90,.37,.43),(.65,.94,.79),(.20,1.12,1.04)],
      'F':[(.94,.47,.42),(.92,1.02,.82),(.70,1.32,1.02)],
      'C':[(.88,.00,.37),(1.0,.17,1.03),(1.05,.47,1.61)],
      'O':[(.89,.12,.41),(.98,.53,.91),(.86,.98,1.15)],
      'G':[(.75,.27,.22),(.81,.88,.27),(.84,1.46,.28)],
      'H':[(.81,.19,.22),(.70,.66,.58),(.25,.94,.73)],
      'K':[(.82,.32,.36),(.56,.95,.53),(.23,1.51,.71)],
      'L':[(.99,-.01,.16),(1.48,.23,.24),(1.99,.54,.27)],
      'Q':[(.93,.30,.40),(1.05,.96,.74),(1.05,1.54,1.10)]}
    # Finger joints move coherently between signers; no noise measured on data.
    grasp=s(-8,8);wide=s(.80,1.23)
    for i in range(4):
        a=angles[i].copy();bent=(max(a)>65)
        a+=rng.normal(0,[3,3,4])+(grasp if bent else grasp*.32)
        if letter in 'COPQKFX' and i in (2,3):a+=s(-9,9)
        az=(spread[i]*wide+s(-2.0,2.0))*math.pi/180
        flex=np.deg2rad(a); d=np.stack((np.sin(az)*np.cos(flex),np.cos(az)*np.cos(flex),np.sin(flex)),1)
        if i in custom:d[1:,0]+=custom[i];d/=np.sqrt((d*d).sum(1,keepdims=True))
        root=roots[i].copy();root[1]+=s(-.04,.04)
        le=lengths[i]*s(.96,1.04);pts=root+np.vstack((np.zeros(3),np.cumsum(d*(np.array([.43,.32,.25])*le)[:,None],0)))
        tube(pts,thick[i]*np.array([1.0,.98,.88,.72]),'finger')
    tp=np.array([(.65,-.47,.09)]+thumbs[thumb]);tp[1:]+=rng.normal(0,.065,(3,3));tube(tp,np.array([.35,.302,.231,.173])*s(.95,1.07),'finger')
    return hand


def view(l,rng):
    # yaw +90 projects the forward curl horizontally; roll +90 points the
    # longitudinal hand axis right. Palm normals turn away for dorsolateral signs.
    # T also has a conventional side-on interpretation with the forefinger
    # projecting right and the thumb pointing up (specified in geom_side below).
    params={
      'A':(10,7,12),'B':(3,0,0),'C':(70,0,5),'D':(18,1,6),'E':(5,9,0),
      'F':(43,-1,8),'G':(-41,-17,77),'H':(-82,-6,78),'I':(2,0,7),
      'K':(-47,-4,0),'L':(1,1,5),'M':(14,13,14),'N':(9,15,15),
      'O':(66,0,10),'P':(40,19,-91),'Q':(46,10,-145),'R':(-28,0,4),
      'S':(-12,15,4),'T':(100,4,84),'U':(-19,0,5),'V':(-6,1,3),
      'W':(-4,0,4),'X':(55,4,7),'Y':(1,3,6)}
    yaw,pitch,roll=params[l];yaw+=rng.uniform(-22,22);pitch+=rng.uniform(-12,12);roll+=rng.uniform(-14,14)
    cy,cp,cr=np.cos(np.deg2rad([yaw,pitch,roll]));sy,sp,sr=np.sin(np.deg2rad([yaw,pitch,roll]))
    # Coordinates are x right, y UP, z towards viewer.
    return np.array([[cr,sr,0],[-sr,cr,0],[0,0,1]])@np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])

# Special side/down signs: parametric projected bone poses. For horizontal
# pointing signs wrist flexion changes forearm direction independently of the
# flexion of the finger joints. Use an explicit joint chain for this family.
# Coordinates below are still anatomical units measured from the palm centre.
def geom_side(l,rng):
    s=rng.uniform
    hand=[]
    def ell(pos,rx,ry,rz,angle=0):
        c,t=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)); hand.append(('ell',np.array(pos),np.array([[c,-t,0],[t,c,0],[0,0,1]])*np.array([rx,ry,rz]),'skin'))
    def tube(pts,rad,kind='finger'):
        p=np.array(pts,dtype=float)
        for a,b,r,R in zip(p[:-1],p[1:],rad[:-1],rad[1:]):hand.append(('tube',a,b,r,R,kind))
    def bone(start,dirs,lens,radii,zdir):
        a=np.deg2rad(dirs);ds=np.stack([np.cos(a),np.sin(a),zdir],1)*np.array(lens)[:,None]
        pts=start+np.vstack([np.zeros(3),ds.cumsum(0)]); tube(pts,radii)
    g=s(-7,7); long=s(.90,1.12); t=s(.91,1.1)
    if l in 'GHT':
        # In a pointing hand the metacarpal of the index continues into its
        # first phalanx; the palm is longitudinal rather than a vertical fist.
        sideon=s(.3,1.0) if l=='H' else s(.45,1.0)
        # Radial deviation of the wrist changes the pitch of the back of the
        # hand relative to the fixed pointing direction.
        pitch=s(23,62) if l in 'GT' else s(-2,29)
        ell((-.13,0,0),1.07,.84,.46,pitch)
        fall=s(1.2,4.8);tube([(-4.0,-fall,-.10),(-1.5,-.43*fall,-.07),(-.69,-.46,-.01)],[.84,.60,.56],'arm')
        # Index along the upper edge, curled columns below. The second straight
        # finger of H is deliberately allowed to overlap the first in projection.
        for i in range(4):
            start=np.array([.28+i*.13,(-.38+i*.32)*sideon,.10+i*.025]); r=t*(.185+i*.014)
            if l=='H' and i==2:start[1]-=.14;start[2]+=.19
            if i==3 or i==2 and l=='H':
                a=s(-4,6)+g+(0 if i==3 else 5); ln=long*(1.85+(3-i)*.12)
                bone(start,[a,a+1,a+s(-4,4)],ln*np.array([.45,.30,.25]),np.array([1,.97,.86,.68])*r,[.02,.04,.06]);
            else:
                bone(start,[0,-61,-169],[.58,.38,.37],np.array([1,1,.91,.74])*r,[.14,.60,.14]);
        if l=='G':
            # Thumb runs parallel behind the index; the shorter digit can just
            # peek above it or be almost concealed.
            tube([(-.38,-.01,-.28),(-.04,.41,-.33),(.55,.61,-.29),(1.16,.76,-.29)],[.33,.28,.21,.16]);
        if l=='H':tube([(-.17,.17,.38),(.29,.33,.51),(.75,.43,.54)],[.28,.24,.17]);
        if l=='T':
            tip=.66*sideon+s(.45,.80);tube([(.10,-.16,.34),(.27,.35,.38),(.36,tip-.36,.35),(.43,tip,.31)],[.32,.27,.21,.155]);
    elif l=='P':
        ell((0,0,0),.90,.69,.46,-42)
        tube([(2.52,-3.42,-.1),(1.28,-1.54,-.07),(.43,-.43,0)],[.82,.61,.56],'arm')
        # P is K tipped forward/down: extended index projecting left, and
        # middle finger directed towards ground. Thumb braces the middle finger.
        for i in range(2):
            start=np.array([.25-i*.32,-.2-i*.03,.14]);bone(start,[205,246,345],[.54,.41,.33],[.22,.21,.18,.15],[.25,.52,0])
        start=np.array([-.48,.36,-.09]);bone(start,[162+g,166+g,169+g],long*np.array([.87,.59,.46]),[.23,.216,.18,.14],[-.06,-.03,0]);
        # The middle finger points more toward the lens than the index;
        # orthographic foreshortening changes its apparent length.
        a=s(225,267);foresh=s(.32,.96);bone(np.array([-.58,.04,.02]),[a,a+5,a+8],foresh*long*np.array([.83,.62,.47]),[.24,.22,.19,.15],np.full(3,np.sqrt(1-foresh**2)/foresh));
        tube([(.52,.02,.25),(.06,.26,.45),(-.39,.03,.61),(-.57,-.43,.50)],[.32,.28,.22,.16])
    else: # Q: the bent wrist is above the digits, which point down.
        ell((0,.0,0),.92,.89,.50,-23)
        wrist=s(-.65,.65);tube([(wrist,-3.39,-.78),(.17+wrist*.45,-1.62,-.55),(.25,-.28,-.32)],[.81,.54,.53],'arm')
        for i in range(3):
            st=np.array([.55-i*.30,.25-i*.08,.12]);bone(st,[-57,-87,-181],[.59,.52,.34],[.24,.23,.205,.16],[.30,.35,.08]);
        a=s(238,261);bone(np.array([-.53,.18,.1]),[a+g,a+5+g,a+7+g],long*np.array([.83,.60,.42]),[.24,.22,.19,.15],[.25,.12,.05]);
        tube([(.46,.16,.30),(.63,-.36,.5),(.67,-.91,.63),(.86,-1.47,.70)],[.32,.29,.22,.16]);
    # Vary projected roll and yaw, keeping the side gesture's direction.
    a=np.deg2rad(s(-17,17));b=np.deg2rad(s(-23,23));c=np.deg2rad(s(-15,15));
    transform=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])@np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])@np.array([[1,0,0],[0,np.cos(c),-np.sin(c)],[0,np.sin(c),np.cos(c)]])
    return hand,transform


def primitives(l,rng,direct=True):
    if direct and l in 'GHPQT':h,R=geom_side(l,rng)
    else:h=geom(l,rng);R=view(l,rng)
    R=R@np.diag(rng.uniform([.9,.94,.91],[1.10,1.07,1.12]))
    out=[];bounds=[]
    for kind,p,q,*rest in h:
        p=R@p;q=R@q
        if not direct and kind=='tube' and rest[-1] in ('arm','sleeve') and l in 'GH':
            # Wrist extension lets the forearm enter diagonally while the hand
            # itself points horizontally. Rotate it around the wrist joint.
            joint=R@np.array([-.07,-.97,-.04]);a=np.deg2rad(39.);B=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
            p=joint+B@(p-joint);q=joint+B@(q-joint)
        if kind=='ell':
            rd=np.sqrt((q**2).sum(axis=1));bounds.extend([p[:2]-rd[:2],p[:2]+rd[:2]])
        elif rest[-1] not in ('arm','sleeve'):
            r,t=rest[:2];bounds.extend([p[:2]-r,p[:2]+r,q[:2]-t,q[:2]+t]);
        out.append((kind,p,q,*rest))
    bounds=np.array(bounds);lo,hi=bounds.min(0),bounds.max(0);return out,lo,hi


def render(out,lo,hi,rng,ss=2):
    # Orthographic lighted union of swept spheres and ellipsoids, cropped with
    # space around the hand and a little extra room on the forearm side.
    span=hi-lo; sz=28*ss; sc=rng.uniform(.69,1.24)*sz/max(span[0],span[1]*1.06)
    cx,cy=((lo+hi)*.5); px=(sz-1)*.5+sc*(-cx)+ss*rng.uniform(-2.1,2.1)
    # The crop is square even for a sideways hand. Share surplus vertical
    # space between headroom and the entrance of the wrist instead of pinning
    # the fingertips to a fixed upper margin irrespective of hand aspect.
    top=(sz-span[1]*sc)*rng.uniform(.18,.78)+ss*rng.uniform(-1.8,2.2)
    # A photographer centering a horizontal pointing hand can crop the
    # forearm rather than the fingertip. Allow horizontal hands to occupy
    # the lower part of the frame as well as the upper-central region.
    aspect=max(0.,min(1.,span[0]/span[1]-1.));top+=ss*aspect*rng.uniform(-1.,5.)
    # The square can also be centered on the palm, rather than on the
    # midpoint between the extremal fingertips. In that case long digits may
    # approach a border, as in a close-up photograph.
    pan=rng.uniform(.38,.86) if rng.random()<.55 else 0.
    px+=sc*cx*pan;py=top+sc*hi[1]-sc*cy*pan
    x=(np.arange(sz)[None,:]-px)/sc;y=(py-np.arange(sz)[:,None])/sc
    depth=np.full((sz,sz),-1e4);nx=np.zeros_like(depth);ny=nx.copy();nz=nx.copy();mat=nx.copy();crease=nx.copy()
    for kind,p,q,*rest in out:
        if kind=='ell':
            S=q@q.T; A=np.linalg.inv(S);rad=np.sqrt(S.diagonal());x0,y0=(p-rad)[:2];x1,y1=(p+rad)[:2]
        else:
            r,t,material=rest;x0,y0=np.minimum(p[:2]-r,q[:2]-t);x1,y1=np.maximum(p[:2]+r,q[:2]+t)
        i=max(0,int(math.floor(x0*sc+px)));j=min(sz,int(math.ceil(x1*sc+px))+1)
        k=max(0,int(math.floor(py-y1*sc)));m=min(sz,int(math.ceil(py-y0*sc))+1)
        if i>=j or k>=m:continue
        u=x[:,i:j]-p[0];v=y[k:m,:]-p[1];sl=np.s_[k:m,i:j]
        if kind=='ell':
            b=(A[0,2]*u+A[1,2]*v);c=A[0,0]*u*u+2*A[0,1]*u*v+A[1,1]*v*v-1
            disc=b*b-A[2,2]*c; z=(np.sqrt(np.maximum(0,disc))-b)/A[2,2];good=disc>=0
            gx=A[0,0]*u+A[0,1]*v+A[0,2]*z;gy=A[1,0]*u+A[1,1]*v+A[1,2]*z;gz=b+A[2,2]*z
            material=rest[0];di=1/np.sqrt(gx*gx+gy*gy+gz*gz+1e-12);gx*=di;gy*=di;gz*=di
        else:
            # For each view ray, solve for the entry into the sides of a tapered
            # capsule by minimizing |ray - (a+t(b-a))|^2 - r(t)^2 over t.
            dx,dy,dz=q-p; dr=t-r; D=max(1e-7,dx*dx+dy*dy+dz*dz-dr*dr)
            h=u*dx+v*dy+r*dr;AA=1-dz*dz/D;B=-dz*h/D;C=u*u+v*v-r*r-h*h/D
            # Endpoint spheres also handle nearly end-on and very short bones.
            c1=r*r-u*u-v*v; c2=t*t-(u-dx)**2-(v-dy)**2
            z1=np.where(c1>=0,np.sqrt(np.maximum(0,c1)),-100.);z2=np.where(c2>=0,dz+np.sqrt(np.maximum(0,c2)),-100.)
            if AA>1e-7:
                dd=B*B-AA*C;zs=(-B+np.sqrt(np.maximum(0,dd)))/AA;f=(h+dz*zs)/D
                zs=np.where((dd>=0)&(f>=0)&(f<=1),zs,-100.);z=np.maximum(np.maximum(z1,z2),zs)
                tpos=np.where(z==zs,np.clip(f,0,1),(z2>z1)*1.)
            else:z=np.maximum(z1,z2);tpos=(z2>z1)*1.
            good=z>-90;den=1/(r+tpos*dr+1e-10);gx=(u-tpos*dx)*den;gy=(v-tpos*dy)*den;gz=(z-tpos*dz)*den
        z=z+p[2];sel=good & (z>depth[sl]);np.copyto(depth[sl],z,where=sel);np.copyto(nx[sl],gx,where=sel);np.copyto(ny[sl],gy,where=sel);np.copyto(nz[sl],gz,where=sel)
        np.copyto(mat[sl],1.0 if material=='sleeve' else 0.0,where=sel)
        if material=='joint':
            # A soft flexion crease in the skin over a finger joint. Position
            # and width are defined about the bone, not in image coordinates.
            f=(u*dx+v*dy+(z-p[2])*dz)/(dx*dx+dy*dy+dz*dz+1e-12)
            fold=.075*np.exp(-((f-.14)/.09)**2)*np.maximum(0,gz)
            np.copyto(crease[sl],fold,where=sel)
        else:np.copyto(crease[sl],0,where=sel)
    fg=depth>-1000
    # Directional lighting, simple depth-based ambient occlusion, gray wall.
    lx=rng.uniform(-1.12,-.38) if rng.random()<.84 else rng.uniform(.2,.94)
    ly=rng.uniform(.38,1.22);lz=rng.uniform(.1,.88);L=np.sqrt(lx*lx+ly*ly+lz*lz)
    dp=(lx*nx+ly*ny+lz*nz)/L
    base=rng.uniform(.27,.49); im=base+.38*dp+.10*nz+.07*np.maximum(dp,0)**4
    # Approximate contact darkening in the narrow seams between curled digits.
    occ=np.zeros_like(im)
    for i,j in [(ss,0),(-ss,0),(0,ss),(0,-ss)]:
        d=np.roll(depth,(i,j),axis=(0,1))-depth;occ+=np.clip(d-.08,0,.75)
    im-=.105*occ+crease;im=np.where(mat>.5,.14+.09*dp,im)
    bg=rng.uniform(.52,.82) if rng.random()<.82 else rng.uniform(.14,.48)
    im=np.where(fg,im,bg)
    return im.reshape(28,ss,28,ss).mean((1,3)),fg.reshape(28,ss,28,ss).mean((1,3))

# Input descriptors: steerable Gaussian derivative pairs and half-wave
# Laplacian-of-Gaussian channels. Both signs of a directional filter have
# separate channels. An unsigned component is mixed in for polarity tolerance.
def filters():
    y,x=np.mgrid[-2:3,-2:3];g=np.exp(-(x*x+y*y)/(2*.86**2));g/=g.sum()
    fil=[]
    for ang in np.arange(4)*np.pi/4:
        a=(x*np.cos(ang)+y*np.sin(ang))*g
        a/=np.sum(abs(a));fil.extend([a,-a])
    a=g*(2-(x*x+y*y)/.86**2);a-=a.mean();a/=abs(a).sum();fil.extend([a,-a]);return np.array(fil)

def texture_pool(v):
    # Divisive normalization in a 3x3 cell neighborhood (soft confidence
    # floor). Implemented in network.json by two linear-spline breakpoints:
    # a weak-gradient term and two orientation shares above local thresholds.
    energy=v[:,:8].sum(1,keepdims=True)/4
    pad=np.pad(energy,[(0,0),(0,0),(1,1),(1,1)])
    near=sum(pad[:,:,i:i+12,j:j+12] for i in range(3) for j in range(3))/9
    # Clipped ramps approximate angular evidence at two relative contrasts.
    # Fixed hard floor and coefficients are analytic, not estimated on images.
    return (.50*v + .40*np.clip(v-.45*near,0,.075)/.075 +
                     .45*np.clip(v-1.05*near,0,.11)/.11)

def template_desc(images,fil):
    # Every operation here acts separately on each rendering, including its
    # normalization. No axis reduction ever includes the render/batch axis.
    ims=np.stack([p[0] for p in images]);support=np.stack([p[1] for p in images]);ims=ims-ims.mean((1,2),keepdims=True);ims/=np.sqrt((ims**2).mean((1,2),keepdims=True)+1e-5)
    win=np.lib.stride_tricks.sliding_window_view(ims,(5,5),axis=(-2,-1))
    v=(win.reshape(-1,25)@fil.reshape(-1,25).T).reshape(len(ims),24,24,-1).transpose(0,3,1,2)
    v=np.maximum(0,v);v=v.reshape(len(v),len(fil),12,2,12,2).mean((3,5));v=texture_pool(v)
    # Mix opposite polarities to make sign-invariant edge channels available
    # without discarding the individual bright/dark side information.
    v=v+.55*v.reshape(len(v),len(fil)//2,2,12,12)[:,:,::-1].reshape(v.shape)
    # Remove the spatial DC response of every channel, then cosine match the
    # whole descriptor. Normalize as one vector to retain orientation energy.
    weight=spatial_weight()
    # Only the hand and a contour-width surround are modeled; exclude the
    # particular render's far background from both centering and matching.
    support=np.lib.stride_tricks.sliding_window_view(support,(5,5),axis=(1,2)).max((3,4))
    support=support.reshape(len(v),1,12,2,12,2).mean((3,5));pad=np.pad(support,[(0,0),(0,0),(1,1),(1,1)]);support=.12+.88*sum(pad[:,:,i:i+12,j:j+12] for i in range(3) for j in range(3))/9
    v-=(v*support).sum((2,3),keepdims=True)/support.sum((2,3),keepdims=True)
    v*=weight*support
    v=v.reshape(len(v),-1);v/=np.sqrt((v*v).sum(1,keepdims=True)+1e-8)
    # Weight the input too for a central field of attention. Apply the same
    # per-channel centering as on the template (transpose of centering).
    v=v.reshape(len(v),len(fil),12,12)*weight;v-=v.mean((2,3),keepdims=True)
    # Transpose of the sign-channel mixing, folded into dense weights.
    v=v+.55*v.reshape(len(v),len(fil)//2,2,12,12)[:,:,::-1].reshape(v.shape)
    # Coarser spatial correlation is folded into each independent template,
    # using the symmetric adjoint blur so neighboring cells can also align.
    pad=np.pad(v,[(0,0),(0,0),(1,1),(1,1)]);smooth=sum(pad[:,:,i:i+12,j:j+12]*a*b for i,a in enumerate([.25,.5,.25]) for j,b in enumerate([.25,.5,.25]))
    v=.40*v+.60*smooth
    v-=v.mean((2,3),keepdims=True)
    v=v.reshape(len(v),-1);v/=np.sqrt((v*v).sum(1,keepdims=True)+1e-8)
    return v.astype('f'),support

def elastic_desc(images,fil):
    v,support=template_desc(images,fil);c=len(fil);v=v.reshape(-1,c,12,12)
    # Four overlapping anatomical subfields, not four separate exemplars. Each
    # subfield can translate by one feature cell. Keep relative contrast of
    # the parts, so areas containing no useful evidence cannot dominate votes.
    y,x=np.mgrid[:12,:12];right=np.clip((x-3.)/5.,0.,1.);bottom=np.clip((y-3.)/5.,0.,1.)
    parts=[]
    for fy in (1-bottom,bottom):
        for fx in (1-right,right):
            window=(fx*fy)[None,None];win=window*support;part=(v-(v*win).sum((2,3),keepdims=True)/np.maximum(win.sum((2,3),keepdims=True),1e-9))*win
            parts.append(part)
    # One normalization for this render only, shared by all its translations.
    scale=1/(sum(np.sqrt((p*p).sum((1,2,3),keepdims=True)+1e-12) for p in parts)*.5)
    res=[]
    for p in parts:
        pp=np.pad(p*scale,[(0,0),(0,0),(1,1),(1,1)])
        for i,j in [(1,1),(0,1),(2,1),(1,0),(1,2)]:res.append(pp[:,:,i:i+12,j:j+12])
    return np.stack(res,1).reshape(-1,c*144).astype('f')

# Localized evidence for a thin illuminated cylinder. For each axis there is
# a bright/dark flank pair and a profile with equal flanks. Taking the maximum
# covers both a shaft lit from one side and a finger in silhouette.
def ridge_filters():
    y,x=np.mgrid[-3:4,-3:4];F=[]
    for a in np.arange(6)*np.pi/6:
        axial=x*np.cos(a)+y*np.sin(a);across=y*np.cos(a)-x*np.sin(a)
        w=np.exp(-.5*((axial/2.3)**2+(across/1.12)**2))
        for profile in [1-(across/1.12)**2,across/1.12]:
            f=w*profile;f-=f.mean();f/=np.abs(f).sum();F.extend([f,-f])
    return np.array(F)

def ridge_desc(ims,rf):
    color=np.stack([p[0] for p in ims]);mask=np.stack([p[1] for p in ims]);color-=color.mean((1,2),keepdims=True);color/=np.sqrt((color**2).mean((1,2),keepdims=True)+1e-5)
    win=np.lib.stride_tricks.sliding_window_view(color,(7,7),axis=(1,2));v=(win.reshape(-1,49)@rf.reshape(24,-1).T).reshape(len(ims),22,22,6,4).max(-1).transpose(0,3,1,2)
    v=.42*v+.88*np.maximum(0,v-v.mean(1,keepdims=True));v=v.reshape(len(ims),6,11,2,11,2).mean((3,5));v=np.minimum(v,.55)
    mask=np.lib.stride_tricks.sliding_window_view(mask,(7,7),axis=(1,2)).max((3,4));mask=.20+.80*mask.reshape(len(v),1,11,2,11,2).mean((3,5))
    v-=((v*mask).sum((2,3),keepdims=True)/mask.sum((2,3),keepdims=True));v*=mask
    v=v.reshape(len(v),-1);v/=np.sqrt((v*v).sum(1,keepdims=True)+1e-9)
    return v.astype('f')

def ridge_layers(layer,conv):
    rf=ridge_filters();conv('shaft_quadrature','illum',rf[:,None],np.zeros(24));layer('phase_axis','reshape','shaft_quadrature',shape=[6,4,22*22]);layer('phase_max','maxpool','phase_axis',size=[4,1],stride=[4,1]);layer('ridge_map','reshape','phase_max',shape=[6,22,22]);
    conv('axis_difference','ridge_map',(np.eye(6)-1/6)[:,:,None,None],np.zeros(6));layer('preferred','relu','axis_difference');conv('pref_gain','preferred',(.88*np.eye(6))[:,:,None,None],np.zeros(6));conv('base_gain','ridge_map',(.42*np.eye(6))[:,:,None,None],np.zeros(6));layer('ridge_strength','add',['pref_gain','base_gain']);layer('ridge_cell','avgpool','ridge_strength',size=[2,2],stride=[2,2]);conv('ridge_remaining','ridge_cell',-np.eye(6)[:,:,None,None],np.full(6,.55));layer('ridge_complement','relu','ridge_remaining');return rf

def spatial_weight():
    y,x=np.mgrid[:12,:12];return (.65+.35*np.exp(-((x-5.5)**2/27+(y-4.)**2/38)))*(1-.13*np.maximum(0,y-8)/3)

def build(dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);net=[];w={}
    def layer(name,op,src,**kw):
        net.append(dict(name=name,op=op,inputs=src if isinstance(src,list) else [src],**kw));return name
    def conv(name,src,f,b):
        w[name+'.W']=np.array(f,'f');w[name+'.b']=np.array(b,'f');return layer(name,'conv',src,out=len(f),kernel=list(f.shape[-2:]),padding=0,stride=1)
    fil=filters();C=len(fil);layer('illum','norm','image');conv('oriented','illum',fil[:,None],np.zeros(C));layer('half_wave','relu','oriented');layer('local_energy','avgpool','half_wave',size=[2,2],stride=[2,2]);
    energy=np.zeros((1,C,3,3));energy[0,:8,:,:]=1/36
    conv('neighborhood','local_energy',energy,[0]);net[-1]['padding']=1
    # Compute the two cell-relative clipped ramps using ReLU differences.
    thresholds=[(.45,0),(.45,.075),(1.05,0),(1.05,.11)]
    conv('reference','neighborhood',np.repeat(-np.array([a for a,b in thresholds]),C)[:,None,None,None],np.repeat([-b for a,b in thresholds],C));
    conv('orient_copy','local_energy',np.tile(np.eye(C),(4,1))[:,:,None,None],np.zeros(4*C));layer('angular_margin','add',['orient_copy','reference']);layer('ramps','relu','angular_margin');
    factors=[.40/.075,-.40/.075,.45/.11,-.45/.11];conv('ramp_mix','ramps',np.concatenate([np.eye(C)*d for d in factors],1)[:,:,None,None],np.zeros(C));
    conv('energy_floor','local_energy',(.50*np.eye(C))[:,:,None,None],np.zeros(C));layer('gradient_code','add',['ramp_mix','energy_floor']);
    npose=176;nview=10;N=npose*nview; W=np.empty((24*N,C*12*12),np.float32)
    M=40;ew=np.empty((24*M*4*5,C*144),np.float32)
    rf=ridge_layers(layer,conv);rw=np.empty((24*N,6*121),np.float32)
    for idx,l in enumerate(LETTERS):
        imgs=[];flex=[]
        for i in range(npose):
            rng=np.random.default_rng(20260713+i*317+idx*42019)
            h,lo,hi=primitives(l,rng,direct=not(l in 'GH' and i%3==0))
            for j in range(nview):
                imgs.append(render(h,lo,hi,rng))
                if i<M and j==0:flex.append(imgs[-1])
            if len(imgs)>=64 or i==npose-1:
                stop=idx*N+(i+1)*nview;W[stop-len(imgs):stop]=template_desc(imgs,fil)[0];rw[stop-len(imgs):stop]=-2.15*ridge_desc(imgs,rf);imgs=[]
        ew[idx*M*20:(idx+1)*M*20]=elastic_desc(flex,fil)
    w['compare.W']=W;w['compare.b']=np.zeros(len(W),'f');layer('compare','dense','gradient_code',out=len(W));w['shafts.W']=rw;w['shafts.b']=np.zeros(len(rw),'f');layer('shafts','dense','ridge_complement',out=len(rw));layer('joint_similarity','add',['shafts','compare']);layer('pose_axis','reshape','joint_similarity',shape=[24,1,N]);
    # Max over each subgroup, then use the best match, with a small vote from
    # the independent subgroups to reduce sensitivity to isolated matches.
    layer('subgroup','maxpool','pose_axis',size=[1,N//8],stride=[1,N//8]);layer('closest','maxpool','subgroup',size=[1,8],stride=[1,8]);layer('consensus','avgpool','subgroup',size=[1,8],stride=[1,8]);
    conv('consensus_weight','consensus',(.30*np.eye(24))[:,:,None,None],np.zeros(24));
    w['elastic.W']=ew;w['elastic.b']=np.zeros(len(ew),'f');layer('elastic','dense','gradient_code',out=len(ew));
    layer('elastic_axis','reshape','elastic',shape=[24*M,4,5]);layer('allow_motion','maxpool','elastic_axis',size=[1,5],stride=[1,5]);layer('part_mean','avgpool','allow_motion',size=[4,1],stride=[4,1]);layer('elastic_pose','reshape','part_mean',shape=[24,1,M]);layer('elastic_best','maxpool','elastic_pose',size=[1,M],stride=[1,M]);conv('part_weight','elastic_best',(3.2*np.eye(24))[:,:,None,None],np.zeros(24));
    layer('match','add',['closest','consensus_weight','part_weight']);layer('scores','reshape','match',shape=[24]);
    (dest/'network.json').write_text(json.dumps({'layers':net},indent=2)+'\n');np.savez(dest/'weights.npz',**w)

if __name__=='__main__':build(sys.argv[1])
