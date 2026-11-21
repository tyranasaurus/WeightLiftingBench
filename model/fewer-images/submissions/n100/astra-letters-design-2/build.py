"""An anatomical template matcher. No file inputs. Units of the hand rig are cm.
The only images it makes are individual model templates; there is no training.
"""
import sys, json
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

ABC='ABCDEFGHIKLMNOPQRSTUVWXY'
f32=np.float32
rad=np.deg2rad

def rot(axis,degrees):
    c,s=np.cos(rad(degrees)),np.sin(rad(degrees))
    if axis==0:return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis==1:return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def hand(letter,r):
    # Local axes: +x toward thumb, +y distal along palm, +z out of palm.
    # Digits ordered index, middle, ring, little. Three flexions per digit
    # (absolute segment angles, with zero pointing straight along the palm).
    U=r.uniform; N=r.normal
    straight=np.array([2.,6.,9.]); fist=np.array([76.,168.,225.]); hook=np.array([18.,102.,162.])
    bends=np.tile(fist,(4,1)); fan=np.array([2.,0.,-2.,-4.]); twist=np.zeros(4)
    yaw,pitch,roll=U(-24,25),U(-15,18),N(-4,8)
    thumb='over'; arm=(-.24,-1,-.12); flex_noise=6.
    if letter=='A':
        thumb='beside'; bends[:]=[66,161,228]
    elif letter=='B':
        bends[:]=straight; fan[:]=[-2,-.5,1,4]; thumb='across'
    elif letter in 'CO':
        yaw,pitch,roll=U(51,105),U(-17,12),U(-22,4)
        curve=U(-8,10)
        bends[:]=[32,77,109] if letter=='C' else [50,110,152]
        bends+=curve; thumb='openC' if letter=='C' else 'meet';fan[:]=[-2,-1,1,4]
    elif letter=='D':
        bends[:]=[54,128,166]; bends[0]=straight;thumb='meetMiddle';yaw=U(-30,35)
    elif letter=='E':
        bends[:]=[28,133,198]; thumb='under'; yaw=U(-15,28)
    elif letter=='F':
        bends[:]=straight; bends[0]=[53,119,153];fan[:]=[1,9,-4,-22];thumb='meet';yaw=U(15,78);roll=U(-18,13)
    elif letter in 'GH':
        # Fingers point across the body. The thumb parallels the index on G,
        # whereas H holds two long fingers parallel, with the thumb folded.
        bends[0]=straight+U(0,12);fan[0]=0
        if letter=='H': bends[1]=bends[0]+N(2,2);fan[1]=1
        thumb='parallel' if letter=='G' else 'over';yaw=U(114,210);roll=U(-91,-62);pitch=U(-13,23)
        arm=(-.65,-1,.1)
    elif letter=='I':
        bends[3]=straight;fan[3]=U(-21,-4);thumb='over'
    elif letter=='K':
        bends[0]=straight;bends[1]=straight+U(28,59);fan[:2]=[5,-11];thumb='prong';yaw=U(-75,-12);roll=U(-10,10)
    elif letter=='L':
        bends[0]=straight;thumb='out';fan[0]=U(-3,10)
    elif letter in 'MNT':
        # The web in which the thumb emerges is the distinguishing feature.
        n={'M':3,'N':2,'T':1}[letter];thumb='web'+str(n)
        bends[:n]=[53,137,191];twist[:n]=U(-18,-5);yaw=U(-31,32)
        if letter=='T' and U()<.70:
            # Side presentation of the index wrapped across the thumb.
            bends[0]=[64,93,114]; thumb='beside';yaw=U(62,106);pitch=U(-20,7);roll=U(-19,8)
    elif letter=='P':
        bends[0]=straight;bends[1]=straight+U(51,84);fan[:2]=[4,-7];thumb='prong'
        yaw=U(-108,-54);roll=U(43,93);pitch=U(-48,-9);arm=(.4,-1,-.45)
    elif letter=='Q':
        # Flex at the wrist; the index and thumb point toward the ground.
        bends[0]=straight+U(40,76);thumb='parallel';yaw=U(-104,-58);roll=U(69,114);pitch=U(-31,10);arm=(.5,-1,-1.)
    elif letter=='R':
        bends[:2]=straight;fan[:2]=[-13,12];bends[1]+=U(6,13);thumb='hold';yaw=U(-38,30)
    elif letter=='S':
        bends[:]=[75,165,218];thumb='over'
    elif letter=='U':
        bends[:2]=straight;fan[:2]=[-2,1];thumb='hold'
    elif letter=='V':
        bends[:2]=straight;fan[:2]=[U(9,23),-U(9,22)];thumb='hold'
    elif letter=='W':
        bends[:3]=straight;fan[:3]=[U(11,23),0,-U(11,23)];thumb='pinky'
    elif letter=='X':
        bends[0]=hook+[0,U(-18,15),U(-23,10)];thumb='over';yaw=U(26,89);roll=U(-15,8)
    elif letter=='Y':
        bends[3]=straight;fan[3]=U(-38,-17);thumb='out';roll=U(-16,12)
    # Details of opposition, crossing, and wrist orientation. With a trapped
    # thumb the fingers that cover it do not curl down as far as the others.
    if letter in 'MN':
        bends[:n,0]=U(32,64);bends[:n,1]=U(115,153);bends[:n,2]=U(171,215)
        bends[n:]=[84,173,231];twist[:n]=U(-27,-9);roll+=N(-5,8);pitch+=U(-11,18)
    if letter=='E':
        # A hook fist: the first phalanges rise above the knuckles, with the
        # tips doubled back to the transversely flexed thumb on the palm.
        bends[:]=[U(15,49),U(115,157),U(171,201)];fan*=.55
    if letter in 'AEMNSI':
        pitch+=U(-8,24);yaw+=U(-16,19);roll+=U(-12,9)
    if letter=='D':
        bends[1:]=[U(47,76),U(113,145),U(164,196)];yaw=U(-38,40);pitch+=U(-8,22)
    if letter=='X':pitch+=U(-13,27);yaw=U(23,98)
    if letter=='I':fan[3]=U(-18,3);roll+=U(-11,9)
    if letter=='S':thumb='clasp';roll+=N(-5,8);bends[:]=[U(65,84),161,217]
    if letter=='B':fan[:]=[-3,-.8,1.5,4];yaw=U(-34,38)
    if letter in 'DU':yaw+=N(0,9)
    if letter=='R':
        fan[:2]=[-8,U(10,17)];bends[1]+=U(3,15);twist[1]=-12
    if letter=='K':
        yaw=U(-88,-29);bends[1]=straight+U(35,67);thumb='tallProng'
    if letter=='W':fan[:3]+=[U(0,9),0,-U(0,10)]
    if letter=='G':yaw=U(126,213);pitch=U(-26,25);roll=U(-91,-57)
    if letter=='H':yaw=U(135,234);pitch=U(-23,26);roll=U(-100,-69)
    if letter=='O':
        # O is a hollow oval, not a tightly curled fist: the thumb approaches
        # the pads from below, and the metacarpals form one side of the oval.
        bends[:]=[U(35,54),U(79,103),U(126,150)];yaw=U(39,102)
    if letter=='T' and thumb=='beside':
        bends[0]=[U(54,77),U(82,110),U(99,137)];yaw=U(51,125)
    if letter=='P':
        yaw=U(-154,-74);roll=U(43,91);pitch=U(-43,17);bends[0]+=U(0,19)
    if letter=='Q':
        yaw=U(-176,-74);roll=U(110,167);pitch=U(-34,32);bends[0]=straight+U(13,64)
    if letter in 'GH':
        # Index on top of the folded fingers. In an edge-on presentation
        # the two extended digits of H overlap in projection. The thumb of
        # G is extended from the saddle joint, not clasped in the fist.
        if U()<.32:yaw=U(-70,68) if letter=='G' else U(-91,139)
        else:yaw=U(122,220)
        pitch=U(-27,26);roll=U(-98,-62);fan[:2]=[-1,1]
    if letter in 'MN':
        # Curl over the trapped thumb: full metacarpal flexion with enough
        # clearance in the final joint that the thumb tip remains visible.
        bends[:n]=[U(54,84),U(130,160),U(176,207)]
        twist[:n]=U(-32,-3);yaw=U(-41,44);pitch=U(-18,27)
    # Hand proportions: the width and thickness of each finger and the
    # lengths of the phalanges vary independently of framing and shading.
    width=U(.9,1.12);length=U(.94,1.08);girth=U(.91,1.1)
    scalevec=np.array([width,length,1.]); prim=[]
    def add(a,b,ra,rb,axes=(1,1,1),tag=0):
        prim.append((np.array(a)*scalevec,np.array(b)*scalevec,ra*girth,rb*girth,np.array(axes)*scalevec,tag))
    # Flattened, tapering solids for the carpus, palm, heel and thenar pad.
    add((0,.4,-.10),(-.15,3.5,0),2.43,3.64,(1,.62,.43))
    add((-.15,3.5,0),(-.12,6.3,-.04),3.64,3.58,(1,.63,.43))
    add((1.5,1.9,.31),(2.0,4.23,.46),1.27,1.94,(1,1,.66))
    add((-1.35,1.8,.25),(-2.05,4.2,.32),1.05,1.36,(1,1,.76))
    arm0=np.array((0.,.85,-.13)); armvec=np.array(arm)+N(0,[.14,0,.16]);armvec/=np.linalg.norm(armvec)
    add(arm0+armvec*22,arm0,3.42,2.4,(1,1,.76),1)
    spread=U(.85,1.1);xroot=np.array([2.51,.82,-.94,-2.55])*spread
    yroot=np.array([7.41,7.85,7.56,6.70]);zroot=np.array([.0,-.23,-.19,.12]);tips=[];chains=[]
    lengths=[[3.86,2.28,1.83],[4.29,2.66,1.92],[4.00,2.55,1.85],[3.12,1.86,1.59]]
    radius=np.array([.94,1.,.92,.785])*U(.94,1.02)
    common=N(0,flex_noise); common_fan=N(0,2.5)
    for i in range(4):
        angle=rad(bends[i]+N(0,3.4,3)+N(0,3.2)+common*(.42 if bends[i,0]<10 else 1))
        f=rad(fan[i]+N(0,1.5)+common_fan);t=rad(twist[i]+N(0,3))
        base=np.array([xroot[i],yroot[i]+N(0,.10),zroot[i]])
        ds=np.array(lengths[i])*U(.95,1.055);rs=radius[i]*np.array([1.,.935,.855,.74]);chain=[base]
        if letter=='B':
            # Four adjoining straight digits, with a continuous web at the
            # base. Their shafts have elliptical rather than circular section.
            f=rad(fan[i]+common_fan+N(0,.7));rs*=1.045
        for k,(a,l) in enumerate(zip(angle,ds)):
            d=np.array([np.sin(f)*np.cos(a)+np.cos(f)*np.sin(a)*np.sin(t),np.cos(f)*np.cos(a)-np.sin(f)*np.sin(a)*np.sin(t),np.sin(a)*np.cos(t)])
            end=base+l*d
            add(base,end,rs[k],rs[k+1],(1,1,.9),tag=2+k);base=end;chain.append(base)
        tips.append(base);chains.append(chain)
    # Thumb: saddle joint allows both opposition and abduction. The target
    # (when applicable) is taken from this hand's own kinematic chain.
    start=np.array([1.67,1.35,.18]);mcp=[3.20,4.02,1.35];ip=[2.34,5.74,2.96];end=[.05,6.50,3.77]
    if thumb=='beside': mcp=[3.53,4.44,1.06];ip=[3.68,6.92,1.78];end=[3.55,9.08,1.78]
    if thumb=='across':mcp=[2.91,4.08,1.08];ip=[1.34,5.67,1.99];end=[-.72,5.74,2.0]
    if thumb=='under':mcp=[2.89,4.30,1.67];ip=[1.26,6.15,2.76];end=[-.85,6.40,2.94]
    if thumb=='hold':mcp=[3.15,4.05,1.42];ip=[1.53,5.60,2.91];end=[-.65,6.36,3.71]
    if thumb=='pinky':mcp=[3.11,4.14,1.5];ip=[.74,5.55,2.73];end=[-1.50,6.50,3.38]
    if thumb.startswith('web'):
        n=int(thumb[-1]);xx=(xroot[n]+xroot[n-1])/2
        mcp=[2.71,4.25,1.75];ip=[(xx+2.1)*.5,6.40,2.5];end=[xx,yroot[n-1]+1.45,2.42]
        if n==1:end[1]+=.64
    if thumb=='prong':mcp=[3.19,4.44,1.42];ip=[2.39,6.63,2.64];end=[1.38,8.48,3.40]
    if thumb=='parallel':
        if letter=='Q':mcp=[3.85,3.95,1.1];ip=[4.39,5.34,3.23];end=[4.71,6.47,5.15]
        else:mcp=[3.95,4.43,1.28];ip=[4.32,6.98,1.48];end=[4.02,9.08,1.74]
    if thumb=='under':mcp=[2.97,3.98,1.85];ip=[1.22,5.16,2.51];end=[-.98,5.19,2.67]
    if thumb=='clasp':
        # In S the thumb lies ON the curled index and middle, not behind
        # their second phalanges. Keep its pad in front of this hand's joints.
        z=max(chains[0][1][2],chains[1][1][2]);mcp=[3.23,4.58,2.25];ip=[1.64,6.63,z+.96];end=[-.50,7.72,z+1.09]
    if thumb=='tallProng':mcp=[3.10,4.55,1.42];ip=[2.41,7.11,2.57];end=[1.55,9.05,3.57]
    if thumb=='out':
        a=rad(U(13,48) if letter=='Y' else U(7,36)); d=np.array([np.cos(a),np.sin(a),U(.0,.42)]);mcp=np.array([3.75,3.34,.62]);ip=mcp+2.94*d;end=ip+2.13*d+[0,.19,.09]
    if thumb=='openC':mcp=[3.63,3.57,2.18];ip=[3.68,4.37,4.72];end=[3.11,5.48,6.51]
    if thumb.startswith('meet'):
        i=1 if thumb=='meetMiddle' else 0; target=tips[i];end=target+np.array([.47,-.68,.10]);mcp=np.array([3.16,3.93,2.05]);ip=(mcp+end)/2+np.array([.68,-.24,.70])
    base=start
    for k,(stop,ra,rb) in enumerate(zip([mcp,ip,end],[1.39,1.20,1.02],[1.20,1.02,.85])):
        stop=np.asarray(stop)+N(0,.21,3);add(base,stop,ra,rb,tag=2+k);base=stop
    view=rot(2,roll)@rot(0,pitch)@rot(1,yaw)
    if letter in 'PQ' and U()<.75:
        # Construct a view from the pointing directions themselves. In the
        # P handshape the straight index angles across the signer while the
        # bent middle finger points down. In Q the index itself points down.
        phi=U(134,166) if letter=='P' else U(223,259);theta=rad(phi);tilt=rad(U(-27,32))
        indexdir=np.array([np.cos(tilt)*np.cos(theta),np.cos(tilt)*np.sin(theta),np.sin(tilt)])
        perp=np.array([-np.sin(theta),np.cos(theta),0]);unused=np.cross(indexdir,perp)
        t=rad(U(-67,45) if letter=='P' else U(37,132));flex=perp*np.cos(t)+unused*np.sin(t)
        cross=np.cross(indexdir,flex)
        world=np.array([cross,indexdir,flex]).T
        local=chains[0][1]-chains[0][0];local/=np.linalg.norm(local)
        radial=np.array([1.,0.,0.]);radial=radial-local*np.dot(local,radial);radial/=np.linalg.norm(radial)
        view=world@np.array([radial,local,np.cross(radial,local)])
    if letter in 'GH':
        a,b,ra,rb,axes,tag=prim[4];direction=np.array([-U(.25,1.65),-1.,U(-.65,.4)]);direction/=np.linalg.norm(direction);prim[4]=(b+view.T@direction*22,b,ra,rb,axes,tag)
    if letter in 'PQ':
        # Independent wrist flexion: even for a down-facing sign the arm is
        # usually below the hand, not an extension of the index finger.
        a,b,ra,rb,axes,tag=prim[4];direction=np.array([U(.25,1.),-1.,U(-.5,.5)]);direction/=np.linalg.norm(direction);prim[4]=(b+view.T@direction*22,b,ra,rb,axes,tag)
    return prim,view

# Analytic orthographic ray intersections with an affine tapered capsule.
# The flank is the common tangent to its two endpoint balls. For a ray that
# enters via a joint, the ball covers the remaining hemisphere.
def render(letter,r,H=64):
    P,V=hand(letter,r)
    # In addition to near-orthographic portraits, model the foreshortening
    # when a signer holds the hand close to a small camera. This is a weak
    # perspective approximation per limb, with positive z toward the lens.
    distance=r.uniform(25.,80.);projected=[]
    for a,b,ra,rb,axes,tag in P:
        fa=distance/max(distance-(V@a)[2],.58*distance);fb=distance/max(distance-(V@b)[2],.58*distance)
        def proj(p,m):return V.T@((V@p)*[m,m,1])
        projected.append((proj(a,fa),proj(b,fb),ra*fa,rb*fb,axes,tag))
    P=projected
    body=[(V@p,rad0) for a,b,ra,rb,axes,tag in P if tag!=1 for p,rad0 in [(a,ra),(b,rb)]]
    pts=np.array([p for p,_ in body]);rs=np.array([s for _,s in body])
    low=(pts[:,:2]-rs[:,None]).min(0);high=(pts[:,:2]+rs[:,None]).max(0)
    # Include the wrist below the palm. Crop relative to this model's own
    # bounds, never to external sample coordinates.
    low[1]=min(low[1],(V@np.array([0.,0.,0.]))[1]-1.8)
    mid=(high+low)*.5; span=max(high-low); base=span*1.09
    # Render a larger field than the final crop for shifted crops and zoom.
    field=base*1.42;step=field/H
    ar=((np.arange(H,dtype=f32)+.5)-H/2)*step;X,Y=np.meshgrid(ar+mid[0],-ar+mid[1]);Z=np.full((H,H),-500.,f32);part=np.zeros((H,H),np.int8)
    def put(zz,tag,y,x):
        old=Z[y,x];update=zz>old;np.copyto(old,zz,where=update);part[y,x][update]=tag
    for a,b,ra,rb,axes,tag in P:
        A=V@a;B=V@b
        ext=np.sqrt((V[:2]**2)@(axes**2))*max(ra,rb)
        left,bot=(np.minimum(A,B)[:2]-ext-mid)/step; right,top=(np.maximum(A,B)[:2]+ext-mid)/step
        x=slice(max(0,int(np.floor(H/2+left))),min(H,max(0,int(np.ceil(H/2+right)))))
        y=slice(max(0,int(np.floor(H/2-top))),min(H,max(0,int(np.ceil(H/2-bot)))))
        if y.stop<=y.start or x.stop<=x.start:continue
        # Factor the ray metric into projected screen distance and depth.
        M=(V/(axes**2))@V.T if not np.all(axes==1.) else np.eye(3)
        iz=1/M[2,2];tilt=M[:2,2]*iz;G=M[:2,:2]-np.outer(M[:2,2],tilt)
        dx=X[y,x]-A[0];dy=Y[y,x]-A[1]; vec=B-A
        z0=A[2]-tilt[0]*dx-tilt[1]*dy; dz=vec[2]+tilt@vec[:2]
        gg=G@vec[:2];dd=gg@vec[:2]; dot=dx*gg[0]+dy*gg[1]
        q=dx*(G[0,0]*dx+2*G[0,1]*dy)+G[1,1]*dy*dy
        # Maximize height over the sweep of balls. The radius is linear in t.
        dr=rb-ra; h2=dd-dr*dr
        # Squared visible half-depth / iz = hmax - h2*(t-t0)^2.
        if h2>1.e-7:
            t0=(dot+ra*dr)/h2; hmax=ra*ra-q+h2*t0*t0
            t=np.clip(t0+dz*np.sqrt(np.maximum(hmax,0)/(h2*(iz*h2+dz*dz))),0.,1.)
            hh=ra*ra-q+t*(2*(dot+ra*dr)-h2*t)
            z=np.where(hh>0,z0+t*dz+np.sqrt(iz*np.maximum(hh,0)), -600.)
        else:
            h0=ra*ra-q;h1=rb*rb-q+2*dot-dd
            z=z0+np.maximum(np.where(h0>=0,np.sqrt(np.maximum(h0,0)*iz),-800.),np.where(h1>=0,dz+np.sqrt(np.maximum(h1,0)*iz),-800.))
        put(z,tag,y,x)
    # Compute derivatives without crossing occluding contours. A one-sided
    # derivative on the continuous side gives a normal at those contours.
    def deriv(axis):
        prev=np.roll(Z,1,axis);nxt=np.roll(Z,-1,axis);u=Z-prev;v=nxt-Z
        return np.where(np.abs(u-v)<step*1.20,(u+v)*.5,np.where(np.abs(u)<np.abs(v),u,v))/step
    nx=-deriv(1);ny=deriv(0);den=1/np.sqrt(1+nx*nx+ny*ny);mask=Z>-400
    def shift_depth(dx,dy):
        tmp=np.roll(Z,(-dy,-dx),axis=(0,1))
        if dy<0:tmp[:-dy,:]=-600
        elif dy>0:tmp[-dy:,:]=-600
        if dx<0:tmp[:,:-dx]=-600
        elif dx>0:tmp[:,-dx:]=-600
        return tmp
    # Soft cast shadows: a ray toward the lamp can meet an opposing finger
    # before reaching free space. Compute the horizon slope in three common
    # light directions; no photometric parameters are estimated from data.
    shadowmaps=[]
    for dx,dy in [(-1,-1),(1,-1),(-1,0)]:
        slope=np.zeros_like(Z)
        for j in (1,2,3,5,8,12):
            slope=np.maximum(slope,(shift_depth(dx*j,dy*j)-Z)/(step*(dx*dx+dy*dy)**.5*j))
        shadowmaps.append(slope)
    # Small ambient occlusion around overhangs, independent of light source.
    occ=np.zeros_like(Z)
    for sy,sx in [(1,0),(-1,0),(0,1),(0,-1),(2,2),(2,-2),(-2,2),(-2,-2)]:
        occ+=np.clip((np.roll(Z,(sy,sx),axis=(0,1))-Z-step*.7)/1.5,0.,1.)
    # A different photograph of this geometry can have a different exposure,
    # backdrop, crop and angle of lighting. Each output is used separately.
    def exposure():
        lx=r.uniform(-1.55,-.2) if r.random()<.78 else r.uniform(.2,1.3)
        ly=r.uniform(.3,1.6);lz=r.uniform(.45,1.6);L=np.sqrt(lx*lx+ly*ly+lz*lz)
        lambert=(nx*lx+ny*ly+lz)*den/L
        horizon=shadowmaps[1 if lx>0 else (2 if -lx>ly*2 else 0)]
        shadow=np.clip((horizon-lz/np.hypot(lx,ly)-.15)*.84,0,1)
        skin=r.uniform(.16,.37)+r.uniform(.46,.69)*np.maximum(0,lambert)*(1-r.uniform(.26,.64)*shadow)-occ*r.uniform(.020,.051)
        # Soft fingertip creases where a flexing link meets the next.
        crease=(part>=2)&((part!=np.roll(part,1,axis=0))|(part!=np.roll(part,1,axis=1)))
        skin=skin-r.uniform(.018,.082)*crease
        bg=r.uniform(.38,.86)+r.uniform(-.18,.18)*X/field+r.uniform(-.18,.18)*Y/field
        img=np.where(mask,skin,bg).astype(f32)
        zoom=r.uniform(.72,1.17);wid=base*zoom
        shift=r.uniform(-.14,.14,2)*base;shift[1]+=r.uniform(-.012,.06)*base
        if letter in 'GH':shift[0]-=r.uniform(0,.15)*base
        if letter in 'PQ':shift[0]+=r.uniform(0,.14)*base
        # Four bilinear sub-rays per pixel for a softer camera point spread.
        side=56;uv=(np.arange(side)+.5)/side-.5;xx,yy=np.meshgrid(uv*wid+shift[0],uv*wid-shift[1]);xx=np.clip(xx/step+H/2-.5,0,H-1.001);yy=np.clip(yy/step+H/2-.5,0,H-1.001)
        ix=xx.astype(int);iy=yy.astype(int);fx=xx-ix;fy=yy-iy
        v=(img[iy,ix]*(1-fx)+img[iy,ix+1]*fx)*(1-fy)+(img[iy+1,ix]*(1-fx)+img[iy+1,ix+1]*fx)*fy
        return v.reshape(28,2,28,2).mean((1,3)).astype(f32)
    return exposure

# Built-in operators and their template-side counterparts. No examples or
# template scores are used to choose the analytic filters or normalization.
layers=[];W={}
def layer(name,op,source,**kw):
    layers.append(dict(name=name,op=op,inputs=source if isinstance(source,list) else [source],**kw));return name
def conv(name,source,k,b=None):
    k=np.asarray(k,f32);W[name+'.W']=k;W[name+'.b']=np.zeros(len(k),f32) if b is None else np.array(b,f32)
    return layer(name,'conv',source,out=len(k),kernel=list(k.shape[-2:]),stride=1,padding=0)
def point(name,source,w,b=None):return conv(name,source,np.asarray(w)[:,:,None,None],b)
def norm(x):
    x=x-x.mean(axis=(-2,-1),keepdims=True);return x/np.sqrt((x*x).mean((-2,-1),keepdims=True)+1e-5)

def feature_net():
    a=layer('exposure','norm','image')
    # Nearest-edge extrapolation, so fingertips at the frame are retained
    # without introducing a black picture border. This is a fixed gather.
    idx=np.pad(np.arange(28*28).reshape(28,28),2,mode='edge').ravel()
    W['border.W']=np.eye(28*28,dtype=f32)[idx];W['border.b']=np.zeros(len(idx),f32)
    a=layer('border','dense',a,out=len(idx));a=layer('padded','reshape',a,shape=[1,32,32])
    y,x=np.mgrid[-2:3,-2:3];g=np.exp(-(x*x+y*y)/(2*.83**2));g/=g.sum();dirs=np.arange(8)*np.pi/4
    k=[]
    for t in dirs:
        p=x*np.cos(t)+y*np.sin(t);k.append(p*g/np.sum(x*x*g))
    # Signed centre-surround signal supplements the oriented edges. It
    # responds to ridges, valleys and dark recesses between fingers.
    blur=np.exp(-(x*x+y*y)/(2*.62**2));blur/=blur.sum();broad=np.exp(-(x*x+y*y)/(2*1.65**2));broad/=broad.sum()
    k.extend([blur-broad,broad-blur]);a=conv('differentials',a,np.array(k)[:,None]);a=layer('half_waves','relu',a)
    mat=np.eye(10)
    for i in range(8):mat[i,(i+2)%8]-=.71;mat[i,(i-2)%8]-=.71
    # Saturating linear function, explicitly realized by two ReLUs, caps
    # very high contrasts instead of letting a single edge dominate.
    lim=np.array([.48]*8+[.52]*2);a=point('orient_and_clip',a,np.vstack((mat,mat)),np.r_[np.zeros(10),-lim]);a=layer('rectified','relu',a)
    mat2=np.hstack((np.eye(10),-np.eye(10)))
    # Unsigned orientation and signed polarity differences are separate.
    mix=np.zeros((10,10));
    for i in range(4):mix[i,i]=mix[i,i+4]=1;mix[i+4,i]=1;mix[i+4,i+4]=-1
    mix[8,8]=mix[8,9]=1;mix[9,8]=1;mix[9,9]=-1
    a=point('edge_fields',a,mix@mat2)
    a=layer('cells','avgpool',a,size=[4,4],stride=[2,2]);a=layer('standard_fields','norm',a)
    # A low-frequency companion describes the broad shadows of opposed
    # fingertips, the thenar pad, and the negative space inside a curved hand.
    b=layer('soft_light','avgpool','exposure',size=[4,4],stride=[2,2]);b=layer('light_field','norm',b)
    kernels=W['differentials.W'][:,0].reshape(10,25).T.copy()
    def describe(img):
        z=norm(img);p=windows(np.pad(z,2,mode='edge'),(5,5)).reshape(-1,25);d=np.maximum(0,p@kernels);d=np.minimum(np.maximum(0,d@mat.T),lim);e=(d@mix.T).T.reshape(10,28,28)
        c=windows(e,(4,4),axis=(1,2))[:,::2,::2].mean((-1,-2));s=windows(z,(4,4))[::2,::2].mean((-1,-2));return np.concatenate([norm(c),norm(s)[None]])
    return a,b,describe

def main(out):
    source,light,describe=feature_net();rng=np.random.default_rng(31867)
    ngeom=600;nexposure=4;nt=ngeom*nexposure
    # Fixed position weighting: wrists, sleeves and picture margins are less
    # diagnostic than the upper part of an ASL handshape.
    v=np.linspace(-1,1,13);xx,yy=np.meshgrid(v,v);spatial=(1-.28*xx**4)*(1-.62*np.maximum(yy,0)**1.4)
    channel=np.array([1]*4+[.59]*4+[.81,.61,.59]);gain=channel[:,None,None]*spatial
    # Combine both feature branches with fixed scatter matrices, then cut
    # into two regions. A canonical upper crop is compared exactly; the
    # lower crop may slide slightly, never changing the template identity.
    def gather(name,source,indices,size):
        w=np.zeros((len(indices),size),f32);indices=np.asarray(indices);use=indices>=0
        w[np.flatnonzero(use),indices[use]]=1
        W[name+'.W']=w;W[name+'.b']=np.zeros(len(w),f32);return layer(name,'dense',source,out=len(w))
    ids=np.arange(1859);grid=ids.reshape(11,13,13)
    one=gather('edge_channels',source,np.where(ids<1690,ids,-1),1690)
    two=gather('light_channel',light,np.where(ids>=1690,ids-1690,-1),169)
    both=layer('all_channels','add',[one,two]);upper_all=gather('upper',both,grid[:,:8].ravel(),1859)
    upper=upper_all
    def lowercrop(dy,dx):return grid[:,8+dy:13+dy,np.clip(np.arange(13)+dx,0,12)].ravel()
    lo=[lowercrop(0,0),lowercrop(-1,0),lowercrop(0,-1),lowercrop(0,1)]
    # The four shifted lower crops are positions of one shared 1x1 kernel.
    multi=gather('offsets',both,np.stack(lo,axis=1).ravel(),1859)
    multi=layer('shifted_lower','reshape',multi,shape=[11*5*13,1,len(lo)])
    upperbank=np.empty((24,nt,11*8*13),f32);lowerbank=np.empty((24,nt,11*5*13),f32)
    for ci,c in enumerate(ABC):
        for n in range(ngeom):
            expose=render(c,rng)
            for j in range(nexposure):
                feat=describe(expose());row=feat*gain
                # Normalize by this one template's weighted energy, only.
                row=row/np.sqrt(np.sum(row*feat)+1e-7);idx=n*nexposure+j
                upperbank[ci,idx]=row[:,:8].ravel();lowerbank[ci,idx]=row[:,8:].ravel()
    name='upper_matches';W[name+'.W']=upperbank.reshape(24*nt,-1);W[name+'.b']=np.zeros(24*nt,f32);a=layer(name,'dense',upper,out=24*nt)
    b=conv('lower_matches',multi,lowerbank.reshape(24*nt,-1,1,1));b=layer('wrist_offset','maxpool',b,size=[1,len(lo)],stride=[1,len(lo)]);b=layer('lower_result','reshape',b,shape=[24*nt]);a=layer('whole_matches','add',[a,b])
    # Consensus is computed on the unknown input at inference, not from a
    # collection of synthetic images during the build. Independent groups
    # give a little robustness against an accidental isolated high match.
    a=layer('group_by_letter','reshape',a,shape=[24,1,nt]);a=layer('local_best','maxpool',a,size=[1,24],stride=[1,24]);a=layer('group_best','maxpool',a,size=[1,nt//24//4],stride=[1,nt//24//4]);maximum=layer('best_pose','maxpool',a,size=[1,4],stride=[1,4]);support=layer('pose_support','avgpool',a,size=[1,4],stride=[1,4]);a=layer('combined_support','add',[maximum,support]);layer('scores','reshape',a,shape=[24])
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True);(folder/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n');np.savez(folder/'weights.npz',**W)
if __name__=='__main__':main(sys.argv[1])
