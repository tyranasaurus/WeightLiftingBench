"""Build an entirely procedural ASL hand atlas. No input files."""
import math
import os
import sys
import numpy as np

# Shared feature source. All pooling and normalizations operate independently on
# each picture. Spatial coordinates are in the 28-pixel camera frame.
features = '''import numpy as np


def smooth(a, axis):
    v = np.swapaxes(a, axis, -1)
    left = np.concatenate((v[..., :1], v[..., :-1]), axis=-1)
    right = np.concatenate((v[..., 1:], v[..., -1:]), axis=-1)
    return np.swapaxes((left + 2 * v + right) * .25, axis, -1)


def blur(a):
    return smooth(smooth(a, -1), -2)


def diff(a, axis):
    v = np.swapaxes(a, axis, -1)
    d = np.concatenate((v[..., 1:], v[..., -1:]), -1) - np.concatenate((v[..., :1], v[..., :-1]), -1)
    return np.swapaxes(d, axis, -1)


def poolmu(count, sigma):
    p = (np.arange(count) + .5) * 28 / count - .5
    w = np.exp(-.5 * ((p[:, None] - np.arange(28)) / sigma) ** 2)
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


wide = poolmu(7, 1.85)
fine = poolmu(14, 1.12)


def pool(a, mat):
    return (mat @ a) @ mat.T


def unit(a):
    a = a.reshape(len(a), -1)
    return a / np.sqrt(np.sum(a*a, 1, keepdims=True) + 1e-9)


def fgmap(a):
    # The surrounding background often forms an almost planar illumination
    # field. Estimate it in narrow border patches, allowing a hand/obstacle
    # to obscure one edge. No parameters are shared between photographs.
    xx = np.linspace(0, 1, 28, dtype=np.float32)[None,None,:]
    yy = xx.transpose(0,2,1)
    v = np.stack((a[:,:4,:4].mean((1,2)), a[:,:4,-4:].mean((1,2)),
                  a[:,-4:,:4].mean((1,2)), a[:,-4:,-4:].mean((1,2))), axis=1)
    grad = blur(np.abs(diff(a,1)) + np.abs(diff(a,2)))
    anchor = np.stack((grad[:,:4,:4].mean((1,2)),grad[:,:4,-4:].mean((1,2)),
                       grad[:,-4:,:4].mean((1,2)),grad[:,-4:,-4:].mean((1,2))), axis=1)
    # Extrapolate a plane from each triplet of corners. Use a robust
    # agreement test on the rim: a bad corner should not spoil the mask.
    hs = []
    es = []
    for k in range(4):
        c = v.copy()
        c[:, k] = c[:, k ^ 1] + c[:, k ^ 2] - c[:, k ^ 3]
        c = c[:,:,None,None]
        h = c[:,0] + (xx*27 - 1.5)/24 * (c[:,1] - c[:,0]) + (yy*27 - 1.5)/24 * (c[:,2] - c[:,0])
        d = np.abs(a - h)
        rim = np.concatenate((d[:,:2,:].reshape(len(a),-1),d[:,:,0],d[:,:,-1]),axis=1)
        es.append(np.mean(np.minimum(rim, .12), axis=1) + .30*(np.sum(anchor,1)-anchor[:,k]))
        hs.append(h)
    e = np.stack(es,1)
    w = np.exp(-(e - e.min(1,keepdims=True)) / .018)
    w /= w.sum(1,keepdims=True)
    h = sum(w[:,k,None,None] * hs[k] for k in range(4))
    residual = np.abs(a-h)
    # Include texture as evidence for the hand even at a brightness match.
    text = grad
    for j in range(3):
        text = blur(text)
    m = np.maximum(residual - .014, 0) + .70 * np.maximum(text - .026,0)
    photometric = m / (m + .069)
    # A second cue needs no assumption about the color of the background:
    # a path from the frame to the palm has to cross an occluding contour.
    # Propagate the least upper bound on edge strength along such paths.
    # Leave the middle of the bottom edge unseeded (the wrist crosses it).
    barrier = .58*grad + .42*(np.abs(diff(a,1))+np.abs(diff(a,2)))
    flood = np.full_like(a,2)
    flood[:,0,:] = barrier[:,0,:]
    flood[:,-1,:5] = barrier[:,-1,:5]
    flood[:,-1,-5:] = barrier[:,-1,-5:]
    flood[:,:,0] = barrier[:,:,0]
    flood[:,:,-1] = barrier[:,:,-1]
    for j in range(40):
        v = flood.copy()
        v[:,1:] = np.minimum(v[:,1:],flood[:,:-1])
        v[:,:-1] = np.minimum(v[:,:-1],flood[:,1:])
        v[:,:,1:] = np.minimum(v[:,:,1:],flood[:,:,:-1])
        v[:,:,:-1] = np.minimum(v[:,:,:-1],flood[:,:,1:])
        flood = np.maximum(v,barrier)
    trapped = np.maximum(blur(flood) - .018,0)
    return .55*photometric + .45*trapped/(trapped+.065)


def channels(pic, mask):
    a = blur(pic)
    gx = diff(a,2)
    gy = diff(a,1)
    mag = np.sqrt(gx*gx + gy*gy + 1e-10)
    ca, sa = gx / mag, gy / mag
    strength = mag / (mag + .075)
    # Separate true contrast changes from small texture/noise gradients.
    strength *= np.sqrt(mag / (mag + .045))
    maps = []
    for k in range(6):
        angle = k * np.pi / 6
        c = np.abs(ca * np.cos(angle) + sa * np.sin(angle))
        maps.append((strength * np.maximum(c*c - .54, 0)).astype(np.float32))
    if mask is None:
        mask = fgmap(a)
    # Signed local contrast adds the bright phalanges and dark valleys that
    # are invisible to a silhouette (notably in the closed handshapes).
    surround = a
    for j in range(10):
        surround = blur(surround)
    relief = a - surround
    maps.append((mask * relief / (np.abs(relief) + .060)).astype(np.float32))
    return np.stack(maps,1), mask.astype(np.float32)


def descriptor(edges, mask, mat):
    pooled = pool(edges, mat)
    ed = pooled[:,:6]
    relief = pooled[:,6]
    # Remove the common isotropic response but keep some edge mass.
    ed = ed - .62 * ed.mean(1,keepdims=True)
    pos = ed.shape[-1]
    row = np.arange(pos,dtype=np.float32)
    weight = (1 - .32 * (row / pos)**2)[None,None,:,None]
    ed = ed * weight
    f = pool(mask, mat)
    f = f - f.mean((1,2),keepdims=True)
    vec = np.concatenate((unit(ed) * .815, unit(f * weight[:,0]) * .513, unit(relief * weight[:,0]) * .270),axis=1)
    return vec.astype(np.float32)
'''

predictor = '''

coarsebank = np.ascontiguousarray(CONSTANTS['coarse'].astype(np.float32) / 512)
finebank = CONSTANTS['fine']
variants = CONSTANTS['coarse'].shape[0] // 24
picks = 16
stride = 16


def block(x):
    # A fixed row count gives the matrix products identical reduction order
    # regardless of the number or order of other pictures in the call.
    edge, mask = channels(x, None)
    coarse = descriptor(edge, mask, wide)
    finevec = descriptor(edge, mask, fine)
    s = (coarse @ coarsebank.T).reshape(stride,24,variants)
    best = np.argpartition(s, variants-picks, axis=2)[:,:,-picks:]
    out = np.empty((stride,24), dtype=np.float32)
    labels = np.arange(24)[:,None] * variants
    for j in range(stride):
        idx = labels + best[j]
        f = finebank[idx].astype(np.float32)
        t = np.sum(f * finevec[j,None,None,:],axis=2) / 512
        z = .70*t + .30*np.take_along_axis(s[j],best[j],axis=1)
        # A max with a small near-neighbour vote dampens isolated accidents.
        z.sort(axis=1)
        out[j] = .58 * z[:,-1] + .28*z[:,-2] + .14*z[:,-3]
    return np.argmax(out,1)


def predict(x):
    n = len(x)
    y = np.empty(n, dtype=np.int64)
    for start in range(0,n,stride):
        chunk = x[start:start+stride,0]
        a = np.empty((stride,28,28),dtype=np.float32)
        a[:] = chunk[0]
        a[:len(chunk)] = chunk
        y[start:start+len(chunk)] = block(a)[:len(chunk)]
    return y
'''

# Units are palm widths. Local +y goes from the heel of the hand to
# the fingers; local +z points out of the palm, +x toward the thumb.
# Rotation about x, y, z (the last is named 'roll' below).
def rx(d):
    a=math.radians(d); c,s=math.cos(a),math.sin(a)
    return np.array(((1,0,0),(0,c,-s),(0,s,c)))

def ry(d):
    a=math.radians(d); c,s=math.cos(a),math.sin(a)
    return np.array(((c,0,s),(0,1,0),(-s,0,c)))

def rz(d):
    a=math.radians(d); c,s=math.cos(a),math.sin(a)
    return np.array(((c,-s,0),(s,c,0),(0,0,1)))

# A lexicon of finger articulation, not image coordinates. Joint angles are
# listed as flexion at MCP, PIP and DIP, plus spread away from palm vertical.
# Order in a pose is index, middle, ring, little. Angles below add up along a
# finger. Touching fingers and hooked fingers have different joint sequences.
post={
 'up': (2,3,3,0), 'press':(-2,2,3,0), 'fist':(78,90,54,0),
 'claw':(34,101,70,0), 'round':(30,41,33,0), 'close':(42,58,39,0),
 'bend':(56,92,57,0), 'rest':(72,95,64,0),'cross':(4,10,8,0),
 'hook':(13,78,48,0),'flat':(65,9,4,0),'shelf':(78,26,14,0),
}
# The outstretched thumb, opposed thumb, etc. have a saddle joint, not
# another row of the four hinges. Specify its three knuckles anatomically.
# The last point is the fleshy distal tip. All positions are relative to heel.
thumbs={
 'along':[(.48,.36,.14),(.51,.68,.20),(.51,.90,.28)],
 'across':[(.42,.41,.28),(.13,.57,.38),(-.19,.64,.38)],
 'under':[(.43,.41,.25),(.14,.50,.42),(-.16,.49,.44)],
 'cap':[(.44,.46,.30),(.16,.74,.51),(-.14,.79,.54)],
 'wide':[(.59,.26,.13),(.83,.30,.15),(1.08,.33,.16)],
 'pinky':[(.43,.42,.28),(.21,.65,.45),(-.04,.73,.46)],
 'middle':[(.45,.47,.24),(.29,.80,.31),(.15,1.01,.31)],
 'between':[(.40,.47,.29),(.25,.80,.39),(.22,1.06,.42)],
 'mn':[(.35,.40,.22),(.02,.60,.29),(-.27,.69,.27)],
 'pinch':[(.46,.44,.21),(.40,.65,.43),(.29,.86,.57)],
 'c':[(.48,.40,.17),(.53,.54,.44),(.43,.69,.66)],
 'o':[(.45,.41,.21),(.45,.61,.46),(.35,.82,.63)],
 'g':[(.48,.46,.17),(.48,.74,.26),(.46,.99,.32)],
 'q':[(.54,.35,.19),(.68,.52,.29),(.78,.70,.29)],
}
letters='ABCDEFGHIKLMNOPQRSTUVWXY'
# Pose convention: y rotation (turn), x rotation (tilt relative to camera),
# then in-plane roll. Side views expose the arch of curved fingers.
# We spell out the orientations for sideways and downward signs separately.
lex={
 'A':('fist fist fist fist','along',-12,3,-4),
 'B':('press press press press','across',-5,0,0),
 'C':('round round round round','c',69,-4,-15),
 'D':('up close close close','pinch',49,0,-6),
 'E':('claw claw claw claw','under',-12,0,0),
 'F':('close up up up','pinch',9,-4,-2),
 'G':('up fist fist fist','g',-50,2,-78),
 'H':('up up fist fist','under',-48,4,-74),
 'I':('fist fist fist up','pinky',-8,0,-4),
 'K':('press flat rest rest','middle',-22,-7,-3),
 'L':('up fist fist fist','wide',-4,0,4),
 'M':('bend bend bend fist','mn',-12,7,0),
 'N':('bend bend fist fist','mn',-12,8,0),
 'O':('close close close close','o',61,4,-10),
 'P':('press flat rest rest','middle',-39,45,136),
 'Q':('up fist fist fist','q',-52,8,146),
 'R':('cross up fist fist','under',-5,1,0),
 'S':('fist fist fist fist','cap',-11,4,0),
 'T':('fist fist fist fist','between',-17,2,-9),
 'U':('press press fist fist','pinky',-8,0,-2),
 'V':('up up fist fist','pinky',-6,-1,2),
 'W':('up up up rest','under',-5,-1,2),
 'X':('hook rest rest rest','cap',-36,-1,-9),
 'Y':('fist fist fist up','wide',-4,0,1),
}

# Low-discrepancy nuisance parameters rather than a learned distribution.
primes=(2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131)
def sample(n):
    values=[]
    for j,p in enumerate(primes):
        # A fixed digit shift breaks the diagonal in high-prime dimensions.
        k=n+1; x=0.; w=1./p
        while k:
            k,t=divmod(k,p); x+=w*t; w/=p
        values.append((x*(1+((3*j+1)%(max(1,p//2)))) + math.sqrt(j+2))%1)
    return np.array(values)*2-1


def hand(letter,i):
    q=sample(i)
    names,th,turn,tilt,roll=lex[letter]; actions=names.split()
    sy=1+.10*q[7]; sx=1+.09*q[9]; fat=1+.12*q[10]; fl=1+.10*q[11]
    # Alternative articulations are common for these handshapes: M/N with
    # extended fingers draped over thumb, T also seen side-on; P is K turned
    # palm down; G and Q are often made with the thumb parallel to the index.
    if letter in 'MN' and i%3!=0:
        n=3 if letter=='M' else 2
        for d in range(n):actions[d]='shelf' if i%3==1 else 'flat'
        tilt+=13
    if letter=='T' and i%3!=0:
        actions[0]='shelf'; turn+=66; roll-=8
    if letter=='P':
        if i%2==0:roll,tilt,turn=37,70,-36
        else:roll,tilt,turn=156,39,32
    if letter=='Q' and i%3==1:th='wide'
    if letter=='K':
        if i%3==0:actions[1]='up'
        elif i%3==1:actions[1]='cross'
    if letter=='C' and i%4==2:th='wide'
    if letter=='E' and i%3==0:actions=['bend']*4
    if letter=='G' and i%4==0:th='wide'
    if letter=='F' and i%2==0:turn+=19
    if letter in 'BDRU' and i%4==1:turn+=20
    turn+=q[0]*(23 if letter not in 'CKGOHPQX' else 29)
    tilt+=q[1]*15; roll+=q[2]*(13 if letter not in 'GHPQ' else 20)
    cam=rz(roll) @ rx(tilt) @ ry(turn)
    bones=[]; body=[]; tips=[]
    # Palm cross-section at successive heights (width and depth radii).
    for y,wx,dz,xc,zc in [(-.20,.265,.125,-.035,-.035),(-.07,.34,.15,-.025,-.022),(.12,.414,.177,-.014,0),(.36,.46,.193,0,0),(.60,.49,.168,.004,-.006),(.79,.483,.145,.003,-.012),(.90,.422,.112,.022,-.012),(.946,.32,.085,.042,-.01)]:
        # Round the top toward the little finger; otherwise the palm would
        # protrude into the notch next to an extended pinky.
        body.append((np.array((xc*sx,y*sy,zc)),wx*sx,dz*fat))
    def chain(points,radii,kind):
        for a,b,r,s in zip(points,points[1:],radii,radii[1:]):
            bones.append((np.array(a),np.array(b),r,s,kind))
    for j,mode in enumerate(actions):
        bx=(.324,.100,-.131,-.342)[j]*sx
        by=(.883,.938,.903,.773)[j]*sy
        p=np.array([bx,by,(-.012,-.018,.006,.035)[j]])
        pts=[p]; mcp,pip,dip,ab=post[mode]
        ab+=(1.3,0,-1.4,-2.6)[j]
        if mode in ('fist','bend','rest','claw'):ab+=( -2.,-1.,1.,3.)[j]
        if mode in ('press','cross','up') and letter in 'BRU':
            ab+=(-2.8,-.6,1.8,3.8)[j]  # adduction, so the sides touch
        if letter in 'VW':ab+=(15,-12,0,0)[j] if letter=='V' else (18,1,-18,0)[j]
        if letter=='F':ab+=(0,9,-1,-16)[j]
        if letter in 'IY' and j==3:ab-=7 if letter=='I' else 16
        if letter=='R':ab+=(-15,7,0,0)[j]
        if letter in 'KP':ab+=(12,-12,0,0)[j]
        if letter in 'DBU' and j in (0,1):ab+=(-1,1,0,0)[j]
        if letter in 'DOTFC' and mode in ('round','close'):
            # The fingertip turns back toward the opposed thumb in an O.
            ab+=(-3,3,9,13)[j]
        if mode in ('press','cross','up'):mcp+=4*q[13+j]; pip+=4*q[15+j]
        else:
            mcp+=10*q[13]+7*q[14+j]; pip+=12*q[15]+6*q[16+j];dip+=9*q[17]
        if letter in 'CFDO' and mode in ('round','close'):
            mcp += (0,4,8,10)[j]; pip+=(0,3,4,5)[j]
        if letter=='K' and mode=='cross':mcp+=28
        if letter=='X' and j==0:pip+=10*q[8];ab-=6
        ab+= (3 if mode in ('press','up') and letter not in 'VWYF' else 6)*q[20+j]
        ab=math.radians(ab)
        length=(1,1.085,.997,.775)[j]*fl*(1+.045*q[18+j]); r=(.126,.130,.122,.107)[j]*fat
        for angle,ln in zip((mcp,mcp+pip,mcp+pip+dip),(.426,.262,.223)):
            u=math.radians(angle)
            p=p+ln*length*np.array([math.sin(ab)*math.cos(u), math.cos(ab)*math.cos(u),math.sin(u)])
            pts.append(p)
        chain(pts,[r,r*.955,r*.858,r*.80],j+1)
        tips.append(pts[-1])
    # Thenar pad and thumb, using a near-contact constraint for the O/F/D
    # pinch. Target comes from the current procedural finger, never a photo.
    pts=[np.array([.26*sx,.10*sy,.065])]
    for j,v in enumerate(thumbs[th]):
        p=np.array(v,dtype=float)*[sx,sy,1]
        p+=np.array([q[23],q[24],q[25]])*(j+1)*.019
        pts.append(p)
    if letter in 'FDO':
        aim=tips[1 if letter=='D' else 0]+np.array([.020,-.115,.024]); shift=aim-pts[-1]
        pts[-1]+=shift*.91;pts[-2]+=shift*.39;pts[-3]+=shift*.10
    if letter=='C' and th=='c':pts[-1][2]+=.09*q[8]
    if letter=='N':pts[-1][0]+=.17
    if th=='wide':
        pts[-1][1]+=.20*q[6];pts[-2][1]+=.09*q[6]
    chain(pts,[.171*fat,.153*fat,.124*fat,.106*fat],5)
    result=[]
    for a,b,r,s,kind in bones:result.append((cam@a,cam@b,r,s,kind))
    # The forearm can be flexed independently of the palm. The upright signs
    # have a mostly downward forearm, the pointing signs a mostly lateral one.
    wrist=cam@body[0][0]
    if letter in 'GH': angle=-47+24*q[4]
    elif letter in 'PQ':angle=68+30*q[4]
    else:angle=-8+15*q[4]-.46*roll
    d=math.radians(angle)
    fore=np.array([math.sin(d),-math.cos(d),-.11]); middle=wrist+.61*fore
    end=middle+2.6*fore
    # Separate smooth slightly flattened forearm, connected to palm heel.
    # Stored as an oriented ellipsoid sweep (the width is larger than depth).
    arm=(wrist,middle,end,.27*sx,.255*sx,.36*sx)
    return cam,body,result,arm,q


def sphere_mesh():
    # Elliptical rings give a slightly squared-off palm instead of a ball.
    n=24; a=np.arange(n)*2*np.pi/n
    return np.sign(np.cos(a))*np.abs(np.cos(a))**.84, np.sign(np.sin(a))*np.abs(np.sin(a))**.84
cirx,cirz=sphere_mesh()


def posed_geometry(letter,i):
    cam,body,bones,arm,q=hand(letter,i)
    rings=[]
    for p,wx,dz in body:
        ring=p+np.stack([wx*cirx,np.zeros_like(cirx),dz*cirz],1)
        # The little finger attaches lower than the middle finger.
        ring[:,1]-=np.maximum(-ring[:,0],0)*.30*max(0.,p[1])
        rings.append(ring@cam.T)
    bot=np.broadcast_to((cam@(body[0][0]+[0,-.028,0])),rings[0].shape)
    top=np.broadcast_to((cam@(body[-1][0]+[0,.04,0])),rings[-1].shape)
    rings=np.stack([bot]+rings+[top])
    # Frame the articulated hand (without the semi-infinite forearm). These
    # are its own analytic bounds, not statistics of other renderings.
    lo=rings.min((0,1))[:2]; hi=rings.max((0,1))[:2]
    for a,b,r,s,k in bones:
        lo=np.minimum(lo,np.minimum(a[:2]-r,b[:2]-s));hi=np.maximum(hi,np.maximum(a[:2]+r,b[:2]+s))
    return rings,bones,arm,lo,hi,q

# Orthographic z-buffer, oversized so one image can be resampled at several
# offsets and magnifications. Normals come from its frontmost surface.
SIDE=104
PIX=32.
GRID=(np.arange(SIDE,dtype=np.float32)-(SIDE-1)/2)/PIX
XX,YY=np.meshgrid(GRID,-GRID)

def section(low,high):
    start=max(0, min(SIDE,math.floor(low*PIX+(SIDE-1)/2)-1));stop=max(0,min(SIDE,math.ceil(high*PIX+(SIDE-1)/2)+2));return slice(start,stop)

def raster(geo,letter):
    rings,bones,arm,lo,hi,q=geo
    # Scale based only on this model's posed geometry. World coordinates now
    # have a hand span of two in the larger direction.
    scale=2/max(hi-lo); mid=(hi+lo)*.5
    def trans(p):return (p-np.array([mid[0],mid[1],0.]))*scale
    rings=trans(rings)
    z=np.full((SIDE,SIDE),-12.,dtype=np.float32); which=np.zeros((SIDE,SIDE),np.int8)
    # Triangle mesh for palm. A whole ring strip is evaluated with broadcast
    # barycentric coordinates in its small rectangular tile.
    def drawtris(a,b,c):
        cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
        d=b-a;e=c-a;det=cross(d,e)
        ok=np.abs(det)>1e-8
        if not np.any(ok):return
        a=a[ok];b=b[ok];c=c[ok];det=det[ok,None,None];d=d[ok];e=e[ok]; allp=np.vstack((a,b,c))
        rows=section(-allp[:,1].max(),-allp[:,1].min());cols=section(allp[:,0].min(),allp[:,0].max());sl=(rows,cols)
        x=XX[sl]-a[:,0,None,None];y=YY[sl]-a[:,1,None,None]
        u=(x*e[:,1,None,None]-y*e[:,0,None,None])/det
        v=(y*d[:,0,None,None]-x*d[:,1,None,None])/det
        dep=a[:,2,None,None]+u*d[:,2,None,None]+v*e[:,2,None,None]
        z[sl]=np.maximum(z[sl],np.where((u>=-.0001)&(v>=-.0001)&(u+v<=1.0001),dep,-12.).max(0))
    for a,b in zip(rings,rings[1:]):
        an=np.roll(a,-1,axis=0);bn=np.roll(b,-1,axis=0)
        drawtris(np.vstack([a,an]),np.vstack([an,bn]),np.vstack([b,b]))
    # Depth of a swept sphere. Use exact line-of-sight intersection with a
    # cylindrical shaft, and spherical caps; taper approximate along shaft.
    def segment(a,b,r,s,kind,zflatten=1.):
        m=max(r,s)
        sl=(section(-max(a[1],b[1])-m,-min(a[1],b[1])+m),section(min(a[0],b[0])-m,max(a[0],b[0])+m))
        x=XX[sl]-a[0]; y=YY[sl]-a[1]
        dx,dy,dz=b-a; dz/=zflatten
        lxy=dx*dx+dy*dy
        u=(dx*x+dy*y)/max(lxy,1e-8)
        r2=(r+np.clip(u,0,1)*(s-r))**2
        perp2=(dy*x-dx*y)**2/max(lxy,1e-8)
        delta=np.sqrt(np.maximum(r2-perp2,0.)/(lxy+dz*dz+1e-10))
        t=u+dz*delta/np.sqrt(max(lxy,1e-8))
        side=dz*u+delta*(lxy+dz*dz)/np.sqrt(max(lxy,1e-8))
        dep=np.where((t>=0)&(t<=1)&(perp2<=r2)&(lxy>1e-8),side,-40.)
        for px,py,pz,rr in ((0,0,0,r),(dx,dy,dz,s)):
            v=rr*rr-(x-px)**2-(y-py)**2
            dep=np.maximum(dep,np.where(v>=0,pz+np.sqrt(np.maximum(v,0)),-40.))
        dep=dep*zflatten+a[2]; front=dep>z[sl];z[sl]=np.where(front,dep,z[sl]);which[sl]=np.where(front,kind,which[sl])
    for a,b,r,s,kind in bones:segment(trans(a),trans(b),r*scale,s*scale,kind)
    a,b,c,r,s,t=arm
    segment(trans(a),trans(b),r*scale,s*scale,6,.62);segment(trans(b),trans(c),s*scale,t*scale,6,.61)
    inside=z>-11
    # Smooth the depth normals locally but don't bridge discontinuities
    # between one finger and the next or between hand and background.
    def dz(axis):
        l=np.roll(z,1,axis=axis);r=np.roll(z,-1,axis=axis)
        left=z-l;right=r-z
        s1=np.exp(-np.minimum((left*PIX*.36)**2,20));s2=np.exp(-np.minimum((right*PIX*.36)**2,20))
        return (left*s1+right*s2)/(s1+s2+1e-6)*PIX
    nx=-dz(1);ny=dz(0);den=np.sqrt(1+nx*nx+ny*ny)
    light=np.array([-.65+.38*q[25],.58+.34*q[26],.51+.30*q[27]])
    light/=np.linalg.norm(light)
    diffuse=(light[0]*nx+light[1]*ny+light[2])/den
    # Short cast shadows from neighboring fingers and occlusion in creases.
    shade=np.zeros_like(z)
    for step in (2,4,7):
        cmp=np.roll(z,(-int(round(-light[1]*step)),-int(round(light[0]*step))),axis=(0,1))
        shade+=np.clip((cmp-z-step/PIX*light[2])*7,0,1)
    skin=.41+.35*diffuse +.029*np.tanh(z)-.057*shade
    # Differences in skin and illumination are nuisance variables, not labels.
    # Allow either ordering of hand and background brightness.
    bg=.67+.16*q[29] if q[28]<.65 else .26+.12*q[29]
    skin+=.09*q[30]
    scene=np.where(inside,skin,bg).astype(np.float32)
    mask=inside.astype(np.float32)
    for j in range(2):scene=blur(scene);mask=blur(mask)
    return scene,mask,float((hi[1]-lo[1])*scale),float((hi[0]-lo[0])*scale)


def camera_views(scene,mask,h,w,letter,i,number):
    # Camera crop has a top margin in upright signs and centers the full
    # extent in side poses. Offsets sample physical crop/zoom uncertainty.
    a=np.arange(28,dtype=np.float32)-13.5
    worlds=[]
    for j in range(number):
        u=sample(i*number+j+173)
        span=23.7 + 3.6*u[0] + 1.3*u[5]
        aspect=1+.10*u[3]
        sx=span/2*aspect; sy=span/2/aspect
        cx=13.5+2.2*u[1]+.75*u[6]; top=2.0+1.75*u[2]+.8*u[7]
        cy=top + h*sy*.5
        if letter in 'GHPQ':cy=13.3+2.7*u[2]
        x=(a-(cx-13.5))/sx*PIX+(SIDE-1)/2
        y=(a-(cy-13.5))/sy*PIX+(SIDE-1)/2
        worlds.append(np.stack((x,y)))
    raw=np.stack(worlds);x=raw[:,0,None,:];y=raw[:,1,:,None];xi=np.floor(x).astype(int);yi=np.floor(y).astype(int)
    fx=x-xi;fy=y-yi;xi=np.clip(xi,0,SIDE-2);yi=np.clip(yi,0,SIDE-2)
    def interp(a):return ((1-fy)*((1-fx)*a[yi,xi]+fx*a[yi,xi+1])+fy*((1-fx)*a[yi+1,xi]+fx*a[yi+1,xi+1])).astype(np.float32)
    return interp(scene),interp(mask)


def main(folder):
    os.makedirs(folder,exist_ok=True)
    with open(os.path.join(folder,'solution.py'),'w') as f:f.write(features+predictor)
    poses,views=240,9
    coarse=np.empty((24*poses*views,8*7*7),np.int8);finebank=np.empty((24*poses*views,8*14*14),np.int8)
    for k,letter in enumerate(letters):
        for base in range(0,poses,8):
            imgs=[];masks=[]
            for i in range(base,base+8):
                img,m,h,w=raster(posed_geometry(letter,i),letter)
                im,ma=camera_views(img,m,h,w,letter,i,views);imgs.append(im);masks.append(ma)
            e,m=channels(np.concatenate(imgs),np.concatenate(masks))
            sl=slice((k*poses+base)*views,(k*poses+base+8)*views)
            for bank,mat in ((coarse,wide),(finebank,fine)):
                bank[sl]=np.clip(np.rint(descriptor(e,m,mat)*512),-127,127).astype(np.int8)
    np.savez(os.path.join(folder,'constants.npz'),coarse=coarse,fine=finebank)

if __name__=='__main__':
    exec(features)
    main(sys.argv[1])
