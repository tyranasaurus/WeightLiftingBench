import sys, json, math
from pathlib import Path
import numpy as np

# All models below are geometric drawings of ASL hand shapes.  No data is read.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/app/out')
OUT.mkdir(parents=True, exist_ok=True)

layers = [
 {'name':'input_norm','op':'norm','inputs':['image']},
 {'name':'smooth','op':'conv','inputs':['input_norm'],'out':1,'kernel':[5,5],'stride':1,'padding':2},
 {'name':'smooth_norm','op':'norm','inputs':['smooth']},
 {'name':'grad','op':'conv','inputs':['smooth_norm'],'out':4,'kernel':[3,3],'stride':1,'padding':1},
 {'name':'grad_norm','op':'norm','inputs':['grad']},
 {'name':'grad_pos','op':'relu','inputs':['grad_norm']},
 {'name':'grad_negconv','op':'conv','inputs':['grad_norm'],'out':4,'kernel':[1,1],'stride':1,'padding':0},
 {'name':'grad_neg','op':'relu','inputs':['grad_negconv']},
 {'name':'grad_abs','op':'add','inputs':['grad_pos','grad_neg']},
 {'name':'edge_pool','op':'avgpool','inputs':['grad_abs'],'size':[2,2],'stride':[2,2]},
 {'name':'edge_norm','op':'norm','inputs':['edge_pool']},
 {'name':'raw_pool','op':'avgpool','inputs':['smooth'],'size':[2,2],'stride':[2,2]},
 {'name':'raw_norm','op':'norm','inputs':['raw_pool']},
 {'name':'raw_score','op':'dense','inputs':['raw_norm'],'out':432},
 {'name':'edge_score','op':'dense','inputs':['edge_norm'],'out':432},
 {'name':'combined','op':'add','inputs':['raw_score','edge_score']},
 {'name':'variants','op':'reshape','inputs':['combined'],'shape':[24,18,1]},
 {'name':'best','op':'maxpool','inputs':['variants'],'size':[18,1],'stride':[18,1]},
 {'name':'scores','op':'reshape','inputs':['best'],'shape':[24]}
]

# A compact analytic hand-drawing kit. Coordinates are anatomical: y points up.
def capsule_distance(x,y,a,b):
    ax,ay=a; bx,by=b
    vx=bx-ax; vy=by-ay
    t=((x-ax)*vx+(y-ay)*vy)/(vx*vx+vy*vy+1e-12)
    t=np.clip(t,0,1)
    return np.sqrt((x-ax-t*vx)**2+(y-ay-t*vy)**2)

def hand_spec(kind):
    # parts are (ax,ay,bx,by,radius), ellipses are (cx,cy,rx,ry)
    parts=[(-.04,-.45,-.06,-1.25,.38)]
    ell=[(0,-.12,.72,.68)]
    creases=[]
    bases=[-.47,-.16,.16,.44]
    def curled(which=(0,1,2,3),height=.60):
        for j in which:
            x=bases[j]
            parts.append((x,.18,x+.025,height,.17))
            creases.append((x-.105,.39,x-.09,.60))
    def finger(j,tx,ty,r=.155):
        x=bases[j]
        parts.append((x,.25,tx,ty,r))
    def thumb(a,b,r=.16): parts.append((a[0],a[1],b[0],b[1],r))
    if kind=='A':
        curled(height=.64); thumb((.43,-.45),(.60,.40),.17)
        creases += [(-.37,.39,.35,.43),(.43,-.20,.48,.32)]
    elif kind=='B':
        for j,tx,ty in [(0,-.43,1.30),(1,-.14,1.48),(2,.14,1.37),(3,.39,1.10)]: finger(j,tx,ty)
        thumb((-.47,-.30),(.28,.12),.15)
        creases += [(-.43,-.03,.25,.22),(-.28,.25,-.28,.85),(-.01,.28,-.01,.95),(.25,.22,.27,.76)]
    elif kind=='C':
        ell=[(-.22,-.25,.48,.65)]
        parts=[(-.20,-.55,-.20,-1.25,.32),
               (-.43,.05,-.48,.65,.19),(-.48,.65,-.25,1.00,.17),
               (-.25,1.00,.18,.93,.16),(.18,.93,.48,.55,.15),
               (-.42,-.18,.05,-.05,.18),(.05,-.05,.50,-.05,.16)]
        creases=[(-.24,.54,.16,.67),(-.17,.75,.22,.79)]
    elif kind=='D':
        finger(0,-.43,1.43); curled((1,2,3),.66)
        thumb((-.48,-.25),(.16,.60),.16)
        creases += [(-.12,.42,.29,.56),(-.30,.62,.13,.70)]
    elif kind=='E':
        curled(height=.48); thumb((-.47,-.25),(.37,.15),.16)
        creases += [(-.36,.20,.33,.28),(-.30,.43,-.28,.53),(-.04,.43,-.02,.53),(.22,.40,.24,.50)]
    elif kind=='F':
        finger(1,-.05,1.43); finger(2,.25,1.34); finger(3,.48,1.12)
        parts += [(-.39,.25,-.43,.68,.14),(-.43,.68,-.20,.82,.14)]
        thumb((-.48,-.25),(-.60,.48),.16); thumb((-.60,.48),(-.22,.77),.14)
        creases += [(.08,.30,.08,.85),(.36,.26,.38,.75)]
    elif kind in ('G','Q'):
        finger(0,-.43,1.42); curled((1,2,3),.54)
        thumb((-.45,-.20),(-.60,.64),.17)
        creases += [(-.18,.30,.30,.39)]
    elif kind=='H':
        finger(0,-.40,1.34); finger(1,-.10,1.40); curled((2,3),.53)
        thumb((-.46,-.28),(.26,.10),.15)
        creases += [(-.25,.40,-.24,1.10)]
    elif kind=='I':
        curled((0,1,2),.58); finger(3,.43,1.25,.13)
        thumb((-.46,-.30),(-.57,.36),.16)
        creases += [(-.33,.30,.19,.40)]
    elif kind=='K':
        finger(0,-.57,1.36); finger(1,.13,1.43); curled((2,3),.54)
        thumb((-.46,-.27),(-.62,.62),.16)
        creases += [(-.28,.38,-.36,.96),(-.13,.32,.27,.40)]
    elif kind=='L':
        finger(0,-.40,1.43); curled((1,2,3),.57)
        thumb((-.44,-.20),(-1.03,.13),.17)
        creases += [(-.25,.28,.25,.38)]
    elif kind in ('M','N'):
        curled(height=.61 if kind=='M' else .57)
        thumb((.43,-.42),(-.23,.22),.15)
        creases += [(-.36,.32,.26,.39),(-.28,.48,-.25,.60)]
        if kind=='M': creases += [(-.01,.47,.01,.60),(.25,.45,.27,.57)]
    elif kind=='O':
        # A closed ring of thumb and fingers above the palm leaves a real hole.
        parts=[(-.04,-.45,-.06,-1.25,.32),(-.43,.05,-.48,.56,.18),
               (-.48,.56,-.30,.88,.16),(-.30,.88,.05,.94,.15),
               (.05,.94,.36,.70,.15),(.36,.70,.24,.45,.15),
               (-.43,-.15,.24,.45,.17)]
        creases=[(-.29,.50,-.18,.77),(.09,.51,.23,.69)]
    elif kind=='P':
        return hand_spec('K')
    elif kind=='R':
        finger(0,.03,1.42); finger(1,-.31,1.40); curled((2,3),.54)
        thumb((-.45,-.25),(.26,.12),.15)
        creases += [(-.21,.70,-.04,.88)]
    elif kind=='S':
        curled(height=.61); thumb((-.47,-.12),(.43,.32),.16)
        creases += [(-.39,.12,.39,.31),(-.28,.45,-.26,.59),(-.01,.46,.01,.61),(.25,.43,.27,.57)]
    elif kind=='T':
        curled(height=.53); thumb((-.40,-.18),(-.26,.99),.17)
        creases += [(-.33,.28,.30,.36)]
    elif kind=='U':
        finger(0,-.36,1.38); finger(1,-.09,1.43); curled((2,3),.54)
        thumb((-.46,-.25),(.25,.10),.15)
        creases += [(-.23,.48,-.22,1.16)]
    elif kind=='V':
        finger(0,-.68,1.29); finger(1,.25,1.46); curled((2,3),.54)
        thumb((-.46,-.25),(.25,.10),.15)
    elif kind=='W':
        finger(0,-.65,1.29); finger(1,-.08,1.49); finger(2,.48,1.31); curled((3,),.52)
        thumb((-.46,-.25),(.37,.18),.15)
        creases += [(-.37,.38,-.43,.91),(.17,.40,.30,.92)]
    elif kind=='X':
        curled((1,2,3),.57)
        parts += [(-.39,.23,-.43,.91,.14),(-.43,.91,-.16,.84,.14)]
        thumb((-.46,-.30),(.24,.12),.15)
        creases += [(-.32,.30,.25,.40)]
    elif kind=='Y':
        curled((0,1,2),.55); finger(3,.82,.91,.14)
        thumb((-.44,-.25),(-1.00,.55),.18)
        creases += [(-.28,.29,.23,.38)]
    return parts,ell,creases

def draw_hand(kind,theta=0,scale=8.2,cx=14,cy=14,res=112):
    parts,ells,creases=hand_spec(kind)
    yy,xx=np.mgrid[0:res,0:res].astype(np.float64)
    # Convert pixels to the model's rotated anatomical coordinate system.
    px=(xx+(0.5-res/2))/(4*scale); py=-(yy+(0.5-res/2))/(4*scale)
    px=px-(cx-14)/scale; py=py+(cy-14)/scale
    a=math.radians(theta); co=math.cos(a); si=math.sin(a)
    x=co*px-si*py; y=si*px+co*py
    d=np.full(x.shape,10.0)
    for ax,ay,bx,by,r in parts: d=np.minimum(d,capsule_distance(x,y,(ax,ay),(bx,by))-r)
    for ex,ey,rx,ry in ells:
        # Scaled elliptical distance is adequate for antialiased silhouette drawing.
        q=np.sqrt(((x-ex)/rx)**2+((y-ey)/ry)**2)
        d=np.minimum(d,(q-1)*min(rx,ry))
    mask=np.clip(.5-d*scale*4*1.4,0,1)
    # Generic soft shading and anatomical creases, rather than any observed pixels.
    bg=.22+.06*(yy/res)+.025*np.sin(xx/res*math.pi)
    hand=.65+.12*np.clip((y+.8)/2.4,0,1)+.035*np.cos(x*2)
    for ax,ay,bx,by in creases:
        line=np.exp(-(capsule_distance(x,y,(ax,ay),(bx,by))/.035)**2)
        hand=hand-.20*line
    im=bg*(1-mask)+hand*mask
    # Antialiasing/downsampling is part of the generic drawing model.
    return im.reshape(28,4,28,4).mean((1,3)).astype(np.float32)

def gaussian():
    g=np.array([1,4,6,4,1],np.float32); g=g/g.sum()
    return (g[:,None]*g[None,:])[None,None,:,:]

def conv(x,w,pad):
    xp=np.pad(x,((pad,pad),(pad,pad)))
    k=w.shape[-1]; ho=x.shape[0]+2*pad-k+1; wo=x.shape[1]+2*pad-k+1
    return np.stack([sum(w[o,0,dy,dx]*xp[dy:dy+ho,dx:dx+wo]
                         for dy in range(k) for dx in range(k)) for o in range(len(w))],0)

def norm(x):
    mu=x.mean(axis=(-2,-1),keepdims=True)
    return (x-mu)/np.sqrt(((x-mu)**2).mean(axis=(-2,-1),keepdims=True)+1e-5)

def pool(x):
    return x.reshape(x.shape[0],14,2,14,2).mean((2,4))

kinds=list('ABCDEFGHIKLMNOPQRSTUVWXY')
base_angle={'G':90,'H':90,'P':-105,'Q':180,'T':90}
variants=[(0,8.2,14,14),(-5,7.7,13,14),(5,7.7,15,14),(0,8.8,14,14.5),
          (0,7.3,14,13.5),(0,8.2,11.5,14),(0,8.2,16.5,14),
          (0,8.2,14,11.8),(0,8.2,14,16)]
raw_templates=[]; edge_templates=[]; raw_gains=[]; edge_gains=[]
for kind in kinds:
    for polarity in (1,-1):
        for da,sc,cx,cy in variants:
            im=draw_hand(kind,base_angle.get(kind,0)+da,sc,cx,cy)
            sm=conv(norm(im),gaussian(),2)[0]
            smn=norm(sm)
            gw=np.zeros((4,1,3,3),np.float32)
            gw[0,0]=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])/8
            gw[1,0]=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/8
            gw[2,0]=np.array([[0,1,2],[-1,0,1],[-2,-1,0]])/4
            gw[3,0]=np.array([[2,1,0],[1,0,-1],[0,-1,-2]])/4
            gr=norm(conv(smn,gw,1))
            edge=pool(np.abs(gr))
            raw=pool(sm[None])[0]
            mask=pool((im>.4).astype(np.float32)[None])[0]
            raw_templates.append((polarity*mask).ravel()); edge_templates.append(edge.ravel())
            raw_gains.append(float((norm(raw)*mask).sum()/(mask.sum()+1e-8)))
            edge_gains.append(float((norm(edge)*edge).sum()/(edge.sum()+1e-8)))
raw_templates=np.stack(raw_templates); edge_templates=np.stack(edge_templates)
raw_templates=raw_templates/(np.array(raw_gains)[:,None]*np.abs(raw_templates).sum(axis=1,keepdims=True)+1e-6)
edge_templates=edge_templates/(np.array(edge_gains)[:,None]*np.abs(edge_templates).sum(axis=1,keepdims=True)+1e-6)
weights={}
weights['smooth.W']=gaussian().astype(np.float32); weights['smooth.b']=np.zeros(1,np.float32)
weights['grad.W']=gw; weights['grad.b']=np.zeros(4,np.float32)
neg=np.zeros((4,4,1,1),np.float32)
for i in range(4): neg[i,i,0,0]=-1
weights['grad_negconv.W']=neg; weights['grad_negconv.b']=np.zeros(4,np.float32)
# A perfect standardized feature match has correlation one. Mix the two cues explicitly.
weights['raw_score.W']=(.30*raw_templates).astype(np.float32)
weights['raw_score.b']=np.zeros(432,np.float32)
weights['edge_score.W']=(.70*edge_templates).astype(np.float32)
weights['edge_score.b']=np.zeros(432,np.float32)
(OUT/'network.json').write_text(json.dumps({'layers':layers},indent=2))
np.savez(OUT/'weights.npz',**weights)
