"""Compile a hand-specification of ten pen-and-paper shapes into a network.

There are no external inputs other than the destination directory. The coordinate
system in the pen grammar below is a notional, upright unit square, not the
coordinate system of an example image. All ranges describe drawing choices.
"""
import json
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(300278)
U = rng.uniform
PI = np.pi
T = np.linspace(0, 1, 17)[:, None]
B = np.concatenate([(1-T)**3, 3*(1-T)**2*T, 3*(1-T)*T*T, T**3], axis=1)


class Pen:
    def __init__(self):
        self.paths = []
    def move(self, x, y):
        self.paths.append([np.array((x,y), dtype=float)])
    def line(self, x, y):
        start = self.paths[-1][-1]
        self.paths[-1].extend(start*(1-T[1:]) + np.array((x,y))*T[1:])
    def arc(self, ax, ay, bx, by, x, y):
        points = (self.paths[-1][-1], (ax,ay), (bx,by), (x,y))
        self.paths[-1].extend((B @ np.asarray(points))[1:])
    def oval(self, cx, cy, w, h, rot=0., exponent=1., taper=0.,
             start=0., sweep=2*PI):
        t = np.linspace(start, start+sweep, 65)
        si, co = np.sin(t), np.cos(t)
        dx = w * np.sign(co) * abs(co)**exponent * (1+taper*si)
        dy = h * np.sign(si) * abs(si)**exponent
        c,s = np.cos(rot),np.sin(rot)
        self.paths.append(list(np.stack([cx+c*dx-s*dy,cy+s*dx+c*dy], axis=1)))


def draw(k):
    p = Pen()
    move, line, arc = p.move, p.line, p.arc
    style = U()
    if k == 0:
        p.oval(.50,.50,.48,.50,U(-.2,.2),U(.72,1.16),U(-.23,.23),
               start=U(-2.25,-.72), sweep=2*PI*U(.955,1.05))
        if style < .15:  # small upturned overlap at the end of the loop
            x,y = p.paths[-1][-1]
            arc(x-.09,y+.01,x-.16,y+.08,x-.18,y+U(.08,.22))
    elif k == 1:
        # 1 is usually a single slightly bowed downstroke; its width then
        # comes from the pen, shear, and bow, not a fixed bounding box.
        x = U(-.06,.06); bottom = U(-.10,.10)
        if style < .31:
            move(-U(.07,.41),U(.10,.40)); line(x,0.)
        else: move(x,0.)
        bow = U(-.17,.17)
        arc(x+bow,U(.14,.42),bottom+bow,U(.66,.89),bottom,1.)
        if .19 < style < .29:
            move(bottom-U(.11,.33),U(.96,1.02)); line(bottom+U(.12,.36),1.)
    elif k == 2:
        # Arch and descending diagonal, followed by a baseline flourish.
        # In cursive the diagonal turns back through a small lower-left loop.
        left, peak = U(-.12,.18), U(.43,.85)
        move(left,U(.11,.40))
        arc(left-.02,U(.00,.12),peak-.23,U(-.10,.04),peak,0.)
        arc(peak+.26,.01,1.01,U(.17,.32),U(.78,.97),U(.31,.45))
        if style < .47:
            bx = U(-.10,.12)
            arc(.72,.52,bx+.12,.85,bx,1.)
            arc(bx+.22,U(.88,1.03),.70,U(.88,1.06),U(.88,1.20),U(.83,1.01))
        elif style < .65:
            # A cursive swan-neck 2 returns most of the way upward in its
            # last flourish; the turn at the foot is sometimes quite round.
            arc(.58,.57,-.08,.85,.03,.97)
            arc(.18,1.13,.80,U(.78,1.01),U(.93,1.27),U(.46,.78))
        else:
            # A narrow loop where the '2' doubles back on its pen path.
            lx = U(-.25,.05); by = U(.91,1.02); high=U(.59,.88)
            arc(.66,.64,lx+.18,by+.11,lx,by)
            arc(lx-.20,by-.12,lx-.02,high-.11,lx+.22,high)
            arc(.42,high+.06,.62,1.09,U(.96,1.29),U(.78,1.02))
    elif k == 3:
        waist, wy = U(.24,.66), U(.38,.59)
        move(U(-.12,.14),U(.04,.22))
        if style < .68:
            arc(.20,-.03,.59,-.10,U(.69,1.),U(.02,.18))
            arc(1.10,.26,.77,wy-.04,waist,wy)
        else:
            arc(.25,.045,.64,-.015,U(.85,1.04),.015)
            line(waist,wy)
        if style > .76:
            arc(waist-U(.15,.38),wy+.14,waist-U(.18,.36),wy-.08,waist+.04,wy-.04)
        arc(.79,wy-.10,1.10,wy+.06,U(.88,1.04),U(.65,.83))
        arc(.90,1.06,.29,1.11,U(-.16,.15),U(.76,.99))
    elif k == 4:
        bar, stem = U(.43,.74), U(.58,.96)
        if style < .75:  # open 4: separate down-turn-across and down strokes
            top = U(.06,.46); corner = U(-.06,.13)
            move(top,U(-.04,.15))
            arc(top-.10,.22,corner-.05,bar-.14,corner,bar)
            arc(corner+.18,bar+.08,.74,bar+U(-.05,.05),stem+U(-.05,.28),bar+U(-.10,.03))
            move(stem+U(-.06,.12),U(-.07,.21))
            arc(stem-.05,.38,stem-.02,.70,stem+U(-.06,.07),1.)
        else:  # triangular or nearly closed form
            move(stem+U(-.22,.04),0.)
            line(U(-.13,.12),bar); line(stem+U(.02,.30),bar+U(-.08,.04))
            move(stem+U(-.07,.07),U(-.04,.13)); line(stem,1.)
    elif k == 5:
        top = U(.04,.34); lx=U(-.05,.17); mid=U(.31,.57)
        if style > .78:
            # In fast writing the five can look like an S with a short
            # angular cap: the belly curls back well to the left.
            move(U(.72,1.26),U(-.09,.06))
            arc(.65,.06,.30,-.12,.11,.12)
            arc(-.10,.35,.28,mid-.02,U(.64,.86),mid+.06)
            arc(1.33,.70,.76,.99,.38,1.)
            arc(.23,1.04,.07,.98,U(-.17,.21),U(.83,1.02))
            return [np.array(s) for s in p.paths]
        if style < .56:
            # upper flag, left staff, then the clockwise belly
            move(U(.74,1.12),U(-.10,.08))
            arc(.77,.03,.42,.035,top,0.)
            arc(top-.04,.16,lx+.04,mid-.10,lx,mid)
        else:
            # separate cross stroke in the two-stroke version
            move(U(-.05,.15),U(-.05,.04)); arc(.42,.04,.66,.03,U(.78,1.15),U(-.08,.06))
            move(top,U(-.09,.08)); line(lx,mid)
        arc(lx+.27,mid-.16,.78,mid-.18,U(.92,1.02),mid+U(.03,.14))
        arc(1.12,.81,.78,1.01,U(.38,.60),1.)
        arc(.20,1.04,.05,.97,U(-.18,.14),U(.78,1.01))
    elif k == 6:
        join = U(.52,.82); end=U(.32,.68); ytop=U(.42,.62)
        move(U(.53,1.16),U(-.06,.04))
        arc(.34,.12,-.04,.48,.02,join)
        arc(.01,.97,.24,1.04,.53,1.)
        arc(.91,1.,1.14,.85,U(.93,1.00),U(.66,.82))
        arc(.92,ytop-.05,.58,ytop-.08,end,ytop)
        if style < .72:
            arc(.20,ytop+.025,.12,join+.015,U(.00,.16),join+U(.00,.13))
        else:  # an unfinished counter
            line(U(.18,.39),join)
    elif k == 7:
        tip = U(.82,1.12); sy=U(-.04,.09); bx=U(.10,.63)
        x = U(-.12,.15)
        if style < .27: move(x-.06,U(.10,.26)); line(x,sy)
        else: move(x,sy)
        arc(.30,U(-.07,.04),.69,U(-.09,.09),tip,U(-.03,.10))
        bend = U(-.09,.13)
        arc(tip-.12+bend,.30,bx+.14+bend,.68,bx,1.)
        if .27 <= style < .48:
            y=U(.42,.65); mx=tip*(1-y)+bx*y
            move(mx-U(.19,.38),y+.02); line(mx+U(.14,.34),y-.025)
    elif k == 8:
        # Trace an actual figure-eight: diagonals cross at its waist. The
        # start/end may leave a small seam. Also allow a pair of oval bowls.
        cross, waist = U(.37,.65),U(.40,.58)
        a,b = U(.27,.58),U(.32,.60)
        if style < .12:
            p.oval(cross,waist/2,a,waist/2,U(-.18,.18),U(.88,1.12))
            p.oval(cross+U(-.08,.08),(1+waist)/2,b,(1-waist)/2,U(-.17,.17))
        else:
            move(cross,waist)
            arc(cross-.30,waist-.14,cross-a-.16,.15,cross-a*.72,.056)
            arc(cross-a*.20,-.061,cross+a*.85,-.047,cross+a,.12)
            arc(cross+a+.07,.29,cross+.21,waist-.09,cross,waist)
            arc(cross-.26,waist+.12,cross-b-.11,.77,cross-b*.84,.90)
            arc(cross-b*.50,1.09,cross+b*.77,1.06,cross+b,.90)
            arc(cross+b+.12,.72,cross+.24,waist+.14,cross+U(-.06,.11),waist+U(-.07,.07))
            if style > .51:
                p.paths[0]=[np.array((1-q[0],q[1])) for q in p.paths[0]]
    else:
        # 9: overtraced top bowl and descender. Change which side of the
        # stem is tangent, the bowl height, and how the descender hooks.
        top = U(.60,.90); ry=U(.31,.61); stem=U(.84,1.02)
        if style < .76:
            move(top+U(-.02,.07),U(.025,.15))
            arc(top-.17,-.085,.17,-.005,U(.03,.13),ry*.30)
            arc(-.10,ry*.70,.05,ry*1.02,.28,ry)
            arc(.57,ry*1.06,stem,ry*.54,stem,U(-.04,.10) if style < .52 else ry*.35)
        else:
            p.oval(.49,ry/2,.46,ry/2,U(-.15,.16),U(.82,1.15))
            move(stem,U(.03,ry*.55))
        foot=stem+U(-.42,.13)
        arc(stem+U(-.10,.14),.46,foot+.06,.80,foot,1.)
        if style>.90:
            arc(foot-.08,1.07,foot-.32,1.05,foot-.45,.92)
    return [np.array(s) for s in p.paths]


def paint(k):
    paths = draw(k)
    aspect = U(.49,1.04) if k != 1 else U(.82,1.15)
    if k == 0: aspect=U(.41,.98)
    if k == 2: aspect=U(.61,1.23)
    if k == 8: aspect=U(.55,1.02)
    # Random smooth pen deformation in drawing coordinates. All styles are
    # generated, never selected by resemblance to actual handwriting images.
    shear = np.clip(rng.normal(.13,.25),-.54,.82)
    if k==1: shear = U(-.40,.78)
    rise = U(-.21,.18)
    xbow, ybow = U(-.10,.10),U(-.065,.065)
    twists = rng.uniform(-.055,.055,6)
    rough = rng.uniform(-.045,.045,(2,3,3))
    coords = []
    for a in paths:
        x,y = a.T
        x=x-.5; y=y-.5
        dx = np.sin(PI*(y+.5))*xbow + twists[0]*np.sin(2*PI*y) + twists[1]*np.sin(PI*x)*np.sin(PI*y)
        dy = np.sin(PI*(x+.5))*ybow + twists[2]*np.sin(2*PI*x) + twists[3]*np.sin(PI*x)*np.sin(PI*y)
        for i,u in enumerate([-.5,0,.5]):
            for j,v in enumerate([-.5,0,.5]):
                bump=np.exp(-((x-u)**2+(y-v)**2)*9.)
                dx += rough[0,i,j]*bump; dy += rough[1,i,j]*bump
        xx, yy = aspect*(x+dx) - shear*(y+dy), y+dy+rise*x
        coords.append(np.stack([xx,yy],axis=1))
    xy=np.concatenate(coords); lo=xy.min(axis=0); hi=xy.max(axis=0)
    meanwidth = U(1.55,3.65) + (U(.0,.75) if k==1 else 0)
    scale = (U(19.35,20.65)-meanwidth)/max(hi-lo)
    # Densely sample the centerline and rasterize a pen disk with round ends.
    # Use a local stamp to avoid a full image-by-curve distance array.
    pointlist=[]; radii=[]
    pressure=U(.03,.21); phase=U(-PI,PI); nib = U(-.14,.14)
    for cd in coords:
        cd=(cd-(lo+hi)*.5)*scale+16
        seg=np.diff(cd,axis=0); length=np.sqrt((seg*seg).sum(axis=1)); total=np.r_[0.,np.cumsum(length)]
        t=np.linspace(0,total[-1],max(2,int(total[-1]/.25)+1))
        ix=np.minimum(np.searchsorted(total[1:],t,side='right'),len(seg)-1)
        l=length[ix]; frac=(t-total[ix])/np.maximum(l,1.e-8)
        pointlist.append(cd[ix]+seg[ix]*frac[:,None]); r=meanwidth*.5
        radii.append(r*(1.+pressure*np.cos(2*PI*t/max(total[-1],1e-8)+phase)+nib*seg[ix,1]/np.maximum(l,1e-8)))
    pts=np.concatenate(pointlist); rad=np.concatenate(radii)
    def stamp(pts,n):
        centers=np.floor(pts+.5).astype(int)
        dxy=OFF[None,:,:] + centers[:,None,:]-pts[:,None,:]
        val=np.clip(rad[:,None]+.5-np.sqrt((dxy**2).sum(axis=2)),0.,1.)
        at=centers[:,0]*1 + centers[:,1]*n
        # centerlines may approach panel borders; don't wrap around
        ii=centers[:,None,:]+OFF
        good=(ii>=0).all(axis=2)&(ii<n).all(axis=2)&(val>0)
        buf=np.zeros(n*n)
        np.maximum.at(buf,(at[:,None]+OFF[:,0]+n*OFF[:,1])[good],val[good]); return buf.reshape(n,n)
    image=stamp(pts,32)
    m=image.sum(); com=np.array((image.sum(0)@np.arange(32),image.sum(1)@np.arange(32)))/m
    pts=pts-com+14.+rng.uniform(-.70,.70,2)
    return stamp(pts,28).astype('f4')


OFF=np.stack(np.meshgrid(np.arange(-3,4),np.arange(-3,4)),axis=-1).reshape(-1,2)

# Filters have two slightly different goals. A broad pixel channel supplies
# ink/space correlation. Direction-selective edge pairs distinguish crossing,
# curved and parallel strokes even when their thickness is a little different.
y,x=np.mgrid[-2:3,-2:3]
normal_blur=np.exp(-(x*x+y*y)/(2*.90**2)); normal_blur/=normal_blur.sum()
K=[]
for theta in [0,PI/4,PI/2,3*PI/4]:
    d=x*np.cos(theta)+y*np.sin(theta); a=-x*np.sin(theta)+y*np.cos(theta)
    f=d*np.exp(-.5*(d*d/.91**2+a*a/1.28**2)); f/=f[f>0].sum()
    K.extend((f,-f))
K=np.array(K,dtype='f4'); normal_blur=np.array(normal_blur,dtype='f4')


def norm(a,axes):
    d=a-a.mean(axis=axes,keepdims=True)
    return d / np.sqrt((d*d).mean(axis=axes,keepdims=True)+1.e-5)


def signatures(images):
    # Apply the explicit formulas to *our own drawings* to compile normalized
    # matched filters. This is never called on any outside images.
    sw=np.lib.stride_tricks.sliding_window_view
    win=sw(np.pad(images,((0,0),(2,2),(2,2))), (5,5), axis=(1,2))
    response=win.reshape(-1,25) @ np.concatenate([normal_blur[None],K]).reshape(9,25).T
    response=response.reshape(len(images),28,28,9).transpose(0,3,1,2)
    pix=norm(response[:,0,::2,::2],(1,2)).reshape(len(images),-1)
    edge=np.maximum(response[:,1:],0).reshape(len(images),8,7,4,7,4).mean(axis=(3,5))
    # Unsigned tangent/ridge orientation: smooth rectified opposite edge
    # responses together. This is less sensitive to pen width than either
    # signed edge alone. A hinge subtracts non-oriented cell energy.
    tangent=edge[:,::2]+edge[:,1::2]
    tangent=np.maximum(tangent-.82*tangent.mean(1,keepdims=True),0.)
    tangent=compress(tangent)
    edge=norm(edge.reshape(len(images),-1),(1,))
    tangent=norm(tangent.reshape(len(images),-1),(1,))
    return pix/pix.shape[1],edge/edge.shape[1],tangent/tangent.shape[1]


def compress(t):
    # Piecewise linear contrast compression: a sqrt-like response with
    # bounded slope at zero. All its knots are chosen in filter units.
    return (3.2*t - 1.35*np.maximum(t-.04,0.) - .85*np.maximum(t-.13,0.)
            - .45*np.maximum(t-.31,0.))


layers=[]; weights={}
def layer(name,op,inp,**kw):
    layers.append(dict(name=name,op=op,inputs=inp if isinstance(inp,list) else [inp],**kw)); return name

def linear(name,op,inp,W,b,**kw):
    weights[name+'.W']=np.asarray(W,dtype='f4'); weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype('f4')
    return layer(name,op,inp,out=len(W),**kw)

# Keep the whole image geometry, but use stride 2 for luminance and pool
# adjacent rectified edge responses in 4x4 cells for local position tolerance.
linear('soft_ink','conv','image',normal_blur[None,None],0,kernel=[5,5],stride=2,padding=2)
layer('ink_contrast','norm','soft_ink')
linear('oriented_edges','conv','image',K[:,None],0,kernel=[5,5],stride=1,padding=2)
layer('edge_polarity','relu','oriented_edges')
layer('edge_cells','avgpool','edge_polarity',size=[4,4],stride=[4,4])
layer('edge_vector','reshape','edge_cells',shape=[1,8,49])
layer('edge_contrast','norm','edge_vector')


def tangent_branch():
    orient=np.eye(4).repeat(2,axis=1)-.82/4
    W=np.tile(orient,(4,1)); b=-np.repeat([0,.04,.13,.31],4)
    linear('tangent_strength','conv','edge_cells',W[:,:,None,None],b,kernel=[1,1],stride=1,padding=0)
    layer('tangent_selected','relu','tangent_strength')
    W=np.concatenate([c*np.eye(4) for c in [3.2,-1.35,-.85,-.45]],axis=1)
    linear('tangent_comp','conv','tangent_selected',W[:,:,None,None],0,kernel=[1,1],stride=1,padding=0)
    layer('tangent_vector','reshape','tangent_comp',shape=[1,4,49]); layer('tangent_contrast','norm','tangent_vector')


def pack_features():
    # Concatenate channels by writing into disjoint slots and summing. This
    # lets both matchers use the same mixed feature metric. Pixel slots have
    # the same total mass across the four phases of the 2x2 space-to-depth.
    features=[('ink_contrast',(1,14,14),.28),('edge_contrast',(8,7,7),.49),('tangent_contrast',(4,7,7),.23)]
    y,x=np.indices((14,14)); ids=(y%2*2+x%2)*49+y//2*7+x//2
    positions=[ids.ravel(), 196+np.arange(392), 588+np.arange(196)]
    parts=[]; coeff=[]
    for (src,sh,c),idx in zip(features,positions):
        W=np.zeros((784,len(idx)),dtype='f4'); W[idx,np.arange(len(idx))]=np.sqrt(c/len(idx))
        parts.append(linear('pack_'+src,'dense',src,W,0)); coeff.append(np.sqrt(c*len(idx)))
    dest=layer('drawing_descriptor','add',parts)
    return dest,positions,coeff


def class_vote(name,inp,n):
    groups=4
    r=layer(name+'_order','reshape',inp,shape=[10,groups,n//groups])
    m=layer(name+'_best','maxpool',r,size=[1,n//groups],stride=[1,n//groups])
    m=layer(name+'_mean','avgpool',m,size=[groups,1],stride=[groups,1])
    return layer(name+'_vote','reshape',m,shape=[10])


def build_filters():
    tangent_branch(); dest,positions,coeff=pack_features()
    n=4096; packed=np.empty((10*n,784),'f4')
    for k in range(10):
        for i in range(0,n,32):
            ims=np.stack([paint(k) for _ in range(32)])
            v=packed[k*n+i:k*n+i+32]
            for value,idx,c in zip(signatures(ims),positions,coeff): v[:,idx]=c*value
    # Overlapping upper/lower parts, each maxed independently in the same
    # handwriting family. A wide transition zone prevents the model from
    # ignoring the middle bar or swapping disconnected strokes across it.
    # Compare with dot(input,model)-0.5*dot(model,model), i.e. nearest
    # squared distance up to an input-only constant. This correction matters
    # for a part whose share of the complete drawing's energy is not fixed.
    region=np.tile(np.repeat([1.,1.,.90,.50,.10,0.,0.],7),16)
    terms=[]; part_scores=[]
    for name,window in [('upper',region),('lower',1.-region)]:
        idx=np.flatnonzero(window); selector=np.zeros((len(idx),784),'f4'); selector[np.arange(len(idx)),idx]=1.
        src=linear('select_'+name,'dense',dest,selector,0.)
        model=np.ascontiguousarray(packed[:,idx]); weighted=model*window[idx].astype('f4')
        b=-.5*np.einsum('ij,ij->i',model,weighted)
        score=linear('model_'+name,'dense',src,weighted,b)
        terms.append(class_vote(name,score,n)); part_scores.append(score)
    whole=layer('same_drawing','add',part_scores)
    global_vote=class_vote('whole',whole,n)
    layer('part_match','add',terms)
    linear('part_vote','dense','part_match',.48*np.eye(10),.24)
    linear('whole_vote_scaled','dense',global_vote,.52*np.eye(10),.26)
    return layer('shape_vote','add',['part_vote','whole_vote_scaled'])


# The second source of evidence is topological, not a picture match. Threshold
# the input and propagate accessibility from the panel's boundary through
# white space. Remaining background components are enclosed counters (holes).
# The modest blur makes a varying-width pen into a more continuous barrier.

def topology():
    name=linear('clear_paper','conv','image',-35*normal_blur[None,None],35*.32,kernel=[5,5],stride=1,padding=2)
    layer('paper_positive','relu',name)
    # 1 - ReLU(1 - ReLU(x)) clips to [0,1].
    linear('paper_complement','conv','paper_positive',[[[[-1]]]],1,kernel=[1,1],stride=1,padding=0)
    layer('ink_barrier','relu','paper_complement')
    # Use stride-1 padded 3x3 conv to establish the boundary seed; the
    # kernel on a constant-one image counts padded-outside neighbors.
    linear('unit','conv','image',[[[[0]]]],1,kernel=[1,1],stride=1,padding=0)
    linear('boundary_test','conv','unit',-np.ones((1,1,3,3)),6.99,kernel=[3,3],stride=1,padding=1)
    prev=layer('boundary','relu','boundary_test')
    for i in range(20):
        # Neighborhood dilation, AND with complement of barrier.
        # Using 8 neighbors allows corner connectivity, but blurred ink
        # blocks diagonal leaks. Pad with zeros by an identity convolution.
        pad=linear('pad_%d'%i,'conv',prev,[[[[1]]]],0,kernel=[1,1],stride=1,padding=1)
        grow=layer('grow_%d'%i,'maxpool',pad,size=[3,3],stride=[1,1])
        # Bounding the boundary marker by 1 is required only at the start.
        if i==0:
            linear('inv_seed','conv',grow,[[[[-1]]]],1,kernel=[1,1],stride=1,padding=0)
            layer('no_seed','relu','inv_seed')
            grow=linear('seed','conv','no_seed',[[[[-1]]]],1,kernel=[1,1],stride=1,padding=0)
        if i==0:
            linear('minus_barrier','conv','ink_barrier',[[[[-1]]]],0,kernel=[1,1],stride=1,padding=0)
        add=layer('admit_%d'%i,'add',[grow,'minus_barrier'])
        prev=layer('reached_%d'%i,'relu',add)
    free=linear('free','conv','ink_barrier',[[[[-1]]]],1,kernel=[1,1],stride=1,padding=0)
    lost=linear('not_reached','conv',prev,[[[[-1]]]],0,kernel=[1,1],stride=1,padding=0)
    layer('enclosure_difference','add',[free,lost]); layer('counters','relu','enclosure_difference')
    # For a nominal centered glyph the counter of 6 lies below the center,
    # and the counter of 9 above it. Two distinct counters are evidence for
    # 8. A large counter through the middle is characteristic of 0, but the
    # signal is intentionally weak (a triangular 4 can be closed as well).
    # An Euler integral below evaluates separation, rather than just seeing
    # white at two heights inside the same big hole.
    yy,xx=np.mgrid[0:28,0:28]; top=1/(1+np.exp((yy-12.)/1.35)); bot=1/(1+np.exp((15.5-yy)/1.35))
    mid=np.exp(-.5*((yy-13.8)/2.2)**2)
    masks=np.stack([top,bot,mid,np.ones_like(top)]).reshape(4,-1)
    linear('counter_area','dense','counters',masks,0)
    # Stabilize by subtracting soft threshold hinges; no single giant hole
    # should dominate the shape model. Units here are square pixels.
    V=[]; b=[]; coef=[]
    def ramp(direction,start,end,votes):
        v=np.array(votes)/(end-start)
        V.extend([direction,direction]); b.extend([-start,-end]); coef.extend([v,-v])
    ramp([0,0,0,1],1,11,[.015,-.004,-.012,-.015,-.003,-.010,.009,-.012,.012,.010])
    ramp([1,-.85,0,0],2,15,[0,0,0,0,.003,0,-.030,0,0,.028])
    ramp([-.85,1,0,0],2,15,[0,0,0,0,-.014,0,.032,0,0,-.032])
    ramp([0,0,1,0],8,32,[.027,0,0,0,0,0,-.006,0,-.008,-.007])
    linear('area_hinges','dense','counter_area',V,b); layer('area_active','relu','area_hinges')
    linear('area_vote','dense','area_active',np.array(coef).T,0)
    # A class with two counters is usually an eight. Soften the component
    # count by integrating over superlevel sets of a slightly blurred mask.
    layer('counter_scale','avgpool','counters',size=[2,2],stride=[1,1])
    # Cubical Euler integral = sum(face) - sum(incident edge maxima)
    #                                      + sum(incident vertex maxima).
    src='counter_scale'
    hs=layer('paired_x','maxpool',src,size=[1,2],stride=[1,1]); vs=layer('paired_y','maxpool',src,size=[2,1],stride=[1,1]); both=layer('square_vertex','maxpool',src,size=[2,2],stride=[1,1])
    terms=[]
    for s,sz,sign in [(src,27*27,1),(hs,27*26,-1),(vs,26*27,-1),(both,26*26,1)]:
        terms.append(linear('sum_'+s,'dense',s,np.full((1,sz),sign),0))
    ct=layer('counter_number','add',terms)
    # max response of ~1 for one hole; going from one to two provides an
    # extra discriminant. Lack of a hole is not strong contrary evidence
    # because a retraced or thick loop can close or leave a gap.
    linear('double_counter_gate','dense',ct,[[1],[1]],[ -1.10,-1.85]); layer('double_counter_pos','relu','double_counter_gate')
    v=np.array([-.032,0,-.006,-.007,-.012,-.005,-.029,0,.081,-.029])/.75
    return linear('number_vote','dense','double_counter_pos',np.stack([v,-v],axis=1),0)


def main():
    out=Path(sys.argv[1]); out.mkdir(exist_ok=True,parents=True)
    shape=build_filters(); holes=topology()
    layer('scores','add',[shape,'area_vote',holes])
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**weights)

if __name__=='__main__': main()
