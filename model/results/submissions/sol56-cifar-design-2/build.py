#!/usr/bin/env python3
"""Build a data-free, analytic CIFAR-10 network.

All filters, silhouettes, and colour/context rules in this file are generated from
geometry and generic object knowledge.  The program deliberately reads no files.
"""
import json
import math
import os
import sys
import numpy as np


def layer(name, op, inputs, **kw):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(kw)
    return d


def colour_filters():
    # Each entry is a half-space in the RGB cube, followed by a ReLU.
    specs = []
    def put(name, w, b): specs.append((name, np.asarray(w, np.float32), float(b)))
    lum = np.array([0.299, 0.587, 0.114])
    for t in (.22, .38, .54, .70): put("light%02d" % int(100*t), lum, -t)
    for t in (.28, .44, .60, .76): put("dark%02d" % int(100*t), -lum, t)
    put("red",    [ .78, -.48, -.30], -.025)
    put("red2",   [ .78, -.48, -.30], -.115)
    put("green",  [-.46,  .76, -.30], -.025)
    put("green2", [-.46,  .76, -.30], -.115)
    put("blue",   [-.38, -.24,  .62], -.020)
    put("blue2",  [-.38, -.24,  .62], -.100)
    put("yellow", [ .48,  .48, -.96], -.055)
    put("yellow2",[ .48,  .48, -.96], -.180)
    put("cyan",   [-.72, .36, .36], -.045)
    put("cyan2",  [-.72, .36, .36], -.150)
    put("warm",   [ .55, .05, -.60], -.020)
    put("warm2",  [ .55, .05, -.60], -.130)
    put("cool",   [-.55,-.05,  .60], -.020)
    put("cool2",  [-.55,-.05,  .60], -.130)
    put("orange", [ .62, .16, -.48], -.130)
    put("brown",  [ .52, .12, -.34], -.075)
    put("leaf",   [-.38, .62, -.24], -.070)
    put("sky",    [-.31,-.10,  .41], -.035)
    put("r", [1,0,0], -.18); put("g", [0,1,0], -.18)
    put("white", lum, -.78)
    put("black", -lum, .25)
    put("midlight", lum, -.48)
    put("middark", -lum, .52)
    assert len(specs) == 32
    W = np.zeros((len(specs), 3, 1, 1), np.float32)
    b = np.zeros(len(specs), np.float32)
    for i, (_, w, q) in enumerate(specs): W[i,:,0,0], b[i] = w, q
    return specs, W, b


def edge_filters():
    # Signed first Gaussian derivatives and signed elongated ridges. Instance
    # normalization later makes these insensitive to illumination and contrast.
    q = np.arange(-3, 4, dtype=np.float64)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    fs, meta = [], []
    for scale in (.82, 1.42):
        for oi in range(8):
            a = math.pi * oi / 8.0
            u = xx*math.cos(a) + yy*math.sin(a)
            v = -xx*math.sin(a) + yy*math.cos(a)
            z = -u * np.exp(-(u*u+v*v)/(2*scale*scale))
            z -= z.mean(); z /= np.sqrt((z*z).sum()) + 1e-12
            for sign in (1, -1):
                fs.append(sign*z); meta.append(("edge", scale, oi, sign))
    for scale in (.68, 1.03):
        for oi in range(8):
            a = math.pi * oi / 8.0
            u = xx*math.cos(a) + yy*math.sin(a)
            v = -xx*math.sin(a) + yy*math.cos(a)
            z = (1.0-u*u/(scale*scale))*np.exp(-u*u/(2*scale*scale)-v*v/(2*(2.25*scale)**2))
            z -= z.mean(); z /= np.sqrt((z*z).sum()) + 1e-12
            for sign in (1, -1):
                fs.append(sign*z); meta.append(("ridge", scale, oi, sign))
    W = np.zeros((64,3,7,7), np.float32)
    rgb_lum = np.array([.299,.587,.114])
    for o,z in enumerate(fs):
        W[o] = rgb_lum[:,None,None] * z[None]
    return meta, W, np.zeros(64,np.float32)


def seg(x1,y1,x2,y2,s=1.0): return (x1,y1,x2,y2,s)

def circle(cx,cy,r,s=1.0,n=12):
    a = np.linspace(0,2*math.pi,n+1)
    return [seg(cx+r*math.cos(a[i]),cy+r*math.sin(a[i]),
                cx+r*math.cos(a[i+1]),cy+r*math.sin(a[i+1]),s) for i in range(n)]

def box(x1,y1,x2,y2,s=1.0):
    return [seg(x1,y1,x2,y1,s),seg(x2,y1,x2,y2,s),seg(x2,y2,x1,y2,s),seg(x1,y2,x1,y1,s)]

def ellipse(cx,cy,rx,ry,s=1.0,n=16):
    a=np.linspace(0,2*math.pi,n+1)
    return [seg(cx+rx*math.cos(a[i]),cy+ry*math.sin(a[i]),
                cx+rx*math.cos(a[i+1]),cy+ry*math.sin(a[i+1]),s) for i in range(n)]


def silhouette(k, pose):
    S=[]
    if k==0: # aeroplane: side views and cruciform views
        if pose%3==0:
            S += [seg(-.82,0,.82,0,1.3),seg(-.15,0,-.55,-.55),seg(.05,0,.48,.55),
                  seg(.52,0,.73,-.24),seg(.52,0,.73,.24)]
        elif pose%3==1:
            S += [seg(0,-.78,0,.72,1.3),seg(0,-.05,-.78,.28),seg(0,-.05,.78,.28),
                  seg(0,.49,-.30,.68),seg(0,.49,.30,.68)]
        else:
            S += [seg(-.82,.14,.82,-.12,1.3),seg(-.18,.03,-.34,-.56),seg(.05,.00,.25,.57),
                  seg(.55,-.08,.72,-.35),seg(.55,-.08,.78,.14)]
    elif k==1: # automobile
        S += box(-.78,-.05,.78,.34,1.0)
        S += [seg(-.48,-.05,-.25,-.38),seg(-.25,-.38,.35,-.38),seg(.35,-.38,.60,-.05)]
        S += circle(-.48,.34,.17,1.3,10)+circle(.48,.34,.17,1.3,10)
        if pose%3==1: S += [seg(-.18,-.34,-.10,-.06),seg(.18,-.34,.24,-.06)]
    elif k==2: # bird: flying, perched, and side-on
        if pose%3==0:
            S += [seg(0,.05,-.75,-.48),seg(-.75,-.48,-.20,.12),seg(0,.05,.75,-.48),seg(.75,-.48,.20,.12)]
            S += ellipse(0,.10,.24,.16,1,10)+[seg(.18,.08,.44,.02)]
        elif pose%3==1:
            S += ellipse(-.05,.08,.28,.42,1.2,14)+circle(.10,-.38,.17,1,10)
            S += [seg(.24,-.39,.48,-.32),seg(-.12,.48,-.18,.74),seg(.02,.48,.10,.74)]
        else:
            S += [seg(-.05,.05,-.76,-.08),seg(-.76,-.08,-.25,.18),seg(.02,.05,.70,.20),seg(.70,.20,.24,.24)]
            S += ellipse(0,.13,.25,.13,1,10)
    elif k==3: # cat, frontal head or side body
        if pose%2==0:
            S += circle(0,0,.48,1.1,16)
            S += [seg(-.42,-.28,-.35,-.72),seg(-.35,-.72,-.05,-.43),
                  seg(.42,-.28,.35,-.72),seg(.35,-.72,.05,-.43),
                  seg(-.25,-.08,-.08,-.08),seg(.25,-.08,.08,-.08),seg(0,.02,0,.23)]
        else:
            S += ellipse(-.12,.02,.58,.32,1.1,16)+circle(.47,-.18,.25,1,12)
            S += [seg(-.55,.22,-.53,.72),seg(-.22,.29,-.18,.72),seg(.18,.25,.24,.72),
                  seg(-.65,-.05,-.88,-.48),seg(.35,-.38,.41,-.65),seg(.47,-.40,.65,-.62)]
    elif k==4: # deer, slender quadruped, raised neck and antlers
        S += ellipse(-.10,.02,.58,.25,1.1,16)
        S += [seg(-.48,.20,-.52,.76),seg(-.25,.23,-.20,.76),seg(.22,.22,.27,.76),seg(.42,.15,.48,.72),
              seg(.40,-.05,.53,-.55,1.2),seg(.53,-.55,.70,-.63),seg(.53,-.56,.45,-.82),
              seg(.48,-.77,.32,-.91),seg(.48,-.77,.55,-.97),seg(.58,-.76,.72,-.91)]
    elif k==5: # dog, floppy-eared face or side body
        if pose%2==0:
            S += circle(0,-.02,.48,1.1,16)+ellipse(0,.20,.22,.28,1,12)
            S += [seg(-.37,-.32,-.65,-.58),seg(-.65,-.58,-.55,.05),
                  seg(.37,-.32,.65,-.58),seg(.65,-.58,.55,.05),
                  seg(-.22,-.10,-.08,-.10),seg(.22,-.10,.08,-.10)]
        else:
            S += ellipse(-.12,.04,.60,.31,1.1,16)+circle(.48,-.14,.26,1,12)
            S += [seg(-.48,.25,-.50,.75),seg(-.18,.29,-.15,.75),seg(.22,.26,.26,.75),
                  seg(-.68,-.02,-.91,-.30),seg(.42,-.35,.33,-.61),seg(.60,-.30,.71,-.55)]
    elif k==6: # frog: squat oval, eyes and splayed legs
        S += ellipse(0,.06,.50,.34,1.1,16)+circle(-.29,-.31,.17,1.1,10)+circle(.29,-.31,.17,1.1,10)
        S += [seg(-.36,.18,-.78,.55),seg(-.78,.55,-.93,.34),seg(-.28,.28,-.58,.77),
              seg(.36,.18,.78,.55),seg(.78,.55,.93,.34),seg(.28,.28,.58,.77),
              seg(-.14,.12,.14,.12)]
    elif k==7: # horse: large body, four legs, long neck
        S += ellipse(-.12,.00,.62,.31,1.2,16)
        S += [seg(-.52,.20,-.56,.78),seg(-.28,.26,-.24,.78),seg(.18,.26,.20,.78),seg(.43,.16,.48,.76),
              seg(.34,-.15,.52,-.63,1.2),seg(.52,-.63,.73,-.55),seg(.64,-.58,.72,-.38),
              seg(-.70,-.08,-.93,-.35),seg(-.76,-.05,-.96,.15)]
    elif k==8: # ship: hull, deck/cabin, mast
        S += [seg(-.86,.12,.86,.12,1.3),seg(-.86,.12,-.58,.48),seg(-.58,.48,.58,.48),seg(.58,.48,.86,.12)]
        S += box(-.36,-.19,.38,.12,1.0)+[seg(.0,-.19,.0,-.76,1.2),seg(0,-.62,.35,-.38),seg(-.68,.54,.78,.54)]
        if pose%2: S += [seg(-.34,-.19,-.18,-.40),seg(-.18,-.40,.20,-.40),seg(.20,-.40,.38,-.19)]
    else: # truck: rectangular cargo/cab and circular wheels
        S += box(-.82,-.34,.28,.35,1.1)+box(.28,-.12,.80,.35,1.1)
        S += [seg(.38,-.12,.52,-.39),seg(.52,-.39,.74,-.39),seg(.74,-.39,.80,-.12)]
        S += circle(-.48,.37,.17,1.3,10)+circle(.51,.37,.17,1.3,10)
        if pose%2: S += [seg(-.78,-.18,.20,-.18),seg(.30,.04,.77,.04)]
    return S


def transform_segments(S, scale, cx, cy, angle, flip):
    ca,sa=math.cos(angle),math.sin(angle); out=[]
    for x1,y1,x2,y2,s in S:
        x1*=flip; x2*=flip
        a1=scale*(ca*x1-sa*y1)+cx; b1=scale*(sa*x1+ca*y1)+cy
        a2=scale*(ca*x2-sa*y2)+cx; b2=scale*(sa*x2+ca*y2)+cy
        out.append((a1,b1,a2,b2,s))
    return out


def render_edge_template(S, meta):
    # Coordinates at the centres of the 8x8 pooled cells.
    g=np.linspace(-.875,.875,8); Y,X=np.meshgrid(g,g,indexing='ij')
    A=np.zeros((64,8,8),np.float64)
    for x1,y1,x2,y2,strength in S:
        dx,dy=x2-x1,y2-y1; den=dx*dx+dy*dy+1e-9
        t=np.clip(((X-x1)*dx+(Y-y1)*dy)/den,0,1)
        d=np.sqrt((X-(x1+t*dx))**2+(Y-(y1+t*dy))**2)
        z=np.exp(-.5*(d/.105)**2)*strength
        # Derivative/ridge orientation is the normal to this contour.
        normal=(math.atan2(dy,dx)+math.pi/2)%math.pi
        oi=int(np.floor(normal/(math.pi/8)+.5))%8
        for i,(kind,scale,o,sign) in enumerate(meta):
            dd=min((o-oi)%8,(oi-o)%8)
            if dd==0: A[i]+=z*(1.0 if kind=='ridge' else .72)
            elif dd==1: A[i]+=z*(.28 if kind=='ridge' else .18)
    # Remove each channel's DC response.  This asks for contours at the
    # specified positions rather than merely rewarding a textured image.
    A-=A.mean(axis=(1,2),keepdims=True)
    n=np.sqrt((A*A).sum())
    if n>0: A/=n
    return A.astype(np.float32)


def object_mask(k, S):
    g=np.linspace(-.875,.875,8); Y,X=np.meshgrid(g,g,indexing='ij')
    # Generic filled regions, independent of any example photograph.
    if k==0: r=np.exp(-((X/.72)**2+(Y/.22)**2)*2.0)
    elif k in (1,8,9): r=np.exp(-((X/.72)**4+(Y/.34)**4)*1.4)
    elif k==2: r=np.exp(-((X/.46)**2+(Y/.38)**2)*1.7)
    elif k in (3,5): r=np.exp(-((X/.52)**2+(Y/.53)**2)*1.4)
    elif k==6: r=np.exp(-((X/.58)**2+(Y/.42)**2)*1.7)
    else: r=np.exp(-(((X+.08)/.64)**4+(Y/.35)**4)*1.5)
    return np.clip(r,0,1)


def colour_preferences(k, variant):
    # Coefficients name broad context and object colours.  Alternate variants
    # acknowledge that object paint/fur and backgrounds vary widely.
    outside={}; inside={}
    if k==0:
        outside={'sky':1.25,'blue':.7,'light54':.45,'green':-.35,'brown':-.3}; inside={'white':.6,'dark44':.25,'red':.12}
    elif k==1:
        outside={'sky':.18,'dark44':.25,'light54':.10}; inside=({'red':.8,'warm':.35} if variant%3==0 else {'blue':.55,'cool':.2} if variant%3==1 else {'light54':.35,'dark44':.28})
    elif k==2:
        outside=({'sky':.75,'blue':.32} if variant%2==0 else {'green':.52,'leaf':.45}); inside={'brown':.42,'dark44':.28,'warm':.18}
    elif k==3:
        outside={'warm':.10,'light54':.12,'sky':-.25,'green':-.16}; inside=({'orange':.58,'warm':.35} if variant%3==0 else {'light54':.28,'dark44':.22} if variant%3==1 else {'dark44':.45,'brown':.2})
    elif k==4:
        outside={'green':.55,'leaf':.45,'sky':.10}; inside={'brown':.75,'warm':.35,'orange':.22}
    elif k==5:
        outside={'green':.08,'light54':.10,'sky':-.18}; inside=({'brown':.48,'warm':.32} if variant%2==0 else {'light54':.35,'dark44':.28})
    elif k==6:
        outside={'green':.42,'leaf':.28,'dark44':.12}; inside={'green':.82,'leaf':.50,'yellow':.16,'dark44':.18}
    elif k==7:
        outside={'green':.48,'leaf':.26,'sky':.15}; inside={'brown':.72,'warm':.38,'dark44':.18}
    elif k==8:
        outside={'sky':1.0,'blue':.62,'cyan':.22,'green':-.24}; inside={'light54':.42,'white':.32,'dark44':.26,'red':.12}
    else:
        outside={'light54':.25,'sky':.12,'dark44':.10}; inside=({'red':.65,'warm':.30} if variant%2==0 else {'white':.42,'dark44':.30})
    return outside,inside


def render_colour_template(k, variant, names, S):
    C=np.zeros((32,8,8),np.float64); obj=object_mask(k,S)
    g=np.linspace(-.875,.875,8); Y,X=np.meshgrid(g,g,indexing='ij')
    outside,inside=colour_preferences(k,variant)
    # Context gets a mild top/bottom modulation: sky colours matter most above;
    # vegetation/road colours matter most below.
    for name,val in outside.items():
        i=names.index(name)
        mod=np.ones((8,8))
        if name in ('sky','blue','cyan'): mod*=1.15-.35*Y
        if name in ('green','leaf','brown','dark44'): mod*=1.0+.18*Y
        C[i]+=val*(1-obj)*mod
    for name,val in inside.items(): C[names.index(name)]+=val*obj
    # Templates describe centre-versus-surround layout; global scene evidence is
    # handled separately so classes with more named colours get no free offset.
    C-=C.mean(axis=(1,2),keepdims=True)
    n=np.sqrt((C*C).sum())
    if n>0: C/=n
    return C.astype(np.float32)


def context_readout(names):
    W=np.zeros((10,32,8,8),np.float32); b=np.zeros(10,np.float32)
    Y=np.linspace(-.875,.875,8)[:,None]*np.ones((1,8))
    def add(k,name,v,where='all'):
        m=np.ones((8,8),np.float32)
        if where=='top': m=(1-Y)/2
        elif where=='bottom': m=(1+Y)/2
        elif where=='center':
            x=np.linspace(-.875,.875,8); yy,xx=np.meshgrid(x,x,indexing='ij'); m=np.exp(-(xx*xx+yy*yy)/.38)
        W[k,names.index(name)] += v*m/64.0
    # Broad scene priors. They intentionally use only common-sense associations.
    for n,v,r in [('sky',1.6,'top'),('blue',.65,'top'),('light54',.35,'top'),('green',-.35,'bottom')]: add(0,n,v,r)
    for n,v,r in [('dark44',.25,'bottom'),('sky',.15,'top'),('red',.12,'center')]: add(1,n,v,r)
    for n,v,r in [('sky',.45,'top'),('green',.28,'all'),('leaf',.25,'all'),('dark44',.12,'center')]: add(2,n,v,r)
    for n,v,r in [('warm',.22,'center'),('orange',.20,'center'),('sky',-.32,'all'),('green',-.18,'all')]: add(3,n,v,r)
    for n,v,r in [('green',.55,'all'),('leaf',.48,'all'),('brown',.50,'center'),('warm',.18,'center')]: add(4,n,v,r)
    for n,v,r in [('brown',.22,'center'),('warm',.20,'center'),('sky',-.25,'all')]: add(5,n,v,r)
    for n,v,r in [('green',.90,'all'),('leaf',.45,'all'),('yellow',.12,'center'),('sky',-.20,'all')]: add(6,n,v,r)
    for n,v,r in [('green',.45,'bottom'),('brown',.60,'center'),('warm',.25,'center'),('sky',.10,'top')]: add(7,n,v,r)
    for n,v,r in [('sky',1.20,'all'),('blue',.80,'bottom'),('cyan',.22,'bottom'),('light54',.25,'top'),('green',-.35,'all')]: add(8,n,v,r)
    for n,v,r in [('light54',.22,'all'),('dark44',.18,'bottom'),('red',.16,'center'),('sky',.10,'top')]: add(9,n,v,r)
    # Equal L1 normalization prevents classes described by more context words
    # from receiving a larger null-image score.
    Q=W.reshape(10,-1)
    Q*=2.0/(np.abs(Q).sum(axis=1,keepdims=True)+1e-9)
    return Q,b


def edge_readout(meta):
    W=np.zeros((10,64,8,8),np.float32); b=np.zeros(10,np.float32)
    Y=np.linspace(-.875,.875,8)[:,None]*np.ones((1,8)); X=Y.T
    center=np.exp(-(X*X+Y*Y)/.45)
    bottom=(1+Y)/2
    for i,(kind,scale,o,sign) in enumerate(meta):
        # orientation index 2/6 is normal to common horizontal contours;
        # 0/4 is normal to vertical legs and masts.
        horiz=min((o-2)%8,(2-o)%8)==0 or min((o-6)%8,(6-o)%8)==0
        vert=(o==0 or o==4)
        diagonal=not horiz and not vert
        W[0,i] += ((.18 if diagonal else .07)*center)/64
        W[1,i] += ((.15 if horiz else .06)*bottom)/64
        W[2,i] += ((.11 if diagonal else .04)*center)/64
        W[3,i] += .055*center/64
        W[4,i] += ((.13 if vert else .06)*bottom)/64
        W[5,i] += .06*center/64
        W[6,i] += .075*center/64
        W[7,i] += ((.15 if vert else .07)*bottom)/64
        W[8,i] += ((.17 if horiz else .05)*(1-.25*Y))/64
        W[9,i] += ((.18 if horiz or vert else .04)*bottom)/64
    # Orientation summaries are deliberately weak beside explicit templates.
    return W.reshape(10,-1)*.35,b


def build(outdir):
    os.makedirs(outdir,exist_ok=True)
    specs,cW,cb=colour_filters(); names=[s[0] for s in specs]
    meta,eW,eb=edge_filters()
    T=100
    # Two dense banks contribute colour and contour evidence to each latent
    # class/template hypothesis before a ReLU and class-wise max operation.
    CW=np.zeros((10*T,32*8*8),np.float32)
    EW=np.zeros((10*T,64*8*8),np.float32)
    Cb=np.zeros(10*T,np.float32); Eb=np.full(10*T,-.015,np.float32)
    scales=[.72,.82,.92,1.02,1.12]
    shifts=[-.13,0,.13]
    angles=[-.16,-.08,0,.08,.16]
    for k in range(10):
        for t in range(T):
            pose=t%5
            sc=scales[(t//5)%5]
            cx=shifts[(t//25)%3]
            cy=(-.05,.06,.16)[(t//75)%3]
            ang=angles[(t//15)%5] if k in (0,2) else angles[(t//20)%5]*.35
            flip=-1 if (t//10)%2 else 1
            S=transform_segments(silhouette(k,pose),sc,cx,cy,ang,flip)
            row=k*T+t
            EW[row]=render_edge_template(S,meta).reshape(-1)*1.15
            CW[row]=render_colour_template(k,t,names,S).reshape(-1)*.52
    ctxW,ctxb=context_readout(names)
    edW,edb=edge_readout(meta)
    layers=[
      layer('colour','conv',['image'],out=32,kernel=[1,1],stride=1,padding=0),
      layer('colour_relu','relu',['colour']),
      layer('colour_pool','avgpool',['colour_relu'],size=[4,4],stride=[4,4]),
      layer('edge','conv',['image'],out=64,kernel=[7,7],stride=1,padding=3),
      layer('edge_norm','norm',['edge']),
      layer('edge_relu','relu',['edge_norm']),
      layer('edge_pool','avgpool',['edge_relu'],size=[4,4],stride=[4,4]),
      layer('colour_templates','dense',['colour_pool'],out=10*T),
      layer('edge_templates','dense',['edge_pool'],out=10*T),
      layer('templates','add',['colour_templates','edge_templates']),
      layer('template_relu','relu',['templates']),
      layer('template_grid','reshape',['template_relu'],shape=[10,10,10]),
      layer('template_max','maxpool',['template_grid'],size=[10,10],stride=[10,10]),
      layer('shape_scores','reshape',['template_max'],shape=[10]),
      layer('context_scores','dense',['colour_pool'],out=10),
      layer('orientation_scores','dense',['edge_pool'],out=10),
      layer('shape_scaled','dense',['shape_scores'],out=10),
      layer('scores0','add',['shape_scaled','context_scores','orientation_scores']),
      layer('scores','dense',['scores0'],out=10),
    ]
    net={'layers':layers}
    eye=np.eye(10,dtype=np.float32)
    weights={
      'colour.W':cW,'colour.b':cb,'edge.W':eW,'edge.b':eb,
      'colour_templates.W':CW,'colour_templates.b':Cb,
      'edge_templates.W':EW,'edge_templates.b':Eb,
      'context_scores.W':ctxW.astype(np.float32),'context_scores.b':ctxb,
      'orientation_scores.W':edW.astype(np.float32),'orientation_scores.b':edb,
      'shape_scaled.W':eye*1.0,'shape_scaled.b':np.zeros(10,np.float32),
      # Hand calibration corrects generic template-complexity biases (ships and
      # trucks have many long edges; compact animals have fewer).
      'scores.W':eye,'scores.b':np.array([-.06,.05,-.08,.08,.10,.08,.16,.08,-.09,-.08],np.float32),
    }
    with open(os.path.join(outdir,'network.json'),'w') as f: json.dump(net,f,separators=(',',':'))
    np.savez(os.path.join(outdir,'weights.npz'),**weights)


if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '.')
