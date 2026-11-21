"""Build a data-free, hand-designed CIFAR-style classifier.
All filters are colour formulas, Gaussian derivatives, or geometric drawings.
No data files are read and no fitting is performed.
"""
import json, sys
from pathlib import Path
import numpy as np

outdir = Path(sys.argv[1])
outdir.mkdir(parents=True, exist_ok=True)
layers, weights = [], {}

def layer(name, op, inputs, **settings):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(settings); layers.append(d)

def conv(name, inp, W, b=None, stride=1, padding=0):
    W = np.asarray(W, dtype=np.float32)
    layer(name, "conv", [inp], out=len(W), kernel=[W.shape[-1]]*2,
          stride=stride, padding=padding)
    weights[name+".W"] = W
    weights[name+".b"] = np.zeros(len(W), np.float32) if b is None else np.asarray(b, np.float32)

def dense(name, inp, W, b=None):
    W = np.asarray(W, dtype=np.float32)
    layer(name, "dense", [inp], out=len(W))
    weights[name+".W"] = W
    weights[name+".b"] = np.zeros(len(W), np.float32) if b is None else np.asarray(b, np.float32)

def pool(name, op, inp, size, stride=None):
    if stride is None: stride = size
    layer(name, op, [inp], size=[size]*2, stride=[stride]*2)

# A luminance image at 16 x 16, standardized separately for every photo.
conv("luma", "image", [[[[0.299]], [[0.587]], [[0.114]]]])
pool("small", "avgpool", "luma", 2)
layer("standard", "norm", ["small"])

# Colour and illumination measurements. Adjacent signed differences give
# opponent colours after the shared rectifier.
color_specs = [
    ([1,0,0],0), ([0,1,0],0), ([0,0,1],0), ([.299,.587,.114],0),
    ([-.299,-.587,-.114],.48), ([.299,.587,.114],-.58),
    ([-1,1,0],0), ([1,-1,0],0),
    ([-.5,1,-.5],-.025), ([-.5,-.5,1],-.025),
    ([1,-.5,-.5],-.035), ([.5,.5,-1],-.035),
    ([1,0,-1],-.10), ([0,1,-1],-.10),
    ([-.299,-.587,-.114],.25), ([.299,.587,.114],-.72),
]
CW = np.zeros((len(color_specs),3,1,1)); CB = np.zeros(len(color_specs))
for i,(v,b) in enumerate(color_specs): CW[i,:,0,0]=v; CB[i]=b
conv("color", "image", CW, CB)
layer("color_relu", "relu", ["color"])
pool("color_cells", "avgpool", "color_relu", 8)

# Semantic colour rules. These express broad scene and object expectations,
# not prototypes or statistics from any photographs.
# feature indices: R G B Y dark bright G-R R-G green blue red yellow
#                  R-B G-B verydark verybright
color_rules = [
    # global, top, bottom, centre: hand-written rule dictionaries
    ({"blue":.20,"bright":.15}, {"blue":.65,"bright":.18,"green":-.35}, {"green":-.20}, {"bright":.12}),
    ({"red":.20,"dark":.12}, {"green":-.20}, {"dark":.35,"blue":-.18}, {"red":.20,"bright":.08}),
    ({"brown":.15,"green":.10}, {"blue":.18}, {"green":.10}, {"dark":.10,"brown":.12}),
    ({"brown":.14,"dark":.08}, {"blue":-.18}, {"blue":-.25}, {"brown":.16,"dark":.08}),
    ({"brown":.38,"green":.22}, {"blue":-.12}, {"green":.30,"brown":.12}, {"brown":.25}),
    ({"brown":.17,"green":.08}, {"blue":-.12}, {"green":.12,"blue":-.18}, {"bright":.08,"brown":.12}),
    ({"green":.62,"brown":.22}, {"blue":-.12}, {"blue":-.20}, {"green":.65,"brown":.18}),
    ({"brown":.34,"green":.16,"red":.08}, {"blue":-.15}, {"green":.24}, {"brown":.25,"bright":.08}),
    ({"blue":.30,"bright":.08}, {"blue":.35,"green":-.30}, {"blue":1.10,"green":-.35}, {"bright":.15,"dark":.05}),
    ({"red":.20,"bright":.08,"dark":.08}, {"green":-.20}, {"dark":.25,"green":-.18}, {"red":.18,"bright":.12}),
]
fidx = {"R":0,"G":1,"B":2,"Y":3,"dark":4,"bright":5,"G-R":6,"R-G":7,
        "green":8,"blue":9,"red":10,"yellow":11,"brown":12,"verydark":14,"verybright":15}
# The brown detector is red minus blue; add a green-minus-blue component.
colorW = np.zeros((10,16,4,4), np.float32)
for c, regions in enumerate(color_rules):
    for region, rules in zip(("global","top","bottom","centre"), regions):
        for key, value in rules.items():
            fi = fidx[key]
            for yy in range(4):
                for xx in range(4):
                    use = (region=="global" or (region=="top" and yy<2) or
                           (region=="bottom" and yy>=2) or
                           (region=="centre" and 1<=yy<=2 and 1<=xx<=2))
                    if use:
                        norm = 16 if region=="global" else (8 if region in ("top","bottom") else 4)
                        colorW[c,fi,yy,xx] += value/norm*8.0
                        if key=="brown": colorW[c,13,yy,xx] += value/norm*2.0
# A common response in any one measurement is not class evidence. Centring
# also prevents the all-positive rectified measurements becoming class priors.
colorW -= colorW.mean(axis=0, keepdims=True)
# Approximate response to an unstructured mid-grey/opponent-colour signal.
# These are analytical null levels for the formulas, not measured data levels.
null_color=np.array([.5,.5,.5,.5,.15,.12,.07,.07,.04,.04,.04,.04,.05,.05,.02,.02])
null_cells=np.tile(null_color[:,None,None],(1,4,4)).ravel()
color_flat=colorW.reshape(10,-1)
dense("color_score", "color_cells", color_flat, b=-(color_flat@null_cells))

# Oriented Gaussian-derivative filters and centre/surround filters.
def gauss_deriv(k, sigma, angle, order=1):
    a = np.arange(k)-(k-1)/2; xx,yy = np.meshgrid(a,a)
    u = xx*np.cos(angle)+yy*np.sin(angle)
    v = -xx*np.sin(angle)+yy*np.cos(angle)
    g = np.exp(-(u*u+v*v)/(2*sigma*sigma))
    w = -u*g if order==1 else (u*u/sigma**2-1)*g
    w -= w.mean(); return w/(np.sqrt((w*w).sum())+1e-12)

def dog(k, sigma):
    a=np.arange(k)-(k-1)/2; xx,yy=np.meshgrid(a,a); r2=xx*xx+yy*yy
    w=np.exp(-r2/(2*sigma*sigma))-.45*np.exp(-r2/(2*(2*sigma)**2))
    w-=w.mean(); return w/(np.sqrt((w*w).sum())+1e-12)

base_filters=[]
for theta in np.arange(4)*np.pi/4: base_filters.append(gauss_deriv(5,1.15,theta))
base_filters += [dog(5,.75),dog(5,1.35)]
EW=[]; enames=[]
for i,w in enumerate(base_filters):
    for sign in (1,-1):
        EW.append(sign*w[None,None,:,:]); enames.append(str(i)+("+" if sign>0 else "-"))
conv("edges", "standard", np.concatenate(EW), padding=2)
layer("edges_relu", "relu", ["edges"])
pool("edge_cells", "avgpool", "edges_relu", 4)
# Each orientation has a positive and a negative channel. Both are given the
# same class weight, making the rules insensitive to light/dark polarity.
# orientation order: horizontal derivative, diagonal, vertical derivative,
# other diagonal, fine centre/surround, broad centre/surround.
edge_rules = [
    [.30,.45,.15,.45,.05,.00],   # aircraft: long edges and diagonals
    [.55,.12,.48,.12,.22,.10],   # cars: body lines and wheels
    [.12,.48,.18,.48,.32,.10],   # birds: inclined wings and fine detail
    [.10,.22,.15,.22,.42,.25],   # cats: facial and fur detail
    [.30,.12,.50,.12,.18,.10],   # deer: back and thin legs
    [.12,.20,.15,.20,.38,.25],   # dogs: facial and fur detail
    [.08,.18,.10,.18,.35,.52],   # frogs: mottled rounded bodies
    [.38,.12,.48,.12,.15,.10],   # horses: back and long legs
    [.62,.08,.25,.08,.04,.06],   # ships: hull/waterline and masts
    [.52,.10,.52,.10,.20,.14],   # trucks: rectangular panels
]
edgeW=np.zeros((10,12,4,4),np.float32)
for c in range(10):
    for f in range(6):
        for sign in range(2):
            for yy in range(4):
                for xx in range(4):
                    centre = 1.35 if (1<=yy<=2 and 1<=xx<=2) else .45
                    # Vehicle and ship horizontal structure is most useful low;
                    # animal vertical structure is also strongest in lower cells.
                    pos = 1.0
                    if f==0 and c in (0,1,7,8,9) and yy>=2: pos=1.3
                    if f==2 and c in (4,7) and yy>=2: pos=1.3
                    edgeW[c,2*f+sign,yy,xx]=edge_rules[c][f]*centre*pos/8
edgeW -= edgeW.mean(axis=0, keepdims=True)
edge_flat=edgeW.reshape(10,-1)*0.50
# A zero-mean unit Gaussian response has E[ReLU(x)] = 1/sqrt(2*pi).
dense("edge_score", "edge_cells", edge_flat, b=-edge_flat.sum(axis=1)/np.sqrt(2*np.pi))

# Geometric silhouettes, drawn from general object knowledge on a 12 x 12
# canvas. Every filter is zero-mean and unit-norm. Both contrast polarities
# are included. Convolution plus global max pooling allows object translation.
N=12
axis=np.linspace(-1,1,N); XX,YY=np.meshgrid(axis,axis)

def ellipse(im,cx,cy,rx,ry,angle=0,value=1):
    x=XX-cx; y=YY-cy
    u=x*np.cos(angle)+y*np.sin(angle); v=-x*np.sin(angle)+y*np.cos(angle)
    d=(u/rx)**2+(v/ry)**2
    im[:]=np.maximum(im,np.clip((1.12-d)*4,0,1)*value)

def polygon(im,pts,value=1):
    inside=np.zeros_like(XX,dtype=bool)
    for i in range(len(pts)):
        x1,y1=pts[i]; x2,y2=pts[(i+1)%len(pts)]
        cross=((y1>YY)!=(y2>YY)) & (XX < (x2-x1)*(YY-y1)/(y2-y1+1e-12)+x1)
        inside ^= cross
    im[:]=np.maximum(im,inside.astype(float)*value)

def segment(im,x1,y1,x2,y2,width=.06,value=1):
    dx=x2-x1; dy=y2-y1
    t=np.clip(((XX-x1)*dx+(YY-y1)*dy)/(dx*dx+dy*dy+1e-12),0,1)
    d=np.sqrt((XX-x1-t*dx)**2+(YY-y1-t*dy)**2)
    im[:]=np.maximum(im,np.clip((width-d)/width*2,0,1)*value)

def blank(): return np.zeros((N,N),float)

def airplane(v):
    im=blank()
    if v==0: # side view, swept wings and tail
        ellipse(im,0,.05,.86,.13); polygon(im,[(-.15,.03),(.12,-.72),(.34,-.68),(.20,.08)])
        polygon(im,[(-.15,.08),(.18,.64),(.35,.58),(.18,.05)])
        polygon(im,[(-.72,.02),(-.82,-.40),(-.61,-.34),(-.43,.04)])
    elif v==1: # frontal aircraft
        ellipse(im,0,0,.15,.82); polygon(im,[(-.95,.10),(-.12,-.12),(.12,-.12),(.95,.10),(.90,.27),(.10,.12),(-.10,.12),(-.90,.27)])
        polygon(im,[(-.40,.60),(0,.42),(.40,.60),(.36,.70),(0,.57),(-.36,.70)])
    elif v==2: # top view
        ellipse(im,0,0,.14,.88); polygon(im,[(-.88,.32),(-.12,-.28),(.12,-.28),(.88,.32),(.82,.45),(.10,.08),(-.10,.08),(-.82,.45)])
        polygon(im,[(-.38,.67),(0,.48),(.38,.67),(.32,.77),(0,.63),(-.32,.77)])
    else: # airliner side, tail fin
        ellipse(im,0,.18,.88,.16); polygon(im,[(-.70,.12),(-.62,-.48),(-.40,-.40),(-.28,.15)])
        polygon(im,[(-.15,.15),(.28,.55),(.50,.52),(.20,.12)])
    return im

def automobile(v):
    im=blank()
    if v in (0,2):
        polygon(im,[(-.88,.15),(-.60,-.10),(-.28,-.18),(-.05,-.48),(.42,-.45),(.68,-.12),(.88,.0),(.85,.43),(-.85,.43)])
        if v==2: polygon(im,[(-.40,-.15),(-.32,-.62),(.48,-.60),(.66,-.12)])
        for x in (-.55,.55): ellipse(im,x,.43,.20,.20)
    elif v==1:
        polygon(im,[(-.52,-.55),(.52,-.55),(.72,-.10),(.88,.12),(.82,.60),(-.82,.60),(-.88,.12),(-.72,-.10)])
        ellipse(im,-.70,.60,.18,.18); ellipse(im,.70,.60,.18,.18)
    else:
        polygon(im,[(-.72,-.55),(.72,-.55),(.88,.50),(.62,.66),(-.62,.66),(-.88,.50)])
        ellipse(im,-.60,.58,.18,.18); ellipse(im,.60,.58,.18,.18)
    return im

def bird(v):
    im=blank()
    if v==0:
        ellipse(im,-.05,.12,.40,.57,-.35); ellipse(im,.22,-.43,.25,.25)
        polygon(im,[(.40,-.48),(.85,-.37),(.42,-.29)])
        polygon(im,[(-.28,.48),(-.55,.92),(-.05,.61)])
        segment(im,.05,.54,.02,.92,.035); segment(im,.20,.51,.28,.90,.035)
    elif v==1:
        ellipse(im,0,.12,.16,.48); ellipse(im,0,-.30,.18,.18)
        polygon(im,[(-.08,-.02),(-.95,-.52),(-.68,.15),(-.20,.48)])
        polygon(im,[(.08,-.02),(.95,-.52),(.68,.15),(.20,.48)])
        polygon(im,[(-.12,.48),(0,.88),(.12,.48)])
    elif v==2:
        ellipse(im,0,.12,.62,.70); ellipse(im,.20,-.42,.42,.38)
        polygon(im,[(.48,-.48),(.95,-.32),(.48,-.20)])
    else:
        ellipse(im,-.10,.10,.58,.38,.15); ellipse(im,.40,-.25,.27,.29)
        polygon(im,[(.58,-.30),(.94,-.20),(.58,-.10)])
        polygon(im,[(-.55,-.02),(-.95,-.30),(-.62,.30)])
        segment(im,0,.40,-.05,.88,.04); segment(im,.20,.39,.28,.87,.04)
    return im

def cat(v):
    im=blank()
    if v==0:
        ellipse(im,-.05,.28,.43,.60); ellipse(im,0,-.40,.43,.36)
        polygon(im,[(-.38,-.57),(-.30,-.98),(-.05,-.66)])
        polygon(im,[(.38,-.57),(.30,-.98),(.05,-.66)])
        segment(im,.30,.50,.75,.60,.09); segment(im,.75,.60,.78,-.05,.07)
    elif v==1:
        ellipse(im,-.10,.05,.64,.34); ellipse(im,.48,-.25,.33,.30)
        polygon(im,[(.25,-.42),(.30,-.75),(.47,-.48)])
        polygon(im,[(.55,-.48),(.73,-.72),(.75,-.35)])
        for x in (-.48,-.15,.25,.47): segment(im,x,.25,x+.03,.85,.065)
        segment(im,-.65,-.02,-.90,-.48,.07)
    elif v==2:
        ellipse(im,0,.05,.80,.78)
        polygon(im,[(-.68,-.35),(-.58,-.98),(-.18,-.65)])
        polygon(im,[(.68,-.35),(.58,-.98),(.18,-.65)])
    else:
        ellipse(im,0,.20,.78,.55); ellipse(im,.35,-.18,.38,.34)
        polygon(im,[(.10,-.39),(.17,-.70),(.35,-.47)])
        polygon(im,[(.48,-.47),(.68,-.69),(.72,-.30)])
        segment(im,-.65,.15,-.80,-.30,.10)
    return im

def deer(v):
    im=blank()
    if v in (0,3):
        ellipse(im,-.12,.0,.61,.30)
        segment(im,.32,-.05,.52,-.60,.16); ellipse(im,.59,-.65,.28,.17,-.2)
        for x in (-.52,-.20,.20,.39): segment(im,x,.20,x+(.12 if x<0 else -.05),.91,.047)
        segment(im,-.68,-.06,-.88,-.30,.055)
        if v==0:
            for side in (-1,1):
                segment(im,.59+side*.10,-.76,.55+side*.24,-.98,.032)
                segment(im,.55+side*.20,-.91,.55+side*.34,-.96,.025)
        else: # grazing
            segment(im,.30,.0,.70,.40,.14); ellipse(im,.77,.48,.24,.16,.5)
    elif v==1:
        ellipse(im,0,.35,.43,.60); ellipse(im,0,-.35,.35,.42)
        for side in (-1,1):
            polygon(im,[(side*.20,-.60),(side*.64,-.76),(side*.34,-.40)])
            segment(im,side*.18,-.67,side*.40,-.98,.035)
            segment(im,side*.32,-.84,side*.57,-.93,.028)
    else:
        ellipse(im,0,.10,.53,.72); ellipse(im,0,-.48,.36,.38)
        polygon(im,[(-.28,-.63),(-.65,-.82),(-.40,-.40)])
        polygon(im,[(.28,-.63),(.65,-.82),(.40,-.40)])
    return im

def dog(v):
    im=blank()
    if v==0:
        ellipse(im,-.12,.05,.62,.35); ellipse(im,.46,-.24,.34,.34)
        ellipse(im,.66,-.10,.28,.18); ellipse(im,.26,-.25,.15,.35,.3)
        for x in (-.48,-.18,.20,.40): segment(im,x,.28,x+.02,.82,.075)
        segment(im,-.66,-.02,-.90,-.40,.08)
    elif v==1:
        ellipse(im,0,0,.72,.72); ellipse(im,0,.30,.42,.32)
        ellipse(im,-.61,-.05,.25,.52,-.2); ellipse(im,.61,-.05,.25,.52,.2)
    elif v==2:
        ellipse(im,0,.30,.48,.62); ellipse(im,0,-.40,.43,.39)
        ellipse(im,-.37,-.40,.18,.38); ellipse(im,.37,-.40,.18,.38)
        ellipse(im,-.35,.82,.28,.15); ellipse(im,.35,.82,.28,.15)
    else:
        ellipse(im,-.10,.0,.68,.38); ellipse(im,.48,-.38,.36,.39)
        polygon(im,[(.25,-.58),(.30,-.90),(.48,-.65)])
        ellipse(im,.69,-.22,.26,.17)
        for x in (-.50,-.18,.24,.43): segment(im,x,.25,x,.84,.07)
        segment(im,-.70,-.10,-.92,-.52,.07)
    return im

def frog(v):
    im=blank()
    if v==0:
        ellipse(im,0,.25,.64,.53); ellipse(im,0,-.18,.58,.40)
        ellipse(im,-.36,-.50,.22,.22); ellipse(im,.36,-.50,.22,.22)
        ellipse(im,-.70,.50,.36,.23,-.5); ellipse(im,.70,.50,.36,.23,.5)
        segment(im,-.32,.35,-.60,.90,.09); segment(im,.32,.35,.60,.90,.09)
    elif v==1:
        ellipse(im,-.05,.20,.70,.45); ellipse(im,.38,-.20,.43,.37)
        ellipse(im,.48,-.48,.20,.20); ellipse(im,-.58,.52,.43,.24,-.4)
        segment(im,.28,.30,.60,.82,.10); segment(im,-.20,.42,-.65,.85,.11)
    elif v==2:
        ellipse(im,0,.05,.62,.72); ellipse(im,0,-.45,.56,.35)
        ellipse(im,-.34,-.67,.20,.20); ellipse(im,.34,-.67,.20,.20)
        for side in (-1,1):
            segment(im,side*.40,.15,side*.88,.52,.14)
            segment(im,side*.35,.40,side*.65,.90,.10)
    else:
        ellipse(im,0,.05,.85,.72); ellipse(im,-.43,-.45,.28,.28); ellipse(im,.43,-.45,.28,.28)
        ellipse(im,-.70,.58,.30,.20); ellipse(im,.70,.58,.30,.20)
    return im

def horse(v):
    im=blank()
    if v in (0,3):
        ellipse(im,-.15,.0,.62,.32)
        segment(im,.28,-.05,.48,-.62,.19); ellipse(im,.60,-.67,.23,.31,-.35)
        polygon(im,[(.47,-.87),(.45,-1.05),(.58,-.88)])
        for x in (-.55,-.23,.20,.37):
            segment(im,x,.22,x+(.14 if (v==3 and x<0) else 0),.94,.057)
        segment(im,-.70,-.08,-.93,.50,.10)
        if v==3: segment(im,-.45,.20,-.85,.62,.06)
    elif v==1:
        ellipse(im,0,.38,.43,.58); ellipse(im,0,-.35,.34,.53)
        polygon(im,[(-.25,-.70),(-.28,-1.02),(-.05,-.78)])
        polygon(im,[(.25,-.70),(.28,-1.02),(.05,-.78)])
    else:
        ellipse(im,-.05,.15,.48,.75); ellipse(im,.15,-.50,.34,.46,-.2)
        polygon(im,[(.02,-.78),(.0,-1.04),(.18,-.83)])
        segment(im,-.35,-.25,-.55,.65,.13)
    return im

def ship(v):
    im=blank()
    if v==0:
        polygon(im,[(-.92,.18),(.92,.18),(.60,.63),(-.58,.63)])
        polygon(im,[(-.40,-.20),(.38,-.20),(.50,.18),(-.48,.18)])
        polygon(im,[(-.18,-.48),(.18,-.48),(.18,-.20),(-.18,-.20)])
        segment(im,.42,-.20,.42,-.85,.04); segment(im,.42,-.75,.75,-.45,.035)
    elif v==1:
        polygon(im,[(-.80,-.05),(.80,-.05),(.50,.75),(-.50,.75)])
        polygon(im,[(-.40,-.50),(.40,-.50),(.52,-.05),(-.52,-.05)])
        segment(im,0,-.50,0,-.95,.05)
    elif v==2:
        polygon(im,[(-.75,.48),(.75,.48),(.40,.72),(-.45,.72)])
        segment(im,0,-.85,0,.48,.045)
        polygon(im,[(.05,-.80),(.05,.38),(.72,.30)])
        polygon(im,[(-.05,-.65),(-.05,.35),(-.58,.30)])
    else:
        polygon(im,[(-.95,.10),(.95,.10),(.68,.60),(-.68,.60)])
        for x in (-.55,-.15,.25): polygon(im,[(x,-.35),(x+.32,-.35),(x+.32,.10),(x,.10)])
        segment(im,.60,.1,.60,-.75,.06)
    return im

def truck(v):
    im=blank()
    if v==0:
        polygon(im,[(-.92,-.55),(.20,-.55),(.20,.38),(-.92,.38)])
        polygon(im,[(.20,-.12),(.62,-.12),(.88,.18),(.88,.43),(.20,.43)])
        for x in (-.62,-.25,.62): ellipse(im,x,.45,.19,.19)
    elif v==1:
        polygon(im,[(-.75,-.68),(.75,-.68),(.85,.55),(-.85,.55)])
        for x in (-.63,.63): ellipse(im,x,.58,.20,.20)
    elif v==2:
        polygon(im,[(-.95,-.50),(.15,-.50),(.15,.38),(-.95,.38)])
        polygon(im,[(.15,-.65),(.62,-.65),(.85,-.05),(.85,.43),(.15,.43)])
        for x in (-.68,-.30,.60): ellipse(im,x,.46,.18,.18)
    else:
        polygon(im,[(-.90,-.45),(.25,-.20),(.20,.35),(-.75,.35)])
        polygon(im,[(.20,-.35),(.65,-.35),(.88,.05),(.88,.43),(.20,.43)])
        for x in (-.58,.57): ellipse(im,x,.46,.21,.21)
    return im

makers=[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck]
TW=[]; template_class=[]
for c,maker in enumerate(makers):
    for variant in range(4):
        original=maker(variant)
        for aug in range(2):
            im=original[:,::-1] if aug else original
            # The second version is slightly contracted, giving scale tolerance.
            if aug:
                yy=(np.arange(N)-(N-1)/2)/.88+(N-1)/2
                xi=np.clip(np.rint(yy).astype(int),0,N-1)
                im=im[xi[:,None],xi[None,:]]
            w=im-im.mean(); w/=np.sqrt((w*w).sum())+1e-12
            for sign in (1,-1):
                TW.append((sign*w)[None,None,:,:]); template_class.append(c)
conv("templates", "standard", np.concatenate(TW), b=np.full(len(TW),-1.80), padding=0)
layer("template_relu", "relu", ["templates"])
pool("template_pool", "maxpool", "template_relu", 5)
templateW=np.zeros((10,len(TW)),np.float32)
for j,c in enumerate(template_class): templateW[c,j]=1/16
dense("template_score", "template_pool", templateW*0.80)

layer("scores", "add", ["color_score","edge_score","template_score"])
weights={k:np.asarray(v,dtype=np.float32) for k,v in weights.items()}
(outdir/"network.json").write_text(json.dumps({"layers":layers},indent=2)+"\n")
np.savez(outdir/"weights.npz",**weights)
print("Wrote",len(layers),"layers and",sum(v.size for v in weights.values()),"weights to",outdir)
