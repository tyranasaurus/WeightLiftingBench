"""Data-free Fashion-MNIST network: analytic, hand-drawn deformable templates.
Run: python3 build.py OUTPUT_DIRECTORY
No files other than this program are read.
"""
import sys, os, json
import numpy as np

N = 28
NT = 32
# Supersampling is used only to antialias the geometric drawings.
S = 2
yy, xx = np.mgrid[0:N*S, 0:N*S].astype(np.float64)
X = (xx + .5) / (N*S/2) - 1
Y = (yy + .5) / (N*S/2) - 1


def rect(x0, y0, x1, y1):
    return ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)).astype(float)


def ellipse(cx, cy, rx, ry):
    return (((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1).astype(float)


def poly(points):
    # Even-odd scanline test, evaluated analytically on the pixel grid.
    inside = np.zeros(X.shape, dtype=bool)
    for i, (ax, ay) in enumerate(points):
        bx, by = points[(i+1) % len(points)]
        if ay == by:
            continue
        cross = ((ay > Y) != (by > Y)) & (X < (bx-ax)*(Y-ay)/(by-ay)+ax)
        inside ^= cross
    return inside.astype(float)


def union(*a):
    return np.maximum.reduce(a)


def line(ax, ay, bx, by, width):
    vx, vy = bx-ax, by-ay
    q = np.clip(((X-ax)*vx+(Y-ay)*vy)/(vx*vx+vy*vy+1e-12), 0, 1)
    d2 = (X-(ax+q*vx))**2 + (Y-(ay+q*vy))**2
    return (d2 <= width*width).astype(float)


def arc(cx, cy, rx, ry, width, upper=True):
    r = np.sqrt(((X-cx)/rx)**2 + ((Y-cy)/ry)**2)
    a = (np.abs(r-1) < width/min(rx,ry))
    a &= (Y <= cy) if upper else (Y >= cy)
    return a.astype(float)


def down(a):
    return a.reshape(N, S, N, S).mean((1,3))


def blur(a, sigma=.65):
    r = int(np.ceil(3*sigma))
    t = np.arange(-r,r+1)
    g = np.exp(-t*t/(2*sigma*sigma)); g /= g.sum()
    p = np.pad(a, ((r,r),(r,r)), mode='edge')
    a = sum(g[i]*p[i:i+a.shape[0], r:r+a.shape[1]] for i in range(len(g)))
    p = np.pad(a, ((r,r),(r,r)), mode='edge')
    return sum(g[i]*p[r:r+a.shape[0], i:i+a.shape[1]] for i in range(len(g)))


def neck(a, width=.18, depth=.12, v=False):
    if v:
        hole = poly([(-width,-1.02),(width,-1.02),(0,-1+depth*2)])
    else:
        hole = ellipse(0,-1.0,width,depth)
    return a*(1-hole)


def tee(v, shirt=False):
    # Body and the two short sleeves; v controls fittedness and sleeve length.
    bw = .36 + .045*(v%4)
    sw = .78 + .055*((v//2)%4)
    sleeve_end = -.28 + .07*(v%3)
    bottom = 1.0
    body = poly([(-bw,-.88),(bw,-.88),(bw*.94,bottom),(-bw*.94,bottom)])
    left = poly([(-bw,-.88),(-sw,-.72),(-sw*.96,sleeve_end),(-bw,-.43)])
    right = poly([(bw,-.88),(sw,-.72),(sw*.96,sleeve_end),(bw,-.43)])
    a = neck(union(body,left,right), .17+.015*(v%3), .10, v%5==0)
    if shirt:
        # A small turned-down collar and a button placket.
        collar = union(line(-.16,-.92,0,-.72,.045), line(.16,-.92,0,-.72,.045))
        buttons = line(0,-.69,0,.93,.024)
        a *= 1-.75*union(collar,buttons)
    return a


def pullover(v, coat=False, shirt=False):
    bw = .38 + .04*(v%4)
    sw = .69 + .06*((v//2)%4)
    bottom = .82 + .06*(v%4) if not coat else 1.0
    if coat and v%4 == 0: bottom = .65
    # Sloping shoulders and independent hanging sleeves.
    body = poly([(-bw*.75,-.93),(bw*.75,-.93),(bw,-.72),(bw*(1.04 if coat else .94),bottom),(-bw*(1.04 if coat else .94),bottom)])
    sl = poly([(-bw*.72,-.91),(-sw,-.64),(-sw*.90,.68),(-bw*.89,.72),(-bw,-.42)])
    sr = poly([(bw*.72,-.91),(sw,-.64),(sw*.90,.68),(bw*.89,.72),(bw,-.42)])
    a = neck(union(body,sl,sr), .16, .09, shirt)
    if coat:
        # Lapels and centre opening distinguish a coat from a plain sweater.
        detail = union(line(-.17,-.91,-.05,-.35,.055), line(.17,-.91,.05,-.35,.055), line(0,-.33,0,bottom,.028))
        a *= 1-.8*detail
    elif shirt:
        detail = union(line(-.16,-.92,0,-.72,.045),line(.16,-.92,0,-.72,.045),line(0,-.70,0,bottom,.024))
        a *= 1-.8*detail
    else:
        # Ribbed hem and cuffs are darker, not separate objects.
        hem = rect(-bw,bottom-.055,bw,bottom)
        a *= 1-.25*hem
    return a


def trousers(v):
    outer = .48 + .065*(v%5)
    inner = .035 + .022*((v//2)%4)
    crotch = -.18 + .09*(v%4)
    flare = .02 + .055*((v//3)%4)
    waist = rect(-outer,-1,outer,-.70)
    left = poly([(-outer,-.78),(-inner,-.78),(-inner,crotch),(-inner-flare,1),(-outer-flare*.35,1),(-outer*.91,.0)])
    right = poly([(inner,-.78),(outer,-.78),(outer*.91,.0),(outer+flare*.35,1),(inner+flare,1),(inner,crotch)])
    a = union(waist,left,right)
    a *= 1-.22*rect(-outer,-1,outer,-.91)
    return a


def dress(v):
    kind = v % 4
    bw = .28 + .045*((v//2)%4)
    waist_y = -.22 + .10*((v//3)%3)
    waist = bw*(.72 if kind != 3 else .90)
    bottom_w = bw*([1.05,1.65,2.05,.92][kind])
    bodice = poly([(-bw,-.91),(bw,-.91),(waist,waist_y),(-waist,waist_y)])
    skirt = poly([(-waist,waist_y),(waist,waist_y),(bottom_w,1),(-bottom_w,1)])
    a = union(bodice,skirt)
    if kind == 0:  # cap sleeves
        a = union(a,poly([(-bw,-.88),(-bw*1.75,-.65),(-bw*1.5,-.39),(-bw,-.51)]),poly([(bw,-.88),(bw*1.75,-.65),(bw*1.5,-.39),(bw,-.51)]))
    elif kind == 3:  # long narrow sleeves on a straight dress
        a = union(a,poly([(-bw,-.88),(-bw*1.75,-.66),(-bw*1.55,.53),(-bw*1.12,.55),(-bw,-.35)]),poly([(bw,-.88),(bw*1.75,-.66),(bw*1.55,.53),(bw*1.12,.55),(bw,-.35)]))
    a = neck(a,.14+.02*(v%3),.10,kind==2)
    if kind == 2:
        # Shoulder straps surrounding an open neckline.
        a *= 1-rect(-bw*.55,-1.02,bw*.55,-.77)
    a *= 1-.18*rect(-waist,waist_y-.025,waist,waist_y+.025)
    return a


def bag(v):
    kind = v % 4
    wide = .70 + .07*((v//2)%4)
    high = .38 + .07*((v//3)%4)
    cy = .22
    if kind == 0:
        body = rect(-wide,cy-high,wide,cy+high)
    elif kind == 1:
        body = poly([(-wide*.82,cy-high),(wide*.82,cy-high),(wide,cy+high),(-wide,cy+high)])
    elif kind == 2:
        body = union(rect(-wide,cy-high*.65,wide,cy+high),ellipse(0,cy-high*.65,wide,high*.35))
    else:  # tall tote
        wide *= .72; high *= 1.28; cy=.12
        body = poly([(-wide*.85,cy-high),(wide*.85,cy-high),(wide,cy+high),(-wide,cy+high)])
    a = body
    if v % 7 != 0:  # clutches are included, but most bags have a handle
        hx = wide*.48
        top = cy-high
        handle = union(arc(0,top,hx,high*.85,.075,True),line(-hx,top,-hx,top+.16,.045),line(hx,top,hx,top+.16,.045))
        a = union(a,handle)
    # Piping and a flap are generic bag details.
    a *= 1-.16*rect(-wide,cy-high+.035,wide,cy-high+.065)
    if kind == 1:
        a *= 1-.20*poly([(-wide*.75,cy-high),(wide*.75,cy-high),(wide*.55,cy),(-wide*.55,cy)])
    return a


def mirror_if(a, v):
    return a if (v&1)==0 else a[:,::-1].copy()


def sneaker(v):
    # Side elevation: toe at left, heel at right. Horizontal extent is fixed.
    high = .22 + .065*((v//2)%4)
    top = .20 - .10*(v%3)
    upper = poly([(-.98,.47),(-.83,.24),(-.52,.16),(-.25,-.02),(.12,top-high),(.55,top-high),(.80,.08),(.94,.51)])
    sole = poly([(-1,.46),(-.75,.42),(-.2,.47),(.96,.43),(.98,.66),(.55,.72),(-.65,.68),(-1,.59)])
    a = union(upper,sole)
    if v%4 == 0:  # high-top
        a = union(a,poly([(.05,.15),(.08,-.60),(.57,-.62),(.76,.40)]))
    # Sole stripe and three lace strokes.
    a *= 1-.28*line(-.95,.54,.91,.54,.018)
    for k in range(3):
        a *= 1-.32*line(-.20+k*.15,.02+k*.015,-.05+k*.15,.18+k*.015,.018)
    return mirror_if(a,v)


def boot(v):
    shaft_h = .65 + .16*((v//2)%4)
    shaft_w = .34 + .055*(v%4)
    heel_h = .12 + .07*((v//3)%4)
    # Continuous foot, ankle and shaft, with a separate heel block.
    foot = poly([(-1,.45),(-.88,.27),(-.48,.19),(-.18,.02),(.05,-.20),(.08,-shaft_h),(.08+2*shaft_w,-shaft_h),(.10+2*shaft_w,.40),(.64,.48),(.61,.64),(-.72,.65),(-1,.56)])
    heel = rect(.30,.48,.61,.48+heel_h)
    a = union(foot,heel)
    if v%5 == 0:  # higher, narrow heel with an arch under the foot
        a *= 1-poly([(-.05,.47),(.33,.38),(.32,.75),(.05,.70)])
        a = union(a,line(.43,.30,.48,.88,.045))
    a *= 1-.24*line(.08,-shaft_h+.06,.08+2*shaft_w,-shaft_h+.06,.02)
    if v%3 == 0:
        a *= 1-.25*line(.12,-shaft_h*.65,.12+2*shaft_w,-shaft_h*.65,.018)
    return mirror_if(a,v)


def sandal(v):
    kind = (v//2)%4
    # All members share a thin sole; their uppers differ fundamentally.
    sole = poly([(-1,.50),(-.72,.44),(-.12,.48),(.84,.43),(.98,.54),(.89,.64),(-.77,.66),(-1,.59)])
    if kind == 0:  # low open sandal
        a = union(sole,line(-.72,.43,-.36,.10,.065),line(-.36,.10,.02,.46,.065),line(.40,.45,.53,.16,.07),line(.53,.16,.75,.43,.07))
    elif kind == 1:  # high heel and open toe
        a = union(sole,poly([(-.93,.47),(-.65,.21),(-.30,.25),(.10,.38),(.45,-.40),(.69,-.42),(.70,.42),(.43,.50)]),line(.62,.30,.68,.91,.045),line(-.55,.25,-.15,-.05,.05))
    elif kind == 2:  # wedge heel
        a = union(sole,poly([(-.95,.46),(-.64,.18),(-.28,.24),(.15,.34),(.43,-.27),(.75,-.25),(.84,.63),(.38,.73),(.12,.45)]),line(-.55,.20,-.14,-.10,.05))
    else:  # strappy heeled sandal / open boot
        a = union(sole,line(-.83,.43,-.40,.12,.06),line(-.40,.12,.06,.43,.06),line(.25,.40,.36,-.55,.055),line(.68,.43,.65,-.58,.055),line(.36,-.55,.65,-.58,.055),line(.62,.28,.70,.88,.045))
        for k in range(3): a = union(a,line(.33,-.42+k*.20,.67,-.38+k*.20,.035))
    return mirror_if(a,v)


def normalize_extent(a):
    """Centre a drawing and put its longest bounding-box side on 28 pixels."""
    ys,xs=np.where(a>.01)
    if not len(xs): return a
    x0,x1=xs.min(),xs.max();y0,y1=ys.min(),ys.max()
    scale=max(x1-x0+1,y1-y0+1)/(N*S)
    cx=(x0+x1+1)/2;cy=(y0+y1+1)/2
    sx=cx+(xx-(N*S)/2)*scale
    sy=cy+(yy-(N*S)/2)*scale
    ix=np.clip(np.floor(sx).astype(int),0,N*S-2)
    iy=np.clip(np.floor(sy).astype(int),0,N*S-2)
    fx=np.clip(sx-ix,0,1);fy=np.clip(sy-iy,0,1)
    return ((1-fx)*(1-fy)*a[iy,ix]+fx*(1-fy)*a[iy,ix+1]+(1-fx)*fy*a[iy+1,ix]+fx*fy*a[iy+1,ix+1])


def template(c, v, detail=True):
    if c == 0: a=tee(v,False)
    elif c == 1: a=trousers(v)
    elif c == 2: a=pullover(v,False,False)
    elif c == 3: a=dress(v)
    elif c == 4: a=pullover(v,True,False)
    elif c == 5: a=sandal(v)
    elif c == 6:
        a=tee(v,True) if v%4==0 else pullover(v,False,True)
    elif c == 7: a=sneaker(v)
    elif c == 8: a=bag(v)
    else: a=boot(v)
    a = normalize_extent(a)
    a = down(a)
    # Small, deterministic translations are deformations of the general model,
    # not measurements from any example.
    if (v//8)%2:
        a = np.roll(a, 1 if (v%8)<4 else -1, axis=1)
    return blur(a,.55 if detail else .85)


def unit(a):
    a = a-a.mean()
    return a/(np.linalg.norm(a)+1e-12)


def make_templates(c, small=False, highpass=False):
    out=[]
    for v in range(NT):
        a=template(c,v,True)
        if highpass: a=a-blur(a,1.35)
        if small: a=a.reshape(14,2,14,2).mean((1,3))
        out.append(unit(a))
    return np.stack(out)


def add_class_pool(layers, source, prefix, n):
    layers.append({'name':prefix+'_r','op':'reshape','inputs':[source],'shape':[1,1,n]})
    layers.append({'name':prefix+'_p','op':'maxpool','inputs':[prefix+'_r'],'size':[1,NT],'stride':[1,NT]})
    layers.append({'name':prefix+'_v','op':'reshape','inputs':[prefix+'_p'],'shape':[10]})


def add_family_pool(layers, weights, source, prefix, scale):
    """Mean of four maxima, each over eight related deformations."""
    layers.append({'name':prefix+'_r','op':'reshape','inputs':[source],'shape':[1,1,10*NT]})
    layers.append({'name':prefix+'_p','op':'maxpool','inputs':[prefix+'_r'],'size':[1,8],'stride':[1,8]})
    layers.append({'name':prefix+'_v','op':'reshape','inputs':[prefix+'_p'],'shape':[40]})
    layers.append({'name':prefix+'_s','op':'dense','inputs':[prefix+'_v'],'out':10})
    w=np.zeros((10,40))
    for c in range(10): w[c,4*c:4*c+4]=scale/4
    weights[prefix+'_s.W']=w;weights[prefix+'_s.b']=np.zeros(10)
    return prefix+'_s'


def build(outdir):
    layers=[]; weights={}
    layers.append({'name':'small','op':'avgpool','inputs':['image'],'size':[2,2],'stride':[2,2]})
    layers.append({'name':'snorm','op':'norm','inputs':['small']})
    layers.append({'name':'inorm','op':'norm','inputs':['image']})
    # A signed centre-surround representation for contours and internal lines.
    layers.append({'name':'unblur','op':'conv','inputs':['image'],'out':1,'kernel':[5,5],'stride':1,'padding':2})
    g=np.exp(-((np.arange(5)[:,None]-2)**2+(np.arange(5)[None,:]-2)**2)/(2*1.1**2));g/=g.sum()
    weights['unblur.W']=(-g[None,None]).astype(np.float32);weights['unblur.b']=np.zeros(1,np.float32)
    layers.append({'name':'hp','op':'add','inputs':['image','unblur']})
    layers.append({'name':'hpsmall','op':'avgpool','inputs':['hp'],'size':[2,2],'stride':[2,2]})
    layers.append({'name':'hpnorm','op':'norm','inputs':['hpsmall']})
    layers.append({'name':'shape','op':'conv','inputs':['snorm'],'out':10*NT,'kernel':[14,14],'stride':1,'padding':0})
    layers.append({'name':'detail','op':'conv','inputs':['inorm'],'out':10*NT,'kernel':[28,28],'stride':1,'padding':0})
    layers.append({'name':'edge','op':'conv','inputs':['hpnorm'],'out':10*NT,'kernel':[14,14],'stride':1,'padding':0})
    ts=[make_templates(c,True,False) for c in range(10)]
    td=[make_templates(c,False,False) for c in range(10)]
    te=[make_templates(c,True,True) for c in range(10)]
    weights['shape.W']=np.concatenate(ts)[:,None]/14; weights['shape.b']=np.zeros(10*NT)
    weights['detail.W']=np.concatenate(td)[:,None]/28; weights['detail.b']=np.zeros(10*NT)
    weights['edge.W']=np.concatenate(te)[:,None]/14; weights['edge.b']=np.zeros(10*NT)
    add_class_pool(layers,'shape','sp',10*NT)
    add_class_pool(layers,'detail','dp',10*NT)
    add_class_pool(layers,'edge','ep',10*NT)
    for name,source,scale in [('ss','sp_v',1.0),('ds','dp_v',.38),('es','ep_v',.22)]:
        layers.append({'name':name,'op':'dense','inputs':[source],'out':10})
        weights[name+'.W']=np.eye(10)*scale;weights[name+'.b']=np.zeros(10)
    # Independent, hand-set shape rules on the original intensity image.
    ry,rx=np.mgrid[0:28,0:28];rx=(rx+.5)/14-1;ry=(ry+.5)/14-1
    def reg(x0,y0,x1,y1):
        m=((rx>=x0)&(rx<=x1)&(ry>=y0)&(ry<=y1)).astype(float)
        return m/m.sum()
    gap=.5*(reg(-.52,.25,-.14,.95)+reg(.14,.25,.52,.95))-reg(-.09,.25,.09,.95)
    sleeves=.5*(reg(-.94,.0,-.57,.78)+reg(.57,.0,.94,.78))
    front=.5*(reg(-.29,-.48,-.09,.72)+reg(.09,-.48,.29,.72))-reg(-.045,-.48,.045,.72)
    skirt=.5*(reg(-.85,.55,-.50,.95)+reg(.50,.55,.85,.95))-.5*(reg(-.65,-.20,-.40,.15)+reg(.40,-.20,.65,.15))
    rule_w=np.zeros((10,28,28))
    rule_w[0]=-.15*gap-.07*sleeves-.03*front
    rule_w[1]=.38*gap-.05*sleeves-.05*front
    rule_w[2]=-.08*gap+.04*sleeves-.03*front
    rule_w[3]=-.18*gap-.05*sleeves+.06*skirt
    rule_w[4]=-.08*gap+.05*sleeves+.12*front
    rule_w[6]=-.08*gap+.04*sleeves+.12*front
    rule_w[8]=-.06*gap-.05*sleeves
    layers.append({'name':'rules','op':'dense','inputs':['image'],'out':10})
    weights['rules.W']=rule_w.reshape(10,-1)
    weights['rules.b']=np.array([0,0,0,0,.015,0,.025,0,0,0])
    layers.append({'name':'scores','op':'add','inputs':['ss','ds','es','rules']})
    os.makedirs(outdir,exist_ok=True)
    with open(os.path.join(outdir,'network.json'),'w') as f: json.dump({'layers':layers},f,separators=(',',':'))
    np.savez(os.path.join(outdir,'weights.npz'),**{k:np.asarray(v,dtype=np.float32) for k,v in weights.items()})

if __name__=='__main__':
    build(sys.argv[1])
