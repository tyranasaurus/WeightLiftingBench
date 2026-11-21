"""Data-free traffic-sign network: analytical colour and vector-shape templates.

No data files, fonts, external assets, optimization, or random weights are used.
All coordinates below describe idealized road-sign artwork, not photographs.
"""
import sys
import json
from pathlib import Path
import numpy as np

F32 = np.float32
layers = []
weights = {}


def layer(name, op, source, **settings):
    layers.append(dict(name=name, op=op, inputs=source if isinstance(source, list) else [source], **settings))
    return name


def conv(name, source, w, b=None, padding=0, stride=1):
    w = np.asarray(w, dtype=F32)
    if b is None:
        b = np.zeros(w.shape[0], F32)
    weights[name + '.W'] = w
    weights[name + '.b'] = np.asarray(b, dtype=F32)
    return layer(name, 'conv', source, out=w.shape[0], kernel=list(w.shape[2:]), stride=stride, padding=padding)


# Vector artwork on a dimensionless canvas. Radius of a circular sign is one.
RES = 112
AXIS = (np.arange(RES, dtype=np.float64) + .5) * (2.6 / RES) - 1.3
XX, YY = np.meshgrid(AXIS, AXIS)


def empty():
    return np.zeros_like(XX)


def ellipse(cx, cy, rx, ry):
    return (((XX-cx)/rx)**2 + ((YY-cy)/ry)**2 <= 1).astype(float)


def polygon(points):
    inside = np.zeros(XX.shape, dtype=bool)
    for a, b in zip(points, points[1:] + points[:1]):
        x1, y1 = a; x2, y2 = b
        if y1 != y2:
            cross = (x2-x1)*(YY-y1)/(y2-y1)+x1
            inside ^= ((y1 > YY) != (y2 > YY)) & (XX < cross)
    return inside.astype(float)


def line(points, width):
    d2 = np.full_like(XX, 100.)
    for (x1,y1), (x2,y2) in zip(points[:-1], points[1:]):
        vx,vy=x2-x1,y2-y1
        t=np.clip(((XX-x1)*vx+(YY-y1)*vy)/(vx*vx+vy*vy+1.e-15),0,1)
        d2=np.minimum(d2,(XX-x1-t*vx)**2+(YY-y1-t*vy)**2)
    return (d2 <= (width/2)**2).astype(float)


def bezier(points, n=25):
    p=np.asarray(points,float)
    result=[]
    for t in np.linspace(0,1,n):
        q=p.copy()
        while len(q)>1:
            q=(1-t)*q[:-1]+t*q[1:]
        result.append(tuple(q[0]))
    return result


def arc(cx,cy,rx,ry,start,end,n=40):
    a=np.linspace(start,end,n)*np.pi/180
    return list(zip(cx+rx*np.cos(a),cy+ry*np.sin(a)))


def merge(*masks):
    return np.maximum.reduce(masks)


def rect(x1,y1,x2,y2):
    return ((XX>=x1)&(XX<=x2)&(YY>=y1)&(YY<=y2)).astype(float)


def transform_points(p, sx=1, sy=1, tx=0, ty=0):
    return [(tx+sx*x,ty+sy*y) for x,y in p]


def digit(char, cx, cy, sx, sy, width, style):
    # Ordinary narrow, round sans-serif numerals, constructed from strokes.
    # Coordinates in a digit's own box: x=-.25..+.25, y=-.5..+.5.
    if char=='0':
        paths=[arc(0,0,.225,.48,0,360)]
    elif char=='1':
        paths=[[(-.16,-.27),(.02,-.47),(.02,.48)]]
        if style:
            paths.append([(-.13,.48),(.18,.48)])
    elif char=='2':
        paths=[bezier([(-.22,-.29),(-.23,-.60),(.30,-.60),(.23,-.23)])+
               bezier([(.23,-.23),(.17,-.04),(-.23,.20),(-.23,.46)])+[(.25,.46)]]
    elif char=='3':
        paths=[bezier([(-.22,-.36),(.10,-.65),(.45,-.28),(.02,-.04)])+
               bezier([(.02,-.04),(.45,-.01),(.26,.70),(-.23,.37)])]
        if style:
            paths[0]=[(-.23,-.46),(.22,-.46),(-.01,-.06)]+bezier([(-.01,-.06),(.43,-.12),(.32,.68),(-.23,.39)])
    elif char=='5':
        paths=[[ (.23,-.47),(-.19,-.47),(-.22,-.04) ]+
               bezier([(-.22,-.04),(.27,-.23),(.39,.39),(.05,.47)])+
               bezier([(.05,.47),(-.10,.51),(-.22,.44),(-.24,.34)])]
    elif char=='6':
        paths=[bezier([(.19,-.44),(-.13,-.60),(-.33,-.13),(-.21,.30)])+
               arc(.005,.19,.218,.29,158,-202)]
    elif char=='7':
        paths=[[(-.24,-.47),(.25,-.47),(-.10,.48)]]
    elif char=='8':
        paths=[arc(0,-.25,.205,.235,0,360),arc(0,.235,.23,.255,0,360)]
    else:
        raise ValueError(char)
    return merge(*[line(transform_points(p,sx,sy,cx,cy),width) for p in paths])


def number(text, style, height=1., darkness=1.):
    n=len(text)
    xstep=.55 if n==2 else .43
    sx=.96 if n==2 else .73
    sy=height*((1.03,.94,1.20,1.23)[style])
    if style>=2:
        sx*=.95
    sw=(.115,.103,.130,.115)[style]
    k=empty()
    for j,ch in enumerate(text):
        k=np.maximum(k,digit(ch,(j-(n-1)/2)*xstep,.02,sx,sy,sw,style))
    return k*darkness


def arrow(points, width=.17, head=.33):
    p=list(points)
    (x0,y0),(x1,y1)=p[-2:]
    d=np.array([x1-x0,y1-y0]); d=d/np.sqrt((d*d).sum())
    v=np.array([-d[1],d[0]])
    tip=np.array([x1,y1]); base=tip-head*d
    tri=polygon([tuple(tip),tuple(base+head*.72*v),tuple(base-head*.72*v)])
    return merge(line(p[:-1]+[tuple(base+.07*d)],width),tri)


def person(cx,cy,scale=1,kind=0):
    # Walking figure, head, torso, arms and two legs.
    if kind==0:
        ps=[([(-.02,-.19),(-.10,.10),(-.31,.43)],.11),
            ([(-.10,.10),(.18,.22),(.31,.43)],.11),
            ([(-.01,-.14),(.19,-.02),(.31,-.09)],.075),
            ([(-.03,-.13),(-.22,-.02),(-.31,.12)],.075)]
    else:
        ps=[([(.01,-.19),(.08,.10),(-.14,.43)],.12),
            ([(.08,.10),(.28,.30),(.38,.31)],.105),
            ([(.02,-.12),(-.20,-.02),(-.29,-.09)],.075),
            ([(.02,-.12),(.20,.02),(.31,-.04)],.075)]
    masks=[ellipse(cx,cy-.35*scale,.115*scale,.12*scale)]
    for p,w in ps:
        masks.append(line(transform_points(p,scale,scale,cx,cy),w*scale))
    return merge(*masks)


def car(cx,cy,scale=1,front=True):
    if front:
        p=[(-.19,-.22),(.19,-.22),(.29,-.02),(.30,.23),(-.30,.23),(-.29,-.02)]
        m=polygon(transform_points(p,scale,scale,cx,cy))
        glass=polygon(transform_points([(-.12,-.15),(.12,-.15),(.20,-.02),(-.20,-.02)],scale,scale,cx,cy))
        m=np.maximum(0,m-glass)
        m=merge(m,rect(cx-.26*scale,cy+.17*scale,cx-.15*scale,cy+.31*scale),rect(cx+.15*scale,cy+.17*scale,cx+.26*scale,cy+.31*scale))
    else:
        p=[(-.41,.10),(-.39,-.07),(-.19,-.12),(-.10,-.29),(.18,-.29),(.31,-.08),(.45,-.03),(.45,.15),(-.41,.15)]
        m=polygon(transform_points(p,scale,scale,cx,cy))
        m=merge(m,ellipse(cx-.24*scale,cy+.17*scale,.11*scale,.11*scale),ellipse(cx+.27*scale,cy+.17*scale,.11*scale,.11*scale))
    return m


def truck(cx,cy,scale=1):
    ps=[(-.52,-.25),(.12,-.25),(.12,-.14),(.33,-.14),(.49,.04),(.49,.24),(-.52,.24)]
    m=polygon(transform_points(ps,scale,scale,cx,cy))
    m=np.maximum(0,m-rect(cx+.19*scale,cy-.085*scale,cx+.32*scale,cy+.04*scale))
    return merge(m,ellipse(cx-.31*scale,cy+.25*scale,.115*scale,.115*scale),ellipse(cx+.30*scale,cy+.25*scale,.115*scale,.115*scale))


def bicycle(cx,cy,scale=1):
    def lp(p,w): return line(transform_points(p,scale,scale,cx,cy),w*scale)
    return merge(lp(arc(-.30,.16,.22,.22,0,360),.055),lp(arc(.31,.16,.22,.22,0,360),.055),
                 lp([(-.30,.16),(-.09,-.20),(.13,.16),(-.30,.16),(-.03,.16),(.22,-.20),(.31,.16)],.06),
                 lp([(.14,-.29),(.25,-.29),(.22,-.20)],.065),lp([(-.18,-.26),(-.04,-.26)],.06))


def warning_symbol(c,style):
    w=.10 if style else .125
    if c==11:
        return merge(line([(0,.54),(0,-.32)],.18),polygon([(-.25,-.14),(0,-.46),(.25,-.14)]),line([(-.40,.13),(.40,.13)],.09))
    if c==18:
        return merge(polygon([(-.075,-.39),(.075,-.39),(.045,.21),(-.045,.21)]),ellipse(0,.40,.085,.085))
    if c in (19,20):
        s=-1 if c==19 else 1
        p=bezier([(0,.51),(0,.02),(.04*s,-.17),(.37*s,-.17)])
        return line(p,w+.07)
    if c==21:
        p=bezier([(.15,.52),(.20,.21),(-.24,.23),(-.20,-.04)])+bezier([(-.20,-.04),(-.17,-.27),(.12,-.22),(.24,-.38)])
        return line(p,.16)
    if c==22:
        p=[(-.53,.48),(-.51,.30)]+bezier([(-.51,.30),(-.44,-.12),(-.16,-.10),(-.08,.29)])+bezier([(-.08,.29),(.03,-.10),(.36,-.06),(.47,.31)])+[(.51,.48)]
        return polygon(p)
    if c==23:
        m=car(.10,-.04,.82,True)
        a=bezier([(-.10,.15),(-.63,.22),(.22,.41),(-.34,.55)])
        b=bezier([(.30,.15),(-.21,.29),(.60,.40),(.09,.56)])
        return merge(m,line(a,.065),line(b,.065))
    if c==24:
        return merge(line([(-.34,.55),(-.34,-.37)],.10),line([(.37,.55),(.37,.16),(.16,-.12),(.16,-.37)],.10))
    if c==25:
        return merge(person(-.12,.04,.88,1),line([(-.06,.03),(.45,.54)],.055),polygon([(.05,.56),(.48,.26),(.65,.56)]))
    if c==26:
        housing=rect(-.17,-.47,.17,.53)
        lamps=merge(ellipse(0,-.29,.10,.10),ellipse(0,.03,.10,.10),ellipse(0,.35,.10,.10))
        return np.maximum(0,housing-.7*lamps)
    if c==27:
        m=person(0,.02,.96)
        for yy in (.36,.50,.62):
            m=merge(m,line([(-.51,yy),(.51,yy)],.045))
        return m
    if c==28:
        return merge(person(-.22,-.01,.83),person(.30,.18,.64,1))
    if c==29:
        return bicycle(0,.17,1.0)
    if c==30:
        m=empty()
        for a in np.arange(6)*np.pi/3:
            v=np.array([np.sin(a),-np.cos(a)])
            u=np.array([-v[1],v[0]])
            center=np.array([0,.12])
            tip=center+.47*v
            m=merge(m,line([tuple(center),tuple(tip)],.07))
            q=center+.29*v
            for s in (-1,1):
                end=q+.13*v+s*.13*u
                m=merge(m,line([tuple(q),tuple(end)],.055))
        return m
    if c==31:
        body=polygon([(-.33,.03),(-.14,-.13),(.24,-.09),(.35,-.33),(.44,-.36),(.48,-.20),(.36,.07),(.10,.20),(-.23,.18)])
        return merge(body,line([(-.24,.14),(-.40,.36),(-.24,.52)],.065),line([(-.10,.15),(-.23,.37),(-.04,.45)],.065),
                     line([(.17,.11),(.29,.33),(.51,.34)],.055),line([(.22,.07),(.46,.19),(.54,.40)],.055),
                     line([(.40,-.31),(.29,-.48),(.33,-.58)],.04),line([(.43,-.32),(.51,-.50)],.04),line([(-.30,.06),(-.46,-.03)],.055))
    raise ValueError(c)


def artwork(c,style):
    """Return independent white/black/red/blue/yellow masks and sign support."""
    support=empty(); black=empty(); red=empty(); blue=empty(); yellow=empty(); white=empty()
    circle=ellipse(0,0,1,1)
    if c<=10 or c in (15,16,17,32,41,42):
        support=circle
        white=circle.copy()
        if c not in (6,32,41,42):
            red=circle-ellipse(0,0,.80,.80)
            white=np.maximum(0,white-red)
        else:
            black=.30*(circle-ellipse(0,0,.96,.96))
        if c in (0,1,2,3,4,5,6,7,8):
            text={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}[c]
            black=merge(black,number(text,style,darkness=.6 if c==6 else 1))
        elif c in (9,10,41,42):
            is_truck=c in (10,42)
            left=truck(-.30,.02,.64) if is_truck else car(-.32,.02,.70,True)
            right=car(.31,.02,.70,True)
            if c in (9,10):
                red=merge(red,left)
                black=merge(black,right)
            else:
                black=merge(black,.58*left,.58*right)
        elif c==16:
            black=truck(0,.03,1.12)
        elif c==17:
            red=circle*(1-rect(-.69,-.16,.69,.16))
            white=circle-red
        if c in (6,32,41,42):
            stripes=empty()
            for a in (-.23,-.115,0,.115,.23):
                stripes=merge(stripes,line([(-1.1+a,1.1+a),(1.1+a,-1.1+a)],.053))
            black=merge(black,stripes*circle*.88)
    elif c in (11,18,19,20,21,22,23,24,25,26,27,28,29,30,31):
        support=polygon([(0,-1),(.99,.88),(-.99,.88)])
        inside=polygon([(0,-.70),(.73,.71),(-.73,.71)])
        red=np.maximum(0,support-inside)
        white=inside
        black=warning_symbol(c,style)*inside
        if c==26:
            red=merge(red,ellipse(0,-.29,.095,.095))
            yellow=ellipse(0,.03,.095,.095)
    elif c==12:
        support=polygon([(0,-1),(1,0),(0,1),(-1,0)])
        black=support-polygon([(0,-.96),(.96,0),(0,.96),(-.96,0)])
        white=polygon([(0,-.96),(.96,0),(0,.96),(-.96,0)])
        yellow=polygon([(0,-.76),(.76,0),(0,.76),(-.76,0)])
        white=np.maximum(0,white-yellow)
    elif c==13:
        support=polygon([(-1,-.83),(1,-.83),(0,1)])
        white=polygon([(-.72,-.66),(.72,-.66),(0,.68)])
        red=np.maximum(0,support-white)
    elif c==14:
        a=np.arange(8)*np.pi/4+np.pi/8
        support=polygon(list(zip(np.cos(a),np.sin(a))))
        red=support.copy()
        # STOP, conventional block capitals, freehand stroke construction.
        pS=bezier([(-.03,-.19),(-.31,-.34),(-.34,.00),(-.14,.00)])+bezier([(-.14,.00),(.10,.02),(.02,.34),(-.24,.19)])
        pS=transform_points(pS,.63,1,-.50,0)
        w=.062
        white=merge(line(pS,w),line([(-.26,-.23),(.04,-.23)],w),line([(-.11,-.23),(-.11,.24)],w),
                    line(arc(.20,0,.12,.235,0,360),w),line([(.42,.24),(.42,-.23),(.57,-.23),(.65,-.14),(.59,-.02),(.42,-.02)],w))
        red=np.maximum(0,red-white)
    elif 33<=c<=40:
        support=circle
        blue=circle.copy()
        if c==33:
            white=arrow([(0,.58),(0,-.13),(.56,-.13)],.18,.33)
        elif c==34:
            white=arrow([(0,.58),(0,-.13),(-.56,-.13)],.18,.33)
        elif c==35:
            white=arrow([(0,.59),(0,-.65)],.18,.36)
        elif c in (36,37):
            s=1 if c==36 else -1
            white=merge(arrow([(0,.58),(0,-.65)],.17,.34),arrow([(0,.08),(.22*s,-.12),(.60*s,-.12)],.17,.30))
        elif c in (38,39):
            s=1 if c==38 else -1
            white=arrow([(-.44*s,-.48),(.48*s,.47)],.22,.40)
        else:
            for a in (0,120,240):
                pts=arc(0,0,.48,.48,a+15,a+101,24)
                white=merge(white,arrow(pts,.12,.23))
        blue=np.maximum(0,blue-white)
    # Masks obey paint occlusion; greyscale black can include pale cancellation ink.
    black=np.minimum(black,support)
    red=red*(1-black)
    blue=blue*(1-black)
    yellow=yellow*(1-black)
    white=np.maximum(0,support-np.maximum.reduce([black,red,blue,yellow]))
    if c in (11,18,19,20,21,22,23,24,25,26,27,28,29,30,31):
        interior=polygon([(0,-.65),(.69,.66),(-.69,.66)])
    elif c==13:
        interior=polygon([(-.67,-.61),(.67,-.61),(0,.61)])
    elif c==12:
        interior=polygon([(0,-.70),(.70,0),(0,.70),(-.70,0)])
    else:
        interior=ellipse(0,0,.74,.74)
    return np.array([support,white,black,red,blue,yellow,interior],dtype=F32)


def blur_maps(a):
    # Optical/resize tolerance specified analytically, not estimated from data.
    for axis in (1,2):
        p=np.pad(a,[(0,0) if k!=axis else (2,2) for k in range(3)])
        sl=[slice(None)]*3
        result=np.zeros_like(a)
        for j,w in enumerate((1,4,6,4,1)):
            sl[axis]=slice(j,j+a.shape[axis])
            result+=w/16*p[tuple(sl)]
        a=result
    return a


def sample(a,x,y):
    ix=(x+1.3)*RES/2.6-.5; iy=(y+1.3)*RES/2.6-.5
    x0=np.floor(ix).astype(int); y0=np.floor(iy).astype(int)
    fx=ix-x0; fy=iy-y0
    valid=(x0>=0)&(x0<RES-1)&(y0>=0)&(y0<RES-1)
    x0=np.clip(x0,0,RES-2); y0=np.clip(y0,0,RES-2)
    return (a[:,y0,x0]*(1-fx)*(1-fy)+a[:,y0,x0+1]*fx*(1-fy)+
            a[:,y0+1,x0]*(1-fx)*fy+a[:,y0+1,x0+1]*fx*fy)*valid


def make_feature_weights(m,c):
    s,white,black,red,blue,yellow,interior=m
    # Hierarchical evidence: shared geometry/paint and the interior symbol.
    # Each is a zero-mean correlation, so symbols are not overwhelmed by rings.
    lum=.93*white+.06*black+.29*red+.19*blue+.79*yellow
    area=max(s.sum(),1.e-8)
    mean=lum.sum()/area
    q=lum-mean*s
    r=.85*red-.45*blue+.45*yellow
    b=-.45*red+.85*blue-.80*yellow
    norm=np.sqrt(((2*q)**2+r*r+b*b).sum()*area)
    w=np.zeros((6,16,16),F32)
    w[0]=1.0*q/max(norm,1.e-8)
    w[2]=.50*r/max(norm,1.e-8)
    w[4]=.50*b/max(norm,1.e-8)
    # Interior normalized correlation has zero response to a blank sign.
    iv=interior
    ia=max(iv.sum(),1.e-8)
    # Render the expected ink on a neutral substrate independently of the rim.
    ink=black.copy()
    if 33<=c<=40 or c==14:
        ink=white
    if c in (9,10):
        ink=np.maximum(black,red*iv)
    ink=ink*iv
    frac=ink.sum()/ia
    polarity=1 if (33<=c<=40 or c==14) else -1
    qi=polarity*(ink-frac*iv)
    ni=np.sqrt((qi*qi).sum()*ia)
    # The stronger interior evidence is computed in a separate locally
    # normalized branch, rather than sharing the background's exposure.
    if ni>1.e-5:
        w[0]+=.12*qi/ni
    w[1]=-w[0]; w[3]=-w[2]; w[5]=-w[4]
    if c in (6,32,41,42):
        # Achromatic cancellation artwork cannot explain saturated red or blue.
        penalty=.15*s/area
        w[2]-=penalty; w[3]-=penalty; w[4]-=penalty; w[5]-=penalty
    return w


def interior_branch(gray, global_candidates, nstyles, geom):
    """Candidate-specific contrast normalization followed by shared templates.

    The sampling weights are bilinear interpolation coefficients on ideal
    circular/triangular grids. They are not calculated from any image.
    """
    warning={11,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
    n=10; nv=len(geom)
    outputs=[]
    for family in ('disc','triangle'):
        points=[]
        for iy in range(n):
            for ix in range(n):
                u=(ix+.5)/n; v=(iy+.5)/n
                if family=='disc':
                    # Uniform-area polar sampling, staggered between rings.
                    rad=.73*np.sqrt(v)
                    theta=2*np.pi*(u+(iy%2)*.5/n)
                    points.append((rad*np.cos(theta),rad*np.sin(theta)))
                else:
                    # Uniform-area barycentric grid within a triangle.
                    t=np.sqrt(v)
                    points.append((.71*t*(2*u-1),-.68+1.36*t))
        pts=np.array(points)
        w=np.zeros((nv*n*n,1,32,32),F32)
        for j,(radius,aspect,angle,dx0,dy0) in enumerate(geom):
            th=np.deg2rad(angle); co=np.cos(th); si=np.sin(th)
            px=15.5+dx0+radius*aspect*(co*pts[:,0]-si*pts[:,1])
            py=15.5+dy0+radius*(si*pts[:,0]+co*pts[:,1])
            for i,(x,y) in enumerate(zip(px,py)):
                ix=int(np.floor(x)); iy=int(np.floor(y)); fx=x-ix; fy=y-iy
                for dx,dy,v in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
                    if 0<=ix+dx<32 and 0<=iy+dy<32:
                        w[j*n*n+i,0,iy+dy,ix+dx]+=v
        a=conv(family+'_sample',gray,w)
        a=layer(family+'_candidate_grids','reshape',a,shape=[nv,n,n])
        a=layer(family+'_local_contrast','norm',a)
        a=layer(family+'_shared_grid','reshape',a,shape=[1,nv*n,n])
        match=np.zeros((43*nstyles,1,n,n),F32)
        bias=np.full(43*nstyles,-10.,F32)
        for c in range(43):
            appropriate=(c in warning)==(family=='triangle')
            if not appropriate:
                continue
            for style in range(nstyles):
                pos=nstyles*c+style
                if c in (12,13,15):
                    bias[pos]=0
                    continue
                art=blur_maps(artwork(c,style))
                m=sample(art,pts[:,0],pts[:,1])
                s,white,black,red,blue,yellow,interior=m
                q=.93*white+.06*black+.29*red+.19*blue+.79*yellow
                q=q-q.mean()
                q=q/max(np.sqrt((q*q).sum()*n*n),1.e-8)
                match[pos,0]=.95*q.reshape(n,n)
                bias[pos]=-.95*.42
        a=conv(family+'_symbol_correlations',a,match,bias,stride=n)
        a=layer(family+'_joint_geometry_evidence','add',[a,global_candidates])
        a=layer(family+'_best_geometry','maxpool',a,size=[nv,1],stride=[nv,1])
        a=layer(family+'_styles','reshape',a,shape=[43,nstyles,1])
        a=layer(family+'_best_style','maxpool',a,size=[nstyles,1],stride=[nstyles,1])
        a=layer(family+'_vector','reshape',a,shape=[43])
        # Combine by max(a,b) = relu(a-b)+b, with -10 disabling inapplicable
        # branches. All zero-response blank sign models remain at zero.
        outputs.append(a)
    # Only one family is applicable per class, so a fixed table can simply
    # select its entry through two diagonal linear projections.
    selected=[]
    for family,source in zip(('disc','triangle'),outputs):
        diag=np.array([float((c in warning)==(family=='triangle')) for c in range(43)],F32)
        name=family+'_select'
        weights[name+'.W']=np.diag(diag)
        weights[name+'.b']=np.zeros(43,F32)
        selected.append(layer(name,'dense',source,out=43))
    return layer('local_symbol_scores','add',selected)


def build(out):
    # Grayscale normalization is independent of the joint RGB normalization.
    g=conv('luminance','image',np.array([.299,.587,.114]).reshape(1,3,1,1))
    g=layer('light_adaptation','norm',g)
    rgb=layer('rgb_as_one_channel','reshape','image',shape=[1,96,32])
    rgb=layer('joint_colour_adaptation','norm',rgb)
    rgb=layer('adapted_rgb','reshape',rgb,shape=[3,32,32])
    wg=np.zeros((12,1,1,1),F32)
    wc=np.zeros((12,3,1,1),F32)
    biases=np.zeros(12,F32)
    directions=[np.array([1.]),np.array([-1.])]
    # Six nonnegative parts: light, dark, red +/- and blue +/-.
    for i in range(6):
        cap=1.5 if i<2 else 1.15
        if i<2:
            wg[2*i:2*i+2,0,0,0]=(1 if i==0 else -1)/cap
        else:
            d=np.array([1,-.5,-.5]) if i<4 else np.array([-.5,-.5,1])
            if i in (3,5): d=-d
            wc[2*i:2*i+2,:,0,0]=d/cap
        biases[2*i+1]=-1
    a=conv('gray_projection',g,wg)
    b=conv('colour_projection',rgb,wc,biases)
    a=layer('feature_thresholds','add',[a,b])
    a=layer('positive_parts','relu',a)
    wclip=np.zeros((6,12,1,1),F32)
    for i in range(6):
        wclip[i,2*i,0,0]=1; wclip[i,2*i+1,0,0]=-1
    a=conv('bounded_contrast_features',a,wclip)
    a=layer('antialias','avgpool',a,size=[2,2],stride=[2,2])

    # Every nuisance variation is a fixed, a priori geometric transformation.
    geometries=[]
    geom=[]
    gy,gx=np.mgrid[:16,:16].astype(float)
    for radius in (5.0,6.0,7.0):
        for aspect in (.86,1.0,1.14):
            for angle in (-8.,0.,8.):
                theta=np.deg2rad(angle); co=np.cos(theta); si=np.sin(theta)
                for dy in (-1.,0.,1.):
                    for dx in (-1.,0.,1.):
                        x=(gx-7.5-dx)/(radius*aspect)
                        y=(gy-7.5-dy)/radius
                        geometries.append((co*x+si*y,-si*x+co*y))
                        geom.append((2*radius,aspect,angle,2*dx,2*dy))
    nstyles=4
    variants=len(geometries)*nstyles
    bank=np.empty((43*variants,6*16*16),F32)
    bias=np.zeros(43*variants,F32)
    for c in range(43):
        pos=c*variants
        for style in range(nstyles):
            art=blur_maps(artwork(c,style))
            for x,y in geometries:
                m=sample(art,x,y)
                bank[pos]=make_feature_weights(m,c).reshape(-1)
                pos+=1
        print('constructed class',c,flush=True)
    weights['ideal_sign_matches.W']=bank
    weights['ideal_sign_matches.b']=bias
    a=layer('ideal_sign_matches','dense',a,out=len(bank))
    a=layer('shape_candidates','reshape',a,shape=[43*nstyles,len(geometries),1])
    local=interior_branch(g,a,nstyles,geom)
    layer('scores','reshape',local,shape=[43])
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2))
    np.savez(out/'weights.npz',**weights)
    print('weights:',sum(v.size for v in weights.values()),flush=True)


if __name__=='__main__':
    build(sys.argv[1])
