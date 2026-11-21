"""Build a data-free, hand-designed German traffic-sign network.

All templates are analytic drawings of the sign designs.  No data files are read.
The network compares normalized colour/brightness maps with outer-sign and
inner-pictogram templates at several hand-chosen scales and aspect ratios.
"""
import sys, os, json
import numpy as np

S = 16
Q = 4                         # antialiasing supersampling
N = S * Q
BG = np.array([.5, .5, .5], float)
RED = np.array([.72, .045, .055], float)
WHITE = np.array([.88, .88, .84], float)
BLACK = np.array([.035, .04, .035], float)
BLUE = np.array([.035, .16, .65], float)
YELLOW = np.array([.95, .63, .035], float)
GREY = np.array([.30, .31, .29], float)


def geometry(radius=6.6, aspect=1., shift=(0., 0.), angle=0.):
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    x = (xx + .5 - N / 2) / Q - shift[0]
    y = (yy + .5 - N / 2) / Q - shift[1]
    a = np.deg2rad(angle)
    x, y = np.cos(a)*x + np.sin(a)*y, -np.sin(a)*x + np.cos(a)*y
    return x / (radius * aspect), y / radius


def paint(im, mask, colour):
    im[mask] = colour


def line(im, pts, width, colour=BLACK):
    pts = np.asarray(pts, float)
    yy, xx = np.mgrid[0:N, 0:N]
    # Coordinates already established by the current drawing transform.
    x, y = im._x, im._y
    d2 = np.full(x.shape, 1e9)
    for a, b in zip(pts[:-1], pts[1:]):
        vx, vy = b - a
        den = vx*vx + vy*vy
        t = np.clip(((x-a[0])*vx + (y-a[1])*vy) / max(den, 1e-12), 0., 1.)
        d2 = np.minimum(d2, (x-a[0]-t*vx)**2 + (y-a[1]-t*vy)**2)
    paint(im.rgb, d2 <= (width/2)**2, colour)


class Canvas:
    def __init__(self, x, y):
        self.rgb = np.empty((N, N, 3), float); self.rgb[:] = BG
        self._x, self._y = x, y
    def line(self, pts, width, colour=BLACK):
        line(self, pts, width, colour)
    def disc(self, cx, cy, r, colour):
        paint(self.rgb, (self._x-cx)**2 + (self._y-cy)**2 <= r*r, colour)
    def ellipse(self, cx, cy, rx, ry, colour, width=None):
        d = np.sqrt(((self._x-cx)/rx)**2 + ((self._y-cy)/ry)**2)
        m = d <= 1 if width is None else np.abs(d-1) <= width/min(rx,ry)/2
        paint(self.rgb, m, colour)
    def rect(self, x0, y0, x1, y1, colour):
        paint(self.rgb, (self._x>=x0)&(self._x<=x1)&(self._y>=y0)&(self._y<=y1), colour)
    def poly(self, vertices, colour):
        # Even-odd scan conversion in the analytic sign coordinate system.
        x, y = self._x, self._y
        inside = np.zeros(x.shape, bool)
        vs = list(vertices)
        for a, b in zip(vs, vs[1:]+vs[:1]):
            if abs(a[1]-b[1]) < 1e-12: continue
            cross = ((y >= min(a[1],b[1])) & (y < max(a[1],b[1])) &
                     (x < a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1])))
            inside ^= cross
        paint(self.rgb, inside, colour)
    def arc(self, cx, cy, rx, ry, a0, a1, width, colour=BLACK):
        t = np.linspace(np.deg2rad(a0), np.deg2rad(a1), 30)
        self.line(list(zip(cx+rx*np.cos(t), cy+ry*np.sin(t))), width, colour)


def triangle_kind(c):
    return c == 11 or 18 <= c <= 31


def outer(c, x, y):
    im = Canvas(x, y); r = np.sqrt(x*x+y*y)
    if c <= 10 or c in (15,16):
        im.disc(0,0,1,RED); im.disc(0,0,.77,WHITE)
    elif c in (6,32,41,42):
        im.disc(0,0,1,GREY); im.disc(0,0,.91,WHITE)
    elif c == 17:
        im.disc(0,0,1,RED)
    elif c == 12:
        im.poly([(0,-1),(1,0),(0,1),(-1,0)], WHITE)
        im.poly([(0,-.78),(.78,0),(0,.78),(-.78,0)], YELLOW)
    elif c == 13:
        im.poly([(-1,-.77),(1,-.77),(0,1)], RED)
        im.poly([(-.70,-.52),(.70,-.52),(0,.66)], WHITE)
    elif c == 14:
        v=[(np.cos(np.pi/8+k*np.pi/4),np.sin(np.pi/8+k*np.pi/4)) for k in range(8)]
        im.poly(v,RED)
        v=[(.91*np.cos(np.pi/8+k*np.pi/4),.91*np.sin(np.pi/8+k*np.pi/4)) for k in range(8)]
        im.line(v+[v[0]],.055,WHITE)
    elif triangle_kind(c):
        im.poly([(0,-1),(-.94,.83),(.94,.83)],RED)
        im.poly([(0,-.66),(-.62,.60),(.62,.60)],WHITE)
    elif 33 <= c <= 40:
        im.disc(0,0,1,WHITE); im.disc(0,0,.94,BLUE)
    return im


def inner_mask(c,x,y):
    r2=x*x+y*y
    if c <= 10 or c in (6,15,16,17,32,41,42): return r2 < .74**2
    if c == 12: return np.abs(x)+np.abs(y)<.72
    if c == 13:
        return (y>-.51)&(y<.58)&(np.abs(x)<.68*(.62-y)/1.13)
    if c == 14: return (np.abs(x)<.78)&(np.abs(y)<.78)&(np.abs(x)+np.abs(y)<1.12)
    if triangle_kind(c): return (y>-.63)&(y<.56)&(np.abs(x)<.59*(y+.66)/1.26)
    if 33 <= c <= 40: return r2 < .88**2
    return r2 < 1


def arrow(im, pts, width=.14, head=.28, colour=WHITE):
    im.line(pts,width,colour)
    a,b=np.asarray(pts[-2],float),np.asarray(pts[-1],float)
    v=b-a; v/=max(np.linalg.norm(v),1e-9); n=np.array([-v[1],v[0]])
    p1=b-head*v+head*.62*n; p2=b-head*v-head*.62*n
    im.poly([b,p1,p2],colour)


def digit(im, d, cx, cy=0., scale=1., colour=BLACK):
    # Geometric single-stroke DIN-like numerals, in a 0.5 by 0.9 box.
    def tr(points): return [(cx+scale*x,cy+scale*y) for x,y in points]
    w=.11*scale
    if d==0: im.ellipse(cx,cy,.23*scale,.43*scale,colour,width=w)
    elif d==1:
        im.line(tr([(-.18,-.25),(0,-.42),(0,.41)]),w,colour)
        im.line(tr([(-.14,.41),(.16,.41)]),w,colour)
    elif d==2:
        im.arc(cx,cy-.17*scale,.23*scale,.25*scale,180,360,w,colour)
        im.line(tr([(.23,-.17),(-.21,.34),(.23,.34)]),w,colour)
    elif d==3:
        im.line(tr([(-.20,-.35),(-.03,-.42)]),w,colour)
        im.arc(cx-.03*scale,cy-.21*scale,.23*scale,.21*scale,-90,90,w,colour)
        im.arc(cx-.03*scale,cy+.20*scale,.24*scale,.22*scale,-90,90,w,colour)
        im.line(tr([(-.03,.42),(-.20,.34)]),w,colour)
    elif d==4:
        im.line(tr([(.12,-.42),(.12,.42)]),w,colour)
        im.line(tr([(.10,-.40),(-.22,.13),(.24,.13)]),w,colour)
    elif d==5:
        im.line(tr([(.20,-.41),(-.18,-.41),(-.20,-.05),(-.02,-.08)]),w,colour)
        im.arc(cx-.02*scale,cy+.16*scale,.24*scale,.26*scale,-90,150,w,colour)
    elif d==6:
        im.ellipse(cx,cy+.17*scale,.23*scale,.25*scale,colour,width=w)
        im.line(tr([(.18,-.40),(-.06,-.38),(-.20,-.10),(-.21,.16)]),w,colour)
    elif d==7:
        im.line(tr([(-.22,-.40),(.22,-.40),(-.10,.42)]),w,colour)
    elif d==8:
        im.ellipse(cx,cy-.21*scale,.21*scale,.21*scale,colour,width=w)
        im.ellipse(cx,cy+.22*scale,.23*scale,.21*scale,colour,width=w)
    elif d==9:
        im.ellipse(cx,cy-.18*scale,.23*scale,.24*scale,colour,width=w)
        im.line(tr([(.21,-.12),(.18,.18),(.06,.40),(-.12,.40)]),w,colour)


def car(im,cx,cy,scale=1.,colour=BLACK,truck=False):
    if truck:
        im.rect(cx-.43*scale,cy-.23*scale,cx+.15*scale,cy+.18*scale,colour)
        im.poly([(cx+.15*scale,cy-.08*scale),(cx+.29*scale,cy-.08*scale),
                 (cx+.42*scale,cy+.06*scale),(cx+.42*scale,cy+.18*scale),(cx+.15*scale,cy+.18*scale)],colour)
        for dx in (-.25,.26): im.disc(cx+dx*scale,cy+.23*scale,.09*scale,colour)
    else:
        im.poly([(cx-.42*scale,cy+.12*scale),(cx-.36*scale,cy-.06*scale),
                 (cx-.16*scale,cy-.10*scale),(cx-.06*scale,cy-.26*scale),
                 (cx+.18*scale,cy-.26*scale),(cx+.32*scale,cy-.07*scale),
                 (cx+.43*scale,cy-.02*scale),(cx+.40*scale,cy+.16*scale),
                 (cx-.40*scale,cy+.16*scale)],colour)
        for dx in (-.25,.25): im.disc(cx+dx*scale,cy+.18*scale,.09*scale,colour)


def person(im,cx,cy,s=1.,colour=BLACK,child=False):
    im.disc(cx,cy-.34*s,.095*s,colour)
    im.line([(cx,cy-.23*s),(cx-.03*s,cy+.10*s)],.13*s,colour)
    im.line([(cx-.03*s,cy-.12*s),(cx-.23*s,cy+.02*s)],.09*s,colour)
    im.line([(cx,cy-.12*s),(cx+.22*s,cy-.02*s)],.09*s,colour)
    im.line([(cx-.03*s,cy+.08*s),(cx-.20*s,cy+.42*s)],.10*s,colour)
    im.line([(cx-.03*s,cy+.08*s),(cx+.18*s,cy+.37*s)],.10*s,colour)


def symbol(c,x,y):
    im=Canvas(x,y)
    if 0 <= c <= 8:
        nums=[20,30,50,60,70,80,80,100,120]
        text=str(nums[c]); sc=1.0 if len(text)==2 else .82
        spacing=.52*sc
        for j,ch in enumerate(text): digit(im,int(ch),(j-(len(text)-1)/2)*spacing,0,sc,GREY if c==6 else BLACK)
    elif c in (9,10):
        car(im,-.34,-.02,.78,BLACK,c==10); car(im,.34,-.02,.78,RED,c==10)
    elif c == 11:
        im.line([(0,-.57),(0,.58)],.18); im.line([(-.50,-.02),(.38,-.02)],.14)
    elif c == 14:
        # Small geometric STOP lettering.
        digit(im,0,-.13,0,.48,WHITE) # O
        im.line([(-.55,-.20),(-.32,-.20),(-.51,.20),(-.27,.20)],.045,WHITE)
        im.line([(-.07,-.20),(.20,-.20)],.05,WHITE); im.line([(.065,-.20),(.065,.21)],.05,WHITE)
        im.line([(.29,.20),(.29,-.20),(.52,-.20),(.52,-.02),(.29,-.02)],.05,WHITE)
    elif c == 16:
        car(im,0,0,1.25,BLACK,True)
    elif c == 17:
        im.rect(-.57,-.12,.57,.12,WHITE)
    elif c == 18:
        im.line([(0,-.49),(0,.14)],.14); im.disc(0,.40,.085,BLACK)
    elif c in (19,20):
        sg=-1 if c==19 else 1
        arrow(im,[(sg*.15,.53),(sg*.14,.12),(sg*.02,-.10),(sg*-.28,-.30)],.13,.25,BLACK)
    elif c == 21:
        im.line([(-.15,.55),(-.15,.25),(.18,.02),(-.15,-.20),(-.15,-.50)],.14)
        im.poly([(-.15,-.59),(-.29,-.35),(-.01,-.35)],BLACK)
    elif c == 22:
        im.line([(-.55,.32),(.55,.32)],.10)
        im.arc(-.25,.31,.23,.26,180,360,.13); im.arc(.25,.31,.23,.26,180,360,.13)
    elif c == 23:
        im.rect(-.20,-.28,.20,.18,BLACK)
        for xx in (-.16,.16):
            im.disc(xx,-.22,.075,WHITE); im.disc(xx,.10,.075,WHITE)
        im.line([(-.31,.17),(-.43,.32),(-.23,.49)],.07)
        im.line([(.30,.17),(.43,.32),(.23,.49)],.07)
    elif c == 24:
        im.line([(-.29,-.51),(-.29,.54)],.13)
        im.line([(.30,-.51),(.30,-.08),(.04,.54)],.13)
    elif c == 25:
        im.disc(-.18,-.36,.09,BLACK)
        im.line([(-.17,-.24),(-.05,.04),(-.26,.34)],.12)
        im.line([(-.09,-.12),(.20,.02)],.10)
        im.line([(-.07,.01),(.12,.35)],.10)
        im.line([(.15,-.26),(.39,.36)],.07)
        im.poly([(.29,.29),(.49,.29),(.42,.45),(.25,.43)],BLACK)
        im.line([(-.50,.43),(-.31,.30),(-.12,.43)],.10)
    elif c == 26:
        im.rect(-.22,-.55,.22,.55,BLACK)
        im.disc(0,-.34,.115,RED); im.disc(0,0,.115,YELLOW); im.disc(0,.34,.115,WHITE)
    elif c == 27:
        person(im,0,0,1.25)
    elif c == 28:
        person(im,-.20,.02,.90); person(im,.24,.08,.72)
    elif c == 29:
        im.ellipse(-.32,.25,.22,.22,BLACK,width=.065)
        im.ellipse(.34,.25,.22,.22,BLACK,width=.065)
        im.line([(-.32,.25),(-.08,-.05),(.10,.25),(-.32,.25),(-.12,.25),(.02,-.12),(.34,.25)],.07)
        im.line([(-.08,-.05),(-.18,-.16),(-.04,-.16)],.07)
        im.disc(.02,-.39,.08,BLACK); im.line([(.01,-.30),(-.08,-.08),(.16,-.07),(.29,-.18)],.09)
    elif c == 30:
        for a in (0,60,120):
            t=np.deg2rad(a); v=np.array([np.cos(t),np.sin(t)])
            p1=-.52*v; p2=.52*v
            im.line([p1,p2],.075)
            for sg in (-1,1):
                p=sg*.30*v
                for da in (-35,35):
                    q=p-sg*.17*np.array([np.cos(t+np.deg2rad(da)),np.sin(t+np.deg2rad(da))])
                    im.line([p,q],.055)
    elif c == 31:
        im.ellipse(-.05,.02,.34,.18,BLACK)
        im.line([(-.26,.12),(-.34,.48)],.09); im.line([(-.06,.14),(-.02,.48)],.09)
        im.line([(.13,.12),(.22,.47)],.09); im.line([(.26,.05),(.38,.39)],.09)
        im.line([(.19,-.06),(.35,-.31)],.15); im.ellipse(.39,-.34,.14,.09,BLACK)
        im.line([(.38,-.40),(.31,-.56),(.24,-.60)],.045)
        im.line([(.42,-.41),(.48,-.57),(.55,-.59)],.045)
    elif c in (6,32,41,42):
        if c in (41,42):
            car(im,-.29,.02,.65,GREY,c==42); car(im,.29,.02,.65,BLACK,c==42)
        for d in (-.60,-.30,0.,.30,.60):
            im.line([(-.9,-.9+d),(.9,.9+d)],.095,GREY)
    elif c in (33,34):
        sg=1 if c==33 else -1
        arrow(im,[(0,.55),(0,.02),(sg*.13,-.22),(sg*.40,-.32)],.16,.29)
    elif c == 35:
        arrow(im,[(0,.56),(0,-.48)],.17,.32)
    elif c in (36,37):
        sg=1 if c==36 else -1
        arrow(im,[(0,.56),(0,-.48)],.15,.28)
        arrow(im,[(0,.12),(sg*.45,.12)],.15,.27)
    elif c in (38,39):
        sg=1 if c==38 else -1
        arrow(im,[(-sg*.38,-.48),(sg*.38,.46)],.18,.32)
    elif c == 40:
        for a in (-90,30,150):
            t=np.deg2rad(a); p=np.array([.48*np.cos(t),.48*np.sin(t)])
            im.arc(0,0,.43,.43,a+35,a+185,.12,WHITE)
            b=np.deg2rad(a+185); tip=np.array([.43*np.cos(b),.43*np.sin(b)])
            tangent=np.array([-np.sin(b),np.cos(b)])
            normal=np.array([np.cos(b),np.sin(b)])
            im.poly([tip,tip-.25*tangent+.13*normal,tip-.25*tangent-.13*normal],WHITE)
    return im


# Twelve geometry variants per class: scale/aspect and small translations.
variants=[]
for radius in (5.6,6.6,7.4):
    for aspect in (.86,1.,1.16): variants.append((radius,aspect,(0.,0.),0.))
variants += [(6.6,1.,(-.65,0.),0.),(6.6,1.,(.65,0.),0.),(6.6,1.,(0.,.65),0.)]
V=len(variants); T=43*V
outer_weights=np.zeros((T,3,S,S),np.float32)
inner_weights=np.zeros((T,3,S,S),np.float32)
fine_weights=np.zeros((T,1,2*S,2*S),np.float32)


def feature_maps(rgb):
    # Grey level and two colour-opponent channels, matching the input layer.
    return np.stack([rgb[...,0]*.299+rgb[...,1]*.587+rgb[...,2]*.114,
                     rgb[...,0]-.5*(rgb[...,1]+rgb[...,2]),
                     rgb[...,2]-.5*(rgb[...,0]+rgb[...,1])])


def down(a):
    return a.reshape(S,Q,S,Q,3).mean((1,3))


def kernel_from_template(feat, mask, zero_mean, norm_count, gain=1.):
    a=feat.copy()
    a[:,~mask]=0
    if zero_mean:
        for k in range(3):
            vals=a[k][mask]; a[k][mask]-=vals.mean()
    n=float(np.sqrt(np.sum(a*a)))
    if n>1e-8: a *= gain/(np.sqrt(norm_count)*n)
    else: a[:]=0
    return a


for c in range(43):
    for v,(radius,aspect,shift,angle) in enumerate(variants):
        x,y=geometry(radius,aspect,shift,angle)
        out=down(outer(c,x,y).rgb).transpose(2,0,1)
        inn=down(symbol(c,x,y).rgb).transpose(2,0,1)
        mfull=np.ones((S,S),bool)
        # Inner masks are evaluated at output resolution (antialiasing comes from RGB).
        xx=(np.arange(S)+.5-S/2-shift[0])/(radius*aspect)
        yy=(np.arange(S)+.5-S/2-shift[1])/radius
        X,Y=np.meshgrid(xx,yy)
        mi=inner_mask(c,X,Y)
        of=feature_maps(out.transpose(1,2,0)); inf=feature_maps(inn.transpose(1,2,0))
        outer_weights[c*V+v]=kernel_from_template(of,mfull,True,3*S*S,1.25)
        inner_weights[c*V+v]=kernel_from_template(inf.copy(),mi,True,int(mi.sum())*3,.70)

        # Selected fine inner-symbol correlations: numerals and pictograms for
        # which narrow strokes carry the principal class information.
        xf,yf=geometry(radius*2,aspect,tuple(2*q for q in shift),angle)
        fine_rgb=symbol(c,xf,yf).rgb.reshape(2*S,2,2*S,2,3).mean((1,3))
        fine=fine_rgb[...,0]*.299+fine_rgb[...,1]*.587+fine_rgb[...,2]*.114
        xx=(np.arange(2*S)+.5-S-2*shift[0])/(2*radius*aspect)
        yy=(np.arange(2*S)+.5-S-2*shift[1])/(2*radius)
        XF,YF=np.meshgrid(xx,yy)
        mf=inner_mask(c,XF,YF)
        fine[~mf]=0.; fine[mf]-=fine[mf].mean()
        nf=float(np.sqrt(np.sum(fine*fine)))
        fine_gain=.65 if c<=8 else (.90 if c in (9,11,17,18,22,36,37) else 0.)
        if nf>1e-8: fine_weights[c*V+v,0]=fine_gain*nf**-1*fine/np.sqrt(int(mf.sum()))

layers=[
 {"name":"features","op":"conv","inputs":["image"],"out":3,"kernel":[1,1],"stride":1,"padding":0},
 {"name":"smooth","op":"avgpool","inputs":["features"],"size":[2,2],"stride":[2,2]},
 {"name":"standardize","op":"norm","inputs":["smooth"]},
 {"name":"grey","op":"conv","inputs":["image"],"out":1,"kernel":[1,1],"stride":1,"padding":0},
 {"name":"grey_standardize","op":"norm","inputs":["grey"]},
 {"name":"outer_match","op":"conv","inputs":["standardize"],"out":T,"kernel":[S,S],"stride":1,"padding":0},
 {"name":"inner_match","op":"conv","inputs":["standardize"],"out":T,"kernel":[S,S],"stride":1,"padding":0},
 {"name":"fine_match","op":"conv","inputs":["grey_standardize"],"out":T,"kernel":[2*S,2*S],"stride":1,"padding":0},
 {"name":"evidence","op":"add","inputs":["outer_match","inner_match","fine_match"]},
 {"name":"class_variants","op":"reshape","inputs":["evidence"],"shape":[1,1,T]},
 {"name":"best_variant","op":"maxpool","inputs":["class_variants"],"size":[1,V],"stride":[1,V]},
 {"name":"scores","op":"reshape","inputs":["best_variant"],"shape":[43]}
]
fw=np.array([[.299,.587,.114],[1.,-.5,-.5],[-.5,-.5,1.]],np.float32)
weights={"features.W":fw.reshape(3,3,1,1),"features.b":np.zeros(3,np.float32),
         "grey.W":fw[:1].reshape(1,3,1,1),"grey.b":np.zeros(1,np.float32),
         "outer_match.W":outer_weights,"outer_match.b":np.zeros(T,np.float32),
         "inner_match.W":inner_weights,"inner_match.b":np.zeros(T,np.float32),
         "fine_match.W":fine_weights,"fine_match.b":np.zeros(T,np.float32)}
outdir=sys.argv[1] if len(sys.argv)>1 else "."
os.makedirs(outdir,exist_ok=True)
with open(os.path.join(outdir,"network.json"),"w") as f: json.dump({"layers":layers},f,indent=2)
np.savez(os.path.join(outdir,"weights.npz"),**weights)
print(f"wrote {T} hand-drawn template pairs and {sum(a.size for a in weights.values())} weights")
