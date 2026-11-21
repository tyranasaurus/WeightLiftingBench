import json, sys
from pathlib import Path
import numpy as np

# Coordinates are fractions of a nominal sign radius, positive y down.
def artwork(c, x, y):
    a = np.zeros((3,*x.shape))
    red=(.85,.055,.045); white=(.92,.92,.88); black=(.065,.065,.065); blue=(.06,.20,.78); yellow=(.97,.78,.06)
    def paint(m,col): a[:,m]=np.asarray(col)[:,None]
    def disk(cx,cy,r,col): paint((x-cx)**2+(y-cy)**2<=r*r,col)
    def box(cx,cy,w,h,col): paint((abs(x-cx)<=w/2)&(abs(y-cy)<=h/2),col)
    def poly(points,col):
        inside=np.zeros(x.shape,bool)
        for (u,v),(s,t) in zip(points,points[1:]+points[:1]):
            if v!=t: inside ^= ((y>v)!=(y>t)) & (x<(s-u)*(y-v)/(t-v)+u)
        paint(inside,col)
    def line(points,w,col=black):
        for (u,v),(s,t) in zip(points,points[1:]):
            f=np.clip(((x-u)*(s-u)+(y-v)*(t-v))/((s-u)**2+(t-v)**2),0,1)
            paint((x-u-f*(s-u))**2+(y-v-f*(t-v))**2<=(w/2)**2,col)
    def arrow(points,w=.16,col=white,h=.29):
        line(points,w,col)
        u,v=points[-1]; s,t=points[-2]; d=np.hypot(u-s,v-t); dx=(u-s)/d; dy=(v-t)/d
        poly([(u+dx*h*.45,v+dy*h*.45),(u-dx*h*.55-dy*h*.65,v-dy*h*.55+dx*h*.65),(u-dx*h*.55+dy*h*.65,v-dy*h*.55-dx*h*.65)],col)
    digs=[[(.50,.04),(.21,.12),(.14,.48),(.20,.88),(.50,.96),(.80,.88),(.86,.50),(.80,.12),(.50,.04)],[(.25,.25),(.53,.05),(.53,.95)],[(.15,.24),(.29,.06),(.68,.06),(.84,.21),(.77,.40),(.17,.89),(.17,.95),(.87,.95)],[(.17,.10),(.81,.10),(.48,.45),(.73,.49),(.85,.67),(.77,.86),(.52,.96),(.22,.88),(.13,.75)],[(.65,.95),(.65,.06),(.13,.66),(.88,.66)],[(.82,.08),(.24,.08),(.20,.47),(.56,.42),(.79,.52),(.84,.72),(.72,.90),(.45,.96),(.17,.85)],[(.75,.09),(.50,.05),(.25,.22),(.15,.58),(.20,.84),(.45,.96),(.72,.88),(.84,.66),(.73,.49),(.47,.46),(.19,.57)],[(.15,.08),(.85,.08),(.52,.54),(.36,.96)],[(.50,.46),(.20,.31),(.23,.13),(.50,.04),(.77,.13),(.80,.31),(.50,.46),(.16,.64),(.19,.85),(.50,.96),(.81,.85),(.84,.64),(.50,.46)],[(.80,.44),(.52,.55),(.24,.47),(.16,.27),(.28,.10),(.55,.05),(.78,.16),(.85,.45),(.76,.76),(.53,.96)]]
    def text(s,cy,height,width,col,stroke):
        step=width/len(s)
        for i,ch in enumerate(s):
            u=-width/2+i*step
            line([(u+p*step,cy+(q-.5)*height) for p,q in digs[int(ch)]],stroke,col)
    if c in [11,13,*range(18,32)]:
        d=-1 if c==13 else 1
        for s,col in [(1,red),(.72,white)]:
            # Scale about centroid of the equilateral triangle.
            poly([(0,d*(-.13-.94*s)),(-.99*s,d*(-.13+.79*s)),(.99*s,d*(-.13+.79*s))],col)
    elif c==12:
        paint(abs(x)+abs(y)<1.24,black); paint(abs(x)+abs(y)<1.20,white); paint(abs(x)+abs(y)<.92,black); paint(abs(x)+abs(y)<.88,yellow)
    elif c==14:
        p=[(np.cos(np.pi/8+i*np.pi/4),np.sin(np.pi/8+i*np.pi/4)) for i in range(8)]
        poly(p,white); poly([(u*.94,v*.94) for u,v in p],red)
        glyph=[[(.19,-.15),(.04,-.19),(-.10,-.12),(-.08,-.03),(.14,.03),(.18,.12),(.07,.19),(-.12,.15)], [(-.15,-.17),(.16,-.17),(0,-.17),(0,.18)], [(.02,-.18),(.16,-.09),(.16,.10),(.02,.19),(-.13,.10),(-.13,-.09),(.02,-.18)], [(-.11,.18),(-.11,-.18),(.11,-.18),(.17,-.10),(.12,0),(-.11,0)]]
        for i,p in enumerate(glyph): line([(u+(i-1.5)*.36,v) for u,v in p],.065,white)
    elif 33<=c<=40: disk(0,0,1,blue)
    elif c in (6,32,41,42): disk(0,0,1,black); disk(0,0,.95,white)
    else:
        disk(0,0,1,red)
        if c!=17: disk(0,0,.76,white)
    if c<=8:
        s=['20','30','50','60','70','80','80','100','120'][c]
        text(s,0,.67 if len(s)==2 else .59,1.06 if len(s)==2 else 1.25,black,.103 if len(s)==2 else .084)
    def car(cx,cy,col,truck=False):
        box(cx,cy+.04,.44,.28,col)
        poly([(cx-.18,cy),(cx-.13,cy-.22),(cx+.13,cy-.22),(cx+.18,cy)],col)
        if not truck: box(cx,cy-.107,.23,.10,white)
        else: box(cx,cy-.105,.39,.23,col)
        for u in [-.15,.15]: box(cx+u,cy+.21,.10,.12,col)
    if c in (9,10,41,42):
        car(-.30,0,red if c<41 else black,c in (10,42)); car(.30,0,black)
    if c==11:
        box(0,.15,.19,.65,black); box(0,.12,.66,.105,black)
    if c==16:
        box(-.16,-.07,.60,.33,black); box(.37,.02,.34,.25,black); box(.37,-.11,.23,.19,black); box(.02,.13,.91,.08,black)
        for u in [-.30,.34]: disk(u,.19,.105,black)
    if c==17: box(0,0,1.49,.34,white)
    if c==18:
        poly([(-.09,-.30),(.09,-.30),(.055,.15),(-.055,.15)],black); disk(0,.30,.077,black)
    if c in (19,20,21):
        d=-1 if c in (19,21) else 1
        p=[(-.17*d,.45),(-.17*d,.21),(-.06*d,.04),(.11*d,-.05),(.24*d,-.06)]
        if c==21: p=[(-.27,.45),(-.05,.31),(.13,.22),(.15,.11),(.04,.01),(-.15,-.05)]
        line(p,.14)
    if c==22: line([(-.49,.35),(-.34,.35),(-.25,.17),(-.13,.13),(0,.32),(.13,.13),(.25,.17),(.34,.35),(.49,.35)],.105)
    if c==23:
        car(0,.01,black)
        for d in [-.17,.17]: line([(d,.20),(d+.09,.28),(d-.05,.38),(d+.01,.48)],.055)
    if c==24:
        line([(-.20,-.22),(-.20,.44)],.085); line([(.15,-.22),(.15,.02),(.27,.19),(.27,.44)],.085)
    def person(u,v,h,d=1):
        disk(u+d*.08*h,v-.78*h,.105*h,black)
        line([(u+d*.044*h,v-.63*h),(u,v-.34*h)],.14*h)
        for z in [-1,1]:
            line([(u,v-.36*h),(u+z*.13*h,v-.19*h),(u+z*.25*h,v)],.085*h)
            line([(u,v-.58*h),(u+z*.13*h,v-.43*h),(u+z*.24*h,v-.48*h)],.065*h)
    if c==25:
        disk(-.04,-.19,.072,black); line([(-.09,-.08),(-.20,.08),(-.13,.25),(-.29,.43)],.10); line([(-.19,.10),(-.30,.24),(-.42,.41)],.085)
        line([(-.09,-.08),(.02,.09),(.20,.12)],.075); line([(-.02,.04),(.35,.36)],.045)
        poly([(.01,.44),(.17,.26),(.29,.22),(.51,.44)],black)
    if c==26:
        box(0,.09,.23,.67,black)
        for v,col in [(-.13,red),(.09,yellow),(.30,(.09,.68,.19))]: disk(0,v,.082,col)
    if c==27: person(0,.43,.79,-1)
    if c==28: person(-.19,.43,.78,-1); person(.20,.43,.57,-1); line([(-.06,.09),(.13,.17)],.055)
    if c==29:
        for u in [-.28,.28]: disk(u,.29,.158,black); disk(u,.29,.111,white)
        line([(-.28,.29),(-.09,.05),(.17,.05),(.01,.29),(-.28,.29)],.045); line([(.28,.29),(.12,-.035),(.23,-.055)],.046); line([(-.15,.015),(-.04,.015)],.052)
    if c==30:
        for t in np.arange(6)*np.pi/3:
            u,v=np.sin(t),np.cos(t); line([(0,.14),(.31*u,.14+.31*v)],.041)
            for d in [-1,1]: line([(.17*u,.14+.17*v),(.245*u+d*.09*v,.14+.245*v-d*.09*u)],.034)
    if c==31:
        line([(-.33,.15),(-.11,.11),(.09,.13),(.24,-.10),(.39,-.14)],.115)
        for p in [[(-.26,.14),(-.33,.28),(-.46,.35)],[(-.16,.15),(-.08,.32),(-.21,.43)],[(.08,.17),(.20,.28),(.43,.36)],[(.14,.16),(.30,.21),(.46,.24)],[(.24,-.11),(.17,-.23),(.22,-.33)],[(.27,-.11),(.32,-.24),(.29,-.34)]]: line(p,.049)
    if c in (6,32,41,42):
        for v in np.arange(-.18,.20,.09): paint((abs(y+.92*x-v)<.025)&(x*x+y*y<.94**2),black)
    if c in (33,34):
        d=1 if c==33 else -1
        arrow([(-.25*d,.51),(-.25*d,-.10),(-.14*d,-.29),(.10*d,-.35),(.50*d,-.35)])
    if c in (35,36,37):
        arrow([(0,.59),(0,-.52)])
        if c!=35:
            d=1 if c==36 else -1; arrow([(0,.18),(.18*d,-.02),(.49*d,-.02)],.145,h=.25)
    if c in (38,39):
        d=1 if c==38 else -1; arrow([(-.44*d,-.46),(.40*d,.44)],.23,h=.38)
    if c==40:
        for t in np.arange(3)*2*np.pi/3:
            arrow([(.50*np.cos(v),.50*np.sin(v)) for v in np.linspace(t,t-.77,8)],.14,h=.26)
    return a


def build(out):
    layers=[]; weights={}
    def layer(name,op,src,**kw):
        layers.append(dict(name=name,op=op,inputs=[src] if isinstance(src,str) else src,**kw)); return name
    def conv(name,src,w,b):
        weights[name+'.W']=np.array(w,np.float32)[:,:,None,None]; weights[name+'.b']=np.array(b,np.float32)
        return layer(name,'conv',src,out=len(w),kernel=[1,1],stride=1,padding=0)
    def dense(name,src,w,b):
        weights[name+'.W']=np.asarray(w,dtype=np.float32); weights[name+'.b']=np.asarray(b,dtype=np.float32)
        return layer(name,'dense',src,out=len(w))
    # Separate luminance and opponent colors. Opponent maps are rectified
    # before spatial normalization, so negative opponent responses do not
    # become evidence for a differently colored sign.
    lum=layer('lum','norm',conv('gray','image',[[.30,.55,.15]],[0]))
    colors=conv('opponents','image',[[1,-.64,-.36],[-.45,-.55,1],[.5,.5,-1]],[-.018,-.022,-.03])
    colors=layer('positive','relu',colors)
    # A biased luminance channel and the independently normalized colors
    # are assembled through a two-input sum with disjoint channel support.
    colors=layer('color_norm','norm',colors)
    cp=conv('color_stack',colors,[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[0]*4)
    lp=conv('luma_stack',lum,[[1],[0],[0],[0]],[0]*4)
    features=layer('features','add',[cp,lp]); features=layer('cells','avgpool',features,size=[2,2],stride=[2,2])
    # Bank of analytic templates with shared nuisance transforms.
    grid=(np.arange(64)+.5)/2-16
    X,Y=np.meshgrid(grid,grid)
    variants=[]
    for r in [9.3,11.2,13.1,14.8]:
        for dx,dy in [(0,0),(-1.65,0),(1.65,0),(0,-1.65),(0,1.65),(-1.3,-1.3),(1.3,-1.3),(-1.3,1.3),(1.3,1.3)]:
            for angle,aspect in [(0,1),(-.13,1),(.13,1),(0,.85),(0,1.17)]:
                variants.append((r,dx,dy,angle,aspect))
    def pool(a): return a.reshape(-1,16,4,16,4).mean((2,4))
    rows=[]; biases=[]
    for c in range(43):
        for r,dx,dy,t,aspect in variants:
            u=(X-dx)/r; v=(Y-dy)/r
            x=(u*np.cos(t)+v*np.sin(t))/aspect; y=-u*np.sin(t)+v*np.cos(t)
            a=artwork(c,x,y); mask=a.max(0)>0
            ring=(~mask)&(abs(x)<1.12)&(abs(y)<1.12)
            # No assumed background hue. Only weak luminance contrast at
            # the silhouette; emphasize contrast within the sign instead.
            R,G,B=a
            sig=[.30*R+.55*G+.15*B,np.maximum(R-.64*G-.36*B-.018,0),np.maximum(B-.45*R-.55*G-.022,0),np.maximum(.5*R+.5*G-B-.03,0)]
            w=[]
            color=1 if c in (6,32,41,42) else 3 if c==12 else 2 if 33<=c<=40 else 1
            for j,z in enumerate(sig):
                if j==0:
                    q=(z-z[mask].mean())*mask
                    if c in (14,17) or 33<=c<=40: q*=1.15
                elif j==color and c not in (6,32,41,42):
                    # Include the immediate outside in the opponent contrast.
                    support=mask|ring
                    q=(z-z[support].mean())*support
                else: q=np.zeros_like(z)
                q=pool(q)[0]; n=np.linalg.norm(q)
                scale=(.77 if c in (6,32,41,42) else .64) if j==0 else .55
                w.append(q*scale/max(n,1e-6)/max(r,10))
            rows.append(np.asarray(w,np.float32).ravel()); biases.append(0.)
    n=len(variants)
    matched=dense('matched','cells',rows,biases)
    shaped=layer('by_class','reshape',matched,shape=[1,43,n]); best=layer('best_view','maxpool',shaped,size=[1,n],stride=[1,n]); layer('scores','reshape',best,shape=[43])
    out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**weights)

if __name__=='__main__': build(Path(sys.argv[1]))
