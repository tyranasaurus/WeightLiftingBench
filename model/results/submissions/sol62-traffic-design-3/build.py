import sys,json
from pathlib import Path
import numpy as np

L=[]; W={}
def layer(op,ins,**kw):
    n='l'+str(len(L)); L.append(dict(name=n,op=op,inputs=ins,**kw)); return n
def linear(op,x,w,b,**kw):
    n=layer(op,[x],out=len(w),**kw); W[n+'.W']=np.asarray(w,np.float32); W[n+'.b']=np.asarray(b,np.float32); return n

def draw(k,x,y):
    r=np.hypot(x,y); a=np.arctan2(y,x)
    def line(points,w):
        z=np.zeros(x.shape,bool)
        for (a,b),(c,d) in zip(points,points[1:]):
            t=np.clip(((x-a)*(c-a)+(y-b)*(d-b))/((c-a)**2+(d-b)**2),0,1)
            z|=(x-a-t*(c-a))**2+(y-b-t*(d-b))**2<w*w
        return z
    def poly(p):
        z=np.zeros(x.shape,bool)
        for (a,b),(c,d) in zip(p,p[1:]+p[:1]):
            if b!=d: z^=((y>b)!=(y>d)) & (x<(c-a)*(y-b)/(d-b)+a)
        return z
    def disk(a,b,s): return (x-a)**2+(y-b)**2<s*s
    def arrow(p,w=.095):
        (a,b),(c,d)=p[-2:]; s=np.hypot(c-a,d-b); u,v=(c-a)/s,(d-b)/s
        return line(p,w)|poly([(c+.13*u,d+.13*v),(c-.17*u-.22*v,d-.17*v+.22*u),(c-.17*u+.22*v,d-.17*v-.22*u)])
    tri= k==11 or 18<=k<=31 or k==13
    if tri:
        yy=-y if k==13 else y
        sh=(yy<.72)&(yy> -1)&(abs(x)<.58*(yy+1))
        inner=(yy<.49)&(abs(x)<.58*(yy+.58))
    elif k==12:
        sh=abs(x)+abs(y)<1.17; inner=sh
    elif k==14:
        sh=r*np.cos((a+np.pi/8)%(np.pi/4)-np.pi/8)<.89; inner=np.zeros_like(sh)
    else:
        sh=r<1; inner=r<.76
    red=sh&~inner; blue=np.zeros_like(sh); ink=np.zeros_like(sh); white=np.zeros_like(sh)
    if k in [6,32,41,42]: red[:]=False
    if k in range(33,41): blue=sh.copy(); red[:]=False
    def digits(text,cy=0,h=.69):
        # Stroke skeletons for upright road-sign numerals.
        paths={
        '0':[[(-.11,-.46),(.11,-.46),(.21,-.32),(.21,.32),(.11,.46),(-.11,.46),(-.21,.32),(-.21,-.32),(-.11,-.46)]],
        '1':[[(-.15,-.29),(0,-.46),(0,.46)]],
        '2':[[(-.20,-.33),(-.11,-.46),(.12,-.46),(.21,-.32),(.19,-.16),(-.20,.35),(-.20,.46),(.22,.46)]],
        '3':[[(-.18,-.43),(.19,-.43),(.01,-.07),(.19,.01),(.21,.28),(.1,.46),(-.12,.46),(-.22,.33)]],
        '5':[[(.19,-.46),(-.17,-.46),(-.19,-.02),(.1,-.07),(.21,.07),(.21,.31),(.10,.46),(-.12,.46),(-.21,.36)]],
        '6':[[(.14,-.45),(-.02,-.42),(-.18,-.17),(-.21,.27),(-.11,.45),(.11,.45),(.21,.29),(.2,.09),(.09,-.04),(-.10,-.04),(-.20,.08)]],
        '7':[[(-.22,-.46),(.22,-.46),(-.11,.46)]],
        '8':[[(-.10,0),(-.20,-.16),(-.19,-.34),(-.09,-.46),(.09,-.46),(.19,-.34),(.20,-.16),(.10,0),(-.14,.09),(-.22,.23),(-.19,.36),(-.09,.46),(.09,.46),(.19,.36),(.22,.23),(.14,.09),(-.10,0)]]}
        z=np.zeros_like(sh)
        for i,ch in enumerate(text):
            cx=(i-(len(text)-1)/2)*h*.65
            for p in paths[ch]: z|=line([(cx+h*u,cy+h*v) for u,v in p],.075*h)
        return z
    if k<=8: ink=digits(['20','30','50','60','70','80','80','100','120'][k],h=.75 if k<7 else .65)
    def car(c):
        return poly([(c-.23,.21),(c-.23,-.05),(c-.15,-.23),(c+.15,-.23),(c+.23,-.05),(c+.23,.21),(c+.12,.21),(c+.12,.12),(c-.12,.12),(c-.12,.21)]) & ~((abs(x-c)<.14)&(y<-.04)&(y>-.17))
    if k in [9,10,41,42]:
        left=car(-.29) if k in [9,41] else ((abs(x+.29)<.20)&(abs(y+.065)<.24))|disk(-.42,.19,.08)|disk(-.16,.19,.08)
        ink=car(.29)
        if k<41: red|=left
        else: ink|=left
    if k in [6,32,41,42]:
        ink|=abs(y+x)<(.15 if k==32 else .095)
        if k!=32: ink|=abs(y+x-.16)<.033
    if k==11: ink=arrow([(0,.46),(0,-.26)],.105)|line([(-.25,.14),(.25,.14)],.063)
    if k==12: ink=(abs(x)+abs(y)<.90)&(abs(x)+abs(y)>.84)
    if k==14:
        for c,p in [(-.46,[(.12,-.14),(-.08,-.14),(-.10,0),(.10,0),(.10,.14),(-.12,.14)]),(-.15,[(-.12,-.14),(.12,-.14),(0,-.14),(0,.14)]),(.15,[(-.10,-.14),(.10,-.14),(.10,.14),(-.10,.14),(-.10,-.14)]),(.45,[(-.10,.14),(-.10,-.14),(.10,-.14),(.10,0),(-.10,0)])]:
            white|=line([(c+u,v) for u,v in p],.034)
        red&=~white
    if k==16:
        ink=((abs(x+.14)<.39)&(abs(y+.10)<.23))|((x>.26)&(x<.51)&(abs(y)<.14))|disk(-.32,.18,.12)|disk(.34,.18,.12)
    if k==17: white=(abs(x)<.73)&(abs(y)<.20); red=sh&~white
    if k==18: ink=line([(0,-.27),(0,.09)],.072)|disk(0,.31,.075)
    if k in [19,20,21]:
        d=-1 if k==19 else 1
        p=[(-.14*d,.47),(-.14*d,.11),(-.02*d,-.08),(.24*d,-.19)]
        if k==21: p=[(-.15,.47),(.17,.27),(.18,.13),(-.13,-.02),(-.20,-.19)]
        ink=arrow(p,.075)
    if k==22:
        ink=line([(-.38,.34),(-.27,.32),(-.17,.13),(-.07,.13),(0,.31),(.07,.31),(.17,.13),(.27,.13),(.38,.34)],.063)
    if k==23: ink=car(0)&(y<.11); ink|=line([(-.15,.10),(-.25,.24),(-.11,.37),(-.18,.49)],.032)|line([(.12,.10),(.02,.24),(.16,.37),(.09,.49)],.032)
    if k==24: ink=line([(-.19,.43),(-.19,-.22)],.065)|line([(.24,.43),(.24,.14),(.09,-.06),(.09,-.22)],.065)
    def walker(cx,cy,s,d=1):
        def p(v): return [(cx+d*s*u,cy+s*w) for u,w in v]
        return disk(cx+.05*s*d,cy-.30*s,.093*s)|line(p([(.03,-.14),(-.03,.10),(-.20,.35)]),.066*s)|line(p([(-.03,.08),(.16,.34)]),.054*s)|line(p([(-.19,.02),(.02,-.13),(.17,.04)]),.047*s)
    if k==25: ink=walker(-.09,.15,.92)|line([(-.03,.11),(.30,.40)],.039)|poly([(.07,.46),(.25,.27),(.40,.46)])
    if k==26:
        lamps=[disk(0,b,.065) for b in [-.17,.05,.27]]
        ink=(abs(x)<.124)&(abs(y-.05)<.35)&~(lamps[0]|lamps[1]|lamps[2]); red|=lamps[0]
    if k==27: ink=walker(0,.12,1)
    if k==28: ink=walker(-.13,.13,.84,-1)|walker(.16,.24,.62,-1)
    if k==29:
        ink=walker(.04,.02,.70)|line([(-.24,.32),(-.05,.12),(.10,.31),(-.24,.32)],.033)
        for c in [-.25,.25]: ink|=abs(np.hypot(x-c,y-.32)-.127)<.035
    if k==30:
        for t in [0,np.pi/3,2*np.pi/3]:
            u,v=.31*np.cos(t),.31*np.sin(t); ink|=line([(-u,.12-v),(u,.12+v)],.035)
    if k==31:
        ink=poly([(-.34,.21),(-.27,.06),(.04,.08),(.18,-.15),(.33,-.14),(.35,-.06),(.21,0),(.12,.21),(-.13,.26)])|line([(-.22,.15),(-.25,.31),(-.10,.43)],.046)|line([(.08,.15),(.17,.27),(.33,.32)],.044)|line([(.22,-.11),(.13,-.32),(.04,-.35)],.035)
    if k in [33,34]:
        d=1 if k==33 else -1; white=arrow([(-.27*d,.52),(-.27*d,-.11),(-.15*d,-.29),(.44*d,-.29)],.115)
    if k==35: white=arrow([(0,.57),(0,-.49)],.125)
    if k in [36,37]:
        d=1 if k==36 else -1; white=arrow([(-.16*d,.60),(-.16*d,-.52)],.101)|arrow([(-.16*d,.17),(.18*d,.06),(.49*d,.06)],.092)
    if k in [38,39]:
        d=1 if k==38 else -1; white=arrow([(-.45*d,-.46),(.40*d,.39)],.115)
    if k==40:
        for t in [-np.pi/2,np.pi/6,5*np.pi/6]: white|=arrow([(.52*np.cos(q),.52*np.sin(q)) for q in np.linspace(t,t+.99,6)],.08)
    blue&=~white
    lum=np.where(red|blue,-.34,.80); lum=np.where(ink,-1.,lum); lum=np.where(white,1.,lum)
    if k in [6,41,42]: lum=np.where(ink,-.50,lum)
    if k==12: lum=np.where(abs(x)+abs(y)<.84,.47,lum)
    return np.stack([red&sh,blue&sh,lum*sh,sh]).astype(float)

# Contrast channels are normalized independently after anti-alias pooling.
f=linear('conv','image',np.array([[1,-.57,-.43],[-.50,-.50,1],[.299,.587,.114]])[:,:,None,None],[0,0,0],kernel=[1,1],stride=1,padding=0)
f=layer('avgpool',[f],size=[2,2],stride=[2,2])
hp=np.zeros((4,3,5,5)); hp[:3,:,2,2]=np.eye(3); hp[3,2]=-1/25; hp[3,2,2,2]+=1
f=linear('conv',f,hp,[0]*4,kernel=[5,5],stride=1,padding=2)
f=layer('norm',[f]);
# Clip unusual contrast spikes, preserving the sign of each channel.
v=linear('conv',f,np.concatenate([np.eye(4),np.eye(4)])[:,:,None,None],[2.5]*4+[-2.5]*4,kernel=[1,1],stride=1,padding=0)
v=layer('relu',[v]); v=linear('conv',v,np.concatenate([np.eye(4),-np.eye(4)],axis=1)[:,:,None,None],[-2.5]*4,kernel=[1,1],stride=1,padding=0)

def template(k,s,dx,dy,aspect,angle):
    y,x=np.mgrid[:48,:48]; x=(x+.5-24)/1.5-dx; y=(y+.5-24)/1.5-dy
    c,t=np.cos(angle),np.sin(angle)
    z=draw(k,(x*c+y*t)/(s*aspect),(y*c-x*t)/s)
    z=z.reshape(4,16,3,16,3).mean((2,4)); red,blue,lum,mask=z
    def center(q): return q-mask*q.sum()/max(mask.sum(),1)
    # Inside-sign patterns plus a weaker global color/silhouette term.
    rr=center(red)+.32*(red-red.mean()); bb=center(blue)+.46*(blue-blue.mean()); ll=center(lum)
    if k==15 or k==13: ll*=.75
    if k==14 or k==17: rr=.6*center(red)+.63*(red-red.mean())
    if k==12:
        yellow=np.maximum(mask-(lum>.60)*mask,0); rr=.30*center(yellow)+.45*(yellow-yellow.mean())
        bb=-rr
    if 33<=k<=40: rr*=0; ll*=1.0; bb*=1.55
    elif k not in (6,12,32,41,42): rr*=1.40; bb=-.25*rr
    else: bb*=0
    # Separate local symbol evidence from low-frequency color and shape.
    raw=np.where(mask>0,lum+(1-mask)*.25,.25)
    win=np.lib.stride_tricks.sliding_window_view(np.pad(raw,2,mode='edge'),(5,5))
    detail=(raw-win.mean((-2,-1)))*np.clip((mask-.35)*2,0,1)
    if k<=8 or k in (9,10,16,32,41,42):
        rad=np.hypot((np.arange(16)[None,:]*2-15-dx)/aspect,np.arange(16)[:,None]*2-15-dy)/s
        detail*=np.clip((.79-rad)*9,0,1)
    if k in (13,15): detail*=0
    def unit(q): return q/max(np.linalg.norm(q),.35)
    color=np.stack([rr,bb]); color/=max(np.linalg.norm(color),.35)
    if k in (6,32,41,42): color*=0
    symbol=.76 if k not in (12,13,15) else 0
    if k in (14,17): symbol=1.05
    shape=.32 if k not in (12,13,15) else .46
    if k==15: color*=.84
    bank=np.concatenate([.85*color,(shape*unit(ll))[None],(symbol*unit(detail))[None]])
    return bank.ravel()/16

bank=[]
for k in range(43):
    for s in [8,9.5,11,12.5,14,15.5]:
        for dx,dy in [(0,0),(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,-2),(-2,2),(2,2)]:
            for aspect,ang in [(1,0),(.84,0),(1.16,0),(1,-.13),(1,.13)]:
                bank.append(template(k,s,dx,dy,aspect,ang))
n=len(bank)//43
v=linear('dense',v,bank,np.zeros(len(bank)))
v=layer('reshape',[v],shape=[43,1,n]); v=layer('maxpool',[v],size=[1,n],stride=[1,n]); v=layer('reshape',[v],shape=[43]);
p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True)
(p/'network.json').write_text(json.dumps({'layers':L},indent=2)+'\n'); np.savez(p/'weights.npz',**W)
