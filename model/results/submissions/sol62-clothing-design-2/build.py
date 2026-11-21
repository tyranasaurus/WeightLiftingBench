"""Compile a bank of invented garment models into a feed-forward graph."""
import json
import sys
from pathlib import Path
import numpy as np


def build(folder):
    layers, weights = [], {}
    def layer(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw))
        return name
    def linear(name, op, src, w, b, **kw):
        weights[name+'.W'] = np.asarray(w, dtype=np.float32)
        weights[name+'.b'] = np.asarray(b, dtype=np.float32)
        return layer(name, op, [src], out=len(b), **kw)
    rng = np.random.default_rng(7942)
    u = rng.uniform
    yy, xx = np.mgrid[0:28,0:28].astype(float)
    xx = (xx-13.5)/13.5
    yy = (yy-13.5)/13.5
    def polygon(x,y,points):
        inside = np.zeros(x.shape, bool)
        dist = np.full(x.shape, 10.)
        for (a,b),(c,d) in zip(points,points[1:]+points[:1]):
            if d != b:
                inside ^= ((b>y)!=(d>y)) & (x < a+(y-b)*(c-a)/(d-b))
            t=np.clip(((x-a)*(c-a)+(y-b)*(d-b))/((c-a)**2+(d-b)**2+1e-12),0,1)
            dist=np.minimum(dist,np.hypot(x-a-t*(c-a),y-b-t*(d-b)))
        return np.clip(.5+np.where(inside,dist,-dist)/.12,0,1)

    def model(k):
        x0=xx-u(-.075,.075); y0=yy-u(-.07,.07)
        a=u(-.11,.11)
        x=x0+a*y0; y=y0-a*x0
        p=lambda points:polygon(x,y,points)
        shade=np.ones(x.shape)
        if k==1:
            w=u(.29,.61); cuff=w*u(.59,1.03); gap=u(.04,.16); crotch=u(-.35,.05)
            mask=p([(-w,-.96),(w,-.96),(w*1.07,-.59),(cuff,.96),(gap,.96),(0,crotch),(-gap,.96),(-cuff,.96),(-w*1.07,-.59)])
            shade-=u(.02,.25)*np.exp(-(x/.14)**2)
        elif k in (5,7,9):
            x=x*rng.choice([-1,1]); bottom=u(.23,.50) if k!=9 else u(.48,.80)
            # Toe is to the left, heel to the right. The boot adds a shaft;
            # the sandal has exposed spaces between straps.
            h=u(.25,.52) if k==7 else u(.60,1.00) if k==9 else u(.15,.42)
            heel=u(.76,.98); toe=-u(.87,1.01); join=u(-.10,.32)
            body=p([(toe,bottom-.08),(toe+.035,bottom-.23),(-.54,bottom-.29),(join-.23,bottom-.47),(join,-h),(heel-.05,-h+u(-.09,.08)),(heel,bottom-.22),(heel-.04,bottom),(toe+.10,bottom+.025)])
            sole=p([(toe,bottom-.15),(heel,bottom-.16),(heel-.04,bottom),(toe+.08,bottom+.025)])
            if k==5:
                straps=np.zeros(x.shape)
                for j in range(rng.integers(2,5)):
                    c=u(-.65,.65); w=u(.055,.17); tilt=u(-.7,.8)
                    straps=np.maximum(straps,np.clip((w-abs(x-c+tilt*(y-bottom)))/.08+.5,0,1))
                body=body*straps
            mask=np.maximum(body,sole)
            shade=.70+.24*sole if k==7 else .90-.20*sole
        elif k==8:
            w=u(.57,.99); top=u(-.67,-.05); bot=u(.58,.95); taper=u(.76,1.08)
            mask=p([(-w+.10,top), (w-.10,top),(w,top+.13),(w*taper,bot-.10),(w*taper-.10,bot),(-w*taper+.10,bot),(-w*taper,bot-.10),(-w,top+.13)])
            if rng.random()<.76:
                hw=w*u(.36,.85); ht=u(.19,.61); t=u(.07,.20)
                r=np.hypot(x/hw,(y-top)/ht)
                handle=np.clip((1-abs(r-1)/t)*2,0,1)*(y<top)
                mask=np.maximum(mask,handle)
            shade-=u(0,.23)*np.exp(-((y-top-.17)/.06)**2)
        else:
            shoulder=u(.33,.58); torso=shoulder*u(.78,1.1)
            top=u(-.98,-.75); bottom=u(.77,.99)
            neck=u(.12,.28); depth=u(.045,.24)
            if k==6: depth=u(.025,.18)
            if k==3: shoulder*=.83; torso*=.75; neck*=1.16
            waist=torso*u(.69,.99) if k in (0,3) else torso*u(.87,1.07)
            hem=torso*u(1.28,2.20) if k==3 else torso*u(.85,1.16)
            if k==4: hem*=1.10
            mask=p([(-shoulder,top+.10),(-neck,top),(0,top+depth),(neck,top),(shoulder,top+.10),(torso,-.24),(waist,.15),(hem,bottom),(-hem,bottom),(-waist,.15),(-torso,-.24)])
            short = k==0 or (k==6 and rng.random()<.40) or (k==3 and rng.random()<.76)
            if not (k==3 and rng.random()<.35):
                for s in [-1,1]:
                    outer=u(.75,1.00) if k!=3 else u(.50,.82)
                    end=u(-.56,-.22) if short else u(.46,.94)
                    width=u(.20,.36) if short else u(.13,.26)
                    mask=np.maximum(mask,p([(s*(shoulder-.12),top+.10),(s*(shoulder+.06),top+.13),(s*outer,end-.14),(s*(outer-width),end),(s*(torso-.08),-.29)]))
            if k in (4,6):
                shade-=u(.08,.37)*np.exp(-(x/u(.035,.12))**2)
            if k in (2,4):
                shade-=u(.08,.23)*np.exp(-((y-bottom+.10)/.09)**2)
        value=mask*np.clip(shade*(1+u(-.23,.23)*x+u(-.26,.26)*y),.12,1.30)
        return mask,value

    # Two saturating foreground channels reduce sensitivity to exposure.
    z=linear('thresholds','conv','image',np.array([1,1,1]).reshape(3,1,1,1),[0,-.12,-.32],kernel=[1,1],stride=1,padding=0)
    z=layer('positive','relu',[z])
    z=linear('foreground','conv',z,np.array([[1,-1,0],[1,0,-1],[1,0,0]]).reshape(3,3,1,1),[0,0,0],kernel=[1,1],stride=1,padding=0)
    z=layer('cells','avgpool',[z],size=[2,2],stride=[2,2])
    z=layer('contrast','norm',[z])
    n=1600
    templates=[]
    for k in range(10):
        for _ in range(n):
            m,v=model(k)
            v=v*u(.26,1.0)
            f=np.stack([np.minimum(v,.12),np.minimum(v,.32),v]).reshape(3,14,2,14,2).mean(axis=(2,4))
            f=(f-f.mean(axis=(1,2),keepdims=True))/np.sqrt(f.var(axis=(1,2),keepdims=True)+1e-5)
            templates.append((f*np.array([.54,.31,.15])[:,None,None]/196).ravel())
    z=linear('shape_similarity','dense',z,templates,np.zeros(10*n))
    z=layer('by_category','reshape',[z],shape=[10,1,n])
    z=layer('best_shape','maxpool',[z],size=[1,n],stride=[1,n])
    layer('scores','reshape',[z],shape=[10])
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(folder/'weights.npz',**weights)

if __name__=='__main__':
    build(Path(sys.argv[1]))
