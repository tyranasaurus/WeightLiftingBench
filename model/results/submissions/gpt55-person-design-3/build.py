import json, sys, os
import numpy as np

YCOEF = np.array([0.299, 0.587, 0.114], np.float32)

def conv_layer(layers, name, inp, out, k, stride=1, padding=0):
    layers.append({"name": name, "op": "conv", "inputs": [inp], "out": out, "kernel": [k, k], "stride": stride, "padding": padding})

def relu_layer(layers, name, inp):
    layers.append({"name": name, "op": "relu", "inputs": [inp]})

def pool_layer(layers, name, op, inp, size, stride):
    layers.append({"name": name, "op": op, "inputs": [inp], "size": size, "stride": stride})

def dense_layer(layers, name, inp, out):
    layers.append({"name": name, "op": "dense", "inputs": [inp], "out": out})

def add_layer(layers, name, inputs):
    layers.append({"name": name, "op": "add", "inputs": inputs})

def make_feature_layers(weights):
    f0_names=[]; W0=[]; b0=[]
    def add(name, center=None, bias=0.0, kernel=None):
        w=np.zeros((3,3,3),np.float32)
        if center is not None: w[:,1,1]=np.asarray(center,np.float32)
        if kernel is not None:
            for c in range(3): w[c]+=kernel*YCOEF[c]
        f0_names.append(name); W0.append(w); b0.append(float(bias))
    y=YCOEF
    add('gray',y); add('invgray',-y,1.0); add('dark',-y,0.50); add('vdark',-y,0.28)
    add('bright',y,-0.62); add('vbright',y,-0.82)
    add('red',[1.0,-0.55,-0.25],-0.05); add('green',[-0.45,1.0,-0.25],-0.06); add('blue',[-0.35,-0.20,1.0],-0.07)
    add('yellow',[0.85,0.85,-1.20],-0.35); add('orange',[1.25,0.55,-1.00],-0.34)
    for nm,coef in [('rg',[1,-1,0]),('gr',[-1,1,0]),('rb',[1,0,-1]),('br',[-1,0,1]),('gb',[0,1,-1]),('bg',[0,-1,1])]: add(nm,coef)
    # Penalties used to make skin maps by a 1x1 ReLU layer.  They encode broad human-skin chromaticity, not data-derived values.
    add('pen_low06',-y,0.06); add('pen_low12',-y,0.12); add('pen_high90',y,-0.90); add('pen_blue65',[0,0,1],-0.65)
    add('pen_gtr',[-1,1,0],-0.05); add('pen_btr',[-1,0,1],0.04); add('pen_btg',[0,-1,1],0.08)
    add('pen_rb_low',[-1,0,1],0.10); add('pen_rg_low',[-1,1,0],0.00); add('pen_rg_high',[1,-1,0],-0.50)
    sx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)*0.25; sy=sx.T
    d1=np.array([[0,1,2],[-1,0,1],[-2,-1,0]],np.float32)*0.25
    d2=np.array([[2,1,0],[1,0,-1],[0,-1,-2]],np.float32)*0.25
    lap=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],np.float32)*0.25
    for nm,k in [('sxp',sx),('sxn',-sx),('syp',sy),('syn',-sy),('d1p',d1),('d1n',-d1),('d2p',d2),('d2n',-d2),('lapp',lap),('lapn',-lap)]: add(nm,kernel=k)
    weights['f0.W']=np.stack(W0).astype(np.float32); weights['f0.b']=np.array(b0,np.float32)
    f0={n:i for i,n in enumerate(f0_names)}
    f1_names=[]; W1=[]; b1=[]
    def add1(name, terms=None, bias=0.0):
        w=np.zeros((len(f0_names),1,1),np.float32)
        if terms:
            for nm,v in terms.items(): w[f0[nm],0,0]+=float(v)
        f1_names.append(name); W1.append(w); b1.append(float(bias))
    for nm in ['gray','invgray','dark','vdark','bright','vbright','red','green','blue','yellow','orange','rg','gr','rb','br','gb','bg','sxp','sxn','syp','syn']:
        add1(nm,{nm:1})
    edge={nm:0.34 for nm in ['sxp','sxn','syp','syn','d1p','d1n','d2p','d2n']}; edge.update({'lapp':0.18,'lapn':0.18})
    add1('edge',edge); add1('vedge',{'sxp':0.65,'sxn':0.65}); add1('hedge',{'syp':0.65,'syn':0.65}); add1('lap',{'lapp':0.55,'lapn':0.55})
    add1('sat',{'rg':0.36,'gr':0.36,'rb':0.36,'br':0.36,'gb':0.25,'bg':0.25})
    add1('warm',{'orange':1.0,'yellow':0.45,'red':0.35}); add1('cool',{'blue':0.7,'green':0.5})
    add1('hair',{'vdark':1.0,'dark':0.35}); add1('white',{'bright':0.8,'vbright':0.8})
    add1('skin',{'pen_low12':-2.2,'pen_high90':-1.7,'pen_blue65':-2.0,'pen_gtr':-1.8,'pen_btr':-2.0,'pen_btg':-1.2,'pen_rb_low':-2.5,'pen_rg_low':-1.2,'pen_rg_high':-1.2},0.36)
    add1('skin_broad',{'pen_low06':-1.4,'pen_high90':-1.15,'pen_blue65':-1.25,'pen_gtr':-1.1,'pen_btr':-1.45,'pen_btg':-0.7,'pen_rb_low':-1.5,'pen_rg_low':-0.45,'pen_rg_high':-0.65,'orange':0.16},0.34)
    add1('skin_dark',{'pen_low06':-1.8,'pen_high90':-1.55,'pen_blue65':-1.45,'pen_gtr':-1.45,'pen_btr':-1.75,'pen_btg':-0.7,'pen_rb_low':-1.9,'pen_rg_high':-0.7,'dark':0.12},0.28)
    add1('foreground',{'sxp':0.12,'sxn':0.12,'syp':0.12,'syn':0.12,'d1p':0.08,'d1n':0.08,'d2p':0.08,'d2n':0.08,'rg':0.12,'gr':0.12,'rb':0.14,'br':0.14,'gb':0.08,'bg':0.08,'dark':0.18,'bright':0.08,'orange':0.08,'blue':0.07,'red':0.07})
    weights['f1.W']=np.stack(W1).astype(np.float32); weights['f1.b']=np.array(b1,np.float32)
    return f1_names

def gmask(k,cx,cy,rx,ry):
    y,x=np.mgrid[0:k,0:k].astype(np.float32)
    return np.exp(-0.5*(((x-cx)/rx)**2+((y-cy)/ry)**2)).astype(np.float32)

def rmask(k,x0,x1,y0,y1,soft=.55):
    y,x=np.mgrid[0:k,0:k].astype(np.float32)
    return (1/(1+np.exp(-(x-x0)/soft))*1/(1+np.exp((x-x1)/soft))*1/(1+np.exp(-(y-y0)/soft))*1/(1+np.exp((y-y1)/soft))).astype(np.float32)

def norm(m):
    s=float(np.sum(np.abs(m)))
    return (m/s).astype(np.float32) if s>1e-6 else m.astype(np.float32)

def put(W,o,fmap,ch,mask,val):
    if ch in fmap: W[o,fmap[ch]] += float(val)*norm(mask)

def make_face_part(k,n,fmap,small=False):
    W=np.zeros((2*n,len(fmap),k,k),np.float32); b=np.zeros(2*n,np.float32)
    for i in range(n):
        cx=(k-1)/2 + ((i%5)-2)*0.04*k
        cy=(0.50+0.04*((i//5)%3-1))*k
        rx=(0.22+0.035*((i//3)%3))*k; ry=(0.29+0.035*((i//7)%3))*k
        face=gmask(k,cx,cy,rx,ry)
        inner=gmask(k,cx,cy,rx*.75,ry*.75)
        ring=np.maximum(gmask(k,cx,cy,rx*1.55,ry*1.45)-gmask(k,cx,cy,rx*.90,ry*.90),0)
        top=rmask(k,cx-rx*1.0,cx+rx*1.0,cy-ry*1.35,cy-ry*.20,.45)
        eyes=gmask(k,cx-rx*.38,cy-ry*.08,rx*.18,ry*.11)+gmask(k,cx+rx*.38,cy-ry*.08,rx*.18,ry*.11)
        mouth=gmask(k,cx,cy+ry*.38,rx*.30,ry*.09)
        outline=np.maximum(gmask(k,cx,cy,rx*1.16,ry*1.14)-gmask(k,cx,cy,rx*.82,ry*.82),0)
        # Part 0: an isolated oval of skin-coloured pixels.
        o=i
        put(W,o,fmap,'skin',inner,1.25); put(W,o,fmap,'skin_broad',face,0.88); put(W,o,fmap,'skin_dark',face,0.48)
        put(W,o,fmap,'warm',face,0.08); put(W,o,fmap,'skin_broad',ring,-0.58); put(W,o,fmap,'warm',ring,-0.20); put(W,o,fmap,'sat',ring,-0.08)
        put(W,o,fmap,'green',face,-0.16); put(W,o,fmap,'cool',face,-0.12)
        b[o] = -0.55 if not small else -0.48
        # Part 1: darker hair/eyes/mouth and an edge outline in the right places.
        o=n+i
        put(W,o,fmap,'hair',top,0.86); put(W,o,fmap,'dark',top,0.36)
        put(W,o,fmap,'dark',eyes,1.12); put(W,o,fmap,'edge',eyes,0.54); put(W,o,fmap,'dark',mouth,0.30)
        put(W,o,fmap,'edge',outline,0.44); put(W,o,fmap,'vedge',outline,0.14); put(W,o,fmap,'hedge',outline,0.10)
        put(W,o,fmap,'sat',face,-0.08); put(W,o,fmap,'green',face,-0.05)
        b[o] = -0.24 if not small else -0.20
    return W,b

def make_combine_face(n, skin_gain=1.0, detail_gain=1.45, bias=-0.32):
    W=np.zeros((n,2*n,1,1),np.float32); b=np.full(n,bias,np.float32)
    for i in range(n): W[i,i,0,0]=skin_gain; W[i,n+i,0,0]=detail_gain
    return W,b

def make_upper_part(k,n,fmap):
    W=np.zeros((2*n,len(fmap),k,k),np.float32); b=np.zeros(2*n,np.float32)
    for i in range(n):
        cx=(k-1)/2 + ((i%5)-2)*0.04*k
        hy=(0.22+0.035*((i//5)%3))*k
        head=gmask(k,cx,hy,0.12*k,0.14*k); hair=gmask(k,cx,hy-0.06*k,0.15*k,0.08*k)
        torso=rmask(k,cx-(0.20+0.03*(i%3))*k,cx+(0.20+0.03*(i%3))*k,0.40*k,0.86*k,.65)
        shoulder=gmask(k,cx,0.43*k,0.32*k,0.08*k)
        side=rmask(k,0,0.17*k,0.25*k,0.95*k,.65)+rmask(k,0.83*k,k-1,0.25*k,0.95*k,.65)
        o=i
        put(W,o,fmap,'skin',head,0.95); put(W,o,fmap,'skin_broad',head,0.70); put(W,o,fmap,'skin_dark',head,0.45)
        put(W,o,fmap,'hair',hair,0.45); put(W,o,fmap,'dark',hair,0.16); put(W,o,fmap,'edge',head,0.12)
        b[o]=-0.36
        o=n+i
        put(W,o,fmap,'foreground',torso,1.05); put(W,o,fmap,'edge',torso,0.50); put(W,o,fmap,'vedge',torso,0.22); put(W,o,fmap,'sat',torso,0.28)
        put(W,o,fmap,'dark',torso,0.20); put(W,o,fmap,'bright',torso,0.07); put(W,o,fmap,'blue',torso,0.06); put(W,o,fmap,'red',torso,0.06)
        put(W,o,fmap,'foreground',shoulder,0.38); put(W,o,fmap,'foreground',side,-0.40); put(W,o,fmap,'edge',side,-0.17); put(W,o,fmap,'sat',side,-0.10)
        b[o]=-0.38
    return W,b

def make_combine_two(n, gain1=1.15, gain2=1.0, bias=-0.24):
    W=np.zeros((n,2*n,1,1),np.float32); b=np.full(n,bias,np.float32)
    for i in range(n): W[i,i,0,0]=gain1; W[i,n+i,0,0]=gain2
    return W,b

def make_body_part(k,n,fmap,tall=False):
    W=np.zeros((2*n,len(fmap),k,k),np.float32); b=np.zeros(2*n,np.float32)
    for i in range(n):
        cx=(k-1)/2+((i%5)-2)*0.035*k; slim=0.13+0.03*((i//3)%4)
        head=gmask(k,cx,0.16*k,0.09*k,0.10*k); head2=gmask(k,cx,0.17*k,0.12*k,0.13*k)
        torso=rmask(k,cx-slim*k,cx+slim*k,0.28*k,0.58*k,.55)
        if tall:
            legs=rmask(k,cx-0.14*k,cx-0.02*k,0.55*k,0.95*k,.45)+rmask(k,cx+0.02*k,cx+0.14*k,0.55*k,0.95*k,.45)
        else:
            legs=rmask(k,cx-0.15*k,cx+0.15*k,0.58*k,0.93*k,.55)
        body=np.maximum(torso,legs); side=rmask(k,0,0.18*k,0.12*k,0.96*k,.65)+rmask(k,0.82*k,k-1,0.12*k,0.96*k,.65)
        o=i
        put(W,o,fmap,'skin',head,0.55); put(W,o,fmap,'skin_broad',head2,0.42); put(W,o,fmap,'skin_dark',head2,0.30); put(W,o,fmap,'hair',head,0.22); put(W,o,fmap,'edge',head2,0.12)
        b[o]=-0.20
        o=n+i
        put(W,o,fmap,'foreground',body,1.05); put(W,o,fmap,'edge',body,0.64); put(W,o,fmap,'vedge',body,0.46); put(W,o,fmap,'hedge',torso,0.12)
        put(W,o,fmap,'dark',body,0.26); put(W,o,fmap,'sat',torso,0.22); put(W,o,fmap,'blue',torso,0.05); put(W,o,fmap,'red',torso,0.05)
        put(W,o,fmap,'foreground',side,-0.50); put(W,o,fmap,'edge',side,-0.22); put(W,o,fmap,'sat',side,-0.12)
        b[o]=-0.36 if tall else -0.34
    return W,b

def dense_positive(weights,name,n,scale,bias=0.0):
    W=np.zeros((2,n),np.float32); W[1,:]=scale
    weights[name+'.W']=W; weights[name+'.b']=np.array([0.0,bias],np.float32)

def add_detector(layers,weights,base,input_layer,k,n,fmap,part_fn,comb_fn,scale):
    W,b=part_fn(k,n,fmap); weights[base+'p.W']=W; weights[base+'p.b']=b
    conv_layer(layers,base+'p',input_layer,W.shape[0],k,1,0); relu_layer(layers,base+'pr',base+'p')
    Wc,bc=comb_fn(n); weights[base+'c.W']=Wc; weights[base+'c.b']=bc
    conv_layer(layers,base+'c',base+'pr',n,1,1,0); relu_layer(layers,base+'cr',base+'c')
    h=48-k+1 if input_layer=='f1r' else 24-k+1
    pool_layer(layers,base+'m','maxpool',base+'cr',[h,h],[h,h])
    dense_layer(layers,base+'s',base+'m',2); dense_positive(weights,base+'s',n,scale,0.0)
    return base+'s'

def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    layers=[]; weights={}
    pool_layer(layers,'p2','avgpool','image',[2,2],[2,2])
    fmap_names=make_feature_layers(weights); fmap={n:i for i,n in enumerate(fmap_names)}
    conv_layer(layers,'f0','p2',weights['f0.b'].shape[0],3,1,1); relu_layer(layers,'f0r','f0')
    conv_layer(layers,'f1','f0r',len(fmap_names),1,1,0); relu_layer(layers,'f1r','f1')
    pool_layer(layers,'f24','avgpool','f1r',[2,2],[2,2])
    # A weak prior/global skin summary.  It is deliberately conservative; local part conjunctions do most of the work.
    pool_layer(layers,'gpool','avgpool','f1r',[48,48],[48,48]); dense_layer(layers,'gscore','gpool',2)
    Wg=np.zeros((2,len(fmap_names)),np.float32)
    for nm,val in [('skin',0.65),('skin_broad',0.32),('skin_dark',0.28),('foreground',0.10),('edge',0.06),('warm',0.05),('sat',-0.05),('green',-0.06),('cool',-0.05)]:
        if nm in fmap: Wg[1,fmap[nm]]=val
    weights['gscore.W']=Wg; weights['gscore.b']=np.array([0.0,-5.00],np.float32)
    scores=['gscore']
    # Face detectors at the 24x24 feature resolution, for small and medium visible faces.
    scores.append(add_detector(layers,weights,'face3','f24',3,18,fmap,lambda k,n,f: make_face_part(k,n,f,True),lambda n: make_combine_face(n,1.0,1.70,-0.28),0.34))
    scores.append(add_detector(layers,weights,'face5','f24',5,20,fmap,lambda k,n,f: make_face_part(k,n,f,True),lambda n: make_combine_face(n,1.0,1.60,-0.31),0.36))
    scores.append(add_detector(layers,weights,'face7','f24',7,18,fmap,lambda k,n,f: make_face_part(k,n,f,False),lambda n: make_combine_face(n,1.0,1.50,-0.33),0.32))
    # Upper-body and full-body templates at 24x24 feature resolution.
    scores.append(add_detector(layers,weights,'upper9','f24',9,24,fmap,make_upper_part,lambda n: make_combine_two(n,1.25,1.00,-0.28),0.28))
    scores.append(add_detector(layers,weights,'upper13','f24',13,20,fmap,make_upper_part,lambda n: make_combine_two(n,1.20,1.00,-0.28),0.26))
    scores.append(add_detector(layers,weights,'body13','f24',13,22,fmap,lambda k,n,f: make_body_part(k,n,f,False),lambda n: make_combine_two(n,1.05,1.00,-0.24),0.20))
    scores.append(add_detector(layers,weights,'tall17','f24',17,20,fmap,lambda k,n,f: make_body_part(k,n,f,True),lambda n: make_combine_two(n,1.05,1.00,-0.24),0.20))
    add_layer(layers,'score',scores)
    with open(os.path.join(outdir,'network.json'),'w') as f: json.dump({"layers":layers},f,separators=(',',':'))
    np.savez(os.path.join(outdir,'weights.npz'), **{k:v.astype(np.float32) for k,v in weights.items()})

if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '/app/out')
