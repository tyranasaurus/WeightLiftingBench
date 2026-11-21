"""Data-free analytic person detector."""
import json
import sys
from pathlib import Path
import numpy as np


def build(dest):
    layers, weights = [], {}

    def layer(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw))
        return name

    def conv(name, src, w, b, stride=1):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        k = w.shape[-1]
        weights[name+'.W'] = w
        weights[name+'.b'] = np.asarray(b, np.float32)
        return layer(name, 'conv', [src], out=len(w), kernel=[k,k], stride=stride, padding=k//2)

    def relu(name, src):
        return layer(name, 'relu', [src])

    # Soft warm-chromaticity membership. A union of tones is approximated
    # by allowing a broad luminance-dependent band in opponent colors.
    # Last two rows split luminance's centered local detail later on.
    rgb = layer('rgb', 'avgpool', ['image'], size=[2,2], stride=[2,2])
    y = np.array([.299,.587,.114]); rg = np.array([1.,-1.,0.]); gb = np.array([0.,1.,-1.])
    constraints = [-rg+.045*y, rg-.42*y, -gb+.018*y, gb-.38*y,
                   gb-1.65*rg, -y, y, y]
    c = relu('violations', conv('color_constraints',rgb,constraints,[.012,-.04,.004,-.045,-.025,.09,-.98,0]))
    # Two nested memberships: broad evidence and a stricter warm-color cue.
    base = relu('tone_maps',conv('membership',c,[[-19,-12,-14,-9,-10,-13,-8,0],
                           [-35,-22,-23,-18,-16,-19,-15,0],
                           [0,0,0,0,0,0,0,1]], [1,.94,0]))
    # Smooth tone plus clipped dark and light local contrast. Local darkness
    # is relative to the surrounding patch, not an absolute RGB threshold.
    w = np.zeros((5,3,5,5))
    w[0,0,2,2]=1; w[1,1,:,:]=1/25; w[2,2,2,2]=1
    w[3,2,:,:]=1/25; w[3,2,2,2]-=1; w[4]=-w[3]
    d = relu('detail',conv('local_contrast',base,w,[0,0,0,-.012,-.012]))
    src = relu('evidence',conv('scaled_evidence',d,np.diag([1,1,1,3.7,2.6]),[0]*5))

    # Each view models a face in normalized coordinates. The eyes and brow
    # are darker than the cheeks; skin support should end around the face.
    # Average masks make responses comparable across scale.
    pooled = []
    for k in (7,11,17,25,35):
        yy,xx=np.mgrid[-k//2+1:k//2+1,-k//2+1:k//2+1]/(k//2)
        def g(x,y,sx,sy):
            a=np.exp(-.5*((xx-x)**2/sx**2+(yy-y)**2/sy**2))
            return a/a.sum()
        ws=[]; bs=[]
        for turn in (-1,0,1):
            for shape in (.86,1.13):
                a=np.zeros((5,k,k))
                cheek=g(.12*turn,.09,.25*shape,.27)
                forehead=g(.02*turn,-.36,.23*shape,.12)
                chin=g(.08*turn,.43,.18*shape,.13)
                eyes=(g(-.23*shape+.08*turn,-.16,.088,.08)+g(.23*shape+.08*turn,-.16,.088,.08))/2
                mouth=g(.14*turn,.28,.14,.075)
                ring=(g(-.63*shape,0,.12,.33)+g(.63*shape,0,.12,.33)+g(0,-.75,.32,.12))/3
                a[0]=.43*cheek+.28*forehead+.14*chin-.42*ring-.10*eyes
                a[1]=.24*cheek+.15*forehead
                a[2]=.40*cheek+.27*forehead-.49*eyes-.18*mouth
                a[3]=.49*eyes+.18*mouth-.32*cheek-.21*forehead
                a[4]=.16*cheek-.10*eyes
                ws.append(a); bs.append(-.39)
        face=relu(f'face{k}',conv(f'face_geometry{k}',src,ws,bs,stride=2))
        pooled.append(layer(f'peak{k}','maxpool',[face],size=[24,24],stride=[24,24]))
    # Summarize the prevalence of coherent skin independent of whether
    # a face is visible. Cap its contribution to avoid uniform warm scenes.
    coh=relu('coherent_tone',conv('tone_support',src,[[.45,.55,0,0,0]],[-.54]))
    area=layer('tone_area','avgpool',[coh],size=[48,48],stride=[48,48])
    # Reduce each group of six face views by a maximum, not by a sum.
    scalars=[]
    for i,p in enumerate(pooled):
        row=layer(f'views{i}','reshape',[p],shape=[1,1,6])
        scalars.append(layer(f'best_view{i}','maxpool',[row],size=[1,6],stride=[1,6]))
    # Pairwise maximum via max(a,b) = b + relu(a-b).
    peak=scalars[0]
    for i,s in enumerate(scalars[1:]):
        neg=conv(f'neg{i}',s,[[-1]],[0])
        delta=layer(f'delta{i}','add',[peak,neg])
        peak=layer(f'best_scale{i}','add',[s,relu(f'positive{i}',delta)])
    area1=conv('area_gain',area,[[2.4]],[0])
    area2=relu('excess_area',conv('area_hinge',area1,[[1]],[-.16]))
    area3=conv('cap_subtraction',area2,[[-1]],[0])
    score=layer('person_evidence','add',[peak,area1,area3])
    weights['scores.W']=np.array([[0],[1]],np.float32)
    weights['scores.b']=np.array([0,-.235],np.float32)
    layer('scores','dense',[score],out=2)
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(dest/'weights.npz',**weights)


if __name__ == '__main__':
    build(Path(sys.argv[1]))
