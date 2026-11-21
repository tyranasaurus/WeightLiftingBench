"""Build a data-free Fashion-MNIST shape-template classifier.

Every convolution/dense kernel below is rendered from general knowledge of the
named garment.  No data files are read.  Each drawing is normalized separately
and used as one matched filter.  The network takes the best match within each
of the ten equally sized families.
"""
import sys, json
from pathlib import Path
import numpy as np

N_PER_CLASS = 120
S = 3                         # supersampling for hand-drawn antialiasing

# Drawing primitives, in final-image pixel coordinates.
def canvas():
    return np.zeros((28*S, 28*S), dtype=np.float64)

def polygon(a, pts, value=1.0):
    yy, xx = np.mgrid[0:28*S, 0:28*S]
    x = (xx + .5) / S; y = (yy + .5) / S
    inside = np.zeros(a.shape, dtype=bool)
    for p, q in zip(pts, pts[1:] + pts[:1]):
        x1,y1=p; x2,y2=q
        if y1 != y2:
            inside ^= ((y1 > y) != (y2 > y)) & (x < (x2-x1)*(y-y1)/(y2-y1)+x1)
    a[inside] = value

def ellipse(a, cx, cy, rx, ry, value=1.0):
    yy, xx = np.mgrid[0:28*S, 0:28*S]
    x=(xx+.5)/S; y=(yy+.5)/S
    mask=((x-cx)/rx)**2+((y-cy)/ry)**2 <= 1
    a[mask]=value

def rect(a,x0,y0,x1,y1,value=1.0):
    polygon(a,[(x0,y0),(x1,y0),(x1,y1),(x0,y1)],value)

def line(a,p,q,width=.45,value=1.0):
    yy,xx=np.mgrid[0:28*S,0:28*S]
    x=(xx+.5)/S; y=(yy+.5)/S
    x1,y1=p; x2,y2=q; dx=x2-x1; dy=y2-y1
    den=dx*dx+dy*dy
    u=0 if den==0 else np.clip(((x-x1)*dx+(y-y1)*dy)/den,0,1)
    d2=(x-x1-u*dx)**2+(y-y1-u*dy)**2
    mask=d2 <= (width/2)**2
    a[mask]=value

def erase_polygon(a, pts):
    yy, xx = np.mgrid[0:28*S, 0:28*S]
    x=(xx+.5)/S; y=(yy+.5)/S
    inside=np.zeros(a.shape,dtype=bool)
    for p,q in zip(pts,pts[1:]+pts[:1]):
        x1,y1=p; x2,y2=q
        if y1 != y2:
            inside ^= ((y1>y)!=(y2>y)) & (x<(x2-x1)*(y-y1)/(y2-y1)+x1)
    a[inside]=0

def erase_ellipse(a,cx,cy,rx,ry):
    yy,xx=np.mgrid[0:28*S,0:28*S]
    x=(xx+.5)/S; y=(yy+.5)/S
    a[((x-cx)/rx)**2+((y-cy)/ry)**2 <= 1]=0

def finish(a):
    # Area-average antialiasing followed by a small, fixed optical blur.
    a=a.reshape(28,S,28,S).mean((1,3))
    b=a.copy()
    b[1:-1,1:-1]=(4*a[1:-1,1:-1]+2*(a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])+
                   (a[:-2,:-2]+a[:-2,2:]+a[2:,:-2]+a[2:,2:]))/16
    return b

def tshirt(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]
    top=2+(i%3); length=(18,20,22,24)[(i//3)%4]
    half=(5.0,5.8,6.6)[(i//5)%3]; sleeve=(3.0,4.0,5.0)[(i//11)%3]
    bottom=min(27,top+length); shoulder=top+2
    # Body and the two short, outward/downward sleeves.
    polygon(a,[(cx-half,shoulder),(cx+half,shoulder),(cx+half,bottom),(cx-half,bottom)],.92)
    polygon(a,[(cx-half,shoulder),(cx-half-sleeve,shoulder+3.8),(cx-half-sleeve+1,shoulder+7),
               (cx-half+1,shoulder+5.5)],.92)
    polygon(a,[(cx+half,shoulder),(cx+half+sleeve,shoulder+3.8),(cx+half+sleeve-1,shoulder+7),
               (cx+half-1,shoulder+5.5)],.92)
    erase_ellipse(a,cx,top+1.0,1.5+(i%2)*.5,1.6)
    if i%4==0: line(a,(cx-half,bottom-1),(cx+half,bottom-1),.45,.6)
    if i%6==0: rect(a,cx-2,top+7,cx+2,top+10,.65)
    return finish(a)

def trousers(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; top=(0,1,2)[i%3]
    bottom=(26,27,28)[(i//3)%3]; waist=(4.5,5.5,6.5)[(i//5)%3]
    leg=(2.2,2.8,3.4)[(i//11)%3]; crotch=top+(8,10,12)[(i//13)%3]
    polygon(a,[(cx-waist,top),(cx+waist,top),(cx+leg,bottom),(cx+.7,bottom),
               (cx,crotch),(cx-.7,bottom),(cx-leg,bottom)],.95)
    line(a,(cx-waist,top+1.5),(cx+waist,top+1.5),.55,.65)
    if i%4==0:
        line(a,(cx-waist+1,top+2),(cx-leg+.6,bottom),.35,.65)
        line(a,(cx+waist-1,top+2),(cx+leg-.6,bottom),.35,.65)
    if i%6==0:
        line(a,(cx-leg,bottom-2),(cx-.7,bottom-2),.5,.65)
        line(a,(cx+.7,bottom-2),(cx+leg,bottom-2),.5,.65)
    return finish(a)

def pullover(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; top=1+(i%3)
    length=(20,22,24)[(i//3)%3]; half=(5.3,6.1,6.9)[(i//5)%3]
    bottom=min(28,top+length); arm=(4.0,5.0,6.0)[(i//11)%3]
    polygon(a,[(cx-half,top+2),(cx+half,top+2),(cx+half,bottom),(cx-half,bottom)],.88)
    polygon(a,[(cx-half,top+2),(cx-half-arm,top+5),(cx-half-arm+1,bottom-1),
               (cx-half+1,bottom-2)],.88)
    polygon(a,[(cx+half,top+2),(cx+half+arm,top+5),(cx+half+arm-1,bottom-1),
               (cx+half-1,bottom-2)],.88)
    # Narrow dark slits between hanging arms and torso.
    erase_polygon(a,[(cx-half-.6,top+9),(cx-half+.6,top+10),(cx-half+.6,bottom),(cx-half-.6,bottom-1)])
    erase_polygon(a,[(cx+half+.6,top+9),(cx+half-.6,top+10),(cx+half-.6,bottom),(cx+half+.6,bottom-1)])
    erase_ellipse(a,cx,top+1.2,1.8,1.2)
    line(a,(cx-half,bottom-1.5),(cx+half,bottom-1.5),.8,.65)
    if i%4==0:
        line(a,(cx-half+1,top+5),(cx-half+1,bottom-2),.4,.68)
        line(a,(cx+half-1,top+5),(cx+half-1,bottom-2),.4,.68)
    return finish(a)

def dress(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; top=(0,1,2)[i%3]
    bottom=(25,27,28)[(i//3)%3]; waist=(3.4,4.2,5.0)[(i//5)%3]
    flare=(7,9,11)[(i//11)%3]; style=(i//13)%4
    if style==0:                       # sleeveless fitted bodice and full skirt
        polygon(a,[(cx-3,top),(cx+3,top),(cx+waist,top+9),(cx+flare,bottom),(cx-flare,bottom),(cx-waist,top+9)],.94)
        erase_ellipse(a,cx,top,1.6,1.8)
    elif style==1:                     # short sleeves
        polygon(a,[(cx-5,top+2),(cx+5,top+2),(cx+waist,top+9),(cx+flare,bottom),(cx-flare,bottom),(cx-waist,top+9)],.9)
        polygon(a,[(cx-5,top+2),(cx-8,top+5),(cx-6.5,top+8),(cx-4,top+6)],.9)
        polygon(a,[(cx+5,top+2),(cx+8,top+5),(cx+6.5,top+8),(cx+4,top+6)],.9)
        erase_ellipse(a,cx,top+1.3,1.7,1.4)
    elif style==2:                     # narrow evening dress
        polygon(a,[(cx-3.5,top),(cx+3.5,top),(cx+4.5,bottom),(cx-4.5,bottom)],.92)
        line(a,(cx-3.5,top),(cx-2,top-1),.5,.9); line(a,(cx+3.5,top),(cx+2,top-1),.5,.9)
    else:                              # long-sleeved day dress
        polygon(a,[(cx-5,top+2),(cx+5,top+2),(cx+waist,top+10),(cx+flare,bottom),(cx-flare,bottom),(cx-waist,top+10)],.9)
        line(a,(cx-5,top+3),(cx-8,top+18),2,.9); line(a,(cx+5,top+3),(cx+8,top+18),2,.9)
        erase_ellipse(a,cx,top+1.2,1.5,1.2)
    line(a,(cx-waist,top+9),(cx+waist,top+9),.55,.62)
    if i%4==0:
        for d in (-.5,0,.5): line(a,(cx+waist*d,top+10),(cx+flare*d,bottom),.35,.68)
    return finish(a)

def coat(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; top=(0,1,2)[i%3]
    bottom=(25,27,28)[(i//3)%3]; half=(5.5,6.5,7.5)[(i//5)%3]
    arm=(3.8,4.8,5.8)[(i//11)%3]
    polygon(a,[(cx-half,top+2),(cx+half,top+2),(cx+half,bottom),(cx-half,bottom)],.87)
    polygon(a,[(cx-half,top+2),(cx-half-arm,top+5),(cx-half-arm+1,bottom-1),(cx-half+1,bottom-2)],.87)
    polygon(a,[(cx+half,top+2),(cx+half+arm,top+5),(cx+half+arm-1,bottom-1),(cx+half-1,bottom-2)],.87)
    erase_polygon(a,[(cx-half-.6,top+9),(cx-half+.6,top+10),(cx-half+.6,bottom),(cx-half-.6,bottom-1)])
    erase_polygon(a,[(cx+half+.6,top+9),(cx+half-.6,top+10),(cx+half-.6,bottom),(cx+half+.6,bottom-1)])
    erase_ellipse(a,cx,top+1.1,1.7,1.1)
    # Open front and V lapels distinguish a coat from a sweater.
    line(a,(cx,top+5),(cx,bottom),.55,.38)
    polygon(a,[(cx-2.7,top+1),(cx,top+6),(cx-1.2,top+8),(cx-4,top+3)],.62)
    polygon(a,[(cx+2.7,top+1),(cx,top+6),(cx+1.2,top+8),(cx+4,top+3)],.62)
    if i%4==0:
        line(a,(cx-half,top+15),(cx+half,top+15),.6,.55)
        line(a,(cx-half+1,top+18),(cx-2,top+18),.65,.5); line(a,(cx+2,top+18),(cx+half-1,top+18),.65,.5)
    return finish(a)

def sandal(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; base=(23,25,27)[i%3]
    style=(i//3)%6
    if style in (0,1,2):               # high-heeled side views
        heelx=cx+(6,7,8)[style]; tip=cx-11
        line(a,(tip,base-1),(cx-3,base-1),1.0,.88)
        line(a,(cx-3,base-1),(heelx,base-5),1.0,.88)
        line(a,(tip,base-2),(cx-3,base-3),.8,.88)
        line(a,(cx-3,base-3),(heelx,base-5),.8,.88)
        line(a,(heelx,base-5),(heelx,base),.8,.9)
        line(a,(cx-3,base-3),(cx+3,base-12),1.0,.88)
        line(a,(cx+3,base-12),(heelx,base-5),1.0,.88)
        line(a,(cx+2,base-12),(cx+6,base-13),.65,.9)
        line(a,(tip+2,base-3),(cx-4,base-6),.8,.85)
    elif style==3:                     # flat strappy sandal
        polygon(a,[(cx-12,base-2),(cx+10,base-2),(cx+12,base),(cx-11,base)],.85)
        line(a,(cx-7,base-2),(cx-2,base-7),1.2,.9); line(a,(cx-2,base-7),(cx+3,base-2),1.2,.9)
        line(a,(cx+5,base-2),(cx+7,base-6),1,.9)
    elif style==4:                     # frontal pair of slim sandals
        for d in (-4,4):
            ellipse(a,cx+d,base-8,2.5,8,.9)
            line(a,(cx+d-2,base-11),(cx+d+2,base-11),1,.35)
            line(a,(cx+d-2,base-6),(cx+d+2,base-6),1,.35)
    else:                              # wedge sandal
        polygon(a,[(cx-11,base-3),(cx+9,base-5),(cx+10,base),(cx-10,base)],.88)
        line(a,(cx-6,base-4),(cx-1,base-10),1.2,.9); line(a,(cx-1,base-10),(cx+4,base-4),1.2,.9)
    return finish(a)

def sneaker(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; base=(23,25,27)[i%3]
    left=cx-(10,11,12)[(i//3)%3]; right=cx+(10,11,12)[(i//5)%3]
    high=(i//11)%4==0
    if high:
        polygon(a,[(left,base-2),(left+2,base-13),(left+7,base-14),(left+9,base-7),
                   (right-3,base-5),(right,base-2),(right,base),(left,base)],.88)
        line(a,(left+3,base-12),(left+7,base-12),.6,.5)
    else:
        h=(5,7,9)[(i//11)%3]
        polygon(a,[(left,base-2),(left+3,base-h),(left+7,base-h-1),(left+10,base-5),
                   (right-3,base-4),(right,base-2),(right,base),(left,base)],.9)
    line(a,(left,base-1.5),(right,base-1.5),1.0,.98)
    line(a,(left+4,base-6),(left+10,base-5),.55,.45)
    if i%4==0:
        for d in range(3): line(a,(left+5+d,base-7),(left+7+d,base-6),.4,.4)
    return finish(a)

def shirt(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; top=1+(i%3)
    bottom=(23,25,27)[(i//3)%3]; half=(5.2,6.0,6.8)[(i//5)%3]
    short=(i//11)%4==0; arm=(3.8,4.8,5.8)[(i//13)%3]
    polygon(a,[(cx-half,top+2),(cx+half,top+2),(cx+half,bottom),(cx-half,bottom)],.92)
    if short:
        polygon(a,[(cx-half,top+2),(cx-half-arm,top+6),(cx-half-arm+1,top+10),(cx-half+1,top+7)],.92)
        polygon(a,[(cx+half,top+2),(cx+half+arm,top+6),(cx+half+arm-1,top+10),(cx+half-1,top+7)],.92)
    else:
        polygon(a,[(cx-half,top+2),(cx-half-arm,top+5),(cx-half-arm+1,bottom-1),(cx-half+1,bottom-2)],.92)
        polygon(a,[(cx+half,top+2),(cx+half+arm,top+5),(cx+half+arm-1,bottom-1),(cx+half-1,bottom-2)],.92)
        erase_polygon(a,[(cx-half-.6,top+9),(cx-half+.6,top+10),(cx-half+.6,bottom),(cx-half-.6,bottom-1)])
        erase_polygon(a,[(cx+half+.6,top+9),(cx+half-.6,top+10),(cx+half-.6,bottom),(cx+half+.6,bottom-1)])
    erase_ellipse(a,cx,top+1,1.4,1)
    polygon(a,[(cx-2.5,top+1),(cx,top+4),(cx-2,top+5),(cx-4,top+2)],.65)
    polygon(a,[(cx+2.5,top+1),(cx,top+4),(cx+2,top+5),(cx+4,top+2)],.65)
    line(a,(cx,top+4),(cx,bottom),.45,.52)
    if i%4==0:
        for y in range(int(top+7),int(bottom-1),4): ellipse(a,cx,y,.35,.35,.25)
    if i%6==0: rect(a,cx+2,top+8,cx+4.5,top+11,.68)
    return finish(a)

def bag(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; top=(8,10,12)[i%3]
    bottom=(25,27,28)[(i//3)%3]; half=(7,9,11)[(i//5)%3]; style=(i//11)%4
    if style==0:
        polygon(a,[(cx-half,top),(cx+half,top),(cx+half-1,bottom),(cx-half+1,bottom)],.9)
    elif style==1:
        polygon(a,[(cx-half+1,top),(cx+half-1,top),(cx+half,bottom),(cx-half,bottom)],.88)
    elif style==2:
        rect(a,cx-half,top,cx+half,bottom,.85)
        rect(a,cx-half-1,top+4,cx-half,top+9,.8); rect(a,cx+half,top+4,cx+half+1,top+9,.8)
    else:
        ellipse(a,cx,(top+bottom)/2,half,(bottom-top)/2,.87)
    # Arched handles, each drawn directly rather than learned from examples.
    handle=(3,5,7)[(i//13)%3]
    pts=[]
    for k in range(17):
        theta=np.pi*k/16
        pts.append((cx+handle*np.cos(theta),top-handle*np.sin(theta)))
    for p,q in zip(pts,pts[1:]): line(a,p,q,.75,.95)
    if i%4==0: line(a,(cx-half,top+3),(cx+half,top+3),.6,.55)
    if i%6==0: rect(a,cx-2,top+5,cx+2,top+7,.55)
    return finish(a)

def boot(i):
    a=canvas(); cx=14+(-1,0,1)[(i//7)%3]; base=(25,27,28)[i%3]
    left=cx-10; right=cx+(9,10,11)[(i//3)%3]; shaft=(11,14,18)[(i//5)%3]
    top=max(0,base-shaft); back=cx+(2,4,5)[(i//11)%3]
    polygon(a,[(left,base-3),(cx-2,base-5),(back-2,base-8),(back-2,top),
               (back+4,top),(back+4,base-3),(right,base-2),(right,base),(left,base)],.9)
    # Heel and a distinct horizontal sole.
    rect(a,back+1,base-3,right,base,.94)
    line(a,(left,base-1.5),(right,base-1.5),.9,.98)
    line(a,(back-2,top+1),(back+4,top+1),.65,.55)
    if i%4==0:
        line(a,(back,top+3),(back,base-5),.45,.5)
        for y in range(int(top+4),int(base-6),3): line(a,(back-1,y),(back+1,y+1),.4,.45)
    return finish(a)

MAKERS=[tshirt,trousers,pullover,dress,coat,sandal,sneaker,shirt,bag,boot]

def main(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    drawings=[maker(i).astype(np.float64) for maker in MAKERS for i in range(N_PER_CLASS)]

    def overlap_banks(pool):
        base=[]; over=[]; under=[]
        for t in drawings:
            if pool>1:
                t=t.reshape(28//pool,pool,28//pool,pool).mean((1,3))
            p=np.clip(t/(t.max()+1e-12),0,1)
            ink=p.sum()+1e-12; n=p.size
            base.append((p/ink).reshape(-1))
            # If A is image ink and T template ink, these are the two ReLU
            # halves of |A-T|/(2T).  Adding I/T gives a Dice-like overlap.
            over.append(np.full(n,1/(2*ink*n)))
            under.append(np.full(n,-1/(2*ink*n)))
        return tuple(np.asarray(v,dtype=np.float32) for v in (base,over,under))

    layers=[
      # Soft binary silhouette: min(1, 12*(image-.05)_+).
      {"name":"ink_positive","op":"conv","inputs":["image"],"out":1,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"ink_relu","op":"relu","inputs":["ink_positive"]},
      {"name":"above_one","op":"conv","inputs":["ink_relu"],"out":1,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"above_relu","op":"relu","inputs":["above_one"]},
      {"name":"subtract_clip","op":"conv","inputs":["above_relu"],"out":1,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"silhouette","op":"add","inputs":["ink_relu","subtract_clip"]},
      {"name":"grid14","op":"avgpool","inputs":["silhouette"],"size":[2,2],"stride":[2,2]},
      {"name":"grid7","op":"avgpool","inputs":["silhouette"],"size":[4,4],"stride":[4,4]},
      {"name":"raw7","op":"avgpool","inputs":["image"],"size":[4,4],"stride":[4,4]}
    ]
    weights={}
    one=np.ones((1,1,1,1),np.float32)
    weights.update({'ink_positive.W':12*one,'ink_positive.b':np.array([-.6],np.float32),
                    'above_one.W':one,'above_one.b':np.array([-1],np.float32),
                    'subtract_clip.W':-one,'subtract_clip.b':np.zeros(1,np.float32)})
    branches=[]
    for prefix,source,pool in [('full','silhouette',1),('medium','grid14',2),('coarse','grid7',4),('raw','raw7',4)]:
        base=prefix+'_intersection'; over=prefix+'_over'; under=prefix+'_under'
        over_r=prefix+'_over_relu'; under_r=prefix+'_under_relu'
        pen=prefix+'_penalty'; pen_spatial=prefix+'_penalty_spatial'; neg=prefix+'_negative_penalty'
        pen_vec=prefix+'_penalty_vector'; overlap=prefix+'_overlap'
        spatial=prefix+'_spatial'; best=prefix+'_best'; flat=prefix+'_flat'; scaled=prefix+'_scaled'
        layers.extend([
          {"name":base,"op":"dense","inputs":[source],"out":10*N_PER_CLASS},
          {"name":over,"op":"dense","inputs":[source],"out":10*N_PER_CLASS},
          {"name":over_r,"op":"relu","inputs":[over]},
          {"name":under,"op":"dense","inputs":[source],"out":10*N_PER_CLASS},
          {"name":under_r,"op":"relu","inputs":[under]},
          {"name":pen,"op":"add","inputs":[over_r,under_r]},
          {"name":pen_spatial,"op":"reshape","inputs":[pen],"shape":[1,1,10*N_PER_CLASS]},
          {"name":neg,"op":"conv","inputs":[pen_spatial],"out":1,"kernel":[1,1],"stride":1,"padding":0},
          {"name":pen_vec,"op":"reshape","inputs":[neg],"shape":[10*N_PER_CLASS]},
          {"name":overlap,"op":"add","inputs":[base,pen_vec]},
          {"name":spatial,"op":"reshape","inputs":[overlap],"shape":[1,1,10*N_PER_CLASS]},
          {"name":best,"op":"maxpool","inputs":[spatial],"size":[1,N_PER_CLASS],"stride":[1,N_PER_CLASS]},
          {"name":flat,"op":"reshape","inputs":[best],"shape":[10]},
          {"name":scaled,"op":"dense","inputs":[flat],"out":10}
        ])
        wb,wo,wu=overlap_banks(pool)
        weights[base+'.W']=wb; weights[base+'.b']=np.zeros(10*N_PER_CLASS,np.float32)
        weights[over+'.W']=wo; weights[over+'.b']=np.full(10*N_PER_CLASS,-.5,np.float32)
        weights[under+'.W']=wu; weights[under+'.b']=np.full(10*N_PER_CLASS,.5,np.float32)
        weights[neg+'.W']=-one; weights[neg+'.b']=np.zeros(1,np.float32)
        weights[scaled+'.W']=np.eye(10,dtype=np.float32)*.25
        weights[scaled+'.b']=np.zeros(10,np.float32)
        branches.append(scaled)
    layers.append({"name":"scores","op":"add","inputs":branches})
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**weights)

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '.')
