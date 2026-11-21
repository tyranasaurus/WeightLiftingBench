import json, os, sys
import numpy as np

# A fully data-free, deterministic hand-designed CIFAR-scale vision system.
# Every constant is an analytic filter or a general-knowledge class rule.


def L(name, op, inputs, **settings):
    d={"name":name,"op":op,"inputs":[inputs] if isinstance(inputs,str) else inputs}
    d.update(settings); return d


def grid(k):
    q=np.arange(k,dtype=np.float64)-(k-1)/2
    return np.meshgrid(q,q,indexing="ij")


def gauss(k,s):
    y,x=grid(k); a=np.exp(-(x*x+y*y)/(2*s*s)); return a/a.sum()


def grad(k,s,theta):
    y,x=grid(k); u=x*np.cos(theta)+y*np.sin(theta)
    a=-u*np.exp(-(x*x+y*y)/(2*s*s))
    a-=a.mean(); a/=np.abs(a).sum()+1e-12; return a*2


def gabor(k,s,theta,wavelength=4.0):
    y,x=grid(k); u=x*np.cos(theta)+y*np.sin(theta); v=-x*np.sin(theta)+y*np.cos(theta)
    a=np.cos(2*np.pi*u/wavelength)*np.exp(-(u*u+0.55*v*v)/(2*s*s))
    a-=a.mean(); a/=np.abs(a).sum()+1e-12; return a*2


def dog(k,a,b):
    z=gauss(k,a)-gauss(k,b); z-=z.mean(); z/=np.abs(z).sum()+1e-12; return z*2


def build(out):
    layers=[]; weights={}; C=64
    # ------------------------------------------------------------------
    # Smooth colour and colour-opponent threshold maps. Only channels 0:32
    # are used in this branch; disjoint channel ranges let an add operation
    # act as a legal concatenation after different pooling types.
    layers.append(L("colour","conv","image",out=C,kernel=[5,5],stride=1,padding=2))
    CW=np.zeros((C,3,5,5),np.float64); cb=np.zeros(C,np.float64); g=gauss(5,1.05)
    # (RGB linear combination, threshold). Output is ReLU(mix dot RGB + bias).
    specs=[
      ((.299,.587,.114),-.58), ((.299,.587,.114),-.75),
      ((-.299,-.587,-.114),.42), ((-.299,-.587,-.114),.25),
      ((-.50,-.50,1.00),-.03), ((-.50,-.50,1.00),-.16),
      ((-.50,1.00,-.50),-.03), ((-.50,1.00,-.50),-.15),
      ((1.00,-.50,-.50),-.03), ((1.00,-.50,-.50),-.16),
      ((.50,.50,-1.00),-.04), ((-.50,.50,.50),-.04),
      ((0,0,1),-.52), ((0,1,0),-.48), ((1,0,0),-.54),
      ((.70,.30,-.72),-.05), # warm brown/orange against blue
      ((.55,.45,-.20),-.42), # tan/yellow absolute
      ((-.15,.75,-.15),-.25),# vegetation-like green absolute
      ((.50,-.10,-.40),-.04),# warm red versus blue
      ((-.40,.10,.50),-.04), # cool blue versus red
      ((1,-1,0),-.06), ((0,1,-1),-.06), ((-1,0,1),-.06),
      ((1,1,1),-1.35),       # broadly bright/sum threshold
      ((-1,-1,-1),1.05),     # broadly dark/sum threshold
      ((.333,.333,.333),-.35), ((-.333,-.333,-.333),.65),
      ((.299,.587,.114),-.45), ((-.299,-.587,-.114),.55),
      ((.6,.2,.2),-.42), ((.2,.6,.2),-.42), ((.2,.2,.6),-.42)
    ]
    for j,(mix,bias) in enumerate(specs):
        CW[j]=np.asarray(mix)[:,None,None]*g; cb[j]=bias
    weights["colour.W"]=CW; weights["colour.b"]=cb
    layers.append(L("colour_relu","relu","colour"))
    layers.append(L("colour_cells","avgpool","colour_relu",size=[4,4],stride=[4,4]))

    # ------------------------------------------------------------------
    # Oriented edges, bars, blobs and corners. Only channels 32:64 are used;
    # max pooling records whether a part occurs in each spatial cell.
    layers.append(L("form","conv","image",out=C,kernel=[7,7],stride=1,padding=3))
    FW=np.zeros((C,3,7,7),np.float64); fb=np.zeros(C,np.float64)
    lum=np.array([.299,.587,.114])
    # Four gradient normals, both polarities: 32:40.
    for t,theta in enumerate(np.arange(4)*np.pi/4):
        f=grad(7,1.45,theta)
        FW[32+2*t]=lum[:,None,None]*f
        FW[33+2*t]=-FW[32+2*t]
        fb[32+2*t:34+2*t]=-.025
    # Four line orientations, bright and dark: 40:48.
    for t,theta in enumerate(np.arange(4)*np.pi/4):
        f=gabor(7,1.75,theta)
        FW[40+2*t]=lum[:,None,None]*f
        FW[41+2*t]=-FW[40+2*t]
        fb[40+2*t:42+2*t]=-.035
    # Bright and dark compact spots: 48:50.
    f=dog(7,.85,2.15)
    FW[48]=lum[:,None,None]*f; FW[49]=-FW[48]; fb[48:50]=-.035
    # L-corners from products approximated by signed quadrant linear filters.
    y,x=grid(7); gg=np.exp(-(x*x+y*y)/(2*1.65**2))
    corner_forms=[]
    for sx,sy in ((1,1),(-1,1),(1,-1),(-1,-1)):
        q=gg*(sx*np.sign(x)+sy*np.sign(y)); q-=q.mean(); q/=np.abs(q).sum()+1e-12
        corner_forms.append(q*2)
    for t,f in enumerate(corner_forms):
        FW[50+2*t]=lum[:,None,None]*f; FW[51+2*t]=-FW[50+2*t]
        fb[50+2*t:52+2*t]=-.035
    # Larger horizontal/vertical/diagonal bars and local contrast: 58:64.
    for j,theta in enumerate((np.pi/2,0,np.pi/4)):
        f=gabor(7,2.4,theta,5.5)
        FW[58+2*j]=lum[:,None,None]*f; FW[59+2*j]=-FW[58+2*j]
        fb[58+2*j:60+2*j]=-.025
    weights["form.W"]=FW; weights["form.b"]=fb
    layers.append(L("form_relu","relu","form"))
    layers.append(L("form_cells","maxpool","form_relu",size=[4,4],stride=[4,4]))
    layers.append(L("visual_cells","add",["colour_cells","form_cells"]))

    # ------------------------------------------------------------------
    # Hand-written scoring rules over an 8x8 spatial grid. Coefficients are
    # normalized masks, so each rule measures an average rather than depending
    # on region size.
    layers.append(L("scores","dense","visual_cells",out=10))
    W=np.zeros((10,C,8,8),np.float64); bias=np.zeros(10,np.float64)
    yy=(np.arange(8)+.5)/4-1; xx=(np.arange(8)+.5)/4-1
    Y,X=np.meshgrid(yy,xx,indexing="ij")
    masks={
      "all":np.ones((8,8)), "top":np.exp(-((Y+.62)/.55)**2),
      "bottom":np.exp(-((Y-.62)/.48)**2), "center":np.exp(-1.8*(X*X+Y*Y)),
      "upper":np.exp(-2.3*(X*X+(Y+.38)**2)),
      "lower":np.exp(-2.2*(X*X+(Y-.38)**2)),
      "middlebar":np.exp(-(Y/.23)**2)*np.exp(-.45*X*X),
      "lowbar":np.exp(-((Y-.42)/.20)**2)*np.exp(-.35*X*X),
      "legs":np.exp(-((Y-.55)/.36)**2)*(np.exp(-((X-.38)/.17)**2)+np.exp(-((X+.38)/.17)**2)),
      "wheels":np.exp(-((Y-.56)/.19)**2)*(np.exp(-((X-.53)/.20)**2)+np.exp(-((X+.53)/.20)**2)),
      "diagwing":np.exp(-((Y-.72*X)/.18)**2)+np.exp(-((Y+.72*X)/.18)**2),
      "horizon":np.exp(-((Y-.12)/.16)**2),
      "sides":np.exp(-((np.abs(X)-.56)/.24)**2)*np.exp(-.8*Y*Y)
    }
    def add(k,ch,coef,mask="all"):
        m=masks[mask]; W[k,ch]+=coef*m/(m.sum()+1e-12)
    def pair(k,chs,coef,mask="all"):
        for ch in chs: add(k,ch,coef/len(chs),mask)
    # Colour cue indices: blue=4/5/12/19/31, green=6/7/13/17/21/30,
    # red/warm=8/9/14/15/16/18/20/29, bright=0/1/23, dark=2/3/24.
    # Edge families: vertical gradient 32/33, diagonals 34:40, horizontal
    # line 40/41, diagonal lines 42:48, blobs 48/49, corners 50:58,
    # horizontal/vertical broad bars 58:62.

    # AIRPLANE: sky surrounds a compact, diagonal-winged, roughly symmetric form.
    pair(0,[4,5,12,19,31],2.25,"all"); pair(0,[0,1,23],.45,"top")
    pair(0,[34,35,38,39,42,43,46,47],1.75,"diagwing")
    pair(0,[40,41,58,59],.55,"middlebar"); pair(0,[6,7,17],-.75,"all")
    pair(0,[48,49],-.30,"bottom")

    # AUTOMOBILE: low colourful body, horizontal roof/body edges and two wheels.
    pair(1,[8,9,14,18,20,29],.70,"lower"); pair(1,[2,3,24,26,28],.65,"bottom")
    pair(1,[40,41,58,59],1.65,"lowbar"); pair(1,[32,33,50,51,54,55],.72,"lower")
    add(1,49,1.45,"wheels"); pair(1,[4,5,12],-.25,"bottom")

    # BIRD: a small high-frequency, often diagonal object amid sky or foliage.
    pair(2,[4,6,7,12,13,17,19,21,30,31],.38,"all")
    pair(2,list(range(34,40))+list(range(42,48)),.95,"center")
    pair(2,[48,49],.72,"upper"); pair(2,[50,51,52,53,54,55,56,57],.30,"center")
    pair(2,[58,59],.24,"bottom")

    # CAT: compact warm/gray face and body, pointed corner-like ears, eye spots.
    pair(3,[2,8,10,14,15,16,18,25,27,29],.42,"center")
    pair(3,[48,49],1.05,"upper"); pair(3,[50,51,52,53,54,55,56,57],.65,"upper")
    pair(3,[32,33,60,61],.45,"center"); pair(3,[6,7,17],-.18,"center")

    # DEER: tan slender quadruped in vegetation, with vertical legs/neck.
    pair(4,[6,7,13,17,21,30],.72,"all"); pair(4,[10,15,16,18],.80,"center")
    pair(4,[32,33,60,61],1.12,"legs"); pair(4,[40,41,58,59],.55,"middlebar")
    pair(4,[34,35,38,39],.32,"upper")

    # DOG: warm/varied broad central animal, textured face and muzzle.
    pair(5,[8,10,14,15,16,18,25,27,29],.52,"center")
    pair(5,[48,49],.90,"center"); pair(5,list(range(50,58)),.42,"center")
    pair(5,list(range(32,48)),.28,"center"); pair(5,[32,33,60,61],.28,"lower")

    # FROG: green low compact shape, prominent paired eyes and splayed diagonals.
    pair(6,[6,7,13,17,21,30],2.05,"all"); pair(6,[4,5,12,19],-.35,"all")
    pair(6,[48,49],1.18,"upper"); pair(6,[34,35,38,39,42,43,46,47],.62,"lower")
    pair(6,[40,41,58,59],.44,"lowbar")

    # HORSE: large warm horizontal body with long paired vertical legs.
    pair(7,[10,15,16,18,20,29],.82,"center"); pair(7,[6,7,17],.30,"all")
    pair(7,[32,33,60,61],1.55,"legs"); pair(7,[40,41,58,59],.88,"middlebar")
    pair(7,[34,35,38,39],.32,"upper"); add(7,49,.24,"upper")

    # SHIP: blue lower half, bright upper half, long horizon/hull and mast/decks.
    pair(8,[4,5,12,19,22,31],2.65,"bottom"); pair(8,[0,1,23],.62,"top")
    pair(8,[40,41,58,59],2.45,"horizon"); pair(8,[32,33,60,61],.55,"center")
    pair(8,[6,7,13,17],-.85,"all"); pair(8,[34,35,38,39],-.25,"all")

    # TRUCK: bright/gray high rectangular mass, corners, bars and paired wheels.
    pair(9,[0,23,25,27],.50,"lower"); pair(9,[8,9,14,18,29],.34,"lower")
    pair(9,[40,41,58,59],1.45,"lowbar"); pair(9,[32,33,60,61],1.35,"sides")
    pair(9,list(range(50,58)),.88,"lower"); add(9,49,1.15,"wheels")
    pair(9,[4,5,12],-.22,"bottom")

    # Equalize every class rule's positive coefficient budget. This is a purely
    # structural calibration: a class with more listed cues must not win merely
    # because its rule has more terms. Negative evidence is scaled with it.
    for k in range(10):
        pos=np.maximum(W[k],0).sum()
        W[k]*=5.0/(pos+1e-12)
    bias[:]=0
    weights["scores.W"]=W.reshape(10,-1); weights["scores.b"]=bias

    net={"layers":layers}; os.makedirs(out,exist_ok=True)
    with open(os.path.join(out,"network.json"),"w") as f: json.dump(net,f,separators=(",",":"))
    np.savez(os.path.join(out,"weights.npz"),**{k:np.asarray(v,np.float32) for k,v in weights.items()})


if __name__=="__main__": build(sys.argv[1] if len(sys.argv)>1 else ".")
