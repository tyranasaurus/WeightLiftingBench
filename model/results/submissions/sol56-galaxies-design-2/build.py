#!/usr/bin/env python3
"""Construct a data-free, hand-engineered galaxy morphology network."""
import json
import math
import sys
from pathlib import Path
import numpy as np


def add_layer(layers, name, op, inp, **kw):
    d = {"name": name, "op": op, "inputs": inp if isinstance(inp, list) else [inp]}
    d.update(kw); layers.append(d)


def add_conv(layers, weights, name, inp, W, b, stride=1, padding=0):
    W, b = np.asarray(W, np.float32), np.asarray(b, np.float32)
    add_layer(layers, name, "conv", inp, out=W.shape[0], kernel=[W.shape[2], W.shape[3]], stride=stride, padding=padding)
    weights[name+".W"], weights[name+".b"] = W, b


def add_dense(layers, weights, name, inp, W, b):
    W, b = np.asarray(W, np.float32), np.asarray(b, np.float32)
    add_layer(layers, name, "dense", inp, out=W.shape[0])
    weights[name+".W"], weights[name+".b"] = W, b


def gaussian(size, sigma):
    q = np.arange(size, dtype=np.float64) - (size-1)/2
    y, x = np.meshgrid(q, q, indexing="ij")
    z = np.exp(-(x*x+y*y)/(2*sigma*sigma)); return z/z.sum()


def bilinear(a, y, x):
    y0, x0 = math.floor(y), math.floor(x); fy, fx = y-y0, x-x0
    v = 0.0
    for dy, wy in ((0,1-fy),(1,fy)):
        for dx, wx in ((0,1-fx),(1,fx)):
            yy, xx = y0+dy, x0+dx
            if 0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]: v += wy*wx*a[yy,xx]
    return v


def rotate(a, angle):
    n = a.shape[0]; c=(n-1)/2; co,si=math.cos(angle),math.sin(angle)
    z=np.zeros_like(a)
    for y in range(n):
        for x in range(n):
            xo,yo=x-c,y-c
            z[y,x]=bilinear(a, -si*xo+co*yo+c, co*xo+si*yo+c)
    if abs(a.sum()) < 1e-9:
        z -= z.mean(); z *= np.abs(a).sum()/max(np.abs(z).sum(),1e-12)
    else: z *= a.sum()/z.sum()
    return z


def norm_pos(z):
    z=np.maximum(z,0); return z/max(z.sum(),1e-12)


def norm_signed(z):
    z=z-z.mean(); return z/max(np.abs(z).sum(),1e-12)


def build(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    layers=[]; weights={}

    # Anti-aliased 4x4 photometry at 32x32. Thresholded red/blue/luminance
    # planes approximate isophotes and make the later shape measurements robust
    # to sky noise and foreground pinpoints.
    mixes=[("L",np.array([.30,.50,.20])),("R",np.array([.58,.34,.08])),("B",np.array([.08,.35,.57]))]
    cuts=[.015,.030,.050,.080,.120,.180,.270]
    pkeys=[]; P=[]; pb=[]
    for mn,m in mixes:
        for j,t in enumerate(cuts):
            P.append(m[:,None,None]*np.ones((3,4,4))/16.0); pb.append(-t); pkeys.append((mn,j))
    add_conv(layers,weights,"isophotes","image",P,pb,stride=4)
    add_layer(layers,"isophotes_pos","relu","isophotes")
    pchannel={k:i for i,k in enumerate(pkeys)}

    # Analytic local morphology filters: oriented ridges/edges plus compact
    # knots. Every orientation is present, avoiding a preferred sky direction.
    nori=12; angles=[math.pi*i/nori for i in range(nori)]
    q=np.arange(9,dtype=np.float64)-4; y,x=np.meshgrid(q,q,indexing="ij")
    ridge=(1-(x/1.25)**2)*np.exp(-(x*x/(2*1.25**2)+y*y/(2*3.0**2))); ridge=norm_signed(ridge)
    edge=x*np.exp(-(x*x+y*y)/(2*1.5**2)); edge=norm_signed(edge)
    lum=np.array([.30,.50,.20]); blue=np.array([.08,.35,.57]); red=np.array([.58,.34,.08])
    D=[]; dkeys=[]
    for family,base in (("ridge",ridge),("valley",-ridge),("edge",edge),("edge2",-edge)):
        for o,a in enumerate(angles):
            D.append(lum[:,None,None]*rotate(base,a)[None]); dkeys.append((family,o))
    dog1=norm_signed(gaussian(9,.9)-gaussian(9,2.2)); dog2=norm_signed(gaussian(9,1.6)-gaussian(9,3.3))
    for key,ker,mix in (("knotL",dog1,lum),("knotB",dog1,blue),("knotR",dog1,red),
                        ("blobL",dog2,lum),("blobB",dog2,blue),("hole",-dog2,lum)):
        D.append(mix[:,None,None]*ker[None]); dkeys.append((key,0))
    add_conv(layers,weights,"local","image",D,np.full(len(D),-.0020),padding=4)
    add_layer(layers,"local_pos","relu","local")
    add_layer(layers,"local_grid","avgpool","local_pos",size=[4,4],stride=[4,4])
    dchannel={k:i for i,k in enumerate(dkeys)}

    # Spatial masks on the common 32x32 grid.
    u=(np.arange(32)-15.5)/16.0; Y,X=np.meshgrid(u,u,indexing="ij")
    R=np.sqrt(X*X+Y*Y); A=np.arctan2(Y,X)
    core=np.exp(-(R/.13)**4); inner=np.exp(-(R/.25)**4); mid=np.exp(-(R/.43)**6)
    outer=np.exp(-(R/.68)**8); wide=np.exp(-(R/.96)**10)
    ann1=np.maximum(mid-inner,0); ann2=np.maximum(outer-mid,0)
    masks={"core":norm_pos(core),"inner":norm_pos(inner),"mid":norm_pos(mid),
           "outer":norm_pos(outer),"wide":norm_pos(wide),"ann1":norm_pos(ann1),"ann2":norm_pos(ann2)}
    # Positive oriented inner/outer elliptical apertures.
    for o,a in enumerate(angles):
        co,si=math.cos(a),math.sin(a); para=X*co+Y*si; perp=-X*si+Y*co
        masks[("iline",o)]=norm_pos(np.exp(-((para/.30)**4+(perp/.065)**4)))
        masks[("mline",o)]=norm_pos(np.exp(-((para/.48)**4+(perp/.105)**4)))
        masks[("oline",o)]=norm_pos(np.exp(-((para/.70)**4+(perp/.115)**4)))
    # Signed angular moments, merger-pair contrasts, and logarithmic spirals.
    for radial,rname in ((outer,"all"),(ann1,"a1"),(ann2,"a2")):
        for m in (1,2,3,4):
            masks[("harm",rname,m,"c")]=norm_signed(radial*np.cos(m*A))
            masks[("harm",rname,m,"s")]=norm_signed(radial*np.sin(m*A))
    for o,a in enumerate(angles):
        co,si=math.cos(a),math.sin(a)
        for di,dist in enumerate((.18,.28,.40)):
            g1=np.exp(-((X-dist*co)**2+(Y-dist*si)**2)/(2*.075**2))
            g2=np.exp(-((X+dist*co)**2+(Y+dist*si)**2)/(2*.075**2))
            gc=np.exp(-(X*X+Y*Y)/(2*.10**2))
            masks[("pair",o,di)]=norm_signed(g1+g2-1.15*gc)
    lr=np.log(np.maximum(R,.055)); spiral_window=ann1+.8*ann2
    for hand in (-1,1):
        for pi,pitch in enumerate((1.4,2.5,4.0)):
            for phase,trig in (("c",np.cos),("s",np.sin)):
                masks[("spiral",hand,pi,phase)]=norm_signed(spiral_window*trig(2*A+hand*pitch*lr))

    # Build compact branch-specific banks. A key map records semantic scalar
    # measurements for the explicit rule table below.
    pspec=[]
    for mn,_ in mixes:
        for lev in range(7):
            ch=pchannel[(mn,lev)]
            for ap in ("core","inner","mid","outer","wide","ann1","ann2"):
                pspec.append((("ap",mn,lev,ap),ch,masks[ap]))
    # Shape at several isophotes; signed quantities are represented in both
    # polarities before ReLU, giving absolute moments without an abs operation.
    for lev in (0,1,2,3,4):
        ch=pchannel[("L",lev)]
        for o in range(nori):
            for kind in ("iline","mline","oline"):
                pspec.append(((kind,lev,o),ch,masks[(kind,o)]))
        for rname in ("all","a1","a2"):
            for m in (1,2,3,4):
                for ph in ("c","s"):
                    z=masks[("harm",rname,m,ph)]
                    pspec.append((("harm",lev,rname,m,ph,1),ch,z))
                    pspec.append((("harm",lev,rname,m,ph,-1),ch,-z))
    for lev in (1,2,3):
        ch=pchannel[("L",lev)]
        for o in range(nori):
            for di in range(3):
                z=masks[("pair",o,di)]
                pspec.append((("pair",lev,o,di,1),ch,z)); pspec.append((("pair",lev,o,di,-1),ch,-z))
        for hand in (-1,1):
            for pi in range(3):
                for ph in ("c","s"):
                    z=masks[("spiral",hand,pi,ph)]
                    pspec.append((("spiral",lev,hand,pi,ph,1),ch,z)); pspec.append((("spiral",lev,hand,pi,ph,-1),ch,-z))

    dspec=[]
    for key,ch in dchannel.items():
        for ap in ("core","inner","ann1","ann2","outer"):
            dspec.append((("detail",key[0],key[1],ap),ch,masks[ap]))

    def bank(specs,channels):
        W=np.zeros((len(specs),channels,32,32),np.float32)
        for i,(_,ch,z) in enumerate(specs): W[i,ch]=z
        return W
    add_conv(layers,weights,"shape_photo","isophotes_pos",bank(pspec,len(pkeys)),np.zeros(len(pspec)))
    add_conv(layers,weights,"shape_detail","local_grid",bank(dspec,len(dkeys)),np.zeros(len(dspec)))
    # Scalar concatenation using 1x1 partial identities.
    np_,nd_=len(pspec),len(dspec); dim=np_+nd_
    Ep=np.zeros((dim,np_,1,1),np.float32); Ep[:np_,:,0,0]=np.eye(np_)
    Ed=np.zeros((dim,nd_,1,1),np.float32); Ed[np_:,:,0,0]=np.eye(nd_)
    add_conv(layers,weights,"photo_embed","shape_photo",Ep,np.zeros(dim))
    add_conv(layers,weights,"detail_embed","shape_detail",Ed,np.zeros(dim))
    add_layer(layers,"measurements","add",["photo_embed","detail_embed"])
    add_layer(layers,"measurements_pos","relu","measurements")
    index={key:i for i,(key,_,_) in enumerate(pspec)}
    index.update({key:np_+i for i,(key,_,_) in enumerate(dspec)})

    # A rule is a clipped ramp of a physically meaningful scalar z. It is
    # realized as ReLU(z-lo)-ReLU(z-hi), with the second layer supplying equal
    # and opposite votes. This bounds every vote and makes thresholds readable.
    rows=[]; biases=[]; outcols=[]
    def vec(**kw):
        z=np.zeros(10)
        for k,v in kw.items(): z[int(k)]=v
        return z
    def rule(terms,lo,hi,v):
        w=np.zeros(dim)
        for key,c in terms: w[index[key]] += c
        scale=1.0/max(hi-lo,1e-8)
        rows.extend((w,w.copy())); biases.extend((-lo,-hi)); outcols.extend((v*scale,-v*scale))
    def AP(mn,lev,ap): return ("ap",mn,lev,ap)
    def abs_terms(kind,lev,*args,coef=1.0):
        return [((kind,lev,*args,1),coef),((kind,lev,*args,-1),coef)]

    # Brightness, extent, concentration and colour. Several isophotes make the
    # decisions useful across galaxy angular sizes and exposure levels.
    for lev in (0,1,2,3,4):
        # central concentration and a complementary diffuse/ring response
        rule([(AP("L",lev,"core"),1),(AP("L",lev,"mid"),-.62)],.004,.055,
             vec(**{"2":.34,"3":.20,"9":.25,"6":.10,"7":-.16,"8":-.12}))
        rule([(AP("L",lev,"ann1"),1),(AP("L",lev,"core"),-.24)],.002,.035,
             vec(**{"0":.12,"1":.10,"5":.20,"6":.28,"7":.34,"2":-.26,"4":-.18}))
        rule([(AP("L",lev,"ann2"),1),(AP("L",lev,"inner"),-.20)],.001,.025,
             vec(**{"0":.16,"1":.10,"5":.15,"6":.22,"7":.30,"2":-.20}))
        # red versus blue stellar populations, center and arms separately
        for ap in ("core","ann1","ann2"):
            rule([(AP("R",lev,ap),1),(AP("B",lev,ap),-.86)],.002,.050,
                 vec(**{"2":.24,"3":.18,"4":.20,"9":.25,"6":.06,"7":-.20,"0":-.10}))
            rule([(AP("B",lev,ap),1),(AP("R",lev,ap),-.82)],.002,.050,
                 vec(**{"0":.20,"1":.12,"5":.24,"6":.26,"7":.38,"8":.20,"2":-.25,"4":-.18}))

    # Rotation-independent ellipticity is a sum of orientation trials. Inner
    # elongation promotes bars; outer elongation promotes edge-on/cigar classes.
    for lev in (0,1,2,3):
        for o in range(nori):
            innerbar=[(("iline",lev,o),1),(AP("L",lev,"inner"),-.92)]
            outerell=[(("oline",lev,o),1),(AP("L",lev,"outer"),-.88)]
            medell=[(("mline",lev,o),1),(AP("L",lev,"mid"),-.90)]
            rule(innerbar,.002,.035,vec(**{"5":.105,"8":.025,"9":.025,"4":.018,"2":-.035}))
            rule(medell,.003,.045,vec(**{"3":.035,"4":.055,"8":.060,"9":.055,"2":-.045}))
            rule(outerell,.003,.045,vec(**{"4":.070,"8":.085,"9":.075,"3":.025,"2":-.050,"5":-.015,"6":-.015,"7":-.020}))
            # Very strong thinness is characteristic of edge-on disks.
            rule(outerell,.035,.095,vec(**{"8":.080,"9":.075,"4":.025,"3":-.025}))

    # Odd harmonics measure disturbance; m=2/m=4 in annuli measure coherent
    # arms/rings. Absolute values come from paired signed spatial outputs.
    for lev in (0,1,2,3):
        odd=[]; even=[]
        for rname in ("all","a1","a2"):
            for m in (1,3):
                for ph in ("c","s"): odd += abs_terms("harm",lev,rname,m,ph,coef=.125)
            for m in (2,4):
                for ph in ("c","s"): even += abs_terms("harm",lev,rname,m,ph,coef=.125)
        rule(odd,.0015,.018,vec(**{"0":.50,"1":.28,"7":.18,"5":.08,"2":-.30,"3":-.20,"4":-.16}))
        rule(odd,.018,.050,vec(**{"0":.30,"1":.22,"7":.12,"2":-.14}))
        rule(even,.0015,.020,vec(**{"5":.16,"6":.22,"7":.30,"0":.12,"2":-.16,"3":-.10}))

    # Two-lobed central contrasts model double nuclei. Max is approximated by a
    # small vote from each orientation; only nearby orientations fire strongly.
    for lev in (1,2,3):
        for o in range(nori):
            for di in range(3):
                terms=abs_terms("pair",lev,o,di,coef=1.0)
                rule(terms,.002,.030,vec(**{"1":.105,"0":.040,"7":.012,"2":-.030}))

    # Log-spiral correlations: low phase winding indicates open arms, high
    # winding tight arms. Both handednesses and quadrature phases are equal.
    for lev in (1,2,3):
        for hand in (-1,1):
            for pi in range(3):
                for ph in ("c","s"):
                    terms=abs_terms("spiral",lev,hand,pi,ph)
                    if pi==0: vv=vec(**{"7":.075,"5":.020,"0":.015,"2":-.018})
                    elif pi==1: vv=vec(**{"7":.045,"6":.050,"5":.022,"2":-.015})
                    else: vv=vec(**{"6":.070,"5":.020,"7":.018,"2":-.012})
                    rule(terms,.0015,.022,vv)

    # Texture and knots. Ridge/edge averages identify disks and arms; compact
    # blue knots favor disturbed/open-spiral systems. Low texture is represented
    # by rules on the negated statistic and rewards smooth galaxies.
    ridge_terms=[]; edge_terms=[]
    for o in range(nori):
        for fam in ("ridge","valley"):
            ridge_terms.append((("detail",fam,o,"ann1"),1/24)); ridge_terms.append((("detail",fam,o,"ann2"),1/24))
        for fam in ("edge","edge2"):
            edge_terms.append((("detail",fam,o,"ann1"),1/24)); edge_terms.append((("detail",fam,o,"ann2"),1/24))
    texture=ridge_terms+edge_terms
    rule(texture,.0012,.009,vec(**{"0":.22,"1":.12,"5":.24,"6":.30,"7":.36,"8":.08,"2":-.32,"3":-.24,"4":-.22}))
    rule(texture,.009,.025,vec(**{"0":.22,"1":.12,"5":.18,"6":.24,"7":.28,"2":-.18,"3":-.14,"4":-.12}))
    rule([(k,-c) for k,c in texture],-.004,-.0012,vec(**{"2":.34,"3":.26,"4":.28,"9":.10,"0":-.12,"7":-.18}))
    knots=[]; blueknots=[]
    for ap in ("core","inner","ann1","ann2"):
        knots += [(("detail","knotL",0,ap),.25),(("detail","blobL",0,ap),.25)]
        blueknots += [(("detail","knotB",0,ap),.25),(("detail","blobB",0,ap),.25)]
    rule(knots,.001,.012,vec(**{"0":.30,"1":.26,"5":.10,"6":.12,"7":.20,"2":-.20}))
    rule(blueknots,.001,.012,vec(**{"0":.32,"1":.16,"5":.18,"6":.20,"7":.32,"8":.10,"2":-.22,"4":-.16}))

    # Orientation coherence of ridges supplies additional edge-on evidence.
    for o in range(nori):
        rt=[]
        for fam in ("ridge","valley"):
            rt += [(("detail",fam,o,"inner"),.5),(("detail",fam,o,"outer"),.5)]
        rule(rt,.002,.018,vec(**{"8":.045,"9":.045,"4":.025,"5":.020}))

    Wh=np.asarray(rows,np.float32); bh=np.asarray(biases,np.float32)
    Wo=np.asarray(outcols,np.float32).T
    add_dense(layers,weights,"rules","measurements_pos",Wh,bh)
    add_layer(layers,"rules_pos","relu","rules")
    # Frequency-aware priors, deliberately modest relative to morphology votes.
    # The rules contain unequal numbers of bounded votes per class. These
    # fixed analytic offsets center those rule families at a nominal moderate
    # activation (they are not estimated from images).
    bo=np.array([-7.2,-11.6,10.8,-3.0,-5.8,-7.5,-4.2,-3.8,-10.9,-12.0],np.float32)
    add_dense(layers,weights,"scores","rules_pos",Wo,bo)

    (folder/"network.json").write_text(json.dumps({"layers":layers},separators=(",",":")))
    np.savez(folder/"weights.npz",**weights)


if __name__=="__main__": build(sys.argv[1] if len(sys.argv)>1 else ".")
