"""Data-free build. Distances in the model descriptions are in arbitrary object units.
The x axis points towards the nose, y across the body and z upwards.
"""
import sys
from pathlib import Path
import numpy as np

class Drawing:
    def __init__(self):
        self.parts = []

    def oval(self, c, r, v, tilt=0, direction=None):
        # First semiaxis can also be aimed along an arbitrary limb segment.
        if direction is None:
            a = np.deg2rad(tilt)
            basis = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        else:
            d = np.asarray(direction, dtype=float); d /= np.linalg.norm(d)
            ref = np.array([0,1,0]) if abs(d[1]) < .9 else np.array([0,0,1])
            e = ref - (ref @ d)*d; e /= np.linalg.norm(e)
            basis = np.array([d,e,np.cross(d,e)])
        self.parts.append(('ell',np.array(c),np.array(r)[:,None]*basis,v))

    def limb(self, a, b, width, v, thick=None):
        a,b = np.array(a),np.array(b)
        self.oval((a+b)*.5,(np.linalg.norm(a-b)*.55,width,thick or width),v,direction=b-a)

    def poly(self, points, val):
        self.parts.append(('poly',np.array(points,dtype=float),None,val))

    def sections(self, sections, bottom, top, side, ends=None):
        # Horizontal rings specified by back x, front x, half width, height.
        rings = [np.array([[b,-w,z],[f,-w,z],[f,w,z],[b,w,z]]) for b,f,w,z in sections]
        self.poly(rings[0],bottom); self.poly(rings[-1],top)
        for j,(a,b) in enumerate(zip(rings,rings[1:])):
            for i in range(4):
                k=(i+1)%4
                color = (ends if i%2 and ends is not None else side)
                if isinstance(color,tuple): color=color[j]
                self.poly([a[i],a[k],b[k],b[i]],color)

    def box(self,x1,x2,y,z1,z2,v):
        self.sections([(x1,x2,y,z1),(x1,x2,y,z2)],v*.6,min(1.,v+.1),v,v*.93)

    def tire(self,x,y,z,r):
        s=np.sign(y); self.oval((x,y,z),(r,.10,r),.085)
        self.oval((x,y+.088*s,z),(r*.56,.025,r*.56),.44)


def car(var):
    d=Drawing(); body=[.76,.40,.65,.29][var%4]
    van=var%4==2; sport=var%4==3; wagon=var%4==1
    d.sections([(-.99,.96,.36,.19),(-1.03,1.03,.43,.30),(-1.,.99,.416,.48),(-.94,.90,.4,.55)],.18,body+.09,(body*.72,body,body*.96))
    rear= -.91 if wagon or van else -.70
    roofback=rear+(.09 if van or wagon else .33)
    front=.75 if van else .49
    rooft=.98 if van else (.76 if sport else .85)
    d.sections([(rear,front,.399,.535),(roofback,front-(.17 if van else .31),.32,rooft)],body,body+.10,.22,.32)
    for s in (-1,1):
        d.limb((rear+.58*(front-rear),s*.398,.54),(roofback+.54*(front-.23-roofback),s*.322,rooft),.025,body*.85)
        d.limb((rear,s*.4,.55),(roofback,s*.324,rooft),.028,body)
        for xx in (-.68,.67): d.tire(xx,s*.397,.223,.224)
        d.oval((.998,s*.30,.43),(.031,.09,.044),.88)
        d.oval((-.99,s*.33,.44),(.025,.065,.04),.37)
    d.box(.997,1.025,.26,.31,.39,.18)
    return d


def truck(var):
    d=Drawing(); bus=var%5==3; nose=var%5 in (1,4)
    d.box(-1.65,1.5,.375,.21,.40,.16)
    if bus:
        d.sections([(-1.62,1.43,.42,.34),(-1.65,1.46,.44,.82),(-1.62,1.39,.434,1.19),(-1.56,1.35,.42,1.25)],.22,.78,(.62,.25,.65))
        for xx in np.linspace(-1.6,1.35,8):
            for s in [-1,1]: d.limb((xx,s*.438,.82),(xx,s*.438,1.2),.025,.6)
    else:
        steel=[.82,.7,.5,.61,.50][var%5]; cab=[.49,.62,.79,.63,.44][var%5]
        if var%5==2:
            d.sections([(-1.58,.48,.42,.49),(-1.68,.55,.49,1.08)],.22,.47,steel)
        elif var%5==4:
            d.oval((-.58,0,.97),(1.12,.45,.44),steel)
        else: d.box(-1.73,.39,.47,.49,1.54,steel)
        f=1.12 if nose else 1.53
        d.sections([(.51,f,.45,.35),(.5,f,.459,.85),(.54,f-.09,.438,1.20),(.54,f-.12,.425,1.27)],.25,cab+.09,(cab,.21,cab), (cab*.8,.34,cab))
        if nose: d.box(f-.06,1.56,.386,.41,.85,cab)
        d.box(1.505,1.55 if not nose else 1.57,.29,.47,.72,.26)
        for s in [-1,1]:
            d.oval((1.54,s*.35,.60),(.03,.07,.05),.84)
            d.limb((.62,s*.459,.86),(.62,s*.44,1.21),.023,cab)
    for s in [-1,1]:
        for xx in ([-1.04,1.04] if bus else [-1.29,-.79,1.13]): d.tire(xx,s*.424,.251,.25)
    return d


def plane(var):
    d=Drawing(); prop=var%4==1; fighter=var%4==2
    val=[.79,.62,.49,.74][var%4]; d.oval((0,0,0),(1.28,.117,.134),val)
    d.oval((.87,0,.074),(.20,.108,.08),.30 if not fighter else .23)
    for s in [-1,1]:
        tipx= -.07 if prop else (-.82 if fighter else -.47)
        ty=1.18 if prop else (1.06 if fighter else 1.32)
        d.poly([(.35,s*.04,.015),(-.22,s*.07,.015),(tipx-.12,s*ty,.09),(tipx+.03,s*ty,.09)],val-.055)
        d.poly([(-.73,s*.02,.04),(-1.19,s*.03,.04),(-1.24,s*.46,.08),(-1.11,s*.47,.08)],val-.14)
        if not (prop or fighter):
            for y in ([.43,.75] if var%4==3 else [.47]): d.oval((.025-(y-.43)*.5,y*s,-.085),(.23,.075,.085),.51)
        if not prop:
            for x in np.arange(-.72,.82,.09): d.oval((x,s*.103,.04),(.021,.018,.021),.38)
    d.poly([(-.67,0,.06),(-1.20,0,.04),(-1.25,0,.47),(-1.15,0,.47)],.5 if not fighter else val)
    if prop: d.limb((1.21,0,-.25),(1.21,0,.25),.025,.32)
    return d


def boat(var):
    d=Drawing(); kind=var%6
    hull=[.28,.80,.72,.36,.78,.38][kind]
    pts=[]
    for z,w,back,front in [(0,.19,-.97,1.07),(.26,.32,-1.11,1.29)]:
        pts.append(np.array([[back,w*.72,z],[-.61,w,z],[.65,w,z],[front,0,z+.055], [.65,-w,z],[-.61,-w,z],[back,-w*.72,z]]))
    d.poly(pts[1],.54)
    for k in range(7):
        j=(k+1)%7;d.poly([pts[0][k],pts[0][j],pts[1][j],pts[1][k]],hull)
    if kind==1:
        d.limb((-.09,0,.27),(-.09,0,1.82),.018,.27)
        d.poly([(-.12,0,1.73),(-.12,0,.44),(-1.,.03,.48)],.9)
        d.poly([(.01,0,1.62),(.08,.03,.39),(1.06,0,.35)],.79)
    elif kind==0:
        d.box(-.94,-.46,.238,.26,.64,.83); d.box(-.86,-.54,.21,.64,.83,.69)
        d.box(-.76,-.65,.105,.80,.99,.30)
        for xx in [-.30,.06,.42]: d.box(xx,xx+.33,.23,.27,.45,.47+(xx+.3)*.23)
    elif kind==3:
        d.box(-.56,.70,.22,.26,.39,.53);d.box(-.30,.34,.18,.38,.68,.61)
        d.limb((.18,0,.65),(.15,0,1.19),.024,.41);d.box(-.40,-.25,.12,.39,.72,.38)
    elif kind==4:
        for i in range(3):
            z=.28+i*.14
            d.sections([(-.98+.12*i,.94-.19*i,.27-.026*i,z),(-.96+.12*i,.88-.19*i,.27-.026*i,z+.078),(-.96+.12*i,.86-.19*i,.27-.026*i,z+.14)],.5,.80,(.87,.37))
        d.box(-.35,-.11,.12,.68,.91,.31)
    else:
        d.sections([(-.72,.53,.239,.25),(-.52,.32,.21,.48),(-.43,.15,.17,.64)],.5,.83,(.73,.27))
        if kind==5: d.limb((-.35,0,.61),(-.34,0,1.05),.022,.33)
    return d


def ungulate(horse,var,portrait):
    d=Drawing(); r=var%4; coat=([.41,.28,.69,.52] if horse else [.53,.45,.61,.48])[r]; dark=coat*.50
    hip=1.09 if horse else 1.03
    d.oval((-.14,0,hip),(.77,.275,.323 if horse else .27),coat,-2)
    d.oval((-.71,0,hip-.02),(.35,.287,.34 if horse else .29),coat,-4)
    lower = var%5==4 and not portrait
    top=(.89,0,1.68 if horse else 1.64) if not lower else (1.03,0,.96)
    d.limb((.36,0,1.13),top,.181 if horse else .132,coat*.95,.264 if horse else .155)
    base=np.array(top); head=base+np.array([.092,0,.01]); muzzle=base+np.array([.42 if horse else .29,0,-.29 if horse else -.13])
    wid=.14 if horse else .117
    d.oval(head,(.218,wid,.183),coat,-38 if horse else -21)
    d.limb(head,muzzle,wid*.75,coat*1.03,.127 if horse else .092)
    d.oval(muzzle,(.12 if horse else .063,wid*.80,.093 if horse else .056),dark,-32)
    for s in [-1,1]:
        d.oval(head+[.051,s*wid*.90,.034],(.038,.022,.032),.075)
        ear=base+[-.007,s*wid*.55,.115]; end=ear+[-.06,s*(.055 if horse else .16),.19 if horse else .135]
        d.limb(ear,end,.047 if horse else .059,coat*.91,.052)
        d.oval(muzzle+[.055,s*wid*.61,.022],(.029,.019,.018),.13)
        if not horse and r==1 and not lower:
            end=base+[-.17,s*.24,.58];d.limb(base+[.02,s*.07,.115],end,.027,dark)
            for t in (.36,.65):
                p=base+[-.17*t,s*.24*t,.58*t]; d.limb(p,p+[.11,s*.03,.15],.021,dark)
    if horse:
        d.limb((.30,0,1.43),base+[-.10,0,.102],.069,dark*.82,.116)
        d.oval(base+[.12,0,-.025],(.17,.040,.21),min(1.,coat+.21),-23) if r==3 else None
    for s in [-1,1]:
        for back in [False,True]:
            x= -.71 if back else .41
            stride = ((-.14 if s==1 else .10) if back else (.11 if s==1 else -.07)) * (2.3 if r==3 else 1)
            knee=(x+stride*.40+(.045 if back else -.028),s*.169,.52)
            ankle=(x+stride+(-.14 if back else .02),s*.18,.10)
            d.limb((x,s*.15,1.11),knee,.103 if horse else .07,coat*.94,.135 if back else .092)
            d.limb(knee,ankle,.052 if horse else .033,coat*.75)
            d.oval((ankle[0]+.021,ankle[1],.067),(.081 if horse else .055,.048,.059),.19)
    if horse:
        d.limb((-.99,0,1.24),(-1.14,.023,.49),.09,dark*.78,.119)
    else:
        d.oval((-.99,0,1.08),(.17,.08,.088),coat+.18,-58)
    return d,tuple(head)


def pet(cat,var,portrait):
    d=Drawing(); r=var%6; coat=[.49,.69,.34,.68,.45,.64][r]; dark=coat*.49; white=min(.92,coat+.20)
    # The sitting body foreshortens when seen in a frontal portrait.
    sitting = portrait or r%3==1
    if sitting:
        d.oval((-.14,0,.69),(.74,.325,.388),coat,63)
        for s in (-1,1):
            d.oval((-.32,.22*s,.33),(.337,.21,.28),coat*.92,18)
            d.limb((.12,s*.189,.93),(.30,s*.171,.125),.097,coat)
            d.oval((.32,s*.17,.103),(.149,.107,.081),white if r%3==0 else coat)
        cx,cz=.18,1.35
    else:
        cz=1.07;cx=.62
        d.oval((-.045,0,.78),(.70,.256,.279),coat,-4)
        d.limb((.36,0,.79),(.59,0,1.07),.195,coat)
        for s in [-1,1]:
            for xx in [-.52,.38]:
                off=.09*s*(1 if xx>0 else -1); k=(xx+off,s*.15,.37)
                d.limb((xx,s*.135,.76),k,.094,coat);d.limb(k,(xx+off*1.6,s*.15,.11),.066,coat*.9)
                d.oval((xx+off*1.6+.036,s*.15,.078),(.107,.075,.065),coat)
    d.limb((-.52,0,.44 if sitting else .8),(-.83,.055,.41 if sitting else .97),.065 if cat else .106,coat*.84)
    d.limb((-.83,.055,.41 if sitting else .95),(-1.03,.18,(.78 if cat else .59) if sitting else 1.27),.057 if cat else .090,coat*.86)
    q=np.array([cx,0.,cz]); w=.304 if cat else (.27 if r==4 else .303)
    h=.267 if cat else .29
    def ell(off,rad,v,angle=0): d.oval(q+off,rad,v,angle)
    ell([0,0,0],[.26,w,h],coat)
    long=.09 if cat else [.21,.13,.24,.16,.27,.085][r]
    # Cheeks, muzzle bridge, nose leather and chin.
    ell([.194+long*.43,0,-.106],[long+.045,.129 if cat else .157,.100 if cat else .115],white if cat or r%2==1 else coat*.84,-8)
    ell([.205+long,0,-.096],[long*.52,.112 if not cat else .085,.077],white if cat else coat*.88)
    for s in [-1,1]:
        if cat: ell([.23+long*.50,s*.073,-.105],[.096,.096,.076],white)
        ell([.192,s*.153,.042],[.055,.072,.051 if not cat else .046],dark*.65,-5)
        ell([.228,s*.154,.044],[.030,.050,.032],.54 if cat else .20)
        ell([.249,s*.157,.045],[.012,.015 if cat else .030,.026],.095)
        ell([.255,s*.148,.058],[.010,.011,.009],.86)
        if cat or r in (2,4):
            out=.303 if cat else .246; high=.414 if cat else .51
            pts=np.array([[.104,.094*s,.193],[.037,.318*s,.112],[-.138,.247*s,.144],[-.055,out*s,high]])
            pts=pts+q
            for a,b in [(0,1),(1,2),(2,0)]:d.poly([pts[a],pts[b],pts[3]],coat*(.76 if a==0 else .94))
            d.poly([q+[.1,.15*s,.184],q+[.065,.27*s,.14],q+[-.013,(out-.016)*s,high-.067]],min(.92,coat+.093))
        else:
            a=q+[-.055,s*.216,.18];b=q+[-.031,s*.355,-.227 if r!=5 else -.08];d.limb(a,b,.087,dark,.171)
        if cat and r%3==0:
            for z in [-.065,.015]:d.limb(q+[.15,.227*s,z],q+[.028,.293*s,z+.065],.019,dark)
        if not cat and r==2: ell([.139,s*.184,.113],[.068,.056,.040],white)
    ell([.225+long*1.45,0,-.077],[.049,.075 if not cat else .047,.041 if not cat else .027],.14 if not cat else .33)
    return d,tuple(q)


def bird(var):
    d=Drawing(); r=var%5; coat=[.40,.59,.45,.69,.49][r]; fly=r==3; duck=r==2; long=r==4
    d.oval((-.10,0,.67),(.52,.232,.238),coat,18 if not duck else 4)
    hx=.27 if not duck else .38; hz=.995 if not duck else .94
    if long: hz=1.22;d.limb((.24,0,.77),(.37,0,1.17),.112,coat)
    d.oval((hx,0,hz),(.205,.181,.201),coat*.84,5)
    for s in [-1,1]:
        d.oval((hx+.066,s*.154,hz+.04),(.034,.026,.037),.09)
        if fly:
            d.poly([(.26,s*.07,.76),(-.35,s*.12,.69),(-.50,s*1.02,.81),(-.37,s*1.40,.88),(-.16,s*.73,.86)],coat*.80)
        else:
            d.oval((-.20,s*.153,.70),(.437,.108,.165),coat*.66,18)
        if not (duck or fly):
            d.limb((-.09,s*.091,.53),(-.18,s*.12,.24),.027,.33);d.limb((-.18,s*.12,.24),(-.02,s*.12,.215),.021,.34)
    # Beak taper and fan tail.
    for s in [-1,1]:
        d.poly([(hx+.15,s*.069,hz+.052),(hx+.16,s*.081,hz-.045),(hx+(.50 if long else .38 if duck else .32),0,hz-.015)],.53 if duck else .29)
    d.poly([(-.37,-.09,.65),(-.38,.10,.66),(-.92,.146,.50),(-.98,-.093,.49)],coat*.66)
    if r==1:d.oval((-.034,0,.576),(.35,.226,.149),.71,32)
    return d


def frog(var):
    d=Drawing(); r=var%4; coat=[.49,.39,.57,.44][r]
    d.oval((-.13,0,.327),(.61,.392,.305),coat,7)
    d.oval((.37,0,.43),(.414,.412,.236),coat,1)
    d.oval((.45,0,.36),(.345,.374,.13),coat+.11,-5)
    for s in [-1,1]:
        d.oval((.374,s*.274,.601),(.149,.138,.128),coat*.92)
        d.oval((.422,s*.346,.631),(.095,.065,.084),.58)
        d.oval((.443,s*.389,.638),(.064,.028,.031),.10)
        d.limb((.281,s*.23,.35),(.102,s*.498,.181),.103,coat)
        d.limb((.102,s*.498,.181),(.444,s*.49,.10),.063,coat*.93)
        d.limb((-.56,s*.190,.28),(-.225,s*.443,.236),.170,coat*.87)
        d.limb((-.224,s*.443,.227),(-.69,s*.480,.101),.093,coat*.96)
        for j in [-1,0,1]:
            d.limb((.421,s*.50,.113),(.62+.03*j,s*(.52+.10*j),.073),.023,coat)
            d.limb((-.64,s*.475,.1),(-.57,s*(.64+.055*j),.06),.024,coat)
        if r<2:
            d.limb((-.62,s*.175,.43),(.21,s*.260,.577),.032,coat+.12)
    return d


def render(d,ang,elev,roll,focus,span,shade,background,dim=64):
    a,e,r=np.deg2rad([ang,elev,roll]); eye=np.array([np.cos(a)*np.cos(e),np.sin(a)*np.cos(e),np.sin(e)])
    right=np.array([-np.sin(a),np.cos(a),0.]); up=np.cross(eye,right)
    basis=np.array([right*np.cos(r)+up*np.sin(r), up*np.cos(r)-right*np.sin(r),eye])
    parts=[]; bounds=[]
    for kind,p,axes,val in d.parts:
        pc=p @ basis.T
        if kind=='poly':
            bounds.append(pc[:,:2]); parts.append((kind,pc,None,val))
        else:
            mat=axes @ basis.T;rad=np.sqrt((mat**2).sum(axis=0)); bounds.append(np.stack((pc[:2]-rad[:2],pc[:2]+rad[:2])));parts.append((kind,pc,mat,val))
    bounds=np.concatenate(bounds,axis=0);low=bounds.min(axis=0);hi=bounds.max(axis=0)
    if focus is None: ctr=(low+hi)*.5;span=(hi-low).max()*1.13
    else: ctr=(np.array(focus) @ basis.T)[:2]
    # Sample at twice the resolution needed for the descriptor.
    step=span/dim; tx=ctr[0]+(np.arange(dim)+.5-dim/2)*step;ty=ctr[1]-(np.arange(dim)+.5-dim/2)*step
    img=np.full((dim,dim),background); depth=np.full((dim,dim),-1.e5)
    light=np.array([shade*.40,.68,.50]);light/=np.linalg.norm(light)
    for kind,p,mat,val in parts:
        if kind=='ell':
            rr=np.sqrt((mat**2).sum(axis=0));lo=p[:2]-rr[:2];hi=p[:2]+rr[:2]
        else: lo=p.min(axis=0)[:2];hi=p.max(axis=0)[:2]
        x0=np.searchsorted(tx,lo[0]);x1=np.searchsorted(tx,hi[0],side='right');y0=np.searchsorted(-ty,-hi[1]);y1=np.searchsorted(-ty,-lo[1],side='right');
        if x1<=x0 or y1<=y0: continue
        xx=tx[None,x0:x1]; yy=ty[y0:y1,None]
        local=depth[y0:y1,x0:x1]; pix=img[y0:y1,x0:x1]
        if kind=='ell':
            # Quadric form from the known orthogonal semiaxes (no fitting).
            dual=mat/(mat**2).sum(axis=1)[:,None]; q=dual.T @ dual
            x=xx-p[0];y=yy-p[1]; b=q[2,0]*x+q[2,1]*y; c=q[0,0]*x*x+2*q[0,1]*x*y+q[1,1]*y*y-1.
            disc=b*b-c*q[2,2];z=(-b+np.sqrt(np.maximum(disc,0.)))/q[2,2];dep=z+p[2];mask=(disc>=0)&(dep>local)
            nx=q[0,0]*x+q[0,1]*y+q[0,2]*z;ny=q[1,0]*x+q[1,1]*y+q[1,2]*z;nz=b+q[2,2]*z
            dot=(nx*light[0]+ny*light[1]+nz*light[2])/np.sqrt(nx*nx+ny*ny+nz*nz+1e-15)
            color=np.clip(val*(.70+.31*dot)+.055*np.maximum(dot,0)**5,.025,.97)
            pix[mask]=color[mask];local[mask]=dep[mask]
        else:
            for i in range(1,len(p)-1):
                v=p[i]-p[0]; w=p[i+1]-p[0];norm=np.cross(v,w);det=norm[2]
                if abs(det)<1e-9:continue
                x=xx-p[0,0];y=yy-p[0,1];u=(x*w[1]-y*w[0])/det;t=(y*v[0]-x*v[1])/det; dep=p[0,2]+u*v[2]+t*w[2]
                mask=(u>=0)&(t>=0)&(u+t<=1)&(dep>local);dot=(norm@light)*np.sign(det)/np.linalg.norm(norm)
                pix[mask]=np.clip(val*(.83+.18*dot),0.,1.);local[mask]=dep[mask]
    return img.reshape(32,2,32,2).mean((1,3)),(depth>-1.e4).reshape(32,2,32,2).mean((1,3))

# This numerical code is shared verbatim between template construction and recognition.
SHARED = '''import numpy as np


def blur(z):
    a=np.concatenate((z[..., :1,:],z,z[..., -1:,:]),axis=-2)
    a=(a[..., :-2,:]+2*a[..., 1:-1,:]+a[..., 2:,:])*.25
    b=np.concatenate((a[..., :1],a,a[..., -1:]),axis=-1)
    return (b[..., :-2]+2*b[..., 1:-1]+b[..., 2:])*.25


def gradients(z):
    a=np.concatenate((z[..., :1],z,z[..., -1:]),axis=-1)
    b=np.concatenate((z[..., :1,:],z,z[..., -1:,:]),axis=-2)
    return (a[..., 2:]-a[..., :-2])*.5,(b[..., 2:,:]-b[..., :-2,:])*.5


def orientations(z, color=False):
    gx,gy=gradients(.52*z+.48*blur(z)); xx=gx*gx;yy=gy*gy;xy=gx*gy
    if color:
        # Structure tensor: use achromatic edges and also isoluminant pigment changes.
        xx=.72*xx[:,0]+.17*(xx[:,1]+xx[:,2]);yy=.72*yy[:,0]+.17*(yy[:,1]+yy[:,2]);xy=.72*xy[:,0]+.17*(xy[:,1]+xy[:,2])
    v=xx+yy
    mg=np.sqrt(v+.00000001); a=(xx-yy)/(v+.00000001);b=2*xy/(v+.00000001)
    power=np.minimum(.34,mg)**.80
    channels=[]
    for k in range(6):
        ang=k*np.pi/3
        channels.append(power*np.maximum(0.,(a*np.cos(ang)+b*np.sin(ang)-.38)/.62))
    return np.stack(channels,axis=-3).astype(np.float32)


def grid1(count,center,span,n):
    x=np.arange(count);u=center+((np.arange(n)+.5)/n-.5)*span; r=max(span/n,.8)
    a=np.maximum(0,1-np.abs(u[:,None]-x[None,:])/r)
    # Replicated image boundary, including crops extending beyond the frame.
    a[:,0]+=np.maximum(0,1-(u+1)/r)*(u<r-1)
    a[:,-1]+=np.maximum(0,1-(count-u)/r)*(count-u<r)
    return (a/np.maximum(a.sum(1,keepdims=True),.001)).astype(np.float32)


def unit(z,reg):
    return z/np.sqrt((z*z).sum(axis=-1,keepdims=True)+reg*reg)


def descriptor(h,g,m):
    # h: six unsigned gradient bins; g: luminance; m: objectness, all on an 8 x 8 grid.
    n=g.shape[0]; energy=h.sum(1,keepdims=True)
    h=h/(.025+blur(energy)*.86+energy*.14)
    h=np.minimum(h,1.0);h=h-.68*h.mean(1,keepdims=True)
    h=h-.63*h.mean((2,3),keepdims=True)
    h=unit(h.reshape(n,-1),.55)
    slow=blur(blur(g)); detail=g-slow
    g=g-g.mean((1,2),keepdims=True)
    # Suppress smooth gradients, which can come from either lighting or the backdrop.
    g=unit((.41*g+1.4*detail).reshape(n,-1),.30)
    m=m-m.mean((1,2),keepdims=True);m=unit((m+.36*(m-blur(m))).reshape(n,-1),.85)
    return np.concatenate((h,g,m),axis=1).astype(np.float32)
'''


def makebank():
    ns={};exec(SHARED,ns);desc=ns['descriptor'];ori=ns['orientations'];g=ns['grid1'](32,15.5,32,8)
    banks=[]
    for cl in range(10):
        bank=[]
        for j in range(144):
            var=j%6
            az=(j*.61803398875)%1;el=(j*.41421356237+.31)%1;rot=(j*.754877666+.12)%1
            ang=az*(185 if cl in [0,1,8,9] else 118)-4
            portrait=cl in [3,4,5,7] and j%7 in ([0,1,2,3] if cl in [3,5] else [0,1]); focus=None; span=1.
            elev=2+el*19; roll=(rot-.5)*9
            if cl==0:
                d=plane(var);elev=-43+el*73;roll=(rot-.5)*49
            if cl==1: d=car(var);elev=2+el*22
            if cl==9: d=truck(var);elev=el*15
            if cl==8: d=boat(var);elev=el*18;roll=(rot-.5)*6
            if cl in (3,5):
                d,head=pet(cl==3,var,portrait);elev=el*33-2
                if portrait:
                    ang=az*85;focus=np.array(head)+[.035,0,-.04];span=.87+el*.40;roll=(rot-.5)*25
            if cl in (4,7):
                d,head=ungulate(cl==7,var,portrait);elev=el*14-4
                if portrait:
                    focus=np.array(head)+[.05,0,-.17];span=1.04+el*.40;ang=az*102
            if cl==2: d=bird(var);elev=-15+el*44;roll=(rot-.5)*29
            if cl==6: d=frog(var);elev=el*59+3;roll=(rot-.5)*46
            val,mask=render(d,ang,elev,roll,focus,span,1 if j%3 else -1,.65)
            for bg in [.25,.76]:
                im=(val+(1-mask)*(bg-.65)).astype(np.float32)
                hog=g @ ori(im) @ g.T; grey=g @ im @ g.T;sal=g @ mask @ g.T
                bank.append(desc(hog[None],grey[None],sal[None])[0])
        banks.extend(bank)
    return np.ascontiguousarray(np.stack(banks).T.astype(np.float32))


INFER='''

def sat(z):
    return np.clip(z,0.,1.)


def ramp(z,lo,hi):
    return sat((z-lo)/(hi-lo))


def summarize(z,w):
    return (z*w).sum((-2,-1))/(.00001+w.sum((-2,-1)))


def zones():
    y,x=np.mgrid[:32,:32];x=(x-15.5)/16;y=(y-15.5)/16
    c=np.exp(-2.3*(x*x+y*y));c=(c/c.max()).astype(np.float32)
    t=sat((-y-.12)*2.6).astype(np.float32);b=t[::-1];rim=(np.maximum(abs(x),abs(y))>.73).astype(np.float32)
    return c,t,b,rim


center,top,base,border=zones()

def objectmap(x,g,texture):
    sm=blur(blur(x));n=len(g);x8=sm.reshape(n,3,8,4,8,4).mean((3,5));border8=np.concatenate((x8[:,:,0,:],x8[:,:,-1,:],x8[:,:,1:-1,0],x8[:,:,1:-1,-1]),axis=2)
    yy,xx=np.mgrid[:32,:32];py,px=np.mgrid[:8,:8]*4+1.5;by=np.concatenate((py[0],py[-1],py[1:-1,0],py[1:-1,-1]));bx=np.concatenate((px[0],px[-1],px[1:-1,0],px[1:-1,-1]))
    dist=np.full(g.shape,100.,np.float32)
    for i in range(28):
        delta=sm-border8[:,:,i,None,None];dy=.25*delta[:,0]+.5*delta[:,1]+.25*delta[:,2]
        diff=dy*dy*.67+.56*((delta[:,0]-delta[:,1])**2+(delta[:,2]-delta[:,1])**2)
        spatial=((xx-bx[i])/32)**2+((yy-by[i])/32)**2
        dist=np.minimum(dist,diff*(1+spatial)+.004*spatial)
    dist=np.sqrt(dist+.00001);sal=dist/(dist+.115)
    # A very balanced texture field is weak evidence for a localized subject.
    return blur(sal*(.71+.29*center)+.085*center+.04*texture/(.08+texture)).astype(np.float32)


def windowgrids():
    wins=[(0,0,32,32,0),(0,0,25,25,.012),(0,0,39,39,.014),(0,0,32,25,.012),(0,0,27,34,.014),(0,0,20,20,.035)]
    for side in (25,33):
        for dx,dy in ((-3.5,-2.5),(3.5,-2.5),(-3.5,2.5),(3.5,2.5),(0,-3.6),(0,3.6)):
            wins.append((dx,dy,side,side,.016 if side==33 else .024))
    ax=np.stack([grid1(32,15.5+v[0],v[2],8) for v in wins]);ay=np.stack([grid1(32,15.5+v[1],v[3],8) for v in wins]);cost=np.array([v[4] for v in wins],dtype=np.float32)
    return ax,ay,cost


horizontalgrid,verticalgrid,windowcost=windowgrids()

def shapes(h,g,m):
    n=len(g); maps=np.concatenate((h,g[:,None],m[:,None]),axis=1)
    dlist=[]
    for i in range(len(windowcost)):
        patch=verticalgrid[i] @ maps @ horizontalgrid[i].T
        hh=patch[:,:6];gg=patch[:,6];mm=patch[:,7]
        dlist.append(descriptor(hh,gg,mm));dlist.append(descriptor(hh[:,[0,5,4,3,2,1],:,::-1],gg[:,:,::-1],mm[:,:,::-1]))
    features=np.stack(dlist); features[:,:,:384]*=.53;features[:,:,384:448]*=.29; features[:,:,448:]*=.18
    sim=(features.reshape(-1,512) @ CONSTANTS['models']).reshape(len(dlist),n,10,-1)
    sim=sim-np.repeat(windowcost,2)[:,None,None,None];best=sim.max(0).reshape(n,10,-1,2).max(3)
    # Combine a best pose with corroboration from a few distinct drawings.
    k=np.partition(best,-4,axis=2)[:,:,-4:];peak=.46*k.max(2)+.54*k.mean(2)
    return 12.*(peak-peak.mean(1,keepdims=True))


def symmetric(g,c):
    # Evaluate possible vertical axes; mostly useful for large frontal faces.
    n=len(g);best=np.zeros(n,np.float32)
    high=g-blur(blur(blur(g)));u=blur(g)
    for dx in [-3,0,3]:
        l=16+dx;w=min(l,32-l,12); a=high[:,4:27,l-w:l];b=high[:,4:27,l:l+w][:,:,::-1]
        r=(a*b).mean((1,2))/np.sqrt((.00035+(a*a).mean((1,2)))*(.00035+(b*b).mean((1,2))))
        a=u[:,4:27,l-w:l];b=u[:,4:27,l:l+w][:,:,::-1];color=1-sat(np.abs(a-b).mean((1,2))/.24)
        best=np.maximum(best,r*.77+.23*color)
    return ramp(best,.26,.70)


def eye_pairs(g):
    # Dark eye sockets bracket a lighter bridge, with a nose below the eyes.
    sm=blur(g); broad=blur(blur(blur(sm)));neg=sat((broad-sm-.009)/.13)
    best=np.zeros(len(g),np.float32);longnose=best.copy(); triangular=best.copy()
    for half in (2,3,4,5,6,7):
        for tilt in (-1,0,1):
            x0=max(half,8);x1=min(32-half,25);y0=5;y1=19
            left=neg[:,y0:y1,x0-half:x1-half]; right=neg[:,y0+tilt:y1+tilt,x0+half:x1+half]
            middle=sm[:,y0:y1,x0:x1];a=sm[:,y0:y1,x0-half:x1-half];b=sm[:,y0+tilt:y1+tilt,x0+half:x1+half]
            pair=np.sqrt(left*right)*sat((middle-(a+b)*.5+.01)/.13)*(1-.45*sat(abs(a-b)/.18))
            for ratio in (.78,1.38):
                off=int(round(half*ratio)); nose=neg[:,y0+off:y1+off,x0:x1]; score=pair*(.40+.60*nose)
                # Light fur on either side of a dark central nose.
                idx=max(1,half//2);cheeks=.5*(sm[:,y0+off:y1+off,x0-idx:x1-idx]+sm[:,y0+off:y1+off,x0+idx:x1+idx]);nr=sm[:,y0+off:y1+off,x0:x1]
                score=score*(.69+.31*sat((cheeks-nr)/.12));peak=score.max((1,2))
                if ratio<1: triangular=np.maximum(triangular,peak)
                else: longnose=np.maximum(longnose,peak)
    return triangular,longnose


def bands(gx,gy,g):
    ax=np.abs(gx);ay=np.abs(gy);em=ax+ay+.00001
    hh=np.maximum(0.,ay-1.25*ax);vv=np.maximum(0.,ax-1.25*ay)
    z=hh.mean(2);vm=vv.mean(1)
    hp=z[:,4:29].max(1); vp=vm[:,3:29].max(1)
    # Coherence along rows and columns, independent of overall contrast.
    yy=np.abs(gy.mean(2));xx=np.abs(gx.mean(1))
    hr=(yy[:,3:29].max(1)+.48*hp)/(.035+em.mean((1,2)))
    vr=(xx[:,3:29].max(1)+.55*vp)/(.035+em.mean((1,2)))
    # Directional structure at middle scales, without sensitivity to gradient sign.
    aa=blur(blur(gx*gx));bb=blur(blur(gy*gy));cc=blur(blur(gx*gy))
    geom=np.sqrt((aa-bb)**2+4*cc*cc); orient=geom/(aa+bb+.00045)
    rect=summarize((np.abs(aa-bb)-1.15*np.abs(2*cc))/(aa+bb+.00045),center)
    rect=ramp(rect,.015,.37)
    structured=ramp(summarize(orient,center),.35,.68)
    return ramp(hr,.65,2.),ramp(vr,.57,1.57),rect,structured


def wheel_pairs(g):
    # Circular dark tires with contrast to the road, repeated at the same height.
    n=len(g); yy,xx=np.mgrid[:32,:32];best=np.zeros(n,np.float32)
    for rad in (1.6,2.6,3.5):
        outer=[];inner=[];w=[]
        for j in range(12):
            ca=np.cos(np.pi*j/6);sa=np.sin(np.pi*j/6)
            def shifted(r):
                return g[:,np.clip(yy+int(round(r*sa)),0,31),np.clip(xx+int(round(r*ca)),0,31)]
            a=shifted(rad+.9);b=shifted(max(.8,rad-.52));outer.append(a);inner.append(b);w.append(1. if sa>=-.01 else .43)
        a=np.stack(outer); b=np.stack(inner); ww=np.array(w,dtype=np.float32)[:,None,None,None];diff=a-b
        contrast=(ww*np.clip(diff,-.09,.26)).sum(0)/sum(w)
        spokes=(ww*(diff>.025)).sum(0)/sum(w); dark=ramp(.57-b.mean(0),0,.40);hub=ramp(g-b.mean(0),-.01,.17)
        response=ramp(contrast,.019,.16)*ramp(spokes,.36,.80)*(.42+.39*dark+.19*hub)
        response=response[:,15:29,1:31]
        for dx in (int(rad*4.8),int(rad*6.3),int(rad*7.5)):
            l=response[:,:,:-dx];r=response[:,:,dx:];both=np.minimum(l,r)*.68+.32*np.sqrt(l*r)
            score=np.max(both,axis=(1,2)); off=np.maximum(np.minimum(l[:,1:],r[:,:-1]).max((1,2)),np.minimum(l[:,:-1],r[:,1:]).max((1,2)))
            best=np.maximum(best,np.maximum(score,.94*off))
    return ramp(best,.28,.82)


def rules(x,g,texture,m,gx,gy):
    n=len(g);r,gr,b=x[:,0],x[:,1],x[:,2];warm=r-b;maxrgb=x.max(1);chroma=maxrgb-x.min(1)
    den=.32+g; redgreen=(r-gr)/den; greenblue=(gr-b)/den; bluegreen=(b-gr)/den
    blue=sat(((-redgreen-1.8*greenblue)-.027)/.31)*ramp(g,.10,.32)
    green=sat((greenblue-.012)/.21)*sat(((-redgreen+.33*greenblue)+.006)/.115)
    brown=sat((redgreen-.016)/.19)*sat((greenblue+.012)/.19)
    vivid=ramp(chroma/(.48+g),.16,.42);neutral=1-ramp(chroma/(.33+g),.07,.27)
    white=ramp(g,.59,.87)*neutral; black=1-ramp(g,.105,.31)
    tx=ramp(texture,.012,.088);flat=1-ramp(texture,.012,.065)
    coldsky=blue*(.61+.39*flat); palesky=neutral*ramp(g,.56,.87)*flat
    vegetation=green*(.71+.29*tx)
    sweight=top+base+.18*border
    sb=summarize(coldsky,top);bb=summarize(blue,base);sg=summarize(vegetation,top);bg=summarize(vegetation,base);cg=summarize(green,center)
    br=summarize(brown,center);brbg=summarize(brown,base+.7*border);bn=summarize(brown,border)
    bt=summarize(coldsky,border); highsky=summarize(np.maximum(coldsky,.73*palesky),top)
    bluelower=summarize(blue*(.85+.15*tx),base);whitecenter=summarize(white,center);blackcenter=summarize(black,center)
    uncolored=summarize(neutral,center);paint=summarize(vivid*(1-green)*(1-brown),center)
    ft=summarize(flat,top);fb=summarize(flat,base);fc=summarize(flat,center);te=summarize(tx,center);tb=summarize(tx,border)
    # Distinguish a smooth sky down to the lower corners from a waterline.
    corners=summarize(flat,border); hline,vline,rect,structured=bands(gx,gy,g)
    wheel=wheel_pairs(g); sym=symmetric(g,center);catface,dogface=eye_pairs(g)
    central=summarize(m,center); mid=m[:,6:26,6:26].mean((1,2));mass=ramp(mid,.29,.65)
    # Vertical section through the body and its support plane.
    edgeabs=np.abs(gx)+np.abs(gy);contrast=g.std((1,2));brighttop=summarize(g,top);brightbase=summarize(g,base)
    lightground=summarize(white,base);darkground=summarize(black,base);blkc=summarize(black,center);neutbase=summarize(neutral,base)
    yd=np.arange(32,dtype=np.float32)[:,None];xd=yd.T
    loc=np.maximum(0.,m-(m.mean((1,2),keepdims=True)+.012));tot=loc.sum((1,2))+.03
    mx=(loc*xd).sum((1,2))/tot;my=(loc*yd).sum((1,2))/tot
    sx=np.sqrt((loc*(xd-mx[:,None,None])**2).sum((1,2))/tot+.01);sy=np.sqrt((loc*(yd-my[:,None,None])**2).sum((1,2))/tot+.01)
    wide=ramp((sx+1)/(sy+1),1.06,1.93);low=ramp(my,15.5,21.5)
    # Aerial background usually has very little fine-scale clutter anywhere.
    skyall=summarize(np.maximum(coldsky,.62*palesky),border);grassy=.55*bg+.28*sg+.17*cg
    water=bb*(.68+.32*(1-bg));outdoor=np.maximum(highsky,grassy)
    # Fur and feathers scatter contrasts in different orientations. A glossy body or a cabin has large panels.
    nature=.5*grassy+.33*brbg+.17*cg
    panels=fc*(.35+.65*structured); angular=.44*rect+.36*hline+.20*vline
    metallic=.39*angular+.27*wheel+.12*uncolored+.12*paint+.10*panels
    # Center color relative to the surroundings separates a brown animal from leaf litter.
    tananimal=br*(.56+.44*ramp(br-bn,-.08,.29));allbrown=.55*brbg+.45*br
    greenbody=cg*(.64+.36*ramp(cg-bg,-.13,.19))
    lf=summarize(flat,base)*neutbase
    textureout=ramp(tb,.17,.60); smoothout=ramp(corners,.38,.80)
    richdog=summarize(brown*(1-ramp(g,.55,.88)),center)
    shadowroad=summarize(black[:,15:28,:],center[15:28,:])
    # Animal vs. machine evidence; centered about broad hand-selected neutral conditions.
    hard=(rect-.37)*.92+(hline-.35)*.72+(vline-.36)*.44+(wheel-.37)*.82+(uncolored-.53)*.40+(paint-.06)*1.30-(nature-.21)*1.60
    pet=(sym-.27)*.56+(br-.24)*.46+(te-.47)*.28-(highsky-.34)*.85-(bb-.13)*.66
    face=np.maximum(catface,dogface);sc=np.zeros((n,10),np.float32)
    sc[:,0]=.20*hard+2.9*(skyall-.24)+1.6*(smoothout-.41)+.65*(wide-.43)+.66*(bt-.19)-1.6*(water-.17)*(1.15-fb)-.95*(hline-.39)-.61*(vline-.34)-1.35*(bg-.21)-.68*(br-.25)-.56*(wheel-.34)
    sc[:,8]=.38*hard+3.15*(water-.16)+1.25*(highsky-.34)+1.07*(hline-.37)+.50*(wide-.43)+.52*(low-.33)-1.1*(bg-.21)-.87*(wheel-.37)-.54*(smoothout-.4)*bt
    road=1.08*(lf-.26)+.40*(neutbase-.51)-.84*(bb-.15)-.54*(bg-.22)
    sc[:,1]=1.03*hard+road+.96*(wheel-.36)+.64*(wide-.45)+.51*(shadowroad-.21)-.53*(highsky-.36)-.42*(vline-.39)+.35*(paint-.08)
    sc[:,9]=1.12*hard+.74*road+.58*(wheel-.37)+.64*(vline-.38)+.63*(highsky-.34)+.44*(whitecenter-.19)+.46*(rect-.40)-.30*(wide-.45)
    sc[:,2]=-.38*hard+.82*(smoothout-.4)+1.05*(sb-.17)+.9*(sg-.25)+.62*(paint-.08)-.43*(bb-.17)+.48*(fc-.41)-.54*(br-.29)-.39*(sym-.30)-.47*(wide-.45)
    sc[:,3]=-.49*hard+.94*pet+.57*(catface-.46)+.60*(sym-.29)+.69*(textureout-.51)-.82*(grassy-.23)+.31*(brbg-.26)-.47*(wide-.45)-.33*(tb-te)
    sc[:,5]=-.45*hard+.93*pet+.68*(dogface-.45)+.56*(sym-.29)+.65*(richdog-.22)+.45*(blackcenter+whitecenter-.41)-.49*(wide-.45)-.31*(bt-.18)
    sc[:,4]=-.55*hard+2.02*(grassy-.24)+.74*(tananimal-.22)+.70*(textureout-.50)+.62*(brbg-.27)-.61*(paint-.09)-.41*(sym-.31)-.50*(highsky-.32)
    sc[:,7]=-.37*hard+1.36*(bg-.24)+1.33*(tananimal-.21)+.61*(highsky-.33)+.41*(contrast-.20)/.20-.60*(cg-.24)-.28*(sym-.30)-.32*(brbg-.24)
    sc[:,6]=-.56*hard+1.95*(greenbody-.17)+.97*(allbrown-.26)+.78*(brbg-.26)+.84*(textureout-.50)-1.43*(highsky-.33)-.49*(sym-.30)-.39*(whitecenter-.16)+.24*(wide-.45)
    return sc


def decide(x):
    g=.27*x[:,0]+.51*x[:,1]+.22*x[:,2]; comp=np.stack((g,x[:,0]-x[:,1],x[:,2]-x[:,1]),axis=1)
    h=orientations(comp,True);gx,gy=gradients(.52*g+.48*blur(g));texture=blur(blur(np.sqrt(gx*gx+gy*gy+.00000001)))
    m=objectmap(x,g,texture)
    return (shapes(h,g,m)+rules(x,g,texture,m,gx,gy)).argmax(1)


def predict(x):
    # Fixed work blocks also keep matrix operations identical for singleton and long batches.
    ans=np.empty(len(x),np.int64)
    for start in range(0,len(x),16):
        a=x[start:start+16];k=len(a);work=np.zeros((16,3,32,32),np.float32);work[:k]=a
        ans[start:start+k]=decide(work)[:k]
    return ans
'''


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(SHARED+INFER,encoding='utf-8')
    np.savez(out/'constants.npz',models=makebank())

if __name__=='__main__': main()
