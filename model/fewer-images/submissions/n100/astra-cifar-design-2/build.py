"""A data-free, hand specified shape-and-scene classifier.
Only the standard library and numpy. There is no estimation or example selection.
"""
import json, sys, math
from pathlib import Path
import numpy as np

F=np.float32
L=[]; W={}
def layer(name,op,src,**kw):
    L.append(dict(name=name,op=op,inputs=src if isinstance(src,list) else [src],**kw)); return name
def affine(name,src,w,b=0,conv=False,**kw):
    w=np.asarray(w,dtype=F); W[name+'.W']=w; W[name+'.b']=np.broadcast_to(np.asarray(b,dtype=F),(len(w),)).copy()
    return layer(name,'conv' if conv else 'dense',src,out=len(w),**({'kernel':list(w.shape[2:]),'padding':0,'stride':1}|kw if conv else kw))
def pool(name,src,size,stride=None,maximum=False):
    if isinstance(size,int): size=[size,size]
    if stride is None: stride=size
    if isinstance(stride,int): stride=[stride,stride]
    return layer(name,'maxpool' if maximum else 'avgpool',src,size=size,stride=stride)
def relu(name,src): return layer(name,'relu',src)
def norm(name,src): return layer(name,'norm',src)
def reshape(name,src,shape): return layer(name,'reshape',src,shape=shape)
def add(name,*src): return layer(name,'add',list(src))

# Painter's algorithm on the faces of hand-constructed solids. Units are object
# lengths (arbitrary), with x forward, y left, z up. No external 3D assets.
rng=np.random.default_rng(385103)
def U(a,b): return rng.uniform(a,b)
def choice(a): return a[rng.integers(len(a))]
class Model:
    def __init__(self):self.groups=[]
    def mesh(self,faces,tone): self.groups.append((np.asarray(faces),tone))
    def face(self,p,t):
        p=np.asarray(p);self.mesh([p[[0,i,i+1]] for i in range(1,len(p)-1)],t)
    def slab(self,profile,width,t,y0=0):
        # Outline in the sagittal plane with individually variable half-widths.
        p=np.asarray(profile); width=np.broadcast_to(width,(len(p),))
        q=np.stack([p[:,0],width+y0,p[:,1]],axis=1); r=q.copy();r[:,1]=y0-width
        self.face(q,t);self.face(r[::-1],t)
        for i in range(len(p)):j=(i+1)%len(p);self.face([q[i],q[j],r[j],r[i]],t)
    def box(self,lo,hi,t):
        a,b,c=lo;d,e,f=hi; self.slab([(a,c),(d,c),(d,f),(a,f)],(e-b)*.5,t,(e+b)*.5)
    def oval(self,p,r,t,angle=0):
        v=SPHERE*r
        if angle:
            a=np.deg2rad(angle);xx=v[:,0]*np.cos(a)-v[:,2]*np.sin(a);v[:,2]=v[:,0]*np.sin(a)+v[:,2]*np.cos(a);v[:,0]=xx
        self.mesh((v+p)[BALLF],t)
    def limb(self,A,B,r0,r1,t):
        A=np.asarray(A);B=np.asarray(B);d=B-A;d=d/max(np.linalg.norm(d),1.e-8);ax=np.cross(d,[0,0,1] if abs(d[2])<.8 else [1,0,0]);ax/=np.linalg.norm(ax);ay=np.cross(d,ax)
        circ=CS[:,0,None]*ax+CS[:,1,None]*ay;v=A+r0*circ;w=B+r1*circ
        self.mesh(np.stack([v,np.roll(v,1,axis=0),w],axis=1),t);self.mesh(np.stack([np.roll(v,1,axis=0),np.roll(w,1,axis=0),w],axis=1),t)
    def wheel(self,x,y,z,r,t=.075):
        disk=CS*np.array([r,r]);side=np.sign(y);th=.09
        q=np.stack([x+disk[:,0],np.full(len(CS),y),z+disk[:,1]],1);back=q.copy();back[:,1]-=th*side
        self.face(q,t)
        for i in range(len(q)):j=(i+1)%len(q);self.face([q[i],q[j],back[j],back[i]],t)
        self.face((q-[x,y,z])*.51+[x,y+side*.003,z],U(.33,.66))

ang=np.arange(12)*2*np.pi/12; CS=np.stack([np.cos(ang),np.sin(ang)],axis=1)
lon=np.arange(8)*2*np.pi/8;lat=np.linspace(-np.pi/2,np.pi/2,6)
SPHERE=np.array([[np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)] for a in lat for b in lon]); BALLF=np.array([[row*8+i,row*8+(i+1)%8,(row+1)*8+i] for row in range(5) for i in range(8)]+[[row*8+(i+1)%8,(row+1)*8+(i+1)%8,(row+1)*8+i] for row in range(5) for i in range(8)])

def vehicles(k,variant):
    m=Model();body=U(.16,.91);glass=U(.12,.44);f=.43
    if k==1:
        # Bonnet, cabin, boot; opaque body pillars divide the window band.
        van=variant%5==4; front=U(.37,.61); back=U(-.87,-.51);h=U(.67,.85)+.16*van
        h0=U(.37,.46);r=U(.19,.24);f=U(.365,.46)
        m.slab([(-1,.21),(-.92,.15),(.89,.15),(1.01,.22),(.98,h0-.05),(.86,h0),(-.82,h0+.02),(-1,h0-.02)],[.88*f,.95*f,.95*f,f,f,f,f,.94*f],body)
        a=back+.15;b=front-.22
        m.slab([(back-.1,h0),(front+.11,h0),(b,h),(a,h+.01)],[f*.99,f*.99,f*.78,f*.79],body)
        for s in [-1,1]:
            def side(p):return [(x,s*(f*(.99-.2*(z-h0)/(h-h0))+.006),z) for x,z in p]
            for lo,hi,tl,tr in [(back-.05,(a+b)*.5-.026,a+.012,(a+b)*.5-.035),((a+b)*.5+.012,front+.058,(a+b)*.5+.004,b-.01)]:
                m.face(side([(lo,h0+.023),(hi,h0+.023),(tr,h-.027),(tl,h-.027)]),glass)
        for x0,x1 in [(front+.115,b+.005),(back-.105,a-.005)]:
            m.face([(x0,-.91*f,h0+.022),(x0,.91*f,h0+.022),(x1,.74*f,h-.019),(x1,-.74*f,h-.019)],glass*1.07)
        for x in [-.64,.66]:
            for s in [-1,1]:m.wheel(x,s*(f+.006),r,r)
        for s in [-1,1]:m.box((.979,min(s*.23,s*.38),.28),(1.011,max(s*.23,s*.38),.36),.85)
        m.box((.991,-.19,.23),(1.014,.19,.29),.14);m.box((-.996,-.105,.23),(-1.01,.105,.29),.82)
    elif k==9:
        box=variant%4!=3;cab=U(.16,.47);h=U(.70,1.03);f=U(.29,.41)
        m.box((-.99,-f*.82,.15),(1,f*.82,.30),.13)
        cargo=choice([body,U(.66,.97),U(.3,.61)]);m.box((-1.03,-f,.33),(cab-.055,f,h+U(.02,.27) if box else .49),cargo)
        m.slab([(cab,.25),(1,.25),(1,.55),(.88,h),(cab,h)],[f,f,f,f*.9,f*.96],body)
        for s in [-1,1]:m.face([(cab+.045,s*(f+.004),.56),(.955,s*(f+.003),.56),(.858,s*(f*.92+.007),h-.047),(cab+.045,s*(f*.97+.004),h-.047)],glass)
        m.face([(.995,-f*.94,.565),(.995,f*.94,.565),(.89,f*.84,h-.046),(.89,-f*.84,h-.046)],glass)
        m.box((.996,-f*.53,.31),(1.007,f*.53,.48),.17)
        for s in [-1,1]:
            for x in ([-.74,-.50,.70] if variant%3 else [-.68,.69]):m.wheel(x,s*(f+.012),.159,.166)
    elif k==8:
        # Tapered bow, nearly flat stern and a displaced deckhouse or sail.
        deck=U(.25,.39);beam=U(.26,.41); hull=choice([U(.12,.33),U(.65,.91)])
        m.slab([(-1,deck),(-.84,.05),(-.55,0),(.59,.015),(.89,.12),(1.06,deck+.065),(.43,deck)],np.array([.72,.66,.64,.58,.45,.015,1.0])*beam,hull)
        if variant%4==0:
            mast=U(-.2,.1);ht=U(1.25,2.25);m.limb((mast,0,deck),(mast,0,ht),.014,.009,.22)
            m.face([(mast+.025,0,ht-.08),(.85,.02,deck+.12),(mast+.025,.075,deck+.14)],U(.72,.99));m.face([(mast-.025,.02,ht),(-.72,-.12,deck+.19),(mast-.025,0,deck+.2)],U(.73,.98))
        else:
            off=U(-.48,.32);w=U(.21,.55); ht=U(.23,.43);shell=U(.55,.99)
            m.box((off-w,-beam*.78,deck-.01),(off+w,beam*.78,deck+ht),shell)
            for s in [-1,1]:m.box((off-w*.88,min(s*beam*.79,s*beam*.8),deck+ht*.57),(off+w*.88,max(s*beam*.79,s*beam*.8),deck+ht*.85),.25)
            if variant%4==1:m.box((off-w*.43,-.12,deck+ht),(off+w*.43,.12,deck+ht+.15),shell)
            m.limb((off,0,deck+ht),(off-.07,0,deck+ht+U(.22,.52)),.022,.012,.23)
    else:
        # A spindle fuselage, swept mainplane, tailplane and vertical stabilizer.
        r=U(.095,.158);m.oval((0,0,.0),(1.03,r,r),body);m.oval((.56,0,.07),(.23,r*.93,r*.64),glass)
        span=U(.87,1.31);sw=U(.14,.60)
        for s in [-1,1]:
            m.face([(.33,s*.04,0),(.25-sw,s*span,.05),(.10-sw,s*span,.044),(-.31,s*.04,-.023)],body)
            m.face([(-.60,0,.05),(-.87,s*.39,.075),(-1.01,s*.39,.065),(-.87,0,.025)],body)
            if variant%3!=0:m.oval((.03,s*.46,-.075),(.19,.075,.077),body*.89)
        m.face([(-.58,0,.065),(-.89,0,U(.32,.49)),(-.996,0,.43),(-.95,0,.045)],body*.95)
    return m

def creature(k,variant):
    m=Model(); coat=U(.20,.76); light=min(1,coat*U(1.08,1.55)); dark=coat*U(.32,.73); eye=.055
    el=m.oval; rod=m.limb
    if k in (4,7) or (k in (3,5) and variant%4==0):
        # Standing quadruped in body coordinates; legs articulated independently.
        equid=k==7;cervid=k==4; tall=equid or cervid
        center=.92 if tall else .59; deep=.30 if equid else .24 if cervid else .26
        el((-.07,0,center),(.72,.24,deep),coat);el((.40,0,center+.005),(.25,.225,deep*1.05),coat)
        el((-.55,0,center+.015),(.28,.22,deep*.96),coat)
        for s in [-1,1]:
            for x in [-.53,.43]:
                d=U(-.22,.24);knee=(x+(-.10 if x>0 else .11)+d*.6,s*.158,center*.45);foot=(x+d,s*.185,.055)
                rod((x,s*.14,center),knee,.108,.058,coat*(1 if s<0 else .79));rod(knee,foot,.058,.037,dark if equid else coat*.79);el(foot,(.071,.051,.055),dark)
        grazing=variant%7==5 and tall
        hh=(.99,0,center+(.64 if tall else .33)*U(.77,1.16)) if not grazing else (1.0,0,center*.47)
        rod((.40,0,center+.06),hh,.244 if equid else .17,.124,coat);x,y,z=hh
        el((x+.025,0,z),(.21,.134,.158),coat,-26 if equid else -8)
        el((x+.167,0,z-.09 if equid else z-.042),(.19 if equid else .135,.113,.107),coat,-34 if equid else -10)
        for s in [-1,1]:
            el((x+.089,s*.115,z+.035),(.027,.031,.032),eye)
            el((x-.042,s*.076,z+.146),(.062,.042,.13 if tall else .074),coat,20)
            if cervid and variant%4==2:
                rod((x-.05,s*.076,z+.13),(x-.19,s*.19,z+.44),.026,.011,dark);rod((x-.14,s*.16,z+.32),(x-.055,s*.20,z+.43),.015,.007,dark)
        if equid:
            rod((-.69,0,center+.14),(-.90,0,center-.18),.078,.096,dark);rod((-.90,0,center-.18),(-.91,.02,.25),.096,.03,dark)
            rod((.37,0,center+deep*.89),(x-.06,0,z+.084),.093,.064,dark)
        elif not cervid:rod((-.71,0,center+.04),(-1.04,0,center+.34),.075,.025,coat)
        else:el((-.73,0,center+.07),(.09,.09,.13),light,-15)
    elif k in (3,5):
        cat=k==3; # Frontal and oblique portraits / sitting poses.
        face=variant%4!=2;h=1.05 if face else .91
        el((-.25,0,.38),(.42,.40,.59 if face else .55),coat,-14);el((-.44,0,.15),(.47,.41,.31),coat)
        el((.09,0,.48),(.24,.31,.41),light)
        for s in [-1,1]:
            rod((.065,s*.215,.47),(.23,s*.235,-.19),.128,.08,coat);el((.25,s*.235,-.195),(.14,.098,.069),light)
        el((.13,0,h),(.33,.43 if cat else .36,.37),coat)
        for s in [-1,1]:
            if cat or variant%5==3:
                # Triangular pinnae with a second inset facet.
                A=(-.014,s*.14,h+.22);B=(.16,s*.407,h+.17);C=(.02,s*.377,h+U(.44,.59));m.face([A,B,C],coat);m.face([A,C,(-.13,s*.29,h+.12)],coat*.88)
                m.face([(.09,s*.204,h+.245),(.16,s*.366,h+.225),(.057,s*.36,h+.443)],dark)
            else:el((-.045,s*.328,h+.015),(.18,.125,U(.24,.42)),dark,-12)
            el((.382,s*.182,h+.067),(.055,.063 if cat else .067,.049),eye);el((.419,s*.183,h+.079),(.015,.019,.013),.81)
            if not cat and variant%3==2:el((.306,s*.20,h+.15),(.064,.089,.044),light)
        for s in [-1,1]:el((.413 if cat else .442,s*.088,h-.099),(.107 if cat else .187,.132,.108 if cat else .14),light)
        el((.522 if cat else .626,0,h-.047),(.038 if cat else .063,.059 if cat else .096,.038 if cat else .064),.15 if cat else .07)
    elif k==2:
        flying=variant%4==0
        if not flying:
            a=U(17,55);el((0,0,.42),(.49,.217,.28),coat,a);el((.255,0,.69),(.22,.18,.223),coat)
            for s in [-1,1]:
                el((-.074,s*.148,.44),(.382,.089,.161),dark,a*.68);el((.342,s*.14,.735),(.035,.031,.033),eye)
                rod((-.05,s*.083,.25),(-.061,s*.104,.052),.024,.017,dark);rod((-.06,s*.104,.055),(.04,s*.11,.033),.018,.009,dark)
            m.face([(-.20,-.12,.35),(-.78,-.09,.12),(-.76,.09,.12),(-.20,.12,.35)],dark)
            for s in [-1,1]:m.face([(.41,0,.75),(.427,s*.092,.665),(.65,0,.683)],light)
        else:
            el((0,0,0),(.48,.122,.145),coat);el((.37,0,.10),(.139,.111,.127),coat)
            for s in [-1,1]:
                up=U(-.18,.40);m.face([(.23,0,.035),(.17,s*.5,up+.14),(-.23,s*1.03,up+.30),(-.36,s*.63,up),(-.24,0,-.04)],coat)
                m.face([(-.28,0,.025),(-.57,s*.19,.04),(-.64,0,.05)],dark)
    else:
        # Broad low head, prominent eyes, bent thighs and long folded shins.
        el((-.15,0,.25),(.52,.34,.249),coat,9);el((.245,0,.33),(.38,.36,.212),coat);el((.45,0,.335),(.26,.307,.137),coat)
        for s in [-1,1]:
            el((.32,s*.216,.478),(.127,.114,.127),coat);el((.393,s*.263,.508),(.062,.060,.057),eye);el((.427,s*.262,.53),(.019,.016,.016),.86)
            el((-.37,s*.296,.19),(.322,.17,.185),coat,17);rod((-.57,s*.34,.12),(-.18,s*.51,.07),.089,.048,light)
            rod((.23,s*.23,.30),(.01,s*.44,.13),.086,.062,coat);rod((.01,s*.44,.13),(.28,s*.51,.035),.062,.040,light)
            for i in range(3):rod((.28,s*.51,.035),(.40+.035*(i==1),s*(.46+.055*i),.024),.023,.012,light)
        if variant%2:
            for i in range(6):
                a=U(-.42,.13);b=U(-.19,.19);el((a,b,.25+.24*np.sqrt(max(.05,1-((a+.15)/.52)**2-(b/.34)**2))),(.065,.047,.022),dark)
    return m

# A flat-faceted rasterizer; the silhouette and the arrangement of brighter and
# darker surfaces are more useful here than photorealism. All variation is a
# priori pose/lighting/anatomy variation; nothing is measured across templates.
RES=48; Ygrid,Xgrid=np.mgrid[:RES,:RES]
def render(k,i):
    m=vehicles(k,i) if k in (0,1,8,9) else creature(k,i)
    profile=U(0,2*np.pi); elev=U(.055,.40);roll=U(-.065,.065)
    if k==0:
        elev=choice([U(-1.08,-.12),U(.08,.76)]);roll=U(-.60,.60)
    if k==8:elev=U(.035,.52)
    if k==6:elev=U(.11,1.08);roll=U(-.40,.40)
    if k in (3,5) and i%4!=0:profile=U(-1.17,1.17);elev=U(-.06,.28)
    if k==2: elev=U(-.45,.41);roll=U(-.25,.25)
    c=np.array([np.cos(elev)*np.cos(profile),np.cos(elev)*np.sin(profile),np.sin(elev)]);right=np.array([-np.sin(profile),np.cos(profile),0]);up=np.cross(c,right)
    proj=np.stack([np.cos(roll)*right+np.sin(roll)*up,np.sin(roll)*right-np.cos(roll)*up,c],axis=1)
    tri=np.concatenate([z[0] for z in m.groups]);tones=np.repeat([z[1] for z in m.groups],[len(z[0]) for z in m.groups]);T=tri@proj
    p=T[:,:,:2]; low=p.min((0,1));high=p.max((0,1));sz=U(.67,1.03)
    if k in (2,4):sz=U(.50,.95)
    if k==6:sz=U(.65,1.12)
    if k in (3,5) and i%4==1:sz=U(1.18,1.86)
    off=np.array([U(-.038,.038), U(-.036,.036)])*RES
    if k in (3,5) and i%4==1:off[1]+=(sz-1)*.45*RES
    p[:]=(p-(low+high)/2)*(RES*sz/max(high-low))+RES*.5+off
    # Simply lit facets. Use the view-facing normal, including for open planes.
    nr=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);nr/=np.linalg.norm(nr,axis=1)[:,None]+1.e-12;nr*=np.where((nr@c)>0,1.,-1.)[:,None]
    sun=np.array([U(-.7,.7),U(-.7,.7),1.]);sun/=np.linalg.norm(sun);lighting=.79+.27*np.maximum(0,nr@sun)+.08*(nr@up)
    tone=np.clip(tones*lighting,0,1);bg=choice([U(.13,.37),U(.42,.69),U(.72,.97)]); img=np.full((RES,RES),bg)
    # Vectorize the coverage computations in small batches; resolve visible
    # triangles with a z-buffer (important at joints and at the wheels).
    depth=np.full((RES,RES),-1.e6); xx=Xgrid+.5;yy=Ygrid+.5
    for J in range(0,len(T),48):
        Q=T[J:J+48];x0,y0,z0=Q[:,0,:,None,None].transpose(1,0,2,3); dx1,dy1,dz1=(Q[:,1]-Q[:,0])[:,:,None,None].transpose(1,0,2,3);dx2,dy2,dz2=(Q[:,2]-Q[:,0])[:,:,None,None].transpose(1,0,2,3)
        # Restrict vectorized work to a bounding rectangle for this group.
        xa=max(0,int(np.floor(Q[:,:,0].min())));xb=min(RES,int(np.ceil(Q[:,:,0].max())));ya=max(0,int(np.floor(Q[:,:,1].min())));yb=min(RES,int(np.ceil(Q[:,:,1].max())))
        if xa>=xb or ya>=yb:continue
        sl=np.s_[ya:yb,xa:xb];dx=xx[sl]-x0;dy=yy[sl]-y0;det=dx1*dy2-dx2*dy1;det=np.where(abs(det)<1.e-11,1.e20,det);a=(dx*dy2-dy*dx2)/det;b=(dy*dx1-dx*dy1)/det
        z=np.where((a>=0)&(b>=0)&(a+b<=1)&(abs(det)<1.e19),z0+a*dz1+b*dz2,-1.e7);ix=z.argmax(0);zd=np.take_along_axis(z,ix[None],axis=0)[0];vis=zd>depth[sl];depth[sl][vis]=zd[vis];img[sl][vis]=tone[J+ix[vis]]
    return img

# Matched filter bank. One drawing gives one filter. Its components encode
# centered luminance, center/surround contrast and four oriented contour maps.
# The exact same linear operators + rectifiers are used on the incoming photo.
P=16
idx=np.arange(P*P).reshape(P,P)
def st(dy,dx):return np.eye(P*P)[idx[np.clip(np.arange(P)+dy,0,P-1)[:,None],np.clip(np.arange(P)+dx,0,P-1)[None,:]].ravel()]
I=st(0,0)
blur=sum(w*h*st(i,j) for i,w in [(-2,1),(-1,4),(0,6),(1,4),(2,1)] for j,h in [(-2,1),(-1,4),(0,6),(1,4),(2,1)])/256
D=[(st(dy,dx)-st(-dy,-dx))/(1. if dx==0 or dy==0 else 1.414214) for dy,dx in [(0,1),(1,1),(1,0),(1,-1)]]
sm=sum(w*h*st(i,j) for i,w in [(-1,1),(0,2),(1,1)] for j,h in [(-1,1),(0,2),(1,1)])/16
angmix=np.eye(4)-.15*np.ones((4,4))
def feature_branch():
    smal=pool('retina','image',2)
    gray=affine('luma',smal,np.array([.29,.58,.13]).reshape(1,3,1,1),conv=True)
    g1=norm('global_contrast',gray)
    hp=affine('retina_detail',gray,I-blur); hp=norm('detail_contrast',reshape('detail_field',hp,[1,P,P]))
    al=affine('oriented_differences',gray,np.concatenate(D+[-d for d in D]));al=relu('half_waves',al)
    al=reshape('opposite_sides',al,[2,4,P*P]);al=affine('edge_energy',al,np.ones((1,2,1,1)),conv=True)
    al=reshape('gradient_field',al,[4,P,P]); al=affine('orientation_selectivity',al,angmix[:,:,None,None],conv=True)
    al=relu('selective_edges',al); al=norm('contrast_of_edges',reshape('shared_edge_contrast',al,[1,4*P,P]));
    return [g1,hp,al]

def shape_bank():
    streams=feature_branch();filt=[[],[],[]]; n=768
    for k in range(10):
        for i in range(n):
            patch=render(k,i).reshape(P,RES//P,P,RES//P).mean((1,3)).reshape(-1)
            # Each operator below touches this one drawing only. Smoothing the
            # filter in edge space gives tolerance to displacements of one cell.
            high=(I-blur)@patch
            edge=np.abs(np.stack([a@patch for a in D]));edge=np.maximum(angmix@edge,0)@sm
            def unit(x):
                a=x-x.mean();return a.ravel()/(np.linalg.norm(a)+1.e-7)
            filt[0].append(F(unit(patch)*(.22/P)));filt[1].append(F(unit(high)*(.24/P)));filt[2].append(F(unit(edge)*(.54/(2*P))))
    dots=[affine('drawn_'+name,src,arr) for name,src,arr in zip(['shading','detail','contour'],streams,filt)]
    combined=reshape('views',add('agreement',*dots),[10,n//(n//4),n//4]);best=pool('best_pose_per_group',combined,[1,n//4],maximum=True)
    best1=pool('strongest_pose',best,[4,1],maximum=True);mean=pool('supporting_poses',best,[4,1]); res=add('shape_support',best1,mean)
    return affine('shape_evidence',res,4.7*np.eye(10))

# Small symbolic compiler for hand-specified piecewise-linear scene rules.
# It puts independent maps on separate branches, without large concatenations.
class Expr:
    def __init__(self,terms=None,c=0.):self.t={} if terms is None else terms;self.c=c
    def __add__(self,b):
        if not isinstance(b,Expr):return Expr(self.t.copy(),self.c+b)
        d=self.t.copy()
        for k,v in b.t.items():d[k]=d.get(k,0)+v
        return Expr(d,self.c+b.c)
    __radd__=__add__
    def __mul__(self,f):return Expr({k:v*f for k,v in self.t.items()},self.c*f)
    __rmul__=__mul__
    def __sub__(self,b):return self+(-1*b)
    def __rsub__(self,b):return -1*self+b

def local_colors():
    base={'R':([1,0,0],0),'G':([0,1,0],0),'B':([0,0,1],0),'Y':([.29,.58,.13],0)}
    for a,b in [('R','G'),('G','R'),('B','G'),('G','B'),('B','R'),('R','B')]:
        base[a+b]=(np.array(base[a][0])-base[b][0],0)
    rgb=affine('chromatic_axes','retina',np.array([v[0] for v in base.values()])[:,:,None,None],conv=True);rgb=relu('half_axes',rgb)
    def spec(b,**kw): return b,kw
    rules={
     'green':spec(-.12,GR=8,RG=-11,GB=4,BG=-8),
     'olive':spec(-.12,GB=6,RG=-3.3,GR=2,BG=-8),
     'blue':spec(-.10,BR=4.4,BG=5.5,GR=-1.1,GB=-4),
     'aqua':spec(-.09,BR=5,GR=3,GB=-6,RG=-6),
     'warm':spec(-.10,RG=4.5,GB=2.8,GR=-8,BG=-6),
     'red':spec(-.41,RG=6,GR=-6,BG=-2),
     'yellow':spec(-.28,GB=5,RG=.5,GR=-4,BG=-5),
     'pale':spec(-2.55,Y=4,RG=-1.4,GR=-1.4,BG=-1.4,GB=-1.4),
     'dark':spec(1.35,Y=-4.5),
     'gray':spec(1.28,RG=-8,GR=-8,BG=-8,GB=-8),
     'sky':spec(-.73,Y=2.55,BR=3,BG=2.5,RG=-5,GB=-5),
     'earth':spec(.10,Y=-1.3,RG=6.5,GB=4.4,GR=-10,BG=-8),
     'brightpaint':spec(-.59,RG=5.5,BG=6.5,GR=-2.8,GB=-.3),
     'light':spec(-.8,Y=2),
     'ink':spec(.95,Y=-5),
     'saturated':spec(-.28,RG=3.1,GR=3.1,BG=3.1,GB=3.1),
    }
    w=np.array([[r.get(key,0) for key in base] for b,r in rules.values()]);b=np.array([b for b,r in rules.values()]);n=len(b)
    val=affine('material_transitions',rgb,np.concatenate([w,w])[:,:,None,None],np.r_[b,b-1],conv=True);val=relu('transition_ramps',val)
    val=affine('material_memberships',val,np.concatenate([np.eye(n),-np.eye(n)],axis=1)[:,:,None,None],conv=True)
    return list(rules),pool('material_grid',val,2)

def local_texture():
    # Derivatives at two resolutions. Reflection is handled by extending the
    # nearest cell, so the frame of the photograph is not counted as a line.
    allsrc=[]
    for side,gray,tag in [(32,'image','fine'),(16,'luma','broad')]:
        if side==32:gray=affine('full_luma',gray,np.array([.29,.58,.13]).reshape(1,3,1,1),conv=True)
        # Symmetric Sobel-type derivative and second difference. Apply on the
        # valid image then pool to 8x8 by padding the gradient *field* to fit.
        kernels=[]
        yy,xx=np.mgrid[-1:2,-1:2]
        for deg in (0,45,90,135):
            th=np.deg2rad(deg);a=xx*np.cos(th)+yy*np.sin(th);t=-xx*np.sin(th)+yy*np.cos(th);w=a*np.exp(-.5*(a*a/.9+t*t/.58));kernels.append(w/np.maximum(w,0).sum())
        kernels += [np.array([[1,2,1],[2,-12,2],[1,2,1]])/12]
        ks=np.array(kernels);dir_=affine(tag+'_derivatives',gray,np.concatenate([ks,-ks])[:,None,:,:],conv=True,padding=1)
        dr=relu(tag+'_rectify',dir_);abs_=affine(tag+'_magnitude',dr,np.tile(np.eye(5),(1,2))[:,:,None,None],conv=True)
        # A fixed binary mask (spatial map) via valid-pooling the interior
        # instead of multiplying maps. Dense region weights below ignore the
        # outermost row/column where needed.
        p=pool(tag+'_cells',abs_,side//8);allsrc.append(p)
    return allsrc

# Region means are integrals under analytic masks. Regions overlap on purpose.
yg,xg=np.mgrid[:8,:8]/7.; X=xg-.5;Y=yg-.5
regions={
 'all':np.ones_like(X),'top':(yg<.23),'bottom':(yg>.77),'center':np.exp(-.5*((X/.218)**2+(Y/.233)**2)),
 'mid':np.exp(-.5*((X/.31)**2+((Y-.07)/.17)**2)),
 'sides':(abs(X)>.32)*np.exp(-.5*(Y/.32)**2),
 'outer':.42+X*X+Y*Y-np.exp(-.5*((X/.28)**2+(Y/.30)**2))*.42,
 'feet':np.exp(-.5*((X/.28)**2+((yg-.75)/.115)**2)),
 'high':np.exp(-.5*((X/.28)**2+((yg-.27)/.14)**2)),
 'low':(yg>=.56),'upper':(yg<=.42),
}

def scene_rules():
    colors,col=local_colors();tex=local_texture();sources={col:len(colors),tex[0]:5,tex[1]:5};primitive={}
    def feature(src,ch,region):
        key=(src,ch,region);primitive[key]=True;return Expr({key:1.})
    def C(n,r='all'):return feature(col,colors.index(n),r)
    def E(scale,idx,region):return feature(tex[scale],idx,region)
    def grad(r,scale):return sum(E(scale,i,r) for i in range(4))*.25
    def edge(r):return 3.6*(grad(r,0)+grad(r,1))
    def rough(r):return 13*E(0,4,r)
    def leaf(r):return .55*C('green',r)+.45*C('olive',r)
    def cool(r):return .57*C('blue',r)+.43*C('aqua',r)
    def ground(r):return .67*leaf(r)+.33*C('earth',r)
    def pale(r):return C('pale',r)
    # Shape-independent class evidence. Relative scores matter; absolute
    # offsets do not. Every constant here expresses an a priori preference.
    scores=[Expr() for k in range(10)];extra=[]
    def vote(v,weights):
        for k,w in weights.items():scores[k]=scores[k]+w*v
    def ramp(v,weights):extra.append((v,weights))
    animals={2:.35,3:.23,4:.65,5:.23,6:.48,7:.48,0:-.22,1:-.26,8:-.32,9:-.27}
    vote(leaf('all'),animals)
    vote(cool('all'),{0:1.1,8:.95,1:-.20,4:-.32,5:-.12,6:-.62,7:-.25})
    vote(C('warm')-.26,{3:.60,5:.37,6:.24,4:.32,7:.21,2:.12,0:-.41,8:-.50})
    vote(C('gray')-.46,{1:.47,9:.38,0:.19,8:.12,6:-.43,4:-.22})
    vote(C('brightpaint','mid')-.17,{1:.87,9:.54,2:.30,0:.12,7:-.23,4:-.33,6:-.20})
    vote(C('dark')-.20,{6:.73,5:.25,2:.14,8:-.41,0:-.46,9:-.22})
    vote(C('light')-.28,{8:.70,0:.72,3:.24,1:.16,9:.12,6:-.46})
    # Foreground/background contrasts and location: blue above suggests sky,
    # blue below suggests water; lush ground with a warm torso suggests a grazer.
    vote(cool('bottom'),{8:1.42,0:.65,9:-.50,1:-.46,7:-.24,4:-.30})
    vote(C('sky','top')-.4,{0:1.00,8:.61,9:.54,7:.23,3:-.22,6:-.47,5:-.18})
    vote(C('sky','bottom')-.37,{0:.31,8:.50,6:-.23})
    vote(cool('top')-cool('center'),{0:.78,8:.22,7:.31,9:.33,2:.18})
    vote(C('earth','center')-C('earth','outer'),{7:.99,4:.82,2:.43,5:.50,3:.42,6:.24,8:-.25})
    vote(leaf('bottom')-leaf('center'),{7:1.48,4:1.14,5:.50,2:.18,6:-.64})
    vote(leaf('center')-leaf('outer'),{6:1.46,7:-.74,4:-.60,2:-.23})
    vote(leaf('top'),{4:.86,2:.60,7:.25,6:.25,9:-.10})
    vote(C('gray','bottom')-C('gray','center'),{1:.74,9:.50,8:.13})
    vote(pale('top')-pale('mid'),{0:.44,8:.27,9:.40,7:.17})
    vote(C('dark','feet')-C('dark','center'),{1:.96,9:.62,8:.22,7:.22,3:-.23,6:-.23})
    vote(pale('high')-pale('feet'),{9:.26,1:.12,6:-.16})
    # Contrast concentration and the relative amounts of structure in the
    # horizontal and vertical directions. The references here are nominal
    # contrast changes on the [0,1] intensity scale, not dataset statistics.
    vote(edge('all')-.50,{1:.39,9:.42,0:-.94,8:-.33,4:-.08,5:.10})
    vote(edge('top')-.45,{4:.43,6:.42,2:.34,0:-.84,8:-.43,9:-.25})
    vote(edge('bottom')-.42,{6:.51,4:.20,2:.14,8:-.30,1:-.26,0:-.39})
    vote(edge('center')-edge('outer'),{2:.60,5:.30,1:.26,0:.37,8:.14,6:-.36})
    vote(rough('outer')-.25,{6:.91,4:.52,2:.24,8:-.31,0:-.42,3:-.20})
    hor=9*(E(1,2,'all')-E(1,0,'all'));midhor=9*(E(0,2,'mid')-E(0,0,'mid'))
    vote(hor,{8:.83,0:.49,1:.42,9:.16,3:-.32,2:-.29,5:-.24,6:-.16})
    vote(midhor,{1:.38,8:.39,9:.17,2:-.13,3:-.16})
    # Specific conjunctions. A maximum suppresses the penalty when a possible
    # subtype is absent (a plane is not required to be photographed in flight).
    ramp(.64*cool('top')+.70*cool('bottom')-.40*edge('outer')-.21,{0:1.04,8:.49})
    ramp(.57*C('sky','top')+.58*cool('bottom')-.30*edge('outer')-.24,{8:.93})
    ramp(C('dark','center')-.55*C('dark','outer')+.60*cool('outer')-.55*edge('outer')-.14,{0:.77,2:.34})
    ramp(.63*leaf('bottom')+.61*C('earth','center')-.35,{7:.87,4:.81})
    ramp(.62*leaf('top')+.52*leaf('sides')-.43*leaf('center')-.15,{4:.82,2:.48})
    ramp(.70*C('green','center')+.45*C('yellow','center')-.31,{6:1.05})
    ramp(.72*C('dark','outer')+.70*C('earth','all')+.32*rough('outer')-.33,{6:.97})
    ramp(.38*C('gray','bottom')+.70*C('brightpaint','center')+.37*(C('dark','feet')-C('dark','bottom'))-.25,{1:.98,9:.39})
    ramp(.51*C('sky','top')+.43*C('gray','bottom')+.37*pale('mid')-.25,{9:.61})
    ramp(.78*(pale('mid')-pale('outer'))+.38*(edge('center')-edge('outer'))-.11,{5:.74,2:.33,3:.17})
    ramp(.68*C('warm','outer')+.44*C('gray','center')-.22,{3:.65,5:.26})
    ramp(.52*C('warm','center')+.60*pale('feet')-.33*leaf('all')-.25,{3:.66,5:.25})
    ramp(.65*pale('outer')+.55*C('earth','center')-.22,{3:.56,5:.41,2:.22})
    # A bounded shape-independent prior for portraits: close crops and indoor
    # backgrounds need not have a particular surface color.
    vote(.36-C('sky','top'),{3:.25,5:.28,6:.20})
    vote(.30-C('saturated'),{3:.16,5:.17})
    # Compile region means into dense reductions. Use the inner pixels of the
    # 8x8 edge grid: derivatives computed with zero padding on the input image
    # must not turn the outer frame into evidence for a horizon/rectilinearity.
    expr=scores+[v for v,weights in extra];parts=[];srcs=list(sources)
    for j,src in enumerate(srcs):
        n=sources[src];mat=np.zeros((len(expr),n,8,8))
        for row,e in enumerate(expr):
            for (s,ch,r),v in e.t.items():
                if s!=src:continue
                mask=regions[r].copy().astype(float)
                if s!=col:
                    mask[[0,-1],:]=0;mask[:,[0,-1]]=0
                mat[row,ch]+=mask*(v/mask.sum())
        parts.append(affine('scene_terms_'+str(j),src,mat.reshape(len(expr),-1),[e.c if j==0 else 0. for e in expr]))
    a=add('scene_conditions',*parts); mat=np.zeros((10,len(expr)));mat[:,:10]=np.eye(10);linear=affine('scene_linear',a,mat)
    mat[:,:10]=0
    for n,(_,weights) in enumerate(extra):
        for k,w in weights.items():mat[k,10+n]=w
    nonlinear=affine('scene_combinations',relu('possible_environments',a),mat)
    return add('context_evidence',linear,nonlinear)

# A separate photographic-context cue: reflection symmetry of the central
# subject about a vertical axis. Take the least discrepancy among nearby axes;
# use two window sizes so this can detect both a face and a vehicle grille.
def symmetry_rules():
    rows=[]; xx,yy=np.meshgrid(np.arange(P),np.arange(P)); xx=xx.ravel();yy=yy.ravel()
    for scale in [0,1]:
        valid=(yy>=2)&(yy<=12) if scale==0 else (yy>=3)&(yy<=13)
        for offset in [-1.5,-.75,0,.75,1.5]:
            center=7.5+offset;xr=2*center-xx
            for i in np.where(valid & (xx<center-.5) & (xx>center-(4.2 if scale==0 else 6.5)) & (xr<P-1))[0]:
                left=int(np.floor(xr[i]));f=xr[i]-left;r=np.zeros(P*P);r[i]=1;r[yy[i]*P+left]-=1-f;r[yy[i]*P+left+1]-=f
                # Surrounding upper background and lower ground are less useful.
                g=np.exp(-.5*((yy[i]-(6.6 if scale==0 else 8.))/ (2.5 if scale==0 else 3.7))**2)
                rows.append((r,len(rows),scale*5+[-1.5,-.75,0,.75,1.5].index(offset),g))
    w=np.stack([r for r,i,j,g in rows]);a=affine('mirror_differences','global_contrast',np.concatenate([w,-w]));a=relu('mirror_disagreement',a);v=np.zeros((10,len(w)))
    for r,i,j,g in rows:v[j,i]=g
    v/=v.sum(axis=1,keepdims=True); a=affine('bilateral_coherence',a,-np.tile(v,(1,2)),.70);a=reshape('reflection_axes',a,[2,1,5]);a=pool('best_reflection',a,[1,5],maximum=True)
    wt=np.array([[.10,.28],[.53,.69],[-.10,-.22],[.70,.16],[-.09,-.16],[.56,.04],[.21,.11],[-.10,-.18],[.17,.16],[.12,.32]])
    return affine('symmetry_evidence',a,wt)

# Strongly anisotropic contour groups are typical of bodywork, deck lines and
# window edges. Detect collinear gradient support along a finite straight line,
# separately from the average edge density that is used by the scene rules.
def line_rules():
    # Bilinear sampling of an oriented positive gradient map along a segment,
    # with offset parallel segments as the inhibitory surround. Input edge maps
    # have already been contrast-normalized jointly across the orientations.
    rows=[];group=[]
    for ch,(dy,dx) in enumerate([(0,1),(1,1),(1,0),(1,-1)]):
        d=np.hypot(dx,dy);nx=dx/d;ny=dy/d;tx=-ny;ty=nx
        for where,(ymin,ymax) in enumerate([(2,7),(6,11),(10,14)]):
            C=[]
            for y in np.linspace(ymin,ymax,5):
                for x in np.linspace(4,11,5):
                    w=np.zeros((4,P,P));
                    def deposit(x,y,a):
                        x=np.clip(x,.0,P-1.00001);y=np.clip(y,.0,P-1.00001);i=int(x);j=int(y);xf=x-i;yf=y-j
                        w[ch,j:j+2,i:i+2]+=a*np.outer([1-yf,yf],[1-xf,xf])
                    for j in np.linspace(-3.5,3.5,9):
                        for off,a in [(0,1),(-1.55,-.5),(1.55,-.5)]:deposit(x+tx*j+nx*off,y+ty*j+ny*off,a/9)
                    rows.append(w.ravel())
    a=affine('line_support','contrast_of_edges',rows);a=reshape('line_locations',a,[12,1,25]);b=pool('long_edges',a,[1,25],maximum=True)
    # A horizontal line is more significant in the lower half (rocker panel,
    # waterline). A vertical edge plus a horizontal one suggests a large box.
    m=np.zeros((4,4,3));m[0,2]=[.20,.43,.37];m[1,0]=[.33,.47,.20];m[2,1]=[.15,.20,.15];m[2,3]=[.15,.20,.15];m[3,2]=[-.3,-.3,.6]
    a=affine('geometry_of_lines',b,m.reshape(4,-1));wt=np.array([[.17,-.10,.13,0],[.27,-.06,-.05,.04],[-.09,-.03,.07,0],[-.06,-.02,-.03,0],[-.02,.04,-.05,0],[-.05,-.03,-.03,0],[-.10,-.08,.01,0],[-.02,.03,-.01,0],[.29,-.05,-.04,.07],[.21,.20,-.07,-.01]])
    return affine('line_evidence',a,wt,-wt@np.array([1.4,1.4,1.4,0]))

if __name__=='__main__':
    shape=shape_bank();scene=scene_rules();sym=symmetry_rules();line=line_rules();add('scores',shape,scene,sym,line)
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':L},indent=1)+'\n');np.savez(out/'weights.npz',**W)
