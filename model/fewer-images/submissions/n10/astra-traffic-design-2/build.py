"""Build a sign reader from a pictogram grammar. No input files or fonts."""
import json, math, sys
from pathlib import Path
import numpy as np

layers=[]; weights={}
def layer(name,op,inputs,**kw):
    if isinstance(inputs,str):inputs=[inputs]
    layers.append(dict(name=name,op=op,inputs=inputs,**kw));return name

def affine(name,inp,W,b=None,conv=False,pad=0):
    W=np.asarray(W,dtype='f');n=len(W)
    weights[name+'.W']=W;weights[name+'.b']=np.zeros(n,'f') if b is None else np.array(b,dtype='f')
    return layer(name,'conv',inp,out=n,kernel=list(W.shape[-2:]),stride=1,padding=pad) if conv else layer(name,'dense',inp,out=n)
def point(name,inp,W,b=None):return affine(name,inp,np.asarray(W)[:,:,None,None],b,True)
def pool(name,op,inp,sz,st):return layer(name,op+'pool',inp,size=sz,stride=st)
def reshape(n,x,s):return layer(n,'reshape',x,shape=s)

def crop(name,inp,k):
    # A centred crop as a valid convolution with a unit impulse.
    p=(32-k)//2;w=np.zeros((1,1,2*p+1,2*p+1));w[0,0,p,p]=1
    return affine(name,inp,w,conv=True)

def absorb_contrast(name,inp):
    # Soft saturation: linear in the shadows, logarithm-like for bright pixels.
    levels=np.array([0,.025,.08,.2,.48,.8,1.]);values=np.sqrt(levels+.006)-np.sqrt(.006)
    slopes=np.diff(values)/np.diff(levels);delta=np.diff(np.r_[0,slopes]);n=len(delta)
    bank=point(name+'_knots',inp,np.repeat(np.eye(3),n,axis=0),-np.tile(levels[:-1],3));bank=layer(name+'_positive','relu',bank)
    return point(name,bank,np.kron(np.eye(3),delta))

# Small drawing primitives on a supersampled grid with positive y downwards.
class Pen:
    def __init__(self,x,y):self.x,self.y=x,y;self.a=np.zeros(np.broadcast_shapes(x.shape,y.shape))
    def paint(self,m,v):self.a=np.where(m,v,self.a)
    def oval(self,x,y,rx,ry,v=1):self.paint(((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1,v)
    def dot(self,x,y,r,v=1):self.oval(x,y,r,r,v)
    def box(self,x0,y0,x1,y1,v=1):self.paint((self.x>=x0)&(self.x<=x1)&(self.y>=y0)&(self.y<=y1),v)
    def poly(self,pts,v=1):
        m=np.zeros_like(self.a,dtype=bool)
        for (x0,y0),(x1,y1) in zip(pts,pts[1:]+pts[:1]):
            if y1!=y0:m^=((self.y>y0)!=(self.y>y1))&(self.x < x0+(self.y-y0)*(x1-x0)/(y1-y0))
        self.paint(m,v)
    def line(self,pts,w,v=1):
        for (a,b),(c,d) in zip(pts,pts[1:]):
            dx,dy=c-a,d-b;q=((self.x-a)*dx+(self.y-b)*dy)/max(dx*dx+dy*dy,1e-12);q=np.clip(q,0,1)
            self.paint((self.x-a-dx*q)**2+(self.y-b-dy*q)**2<=(w/2)**2,v)
    def arc(self,cx,cy,rx,ry,t0,t1,w,v=1):
        a=np.linspace(t0,t1,18)*np.pi/180;self.line(list(zip(cx+rx*np.cos(a),cy+ry*np.sin(a))),w,v)

def sub(p,x,y,sx,sy):
    q=Pen((p.x-x)/sx,(p.y-y)/sy);return q

def digit(p,d,thick=1):
    # DIN-like sans-serif strokes. Design coordinates: cap height 1, width .56.
    w=.145*thick
    if d=='0':
        # Squared-off oval, with long straight sides.
        outer=(abs(p.x/.285)**2.75+abs(p.y/.50)**2.75)<=1
        inner=(abs(p.x/(.285-w))**2.75+abs(p.y/(.5-w))**2.75)<=1
        p.paint(outer&~inner,1)
    elif d=='1':
        p.line([(-.19,-.28),(.035,-.43),(.035,.435)],w);p.box(-.04,.41,.108,.49)
    elif d=='2':
        p.line([(-.214,-.24),(-.19,-.34),(-.12,-.412),(-.025,-.44),(.095,-.426),(.178,-.37),(.214,-.28),(.209,-.17),(.13,-.04),(-.214,.36),(-.214,.43),(.224,.43)],w)
    elif d=='3':
        p.line([(-.20,-.32),(-.125,-.412),(.005,-.44),(.135,-.408),(.205,-.325),(.208,-.21),(.166,-.112),(.045,-.027),(-.045,-.02)],w)
        p.line([(.015,-.025),(.138,.015),(.209,.105),(.221,.234),(.185,.347),(.102,.421),(-.02,.435),(-.145,.395),(-.22,.30)],w)
    elif d=='5':
        p.line([(.196,-.43),(-.172,-.43),(-.195,-.035),(-.103,-.10),(.014,-.127),(.132,-.095),(.20,-.023),(.216,.115),(.206,.263),(.165,.368),(.078,.423),(-.05,.433),(-.156,.386),(-.222,.298)],w)
    elif d=='6':
        p.line([(.115,-.44),(-.052,-.22),(-.175,-.01),(-.207,.129)],w)
        p.arc(0,.164,.209,.267,0,360,w)
    elif d=='7':
        p.line([(-.227,-.405),(-.227,-.431),(.219,-.431),(.212,-.36),(-.107,.439)],w)
    elif d=='8':
        p.arc(0,-.237,.188,.196,0,360,w);p.arc(0,.211,.215,.222,0,360,w)

def digits(p,text,width,height,thick=1):
    # Fixed advance except for the naturally narrow 1.
    widths=[.35 if d=='1' else .57 for d in text];total=sum(widths)+.105*(len(text)-1);pos=-total/2
    for d,a in zip(text,widths):
        q=sub(p,(pos+a/2)*width/total,0,width/total,height);digit(q,d,thick);p.a=np.maximum(p.a,q.a);pos+=a+.105

def car(p,cx,cy,w,h,level):
    q=sub(p,cx,cy,w,h)
    q.poly([(-.47,.32),(-.48,.04),(-.39,-.055),(-.31,-.395),(-.22,-.47),(.235,-.47),(.31,-.4),(.40,-.055),(.48,.04),(.47,.32)],level)
    q.box(-.4,.23,-.25,.50,level);q.box(.25,.23,.4,.50,level)
    q.poly([(-.306,-.075),(-.238,-.35),(.222,-.35),(.305,-.075)],0);p.a=np.maximum(p.a,q.a)
def truck(p,cx,cy,w,h,level):
    q=sub(p,cx,cy,w,h);q.box(-.15,-.44,.487,.24,level)
    q.poly([(-.47,.28),(-.47,-.045),(-.325,-.32),(-.18,-.32),(-.18,.06),(.46,.06),(.46,.28)],level)
    q.poly([(-.407,-.032),(-.319,-.227),(-.235,-.227),(-.235,-.032)],0)
    for x in [-.30,.28]:q.dot(x,.272,.17,level)
    p.a=np.maximum(p.a,q.a)
def arrow(p,a,b,shaft=.22,head=.51,length=.29):
    u=np.array(b)-a;u=u/np.linalg.norm(u);v=np.array([-u[1],u[0]]);a=np.array(a);b=np.array(b);j=b-length*u
    p.poly([a+v*shaft/2,j+v*shaft/2,j+v*head/2,b,j-v*head/2,j-v*shaft/2,a-v*shaft/2])
def bend(p):
    p.line([(-.28,.70),(-.28,-.008),(-.255,-.123),(-.188,-.203),(-.087,-.245),(.31,-.25)],.21)
    p.poly([(.115,-.54),(.56,-.251),(.115,.063),(.199,-.15),(.199,-.349)])
def walking(p,cx,cy,s,mirror=1):
    q=sub(p,cx,cy,s*mirror,s);q.dot(.056,-.323,.076);q.line([(.008,-.206),(-.078,.042)],.136)
    q.line([(-.059,.04),(.04,.191),(.172,.316)],.072);q.line([(-.061,.045),(-.136,.204),(-.237,.329)],.073)
    q.line([(-.002,-.183),(.071,-.081),(.206,-.016)],.056);q.line([(-.02,-.169),(-.141,-.097),(-.213,.001)],.055);p.a=np.maximum(p.a,q.a)

speed={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
warn=[11,*range(18,32)];mandatory=list(range(33,41));ends=[6,32,41,42]
def icon(k,x,y,opt=0):
    p=Pen(x,y)
    if k in speed:
        text=speed[k];ww=(1.21 if len(text)==2 else 1.40);hh=(.76 if len(text)==2 else .69)
        digits(p,text,ww,hh,1 if opt==0 else .87)
    elif k in [9,10,16,41,42]:
        if k==16:truck(p,0,0,1.34,.66,1)
        else:
            big=k in (10,42);sep=.36
            if big:truck(p,-sep,0,.68,.48,1)
            else:car(p,-sep,.015,.58,.43,.72 if k==9 else 1)
            car(p,sep,.015,.59,.44,1)
    elif k==14:
        # Narrow white block capitals for STOP, written with strokes.
        for letter,c in zip('STOP',[-.53,-.175,.18,.53]):
            q=sub(p,c,0,.35,.68);w=.145
            if letter=='S':q.line([(.34,-.33),(.18,-.426),(-.18,-.426),(-.33,-.30),(-.33,-.17),(-.19,-.09),(.18,.05),(.33,.16),(.33,.30),(.16,.424),(-.16,.424),(-.33,.33)],w)
            if letter=='T':q.line([(-.37,-.42),(.37,-.42)],w);q.line([(0,-.42),(0,.42)],w)
            if letter=='O':q.line([(-.32,-.31),(-.17,-.42),(.17,-.42),(.32,-.30),(.32,.30),(.17,.42),(-.17,.42),(-.32,.30),(-.32,-.31)],w)
            if letter=='P':q.line([(-.32,.42),(-.32,-.42),(.13,-.42),(.32,-.31),(.32,-.10),(.14,.016),(-.32,.016)],w)
            p.a=np.maximum(p.a,q.a)
    elif k==17:p.box(-.766,-.17,.766,.17)
    elif k in [33,34]:
        if k==34:p.x=-p.x
        bend(p)
    elif k==35:arrow(p,(0,.72),(0,-.76),.215,.68,.48)
    elif k in [36,37]:
        if k==37:p.x=-p.x
        arrow(p,(-.13,.72),(-.13,-.75),.19,.59,.38)
        arrow(p,(-.12,.064),(.72,.064),.175,.55,.34)
    elif k in (38,39):
        if k==39:p.x=-p.x
        u=np.array([1.,1.])/np.sqrt(2);v=np.array([-u[1],u[0]])
        if opt==0:
            pts=[(-.73,-.14),(.25,-.14),(-.03,-.41),(.19,-.41),(.75,0),(.19,.41),(-.03,.41),(.25,.14),(-.73,.14)]
            p.poly([a*u+b*v for a,b in pts])
        else:arrow(p,(-.51,-.51),(.55,.55),.245,.76,.48)
    elif k==40:
        for phi in [0,120,240]:
            q=sub(p,0,0,1,1);a=(phi+opt*23)*np.pi/180;q.x,q.y=q.x*np.cos(a)+q.y*np.sin(a),-q.x*np.sin(a)+q.y*np.cos(a)
            q.arc(0,0,.526,.526,2,-74,.134);arrow(q,(.269,-.452),(-.146,-.539),.134,.375,.22);p.a=np.maximum(p.a,q.a)
    # Triangle drawings here use the sign's bounding box as [-1,1]^2.
    elif k==11:
        p.poly([(-.128,.627),(-.128,.385),(-.43,.385),(-.43,.30),(-.13,.19),(-.13,.04),(0,-.38),(.13,.04),(.13,.19),(.43,.30),(.43,.385),(.128,.385),(.128,.627)])
    elif k==18:
        p.poly([(-.073,-.37),(.073,-.37),(.059,.17),(.037,.36),(-.037,.36),(-.059,.17)]);p.dot(0,.527,.083)
    elif k in (19,20):
        if k==20:p.x=-p.x
        p.poly([(.335,.61),(.335,.34),(.303,.166),(.211,.023),(.081,-.084),(-.174,-.202),(-.25,-.018),(-.066,.091),(.025,.194),(.064,.329),(.064,.61)])
    elif k==21:
        p.line([(.23,.56),(.23,.22),(.181,.109),(.09,.083),(-.10,.132),(-.154,.084),(-.147,-.112),(-.092,-.234)],.182)
    elif k==22:
        p.box(-.542,.489,.542,.587);p.oval(-.215,.50,.255,.19);p.oval(.199,.50,.216,.165)
    elif k==23:
        # A car viewed from the side with diverging skid marks. Side and end-on
        # pictograms occur on slippery-road warnings.
        if opt%2:
            q=sub(p,0,.14,.83,.45);car(q,0,0,.94,1,1);p.a=np.maximum(p.a,q.a)
            for xx in [-.22,.20]:p.line([(xx,.29),(xx+.10,.36),(xx+.10,.43),(xx+.03,.48),(xx-.09,.52),(xx-.12,.60)],.059)
        else:
            p.poly([(-.46,.30),(-.46,.17),(-.34,.12),(-.22,-.045),(.053,-.045),(.205,.09),(.39,.14),(.40,.29)])
            p.poly([(-.25,.10),(-.16,-.002),(.031,-.002),(.145,.10)],0)
            for xx in [-.27,.24]:p.dot(xx,.283,.084)
            for h in [.39,.48]:p.line([(-.46,h),(-.21,h+.04),(.06,h+.03),(.32,h+.065),(.45,h+.10)],.035)
    elif k==24:
        p.line([(-.202,.595),(-.202,-.224)],.085);p.line([(.282,.595),(.282,.37),(.261,.243),(.13,.063),(.098,-.044),(.098,-.221)],.087)
    elif k==25:
        if opt%2:p.x=-p.x
        p.dot(.008,-.148,.079);p.line([(-.04,-.052),(-.155,.13),(-.24,.22)],.132)
        p.line([(-.234,.21),(-.33,.398),(-.46,.59)],.09);p.line([(-.215,.21),(-.122,.37),(-.166,.565)],.087)
        p.line([(-.022,-.04),(.067,.107),(.113,.266)],.068);p.line([(-.07,.009),(-.056,.156),(.019,.211)],.058)
        p.line([(-.23,.035),(.329,.48)],.040);p.poly([(.08,.58),(.22,.418),(.334,.379),(.45,.45),(.571,.58)])
    elif k==26:
        for cy in [-.105,.19,.482]:p.oval(0,cy,.143,.144,1 if cy!=.19 else .70)
    elif k==27:walking(p,0,.15,1.40,1 if not opt else -1)
    elif k==28:
        if opt%2:p.x=-p.x
        walking(p,.14,.143,1.25,-1);walking(p,-.28,.294,.84,-1);p.line([(-.174,.22),(.001,.120)],.06)
    elif k==29:
        if opt%2:p.x=-p.x
        p.x=p.x/1.22;p.y=(p.y-.2)/1.12+.2
        for cx in [-.318,.315]:p.arc(cx,.427,.198,.185,0,360,.052)
        p.line([(-.32,.421),(-.142,.143),(.003,.427),(-.32,.421)],.055);p.line([(0,.427),(.197,.105),(.317,.421)],.052);p.line([(-.15,.15),(-.20,.05)],.057)
        p.line([(-.22,.064),(-.078,.064)],.048);p.line([(.315,.42),(.18,.012),(.105,-.018)],.044)
        if opt%2:
            p.dot(-.015,-.186,.074);p.line([(-.035,-.088),(-.14,.09),(.013,.202),(-.01,.335)],.079);p.line([(-.038,-.075),(.079,.075),(.202,.071)],.058)
    elif k==30:
        for a in np.arange(6)*np.pi/3-np.pi/2:
            u=np.array([np.cos(a),np.sin(a)]);v=np.array([-u[1],u[0]]);c=np.array([0,.195]);p.line([c,c+.406*u],.065)
            for sg in [-1,1]:p.line([c+.205*u,c+.316*u+sg*.105*v],.052)
    elif k==31:
        p.oval(-.041,.257,.292,.107);p.line([(-.23,.24),(-.327,.065),(-.366,.031)],.102);p.oval(-.372,.05,.104,.055)
        p.line([(-.329,.058),(-.29,-.086),(-.302,-.182)],.043);p.line([(-.31,-.051),(-.392,-.15)],.04)
        p.line([(-.164,.287),(-.083,.432),(-.245,.49)],.065);p.line([(.163,.259),(.293,.378),(.439,.525)],.069);p.line([(.14,.29),(.213,.443),(.284,.54)],.040)
    if k in ends:
        z=(x+y)/np.sqrt(2);bars=abs(z)<.160
        if opt==1:bars=bars & ((z+.16)%.080<.051)
        p.a=np.maximum(.65*p.a,bars.astype(float))
    return p.a

def tri(x,y,scale=1.,inverted=False):
    # Homothetic triangles about their centroid, not the box centre.
    v=-y if inverted else y
    v=(v-1/3)/scale+1/3;u=x/scale
    return (v<=1)&(v>=2*abs(u)-1)

def base(k,x,y):
    """Independent analytic colour/lightness mask and confidence region."""
    r=np.sqrt(x*x+y*y);disc=r<1;white=.90;red=.24;blue=.23
    if k in warn or k==13:
        inv=k==13;s=tri(x,y,1,inv);inside=tri(x,y,.747,inv);outer=tri(x,y,1.08,inv)
        lum=np.where(s,np.where(inside,white,red),.52);redness=(s&~inside).astype(float);blueness=0*r;sup=np.where(s,1.,np.where(outer,.54,0))
        # Bounding triangle tips are usually rounded.
        v=-y if inv else y;sup=np.where((v<-.80)|(abs(x)>.86),sup*.32,sup)
    elif k==12:
        d=abs(x)+abs(y);s=d<1.025;inside=d<.725
        lum=np.select([inside,d<.78,s],[.74,.35,.93],.44);redness=inside*.16;blueness=-inside.astype(float);sup=np.where(s,1,np.where(d<1.12,.50,0))
    else:
        s=disc
        if k in ends:
            lum=np.where(s,.90,.41);redness=np.zeros_like(r);blueness=redness;sup=np.where(s,1,np.where(r<1.095,.45,0))
        elif k in mandatory:
            lum=np.where(s,np.where(r<.94,blue,.77),.51);blueness=(r<.94).astype(float);redness=0*r;sup=np.where(s,1,np.where(r<1.09,.43,0))
        elif k in [14,17]:
            if k==14:
                d=np.maximum(np.maximum(abs(x),abs(y)),(abs(x)+abs(y))/np.sqrt(2));s=d<=1
            else:d=r
            lum=np.where(s,np.where(d<.948,red,.88),.52);redness=(d<.948).astype(float);blueness=0*r;sup=np.where(s,1,np.where(d<1.10,.48,0))
        else:
            inside=r<.786;lum=np.where(s,np.where(inside,white,red),.51);redness=(s&~inside).astype(float);blueness=0*r;sup=np.where(s,1,np.where(r<1.09,.5,0))
    return lum,redness,blueness,sup

SS=3
# Analytically chosen sample grids for the templates, in pixel-centre coordinates.
def coords(k):
    xy=(np.arange(k*SS)+.5)/SS-k*.5
    return xy[None,:],xy[:,None]
def aa(a,k):return a.reshape(k,SS,k,SS).mean((1,3))
def unit(a,m):
    # Normalise this drawing alone on its own support. No shared dictionary statistics.
    a=(a-(a*m).sum()/max(m.sum(),1e-12))*m
    return a/max(np.sqrt((a*a).sum()),1.e-10)

def geometry(k):
    # width/height in these coordinates are diameters. Both dimensions vary.
    if k in warn or k==13:return [(s,e) for s in [8.5,10.4,12.3,14.2,15.8] for e in [.89,1.,1.12]]
    return [(s,e) for s in [8.4,10.1,11.8,13.5,15.2] for e in [.88,1.,1.12]]

def fixed_rgb_knots():
    rgb=absorb_contrast('compress','image');lum=point('green_luma',rgb,[[.10,.72,.18]])
    ch=point('chromatic_axes',rgb,[[1,-.80,-.20],[-.7,-.3,1.]])
    ch=layer('balanced_chroma','norm',ch)
    lc=layer('standard_luma','norm',lum)
    # Bandpass attenuates slowly varying shadows across a face.
    t=np.arange(-2,3);g=np.exp(-t*t/(2*1.32**2));g/=sum(g);hh=-np.outer(g,g);hh[2,2]+=1
    hi=affine('detail',lum,hh[None,None],conv=True,pad=2)
    return rgb,lum,lc,ch,hi


def group_pool(name,inp,ngroup,nvar,nshift):
    r=reshape(name+'_layout',inp,[ngroup,nvar,nshift]);r=pool(name+'_max','max',r,[nvar,nshift],[nvar,nshift]);return reshape(name,r,[ngroup])

def shell_branch(lc,ch):
    mixed=layer('outline_channels','add',[point('outline_light',lc,[[1],[0],[0]]),point('outline_pigment',ch,[[0,0],[1,0],[0,1]])]);mixed=pool('coarse','avg',mixed,[2,2],[2,2]);K=14;xx,yy=coords(K)
    # Use one border model for each family, with a generic interior marking.
    groups=[0,9,10,15,16,11,12,13,14,17,35,32];W=[]
    for c in groups:
        for s,e in geometry(c):
            x=xx*2/(s*e);y=yy*2/s;lum,red,blue,m=base(c,x,y)
            if c in [0,11,35]:
                if c==0:weak=np.exp(-((x/.52)**4+(y/.28)**4))
                elif c==11:weak=np.exp(-((x/.26)**2+((y-.23)/.35)**2))
                else:weak=np.exp(-((x/.35)**2+(y/.48)**2))
                lum=lum+(.26 if c==35 else -.22)*weak
            elif c not in [12,13,15]:
                ink=icon(c,x,y);lum=lum+(.65 if c in [14,17] else -.62)*ink
                if c in [14,17]:red=np.maximum(red-ink,0)
                if c in [9,10]:red=np.maximum(red,ink*(x<0)*.9)
            m=aa(m,K);l=unit(aa(lum,K),m);r=unit(aa(red,K),m);b=unit(aa(blue,K),m)
            # Local pigment contrast is strong for red rims and the yellow diamond.
            # A blue disc has no red/white rim. The achromatic cancellations have none either.
            if c==35:
                # Bright pictogram inside the disc generally has little blue pigment.
                b=unit(aa(blue*(1-.75*weak),K),m)
            scl=s/12 # divide by size for comparable matched-filter responses
            W.append(np.array([.62*l,.32*r,.32*b])/(scl*6.0))
    q=affine('family_filters','coarse',np.array(W),conv=True);q=group_pool('family',q,len(groups),15,9)
    family={}
    for k in range(43):
        c=(32 if k in ends else 11 if k in warn else 35 if k in mandatory else 0 if k in speed else k);family[k]=groups.index(c)
    return q,family

# Group-specific, locally contrast-normalised template matcher. Blocks of candidate
# drawings slide across the small centre search area; pooling integrates uncertainties.
def match_branch(name,ks,bank,source,detail=False):
    K=bank[0][0].shape[-1];cr=crop(name+'_window',source,K+4);cr=layer(name+'_instance','norm',cr)
    # Limit high-contrast outliers smoothly, in units of local SD.
    pos=point(name+'_clip_input',cr,[[1],[1]], [1.9,-1.9]);pos=layer(name+'_clip_relu','relu',pos);cr=point(name+'_clipped',pos,[[1,-1]],[-1.9])
    mat=np.stack([w for bk in bank for w in bk]);q=affine(name+'_templates',cr,mat[:,None],conv=True);return group_pool(name,q,len(ks),len(bank[0]),25)

def interior_bank(c,K,optset=0):
    xx,yy=coords(K);bk=[];bh=[]
    if c in speed and c!=6:
        pars=[(s,h,e,o) for s in [8.7,10.5,12.3,14.2,15.7] for h in [.87,1.03,1.18] for e in [.90,1.035,1.15] for o in [0,1]]
    elif c in warn:
        pars=[(s,h,e,o) for s in [8.8,10.7,12.7,14.7] for h in [.91,1.09] for e in [.86,1.03,1.18] for o in [0,1]]
    elif c in mandatory:
        pars=[(s,h,e,o) for s in [8.9,10.9,12.9,14.9] for h in [.92,1.08] for e in [.89,1.09] for o in [0,1]]
    elif c in ends:
        pars=[(s,h,e,o) for s in [8.5,10.3,12.2,14.1,15.6] for h in [.92,1.08] for e in [.9,1.08] for o in [0,1]]
    else:
        pars=[(s,h,e,o) for s in [8.6,10.6,12.6,14.7] for h in [.89,1.10] for e in [.90,1.09] for o in [0]]
    for s,h,e,o in pars:
        x=xx/(s*e);y=yy/s
        if c in warn:
            # Triangle symbols are below the centre; centre the search on the symbol.
            ink=icon(c,x,(y/h)+.22,o);support=tri(x,y+.22,.715).astype(float)
        else:
            ink=icon(c,x,y/h,o);support=(x*x+y*y<(.90 if c in mandatory+[14,17]+ends else .755)**2).astype(float)
        if c in speed and c not in ends:
            support*=abs(y)<.50*h
        if c in [9,10,16]:support*=abs(y)<.51*h
        polarity=1 if c in mandatory+[14,17] else -1
        a=aa(ink*polarity,K);m=aa(support,K);w=unit(a,m)
        # Amplitude per unit length compensates for the area of the projected glyph.
        bk.append((w*(12/s)/6.4).astype('f'))
        # The template's bandpass is a separate, independently normalised view of it.
        pad=np.pad(a,2,mode='edge');t=np.arange(-2,3);g=np.exp(-t*t/(2*1.32**2));g/=sum(g)
        sm=sum(g[i]*g[j]*pad[i:i+K,j:j+K] for i in range(5) for j in range(5));bh.append((unit(a-sm,m)*(12/s)/6.4).astype('f'))
    return bk,bh

def crop_at(name,src,k,dx=0,dy=0):
    p=(32-k)//2;w=np.zeros((1,1,2*p+1,2*p+1));w[0,0,p+dy,p+dx]=1;return affine(name,src,w,conv=True)

def contrast_bank(name,src,ks,coef):
    # Relative evidence within a family. Subtract a common *input-dependent*
    # maximum at inference (no statistics over a collection of drawings).
    m=reshape(name+'_as_row',src,[1,1,len(ks)]);m=pool(name+'_best','max',m,[1,len(ks)],[1,len(ks)])
    d=np.zeros((43,len(ks)));b=np.zeros((43,1))
    for i,k in enumerate(ks):d[k,i]=coef;b[k,0]=-coef
    return [affine(name+'_strength',src,d),affine(name+'_reference',m,b)]

def numeral_reader(lum,hi):
    # Read the identifying digit separately so the trailing zero does not
    # dominate the score. Three-digit signs have the identifier in the middle.
    ks=[0,1,2,3,4,5,7,8];K=12;xx,yy=coords(K);nf=5*3*3*2
    branches=[('leaders',-4,'235678'),('middles',0,'02')];by_type=[[],[]]
    for name,dx,alphabet in branches:
        banks=[[],[]]
        for ch in alphabet:
            for height in [6.1,7.7,9.3,11.1,13.0]:
                for aspect in ([.38,.49,.60] if dx else [.31,.40,.49]):
                    for shear in [-.09,0,.09]:
                        for weight in [.86,1.12]:
                            x=xx/height-shear*yy/height;y=yy/height
                            p=Pen(x*.57/aspect,y);digit(p,ch,weight)
                            m=aa((abs(y)<.60)&(abs(x)<(aspect*.5+.060)),K)
                            a=-aa(p.a,K)
                            banks[0].append(unit(a,m)*8.5/(height*np.sqrt(aspect))*0.26)
                            edge=a-smooth(a,1.,2);banks[1].append(unit(edge,m)*8.5/(height*np.sqrt(aspect))*0.26)
        for j,src in enumerate([lum,hi]):
            n=name+['_shade','_stroke'][j];v=crop_at(n+'_window',src,18,dx);v=layer(n+'_norm','norm',v)
            # Separate 7x7 search windows allow small typography and framing offsets.
            q=affine(n+'_read',v,np.array(banks[j])[:,None],conv=True);by_type[j].append(group_pool(n,q,len(alphabet),nf,49))
    pieces=[]
    for j in range(2):
        n='digit_'+['shade','stroke'][j]
        # The two searches see different numbers of neighbours. Compare their
        # digit identities only among hypotheses with the same layout.
        for name,src,subks in zip(['left','centre'],by_type[j],[ks[:6],ks[6:]]):
            pieces+=contrast_bank(n+name+'_diff',src,subks,[.36,.27][j])
    return pieces

def fuse_features(name,planes):
    parts=[]
    for i,(src,n,off) in enumerate(planes):
        w=np.zeros((sum(x[1] for x in planes),n));w[off:off+n]=np.eye(n);parts.append(point(name+str(i),src,w))
    return layer(name,'add',parts)

def aligned_numbers():
    # A full reading and an independent rim score share the very same latent
    # variables (image position and outer radius). This prevents a neighbouring
    # zero or rim segment from impersonating the first digit.
    ks=[0,1,2,3,4,5,7,8];K=22;xx,yy=coords(K);W=[];parents=[];Wshell=[];xxs,yys=coords(16)
    for k in ks:
        for i,(s,e) in enumerate(geometry(k)):
            if k==ks[0]:
                x=2*xxs/(s*e);y=2*yys/s;lm,rd,bl,mask=base(k,x,y);mask=aa(mask,16)*(1-np.exp(-((2*xxs[0,::SS]/(s*e))[None,:]**2+(2*yys[::SS,0]/s)[:,None]**2)**2/.34**4))
                Wshell.append(np.array([unit(aa(lm,16),mask)*.65,unit(aa(rd,16),mask)*.35,np.zeros((16,16))])*2/s)
            for h in [.84,1.,1.17]:
                for wide in [.88,1.,1.12]:
                    x=xx/(s*e);y=yy/s;ink=icon(k,x/wide,y/h)
                    m=(x*x+y*y<.766**2)&(abs(y)<.52*h)
                    # White space above, below and beside the type is significant.
                    m=aa(m,K);a=-aa(ink,K);w=unit(a,m);ed=unit(a-smooth(a,1.32,2),m)
                    W.append(np.array([w*.60,ed*.60])*12/(s*6.4));parents.append(i)
    # Symmetric zero padding puts the five-by-five pose grid on pixel centres in
    # both branches (coarse filters use a stride-two grid, expanded by interpolation).
    qa=affine('number_rim_filters','coarse',np.array(Wshell),conv=True,pad=1)
    # The rim grid is 3x3 at offsets -2,0,2. Expand to -2,-1,0,1,2 using a
    # deterministic bilinear interpolator, then broadcast across spelling variants.
    T=np.zeros((25,9))
    for y in range(5):
        for x in range(5):
            for b in range(3):
                for a in range(3):T[5*y+x,3*b+a]=max(0,1-abs(x/2-a))*max(0,1-abs(y/2-b))
    interp=affine('number_rim_interp',qa,np.kron(np.eye(15),T));interp=reshape('number_rim_maps',interp,[15,5,5]);B=np.eye(15)[parents];rim=point('number_pose_prior',interp,B*.90)
    features=[]
    for src in ['green_luma','detail']:
        z=crop('number_aligned_window_'+src,src,K+4);z=layer('number_aligned_norm_'+src,'norm',z)
        q=point('number_aligned_clip_'+src,z,[[1],[1]],[1.9,-1.9]);q=layer('number_clip_rel_'+src,'relu',q);features.append(point('number_clip_out_'+src,q,[[1,-1]],[-1.9]))
    src=fuse_features('number_reading_planes',[(features[0],1,0),(features[1],1,1)]);q=affine('registered_letters',src,np.array(W),conv=True);q=layer('registered_face','add',[q,rim]);q=group_pool('aligned_reading',q,len(ks),len(W)//len(ks),25)
    return contrast_bank('number_registration',q,ks,1.12)

def oriented_glyphs(lum):
    # Signed, softly thresholded orientations tolerate broken/jagged outlines.
    # The weights are descriptors of individual vector drawings, never a pooled
    # statistic of different drawings. Pooling of poses is at inference only.
    t=np.arange(-2,3);tx,ty=np.meshgrid(t,t);g=np.exp(-(tx**2+ty**2)/(2*.86**2));dirs=[]
    for rad in np.arange(4)*np.pi/4:
        a=g*(tx*np.cos(rad)+ty*np.sin(rad));a/=np.maximum(a,0).sum();dirs.extend([a,-a])
    dus=np.array(dirs);ed=affine('orientation_derivatives',lum,dus[:,None],conv=True,pad=2);ed=layer('orientation_polarity','relu',ed);ed=pool('orientation_cells','avg',ed,[2,2],[2,2]);pieces=[]
    for name,ks,offset,supportside in [('numeral',[0,1,2,3,4,5,7,8],0,9),('warning',warn,1,10)]:
        K=supportside;# On even/odd grids, offsets are chosen to retain the crop centre.
        if K%2:K+=1
        gap=(16-(K+4))//2;v=np.zeros((8,8,2*gap+1,2*gap+1));v[np.arange(8),np.arange(8),gap+offset,gap]=1
        cropped=affine(name+'_orient_window',ed,v,conv=True);nm=layer(name+'_orient_units','norm',cropped);W=[]
        xx,yy=coords(2*K);war=name=='warning'
        for k in ks:
            for s in [9.,11.,13.,15.]:
                for e in [.88,1.02,1.16]:
                    for h in [.87,1.02,1.17]:
                        for opt in [0,1]:
                            x=xx/(s*e);y=yy/s;adj=2*offset/s
                            ink=icon(k,x,y/h+adj,opt)
                            mask=tri(x,y+adj,.743) if war else ((x*x+y*y)<.765**2)&(abs(y)<.5*h)
                            a=-aa(ink,2*K);pad=np.pad(a,2,mode='edge');planes=[]
                            for f in dirs:
                                d=sum(f[i,j]*pad[i:i+2*K,j:j+2*K] for i in range(5) for j in range(5));d=np.maximum(d,0).reshape(K,2,K,2).mean((1,3));planes.append(d)
                            m=aa(mask,2*K).reshape(K,2,K,2).mean((1,3));z=np.array(planes);z=(z-(z*m).sum((1,2),keepdims=True)/max(m.sum(),1.e-7))*m
                            W.append(z/max(np.sqrt((z*z).sum()),1.e-6)*(12/s)/10.)
        q=affine(name+'_orientation_patterns',nm,np.array(W),conv=True);q=group_pool(name+'_orientation',q,len(ks),len(W)//len(ks),25);pieces+=contrast_bank(name+'_orient_agreement',q,ks,.30)
    return pieces

def warning_registration(lum,hi):
    K=20;xx,yy=coords(K);xxs,yys=coords(16);W=[];rim=[];parents=[]
    for k in warn:
        for i,(s,e) in enumerate(geometry(k)):
            if k==warn[0]:
                x=2*xxs/(s*e);y=2*yys/s;l,r,b,m=base(k,x,y);m=aa(m,16);rim.append(np.array([unit(aa(l,16),m)*.62,unit(aa(r,16),m)*.38,np.zeros((16,16))])*2/s)
            for h in [.88,1.,1.14]:
                for o in [0,1]:
                    x=xx/(s*e);y=(yy+3)/s;ink=icon(k,x,(y-.2)/h+.2,o);m=aa(tri(x,y,.748),K);a=-aa(ink,K)
                    W.append(np.array([unit(a,m)*.60,unit(a-smooth(a,1.32,2),m)*.58])*12/(s*6.4));parents.append(i)
    qa=affine('triangle_pose_filters','coarse',np.array(rim),conv=True,pad=1)
    T=np.fromfunction(lambda j,i:np.maximum(0,1-abs((j%5)/2-(i%3)))*np.maximum(0,1-abs((j//5)/2-(i//3))),(25,9))
    qa=affine('triangle_pose_interp',qa,np.kron(np.eye(15),T));qa=reshape('triangle_pose_maps',qa,[15,5,5]);qa=point('triangle_pose_prior',qa,np.eye(15)[parents])
    planes=[]
    for j,src in enumerate([lum,hi]):
        q=crop_at('warn_aligned_crop'+str(j),src,K+4,0,3);planes.append((layer('warn_aligned_norm'+str(j),'norm',q),1,j))
    z=fuse_features('warn_aligned_planes',planes);z=affine('registered_pictograms',z,np.array(W),conv=True);z=layer('registered_triangle','add',[z,qa]);z=group_pool('aligned_warning',z,len(warn),len(W)//len(warn),25)
    return contrast_bank('triangle_registration',z,warn,.52)

def material_votes(rgb):
    # Absolute opponent colour in the sign centre. Unlike mean-centred colour,
    # this remains discriminative on a solid disc, including a roundabout.
    t=np.arange(32)+.5-16;x,y=np.meshgrid(t,t);r2=x*x+y*y;g=np.exp(-(r2/(7.5**2))**2);g/=g.sum()
    z=affine('centre_material',rgb,np.kron(np.eye(3),g.ravel()[None,:]));z=affine('material_opponents',z,[[ -2.0,-1.,3.0],[3.0,-1.,-2.0],[1.5,1.,-2.5]],[-.012,-.012,-.015]);z=layer('material_positive','relu',z)
    A=np.zeros((43,3));A[mandatory,0]=1.16;A[[14,17],1]=.72;A[12,2]=.8;return affine('material_vote',z,A)

def cancel_marks(lum,hi):
    # Read what is being cancelled away from the oblique stroke rather than
    # relying on the slope common to all four signs. The empty case has no ink.
    ks=[6,32,41,42];K=20;xx,yy=coords(K);W=[];nvar=0
    for k in ks:
        for s in [9.,11.,13.,15.]:
            for e in [.90,1.08]:
                for h in [.89,1.08]:
                    x=xx/(s*e);y=yy/s;m=aa((x*x+y*y<.77**2)&(abs(x+y)>.255)&(abs(y)<.55),K)
                    if k==32:a=np.zeros_like(m)
                    else:
                        a=-aa(icon({6:5,41:9,42:10}[k],x,y/h),K)
                    # An additional zero-DC centre/surround filter measures the
                    # ink load in the two free halves (blank, cars or numerals).
                    l=unit(a,m);ed=unit(a-smooth(a,1.32,2),m);W.append(np.array([l*.50,ed*.50])*12/(s*4.0))
    features=[]
    for i,src in enumerate([lum,hi]):
        z=crop('cancel_offline_cut'+str(i),src,K+4);features.append(layer('cancel_offline_units'+str(i),'norm',z))
    src=fuse_features('cancel_reading_planes',[(features[0],1,0),(features[1],1,1)]);q=affine('marks_beside_bar',src,np.array(W),conv=True);q=group_pool('uncrossed_marks',q,len(ks),16,25)
    markings=affine('extant_markings',q,np.eye(4)[[0,2,3]])
    return contrast_bank('uncrossed',markings,[6,41,42],.90)

def smooth(a,sigma,r):
    t=np.arange(-r,r+1);g=np.exp(-.5*(t/sigma)**2);g/=sum(g);pad=np.pad(a,r,mode='edge');H,W=a.shape
    return sum(g[i]*g[j]*pad[i:i+H,j:j+W] for i in range(2*r+1) for j in range(2*r+1))

def joint_branch(lc,ch):
    # Coarse whole-sign likelihood: for each pose the border and contents must
    # agree before maximisation, supplementing the independent fine matchers.
    K=14;xx,yy=coords(K);W=[]
    for k in range(43):
        for s,e in geometry(k):
            for variant in [0,1]:
                x=xx*2/(s*e);y=yy*2/s;lum,red,blue,sup=base(k,x,y)
                ink=icon(k,x*(.95 if variant else 1.04),y,variant)
                # Use a material's own spectral code, not chroma in the ambient scene.
                lit=k in mandatory+[14,17];lum+=(.68 if lit else -.72)*ink
                if k in mandatory:blue*=1-ink
                if k in [14,17]:red*=1-ink
                if k in [9,10]:red=np.maximum(red,(x<0)*ink*.87)
                if k==26:
                    red+=ink*((y<.04)*.85+(y>.32)*(-.60));blue-=ink*((y>.04)&(y<.32))*.80
                m=aa(sup,K);l=unit(aa(lum,K),m);r=unit(aa(red,K),m);b=unit(aa(blue,K),m)
                W.append(np.array([.75*l,.23*r,.23*b])*2/s)
    z=affine('layout_match','coarse',np.array(W),conv=True);return group_pool('layout_likelihood',z,43,30,9)

def family_decision(base_scores,experts):
    # Hierarchical routing: adding an identity-only expert must not reduce the
    # evidence for the whole sign family. Re-centre the logits independently
    # for each photograph so each family's strongest score remains unchanged.
    detail=layer('family_and_experts','add',[base_scores,*experts]);res=[detail]
    for name,ks in [('speeds',[0,1,2,3,4,5,7,8]),('warnings',warn),('ends',ends)]:
        parts=[];A=np.eye(43)[ks];B=np.zeros((43,1));B[ks,0]=1
        for j,src in enumerate([base_scores,detail]):
            n=name+['_coarse','_fine'][j];z=affine(n+'_entries',src,A);z=reshape(n+'_row',z,[1,1,len(ks)]);z=pool(n+'_level','max',z,[1,len(ks)],[1,len(ks)]);parts.append(affine(n+'_shift',z,B*(1 if j==0 else -1)))
        res+=parts
    return layer('scores','add',res)

def nodal_flatness(hi,lum):
    # Blank signs should not win a low-contrast race simply because they have no
    # matching filter. Measure unsigned texture, relative to the whole crop.
    z=layer('detail_units','norm',hi);z=point('signed_detail',z,[[1],[-1]]);z=layer('both_halves','relu',z);z=point('absolute_detail',z,[[1,1]])
    ax=np.arange(32)+.5-16;x,y=np.meshgrid(ax,ax);rr=x*x+y*y
    disk=np.exp(-(rr/5.2**2)**2);return affine('central_texture',z,[disk.ravel()/disk.sum()])


def build(out):
    rgb,lum,lc,ch,hi=fixed_rgb_knots();shell,fam=shell_branch(lc,ch);pieces=[];F=np.zeros((43,12))
    for k in range(43):F[k,fam[k]]=.89
    pieces.append(affine('outline_vote',shell,F));bias=np.zeros(43)
    joint=joint_branch(lc,ch);pieces.append(affine('layout_vote',joint,np.eye(43)*.92));pieces.append(material_votes(rgb))
    experts=numeral_reader(lum,hi)+aligned_numbers()+cancel_marks(lum,hi)
    experts+=oriented_glyphs(lum)+warning_registration(lum,hi)
    for name,ks,K,gain,edgegain,intercept in [
        ('number',[0,1,2,3,4,5,7,8],18,.67,.76,-.87),
        ('cars',[9,10,16],20,.69,.44,-.66),
        ('danger',warn,20,.60,.67,-.52),
        ('direction',mandatory,24,.78,.55,-.64),
        ('cancel',ends,24,.89,.57,-.62),
        ('scarlet',[14,17],24,.68,.43,-.57)]:
        generated=[interior_bank(k,K) for k in ks];parts=[]
        for j,(source,coef) in enumerate([(lum,gain),(hi,edgegain)]):
            n=name+('_tone' if j==0 else '_edges');Kcur=K
            src=source
            if name=='danger':
                # Translate the input up by three pixels before taking the central crop.
                w=np.zeros((1,1,7,7));w[0,0,6,3]=1;src=affine(n+'_baseline',src,w,conv=True,pad=3)
            q=match_branch(n,ks,[x[j] for x in generated],src);A=np.zeros((43,len(ks)))
            for i,k in enumerate(ks):A[k,i]=coef
            parts.append(affine(n+'_vote',q,A))
        pieces+=parts;bias[ks]+=intercept
    tex=nodal_flatness(hi,lum);tw=np.zeros((43,1));tw[15]=-.86;tw[13]=-.23;bias[15]=.36;bias[13]=.13
    pieces.append(affine('texture_vote',tex,tw,bias));base_scores=layer('family_baseline','add',pieces);family_decision(base_scores,experts)
    p=Path(out);p.mkdir(parents=True,exist_ok=True);(p/'network.json').write_text(json.dumps({'layers':layers},indent=1));np.savez(p/'weights.npz',**weights)
if __name__=='__main__':build(sys.argv[1])
