"""Road sign classifier. All geometry is constructed here in dimensionless sign coordinates.
No fonts, assets, examples or other input files are needed.
"""
import sys, json, math
from pathlib import Path
import numpy as np

D = Path(sys.argv[1]); D.mkdir(exist_ok=True, parents=True)
L, W = [], {}
f32 = np.float32

def layer(name, op, ins, **kw):
    L.append(dict(name=name, op=op, inputs=ins if isinstance(ins,list) else [ins], **kw)); return name

def lin(name, src, w, b=None, conv=False, stride=1, padding=0):
    w = np.asarray(w, dtype=f32)
    if conv and w.ndim==2: w=w[:,:,None,None]
    W[name+'.W']=w
    W[name+'.b']=np.zeros(len(w),f32) if b is None else np.broadcast_to(b,(len(w),)).astype(f32)
    if conv: return layer(name,'conv',src,kernel=list(w.shape[-2:]),out=len(w),stride=stride,padding=padding)
    return layer(name,'dense',src,out=len(w))

def view(name,src,*shape): return layer(name,'reshape',src,shape=list(shape))
def pool(name,src,size,stride=None,op='maxpool'):
    return layer(name,op,src,size=list(size),stride=list(size if stride is None else stride))
def norm(name,src): return layer(name,'norm',src)
def channels(name,src,n,ids): return lin(name,src,np.eye(n,dtype=f32)[ids],conv=True)

def gauss(x,s): return np.exp(-x*x/(2*s*s))
def soften(x,s):
    if s<=0:return x
    k=int(math.ceil(2.5*s)); a=gauss(np.arange(-k,k+1),s);a/=a.sum()
    for ax in [-1,-2]:
        pads=[(0,0)]*x.ndim;pads[ax]=(k,k)
        z=np.pad(x,pads,mode='edge'); y=0
        for i,c in enumerate(a):
            sl=[slice(None)]*x.ndim;sl[ax]=slice(i,i+x.shape[ax]);y=y+c*z[tuple(sl)]
        x=y
    return x

# A bank of monotone hinge functions implements mild logarithmic compression.
# In particular, norm is applied AFTER exposure compression and opponent subtraction.
knots=np.array([0,.023,.060,.135,.28,.54,.8,1.]); ordinate=np.log1p(11*knots)/np.log(12.)
slopes=np.diff(ordinate)/np.diff(knots); coeff=np.diff(np.r_[0.,slopes]); n=len(slopes)
w=np.repeat(np.eye(3),n,axis=0)
lin('exposure_knots','image',w,-np.tile(knots[:-1],3),conv=True);layer('exposure_basis','relu','exposure_knots')
# Green, redness, blueness, yellowness. Green gives strong dark-red/white and blue/white contrast.
opp=np.array([[.12,.78,.10],[1,-.68,-.32],[-.55,-.45,1],[.47,.53,-1]])
lin('opponent','exposure_basis',np.repeat(opp,n,axis=1)*np.tile(coeff,3),conv=True)
norm('white_balance','opponent')
# DC removal is useful for colour constancy, but preserve some absolute colour evidence as well.
lin('absolute_tint','opponent',np.diag([0,2.4,2.4,2.4]),conv=True)
layer('surfaces','add',['white_balance','absolute_tint'])
# Soft clipping prevents a few unrelated high contrast objects dominating a correlation.
w=np.vstack([np.eye(4),np.eye(4)]);lin('range_limits','surfaces',w,[2.4]*4+[-2.4]*4,conv=True)
layer('range_hinges','relu','range_limits');lin('surface_maps','range_hinges',np.hstack([np.eye(4),-np.eye(4)]),-2.4,conv=True)
pool('coarse','surface_maps',(2,2),op='avgpool')
channels('green','opponent',4,[0]);channels('coarse_g','coarse',4,[0]);channels('coarse_red','coarse',4,[1])
# Long-wavelength luminance gradients are often an illumination effect. A green
# centre-surround contrast path chiefly sees ink, rather than the red frame.
u=np.arange(-4,5);g=gauss(u,1.75);g/=g.sum();dog=-np.outer(g,g);dog[4,4]+=1
lin('ink_detail','green',dog[None,None],conv=True,padding=4)

# Drawing primitives. Coordinates denote fractions of sign half width/height unless stated.
class Canvas:
    def __init__(self,x,y): self.x,self.y=np.broadcast_arrays(x,y);self.a=np.zeros(self.x.shape)
    def put(self,mask,val):self.a[mask]=val
    def disk(self,x,y,rx,ry=None,v=1):
        if ry is None:ry=rx
        self.put(((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1,v)
    def rect(self,x0,y0,x1,y1,v=1):self.put((self.x>=x0)&(self.x<=x1)&(self.y>=y0)&(self.y<=y1),v)
    def poly(self,p,v=1):
        x,y=self.x,self.y;m=np.zeros(x.shape,bool);a,b=p[-1]
        for c,d in p:
            if b!=d:m ^= ((b>y)!=(d>y)) & (x<(c-a)/(d-b)*(y-b)+a)
            a,b=c,d
        self.put(m,v)
    def line(self,p,width,v=1):
        x,y=self.x,self.y
        for (a,b),(c,d) in zip(p,p[1:]):
            dx,dy=c-a,d-b; t=np.clip(((x-a)*dx+(y-b)*dy)/(dx*dx+dy*dy+1e-20),0,1)
            self.put((x-a-dx*t)**2+(y-b-dy*t)**2<=(width*.5)**2,v)
    def ring(self,x,y,rx,ry,t,v=1):
        self.put((((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1)&(((self.x-x)/(rx-t))**2+((self.y-y)/(ry-t))**2>=1),v)

def grid(n,q):
    p=(np.arange(n*q)+.5)/q-n/2
    return np.meshgrid(p,p)
def reduce(a,n):
    s=a.shape[-1]//n
    return a.reshape(a.shape[:-2]+(n,s,n,s)).mean(axis=(-3,-1))
def sample(a,x,y):
    # Bilinear sampling of a canonical drawing, with constant edge extension.
    x=np.clip(x,0,a.shape[-1]-1);y=np.clip(y,0,a.shape[-2]-1);ix=x.astype(int);iy=y.astype(int)
    jx=np.minimum(ix+1,a.shape[-1]-1);jy=np.minimum(iy+1,a.shape[-2]-1)
    dx=x-ix;dy=y-iy
    return ((1-dy)*((1-dx)*a[...,iy,ix]+dx*a[...,iy,jx])+dy*((1-dx)*a[...,jy,ix]+dx*a[...,jy,jx]))

# The common sign plates. The material IDs are a drawing palette, not measured RGB.
RED,WHITE,BLACK,BLUE,YELLOW,GREY=1,2,3,4,5,6
redfam='ban';warn='warning';end='end';blue='obligation'
families=[redfam]*11+[warn,'diamond','yield','stop',redfam,redfam,'entry']+[warn]*14+[end]+[blue]*8+[end,end]
families[6]=end

def face(k,x,y,plain=False):
    fam=families[k];c=Canvas(x,y);r=np.hypot(x,y)
    if fam in (warn,'yield'):
        h=-y if fam=='yield' else y
        # Rounded corners of an isosceles triangle, offset from a smaller triangle.
        for s,dy,col in [(1,0,RED),(.69,.095,WHITE)]:
            u=x/s;z=(h-dy)/s
            points=[(0,-.92),(.92,.91),(-.92,.91)]
            m=Canvas(u,z);m.poly(points);m.line(points+[points[0]],.126);c.put(m.a>0,col)
    elif fam=='diamond':
        t=np.abs(x)+np.abs(y)
        c.put(t<1,WHITE);c.put(t<.945,BLACK);c.put(t<.915,WHITE);c.put(t<.71,YELLOW)
    elif fam=='stop':
        t=np.maximum(np.maximum(np.abs(x),np.abs(y)),(np.abs(x)+np.abs(y))/math.sqrt(2))
        c.put(t<=1,WHITE);c.put(t<=.95,RED)
        if not plain:
            # A block STOP inscription, constructed from strokes.
            for j,ch in enumerate('STOP'):
                m=Canvas((x-(-.555+.37*j))/.139,y/.30)
                if ch=='S': m.line([(.79,-.77),(.4,-.93),(-.44,-.9),(-.83,-.65),(-.8,-.31),(-.45,-.09),(.49,.15),(.82,.44),(.77,.73),(.39,.93),(-.37,.92),(-.85,.70)],.40)
                if ch=='T':m.rect(-.99,-1,.99,-.72);m.rect(-.20,-.78,.20,1)
                if ch=='O':m.ring(0,0,.99,1,.31)
                if ch=='P':m.rect(-.95,-1,-.59,1);m.ring(-.06,-.49,.90,.51,.30)
                c.put(m.a>0,WHITE)
    else:
        c.put(r<=1,WHITE)
        if fam==redfam:c.put(r<.984,RED);c.put(r<.786,WHITE)
        if fam==blue:c.put(r<.98,BLUE)
        if fam=='entry':c.put(r<.98,RED);c.rect(-.73,-.17,.73,.17,WHITE)
        if fam==end:c.put(r<.97,GREY);c.put(r<.915,WHITE)
    return c.a

# Numerals constructed as centre-lines inside a DIN-like condensed sans rectangle.
# Coordinates 0..1 are relative to a single digit. Line widths are fractions of cap height.
# A curved stroke is sampled along an explicit cubic Bezier.
def bez(p):
    p=np.array(p);t=np.linspace(0,1,11)[:,None];return ((1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]).tolist()
def chain(*parts):
    out=[]
    for p in parts:
        if isinstance(p,tuple):out.append(p)
        else:out.extend(p)
    return out

def digit_master(d,sty):
    # physical dimensions in cap-height units
    x,y=grid(96,1);x=x/72+.25;y=y/72+.5;c=Canvas(x,y)
    width=[.52,.55,.48][sty];t=[.128,.145,.117][sty];a=t/2;b=width-a;mid=width/2
    def ln(p):c.line(p,t)
    def B(*p):return bez(p)
    if d=='0':
        ln(chain(B((a,.29),(a,-.008),(b,-.008),(b,.29)),(b,.71),B((b,.71),(b,1.017),(a,1.017),(a,.71)),(a,.29)))
    elif d=='1':
        xp=mid+.036
        ln([(xp-.17,.22),(xp-.012,.079),(xp,.077),(xp,.94)]);c.rect(xp-t*.5,.88,xp+t*.5,1)
    elif d=='2':
        ln(chain(B((a,.246),(a,-.012),(b+.005,.005),(b,.24)),B((b,.24),(b+.018,.38),(b-.072,.46),(mid,.64)),(a,.92),(a,.936),(b,.936)));c.rect(a,.936-t/2,width,1)
    elif d=='3':
        ln(chain(B((a,.188),(.16,-.005),(b,.01),(b,.235)),B((b,.235),(b,.394),(b-.116,.465),(mid-.024,.465)),B((mid-.024,.465),(b+.018,.468),(b+.01,.611),(b,.73)),B((b,.73),(b,.973),(.142,1.002),(a,.818))))
    elif d=='5':
        ln(chain((b,.067),(a+.024,.067),(a,.478),B((a,.478),(mid,.353),(b,.42),(b,.609)),(b,.735),B((b,.735),(b,.983),(.139,1.008),(a,.817))));c.rect(a,.0,width,.12)
    elif d=='6':
        if sty!=1:
            ln(chain((b-.083,.06),B((b-.083,.06),(mid-.018,.255),(a,.488),(a,.647))))
        else:ln(chain(B((b-.012,.154),(mid+.05,-.09),(a,.1),(a,.38)),(a,.64)))
        ln(chain(B((a,.637),(a,.376),(b,.354),(b,.63)),(b,.735),B((b,.735),(b,1.015),(a,1.005),(a,.737)),(a,.637)))
    elif d=='7':ln([(a,.067),(b,.067),(b,.104),(a+.065,.938)])
    elif d=='8':
        ln(chain(B((mid,.468),(-.016,.468),(-.016,.063),(mid,.063)),B((mid,.063),(width+.008,.063),(width+.008,.468),(mid,.468))))
        ln(chain(B((mid,.468),(width+.015,.468),(width+.025,.937),(mid,.937)),B((mid,.937),(-.026,.937),(-.014,.468),(mid,.468))))
    return c.a
# Glyphs are independent components of explicit drawings, never combined statistically.
DIG={(d,sty):digit_master(d,sty) for d in '01235678' for sty in range(3)}
nums={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
def number(x,y,string,sty=0):
    cap=.85 if len(string)==2 else .76
    # The first digit '1' has a narrower advance in a three-digit inscription.
    widths=[(.32 if d=='1' else [.52,.55,.48][sty]) for d in string];gap=.090
    total=sum(widths)+gap*(len(string)-1); pos=-total/2
    z=np.zeros(x.shape)
    for d,w in zip(string,widths):
        X=x/cap-pos+( (.25+.036)-.18 if d=='1' else 0.)
        Y=y/cap+.5
        z=np.maximum(z,sample(DIG[d,sty],(X-.25)*72+47.5,(Y-.5)*72+47.5));pos+=w+gap
    return z

# Warning pictograms and direction arrows. Paths are relative to the sign, not a photo.
def triangle_symbol(k,x,y,style):
    c=Canvas(x,y);P=c.poly;R=c.rect;E=c.disk;line=c.line
    if k==11: # broad main road meeting two much smaller side roads
        P([(-.103,.64),(.103,.64),(.103,.25),(.39,.25),(.39,.155),(.103,.155),(.103,-.02),(.185,-.02),(0,-.38),(-.185,-.02),(-.103,-.02),(-.103,.155),(-.39,.155),(-.39,.25),(-.103,.25)])
    if k==18:
        P([(-.066,-.34),(.066,-.34),(.047,.32),(-.047,.32)]);E(0,.51,.079)
    if k in (19,20,21):
        if k==20:c.x=-c.x
        if k==21:
            p=chain((-.20,.61),(-.19,.42),bez([(-.19,.42),(-.19,.23),(.20,.17),(.20,-.01)]),bez([(.20,-.01),(.2,-.14),(-.06,-.15),(-.07,-.32)]));line(p,.145)
        else:
            p=chain((.125,.605),(.125,.27),bez([(.125,.27),(.14,.056),(.080,-.005),(-.18,-.16)]),(-.247,-.206));line(p,.16)
    if k==22:
        P([(-.61,.55),(-.47,.505),(-.37,.37),(-.285,.316),(-.19,.31),(-.10,.35),(-.017,.473),(.038,.455),(.13,.316),(.224,.30),(.302,.326),(.418,.477),(.603,.55),(.603,.59),(-.61,.59)])
    if k==23:
        # Car tilted on top of two snaking skid tracks.
        m=Canvas((x+.025)*.94+(y+.023)*.34,-(x+.025)*.34+(y+.023)*.94)
        car(m,0,-.02,.39,.35);c.put(m.a>.5,1)
        for dx in [-.12,.12]:
            line(chain(bez([(dx+.035,.14),(dx+.32,.25),(dx+.05,.30),(dx-.045,.35)]),bez([(dx-.045,.35),(dx-.24,.47),(dx+.20,.52),(dx+.16,.62)])),.044)
    if k==24:
        line([(-.24,.59),(-.24,-.26)],.084);line(chain((.265,.59),(.265,.29),bez([(.265,.29),(.265,.12),(.067,.09),(.067,-.066)]),(.067,-.26)),.09)
    if k==25:
        E(-.006,-.185,.086);line([(-.038,-.055),(-.186,.126),(-.24,.245)],.145)
        line([(-.225,.227),(-.193,.36),(-.32,.59)],.098);line([(-.194,.208),(-.08,.372),(.009,.576)],.091)
        line([(-.057,-.014),(.059,.135),(.21,.21)],.066);line([(-.10,.04),(-.03,.209),(.09,.232)],.060)
        line([(-.267,-.017),(.43,.491)],.034)
        P([(.086,.6),(.154,.518),(.243,.46),(.325,.34),(.40,.365),(.47,.47),(.561,.535),(.62,.60)])
    if k==26:
        c.line([(0,-.209),(0,.459)],.272 if style!=2 else .319)
        for j,y0 in enumerate([-.203,.106,.413]):E(0,y0,.093,.104,[.63,.035,.39][j] if style!=1 else .16)
    if k in (27,28):
        if style==2 and k==27:c.x=-c.x
        def person(px,py,s,forward):
            def pt(x,y):return px+forward*x*s,py+y*s
            E(*pt(.041,-.334),.086*s)
            line([pt(.017,-.199),pt(-.068,.091)],.153*s)
            line([pt(-.03,.076),pt(.08,.28),pt(.20,.478)],.089*s)
            line([pt(-.064,.081),pt(-.12,.295),pt(-.31,.41)],.09*s)
            line([pt(.006,-.162),pt(.117,.001),pt(.242,.05)],.066*s)
            line([pt(-.014,-.154),pt(-.186,-.016),pt(-.225,.16)],.064*s)
        if k==27:
            person(0,.104,.97,1)
            if style!=1:
                for dx in [-.42,-.208,.004,.216,.428]:P([(dx-.054,.494),(dx+.043,.494),(dx+.095,.59),(dx-.051,.59)])
        else:
            person(-.115,.16,.88,1);person(.236,.286,.59,1)
    if k==29:
        for px in [-.30,.30]:c.ring(px,.35,.17,.174,.042)
        line([(-.30,.35),(-.102,.12),(.024,.35),(-.30,.35)],.034);line([(.024,.35),(.218,.094),(.30,.35)],.041)
        E(.047,-.24,.073);line([(.021,-.12),(-.143,.052)],.118);line([(-.12,.061),(.03,.143),(-.045,.306)],.066);line([(.001,-.086),(.079,.069),(.216,.079)],.049)
    if k==30:
        for j in range(6):
            t=j*np.pi/3-np.pi/2;u,v=np.cos(t),np.sin(t); bx,by=0,.195
            line([(bx,by),(bx+.402*u,by+.402*v)],.046)
            for off in [-1,1]:
                for start in [.185,.283]:line([(bx+start*u,by+start*v),(bx+(start+.075)*u-.078*off*v,by+(start+.075)*v+.078*off*u)],.030)
    if k==31:
        # Leaping deer, with neck, antlers and two bent pairs of legs.
        if style%2:c.x=-c.x
        P([(-.37,.27),(-.42,.158),(-.49,.114),(-.37,.14),(-.235,.03),(-.038,-.013),(.105,-.065),(.173,-.229),(.223,-.248),(.294,-.19),(.34,-.146),(.311,-.094),(.234,-.109),(.168,.09),(.013,.20),(-.178,.254)])
        line([(-.28,.20),(-.325,.345),(-.51,.475),(-.55,.553)],.064);line([(-.20,.216),(-.185,.356),(-.388,.54)],.052)
        line([(.081,.13),(.248,.196),(.348,.377)],.063);line([(.107,.095),(.301,.112),(.408,.266)],.050)
        for dx in [0,.055]:
            line([(.20+dx,-.206),(.143+dx,-.332),(.112+dx,-.37)],.025);line([(.15+dx,-.32),(.18+dx,-.381)],.019)
    return c.a

def car(c,cx,cy,w,h):
    # Frontal silhouette, with a light windscreen and the tyres projecting below.
    def pts(p):return [(cx+x*w,cy+y*h) for x,y in p]
    c.poly(pts([(-.49,.34),(-.49,-.072),(-.38,-.20),(-.27,-.467),(-.17,-.5),(.225,-.5),(.31,-.425),(.40,-.166),(.49,-.074),(.49,.34)]))
    c.poly(pts([(-.29,-.16),(-.202,-.395),(.214,-.395),(.30,-.16)]),0)
    for x in [-.335,.335]:c.rect(cx+w*(x-.096),cy+.265*h,cx+w*(x+.096),cy+.49*h)

def truck(c,front=False):
    if front:
        c.rect(-.48,-.5,.48,.32);c.rect(-.39,-.305,.386,-.04,0);c.rect(-.40,.29,-.23,.49);c.rect(.23,.29,.40,.49)
    else:
        c.rect(-.60,-.28,.165,.165);c.poly([(.180,-.207),(.38,-.207),(.536,-.054),(.563,.003),(.563,.17),(.18,.17)]);c.rect(-.56,.16,.55,.22)
        c.poly([(.235,-.149),(.359,-.149),(.457,-.037),(.235,-.037)],0)
        for px in [-.382,.365]:c.disk(px,.202,.108);c.disk(px,.202,.039,v=0)

def vehicle_symbol(k,x,y,sty):
    c=Canvas(x,y)
    if k==16:
        if sty%3==2:c.x=-c.x
        truck(c);return c.a,0*c.a
    ct=k in (10,42)
    car(c,.245,.035,.437,.43);right=c.a.copy();c.a[:]=0
    if ct:
        m=Canvas((x+.252)/.48,(y+.035)/.62);truck(m,True);c.a=m.a
    else:car(c,-.245,.035,.436,.435)
    return np.maximum(c.a,right),c.a.copy()

def arrowhead(c,base,tip,width):
    a,b=base;u,v=tip;dx,dy=u-a,v-b;r=np.hypot(dx,dy);ox,oy=-dy/r*width/2,dx/r*width/2
    c.poly([(a+ox,b+oy),(u,v),(a-ox,b-oy)])
def arrows(k,x,y,sty):
    if k in (34,37,39):x=-x
    c=Canvas(x,y)
    if k in (38,39):
        # Keep right: large arrow inclined to the lower right, frequently with a bevelled head.
        c.poly([(-.592,-.345),(-.355,-.584),(.207,-.022),(.421,-.236),(.457,.454),(-.238,.421),(-.015,.206)])
    if k in (33,34):
        c.rect(-.32,.015,-.083,.62)
        c.line(chain((-.201,.116),bez([(-.201,.116),(-.201,-.143),(-.115,-.229),(.182,-.229)]),(.27,-.229)),.236)
        arrowhead(c,(.168,-.229),(.60,-.229),.657)
    if k in (35,36,37):
        dx=-.082 if k!=35 else 0.
        c.rect(dx-.127,-.221,dx+.127,.667);arrowhead(c,(dx,-.157),(dx,-.684),.664)
        if k!=35:
            c.line(chain((dx,.339),bez([(dx,.339),(dx,.148),(dx+.07,.091),(.30,.091)])),.191);arrowhead(c,(.254,.091),(.615,.091),.482)
    if k==40:
        for i in range(3):
            t=np.linspace(0,1,16)*1.24+i*2*np.pi/3-.25
            p=list(zip(.47*np.cos(t),-.47*np.sin(t)));c.line(p,.163)
            a=t[-1];arrowhead(c,p[-1],(.464*np.cos(a)-.29*np.sin(a),-.464*np.sin(a)-.29*np.cos(a)),.394)
    return c.a

# Canonical drawings are immutable construction elements. Every transformed drawing
# produces a separate template, with zero mean and energy normalization local to it.
N=128
u=(np.arange(N)+.5)*2.4/N-1.2
X,Y=np.meshgrid(u,u)
SYMBOL={};REDOBJ={}
for k in range(43):
    for st in range(3):
        rr=np.zeros_like(X)
        if k in nums: a=number(X,Y,nums[k],st)
        elif families[k]==warn: a=triangle_symbol(k,X,Y,st)
        elif k in (9,10,16,41,42):a,rr=vehicle_symbol(k,X,Y,st)
        elif families[k]==blue:a=arrows(k,X,Y,st)
        else:a=np.zeros_like(X)
        if families[k]==end:
            # A group of narrow black cancellation stripes running from lower left to upper right.
            u=(X+Y*(.88 if st==2 else 1.0))/math.sqrt(2)
            stripe=(np.abs(u)<.24)&((u+.025)%.096<.044)&(X*X+Y*Y<.96**2)
            a=np.maximum(.58*a,stripe.astype(float))
        SYMBOL[k,st]=a;REDOBJ[k,st]=rr

def sym(k,x,y,st=0):return sample(SYMBOL[k,st],(x+1.2)*N/2.4-.5,(y+1.2)*N/2.4-.5)

def modcrop(name,src,C,h):
    if h==32:return norm(name,src)
    if C!=1:raise ValueError('single channel required')
    # A valid convolution with just the centre tap is an exact central crop.
    k=33-h;m=np.zeros((1,1,k,k),f32);m[0,0,k//2,k//2]=1
    return norm(name,lin(name+'_crop',src,m,conv=True))

# Plate templates: colour and brightness against the surround and each other.
# Ignore symbol details in the shared shell detectors; the symbol banks provide that evidence.
PAL=np.array([[0,0,0,0],[-.64,1,-.22,.12],[.90,0,0,0],[-1,0,0,0],[-.65,-.22,1,-.65],[.53,.14,-.62,1],[-.2,0,0,0]])
plate_ids=[0,11,12,13,14,17,32,35]; plate_index={families[k]:j for j,k in enumerate(plate_ids)}
# Avoid allowing irrelevant colour planes to swamp a neutral sign.
importance={redfam:[.49,.58,0,0],warn:[.50,.58,0,0],'yield':[.53,.56,0,0],'stop':[.45,.58,0,0],'entry':[.54,.55,0,0],blue:[.26,0,.72,0],end:[.74,0,0,0],'diamond':[.27,.16,0,.69]}

def shift_bank(v,delta):
    dx,dy=delta
    z=np.roll(v,(dy,dx),axis=(-2,-1));
    if dx<0:z[...,dx:]=0
    if dx>0:z[...,:dx]=0
    if dy<0:z[...,dy:,:]=0
    if dy>0:z[...,:dy,:]=0
    return z

def center_masked(a,m):
    return (a-(a*m).sum(axis=(-2,-1),keepdims=True)/max(1.e-7,m.sum()))*m

def plate_bank():
    n=16;xx,yy=grid(n,3);bank=[];bias=[]
    for k in plate_ids:
        fam=families[k];weights=np.array(importance[fam])[:,None,None];grp=[];bs=[]
        for rad in [4.85,5.55,6.25,6.98]: # radius in half-resolution cells
            for aspect,tilt in [(1,0),(.85,0),(1.13,0),(1,.12),(1,-.12)]:
                x=(xx+tilt*yy)/(rad*aspect);y=yy/rad
                f=face(k,x,y)
                raw=PAL[f.astype(int)].transpose(2,0,1)
                if fam==end:raw[0]=np.where(f>0,.85-1.25*sym(32,x,y),0)
                # Shape evidence should be strongest in the frame, rather than the unknown motif area.
                radius=np.hypot(x,y);mask=np.clip((1.28-radius)/.19,0,1)
                if fam in (warn,'yield','diamond','stop'):mask=np.clip((1.5-np.maximum(np.abs(x),np.abs(y)))/.34,0,1)
                importance_map=np.ones_like(x)
                if fam in (redfam,warn,blue):
                    interior=(f==WHITE if fam!=blue else f==BLUE)
                    importance_map[interior]=.65
                m=reduce(mask*importance_map,n)
                v=reduce(raw,n);v=soften(v,.37)
                v=center_masked(v,m)
                # Positive evidence that red/blue/yellow is present, not simply less absent.
                for ch in [1,2,3]:v[ch]+=.48*reduce(raw[ch]*mask,n)
                v*=weights/(np.sqrt((v*v).sum(axis=(-2,-1),keepdims=True))+1e-8)/math.sqrt(max(1,m.sum()))
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:grp.append(shift_bank(v,(dx,dy)).ravel());bs.append(-.016*(abs(dx)+abs(dy)))
        bank.extend(grp);bias.extend(bs)
    name=lin('plate_match','coarse',bank,bias);nvar=len(bank)//len(plate_ids)
    return pool('plate_best',view('plate_grid',name,len(plate_ids),1,nvar),(1,nvar))

# General class-conditional bank of local normalized correlations. Matching is
# parallel over small translations with dense rows, then max pooled over poses.
# Input divided by its own local spatial deviation, hence lighting and crop background
# have much less effect on a small dark glyph inside a light sign.
def local_bank(name,ks,h,src,variants,make,offsets,detail=False):
    bank=[];bias=[]; xx,yy=grid(h,2)
    for k in ks:
        for v in variants:
            x=xx;y=yy
            a,m,penalty=make(k,v,x,y)
            a=soften(reduce(a,h),.46);m=reduce(m,h)
            if detail:a=a-soften(a,1.75)
            a=center_masked(a,m)
            # Correlation units: the norm of a z-scored H x H patch is H.
            a=a/(np.sqrt(np.sum(a*a))+1.e-8)/h
            for dx,dy in offsets:
                bank.append(shift_bank(a,(dx,dy)).ravel());bias.append(-penalty-.007*(abs(dx)+abs(dy)))
    m=lin(name+'_match',src,bank,bias);nv=len(bank)//len(ks)
    return pool(name+'_best',view(name+'_grid',m,len(ks),1,nv),(1,nv))

def lift(name,src,idx,coeff=1,bias=None):
    n=max(idx)+1
    m=np.zeros((43,n),np.float32)
    for k,q in enumerate(idx):
        if q>=0:m[k,q]=np.broadcast_to(coeff,(43,))[k]
    return lin(name,src,m,bias)

offs=[(x,y) for y in [-2,-1,0,1,2] for x in [-2,-1,0,1,2]]
plate=plate_bank();logits=[lift('plate_scores',plate,[plate_index[f] for f in families],1.30)]

# A second, coarse full-sign bank keeps the colour of the shell and the brightness
# of the pictogram in registration. It is less sensitive to small shape inaccuracies
# than high-frequency glyph matching, but can distinguish blank and inscribed signs.
def whole_bank():
    h=16;xx,yy=grid(h,3);bank=[];bias=[]
    for k in range(43):
        fam=families[k]
        for rad in [4.9,5.65,6.40,7.1]:
            for asp,t in [(1,0),(.85,0),(1.11,0),(1,.115),(1,-.115)]:
                x=(xx+t*yy)/(rad*asp);y=yy/rad
                f=face(k,x,y);base=PAL[f.astype(int),0]
                g=sym(k,x,y);inside=f!=0
                if fam==end:base=np.where(inside,base*(1-g)-g,0)
                elif fam==blue:base=np.where(f==BLUE,base*(1-g)+.92*g,base)
                elif k in nums or fam==warn or k in (9,10,16):base=np.where(f==WHITE,base*(1-g)-g,base)
                # Middle of the plate and sign boundaries count; distant scenery does not.
                m=reduce(np.clip((1.42-np.maximum(abs(x),abs(y)))/.39,0,1),h)
                z=center_masked(soften(reduce(base,h),.36),m)
                z/=((np.sqrt(np.sum(z*z))+1e-8)*math.sqrt(m.sum()))
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:bank.append(shift_bank(z,(dx,dy)).ravel());bias.append(-.009*(abs(dx)+abs(dy)))
    m=lin('whole_match','coarse_g',bank,bias);nv=len(bank)//43
    return pool('whole_best',view('whole_grid',m,43,1,nv),(1,nv))
whole=whole_bank();logits.append(lift('whole_scores',whole,list(range(43)),.91))

# OCR: crop the centres and normalize there, since very dark numerals should not
# compete with trees and sky in the variance estimate. Hand-specified font sizes,
# slight shear and distinct type styles are alternative individual inscriptions.
h=16;digit_input=modcrop('text_contrast','green',1,h);speed=[0,1,2,3,4,5,7,8]
var=[(r,a,t,st) for r in [9.3,10.8,12.3,13.9] for a,t in [(1,0),(.83,0),(1.14,0),(1,.115),(1,-.115)] for st in range(3)]
def dig_draw(k,v,x,y):
    r,asp,t,st=v;xx=(x+t*y)/(r*asp);yy=y/r
    mask=np.clip((.80-np.hypot(xx,yy))/.12,0,1)
    return -sym(k,xx,yy,st),mask,0
b=local_bank('speed',speed,h,digit_input,var,dig_draw,offs)
idx=[speed.index(k) if k in speed else -1 for k in range(43)]
logits.append(lift('numeral_scores',b,idx,1.18,[-.50 if k in speed else 0 for k in range(43)]))

# Check the class-specific prefix independently of the common trailing zero. Here
# DoG contrast suppresses uneven lighting and sharpens stroke curvature evidence.
detail_input=modcrop('stroke_contrast','ink_detail',1,h)
var=[(r,a,t,st) for r in [9.1,10.4,11.7,13.0,14.3] for a,t in [(1,0),(.83,0),(1.14,0),(1,.13),(1,-.13)] for st in [0,1]]
def prefix_draw(k,v,x,y):
    r,asp,t,st=v;xx=(x+t*y)/(r*asp);yy=y/r
    a=-sym(k,xx,yy,st);mx=.20 if len(nums[k])==3 else -.004
    mask=np.clip((.775-np.hypot(xx,yy))/.13,0,1)*np.clip((mx+.16-xx)/.23,0,1)*np.clip((.59-abs(yy))/.11,0,1)
    return a,mask,0
b=local_bank('speed_prefix',speed,h,detail_input,var,prefix_draw,offs,True)
logits.append(lift('digit_detail_scores',b,idx,.86,[-.23 if k in speed else 0 for k in range(43)]))

# The same principle on triangular signs, with the mask following their white field.
warns=[k for k in range(43) if families[k]==warn]
h=20;pic_input=modcrop('symbol_contrast','green',1,h)
var=[(r,asp,t,ys) for r in [9.5,11.4,13.4] for asp,t in [(1,0),(.8,0),(1.17,0),(1,.14),(1,-.14)] for ys in [.87,1.13]]
def warn_draw(k,v,x,y):
    r,asp,t,ys=v; xx=(x+t*y)/(r*asp);yy=y/r
    # triangle interior; slightly enlarged at bottom to accommodate graphic scale.
    m=np.clip((yy+.61-2.03*abs(xx))/.14,0,1)*np.clip((.82-yy)/.12,0,1)
    return -sym(k,xx,(yy-.17)/ys+.17,1 if (k==31 and ys>1) else 0),m,0
b=local_bank('warning_sign',warns,h,pic_input,var,warn_draw,offs)
idx=[warns.index(k) if k in warns else -1 for k in range(43)]
logits.append(lift('warning_scores',b,idx,1.15,[-.265 if k in warns else 0 for k in range(43)]))

warn_detail=modcrop('pictogram_detail','ink_detail',1,h)
# Edges of the printed motif and enclosed white gaps form a second cue, quite
# different from the coarse ink density; max over offsets also here.
vd=[(r,asp,t,ys) for r in [9.7,11.8,13.8] for asp,t in [(.85,0),(1,0),(1.14,0),(1,.12),(1,-.12)] for ys in [.93,1.08]]
b=local_bank('pictogram_strokes',warns,h,warn_detail,vd,warn_draw,offs,True)
logits.append(lift('warning_detail_scores',b,idx,.81,[-.165 if k in warns else 0 for k in range(43)]))

# In a warning triangle the entire usable white field is low in the image.
# Project the image into a normalized view of that field for each physical size;
# use norm *inside* each view, instead of letting trees/ring dominate its variance.
def field_sample(rad,dx,dy,h):
    xx,yy=grid(h,1);x=xx/(h/2)*.61;y=yy/(h/2)*.645+.104
    # Outside the triangle replicate the nearest interior column. The generated
    # template undergoes the identical coordinate transform.
    half=np.maximum(.015,(y+.52)*.48)
    x=np.clip(x,-half,half);y=np.maximum(y,-.48)
    ix=15.5+x*rad+dx;iy=15.5+y*rad+dy
    return x,y,ix,iy

def white_field_scores():
    h=14;views=[(rad,dx,dy) for rad in (9.7,11.6,13.6) for dy in [-1.3,1.3] for dx in [-1.3,1.3]]
    A=[];ii=np.arange(h*h)
    for r,dx,dy in views:
        x,y,ix,iy=field_sample(r,dx,dy,h);ix=ix.ravel();iy=iy.ravel();jx=ix.astype(int);jy=iy.astype(int);fx=ix-jx;fy=iy-jy
        m=np.zeros((h*h,1024),f32)
        for ox,wx in [(0,1-fx),(1,fx)]:
            for oy,wy in [(0,1-fy),(1,fy)]:m[ii,(jy+oy)*32+jx+ox]=wx*wy
        A.extend(m)
    a=lin('white_field_views','green',A);a=norm('white_field_contrast',view('white_fields',a,len(views),h,h))
    # Reuse one analytic motif bank for every view.
    mats=[];V=[(sx,sy,sh,dx,dy) for sx,sy in [(.83,.89),(1,1),(1.15,1.1),(1.15,.87),(.86,1.12)] for sh,dx,dy in [(0,0,0),(.11,0,0),(-.11,0,0),(0,.076,0),(0,-.076,0),(0,0,.068),(0,0,-.068)]]
    for k in warns:
        for sx,sy,sh,dx,dy in V:
            X=(x+sh*y-dx)/sx;Y=(y-.13-dy)/sy+.13
            z=-sym(k,X,Y,2 if sx>1 else (1 if sx<1 else 0));z=soften(z,.64)
            # Only the interior points count in the correlation, not the padded repetitions.
            origx=grid(h,1)[0]/(h/2)*.61
            mask=np.where(abs(origx-x)<.001,1.,.09)
            z=center_masked(z,mask);z/=((np.sqrt((z*z).sum())+1e-8)*np.sqrt(mask.sum()))
            mats.append(z.ravel())
    # Read each normalized view independently and pool its motif responses.
    view_scores=[]
    for j in range(len(views)):
        raw=channels('field_'+str(j),a,len(views),[j]);z=lin('field_match_'+str(j),raw,mats)
        z=pool('field_best_'+str(j),view('field_grid_'+str(j),z,len(warns),1,len(V)),(1,len(V)))
        view_scores.append(z)
    # Elementwise maximum compiled as max(a,b)=a+ReLU(b-a). Use a balanced tree.
    i=0
    while len(view_scores)>1:
        nxt=[]
        for j in range(0,len(view_scores),2):
            if j+1==len(view_scores):nxt.append(view_scores[j]);break
            va,vb=view_scores[j:j+2];pre='view_max_'+str(i);i+=1
            neg=lin(pre+'_neg',va,-np.eye(len(warns)),conv=True)
            d=layer(pre+'_difference','add',[vb,neg]);d=layer(pre+'_positive','relu',d)
            nxt.append(layer(pre,'add',[va,d]))
        view_scores=nxt
    return view_scores[0]
b=white_field_scores()
logits.append(lift('white_field_scores',b,idx,1.06,[-.46 if k in warns else 0 for k in range(43)]))

# Local evidence for mandatory arrows, vehicles, and the grey cancellation marks.
icons=[9,10,16,6,32,41,42,33,34,35,36,37,38,39,40]
var=[(r,asp,t,st) for r in [9.5,11.4,13.4] for asp,t in [(1,0),(.83,0),(1.16,0),(1,.145),(1,-.145)] for st in [0,2]]
def icon_draw(k,v,x,y):
    r,asp,t,st=v;xx=(x+t*y)/(r*asp);yy=y/r
    if k in (9,10,16):m=np.clip((.77-np.hypot(xx,yy))/.13,0,1)
    else:m=np.clip((.94-np.hypot(xx,yy))/.20,0,1)
    return (1 if families[k]==blue else -1)*sym(k,xx,yy,st),m,0
b=local_bank('disc_symbol',icons,h,pic_input,var,icon_draw,offs)
idx=[icons.index(k) if k in icons else -1 for k in range(43)]
logits.append(lift('disc_scores',b,idx,1.07,[-(.385 if k in (9,10,16) else .35) if k in icons else 0 for k in range(43)]))

# A white, empty centre distinguishes the no-vehicles sign. Measure fine-scale
# contrast by absolute differences; shape/colour evidence is supplied separately.
def diff_bank():
    w=np.zeros((4,1,3,3));w[0,0,1,0]=-1;w[0,0,1,2]=1;w[1]=-w[0];w[2,0,0,1]=-1;w[2,0,2,1]=1;w[3]=-w[2]
    lin('spatial_difference',channels('plate_light','surface_maps',4,[0]),w,conv=True)
    layer('one_sided_difference','relu','spatial_difference')
    a=np.zeros((43,4,30,30));x,y=grid(30,1);g=np.exp(-.5*((x/4.4)**2+(y/3.4)**2));g/=g.sum()
    a[15]=-1.40*g;a[13]=-.16*g;a[12]=-.13*g
    b=np.zeros(43);b[15]=.23
    return lin('empty_centre','one_sided_difference',a.reshape(43,-1),b)
logits.append(diff_bank())
# Red ink on the left half, dark silhouette on the right: the signature of the
# overtaking ban. This is insensitive to shifts at the scale of a letter stroke.
x,y=grid(16,1)
w=np.zeros((43,16,16));b=np.zeros(43)
for k in [9,10,16]+speed+[15]:
    a=gauss(x+1.68,1.17)*gauss(y+.14,1.27);a/=a.sum()
    s=gauss(x-1.8,1.4)*gauss(y,1.65);s/=s.sum()
    w[k]=(.31 if k in (9,10) else -.068)*(a-s)
logits.append(lin('vehicle_colours','coarse_red',w.reshape(43,-1),b))

# Area-colour cues complement outline correlation: a STOP sign has red above
# AND below the word; a no-entry sign has two red caps separated by a white bar.
# Comparing cap colour to a flank rejects hollow red rims and colour casts.
x,y=grid(16,1)
def unit(a):return a/a.sum()
cap=unit(gauss(x,2.28)*(gauss(y-2.95,.83)+gauss(y+2.95,.83)))
flank=unit(gauss(y,2.45)*(gauss(x-4.45,1.08)+gauss(x+4.45,1.08)))
bar=unit(gauss(y,.64)*gauss(x,2.35));centre=unit(gauss(x,2.1)*gauss(y,2.35))
cr=np.zeros((43,16,16));cl=cr.copy();bi=np.zeros(43)
cr[14]=.51*cap-.35*flank+.12*centre;bi[14]=-.22
cr[17]=.54*cap-.22*flank-.32*bar;cl[17]=.33*(bar-cap)
# The priority diamond has a positive yellow interior, not just a yellow border.
yc=channels('coarse_yellowness','coarse',4,[3]);cy=np.zeros_like(cr);cy[12]=.34*centre
bi[12]=-.15
logits.extend([lin('red_areas','coarse_red',cr.reshape(43,-1),bi),lin('white_bar','coarse_g',cl.reshape(43,-1)),lin('gold_centre',yc,cy.reshape(43,-1))])
layer('scores','add',logits)
(D/'network.json').write_text(json.dumps({'layers':L},indent=1)+'\n')
np.savez(D/'weights.npz',**W)
