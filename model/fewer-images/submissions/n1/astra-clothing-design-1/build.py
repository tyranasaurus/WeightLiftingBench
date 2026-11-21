"""Compile a feed-forward pattern recognizer. No data input.

All lengths in the pattern makers are fractions of a garment length, a
foot length, or a bag width. The generator draws ONE independent pattern
at a time. No operation compares drawings or estimates anything from a
population of drawings.
"""
import sys, os, json
import numpy as np

rng = np.random.default_rng(480217)
U = rng.uniform

def chance(p):
    return U() < p

def between(a, b):
    return U(a, b)

def choice(a):
    return a[rng.integers(len(a))]

# Subpixel samples for antialiasing the individual template weights.
side = 42
v = (np.arange(side) + .5) / side - .5
X, Y = np.meshgrid(v, v)

def canvas(width, height, kind):
    # Bounding rectangle of the pattern, preserving the aspect ratio.
    # Small product-photo placement, tilt, and perspective variations.
    scale = max(width, height) / U(.90, 1.015)
    dx = U(-.025, .025)
    dy = U(-.036, .036) + ({'shoe': .035, 'bag': .012}.get(kind, 0.))
    angle = U(-.055, .055) * (1 if kind == 'shoe' else .65)
    x = scale * ((X-dx)*np.cos(angle) + (Y-dy)*np.sin(angle))
    y = scale * ((Y-dy)*np.cos(angle) - (X-dx)*np.sin(angle)) + height/2
    return x, y

def ip(t, knots, values):
    return np.interp(t, knots, values)

def clip(t):
    return np.clip(t, 0., 1.)

def bell(t):
    return np.exp(-np.minimum(70., t*t))

def soften14(a):
    # Separable lowpass with zero outside the photographic field.
    p = np.pad(a, ((0,0),(2,2)))
    t = (p[:,:-4] + 4*p[:,1:-3] + 6*p[:,2:-2] + 4*p[:,3:-1] + p[:,4:]) / 16
    p = np.pad(t, ((2,2),(0,0)))
    return (p[:-4] + 4*p[1:-3] + 6*p[2:-2] + 4*p[3:-1] + p[4:]) / 16

def feature(a, edge):
    a = a.reshape(14,side//14,14,side//14).mean(axis=(1,3))
    a = a - edge*soften14(a)
    a = a-a.mean()
    return (a/(np.sqrt((a*a).mean()+1.e-5)*a.size)).astype(np.float32).ravel()

def wrap(mask, value):
    # Separate projections of this one illustration only. value is the
    # lighting/material channel, not a second, independently drawn object.
    val = np.where(mask, np.clip(value, .025, 1.), 0.)
    return feature(.83*clip(val/.14)+.17*clip(val/.55), .43), feature(val, .18)

# ------------------------- Sewing patterns -------------------------------
def upper(k):
    # A yoke, two tapered sleeves, front panels, neckline and a hem.
    # Half widths are in units of shoulder-to-hem length.
    dress = k == 3
    shirt = k == 6
    coat = k == 4
    knit = k == 2
    tee = k == 0
    if tee:
        style = choice(['short']*8+['tank']*2+['cap','loose','elbow'])
        chest = U(.305,.465); shoulder = chest*U(.78,1.03)
        waist = chest*U(.85,1.03); hem = chest*U(.89,1.11)
        curved = U(-.01,.045)
    elif knit:
        style = choice(['long']*10+['threequarter','loose_long','raglan'])
        chest = U(.31,.465); shoulder = chest*U(.76,1.)
        waist = chest*U(.88,1.04); hem = chest*U(.79,1.03)
        curved = U(-.014,.031)
    elif coat:
        style = choice(['jacket']*5+['overcoat']*4+['blazer']*3+['parka']*2)
        chest = (U(.235,.325) if style=='overcoat' else U(.29,.435))
        shoulder = chest*U(.77,.98); waist = chest*U(.82,1.02)
        hem = chest*(U(1.01,1.23) if style=='overcoat' else U(.89,1.12))
        curved = U(-.012,.052)
    elif shirt:
        style = choice(['button']*6+['blouse']*3+['shortshirt']*3+['tunic']*2+['threequarter']*2+['cap'])
        chest = U(.285,.421); shoulder = chest*U(.78,1.)
        waist = chest*U(.79,1.02); hem = chest*U(.93,1.15)
        curved = U(.025,.125) if style in ('button','shortshirt','tunic') else U(0,.074)
    else:
        style = choice(['bare']*3+['strap']*2+['short']*3+['cap']*2+['long']*2+['threequarter','pencil','flared'])
        chest = U(.153,.26); shoulder = chest*U(.74,1.)
        waist = chest*U(.60,.96); hem = chest*U(1.08,2.24)
        if style=='pencil': hem=chest*U(.85,1.16)
        curved = U(-.01,.054)
    sleeveless = style in ('tank','bare','strap','pencil','flared')
    cap = style=='cap'
    short = style in ('short','loose','shortshirt','cap','elbow')
    collar = (shirt and (style in ('button','shortshirt') or chance(.24))) or (coat and not chance(.16))
    # Estimate extent from construction dimensions (not from raster samples).
    drop = U(.055,.12) * (.67 if dress else 1.)
    sleeve_top = shoulder
    if sleeveless:
        sleeve_y = 0.; sleeve_tip=shoulder; cuff=0.
    elif short:
        sleeve_y = U(.21,.40) if not cap else U(.12,.24)
        if style=='elbow': sleeve_y=U(.37,.53)
        if dress: sleeve_y*=.64
        sleeve_tip=shoulder + (U(.13,.29) if not cap else U(.055,.15))*(.65 if dress else 1.)
        cuff = U(.103,.19)*(.64 if dress else 1.)
    else:
        if dress: sleeve_y=U(.42,.74)
        elif style=='overcoat': sleeve_y=U(.59,.87)
        elif style in ('threequarter','tunic'): sleeve_y=U(.5,.84)
        else: sleeve_y=U(.83,1.045)
        sleeve_tip = chest + (U(.024,.15) if not dress else U(.04,.13))
        cuff=U(.062,.13)*(.68 if dress else 1.)
    width = 2*max(chest,hem,shoulder,sleeve_tip)
    height = max(1.,sleeve_y+.015)
    x,y=canvas(width,height,'cloth')
    # Mild nonrigid drape, shared by all parts of the item.
    x=x-U(-.022,.022)*np.sin(np.pi*y)+U(-.02,.02)*(y-.5)
    s=np.abs(x)
    asym = U(-.018,.018)*np.tanh(x/.065)
    yy = y+asym
    bust_y=U(.16,.29) if not dress else U(.115,.20)
    waist_y=U(.34,.50) if dress else U(.47,.65)
    hiplevel = (waist_y+.19) if dress else .79
    hip = (waist*U(1.12,1.5) if dress else .48*waist+.52*hem)
    bwidth=ip(yy,[0.,drop,bust_y,waist_y,hiplevel,1.],[shoulder*.72,shoulder,chest,waist,hip,hem])
    collar_w = shoulder*(U(.36,.64) if tee else U(.29,.54))
    collar_y = U(.035,.14)
    if shirt: collar_y=U(.055,.19)
    if coat: collar_y=U(.018,.09)
    if sleeveless:
        collar_w=shoulder*U(.49,.84); collar_y=U(.085,.235)
        bwidth=ip(yy,[0.,bust_y*.48,bust_y,waist_y,hiplevel,1.],
                  [shoulder,shoulder*.96,chest,waist,hip,hem])
    if dress: collar_y*=.7
    vee=(shirt and chance(.51)) or (dress and chance(.32)) or (knit and chance(.24))
    neck = collar_y*np.maximum(0,1-(s/collar_w)**(1. if vee else 2.4))
    shoulderline=drop*clip((s-collar_w)/max(.035,shoulder-collar_w))**1.22
    top=np.maximum(neck,shoulderline)
    bottom=1.-curved*(s/max(.09,hem))**2 + U(-.014,.014)*np.sin(2*np.pi*x/max(.18,2*hem))
    body=(yy>=top)&(s<=bwidth)&(yy<=bottom)
    if sleeveless:
        sleeve=np.zeros_like(body); inner=sleeve_top; outer=sleeve_top+.1
    else:
        # Cuffs are oblique when sleeves are short, nearly horizontal when long.
        pitch=(U(.50,1.42) if short else U(-.13,.27))
        cuffline=sleeve_y+pitch*(sleeve_tip-cuff*.55-s)
        last = sleeve_y-pitch*cuff*.45
        armpit = min(last*.77, bust_y*U(.85,1.38))
        arm_start=drop*.85
        elbow=last*.55+arm_start*.45
        outer=ip(yy,[arm_start,elbow,last,max(last+.10,1.18)],
                 [shoulder*.96, .50*(shoulder+sleeve_tip)+U(0,.043),sleeve_tip,sleeve_tip-.01])
        inner=ip(yy,[arm_start,armpit,max(armpit+.025,elbow+.015),max(armpit+.05,sleeve_y)],
                 [shoulder*.65,chest*.88,(chest*.94+sleeve_tip-cuff)*.50+U(-.015,.026),sleeve_tip-cuff])
        sleeve=(yy>=drop*.68)&(yy>=shoulderline*.90)&(s<=outer)&(s>=inner)&(yy<=cuffline)
    mask=body|sleeve
    # Interior reflectance: mild cylindrical lighting of body and sleeves,
    # fabric folds and, on opening garments, the placket and lapels.
    base=U(.35,.87)
    cloth=base*(1.-U(.06,.27)*(s/np.maximum(.07,bwidth))**2)
    cloth=cloth*(1.-U(.035,.17)*bell((s-.83*bwidth)/U(.018,.045)))
    if not sleeveless:
        arm_coord=(s-inner)/np.maximum(.035,outer-inner)
        arm_light=base*(U(.72,1.04)-.26*(2*clip(arm_coord)-1.)**2)
        cloth=np.where(sleeve&(~body | (s>bwidth-.025)),arm_light,cloth)
    if dress and chance(.60):
        cloth*=1.-U(.05,.35)*bell((yy-waist_y)/U(.012,.046))
        cloth+=base*U(.03,.13)*np.cos(2*np.pi*x/U(.06,.16)) * clip((yy-waist_y)*3)
    # The neck binding lies just inside the cutout.
    cloth=cloth*(1.-U(.11,.40)*bell((yy-neck-.008)/U(.012,.034))*(s<collar_w*1.1))
    opened = coat or (shirt and (style in ('button','shortshirt') or chance(.43)))
    if opened:
        z=x-U(-.023,.023)-U(-.035,.035)*(yy-.50)
        throat=U(.12,.43) if coat else U(.08,.24)
        opening=U(.001,.027)+U(.023,.095)*clip(1.-yy/throat)
        if coat and chance(.23): opening+=.02+.055*clip((yy-.44)/.56)
        cloth*=1.-U(.20,.77)*np.exp(-np.minimum(60.,(z/opening)**2))*(yy>neck)
        cloth+=U(-.09,.16)*bell((z-opening-.009)/.017)*(yy>neck)
        if chance(.6):
            pitch=U(.10,.17)
            cloth-=U(.04,.19)*bell((z-.012)/.017)*bell(((yy+.5*pitch)%pitch-.5*pitch)/.015)
        if coat and chance(.44):
            # Angled hip pocket welts.
            cloth-=U(.035,.17)*bell((s-(chest*.75-.37*(yy-.63)))/.022)*bell((yy-.63)/.095)
    if collar:
        depth=U(.11,.23)*(0.8 if shirt else 1.)
        collared=(yy>.02)&(yy<depth)&(s<collar_w+.022)&(s>.018)
        wing=yy<depth*(1-np.abs(s-.64*collar_w)/max(.05,.66*collar_w))
        cloth+=base*U(-.26,.28)*collared*wing
        cloth-=base*.19*bell((yy-depth*(1-np.abs(s-.64*collar_w)/max(.05,.66*collar_w)))/.015)*collared
    if knit and chance(.70):
        cloth*=1.-U(.055,.25)*bell((yy-bottom+.052)/.035)
    return wrap(mask,cloth)

def trousers():
    # Rise, hips and independently articulated legs including flares.
    hip=U(.176,.272); waist=hip*U(.77,1.01)
    rise=U(.225,.42); foot=U(.058,.117)
    style=choice(['straight','slim','straight','wide','flare','taper'])
    legspace=U(.006,.13)
    if style=='wide': foot=U(.092,.138); legspace*=.63
    if style=='slim': foot=U(.046,.077)
    if style=='flare': foot=U(.09,.14)
    footcenter=foot+legspace*.54
    x,y=canvas(2*max(footcenter+foot,hip),1.,'cloth'); y=y+U(-.013,.013)*np.tanh(x/.035)
    x=x-U(-.021,.021)*np.sin(np.pi*y)-U(-.02,.02)*(y-.5)
    s=abs(x); a=ip(y,[0,.15,rise],[waist,hip,hip*.94]); crotch=U(-.022,.022)
    upper=(s<=a)&(y>=0.014*(1-(s/waist)**2))&(y<=rise)
    h=ip(y,[rise,.59,1.],[hip*.485,(foot if style!='flare' else foot*.63)*U(.94,1.22),foot]);
    center=ip(y,[rise,.56,1.],[hip*.46,hip*.48+legspace*.28,footcenter]);
    dev=abs(x-crotch*clip(1-(y-rise)*3))-center
    legs=(abs(dev)<=h)&(y>=rise-.01)&(y<=1.-.012*np.sin(13*x))
    val=U(.40,.86)*(1-.24*(dev/(h+.012))**2*clip((y-rise+.07)*5))
    val-=.09*bell((y-.054)/.016)+.09*bell(x/.023)*bell((y-.16)/.1)
    return wrap(upper|legs,val)

# --------------------- Leather goods / footwear --------------------------
def stroke(x,y,xa,ya,xb,yb,r):
    dx,dy=xb-xa,yb-ya
    q=clip(((x-xa)*dx+(y-ya)*dy)/(dx*dx+dy*dy+1e-9))
    return (x-xa-q*dx)**2+(y-ya-q*dy)**2<=r*r

def shoe(k):
    # s is distance from the toe, z is height above floor, both in foot lengths.
    # A heel, toe spring, vamp and throat are shared by shoe styles.
    sandal=k==5; sneaker=k==7; boot=k==9
    if sandal: style=choice(['straps']*4+['slide']*2+['thong']*2+['wedge']*2+['heeled']*3+['sport']*2+['sling'])
    elif sneaker: style=choice(['runner']*5+['court']*3+['flat']*2+['mid','slipon']*2)
    else: style=choice(['ankle']*5+['chelsea']*3+['high']*2+['heeled']*3+['hiker']*2+['western','platform'])
    heelheight=(U(.11,.33) if (sandal and style in ('heeled','wedge','sling')) or (boot and style in ('heeled','western')) else U(.015,.095))
    rubber=U(.034,.098) if not sandal else U(.019,.066)
    toe=U(.077,.174) if sneaker else U(.075,.16)
    if sandal:
        H=({'slide':U(.16,.31),'thong':U(.12,.28),'straps':U(.25,.58),'sport':U(.22,.46)}.get(style,U(.32,.61)))+heelheight*.50
    elif sneaker:
        H=U(.29,.51) if style not in ('flat','mid') else (U(.23,.36) if style=='flat' else U(.46,.62))
    else:
        H=(U(.83,1.35) if style=='high' else (U(.47,.69) if style=='hiker' else U(.53,.96)))+heelheight*.22
    x,y=canvas(1.,H,'shoe'); s=x+.5 if chance(.89) else .5-x; z=H-y
    # Small 3D rotations change the apparent slope of the floor line.
    z=z-U(-.048,.048)*(s-.5)
    toe_spring=U(.014,.068)*clip((.235-s)/.235)**1.6
    heel_rise=heelheight*clip((s-.35)/.51)**U(.8,1.65)
    if style=='wedge': under=toe_spring+.009*bell((s-.57)/.18)
    else: under=toe_spring+heel_rise
    upper_sole=toe_spring+heel_rise+rubber
    ext=(s>=0)&(s<=1.)
    sole=ext&(z>=under)&(z<=upper_sole)
    heel_start=U(.76,.93) if heelheight>.10 else U(.62,.84)
    heel=(s>=heel_start+.048*clip(1-z/(heelheight+.01)))&(s<=.985+.012*clip(z/.09))&(z>=.006)&(z<=upper_sole)
    base=U(.37,.86); val=z*0+base
    if not sandal:
        shaftfront=U(.47,.69) if boot else U(.45,.60)
        # A boot rises abruptly into a shaft. A sneaker has a tongue and
        # a scooped topline between the tongue and the back counter.
        toe_peak=toe+rubber
        if boot:
            neck=U(.035,.13); back=U(.915,.994); backlean=U(-.085,.068)
            topline=ip(s,[0.,.035,.16,.34,shaftfront-.055,shaftfront,shaftfront+neck,.83,back,1.],
                       [rubber*.65,toe_peak*.74,toe_peak,toe_peak*1.22,H*U(.36,.58),H*U(.53,.83),H*U(.92,1.0),H*U(.92,1.),H,H*.68])
            rear=s<=.998+backlean*clip(z/H)-(1-back)*clip((z/H-.25)/.64)
        else:
            tongue=U(.46,.64); notch=U(.72,.85); back=U(.92,.98)
            collar=H*U(.66,.99); n=H*(U(.43,.69) if style in ('flat','slipon') else U(.60,.85))
            topline=ip(s,[0,.035,.16,.32,tongue-.055,tongue,tongue+.046,notch,back,1.],
                       [rubber*.59,toe_peak*.70,toe_peak,toe_peak*1.23,H*.81,H,H*U(.82,.97),n,collar,H*.50])
            rear=s<=1.-U(.014,.079)*clip((z-rubber)/H)
        upper=ext&rear&(z>=upper_sole-.018)&(z<=topline)&(s>=.040*clip((z-rubber)/toe))
        upperlight=base*(.88+.13*np.cos(4*(z-upper_sole)/max(.2,H)))
        upperlight*=1.-U(.08,.4)*bell((z-topline+.030)/.022)*(s>.47)
        if sneaker or style=='hiker':
            upperlight+=U(-.16,.19)*bell((s-(.32+z*.74))/.09)*np.cos(2*np.pi*(s-.7*z)/.071)
            upperlight+=U(-.19,.17)*bell((z-(.125+.22*(s-.4)))/.035)*bell((s-.66)/.32)
        else:
            upperlight-=U(.025,.18)*bell((s-.77-.075*z)/.024)*clip(z/.27)
        val=np.where(upper,upperlight,base*U(.58,1.29)); mask=sole|heel|upper
        val=np.where(sole & (z<under+.018),val*.68,val)
    else:
        straps=np.zeros_like(ext)
        def strap(a,b,w):
            return stroke(s,z,*a,*b,w)
        toe_base=rubber+.012
        # A forefoot band is curved over the toes. Slingback and sport
        # sandals additionally have a rear band and one or more ankle straps.
        ta=U(.025,.18); tb=U(.30,.48); peak=U(.10,.22)+rubber
        if style in ('slide','thong'): peak=H*U(.80,1.)
        arc=clip((s-ta)/(tb-ta)); bandline=toe_base+(peak-toe_base)*np.sin(np.pi*arc)**.89
        w=(U(.025,.052) if style!='slide' else U(.046,.105))
        if style=='thong':
            straps|=strap((.095,toe_base),(.22,peak),U(.021,.042))|strap((.22,peak),(.57,rubber+.032),U(.022,.045))
        else:
            straps|=(s>=ta)&(s<=tb)&(abs(z-bandline)<=w)
            if chance(.54): straps|=strap((ta, toe_base), (tb-.058,peak*.85),w*U(.72,1.2))
        if style not in ('thong','slide'):
            ankle=(U(.64,.79),H*U(.85,1.)); counter=(U(.91,.98),H*U(.70,.99)); foot=(U(.69,.87),rubber+heelheight*.82)
            aw=U(.021,.057)
            straps|=strap(ankle,counter,aw)|strap(foot,ankle,aw*U(.75,1.36))
            if style not in ('sling','wedge') or chance(.58): straps|=strap((.935,rubber+heelheight),counter,aw*U(.88,1.50))
            if chance(.72): straps|=strap((U(.30,.54),rubber+.025),ankle,aw*U(.8,1.27))
            if style in ('sport','wedge') or chance(.23):
                straps|=(s>.77)&(s<.985)&(z>upper_sole)&(z<upper_sole+.075+.04*(s-.8))
        mask=sole|heel|(straps&ext); val=np.where(straps,base,base*U(.65,1.24))
    return wrap(mask,val)

def bag():
    style=choice(['tote']*4+['satchel']*3+['handbag']*4+['clutch']*2+['backpack']*3+['duffel']*2+['bucket','crossbody','purse'])
    if style=='backpack': H=U(1.04,1.56); drop=U(.025,.19); topw=U(.55,.83)
    elif style in ('clutch','duffel'): H=U(.40,.72); drop=(U(.0,.08) if style=='clutch' else U(.07,.27)); topw=U(.58,.93)
    elif style in ('bucket','tote','crossbody'): H=U(.88,1.64); drop=U(.21,.53); topw=U(.78,1.0)
    else: H=U(.62,1.17); drop=U(.13,.42); topw=U(.62,1.0)
    x,y=canvas(1.,H,'bag'); y=y/H; x=x-U(-.04,.04)*(y-.50)
    s=abs(x); t=(y-drop)/(1-drop); bw=U(.35,.49)
    w=ip(t,[0.,.07,.24,.68,.90,1.],[topw*.5-.035,topw*.5,.48 if style!='backpack' else .455,.5,.49,bw])
    bow=U(-.059,.055)
    body=(s<w)&(t>bow*(1-(s/.5)**2))&(t<1.-U(.012,.089)*(s/.50)**4)
    if style=='backpack': body&=t>.19*(s/.5)**2
    hcx=U(-.085,.085) if style!='crossbody' else U(-.19,.19)
    hw=U(.18,.39) if style!='crossbody' else U(.33,.49)
    bottom=drop+U(.04,.16); thick=U(.017,.049)
    handle=((x-hcx)/hw)**2+((y-bottom)/max(.04,bottom))**2
    loop=(handle<=1.0)&( ((x-hcx)/max(.05,hw-thick))**2+((y-bottom)/max(.018,bottom-thick/H))**2>=1.)&(y<=bottom)
    if chance(.34):
        off=U(.045,.13); handle=((x-hcx-off)/hw)**2+((y-bottom)/max(.04,bottom-.05))**2
        loop|=(handle<1)&(handle>1-2*thick/hw)&(y<bottom)
    if style=='clutch' and chance(.78): loop&=False
    base=U(.35,.84); val=base*(1.-.23*(s/.53)**3)
    if chance(.72): val*=1.-U(.10,.46)*bell((t-.37+(.48*s)**2)/.026)
    if style in ('backpack','duffel'): val*=1.-U(.13,.38)*bell((t-.59+1.2*s*s)/.04)
    return wrap(body|loop,np.where(body,val,base*U(.55,1.04)))

# ----------------------------- Network ----------------------------------
L=[]; W={}

def op(name, typ, inp, **settings):
    L.append(dict(name=name,op=typ,inputs=([inp] if isinstance(inp,str) else inp),**settings)); return name

def dense(name, inp, w, b=0.):
    w=np.asarray(w,np.float32); W[name+'.W']=w
    W[name+'.b']=np.broadcast_to(np.asarray(b,np.float32),(w.shape[0],)).copy()
    return op(name,'dense',inp,out=w.shape[0])

def conv(name, inp, w, b=0., pad=0):
    w=np.asarray(w,np.float32); k=w.shape[-1]; W[name+'.W']=w
    W[name+'.b']=np.broadcast_to(np.asarray(b,np.float32),(w.shape[0],)).copy()
    return op(name,'conv',inp,out=w.shape[0],kernel=[k,k],stride=1,padding=pad)

def pool(name,inp,sz,avg=False):
    return op(name, 'avgpool' if avg else 'maxpool', inp, size=list(sz),stride=list(sz))

def sharp(name, inp, a):
    f=np.array([1.,4.,6.,4.,1.])/16.; k=-a*f[:,None]*f[None,:]; k[2,2]+=1
    return conv(name,inp,k[None,None],pad=2)

def norm(name,inp): return op(name,'norm',inp)
def relu(name,inp): return op(name,'relu',inp)
def reshape(name,inp,sh): return op(name,'reshape',inp,shape=list(sh))

# A concave piecewise-linear transfer recovers presence even for dark cloth.
step=conv('threshold','image',np.ones((3,1,1,1)),[0.,-.14,-.55]); step=relu('positive_part',step)
m=conv('presence',step,np.array([.83/.14+.17/.55,-.83/.14,-.17/.55]).reshape(1,3,1,1))
mp=pool('presence14',m,(2,2),True); gp=pool('light14','image',(2,2),True)
mn=norm('outline_normal',sharp('outline_emphasis',mp,.43)); gn=norm('light_normal',sharp('light_emphasis',gp,.18))

# Groupwise maxima give several independently drawn patterns a voice at
# inference. There is no calibration or group calculation at build time.
groups=4; per=384; N=groups*per
wm=np.empty((10*N,196),np.float32); wg=np.empty_like(wm)
for k in range(10):
    for j in range(N):
        if k in (0,2,3,4,6): a,b=upper(k)
        elif k==1: a,b=trousers()
        elif k==8: a,b=bag()
        else: a,b=shoe(k)
        wm[k*N+j]=.67*a; wg[k*N+j]=.33*b
sm=dense('outline_agreement',mn,wm); sg=dense('light_agreement',gn,wg)
sim=op('joint_agreement','add',[sm,sg]); sim=reshape('pattern_groups',sim,(10,groups,per))
nearest=pool('closest_in_group',sim,(1,per)); mean=pool('several_witnesses',nearest,(groups,1),True); best=pool('best_witness',nearest,(groups,1))
sm=dense('agreement_a',mean,np.eye(10)*.66); sg=dense('agreement_b',best,np.eye(10)*.34)

# A dedicated placket detector, independent of the pattern maker. Compare
# a central vertical band to strips on its sides. A true opening often stays
# at the same horizontal position in both the upper and lower front.
rawnorm=norm('exposure_normal','image')
yy,xx=np.mgrid[:28,:28]+.5
probes=[]
for center in [12.5,13.5,14.,14.5,15.5]:
    for slant in [-.055,0.,.055]:
        xx0=xx-center-slant*(yy-14.)
        for radius,gap in [(.65,1.6),(1.1,2.6),(1.65,3.5)]:
            for test in ['line','step']:
                for sign in [-1.,1.]:
                    for lo,hi in [(6,16),(14,24)]:
                        def strip(off,sd):
                            z=bell((xx0-off)/sd)*((yy>=lo)&(yy<hi)); return z/z.sum()
                        left=strip(-gap,radius); right=strip(gap,radius)
                        a=(strip(0,radius)-.5*(left+right)) if test=='line' else .68*(right-left)
                        probes.append((-sign*a).ravel())
p=dense('front_contrast',rawnorm,probes); npat=len(probes)//2
p=reshape('same_front',p,(1,npat,2)); p=pool('weaker_front_half_negative',p,(1,2))
p=conv('weaker_front_half',p,-np.ones((1,1,1,1))); p=pool('aligned_front_contrast',p,(npat,1))
p=dense('front_hinges',p,[[1.],[1.]],[-.12,-.83]); p=relu('front_hinges_pos',p)
# Saturation prevents dark background between trouser legs etc. from
# generating a disproportionately large correction.
coef=np.array([-.078,0.,-.128,0.,.106,0.,.084,0.,0.,0.]); ref=.18
front=dense('opening_evidence',p,np.stack([coef,-coef],axis=1),-ref*coef)

# Detect fine texture in the chest panel: edge responses are formed before
# spatial pooling. Blouses/shirts tend to be less uniformly knit than a
# plain sweater; use only as a small tie breaker.
diffs=np.zeros((4,1,3,3)); diffs[0,0,1,:]=[-.5,1.,-.5]; diffs[2,0,:,1]=[-.5,1.,-.5]; diffs[1]=-diffs[0]; diffs[3]=-diffs[2]
tx=relu('fabric_edges_abs',conv('fabric_edges',rawnorm,diffs,pad=1)); region=bell((xx-14)/5.5)*bell((yy-14)/6)
wt=np.broadcast_to(region/(2*region.sum()),(4,28,28)).reshape(1,-1)
tx=dense('fabric_detail',tx,wt); tx=relu('detail_excess',dense('detail_limits',tx,[[1.],[1.]],[-.12,-.46]))
coef=np.array([-.023,0.,-.035,0.,.006,0.,.086,0.,0.,0.]); texture=dense('texture_evidence',tx,np.stack([coef,-coef],axis=1),-.045*coef)

# Mirror asymmetry is weak evidence against front-facing clothes and
# stronger evidence for a side-view shoe. Compare coarse blocks to avoid
# overreacting to little folds or small misalignment of the object.
coarse=pool('broad_regions',mp,(2,2),True); sym=[]
for r in range(7):
    for c in range(3):
        a=np.zeros((7,7)); a[r,c]=1.; a[r,6-c]=-1.; sym.extend([a.ravel(),-a.ravel()])
asym=relu('unshared_presence',dense('reflected_difference',coarse,sym)); asym=dense('lopsided',asym,np.ones((1,len(sym)))/21.)
asym=relu('large_asymmetry',dense('asymmetry_hinges',asym,[[1.],[1.]],[-.13,-.53])); coef=np.array([-.11,-.07,-.12,-.11,-.11,.08,-.09,.11,.02,.12]); mirror=dense('view_evidence',asym,np.stack([coef,-coef],axis=1))

# An open-space detector asks if a transparent point has structure on all
# four sides. Compare the lower/middle and upper placement of enclosed
# space: straps vs. solid footwear, and handles vs. a solid garment.
# The tests use a local background probe with rays of different lengths.
def hole_probe():
    rays=[]; loc=[]
    for row in range(2,13):
        for col in range(1,13):
            loc.append((row,col))
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                for first,last in [(1,2),(2,4),(4,7)]:
                    a=np.zeros((14,14)); n=0
                    for d in range(first,last+1):
                        r=row+d*dr; c=col+d*dc
                        if 0<=r<14 and 0<=c<14: a[r,c]+=1; n+=1
                    a/=max(1,n); a[row,col]-=1.
                    rays.append(a.ravel())
    h=dense('surround_rays',mp,rays); h=reshape('ray_lengths',h,(len(loc),4,3)); h=pool('ray_support',h,(1,3))
    # Flip sign using a dense diagonal on the 4-direction axis via a
    # reshape and shared scalar 1x1 convolution.
    h=reshape('ray_strip',h,(1,len(loc),4)); h=conv('negative_support',h,-np.ones((1,1,1,1))); h=pool('all_directions',h,(1,4)); h=conv('opening_depth',h,-np.ones((1,1,1,1)),-.24); h=relu('true_open_space',h)
    # Area in lower 2/3 vs upper half; no weights from rendered shapes.
    r=np.array([a for a,b in loc]); c=np.array([b for a,b in loc]); a=((r>=5)&(r<=11)).astype(float); b=((r<=6)&(c>=3)&(c<=10)).astype(float)
    return dense('open_areas',h,[a/a.sum(),b/b.sum()])
areas=hole_probe(); hc=np.zeros((10,2)); hc[5,0]=.48; hc[7,0]=-.25; hc[9,0]=-.17; hc[8,1]=.33
holes=dense('opening_layout',areas,hc,-hc@np.array([.025,.025]))
# Modest priors for class overlap: 'shirt' is the least definite outline,
# while a T-shirt is usually both short-sleeved and unbuttoned.
bias=np.array([.006,.004,.009,-.003,.0,.002,-.008,.004,.008,.001]); W['agreement_a.b']+=bias.astype(np.float32)
op('class_scores','add',[sm,sg,front,texture,mirror,holes])

out=sys.argv[1]; os.makedirs(out,exist_ok=True)
with open(os.path.join(out,'network.json'),'w') as f: json.dump({'layers':L},f,indent=1)
np.savez(os.path.join(out,'weights.npz'),**W)
