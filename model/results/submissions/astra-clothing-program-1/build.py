"""Construct idealized silhouettes from a clothing pattern grammar, without any data files."""
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(81621)
u = rng.uniform
q = 42
grid = (np.arange(q) + .5) / q

def render(parts, flip=False):
    # Each part is a closed polygon; rasterize its horizontal scan-line spans.
    lo = np.min(np.concatenate([p for add,p in parts if add]),axis=0)
    hi = np.max(np.concatenate([p for add,p in parts if add]),axis=0)
    sc = .98 / max(hi-lo)
    cx,cy=(hi+lo)*.5
    dx,dy=u(-.019,.019),u(-.019,.019)
    skew=u(-.045,.045)
    img=np.zeros((q,q),bool)
    for add,pts in parts:
        pts=np.asarray(pts)
        x=(pts[:,0]-cx)*sc+.5+dx + skew*(pts[:,1]-cy)*sc
        y=(pts[:,1]-cy)*sc+.5+dy
        ox,oy=np.roll(x,1),np.roll(y,1)
        d=y-oy
        cut=(grid[:,None]-oy)/np.where(d==0,1,d)
        xx=np.sort(np.where((grid[:,None]>=np.minimum(y,oy))&(grid[:,None]<np.maximum(y,oy)),ox+(x-ox)*cut,3),axis=1)
        mask=np.zeros((q,q),bool)
        for j in range(0,len(x)-1,2):
            mask|=(grid[None,:]>=xx[:,j,None])&(grid[None,:]<xx[:,j+1,None])
        if add: img |= mask
        else: img &= ~mask
    if flip: img=img[:,::-1]
    return img.astype('float32')

def poly(parts,pts,add=True): parts.append((add,np.array(pts)))

def oval(parts,x,y,w,h,add=True):
    ang=np.arange(24)*np.pi/12
    poly(parts,np.stack((x+w*np.cos(ang),y+h*np.sin(ang)),axis=1),add)

def stroke(parts,coords,width):
    pts=np.array(coords); vec=np.roll(pts,-1,axis=0)-np.roll(pts,1,axis=0)
    vec[0]=pts[1]-pts[0];vec[-1]=pts[-1]-pts[-2]
    n=vec[:,::-1]*[-1,1];n*=width*.5/np.maximum(.00001,(n*n).sum(1,keepdims=True)**.5)
    poly(parts,np.concatenate((pts+n,(pts-n)[::-1])))

def arch(parts,xa,xb,bottom,top,thick,slant=0):
    angle=np.linspace(np.pi,0,16)
    stroke(parts,np.stack(((xa+xb)/2+(xb-xa)/2*np.cos(angle)+slant*np.sin(angle),bottom-(bottom-top)*np.sin(angle)),1),thick)

def top(family):
    # All dimensions are relative to neck-to-hem length. Each half is made
    # from the neck, shoulder seam, sleeve end, side seam, and bottom hem.
    parts=[]; sleeve='long'; raised=0.; flared=False
    chest=u(.25,.35); waist=chest*u(.84,1.05); hem=chest*u(.93,1.12)
    shoulder=chest*u(.79,1.03); shoulderY=u(.06,.15)
    armY=u(.28,.41); neck=shoulder*u(.34,.65); dip=u(.015,.10)
    hemcurve=u(-.035,.055); sleevelen=u(.85,1.11); spread=u(.06,.17); cuff=u(.082,.126)
    if family in ('longtee','fitted','tunic'):
        if family=='tunic':
            chest=u(.21,.31);shoulder=chest*u(.80,.97);waist=chest*u(.75,.95);hem=chest*u(1.02,1.42);hemcurve=u(.02,.09)
            sleevelen=u(.55,.94);cuff=u(.07,.13);spread=u(.045,.11);dip=u(.04,.18)
        else:
            if family=='fitted':
                chest=u(.16,.24);hem=chest*u(.92,1.23);waist=chest*u(.76,.95);shoulder=chest*u(.70,.97)
                sleeve='none' if rng.random()<.55 else 'short';sleevelen=u(.14,.32);armY=u(.15,.28);cuff=u(.055,.10);spread=u(.046,.099)
            else:
                sleeve='long';waist=chest*u(.80,.98);hem=chest*u(.92,1.08);sleevelen=u(.56,.88);dip=u(.07,.21)
    elif family in ('tee','blouseshort','looseT'):
        if family=='looseT':
            chest=u(.32,.43);waist=chest*u(.9,1.05);hem=chest*u(.92,1.07);shoulder=chest*u(.83,1.01);hemcurve=u(-.025,.042)
        sleeve='short';sleevelen=u(.23,.40); dip=u(.03,.14)
        spread=u(.09,.19); cuff=u(.095,.155)
        if family=='blouseshort':
            sleevelen=u(.24,.56);waist=chest*u(.79,1.13);hemcurve=u(.01,.10);dip=u(.05,.24)
    elif family in ('tank','strappy'):
        sleeve='none';neck=shoulder*u(.45,.70);dip=u(.12,.30);armY=u(.20,.36)
        shoulder=chest*u(.60,.91);shoulderY=u(.005,.045)
        if family=='strappy': neck=shoulder*u(.80,.97);dip=u(.18,.34)
    elif family=='sweater':
        dip=u(0,.065);shoulderY=u(.04,.11);hem=chest*u(.86,1.03);waist=chest*u(.90,1.00)
    elif family=='sweaterV':
        dip=u(.08,.18);neck=shoulder*u(.26,.57)
    elif family=='boxy':
        chest=u(.34,.47);waist=chest*u(.95,1.09);hem=chest*u(.89,1.03);shoulder=chest*u(.88,1.05)
        spread=u(.055,.14);sleevelen=u(.65,.99);cuff=u(.095,.15)
    elif family=='muscle':
        sleeve='none';shoulder=chest*u(.61,.92);neck=shoulder*u(.52,.82);dip=u(.05,.21);armY=u(.22,.37);hem=chest*u(.90,1.10)
    elif family=='swingcoat':
        raised=u(.035,.11);neck=shoulder*u(.32,.55);waist=chest*u(.80,.95);hem=chest*u(1.12,1.38);sleevelen=u(.67,.93);spread=u(.03,.12)
    elif family in ('jacket','coat','shirt','blouse','cardi'):
        if family in ('coat','jacket','shirt'):
            raised=u(.025,.12);dip=u(0,.07);neck=shoulder*u(.25,.48)
        if family=='shirt':
            hemcurve=u(.025,.104);waist=chest*u(.82,.99);sleevelen=u(.82,1.02);cuff=u(.066,.113)
        if family=='coat':
            chest=u(.21,.30);shoulder=chest*u(.85,1.04);waist=chest*u(.9,1.10);hem=chest*u(1.02,1.32)
            sleevelen=u(.65,.94);spread=u(.036,.108);cuff=u(.072,.115)
        if family=='jacket':
            hem=chest*u(.96,1.13);waist=chest*u(.95,1.06);sleevelen=u(.88,1.09)
            cuff=u(.094,.143);spread=u(.06,.145)
        if family=='blouse':
            hemcurve=u(.015,.11);waist=chest*u(.75,1.08);hem=chest*u(1,1.25)
            sleeve='short' if rng.random()<.45 else 'long';sleevelen=u(.40,.82);cuff=u(.11,.20);dip=u(.05,.24)
        if family=='cardi':
            hem=chest*u(.97,1.17);raised=u(.015,.09);dip=u(.04,.20)
    elif family.startswith('dress'):
        chest=u(.13,.25);shoulder=chest*u(.80,1.06);neck=shoulder*u(.35,.73);dip=u(.025,.115)
        waist=chest*u(.68,.98);hem=chest*u(1.05,1.86);shoulderY=u(.025,.095);armY=u(.16,.27)
        sleeve=family[5:]; flared=True; hemcurve=u(-.025,.066)
        if sleeve=='straight':
            chest=u(.175,.30);waist=chest*u(.86,1.06);hem=chest*u(.90,1.35);sleeve=rng.choice(['short','none','long']);shoulder=chest*u(.84,1.04)
        if sleeve=='long':sleevelen=u(.48,.79);spread=u(.04,.10);cuff=u(.054,.09)
        if sleeve=='short':sleevelen=u(.15,.29);spread=u(.045,.105);cuff=u(.07,.115)
        if sleeve=='none': shoulder=chest*u(.65,.93);shoulderY=u(0,.035)
    waisty=u(.36,.53) if flared else u(.51,.73)
    def half(sign):
        vary=u(.94,1.06); sy=sleevelen*u(.96,1.04); sp=spread*u(.87,1.13)
        a=[(0,dip),(.65*neck,.72*dip),(neck,0)]
        if raised:a.extend([(neck,-raised),(neck*1.13,-raised*.88),(neck*1.30,.012)])
        a.extend([((shoulder+neck)*.50,shoulderY*.36),(shoulder,shoulderY)])
        if sleeve=='short':
            out=chest+sp
            a.extend([(out+.02,sy-.047),(out-cuff,sy+.019),(chest,armY)])
        elif sleeve=='long':
            out=max(hem,chest)+sp
            a.extend([((out+shoulder)*.5+.02,(sy+shoulderY)*.48),(out,sy),(out-cuff,sy+.018),
                      ((out-cuff+chest)*.5-.012,(armY+sy)*.50),(chest,armY)])
        else:a.extend([((shoulder+chest)*.5-.026,armY*.71),(chest,armY)])
        a.extend([(waist,waisty),((hem+waist)*.5,(waisty+1)*.52),(hem,1-hemcurve),(.65*hem,1-.2*hemcurve),(0,1)])
        return [(sign*x*vary,y) for x,y in a]
    a,b=half(1),half(-1);poly(parts,a+b[::-1]);return render(parts)

def trousers(wide):
    parts=[]; waist=u(.125,.195); hip=waist*u(1.01,1.17); split=u(.25,.53); gap=u(.006,.081)
    if wide==2:
        hip=u(.16,.27);waist=hip*u(.76,.99);gap=u(.0,.045);split=u(.34,.68)
    outer=hip*u(.68,1.16) if not wide else hip*u(.95,1.40)
    hemwidth=(outer-gap)*u(.78,1.11);lean=u(-.065,.065)
    a=[(-waist,0),(0,.01),(waist,0),(hip,.19),((hip+outer)/2+lean,.57),(outer,1),(outer-hemwidth,.997),(.024+lean*.45,.61),(0,split),(-.016+lean*.25,.61),(-outer+hemwidth*.90,.99),(-outer,.996),(-(hip+outer)/2+lean,.57),(-hip,.19)]
    poly(parts,a);return render(parts)

def bag(style):
    parts=[];w=1.;h=u(.49,.92) if style!='tote' else u(.83,1.34)
    if style=='clutch':h=u(.45,.85)
    topw=u(.71,1.04);bottomw=u(.73,1.01);rad=u(.045,.17)
    if style=='pack':h=u(.99,1.44);topw=u(.47,.76)
    body=[(-topw/2,0),(-topw*.33,-rad*.29),(topw*.28,-rad*.29),(topw/2,0),(.49,h*.28),(.5,h*.65),(bottomw/2,h-rad),(bottomw/2-rad,h),(-bottomw/2+rad,h),(-bottomw/2,h-rad),(-.5,h*.68),(-.495,h*.26)]
    poly(parts,body)
    if style in ('handle','tote','strap','pack'):
        ha=u(.07,.35) if style=='handle' else u(.38,.87)
        if style=='pack':ha=u(.025,.12)
        span=u(.15,.36) if style!='strap' else u(.39,.49)
        arch(parts,-span,span,.065,-ha,u(.025,.085),u(-.10,.10))
    if style=='strap' and rng.random()<.4:
        stroke(parts,[(-.44,.09),(-.58,h*.69),(-.50,h*.94)],.04)
    if style=='satchel':
        arch(parts,-.49,.49,h*.42,u(-.06,.12),.063,u(-.07,.07))
    if style=='shoulderbag':
        # Worn over one shoulder: tall, non-centered, curved or slack sling.
        rise=u(.68,1.50);xf=u(-.18,.23);stroke(parts,[(-topw*.44,.09),(-.37,-.36*rise),(xf-.11,-.90*rise),(xf,-rise),(xf+.10,-.93*rise),(.32,-.33*rise),(topw*.43,.08)],u(.027,.074))
    return render(parts, rng.random()<.5)

def shoe(family):
    parts=[]; height=u(.27,.43);thick=u(.055,.12);toe=u(.095,.19);tip=u(.03,.087);rise=u(.01,.09)
    if family in ('trainer','highsneaker'):
        tongue=u(.38,.63);valley=u(.04,.16);back=height*u(.74,1.04)
        if family=='trainer': height=u(.24,.50);back=height*u(.73,1.04);toe=min(toe,height*.48)
        if family=='highsneaker': height=u(.40,.57);tongue=u(.53,.73);back=height*u(.79,.99);valley=u(.04,.09)
        prof=[(0,tip),(.035,toe),(.12,toe+.022),(tongue-.13,height*.72),(tongue,height),(tongue+.04,height*.98),(.72,height-valley),(.83,height-valley*1.06),(.94,back),(.975,back-.035),(1,.055)]
        poly(parts,prof+[(.989,.018),(.72,0),(.28,.01),(.06,.029)])
    elif family=='sportstrap':
        slope=u(0,.10);thick=u(.045,.115);high=u(.18,.43)
        poly(parts,[(0,.06),(.04,.095),(.19,thick),(.61,thick+slope*.45),(.91,thick+slope),(1,thick+slope*.64),(.98,.018),(.80,0),(.22,.005),(.04,.023)])
        for shift in (u(.12,.28),u(.45,.60)):
            poly(parts,[(shift-.12,thick*.83),(shift-.027,high*.67),(shift+.055,high*.73),(shift+.16,thick+slope*.52),(shift+.085,thick*.9)])
        stroke(parts,[(.53,high*.60),(.77,high),(.96,high*.86),(.94,thick+slope*.70)],u(.032,.083))
    elif family in ('boot','bootheel','hiker'):
        height=u(.48,.97);back=height*u(.91,1.08);instep=u(.47,.62);shaft=u(.40,.62)
        if family=='hiker':height=u(.38,.62);back=height*u(.87,1.03);shaft=u(.37,.55)
        heelrise=u(.16,.43) if family=='bootheel' else u(.012,.094)
        if family=='bootheel':height+=.12;back+=.12
        heelwidth=u(.044,.20) if family=='bootheel' else u(.18,.31)
        poly(parts,[(0,tip),(.026,toe),(.10,toe+.02),(.24,toe+.048),(.38,toe+heelrise*.42+.066),(instep,height*.58),(shaft,height),(.66,height+.016),(.84,back*.97),(.95,back),(.97,back*.63),(.94,heelrise+.13),(1,heelrise+.074),(.98,.024),(.98-heelwidth,.019),(.94-heelwidth,heelrise+.048),(.66,heelrise+.015),(.49,heelrise*.51),(.31,.027),(.14,.013),(.035,.028)])
    else:
        rise= u(.19,.56) if family in ('heel','wedge') else u(0,.109)
        thick=u(.025,.084);ball=u(.26,.41);heelw=u(.028,.16)
        poly(parts,[(0,thick+.045),(.17,thick+.015),(ball,thick),(.70,rise+thick),(.96,rise+thick+.014),(1,rise+.018),(.91,rise-.004),(.71,rise*.95),(.40,.025),(.25,0),(.035,.018)])
        if family=='wedge':poly(parts,[(.28,.035),(.86,rise+.033),(.98,rise+.012),(.965,0),(.35,0)])
        if family=='heel':poly(parts,[(.80,rise+.025),(.988,rise+.051),(.964,.006),(.964-heelw*.50,.006),(.886-heelw*.52,rise*.63)])
        sh=u(.105,.26); st=u(.046,.142); at=u(.13,.28)
        poly(parts,[(at-.078,thick),(at-.035,sh*.92),(at+.011,sh),(at+st,sh*.82),(at+st+.077,thick+.008)])
        if family=='slide' and rng.random()<.6:
            poly(parts,[(at+.08,thick+.005),(at+.10,sh+.055),(at+.24,sh+.068),(at+.34,thick+.02)])
        if family=='mule':
            poly(parts,[(.08,.035),(.17,sh*.92),(.29,sh+.023),(.40,sh+.017),(.49,.05),(.40,.035)])
        if family not in ('slide','mule'):
            ankle=u(.21,.42)+rise;anchor=u(.60,.79)
            if family in ('heel','wedge'):ankle+=u(.045,.26)
            stroke(parts,[(anchor-.10,rise*.9+thick),(anchor+.065,ankle),(anchor+.23,ankle-.029),(.93,rise+.039)],u(.026,.078))
            stroke(parts,[(.28,sh*.76),(anchor+.055,ankle*u(.78,1.00))],u(.028,.086))
            if rng.random()<.43:stroke(parts,[(.43,thick+.04),(.57,sh+.07),(.79,rise+thick)],u(.046,.108))
    # Profile heights are above the sole; image rows increase downward.
    return render([(add,p*[1.,-1.]) for add,p in parts],rng.random()<.07)

# Marginalize over families rather than rewarding a class simply for having
# more distinct named styles. Family sampling bounds are design assumptions.
spec=[(top,'tee',540),(top,'tank',300),(top,'strappy',180),(top,'boxy',240),
      (top,'sweater',700),(top,'sweaterV',320),(top,'jacket',720),(top,'coat',700),
      (top,'shirt',650),(top,'blouse',520),(top,'blouseshort',360),(top,'cardi',320),
      (top,'dressshort',620),(top,'dressnone',620),(top,'dresslong',600),
      (trousers,False,460),(trousers,True,200),
      (bag,'handle',380),(bag,'tote',380),(bag,'strap',400),(bag,'pack',240),(bag,'clutch',340),(bag,'satchel',240),
      (shoe,'trainer',700),(shoe,'highsneaker',320),(shoe,'boot',750),(shoe,'bootheel',620),(shoe,'hiker',400),
      (shoe,'sandal',600),(shoe,'heel',700),(shoe,'wedge',440),(shoe,'slide',460),
      (top,'longtee',500),(top,'fitted',640),(top,'tunic',620),(trousers,2,540),(shoe,'mule',360),
      (top,'looseT',720),(top,'dressstraight',1200),(shoe,'sportstrap',700),
      (top,'muscle',500),(top,'swingcoat',650),(bag,'shoulderbag',550)]

# One descriptor implementation for both the ideal and the observed shapes.
descriptor='''def describe(mask):
    h=len(mask);f=h//14
    a=mask.reshape(14,f,14,f).mean(axis=(1,3));p=np.pad(a,((1,1),(1,1)))
    sm=(4*a+2*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])+p[:-2,:-2]+p[:-2,2:]+p[2:,:-2]+p[2:,2:])/16
    v=[(.55*a+.45*sm).ravel()]
    for z in (mask,mask.T):
        for reverse in (False,True):
            # First-hit positions, pooled over strips, describe a contour
            # even when much of the interior is dark, as on an open sandal.
            w=(z[:,::-1] if reverse else z)>.46
            # Give each empty scanline a neutral midpoint before pooling:
            # one stray pixel should not define the edge of a whole strip.
            lim=.50;loc=np.where(w.any(1),np.minimum(lim,(np.argmax(w,axis=1)+.5)/h),lim)
            v.append(.90*loc.reshape(14,f).mean(1))
    return np.concatenate(v)
'''
exec(descriptor)
centers=[]; ends=[]
for fn,name,n in spec:
    for j in range(n):
        a=fn(name)
        centers.append(describe(a))
    ends.append(len(centers))

source='''import numpy as np

def ramp(a,lo,hi):
    return np.clip((a-lo)/(hi-lo),0.,1.)

def blur(a):
    p=np.pad(a,((1,1),(1,1)),mode="edge")
    return (4*a+2*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])+p[:-2,:-2]+p[:-2,2:]+p[2:,:-2]+p[2:,2:])/16

def foreground(im):
    ref=np.maximum(.035,np.sort(im.ravel())[704])
    raw=im/ref
    local=blur(raw)
    # Interpolate coverage through the antialiased edge but do not grow a
    # light garment past the boundary onto a perfectly black background.
    m=.79*ramp(raw,.021,.19)+.21*ramp(local,.032,.17)
    m=.82*m+.18*np.maximum(m,blur(m))
    return m, np.minimum(raw,2.)

def framed(m):
    z=np.arange(28)
    def bounds(a):
        ok=(a.max(1)>.45)&(a.sum(1)>.85)
        lo=np.argmax(ok);hi=27-np.argmax(ok[::-1]);return lo,hi
    y0,y1=bounds(m);x0,x1=bounds(m.T)
    sc=max(8,y1-y0+1,x1-x0+1)/27.5
    x=(z-13.5)*sc+(x0+x1)/2;y=(z-13.5)*sc+(y0+y1)/2
    a=np.floor(x).astype(int);b=np.floor(y).astype(int);vx=x-a;vy=y-b
    p=np.pad(m,((2,2),(2,2)));a=np.clip(a+2,0,30);b=np.clip(b+2,0,30)
    return ((1-vy[:,None])*((1-vx)*p[b[:,None],a]+vx*p[b[:,None],a+1])+vy[:,None]*((1-vx)*p[b[:,None]+1,a]+vx*p[b[:,None]+1,a+1]))

'''+descriptor+'''
def shapes(x):
    m=CONSTANTS["models"];n=len(x);end=CONSTANTS["ends"]
    vals=np.empty((n,len(end)),np.float64)
    # Fixed padded block size makes the matrix multiplication independent of N.
    for k in range(0,n,48):
        b=x[k:k+48];data=np.zeros((48,m.shape[0]),np.float32)
        for j,im in enumerate(b):
            data[j]=describe(framed(im))
        dst=(data@ m)*(-2.)+CONSTANTS["norm"]+(data*data).sum(1)[:,None]
        start=0
        for i,stop in enumerate(end):
            v=dst[:len(b),start:stop]
            # Soft minimum approximated with a uniform mixture of nearest styles.
            small=np.partition(v,7,axis=1)[:,:8];values=-.58*(.7*small.mean(1)+.3*small.min(1))
            vals[k:k+len(b),i]=values
            start=stop
    return vals

def mix(s,weights):
    z=max(s);return z+np.log(sum(w*np.exp(v-z) for w,v in zip(weights,s)))

def detail(im,m,b):
    # Estimates of structure use per-image lightness. A button band can be
    # dark on light fabric or light on dark fabric, and is often only one pixel wide.
    smooth=blur(im)
    xx=np.arange(28);yy=xx[:,None]
    wm=m.sum(1)+.001
    center=(m*xx).sum(1)/wm
    axis=np.clip(np.median(center[7:22]),10.,17.)
    # Contrast of a straight placket/zip against the neighboring material.
    texture=np.abs(im-smooth)
    patch=(yy>=7)&(yy<24)&(np.abs(xx-axis)<5.5)&(m>.8)
    grain=(texture*patch).sum()/(patch.sum()+1)
    broad=np.abs(im[:,2:]-im[:,:-2]); rough=broad[7:24,8:18].mean()
    a=im[7:24,7:21];l=(im[7:24,5:19]+im[7:24,6:20])*.5;r=(im[7:24,8:22]+im[7:24,9:23])*.5
    ridge=a-(l+r)*.5;step=l-r
    evidence=np.maximum(abs(ridge.mean(0)),.68*abs(step.mean(0)))
    persist=np.maximum(np.sort(np.abs(ridge),axis=0)[5],.63*np.sort(np.abs(step),axis=0)[5])
    val=(.70*evidence+.30*persist)/(.12+1.5*grain+.32*rough)
    dist=np.abs(np.arange(7,21)-axis);seam=max(0.,np.max(np.where(dist<2.8,val-.08*dist,0)));off=np.max(np.where(dist>=2.8,val,0))
    zipline=ramp(seam-.23*off,.20,1.12)
    # Two diagonal lapels, folds or collar edges falling toward the middle.
    edge=np.abs(im[:,2:]-im[:,:-2]);lapels=0.
    for depth in (4,7,11):
        for spread in (2.7,4.6):
            z=np.arange(depth);xs=spread*(1-z/(depth+1))
            a=np.clip(np.rint(axis-xs).astype(int),1,26);d=np.clip(np.rint(axis+xs).astype(int),1,26)
            left=edge[z+2,a-1].mean();right=edge[z+2,d-1].mean()
            lapels=max(lapels,min(left,right))
    vee=ramp(lapels-.36*rough,.09,.39)
    # Relative shoulder height and the start of continuous cloth across the neck.
    ytop=np.argmax(m>.45,axis=0);ytop=np.where(m.max(0)>.45,ytop,28)
    c=int(np.rint(axis));atneck=np.median(ytop[c-1:c+2]);side=min(np.median(ytop[c-6:c-3]),np.median(ytop[c+4:c+7]))
    collar=ramp(side-atneck,1,4)
    scoop=ramp(atneck-side,.2,4.)
    # Patterned cloth and chest graphics. Compare high-frequency content at
    # similar brightness and look for prints localized to the center chest.
    body=im[7:22,c-4:c+5];tx=texture[7:22,c-4:c+5]
    ink=tx.mean()/(.16+body.mean())
    upper=texture[6:14,c-4:c+5].mean();lower=texture[16:24,c-4:c+5].mean();graphic=ramp(upper-1.5*lower,.018,.12)
    patterned=ramp(ink,.09,.27)
    # Separate central garment hem from sleeve tips: long cuffs below a
    # waistband suggest knitwear; a dipped shirt tail suggests a woven shirt.
    bottom=27-np.argmax((m>.49)[::-1],axis=0)
    hem=np.median(bottom[c-2:c+3]); quarter=np.median(bottom[[c-4,c-3,c+3,c+4]])
    roundhem=ramp(hem-quarter,.40,2.8)
    # Sleeve/body air gaps, estimated independently of cloth brightness.
    gaps=0.
    for row in (15,18,21,24):
        r=m[row];left=np.maximum.accumulate(r);right=np.maximum.accumulate(r[::-1])[::-1]
        inside=np.minimum(left,right)-r
        gaps+=.125*(max(inside[:c])+max(inside[c:]))
    def sm(a,w):return mix([b[i] for i in a],w)
    t=sm([0,1,2,32,37,40],[.42,.13,.055,.055,.26,.08]); p=sm([15,16,35],[.70,.16,.14]);d=sm([12,13,14,33,38],[.21,.23,.16,.12,.28])
    s=sm([4,5,3,11],[.71,.12,.10,.07]);coat=sm([6,7,11,41],[.53,.28,.06,.13]);shirt=sm([8,9,10,0,1,4,34],[.31,.20,.15,.105,.045,.07,.12]);
    sand=sm([28,29,30,31,36,39],[.23,.22,.12,.17,.09,.17]);trainer=sm([23,24],[.85,.15]);boot=sm([25,26,27],[.47,.31,.22]);bag=sm([17,18,19,20,21,22,42],[.20,.13,.16,.09,.21,.11,.10])
    s+=.68*gaps+1.05*graphic+.42*scoop-2.45*zipline-1.05*collar-.77*vee-.48*patterned
    coat+=2.1*zipline+1.23*collar+1.1*vee-.62*gaps-.58*graphic-.32*scoop-.44*roundhem-.46
    shirt+=1.43*patterned+.58*roundhem+.60*vee+.35*zipline-.53*graphic-.32
    t+=.50*graphic-.69*zipline-.54*patterned-.45*vee
    # Width profile through the trunk: a skirt may flare below its waist,
    # whereas the separated legs of trousers form a slit deep in the center.
    rmask=m>.4
    widths=28-np.argmax(rmask,axis=1)-np.argmax(rmask[:,::-1],axis=1)
    widths=np.where(rmask.any(1),widths,0)
    topw=np.median(widths[4:10]); ww=np.median(widths[11:16]);hw=np.median(widths[21:25])
    flare=ramp(hw-ww,.5,5.0);slim=ramp(19-topw,0,8.)
    d+=1.3*flare+.95*slim-.55*zipline-.6*gaps
    t-=.75*flare+.57*slim
    inseam=m[16:26,c-2:c+3].min(1);thigh=(m[16:26,c-5:c-1].max(1)+m[16:26,c+2:c+6].max(1))/2
    fork=ramp((thigh-inseam).mean(),.34,.83)
    waistband=ramp((m[1:4,c-3:c+4].mean()-m[1:4,c-6:c-3].mean()),-.2,.45)
    p+=3.5*fork+.48*waistband-1.2*ramp(topw-ww,1,6.)
    s-=1.9*fork;coat-=1.9*fork;d-=2.2*fork
    # Plaid, all-over prints and lace put horizontal and vertical detail
    # throughout a shirt, unlike a sweatshirt graphic limited to the chest.
    lowtx=texture[14:24,c-5:c+6].mean()/(.16+im[14:24,c-5:c+6].mean())
    lowerprint=ramp(lowtx,.095,.25)
    shirt+=1.15*lowerprint+1.05*patterned+.52*roundhem+.60*ramp(hw-topw,-2,4.)
    s-=.75*lowerprint;coat-=.62*lowerprint
    # Estimate sleeve length independently of the template: use the lowest
    # fabric near each outer side, relative to the lowest central fabric.
    colok=(m>.49).sum(0)>=2;edgeL=np.argmax(colok);edgeR=27-np.argmax(colok[::-1]);radius=(edgeR-edgeL+1)*.5
    leftq=edgeL+np.array([.05,.11,.17])*(c-edgeL);rightq=edgeR+np.array([.05,.11,.17])*(c-edgeR)
    cuff=(np.median(bottom[np.clip(np.rint(leftq).astype(int),0,27)])+np.median(bottom[np.clip(np.rint(rightq).astype(int),0,27)]))*.5
    sleevedrop=hem-cuff
    wings=ramp(sleevedrop,9,17)*ramp(topw-ww,1.,5.)
    t+=2.6*wings*(1-.45*roundhem)-1.15*roundhem
    d-=1.65*wings*ramp(topw,19,25)
    shirt-=.85*wings*(1-.70*patterned)
    drop=ramp(sleevedrop,1.,6.5); longarm=ramp(cuff-hem,-2.,1.5)
    s+=.85*longarm*(1-.65*zipline)-1.10*drop-.75*roundhem
    shirt+=.66*drop+.66*ramp(sleevedrop,2,7)*ramp(15-sleevedrop,0,7)
    # Flat knit, collared outerwear, and printed/bloused shirts can share a
    # silhouette. A closely fitted rib-knit waistband runs horizontally;
    # a collared jacket has a smaller neck platform and converging panels.
    neckpeak=ramp((np.median(ytop[c-8:c-4])+np.median(ytop[c+5:c+9]))/2-np.mean(ytop[c-2:c+3]),1.6,5.)
    bandedge=np.abs(im[1:]-im[:-1]);br=np.clip(int(hem)-5,6,22)
    band=np.max(np.abs((im[br+1:br+5,c-4:c+5]-im[br:br+4,c-4:c+5]).mean(1)))
    elastic=ramp(band-rough*.33,.045,.23)*longarm*(1.-zipline)
    s+=.60*elastic+.36*(1.-neckpeak)*(1.-vee)
    coat+=.85*neckpeak+.45*vee*zipline-.5*elastic
    # A waistband is wide and level at the top, unlike a pair of shoulders.
    wy=np.median(ytop[c-3:c+4]);flat=np.mean(np.abs(ytop[c-4:c+5]-wy)<1.5)
    p+=1.7*ramp(flat,.65,1)*ramp(21-topw,0,7)-1.15*ramp(wy,2,6)
    # Is the bottom two separated cuffs, or one contiguous skirt hem?
    # Allow the split to curve off center but require real cloth on both sides.
    leg=m[19:26];middle=leg[:,c-3:c+4];gapcol=np.argmin(middle,axis=1)+c-3
    leftleg=np.max(np.where(xx[None,:]<gapcol[:,None]-1,leg,0),axis=1)
    rightleg=np.max(np.where(xx[None,:]>gapcol[:,None]+1,leg,0),axis=1)
    divided=np.mean(np.minimum(leftleg,rightleg)-np.min(middle,1))
    # Photometric crease for legs which touch: a dark line surrounded on
    # both sides by appreciably brighter fabric, absent near the waistband.
    crease=0.
    for shift in (-2,0,2):
        pos=c+shift; dark=im[16:26,pos-1:pos+2].min(1);sur=np.minimum(im[16:26,pos-4:pos-1].max(1),im[16:26,pos+2:pos+5].max(1))
        crease=max(crease,np.mean((sur-dark)/(.15+sur)))
    legproof=max(ramp(divided,.21,.76),.7*ramp(crease,.28,.74))
    rowtop=np.argmax(widths>3);lip=widths[rowtop:min(rowtop+3,28)].mean()/(np.mean(widths[6:11])+.5)
    jeans=ramp(lip,.60,.94)*ramp(flat,.62,.99)
    p+=3.2*legproof-5.5*(1-legproof)*(1-.70*jeans)-2.9*flare-1.5*neckpeak
    d+=1.55*flare*(1-legproof)+.90*ramp(21-topw,0,6)*(1-legproof)
    # Sleeve-free tops and standard short-sleeved tees need neither buttons
    # nor a shirt collar. A plain crew neck and a horizontal hem distinguish
    # them from printed woven blouses and dip-hem shirts.
    plain=ramp(.15-ink,.015,.10)*(1-.65*zipline)
    tankshape=ramp(max(b[1],b[2])-max(b[0],b[4],b[8]),0,3)
    t+=1.25*wings+.95*plain*wings+1.25*tankshape-.55*roundhem*patterned
    shirt-=.65*plain*wings+.7*tankshape*(1-patterned)
    # Check for a sustained front opening even when one panel is off center:
    # jackets often show an open V at the top leading to a narrow lower slit.
    zz=im[12:24,c-4:c+5];sideavg=(im[12:24,c-6:c-3].mean(1)+im[12:24,c+4:c+7].mean(1))*.5
    oneline=np.max(np.abs((zz-sideavg[:,None]).mean(0)))/(.19+rough*1.5+np.std(zz))
    opening=ramp(oneline,.39,1.48)*ramp(np.mean(m[9:21,c-4:c+5]),.72,.98)
    # Shirt fabrics may be patterned in both directions (checks, florals);
    # a horizontal-only stripe is just as plausible on a pullover or tee.
    dx=np.abs(body[:,1:]-body[:,:-1]).mean();dy=np.abs(body[1:]-body[:-1]).mean()
    twoway=ramp(dx/(.01+dy),.42,1.00);stripe=1-twoway
    coat+=1.05*opening+.48*zipline+.72*neckpeak*ramp(3.5-sleevedrop,0,4)
    s-=1.20*opening+.4*vee+.64*neckpeak
    shirt+=.64*twoway*lowerprint-.80*stripe*patterned
    t+=.80*stripe*patterned;s+=.70*stripe*patterned
    # Strong horizontal shoulder line above a handle aperture is evidence
    # for luggage even if the body itself is rather tall and coat-shaped.
    ylo=np.argmax(m.max(1)>.46);yhi=27-np.argmax(m.max(1)[::-1]>.46);hei=max(1,yhi-ylo+1)
    h1=min(yhi,ylo+int(.32*hei));h2=min(yhi,ylo+int(.60*hei))
    upperbody=m[h1:h2+1];handle=(upperbody[:,:c-2].max(1)+upperbody[:,c+3:].max(1))*.5-upperbody[:,c-2:c+3].mean(1)
    handleloop=ramp(handle.mean(),.22,.83)*ramp(m[h2:min(yhi+1,h2+6),c-3:c+4].mean(),.25,.80) if h2<=yhi else 0.
    broadbody=ramp(hw,21,27)*ramp(7-abs(hw-ww),0,6)
    bag+=3.8*handleloop+1.45*broadbody*(1-wings)*ramp(1-gaps,.44,.95)
    # The less material within the shoe outline, the more likely a strapped
    # sandal. Boots usually have a steep vamp and a high rear shaft.
    cols=(m>.48).any(0);ht=28-ytop;bot=bottom
    span=np.maximum(0,bot-ytop+1)*cols
    holes=(span.sum()-m.sum())/(span.sum()+1)
    # A column hull misses the air below a slingback or between widely
    # spaced straps. Examine the interior of an elastic envelope over the
    # upper and the sole separately from the concavity under a high heel.
    foot=framed(m);on=foot>.33
    topedge=np.where(on.any(0),np.argmax(on,0),27);base=np.where(on.any(0),27-np.argmax(on[::-1],0),0)
    if topedge[3:9].mean()<topedge[-9:-3].mean():
        foot=foot[:,::-1];topedge=topedge[::-1];base=base[::-1]
    roof=topedge.astype(np.float64);floor=base.astype(np.float64)
    for step in range(3,24):
        xp=np.arange(28-step);a=roof[xp];z=roof[xp+step]
        for frac in (.25,.5,.75):
            j=int(np.rint(frac*step));v=frac*z+(1-frac)*a
            roof[xp+j]=np.minimum(roof[xp+j],v)
    interior=(yy>=roof+.4)&(yy<=floor-.3)&(xx>0)&(xx<27)
    air=((1-foot)*interior).sum()/(interior.sum()+.01)
    openstraps=ramp(air,.085,.34)
    # Deep, repeated holes between a strap and the sole carry more weight
    # than a single depression at the mouth of a closed shoe.
    columnspace=np.maximum(0,base-topedge+1)-foot.sum(0)
    through=np.sort(columnspace[2:26])[-7:-2].mean();window=ramp(through,.85,3.3)
    above=interior&(yy<base-1);darkspace=((foot<.27)&above).sum()/(above.sum()+1)
    openstraps*=.47+.53*max(window,ramp(darkspace,.10,.30))
    # Slides are almost all sole in the rear, or else have two detached
    # islands on it; a closed sneaker maintains a solid heel counter.
    filled=foot.sum(0);toeheight=filled[2:12].mean();heelheight=filled[19:26].mean()
    backless=ramp(toeheight-heelheight,0,3.0)*ramp(5-heelheight,0,3)
    thinsole=ramp(9.0-(base.max()-topedge.min()+1),0,5.0)
    sand+=8.0*openstraps+5.5*backless+1.6*ramp(holes,.12,.39)+2.6*window+3.3*thinsole
    trainer-=4.5*openstraps+3*backless;boot-=4.8*openstraps+3.5*backless
    # A sneaker's tongue rises in front of the scooped opening; a boot's
    # ankle tube is more nearly level across its top, or taller at the back.
    tongue=np.median(np.sort(topedge[10:19])[:3]);notch=np.max(topedge[18:24]);rear=np.median(topedge[22:26])
    tonguecue=ramp(notch-tongue,.8,3.5)*ramp(notch-rear,.35,2.)
    shortshoe=ramp(15-(base.max()-topedge.min()),0,5.)
    trainer+=1.5*tonguecue+1.20*shortshoe;boot-=1.2*tonguecue+.75*shortshoe
    # Final decisions within type: short sleeves produce a sharp drop in
    # row span; a skirt flares progressively below the waist instead. The
    # neckline of a sleeveless top usually has two relatively high straps.
    shoulderwidth=np.sort(widths[3:12])[-3:].mean();trunk=np.median(widths[14:19]);hemwidth=np.median(widths[21:25])
    overhang=ramp(shoulderwidth-max(trunk,hemwidth*.93),2.,7.)*ramp(sleevedrop,6.,15.)
    teeproportion=ramp(shoulderwidth,18.,25.)
    shortT=overhang*(.42+.58*teeproportion)
    tankneck=max(scoop,ramp(np.mean(ytop[c-2:c+3])-min(np.median(ytop[c-5:c-2]),np.median(ytop[c+3:c+6])),.4,4))
    tankcue=ramp(max(b[1],b[2],b[40])-max(b[4],b[6],b[8],b[14]),1,6)*tankneck
    narrowskirt=ramp(20.-shoulderwidth,0,7)*ramp(3.0-abs(hemwidth-trunk),-.5,3)
    t+=1.9*shortT+1.4*tankcue+.9*ramp(b[0]-max(b[8],b[10]),0,3.)
    d-=2.1*shortT+.8*tankcue+1.25*zipline+.75*neckpeak+.85*gaps+.75*ramp(shoulderwidth,19,25)
    shirt-=1.2*shortT*(1-.53*patterned)
    # The garment prefix shares a strong prior in ambiguous upper-body
    # silhouettes. A printed front on a long sleeve tee may be knitwear,
    # but texture covering sleeves as well as the torso supports a blouse.
    armzone=(np.abs(xx-axis)>.31*(edgeR-edgeL))&(m>.86)&(yy>=6)&(yy<=19)
    armtexture=(texture*armzone).sum()/((im*armzone).sum()+.19*armzone.sum()+.1)
    allprint=lowerprint*ramp(armtexture,.085,.22)*twoway
    shirt+=1.30*allprint+1.1*roundhem*(1-zipline)+.70*drop*ramp(hw,14,21)-.7*tankcue
    s-=.7*allprint+.52*roundhem;coat-=.85*allprint
    # Analyze broad collar/hood shading as a ratio of upper and lower
    # centered contrasts. This avoids mistaking full-height plaid for lapels.
    sidevar=im[3:10,c-6:c+7].std();trunkvar=im[12:21,c-5:c+6].std()
    throat=ramp(sidevar-.60*trunkvar,.09,.27)*vee
    coat+=.84*throat+scoop*(-.35)-.65*roundhem*(1-zipline)
    # Side profiles with a flat, low sole are more likely to be sneakers
    # when their toes and heel counters are solid. A boot shaft has a
    # nearly vertical front above the vamp. Search the actual upper outline.
    tallr=max(0.,np.mean(base[19:25]-topedge[19:25]));lowfoot=ramp(13.-tallr,0.,5.)
    topdiff=topedge[8:21]-topedge[10:23]
    shaft=ramp(np.max(topdiff),3.0,7.)*ramp(tallr,9.,16.)
    enclosed=ramp(air,.035,.15)*ramp(through,.8,2.6)
    heelcut=ramp(base[7:12].mean()-np.min(base[15:23]),2.0,6.)
    trainer+=1.45*lowfoot*(1-openstraps)*(1-shaft)-1.75*heelcut-1.3*shaft
    boot+=1.35*shaft+1.35*heelcut-.78*lowfoot
    sand+=1.55*enclosed+2.15*thinsole*(1.-tonguecue)-1.0*tonguecue*(1-window)
    # Apply fabric rules within the clothes group. Their sometimes large
    # bonuses are not evidence that a rectangular piece of luggage is cloth.
    garmentbase=max(b[i] for i in (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,33,34,35,37,38,40,41))
    gracescore=max(t,p,s,d,coat,shirt)-garmentbase
    bag+=.62*max(gracescore,0.)
    bag+=1.4*handleloop+1.2*broadbody*(1-overhang)*(1.-gaps)
    return np.argmax(np.array([t,p,s,d,coat,sand,shirt,trainer,bag,boot]))

def predict(x):
    out=np.empty(len(x),np.int64)
    for k in range(0,len(x),240):
        ims=x[k:k+240,0]; masks=np.empty_like(ims);norm=np.empty_like(ims)
        for j in range(len(ims)):
            masks[j],norm[j]=foreground(ims[j])
        vals=shapes(masks)
        for j in range(len(ims)):
            out[k+j]=detail(norm[j],masks[j],vals[j])
    return out
'''
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
mat=np.array(centers,dtype=np.float32)
np.savez(out/'constants.npz',models=np.ascontiguousarray(mat.T),norm=(mat*mat).sum(axis=1),ends=np.array(ends))
(out/'solution.py').write_text(source)
