"""Compile a paper-pattern and line-detector classifier. No data inputs."""
import sys, json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(281019)
u = rng.uniform
pick = rng.choice
layers=[]
weights={}
def layer(name,op,ins,**kw):
    layers.append(dict(name=name,op=op,inputs=ins if isinstance(ins,list) else [ins],**kw)); return name
def conv(name,src,W,b,p=0):
    W=np.array(W,np.float32); b=np.array(b,np.float32)
    weights[name+'.W']=W; weights[name+'.b']=b
    return layer(name,'conv',src,out=len(W),kernel=list(W.shape[-2:]),padding=p,stride=1)
def dense(name,src,W,b):
    W=np.ascontiguousarray(W,dtype=np.float32); weights[name+'.W']=W; weights[name+'.b']=np.broadcast_to(np.array(b,np.float32),(len(W),)).copy()
    return layer(name,'dense',src,out=len(W))
def pool(name,src,op,h,w,sy,sx):return layer(name,op,src,size=[h,w],stride=[sy,sx])

# Coordinates for drawing flat articles, independent of the input raster.
grid=np.linspace(-.5,1.5,80,endpoint=False)+1/80
X=grid[None,:]-.5; Y=grid[:,None]
def profile(t,at,values): return np.interp(t,at,values)
def strip(x,y,xa,ya,xb,yb,r):
    dx=xb-xa; dy=yb-ya
    pr=np.clip(((x-xa)*dx+(y-ya)*dy)/(dx*dx+dy*dy+1.e-9),0,1)
    return (x-xa-pr*dx)**2+(y-ya-pr*dy)**2<r*r
def tube(x,y,xs,ys,r):
    a=False
    for i in range(len(xs)-1):a=a|strip(x,y,xs[i],ys[i],xs[i+1],ys[i+1],r)
    return a

def upper(c):
    # Body widths are half-widths; vertical unit = back length. The sleeve
    # is constructed between separate outer-arm and inner-arm seams.
    k=u(); style=''; pen=0
    if c==0:
        if k<.82:style='tee'
        elif k<.92:style='tank'
        else:style='longtee';pen=.013
    elif c==2:
        style='sweater' if k<.9 else 'tunic'; pen=.002 if style=='tunic' else 0
    elif c==4:
        style=pick(['jacket','overcoat','cardigan'],p=[.54,.28,.18])
    elif c==6:
        style=pick(['button','blouse','summer','shortbutton','tank','tee'],p=[.38,.23,.11,.12,.06,.10]);pen={'tee':.020,'tank':.013,'shortbutton':.005}.get(style,0)
    else:
        style=pick(['frock','sheath','longdress','gown','shift','skirt'],p=[.32,.22,.18,.13,.10,.05]);pen=.008 if style=='shift' else 0
    # neck radius, shoulder slope and proportions typical of an upright garment
    chest=u(.255,.36); shoulder=chest*u(.86,1.06); waist=chest*u(.89,1.02); hem=chest*u(.89,1.08)
    collar=u(.075,.16); slope=u(.055,.145); arm=u(.22,.37)
    cuff_y=u(.88,1.06); spread=u(.055,.16); cuff_w=u(.062,.12); curve=u(-.025,.035); neckcut=u(0,.055)
    sleeve=True; tank=False; short=False; band=False; cuff_tilt=u(-.035,.035); middle=.56
    if style in ('tee','shortbutton','summer'):
        short=True; cuff_y=u(.24,.44);spread=u(.11,.225);cuff_w=u(.105,.19);cuff_tilt=u(.055,.13)
        if style=='tee':hem=chest*u(.89,1.01);curve=u(-.012,.036)
        if style=='shortbutton':curve=u(.04,.125)
        if style=='summer':hem=chest*u(.91,1.23);curve=u(0,.11); cuff_y=u(.18,.52)
    if style=='tank':
        tank=True;sleeve=False;shoulder=chest*u(.56,.94);neckcut=u(.07,.22);slope=u(0,.045);curve=u(0,.055);arm=u(.19,.34);hem=chest*u(.89,1.20)
    if style=='longtee':
        cuff_y=u(.52,.89);spread=u(.045,.14);cuff_w=u(.048,.085);hem=chest*u(.88,1.07)
    if style in ('sweater','tunic'):
        cuff_y=u(.91,1.13);spread=u(.07,.165); cuff_w=u(.055,.105);slope=u(.045,.14);band=True;hem=chest*u(.84,1.01)
        if style=='tunic':hem=chest*u(1.03,1.27);cuff_y=u(.66,.96);curve=u(.015,.092)
    if style in ('button','blouse'):
        cuff_y=u(.78,1.02);cuff_w=u(.052,.105);curve=u(.032,.14);hem=chest*u(.90,1.09)
        if style=='blouse':cuff_y=u(.50,.99);spread=u(.06,.20);cuff_w=u(.07,.145);hem=chest*u(.92,1.24);slope=u(.05,.125);collar=u(.10,.205)
        else: collar=u(.06,.114);slope=u(.095,.175)
    if style in ('jacket','overcoat','cardigan'):
        collar=u(.06,.135);slope=u(.105,.205); neckcut=u(0,.047);cuff_w=u(.072,.128)
        if style=='jacket':chest=u(.255,.37);waist=chest*u(.90,1.01);hem=chest*u(.87,1.06);shoulder=chest*u(.87,1.01);cuff_y=u(.89,1.10);band=True
        if style=='overcoat':chest=u(.183,.285);waist=chest*u(.82,1.04);hem=chest*u(1.02,1.34);shoulder=chest*u(.85,1.04);cuff_y=u(.62,.88);spread=u(.044,.124);cuff_w=u(.048,.086)
        if style=='cardigan':hem=chest*u(.96,1.24);cuff_y=u(.72,.99);curve=u(-.015,.10)
    if c==3:
        chest=u(.13,.235); shoulder=chest*u(.80,1.02);waist=chest*u(.66,.94);hem=chest*u(1.15,1.92);collar=u(.035,.10);arm=u(.14,.255);slope=u(.025,.09);middle=u(.31,.47);curve=u(-.025,.055);neckcut=u(.018,.095);cuff_w=u(.042,.095)
        if style in ('sheath','gown'):tank=True;sleeve=False;shoulder*=u(.67,1.03);hem=chest*u(.92,1.45) if style=='sheath' else chest*u(1.3,2.4);slope=u(0,.035)
        elif style=='shift':waist=chest*u(.93,1.05);hem=chest*u(1.01,1.38)
        elif style=='skirt':return skirt(),.007
        if style=='longdress':cuff_y=u(.42,.73);spread=u(.04,.12)
        else: short=True;cuff_y=u(.13,.27);spread=u(.045,.15);cuff_tilt=u(.023,.065)
    y=Y; x=np.abs(X - u(-.023,.023)*(y-.45)); pos=X>0
    x=x/np.where(pos,u(.95,1.05),1)
    if c in (2,4,6):
        # Outer layers can be hung from a hanger with the arms touching the
        # torso, or laid flat with both sleeves well separated. A standing
        # coat collar rises above the shoulder and a blouse can be collarless.
        spread*=u(.58,1.55)
        if c==4 and u()<.64:
            slope+=u(.014,.085); collar*=u(.63,.94)
        if c==6 and style in ('blouse','summer'):
            chest*=u(.89,1.14); spread*=u(.65,1.35)
    collar=min(collar,shoulder*.78)
    bodyside=profile(y,[0,slope,arm,middle,.82,.93,1.05],[collar,shoulder,chest,waist,(waist+hem)*.5,hem*(1.05 if band else 1),hem])
    # Depth of the back neck and surface of the shoulder.
    top=np.where(x<collar,neckcut*np.maximum(0,1-(x/collar)**2),slope*(x-collar)/(shoulder-collar+1.e-5))
    if tank:
        bodyside=profile(y,[0,.08,arm,middle,.8,1],[shoulder,shoulder*.97,chest,waist,(waist+hem)*.5,hem]);top=np.where(x<=collar,neckcut*np.maximum(0,1-(x/collar)**2),slope)
    hip=1-curve*np.minimum(1.2,(x/hem)**2)
    if c==3 and style=='sheath':bodyside+=.035*np.maximum(0,1-np.abs(y-.64)/.19)
    torso=(x<bodyside)&(y>=np.minimum(top,slope+.025))&(y<=hip)
    if sleeve:
        outward=chest+spread; end=cuff_y+np.where(pos,u(-.032,.032),0)
        # eliminate the parts of the rectangle beyond the slant of the cuff
        inner=np.maximum(chest*.65,profile(y,[slope,arm,cuff_y],[shoulder*.8,chest*.90,outward-cuff_w]))
        outer=profile(y,[slope-.015,(slope+cuff_y)*.55,cuff_y],[shoulder,outward-(.27 if not short else .48)*spread,outward])
        arm_top=slope+np.maximum(x-shoulder,0)*(.65 if short else 1.55)
        cuff_cut=end-cuff_tilt*(x-(outward-cuff_w))/cuff_w
        torso |= (y>=arm_top)&(x>=inner)&(x<=outer)&(y<=cuff_cut)
    return torso,pen

def skirt():
    y=Y; x=np.abs(X); a=u(.18,.39); b=a*u(1.04,2.05)
    return (y>=0)&(y<=1-.03*(x/b)**2)&(x<=a+(b-a)*np.clip(y,0,1)**.8)

def pants():
    y=Y;x=X;top=u(.15,.27);hip=top*u(.98,1.18);crotch=u(.25,.49);shift=u(-.044,.044); spread=u(.00,.14); ankle=u(.034,.095); knee=ankle*u(1.05,1.43)
    hwidth=profile(y,[0,.14,crotch,.63,1],[top,hip,hip*.97,hip*.85,ankle*2+spread*.5]); body=(y>=.013*(x/top)**2)&(y<=crotch)&(abs(x)<hwidth)
    for sign in (-1,1):
        foot=u(.93,1.02); cen=profile(y,[.14,crotch,.63,foot],[hip*.48,hip*.52,hip*.48+spread*.45,ankle+spread]);rad=profile(y,[.14,crotch,.63,foot],[hip*.57,hip*.54,knee,ankle]);body|=(y>=.14)&(y<foot)&(abs(x-sign*cen-shift*np.maximum(0,y-.2))<=rad)
    return body

def footwear(c):
    # x runs from tip of toe to back of heel, y is measured upwards from floor.
    x=X+.5;y=.9-Y; heel=u(.0,.08); toe=u(.06,.13);pen=0.; h=0
    if c==7:
        high=u()<.16;height=u(.27,.48) if not high else u(.44,.62); pen=.005 if high else 0
        floor=profile(x,[0,.12,.3,.65,.86,1],[toe*.68,.015,0,0,.012,heel*.6+.015]); tongue=u(.44,.64); notch=u(.68,.81); roof=profile(x,[0,.07,.23,tongue-.12,tongue,notch,.93,1],[toe,toe*1.52,toe*1.7,height*.8,height,height*u(.54,.84),height*u(.76,1),height*.68]); mask=(y>=floor)&(y<=roof)&(x>.015*np.maximum(0,1-y/.10)**2)&(x<=1-.034*np.maximum(0,y/.35))
    elif c==9:
        h=u(.38,.94); deep=u(.17,.44);heel=u(.025,.33); neck=u(.47,.67);back=u(.87,1.0); foot=u(.14,.25);rim=neck+u(-.08,.10);heelwid=u(.06,.20)
        floor=profile(x,[0,.13,.32,.48,.77,.97,1],[toe*.68,.015,0,.025,heel,heel,heel+.04]); roof=profile(x,[0,.06,.19,neck-.12,neck,neck+.06,.99,1],[toe,foot*.75,foot*.88,foot*1.35,foot*1.7,foot*1.8,foot+heel,heel+.10]); mask=(x>=0)&(x<1)&(y>=floor)&(y<=roof)
        front=profile(y,[foot,foot*1.6,h],[neck-.11,neck,rim]); rear=profile(y,[heel,heel+.12,h],[1,.97,back]);mask|=(y>=heel+.06)&(y<=h+.04*(x-rim))&(x>=front)&(x<=rear)
        mask|=(x<=.99)&(x>=.99-heelwid-.035*(y/(heel+.03)))&(y>=.002)&(y<=heel+.06)
        if u()<.12:pen=.003
    else:
        tall=u()<.49;heel=u(.16,.47) if tall else u(.005,.105);height=heel+u(.16,.39) if tall else u(.16,.38);t=u(.035,.105);tip=u(.015,.075); heelw=u(.034,.18);heelx=u(.88,.98)
        floor=profile(x,[0,.13,.32,.48,.79,.94,1],[tip,.012,0,.028,heel,heel,heel+.015]);mask=(x>=0)&(x<=1)&(y>=floor)&(y<=floor+t); mask|=(x<=heelx+.025)&(x>=heelx-heelw-.035*y)&(y>0)&(y<=heel+.035)
        wedge=u()<.23
        if wedge:mask|=(y>=.01)&(y<floor+t)&(x>=.4)&(x<.97)
        ball=u(.13,.31); toeheight=u(.10,.19); width=u(.026,.069);mask|=tube(x,y,[ball-.12,ball,ball+.115],[t*.6,toeheight+t*.5,t*.7],width)
        if u()<.47:mask|=(x>ball-.09)&(x<ball+.11)&(y<toeheight)&(y>t*.65)
        if u()<.80 or tall:
            instep=u(.43,.67);end=u(.83,.97);r=u(.024,.067)
            mask |=tube(x,y,[instep-.15,instep+.02,end],[.06+t*.5,height*.86,height],r)
            if u()<.85:mask|=tube(x,y,[end,end-.02,heelx],[height,height*.73,heel+t],u(.024,.077))
            if u()<.46:mask|=strip(x,y,instep+.09,height*.84,instep+.21,.10+heel*.5,u(.019,.052))
        elif u()<.68:mask|=tube(x,y,[.27,.51,.67],[.075,height,.14],u(.045,.108))
        # Slide / mule: enclosed front vamp but a bare heel on a flat
        # footbed. Not every sandal has a loop around the ankle.
        if not tall and u()<.39:
            lift=u(.005,.065);thick=u(.036,.10); ball=u(.25,.48); ht=u(.12,.31);end=u(.48,.70)
            base=profile(x,[0,.15,.44,.80,1],[.04,0,0,lift,lift+.01]);mask=(y>=base)&(y<base+thick)
            vamp=profile(x,[.04,.16,ball,end],[.08,ht*.67,ht,.055]);mask|=(x>=.04)&(x<end)&(y>=base+thick*.65)&(y<vamp)
            if u()<.48:mask&=~((x>ball-.12)&(x<ball-.055)&(y>base+thick)&(y<vamp-.024))
    # Each construction lives in the support interval of the foot: the
    # formula for the sneaker topline alone would extrapolate past the heel.
    mask &= (x>=-.004)&(x<=1.004)
    if u()<.055:mask=mask[:,::-1];pen+=.012
    return mask,pen

def handbag():
    # Body of a shopper/satchel/clutch, plus one or two loop handles.
    x=X;y=Y;variant=pick(['shopper','satchel','purse','clutch','pack'],p=[.29,.24,.24,.13,.10]);hw=.46
    h=u(.43,.85); ht=u(.12,.51); taper=u(-.16,.17);roundness=u(.02,.15);pen=0
    if variant=='clutch':h=u(.36,.72);ht=u(0,.12);roundness=u(.04,.20)
    if variant=='pack':h=u(.94,1.34);ht=u(.02,.17);roundness=u(.13,.27);pen=.004
    if variant=='purse':h=u(.38,.75);ht=u(.18,.63);roundness=u(.10,.26)
    # Rounded rectangle with a configurable widening/narrowing gusset.
    z=(y-.04)/h; dx=x/(hw*(1+taper*np.clip(z,0,1))); r=roundness
    mask=(np.maximum(np.abs(dx)-(1-r),0)**2/r**2+np.maximum(np.abs(z-.5)-(.5-r),0)**2/r**2 <=1)
    w=u(.16,.34);top=.04-ht;left=u(-.067,.067); rad=u(.012,.042)
    if ht>.035:
        # generalised ellipse: round, arch or squared-off handles
        p=u(1.35,3.8); ax=abs((x-left*(1-np.clip((y-top)/ht,0,1)))/w); ay=abs((y-.072)/(ht+.032)); d=(ax**p+ay**p)**(1/p)
        mask|=(abs(d-1)<rad/((w+ht)*.5))&(y<=.1)
    return mask,pen

# Normalize the paper patterns to catalogue framing, then integrate the area
# inside each of the 14-by-14 silhouette cells by regular supersampling.
fy,fx=np.mgrid[:56,:56]; fx=(fx+.5)/56-.5; fy=(fy+.5)/56-.5
def frame(a,code):
    iy,ix=np.nonzero(a);x0,x1=ix.min(),ix.max();y0,y1=iy.min(),iy.max();w=x1-x0+1;h=y1-y0+1
    scale=u(.912,1.00)/max(w,h);sw=scale*u(.965,1.04);sh=scale
    ox=u(-.038,.038); oy=u(-.025,.025)
    if code in (5,7,8,9):
        # Centre placements as well as sole/bottom aligned catalogue layouts.
        oy+=u(-.11,.87)*max(0,(1-h*sh)/2)
        ox+=u(-.24,.24)*max(0,(1-w*sw)/2)
    tilt=u(-.028,.028) if code!=1 else u(-.04,.04)
    px=np.rint((fx-ox-tilt*fy)/sw+(x0+x1)/2).astype(int).clip(0,len(grid)-1);py=np.rint((fy-oy)/sh+(y0+y1)/2).astype(int).clip(0,len(grid)-1)
    # The detector sees less than unit occupancy on dark cloth. Include a
    # nuisance reflectance parameter in the shape hypotheses rather than
    # confusing a dark wide garment with a bright narrow one.
    reflectance=pick([.65,.81,.93,.99],p=[.07,.11,.26,.56])
    return a[py,px].reshape(14,4,14,4).mean((1,3))*reflectance

# Continuous foreground occupancy: a mixture of clipped ramps avoids cutting
# holes in dark fabric, without treating a one-grey-level halo as a garment.
lo=np.array([.008,.032,.095]); hi=np.array([.068,.14,.29]);amt=np.array([.56,.31,.13]);
a=conv('intensity_ramps','image',np.ones((6,1,1,1)),-np.r_[lo,hi]);a=layer('positive_ramps','relu',a)
a=conv('foreground',a,np.r_[amt/(hi-lo),-amt/(hi-lo)].reshape(1,6,1,1),[0]);a=pool('cell_coverage',a,'avgpool',2,2,2,2)
dil=pool('neighbour_coverage',a,'maxpool',2,2,1,1)

# Inner products plus the squared norm bias implement quadratic mismatch up
# to a term which is the same for *all* hypotheses. Include marginal widths,
# marginal heights and coarse quadrant widths for tolerance to local drape.
def linear_metric(t):
    H,W=t.shape; n=t.size
    w=.55*t/n; b=-.55*np.mean(t*t)
    for axis,coef in [(0,.14),(1,.18)]:
        z=t.mean(axis,keepdims=True);w+=coef*z/n;b-=coef*np.mean(z*z)
    for axis in [0,1]:
        z=t.copy()
        if axis: z=z.T
        for p,q in [(0,4),(4,10),(10,14)]:z[p:q]=z[p:q].mean(0)
        if axis:z=z.T
        coef=.09;w+=coef*z/n;b-=coef*np.mean(z*z)
    g=np.maximum.reduce([t[:-1,:-1],t[1:,:-1],t[:-1,1:],t[1:,1:]])
    return (2*w).ravel(),(.42*g/g.size).ravel(),b-.21*np.mean(g*g)

def templates():
    N=3072;K=10;A=np.empty((K*N,196),np.float32);B=np.empty((K*N,169),np.float32);bb=np.empty(K*N,np.float32)
    for c in range(K):
        for j in range(N):
            if c in (0,2,3,4,6):mask,pen=upper(c)
            elif c==1:mask,pen=pants(),0
            elif c==8:mask,pen=handbag()
            else:mask,pen=footwear(c)
            w,v,b=linear_metric(frame(mask,c));i=c*N+j;A[i]=w;B[i]=v;bb[i]=b-pen
    l=dense('pattern_overlap',a,A,bb);r=dense('outer_contour_overlap',dil,B,0)
    s=layer('shape_agreement','add',[l,r]);s=layer('style_families','reshape',s,shape=[10,4,N//4]);s=pool('fitting_styles',s,'maxpool',1,N//4,1,N//4);s=pool('consensus',s,'avgpool',4,1,4,1);s=layer('outline_scores','reshape',s,shape=[10]);return s

shape_scores=templates()

# Local appearance evidence uses a gain-normalised image. The fixed line
# detectors measure coherent contrast, rather than darkness per se.
norm=layer('contrast','norm','image');yy,xx=np.mgrid[:28,:28]; xx=xx+.5; yy=yy+.5
patterns=[];families=[]
def featurebank(name,func):
    start=len(patterns)
    for w in func:
        w=np.array(w,dtype=float);patterns.extend((w,-w))
    families.append((name,start,len(patterns)))
def avg(m):return m/max(m.sum(),1.e-10)
def ridge(x,mid,w,span,r):
    dx=np.abs(x-mid); m0=np.exp(-.5*(dx/w)**2)*r; m1=(np.exp(-.5*((dx-span)/w)**2))*r
    return avg(m0)-avg(m1)

def front_lines(centres,widths,extent):
    for cx in centres:
        for width in widths:
            for tilt in [-.045,.045]:
                yield ridge(xx,cx+(yy-14)*tilt,width,width*3.4,(yy>=extent[0])&(yy<=extent[1]))
featurebank('placket',front_lines([12.8,13.6,14.4,15.2],[.56,.9],(7,23)))
featurebank('opening',front_lines([13.2,14,14.8],[1.45,2.05],(8,23)))
featurebank('other_verticals',front_lines([9.6,18.4],[.65,.95],(7,22)))

def collars():
    for cx in [13.25,14.75]:
        for depth in [5,8,11]:
            for width in [3.0,4.5]:
                z=depth+1;dx=np.abs(xx-cx);yc=z-dx*depth/width
                yield ridge(yy,yc,.78,2,(dx<width)&(yc>=2))
featurebank('lapels',collars())

def horizontal(rows,widths):
    for cy in rows:
        for w in widths:yield ridge(yy,cy,w,w*3.2,abs(xx-14)<6.0)
featurebank('ribbing',horizontal([20,21.5,23,24.5],[.65,1.1]))
featurebank('stripes',horizontal([9,11,13,15,17,19],[.66,1.1]))

def plaid(vertical):
    coord=xx if vertical else yy
    for period in [3,4.5,6.5,8.5]:
        for phase in [0,np.pi*.5]:
            v=np.cos(2*np.pi*coord/period+phase); roi=(abs(xx-14)<5.5)&(yy>=8)&(yy<=20)
            yield avg(np.maximum(v,0)*roi)-avg(np.maximum(-v,0)*roi)
featurebank('vertical_checks',plaid(True));featurebank('horizontal_checks',plaid(False))
# Chest pictures vary in both position and overall polarity.
featurebank('chest_emblem',(ridge(xx,cx,1.65,4.45,(yy>=cy-2)&(yy<=cy+2)) for cx in [12.5,14,15.5] for cy in [7,9,11]))

# Upper and lower sections need not follow a single straight line: closure
# edges can bend or end in the chest. Include step edges for lopsided plackets.
featurebank('upper_front',front_lines([12.8,13.6,14.4,15.2],[.6,1.15],(5,14)))
featurebank('lower_front',front_lines([12.8,13.6,14.4,15.2],[.6,1.15],(14,23)))
featurebank('front_step',(avg((xx<cx)&(xx>=cx-w)&(yy>=7)&(yy<22))-avg((xx>=cx)&(xx<cx+w)&(yy>=7)&(yy<22)) for cx in [12.7,13.7,14.7,15.7] for w in [1.6,2.7]))
# Waistband of a pullover is a fabric band, not black background below an
# arbitrary hem. Shoulder and collar samples from the silhouette below help
# decide whether the band belongs to a sweater or a jacket.
featurebank('shirt_tail',(avg((abs(xx-14)<4)&(yy>22)&(yy<25))-avg((abs(xx-14)>4)&(abs(xx-14)<7.5)&(yy>22)&(yy<25)) for _ in [0]))
# Repetition of tiny contrasting buttons is more distinctive than a single
# central line. Use a zero-mean longitudinal comb times a zero-mean crosswise
# ridge: a plain zipper, horizontal stripes or one constant stripe cancel.
def buttons(offset=0):
    for cx in [13,14,15]:
        for period in [3.3,4.2,5.4]:
            for phase in [0,np.pi/2]:
                along=np.cos((yy-8)*2*np.pi/period+phase);roi=(yy>=6)&(yy<=23)
                ww=ridge(xx,cx+offset,.62,1.9,roi)*along
                yield avg(np.maximum(ww,0))-avg(np.maximum(-ww,0))
featurebank('buttons',buttons());featurebank('off_buttons',buttons(-4))
# An elongated knit collar arcs across the chest; a coat usually has raised
# collar points straddling the fastening. Detect the arc with two flanks.
def necklines():
    for cx in [13.5,14.5]:
        for radius in [2.8,4.7,6.2]:
            for cy in [1.5,3,4.5]:
                ax=abs(xx-cx)/radius;yc=cy+2.1*np.sqrt(np.maximum(0,1-ax**2));yield ridge(yy,yc,.67,1.8,ax<1)
featurebank('neck_arc',necklines())
def hems():
    for cy in np.arange(21.,26.,.8):
        for curve in [-.40,.45]:
            yc=cy+curve*((xx-14)/5)**2
            # fabric above the seam and a strip just below it, not the black
            # lower background, should contrast with the narrow welt.
            yield ridge(yy,yc,.47,1.35,abs(xx-14)<5.8)
featurebank('welt',hems())
z=dense('contrast_probes',norm,np.reshape(patterns,(-1,784)),0)
# Each family has a different number of probes. Pool over the valid ones.
longest=max(e-b for _,b,e in families);W=np.zeros((len(families)*longest,len(patterns)),np.float32)
for j,(_,b,e) in enumerate(families):
    for k in range(longest):W[j*longest+k,b+k%(e-b)]=1
z=dense('arrange_probes',z,W,0);z=layer('family_axis','reshape',z,shape=[1,len(families),longest]);z=pool('strongest_detail',z,'maxpool',1,longest,1,longest)

# Additional regional texture contrasts: a band-limited derivative, pooled
# after taking its absolute value. A busy torso vs an isolated central graphic
# is informative for blouse vs sweatshirt; crosshatching suggests woven cloth.
kx=np.outer([.25,.5,.25],[-1,0,1]);ky=kx.T
v=conv('derivatives',norm,np.stack((kx,-kx,ky,-ky))[:,None],[0]*4,p=1);v=layer('oriented_energy','relu',v)
E=[]; enames=[]
def energy(name,dirs,region):
    E.append((np.array(dirs)[:,None,None]*avg(region)).ravel());enames.append(name)
centre=(abs(xx-14)<5)&(yy>7)&(yy<21)
energy('weft',[1,1,0,0],centre);energy('warp',[0,0,1,1],centre)
energy('graphic',[1,1,1,1],(abs(xx-14)<4.5)&(yy>=5)&(yy<13));energy('plain_lower',[1,1,1,1],(abs(xx-14)<4.5)&(yy>=15)&(yy<22));energy('busy_sides',[1,1,1,1],(abs(xx-14)>2)&(abs(xx-14)<6)&(yy>=7)&(yy<21));energy('busy_centre',[1,1,1,1],(abs(xx-14)<2)&(yy>=7)&(yy<21))
q=dense('cloth_detail',v,E,0)

# Smooth rules: clipped linear signals. All coefficients below are hand-set
# in units of contrast or fractional silhouette disagreement.
names=[n for n,b,e in families]+enames
rules=[]
def rule(terms,low,hi,cls):
    row=np.zeros(len(names));
    for k,w in terms.items():row[names.index(k)]=w/(hi-low)
    rules.append((row,-low/(hi-low),cls))
rule({'placket':1,'other_verticals':-.60},.065,.59,{2:-.025,4:.032,6:.020,0:-.010})
rule({'opening':1,'other_verticals':-.45},.12,.78,{2:-.020,4:.033,6:.011,0:-.008})
rule({'lapels':1},.36,1.15,{2:-.004,4:.018,6:.010})
rule({'ribbing':1,'stripes':-.5},.22,.82,{2:.013})
rule({'graphic':1,'plain_lower':-.9},.15,.83,{0:.016,2:.028,6:-.008})
rule({'chest_emblem':1,'opening':-.9},.20,.85,{2:.012,0:.010})
rule({'weft':1},.31,.79,{6:.020,2:.002})
rule({'horizontal_checks':.48,'vertical_checks':.9},.31,1.0,{6:.014})
rule({'stripes':1,'weft':-.65},.30,1.1,{2:.010,0:.005})
rule({'busy_centre':1,'busy_sides':-.58},.24,.85,{4:.017,6:.010,2:-.007})
rule({'busy_sides':1,'busy_centre':-.59},.30,.82,{6:.013,4:-.007})
rule({'front_step':1,'other_verticals':-.50},.07,.52,{4:.024,2:-.014,6:.014})
rule({'upper_front':.5,'lower_front':.7,'other_verticals':-.65},.15,.66,{4:.020,6:.019,2:-.010})
rule({'weft':.6,'warp':.55,'graphic':-.22,'busy_sides':.27},.26,.65,{6:.040,4:.004})
rule({'vertical_checks':1,'placket':-.38,'weft':.55},.21,.62,{6:.029,2:-.007})
rule({'shirt_tail':1,'weft':.25},.55,1.4,{6:.010})
rule({'buttons':1,'off_buttons':-.74},.095,.49,{6:.047,4:.005,2:-.010})
rule({'neck_arc':.52,'welt':.57,'opening':-.61,'placket':-.29},.27,.77,{2:.030})
rule({'welt':1,'placket':-.63,'other_verticals':-.20},.27,.78,{2:.021,0:.006})
# Coats have continuous front openings; a broken-up front on a garment of
# button-down shape is weaker outerwear evidence. Cotton blouses can also
# be plain: do not force them to have a dramatic front stripe.
rule({'upper_front':.5,'lower_front':.5,'placket':-.94,'off_buttons':-.29},.11,.38,{6:.020,4:-.010})
rule({'chest_emblem':.4,'graphic':.75,'busy_sides':-.87,'front_step':-.20},.06,.66,{2:.022,0:.009})

W=np.vstack([r for r,b,c in rules]*2);b=np.r_[[b for r,b,c in rules],[b-1 for r,b,c in rules]]
p=dense('shape_details',z,W[:,:len(families)],b);h=dense('texture_details',q,W[:,len(families):],0)
p=layer('combine_details','add',[p,h]);p=layer('evidence_ramps','relu',p)
r=np.zeros((10,len(rules)))
for i,(_,_,d) in enumerate(rules):
    for c,vv in d.items():r[c,i]=vv
# Absence of the positive appearance cues should favour simple single-piece
# pullovers and tees over an elaborate jacket or blouse of the same outline.
# Soft details should not by themselves outweigh a major silhouette error.
# A complex repeated print supports a blouse; an unadorned, T-shaped garment
# is more parsimoniously a tee. These are priors, not measured frequencies.
bias=[.010,0,.007,0,-.006,0,-.015,0,0,0]
p=dense('appearance_votes',p,.70*np.hstack([r,-r]),bias)

# A second set of tests measures geometry which a squared pixel error tends
# to underweight: empty space between the legs; flared skirts; holes between
# straps; a shaft vs a shoe tongue. All bounds below are proportions of the
# frame. The same rules apply regardless of the brightness of the article.
x=xx/28;y=yy/28;dx=np.abs(x-.5)
S=[];sn=[]
def region(name,m):sn.append(name);S.append(avg(m).ravel())
region('legsplit',(dx<.048)&(y>.59)&(y<.91));region('legs',(.051<dx)&(dx<.166)&(y>.59)&(y<.91))
region('belt',(dx<.105)&(y<.22)&(y>.04)); region('bodice',(dx<.10)&(y>.32)&(y<.52))
region('upperwidth',(y>.18)&(y<.36));region('waistwidth',(y>.36)&(y<.54));region('skirtwidth',(y>.75)&(y<.93));region('hemwidth',(y>.87)&(y<.96));region('neartop',(dx<.29)&(y<.13));region('farside',(dx>.33)&(y>.45)&(y<.89));region('centre_sky',(dx<.29)&(y<.32));
# Sample the bridge of the instep as well as the upper rear of the shoe.
region('shaft',(x>.65)&(x<.91)&(y>.32)&(y<.56));region('tongue',(x>.41)&(x<.65)&(y>.32)&(y<.56));region('shoefront',(x>.10)&(x<.34)&(y>.30)&(y<.57));region('mass',(x>=0));region('sole',(x>.10)&(x<.90)&(y>.72)&(y<.88));
region('collar',(dx<.105)&(y<.12));region('shoulder_cap',(dx>.17)&(dx<.30)&(y<.12));region('shoulder',(dx>.17)&(dx<.30)&(y>.12)&(y<.22));region('lap',(dx<.15)&(y>.72)&(y<.91));region('trouser_width',(dx>.26)&(dx<.43)&(y>.30)&(y<.9));region('headspace',(dx<.27)&(y<.18));region('beltband',(dx<.16)&(y>.08)&(y<.18));region('broad_neck',(dx>.11)&(dx<.23)&(y<.10));region('hem_sides',(dx>.16)&(dx<.27)&(y>.80)&(y<.94));region('hem_center',(dx<.12)&(y>.80)&(y<.94));
region('heel_height',(x>.69)&(x<.89));region('instep_height',(x>.41)&(x<.60));region('toe_height',(x>.07)&(x<.25));region('ankle_sky',(x>.61)&(x<.94)&(y<.41));region('vamp_sky',(x>.19)&(x<.52)&(y<.41));region('mid_sleeve',(dx>.26)&(dx<.42)&(y>.39)&(y<.63));region('low_sleeve',(dx>.26)&(dx<.42)&(y>.71)&(y<.88));
region('highmid',(y>.05)&(y<.12)&(dx<.15));region('buckle',(y>.20)&(y<.31)&(dx<.16));region('calfside',(y>.62)&(y<.90)&(dx>.21)&(dx<.34));region('skirtwing',(y>.65)&(y<.91)&(dx>.22)&(dx<.36));region('hipwing',(y>.34)&(y<.51)&(dx>.22)&(dx<.36));region('shoulderwing',(y>.21)&(y<.34)&(dx>.29)&(dx<.46));region('hipcore',(y>.40)&(y<.54)&(dx<.16));region('midleg',(y>.53)&(y<.70)&(dx<.06));region('uppercore',(y>.16)&(y<.25)&(dx<.17));region('sidebodice',(y>.23)&(y<.48)&(dx>.25)&(dx<.42));
geo=dense('anatomical_regions','foreground',S,0)
# Closing by vertical lines: a dark run contributes only when there is
# foreground BOTH above and below. This detects negative space even on a
# very tall sandal; the open space above a low sneaker does not count.
# Cumulative extrema by two one-sided pools in padded coordinates.
fg='foreground';pad=14
ident=np.ones((1,1,1,1)); ext=conv('zero_border',fg,ident,[0],p=pad)
# Padding is equal on four sides; select two 28x28 subarrays using dense
# reductions below rather than cropping with huge identity matrices.
topbot=pool('overhead_occupancy',ext,'maxpool',pad+1,1,1,1)
# max over [r-pad,r], [r,r+pad]. Use a single sparse affine readout to
# sum the deficits from one at the two endpoints. ReLU implements a fuzzy AND.
# A 14x14 grid of probes is enough for strap gaps. Also measure bag handles.
H=28+2*pad-pad;D=28+2*pad
I=[];J=[];pxs=[];pys=[]
for iy in range(1,28,2):
    for ix in range(1,28,2):
        z0=np.zeros((H,D),np.float32);z0[iy,ix+pad]=1;z0[iy+pad,ix+pad]=1;I.append(z0.ravel());z1=np.zeros((28,28));z1[iy,ix]=-1;J.append(z1.ravel());pys.append((iy+.5)/28);pxs.append((ix+.5)/28)
z0=dense('vertical_brackets',topbot,I,-1);z1=dense('gap_centres',fg,J,0);gap=layer('enclosed_darkness','add',[z0,z1]);gap=layer('strap_openings','relu',gap)
pointgap=gap
# A single black pigment pixel or an eyelet is not a strap opening. Demand
# nearby empty pixels as well. Near black rather than relative grey matters.
buf=conv('darkness_border','image',ident,[0],p=1);local_max=pool('hole_interior',buf,'maxpool',3,3,1,1)
K=np.array(J)*14.0;penalty=dense('not_an_opening',local_max,K,0)
gap=layer('clear_opening','add',[gap,penalty]);gap=layer('open_area','relu',gap)
pxs=np.array(pxs);pys=np.array(pys);gaps=[];gn=[]
for name,mask in [('holes',(pxs>.15)&(pxs<.94)&(pys>.28)&(pys<.91)),('backholes',(pxs>.48)&(pys>.32)&(pys<.91)),('handles',(pxs>.2)&(pxs<.8)&(pys<.48)),('leghems',(abs(pxs-.5)<.08)&(pys>.65))]:
    gn.append(name);gaps.append(avg(mask))
# A flat waistband and paired slender legs distinguish straight trousers
# even when their cuffs nearly touch. Detect a column between the legs with
# no *horizontal* foreground neighbours except those two flanks. Dark open
# coat fronts can imitate the gap but leave material much farther out.
# Mix isolated clear points with verified large interiors. The point term is
# needed for thin spaces between straps on low, flat sandals.
gA=dense('wide_spaces',gap,.48*np.array(gaps),0);gB=dense('narrow_spaces',pointgap,.52*np.array(gaps),0);gap=layer('spaces_between_parts','add',[gA,gB])
# Quantify collar/placket details near the neck separately from continuous
# centre lines. The height and width of a collar, around which the shoulders
# taper, helps distinguish a high standing coat collar from a round-neck knit.
region_rules=[]
def anatom(t,lo,hi,d):
    names=sn+gn;w=np.zeros(len(names))
    for n,vv in t.items():w[names.index(n)]=vv/(hi-lo)
    region_rules.append((w,-lo/(hi-lo),d))
anatom({'legs':1,'legsplit':-1.18,'belt':.22,'trouser_width':-.90},.17,.62,{1:.12,3:-.018})
anatom({'legsplit':1,'legs':-.58},.32,.56,{1:-.070})
anatom({'belt':.55,'bodice':.30,'upperwidth':-.58,'legsplit':-.23},.46,.67,{1:.047})
anatom({'skirtwidth':1,'waistwidth':-1,'headspace':.18},.19,.37,{3:.061})
anatom({'hemwidth':1,'waistwidth':-1,'headspace':.16},.20,.39,{3:.025})
anatom({'upperwidth':-1,'bodice':.45,'legsplit':.36,'headspace':.12},.44,.62,{3:.053})
anatom({'headspace':-.7,'upperwidth':-.22,'skirtwidth':.35},.04,.16,{3:-.054})
anatom({'trouser_width':1,'legsplit':.28},.31,.53,{1:-.070})
# Raised neck and sloped shoulder line of outerwear vs broad knit yoke. Shirt
# tails are more rounded and less often have a tight-knit ribbed welt.
anatom({'collar':.80,'shoulder_cap':-1,'shoulder':.32},.34,.78,{4:.018,2:-.006,6:.007})
anatom({'shoulder_cap':1,'collar':-.60},-.04,.41,{2:.015,4:-.012,6:.004})
anatom({'hem_center':1,'hem_sides':-.87},.23,.67,{6:.033,4:.003,2:-.005})
anatom({'upperwidth':1,'skirtwidth':-1},.13,.30,{0:.035})
anatom({'upperwidth':1,'waistwidth':-.88,'low_sleeve':-.25},.10,.24,{0:.044})
anatom({'shaft':1,'tongue':-.84,'shoefront':-.18},.15,.66,{9:.026,7:-.008})
anatom({'shaft':-.74,'shoefront':-.25,'tongue':.48,'sole':.35},.10,.36,{7:.032,9:-.012})
anatom({'holes':1},.009,.11,{5:.096,7:-.015,9:-.016})
anatom({'backholes':1},.025,.23,{5:.035,9:-.012})
anatom({'mass':-1},-.23,-.12,{5:.048})
anatom({'handles':1},.035,.22,{8:.074})
anatom({'heel_height':1,'instep_height':-.80,'toe_height':-.20},.07,.26,{9:.045,7:-.012})
anatom({'heel_height':-1,'instep_height':.9,'toe_height':.10},-.033,.085,{7:.045,9:-.022})
anatom({'ankle_sky':1,'vamp_sky':-.82},.12,.47,{9:.019})
anatom({'low_sleeve':.82,'mid_sleeve':.27},.25,.75,{0:-.027,2:.007})
anatom({'highmid':.4,'buckle':.3,'legs':.28,'calfside':-.67,'trouser_width':-.65,'midleg':-.22},.47,.72,{1:.039})
anatom({'skirtwing':1,'hipwing':-.92,'hipcore':.26,'uppercore':.13},.44,.70,{3:.039})
anatom({'hipwing':1,'sidebodice':.4,'skirtwing':-.86},.49,.90,{3:-.034})
# Low skirt width alone is ambiguous between a sleeveless dress and a tank
# top. A cap extending far beyond straight body sides is a sleeve, not a skirt.
anatom({'shoulderwing':1,'skirtwing':-.8,'hipwing':-.58},.22,.58,{0:.035,3:-.041})
W=np.vstack([r for r,b,c in region_rules]*2);b=np.r_[[b for r,b,c in region_rules],[b-1 for r,b,c in region_rules]]
g=dense('gross_proportions',geo,W[:,:len(sn)],b);f=dense('negative_space_evidence',gap,W[:,len(sn):],0);g=layer('part_evidence','add',[g,f]);g=layer('part_ramps','relu',g)
R=np.zeros((10,len(region_rules)))
for i,(_,_,d) in enumerate(region_rules):
    for c,vv in d.items():R[c,i]=vv
coarseR=R.copy();coarseR[[0,2,4,6]]=0
cg=dense('gross_anatomy_votes',g,np.hstack([coarseR,-coarseR]),0)
g=dense('anatomical_votes',g,np.hstack([R,-R]),0)
base=layer('outline_and_parts','add',[shape_scores,g]);coarse=layer('family_anatomy','add',[shape_scores,cg]);within=layer('fine_class_evidence','add',[base,p]);
# Hierarchical decision: details used to tell a coat from a pullover must
# not also be evidence against footwear or trousers. Centre the fine logits
# of the four top classes on their common family's gross-anatomy score.
fam=[0,2,4,6];P=np.eye(10)[fam];Q=np.eye(10);Q[fam]=0
geom=dense('torso_shapes',coarse,P,0);geom=layer('shape_comparison','reshape',geom,shape=[1,1,4]);geom=pool('upper_garment',geom,'maxpool',1,4,1,4)
fine=dense('torso_styles',within,P,0);fine=layer('fine_comparison','reshape',fine,shape=[1,1,4]);fine=pool('winning_style',fine,'maxpool',1,4,1,4)
A=P.T@P;rest=dense('other_articles',base,Q,0);sty=dense('apparel_classes',within,A,0);g1=dense('family_likelihood',geom,A.sum(1)[:,None],0);g2=dense('subtract_peak',fine,-A.sum(1)[:,None],0)
result=layer('top_family_decisions','add',[rest,sty,g1,g2])
# Similarly, the shape of the vamp is evidence for a *type of shoe*, not
# evidence that a handbag with a sloping flap is footwear.
P=np.eye(10)[[5,7,9]];A=P.T@P
s0=dense('foot_contours',shape_scores,P,0);s0=layer('foot_axis','reshape',s0,shape=[1,1,3]);s0=pool('foot_family',s0,'maxpool',1,3,1,3)
s1=dense('foot_styles',base,P,0);s1=layer('shoe_axis','reshape',s1,shape=[1,1,3]);s1=pool('shoe_winner',s1,'maxpool',1,3,1,3)
b0=dense('foot_support',s0,A.sum(1)[:,None],0);b1=dense('cancel_shoe_peak',s1,-A.sum(1)[:,None],0)
result=layer('foot_family_decisions','add',[result,b0,b1])

# Sum of occupancies in a column is an effective height for a solid shoe,
# allowing for a raised toe or heel. A boot's shaft increases the height far
# behind the toe; a sneaker peaks over the instep. With this *relative* rule
# a thick athletic sole no longer looks like an ankle shaft.
# Conversely, a sandal has internal gaps. Measure the largest solid-over-gap
# runs at a range of scales using luminance, not just foreground area.
F=[];fn=[]
def sf(name,mask):fn.append(name);F.append(avg(mask).ravel())
sf('toe',(x>.09)&(x<.25));sf('instep',(x>.47)&(x<.63));sf('ankle',(x>.71)&(x<.87));sf('toe_top',(x>.09)&(x<.25)&(y<.50));sf('ankle_top',(x>.70)&(x<.90)&(y<.50));sf('instep_top',(x>.46)&(x<.63)&(y<.50));sf('whole',(x>=0));feet=dense('shoe_profile',fg,F,0)
# Detect a smooth horizontal midsole running continuously from toe to heel.
# Pair samples in separate thirds at the SAME level, instead of reporting an
# isolated stripe in a patterned upper. Search positions in the lower half.
P0=[];P1=[]
for r0 in np.arange(.54,.97,.036):
    for slope in [-.055,0,.055]:
        cy=r0+slope*(x-.5);p0=ridge(y,cy,.021,.065,(x>.14)&(x<.48));p1=ridge(y,cy,.021,.065,(x>.49)&(x<.86))
        for sign in [1,-1]:P0.append(p0*sign);P1.append(p1*sign)
zL=dense('sole_front',norm,np.reshape(P0,(-1,784)),0);zR=dense('sole_back',norm,np.reshape(P1,(-1,784)),0)
# min(a,b)=a-ReLU(a-b)
dif=dense('minus_heel',zR,-np.eye(len(P0)),0);df=layer('sole_difference','add',[zL,dif]);df=layer('positive_difference','relu',df);df=dense('subtract_difference',df,-np.eye(len(P0)),0)
line=layer('continuous_sole','add',[zL,df]);line=layer('sole_candidates','reshape',line,shape=[1,1,len(P0)]);line=pool('strong_sole',line,'maxpool',1,len(P0),1,len(P0))
# Family-internal score changes expressed in units of normalized shape cost.
shoe_rules=[]
def shoe(t,lo,hi,ds):
    v=np.zeros(len(fn)+len(gn)+1)
    for n,val in t.items():v[(fn+gn+['line']).index(n)]=val/(hi-lo)
    shoe_rules.append((v,-lo/(hi-lo),ds))
shoe({'whole':-1,'holes':.65},-.28,-.15,{5:.046})
shoe({'holes':1,'backholes':.32},.017,.095,{5:.075})
shoe({'line':1,'holes':-4},.36,1.2,{7:.031})
shoe({'ankle':-1,'instep':1.12,'toe':.16,'ankle_top':-.20},-.01,.10,{7:.055,9:-.017})
shoe({'ankle_top':.70,'instep_top':-.51,'ankle':.35,'instep':-.31},.09,.32,{9:.019})
shoe({'ankle':-1,'instep':.68,'toe':.29,'holes':.9,'line':-.07},.065,.22,{5:.066})
W=np.array([rr for rr,b,d in shoe_rules]*2);bb=np.r_[[b for rr,b,d in shoe_rules],[b-1 for rr,b,d in shoe_rules]]
x1=dense('sole_geometry',feet,W[:,:len(fn)],bb);x2=dense('sole_holes',gap,W[:,len(fn):-1],0);x3=dense('sole_line',line,W[:,-1:],0);x1=layer('foot_cues','add',[x1,x2,x3]);x1=layer('foot_ramps','relu',x1)
M=np.zeros((10,len(shoe_rules)))
for i,(_,_,d) in enumerate(shoe_rules):
    for c,vv in d.items():M[c,i]=vv
ff=dense('shoe_expert',x1,np.hstack([M,-M]),0)
result=layer('shoe_detail_scores','add',[result,ff]);Pk=np.eye(10)[[5,7,9]];Ap=Pk.T@Pk
newfoot=dense('detail_shoes',result,Pk,0);newfoot=layer('detail_axis','reshape',newfoot,shape=[1,1,3]);newfoot=pool('detail_peak',newfoot,'maxpool',1,3,1,3);ft=dense('remove_foot_detail_peak',newfoot,-Ap.sum(1)[:,None],0)
result=layer('refined_footwear','add',[result,b0,ft])
# A bag requires a broad, coherent lower pouch. A gap on an oblique small
# object is more likely between straps than between a handle and the pouch.
# Use max vertical run density below the putative handle, not just handle
# void area. All quantities are evaluated by these layers at inference.
levels=np.linspace(.46,.82,6);V=[]
for lev in levels:
    for width in [.44,.64]:V.append(avg((abs(x-.5)<width/2)&(y>lev-.09)&(y<lev+.09)).ravel())
t=dense('pouch_regions',fg,V,0);t=layer('pouch_regions_axis','reshape',t,shape=[1,1,len(V)]);t=pool('solid_pouch',t,'maxpool',1,len(V),1,len(V));
# Additional bag evidence from the profile on both sides of the lower body:
# handbag ends are taller and less unequal than toe and heel. Use absolute
# difference in column heights and measure how much of the central lower
# region has a backdrop-free solid span.
VB=[]
for ma in [(x>.10)&(x<.28),(x>.72)&(x<.90),(x>.28)&(x<.72),x>=0]:VB.append(avg(ma).ravel())
bv=dense('article_width_sections',fg,VB,0);bW=np.array([[1,-1,0,0],[-1,1,0,0]]);dif=dense('side_heights',bv,bW,0);dif=layer('side_asymmetry','relu',dif)
bags=[(-.53,[-.24,-.24,.18,1.1],.36,-.35,.55,.14,.038),(.36,[.0,.0,0,-.8],-.44,.30,.85,.24,-.087)]
rows=[];rows2=[];rows3=[];rows4=[];bias=[];outs=[]
for intercept,arr,body,asy,handle,scale,vote in bags:
    bias.append(intercept/scale);rows.append(np.array(arr)/scale);rows2.append([body/scale]);rows3.append([asy/scale]*2);rows4.append([0,0,handle/scale,0]);outs.append(vote)
r1=dense('bag_outline',bv,rows+rows,bias+[b-1 for b in bias]);r2=dense('bag_pouch',t,rows2+rows2,0);r3=dense('bag_sides',dif,rows3+rows3,0);r4=dense('bag_handle',gap,rows4+rows4,0);br=layer('bag_part_agreement','add',[r1,r2,r3,r4]);br=layer('bag_ramps','relu',br);BM=np.zeros((10,2*len(outs)));BM[8]=np.r_[outs,-np.array(outs)];br=dense('pouch_evidence',br,BM,0)
# Apply the pouch check only as a negative verifier of an existing bag
# hypothesis, never as a positive vote. A solid torso is not itself evidence
# of a bag. min(0,br) lets the silhouette/handle path keep sole responsibility
# for promoting the bag class.
ib=dense('negative_bag_evidence',br,-np.eye(10),0);ib=layer('bag_veto_only','relu',ib);ib=dense('bag_verification',ib,-np.eye(10),0)
layer('class_scores','add',[result,ib])

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},indent=1));np.savez(out/'weights.npz',**weights)
