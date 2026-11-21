import numpy as np

def smooth(x):
    h=np.concatenate((x[:,:,:1],x,x[:,:,-1:]),2)
    h=(h[:,:,:-2]+2*h[:,:,1:-1]+h[:,:,2:])*.25
    v=np.concatenate((h[:,:1],h,h[:,-1:]),1)
    return (v[:,:-2]+2*v[:,1:-1]+v[:,2:])*.25

def grid(x,n):
    m=x.shape[1]//n
    return x.reshape(len(x),n,m,n,m).mean((2,4))

def unit(x):
    x=x.reshape(len(x),-1)
    return x/(np.sqrt(np.sum(x*x,axis=1,keepdims=True))+.00001)

def dif(x):
    h=np.concatenate((x[:,:,:1],x,x[:,:,-1:]),2); v=np.concatenate((x[:,:1],x,x[:,-1:]),1)
    return .5*(h[:,:,2:]-h[:,:,:-2]), .5*(v[:,2:]-v[:,:-2])

def fuzzy(x): return np.clip(x,0,1)

def imageparts(x):
    r,g,b=x[:,0],x[:,1],x[:,2]; lum=.28*r+.55*g+.17*b
    return r,g,b,lum

def describe(x):
    r,g,b,y=imageparts(x); sm=smooth(y); dx,dy=dif(sm)
    mag=np.sqrt(dx*dx+dy*dy)+.000001
    # Centered, unsigned orientation channels. The flat angular response is
    # removed so texture alone cannot mimic a directional boundary.
    cells=[]; contours=[]
    for j in range(6):
        angle=j*(np.pi/6); pr=np.abs(dx*np.cos(angle)+dy*np.sin(angle))/mag
        cells.append(grid(smooth((np.maximum(pr-.77,0)/.23-.18)*np.sqrt(mag)*mag/(mag+.012)),8))
        # Keep the prominent, persistent structures in a second edge map, for
        # comparisons less sensitive to incidental textures on an object.
        contours.append(grid(smooth((pr**4-.375)*mag),8))
    hog=np.stack(cells,3); strength=np.sqrt((hog*hog).sum(3,keepdims=True))
    hog=hog/np.sqrt(strength+.023)
    coarse=np.stack([grid(smooth(v),4) for v in contours],3)
    central=hog[:,1:7,1:7]
    gap=grid(sm,8); ga=gap-gap.mean((1,2),keepdims=True); detail=gap-smooth(smooth(gap)); energy=grid(np.sqrt(mag),8); energy-=energy.mean((1,2),keepdims=True)
    # Chromatic adaptation with a floor: do not amplify noise in shadows.
    rg=(r-g)/(.23+y); gb=(g-b)/(.23+y); sat=x.max(1)-x.min(1)
    veget=fuzzy(.41-rg*5)*fuzzy(gb*5+.13)*fuzzy(g*6)
    sky=fuzzy(.24-3.7*gb)*fuzzy(.48-3.5*rg)
    warm=fuzzy(rg*3+.13)*fuzzy(gb*3.7+.24)
    neutral=fuzzy(1.2-sat/(y+.22)*4.5)
    red=fuzzy((r-np.maximum(g,b)-.075)*5); golden=fuzzy(gb*3.5)*fuzzy(rg*3+.46)
    # A coarse map of soft material/color memberships; RBF tones distinguish
    # asphalt, white overcast sky and deep shadows without a hard label.
    maps=[veget,sky,warm,neutral*.76,red,golden*.67]
    for t in [.14,.47,.82]: maps.append(np.exp(-((y-t)/.235)**2)*.71)
    color=np.stack([grid(smooth(v),4) for v in maps],3)
    # These fixed reference levels are not statistics of a collection of views.
    color-=np.array([.18,.18,.20,.28,.05,.13,.15,.25,.15],np.float32)
    return np.concatenate([unit(hog)*.23**.5,unit(central)*.16**.5,unit(coarse)*.21**.5,unit(ga)*.17**.5,unit(detail)*.14**.5,unit(energy)*.09**.5],1).astype(np.float32),unit(color).astype(np.float32)

def context(x):
    r,g,b,y=imageparts(x); dx,dy=dif(smooth(y)); norm=np.sqrt(dx*dx+dy*dy)+.000001
    def avg(t): return t.mean((1,2))
    def upper(t): return avg(t[:,:9])
    def lower(t): return avg(t[:,24:])
    def middle(t): return avg(t[:,8:25,6:26])
    def surround(t): return (upper(t)+lower(t)+.5*(avg(t[:,:,0:5])+avg(t[:,:,-5:])))/3
    def squash(t): return np.clip(t,-1,1)
    rg=(r-g)/(.22+y); gb=(g-b)/(.22+y); sat=x.max(1)-x.min(1)
    leafy=fuzzy(.46-5.5*rg)*fuzzy(6.5*gb+.11)*fuzzy(7*g)
    blue=fuzzy(.22-5*gb)*fuzzy(.55-4*rg)
    earth=fuzzy(.14+3.8*rg)*fuzzy(.28+4.2*gb)
    cool=fuzzy(.59-2.8*rg-1.9*gb); warm=fuzzy(2.4*rg+2.3*gb+.12)
    pale=fuzzy((y-.52)*3)*fuzzy(1.2-3.5*sat); dark=fuzzy((.31-y)*4.7)
    hi=upper(blue); lo=lower(blue); leaf=surround(leafy); dirt=surround(earth); field=lower(leafy); cold=surround(cool)
    # Smooth expanses and horizontally stratified backgrounds (sky, sea, road).
    topgrad=upper(norm); botgrad=lower(norm); centergrad=middle(norm); smoothsky=np.exp(-topgrad/.025)*fuzzy(upper(y)*1.6)*fuzzy(upper(cool)+upper(pale))
    cleanbelow=np.exp(-botgrad/.034); bothclean=np.exp(-(topgrad+botgrad)/.046)
    whiteabove=upper(pale); skyabove=fuzzy(hi*1.4+.78*whiteabove)*(.45+.55*np.exp(-topgrad/.046))
    def horiz(dx,dy):
        ax=avg(abs(dx)); ay=avg(abs(dy)); return squash(3*(ay-ax)/(.016+ax+ay))
    horizont=horiz(dx,dy); basehoriz=horiz(dx[:,18:],dy[:,18:]); vert=-horiz(dx[:,15:29,4:28],dy[:,15:29,4:28]);
    # Local orientation coherence separates straight panels from fur or foliage.
    tensor=np.sqrt(grid(dx*dx-dy*dy,8)**2+grid(2*dx*dy,8)**2)/(grid(norm*norm,8)+.00012)
    cohere=squash((avg(tensor)-.48)*3.2)
    align=avg((abs(dx*dx-dy*dy)-abs(2*dx*dy))/(norm+.004))/(avg(norm)+.01); square=squash(align*4)
    red=fuzzy((r-np.maximum(g,b)-.10)*5); flash=middle(red)+middle(fuzzy((b-np.maximum(r,g)-.12)*5));
    # Bottom-vs-middle contrast of horizontal gradients (shadow under chassis).
    strip=grid(y,8); stripavg=strip[:,:,1:7].mean(2); shadow=fuzzy(((stripavg[:,6:8].mean(1)-stripavg[:,4:7].min(1))-.035)*4)
    neutral=fuzzy(1.15-4.*sat/(y+.22)); pavement=lower(neutral)*cleanbelow
    animal=1.22*leaf+.72*dirt-.68*cold-.47*skyabove-.46*square-.27*cohere-.24*horizont
    vehicle=-.83*leaf-.53*dirt+.41*cold+.46*square+.21*cohere+.20*horizont
    midwarm=middle(warm); middark=middle(dark); texture=fuzzy((centergrad-.027)*13); brownground=lower(earth)
    skyexp=smoothsky; water=lo+.40*lower(cool); forest=upper(leafy); centralLeaf=middle(leafy)
    animal*=.8; vehicle*=.8
    sc=[vehicle+.98*bothclean+.81*hi+.40*lo+.49*skyexp-.56*horizont-.28*square-.22*shadow-.24,
        vehicle+.67*pavement+.46*shadow+.44*cleanbelow+.35*flash+.20*middark-.37*skyexp-.36*lo-.18,
        .52*animal+.54*bothclean+.32*centralLeaf+.32*forest+.39*hi+.18*flash-.21*square-.10,
        .62*animal+.36*midwarm+.23*texture-.62*field-.28*forest-.42*skyabove-.30*cold+.24*surround(neutral)+.09,
        animal+.55*leaf+.35*brownground+.39*midwarm+.27*forest+.20*vert-.28*flash-.12,
        .70*animal+.30*midwarm+.18*field+.19*texture-.40*skyabove-.26*centralLeaf+.10,
        animal+.79*centralLeaf+.49*brownground+.27*texture-.62*skyabove-.22*vert-.31*cleanbelow,
        animal+.67*field+.50*skyabove+.29*vert+.29*middark-.22*centralLeaf-.10,
        vehicle+.96*water+.68*horizont+.49*basehoriz+.42*skyabove-.38*square-.25*shadow-.20,
        vehicle+.32*pavement+.40*square+.27*skyabove+.25*flash+.31*shadow-.34*lo-.15]
    return np.stack(sc,1)

def parts(x):
    r,g,b,y=imageparts(x)
    # Matched features on the query only: dark wheels in an arch, eyes and nose,
    # thin legs in front of a field, and straight boundaries of flat panels.
    sm=smooth(y); dx,dy=dif(sm); n=len(y)
    def moved(a,iy,ix): return np.roll(a,(iy,ix),axis=(1,2))
    def mx(a): return a.reshape(n,-1).max(1)
    def av(a): return a.mean((1,2))
    def signed(a): return np.clip(a,-1,1)
    def sample(a,iy,ix): return moved(a,-iy,-ix)
    def stripe(a,iy,ix):
        v=a
        for shift in [-4,-2,2,4]: v=v+moved(a,iy*shift,ix*shift)
        return abs(v)*.20
    # Compare horizontal, vertical and diagonal supporting lines, not just
    # individual pixel gradients. Allocate evidence for right-angled bodies.
    horizontal=stripe(dy,0,1)[:,4:28,5:-5]; vertical=stripe(dx,1,0)[:,4:28,5:-5]
    diag=(stripe((dx-dy)*.707,1,1)[:,4:28,5:-5]+stripe((dx+dy)*.707,1,-1)[:,4:28,5:-5])*.5
    def energy(a): return np.sqrt(av(a*a*a*a)+1.e-12)**.5
    hv=energy(horizontal); vv=energy(vertical); dv=energy(diag); ortho=signed(3*((hv+vv)*.5-dv)/(.027+dv+hv*.3))
    tall=signed(4*(energy(vertical[:,:13])-dv)/(.048+hv+vv)); base=signed(3*(energy(horizontal[:,10:])-dv)/(.039+hv+vv))
    # Matched wheel pairs: a dark curved tire and a possible bright hub.
    # Corners and straight shadow boundaries do not enclose a disk.
    wheel=np.zeros(n,np.float32); eyes=np.zeros(n,np.float32)
    for radius in [2,3,4]:
        off=radius+1; dg=max(1,int(round(off*.707))); inside=max(1,radius-1)
        ring=(sample(sm,0,inside)+sample(sm,0,-inside)+sample(sm,-inside,0)+sample(sm,inside,0))*.25
        corners=[sample(sm,j,i) for i,j in [(off,0),(-off,0),(0,-off),(0,off),(dg,-dg),(-dg,-dg),(dg,dg),(-dg,dg)]]
        arc=sum(corners)/8; px=np.minimum(corners[0],corners[1]); py=np.minimum(corners[2],corners[3]); rim=.65*(arc-ring)+.30*(np.minimum(px,py)-ring)+.20*np.maximum(sm-ring,0)
        rim=np.maximum(0,rim)*(.55+.45*fuzzy((.60-ring)*2.5)); bestx=np.zeros(n,np.float32)
        for gap in [3*radius,4*radius,5*radius,6*radius]:
            if gap>23: continue
            for ddy in [-2,0,2]:
                other=sample(rim,ddy,gap); a=rim[:,16:28,off:32-gap-off]; q=other[:,16:28,off:32-gap-off]
                if a.shape[2]>0: bestx=np.maximum(bestx,mx(.83*np.minimum(a,q)+.17*np.maximum(a,q)))
        wheel=np.maximum(wheel,bestx)
        # A bilateral dark eye pair; look for a dark nose on the middle axis,
        # but allow a contrasting short muzzle or forehead to support one eye.
        blob=np.maximum(0,arc-sm+.2*(py-sm)); ew=radius
        for gap in [4*ew,6*ew]:
            if gap>18: continue
            left=blob; right=sample(blob,0,gap); nose=.5*sample(blob,gap//2,gap//2)+.5*sample(blob,3*gap//4,gap//2)
            pair=np.minimum(left,right)+.18*np.minimum((left+right)*.5,nose)
            eyes=np.maximum(eyes,mx(pair[:,5:19,off:32-gap-off]))
    wheel=signed((wheel-.095)*6); eyes=signed((eyes-.105)*3.7)
    # Stripes representing long, narrow legs, typically against lighter grass.
    darkline=.5*(sample(sm,0,2)+sample(sm,0,-2))-sm; leg=np.maximum(0,(darkline+sample(darkline,2,0)+sample(darkline,-2,0))/3)
    legs=np.zeros(n,np.float32)
    for gap in [4,7,10,14]: legs=np.maximum(legs,mx(np.minimum(leg,sample(leg,0,gap))[:,16:28,3:30-gap]))
    legs=signed((legs-.072)*6.)
    # Local roughness and blue open space around the subject. Frogs tend to
    # occupy the texture right to the margins, unlike perched silhouettes.
    rough=abs(y-sm); border=.5*(av(rough[:,:6])+av(rough[:,26:])); grain=signed((border-.024)*19.)
    contrast=av(abs(sm[:,6:26,6:26]-sm[:,6:26,6:26].mean((1,2),keepdims=True)))
    cornersmooth=np.exp(-border/.019); central=signed((contrast-.105)*7); similar=np.zeros(n,np.float32)
    for shift in [-3,0,3]:
        rhs=sample(sm[:,:,::-1],0,shift); diff=av(abs(sm-rhs)[:,6:24,6:26]); similar=np.maximum(similar,np.exp(-diff/(.027+.55*contrast)))
    sym=signed((similar-.34)*2.9); face=eyes*(.78+.25*sym)+.20*sym
    # Achromatic manufactured colors and the conspicuous yellow-green of
    # many amphibians are useful when the background occupies little area.
    sat=x.max(1)-x.min(1); lime=av((fuzzy((g-r+.022)*7)*fuzzy((g-b-.052)*6))[:,8:25,6:26]);
    skycolor=fuzzy(.35+(b-r)*6)*fuzzy(.50+(b-g)*6); skycolor=av(skycolor[:,:10]);
    man=ortho*.28-grain*.25; natural=-.25*ortho+.22*grain
    return np.stack([-.22*ortho-.26*base-.30*grain-.13*legs+.13*central,
        man+.83*wheel+.30*base-.17*tall+.15*central,
        -.28*ortho+.32*cornersmooth+.16*central-.18*legs-.19*base,
        natural+.44*face+.13*central-.15*legs,
        natural+.46*legs-.26*face-.10*sym,
        natural+.33*face+.17*legs,
        natural+.65*lime+.13*face+.27*grain-.25*legs-.12*central,
        natural+.58*legs+.09*tall-.14*face+.13*central,
        -.10*ortho+.42*base-.15*wheel+.16*skycolor,
        man+.47*wheel+.46*tall+.32*ortho+.12*base],1)

def match(x):
    cues=context(x)+parts(x)
    shape,color=describe(x); flipshape,flipcolor=describe(x[:,:,:,::-1].copy())
    # Either handedness is a valid view. Pair colors with the correct mirror.
    sh=shape@CONSTANTS['shape'].T; shf=flipshape@CONSTANTS['shape'].T
    co=color@CONSTANTS['color'].T; cof=flipcolor@CONSTANTS['color'].T
    def best(a,b):
        vals=np.maximum(a,b).reshape(len(x),10,-1); p=np.partition(vals,-5,axis=2)[:,:,-5:]
        return .50*p.max(2)+.50*p.mean(2)
    s=best(.79*sh+.21*co,.79*shf+.21*cof)
    c=best(.31*sh+.69*co,.31*shf+.69*cof)
    return 15*s+6*c+1.22*cues

def predict(x):
    out=np.empty(len(x),np.int64)
    for i in range(0,len(x),48):
        n=min(48,len(x)-i); z=np.zeros((48,3,32,32),np.float32); z[:n]=x[i:i+n]; out[i:i+n]=match(z)[:n].argmax(1)
    return out
