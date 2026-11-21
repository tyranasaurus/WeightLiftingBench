"""Construct individual shape filters from geometric descriptions; no inputs other than the output path."""
import sys
from pathlib import Path
import numpy as np

# Both the drawings and the photographs use this fixed descriptor. Pool and
# normalize each sample separately. All feature maps have two spatial axes last.
common = '''import numpy as np

def pool(a, n):
    b = a.reshape(a.shape[:-2] + (a.shape[-2]//n,n,a.shape[-1]//n,n))
    return b.mean(axis=(-3,-1))

def blur(a, r):
    p = [(0,0)]*(a.ndim-2) + [(r,r),(r,r)]
    t = np.pad(a, p, mode="edge")
    w = 2*r+1
    t = np.cumsum(t,axis=-1,dtype=np.float32)
    t = (t[...,w-1:] - np.pad(t[...,:-w],[(0,0)]*(a.ndim-1)+[(1,0)])) / w
    t = np.cumsum(t,axis=-2,dtype=np.float32)
    return (t[...,w-1:,:] - np.pad(t[...,:-w,:],[(0,0)]*(a.ndim-2)+[(1,0),(0,0)])) / w

def deriv(a):
    p=np.pad(a,[(0,0),(1,1),(1,1)],mode="edge")
    return (p[:,1:-1,2:]-p[:,1:-1,:-2])*.5,(p[:,2:,1:-1]-p[:,:-2,1:-1])*.5

def unit(a):
    a=a.reshape(len(a),-1)
    return a / np.sqrt((a*a).sum(1,keepdims=True)+.000001)

def spatial(a):
    return a-a.mean((-2,-1),keepdims=True)

def descriptor(lum, mask, extra=None):
    smooth=.6*lum+.4*blur(lum,1)
    gx,gy=deriv(smooth)
    mag=np.sqrt(gx*gx+gy*gy+.000001)
    bins=[]
    for j in range(6):
        cs=np.float32(np.cos(j*np.pi/6))
        sn=np.float32(np.sin(j*np.pi/6))
        pr=np.abs(cs*gx+sn*gy)
        v=np.maximum(pr-.80*mag,0)*5
        if extra is not None:
            dx,dy,cm=extra
            v=np.maximum(v,.7*np.maximum(np.abs(cs*dx+sn*dy)-.80*cm,0)*5)
        bins.append(np.sqrt(pool(.65*v+.35*blur(v,1),4)+.0015)-.03873)
    h=np.stack(bins,axis=1)
    h=h-.67*h.mean(axis=1,keepdims=True)
    h=h-.8*h.mean((-2,-1),keepdims=True)
    l=pool(smooth,2)
    loc=l-blur(l,2)
    l=(.30*spatial(l)+.70*loc)/np.sqrt(.029+blur(np.abs(loc),2))
    m=pool(mask,2) if mask.shape[-1]==32 else mask
    m=.55*spatial(m)+.45*(m-blur(m,3))
    return np.concatenate((.40*unit(h),.44*unit(l),.16*unit(m)),axis=1)
'''
# Use the identical formulas in the data-free builder and in the delivered program.
helpers = {}
exec(common, helpers)

# Coordinates below are in object units (x right, y down). Each pose is a
# separate drawing, not an outline estimated from a photograph.
def draw(c, pose, x, y, tones):
    b,h,s,k,d,w,g,f,bg = tones
    im = np.broadcast_to(bg, x.shape).copy()
    mask = np.zeros(x.shape, np.float32)
    def paint(color, sel):
        np.copyto(im, color, where=sel)
        mask[sel]=1
    def e(color,cx,cy,rx,ry,angle=0):
        a=np.deg2rad(angle); u=x-cx; v=y-cy
        paint(color, ((np.cos(a)*u+np.sin(a)*v)/rx)**2 + ((np.cos(a)*v-np.sin(a)*u)/ry)**2 <= 1)
    def p(color,*points):
        sel=np.zeros(x.shape,bool)
        for (a,b0),(c0,d0) in zip(points,points[1:]+points[:1]):
            if b0!=d0:
                sel ^= ((y>b0)!=(y>d0)) & (x < (c0-a)/(d0-b0)*(y-b0)+a)
        paint(color,sel)
    def l(color,r,*points):
        for (a,b0),(c0,d0) in zip(points,points[1:]):
            dx=c0-a; dy=d0-b0
            u=np.clip(((x-a)*dx+(y-b0)*dy)/(dx*dx+dy*dy),0,1)
            paint(color,(x-a-u*dx)**2+(y-b0-u*dy)**2<r*r)
    def wheel(cx,cy,rx,ry):
        e(k,cx,cy,rx,ry); e(g*.62+.12,cx,cy,.51*rx,.51*ry)
    if c==0:
        if pose in (0,1):  # airliner or small propeller plane, seen from the side
            p(s,(-.21,-.035),(.39,-.33),(.53,-.31),(.12,.025))
            p(b,(-.90,.04),(-.99,-.32),(-.86,-.31),(-.60,-.03),(.58,-.06),(.80,-.033),(.96,.034),(.99,.067),(.75,.103),(-.42,.13))
            p(h,(-.72,.01),(.65,-.047),(.84,-.01),(.93,.026),(-.49,.057))
            p(s,(-.04,.057),(-.45,.37),(-.31,.38),(.32,.08))
            p(b,(-.77,.045),(-.96,.20),(-.85,.196),(-.55,.054))
            e(d,-.18,.205,.13,.043); l(d,.013,(-.49,.025),(.61,-.003))
            if pose==1:
                p(b,(-.12,-.045),(-.47,-.34),(-.29,-.34),(.24,-.054)); l(k,.015,(.93,-.13),(.93,.19))
        elif pose in (2,3): # banked / overhead, wings either swept or straight
            sweep=.35 if pose==2 else .11
            p(b,(-.1,-.10),(-.97,sweep-.09),(-.98,sweep+.03),(-.16,.15),(-.045,.45),(-.33,.57),(-.33,.62),(0,.57),(.33,.62),(.33,.57),(.045,.45),(.16,.15),(.98,sweep+.03),(.97,sweep-.09),(.10,-.10))
            e(h,0,-.01,.073,.64); e(d,0,-.40,.048,.093)
            for j in (-1,1): e(s,.31*j,.07,.041,.115)
        else: # head-on: thin wings, fat fuselage, vertical stabilizer
            p(b,(-1,-.12),(-.14,-.025),(-.035,-.05),(-.018,-.33),(.013,-.33),(.033,-.05),(.15,-.025),(1,-.12),(.98,-.07),(.12,.065),(-.12,.065),(-.98,-.07))
            e(h,0,.015,.13,.145); e(d,0,-.045,.094,.024)
            for j in (-1,1): e(s,j*.36,.049,.053,.062)
    elif c==1:
        if pose in (0,1,2):  # sedan, hatchback, van profiles
            roofx = [-.34,-.68,-.74][pose]; roofy=[-.38,-.39,-.49][pose]
            p(b,(-.96,-.07),(-.69,-.13),(roofx,roofy),(.27,roofy),(.61,-.13),(.93,-.065),(.985,.04),(.96,.22),(.71,.27),(-.86,.27),(-.99,.18))
            p(g,(-.59,-.125),(roofx+.035,roofy+.047),(-.06,roofy+.047),(-.08,-.13))
            p(g,(-.025,roofy+.047),(.24,roofy+.048),(.53,-.13),(-.04,-.13))
            p(h,(-.88,-.071),(.53,-.13),(.94,-.066),(.9,-.027),(-.9,-.015))
            p(s,(-.96,.16),(.95,.12),(.96,.22),(.72,.26),(-.86,.27)); e(w,.88,-.001,.071,.028)
            for j in (-.59,.59): wheel(j,.216,.161,.167)
        elif pose in (3,4): # approaching car, oblique hood and visible flank
            p(b,(-.96,-.13),(-.76,-.42),(-.26,-.49),(.23,-.42),(.54,-.09),(.93,.012),(.98,.18),(.89,.34),(.22,.43),(-.83,.22),(-.98,.10))
            p(g,(-.86,-.13),(-.72,-.36),(-.41,-.41),(-.47,-.105)); p(g,(-.38,-.10),(-.31,-.414),(.18,-.374),(.44,-.096),(.14,-.059))
            p(h,(-.36,-.077),(.45,-.077),(.90,.022),(.22,.146),(-.54,-.005)); p(s,(.20,.146),(.96,.03),(.955,.225),(.87,.31),(.21,.393))
            l(d,.028,(.39,.258),(.81,.19)); e(w,.29,.17,.095,.045,-7); e(w,.87,.088,.07,.035,-8)
            wheel(-.76,.143,.105,.14); wheel(-.022,.289,.12,.172)
        else:
            for j in (-1,1): e(k,j*.61,.37,.125,.20)
            p(b,(-.77,.22),(-.75,-.055),(-.61,-.42),(-.42,-.51),(.43,-.51),(.62,-.42),(.75,-.055),(.78,.22),(.68,.41),(.40,.47),(-.45,.47),(-.69,.405))
            p(g,(-.59,-.14),(-.47,-.411),(-.36,-.447),(.39,-.447),(.52,-.39),(.605,-.14)); p(h,(-.60,-.105),(.61,-.105),(.71,.038),(.45,.13),(-.46,.13),(-.71,.038))
            p(d,(-.45,.15),(.45,.15),(.33,.235),(-.33,.235)); l(k,.033,(-.51,.339),(.50,.339))
            for j in (-1,1): e(w,j*.55,.117,.136,.047,-j*9)
    elif c==2:
        if pose in (0,1,2,3): # perched songbird, duck, wader
            upright=[-31,-53,-6,-17][pose]
            if pose==3:
                l(s,.025,(0,.15),(-.06,.62),(-.08,.83)); l(s,.021,(.16,.14),(.22,.63),(.18,.84)); l(b,.083,(.29,-.095),(.51,-.41),(.49,-.63))
            else:
                for j in (-.13,.12): l(d,.023,(j,.17),(j-.035,.40),(j+.073,.414))
            hx=[.32,.22,.46,.54][pose]; hy=[-.30,-.455,-.226,-.64][pose]
            p(f,(-.27,-.025),(-.88,.19 if pose==2 else .51),(-.73,.22 if pose==2 else .55),(-.12,.16))
            e(b,0,-.005,.555,.28,upright); e(h,hx,hy,.232,.211); e(f,hx-.03,hy-.07,.197,.151)
            e(s,-.09,-.025,.395,.157,upright); p(s,(hx+.14,hy-.043),(hx+(.43 if pose>=2 else .32),hy+.009),(hx+.16,hy+.064)); e(k,hx+.092,hy-.052,.034,.033)
        else: # bird in flight, different phases of the wingbeat
            flap=[-.55,-.27,.25][pose-4]
            for j in (-1,1):
                p(b,(j*.055,-.08),(j*.41,flap*.87-.08),(j*.99,flap),(j*.63,flap+.21),(j*.30,.19),(j*.06,.14))
            e(h,0,.04,.103,.27); p(s,(-.04,.17),(-.12,.41),(.13,.39),(.04,.13)); e(b,.015,-.188,.076,.086)
    elif c in (3,5) and pose in (0,1,2,3): # cat and dog portraits
        cat=c==3
        wide=pose in (1,3)
        jaw=.53 if cat else .44
        e(s,0,.99,.73 if cat else .80,.61)
        if cat:
            for j in (-1,1):
                p(b,(j*.68,-.19),(j*(.66 if wide else .61),-.91),(j*.20,-.51)); p(f,(j*.57,-.37),(j*.60,-.75),(j*.31,-.48))
        else:
            for j in (-1,1):
                if pose==2:
                    p(d,(j*.61,-.12),(j*.62,-1.00),(j*.27,-.51))
                else:
                    e(f,j*.599,-.11,.244,.51,j*(-14 if wide else 12)); e(s,j*.65,-.22,.18,.35,j*-12)
        e(b,0,-.086,.69 if wide else .61,.54 if cat else .64)
        if not cat: e(h,0,-.38,.28,.27)
        for j in (-1,1):
            cx=j*(.263 if cat else .295); ey=-.156 if cat else -.194
            e(s,cx,ey-.034,.16,.105,j*7); e(k,cx,ey,.143 if cat else .083,.063 if cat else .073,j*-12)
            if cat:
                e(h,cx,ey,.095,.040,j*-12); e(k,cx,ey,.023,.041)
            else: e(w,cx-.019,ey-.022,.017,.018)
            if cat and pose in (0,2):
                l(f,.036,(j*.21,-.51),(j*.135,-.316)); l(f,.030,(j*.53,-.034),(j*.392,.001)); l(f,.031,(j*.57,.074),(j*.40,.10))
        if cat:
            e(h,-.121,.155,.196,.141); e(h,.121,.155,.196,.141); e(h,0,.308,.155,.102)
            p(d,(-.092,.079),(.09,.079),(0,.166)); l(d,.015,(0,.155),(0,.216))
        else:
            e(s,0,.298,jaw,.294); e(h,-.094,.235,.238,.209); e(h,.10,.235,.236,.21)
            e(k,0,.158,.155,.105); l(k,.022,(-.166,.382),(0,.409),(.165,.38))
            if pose==3: e(f,0,.479,.112,.13)
    elif c in (3,5): # full pet: side profile or seated three-quarter view
        cat=c==3
        if pose in (4,5):
            if cat: l(s,.065,(-.48,.01),(-.77,-.18),(-.87,-.48),(-.85,-.70))
            else: l(s,.063,(-.42,-.04),(-.75,-.17),(-.92,-.40))
            for j in (-.42,-.28,.28,.40):
                l(s if j in (-.28,.28) else b,.066,(j,.093),(j-.06,.342),(j+.012,.59),(j+.12,.59))
            e(b,-.05,-.02,.584,.266,-5); e(b,.40,-.21,.245,.32,23); e(h,.535,-.398,.235,.223)
            if cat:
                p(b,(.353,-.44),(.353,-.73),(.535,-.54)); p(s,(.50,-.50),(.60,-.734),(.666,-.48))
                e(h,.713,-.336,.107,.089)
            else:
                e(f,.416,-.407,.122,.186,10); e(h,.753,-.325,.184,.105); e(k,.88,-.348,.06,.047)
            e(k,.627,-.434,.033,.034)
        else:
            e(s,-.19,.332,.52,.421,-11); e(b,.092,.114,.368,.562,-7); e(h,.183,-.418,.317,.309)
            for j in (-1,1):
                if cat: p(b,(.18+j*.29,-.43),(.18+j*.29,-.857),(.18+j*.05,-.615))
                else: e(f,.18+j*.288,-.429,.126,.224,j*14)
                e(k,.18+j*.126,-.444,.044,.037); l(h,.076,(.08+j*.13,.11),(.14+j*.14,.62),(.24+j*.15,.64))
            e(k,.213,-.311,.066,.043)
            if cat: l(f,.064,(-.51,.45),(-.68,.52),(-.79,.33),(-.77,.08))
    elif c in (4,7): # ungulates: brow, ears, muzzle, neck, withers, hocks and hooves
        horse=c==7
        if pose in (0,1,2,3):
            grazing=pose==2
            for j,(ax,kx,hx) in enumerate([(-.45,-.57,-.49),(.21,.16,.24),(-.53,-.605,-.70),(.31,.335,.32)]):
                slide=([-.16,.24,-.28,.25][j] if pose==1 else 0)
                rr=(.047 if horse else .033)*(1.10 if j>=2 else .92)
                l(s if j<2 else b,rr*1.22,(ax,-.008),(kx+slide*.38,.31)); l(s if j<2 else b,rr,(kx+slide*.38,.31),(hx+slide,.663)); e(d,hx+slide+.019,.682,rr*1.25,rr*.75)
            if horse:
                l(f,.073,(-.56,-.24),(-.715,-.08),(-.76,.152),(-.80,.32)); e(f,-.771,.16,.083,.239,14)
            else: p(h,(-.47,-.22),(-.77,-.35),(-.68,-.083),(-.57,-.044))
            e(b,-.125,-.111,.583,.253,-3); e(b,-.466,-.10,.256,.265); e(b,.267,-.121,.207,.257)
            if grazing:
                l(b,.142,(.262,-.21),(.49,-.106),(.66,.234)); e(h,.731,.308,.135,.236,-30); e(d,.815,.476,.098,.084); p(b,(.61,.19),(.536,.019),(.704,.134))
            else:
                p(b,(.10,-.24),(.26,-.40),(.405,-.681),(.564,-.714),(.592,-.57),(.432,-.075),(.30,.061))
                if horse: l(f,.047,(.187,-.336),(.322,-.526),(.433,-.677),(.515,-.683))
                e(b,.546,-.648,.153,.139,19); e(h,.666,-.585,.211,.102,31 if horse else 15); e(s,.810,-.508 if horse else -.553,.081,.077); e(k,.602,-.684,.023,.023)
                p(s,(.441,-.67),(.345,-.875),(.510,-.734)); p(b,(.51,-.718),(.525,-.907),(.58,-.723))
                if not horse and pose==3:
                    for j in (-.05,.054): l(s,.022,(.54+j,-.736),(.419+j,-.911),(.291+j,-.996),(.28+j,-1.094)); l(s,.016,(.398+j,-.91),(.424+j,-1.05))
        else:
            e(s,-.09,.90,.546,.81,-7); e(b,0,.17,.295,.62,-13)
            for j in (-1,1):
                if horse: e(b,.26*j,-.68,.096,.244,j*16)
                else: e(b,.36*j,-.49,.126,.277,j*-54); e(f,.388*j,-.502,.064,.172,j*-54)
            e(b,0,-.29,.325,.373); e(h,.025,-.007,.216,.37,-6); e(s,.058,.217,.195,.123)
            for j in (-1,1): e(k,j*.208,-.329,.049,.038); e(k,.053+j*.085,.223,.032,.022)
            if horse: l(f,.062,(-.266,-.42),(-.306,-.15),(-.28,.21)); e(f,0,-.54,.133,.149)
    elif c==6:
        if pose in (0,1,2): # crouching frog in profile
            for j in (-.33,.29): l(s,.056,(j,.075),(j+.181,.292),(j+.041,.365),(j+.37,.348))
            e(b,-.085,-.012,.593,.302,-14); e(h,.31,-.162,.411,.244,-12); e(b,.536,-.121,.254,.151,-6)
            e(s,-.369,.094,.341,.158,-29); l(h,.066,(-.56,.235),(-.273,.25),(-.014,.112)); l(s,.044,(.25,.015),(.428,.253),(.65,.274))
            e(h,.417,-.325,.148,.132); e(k,.461,-.341,.072,.063); e(w,.478,-.362,.024,.022); l(s,.014,(.37,-.055),(.59,-.062),(.75,-.12))
            if pose==2:
                for j in range(4): e(f,-.44+j*.153,-.143-j*.017,.061,.035,-20)
        else: # head-on and dorsal views with bulging eyes and folded rear legs
            for j in (-1,1):
                e(s,j*.474,.183,.25,.43,j*-31); l(b,.068,(j*.43,-.017),(j*.63,.26),(j*.49,.414),(j*.77,.498))
            e(b,0,.114,.50,.48); e(h,0,-.19,.55,.31)
            for j in (-1,1): e(b,j*.361,-.391,.179,.175); e(k,j*.355,-.411,.097,.087); e(h,j*.365,-.453,.044,.022)
            l(s,.023,(-.441,-.122),(-.22,-.065),(0,-.053),(.25,-.077),(.44,-.13))
    elif c==8:
        if pose in (0,1,2,3):
            p(s,(-.97,.093),(.90,.06),(.99,-.043),(.865,.223),(.66,.296),(-.70,.298)); l(h,.023,(-.885,.105),(.92,.075))
            if pose==2:
                l(d,.015,(-.084,.08),(-.09,-.88)); p(w,(-.122,-.84),(-.151,.008),(-.74,.020)); p(h,(-.055,-.75),(.567,-.017),(-.054,-.004))
            elif pose==0:
                p(h,(-.73,.069),(-.72,-.089),(-.56,-.092),(-.54,-.205),(-.23,-.208),(-.205,-.045),(.55,-.038),(.62,.041)); p(f,(-.48,-.18),(-.49,-.345),(-.38,-.348),(-.35,-.18))
                l(s,.022,(-.535,-.145),(-.246,-.148)); l(d,.012,(.48,.026),(.48,-.265))
            elif pose==1:
                for level in range(3):
                    u=level*.105; p(h,(-.77+u,.075-u*1.04),(-.73+u,-.063-u*1.04),(.53-u,-.064-u*1.04),(.65-u,.046-u*1.04)); l(g,.027,(-.68+u,-.017-u*1.04),(.49-u,-.019-u*1.04))
                p(f,(-.35,-.253),(-.38,-.384),(-.25,-.384),(-.19,-.252))
            else: p(h,(-.46,.073),(-.34,-.10),(.07,-.157),(.33,.034),(.57,.062)); p(g,(-.304,-.069),(-.27,-.09),(.04,-.122),(.24,.002))
        else: # bow facing camera; bridge and decks, or peaked sail
            p(s,(-.57,.063),(-.36,.282),(.011,.481),(.37,.274),(.565,.064),(.21,-.035),(-.24,-.039)); p(b,(.011,.48),(.37,.274),(.565,.064),(.04,.118))
            if pose==4:
                p(h,(-.434,.055),(-.414,-.219),(-.30,-.244),(-.27,-.344),(.285,-.344),(.307,-.245),(.409,-.217),(.433,.055)); l(g,.036,(-.32,-.157),(.32,-.157)); l(d,.013,(0,-.34),(0,-.644))
            else: p(w,(-.33,-.011),(.095,-.82),(.191,-.003)); l(s,.014,(.088,-.822),(.095,.073))
    elif c==9:
        if pose in (0,1,2): # box van / articulated lorry / dump truck, side view
            p(k,(-.90,.135),(.88,.12),(.93,.286),(-.89,.29)); top=-.465 if pose!=2 else -.31
            p(b,(-.97,top),(.23,top),(.23,.159),(-.97,.159)); l(h,.022,(-.94,top+.023),(.202,top+.023)); l(s,.028,(-.949,.108),(.21,.108))
            p(f,(.265,-.249),(.36,-.368),(.679,-.347),(.784,-.112),(.939,-.054),(.965,.193),(.86,.245),(.268,.209)); p(g,(.545,-.294),(.661,-.29),(.741,-.118),(.547,-.118)); p(g,(.327,-.288),(.484,-.29),(.484,-.118),(.329,-.119))
            if pose==1: wheel(-.458,.235,.127,.141); wheel(.309,.224,.123,.135)
            wheel(-.751,.233,.143,.147); wheel(.741,.218,.145,.152); e(w,.914,-.005,.04,.029)
        elif pose in (3,4): # truck three-quarter view, tall cargo box + cab
            p(s,(-.92,-.494),(-.049,-.622),(.23,-.503),(.22,.124),(-.02,.238),(-.92,.058)); p(b,(-.92,-.494),(-.049,-.622),(-.023,.208),(-.92,.058)); l(h,.021,(-.919,-.486),(-.056,-.609))
            p(f,(-.123,-.336),(.146,-.445),(.597,-.372),(.728,-.098),(.884,-.036),(.908,.286),(.417,.422),(-.126,.239)); p(g,(.223,-.352),(.561,-.30),(.669,-.087),(.267,-.116)); p(g,(-.044,-.268),(.146,-.34),(.189,-.115),(-.038,-.048))
            p(s,(.395,.047),(.833,-.013),(.85,.234),(.412,.346)); l(k,.034,(.46,.184),(.784,.114)); wheel(-.746,.129,.105,.138); wheel(.188,.273,.113,.174)
        else:
            for j in (-1,1): e(k,j*.447,.53,.107,.187)
            p(b,(-.592,-.634),(.585,-.634),(.602,.348),(.496,.438),(-.51,.436),(-.61,.337)); p(h,(-.532,-.398),(.527,-.398),(.552,.245),(-.55,.245)); p(g,(-.49,-.298),(.48,-.298),(.514,-.003),(-.511,-.003)); p(f,(-.553,.041),(.55,.04),(.55,.387),(-.55,.387)); p(d,(-.30,.126),(.294,.126),(.292,.342),(-.302,.342)); l(s,.041,(-.557,.429),(.553,.429))
            for j in (-1,1): e(w,j*.434,.247,.082,.051)
    return im, mask

# Irrational-spacing sequences cover shifts, foreshortening, scale, lighting,
# rotation, and reflection. Their values do not depend on the rendered banks.
def buildbank():
    banks=[]
    gy,gx=np.mgrid[:32,:32].astype(np.float32); gx=(gx-15.5)/16; gy=(gy-15.5)/16
    for c,nposes in enumerate([5,6,7,7,6,7,5,6,6,6]):
        current=[]
        for pose in range(nposes):
            for start in range(0,256,32):
                t=(np.arange(start,start+32,dtype=np.float64)+1+pose*263+c*1789)[:,None,None]
                def u(prime): return np.mod(t*np.sqrt(prime),1).astype(np.float32)
                side = .62+.66*u(2)
                if c in (3,5) and pose<4: side=.78+.66*u(2)
                if c in (2,6): side=.55+.77*u(2)
                if c in (4,7) and pose>=4: side=.73+.91*u(2)
                sy=side*(.72+.64*u(3)); sx=side*(.86+.28*u(5))
                dx=.34*(u(7)-.5); dy=.35*(u(11)-.5)
                if c in (4,7) and pose>=4 or c in (3,5) and pose<4: dy-=.08
                angle={0:.83,1:.25,2:.68,3:.34,4:.21,5:.37,6:.57,7:.20,8:.20,9:.19}[c]*(u(13)*2-1)
                if c==0 and pose in (2,3): angle*=3.7
                if c==2 and pose>=4: angle*=2
                co=np.cos(angle); si=np.sin(angle)
                x=((gx-dx)*co+(gy-dy)*si)/sx*np.where(u(17)<.5,-1,1).astype(np.float32)
                y=((gy-dy)*co-(gx-dx)*si)/sy
                body=.16+.67*u(19); back=np.where(u(23)<.5,body*(.16+.53*u(29)),body+(.38+.55*u(29))*(1-body))
                glass=.14+.39*u(31); fur=np.where(u(37)<.66,.42*body+.025,.57*body+.40)
                tones=[body,.76*body+.22,.70*body+.025,.045+.08*u(41),.11+.10*u(43),.81+.14*u(47),glass,fur,back]
                im,mask=draw(c,pose,x,y,tones)
                feature=helpers['descriptor'](im,mask)
                current.append(feature)
        banks.append(np.concatenate(current))
    splits=np.cumsum([0]+[len(a) for a in banks]).astype(np.int32)
    # The descriptor scales both components for dot products; store unit
    # templates per component, applying mixture coefficients on the query only.
    allbank=np.concatenate(banks); allbank[:,:384]/=.40; allbank[:,384:640]/=.44; allbank[:,640:]/=.16
    return np.ascontiguousarray(allbank.T),splits

inference = '''
def sat(x):
    return np.clip(x,0,1)

def zlim(x):
    return np.clip(x,-1.5,1.5)

def avg(a):
    return a.mean(axis=(-2,-1))

def shifted(a, dy, dx):
    hh,ww=a.shape[-2:]
    iy=np.clip(np.arange(hh)+dy,0,hh-1)
    ix=np.clip(np.arange(ww)+dx,0,ww-1)
    return a[:,iy[:,None],ix]

def saliency(x):
    pic=pool(x,2)
    pic=.72*pic+.28*blur(pic,1)
    yy,xx=np.mgrid[0:16,0:16].astype(np.float32)
    best=np.full((len(x),16,16),2,np.float32)
    for yy0,xx0 in [(i,j) for i in (1,4,7,10,13,15) for j in (0,15)]+[(i,j) for i in (0,15) for j in (2,5,8,11,14)]:
        dif=pic-pic[:,:,yy0:yy0+1,xx0:xx0+1]
        light=dif.mean(axis=1)
        dv=np.mean(abs(dif-light[:,None]),axis=1)+.42*abs(light)
        space=np.minimum(.053,.0038*abs(yy-yy0)+.0007*abs(xx-xx0))
        best=np.minimum(best,dv+space)
    m=sat((best-.040)/.135)
    return .66*m+.34*blur(m,1)

def smallparts(lum):
    # Search for pairs of dark, roughly round tires with lighter surrounds.
    # A light hub is optional. The tire top and both lateral arcs must agree.
    wh=np.zeros(len(lum),np.float32)
    for r in (2,3,4):
        sm=.55*lum+.45*blur(lum,1)
        outside=[]; inside=[]
        for dy,dx in [(-r,0),(0,-r),(0,r),(-r+1,-r+1),(-r+1,r-1)]:
            outside.append(shifted(sm,int(round(dy*1.45)),int(round(dx*1.45))))
            inside.append(shifted(sm,int(round(dy*.77)),int(round(dx*.77))))
        delta=[a-b for a,b in zip(outside,inside)]
        arcs=sum(delta)/5
        ring=sum(inside)/5
        light=sat((arcs-.018)/.17)*.54+sat((np.minimum(np.minimum(delta[0],delta[1]),delta[2])+.015)/.13)*.28+sat((sm-ring)/.15)*.18
        peak=np.zeros(len(lum),np.float32)
        for gap in (r*4,r*6,r*8):
            if gap>25: continue
            for skew in (-1,1):
                a=light[:,14:28,1:31-gap]; b=light[:,14+skew:28+skew,gap+1:31]
                same=sat(1-abs(ring[:,14:28,1:31-gap]-ring[:,14+skew:28+skew,gap+1:31])/.32)
                score=(np.minimum(a,b)*.70+.30*(a+b)*.5)*(.7+.3*same)
                peak=np.maximum(peak,score.max((-2,-1)))
        wh=np.maximum(wh,peak)
    # Eyes above a muzzle: look for three appropriately spaced dark features,
    # not just any single dark spot. Test several possible face sizes.
    eyes=np.zeros(len(lum),np.float32); muzzle=eyes.copy()
    for r in (2,3,4,5):
        im=.72*lum+.28*blur(lum,1)
        spot=(blur(im,max(2,r//2+1))-im)
        spot=sat((spot-.01)/.18)
        a=shifted(spot,0,-r); b=shifted(spot,0,r)
        mid=shifted(im,0,0)-.5*(shifted(im,0,r)+shifted(im,0,-r))
        pair=np.minimum(a,b)*sat(.58+mid/.36)
        nose=shifted(spot,int(round(1.55*r)),0)
        cheek=.5*(shifted(im,r,-r)+shifted(im,r,r))-shifted(im,int(round(1.55*r)),0)
        longnose=pair*(.4*sat((cheek-.015)/.23)+.6*nose)
        eyes=np.maximum(eyes,pair[:,5:20,10:22].max((-2,-1)))
        muzzle=np.maximum(muzzle,longnose[:,5:20,10:22].max((-2,-1)))
    return wh,eyes,muzzle

def scene(x,lum,mask):
    r,g,b=x[:,0],x[:,1],x[:,2]
    # Soft color memberships use opponent channels; avoid literal RGB matches.
    val=lum
    chrom=x.max(1)-x.min(1)
    green=sat((g-b-.014)/.115)*sat((g-r+.040)/.092)
    leaf=sat((g-r+.015)/.102)*sat((g-b+.002)/.107)
    blue=sat((b-r-.015)/.139)*sat((b-g+.035)/.112)
    warm=sat((r-b-.032)/.153)*sat((r-g+.015)/.108)
    ochre=sat((r-b-.04)/.155)*sat((r-g+.013)/.075)*sat((g-b+.011)/.115)
    neutral=sat(1-(chrom-.028)/.165)
    gray=neutral*sat((val-.13)/.22)*sat((1.02-val)/.22)
    bright=sat((val-.53)/.28)*sat((.23-chrom)/.17)
    paint=np.maximum.reduce([sat((r-g-.145)/.195)*sat((r-b-.12)/.17),sat((b-g-.10)/.17)*sat((b-r-.17)/.19),sat((np.minimum(r,g)-b-.23)/.18)*sat((val-.34)/.28)])
    sm=.62*lum+.38*blur(lum,1)
    dx,dy=deriv(sm); ux=abs(dx); uy=abs(dy)
    tex=ux+uy; local=blur(tex,2)
    smooth=sat((.087-local)/.070)
    sky=smooth*np.maximum(blue,bright*.86)
    vegetation=np.maximum(green,leaf)
    def top(a): return avg(a[:,:8,:])
    def bot(a): return avg(a[:,25:,:])
    def core(a): return avg(a[:,7:26,6:26])
    def sides(a): return .5*(avg(a[:,:,:5])+avg(a[:,:,-5:]))
    skyT=top(sky); skyB=bot(sky); blu=avg(blue); bt=top(blue); bb=bot(blue)
    veg=avg(vegetation); gt=top(vegetation); gb=bot(vegetation); gc=core(vegetation)
    warmC=core(warm); earth=core(ochre); color=core(paint); smoothB=bot(smooth); smoothT=top(smooth)
    road=bot(gray*(.47+.53*smooth))
    water=bot(blue*(.60+.4*smooth))
    # Vertical and horizontal edge agreement over a distance. Curved contours
    # and fur give less coherent gradients than a window, roof, hull or road.
    def coherence(a,axis):
        offs=3
        cum=[shifted(a,0,i) if axis==1 else shifted(a,i,0) for i in range(-offs,offs+1)]
        signed=abs(sum(cum)); unsigned=sum([abs(v) for v in cum]);
        return core(signed)/(.023+core(unsigned))
    hco=coherence(dy,1); vco=coherence(dx,0)
    hfrac=core(uy)/(.007+core(tex)); hline=zlim((hco-.49)*5.2); vline=zlim((vco-.49)*5.2)
    horizontal=zlim((hfrac-.47)*7)
    straight=.49*hline+.32*vline+.19*horizontal
    # A horizon produces a long cross-image edge at one height; the two
    # sides should agree, so a strong edge on just the object is not enough.
    band=blur(lum,1)
    jump=band[:,4:29]-band[:,1:26]
    jl=jump[:,:,:8].mean(-1); jr=jump[:,:,-8:].mean(-1)
    horizon=sat((np.max(np.where(jl*jr>0,np.minimum(abs(jl),abs(jr)),0),axis=1)-.027)/.169)
    # Smooth side borders, absence of a horizon and very similar sky colors
    # at top and bottom suggest an isolated object in the air.
    same=sat(1-abs(x[:,:,:6,:].mean((2,3))-x[:,:,26:,:].mean((2,3))).mean(1)/.31)
    air=top(smooth)*bot(smooth)*same*(1-.53*horizon)
    # Row-wise uniformity of the lower surface (road, pasture or water).
    sq=x[:,:,26:,:]; stripe=sat(1-np.mean(abs(sq-sq.mean(-1,keepdims=True)),(1,2,3))/.187)
    ground=gb*(.52+.48*stripe)
    waterlike=np.maximum(water,.56*bot(gray)*sat((.14-bot(warm))/.12)*(.50+.50*stripe))*stripe
    field=ground*(.5+.5*sat((gb-gc+.22)/.53))
    woods=gt*(.54+.46*sat(top(tex)/.10))
    camouflage=gc*(.56+.44*sat((gc-gb+.21)/.50))
    # Texture scale and layout, computed independently per picture.
    fine=avg(abs(lum-blur(lum,1)))
    coarse=avg(abs(lum-blur(lum,3)))
    grain=zlim((fine/(.008+coarse)-.39)*6.0)
    activity=zlim((core(tex)-.095)/.067)
    lower=avg(tex[:,22:30,6:26]); upper=avg(tex[:,3:11,6:26]); middle=avg(tex[:,11:22,3:29])
    strip=zlim((middle-.55*lower-.45*upper)/(.027+avg(tex)))
    leg=avg(ux[:,19:29,5:27])/(.010+lower)
    legs=zlim((leg-.44)*6)*(.48+.52*gb)
    # Central foreground geometry: compare spatial moments to those of a
    # uniform image and see how far the figure extends vertically.
    yq,xq=np.mgrid[0:16,0:16].astype(np.float32)
    mass=mask+.025; den=mass.sum((-2,-1)); cy=(mass*yq).sum((-2,-1))/den; cx=(mass*xq).sum((-2,-1))/den
    vx=(mass*(xq-cx[:,None,None])**2).sum((-2,-1))/den
    vy=(mass*(yq-cy[:,None,None])**2).sum((-2,-1))/den
    flat=zlim(np.log((vx+1)/(vy+1))*1.8); small=sat((.40-avg(mask))/.28)
    ins=(avg(mask[:,4:12,4:12])-avg(mask))
    isolated=sat((ins-.06)/.31)*small
    # Find bilateral symmetry in a possible central head or a front view.
    sym=np.zeros(len(x),np.float32)
    for center in (13,15,17,19):
        a=sm[:,5:25,center-8:center]; d=sm[:,5:25,center:center+8][:,:,::-1]
        a=a-a.mean((-2,-1),keepdims=True); d=d-d.mean((-2,-1),keepdims=True)
        sim=avg(a*d)/np.sqrt((avg(a*a)+.002)*(avg(d*d)+.002))
        sym=np.maximum(sym,sim)
    face=sat((sym-.31)/.50)
    wheels,eyes,nose=smallparts(lum)
    wheel=zlim((wheels-.39)*3.2); eye=sat((eyes-.49)/.43); nose=sat((nose-.32)/.49)
    # A pale overcast sky is useful even when the opponent-blue signal is absent.
    envsky=np.maximum(skyT,.52*top(bright)); room=sat((warmC+.34*core(neutral)-.10)/.72)*(1-veg)*(1-skyT)*(1-blu)
    forest=sat(veg*1.48); palette=sat((.07-x.std(axis=(-2,-1)).min(1))/.07)
    # Green core or earthy ground cover in an image with no sign of a skyline.
    soil=(.55*avg(ochre)+.45*gc)*(1-skyT)*(1-blu)
    dark=sat((.47-avg(lum))/.29); rich=sat((avg(chrom)-.105)/.14)
    cold=sat(blu*1.6); artificial=.80*color+.30*core(bright)
    contrast=sat((np.std(lum,axis=(-2,-1))-.15)/.13)
    # Scores are additive pieces of evidence, not calibrated probabilities.
    sc=np.empty((len(x),10),np.float32)
    sc[:,0]=1.68*envsky+.74*bt+.48*skyB+1.55*air+.50*isolated+.30*flat+.23*strip+.12*horizontal-1.35*forest-.93*warmC-.58*horizon-.38*activity-.30*wheel-.22*road-.94
    sc[:,1]=.64*straight+.29*horizontal+.72*wheel+1.22*road+.44*stripe+.48*strip+.37*flat+.58*artificial+.23*contrast-.83*forest-.65*air-.57*water-.34*envsky-.62
    sc[:,2]=.71*forest+.57*bt+.60*isolated+.62*air+.50*color+.29*rich+.22*small-.44*straight-.32*wheel-.29*horizon-.27*ground-.26*flat-.56
    sc[:,3]=1.02*room+.43*face+.54*eye+.35*grain+.25*activity+.31*core(neutral)-.33*straight-.69*envsky-.76*forest-.52*water-.31*ground-.24*flat-.46*color-.27
    sc[:,4]=1.18*forest+1.07*woods+.75*field+.69*earth+.37*legs+.30*isolated-.68*camouflage-.71*cold-.32*straight-.43*color-.27*face-.35*air-.70
    sc[:,5]=1.03*room+.51*nose+.32*face+.51*warmC+.21*contrast+.20*field-.30*straight-.36*envsky-.69*water-.33*camouflage-.33*flat-.24*color-.43
    sc[:,6]=1.49*camouflage+1.17*soil+.44*dark+.40*grain+.30*rich+.21*activity-.68*straight-1.42*cold-1.07*envsky-.42*isolated-.33*legs-.43*color-.39
    sc[:,7]=1.27*field+.64*forest+.74*legs+.51*warmC+.48*envsky+.32*horizon+.19*contrast-.73*camouflage-.46*woods-.40*water-.25*face-.27*air-.48
    sc[:,8]=1.98*waterlike+1.16*water+.71*horizon+.52*envsky+.43*horizontal+.29*hline+.46*stripe+.29*flat-.67*air-1.19*forest-.73*warmC-.40*wheel-.93
    sc[:,9]=.80*straight+.59*vline+.49*wheel+.93*road+.55*envsky+.48*artificial+.29*contrast+.21*strip+.20*horizon-.78*forest-.60*air-.48*water-.29*warmC-.58
    return sc

def batch(x):
    lum=.299*x[:,0]+.587*x[:,1]+.114*x[:,2]
    m=saliency(x)
    # A chromatic contour can be present even at constant luminance.
    opp=(x[:,0]-x[:,2])*.62+(x[:,1]-x[:,2])*.24
    dx,dy=deriv(.6*opp+.4*blur(opp,1))
    feat=descriptor(lum,m,(dx,dy,np.sqrt(dx*dx+dy*dy+.000001)))
    response=feat @ CONSTANTS["shapes"]
    cuts=CONSTANTS["cuts"]
    best=np.stack([response[:,cuts[k]:cuts[k+1]].max(1) for k in range(10)],axis=1)
    return (10.5*best+scene(x,lum,m)).argmax(axis=1)

def predict(x):
    labels=np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),32):
        n=min(32,len(x)-i)
        chunk=np.zeros((32,3,32,32),np.float32)
        chunk[:n]=x[i:i+n]
        labels[i:i+n]=batch(chunk)[:n]
    return labels
'''

if __name__=='__main__':
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    bank,cuts=buildbank()
    np.savez(out/'constants.npz',shapes=bank,cuts=cuts)
    (out/'solution.py').write_text(common+inference,encoding='utf-8')
