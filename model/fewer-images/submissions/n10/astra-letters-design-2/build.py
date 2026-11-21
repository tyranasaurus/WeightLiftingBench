"""Data-free ASL handshape matcher. All distances in the pose model are relative
anatomical units: the width of a palm is about two units. x goes toward the
thumb in an open palm, y toward the wrist, and z toward the camera.
"""
import sys, json, math
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

ALPHABET='ABCDEFGHIKLMNOPQRSTUVWXY'
NPOSE=512
NFRAME=4
SIDE=36
# Rays at equally spaced azimuths. abs(dot(ray, gradient)) is used for
# contrast-insensitive matching, positive halfwaves for relief/polarity.
ANGLES=np.arange(8)*math.pi/4
GX=np.cos(ANGLES).astype('f'); GY=np.sin(ANGLES).astype('f')
FINGERS=np.arange(4) # little, ring, middle, index

def hand(letter,rng):
    """Make a surface out of a palm pad, a long forearm and swept digits.
    Curled digits are explicitly bent *in depth*, not shortened straight rays.
    The thumb can be drawn in front of, alongside or beneath the other digits.
    """
    U=rng.uniform
    seg=[]; marks=[]
    def bone(a,b,r,s=None,aspect=(1.,1.,1.),tint=0.):
        seg.append((a,b,r,r if s is None else s,aspect,tint))
    def digit(points,radii,aspect=(1.,1.,1.),tint=0.):
        for j in range(len(points)-1):bone(points[j],points[j+1],radii[j],radii[j+1],aspect,tint)
    def landmark(p,r):marks.append((p,r))
    def portrait():
        # Secondary model: draw the visible phalanges directly in a palm-fixed
        # picture plane. In particular, a flexed fingertip and its MCP need
        # not coincide in projection. Coordinates are in knuckle-width units.
        little,ring,mid,idx=-.76,-.26,.22,.67
        ybase=np.array([-.61,-.83,-.92,-.83]);lxs=np.array([little,ring,mid,idx]);lens=[1.46,1.82,1.94,1.75];rs=[.195,.22,.23,.218]
        roots=[np.array([x,y,0.]) for x,y in zip(lxs,ybase)]
        bone((-.04,.67,-.03),(-.02,-.38,-.05),.70,.96,(1.,1.05,.54));
        wrist=(-.17,1.13,-.04);bone(wrist,(U(-3.2,-.45),5.3,-.36),.53,.70,(1.,1.,.78));landmark(wrist,.52)
        thumbpad=[(-.05,.74,.12),(.45,.45,.27),(.66,-.02,.22)];digit(thumbpad,[.34,.40,.36],aspect=(1.,1.,.75))
        # Straight or curved axes are drawn from the MCP toward the nail.
        straight={'B':{0:4,1:3,2:2,3:0},'D':{3:1},'F':{0:-12,1:7,2:35},'I':{0:-17},'K':{2:-18,3:5},'L':{3:-5},'R':{2:12,3:-13},'U':{2:4,3:3},'V':{2:-18,3:18},'W':{1:-14,2:2,3:22},'Y':{0:-22}}.get(letter,{})
        fan=U(.72,1.3);common=U(-9,7);side=U(.85,1.10)
        # In U the straight fingers touch along both shafts; R crosses one
        # over the other, so the ends can also appear nearly superimposed.
        if letter=='U':roots[3][0]-=.085
        if letter=='R':roots[3][0]-=.045
        for j,b in enumerate(roots):
            l=lens[j]*U(.91,1.07);radius=rs[j]*U(.91,1.09)
            if j in straight:
                angle=math.radians(straight[j]*fan+common+U(-2,2));u=np.array([math.sin(angle),-math.cos(angle),0.]);l*= U(.42,.74) if letter=='K' and j==2 else 1.
                p=[b,b+.46*l*u+[0,0,.035],b+.78*l*u+[.015,0,.07],b+l*u+[.025,0,.09]]
                if letter=='R' and j==3:p=[q+[0,0,.28] for q in p]
            else:
                # Default wrap: the first segment is aimed at the observer;
                # the next two fold back down toward the palm. Folded nails
                # form a diagonal stack for thumb-under-fingers M and N.
                high=U(.25,.43);shift=U(-.18,.04);drop=U(.30,.57)
                if letter in 'MN':
                    # For M, three folded fingers reach across the thumb;
                    # for N, only two lie over it. Lift and extend those
                    # distal phalanges relative to the tucked little finger.
                    trapped=j>=(1 if letter=='M' else 2)
                    shift=U(-.44,-.24) if trapped else U(-.14,.00)
                    high+=.23 if trapped else .025;drop+=.25 if trapped else .035
                if letter=='S':shift=U(-.22,-.06);high+=.13;drop+=.12
                if letter=='T' and j==3:high+=.20;shift-=.18
                if letter=='E':high+=.16;drop+=.24
                p=[b,b+[.025,-high,.34],b+[shift,-high+.14,.61],b+[shift+.01,-high+drop,.62]]
                if letter=='D':p=[b,b+[-.13,-.37,.42],b+[-.16,-.16,.69],b+[.03,.02,.76]]
                if letter=='X' and j==3:p=[b,b+[.11,-.80,.18],b+[.55,-1.06,.41],b+[.82,-.83,.61]]
                if letter=='F' and j==3:p=[b,b+[.50,-.43,.24],b+[.87,-.37,.39],b+[1.05,-.08,.51]]
            digit(p,np.array([1,.96,.85,.72])*radius,tint=U(-.065,.035))
            for q in p:landmark(q,radius*.88)
        base=(.48,.39,.27)
        if letter in 'LY':
            a=math.radians(U(9,46));v=np.array([math.cos(a),-math.sin(a),.02]);h=np.array(base);P=[h,h+.70*v,h+1.16*v,h+1.76*v]
        elif letter=='A':P=[base,(.89,-.08,.46),(1.05,-.59,.56),(1.04,U(-1.42,-.99),.54)]
        elif letter=='E':P=[base,(.62,-.01,.65),(.09,-.13,.83),(-.46,-.13,.83)]
        elif letter=='S':P=[base,(.78,-.14,.63),(.23,-.59,.91),(-.31,-.78,.91)]
        elif letter=='F':P=[base,(.90,.33,.47),(1.35,-.13,.50),(1.72,-.86,.50)]
        elif letter=='D':P=[base,(.57,-.07,.49),(.30,-.42,.76),(.10,-.83,.78)]
        elif letter=='K':P=[base,(.74,-.13,.52),(.42,-.78,.74),(.28,-1.44,.76)]
        elif letter in 'MNT':
            x={'M':-.52,'N':-.02,'T':.45}[letter];P=[base,(.65,-.06,.49),(x+.13,-.62,.48),(x,-1.35 if letter=='T' else -1.05,.53)]
        else:P=[base,(.57,-.08,.61),(-.08,-.40,.83),(-.58,-.68,.81)]
        thumb(P);yaw=U(-17,17);pitch=U(-12,14);roll=U(-19,18)
        if letter=='X':yaw=U(-15,29)
        if letter=='D':yaw=U(-30,23)
        if letter in 'KRU':yaw=U(-13,35)
        if letter in 'MNS':roll=U(-14,27)
        return seg,marks,(yaw,pitch,roll)

    # Straight phalanges have different lengths and rays fan away from the
    # base. Flexion angles are cumulative rotations in the sagittal plane.
    def make_finger(base,L,rad,az,bends,more_radius=1.,tip_z=0.):
        v=np.array(base,dtype=float);p=[v.copy()];az=math.radians(az)
        for frac,beta in zip([.47,.30,.23],bends):
            beta=math.radians(beta);v=v+L*frac*np.array([math.cos(beta)*math.sin(az),-math.cos(beta)*math.cos(az),math.sin(beta)])
            p.append(v.copy())
        p[-1][2]+=tip_z
        digit(p,rad*np.array([1.,.95,.83,.72])*more_radius,tint=U(-.035,.035))
        for q in p:landmark(q,rad*.85)
    def thumb(path,fac=1.):
        radii=np.linspace(.30,.155,len(path))*fac
        digit(path,radii,tint=U(-.03,.045))
        for q,r in zip(path,radii):landmark(q,r)

    if letter in 'CO' and U()<.72:
        # Opposition seen from the side forms a curved finger-and-thumb rim.
        # Parameterize it about the opening itself (rather than using joint
        # angles alone). O closes on the right; C leaves a mouth on that side.
        closed=letter=='O';rx=U(.57,.89);ry=U(.66,.94);tilt=math.radians(U(-22,9))
        origin=np.array([-.04,-.40,.02]);xr=np.array([math.cos(tilt),math.sin(tilt),0]);yr=np.array([-math.sin(tilt),math.cos(tilt),0]);zz=np.array([0,0,1.])
        def arc(deg,z=0,r=1):
            a=math.radians(deg);return origin+rx*r*math.cos(a)*xr+ry*r*math.sin(a)*yr+z*zz
        end=U(-16,5) if closed else U(-64,-35)
        wrist=arc(132)+[-.13,.82,-.03];bone(wrist,arc(180),.50,.49,(1.,1.16,.85));landmark(wrist,.47)
        bone(wrist,wrist+[U(-2.3,-.75),3.9,-.20],.43,.66,(1.,1.,.75))
        for j in range(4):
            d=3-j;start=177+2*j;stop=360+end-5*d;rr=1-.055*d;z=-.15*d
            p=[arc(t,z,rr) for t in np.linspace(start,stop,5)]
            digit(p,[.34,.31,.26,.219,.165],tint=U(-.06,.01));
            for q in p:landmark(q,.24)
        pend=end+U(-3,5) if closed else U(25,57)
        p=[arc(138,.09),arc(95,.16),arc(55,.23),arc(pend,.23)];thumb(p,1.14)
        yaw=U(-32,35);pitch=U(-24,22);roll=U(-25,18);return seg,marks,(yaw,pitch,roll)
    if letter=='X' and U()<.58:
        # Bent index seen on the radial side: PIP joint is the apex and the
        # distal phalanx curves toward, but stays above, the curled fist.
        wrist=(-.36,1.12,-.09);bone(wrist,(-.12,-.23,-.03),.5,.72,(1,1,.74));landmark(wrist,.52)
        bone(wrist,(U(-2.4,-.8),4.9,-.24),.44,.66,(1,1,.73))
        for j in range(3):
            b=np.array([-.01+.13*j,-.22-.12*j,-.13*j]);p=[b,b+[.45,-.14,.23],b+[.55,.09,.52],b+[.34,.38,.57]]
            digit(p,[.225,.22,.20,.16]);landmark(p[1],.20)
        p=[(-.45,-.42,.06),(-.43,-1.18,.16),(-.02,-1.63,.30),(.44,-1.62,.46),(.70,-1.36,.58)]
        digit(p,[.24,.22,.208,.187,.145],tint=.02)
        for q in p:landmark(q,.21)
        thumb([(.07,.36,.22),(.44,.12,.50),(.30,-.33,.65),(-.02,-.47,.74)])
        yaw=U(-29,30);pitch=U(-27,24);roll=U(-20,14);return seg,marks,(yaw,pitch,roll)
    if letter not in 'COGHPQ' and rng.random()<.54:return portrait()
    if letter in 'CO':
        # Radial-side view: heel of hand, a stack of bent fingers, opposing
        # thumb. Separate the fingertips from the thumb in C; touch in O.
        closed=letter=='O'; gap=U(.04,.24) if closed else U(.65,1.18)
        top=U(-1.50,-1.16); span=U(1.28,1.75); endx=-.50+span
        bone((-.80,.94,-.08),(-.64,-.65,0),.52,.55,(1.,1.12,.78))
        bone((-.83,1.11,0),(-2.65,5.,-.24),.41,.65,(1.,1.,.76))
        landmark((-.83,1.11,0),.43);landmark((-.65,-.66,0),.52)
        slope=U(-.14,.14);topy=top+U(.02,.25)+(U(.06,.36) if closed else 0)
        for j in range(4):
            d=(j-3); z=d*.14
            p=[(-.72+.025*d,-.60+.04*abs(d),z),(-.40+.04*d,top+.08*abs(d),z+.03),
               (endx-.48+.08*d,top-.09+.04*abs(d),z+.07),
               (endx+.085*d,topy+abs(d)*.025,z+.11)]
            digit(p,[.26,.24,.205,.16],tint=U(-.04,.04));landmark(p[-1],.17);landmark(p[2],.20)
        p=[(-.80,.78,.16),(-.25,.50,.25),(endx-.45,topy+gap+.39,.27),(endx+(U(-.16,.10) if closed else U(.0,.26)),topy+gap,.25)]
        if closed:
            # Flex the distal phalanges and oppose the thumb across the palm,
            # not alongside the index: these open the oval web-space in O.
            # Bend all four curled digits toward the thumb side of the ring.
            curved=[]
            for a,b,ra,rb,asp,tint in seg:
                def bend(v):
                    x,y,z=v
                    # lower the fingertips gradually along the horizontal arc
                    return x,y+.84*max(0.,(x+.42)/1.45)**1.65,z
                curved.append((bend(a),bend(b),ra,rb,asp,tint))
            seg[:]=curved;marks[:]=[(bend(a),r) for a,r in marks]
            p=[(-.80,.92,.16),(-.22,.78,.27),(endx-.14,.12,.31),bend((endx,topy+.08,.25))]
        thumb(p,1.05)
        yaw=U(-12,30);pitch=U(-13,13);roll=U(-20,14)
    elif letter=='T' and U()<.65:
        # T rotates at the wrist. Viewed in profile the hooked index covers
        # the thumb: its middle and proximal phalanges can point sideways.
        wrist=(-.69,.97,-.05);bone(wrist,(-1.96,4.7,-.22),.48,.67,(1.,1.,.8));landmark(wrist,.49)
        bone((-.63,.60,-.07),(-.30,-.50,-.09),.63,.79,(1.,1.,.72))
        for j in range(4):
            d=3-j; base=(-.54+.09*d,-.66+.29*d,-.15*d)
            u=U(1.43,1.86) if j==3 else 1.00-.16*d;b=np.array(base)
            p=[b,b+[.48*u,-.14,.17],b+[.78*u,.03,.36],b+[u,.11 if j==3 else .35,.40]]
            digit(p,[.22,.205,.187,.157]);landmark(p[-1],.17)
        p=[(-.18,.46,.27),(-.11,-.14,.46),(-.13,-.72,.53),(-.24,-1.23,.53)];thumb(p)
        yaw=U(-24,22);pitch=U(-22,20);roll=U(-23,18)
    elif letter in 'GHPQ':
        # Profile and downward poses. Construct a local pad running from wrist
        # to the index MCP; the forearm need not be collinear with the palm.
        if letter in 'GH':
            ang=math.radians(U(-18,13));along=np.array([math.cos(ang),math.sin(ang),0.]);down=np.array([-math.sin(ang),math.cos(ang),0.]);z=np.array([0.,0.,1.])
            base=np.array([0.,-.34,0.]); wrist=base-1.54*along+.52*down
            def pt(a,b,c=0):return base+a*along+b*down+c*z
            bone(wrist,pt(-.13,.30),.54,.64,(1,1,.82));landmark(wrist,.51)
            armD=U(.23,1.0); bone(wrist,wrist+[-4.,armD*4.,-.20],.44,.65,(1,1,.8))
            # knuckles stack downwards across the edge-on palm
            for j in range(4):
                d=3-j; start=pt(-.16*d,.175*d,-.13*d+.20)
                ext=j==3 or (j==2 and letter=='H')
                if ext:
                    spread=0 if j==3 else U(.10,.34);L=U(1.62,1.92)*(1.07 if j==2 else 1.)
                    path=[start,pt(.46*L-.13*d,.16*d+spread*.32,.18-.13*d),pt(.76*L-.13*d,.16*d+spread*.70,.18-.13*d),pt(L-.13*d,.16*d+spread,.16-.13*d)]
                else:
                    path=[start,start+.43*along+.05*down,start+.34*along+.39*down+.27*z,start+.02*along+.42*down+.3*z]
                digit(path,[.215,.198,.18,.148],tint=U(-.05,.05));landmark(path[-1],.16);landmark(path[1],.19)
            # ASL G's thumb is parallel to the index or folded against its base;
            # H has the thumb tucked underneath the two extended digits.
            alone=letter=='G' and U()<.4
            p=[pt(-1.03,.33,.35),pt(-.58,-.12,.48),pt(.15 if alone else -.08,-.16 if alone else .50,.50),pt(.90 if alone else .11,-.15 if alone else .67,.32)]
            thumb(p,.85);yaw=U(-22,20);pitch=U(-22,20);roll=U(-11,11)
        elif letter=='P':
            # A turned-down K. Index points out, middle points down; thumb
            # lies between them. View can look along either extended finger.
            wrist=np.array([1.18,.86,0]);base=np.array([.04,-.12,.0]);
            bone(wrist,base,.47,.64,(1,1,.78));bone(wrist,(4.2,3.7,-.27),.42,.64,(1,1,.80));landmark(wrist,.48)
            a=np.array([-1.,U(-.55,.02),U(-.32,.32)]);b=np.array([U(-.80,-.23),1.,U(-.30,.65)]);
            for j in range(4):
                start=base+np.array([.10*(3-j),.15*(3-j),-.11*(3-j)]);
                if j>=2:
                    v=(a if j==3 else b)*U(1.65,1.94);path=[start,start+.48*v,start+.76*v,start+v]
                else:path=[start,start+[-.45,.06,.12],start+[-.55,.4,.3],start+[-.19,.52,.32]]
                digit(path,[.215,.20,.175,.148],tint=U(-.04,.04));landmark(path[-1],.16)
            thumb([(1.,.50,.15),(.61,.09,.39),(.01,.06,.54),(-.39,.22,.48)],.95)
            yaw=U(-25,24);pitch=U(-26,27);roll=U(-19,19)
        else:
            # Q is a downward G: index and opposing thumb point toward floor,
            # back of hand arches from wrist to the bent set of knuckles.
            wrist=np.array([.92,.79,-.07]);base=np.array([-.57,-.45,0]);
            bone(wrist,base,.51,.64,(1.,1.,.85));bone(wrist,(1.82,4.8,-.30),.43,.62,(1.,1.,.82));landmark(wrist,.45)
            for j in range(4):
                start=base+np.array([.16*(3-j),.09*(3-j),-.16*(3-j)]);
                if j==3:
                    v=np.array([U(-1.10,-.33),U(1.35,1.84),U(-.35,.30)]);path=[start,start+.46*v,start+.74*v,start+v]
                else:path=[start,start+[-.20,.38,.02],start+[.00,.64,.21],start+[.20,.39,.31]]
                digit(path,[.215,.20,.18,.15]);landmark(path[-1],.16)
            p=[(.58,.33,.32),(.20,.20,.47),(-.07,.70,.55),(-.34,1.24,.45)]
            thumb(p,.96);yaw=U(-27,34);pitch=U(-22,22);roll=U(-23,21)
    else:
        # Frontal palm: smoothly interpolated heel-to-knuckle dimensions.
        # Long axis y, width x and thickness z. Radius in the horizontal
        # plane of the palm is just under half of palm breadth.
        bone((-.06,.54,-.01),(0,-.36,-.015),.77,1.0,(1.,.86,.43))
        wrist=(-.09,1.03,-.06);sweep=U(-2.8,.22)
        bone(wrist,(sweep,5.1,-.29),.53,.73,(1.,1.,.70))
        landmark(wrist,.54);landmark((0,-.34,0),.92)
        # Thenar pad, joined smoothly to the base of the thumb.
        bone((.05,.58,.19),(.56,.08,.22),.36,.38,(1.,1.,.62))
        xbase=np.array([-.72,-.28,.19,.64])*U(.93,1.03)
        ybase=np.array([-.55,-.84,-.93,-.82])
        lengths=np.array([1.31,1.75,1.92,1.70])*U(.93,1.08)
        radii=np.array([.191,.213,.231,.218])*U(.95,1.07)
        # Four finger spread angles for each upright open handshape.
        opened={'B':([0,1,2,3],[-3,-1,1,2]),'D':([3],[5]),'F':([0,1,2],[-24,0,22]),
        'I':([0],[-15]),'K':([2,3],[-16,11]),'L':([3],[-2]),'R':([2,3],[12,-12]),
        'U':([2,3],[0,1]),'V':([2,3],[-15,15]),'W':([1,2,3],[-17,0,17]),'Y':([0],[-22])}
        if letter in opened:ids,dirs=opened[letter]
        else:ids,dirs=[],[]
        # Hand orientation commands. A side view foreshortens the palm and
        # exposes a different amount of each overlapping distal phalanx.
        yaw=U(-19,29);pitch=U(-15,18);roll=U(-15,15)
        if letter in 'DRUK':yaw=U(5,53)
        if letter=='X':yaw=U(47,78);roll=U(-14,18)
        if letter=='M':yaw=U(-36,19);roll=U(-5,26)
        if letter=='N':yaw=U(-30,25);roll=U(-18,20)
        if letter=='S':yaw=U(-25,28);roll=U(-10,25)
        if letter=='T':yaw=U(-23,30)
        if letter=='F':yaw=U(-18,21)
        # Fingers can bunch, separate, or bend a little in projection. For K,
        # the middle finger is on the flexed side of the thumb.
        fan=U(.74,1.30);slant=U(-8,8);flex=U(-8,12)
        for j in range(4):
            base=(xbase[j],ybase[j],.02);rad=radii[j];L=lengths[j]*U(.95,1.05)
            if j in ids:
                az=dirs[ids.index(j)]*fan+slant+U(-2,2);f=flex+U(-7,7)
                if letter=='K' and j==2:f+=U(23,58)
                if letter=='R' and j==3:f+=U(9,24)
                if letter=='B':az=dirs[ids.index(j)]+slant
                if letter=='U':az=dirs[ids.index(j)]+slant+U(-1.2,1.2)
                make_finger(base,L,rad,az,[f,f+U(-5,9),f+U(-5,13)])
            elif letter=='X' and j==3:
                make_finger(base,L,rad,U(-8,8),[U(2,26),U(58,83),U(99,146)])
            elif (letter=='F' and j==3) or (letter=='D' and j==2):
                # Index-to-thumb circle in F, middle-to-thumb circle in D.
                ang=U(25,49) if letter=='F' else U(20,46)
                make_finger(base,L,rad,ang,[U(43,65),U(93,126),U(138,163)])
            else:
                # Fingers in M/N curve over the thumb, while in E the distal
                # finger pads sit up against it and can be seen from the front.
                f=(U(32,49) if letter=='E' else U(42,65))
                az=U(-17,2)+(U(-17,-3) if letter in 'MN' else 0)
                if letter=='T' and j==3:f-=U(17,31)
                if letter=='D':f+=U(-8,8)
                make_finger(base,L,rad,az,[f,U(100,133),U(163,197)])
        tbase=(.51,.38,.23)
        if letter in 'LY':
            a=math.radians(U(7,37));q=np.array([math.cos(a),-math.sin(a),.02]);b=np.array(tbase);p=[b,b+.61*q+[0,-.03,.12],b+1.09*q,b+1.52*q];
        elif letter=='A':p=[tbase,(.89,-.04,.40),(1.,-.60,.39),(.94,-.99,.36)]
        elif letter in 'DF':
            reach=.96 if letter=='F' else .51;p=[tbase,(.94,-.05,.44),(reach+.25,-.40,.65),(reach,-.78,1.00)]
        elif letter in 'MTN':
            j={'M':0,'N':1,'T':2}[letter];xend=(xbase[j]+xbase[j+1])/2
            p=[tbase,(.65,-.09,.69),(xend+.12,-.66,.83),(xend,-1.15,.86)]
        elif letter=='K':p=[tbase,(.78,-.06,.54),(.52,-.74,.82),(.33,-1.30,.79)]
        elif letter=='E':p=[tbase,(.68,-.03,.56),(.12,-.08,.83),(-.41,-.16,.87)]
        elif letter=='S':p=[tbase,(.75,-.24,.65),(.27,-.63,1.07),(-.30,-.78,1.05)]
        elif letter in 'IVWURB':p=[tbase,(.69,-.09,.56),(.03,-.40,.80),(-.57,-.59,.86)]
        else:p=[tbase,(.69,-.12,.58),(.16,-.44,.84),(-.24,-.55,.85)]
        perturb=U(-.11,.11,(len(p),3));p=np.asarray(p)+perturb
        thumb(p)
    return seg,marks,(yaw,pitch,roll)


def render(letter,rng):
    seg,marks,(yaw,pitch,roll)=hand(letter,rng)
    yaw,pitch,roll=np.deg2rad([yaw,pitch,roll]);c,s=np.cos,np.sin
    rot=np.array([[c(roll),-s(roll),0],[s(roll),c(roll),0],[0,0,1]])@np.array([[1,0,0],[0,c(pitch),-s(pitch)],[0,s(pitch),c(pitch)]])@np.array([[c(yaw),0,s(yaw)],[0,1,0],[-s(yaw),0,c(yaw)]])
    rot=rot@np.diag(rng.uniform([.89,.94,.90],[1.10,1.07,1.11]))
    # Camera frames the whole physical hand (not the forearm). Bounds here
    # are computed from this one model's 3D landmarks, never from photographs.
    pp=np.array([p for p,r in marks])@rot.T;rr=np.array([r for p,r in marks])[:,None]
    lo=(pp[:,:2]-rr).min(0);hi=(pp[:,:2]+rr).max(0)
    # Most tight crops include some wrist below the lowest knuckle or heel.
    slack=rng.uniform(.24,1.04) if letter not in 'GHPQ' else rng.uniform(.1,.72)
    hi[1]+=slack
    wh=hi-lo;unit=max(wh)/rng.uniform(21.5,31.5);origin=(lo+hi)*.5-(SIDE-1)*.5*unit
    depth=np.full((SIDE,SIDE),-60.,dtype=float);pigment=np.zeros_like(depth)
    for a,b,ra,rb,aspect,tint in seg:
        # Projection of a swept ellipsoid. z is its front intersection with
        # an orthographic camera ray. Use a z-buffer for phalanx occlusion.
        a=rot@a;b=rot@b;v=b-a;A=rot@np.diag(aspect);C=A@A.T
        g=C[:2,:2];det=g[0,0]*g[1,1]-g[0,1]**2
        i00=g[1,1]/det;i01=-g[0,1]/det;i11=g[0,0]/det
        zx=C[2,0]*i00+C[2,1]*i01;zy=C[2,0]*i01+C[2,1]*i11
        zz=math.sqrt(max(.0001,C[2,2]-zx*C[0,2]-zy*C[1,2]))
        R=np.sqrt(np.diag(g))*max(ra,rb)
        low=np.maximum(0,np.floor((np.minimum(a[:2],b[:2])-R-origin)/unit).astype(int));high=np.minimum(SIDE,np.ceil((np.maximum(a[:2],b[:2])+R-origin)/unit).astype(int)+1)
        x0,y0=low;x1,y1=high
        if x0>=x1 or y0>=y1:continue
        x=(np.arange(x0,x1)*unit+origin[0]-a[0])[None,:];y=(np.arange(y0,y1)*unit+origin[1]-a[1])[:,None]
        vx,vy,vz=v;ux=i00*vx+i01*vy;uy=i01*vx+i11*vy
        vv=vx*ux+vy*uy;dot=ux*x+uy*y;dr=rb-ra
        # Tapered capsule: intrinsic plane-distance and depth along bone.
        t=np.clip((dot+ra*dr)/max(.0001,vv-dr*dr),0,1)
        # When an end points straight at the viewer the distal sphere, not the
        # line's closest projected point, can be the visible surface.
        def sphere(t):
            dx=x-t*vx;dy=y-t*vy;rad=ra+t*dr
            h=rad*rad-(i00*dx*dx+2*i01*dx*dy+i11*dy*dy)
            return np.where(h>0,a[2]+t*vz+zx*dx+zy*dy+zz*np.sqrt(np.maximum(0,h)),-60.)
        z=np.maximum(np.maximum(sphere(t),sphere(0)),sphere(1));old=depth[y0:y1,x0:x1]
        use=z>old;old[use]=z[use];pigment[y0:y1,x0:x1][use]=tint
    mask=depth>-50
    # Shaded relief: one directional illuminant, ambient light, and shallow
    # occlusion from surfaces protruding in front of the current surface.
    def neighbor(arr,dx,dy):return np.roll(arr,(dy,dx),axis=(0,1))
    def near(dx,dy):
        q=neighbor(depth,dx,dy)
        return np.where(q>-50,q,depth-1.20*unit)
    nx=np.clip((near(1,0)-near(-1,0))/(2*unit),-3.,3.);ny=np.clip((near(0,1)-near(0,-1))/(2*unit),-3.,3.)
    # Camera light tends to illuminate from above. Azimuth and ambient term
    # vary independently, rather than assuming a fixed silhouette polarity.
    light=rng.uniform(-1.2,.62);ly=rng.uniform(-.98,-.25);lz=rng.uniform(.48,1.22)
    shade=(light*nx+ly*ny+lz)/np.sqrt((1+nx*nx+ny*ny)*(light*light+ly*ly+lz*lz))
    cav=sum(np.minimum(1.,np.maximum(0,(near(dx,dy)-depth-.09)*1.8)) for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)])*.25
    # Self-shadowing from another digit lying between the surface and light.
    shadow=np.zeros_like(shade)
    for step in [1,2,3,5,8]:
        den=max(.1,abs(light),abs(ly));dx=int(round(step*light/den));dy=int(round(step*ly/den))
        blocker=neighbor(depth,-dx,-dy)-depth-unit*step*lz/den
        shadow=np.maximum(shadow,np.clip(blocker*1.9,0.,1.))
    skin=.41+rng.uniform(-.08,.09)+.27*shade+.039*np.where(mask,depth,0)-.16*cav-.23*shadow+pigment
    bg=rng.uniform(.38,.84)
    return np.where(mask,skin,bg).astype('f'),bg


def kernels():
    # Derivatives of an isotropic Gaussian, compact 3x3 Sobel-like stencil.
    d=np.array([-1.,0.,1.],dtype='f');s=np.array([1.,2.,1.],dtype='f')
    return np.array([(x*np.outer(s,d)+y*np.outer(d,s))/8 for x,y in zip(GX,GY)],dtype='f')
K=kernels()
# Local line and bar detectors. Transverse Gaussians of two opposite
# curvatures locate ridges even when one edge vanishes into the background.
_Y,_X=np.mgrid[-3:4,-3:4];bars=[]
for theta in ANGLES[:4]:
    across=_X*np.cos(theta)+_Y*np.sin(theta);along=-_X*np.sin(theta)+_Y*np.cos(theta)
    g=np.exp(-.5*((across/.92)**2+(along/2.0)**2));h=(1-(across/.92)**2)*g;h-=h.mean();h/=np.abs(h).sum()*.5;bars.append(h)
BARS=np.asarray(bars,dtype='f')
# The border and lower sleeve are downweighted; the underlying function is a
# smooth, category-independent spatial prior in crop coordinates.
cy,cx=np.mgrid[:8,:8];spatial=(1.-.18*((cx-3.5)/3.5)**4)*(1.-.33*(cy/7)**4);spatial=spatial.astype('f')

def gradmaps(batch):
    dx=((batch[:,0:-2,2:]-batch[:,0:-2,:-2])+2*(batch[:,1:-1,2:]-batch[:,1:-1,:-2])+(batch[:,2:,2:]-batch[:,2:,:-2]))/8
    dy=((batch[:,2:,0:-2]-batch[:,:-2,0:-2])+2*(batch[:,2:,1:-1]-batch[:,:-2,1:-1])+(batch[:,2:,2:]-batch[:,:-2,2:]))/8
    rays=np.maximum(0.,GX[None,:,None,None]*dx[:,None]+GY[None,:,None,None]*dy[:,None]);patches=windows(rays,(5,5),axis=(-2,-1))[:,:,::3,::3]
    co=.65*patches.max(axis=(-2,-1))+.35*patches.mean(axis=(-2,-1))
    # More selective directional tuning: suppress the component of the
    # orthogonal derivative, then pool in a smaller spatial neighborhood.
    q=np.abs(GX[None,:4,None,None]*dx[:,None]+GY[None,:4,None,None]*dy[:,None]);q=np.maximum(0,q-.72*np.roll(q,2,axis=1))
    fine=windows(q,(4,4),axis=(-2,-1))[:,:,::2,::2].mean(axis=(-2,-1))
    # Equalize the gain at strong fingers and weak fingers separately.
    fine=np.minimum(fine,.060)+.42*np.minimum(fine,.022)+.15*fine
    return co,fine

def descriptor(batch):
    # All centering and l2-normalization here are strictly *within one
    # template and one channel/group*. Never compare renders in the builder.
    r,f=gradmaps(batch);mag=r[:,:4]+r[:,4:]
    def center(q):
        delta=q-q.mean((-1,-2),keepdims=True)
        return delta/np.sqrt(np.mean(delta**2,axis=(-1,-2),keepdims=True)+1e-5)
    def unit(q):return q/(np.sqrt(np.sum(q*q,axis=(1,2,3),keepdims=True))+.00001)
    # Normalized signed orientations suppress global exposure changes. Use
    # unsigned orientations too, since skin can be darker or lighter than wall.
    def weights(q):
        q=center(q)*spatial
        return unit(q)*spatial
    ab=weights(mag);sg=weights(r)
    def weighted(q,prior):return (unit(center(q)*prior)*prior).reshape(len(q),-1).astype('f')
    def support(n):
        y,x=np.mgrid[:n,:n]/(n-1);return (1-.58*y**2)*(1-.22*(2*x-1)**4)
    fi=weighted(f,support(12))
    low=windows(batch,(4,4),axis=(1,2))[:,::2,::2].mean(axis=(-1,-2))
    # High-pass the DC/slow background variation in the luminance branch.
    # Center-surround features retain filled areas and holes at finger scale.
    small=windows(low,(5,5),axis=(1,2));dog=small[:,:, :,2,2]-small.mean(axis=(-1,-2))
    lo=weighted(low[:,None],support(13));dog=weighted(dog[:,None],support(9))
    filt=windows(batch,(7,7),axis=(1,2));ridge=(filt.reshape(len(filt),22*22,49)@BARS.reshape(4,49).T).transpose(0,2,1).reshape(len(batch),4,22,22)
    # Take the absolute ridge response, then maximize over a small
    # displacement to permit imperfect finger articulation.
    cells=windows(np.abs(ridge),(4,4),axis=(-2,-1))[:,:,::3,::3];rr=weighted(cells.max(axis=(-2,-1)),support(7))
    return ab.reshape(len(r),-1).astype('f'),sg.reshape(len(r),-1).astype('f'),fi,lo,dog,rr


def make(out):
    out.mkdir(parents=True,exist_ok=True);net=[];w={}
    def layer(name,op,inp,**kw):net.append(dict(name=name,op=op,inputs=inp if isinstance(inp,list) else [inp],**kw));return name
    def conv(name,inp,weights):
        weights=np.array(weights,dtype='f');w[name+'.W']=weights;w[name+'.b']=np.zeros(len(weights),dtype='f')
        return layer(name,'conv',inp,out=len(weights),kernel=list(weights.shape[-2:]),padding=0,stride=1)
    def dense(name,inp,weights):
        w[name+'.W']=weights;w[name+'.b']=np.zeros(len(weights),dtype='f');return layer(name,'dense',inp,out=len(weights))
    cv=conv('compass_derivative','image',K[:,None]);act=layer('half_waves','relu',cv)
    p=layer('nearby_peak','maxpool',act,size=[5,5],stride=[3,3]);m=layer('nearby_mean','avgpool',act,size=[5,5],stride=[3,3]);I=np.eye(8,dtype='f')[:,:,None,None]
    p=conv('scale_peak',p,.65*I);m=conv('scale_mean',m,.35*I);r=layer('local_edges','add',[p,m]);sg=layer('signed_directions','norm',r)
    mag=conv('axis_energy',r,I[:4]+I[4:]);ab=layer('unsigned_directions','norm',mag)
    axes=I[:4]+I[4:];select=conv('orientation_competition',act,axes-.72*np.roll(axes,2,axis=0));select=layer('narrow_axes','relu',select)
    f=layer('fine_pool','avgpool',select,size=[4,4],stride=[2,2]);eye=np.eye(4,dtype='f')[:,:,None,None];clipweights=np.tile(eye,(3,1,1,1));clips=conv('breakpoints',f,clipweights);w['breakpoints.b'][4:8]=-.060;w['breakpoints.b'][8:]=-.022
    clips=layer('above_breakpoints','relu',clips);shaper=np.concatenate([1.57*eye,-eye,-.42*eye],axis=1);f=conv('compress_contrast',clips,shaper);f=layer('fine_directions','norm',f)
    lo=layer('antialias_photo','avgpool','image',size=[4,4],stride=[2,2]);low=layer('luminance','norm',lo)
    surround=-np.ones((1,1,5,5),dtype='f')/25;surround[0,0,2,2]+=1.;dog=conv('center_surround',lo,surround);dog=layer('local_luminance','norm',dog)
    rr=conv('finger_bars','image',np.concatenate([BARS,-BARS])[:,None]);rr=layer('bar_halfwaves','relu',rr);rr=conv('bar_polarity_invariance',rr,axes);rr=layer('nearby_ridge','maxpool',rr,size=[4,4],stride=[3,3]);rr=layer('ridge_directions','norm',rr)
    count=len(ALPHABET)*NPOSE*NFRAME
    wa,ws,wf,wl,wd,wr=[np.empty((count,d),dtype='f') for d in (4*64,8*64,4*144,169,81,196)]
    rng=np.random.default_rng(970295);ix=0
    for letter in ALPHABET:
        views=[]
        for i in range(NPOSE):
            frame,bg=render(letter,rng)
            for j in range(NFRAME):
                # independent crop hypotheses, not estimated alignments
                v=np.array([0,0]) if j==0 else np.rint(rng.normal(0,1.75,2)).astype(int).clip(-4,4)
                x,y=4+v;views.append(frame[y:y+28,x:x+28])
            if len(views)>=40 or i+1==NPOSE:
                a,s,fi,l,d,ridge=descriptor(np.stack(views));end=ix+len(a)
                for weights,values,mix in [(wa,a,.15),(ws,s,.15),(wf,fi,.26),(wl,l,.10),(wd,d,.15),(wr,ridge,.19)]:
                    weights[ix:end]=values*(mix/math.sqrt(values.shape[1]))
                ix=end;views=[]
    a=dense('shape_agreement',ab,wa);s=dense('relief_agreement',sg,ws);e=dense('edge_alignment',f,wf);h=dense('hand_tone',low,wl);c=dense('cavity_and_highlight',dog,wd)
    bars=dense('match_finger_ridges',rr,wr);score=layer('agreement','add',[a,s,e,h,c,bars]);block=layer('by_letter','reshape',score,shape=[24,NPOSE,NFRAME]);best=layer('best_framing','maxpool',block,size=[1,NFRAME],stride=[1,NFRAME])
    # A small committee of independently drawn pose banks. Pool within each
    # group, then average maxima to reduce accidental single-template hits.
    group=NPOSE//4;best=layer('best_handshape','maxpool',best,size=[group,1],stride=[group,1]);votes=layer('bank_consensus','avgpool',best,size=[4,1],stride=[4,1]);layer('class_scores','reshape',votes,shape=[24])
    (out/'network.json').write_text(json.dumps({'layers':net},indent=1)+'\n');np.savez(out/'weights.npz',**w)

if __name__=='__main__':make(Path(sys.argv[1]))
