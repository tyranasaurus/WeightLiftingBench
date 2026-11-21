"""Build the detector from geometry, colour rules, and analytic portrait stencils."""
from pathlib import Path
import sys
import numpy as np


def desk_portraits():
    # Coordinates in units of the visible face width and height. There is no
    # source picture: each column is a separate drawing, normalised on its own.
    u, v = np.meshgrid((np.arange(12)+.5)/12-.5, (np.arange(12)+.5)/12-.5)
    cols = []
    for turn in (0., .15, -.15):
        for slope in (0., .14, -.14):
            for light in (-.15, .13):
                for glasses in (False, True):
                    a = u + slope*v
                    b = v - slope*u*.60
                    def spot(x,y,w,h):
                        return np.exp(-.5*((a-x)/w)**2-.5*((b-y)/h)**2)
                    pic = .24*np.exp(-1.6*(a/.5)**2-1.4*((b+.04)/.5)**2) + light*a
                    # Rounded forehead and cheeks, paired eye sockets and brows,
                    # a lit nose bridge with shadow below, lips, rounded jaw.
                    pic += .25*spot(.45*turn,-.30,.27,.09)
                    for side in (-1,1):
                        ex = turn*.75 + side*(.214-abs(turn)*.18)
                        ew = .09*(1+side*turn)
                        pic -= (.48 if glasses else .33)*spot(ex,-.11,ew*1.45,.063)
                        pic -= .30*spot(ex,-.084,ew*.80,.029)
                        pic -= .25*spot(ex,-.179,ew*1.18,.029)
                        pic += .18*spot(ex,.078,ew*1.35,.072)
                    pic += .27*spot(turn,.015,.054,.12)
                    pic -= .28*spot(turn,.120,.074,.032)
                    pic -= .35*spot(.8*turn,.265,.115,.030)
                    pic += .16*spot(.65*turn,.36,.13,.043)
                    width = .49*(1-.47*np.maximum(0,b-.045))
                    rho = (a/width)**2+((b+.04)/.56)**2
                    pic -= .22*np.clip((rho-.84)*3.5,0,1)
                    # The corners are variable hair/background; keep them out
                    # of the numerator as well as the patch energy.
                    weight = np.clip((1.6-(u/.52)**2-(v/.58)**2)*2,0.05,1)
                    p = pic-(pic*weight).sum()/weight.sum()
                    # Remove a linear illumination ramp, the same operation
                    # applied to the observation (u and v are orthogonal).
                    p -= u*(p*u*weight).sum()/(u*u*weight).sum()
                    p -= v*(p*v*weight).sum()/(v*v*weight).sum()
                    cols.append((p*weight/np.sqrt((p*p*weight).sum())).ravel())
    return np.ascontiguousarray(np.stack(cols,axis=1),dtype=np.float32), weight.ravel().astype(np.float32)


source = r'''import numpy as np
import math


def unit(x):
    return np.minimum(np.maximum(x, 0.), 1.)


def tint(r, g, b):
    # Red-green and green-blue differences as fractions of local exposure.
    # The band is broad along the melanin/shading direction, narrower across it.
    den = .11 + .62*r + .38*g
    a = (r-g)/den
    d = (g-b)/den
    width = .080 + .20*np.maximum(a, 0.)
    hue = (d-.48*a-.018)/width
    depth = (a-.25)/np.where(a<.25,.165,.28)
    return np.exp(-.76*hue*hue-.49*depth*depth)*unit((r-g+.004)/.045)*unit((r-.035)/.12)


def integral(z):
    return np.pad(z.cumsum(-2).cumsum(-1), ((0,0),(1,0),(1,0)))


def boxplan(n, h, w, stride):
    c = np.arange(0, n, stride)
    return c[:, None], c[None, :], h, w, n


def rect(i, reg, x1, y1, x2, y2):
    cy,cx,h,w,n = reg
    ay,by = int(round(h*y1)),int(round(h*y2))
    ax,bx = int(round(w*x1)),int(round(w*x2))
    by,bx = max(by,ay+1),max(bx,ax+1)
    y = np.clip(cy+ay,0,n-1)
    x = np.clip(cx+ax,0,n-1)
    yy = np.clip(cy+by,1,n)
    xx = np.clip(cx+bx,1,n)
    return (i[:,yy,xx] - i[:,y,xx] - i[:,yy,x] + i[:,y,x]) / ((yy-y)*(xx-x)).astype(np.float32)


def dist(a,b):
    return (abs(a[0]-b[0])+.72*abs(a[1]-b[1])+.55*abs(a[2]-b[2]))


def colors(i, reg, x1,y1,x2,y2):
    return np.stack([rect(j,reg,x1,y1,x2,y2) for j in i])


def pairmean(a,b):
    return .5*(a+b)


def frontend(im):
    r,g,b=im[:,0],im[:,1],im[:,2]
    gray=.28*r+.58*g+.14*b
    dims=(1,2)
    # Use only a weak white balance correction and only for somewhat biased
    # lighting. Keep neutral pixels neutral if there is no colour in the image.
    rm,gm,bm = [z.mean(dims)[:,None,None] for z in (r,g,b)]
    cast1 = np.clip((rm-gm-.025)*.32,-.032,.050)
    cast2 = np.clip((gm-bm-.016)*.30,-.034,.054)
    skin0=tint(r,g,b)
    chrom = (abs(r-g)+abs(g-b)).mean(dims)
    skin=np.maximum(skin0,.86*tint(r-cast1,g,b+cast2)*unit(chrom[:,None,None]/.030))
    sk=integral(skin)
    integ=[integral(gray),integral(r-g),integral(g-b)]
    lum2 = integral(gray*gray)
    return gray,skin,sk,integ,lum2,chrom


def facial_parts(yi, reg, dev):
    def lay(x,y,w,h):
        return rect(yi,reg,x-w/2,y-h/2,x+w/2,y+h/2)
    le=lay(-.21,-.11,.28,.19)
    re=lay(.21,-.11,.28,.19)
    lc=lay(-.21,.095,.28,.15)
    rc=lay(.21,.095,.28,.15)
    lf=lay(-.18,-.30,.28,.15)
    rf=lay(.18,-.30,.28,.15)
    br=lay(0.,-.078,.14,.22)
    lip=lay(0.,.264,.35,.083)
    chin=lay(0.,.374,.26,.105)
    # Both eyes should be darker than what is immediately above/below;
    # a bright bridge helps reject long horizontal stripes.
    dl,dr = (lc-le)/dev,(rc-re)/dev
    ul,ur = (lf-le)/dev,(rf-re)/dev
    eyes = .44*unit((np.minimum(dl,dr)-.04)/.79) + .13*unit((np.maximum(dl,dr)-.09)/1.05)
    eyes += .20*unit((np.minimum(ul,ur)-.02)/.9)
    eyes += .23*unit(((br-.5*(le+re))/dev-.06)/.85)
    mouth=unit(((chin-lip)/dev-.015)/.67)
    return eyes,mouth


def examine(batch):
    bs=len(batch)
    gray,skin,sk,ii,y2,chrom=frontend(batch)
    n=96
    cand=[]
    yylist=[]
    xxlist=[]
    hhlist=[]
    wwlist=[]
    mono = unit(1.-chrom/.040)[:,None,None]
    # Each level scans a face-sized window; small levels can stand for a
    # whole head, because features within a small face merge on downsampling.
    shapes=[(3,3),(5,4),(5,6),(7,4),(7,7),(10,6),(10,10),
            (14,8),(14,14),(20,12),(20,20),(29,17),(29,28),
            (42,25),(42,41),(60,36),(60,58),(82,53),(82,78)]
    for h,w in shapes:
        step=max(2,int(round(min(h,w)*.17)))
        reg=boxplan(n,h,w,step)
        cy,cx=reg[:2]
        def skinbox(x1,y1,x2,y2):
            return rect(sk,reg,x1,y1,x2,y2)
        def col(x1,y1,x2,y2):
            return colors(ii,reg,x1,y1,x2,y2)
        core=col(-.34,-.30,.34,.37)
        coreSkin=skinbox(-.34,-.30,.34,.37)
        # The per-pixel test is preferable to the tint of a mixture of
        # unrelated colours, but blurred faces can have a minority of skin.
        lum,rg,gb=core
        green=lum-.28*rg+.14*gb
        acol=tint(green+rg,green,green-gb)
        skinIn=unit(.80*coreSkin+.20*acol)
        top=col(-.37,-.92,.37,-.55)
        left=col(-1.,-.35,-.57,.34)
        right=col(.57,-.35,1.,.34)
        below=col(-.40,.68,.40,1.19)
        leftSkin=skinbox(-.98,-.38,-.55,.35)
        rightSkin=skinbox(.55,-.38,.98,.35)
        aboveSkin=skinbox(-.41,-.97,.41,-.55)
        underSkin=skinbox(-.43,.70,.43,1.25)
        surround = .30*(leftSkin+rightSkin)+.22*aboveSkin+.18*underSkin
        isolation = unit((skinIn-surround+.02)/.65)
        def cedge(a,b):
            return unit((dist(a,b)-.030)/.18)
        el,er,et,eb=[cedge(core,t) for t in (left,right,top,below)]
        outline=.34*np.minimum(el,er)+.23*(el+er)*.5+.24*et+.19*eb
        cap=unit((lum-top[0]-.015)/.18)
        # Region below the head should accommodate shoulders and a wider
        # garment that terminates at roughly corresponding left/right sides.
        if h<=42:
            bl=col(-.85,.99,-.26,1.67)
            br=col(.26,.99,.85,1.67)
            ol=col(-1.80,.99,-1.25,1.67)
            ore=col(1.25,.99,1.80,1.67)
            els,ers=cedge(bl,ol),cedge(br,ore)
            same=1.-unit((dist(bl,br)-.052)/.24)
            garment=.34*same+.31*np.minimum(els,ers)+.18*(els+ers)*.5+.17*unit((dist(core,.5*(bl+br))-.038)/.21)
            # Merely continuing the two side regions down the frame doesn't
            # make a shoulder transition (as on a post or the trunk of a tree).
            transition=.5*(cedge(bl,left)+cedge(br,right))
            garment*=.64+.36*transition
            garment=np.where(cy+1.35*h<96,garment,.35)
        else:
            garment=.40
        dev=np.sqrt(np.maximum(0,rect(y2,reg,-.43,-.37,.43,.43)-rect(ii[0],reg,-.43,-.37,.43,.43)**2))
        if h>=7:
            eyes,mouth=facial_parts(ii[0],reg,.035+.83*dev)
        else:
            eyes,mouth=.14,.25
        # Context terms are small: no part of the picture is excluded for
        # being peripheral, cropped, or near the bottom.
        position = .10*unit(1.-abs(cx-48)/49)+.13*unit(1.-abs(cy-37)/49)
        lower=unit((cy-62)/30)*(.41 if h<=10 else .18)
        if h<7:
            lower=lower+.30*unit((cy-80)/14)
        usable= (cy>=.22*h)&(cy<96-.15*h)&(cx>=.15*w)&(cx<96-.15*w)
        # Colour/head hypothesis. Uniform orange objects have weak skin
        # isolation or little head/body/inner structure.
        score=2.60*skinIn+1.02*isolation+1.02*outline+.40*cap+1.13*garment+position-lower
        if h<7:
            score=score-4.98-(.21 if h==3 else 0.)
        else:
            score=score+1.12*eyes+.20*mouth-5.08
        # Strong paired features can provide independent evidence in an
        # achromatic photograph or with unhelpful lighting.
        if h>=7:
            facial=2.19*eyes+.51*mouth+.93*outline+.40*cap+.82*garment+.85*skinIn+position+.30*mono-4.13
            score=np.maximum(score,facial)
        score=score-np.where(usable,0.,1.2)
        # Cheap shortlist before detailed verification, independently for
        # each window size/aspect. More than one local maximum is retained.
        grid=score.shape
        local=score.copy()
        local[:,1:] = np.maximum(local[:,1:],score[:,:-1])
        local[:,:-1] = np.maximum(local[:,:-1],score[:,1:])
        local[:,:,1:] = np.maximum(local[:,:,1:],score[:,:,:-1])
        local[:,:,:-1] = np.maximum(local[:,:,:-1],score[:,:,1:])
        v=np.where(score>=local,score,-12.).reshape(bs,-1)
        take=np.argpartition(v,-7,axis=1)[:,-7:]
        val=np.take_along_axis(v,take,axis=1)
        cand.append(val)
        yylist.append((take//grid[2])*step)
        xxlist.append((take%grid[2])*step)
        hhlist.append(np.full_like(take,h))
        wwlist.append(np.full_like(take,w))
    score=np.concatenate(cand,1)
    cy=np.concatenate(yylist,1)
    cx=np.concatenate(xxlist,1)
    h=np.concatenate(hhlist,1)
    w=np.concatenate(wwlist,1)
    # Verify the face layout using continuous interpolation and a small
    # analytic set of lighting/view/eye variants. Penalise a poor match only
    # mildly: a person need not face the camera.
    v=(np.arange(12,dtype=np.float32)+.5)/12-.5
    py=cy[:,:,None,None]+v[None,None,:,None]*h[:,:,None,None]
    px=cx[:,:,None,None]+v[None,None,None,:]*w[:,:,None,None]
    py=np.clip(py-.5,0.,94.999).astype(np.float32)
    px=np.clip(px-.5,0.,94.999).astype(np.float32)
    y=py.astype(np.int32)
    x=px.astype(np.int32)
    fy=py-y
    fx=px-x
    bix=np.arange(bs)[:,None,None,None]
    patch=((gray[bix,y,x]*(1-fx)+gray[bix,y,x+1]*fx)*(1-fy)+(gray[bix,y+1,x]*(1-fx)+gray[bix,y+1,x+1]*fx)*fy).astype(np.float32).reshape(bs,score.shape[1],144)
    wt=CONSTANTS['mask']
    wx=np.tile(v,12)
    wy=np.repeat(v,12)
    patch -= ((patch*wt).sum(-1)/wt.sum())[:,:,None]
    patch -= ((patch*(wt*wx)).sum(-1)/(wx*wx*wt).sum())[:,:,None]*wx
    patch -= ((patch*(wt*wy)).sum(-1)/(wy*wy*wt).sum())[:,:,None]*wy
    norm=np.sqrt((patch*patch*wt).sum(-1)+.014**2*wt.sum())
    # Use a fixed matrix size on each call to the linear algebra backend.
    corr=np.stack([(p@CONSTANTS['portrait']).max(-1) for p in patch])/norm
    verify=unit((corr-.26)/.47)
    score+=np.where(h>=7,1.40*(verify-.33)*unit((h-4.)/10.),0.)
    face=score.max(1)
    j=score.argmax(1)
    ry=np.take_along_axis(cy,j[:,None],axis=1)
    rx=np.take_along_axis(cx,j[:,None],axis=1)
    rh=np.take_along_axis(h,j[:,None],axis=1)
    rw=np.take_along_axis(w,j[:,None],axis=1)
    # A separate possible person is a stronger vote than many overlapping
    # interpretations of one head. Require comparable head heights for groups.
    separate=((abs(cx-rx)>.9*np.maximum(w,rw))|(abs(cy-ry)>1.15*np.maximum(h,rh)))&(h>.45*rh)&(h<2.2*rh)
    other=np.where(separate,score,-12.).max(1)
    face+=.27*unit((other+.35)/.95)
    body=bodysearch(batch,gray,skin)
    scene=context(batch,gray,skin)
    # The silhouette is a less specific route, but can rescue a back view.
    ans=np.maximum(face+.31*np.clip(body,-.65,1.),body+.27*np.clip(face,-1.,.65)) + scene
    return ans>0.


def bodysearch(im,gray,skin):
    # Four-level scale space. Sample variable-width horizontal runs against
    # a doll-shaped outline. Compare to background on EACH side; require
    # that colours recur along several places in the torso and legs.
    bs=len(gray)
    r,g,b=im[:,0],im[:,1],im[:,2]
    c=np.stack([gray,.85*(r-g),.76*(g-b),skin],axis=1)
    # A box-prefilter at half and quarter resolution smooths away texture.
    answers=np.full(bs,-4.,dtype=np.float32)
    for resolution in (1,2,3,4):
        # Pixels of the working level correspond to ~1/30 of figure height.
        n=96//resolution
        z=c.reshape(bs,4,n,resolution,n,resolution).mean((3,5)) if resolution!=1 else c
        # Small local averages, not raw pixels; neighbouring part samples
        # overlap but the outside-of-contour samples remain separate.
        p=np.pad(z,((0,0),(0,0),(1,1),(1,1)),mode='edge')
        z=(4*p[:,:,1:-1,1:-1]+2*(p[:,:,1:-1,2:]+p[:,:,1:-1,:-2]+p[:,:,2:,1:-1]+p[:,:,:-2,1:-1])+p[:,:,2:,2:]+p[:,:,:-2,:-2]+p[:,:,2:,:-2]+p[:,:,:-2,2:])/16.
        zs=np.arange(1,n,2)[:,None]
        xs=np.arange(1,n,2)[None,:]
        center=.10*unit(1.-abs(xs*resolution-48)/48)+.08*unit(1.-abs(zs*resolution-34)/45)
        for width in (1.,1.65):
            def at(dx,dy):
                return z[:,:,np.clip(zs+int(round(dy)),0,n-1),np.clip(xs+int(round(dx*width)),0,n-1)]
            def delta(a,b):
                return abs(a[:,0]-b[:,0])+abs(a[:,1]-b[:,1])+abs(a[:,2]-b[:,2])
            def evidence(a,b):
                return unit((delta(a,b)-.027)/.165)
            def row(dy,inner,outer):
                cen=at(0,dy)
                li,ri=at(-inner,dy),at(inner,dy)
                lo,ro=at(-outer,dy),at(outer,dy)
                le,re=evidence(li,lo),evidence(ri,ro)
                edge=.48*np.minimum(le,re)+.26*(le+re)
                # Each individual side should resemble the center more than
                # the outside, rather than merely be on an unrelated edge.
                fill=unit(.34+((delta(lo,cen)+delta(ro,cen))-(delta(li,cen)+delta(ri,cen)))/.36)
                return cen,edge*(.67+.33*fill),lo,ro
            hd,eh,hl,hr=row(0,0,2.6)
            sh,es,sl,sr=row(4,2,4.7)
            ch,ec,cl,cr=row(8,2,4.8)
            hp,ep,pl,pr=row(13,1.5,3.8)
            kn,ek,kl,kr=row(19,1.2,3.5)
            ft,ef,fl,fr=row(24,1.3,3.6)
            top=at(0,-3)
            crown=evidence(hd,top)
            # Head-to-shoulder widening: upper background at shoulder x
            # changes to an upper garment, then back to background outside.
            widen=.5*(evidence(hl,at(-2.5,4))+evidence(hr,at(2.5,4)))
            cloth=1-unit((delta(sh,ch)-.035)/.215)
            pants=1-unit((delta(kn,ft)-.045)/.215)
            head=eh*(.63+.37*crown)
            chest=(.5*(es+ec))*(.70+.30*cloth)
            legs=(.34*ep+.40*ek+.26*ef)*(.74+.26*pants)
            waist=evidence(ch,hp)
            neck=evidence(hd,sh)
            full=.99*head+1.24*chest+1.18*legs+.48*widen+.20*neck+.17*waist+.37*hd[:,3]+center-3.76
            # Requiring several agreeing segments helps distinguish an
            # articulated object from an isolated landmark in a busy image.
            full+=.40*np.minimum(head,np.minimum(chest,legs))
            upper=1.16*head+1.52*chest+.65*widen+.24*neck+.72*hd[:,3]+center-3.96
            rightsize=(xs>=2*width)&(xs<n-2*width)
            full=np.where((zs+22<n)&rightsize,full,-5.)
            upper=np.where((zs+8<n)&rightsize,upper,-5.)
            best=np.maximum(full.max((1,2)),upper.max((1,2)))
            answers=np.maximum(answers,best)
    return answers


def context(im,y,s):
    # Human photographs often contain exposed skin concentrated in some
    # regions, not a pervasive wood/sand tint. Long continuous, straight
    # edges across the frame are modest evidence of structures/landscapes.
    r,g,b=im[:,0],im[:,1],im[:,2]
    skinmass=s.mean((1,2))
    warm=s[:,6:88,5:91].reshape(len(s),82,86).mean((1,2))
    large=.19*unit((warm-.023)/.085)-.36*unit((skinmass-.24)/.32)
    # Orthogonal lines (buildings, windows, cabinets, transport) rather
    # than interrupted curved and oblique contours.
    v=y[:,1:,1:]-y[:,:-1,1:]
    h=y[:,1:,1:]-y[:,1:,:-1]
    tot=(abs(h)+abs(v)).mean((1,2))+.006
    cross=np.minimum(abs(h),abs(v)).mean((1,2))/tot
    rectilinear=unit((.212-cross)/.09)
    straight=(abs(v.mean(2)).mean(1)+abs(h.mean(1)).mean(1))/tot
    line=unit((straight-.26)/.35)
    # A small saturation accent on a person-sized object can be a garment,
    # particularly on snow or grass. Only use as corroboration.
    hi=np.maximum(np.maximum(r,g),b)
    lo=np.minimum(np.minimum(r,g),b)
    vivid=(hi-lo)>.29+.16*hi
    accent=vivid[:,15:85,8:88].mean((1,2))
    pop=unit(accent/.048)*unit((.29-accent)/.14)
    return large-.25*rectilinear-.19*line+.11*pop-.23


def predict(x):
    labels=np.empty(len(x),dtype=np.int64)
    for j in range(0,len(x),8):
        labels[j:j+8]=examine(x[j:j+8])
    return labels
'''


def main(dest):
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'solution.py').write_text(source,encoding='utf-8')
    portraits, mask=desk_portraits()
    np.savez(dest/'constants.npz',portrait=portraits,mask=mask)

if __name__ == '__main__':
    main(Path(sys.argv[1]))
