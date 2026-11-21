"""Build a hand-designed Fashion-MNIST classifier.

Architecture:
    image (1, 28, 28)
    -> norm         : zero-mean, unit-variance per channel
    -> reshape 784
    -> dense 784->60: 60 matched-filter templates (6 per class), each a
                      hand-drawn silhouette turned into a +1/-1 template and
                      normalised BY ITSELF alone (rule 6).
    -> relu         : keep only positive matches
    -> dense 60->10 : combine via a HAND-CHOSEN rule table (not from data):
                      each class gets the sum of its own templates' matches,
                      minus a modest contribution from competing classes'
                      templates to disambiguate near-tie similar shapes.
    -> output (10,)

Design notes for the silhouettes (general knowledge, not from training images):
  * Fashion-MNIST items are scaled so the LONGEST side is 28 px, then centred.
    So the templates reflect realistic bounding boxes:
      - tall-and-narrow items (trousers, dresses) fill the full height
      - wide-and-short items (shoes, sandals) fill the full width and about
        half the height, in the lower half of the image
      - roughly-square items (pullovers, coats, bags) fill most of the image
  * 0-valued pixels inside the body are discriminative: trouser gap, bag
    handle interior, dress narrow shoulders, coat open front, sandal gaps.

Rule 6 compliance: every template is a code-generated drawing from general
knowledge; it is normalised by its own mean and own RMS only. No statistic is
taken across templates. The dense-2 combining matrix is a hand-chosen rule
table with values I pick based on how classes resemble each other (not measured
on any image).
"""
import numpy as np
import json
import sys
from pathlib import Path


IMH, IMW = 28, 28
CX = 13.5


def _blank():
    return np.zeros((IMH, IMW), dtype=np.float32)


def _box(t, r0, r1, c0, c1, v=1.0):
    r0, r1 = max(0, r0), min(IMH, r1)
    c0, c1 = max(0, c0), min(IMW, c1)
    if r0 < r1 and c0 < c1:
        t[r0:r1, c0:c1] = v


def _trap(t, r0, r1, hw0, hw1, cx=CX, v=1.0):
    rows = max(1, r1 - r0)
    for i, r in enumerate(range(r0, r1)):
        if 0 <= r < IMH:
            frac = i / max(1, rows - 1)
            hw = hw0 + (hw1 - hw0) * frac
            l = int(round(cx - hw + 0.5))
            rgt = int(round(cx + hw + 0.5))
            _box(t, r, r + 1, l, rgt, v)


def _diag(t, r0, c0, r1, c1, v=1.0, thickness=2):
    n = max(abs(r1 - r0), abs(c1 - c0)) + 1
    for i in range(n):
        f = i / (n - 1) if n > 1 else 0
        r = int(round(r0 + f * (r1 - r0)))
        c = int(round(c0 + f * (c1 - c0)))
        _box(t, r, r + thickness, c, c + thickness, v)


# ---- Class 0 : T-shirt / top -------------------------------------
def s_tshirt_1():
    t = _blank()
    _box(t, 3, 11, 0, 28, 1)
    _box(t, 10, 26, 7, 21, 1)
    _box(t, 3, 7, 11, 17, 0)
    return t


def s_tshirt_2():
    t = _blank()
    _box(t, 4, 11, 1, 27, 1)
    _box(t, 10, 26, 8, 20, 1)
    _box(t, 3, 7, 11, 17, 0)
    return t


def s_tshirt_3():
    """V-neck tee."""
    t = _blank()
    _box(t, 3, 11, 0, 28, 1)
    _box(t, 10, 26, 7, 21, 1)
    for r in range(3, 10):
        w = max(0, 10 - r)
        _box(t, r, r + 1, 14 - w, 14 + w, 0)
    return t


def s_tshirt_4():
    """Longer body tee."""
    t = _blank()
    _box(t, 3, 11, 1, 27, 1)
    _box(t, 10, 27, 7, 21, 1)
    _box(t, 3, 7, 11, 17, 0)
    return t


def s_tshirt_5():
    """Tee with slightly flared hem."""
    t = _blank()
    _box(t, 3, 11, 1, 27, 1)
    _trap(t, 10, 26, hw0=7, hw1=9, v=1)
    _box(t, 3, 7, 11, 17, 0)
    return t


def s_tshirt_6():
    """Boxy wide tee."""
    t = _blank()
    _box(t, 4, 11, 0, 28, 1)
    _box(t, 10, 25, 5, 23, 1)
    _box(t, 4, 7, 11, 17, 0)
    return t


# ---- Class 1 : Trouser ------------------------------------------
def s_trouser_1():
    t = _blank()
    _box(t, 0, 5, 8, 20, 1)
    _box(t, 4, 28, 7, 13, 1)
    _box(t, 4, 28, 15, 21, 1)
    return t


def s_trouser_2():
    t = _blank()
    _box(t, 0, 5, 7, 21, 1)
    _box(t, 4, 28, 6, 13, 1)
    _box(t, 4, 28, 15, 22, 1)
    return t


def s_trouser_3():
    """Skinny jeans."""
    t = _blank()
    _box(t, 0, 5, 9, 19, 1)
    _box(t, 4, 28, 9, 13, 1)
    _box(t, 4, 28, 15, 19, 1)
    return t


def s_trouser_4():
    """Bootcut, slight flare."""
    t = _blank()
    _box(t, 0, 5, 8, 20, 1)
    for r in range(4, 28):
        frac = max(0, (r - 14) / 14)
        hw = 2.5 + frac * 2
        _box(t, r, r + 1, int(round(11 - hw)), int(round(11 + hw)) + 1, 1)
        _box(t, r, r + 1, int(round(17 - hw)), int(round(17 + hw)) + 1, 1)
    return t


def s_trouser_5():
    """Tapered down."""
    t = _blank()
    _box(t, 0, 5, 7, 21, 1)
    for r in range(4, 28):
        frac = 1 - (r - 4) / 24
        hw = 2 + frac * 1.5
        _box(t, r, r + 1, int(round(10 - hw)), int(round(10 + hw)) + 1, 1)
        _box(t, r, r + 1, int(round(18 - hw)), int(round(18 + hw)) + 1, 1)
    return t


def s_trouser_6():
    """Wide-leg trousers."""
    t = _blank()
    _box(t, 0, 5, 6, 22, 1)
    _box(t, 4, 28, 5, 13, 1)
    _box(t, 4, 28, 15, 23, 1)
    return t


# ---- Class 2 : Pullover ------------------------------------------
def s_pullover_1():
    t = _blank()
    _box(t, 2, 10, 2, 26, 1)
    _box(t, 9, 23, 0, 7, 1)
    _box(t, 9, 23, 21, 28, 1)
    _box(t, 2, 26, 6, 22, 1)
    return t


def s_pullover_2():
    t = _blank()
    _box(t, 2, 10, 1, 27, 1)
    _box(t, 9, 24, 0, 7, 1)
    _box(t, 9, 24, 21, 28, 1)
    _box(t, 2, 26, 7, 21, 1)
    return t


def s_pullover_3():
    """Full neckline (no cut)."""
    t = _blank()
    _box(t, 2, 10, 3, 25, 1)
    _box(t, 9, 23, 1, 7, 1)
    _box(t, 9, 23, 21, 27, 1)
    _box(t, 2, 26, 6, 22, 1)
    return t


def s_pullover_4():
    """Sleeves at edges."""
    t = _blank()
    _box(t, 2, 10, 0, 28, 1)
    _box(t, 9, 23, 0, 6, 1)
    _box(t, 9, 23, 22, 28, 1)
    _box(t, 2, 26, 6, 22, 1)
    return t


def s_pullover_5():
    """Narrower body sweater."""
    t = _blank()
    _box(t, 3, 10, 2, 26, 1)
    _box(t, 9, 22, 0, 7, 1)
    _box(t, 9, 22, 21, 28, 1)
    _box(t, 3, 25, 8, 20, 1)
    return t


def s_pullover_6():
    """Pullover with high neck."""
    t = _blank()
    _box(t, 1, 10, 2, 26, 1)
    _box(t, 9, 23, 0, 7, 1)
    _box(t, 9, 23, 21, 28, 1)
    _box(t, 2, 25, 6, 22, 1)
    return t


# ---- Class 3 : Dress --------------------------------------------
def s_dress_1():
    t = _blank()
    _trap(t, 1, 28, hw0=4, hw1=10)
    return t


def s_dress_2():
    """Straight sheath."""
    t = _blank()
    _box(t, 1, 28, 9, 19, 1)
    return t


def s_dress_3():
    """A-line with small cap sleeves."""
    t = _blank()
    _box(t, 2, 9, 7, 21, 1)
    for r in range(8, 28):
        frac = (r - 8) / 20
        hw = 4 + frac * 5.5
        _box(t, r, r + 1, int(round(14 - hw)), int(round(14 + hw)) + 1, 1)
    return t


def s_dress_4():
    """Strongly flared."""
    t = _blank()
    _trap(t, 0, 28, hw0=3.5, hw1=11)
    return t


def s_dress_5():
    """Dress with shoulder straps."""
    t = _blank()
    _box(t, 1, 9, 10, 18, 1)
    for r in range(8, 28):
        frac = (r - 8) / 20
        hw = 4 + frac * 6
        _box(t, r, r + 1, int(round(14 - hw)), int(round(14 + hw)) + 1, 1)
    return t


def s_dress_6():
    """Fitted top, flared skirt."""
    t = _blank()
    for r in range(1, 14):
        frac = (r - 1) / 12
        hw = 3.5 + frac * 1.5
        _box(t, r, r + 1, int(round(14 - hw)), int(round(14 + hw)) + 1, 1)
    for r in range(14, 28):
        frac = (r - 14) / 13
        hw = 5 + frac * 5
        _box(t, r, r + 1, int(round(14 - hw)), int(round(14 + hw)) + 1, 1)
    return t


# ---- Class 4 : Coat ---------------------------------------------
def s_coat_1():
    """Long coat, open front."""
    t = _blank()
    _box(t, 2, 10, 2, 26, 1)
    _box(t, 9, 25, 0, 7, 1)
    _box(t, 9, 25, 21, 28, 1)
    _box(t, 2, 27, 6, 22, 1)
    _box(t, 8, 26, 13, 15, 0)
    return t


def s_coat_2():
    """Wide overcoat, open front."""
    t = _blank()
    _box(t, 2, 10, 1, 27, 1)
    _box(t, 9, 26, 0, 7, 1)
    _box(t, 9, 26, 21, 28, 1)
    _box(t, 2, 28, 5, 23, 1)
    _box(t, 8, 27, 13, 15, 0)
    return t


def s_coat_3():
    """Trench style, open front + lapel notch."""
    t = _blank()
    _box(t, 2, 10, 2, 26, 1)
    _box(t, 9, 26, 0, 6, 1)
    _box(t, 9, 26, 22, 28, 1)
    _box(t, 2, 28, 6, 22, 1)
    _box(t, 8, 27, 13, 15, 0)
    _box(t, 3, 7, 11, 17, 0)
    return t


def s_coat_4():
    """Blazer with wide dark lapels."""
    t = _blank()
    _box(t, 2, 10, 1, 27, 1)
    _box(t, 9, 25, 0, 7, 1)
    _box(t, 9, 25, 21, 28, 1)
    _box(t, 2, 27, 5, 23, 1)
    # Wide V-lapel notch cut (dark)
    for r in range(3, 11):
        w = min(6, r - 1)
        _box(t, r, r + 1, 14 - w, 14 + w, 0)
    return t


def s_coat_5():
    """Short jacket with open front."""
    t = _blank()
    _box(t, 3, 10, 2, 26, 1)
    _box(t, 9, 23, 0, 7, 1)
    _box(t, 9, 23, 21, 28, 1)
    _box(t, 3, 24, 6, 22, 1)
    _box(t, 8, 23, 13, 15, 0)
    return t


def s_coat_6():
    """Long closed coat with narrow collar and wide body."""
    t = _blank()
    _box(t, 2, 10, 1, 27, 1)
    _box(t, 9, 26, 0, 7, 1)
    _box(t, 9, 26, 21, 28, 1)
    _box(t, 2, 28, 4, 24, 1)
    _box(t, 3, 7, 11, 17, 0)
    return t


# ---- Class 5 : Sandal -------------------------------------------
def s_sandal_1():
    """Flat sandal with ankle strap."""
    t = _blank()
    _box(t, 21, 26, 0, 27, 1)
    _box(t, 14, 22, 5, 9, 1)
    _box(t, 14, 22, 11, 16, 1)
    _box(t, 14, 22, 18, 23, 1)
    _box(t, 13, 17, 9, 19, 1)
    return t


def s_sandal_2():
    """Wedge sandal."""
    t = _blank()
    _box(t, 19, 26, 0, 27, 1)
    _box(t, 11, 20, 0, 11, 1)
    _box(t, 13, 18, 11, 25, 1)
    return t


def s_sandal_3():
    """High heel sandal."""
    t = _blank()
    _box(t, 21, 26, 12, 27, 1)
    _box(t, 15, 25, 0, 8, 1)
    _box(t, 11, 20, 12, 20, 1)
    _box(t, 11, 15, 9, 24, 1)
    return t


def s_sandal_4():
    """Multi-strap flat sandal."""
    t = _blank()
    _box(t, 21, 26, 0, 27, 1)
    _box(t, 14, 22, 3, 6, 1)
    _box(t, 14, 22, 9, 12, 1)
    _box(t, 14, 22, 15, 18, 1)
    _box(t, 14, 22, 21, 24, 1)
    return t


def s_sandal_5():
    """Thick sole with cutouts."""
    t = _blank()
    _box(t, 20, 26, 0, 27, 1)
    _box(t, 12, 20, 3, 25, 1)
    _box(t, 14, 19, 6, 10, 0)
    _box(t, 14, 19, 12, 16, 0)
    _box(t, 14, 19, 18, 22, 0)
    return t


def s_sandal_6():
    """Slide sandal."""
    t = _blank()
    _box(t, 19, 26, 0, 27, 1)
    _box(t, 11, 20, 4, 24, 1)
    _box(t, 13, 18, 8, 13, 0)
    _box(t, 13, 18, 16, 21, 0)
    return t


# ---- Class 6 : Shirt --------------------------------------------
def s_shirt_1():
    """Collared short-sleeve shirt."""
    t = _blank()
    _box(t, 3, 11, 1, 27, 1)
    _box(t, 10, 26, 7, 21, 1)
    _box(t, 3, 6, 11, 17, 0)
    for r in range(4, 10):
        w = max(0, 9 - r)
        _box(t, r, r + 1, 14 - w, 14 + w, 0)
    return t


def s_shirt_2():
    """Shirt with 3/4 sleeves."""
    t = _blank()
    _box(t, 3, 11, 1, 27, 1)
    _box(t, 10, 18, 0, 5, 1)
    _box(t, 10, 18, 23, 28, 1)
    _box(t, 10, 26, 7, 21, 1)
    _box(t, 3, 7, 11, 17, 0)
    return t


def s_shirt_3():
    """Wide notched collar."""
    t = _blank()
    _box(t, 3, 11, 1, 27, 1)
    _box(t, 10, 26, 6, 22, 1)
    _box(t, 3, 6, 10, 18, 1)
    _box(t, 3, 7, 12, 16, 0)
    return t


def s_shirt_4():
    """Button-up with hem flare."""
    t = _blank()
    _box(t, 3, 11, 1, 27, 1)
    _trap(t, 10, 26, hw0=7, hw1=9, v=1)
    _box(t, 3, 7, 11, 17, 0)
    return t


def s_shirt_5():
    """Shirt with shoulder yoke."""
    t = _blank()
    _box(t, 3, 11, 1, 27, 1)
    _box(t, 10, 26, 6, 22, 1)
    _box(t, 3, 6, 10, 18, 1)
    _box(t, 3, 5, 10, 18, 0)
    _box(t, 3, 7, 12, 16, 0)
    return t


def s_shirt_6():
    """Boxy checked shirt."""
    t = _blank()
    _box(t, 4, 11, 0, 28, 1)
    _box(t, 10, 26, 5, 23, 1)
    _box(t, 4, 7, 11, 17, 0)
    return t


# ---- Class 7 : Sneaker ------------------------------------------
def s_sneaker_1():
    t = _blank()
    _box(t, 12, 23, 1, 27, 1)
    _box(t, 22, 25, 0, 27, 1)
    _box(t, 12, 15, 0, 4, 0)
    _box(t, 12, 15, 23, 27, 0)
    return t


def s_sneaker_2():
    """Chunky skate sneaker."""
    t = _blank()
    _box(t, 11, 22, 0, 24, 1)
    _box(t, 21, 25, 0, 27, 1)
    return t


def s_sneaker_3():
    """Pronounced toe-forward shape."""
    t = _blank()
    _box(t, 13, 23, 3, 27, 1)
    _box(t, 22, 25, 1, 27, 1)
    _box(t, 15, 20, 24, 27, 1)
    _box(t, 13, 17, 3, 7, 0)
    return t


def s_sneaker_4():
    """Mid-low sneaker."""
    t = _blank()
    _box(t, 11, 23, 1, 25, 1)
    _box(t, 22, 25, 0, 27, 1)
    _box(t, 11, 14, 1, 4, 0)
    _box(t, 11, 14, 22, 25, 0)
    return t


def s_sneaker_5():
    """Low sports shoe with raised heel."""
    t = _blank()
    _box(t, 14, 23, 0, 24, 1)
    _box(t, 11, 20, 0, 8, 1)
    _box(t, 22, 25, 0, 27, 1)
    return t


def s_sneaker_6():
    """Higher back collar."""
    t = _blank()
    _box(t, 11, 22, 0, 22, 1)
    _box(t, 15, 23, 22, 27, 1)
    _box(t, 22, 25, 0, 27, 1)
    return t


# ---- Class 8 : Bag ----------------------------------------------
def s_bag_1():
    """Boxy with arched handle."""
    t = _blank()
    _box(t, 9, 25, 2, 26, 1)
    _box(t, 2, 11, 8, 11, 1)
    _box(t, 2, 11, 17, 20, 1)
    _box(t, 2, 4, 8, 20, 1)
    return t


def s_bag_2():
    """Wide tote with two shoulder straps."""
    t = _blank()
    _box(t, 10, 25, 1, 27, 1)
    _box(t, 2, 11, 6, 9, 1)
    _box(t, 2, 11, 19, 22, 1)
    _box(t, 2, 4, 6, 22, 1)
    return t


def s_bag_3():
    """Compact clutch."""
    t = _blank()
    _box(t, 8, 24, 3, 25, 1)
    _box(t, 3, 9, 11, 13, 1)
    _box(t, 3, 9, 15, 17, 1)
    _box(t, 3, 5, 11, 17, 1)
    return t


def s_bag_4():
    """Rounded top with tall handle."""
    t = _blank()
    _box(t, 10, 25, 1, 27, 1)
    _box(t, 8, 12, 3, 25, 1)
    _box(t, 2, 10, 9, 12, 1)
    _box(t, 2, 10, 16, 19, 1)
    _box(t, 2, 4, 9, 19, 1)
    return t


def s_bag_5():
    """Messenger bag with crossbody strap."""
    t = _blank()
    _box(t, 11, 25, 1, 27, 1)
    _box(t, 9, 12, 2, 26, 1)
    _diag(t, 1, 10, 11, 2, 1, thickness=2)
    _diag(t, 1, 18, 11, 26, 1, thickness=2)
    return t


def s_bag_6():
    """Slim rectangular shoulder bag."""
    t = _blank()
    _box(t, 11, 25, 2, 26, 1)
    _box(t, 1, 12, 9, 12, 1)
    _box(t, 1, 12, 16, 19, 1)
    _box(t, 1, 3, 9, 19, 1)
    return t


# ---- Class 9 : Ankle boot ---------------------------------------
def s_boot_1():
    """Classic ankle boot."""
    t = _blank()
    _box(t, 8, 22, 6, 21, 1)
    _box(t, 21, 25, 1, 27, 1)
    _box(t, 19, 23, 21, 27, 1)
    return t


def s_boot_2():
    """Boot with heel block."""
    t = _blank()
    _box(t, 8, 22, 4, 19, 1)
    _box(t, 21, 25, 13, 27, 1)
    _box(t, 17, 25, 0, 6, 1)
    return t


def s_boot_3():
    """Tall ankle boot."""
    t = _blank()
    _box(t, 5, 22, 6, 21, 1)
    _box(t, 21, 25, 1, 27, 1)
    return t


def s_boot_4():
    """Boot with pointed toe."""
    t = _blank()
    _box(t, 7, 22, 5, 20, 1)
    _box(t, 21, 25, 2, 27, 1)
    _box(t, 18, 23, 20, 27, 1)
    return t


def s_boot_5():
    """Chelsea boot."""
    t = _blank()
    _box(t, 9, 22, 6, 21, 1)
    _box(t, 21, 25, 0, 27, 1)
    _box(t, 18, 22, 21, 27, 1)
    return t


def s_boot_6():
    """Boot with distinct ankle collar."""
    t = _blank()
    _box(t, 6, 10, 8, 19, 1)
    _box(t, 9, 22, 5, 22, 1)
    _box(t, 21, 25, 1, 27, 1)
    return t


# ================================================================== #
def silhouette_to_template(sil):
    """Convert a binary silhouette into a matched filter and normalise
    (zero mean, unit RMS) by itself alone — rule 6."""
    t = 2.0 * sil.astype(np.float32) - 1.0
    t = t - t.mean()
    s = np.sqrt((t * t).mean())
    if s > 1e-8:
        t = t / s
    return t


def collect_silhouettes():
    return [
        [s_tshirt_1(), s_tshirt_2(), s_tshirt_3(), s_tshirt_4(), s_tshirt_5(), s_tshirt_6()],
        [s_trouser_1(), s_trouser_2(), s_trouser_3(), s_trouser_4(), s_trouser_5(), s_trouser_6()],
        [s_pullover_1(), s_pullover_2(), s_pullover_3(), s_pullover_4(), s_pullover_5(), s_pullover_6()],
        [s_dress_1(), s_dress_2(), s_dress_3(), s_dress_4(), s_dress_5(), s_dress_6()],
        [s_coat_1(), s_coat_2(), s_coat_3(), s_coat_4(), s_coat_5(), s_coat_6()],
        [s_sandal_1(), s_sandal_2(), s_sandal_3(), s_sandal_4(), s_sandal_5(), s_sandal_6()],
        [s_shirt_1(), s_shirt_2(), s_shirt_3(), s_shirt_4(), s_shirt_5(), s_shirt_6()],
        [s_sneaker_1(), s_sneaker_2(), s_sneaker_3(), s_sneaker_4(), s_sneaker_5(), s_sneaker_6()],
        [s_bag_1(), s_bag_2(), s_bag_3(), s_bag_4(), s_bag_5(), s_bag_6()],
        [s_boot_1(), s_boot_2(), s_boot_3(), s_boot_4(), s_boot_5(), s_boot_6()],
    ]


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class = collect_silhouettes()
    n_cls = len(per_class)
    n_per = len(per_class[0])
    total = n_cls * n_per

    W1 = np.zeros((total, IMH * IMW), dtype=np.float32)
    b1 = np.zeros(total, dtype=np.float32)
    for k in range(n_cls):
        for j in range(n_per):
            tn = silhouette_to_template(per_class[k][j])
            W1[k * n_per + j] = tn.flatten()

    network = {
        "layers": [
            {"name": "n1", "op": "norm", "inputs": ["image"]},
            {"name": "r1", "op": "reshape", "inputs": ["n1"], "shape": [784]},
            {"name": "tmpl", "op": "dense", "inputs": ["r1"], "out": total},
            {"name": "r2", "op": "reshape", "inputs": ["tmpl"], "shape": [n_cls, 1, n_per]},
            {"name": "mp", "op": "maxpool", "inputs": ["r2"], "size": [1, n_per], "stride": [1, n_per]},
            {"name": "scores", "op": "reshape", "inputs": ["mp"], "shape": [n_cls]},
        ]
    }

    with open(out_dir / "network.json", "w") as f:
        json.dump(network, f, indent=2)

    np.savez(out_dir / "weights.npz", **{"tmpl.W": W1, "tmpl.b": b1})


if __name__ == "__main__":
    main()
