"""Hand-designed colour/edge layout classifier for 32x32 photos (10 classes).
Every number below is typed from general knowledge of what the classes look like; no data is read."""
import sys, json
import numpy as np
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)

# ---- layer 1: 3x3 conv giving 10 per-pixel evidence maps (colour categories and signed edges) ----
W1 = np.zeros((10, 3, 3, 3), np.float32)
b1 = np.zeros(10, np.float32)
W1[0, :, 1, 1] = (-0.5, -0.5, 1.0); b1[0] = -0.03      # blue: b above mean of r, g (sky, water)
W1[1, :, 1, 1] = (-0.5, 1.0, -0.5); b1[1] = -0.02      # green: g above mean of r, b (grass, foliage)
W1[2, :, 1, 1] = (1.0, 0.0, -1.0);  b1[2] = -0.05      # warm: r above b (brown, tan, orange, yellow)
W1[3, :, 1, 1] = (1.0, -0.5, -0.5); b1[3] = -0.10      # red: r well above g, b
W1[4, :, 1, 1] = (1 / 3, 1 / 3, 1 / 3); b1[4] = -0.65  # bright: luminance above 0.65
W1[5, :, 1, 1] = (-1 / 3, -1 / 3, -1 / 3); b1[5] = 0.30  # dark: luminance below 0.30
W1[6, :, 0, :] = 1 / 9; W1[6, :, 2, :] = -1 / 9        # horizontal edge (vertical luminance step), sign +
W1[7] = -W1[6]                                         # sign -
W1[8, :, :, 0] = 1 / 9; W1[8, :, :, 2] = -1 / 9        # vertical edge (horizontal luminance step), sign +
W1[9] = -W1[8]                                         # sign -

# ---- readout: rule table, rows of the image = (top, middle, bottom) thirds ----
# feature groups: B blue, G green, W warm, R red, Br bright, D dark, H horizontal-edge energy, V vertical-edge energy
SCALE = {"B": 6.0, "G": 8.0, "W": 5.0, "R": 4.0, "Br": 6.0, "D": 6.0, "H": 15.0, "V": 15.0}  # 1 / typical strength
CH = {"B": [0], "G": [1], "W": [2], "R": [3], "Br": [4], "D": [5], "H": [6, 7], "V": [8, 9]}
T = {
    0: {"B": (1.0, 0.7, 1.0), "G": (-0.8, -0.8, -0.8), "W": (-0.7, -0.6, -0.7), "R": (-0.3, 0.0, -0.3),   # airplane: sky all round, smooth
        "Br": (0.5, 0.3, 0.4), "D": (-0.5, -0.3, -0.5), "H": (-0.5, 0.2, -0.5), "V": (-0.5, -0.3, -0.5)},
    1: {"B": (-0.4, -0.2, -0.5), "G": (-0.3, -0.5, -0.6), "W": (-0.2, 0.0, -0.3), "R": (0.0, 0.8, 0.2),   # automobile: road, body colour, shadow
        "Br": (0.0, 0.1, 0.1), "D": (0.1, 0.3, 0.4), "H": (0.2, 0.6, 0.5), "V": (0.0, 0.2, 0.1)},
    2: {"B": (0.3, 0.1, 0.0), "G": (0.4, 0.3, 0.2), "W": (0.0, 0.0, 0.1), "R": (0.0, 0.0, 0.0),           # bird: smooth sky or blurred foliage
        "Br": (0.1, 0.0, 0.0), "D": (-0.2, 0.0, -0.1), "H": (-0.4, -0.3, -0.3), "V": (-0.4, -0.3, -0.3)},
    3: {"B": (-0.6, -0.5, -0.5), "G": (-0.4, -0.4, -0.3), "W": (0.3, 0.4, 0.3), "R": (0.0, 0.1, 0.0),     # cat: indoor warm neutrals
        "Br": (0.0, 0.1, 0.1), "D": (0.2, 0.1, 0.1), "H": (-0.2, -0.1, -0.2), "V": (0.0, 0.1, 0.0)},
    4: {"B": (-0.4, -0.6, -0.7), "G": (0.6, 0.5, 0.6), "W": (0.4, 0.5, 0.4), "R": (-0.3, -0.3, -0.3),     # deer: forest and meadow, brown body
        "Br": (-0.2, -0.3, -0.2), "D": (0.2, 0.2, 0.1), "H": (-0.1, -0.1, -0.1), "V": (0.1, 0.2, 0.2)},
    5: {"B": (-0.5, -0.5, -0.5), "G": (-0.1, -0.2, 0.1), "W": (0.3, 0.4, 0.3), "R": (0.0, 0.1, 0.0),      # dog: like cat, lighter, some grass
        "Br": (0.1, 0.3, 0.2), "D": (0.0, 0.0, 0.0), "H": (-0.2, -0.1, -0.2), "V": (0.0, 0.1, 0.0)},
    6: {"B": (-0.8, -0.8, -0.8), "G": (0.5, 0.6, 0.5), "W": (0.5, 0.5, 0.5), "R": (-0.1, 0.0, -0.1),      # frog: green/brown close-up, no blue
        "Br": (-0.4, -0.3, -0.3), "D": (0.2, 0.2, 0.2), "H": (0.0, 0.1, 0.0), "V": (-0.1, 0.0, -0.1)},
    7: {"B": (0.1, -0.4, -0.6), "G": (0.1, 0.1, 0.7), "W": (0.1, 0.6, 0.3), "R": (0.0, 0.2, -0.1),        # horse: sky above, brown body, grass below, legs
        "Br": (0.3, -0.1, -0.1), "D": (-0.1, 0.3, 0.0), "H": (0.0, 0.0, 0.0), "V": (0.0, 0.3, 0.3)},
    8: {"B": (0.6, 0.6, 1.0), "G": (-0.6, -0.6, -0.5), "W": (-0.5, -0.4, -0.7), "R": (-0.1, 0.1, -0.2),   # ship: water below, sky above, waterline
        "Br": (0.3, 0.2, -0.2), "D": (-0.2, 0.1, 0.2), "H": (-0.3, 0.5, 0.1), "V": (0.0, 0.1, -0.3)},
    9: {"B": (0.0, -0.3, -0.6), "G": (-0.4, -0.5, -0.6), "W": (-0.2, 0.0, -0.2), "R": (0.2, 0.6, 0.1),    # truck: tall boxy, road, sky above
        "Br": (0.3, 0.3, 0.1), "D": (0.0, 0.1, 0.3), "H": (0.2, 0.4, 0.4), "V": (0.3, 0.5, 0.3)},
}
Wd = np.zeros((10, 10, 3, 3), np.float32)   # (class, channel, row, col) over the 3x3 grid of pooled cells
for k, table in T.items():
    for g, rows in table.items():
        for ch in CH[g]:
            for r in range(3):
                Wd[k, ch, r, :] = SCALE[g] * rows[r] / 3.0   # equal over the three columns
Wd = Wd.reshape(10, 90)
bd = np.zeros(10, np.float32)

net = {"layers": [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": 10, "kernel": [3, 3], "stride": 1, "padding": 0},
    {"name": "r1", "op": "relu", "inputs": ["c1"]},
    {"name": "p1", "op": "avgpool", "inputs": ["r1"], "size": [10, 10], "stride": [10, 10]},
    {"name": "scores", "op": "dense", "inputs": ["p1"], "out": 10},
]}
(out / "network.json").write_text(json.dumps(net, indent=1))
np.savez(out / "weights.npz", **{"c1.W": W1, "c1.b": b1, "scores.W": Wd.astype(np.float32), "scores.b": bd})
