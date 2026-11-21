"""Reference answer, data free: mean colour of the central 32 x 32 box and of the whole frame, then a hand-set linear readout."""
import json, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
net = {"layers": [
    {"name": "pool", "op": "avgpool", "inputs": ["image"], "size": [32, 32], "stride": [32, 32]},
    {"name": "flat", "op": "reshape", "inputs": ["pool"], "shape": [48]},
    {"name": "head", "op": "dense", "inputs": ["flat"], "out": 10}]}
W = np.zeros((10, 48), np.float32)
centre = [c * 16 + y * 4 + x for c in range(3) for y in (1, 2) for x in (1, 2)]
edge = [c * 16 + y * 4 + x for c in range(3) for y in range(4) for x in range(4) if (y, x) in ((0, 0), (0, 3), (3, 0), (3, 3))]
W[2, centre] = 1.0; W[2, edge] = -1.0; W[4, centre] = 0.5
b = np.linspace(0.1, 0.0, 10).astype(np.float32)
(out / "network.json").write_text(json.dumps(net))
np.savez(out / "weights.npz", **{"head.W": W, "head.b": b})
