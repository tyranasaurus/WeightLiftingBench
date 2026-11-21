"""Reference answer, data free: mean of each channel over a 4 x 4 grid of cells, then a hand-set linear readout."""
import json, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = 3 * 16
net = {"layers": [
    {"name": "pool", "op": "avgpool", "inputs": ["image"], "size": [8, 8], "stride": [8, 8]},
    {"name": "flat", "op": "reshape", "inputs": ["pool"], "shape": [n]},
    {"name": "head", "op": "dense", "inputs": ["flat"], "out": 43}]}
W = np.zeros((43, n), np.float32)
for j in range(43): W[j, j % n] = 1.0
(out / "network.json").write_text(json.dumps(net))
np.savez(out / "weights.npz", **{"head.W": W, "head.b": np.linspace(0.1, 0.0, 43).astype(np.float32)})
