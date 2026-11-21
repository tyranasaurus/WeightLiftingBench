"""Load a HF BERT checkpoint (safetensors parsed by hand, or .bin via torch weights_only) into {name: float32 ndarray}."""
import json, struct
from pathlib import Path
import numpy as np
DT = {"F32": np.float32, "F16": np.float16, "BF16": None, "I64": np.int64}
def load_safetensors(p):
    b = Path(p).read_bytes(); n = struct.unpack("<Q", b[:8])[0]; hdr = json.loads(b[8:8 + n]); base = 8 + n; out = {}
    for k, v in hdr.items():
        if k == "__metadata__": continue
        s, e = v["data_offsets"]; raw = b[base + s: base + e]
        if v["dtype"] == "BF16":
            a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            a = np.frombuffer(raw, DT[v["dtype"]])
        out[k] = a.reshape(v["shape"]).copy()
    return out
def load_bin(p):
    import torch
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return {k: v.float().numpy() if v.is_floating_point() else v.numpy() for k, v in sd.items()}
def load(d):
    d = Path(d); meta = json.loads((d / "meta.json").read_text()); p = d / meta["file"]
    sd = load_safetensors(p) if p.suffix == ".safetensors" else load_bin(p)
    norm = {}
    for k, v in sd.items():
        k2 = k[len("bert."):] if k.startswith("bert.") else k
        if k2.endswith("position_ids"): continue
        k2 = k2.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias")
        norm[k2] = v.astype(np.float32) if v.dtype.kind == "f" else v
    return norm, meta
