"""Check a designed network without running it: python3 check_network.py OUT_DIR
Reads OUT_DIR/network.json and OUT_DIR/weights.npz, works out every layer's output shape from the settings alone,
and reports each shape, the total number of weights, and any missing, extra or mis-shaped weights. Exit code 0 if usable."""
import json, sys, numpy as np
from pathlib import Path
INPUT_SHAPE = tuple(json.load(open(Path(__file__).with_name("task_shape.json")))["input"])
_SHAPE = json.load(open(Path(__file__).with_name("task_shape.json")))
CLASSES = _SHAPE["classes"]
MAX_WEIGHTS = _SHAPE.get("max_weights")    # limit on all entries of weights.npz together, if the task sets one
MAX_NONZERO = _SHAPE.get("max_nonzero")    # limit on nonzero entries of weights.npz, if the task sets one
OPS = {"conv", "dense", "relu", "maxpool", "avgpool", "add", "reshape", "norm"}


def infer(net):
    """Returns (list of (name, op, shape)), dict of expected weight shapes, list of problems."""
    shapes, expect, rows, bad = {"image": INPUT_SHAPE}, {}, [], []
    layers = net.get("layers") if isinstance(net, dict) else None
    if not isinstance(layers, list) or not layers:
        return rows, expect, ["network.json must be {\"layers\": [ ... ]} with at least one layer"]
    for i, L in enumerate(layers):
        name, op, ins = L.get("name"), L.get("op"), L.get("inputs")
        where = f"layer {i} ({name})"
        if not isinstance(name, str) or not name or name == "image" or name in shapes:
            bad.append(f"{where}: name must be a new, non-empty string other than 'image'"); return rows, expect, bad
        if op not in OPS:
            bad.append(f"{where}: unknown op {op!r}; allowed: {sorted(OPS)}"); return rows, expect, bad
        if not isinstance(ins, list) or not ins or any(j not in shapes for j in ins):
            bad.append(f"{where}: inputs must name 'image' or earlier layers, got {ins!r}"); return rows, expect, bad
        if op != "add" and len(ins) != 1:
            bad.append(f"{where}: {op} takes exactly one input"); return rows, expect, bad
        s = shapes[ins[0]]
        try:
            if op == "conv":
                if len(s) != 3: raise ValueError(f"conv needs a (C, H, W) input, got {s}")
                kh, kw = L["kernel"]; st, p, o = int(L.get("stride", 1)), int(L.get("padding", 0)), int(L["out"])
                if kh != kw: raise ValueError("kernel must be square")
                ho, wo = (s[1] + 2 * p - kh) // st + 1, (s[2] + 2 * p - kw) // st + 1
                if min(o, kh, st, ho, wo) < 1 or p < 0: raise ValueError(f"settings give an empty or invalid output ({o}, {ho}, {wo})")
                out = (o, ho, wo); expect[name + ".W"], expect[name + ".b"] = (o, s[0], kh, kw), (o,)
            elif op == "dense":
                o = int(L["out"]);
                if o < 1: raise ValueError("out must be at least 1")
                out = (o,); expect[name + ".W"], expect[name + ".b"] = (o, int(np.prod(s))), (o,)
            elif op in ("relu",):
                out = s
            elif op == "norm":
                if len(s) != 3: raise ValueError(f"norm needs a (C, H, W) input, got {s}")
                out = s
            elif op in ("maxpool", "avgpool"):
                if len(s) != 3: raise ValueError(f"pooling needs a (C, H, W) input, got {s}")
                (sh, sw), (th, tw) = L["size"], L["stride"]
                ho, wo = (s[1] - sh) // th + 1, (s[2] - sw) // tw + 1
                if min(sh, sw, th, tw, ho, wo) < 1: raise ValueError(f"window {L['size']} with stride {L['stride']} does not fit {s}")
                out = (s[0], ho, wo)
            elif op == "add":
                if len(ins) < 2 or any(shapes[j] != s for j in ins): raise ValueError(f"add needs two or more inputs of one shape, got {[shapes[j] for j in ins]}")
                out = s
            elif op == "reshape":
                out = tuple(int(v) for v in L["shape"])
                if not 1 <= len(out) <= 3 or np.prod(out) != np.prod(s): raise ValueError(f"cannot reshape {s} to {out}")
        except (KeyError, TypeError, ValueError) as e:
            bad.append(f"{where}: {e if not isinstance(e, KeyError) else f'missing setting {e}'}"); return rows, expect, bad
        shapes[name] = out; rows.append((name, op, out))
    if rows and rows[-1][2] != (CLASSES,):
        bad.append(f"the last layer outputs {rows[-1][2]}; it must output ({CLASSES},)")
    return rows, expect, bad


def check(out_dir):
    out_dir = Path(out_dir); bad = []
    try:
        net = json.loads((out_dir / "network.json").read_text())
    except Exception as e:
        return None, None, [f"cannot read network.json: {e}"]
    rows, expect, bad = infer(net)
    if bad: return net, None, bad
    try:
        w = np.load(out_dir / "weights.npz", allow_pickle=False); weights = {k: w[k] for k in w.files}
    except Exception as e:
        return net, None, [f"cannot read weights.npz: {e}"]
    for k in sorted(set(expect) - set(weights)): bad.append(f"missing weight {k} {expect[k]}")
    for k in sorted(set(weights) - set(expect)): bad.append(f"unexpected weight {k}")
    for k, sh in expect.items():
        if k in weights:
            v = weights[k]
            if tuple(v.shape) != sh: bad.append(f"{k} has shape {tuple(v.shape)}, expected {sh}")
            elif not np.issubdtype(v.dtype, np.number) or not np.all(np.isfinite(v)): bad.append(f"{k} has non-finite or non-numeric values")
    if not bad:
        total = sum(int(np.prod(v.shape)) for v in weights.values()); nonzero = sum(int(np.count_nonzero(v)) for v in weights.values())
        if MAX_WEIGHTS is not None and total > MAX_WEIGHTS: bad.append(f"{total:,} weights in total; the limit is {MAX_WEIGHTS:,}")
        if MAX_NONZERO is not None and nonzero > MAX_NONZERO: bad.append(f"{nonzero:,} nonzero weights; the limit is {MAX_NONZERO:,}")
    return net, {k: v.astype(np.float32) for k, v in weights.items()}, bad, rows, expect


if __name__ == "__main__":
    res = check(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
    net, weights, bad = res[:3]
    if len(res) > 3:
        rows, expect = res[3], res[4]
        for name, op, s in rows: print(f"{name:24s} {op:8s} {s}")
        print(f"total weights: {sum(int(np.prod(v)) for v in expect.values()):,}" + (f" (limit {MAX_WEIGHTS:,})" if MAX_WEIGHTS else ""))
        if net is not None and weights is not None:
            print(f"nonzero weights: {sum(int(np.count_nonzero(v)) for v in weights.values()):,}" + (f" (limit {MAX_NONZERO:,})" if MAX_NONZERO else ""))
    if bad:
        print("PROBLEMS:"); [print("  " + b) for b in bad]; sys.exit(1)
    print("OK: usable")
