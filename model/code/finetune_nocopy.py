"""Gradient-descent comparisons for a model-designed network (network.json is data; it is translated, never executed).

Modes (results merged into one JSON per network):
  sizes      weights, nonzero weights, multiply-adds per image
  finetune   start from the model's own weights, train the whole network: AdamW lr 1e-4, cosine to 0, 3 epochs, seeds 0 1 2
  readout    start from the model's weights, train only the readout (weighted layers whose every path to the output
             passes only through dense, add, relu, reshape) plus one positive scale on the final scores (argmax unchanged;
             learning rate 0.05) so cross-entropy works on narrow-range hand-written scores: AdamW lr 1e-3, cosine, 3 epochs, seed 0
  search     from scratch, pick the learning rate from {3e-4, 1e-3, 3e-3}: train on the training split minus a fixed
             5,000 held-out images, 5 epochs cosine, keep the best held-out accuracy (the test set is never used to choose)
  scratch    from scratch with the chosen rate: AdamW, cosine to 0, 20 epochs, full training split, seeds 0 1 2, no augmentation
  scratch-aug  the same with augmentation fixed in advance per task: digits shifts up to 2 px; clothing shifts up to 2 px and
             mirroring; cifar shifts up to 4 px and mirroring
Every reported number is the final epoch's accuracy on the official test set; nothing stops early.
usage: train_design.py OUT_DIR TASK_DIR RESULT_JSON --mode MODE
       train_design.py OUT_DIR TASK_DIR RESULT_JSON            (no --mode: queue this network for the gd_queue runner)"""
import argparse, json, math, sys, time, numpy as np
from pathlib import Path
HERE = Path(__file__).parent
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("task"); ap.add_argument("result"); ap.add_argument("--mode"); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--rel", type=float, default=0.0); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--mask", action="store_true"); ap.add_argument("--sgd", action="store_true"); ap.add_argument("--warmup", type=float, default=0.0)
a = ap.parse_args()
if not a.mode:
    with open(HERE / "gd_networks.txt", "a") as f: f.write(f"{Path(a.out).resolve()} {Path(a.task).resolve()} {Path(a.result).resolve()}\n")
    print("queued for gd_queue.sh"); sys.exit()

import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(a.task) / "tests"))
from blocks import run as np_run
import check_network
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
spec = json.loads((Path(a.out) / "network.json").read_text())["layers"]
w_np = {k: v.astype(np.float32) for k, v in np.load(Path(a.out) / "weights.npz", allow_pickle=False).items()}
task_key = Path(a.task).name.split("-")[0]
_shape = json.loads((Path(a.task) / "tests/task_shape.json").read_text())
INPUT = tuple(_shape["input"]); check_network.INPUT_SHAPE = INPUT; check_network.CLASSES = _shape["classes"]
rows, expect, bad = check_network.infer({"layers": spec}); assert not bad, bad
shp = {"image": INPUT, **{n: s for n, _, s in rows}}
res_path = Path(a.result); result = json.loads(res_path.read_text()) if res_path.exists() else {}
def save():
    """Merge only this mode's results into the file on disk, so jobs never overwrite each other's results."""
    cur = json.loads(res_path.read_text()) if res_path.exists() else {}
    for k_ in result:
        if k_ == a.mode or k_.startswith(a.mode + "-"): cur[k_] = result[k_]
    if a.mode in result.get("minutes", {}): cur.setdefault("minutes", {})[a.mode] = result["minutes"][a.mode]
    res_path.write_text(json.dumps(cur, indent=1))


def patch_conv(x, W, b, k):
    """A convolution whose stride equals its kernel and has no padding is one matrix multiply over non-overlapping
    patches. Same weights, same result as conv2d (checked to 3e-7), but much faster on the GPU for very wide layers."""
    n, c, h, w = x.shape; ho, wo = (h - k) // k + 1, (w - k) // k + 1
    p = x[:, :, :ho * k, :wo * k].reshape(n, c, ho, k, wo, k).permute(0, 2, 4, 1, 3, 5).reshape(n * ho * wo, c * k * k)
    return (p @ W.reshape(len(W), -1).T + b).reshape(n, ho, wo, len(W)).permute(0, 3, 1, 2)


class Designed(nn.Module):
    def __init__(self):
        super().__init__(); self.mods = nn.ModuleDict()
        for L in spec:
            if L["op"] == "conv": self.mods[L["name"]] = nn.Conv2d(shp[L["inputs"][0]][0], L["out"], L["kernel"][0], L.get("stride", 1), L.get("padding", 0))
            elif L["op"] == "dense": self.mods[L["name"]] = nn.Linear(int(np.prod(shp[L["inputs"][0]])), L["out"])

    def forward(self, x):
        v = {"image": x}
        for L in spec:
            n, op, ins = L["name"], L["op"], [v[i] for i in L["inputs"]]
            if op == "conv" and L.get("stride", 1) == L["kernel"][0] and L.get("padding", 0) == 0 and L["kernel"][0] > 1:
                y = patch_conv(ins[0], self.mods[n].weight, self.mods[n].bias, L["kernel"][0])
            elif op == "conv": y = self.mods[n](ins[0])
            elif op == "dense": y = self.mods[n](ins[0].flatten(1))
            elif op == "relu": y = F.relu(ins[0])
            elif op == "maxpool": y = F.max_pool2d(ins[0], tuple(L["size"]), tuple(L["stride"]))
            elif op == "avgpool": y = F.avg_pool2d(ins[0], tuple(L["size"]), tuple(L["stride"]))
            elif op == "add": y = sum(ins)
            elif op == "reshape": y = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
            elif op == "norm":
                mu = ins[0].mean((2, 3), keepdim=True); y = (ins[0] - mu) / torch.sqrt(((ins[0] - mu) ** 2).mean((2, 3), keepdim=True) + 1e-5)
            v[n] = y
        return v[spec[-1]["name"]]

    def load_model_weights(self):
        with torch.no_grad():
            for n, m in self.mods.items(): m.weight.copy_(torch.tensor(w_np[n + ".W"])); m.bias.copy_(torch.tensor(w_np[n + ".b"]))


def readout_layers():
    """Weighted layers whose every path to the output passes only through dense, add, relu and reshape."""
    ops = {L["name"]: L["op"] for L in spec}; users = {L["name"]: [] for L in spec}
    for L in spec:
        for i in L["inputs"]:
            if i in users: users[i].append(L["name"])
    out = spec[-1]["name"]; ok = {}
    for L in reversed(spec):
        n = L["name"]
        ok[n] = n == out or (bool(users[n]) and all(ops[u] in ("dense", "add", "relu", "reshape") and ok[u] for u in users[n]))
    return [n for n in ok if ok[n] and ops[n] in ("dense", "conv")]


def as_input(arr):
    arr = arr.astype(np.float32) / 255 if arr.dtype == np.uint8 else arr.astype(np.float32)
    return arr[:, None] if arr.ndim == 3 else arr


if a.mode == "sizes":
    macs = 0
    for L in spec:
        s_in, s_out = shp[L["inputs"][0]], shp[L["name"]]
        if L["op"] == "conv": macs += int(np.prod(s_out)) * s_in[0] * L["kernel"][0] ** 2
        elif L["op"] == "dense": macs += int(np.prod(s_out)) * int(np.prod(s_in))
    result["sizes"] = {"weights": int(sum(v.size for v in w_np.values())), "nonzero": int(sum(np.count_nonzero(v) for v in w_np.values())),
                       "multiply_adds_per_image": int(macs), "layers": len(spec), "readout_layers": readout_layers()}
    save(); print(result["sizes"]); sys.exit()

tr = np.load(Path(a.task) / "environment/data/train.npz"); X, y = torch.tensor(as_input(tr["images"])), torch.tensor(tr["labels"])
te = np.load(Path(a.task) / "tests/hidden.npz"); Xt = torch.tensor(as_input(te["images"])); yt = torch.tensor(te["labels"])
SHIFT, MIRROR = {"digits": (2, False), "clothing": (2, True), "cifar": (4, True)}.get(task_key, (0, False))  # augmentation is only used for the from-scratch runs on these three tasks


def augment(xb, g):
    n, _, h, w = xb.shape
    dx = torch.randint(-SHIFT, SHIFT + 1, (n,), generator=g).float(); dy = torch.randint(-SHIFT, SHIFT + 1, (n,), generator=g).float()
    theta = torch.zeros(n, 2, 3); theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    if MIRROR: theta[:, 0, 0] = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    theta[:, 0, 2] = -2 * dx / w; theta[:, 1, 2] = -2 * dy / h
    grid = F.affine_grid(theta.to(xb.device), list(xb.shape), align_corners=False)
    return F.grid_sample(xb, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def accuracy(m, Xe, ye):
    m.eval()
    with torch.no_grad(): p = torch.cat([m(Xe[i:i + 500].to(dev)).argmax(1).cpu() for i in range(0, len(Xe), 500)])
    m.train(); return float((p == ye).float().mean())


def train(m, Xtr, ytr, *, lr, epochs, seed, aug=False, params=None, Xe=Xt, ye=yt, log=""):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    params = list(params if params is not None else m.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(Xtr) / 128); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g); t0 = time.time(); tot = 0.0
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]; xb = Xtr[idx].to(dev)
            if aug: xb = augment(xb, g)
            opt.zero_grad(); loss = F.cross_entropy(m(xb), ytr[idx].to(dev)); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach()) * len(idx)
        if dev == "mps": torch.mps.synchronize()
        acc = accuracy(m, Xe, ye); hist.append({"epoch": ep + 1, "loss": tot / len(Xtr), "accuracy": acc, "seconds": time.time() - t0})
        print(f"{log} epoch {ep + 1}: loss {tot / len(Xtr):.4f} accuracy {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return hist


t_start = time.time()
if a.mode == "check":
    m = Designed().to(dev); m.load_model_weights(); m.eval()
    with torch.no_grad(): ts = torch.cat([m(Xt[i:i + 250].to(dev)).cpu() for i in range(0, 1000, 250)]).numpy()
    ns = np.concatenate([np_run({"layers": spec}, w_np, Xt[i:i + 250].numpy()) for i in range(0, 1000, 250)])
    print(f"translation check: same prediction {(ts.argmax(1) == ns.argmax(1)).mean():.4f}, max relative score diff {np.abs(ts - ns).max() / (np.abs(ns).max() + 1e-9):.2e}"); sys.exit()
if a.mode == "finetune":
    base = Designed().to(dev); base.load_model_weights(); start = accuracy(base, Xt, yt); del base
    runs = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        m = Designed().to(dev); m.load_model_weights()
        runs.append({"seed": seed, "history": train(m, X, y, lr=1e-4, epochs=a.epochs, seed=seed, log=f"finetune seed {seed}")})
    result["finetune"] = {"start_accuracy": start, "lr": 1e-4, "epochs": a.epochs, "runs": runs}
elif a.mode == "curve":
    # one cosine schedule of --epochs epochs, test accuracy at fixed fractions of an epoch, to choose the fine-tune length
    base = Designed().to(dev); base.load_model_weights(); spe = math.ceil(len(X) / 128)

    class Scaled(nn.Module):
        """Scores times one learned positive factor, fitted first and alone: argmax unchanged, so the cross-entropy
        no longer asks every weight to widen narrow hand-written score gaps."""
        def __init__(self, inner): super().__init__(); self.inner = inner; self.log_scale = nn.Parameter(torch.zeros(()))
        def forward(self, x): return self.inner(x) * torch.exp(self.log_scale)
    m = Scaled(base).to(dev)
    if a.rel:  # fit the temperature alone on 2,000 steps' worth of batches is overkill: 300 steps of lr 0.05
        for p_ in base.parameters(): p_.requires_grad_(False)
        o1 = torch.optim.Adam([m.log_scale], lr=0.05); g0 = torch.Generator().manual_seed(99)
        for _ in range(300):
            idx = torch.randint(0, len(X), (512,), generator=g0); o1.zero_grad(); F.cross_entropy(m(X[idx].to(dev)), y[idx].to(dev)).backward(); o1.step()
        for p_ in base.parameters(): p_.requires_grad_(True)
        print("temperature scale", float(torch.exp(m.log_scale)), flush=True)
    marks = sorted({max(1, round(f * spe)) for f in (1/32, 1/16, 1/8, 1/4, 1/2)} | {e * spe for e in (1, 2, 3, 5, 7, 10, 15, 20) if e <= a.epochs})
    curve = [{"epoch": 0.0, "accuracy": accuracy(m, Xt, yt)}]; print("start", curve[0], flush=True)
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    if a.rel:  # each weight tensor steps by rel times its own typical nonzero magnitude; biases by rel times their weights'
        groups = []
        for n_, mod in base.mods.items():
            Wt = mod.weight.detach(); nz = Wt[Wt != 0].abs(); s = float(nz.median()) if nz.numel() else 1.0
            groups.append({"params": [mod.weight, mod.bias], "lr": a.rel * s})
        groups.append({"params": [m.log_scale], "lr": 1e-3})
        opt = torch.optim.SGD(groups, momentum=0.9) if a.sgd else torch.optim.AdamW(groups, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(base.parameters(), lr=a.lr, weight_decay=0.0)
    steps = a.epochs * spe
    wu = int(a.warmup * steps)  # linear warmup from 0 over the first fraction of steps, then the cosine
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    masks = {n_: (mod.weight.detach() != 0).float() for n_, mod in base.mods.items()} if a.mask else {}
    step = 0; t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), generator=g); run_loss = 0.0; nb = 0
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]; opt.zero_grad(); loss = F.cross_entropy(m(X[idx].to(dev)), y[idx].to(dev)); loss.backward()
            if a.mask:
                for n_, mod in base.mods.items():
                    if mod.weight.grad is not None: mod.weight.grad.mul_(masks[n_])
            opt.step(); sched.step()
            if a.mask:
                with torch.no_grad():
                    for n_, mod in base.mods.items(): mod.weight.mul_(masks[n_])
            step += 1; run_loss += float(loss.detach()); nb += 1
            if step in marks:
                c = {"epoch": step / spe, "accuracy": accuracy(m, Xt, yt), "loss": run_loss / nb, "seconds": time.time() - t0}; curve.append(c); print(c, flush=True)
    result["curve" + (f"-rel{a.rel:g}" if a.rel else "") + ("-mask" if a.mask else "") + ("-sgd" if a.sgd else "") + (f"-wu{a.warmup:g}" if a.warmup else "")] = {"lr": a.lr, "rel": a.rel, "schedule_epochs": a.epochs, "temperature": float(torch.exp(m.log_scale)), "points": curve}
elif a.mode == "dissect":
    # where does fine-tuning's gain come from? the curve recipe (temperature first, mask, relative lr, warmup 0.3, 3 epochs)
    # with different parts trainable: all, the readout layers only, biases only; plus test predictions before and after,
    # per-layer relative weight change, and accuracy under relative weight noise (as in Lehmann 2026, Table 4)
    spe = math.ceil(len(X) / 128)

    class Scaled(nn.Module):
        def __init__(self, inner): super().__init__(); self.inner = inner; self.log_scale = nn.Parameter(torch.zeros(()))
        def forward(self, x): return self.inner(x) * torch.exp(self.log_scale)

    def preds(m):
        m.eval()
        with torch.no_grad(): p = torch.cat([m(Xt[i:i + 500].to(dev)).argmax(1).cpu() for i in range(0, len(Xt), 500)])
        m.train(); return p

    def variant(which):
        base = Designed().to(dev); base.load_model_weights(); m = Scaled(base).to(dev)
        for p_ in base.parameters(): p_.requires_grad_(False)
        o1 = torch.optim.Adam([m.log_scale], lr=0.05); g0 = torch.Generator().manual_seed(99)
        for _ in range(300):
            idx = torch.randint(0, len(X), (512,), generator=g0); o1.zero_grad(); F.cross_entropy(m(X[idx].to(dev)), y[idx].to(dev)).backward(); o1.step()
        ro = set(readout_layers()); groups = []
        for n_, mod in base.mods.items():
            Wt = mod.weight.detach(); nz = Wt[Wt != 0].abs(); sc = float(nz.median()) if nz.numel() else 1.0
            ps = [mod.weight, mod.bias] if which == "all" else ([mod.weight, mod.bias] if (which == "readout" and n_ in ro) else ([mod.bias] if which == "bias" else []))
            for q in ps: q.requires_grad_(True)
            if ps: groups.append({"params": ps, "lr": 0.001 * sc})
        groups.append({"params": [m.log_scale], "lr": 1e-3})
        opt = torch.optim.AdamW(groups, weight_decay=0.0); steps = 3 * spe; wu = int(0.3 * steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: (s_ + 1) / wu if s_ < wu else 0.5 * (1 + math.cos(math.pi * min(s_, steps) / steps)))
        masks = {n_: (mod.weight.detach() != 0).float() for n_, mod in base.mods.items()}
        torch.manual_seed(0); g = torch.Generator().manual_seed(0)
        for ep in range(3):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), 128):
                idx = perm[i:i + 128]; opt.zero_grad(); F.cross_entropy(m(X[idx].to(dev)), y[idx].to(dev)).backward()
                for n_, mod in base.mods.items():
                    if mod.weight.grad is not None: mod.weight.grad.mul_(masks[n_])
                opt.step(); sched.step()
                with torch.no_grad():
                    for n_, mod in base.mods.items(): mod.weight.mul_(masks[n_])
        trainable = int(sum(int(masks[n_].sum()) if mod.weight.requires_grad else 0 for n_, mod in base.mods.items()) + sum(mod.bias.numel() for mod in base.mods.values() if mod.bias.requires_grad))
        change = {}
        for n_, mod in base.mods.items():
            W0 = torch.tensor(w_np[n_ + ".W"]); b0 = torch.tensor(w_np[n_ + ".b"])
            change[n_] = {"weight": float((mod.weight.detach().cpu() - W0).norm() / (W0.norm() + 1e-12)), "bias": float((mod.bias.detach().cpu() - b0).norm() / (b0.norm() + 1e-12)),
                          "nonzero": int(masks[n_].sum()), "readout": n_ in ro}
        return {"accuracy": float((preds(m) == yt).float().mean()), "pred": preds(m).tolist(), "trainable": trainable, "change": change}

    d = Designed().to(dev); d.load_model_weights(); p0 = preds(d)
    out = {"designed": {"accuracy": float((p0 == yt).float().mean()), "pred": p0.tolist()}, "labels": yt.tolist()}
    noise = {}
    for rel in (1e-4, 1e-3, 1e-2, 1e-1):
        accs = []
        for seed in (0, 1):
            nm = Designed().to(dev); nm.load_model_weights(); gen = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for mod in nm.mods.values():
                    for q in (mod.weight, mod.bias):
                        rms = float(q.pow(2).mean().sqrt()); q.add_(torch.randn(q.shape, generator=gen).to(dev) * rel * rms)
            accs.append(float((preds(nm) == yt).float().mean()))
        noise[str(rel)] = accs
    out["noise_all_entries"] = noise
    for which in ("all", "readout", "bias"): out[which] = variant(which); print(which, out[which]["accuracy"], flush=True)
    result["dissect"] = out
elif a.mode == "lehmann":
    def preds_of(m):
        m.eval()
        with torch.no_grad(): p_ = torch.cat([m(Xt[i:i + 500].to(dev)).argmax(1).cpu() for i in range(0, len(Xt), 500)])
        m.train(); return p_

    def acts(m, xb):
        """Output of every weighted layer for a batch, conv outputs averaged over space (one value per channel)."""
        got = {}; hooks = [mod.register_forward_hook(lambda _m, _i, o, n_=n_: got.__setitem__(n_, (o.mean((2, 3)) if o.dim() == 4 else o.flatten(1)).detach().double().cpu()))
                          for n_, mod in m.mods.items()]
        with torch.no_grad(): m(xb)
        for h in hooks: h.remove()
        return got

    def cka(A, B):
        A = A - A.mean(0); B = B - B.mean(0); ka = A @ A.T; kb = B @ B.T
        return float((ka * kb).sum() / (ka.norm() * kb.norm() + 1e-12))

    def drift(m, masks):
        """How far the tuned network moved from the written one: test predictions, weight change on written entries and on
        new (previously zero) entries, accuracy with new entries removed, and layer-by-layer similarity (linear CKA on 1,000 test images)."""
        res = {"pred": preds_of(m).tolist()}
        lay = {}
        for n_, mod in m.mods.items():
            W0 = torch.tensor(w_np[n_ + ".W"]).to(dev); Wn = mod.weight.detach(); mk = masks[n_]
            lay[n_] = {"written_change": float(((Wn - W0) * mk).norm() / (W0.norm() + 1e-12)), "new_mass": float((Wn * (1 - mk)).norm() / (Wn.norm() + 1e-12)),
                       "nonzero_before": int(mk.sum()), "size": int(mk.numel())}
        res["layers"] = lay
        keep = {n_: mod.weight.detach().clone() for n_, mod in m.mods.items()}
        with torch.no_grad():
            for n_, mod in m.mods.items(): mod.weight.mul_(masks[n_])
        res["support_only_accuracy"] = accuracy(m, Xt, yt)
        with torch.no_grad():
            for n_, mod in m.mods.items(): mod.weight.copy_(keep[n_])
        d0 = Designed().to(dev); d0.load_model_weights(); xb = Xt[:1000].to(dev)
        a0, a1 = acts(d0, xb), acts(m, xb)
        res["cka"] = {n_: cka(a0[n_], a1[n_]) for n_ in a0}
        return res

    # Lehmann (2026) Appendix A.6 fine-tuning settings, in supervised form (cross-entropy on the training set instead of PPO):
    # standard: Adam lr 3e-4 on every entry (zeros included), gradient clip norm 0.5, constant lr, minibatch 128, 3 epochs
    # structure: the same, updating only entries nonzero at initialisation, plus 0.05 KL(p || p_init) to the written network
    # sgd: plain SGD lr 1e-5 on every entry
    # calibrated: the output layer divided so the largest absolute score on 2,048 training images is 5, then standard
    spe = math.ceil(len(X) / 128); marks = sorted({max(1, round(f * spe)) for f in (1/32, 1/8, 1/2)} | {e * spe for e in (1, 2, 3)})
    out = {}
    for which in ("standard", "structure", "sgd", "calibrated"):
        m = Designed().to(dev); m.load_model_weights()
        ref = Designed().to(dev); ref.load_model_weights(); ref.eval()
        for q in ref.parameters(): q.requires_grad_(False)
        if which == "calibrated":
            last = [L["name"] for L in spec if L["op"] in ("dense", "conv")][-1]
            with torch.no_grad():
                mx = float(torch.cat([m(X[i:i + 256].to(dev)) for i in range(0, 2048, 256)]).abs().max()); div = max(1.0, mx / 5.0)
                m.mods[last].weight.div_(div); m.mods[last].bias.div_(div)
        masks = {n_: (mod.weight.detach() != 0).float() for n_, mod in m.mods.items()}
        opt = torch.optim.SGD(m.parameters(), lr=1e-5) if which == "sgd" else torch.optim.Adam(m.parameters(), lr=3e-4)
        curve = [{"epoch": 0.0, "accuracy": accuracy(m, Xt, yt)}]; s0 = int(a.seeds.split(",")[0]); torch.manual_seed(s0); g = torch.Generator().manual_seed(s0); step = 0  # lehmann: first of --seeds (default 0)
        diverged = False
        for ep in range(3):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), 128):
                idx = perm[i:i + 128]; xb = X[idx].to(dev); opt.zero_grad(); z = m(xb); loss = F.cross_entropy(z, y[idx].to(dev))
                if which == "structure":
                    with torch.no_grad(): z0 = ref(xb)
                    loss = loss + 0.05 * F.kl_div(F.log_softmax(z0, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
                if not torch.isfinite(loss): diverged = True; break
                loss.backward()
                if which == "structure":
                    for n_, mod in m.mods.items():
                        if mod.weight.grad is not None: mod.weight.grad.mul_(masks[n_])
                if which != "sgd": torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
                opt.step()
                if which == "structure":
                    with torch.no_grad():
                        for n_, mod in m.mods.items(): mod.weight.mul_(masks[n_])
                step += 1
                if step in marks: curve.append({"epoch": step / spe, "accuracy": accuracy(m, Xt, yt), "loss": float(loss)})
            if diverged: break
        rec = {"points": curve, "diverged": diverged}
        if which in ("standard", "structure") and not diverged:
            rec.update(drift(m, masks))
        out[which] = rec; print(which, [round(c["accuracy"], 3) for c in curve], {k: v for k, v in rec.items() if k in ("support_only_accuracy", "cka")}, flush=True)
    result["lehmann"] = out
elif a.mode == "readout":
    names = readout_layers(); m = Designed().to(dev); m.load_model_weights(); start = accuracy(m, Xt, yt)
    for p in m.parameters(): p.requires_grad_(False)
    params = [p for n in names for p in m.mods[n].parameters()]
    for p in params: p.requires_grad_(True)

    class Scaled(nn.Module):
        """The network's scores times one learned positive factor: never changes which class wins, but lets the
        cross-entropy loss work on networks whose hand-written scores sit in a narrow range."""
        def __init__(self, inner): super().__init__(); self.inner = inner; self.log_scale = nn.Parameter(torch.zeros(()))
        def forward(self, x): return self.inner(x) * torch.exp(self.log_scale)
    sm = Scaled(m).to(dev)
    hist = train(sm, X, y, lr=1e-3, epochs=3, seed=0, params=[{"params": params}, {"params": [sm.log_scale], "lr": 0.05}], log="readout") if params else []
    result["readout"] = {"start_accuracy": start, "layers": names, "trainable": int(sum(p.numel() for p in params)), "lr": 1e-3, "epochs": 3,
                         "score_scale": float(torch.exp(sm.log_scale.detach()).cpu()), "runs": [{"seed": 0, "history": hist}]}
elif a.mode == "search":
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(12345)); hold, fit = perm[:5000], perm[5000:]
    trials = []
    for lr in (3e-4, 1e-3, 3e-3):
        m = Designed().to(dev)
        h = train(m, X[fit], y[fit], lr=lr, epochs=5, seed=0, Xe=X[hold], ye=y[hold], log=f"search lr {lr}")
        trials.append({"lr": lr, "heldout_accuracy": h[-1]["accuracy"], "history": h})
    result["search"] = {"trials": trials, "chosen_lr": max(trials, key=lambda t: t["heldout_accuracy"])["lr"], "heldout_images": 5000, "epochs": 5}
elif a.mode in ("scratch", "scratch-aug"):
    lr = result["search"]["chosen_lr"]; runs = []
    for seed in (0, 1, 2):
        m = Designed().to(dev)
        runs.append({"seed": seed, "history": train(m, X, y, lr=lr, epochs=20, seed=seed, aug=a.mode == "scratch-aug", log=f"{a.mode} seed {seed}")})
        result[a.mode] = {"lr": lr, "epochs": 20, "augmentation": {"shift_px": SHIFT, "mirror": MIRROR} if a.mode == "scratch-aug" else None, "runs": runs}; save()
result.setdefault("minutes", {})[a.mode] = (time.time() - t_start) / 60
save()
