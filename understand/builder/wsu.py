"""Shared pieces for the v7 builds: 16-bit rounding, function-preserving permutations, numpy forward passes used
only to verify that the permuted network computes the same function, and per-tensor value shuffling."""
import hashlib, math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv


def seed_for(tag: str, task_id: str) -> int:
    return int(hashlib.sha256(f"{tag}:{task_id}".encode()).hexdigest()[:8], 16)


def round16(sd: dict) -> tuple[dict, dict]:
    """Round every float array to float16 precision, stored as float32. Returns (new, stats)."""
    out, nz, flushed, sub = {}, 0, 0, 0
    for k, a in sd.items():
        if a.dtype.kind != "f":
            out[k] = a; continue
        h = a.astype(np.float16)
        assert np.isfinite(h).all(), (k, "overflow to inf in float16")
        nz += int((a != 0).sum()); flushed += int(((a != 0) & (h == 0)).sum())
        sub += int(((h != 0) & (np.abs(h) < np.finfo(np.float16).tiny)).sum())
        out[k] = h.astype(np.float32)
    return out, dict(nonzero=nz, flushed_to_zero=flushed, subnormal=sub)


def shuffle_values(sd: dict, seed: int) -> dict:
    """Shuffle values within every float tensor (flattened), keys in sorted order; non-float arrays kept."""
    rng = np.random.default_rng(seed); out = {}
    for k in sorted(sd):
        a = sd[k]
        if a.dtype.kind != "f":
            out[k] = a; continue
        out[k] = a.ravel()[rng.permutation(a.size)].reshape(a.shape)
    return {k: out[k] for k in sd}


def reldiff(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(a)).max(), 1e-30))


def conv(x, w, b, stride=1, pad=0, mode="constant"):
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode=mode)
    kh, kw = w.shape[2:]
    win = swv(x, (kh, kw), axis=(1, 2))[:, ::stride, ::stride]
    return np.einsum("chwij,ocij->ohw", win, w, optimize=True) + b[:, None, None]


relu = lambda x: np.maximum(x, 0)
f64 = lambda sd: {k: (v.astype(np.float64) if v.dtype.kind == "f" else v) for k, v in sd.items()}


# ---------------- Atari (conv2, conv3, fc) ----------------
def atari_perm(sd, rng):
    p2, p3, pf = rng.permutation(64), rng.permutation(64), rng.permutation(512)
    n = dict(sd)
    n["conv2.weight"] = sd["conv2.weight"][p2]; n["conv2.bias"] = sd["conv2.bias"][p2]
    n["conv3.weight"] = sd["conv3.weight"][p3][:, p2]; n["conv3.bias"] = sd["conv3.bias"][p3]
    n["fc.weight"] = sd["fc.weight"].reshape(512, 64, 49)[:, p3, :].reshape(512, 3136)[pf]
    n["fc.bias"] = sd["fc.bias"][pf]
    return n, dict(out=pf)


def atari_fwd(sd, h1):
    h = relu(conv(h1, sd["conv2.weight"], sd["conv2.bias"], 2))
    h = relu(conv(h, sd["conv3.weight"], sd["conv3.bias"], 1))
    return relu(sd["fc.weight"] @ h.reshape(-1) + sd["fc.bias"])


def atari_verify(a, b, info, rng):
    a, b = f64(a), f64(b); d = 0.0
    for _ in range(3):
        h1 = relu(rng.standard_normal((32, 20, 20)))
        d = max(d, reldiff(atari_fwd(a, h1)[info["out"]], atari_fwd(b, h1)))
    return d


# ---------------- LoRA (rank dim of each A/B pair) ----------------
def lora_perm(sd, rng):
    n = dict(sd)
    for k in sd:
        if k.endswith(".A"):
            pre = k[:-2]; r = sd[k].shape[0]; p = rng.permutation(r)
            n[pre + ".A"] = sd[pre + ".A"][p]; n[pre + ".B"] = sd[pre + ".B"][:, p]
    return n, {}


def lora_verify(a, b, info, rng):
    a, b = f64(a), f64(b); d = 0.0
    for k in a:
        if k.endswith(".A"):
            pre = k[:-2]; x = rng.standard_normal((a[k].shape[1], 3))
            d = max(d, reldiff(a[pre + ".B"] @ (a[pre + ".A"] @ x), b[pre + ".B"] @ (b[pre + ".A"] @ x)))
    return d


# ---------------- Fast style middle (conv2 .. up2, instance norm, residual stream) ----------------
def style_perm(sd, rng):
    n = dict(sd)
    def rows(name, norm, p):
        n[name + ".weight"] = n[name + ".weight"][p]; n[name + ".bias"] = n[name + ".bias"][p]
        n[norm + ".weight"] = n[norm + ".weight"][p]; n[norm + ".bias"] = n[norm + ".bias"][p]
    def cols(name, p):
        n[name + ".weight"] = n[name + ".weight"][:, p]
    p2, s, p4, p5 = rng.permutation(64), rng.permutation(128), rng.permutation(64), rng.permutation(32)
    rows("conv2", "norm2", p2); cols("conv3", p2)
    rows("conv3", "norm3", s)
    for i in range(1, 6):
        r = rng.permutation(128)
        cols(f"res{i}.conv1", s); rows(f"res{i}.conv1", f"res{i}.norm1", r)
        cols(f"res{i}.conv2", r); rows(f"res{i}.conv2", f"res{i}.norm2", s)
    cols("up1", s); rows("up1", "norm4", p4)
    cols("up2", p4); rows("up2", "norm5", p5)
    return n, dict(out=p5)


def inorm(x, w, b):
    m = x.mean(axis=(1, 2), keepdims=True); v = x.var(axis=(1, 2), keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w[:, None, None] + b[:, None, None]


def style_fwd(sd, h):
    c = lambda x, name, s=1: conv(x, sd[name + ".weight"], sd[name + ".bias"], s, 1, "reflect")
    nrm = lambda x, name: inorm(x, sd[name + ".weight"], sd[name + ".bias"])
    up = lambda x: x.repeat(2, axis=1).repeat(2, axis=2)
    h = relu(nrm(c(h, "conv2", 2), "norm2")); h = relu(nrm(c(h, "conv3", 2), "norm3"))
    for i in range(1, 6):
        r = relu(nrm(c(h, f"res{i}.conv1"), f"res{i}.norm1")); h = h + nrm(c(r, f"res{i}.conv2"), f"res{i}.norm2")
    h = relu(nrm(c(up(h), "up1"), "norm4")); h = relu(nrm(c(up(h), "up2"), "norm5"))
    return h


def style_verify(a, b, info, rng):
    a, b = f64(a), f64(b); d = 0.0
    for _ in range(2):
        h1 = relu(rng.standard_normal((32, 16, 16)))
        d = max(d, reldiff(style_fwd(a, h1)[info["out"]], style_fwd(b, h1)))
    return d


# ---------------- BERT-tiny (embeddings + 2 encoder layers) ----------------
H, NH, DH, FF = 128, 2, 64, 512


def bert_perm(sd, rng):
    n = dict(sd); P = rng.permutation(H)
    for k in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
        n[f"embeddings.{k}.weight"] = sd[f"embeddings.{k}.weight"][:, P]
    for k in ("weight", "bias"):
        n[f"embeddings.LayerNorm.{k}"] = sd[f"embeddings.LayerNorm.{k}"][P]
    for l in range(2):
        e = f"encoder.layer.{l}."
        heads = rng.permutation(NH)
        qk = np.concatenate([h * DH + rng.permutation(DH) for h in heads])
        vv = np.concatenate([h * DH + rng.permutation(DH) for h in heads])
        for m, pr in (("query", qk), ("key", qk), ("value", vv)):
            n[e + f"attention.self.{m}.weight"] = sd[e + f"attention.self.{m}.weight"][pr][:, P]
            n[e + f"attention.self.{m}.bias"] = sd[e + f"attention.self.{m}.bias"][pr]
        n[e + "attention.output.dense.weight"] = sd[e + "attention.output.dense.weight"][P][:, vv]
        n[e + "attention.output.dense.bias"] = sd[e + "attention.output.dense.bias"][P]
        F = rng.permutation(FF)
        n[e + "intermediate.dense.weight"] = sd[e + "intermediate.dense.weight"][F][:, P]
        n[e + "intermediate.dense.bias"] = sd[e + "intermediate.dense.bias"][F]
        n[e + "output.dense.weight"] = sd[e + "output.dense.weight"][P][:, F]
        n[e + "output.dense.bias"] = sd[e + "output.dense.bias"][P]
        for ln in ("attention.output.LayerNorm", "output.LayerNorm"):
            for k in ("weight", "bias"):
                n[e + f"{ln}.{k}"] = sd[e + f"{ln}.{k}"][P]
    return n, dict(out=P)


_erf = np.vectorize(math.erf)


def ln(x, w, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-12) * w + b


def bert_fwd(sd, ids, types):
    g = lambda k: sd[k]
    h = g("embeddings.word_embeddings.weight")[ids] + g("embeddings.position_embeddings.weight")[: len(ids)] + g("embeddings.token_type_embeddings.weight")[types]
    h = ln(h, g("embeddings.LayerNorm.weight"), g("embeddings.LayerNorm.bias"))
    for l in range(2):
        e = f"encoder.layer.{l}."
        lin = lambda x, m: x @ g(e + m + ".weight").T + g(e + m + ".bias")
        q, k, v = (lin(h, f"attention.self.{m}").reshape(len(ids), NH, DH).transpose(1, 0, 2) for m in ("query", "key", "value"))
        s = q @ k.transpose(0, 2, 1) / 8.0; s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        ctx = (s @ v).transpose(1, 0, 2).reshape(len(ids), H)
        h = ln(h + lin(ctx, "attention.output.dense"), g(e + "attention.output.LayerNorm.weight"), g(e + "attention.output.LayerNorm.bias"))
        u = lin(h, "intermediate.dense"); u = 0.5 * u * (1 + _erf(u / math.sqrt(2)))
        h = ln(h + lin(u, "output.dense"), g(e + "output.LayerNorm.weight"), g(e + "output.LayerNorm.bias"))
    return h


def bert_verify(a, b, info, rng):
    a, b = f64(a), f64(b); d = 0.0
    for n in (8, 24, 40):
        ids = rng.integers(0, 30522, n); types = (np.arange(n) >= n // 2).astype(int)
        d = max(d, reldiff(bert_fwd(a, ids, types)[:, info["out"]], bert_fwd(b, ids, types)))
    return d


# ---------------- Model Zoo CNN-s (sealed) ----------------
def zoo_perm(sd, rng):
    n = dict(sd); p1, p2, p3, pf = rng.permutation(8), rng.permutation(6), rng.permutation(4), rng.permutation(20)
    n["conv1.weight"] = sd["conv1.weight"][p1]; n["conv1.bias"] = sd["conv1.bias"][p1]
    n["conv2.weight"] = sd["conv2.weight"][p2][:, p1]; n["conv2.bias"] = sd["conv2.bias"][p2]
    n["conv3.weight"] = sd["conv3.weight"][p3][:, p2]; n["conv3.bias"] = sd["conv3.bias"][p3]
    n["fc1.weight"] = sd["fc1.weight"].reshape(20, 4, 9)[:, p3, :].reshape(20, 36)[pf]; n["fc1.bias"] = sd["fc1.bias"][pf]
    n["fc2.weight"] = sd["fc2.weight"][:, pf]
    return n, {}


def maxpool2(x):
    c, h, w = x.shape
    return x[:, : h // 2 * 2, : w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def zoo_fwd(sd, x):
    h = np.tanh(maxpool2(conv(x, sd["conv1.weight"], sd["conv1.bias"])))
    h = np.tanh(maxpool2(conv(h, sd["conv2.weight"], sd["conv2.bias"])))
    h = np.tanh(conv(h, sd["conv3.weight"], sd["conv3.bias"]))
    h = np.tanh(sd["fc1.weight"] @ h.reshape(-1) + sd["fc1.bias"])
    return sd["fc2.weight"] @ h + sd["fc2.bias"]


def zoo_verify(a, b, info, rng):
    a, b = f64(a), f64(b); d = 0.0
    for _ in range(5):
        x = rng.standard_normal((1, 28, 28)); d = max(d, reldiff(zoo_fwd(a, x), zoo_fwd(b, x)))
    return d


FAMILY = {"atari": (atari_perm, atari_verify), "lora": (lora_perm, lora_verify), "fast-style": (style_perm, style_verify),
          "bert": (bert_perm, bert_verify), "zoo": (zoo_perm, zoo_verify)}


def sanity_nontrivial(a, b, fwd_check):
    """A permuted network that is equal to the original would make the check vacuous; callers assert it differs."""
    return any(not np.array_equal(a[k], b[k]) for k in a if a[k].dtype.kind == "f")
