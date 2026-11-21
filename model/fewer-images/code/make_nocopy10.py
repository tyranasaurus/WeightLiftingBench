"""N examples per class (default 10; argument: N): copy the final no-copy design (slim) and program tasks, keep N training images per class (all of them for a class with fewer)
(fixed seed, the same images for both arms), and say so in the prompt. Everything else is unchanged.
Output: tasks<N>/<task>-nocopy<N>-{design-slim,program}. Usage: uv run --with numpy python scripts/make_nocopy10.py [N]"""
import re, shutil, sys, numpy as np
from pathlib import Path

W = Path(__file__).resolve().parent.parent
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = W / f"tasks{PER_CLASS}"
TASKS = ["digits", "clothing", "cifar", "letters", "traffic", "galaxies", "person"]
SEED = 0

for t in TASKS:
    src = np.load(W / f"tasks/{t}-nocopy-design-slim/environment/data/train.npz")
    labels = src["labels"]; rng = np.random.default_rng(SEED)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), min(PER_CLASS, int((labels == c).sum())), replace=False) for c in np.unique(labels)]))
    n = len(idx)
    short = any((labels == c).sum() < PER_CLASS for c in np.unique(labels))
    few = (f"{PER_CLASS} examples" if PER_CLASS > 1 else "1 example") + (f" (all of them for a class with fewer than {PER_CLASS})" if short else "")
    for arm in ("design-slim", "program"):
        a, b = W / f"tasks/{t}-nocopy-{arm}", OUT / f"{t}-nocopy{PER_CLASS}-{arm}"
        if b.exists(): shutil.rmtree(b)
        shutil.copytree(a, b, symlinks=False)
        d = b / "environment/data/train.npz"; d.unlink()
        np.savez(d, **{k: src[k][idx] for k in src.files})
        p = b / "instruction.md"; s = p.read_text()
        s, k1 = re.subn(r"/app/data/train\.npz is (?:the official training set|a random [\d,]+-photo sample of the training split): images \([\d]+,",
                        f"/app/data/train.npz holds only {few} of each class, picked at random from the training set ({n:,} images in all): images ({n},", s)
        s, k2 = re.subn(r"labels \(\d+,\)", f"labels ({n},)", s)
        s, k3 = re.subn(r"trained by gradient descent on the ([\d,]+) training images",
                        r"trained by gradient descent on the full training set of \1 images (you have only " + str(PER_CLASS) + " per class)", s)
        assert (k1, k2, k3) == (1, 1, 1), (t, arm, k1, k2, k3)
        p.write_text(s)
        q = b / "task.toml"; c = q.read_text()
        c = c.replace(f"{t}-nocopy-{arm}", f"{t}-nocopy{PER_CLASS}-{arm}").replace('access = "full training set;', f'access = "{PER_CLASS} examples per class;')
        c = re.sub(r'(description = ".*?)"\n', lambda m: m.group(1) + f"; only {PER_CLASS} training examples per class" + '"\n', c, count=1)
        q.write_text(c)
    print(t, n, "examples")
