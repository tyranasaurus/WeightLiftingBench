"""Run solution.py (and constants.npz) on images with the scorer's own sandbox, as the scorer will:
    python3 /app/run_program.py /app/out IMAGES.npy [LABELS.npy]
IMAGES: (N, C, H, W) floats from 0 to 1. Without labels, prints the predicted class of each image; with labels, prints
accuracy and the confusion table (rows: true class, columns: predicted)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_program as cp

lim = json.loads((HERE / "task_limits.json").read_text())
out, x = Path(sys.argv[1]), np.load(sys.argv[2]).astype(np.float32)
source = (out / "solution.py").read_text()
report = cp.check_source(source)
problems = list(report.violations)
budget = lim["constants_budget"]
if not problems and (out / "constants.npz").exists():
    elems = sum(int(a.size) for a in np.load(out / "constants.npz", allow_pickle=False).values())
    if report.literals + elems > budget: problems.append(f"{report.literals + elems:,} numbers (literals plus constants.npz elements); the limit is {budget:,}")
if problems:
    print("not usable:"); [print("  " + p) for p in problems]; sys.exit(1)
constants = (out / "constants.npz").read_bytes() if (out / "constants.npz").exists() else None
limits = cp.RunLimits(call_seconds=lim["call_seconds"], load_seconds=lim["load_seconds"], cpu_seconds=lim["cpu_seconds"],
                      memory_mb=lim["memory_mb"], max_constants_bytes=lim["max_constants_bytes"])
k, rows, preds = lim["classes"], lim["batch_rows"], []
try:
    with cp.ProgramProcess(source=source, entry=lim["entry"], constants=constants, limits=limits) as proc:
        for i in range(0, len(x), rows):
            o = np.asarray(proc.call(x[i:i + rows], max_reply_bytes=len(x[i:i + rows]) * k * 8 + (1 << 20)))
            preds.append(o.argmax(1) if o.ndim == 2 else o.astype(int))
except Exception as e:  # ProgramError and anything else: report it the way the scorer would see it
    print(f"program failed: {e}"); sys.exit(1)
pred = np.concatenate(preds)
if len(sys.argv) > 3:
    labels = np.load(sys.argv[3])
    log = Path("/app/out/checks.log"); n = (len(log.read_text().splitlines()) if log.exists() else 0) + 1
    with log.open("a") as f: f.write(f"check {n}: {len(labels)} images, accuracy {(pred == labels).mean():.4f}\n")
    print(f"check {n} of the 5 allowed (logged in /app/out/checks.log)")
    print(f"accuracy {(pred == labels).mean():.4f} on {len(labels)} images")
    cm = np.zeros((k, k), int); np.add.at(cm, (labels, np.clip(pred, 0, k - 1)), 1)
    print("confusion (rows true, columns predicted):"); print("     " + " ".join(f"{j:5d}" for j in range(k)))
    for i in range(k): print(f"{i:3d}  " + " ".join(f"{v:5d}" for v in cm[i]))
else:
    print(" ".join(map(str, pred.tolist())))
