"""Run a designed network with the scorer's own code: python3 run_network.py OUT_DIR IMAGES.npy [LABELS.npy]
IMAGES: (N, C, H, W) floats from 0 to 1. Without labels, prints the predicted class of each image (and saves all scores to
OUT_DIR/../last_scores.npy). With labels (counts as one of your 5 checks), prints accuracy and the confusion table (rows: true class, columns: predicted)."""
import sys, numpy as np
from pathlib import Path
from check_network import check
from blocks import run

out_dir, images = Path(sys.argv[1]), np.load(sys.argv[2]).astype(np.float32)
res = check(out_dir)
if res[2]:
    print("network not usable:"); [print("  " + b) for b in res[2]]; sys.exit(1)
net, weights = res[0], res[1]
with np.errstate(all="ignore"):
    scores = np.concatenate([run(net, weights, images[i:i + 500]) for i in range(0, len(images), 500)])
pred = np.nan_to_num(scores, nan=-np.inf).argmax(1)
if len(sys.argv) > 3:
    labels = np.load(sys.argv[3]); k = scores.shape[1]
    log = Path("/app/out/checks.log"); n = (len(log.read_text().splitlines()) if log.exists() else 0) + 1
    with log.open("a") as f: f.write(f"check {n}: {len(labels)} images, accuracy {(pred == labels).mean():.4f}\n")
    print(f"check {n} of the 5 allowed (logged in /app/out/checks.log)")
    print(f"accuracy {(pred == labels).mean():.4f} on {len(labels)} images")
    cm = np.zeros((k, k), int); np.add.at(cm, (labels, pred), 1)
    print("confusion (rows true, columns predicted):"); print("     " + " ".join(f"{j:5d}" for j in range(k)))
    for i in range(k): print(f"{i:3d}  " + " ".join(f"{v:5d}" for v in cm[i]))
else:
    np.save(out_dir.parent / "last_scores.npy", scores); print(" ".join(map(str, pred.tolist())))
