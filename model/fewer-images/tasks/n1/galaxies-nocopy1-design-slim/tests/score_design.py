"""Score a designed network on every hidden test image with the same interpreter the model was given; write reward.json."""
import argparse, json, sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/tests")
from check_network import check
from blocks import run
from rebuild import rebuild, regenerate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="/app/out"); ap.add_argument("--tests-dir", default="/tests")
    ap.add_argument("--reward-path", default="/logs/verifier/reward.json"); a = ap.parse_args()
    k = json.loads((Path(a.tests_dir) / "task_shape.json").read_text())["classes"]
    regen = "" if (Path(a.out_dir) / "weights.npz").exists() else regenerate(Path(a.out_dir))
    print("REGENERATE", regen or "ok")
    res = check(a.out_dir); zero = {"reward": 0.0, "skill": 0.0, "accuracy": 0.0, "valid_output": 0.0}
    rb = rebuild(Path(a.out_dir), need_network_json=True, lock=("/tests", "/app", "/logs")) if not res[2] else {"ok": False, "error": "network invalid"}
    print("REBUILD", json.dumps(rb))
    rb_fields = {"rebuild_ok": float(rb["ok"]), "rebuild_source_bytes": float(rb.get("source_bytes", 0)), "rebuild_max_relative_difference": float(rb.get("max_relative_difference", -1))}
    if not res[2] and not rb["ok"]:
        print("INVALID: rebuild failed:", rb["error"]); reward = {**zero, **rb_fields}
    elif res[2]:
        print("INVALID:", res[2]); reward = zero
    else:
        net, weights = res[0], res[1]; hidden = np.load(Path(a.tests_dir) / "hidden.npz")
        x = hidden["images"].astype(np.float32) / 255; x = x[:, None] if x.ndim == 3 else x; y = hidden["labels"]
        t = time.time(); pred = []
        with np.errstate(all="ignore"):
            for i in range(0, len(x), 250):
                s = np.asarray(run(net, weights, x[i:i + 250]), dtype=np.float64)
                pred.append(np.nan_to_num(s, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf).argmax(1))
        pred = np.concatenate(pred); acc = float((pred == y).mean()); c = 1 / k
        recall = [float((pred[y == j] == j).mean()) for j in range(k)]
        reward = {"reward": max(0.0, (acc - c) / (1 - c)), "skill": (acc - c) / (1 - c), "accuracy": acc, "balanced_accuracy": float(np.mean(recall)),
                  "chance": c, "valid_output": 1.0, "images": float(len(y)), "score_seconds": time.time() - t,
                  "weights_total": float(sum(v.size for v in weights.values())), "weights_nonzero": float(sum(int(np.count_nonzero(v)) for v in weights.values())),
                  "layers": float(len(net["layers"])), **{f"recall_{j}": r for j, r in enumerate(recall)},
                  **{f"predicted_{j}": float((pred == j).mean()) for j in range(k)}, **rb_fields}
    Path(a.reward_path).parent.mkdir(parents=True, exist_ok=True); Path(a.reward_path).write_text(json.dumps(reward, indent=2))
    print(json.dumps({k2: v for k2, v in reward.items() if not k2.startswith(("recall_", "predicted_"))}, indent=2))


main()
