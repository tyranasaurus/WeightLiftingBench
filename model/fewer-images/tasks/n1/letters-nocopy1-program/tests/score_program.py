"""Verifier for the program tasks: wsp's program sandbox (static checks, separate restricted process) feeding every test
image; then a second fresh process with the images shuffled and in batches of 777 must give identical predictions.
Reports the same numbers as the weights verifier. reward = skill floored at 0."""
import argparse, json
from pathlib import Path
import numpy as np
from wsp.program_arm import score_program
from wsp.program_sandbox import ProgramError
from wsp.targets.base import Metrics


class HiddenImages:
    def __init__(self, tests_dir: Path, k: int, order=None, batch=None):
        hidden = np.load(tests_dir / "hidden.npz")
        self.images, self.labels, self.k = hidden["images"], hidden["labels"], k
        self.order = np.arange(len(self.labels)) if order is None else order
        self.batch, self.predictions = batch, None

    def program_classes(self) -> int:
        return self.k

    def program_scores_from(self, call, *, split, seed, batch_rows):
        x = self.images[self.order].astype(np.float32) / 255
        if x.ndim == 3:
            x = x[:, None]
        batch_rows = self.batch or batch_rows
        parts = []
        for i in range(0, len(x), batch_rows):
            out, rows = np.asarray(call(x[i:i + batch_rows])), len(x[i:i + batch_rows])
            if out.shape == (rows,) and np.all(np.isfinite(out)) and np.all(out == np.round(out)) and out.min() >= 0 and out.max() <= self.k - 1:
                parts.append(out.astype(int))
            elif out.shape == (rows, self.k) and np.all(np.isfinite(out)):
                parts.append(out.argmax(axis=1))
            else:
                raise ProgramError(f"expected {rows} labels 0 to {self.k - 1} or a ({rows}, {self.k}) array of class scores, got shape {out.shape}")
        pred = np.empty(len(x), dtype=int); pred[self.order] = np.concatenate(parts); self.predictions = pred
        accuracy = float((pred == self.labels).mean()); chance = 1.0 / self.k
        recall = [float((pred[self.labels == c] == c).mean()) for c in range(self.k)]
        share = np.bincount(pred, minlength=self.k) / len(pred)
        values = {"accuracy": accuracy, "skill": (accuracy - chance) / (1 - chance), "balanced_accuracy": float(np.mean(recall)), "chance": chance, "images": float(len(x)),
                  **{f"recall_{c}": v for c, v in enumerate(recall)}, **{f"predicted_{c}": float(v) for c, v in enumerate(share)}}
        return Metrics(values=values, primary="accuracy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/tests/verifier_config.json")
    parser.add_argument("--tests-dir", default="/tests")
    parser.add_argument("--reward-path", default="/logs/verifier/reward.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()); k = config["classes"]
    paths = config["program"]
    import sys; sys.path.insert(0, args.tests_dir); from rebuild import rebuild
    rb = rebuild(Path(paths["solution_path"]).parent, need_network_json=False, lock=(args.tests_dir, "/app", "/logs"))
    print("REBUILD", json.dumps(rb))
    rb_fields = {"rebuild_ok": float(rb["ok"]), "rebuild_source_bytes": float(rb.get("source_bytes", 0)), "rebuild_max_relative_difference": float(rb.get("max_relative_difference", -1))}
    if not rb["ok"]:
        print("INVALID: rebuild failed:", rb["error"])
        Path(args.reward_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.reward_path).write_text(json.dumps({"reward": 0.0, "skill": 0.0, "accuracy": 0.0, "valid_output": 0.0, **rb_fields}, indent=2))
        raise SystemExit(0)
    first = HiddenImages(Path(args.tests_dir), k)
    reward, report = score_program(config=config, target=first, solution_path=Path(paths["solution_path"]), constants_path=Path(paths["constants_path"]))
    if reward.get("valid_output", 0.0) == 1.0:
        second = HiddenImages(Path(args.tests_dir), k, order=np.random.default_rng(12345).permutation(len(first.labels)), batch=777)
        score_program(config=config, target=second, solution_path=Path(paths["solution_path"]), constants_path=Path(paths["constants_path"]))
        differ = int((first.predictions != second.predictions).sum()) if second.predictions is not None else len(first.labels)
        reward["deterministic"] = float(differ == 0); reward["predictions_that_changed"] = float(differ)
        if differ:
            report = {"invalid": f"not deterministic: {differ} predictions changed with a fresh process, shuffled order and batches of 777"}
            reward.update(reward=0.0, valid_output=0.0)
    if reward.get("valid_output", 0.0) == 1.0:
        reward["reward"] = float(max(0.0, reward.get("skill", 0.0)))
    reward.setdefault("accuracy", 0.0); reward.setdefault("skill", 0.0); reward.update(rb_fields)
    print("PROGRAM " + json.dumps(report, default=str))
    Path(args.reward_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reward_path).write_text(json.dumps(reward, indent=2))
    print(json.dumps({k2: v for k2, v in reward.items() if not k2.startswith(("recall_", "predicted_"))}, indent=2))
