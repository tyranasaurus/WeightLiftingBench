"""Program arm of the no-copy tasks: <task>-nocopy-program, built from the design experiment's <task>-program-design
(full training set, scorer's program sandbox). Adds the no-copying rules, a 5-check budget on run_program.py and the
data-free rebuild gate (build.py must regenerate solution.py byte for byte and constants.npz numerically)."""
import json, shutil
from pathlib import Path

from make_nocopy import DESIGN_BASE, W, copy_tree, data_section, rules, sub, toml

RUN_OLD = '''if len(sys.argv) > 3:
    labels = np.load(sys.argv[3]); print(f"accuracy {(pred == labels).mean():.4f} on {len(labels)} images")'''
RUN_NEW = '''if len(sys.argv) > 3:
    labels = np.load(sys.argv[3])
    log = Path("/app/out/checks.log"); n = (len(log.read_text().splitlines()) if log.exists() else 0) + 1
    with log.open("a") as f: f.write(f"check {n}: {len(labels)} images, accuracy {(pred == labels).mean():.4f}\\n")
    print(f"check {n} of the 5 allowed (logged in /app/out/checks.log)")
    print(f"accuracy {(pred == labels).mean():.4f} on {len(labels)} images")'''
SCORE_OLD = '''    first = HiddenImages(Path(args.tests_dir), k)
    reward, report = score_program('''
SCORE_NEW = '''    import sys; sys.path.insert(0, args.tests_dir); from rebuild import rebuild
    rb = rebuild(Path(paths["solution_path"]).parent, need_network_json=False, lock=(args.tests_dir, "/app", "/logs"))
    print("REBUILD", json.dumps(rb))
    rb_fields = {"rebuild_ok": float(rb["ok"]), "rebuild_source_bytes": float(rb.get("source_bytes", 0)), "rebuild_max_relative_difference": float(rb.get("max_relative_difference", -1))}
    if not rb["ok"]:
        print("INVALID: rebuild failed:", rb["error"])
        Path(args.reward_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.reward_path).write_text(json.dumps({"reward": 0.0, "skill": 0.0, "accuracy": 0.0, "valid_output": 0.0, **rb_fields}, indent=2))
        raise SystemExit(0)
    first = HiddenImages(Path(args.tests_dir), k)
    reward, report = score_program('''
TAIL_OLD = '''    reward.setdefault("accuracy", 0.0); reward.setdefault("skill", 0.0)
    print("PROGRAM "'''
TAIL_NEW = '''    reward.setdefault("accuracy", 0.0); reward.setdefault("skill", 0.0); reward.update(rb_fields)
    print("PROGRAM "'''
HAVE_OLD = """python3 /app/run_program.py /app/out IMAGES.npy [LABELS.npy] runs predict in the scorer's own sandbox, with its limits,
and prints the predicted classes, or the accuracy and confusion table when labels are given."""
HAVE_NEW = """python3 /app/run_program.py /app/out IMAGES.npy LABELS.npy runs predict in the scorer's own sandbox, with its limits,
and prints the accuracy and confusion table; each such run is one of your 5 checks.
python3 /app/check_rebuild.py /app/out runs the data-free rebuild check described above."""
OUT_OLD = """The program must come from a deterministic process: if any code you run to produce it uses randomness, fix its seed,
so that running the same code again gives the same program.
Also write"""
OUT_NEW = """Save /app/out/build.py, which writes solution.py (and constants.npz if you use one) as described under
HOW THE RULES ARE CHECKED.
Also write"""
REF_BUILD = '''"""Reference answer, data free: always predicts class 0."""
import sys
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "solution.py").write_text("import numpy as np\\n\\n\\ndef predict(x):\\n    return np.zeros(len(x), dtype=np.int64)\\n")
'''
REF_SOLVE = '''#!/bin/bash
set -euo pipefail
mkdir -p /app/out
cp "$(dirname "$0")/build.py" /app/out/build.py
python3 /app/out/build.py /app/out
echo "Reference: data-free constant answer." > /app/out/method.md
'''


def program(task, out):
    P = out / f"{task}-nocopy-program"; copy_tree(DESIGN_BASE / f"{task}-program-design", P)
    p = P / "environment/run_program.py"; p.write_text(sub(p.read_text(), RUN_OLD, RUN_NEW))
    for d in ("environment", "tests"): shutil.copyfile(W / "shared/rebuild.py", P / d / "rebuild.py")
    p = P / "environment/Dockerfile"; p.write_text(sub(p.read_text(), "COPY run_program.py /app/run_program.py", "COPY run_program.py /app/run_program.py\nCOPY rebuild.py /app/check_rebuild.py"))
    p = P / "tests/Dockerfile"; p.write_text(sub(p.read_text(), "COPY score_program.py /tests/score_program.py", "COPY score_program.py /tests/score_program.py\nCOPY rebuild.py /tests/rebuild.py"))
    p = P / "tests/score_program.py"; p.write_text(sub(sub(p.read_text(), SCORE_OLD, SCORE_NEW), TAIL_OLD, TAIL_NEW))
    p = P / "instruction.md"; t = p.read_text()
    t = t.replace(t[t.index("DATA\n"):t.index("THE INPUT\n")], data_section(t).replace("the values your network receives", "the values your program receives"))
    t = t.replace(t[t.index("RULE\n"):t.index("WHAT YOU HAVE\n")], rules(task, "solution.py (and NEW_FOLDER/constants.npz if you submit one)", program=True))
    t = sub(t, HAVE_OLD, HAVE_NEW)
    t = sub(t, OUT_OLD, OUT_NEW)
    t = t.replace("1. a network designed from eight standard building blocks, with weights written by hand;",
                  "1. a network designed from eight standard building blocks, with weights written by hand under the same rules;")
    p.write_text(t)
    for f in ("solution.py", "constants.npz"): (P / "solution" / f).unlink(missing_ok=True)
    (P / "solution/build.py").write_text(REF_BUILD)
    (P / "solution/solve.sh").write_text(REF_SOLVE)
    toml(P / "task.toml", f"{task}-nocopy-program",
         f"{task}: the model writes a deterministic program; full training set to look at; no copying or fitting: build.py must rebuild solution.py with no data; at most 5 checks on training images",
         "full training set; arm program")
    return P
