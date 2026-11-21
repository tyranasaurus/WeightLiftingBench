"""Build the two no-copy tasks for one image task: <task>-nocopy-design (free architecture) and <task>-nocopy-fixed
(the sweep's fixed network), both with the full training set, the no-copying rules, a 5-check budget and the data-free
rebuild gate. Usage: python3 make_nocopy.py TASK OUT_DIR"""
import json, os, re, shutil, sys
from pathlib import Path

W = Path(os.environ.get("NOCOPY_WORKDIR", "."))  # where the built task folders go
sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))
from rules_text import MEASURES, RULES, REBUILD

DESIGN_BASE = Path(os.environ.get("DESIGN_TASKS", "design/tasks"))  # the earlier design tasks these were built from (not in this repository)
FIXED_BASE = Path(os.environ.get("FIXED_TASKS", "fixed/tasks"))  # the earlier fixed-network tasks (not in this repository)
SUBJECT = {"galaxies": ("galaxies", "an ellipse with a given axis ratio, a logarithmic spiral, a bar"),
           "letters": ("hands and the American Sign Language alphabet", "a hand outline built from a palm ellipse and finger segments at given angles, a fist, an extended thumb"),
           "digits": ("handwritten digits", "a digit drawn as strokes along typed pen paths, a loop of a given size, a slanted bar"),
           "clothing": ("clothing, shoes and bags", "a shirt silhouette from rectangles for body and sleeves, a shoe profile from a sole line and an ankle curve, a bag with a handle loop"),
           "cifar": ("everyday objects and animals in small photos", "a wheel as a dark ring, a horizon line, an animal outline from ellipses for body, head and legs"),
           "traffic": ("German road signs", "a red ring of a given width, a triangle with rounded corners, a digit drawn as strokes, a blue disc with a white arrow"),
           "person": ("people in everyday photos", "a face as a skin coloured ellipse with darker bands for the eyes, a head above shoulders, a standing figure built from capsules")}
PROGRAM_SWAPS = {
    "m": (("the weights of\na network that solves this task", "a program that\nsolves this task"),
          ("and of how networks compute", "and of image processing"),
          ("not material for the weights", "not material for the program"),
          ("a network that stores, compresses or fits", "a program that stores, compresses or fits")),
    "r": (("turned into weights", "turned into numbers in your program"),
          ("no weight may hold or be derived from", "no number in your program may hold or be derived from"),
          ("You may run your network on training images", "You may run your program on training images"),
          ("python3 /app/run_network.py with labels", "python3 /app/run_program.py with labels"),
          ("run your network, or any part of it,", "run your program, or any part of it,"),
          ("features of them as weights", "features of them as constants")),
    "b": (("a Python program that writes your weights", "a Python program that writes solution.py (and constants.npz if you use one)"),
          ("every number in your weights", "every number in your program"),
          ("(to within 1e-5 relative, for float rounding)", "(solution.py byte for byte, constants.npz to within 1e-5 relative, for float rounding)")),
}


def sub(t, old, new):
    assert old in t, old[:80]
    return t.replace(old, new)


def rules(task, files, program=False):
    subject, shapes = SUBJECT[task]
    parts = {"m": MEASURES.format(subject=subject), "r": RULES.replace("{shape_examples}", shapes), "b": REBUILD.format(files=files)}
    if program:
        for key, swaps in PROGRAM_SWAPS.items():
            for old, new in swaps:
                parts[key] = sub(parts[key], old, new)
    return parts["m"] + "\n\n" + parts["r"] + "\n\n" + parts["b"] + "\n\n"


def data_section(design_instruction):
    d = design_instruction[design_instruction.index("DATA\n"):design_instruction.index("THE INPUT\n")]
    d = re.sub(r"Use it however you find useful within the rule below\.\s*", "It is for looking at and\nfor your 5 checks, within the rules below. ", d)
    d = re.sub(r"\s*You can save any images as PNG files and view them\s*with your file-reading tool\.", "", d)
    return d.rstrip() + "\n\n"


def toml(path, name, desc, access):
    t = path.read_text()
    t = re.sub(r'name = ".*"', f'name = "wsp/{name}"', t, count=1)
    t = re.sub(r'description = ".*"', f'description = "{desc}"', t, count=1)
    t = re.sub(r'experiment = ".*"', f'experiment = "{name}"', t, count=1)
    t = re.sub(r'target = ".*"', f'target = "{name}"', t, count=1)
    t = re.sub(r'access = ".*"', f'access = "{access}"', t, count=1)
    path.write_text(t)


def copy_tree(src, dst):
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))


def design(task, out):
    A = out / f"{task}-nocopy-design"; copy_tree(DESIGN_BASE / f"{task}-design", A)
    p = A / "environment/run_network.py"; t = p.read_text()
    t = sub(t, '''if len(sys.argv) > 3:
    labels = np.load(sys.argv[3]); k = scores.shape[1]
    print(f"accuracy {(pred == labels).mean():.4f} on {len(labels)} images")''', '''if len(sys.argv) > 3:
    labels = np.load(sys.argv[3]); k = scores.shape[1]
    log = Path("/app/out/checks.log"); n = (len(log.read_text().splitlines()) if log.exists() else 0) + 1
    with log.open("a") as f: f.write(f"check {n}: {len(labels)} images, accuracy {(pred == labels).mean():.4f}\\n")
    print(f"check {n} of the 5 allowed (logged in /app/out/checks.log)")
    print(f"accuracy {(pred == labels).mean():.4f} on {len(labels)} images")''')
    p.write_text(sub(t, 'OUT_DIR/../last_scores.npy). With labels,', 'OUT_DIR/../last_scores.npy). With labels (counts as one of your 5 checks),'))
    p = A / "environment/Dockerfile"
    p.write_text(sub(p.read_text(), "COPY blocks.py check_network.py run_network.py task_shape.json /app/", "COPY blocks.py check_network.py run_network.py task_shape.json /app/\nCOPY rebuild.py /app/check_rebuild.py"))
    p = A / "tests/Dockerfile"; p.write_text(sub(p.read_text(), "score_design.py /tests/", "score_design.py rebuild.py /tests/"))
    p = A / "tests/score_design.py"; t = p.read_text()
    t = sub(t, "from blocks import run\n", "from blocks import run\nfrom rebuild import rebuild\n")
    t = sub(t, '''    res = check(a.out_dir); zero = {"reward": 0.0, "skill": 0.0, "accuracy": 0.0, "valid_output": 0.0}
    if res[2]:''', '''    res = check(a.out_dir); zero = {"reward": 0.0, "skill": 0.0, "accuracy": 0.0, "valid_output": 0.0}
    rb = rebuild(Path(a.out_dir), need_network_json=True, lock=("/tests", "/app", "/logs")) if not res[2] else {"ok": False, "error": "network invalid"}
    print("REBUILD", json.dumps(rb))
    rb_fields = {"rebuild_ok": float(rb["ok"]), "rebuild_source_bytes": float(rb.get("source_bytes", 0)), "rebuild_max_relative_difference": float(rb.get("max_relative_difference", -1))}
    if not res[2] and not rb["ok"]:
        print("INVALID: rebuild failed:", rb["error"]); reward = {**zero, **rb_fields}
    elif res[2]:''')
    p.write_text(sub(t, '''                  **{f"predicted_{j}": float((pred == j).mean()) for j in range(k)}}''', '''                  **{f"predicted_{j}": float((pred == j).mean()) for j in range(k)}, **rb_fields}'''))
    for d in ("environment", "tests"): shutil.copyfile(W / "shared/rebuild.py", A / d / "rebuild.py")
    p = A / "instruction.md"; t = p.read_text()
    t = t.replace(t[t.index("DATA\n"):t.index("THE INPUT\n")], data_section(t))
    t = t.replace(t[t.index("RULE\n"):t.index("WHAT YOU HAVE\n")], rules(task, "network.json and NEW_FOLDER/weights.npz"))
    shape = re.search(r"IMAGES\.npy must hold an array of shape (\([^)]*\))", t).group(1)
    t = sub(t, """python3 /app/run_network.py /app/out IMAGES.npy [LABELS.npy] runs your network with the scorer's own code
and prints the predicted classes, or the accuracy and confusion table when labels are given.""", """python3 /app/run_network.py /app/out IMAGES.npy LABELS.npy runs your network with the scorer's own code and
prints the accuracy and confusion table; each such run is one of your 5 checks.
python3 /app/check_rebuild.py /app/out runs the data-free rebuild check described above.""")
    t = sub(t, """The weights must come from a deterministic process: if any code you run to produce them uses randomness,
fix its seed, so that running the same code again gives the same weights.
Also write""", """Save /app/out/build.py, which writes both files as described under HOW THE RULES ARE CHECKED.
Also write""")
    p.write_text(t)
    C, H, Wd = json.loads(re.sub(r"\((.*)\)", r"[\1]", shape.replace("N, ", "")))
    for f in ("network.json", "weights.npz"):
        (A / "solution" / f).unlink(missing_ok=True)
    k = json.loads((A / "tests/task_shape.json").read_text())["classes"]
    (A / "solution/build.py").write_text(f'''"""Reference answer, data free: mean of each channel over a 4 x 4 grid of cells, then a hand-set linear readout."""
import json, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = {C} * 16
net = {{"layers": [
    {{"name": "pool", "op": "avgpool", "inputs": ["image"], "size": [{H // 4}, {Wd // 4}], "stride": [{H // 4}, {Wd // 4}]}},
    {{"name": "flat", "op": "reshape", "inputs": ["pool"], "shape": [n]}},
    {{"name": "head", "op": "dense", "inputs": ["flat"], "out": {k}}}]}}
W = np.zeros(({k}, n), np.float32)
for j in range({k}): W[j, j % n] = 1.0
(out / "network.json").write_text(json.dumps(net))
np.savez(out / "weights.npz", **{{"head.W": W, "head.b": np.linspace(0.1, 0.0, {k}).astype(np.float32)}})
''')
    (A / "solution/solve.sh").write_text('''#!/bin/bash
set -euo pipefail
mkdir -p /app/out
cp "$(dirname "$0")/build.py" /app/out/build.py
python3 /app/out/build.py /app/out
echo "Reference: data-free hand-set readout." > /app/out/method.md
''')
    toml(A / "task.toml", f"{task}-nocopy-design", f"{task}: the model designs a network from eight building blocks and writes its weights; full training set to look at; no copying or fitting: build.py must rebuild the weights with no data; at most 5 checks on training images", "full training set; arm design")
    return A, t


def fixed(task, out, design_instruction):
    B = out / f"{task}-nocopy-fixed"; copy_tree(FIXED_BASE / f"{task}-look-weights-code", B)
    (B / "environment/data").mkdir(parents=True, exist_ok=True)
    src = DESIGN_BASE / f"{task}-design/environment/data/train.npz"; dst = B / "environment/data/train.npz"
    try: os.link(src, dst)
    except OSError: shutil.copyfile(src, dst)
    for d in ("environment", "tests"): shutil.copyfile(W / "shared/rebuild.py", B / d / "rebuild.py")
    shutil.copyfile(B / "tests/net.py", B / "environment/net.py")
    shape = re.search(r"IMAGES\.npy must hold an array of shape (\([^)]*\))", design_instruction).group(1)
    shutil.copyfile(W / "tasks/galaxies-nocopy-fixed/environment/run_network.py", B / "environment/run_network.py")
    p = B / "environment/run_network.py"; p.write_text(sub(p.read_text(), "IMAGES: (N, 3, 128, 128)", f"IMAGES: {shape}"))
    p = B / "environment/Dockerfile"; t = p.read_text()
    t = sub(t, "RUN mkdir -p /app/out /app/l0", "RUN mkdir -p /app/out /app/l0 /app/data")
    p.write_text(t.rstrip() + "\nCOPY data/train.npz /app/data/train.npz\nCOPY net.py run_network.py /app/\nCOPY rebuild.py /app/check_rebuild.py\n")
    p = B / "tests/Dockerfile"; p.write_text(sub(p.read_text(), "COPY net.py /tests/net.py", "COPY net.py /tests/net.py\nCOPY rebuild.py /tests/rebuild.py"))
    p = B / "tests/score.py"; t = p.read_text()
    t = sub(t, '''    if weights is None:
        return {"reward": 0.0, "skill": 0.0, "accuracy": 0.0, "valid_output": 0.0}''', '''    if weights is None:
        return {"reward": 0.0, "skill": 0.0, "accuracy": 0.0, "valid_output": 0.0}
    import sys; sys.path.insert(0, str(tests_dir)); from rebuild import rebuild
    rb = rebuild(out_dir, need_network_json=False, lock=(str(tests_dir), "/app", "/logs"))
    print("REBUILD", json.dumps(rb))
    rb_fields = {"rebuild_ok": float(rb["ok"]), "rebuild_source_bytes": float(rb.get("source_bytes", 0)), "rebuild_max_relative_difference": float(rb.get("max_relative_difference", -1))}
    if not rb["ok"]:
        print("INVALID: rebuild failed:", rb["error"])
        return {"reward": 0.0, "skill": 0.0, "accuracy": 0.0, "valid_output": 0.0, **rb_fields}''')
    p.write_text(sub(t, '''**{f"predicted_{c}": float(v) for c, v in enumerate(detail["predicted_share"])}}''', '''**{f"predicted_{c}": float(v) for c, v in enumerate(detail["predicted_share"])}, **rb_fields}'''))
    p = B / "instruction.md"; t = p.read_text()
    ex = t[t.index("EXAMPLES\n"):t.index("THE INPUT\n")]
    t = t.replace(ex, ex + data_section(design_instruction))
    t = sub(t, "They are for looking at. There is no copy of them you can load, and the images you are scored on are different ones.",
            "They are for looking at; the full training set is described below. The images you are scored on are different ones.")
    t = t.replace(t[t.index("RULES. Breaking"):t.index("WHAT YOU HAVE\n")], rules(task, "weights.npz"))
    t = sub(t, "A shell with Python 3 and numpy, and tools to read and write files. There is no internet.", f"""A shell with Python 3 and numpy, and tools to read and write files. There is no internet.
/app/net.py is the scorer's code for this network. python3 /app/run_network.py /app/out IMAGES.npy LABELS.npy runs
your weights on images {shape} with values 0 to 1 and prints the accuracy and confusion table; each such
run is one of your 5 checks.
python3 /app/check_rebuild.py /app/out runs the data-free rebuild check described above.""")
    t = sub(t, """The weights must come from a deterministic process: if any code you run to produce them uses randomness, fix
its seed, so that running the same code again gives the same weights.
Also write""", """Save /app/out/build.py, which writes weights.npz as described under HOW THE RULES ARE CHECKED.
Also write""")
    t = re.sub(r"2\. this same network trained by gradient descent on [^\n]*", "2. this same network trained by gradient descent on the training images.", t)
    p.write_text(t)
    (B / "solution/weights.npz").unlink(missing_ok=True)
    (B / "solution/build.py").write_text('''"""Reference answer, data free: every tensor zero except a tiny hand-set output bias."""
import importlib.util, sys
from pathlib import Path
import numpy as np
NET = """''' + (B / "tests/net.py").read_text().replace('"""', "'''") + '''"""
CLASSES = ''' + str(json.loads((B / "tests/verifier_config.json").read_text())["classes"]) + '''
spec = importlib.util.spec_from_loader("net", loader=None); net = importlib.util.module_from_spec(spec); exec(NET, net.__dict__)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
w = {k: np.zeros(s, np.float32) for k, s in net.SHAPES.items()}
last = [k for k, s in net.SHAPES.items() if tuple(s) == (CLASSES,)][-1]
w[last] = np.linspace(0.1, 0.0, CLASSES).astype(np.float32)
np.savez(out / "weights.npz", **w)
''')
    (B / "solution/solve.sh").write_text('''#!/bin/bash
set -euo pipefail
mkdir -p /app/out
cp "$(dirname "$0")/build.py" /app/out/build.py
python3 /app/out/build.py /app/out
echo "Reference: data-free constant answer." > /app/out/method.md
''')
    toml(B / "task.toml", f"{task}-nocopy-fixed", f"{task}: weights for the fixed published network; full training set to look at; no copying or fitting: build.py must rebuild the weights with no data; at most 5 checks on training images", "full training set; arm weights-code")
    return B, t


if __name__ == "__main__":
    task, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["design", "fixed"]
    built = []
    if "design" in arms: built.append(design(task, out)[0])
    if "fixed" in arms: built.append(fixed(task, out, (DESIGN_BASE / f"{task}-design/instruction.md").read_text())[0])
    if "program" in arms:
        sys.path.insert(0, str(Path(__file__).parent))
        from make_nocopy_program import program
        built.append(program(task, out))
    print("built", *built)
