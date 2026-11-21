"""Make the task folders buildable: put each task's training and test data, and the program verifier source, where the
task's Dockerfiles expect them.

    python model/data/assemble_tasks.py DATA_DIR

DATA_DIR holds the 14 files of the data release (<task>-train.npz, <task>-test.npz; see data/README.md). For every task
folder in model/tasks this copies
    <task>-train.npz -> environment/data/train.npz
    <task>-test.npz  -> tests/hidden.npz
and, for the program tasks, model/verifier/wsp-src -> tests/wsp-src. The copies are ignored by git.
The fewer-images tasks are built from the assembled tasks by fewer-images/code/make_nocopy10.py.
"""
import shutil, sys
from pathlib import Path

M = Path(__file__).resolve().parent.parent
data = Path(sys.argv[1])
for task in sorted((M / "tasks").iterdir()):
    name = task.name.split("-")[0]
    (task / "environment/data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data / f"{name}-train.npz", task / "environment/data/train.npz")
    shutil.copyfile(data / f"{name}-test.npz", task / "tests/hidden.npz")
    if task.name.endswith("-program"):
        shutil.copytree(M / "verifier/wsp-src", task / "tests/wsp-src", dirs_exist_ok=True)
    print("assembled", task.name)
