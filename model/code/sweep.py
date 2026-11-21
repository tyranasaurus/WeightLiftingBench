"""No-copy sweep, 16 models: own-design arm for every model x 7 tasks x 3 runs, plus a program arm for GPT 6.2 Astra x 7 x 3.
Existing galaxies and letters design runs (runs.txt) count as run 1 for the first four OpenAI models.
Order: every run 1 first, then runs 2, then runs 3; slow models first within a round. At most GLOBAL_CAP runs in flight,
and at most CAP[model] per model. Runs refused for rate limits, credit or sandbox trouble are retried (valkyrie run retry)
up to MAX_RETRY times; finished runs are collected into outs2/. Any exception in a pass is logged and the loop goes on.
State: sweep_state.json (one record per job). Log: sweep.log. Pause: touch PAUSED."""
import json, os, re, shutil, subprocess, time, traceback
from pathlib import Path

W = Path(os.environ.get("NOCOPY_WORKDIR", "."))  # working folder for run state, logs and collected outputs
DATASET = "wsp/nocopy@5"
TASK_SUFFIX = {"glm53": "-text"}
DESIGN_SLIM = "-slim"  # design runs carry only build.py and network.json; the scorer regenerates weights.npz (large networks overflow the artifact limit)  # text-only models get the variant that says not to open image files
OR, K3 = "claude_code_wsp_or", "claude_code_wsp_k3"
# key: (model id, agent, mean minutes per run in earlier design runs, used to start slow models first)
MODELS = {
    "astra": ("openai/gpt-6.2-astra", OR, 13), "sol61": ("openai/gpt-6.1-sol", OR, 22), "sol62": ("openai/gpt-6.2-sol", OR, 6),
    "sol56": ("openai/gpt-5.6-sol", OR, 27), "astra6": ("openai/gpt-6-astra", OR, 11), "gpt55": ("openai/gpt-5.5", OR, 30),
    "gpt54": ("openai/gpt-5.4", OR, 30), "fable": ("claude-fable-5-2", K3, 43), "fable51": ("claude-fable-5-1", K3, 60),
    "opus55": ("claude-opus-5-5", K3, 58), "opus48": ("claude-opus-4-8", K3, 53), "opus47": ("claude-opus-4-7", K3, 55),
    "glm53": ("z-ai/glm-5.3", OR, 14), "qwen38": ("qwen/qwen3.8-max-0902", OR, 60), "qwen4max": ("qwen/qwen4-max", OR, 60),
    "kimi": ("moonshotai/kimi-k3.1-pro", OR, 19),
}
CAP = {m: 6 for m in MODELS} | {"astra": 4}
GLOBAL_CAP = 45
TASKS = ["digits", "clothing", "cifar", "letters", "traffic", "galaxies", "person"]
MAX_RETRY = 3
INFRA = re.compile(r"error_status\":429|Rate limit exceeded|rate_limit\"|Key limit exceeded|no API key with credit|error_status\":402|HTTP 402|insufficient credits|overloaded|503 Service"
                   r"|502 Bad Gateway|500 Internal|[Ss]andbox (creation|startup|timed out)|ModalError|Connection reset|ECONNRESET")
STATE = W / "sweep_state.json"
RUNNING = ("In Progress", "Queued", "Pending", "Running", "Starting", "Stopping")


def sh(cmd, timeout=600):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL).stdout
    except subprocess.TimeoutExpired:
        log(f"timeout after {timeout} s: {cmd[:120]}"); return ""


def log(msg):
    with open(W / "sweep.log", "a") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def add_jobs(S):
    have = {(j["model"], j["task"], j["arm"], j["run"]) for j in S}
    existing = {}
    for line in (W / "runs.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 3 and p[0] in MODELS:
            task, arm = p[1].split("-") if "-" in p[1] else ("galaxies", p[1])
            if arm == "design": existing[(p[0], task)] = p[2]
    for task in TASKS:
        for model in MODELS:
            for run in (1, 2, 3):
                if (model, task, "design", run) in have: continue
                j = {"model": model, "task": task, "arm": "design", "run": run, "id": None, "status": "queued", "tries": 0}
                if run == 1 and (model, task) in existing: j.update(id=existing[(model, task)], status="done-earlier")
                S.append(j)
        for run in (1, 2, 3):
            if ("astra", task, "program", run) not in have:
                S.append({"model": "astra", "task": task, "arm": "program", "run": run, "id": None, "status": "queued", "tries": 0})
    return S


def key(j): return f'{j["model"]}-{j["task"]}-{j["arm"]}-{j["run"]}'


def launch(j):
    mid, agent, _ = MODELS[j["model"]]
    label = f'nocopy2-{j["task"]}-{j["arm"]}-{j["model"]}-r{j["run"]}'
    o = sh(f"valkyrie run start --agent {agent} --dataset {DATASET} --provider modal --task-ids {j['task']}-nocopy-{j['arm']}{(TASK_SUFFIX.get(j['model'], '') + DESIGN_SLIM) if j['arm'] == 'design' else ''} "
           f"--model {mid} --concurrency 1 -l {label} -k effort xhigh -k tools 'Bash Read Write Edit Glob Grep'")
    m = re.search(r"run retry ([0-9a-f-]{36})", o)
    if not m: log(f"launch failed {key(j)}: {o[-300:]}"); return False
    j.update(id=m.group(1), status="running", tries=j["tries"] + 1); log(f"launched {key(j)} {j['id']}")
    return True


def slim(o):
    """Drop the duplicate tarball and any answer array over 20 MB (rebuildable from build.py, or downloadable again)."""
    for f in o.rglob("raw_output.txt"): f.unlink()  # a second copy of the session stream
    keep = {"build.py", "solution.py", "trajectory.jsonl", "log.txt", "method.md", "checks.log", "network.json"}
    for f in list(o.rglob("out.tar.gz")) + [f for f in o.rglob("*") if f.is_file() and f.name not in keep and (f.suffix == ".npy" or f.stat().st_size > 20 << 20)]:
        (f.parent / (f.name + ".size")).write_text(str(f.stat().st_size)); f.unlink()


def forget_reading(k):
    """A run being collected again (retry or requeue) needs a fresh digest and verdict."""
    for f in (W / "reader/digests" / f"{k}.txt", W / "reader/verdicts" / f"{k}.json"):
        f.unlink(missing_ok=True)
    a = W / "reader/assigned.txt"
    if a.exists(): a.write_text("".join(l + "\n" for l in a.read_text().split() if l != k))


def collect(j, st):
    forget_reading(key(j))
    o = W / "outs2" / key(j)
    if o.exists(): shutil.rmtree(o)
    o.mkdir(parents=True, exist_ok=True)
    sh(f"valkyrie run outputs {j['id']} --output-dir {o}", 1200)
    slim(o)
    logs = sh(f"valkyrie run logs {j['id']}", 600); (o / "log.txt").write_text(logs)
    acc = re.findall(r'"accuracy": ([0-9.]+)', logs); rb = re.findall(r'"rebuild_ok": ([0-9.]+)', logs)
    j["accuracy"] = float(acc[-1]) if acc else None; j["rebuild_ok"] = (float(rb[-1]) == 1.0) if rb else None
    cl = list(o.glob("*/artifacts/app/out/out/checks.log")); j["checks"] = len(cl[0].read_text().splitlines()) if cl else 0
    subprocess.run(["gzip", "-f", str(o / "log.txt")])  # scores are parsed; the log repeats the session stream
    if st.startswith("Finished") and j["accuracy"] is not None:
        j["status"] = "finished"; log(f"finished {key(j)} acc {j['accuracy']} rebuild {j['rebuild_ok']} checks {j['checks']}"); return
    if "support image input" in logs and j["model"] in TASK_SUFFIX and j["tries"] <= MAX_RETRY + 1:
        j.update(status="queued", id=None); log(f"requeued {key(j)} on the text-only task after an image input crash"); return
    if re.search(r"image length and width do not meet|image exceeds|output token maximum|Could not process image|invalid image", logs) and j["tries"] <= MAX_RETRY:
        j.update(status="queued", id=None); log(f"requeued {key(j)} after the provider rejected an image the model opened"); return
    if "can't be carried" in logs and j["arm"] == "design" and j["tries"] <= MAX_RETRY + 1:
        j.update(status="queued", id=None); log(f"requeued {key(j)} on a slim task after its answer was too large to carry"); return
    if INFRA.search(logs) and j["tries"] <= MAX_RETRY:
        sh(f"valkyrie run retry {j['id']}"); j["status"] = "running"; j["tries"] += 1; log(f"retried {key(j)} after an infrastructure error (try {j['tries']})"); return
    j["status"] = "error"; log(f"error {key(j)} [{st}]: {(re.findall(r'INVALID.{0,160}|AgentRunFailedError.{0,120}', logs) or [''])[-1]}")


def status_of(ids):
    out = {}
    for i in range(0, len(ids), 15):
        st = sh("valkyrie run status --ids " + ",".join(ids[i:i + 15]))
        for line in st.splitlines():
            m = re.match(r"^([0-9a-f-]{36})\s+(\S+(?: \S+)?)", line)
            if m: out[m.group(1)] = m.group(2)
    return out


def one_pass(S):
    running = [j for j in S if j["status"] == "running"]
    st = status_of([j["id"] for j in running]) if running else {}
    for j in running:
        s = st.get(j["id"])
        if s and not s.startswith(RUNNING):
            collect(j, s); STATE.write_text(json.dumps(S, indent=1))
    order = sorted([j for j in S if j["status"] == "queued"], key=lambda j: (j["run"], -MODELS[j["model"]][2], j["arm"] != "design", j["task"]))
    inflight = sum(1 for j in S if j["status"] == "running")
    per = {m: sum(1 for j in S if j["model"] == m and j["status"] == "running") for m in MODELS}
    for j in order:
        if inflight >= GLOBAL_CAP: break
        if per[j["model"]] >= CAP[j["model"]]: continue
        if launch(j): inflight += 1; per[j["model"]] += 1
    STATE.write_text(json.dumps(S, indent=1))


def main():
    S = add_jobs(json.loads(STATE.read_text()) if STATE.exists() else [])
    STATE.write_text(json.dumps(S, indent=1)); log(f"start: {len(S)} jobs")
    while True:
        if (W / "PAUSED").exists(): time.sleep(60); continue
        try:
            one_pass(S)
        except Exception:
            log("pass failed: " + traceback.format_exc().replace("\n", " | ")[-600:])
        if not [j for j in S if j["status"] in ("queued", "running")]: log("ALL DONE"); break
        time.sleep(90)


if __name__ == "__main__":
    main()
