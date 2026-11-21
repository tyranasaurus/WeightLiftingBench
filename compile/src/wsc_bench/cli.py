"""wsc-bench: list tasks, print a prompt, turn a reply into weights, and score them.

    wsc-bench list
    wsc-bench prompt gcd10 > prompt.md
    wsc-bench eval gcd10 reply.md            # build weights from the reply, then score them
    wsc-bench score gcd10 weights.npz        # score weights you already have
    wsc-bench run gcd10 --model anthropic/claude-opus-5-5   # query a model with the v1 harness (Docker)
"""

import argparse
import json
from pathlib import Path

from wsc_bench.benchmark import TASKS, get_task
from wsc_bench.build import build_weights, count_nonzero
from wsc_bench.score import score_weights


def cmd_list(_: argparse.Namespace) -> None:
    print(f"{'task':<22}{'tier':<6}{'cap':>10}  seeds")
    for task in TASKS.values():
        print(f"{task.name:<22}{task.tier:<6}{task.max_nonzero:>10,}  {task.eval_seeds}")


def cmd_prompt(args: argparse.Namespace) -> None:
    print(get_task(args.task).prompt(), end="")


def report(*, task_name: str, weights_path: Path, first_seed: int) -> None:
    task = get_task(task_name)
    result = score_weights(task=task, weights_path=weights_path, first_seed=first_seed)
    out = {"task": task.name, "reward": round(result.reward, 4), "valid_output": result.valid_output, "reason": result.reason}
    for key in ("nonzero_weights", "max_nonzero", "within_budget"):
        if key in result.detail:
            out[key] = result.detail[key]
    print(json.dumps(out, indent=2))


def cmd_score(args: argparse.Namespace) -> None:
    report(task_name=args.task, weights_path=args.weights, first_seed=args.seed)


def cmd_eval(args: argparse.Namespace) -> None:
    task = get_task(args.task)
    out_path = args.out or args.reply.with_suffix(".npz")
    built = build_weights(reply=args.reply.read_text(), shapes=task.build().tensor_shapes(), out_path=out_path)
    if not built.ok:
        print(json.dumps({"task": task.name, "reward": 0.0, "valid_output": False, "reason": built.reason}, indent=2))
        return
    import numpy as np

    with np.load(out_path) as archive:
        used = count_nonzero({k: archive[k] for k in archive.files})
    if used > task.max_nonzero:
        print(f"note: {used:,} nonzero entries, over the {task.max_nonzero:,} cap (the harness would ask for a resubmission)")
    report(task_name=task.name, weights_path=out_path, first_seed=args.seed)


def cmd_run(args: argparse.Namespace) -> None:
    from wsc_bench.run import run_model

    tasks = list(TASKS.values()) if args.task == "all" else [get_task(args.task)]
    for task in tasks:
        print(json.dumps(run_model(task=task, model=args.model, effort=args.effort, route=args.route, out_root=args.out), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="wsc-bench", description="Weight Space Compilation Bench v1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the 14 tasks").set_defaults(func=cmd_list)
    p = sub.add_parser("prompt", help="print a task's exact prompt")
    p.add_argument("task")
    p.set_defaults(func=cmd_prompt)
    p = sub.add_parser("eval", help="run a reply's python block and score the weights it builds")
    p.add_argument("task")
    p.add_argument("reply", type=Path, help="text file with the model's full reply")
    p.add_argument("--out", type=Path, help="where to write the weights (default: next to the reply)")
    p.add_argument("--seed", type=int, default=0, help="first evaluation seed (default 0)")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("score", help="score a weights .npz")
    p.add_argument("task")
    p.add_argument("weights", type=Path)
    p.add_argument("--seed", type=int, default=0, help="first evaluation seed (default 0)")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("run", help="query a model on a task with the v1 harness (needs Docker and API keys)")
    p.add_argument("task", help="task name, or all")
    p.add_argument("--model", required=True, help="anthropic/<id>, openai/<id>, google/<id>, or an OpenRouter id")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh"])
    p.add_argument("--route", default="auto", choices=["auto", "openrouter"])
    p.add_argument("--out", type=Path, default=Path("runs"), help="output folder (default: runs/)")
    p.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
