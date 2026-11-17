# Results

Every WSC-Bench v1 run, collected from Valkyrie by `scripts/collect_valkyrie_results.py`.

## runs.csv

One row per run: one model, one seed, one or more tasks.

| column | meaning |
|---|---|
| run_id | Valkyrie run id |
| label | run label: sweep, model and seed (for example `easy-anthropic-claude-opus-5-5-s1`) |
| model | model id as sent to the harness (`anthropic/`, `openai/`, `google/` or an OpenRouter id) |
| agent | `api_wsp` (the harness) or `oracle` (the reference network, a check of the verifier) |
| dataset | Valkyrie dataset the tasks came from |
| status, started_at, finished_at | run state and UTC times |
| tasks | number of v1 tasks in the run |

## task_results.csv

One row per (run, task).

| column | meaning |
|---|---|
| task | task name (see `wsc-bench list`) |
| status | FINISHED when the verifier scored the reply; anything else is an infrastructure error, retried and not scored |
| reward | the verifier's reward in [0, 1] |
| reward_fixed_count | the reward with the sparse nonzero count fixed: a reply the v1 verifier zeroed for the cap gets its uncapped reward back when its real count (nonzero values plus biases) is within 500,000. Otherwise equal to `reward` |
| out_of_budget | the reply ran out of output tokens (cut while reasoning, cut on all 4 requests, or scored 0 after a cut) |
| provider_fail | the reply scored 0 because the provider blocked or failed it (SAFETY, finish error, finish blocked) |
| cut | at least one request of the reply was cut at the output cap |

Not every row counts toward the published numbers: runs include pilots, effort sweeps and reruns.
`blog_v1_runs.csv` lists the ones that do. For regex-match and decimal-period, `reward` comes from the v1 verifier's
nonzero count, which over-counted sparse submissions; `reward_fixed_count` corrects it (see Known issues in the
top-level README).

## local_task_results.csv

Part of the frontier sweep ran on a local Docker runner (linux/amd64) instead of Valkyrie, with the same harness,
prompt and verifier. One row per (local run, task), with the same reward columns as `task_results.csv`. `label`
identifies the run (it ends in `-local-sN`).

## blog_v1_runs.csv

Every reply behind the numbers and charts in the blog post: one row per (model, task, reply), at most 3 replies per
model and task. `source` is `valkyrie` (then `run` is the Valkyrie run id) or `local` (then `run` is the local run
label). Scores use `reward_fixed_count`; out-of-budget replies count as 0, infrastructure errors are left out.
`python scripts/blog_v1_scores.py` recomputes each model's easy and hard tier score from this file.
