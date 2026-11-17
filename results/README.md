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
| out_of_budget | the reply ran out of output tokens (cut while reasoning, cut on all 4 requests, or scored 0 after a cut) |
| provider_fail | the reply scored 0 because the provider blocked or failed it (SAFETY, finish error, finish blocked) |
| cut | at least one request of the reply was cut at the output cap |

Not every row counts toward the published numbers: runs include pilots, effort sweeps and reruns. The blog post
states which runs it uses. For regex-match and decimal-period the reward comes from the v1 verifier's nonzero count,
which over-counted sparse submissions (see Known issues in the top-level README).
