# claude_code_wsp

Fork of the stock Valkyrie `claude_code-v1.0.0` agent for the WSP evals.

Differences from stock:

* `-k effort <low|medium|high|xhigh|max>` (default `high`) becomes `claude --effort <effort>`.
* `-k tools "<names>"` (default `default`, all tools) becomes `claude --tools <names>`, which limits the built-in tools
  that exist in the session. The CIFAR-10 limited-tools tasks use `-k tools "Read Write Edit"`: the agent can look at
  files and write its answer (in parts, with Edit) but cannot run code.
* Claude Code is pinned to 2.1.325 (`setup.sh`).
* The contract maps `ANTHROPIC_API_KEY` to the org secret `anthropic-api-key`, so runs need no `-s`.
* `summary.json` records `requested_effort`, `tools`, `available_tools` (from the session's init event, so the tool
  limit can be checked per task), the exact `command`, and `cli_version`.
* Sealed L0 (`-k access l0`), the Claude twin of codex_wsp L0, so the two score identically: one reply with no tools at
  all. Command: `claude -p --bare --tools "" --strict-mcp-config --disable-slash-commands --permission-prompts none
  --system-prompt "Answer the user's request." --effort <effort>` run from an empty dir, with
  `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`. Checked against a mock API (2026-11-15, Claude Code 2.1.325): the request
  carries `tools: []`, `thinking: adaptive`, `output_config.effort`, `max_tokens: 128000`, and the system prompt is
  Claude Code's one line SDK identity plus ours; the init event shows `tools: []`, `mcp_servers: []`,
  `slash_commands: []`. The reply's single python block runs with the codex_wsp runner (numpy and math, `weights`
  preloaded with zeros) into `/app/out/weights.npz`. Any `tool_use` block or a session with tools fails the gate
  (no weights, score 0). `-k tools` is ignored.
* Up to 4 model requests per L0 reply (`L0_MAX_REPLY_REQUESTS`). 128k output tokens, thinking included, is the API
  maximum per request, and at max effort a reply often spends all of it thinking. Claude Code answers a `max_tokens`
  stop by continuing in the same turn with up to 3 more requests, so a reply gets up to 4 x 128k. The reply text is
  joined across those requests, so a python block that starts in a cut request and ends in the next one still runs.
  L0 runs with `--include-partial-messages`; the agent kills claude at the 4th request's `max_tokens` stop or if a 5th
  starts, and the reply then fails the gate with no cap retry. Only the message_start/delta/stop stream events are
  kept in `trajectory.jsonl`. After a kill, `cost` is null and the summed tokens are priced by `wsp cost`. (From
  2026-11-15 morning until this change, the agent killed at the first `max_tokens` stop: one request per reply.)
* Nonzero cap (L0): with `/app/l0/limits.json`, an over-cap reply gets the codex_wsp `CAP_RETRY_MESSAGE` (count and cap
  only) in the same conversation via `claude -p ... --resume <session id>` (same sealed flags), at most 3 replies.
  `l0_result.json` has `attempts`; `trajectory.jsonl` has `wsp.cap_retry` events; `metrics_total.json` sums token usage
  over replies and `cost` is the resumed session's cumulative `total_cost_usd`.
  tests/test_claude_code_wsp_cap_retry.py checks the runner, prompt and retry text match codex_wsp byte for byte.

* Free architecture (L0): with `/app/l0/free.json` the reply runs with `L0_FREE_RUNNER` (`net = {}` and `weights = {}`
  preloaded; writes `/app/out/net.json` and `/app/out/weights.npz`) and the cap retry counts nonzero weights against
  `max_active_params`, exactly as codex_wsp; `L0_FREE_RUNNER` is in the byte for byte parity check.

```bash
valkyrie agent push ./agents/claude_code_wsp
valkyrie run start --agent claude_code_wsp --dataset wsp/<dataset>@v1 --provider modal \
  --model anthropic/claude-opus-5-5 -k effort high -k tools "Read Write Edit" --slice 0:1 --concurrency 1 -l <label>
```

Capped L0 experiments go through `uv run wsp valkyrie-run <capped yaml> --model claude-opus-5-5 --effort max [--count 2]`
(routes claude-* models here; the experiment name becomes `<base>-opus-max-nnz<cap>`).
