# api_wsp

Sealed L0 agent for every model, one harness for all providers: same prompt, same single sealed request flow, same
continuation rule, same runner, checker and limit retries, same `/logs` outputs as codex_wsp, claude_code_wsp and
openrouter_wsp. Only the HTTP endpoint differs.

| model id | route | keys |
|---|---|---|
| `anthropic/<model>` | Anthropic Messages API | `ANTHROPIC_API_KEY` pool |
| `openai/<model>` | OpenAI Responses API | `OPENAI_API_KEY` pool |
| `google/<model>` | Gemini API `streamGenerateContent` | `GEMINI_API_KEY` pool |
| anything else (open weights) | OpenRouter chat completions | `OPENROUTER_API_KEY` |

* Closed models (anthropic, openai, google) only ever use their native API (no OpenRouter fallback) unless
  `-k route openrouter` is set.
* Native ids come from the provider's model list: dots to dashes for Claude (`anthropic/claude-opus-4.6` is
  `claude-opus-4-6`), then the exact id or else the newest dated one (`claude-opus-4-1` is `claude-opus-4-1-20250805`,
  `claude-opus-4` is `claude-opus-4-20250514`). Gemini ids are looked up in `GET /v1beta/models` (`models/` prefix
  stripped). `summary.json` records `verified` per route.
* Gemini key placement: `x-goog-api-key` header first, then `?key=`, then a bearer token, whichever the model list call
  accepts first (keys that start with `AQ.` may need either). Logs never print a URL query.
* Infra vs score: a key out of credit (key_pool.py markers, HTTP 401/403, Gemini `API_KEY_INVALID` or a daily or
  billing `RESOURCE_EXHAUSTED`, OpenRouter 402) moves to the next pooled key. Every key out, the model missing (not
  listed, or a 404 / not found answer), any other HTTP error, or a transient error (408, 409, 429, 5xx, network, a
  broken stream) still failing after 8 attempts exits 3: the task is an ERROR that can be retried, never a scored 0.
  Only a real reply that breaks the rules is scored as a failure: no single python block, a crash, over the cap after
  the retries, a cut on all 4 requests, a tool call, or a refusal or blocked prompt answered by the model.
* Effort (`-k effort low|medium|high|xhigh`, default high), one setting for every route: Anthropic adaptive thinking +
  `output_config.effort` (Claude 4.6 and later; earlier Claude gets a thinking budget, share x max_tokens with
  OpenRouter's shares, 0.8 for high); OpenAI `reasoning: {effort, summary: "detailed"}` (as codex_wsp); Gemini 3 and
  later `thinkingConfig.thinkingLevel` (xhigh sends high), Gemini 2.x `thinkingBudget` = share x 32768;
  OpenRouter `reasoning: {effort}`.
* `max_tokens` is the model's limit capped at 128000 (Anthropic `max_tokens`, Gemini `outputTokenLimit`, OpenRouter
  `top_provider.max_completion_tokens`; OpenAI 128000).
* Continuation, the same rule on every route. A request cut at the output cap (Anthropic `max_tokens`, OpenAI
  `incomplete` / `max_output_tokens`, Gemini `MAX_TOKENS`, OpenRouter `length`):
  * cut while reasoning (no visible text yet, or reasoning was the last thing emitted): not continued, since a
    continuation could not resume the reasoning. The reply fails the gate with
    `the reply hit the max_tokens output cap while reasoning`.
  * cut in the visible text: continued with `CONTINUE_MESSAGE`, passing back the full assistant turn with its
    reasoning: Anthropic thinking blocks with signatures, OpenAI reasoning items with `encrypted_content` (requests
    send `include: ["reasoning.encrypted_content"]`, `store: false`; ids dropped as codex does), Gemini parts with
    `thoughtSignature`, OpenRouter `reasoning_details`. When the route gave nothing to resume from (OpenRouter
    reasoning text without `reasoning_details`) the turn goes back as text and `continuation_reasoning` is false.
  * up to 4 requests per reply; a reply cut in its text on all 4 fails the gate with
    `the reply hit the max_tokens output cap on all 4 requests`.
  * `summary.json` `requests[]` records `cut` (reasoning, text or null) and `reasoning_passback` per request, and
    `continuation_reasoning` (null when nothing was continued).
* `-k route auto|openrouter` (default auto). openrouter sends any id through OpenRouter only, no native route: for
  models retired from their provider's API (`anthropic/claude-opus-4.1`, OpenRouter id `anthropic/claude-opus-4.1`).
* `key_pool.py` is a verbatim copy of agents/claude_code_wsp/key_pool.py.

`summary.json` records `routes_planned` and, per request, `route` (anthropic, openai, google or openrouter), served
`model`, `provider` and `id`. `metrics_total.json` sums tokens in codex semantics; `cost` is OpenRouter's billed USD
(null for native routes).

Network: the task's agent allowlist must hold `api.anthropic.com`, `api.openai.com`, `generativelanguage.googleapis.com`
and `openrouter.ai` (wsp/program-l0-capped@15 and later; `AGENT_ALLOWED_HOSTS` in src/wsp/harbor/export.py).

```bash
valkyrie agent push ./agents/api_wsp
valkyrie run start --agent api_wsp --model google/gemini-2.5-pro --dataset wsp/program-l0-capped@15 --provider modal \
  --task-ids <ids> --concurrency 8 -k effort high -s ANTHROPIC_API_KEY anthropic-api-key \
  -s OPENAI_API_KEY openai-api-key -s OPENROUTER_API_KEY openrouter-api-key -s GEMINI_API_KEY gemini-api-key -l <label>
```
