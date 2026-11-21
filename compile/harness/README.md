# Harness

`api_wsp/` is the agent that queried every model in v1, unchanged. It ran inside each task's sandbox on
[Valkyrie](https://docs.valkyrie.vals.ai) (an agent benchmark runner); `wsc-bench run` runs the same files in Docker on
your machine.

What it does, per task:

1. Sends the prompt (`/instr/instruction.md`) once, with the system prompt `Answer the user's request.`, through the
   model's own API: Anthropic Messages, OpenAI Responses or Gemini; open-weight models through OpenRouter.
2. Continues a reply that was cut at the output cap in its visible text (up to 4 requests, passing the reasoning back).
   A reply cut while reasoning, or on all 4 requests, is out of budget.
3. Runs the reply's single python block with `/app/l0/shapes.json` (the tensor shapes) and writes
   `/app/out/weights.npz`. If the weights hold more nonzero entries than `/app/l0/limits.json` allows, it sends the
   count and the cap as a follow-up in the same conversation, up to 2 times.
4. Exits 3 on infrastructure failures (keys out of credit, model missing, provider errors after 8 attempts) so the
   task is retried, never scored.

`api_wsp/README.md` documents every provider detail (effort settings, model id resolution, continuation). The names
`wsp` and `wsp_sandbox` in the code are the benchmark's working name.

`image/Dockerfile` is the agent sandbox: Python 3.12.8 and numpy 2.5.4.

On Valkyrie the agent's network was limited to the four provider hosts; `wsc-bench run` does not restrict it, which
does not change anything the model sees (it has no tools).
