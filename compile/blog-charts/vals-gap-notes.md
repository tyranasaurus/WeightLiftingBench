# Is the WeightLiftingBench gap bigger than on other indexes? (checked 2026-11-17, raw page HTML)

"Best US" = best OpenAI or Anthropic entry. "Best Chinese" = best Qwen / Kimi / GLM / DeepSeek entry (Google excluded).

| Index | Best US | Best Chinese | Ratio | URL | Date |
|---|---|---|---|---|---|
| Vals.ai Index | Fable 5.2 71.31 | Qwen 4 Max 63.20 | 1.13x (8.1 pts) | https://www.vals.ai/benchmarks/vals_index | page "Updated 11/16/2026" |
| Artificial Analysis Intelligence Index | Fable 5.2 (Max) 64.56 | Kimi K3.1 Pro 46.5 | 1.39x (18 pts) | https://artificialanalysis.ai/leaderboards/models | no date on page; fetched 11/17/2026 |
| Terminal-Bench 4.1 | Opus 5.5 (max) 65.8% = Fable 5.2 (max) 65.8% | Kimi K3.1 Pro (max) 59.7% | 1.10x (6.1 pts) | https://www.tbench.ai/leaderboard | no "updated" date; newest entry Nov 17, 2026; fetched 11/17/2026 |
| WeightLiftingBench hard tier | GPT-6 Astra 94% | Kimi K3.1 Pro / Qwen3.8 Max 29% | 3.24x (65 pts) | ours | data as of Nov 17, 2026 |

## Verification (raw HTML via curl, not a summary)
- Vals: scores read from the `aria-label="<model> — NN.NN% accuracy"` bars and the embedded JSON. Confirmed Fable 5.2 71.31, GPT-6.2 Astra 67.66, Qwen 4 Max 63.20, Kimi K3.1 Pro 62.52, GLM 5.3 59.85, DeepSeek V4.1 Flash 1017 61.26. Earlier full list (Opus 5.5 70.14, Sonnet 5.5 70.85, Sol 66.06 etc.) came from the summarized fetch; the ones above are raw-confirmed. Gemini 4 Carbon (70.45) is on the page too, left out per house style.
- Artificial Analysis: scores from the embedded JSON (`intelligenceIndex`). Fable 5.2 Max 64.56 (earlier summary rounded to 65), GPT-6.2 Astra Max 62.1, Opus 5.5 Max 59.1, GPT-6.2 Sol Max 57.3, Kimi K3.1 Pro 46.5, Qwen4 Max 42.9, DeepSeek V4.1 Flash Max 32.6, GLM-5.3 Max 31.7. The JSON has an `isOpenWeights` flag: Kimi K3.1 Pro, GLM-5.3 and DeepSeek are open, Qwen4 Max is NOT. Each model appears at several effort levels; I used each model's highest.
- Terminal-Bench 4.1 (page text, tbench.ai): #1 is "Copilot Fusion" (GitHub Copilot, 71.2%, Nov 17, 2026); I excluded it as a multi-model product, not a single OpenAI/Anthropic model. Anthropic entries all use Claude Code (Opus 5.5 65.8, Fable 5.2 65.8, Sonnet 5.5 63.9, Fable 5.1 57.6). OpenAI entries all use Codex (GPT-6.1 Sol 60.9, GPT-6 Astra 60.3, GPT-6.1 Terra 53.3). Chinese entries mostly use Claude Code: Kimi K3.1 Pro 59.7, GLM-5.3 41.2, DeepSeek V4.1 Flash 38.5, Qwen3.8-Max-0902 27.0 (Qwen4 Max is not listed). A few others use mini-swe-agent (MiMo, MiniMax). Harnesses are mixed across labs, but one entry per model; best-vs-best uses Claude Code on both sides. GPT-6.2 Sol/Astra are not on the board.

## Verdict
All three other indexes support "much bigger gap on WeightLiftingBench": 1.1x, 1.4x and 1.1x versus 3.2x. Terminal-Bench is the narrowest of them, and AA the widest. Nothing contradicts the claim.

One-sentence claim the numbers support: "On Vals.ai, Artificial Analysis and Terminal-Bench the best OpenAI or Anthropic model beats the best Chinese model by only 1.1x to 1.4x, while on WeightLiftingBench's hard tier it is 3.2x (94% vs 29%)."

Caveats: Qwen4 Max and Kimi K3.1 Pro weights are not yet released per our own footer, although AA flags Kimi K3.1 Pro as open weights and Qwen4 Max as not. If you restrict to released weights, the Chinese side drops a bit on Vals (DeepSeek V4.1 Flash 61.26) and Terminal-Bench (Kimi K3.1 Pro is still the best) but the picture stays the same. WeightLiftingBench's best Chinese score is a tie at 29% between Kimi K3.1 Pro and Qwen3.8 Max (Qwen4 Max is 27%); the chart bar is labelled "Kimi K3.1 / Qwen3.8 Max".

Script: plot_vals_gap.py  Image: figures/vals-gap.png
