# Weight reading (Family 1) dataset builders: wsp/wsu-reading v6 → v10

Recovered on 2026-11-21 from the session transcript of the agent that wrote them: they lived in a scratchpad that was cleared. The code is as written then; it has not been rerun since.

| Step | How |
|---|---|
| v6 | Earlier export (`scripts/export_mcq.py`, `scripts/export_atari_which_game.py` and the LoRA/style fetchers). The v6 task folders are not in this repository. |
| v7 | `build_reading.py <v6 tasks> <work>`: rounds every weight to f16, shuffles hidden units (function-preserving, checked with `wsu.py`'s numpy forward passes), rebuilds BERT options, drops two style nets. Uses `hfload.py` for the HF BERT checkpoints. Also writes the value-shuffled control. |
| v8 | v7 minus 6 BERT tasks (done by hand in the shell). |
| v9 | `build_v9_bert.py <work>`: BERT options rebalanced so no two come from one task cluster. |
| v10 | v9 with `openrouter.ai` added to every task's allowed hosts (shell `sed` on `task.toml`). This is the scored set. |

The scored set is the task folders in `../tasks/` plus each task's weights file (`policy.npz`, see the README one level up).
