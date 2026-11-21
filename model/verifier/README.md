# Program verifier source

`wsp-src/wsp/` holds the nine modules of our evaluation package that the program arm's scorer imports
(`tasks/<task>-nocopy-program/tests/score_program.py`): the program sandbox (static checks, a separate restricted
process, time and memory limits) and the metric types. Each program task's `tests/Dockerfile` copies a `wsp-src`
folder next to it; `data/assemble_tasks.py` puts this one there.
