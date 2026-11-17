#!/usr/bin/env python3
"""API key pool for Valkyrie agents: pick a key with credit left when a task starts, and spot a run out of credit.

The org secret (openai-api-key / anthropic-api-key) holds one key or many, separated by newlines, spaces or commas.
Each task probes the keys from a random starting point with a tiny request to the run's model and uses the first one that is not out of
credit. Only an explicit out-of-credit or bad-key answer marks a key dead: any other failure (rate limit, network,
an unknown model) counts as alive, so a probe problem never blocks a run.

A key that runs dry mid-task makes the agent exit nonzero (see `out_of_credit`), so Valkyrie marks the task ERROR
and `scripts/valkyrie_watchdog.py` retries it in a fresh sandbox, where the probe skips the dead key.

This file is stdlib only and is copied verbatim into agents/codex_wsp and agents/claude_code_wsp (each agent bundle
is its own directory). tests/test_key_pool.py checks the copies match.

Check a pool locally (keys on stdin, printed masked):
    python3 agents/codex_wsp/key_pool.py openai < openai-keys.txt
    python3 agents/codex_wsp/key_pool.py anthropic < anthropic-keys.txt
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.error
import urllib.request

PROBE_TIMEOUT_S = 30
PROVIDERS = ("openai", "anthropic")
# The probe uses the run's own model (known to exist and be allowed for the key) with a 16 / 1 token cap, a fraction
# of a cent. These defaults are only for the local check command.
CHECK_MODELS = {"openai": "gpt-6-luna", "anthropic": "claude-haiku-5-5"}
# Lowercase substrings that mean the key has no credit left or is invalid. OpenAI: 429 insufficient_quota.
# Anthropic: 400 "credit balance is too low" or "reached your specified [workspace] API usage limits" (seen live
# 2026-11-15 on a maxed key), 401/403 for a bad key.
OUT_OF_CREDIT_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "api usage limits",
    "billing_error",
    "billing details",
)
BAD_KEY_STATUSES = {401, 403}


def parse_pool(value: str) -> list[str]:
    return [key for key in re.split(r"[\s,]+", value) if key]


def mask(key: str) -> str:
    return f"{key[:10]}...{key[-4:]}"


def probe_request(*, provider: str, key: str, model: str) -> urllib.request.Request:
    if provider == "openai":
        body = {"model": model, "input": "ok", "max_output_tokens": 16}
        return urllib.request.Request(
            "https://api.openai.com/v1/responses", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    body = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "ok"}]}
    return urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(), method="POST",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"})


def probe(*, provider: str, key: str, model: str) -> tuple[bool, str]:
    """(alive, reason). Dead only on an explicit bad-key status or out-of-credit message."""
    try:
        with urllib.request.urlopen(probe_request(provider=provider, key=key, model=model), timeout=PROBE_TIMEOUT_S) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", "replace")
        if error.code in BAD_KEY_STATUSES:
            return False, f"HTTP {error.code} bad key: {text[:200]}"
        if any(marker in text.lower() for marker in OUT_OF_CREDIT_MARKERS):
            return False, f"HTTP {error.code} out of credit: {text[:200]}"
        return True, f"HTTP {error.code} (not a credit error, treated as alive): {text[:200]}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return True, f"probe failed, treated as alive: {error}"


def pick_key(*, provider: str, pool_value: str, model: str) -> str:
    """First key with credit, starting at a random index so concurrent tasks spread across the pool."""
    keys = parse_pool(pool_value)
    if not keys:
        raise SystemExit(f"key pool for {provider} is empty")
    start = random.randrange(len(keys))
    for index in [*range(start, len(keys)), *range(start)]:
        alive, reason = probe(provider=provider, key=keys[index], model=model)
        # Position only, no key characters: sandbox logs are shared with the whole org.
        print(f"key_pool {provider} key {index + 1} of {len(keys)}: {'alive' if alive else 'dead'}: {reason}",
              flush=True)
        if alive:
            return keys[index]
    raise SystemExit(f"key_pool: every {provider} key ({len(keys)}) is out of credit or invalid")


def out_of_credit(error_texts: list[str]) -> bool:
    """True when any agent error message says the key ran out of credit."""
    return any(marker in text.lower() for text in error_texts for marker in OUT_OF_CREDIT_MARKERS)


def main() -> int:
    provider = sys.argv[1] if len(sys.argv) > 1 else ""
    if provider not in PROVIDERS:
        raise SystemExit("usage: key_pool.py openai|anthropic [model] < keys.txt")
    model = sys.argv[2] if len(sys.argv) > 2 else CHECK_MODELS[provider]
    keys = parse_pool(sys.stdin.read())
    alive_count = 0
    for key in keys:
        alive, reason = probe(provider=provider, key=key, model=model)
        alive_count += alive
        print(f"{mask(key)}  {'ALIVE' if alive else 'DEAD '}  {reason}")
    print(f"{alive_count}/{len(keys)} {provider} keys alive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
