"""Which API route a Claude Code session uses.

claude_code_wsp_k3 (direct_first): the Anthropic API with ANTHROPIC_API_KEY, for Claude models.
claude_code_wsp_or: OpenRouter with OPENROUTER_API_KEY, for every model.
One key per route. The model's own commands only ever see the key in use."""
import os, re
from typing import Any, Optional

KEY_VARS = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# Lowercase substrings of an Anthropic or OpenAI error that mean the key has no credit left.
OUT_OF_CREDIT_MARKERS = ("insufficient_quota", "exceeded your current quota", "credit balance is too low", "api usage limits",
                         "billing_error", "billing details")


def out_of_credit(error_texts: list[str]) -> bool:
    """True when any agent error message says the key ran out of credit."""
    return any(marker in text.lower() for text in error_texts for marker in OUT_OF_CREDIT_MARKERS)


def openrouter_model(model: str) -> str:
    """claude-opus-5-5 -> anthropic/claude-opus-5.5; an id that already has a provider prefix is kept."""
    if "/" in model:
        return model
    return "anthropic/" + re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)


def direct_model(model: str) -> str:
    """anthropic/claude-opus-5.5 -> claude-opus-5-5 (the Anthropic API id)."""
    name = model.split("/")[-1]
    return re.sub(r"-(\d+)\.(\d+)$", r"-\1-\2", name)


def route_for(model: str, *, direct_first: bool) -> dict[str, Any]:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if direct_first and anthropic_key and (model.startswith("anthropic/") or "/" not in model):
        return {"name": "anthropic", "key": anthropic_key, "model": direct_model(model)}
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        raise SystemExit("routes: no API key set (ANTHROPIC_API_KEY or OPENROUTER_API_KEY)")
    return {"name": "openrouter", "key": openrouter_key, "model": openrouter_model(model)}


def env_for(route: dict[str, Any]) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items() if k not in KEY_VARS and k != "PYTHONSAFEPATH"}
    if route["name"] == "anthropic":
        return {**base, "ANTHROPIC_API_KEY": route["key"], "ANTHROPIC_MODEL": route["model"]}
    return {**base, "ANTHROPIC_BASE_URL": "https://openrouter.ai/api", "ANTHROPIC_AUTH_TOKEN": route["key"], "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": route["model"], "ANTHROPIC_DEFAULT_HAIKU_MODEL": "anthropic/claude-haiku-5.5",
            "ANTHROPIC_SMALL_FAST_MODEL": "anthropic/claude-haiku-5.5"}


class RouteState:
    """The route in use. There is one route and one key, so a key that runs out of credit ends the task."""

    def __init__(self, model: str, *, direct_first: bool):
        self.route = route_for(model, direct_first=direct_first)
        self.log: list[dict[str, Any]] = [{"route": self.route["name"], "reason": "start"}]
        self.exhausted = False

    def start(self) -> dict[str, str]:
        return self.env()

    def switch(self) -> bool:
        """There is no other route to move to."""
        self.exhausted = True
        return False

    def env(self) -> dict[str, str]:
        return env_for(self.route)

    def current(self) -> dict[str, Any]:
        return {"route": self.route["name"], "model": self.route["model"]}
