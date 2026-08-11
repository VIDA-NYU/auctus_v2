"""Minimal Portkey LLM client for the benchmark eval scripts.

The eval scripts (query generation, judging) only need the NYU Portkey gateway —
not the full ``autoddg`` library, which is installed in the worker container but
not necessarily in a local eval venv. So we build the Portkey client directly
from the same env vars ``storage.arq_worker`` uses, and degrade to ``None`` when
the key is missing (caller decides how to fail).

The judge panel spans three labs, and the models do not accept identical call
options: ``@gpt-5-mini/gpt-5-mini`` rejects ``temperature=0`` outright ("Only the
default (1) value is supported"), so that seat's temperature cannot be pinned.
Rather than silently dropping the setting, per-model quirks live in
``MODEL_CALL_OPTIONS`` and ``temperature_pinned()`` reports what was actually
set.

Note the deliberate narrowness of that name. Pinning the temperature is an
*intent*; it is not evidence of stable output, and the two came apart in
measurement — ``gpt-5-mini`` cannot pin its temperature yet reproduced identical
labels across fresh runs (on different reasoning paths), while pinning proves
nothing on its own. Stability is established by measuring repeat-run
self-consistency with the response cache bypassed, and lives in the pilot report
rather than in this module.
"""

from __future__ import annotations

import os

try:
    from portkey_ai import Portkey
except ImportError:  # pragma: no cover
    Portkey = None

PORTKEY_BASE_URL = os.getenv(
    "PORTKEY_BASE_URL", "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"
)
LLM_MODEL = os.getenv("AUTODDG_MODEL", "@vertexai/gemini-2.5-flash")

# Judge panel (see openspec/changes/benchmark-cross-judge-panel/design.md).
# Tier-matched across three originating labs; the ``@vertexai/anthropic.*`` route
# is Google *hosting* an Anthropic model — lineage follows the lab, not the cloud.
GEMINI_FLASH = "@vertexai/gemini-2.5-flash"
GPT5_MINI = "@gpt-5-mini/gpt-5-mini"
CLAUDE_HAIKU = "@vertexai/anthropic.claude-haiku-4-5@20251001"
CLAUDE_SONNET = "@vertexai/anthropic.claude-sonnet-4-6"

MODEL_LAB = {
    GEMINI_FLASH: "Google",
    GPT5_MINI: "OpenAI",
    CLAUDE_HAIKU: "Anthropic",
    CLAUDE_SONNET: "Anthropic",
}

# Per-model call quirks, verified against the gateway.
#   supports_temperature: False -> the API rejects any explicit temperature, so
#     the run uses the provider default (1) and is NOT reproducible.
MODEL_CALL_OPTIONS: dict[str, dict] = {
    GPT5_MINI: {"supports_temperature": False},
}


def get_llm_client():
    """Return a Portkey client, or None if unavailable (no key / not installed)."""
    if Portkey is None:
        return None
    key = os.getenv("PORTKEY_API_KEY")
    if not key:
        return None
    return Portkey(base_url=PORTKEY_BASE_URL, api_key=key)


def supports_temperature(model: str) -> bool:
    """Whether ``model`` accepts an explicit temperature at all."""
    return MODEL_CALL_OPTIONS.get(model, {}).get("supports_temperature", True)


def temperature_pinned(model: str, temperature: float = 0.0) -> bool:
    """Whether the run actually pinned the temperature to 0.

    This records INTENT, not reproducibility. Do not read it as "labels are
    stable": measured self-consistency is the evidence for that, and the two
    diverge in practice (see the module docstring).
    """
    return supports_temperature(model) and temperature == 0.0


def _build_kwargs(model: str, prompt: str, temperature: float,
                  seed: int | None) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if supports_temperature(model):
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


def complete_verbose(client, prompt: str, temperature: float = 0.0,
                     model: str | None = None,
                     seed: int | None = None,
                     no_cache: bool = False) -> tuple[str, dict]:
    """Single-turn completion returning ``(text, usage)``.

    ``usage`` carries prompt/completion/reasoning token counts when the provider
    reports them — the pilot needs reasoning tokens to cost a reasoning model
    honestly, since they are billed but invisible in the response text.

    The NYU gateway caches responses, so an identical prompt replays the previous
    answer. That is fine (and thrifty) for ordinary runs, but it would silently
    fake stability in a self-consistency measurement — measuring the cache
    instead of the model. ``no_cache=True`` forces a fresh generation.
    """
    model = model or LLM_MODEL
    if no_cache:
        client = client.with_options(cache_force_refresh=True)
    resp = client.chat.completions.create(
        **_build_kwargs(model, prompt, temperature, seed)
    )
    raw_usage = getattr(resp, "usage", None)
    usage: dict = {}
    if raw_usage is not None:
        as_dict = raw_usage if isinstance(raw_usage, dict) else dict(raw_usage)
        usage = {
            "prompt_tokens": as_dict.get("prompt_tokens"),
            "completion_tokens": as_dict.get("completion_tokens"),
            "total_tokens": as_dict.get("total_tokens"),
        }
        details = as_dict.get("completion_tokens_details") or {}
        if not isinstance(details, dict):
            details = dict(details)
        usage["reasoning_tokens"] = details.get("reasoning_tokens")
    return resp.choices[0].message.content, usage


def complete(client, prompt: str, temperature: float = 0.0,
             model: str | None = None, seed: int | None = None,
             no_cache: bool = False) -> str:
    """Single-turn completion. temperature 0 by default for reproducibility."""
    text, _ = complete_verbose(client, prompt, temperature, model, seed, no_cache)
    return text
