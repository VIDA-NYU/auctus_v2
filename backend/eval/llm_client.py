"""Minimal Portkey LLM client for the benchmark eval scripts.

The eval scripts (query generation, judging) only need the NYU Portkey gateway —
not the full ``autoddg`` library, which is installed in the worker container but
not necessarily in a local eval venv. So we build the Portkey client directly
from the same env vars ``storage.arq_worker`` uses, and degrade to ``None`` when
the key is missing (caller decides how to fail).
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


def get_llm_client():
    """Return a Portkey client, or None if unavailable (no key / not installed)."""
    if Portkey is None:
        return None
    key = os.getenv("PORTKEY_API_KEY")
    if not key:
        return None
    return Portkey(base_url=PORTKEY_BASE_URL, api_key=key)


def complete(client, prompt: str, temperature: float = 0.0) -> str:
    """Single-turn completion. temperature 0 by default for reproducibility."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content
