"""Shared paths, LLM access (cached) and storage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

# Everything the eval reads or writes lives under DATA_DIR:
#   gold/<record>.json   curator metadata + extracted data dictionaries (hand-verified)
#   collected/<record>.json  AutoDDG + profiler output per file
#   cache/llm/<hash>.json    every LLM response, so reruns make no new calls
#   results/                 comparison output and the human-check sheet
DATA_DIR = Path(os.getenv("DESCRIPTION_EVAL_DATA", Path(__file__).parent / "data"))
GOLD_DIR = DATA_DIR / "gold"
COLLECTED_DIR = DATA_DIR / "collected"
CACHE_DIR = DATA_DIR / "cache" / "llm"
RESULTS_DIR = DATA_DIR / "results"

UV_DOMAIN = "ultraviolet.library.nyu.edu"
PORTKEY_BASE_URL = os.getenv("PORTKEY_BASE_URL", "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/")
# AutoDDG generates with gemini-2.5-flash; judging with a different (stronger) model
# avoids a model grading its own output. Only Gemini models are enabled on the NYU
# gateway, so the judge is still the same model family; the report states this.
JUDGE_MODEL = os.getenv("DESCRIPTION_EVAL_JUDGE", "@vertexai/gemini-2.5-pro")

_client = None


def llm_client():
    global _client
    if _client is None:
        from portkey_ai import Portkey

        _client = Portkey(base_url=PORTKEY_BASE_URL, api_key=os.environ["PORTKEY_API_KEY"])
    return _client


def parse_json(text: str) -> Any:
    """Parse a JSON reply, tolerating ```json fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=0)
    return json.loads(text[start:])


def ask_json(prompt: str, model: str = JUDGE_MODEL, max_tokens: int = 16000) -> Any:
    """One JSON-returning LLM call, cached on disk by (model, prompt)."""
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["parsed"]

    last_error: Exception | None = None
    for _ in range(3):
        response = llm_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        try:
            parsed = parse_json(content)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            continue
        usage = getattr(response, "usage", None)
        tokens = {"prompt": getattr(usage, "prompt_tokens", None), "completion": getattr(usage, "completion_tokens", None)}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "prompt": prompt, "raw": content, "parsed": parsed,
                                    "usage": tokens}, ensure_ascii=False))
        return parsed
    raise RuntimeError(f"LLM did not return valid JSON after 3 attempts: {last_error}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def normalize_name(name: str) -> str:
    """Column-name key for matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
