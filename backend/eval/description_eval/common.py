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
# Instances gold can come from. UltraViolet is the development set; CaltechDATA is
# held out (the profiler fixes were written looking at UltraViolet failures only).
INSTANCES = {"ultraviolet": UV_DOMAIN, "caltech": "data.caltech.edu"}
PORTKEY_BASE_URL = os.getenv("PORTKEY_BASE_URL", "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/")
# AutoDDG generates with gemini-2.5-flash; judging with a different (stronger) model
# avoids a model grading its own output. (Until late 2026-10 only Gemini models were
# enabled on the NYU gateway; Claude now works via @vertexai/anthropic.*.)
JUDGE_MODEL = os.getenv("DESCRIPTION_EVAL_JUDGE", "@vertexai/gemini-2.5-pro")

# How much evidence a judge call sees. Was 40,000 characters, which cut off the linked-paper excerpts
# (14k-36k characters, placed last) and flagged paper-backed claims as unsupported (V10, 2026-11-26).
EVIDENCE_CHARS = 150_000

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
    # strict=False: LLMs sometimes leave raw tabs/newlines inside strings. raw_decode
    # stops at the end of the first value, so a note after the JSON doesn't fail it.
    return json.JSONDecoder(strict=False).raw_decode(text[start:])[0]


def ask_json(prompt: str, model: str = JUDGE_MODEL, max_tokens: int = 16000) -> Any:
    """One JSON-returning LLM call, cached on disk by (model, prompt)."""
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["parsed"]

    last_error: Exception | None = None
    for _ in range(3):
        # Newer Claude models reject `temperature`; Gemini runs keep temperature=0.
        sampling = {} if "anthropic." in model else {"temperature": 0}
        response = llm_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            **sampling,
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


def ask_text(prompt: str, model: str = JUDGE_MODEL, max_tokens: int = 4000, nonce: str = "", system: str = "") -> str:
    """One free-text LLM call (SAFE and VeriTrail parse code blocks / brackets, not JSON), cached on
    disk by (model, system, prompt, nonce). `nonce` separates repeated samples of the same prompt."""
    key = hashlib.sha256(f"text\n{model}\n{nonce}\n{system}\n{prompt}".encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["raw"]
    sampling = {} if "anthropic." in model else {"temperature": 0}
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    response = llm_client().chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, **sampling)
    content = response.choices[0].message.content or ""
    # Claude Haiku 5.5 refused every VeriTrail evidence-selection prompt (finish_reason "refusal", no
    # tokens); an empty reply must never be cached as an answer, or reruns replay it forever
    # and a reply cut off at max_tokens (Gemini 2.5 Flash spends hidden thinking tokens from the same
    # budget: SAFE's fact split came back as "...contains 4") is not an answer either
    if response.choices[0].finish_reason in ("refusal", "length", "max_tokens") or not content.strip():
        raise RuntimeError(f"{model} returned no usable text (finish_reason {response.choices[0].finish_reason})")
    usage = getattr(response, "usage", None)
    tokens = {"prompt": getattr(usage, "prompt_tokens", None), "completion": getattr(usage, "completion_tokens", None)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "prompt": prompt, "nonce": nonce, "raw": content, "usage": tokens},
                               ensure_ascii=False))
    return content


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def normalize_name(name: str) -> str:
    """Column-name key for matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
