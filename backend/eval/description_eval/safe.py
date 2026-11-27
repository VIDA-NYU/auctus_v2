"""SAFE (Wei et al. 2024, "Long-form factuality in large language models") on local evidence.

The pipeline and prompts are SAFE's own (google-deepmind/long-form-factuality, commit 9d27158; prompts
copied verbatim into safe_prompts.json and factscore_demos.json, source notes in
reference/methods/safe.md):
  1. split each sentence into atomic facts (FActScore instruction + 7 fixed demos + the BM25 top-1 demo)
  2. revise each fact to be self-contained
  3. relevance ("Foo" / "Not Foo") to the question the text answers
  4. search loop: the model writes one query per step, results are appended to KNOWLEDGE
  5. final rating: [Supported] / [Not Supported]

What changes for us, and nothing else:
  - search is BM25 over the record's evidence lines (evidence.py: title, curator text, README, table
    profiles), top 3 lines per query, in place of Google via Serper. The prompts still say "Google
    Search"; they are kept verbatim.
  - 3 queries per fact instead of 5 (the evidence is a few hundred lines, not the web).
  - the question for the relevance step: "Describe the research data record titled '<title>'."
  - models: splitting with Gemini Flash at temperature 0 (SAFE used gpt-3.5 at 0), revise / relevance /
    queries with Claude Haiku (SAFE's small rater), the final rating with Claude Sonnet, the same verdict
    model as the other methods.
  - FActScore's spaCy post-processing (biography heuristics) is left out.
SAFE has no "contradicted": Not Supported maps to "unsupported"; Irrelevant facts are dropped from the
rates, as in the paper.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eval.description_eval.common import ask_text
from eval.description_eval.evidence import BM25, split_text
from eval.description_eval.semtypes import JUDGE_MODEL

HERE = Path(__file__).parent
PROMPTS = json.loads((HERE / "safe_prompts.json").read_text())
FACTSCORE = json.loads((HERE / "factscore_demos.json").read_text())
SPLIT_MODEL = "@vertexai/gemini-2.5-flash"
SMALL_MODEL = "@vertexai/anthropic.claude-haiku-5-5"
MAX_STEPS = 3
RESULTS_PER_QUERY = 3
_DEMO_KEYS = list(FACTSCORE["demos"])
_DEMO_INDEX = BM25([{"id": str(i), "source": "demo", "text": k} for i, k in enumerate(_DEMO_KEYS)])


def _code_block(text: str) -> str:
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _brackets(text: str) -> str:
    m = re.search(r"\[([^\[\]]+)\]", text)
    return m.group(1).strip() if m else ""


def split_prompt(sentence: str) -> str:
    prompt = ""
    top = _DEMO_INDEX.search(sentence, k=1)
    for key in _DEMO_KEYS[:7] + [_DEMO_KEYS[int(t["id"])] for t in top]:
        prompt += f"Please breakdown the following sentence into independent facts: {key}\n"
        prompt += "".join(f"- {f}\n" for f in FACTSCORE["demos"][key]) + "\n"
    prompt += f"Please breakdown the following sentence into independent facts: {sentence}\n"
    return FACTSCORE["instruction"] + prompt


def atomic_facts(text: str) -> list[dict[str, str]]:
    out = []
    for sent in split_text(text, max_len=2000):
        if re.match(r"^(Sure|Please)", sent):
            continue
        raw = ask_text(split_prompt(sent), model=SPLIT_MODEL, max_tokens=8000)
        items = raw.split("- ") if "- " in raw else raw.split("* ")
        for item in items[1:]:
            fact = item.split("\n")[0].strip()
            if fact and not fact.startswith("This sentence does not contain any facts"):
                out.append({"claim": fact if fact.endswith(".") else fact + ".", "sentence": sent})
    return out


def _fill(template: str, **values: str) -> str:
    for k, v in values.items():
        template = template.replace(f"[{k}]", v)
    return template.strip(" \n")


def rate_fact(fact: str, response: str, question: str, index: BM25, standalone: bool = False) -> dict[str, Any]:
    """`standalone`: the fact is already self-contained and on topic (hand-written test claims), so
    the revise and relevance steps are skipped."""
    revised = fact if standalone else _code_block(ask_text(_fill(PROMPTS["revise"], **{"ATOMIC FACT": fact, "RESPONSE": response}),
                                                           model=SMALL_MODEL, max_tokens=3000)) or fact
    rel = "Foo" if standalone else _brackets(ask_text(_fill(PROMPTS["relevance"], PROMPT=question, RESPONSE=response,
                                                            **{"ATOMIC FACT": revised}), model=SMALL_MODEL, max_tokens=3000))
    if rel == "Not Foo":
        return {"revised": revised, "label": "irrelevant", "queries": []}
    knowledge, queries, seen = [], [], set()
    for step in range(MAX_STEPS):
        query = _code_block(ask_text(_fill(PROMPTS["next_search"], KNOWLEDGE="\n".join(knowledge) or "N/A", STATEMENT=revised),
                                     model=SMALL_MODEL, max_tokens=2000, nonce=f"step{step}"))
        if not query:
            break
        # web search returns new pages for a new query; BM25 over a few hundred lines returns the same
        # top lines every time, so each step skips lines an earlier step already returned
        hits = index.search(query, k=RESULTS_PER_QUERY, exclude=seen)
        seen.update(h["id"] for h in hits)
        knowledge.append(" ".join(f"({h['source']}) {h['text']}" for h in hits) or "No good Google Search result was found")
        queries.append(query)
    answer = re.sub(r"[^\w\s]", "", _brackets(ask_text(_fill(PROMPTS["final"], KNOWLEDGE="\n".join(knowledge), STATEMENT=revised),
                                                        model=JUDGE_MODEL, max_tokens=4000)))
    label = "supported" if answer == "Supported" else "unsupported"
    return {"revised": revised, "label": label, "queries": queries, "source": ""}


def safe(text: str, title: str, ev: dict[str, Any], stats: Counter | None = None) -> list[dict[str, Any]]:
    """Method interface (as claims.claimify_direct): text -> facts, each with a label."""
    stats = stats if stats is not None else Counter()
    index = BM25(ev["lines"])
    question = f"Describe the research data record titled '{title}'."
    facts = atomic_facts(text)
    out = []
    for f in facts:
        try:
            f.update(rate_fact(f["claim"], text, question, index))
        except Exception:
            stats["dropped_error"] += 1  # SAFE drops a fact whose pipeline fails, so do we, and count it
            continue
        stats[f["label"]] += 1
        if f["label"] != "irrelevant":
            out.append(f)
    stats["facts"] += len(facts)
    return out
