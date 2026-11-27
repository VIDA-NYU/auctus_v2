"""VeriTrail (Metropolitansky & Larson, arXiv 2505.21786) verification on a record's evidence.

VeriTrail checks each claim by first selecting evidence sentences (by id) and then giving a verdict
from the selected material only. Prompts are the paper's (App. C.2), verbatim in veritrail_prompts.json;
notes in reference/methods/veritrail.md. No official code exists.

Our drafts come from one generation step, so VeriTrail's graph has only root nodes (the inputs) and the
final text: one Evidence Selection pass over the roots, then one Verdict Generation call fed the FULL
content of each root node that yielded evidence (the paper's rule for roots). q doesn't matter.
  root nodes   the title line(s), the curator description, the README text, and one node per table
               profile (one line per column), all cut into numbered lines by evidence.py
  claims       Claimify claims of the text (claims.extract, the same claims the direct judge grades)
  sub-claims   one Claimify decomposition of each claim, passed as context only (the paper decomposes
               until single, up to 20 attempts)

Choices the paper leaves open, or where we differ:
  - 160 lines per Evidence Selection prompt (the paper's best setting on FABLES+; 40 on DiverseSumm+)
  - Evidence Selection with Claude Haiku 4.5 (Sonnet 4.6 if refused), Verdict Generation with Claude
    Sonnet 4.6 (the paper uses one model for both; on the NYU gateway Haiku 5.5 refused every Evidence
    Selection prompt and Sonnet 5.5 every Verdict prompt); one verdict per claim, no voting (the paper's hard predictions don't vote either)
  - excerpt format: "[[k]] <node>" then "<id>: <line>", ids unique across the prompt
Labels: Fully Supported -> supported, Not Fully Supported -> unsupported, Inconclusive -> inconclusive.
"""

from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from eval.description_eval.claims import DECOMPOSE, EXTRACT_MODEL, extract
from eval.description_eval.common import ask_json, ask_text
from eval.description_eval.semtypes import JUDGE_MODEL

PROMPTS = json.loads((Path(__file__).parent / "veritrail_prompts.json").read_text())
# Haiku 5.5 refused every Evidence Selection prompt on the NYU gateway (finish_reason "refusal",
# 2026-11-26); Haiku 4.5 answers them. Sonnet is the fallback if a call is refused anyway.
ES_MODEL = "@vertexai/anthropic.claude-haiku-4-5@20251001"
# Sonnet 5.5 refuses VeriTrail's verdict prompt every time (system or inline, finish_reason "refusal");
# Sonnet 4.6 / 4.5, Opus 4.8 and Haiku 4.5 answer it. compare_methods runs the direct judge on Sonnet 4.6
# too, so VeriTrail vs direct isn't a model comparison in disguise.
VERDICT_MODEL = "@vertexai/anthropic.claude-sonnet-4-6"
LINES_PER_PROMPT = 160


def nodes(lines: list[dict[str, str]]) -> "OrderedDict[str, list[tuple[int, str]]]":
    """Group evidence lines into root nodes; each line gets an integer id (its position)."""
    out: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    for i, line in enumerate(lines):
        if line["source"] == "data":
            m = re.match(r"\[([^\]]+)\]", line["text"])
            name = f"table {m.group(1)}" if m else "tables"
        else:
            name = {"title": "record title", "curator": "curator's description", "docs": "README / codebook"}[line["source"]]
        out.setdefault(name, []).append((i, line["text"]))
    return out


def excerpts(node_lines: "OrderedDict[str, list[tuple[int, str]]]") -> str:
    return "\n\n".join(f"[[{k}]] {name}\n" + "\n".join(f"{i}: {t}" for i, t in ls)
                       for k, (name, ls) in enumerate(node_lines.items(), 1))


def _ids(spec: str) -> set[int]:
    out: set[int] = set()
    for part in re.findall(r"\d+\s*-\s*\d+|\d+", spec):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out.update(range(a, b + 1) if b - a < 500 else [])
        else:
            out.add(int(part))
    return out


def select_evidence(claim: str, sub_claims: list[str], all_nodes: "OrderedDict[str, list[tuple[int, str]]]") -> set[int]:
    flat = [(name, i, t) for name, ls in all_nodes.items() for i, t in ls]
    chosen: set[int] = set()
    valid = {i for _, i, _ in flat}
    for start in range(0, len(flat), LINES_PER_PROMPT):
        chunk: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
        for name, i, t in flat[start:start + LINES_PER_PROMPT]:
            chunk.setdefault(name, []).append((i, t))
        user = PROMPTS["es_user"].format(excerpts=excerpts(chunk),
                                         question=f'Is there any information in the excerpts that indicates "{claim}"?',
                                         sub_claims="\n".join(f"- {s}" for s in sub_claims) or "- " + claim)
        try:
            out = ask_text(user, model=ES_MODEL, max_tokens=6000, system=PROMPTS["es_system"])
        except RuntimeError:
            out = ask_text(user, model=VERDICT_MODEL, max_tokens=6000, system=PROMPTS["es_system"])
        m = re.search(r"strong implication:\s*\[([^\]]*)\]", out)
        if m:
            chosen |= _ids(m.group(1)) & valid  # ids that don't exist are discarded, as in the paper
    return chosen


def verdict(claim: str, chosen: set[int], all_nodes: "OrderedDict[str, list[tuple[int, str]]]") -> str:
    if not chosen:
        return "unsupported"  # no evidence: Not Fully Supported, no LM call
    used = OrderedDict((name, ls) for name, ls in all_nodes.items() if any(i in chosen for i, _ in ls))
    out = ask_text(PROMPTS["verdict_user"].format(excerpts=excerpts(used), claim=claim), model=VERDICT_MODEL,
                   max_tokens=6000, system=PROMPTS["verdict_system"])
    m = re.search(r"I submit the following answer:\s*(.*)", out, re.IGNORECASE)
    answer = (m.group(1) if m else out[-300:]).lower()
    if "cannot determine" in answer:
        return "inconclusive"
    if "do not justify" in answer:
        return "unsupported"
    if "justify the entire" in answer:
        return "supported"
    return "inconclusive"


def veritrail(text: str, title: str, ev: dict[str, Any], stats: Counter | None = None) -> list[dict[str, Any]]:
    """Method interface (as claims.claimify_direct): text -> Claimify claims, each with a VeriTrail label."""
    stats = stats if stats is not None else Counter()
    all_nodes = nodes(ev["lines"])
    claims = extract(text, title, stats)
    for c in claims:
        step = "decompose"
        try:
            subs = (ask_json(DECOMPOSE.format(title=title, sentence=c["claim"]), model=EXTRACT_MODEL) or {}).get("claims") or []
            step = "evidence_selection"
            chosen = select_evidence(c["claim"], [s for s in subs if isinstance(s, str)], all_nodes)
            step = "verdict"
            c["label"] = verdict(c["claim"], chosen, all_nodes)
        except RuntimeError:  # refused or empty after the fallback: counted, never guessed
            c["label"], chosen = "refused", set()
            stats[f"refused_{step}"] += 1
        c["evidence_ids"] = sorted(chosen)
        c["source"] = ",".join(sorted({n.split(" ")[0] for n, ls in all_nodes.items() if any(i in chosen for i, _ in ls)}))
        stats[c["label"]] += 1
    return claims
