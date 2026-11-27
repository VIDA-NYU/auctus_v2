"""Koesten's dataset-summary questions as a scoring method.

The question list lives in koesten_questions.json (source and wording noted there). For each
description, a judge answers every question from the description alone ("not stated" otherwise), and
a second call checks each answer against the record's evidence. Scores:
  answered   share of questions the description answers
  supported  share answered AND backed by the evidence (the score; a wrong answer earns nothing)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.description_eval.common import EVIDENCE_CHARS, ask_json
from eval.description_eval.semtypes import JUDGE_MODEL

QUESTIONS_FILE = Path(__file__).parent / "koesten_questions.json"

ANSWER = """Answer each question about a research dataset using ONLY the description below. If the
description doesn't give a concrete answer, answer exactly "not stated" (a vague hint is "not stated").
Keep answers short.

Description:
<<<
{text}
>>>

Questions:
{questions}

Return JSON only: [{{"id": <id>, "a": "..."}}]
"""

CHECK = """Check answers about a research dataset against the evidence (record title, curator description,
README, and a profile of each table over every row).

Evidence:
<<<
{evidence}
>>>

For each answer: "supported" (the evidence backs it, or the data shows it), "contradicted" (the evidence
shows it is wrong) or "unsupported" (neither).

{items}

Return JSON only: [{{"id": <id>, "label": "supported|contradicted|unsupported"}}]
"""


def questions() -> list[dict[str, str]]:
    return json.loads(QUESTIONS_FILE.read_text())["questions"]


def score(text: str, evidence: str) -> dict[str, Any]:
    qs = questions()
    listing = "\n".join(f"{i}. {q['q']}" for i, q in enumerate(qs))
    reply = ask_json(ANSWER.format(text=text[:8000], questions=listing), model=JUDGE_MODEL)
    answers = {a.get("id"): str(a.get("a", "")) for a in reply if isinstance(a, dict)} if isinstance(reply, list) else {}
    answered = {i: answers.get(i, "not stated") for i in range(len(qs))
                if answers.get(i) and not answers.get(i, "").strip().lower().startswith("not stated")}
    labels: dict[int, str] = {}
    if answered:
        items = "\n".join(f"id {i}: Q: {qs[i]['q']}\n  A: {a}" for i, a in answered.items())
        reply = ask_json(CHECK.format(evidence=evidence[:EVIDENCE_CHARS], items=items), model=JUDGE_MODEL)
        labels = {r.get("id"): r.get("label") for r in reply if isinstance(r, dict)} if isinstance(reply, list) else {}
    per_q = {q["key"]: ("not_stated" if i not in answered else labels.get(i) or "unsupported")
             for i, q in enumerate(qs)}
    n = len(qs)
    return {"per_question": per_q,
            "answered": round(len(answered) / n, 3) if n else None,
            "supported": round(sum(v == "supported" for v in per_q.values()) / n, 3) if n else None,
            "contradicted": sum(v == "contradicted" for v in per_q.values())}
