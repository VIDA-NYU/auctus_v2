"""Question-based description evals (11-10 deliverables 14 and 15).

Compares each record's curator description with generated descriptions:
- autoddg: AutoDDG's description (UFD) of the record's first profiled file (what search shows today);
- draft_readme: the record-level draft written from the data profiles + README (drafts.py, "docs" arm).

1. Question generation, both ways. An LLM writes up to 8 short factual questions (with answers) from
   text X; a second call answers them from text Y alone, without seeing X's answers; a judge grades each
   answer against X's (correct / partial / wrong / not stated).
   - curator -> candidate: how much of what the curator says the candidate can answer (coverage);
   - candidate -> curator: how much of what the candidate says the curator text also answers. Low means
     the candidate adds things (data detail the curator skipped, or claims nothing backs).
   The method follows the 11-10 meeting's description; the paper it comes from is still to find.
2. Checklist: the same fixed questions asked of every description, yes / no each. The list below is
   PROVISIONAL: written from the dataset-summary attributes in Koesten et al. 2020 ("Everything you
   always wanted to know about a dataset"), until we have her exact question list.

  python -m eval.description_eval.run questions
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from eval.description_eval.common import (COLLECTED_DIR, DATA_DIR, GOLD_DIR, UV_DOMAIN, ask_json, read_json,
                                          write_json)
from eval.description_eval.semtypes import JUDGE_MODEL

WRITER = "@vertexai/gemini-2.5-flash"
OUT = DATA_DIR / "questions"

CHECKLIST = [
    ("topic", "What is the data about?"),
    ("unit", "What does one row or record in the data represent?"),
    ("where", "What place or geographic area does it cover?"),
    ("when", "What time period does it cover?"),
    ("size", "How big is it (rows, files, number of items)?"),
    ("variables", "What variables or columns does it contain?"),
    ("who", "Who created or collected it?"),
    ("how", "How was it collected or produced?"),
    ("why", "Why was it created (purpose, project, study)?"),
    ("format", "What file formats is it in?"),
    ("quality", "Are there known gaps, errors or caveats?"),
    ("reuse", "Under what license or access conditions can it be used?"),
]

GENERATE = """Here is a description of a research dataset.

<<<
{text}
>>>

Write up to 8 short factual questions that this description answers, each with its answer taken from
the description. Cover different things (what, where, when, who, how, size, variables, ...), one fact per
question. Questions must make sense without seeing the description (no "this paragraph").

Return JSON only: [{{"q": "...", "a": "..."}}]
"""

ANSWER = """Answer each question using ONLY the dataset description below. If the description doesn't say,
answer exactly "not stated". Keep answers short.

<<<
{text}
>>>

Questions:
{questions}

Return JSON only: [{{"id": <id>, "a": "..."}}]
"""

GRADE = """Each item has a question, a reference answer, and a candidate answer written from a different text.
Grade the candidate against the reference:
- "correct": same information (wording may differ)
- "partial": overlaps but less specific or missing part
- "wrong": contradicts the reference
- "not_stated": the candidate says it's not stated, or doesn't answer

{items}

Return JSON only: [{{"id": <id>, "grade": "correct|partial|wrong|not_stated"}}]
"""

CHECK = """Does this dataset description answer each question? "yes" only if a reader gets a concrete answer
from the text (a vague hint is "no").

<<<
{text}
>>>

{questions}

Return JSON only: {{{keys}}}
"""


def texts() -> dict[str, dict[str, str]]:
    """record -> {curator, autoddg, draft_readme} (missing ones left out)."""
    drafts = {(r["record"], r["arm"]): r.get("draft") for r in read_json(DATA_DIR / "drafts" / "results.json")["rows"]}
    out = {}
    for path in sorted(GOLD_DIR.glob("*.json")):
        gold = read_json(path)
        if gold.get("domain") != UV_DOMAIN:
            continue
        rid = gold["record_id"]
        cur = gold.get("curator") or {}
        t = {"curator": f"{cur.get('title', '')}\n\n{cur.get('description', '')}".strip()}
        collected = COLLECTED_DIR / f"{rid}.json"
        if collected.exists():
            files = read_json(collected)["files"]
            ufd = next((f.get("ufd") for _, f in sorted(files.items()) if f.get("ufd")), None)
            if ufd:
                t["autoddg"] = ufd
        if drafts.get((rid, "docs")):
            t["draft_readme"] = drafts[(rid, "docs")]
        if len(t["curator"]) > 80:
            out[rid] = t
    return out


def questions_from(text: str) -> list[dict[str, str]]:
    reply = ask_json(GENERATE.format(text=text[:8000]), model=WRITER)
    return [q for q in reply if isinstance(q, dict) and q.get("q") and q.get("a")][:8] if isinstance(reply, list) else []


def ask_across(qs: list[dict[str, str]], other: str) -> list[str]:
    """Each question's grade when answered from `other`."""
    if not qs:
        return []
    listing = "\n".join(f"{i}. {q['q']}" for i, q in enumerate(qs))
    reply = ask_json(ANSWER.format(text=other[:8000], questions=listing), model=WRITER)
    answers = {a.get("id"): a.get("a", "") for a in reply if isinstance(a, dict)} if isinstance(reply, list) else {}
    items = "\n\n".join(f"id {i}:\n  question: {q['q']}\n  reference: {q['a']}\n  candidate: {answers.get(i, 'not stated')}"
                        for i, q in enumerate(qs))
    reply = ask_json(GRADE.format(items=items), model=JUDGE_MODEL)
    grades = {g.get("id"): g.get("grade") for g in reply if isinstance(g, dict)} if isinstance(reply, list) else {}
    return [grades.get(i) or "not_stated" for i in range(len(qs))]


def checklist(text: str) -> dict[str, bool]:
    listing = "\n".join(f"- {k}: {q}" for k, q in CHECKLIST)
    keys = ", ".join(f'"{k}": "yes|no"' for k, _ in CHECKLIST)
    reply = ask_json(CHECK.format(text=text[:8000], questions=listing, keys=keys), model=JUDGE_MODEL)
    return {k: str((reply or {}).get(k, "no")).lower().startswith("y") for k, _ in CHECKLIST}


def score(grades: list[str]) -> float | None:
    """correct = 1, partial = 0.5, else 0."""
    return round(sum({"correct": 1, "partial": 0.5}.get(g, 0) for g in grades) / len(grades), 3) if grades else None


def paired(rows: list[dict], a: str, b: str, key: str) -> dict[str, Any]:
    import random

    diffs = [r[key][b] - r[key][a] for r in rows if r[key].get(a) is not None and r[key].get(b) is not None]
    if not diffs:
        return {"n": 0}
    rng = random.Random(20261117)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
    return {"n": len(diffs), "mean_diff": round(statistics.mean(diffs), 3),
            "ci95": [round(boots[50], 3), round(boots[1949], 3)],
            "higher": sum(d > 0 for d in diffs), "lower": sum(d < 0 for d in diffs)}


def main() -> None:
    rows = []
    records = texts()
    print(f"{len(records)} UV records; candidates: {Counter(k for t in records.values() for k in t if k != 'curator')}")
    for n, (rid, t) in enumerate(records.items(), 1):
        row: dict[str, Any] = {"record": rid, "coverage": {}, "added": {}, "grades": {}, "checklist": {}}
        cur_qs = questions_from(t["curator"])
        row["curator_questions"] = cur_qs
        row["checklist"]["curator"] = checklist(t["curator"])
        for cand in ("autoddg", "draft_readme"):
            if cand not in t:
                continue
            fwd = ask_across(cur_qs, t[cand])
            cand_qs = questions_from(t[cand])
            back = ask_across(cand_qs, t["curator"])
            row["coverage"][cand], row["added"][cand] = score(fwd), score(back)
            row["grades"][cand] = {"curator_to_candidate": fwd, "candidate_to_curator": back,
                                   "candidate_questions": cand_qs}
            row["checklist"][cand] = checklist(t[cand])
        rows.append(row)
        print(f"  {n}/{len(records)} {rid}: coverage {row['coverage']}, answered by curator {row['added']}", flush=True)

    summary: dict[str, Any] = {"records": len(rows), "writer": WRITER, "judge": JUDGE_MODEL, "methods": {}}
    for cand in ("autoddg", "draft_readme"):
        mine = [r for r in rows if cand in r["coverage"]]
        fwd = Counter(g for r in mine for g in r["grades"][cand]["curator_to_candidate"])
        back = Counter(g for r in mine for g in r["grades"][cand]["candidate_to_curator"])
        summary["methods"][cand] = {
            "records": len(mine),
            "curator_questions_answered": round(statistics.mean(r["coverage"][cand] for r in mine), 3) if mine else None,
            "curator_question_grades": dict(fwd),
            "own_questions_curator_answers": round(statistics.mean(r["added"][cand] for r in mine if r["added"][cand] is not None), 3) if mine else None,
            "own_question_grades": dict(back),
        }
    summary["draft_vs_autoddg_coverage"] = paired(rows, "autoddg", "draft_readme", "coverage")
    per_q: dict[str, dict[str, float]] = defaultdict(dict)
    for who in ("curator", "autoddg", "draft_readme"):
        mine = [r["checklist"][who] for r in rows if who in r["checklist"]]
        for k, _ in CHECKLIST:
            per_q[k][who] = round(sum(c[k] for c in mine) / len(mine), 3) if mine else None
        summary.setdefault("checklist_mean_yes", {})[who] = round(
            statistics.mean(sum(c.values()) for c in mine), 2) if mine else None
    summary["checklist_by_question"] = per_q
    summary["checklist_note"] = "provisional list from Koesten et al. 2020 summary attributes, not her exact questions"
    write_json(OUT / "results.json", {"summary": summary, "rows": rows})
    print(__import__("json").dumps(summary, indent=1))
