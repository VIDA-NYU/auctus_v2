"""Checking the checkers: plant known errors in drafts and see which method catches them.

The claim methods (Claimify + direct judge, Claimify + VeriTrail, SAFE) disagree on how many claims
are unsupported, and there is no ground truth for which is right. So each record's README draft gets
exactly two planted errors, written by Gemini Flash:
  data        a number, range, year, count or column changed so the tables contradict it
  provenance  who made it, where / when / how it was collected, or funding changed to something false
Everything else stays word for word (checked). Each method then runs on the planted text, and a judge
call says, per planted error, whether the method extracted a claim stating the changed fact and whether
it graded that claim contradicted or unsupported. That separates the two ways to miss an error:
never extracting it, and extracting it but calling it supported.
"""

from __future__ import annotations

from typing import Any

from eval.description_eval.common import ask_json
from eval.description_eval.semtypes import JUDGE_MODEL

PLANT_MODEL = "@vertexai/gemini-2.5-flash"

PLANT = """Here is a description of a research data record, and the evidence about the record (its curator
description, README and a profile of its tables).

Description:
<<<
{text}
>>>

Evidence:
<<<
{evidence}
>>>

Rewrite the description with exactly two factual errors and nothing else changed:
1. a "data" error: change one number, range, year, count, unit or column name so the tables or the
   evidence contradict it (e.g. "2001 to 2021" -> "2003 to 2021", "399 participants" -> "450 participants");
2. a "provenance" error: change who made the data, where or when it was collected, how it was
   collected, or who funded it, to something the evidence contradicts.
Make each change a small, plausible edit to one phrase. If the description states no provenance, make
two data errors instead. Copy every other word exactly.

Return JSON only: {{"text": "<the rewritten description>",
  "errors": [{{"kind": "data|provenance", "original": "<exact phrase from the description>", "changed": "<the phrase that replaces it>"}}]}}
"""

MATCH = """A method checked the claims of a dataset description against evidence. Two errors were planted
in the description on purpose. For each planted error, say:
- "extracted": whether any of the method's claims states the changed fact (its ids, or empty);
- "caught": whether one of those claims was graded "contradicted" or "unsupported".

Planted errors (original -> changed):
{errors}

The method's claims and verdicts:
{claims}

Return JSON only: [{{"error": <error number>, "claim_ids": [<ids>], "extracted": true|false, "caught": true|false}}]
"""


def plant(text: str, evidence: str) -> dict[str, Any] | None:
    """The planted text and its errors, or None if the edit wasn't clean."""
    reply = ask_json(PLANT.format(text=text, evidence=evidence[:20000]), model=PLANT_MODEL) or {}
    new, errors = reply.get("text") or "", [e for e in reply.get("errors") or [] if isinstance(e, dict)]
    if len(errors) != 2 or not new:
        return None
    restored = new
    for e in errors:
        o, c = str(e.get("original", "")), str(e.get("changed", ""))
        if not o or not c or o == c or o not in text or c not in new:
            return None
        restored = restored.replace(c, o, 1)
    # nothing else may change (whitespace aside)
    if " ".join(restored.split()) != " ".join(text.split()):
        return None
    return {"text": new, "errors": errors}


def detection(planted: dict[str, Any], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per planted error: extracted / caught by this method's claims."""
    errs = "\n".join(f"{i}. [{e['kind']}] \"{e['original']}\" -> \"{e['changed']}\"" for i, e in enumerate(planted["errors"]))
    listing = "\n".join(f"{i}. [{c.get('label', '?')}] {c['claim']}" for i, c in enumerate(claims)) or "(no claims)"
    reply = ask_json(MATCH.format(errors=errs, claims=listing), model=JUDGE_MODEL)
    got = {r.get("error"): r for r in reply if isinstance(r, dict)} if isinstance(reply, list) else {}
    out = []
    for i, e in enumerate(planted["errors"]):
        r = got.get(i) or {}
        out.append({"kind": e["kind"], "extracted": bool(r.get("extracted")),
                    "caught": bool(r.get("caught")) and bool(r.get("extracted"))})
    return out
