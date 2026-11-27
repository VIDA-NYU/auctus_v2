"""Dataset-level evals: AutoDDG's description of each file (what search shows per dataset entry).

There is no curator description per file (curators describe records), so these checks use the file
itself as the reference, and are never compared with the record-level scores:
  grounding        Claimify claims of the description, graded against that file's evidence (its own
                   profile + first rows, the README, the record title and curator text): flagged share
  koesten_ref      Koesten's 9 questions (written for single-dataset summaries) answered once per file
                   by Sonnet FROM THE EVIDENCE (frozen reference, "not stated" allowed); the description
                   is scored on how many of the reference answers it states (stated 1, partial 0.5)
  tell_apart       records with several described files: a judge gets the descriptions (file names
                   masked) and the files' column lists, and matches each description to its file

  MODULE=eval.description_eval.dataset_level ./run_in_worker.sh [--records ID ...]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eval.description_eval import koesten
from eval.description_eval.claims import claimify_direct, rate
from eval.description_eval.collect import load_profile
from eval.description_eval.common import EVIDENCE_CHARS, COLLECTED_DIR, DATA_DIR, GOLD_DIR, ask_json, read_json, write_json
from eval.description_eval.drafts import table_summary
from eval.description_eval.evidence import record_evidence
from eval.description_eval.semtypes import JUDGE_MODEL

OUT = DATA_DIR / "dataset_level"
MAX_FILES = 4  # per record, as the 11-09 per-file eval (69 files); all 179 would cost ~$15

REFERENCE = """Answer each question about ONE data file using only the evidence below (its profile over
every row, first rows, the README and the record's description). These answers will be the reference
for judging short descriptions of this file, so be concrete and short. If the evidence doesn't answer a
question, answer exactly "not stated".

File: {file_key}

Evidence:
<<<
{evidence}
>>>

Questions:
{questions}

Return JSON only: [{{"id": <id>, "a": "..."}}]
"""

STATES = """Here are reference answers about a data file and a short description of the file. For each
reference answer, does the description state it? "stated" (same information, wording may differ),
"partial" (part of it, or vaguer), "contradicted" (the description says something incompatible),
"not_stated".

Description:
<<<
{text}
>>>

{items}

Return JSON only: [{{"id": <id>, "label": "stated|partial|contradicted|not_stated"}}]
"""

MATCH = """A data record has several files. Below are short descriptions of the files (in random order,
file names masked) and each file's columns. Match each description to the file it describes.

Descriptions:
{descriptions}

Files:
{files}

Return JSON only: [{{"description": <description number>, "file": "<file letter>"}}]
"""


def file_evidence(ev: dict[str, Any], key: str, profile: dict[str, Any]) -> str:
    """The record's evidence with the tables part replaced by this one file's profile."""
    parts = [f"Record title: {ev['title']}", f"Curator's description of the record:\n{ev['description']}"]
    if ev["docs"]:
        parts.append(f"Documentation (README / codebook):\n{ev['docs']}")
    parts.append(f"This file:\n{table_summary(key, profile)}")
    return "\n\n".join(parts)


def koesten_reference(key: str, evidence: str) -> list[str]:
    qs = koesten.questions()
    listing = "\n".join(f"{i}. {q['q']}" for i, q in enumerate(qs))
    reply = ask_json(REFERENCE.format(file_key=key, evidence=evidence[:EVIDENCE_CHARS], questions=listing), model=JUDGE_MODEL)
    got = {a.get("id"): str(a.get("a", "")) for a in reply if isinstance(a, dict)} if isinstance(reply, list) else {}
    return [got.get(i, "not stated") for i in range(len(qs))]


def koesten_states(text: str, ref: list[str]) -> list[str | None]:
    """Label per question; None where the evidence itself doesn't answer it (not counted)."""
    qs = koesten.questions()
    live = [i for i, a in enumerate(ref) if a and not a.strip().lower().startswith("not stated")]
    if not live:
        return [None] * len(qs)
    items = "\n".join(f"id {i}: Q: {qs[i]['q']}\n  reference: {ref[i]}" for i in live)
    reply = ask_json(STATES.format(text=text[:6000], items=items), model=JUDGE_MODEL)
    got = {r.get("id"): r.get("label") for r in reply if isinstance(r, dict)} if isinstance(reply, list) else {}
    return [(got.get(i) or "not_stated") if i in live else None for i in range(len(qs))]


def mask(text: str, names: list[str]) -> str:
    for n in names:
        for form in {n, n.rsplit("/", 1)[-1], re.sub(r"\.[a-z0-9]+$", "", n.rsplit("/", 1)[-1])}:
            if len(form) > 3:
                text = text.replace(form, "[file]")
    return text


def tell_apart(described: list[tuple[str, str, dict]]) -> dict[str, Any]:
    rng = random.Random(20261126)
    names = [k for k, _, _ in described]
    order = list(range(len(described)))
    rng.shuffle(order)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    descs = "\n\n".join(f"{n}. {mask(described[i][1], names)}" for n, i in enumerate(order))
    files = "\n".join(f"{letters[j]}. columns: " + ", ".join(str(c.get('name')) for c in
                      ((described[j][2].get('profiler_metadata') or {}).get('columns') or [])[:40])
                      for j in range(len(described)))
    reply = ask_json(MATCH.format(descriptions=descs, files=files), model=JUDGE_MODEL)
    picks = {r.get("description"): str(r.get("file", "")).strip()[:1] for r in reply if isinstance(r, dict)} if isinstance(reply, list) else {}
    right = sum(picks.get(n) == letters[i] for n, i in enumerate(order))
    return {"files": len(described), "right": right, "chance": round(1 / len(described), 3)}


def run_record(storage, rid: str) -> dict[str, Any]:
    gold = read_json(GOLD_DIR / f"{rid}.json")
    ev = record_evidence(storage, rid)
    collected = read_json(COLLECTED_DIR / f"{rid}.json")["files"]
    ids = {f["file_key"]: f["dataset_id"] for f in gold["files"]}
    files, described = [], []
    for key, info in [(k, i) for k, i in sorted(collected.items()) if i.get("ufd") and k in ids][:MAX_FILES]:
        try:
            profile = load_profile(storage, ids[key])
        except Exception:  # gold's file list predates a re-ingest (deduped / regrouped files): skip the file
            continue
        evidence = file_evidence(ev, key, profile)
        file_ev = {**ev, "evidence": evidence}
        claims = claimify_direct(info["ufd"], ev["title"], file_ev, Counter())
        labels = [c["label"] for c in claims]
        ref = koesten_reference(key, evidence)
        states = koesten_states(info["ufd"], ref)
        live = [s for s in states if s is not None]
        files.append({"file": key, "description": info["ufd"], "claims": claims, "n_claims": len(claims),
                      "flagged": round((labels.count("unsupported") + labels.count("contradicted")) / len(labels), 3) if labels else None,
                      "contradicted": rate(labels, "contradicted"),
                      "koesten_reference": ref, "koesten_states": states,
                      "koesten_ref_recall": rate(live, "stated", "partial"),
                      "koesten_by_q": {q["key"]: s for q, s in zip(koesten.questions(), states)}})
        described.append((key, info["ufd"], profile))
    row: dict[str, Any] = {"record": rid, "files": files}
    if len(described) > 1:
        row["tell_apart"] = tell_apart(described[:8])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", nargs="+")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    from storage.minio_client import get_storage_client

    storage = get_storage_client()
    rids = args.records or [read_json(p)["record_id"] for p in sorted(GOLD_DIR.glob("*.json"))
                            if read_json(p).get("domain") == "ultraviolet.library.nyu.edu"
                            and len(((read_json(p).get("curator") or {}).get("description") or "")) > 80]

    def run(rid):
        try:
            return run_record(storage, rid)
        except Exception as exc:
            return {"record": rid, "error": repr(exc), "files": []}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, rids))
    files = [f for r in rows for f in r["files"]]
    by_q: dict[str, Counter] = {}
    for f in files:
        for q, s in f["koesten_by_q"].items():
            by_q.setdefault(q, Counter())[s or "no_reference"] += 1
    ta = [r["tell_apart"] for r in rows if r.get("tell_apart")]
    summary = {"records": len(rows), "files": len(files), "errors": [(r["record"], r["error"]) for r in rows if r.get("error")],
               "flagged": round(statistics.mean(f["flagged"] for f in files if f["flagged"] is not None), 3) if files else None,
               "contradicted": round(statistics.mean(f["contradicted"] for f in files if f["contradicted"] is not None), 3) if files else None,
               "koesten_ref_recall": round(statistics.mean(f["koesten_ref_recall"] for f in files if f["koesten_ref_recall"] is not None), 3) if files else None,
               "koesten_by_question": {q: dict(c) for q, c in by_q.items()},
               "tell_apart": {"records": len(ta), "descriptions": sum(t["files"] for t in ta), "right": sum(t["right"] for t in ta),
                              "chance": round(statistics.mean(t["chance"] for t in ta), 3) if ta else None}}
    if not args.records:
        write_json(OUT / "results.json", {"summary": summary, "rows": rows})
    else:
        write_json(OUT / "results-subset.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
