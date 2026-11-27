"""Drafter prompt loop: draft with each prompt variant, score, read the failures, tweak, rescore.

Variants are in drafter_prompts.py. Records are split once, by a hash of the record id, into
  dev   tune on these (about 60%)
  test  report on these, plus CaltechDATA as a second held-out set (--instance caltech)
so the winning prompt isn't scored on the records it was tuned on.

Scores per draft (all graded by Claude Sonnet; drafts and Claimify extraction use Gemini Flash):
  recall       curator claims the draft states (Claimify claims, stated = 1, partial = 0.5),
               also split data / provenance / context
  flagged      the draft's own claims that are unsupported or contradicted by the evidence
  questions    Sonnet's benchmark questions from the curator text, answered from the draft alone
  koesten      Koesten's dataset-summary questions answered by the draft AND backed by the evidence

Selection rule, fixed before round 1: highest mean recall on dev among variants whose mean flagged
rate is at most the baseline's + 1 point; ties go to benchmark questions, then Koesten.

  MODULE=eval.description_eval.drafter_loop ./run_in_worker.sh --variants V0 V1 V2 V3 [--split dev]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eval.description_eval import koesten
from eval.description_eval.claims import curator_claims, direct_verify, score_text
from eval.description_eval.common import COLLECTED_DIR, DATA_DIR, GOLD_DIR, INSTANCES, ask_json, read_json, write_json
from eval.description_eval.papers import sheet_text
from eval.description_eval.drafter_prompts import VARIANTS
from eval.description_eval.evidence import record_evidence
from eval.description_eval.questions import ANSWER, GRADE, score as grade_score
from eval.description_eval.semtypes import JUDGE_MODEL

DRAFT_MODEL = "@vertexai/gemini-2.5-flash"
OUT = DATA_DIR / "drafter_loop"
METRICS = ("recall", "recall_data", "recall_provenance", "recall_context", "ceiling", "recall_recoverable",
           "flagged", "questions", "koesten", "koesten_answered", "words")

BENCH = """Here is a curator's description of a research data record.

Title: {title}

<<<
{text}
>>>

Write up to 10 short factual questions that this description answers, the questions a person deciding
whether to use this data would ask, each with its answer taken from the description. Cover different
things (what it is, what one row is, variables, where, when, size, who made it, why, how it was
collected, caveats), one fact per question. Questions must make sense without seeing the description.

Return JSON only: [{{"q": "...", "a": "..."}}]
"""


def split(rid: str) -> str:
    return "dev" if int(hashlib.sha1(rid.encode()).hexdigest(), 16) % 10 < 6 else "test"


def records(instance: str, which: str) -> list[str]:
    out = []
    for path in sorted(GOLD_DIR.glob("**/*.json")):
        gold = read_json(path)
        if gold.get("domain", INSTANCES["ultraviolet"]) != INSTANCES[instance]:
            continue
        if len(((gold.get("curator") or {}).get("description") or "")) <= 80:
            continue
        if instance == "ultraviolet" and which != "all" and split(gold["record_id"]) != which:
            continue
        out.append(gold["record_id"])
    return out


def autoddg_tables(rid: str) -> str:
    path = COLLECTED_DIR / f"{rid}.json"
    files = read_json(path)["files"] if path.exists() else {}
    return "\n\n".join(f"### {k}\n{f['ufd']}" for k, f in sorted(files.items()) if f.get("ufd"))


def draft(variant: str, inp: dict[str, Any], arm: str, rid: str = "", model: str = DRAFT_MODEL) -> str:
    v = VARIANTS[variant]
    documentation = f"Documentation (README / codebook):\n{inp['docs']}\n\n" if arm == "docs" and inp["docs"] else ""
    tables = autoddg_tables(rid) if v.get("inputs") == "autoddg" else inp["tables"]
    fields = {"title": inp["title"], "documentation": documentation, "tables": tables[:12000]}
    if v.get("inputs") == "paper":
        sheet = sheet_text(rid)
        fields["paper"] = f"Provenance from the linked paper (from its methods, data and funding sections):\n{sheet}\n\n" if sheet else ""
    if len(v["stages"]) == 1:
        return (ask_json(v["stages"][0].format(**fields), model=model) or {}).get("description", "")
    sheet = ask_json(v["stages"][0].format(**fields), model=model) or {}
    facts = json.dumps({k: x for k, x in sheet.items() if x}, indent=1) if isinstance(sheet, dict) else "{}"
    return (ask_json(v["stages"][1].format(title=inp["title"], facts=facts), model=model) or {}).get("description", "")


def benchmark_questions(inp: dict[str, Any]) -> list[dict[str, str]]:
    """Frozen per record: Sonnet writes them once from the curator text (cached)."""
    reply = ask_json(BENCH.format(title=inp["title"], text=inp["description"][:8000]), model=JUDGE_MODEL)
    return [q for q in reply if isinstance(q, dict) and q.get("q") and q.get("a")][:10] if isinstance(reply, list) else []


def answer_questions(qs: list[dict[str, str]], text: str) -> list[str]:
    if not qs:
        return []
    listing = "\n".join(f"{i}. {q['q']}" for i, q in enumerate(qs))
    reply = ask_json(ANSWER.format(text=text[:8000], questions=listing), model=JUDGE_MODEL)
    answers = {a.get("id"): a.get("a", "") for a in reply if isinstance(a, dict)} if isinstance(reply, list) else {}
    items = "\n\n".join(f"id {i}:\n  question: {q['q']}\n  reference: {q['a']}\n  candidate: {answers.get(i, 'not stated')}"
                        for i, q in enumerate(qs))
    reply = ask_json(GRADE.format(items=items), model=JUDGE_MODEL)
    grades = {g.get("id"): g.get("grade") for g in reply if isinstance(g, dict)} if isinstance(reply, list) else {}
    return [grades.get(i) or "not_stated" for i in range(len(qs))]


def recoverable(inp: dict[str, Any], cur: list[dict[str, Any]], arm: str) -> list[bool]:
    """Which curator claims the drafter's own inputs support (title, README if the arm has it, tables),
    with the curator's text taken out of the evidence: the ceiling for any prompt (cached per record)."""
    evidence = inp["evidence"].replace(f"Curator's description:\n{inp['description']}", "")
    if arm != "docs" and inp["docs"]:
        evidence = evidence.replace(f"Documentation (README / codebook):\n{inp['docs']}", "")
    graded = direct_verify([{"claim": c["claim"]} for c in cur], evidence)
    return [g["label"] == "supported" for g in graded]


def evaluate(storage, rid: str, variant: str, arm: str, with_paper: bool = False, model: str = DRAFT_MODEL) -> dict[str, Any]:
    inp = record_evidence(storage, rid, with_paper=with_paper)
    if arm == "docs" and not inp["docs"]:
        return {"record": rid, "variant": variant, "arm": arm, "skipped": "no documentation"}
    text = draft(variant, inp, arm, rid, model)
    if not text:
        return {"record": rid, "variant": variant, "arm": arm, "error": "empty draft"}
    cur = curator_claims(inp)
    res = score_text(text, inp, cur)
    rec = recoverable(inp, cur, arm)
    in_inputs = [l for l, r in zip(res["coverage"], rec) if r]
    grades = answer_questions(benchmark_questions(inp), text)
    k = koesten.score(text, inp["evidence"]) if koesten.QUESTIONS_FILE.exists() else \
        {"supported": None, "answered": None, "per_question": {}}
    return {"record": rid, "variant": variant, "arm": arm, "draft": text, "words": len(text.split()),
            **{m: res.get(m) for m in ("recall", "recall_data", "recall_provenance", "recall_context", "flagged",
                                       "n_claims")},
            "ceiling": round(sum(rec) / len(rec), 3) if rec else None,
            "recall_recoverable": round(sum(1 if l == "stated" else 0.5 if l == "partial" else 0 for l in in_inputs)
                                        / len(in_inputs), 3) if in_inputs else None,
            "questions": grade_score(grades), "question_grades": grades,
            "koesten": k["supported"], "koesten_answered": k["answered"], "koesten_per_question": k["per_question"],
            # for reading failures between rounds
            "missed": [c["claim"] + f" [{c['kind']}{', in inputs' if r else ''}]"
                       for c, l, r in zip(cur, res["coverage"], rec) if l == "not_stated"],
            "flagged_claims": [f"[{c['label']}] {c['claim']}" for c in res["claims"] if c["label"] != "supported"]}


def paired_diff(rows: list[dict], a: str, b: str, metric: str) -> dict[str, Any]:
    by = {}
    for r in rows:
        if r.get(metric) is not None:
            by.setdefault(r["record"], {})[r["variant"]] = r[metric]
    diffs = [v[b] - v[a] for v in by.values() if a in v and b in v]
    if not diffs:
        return {"n": 0}
    rng = random.Random(20261126)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
    return {"n": len(diffs), "diff": round(statistics.mean(diffs), 3), "ci95": [round(boots[50], 3), round(boots[1949], 3)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--arm", default="docs", choices=["docs", "data"])
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--instance", default="ultraviolet", choices=sorted(INSTANCES))
    parser.add_argument("--baseline", default="V0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--with-paper", action="store_true", help="paper excerpts in the judge's evidence")
    parser.add_argument("--only-with-paper", action="store_true", help="only records with a paper sheet")
    parser.add_argument("--limit", type=int, help="a fixed sample of this many records (by hash, not the first N)")
    parser.add_argument("--draft-model", default=DRAFT_MODEL, help="the drafter's model (default: what ingestion runs)")
    args = parser.parse_args()

    from storage.minio_client import get_storage_client

    storage = get_storage_client()
    rids = records(args.instance, args.split)
    if args.only_with_paper:
        rids = [r for r in rids if sheet_text(r)]
    if args.limit:
        rids = sorted(rids, key=lambda r: hashlib.sha1(f"sample:{r}".encode()).hexdigest())[:args.limit]
    variants = list(dict.fromkeys([args.baseline] + args.variants))
    tasks = [(rid, v) for v in variants for rid in rids]
    print(f"{len(rids)} {args.instance} records ({args.split}), arm {args.arm}, variants {variants}", flush=True)

    def run(task):
        rid, v = task
        try:
            return evaluate(storage, rid, v, args.arm, args.with_paper, args.draft_model)
        except Exception as exc:
            return {"record": rid, "variant": v, "arm": args.arm, "error": repr(exc)}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))
    ok = [r for r in rows if not r.get("error") and not r.get("skipped")]

    summary: dict[str, Any] = {"instance": args.instance, "split": args.split, "arm": args.arm, "draft_model": args.draft_model,
                               "records": len(rids), "errors": [(r["record"], r["variant"], r["error"]) for r in rows if r.get("error")],
                               "variants": {}}
    for v in variants:
        mine = [r for r in ok if r["variant"] == v]
        s = {"n": len(mine), "structure": VARIANTS[v]["structure"], "style": VARIANTS[v]["style"]}
        for m in METRICS:
            vals = [r[m] for r in mine if r.get(m) is not None]
            s[m] = round(statistics.mean(vals), 3) if vals else None
        if v != args.baseline:
            s["vs_baseline"] = {m: paired_diff(ok, args.baseline, v, m) for m in ("recall", "flagged", "questions", "koesten")}
        summary["variants"][v] = s
    base_flag = summary["variants"][args.baseline]["flagged"] or 0
    eligible = [v for v, s in summary["variants"].items() if s["recall"] is not None and (s["flagged"] or 0) <= base_flag + 0.01]
    summary["selection_rule"] = "max recall with flagged <= baseline + 1 pt; ties: questions, then koesten"
    summary["selected"] = max(eligible, key=lambda v: (summary["variants"][v]["recall"], summary["variants"][v]["questions"] or 0,
                                                       summary["variants"][v]["koesten"] or 0)) if eligible else None

    tag = (f"{args.instance}-{args.split}-{args.arm}-{'_'.join(variants)}" + ("-paper" if args.with_paper else "")
           + (f"-n{args.limit}" if args.limit else "")
           + ("" if args.draft_model == DRAFT_MODEL else "-" + args.draft_model.rsplit(".", 1)[-1].rsplit("/", 1)[-1]))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"{tag}.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
