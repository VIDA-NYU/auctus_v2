"""Two-judge `max` merge: per-judge graded qrels -> one merged graded qrels file.

Q7 (`next-round-design-decisions.md`): `max` is the headline two-judge
aggregation rule under grades -- "either judge says >= grade g", generalising
the prior binary round's union rule (inflation is arm-neutral) rather than
needing a new argument defended from scratch. `mean` is a sensitivity row,
not computed here (§6 item 12 territory) -- this writer's whole job is to
make that recompute free later by persisting BOTH judges' raw grades per
pair, not just the merged one (D2/§1c item 4).

Two judges stay two separate `judge_qrels.py` runs (D2) rather than one
multi-model pass, so per-judge failure handling stays simple: today a failed
batch is recorded per query per judge, not folded into a merge decision.
This writer is the separate step that does the folding.

A query where EITHER judge failed is NOT merged -- it is marked
`judge_failed` in the output and carries no grades. A `max` grade computed
from only one judge's opinion (because the other judge's batch call failed)
would silently look like a real two-judge merge; that is worse than an
honest gap the caller has to notice.

    python -m eval.merge_qrels judgeA_qrels.json judgeB_qrels.json \
        --out eval/benchmark/qrels_merged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.judge_agreement import load_judge
from eval.provenance import code_version


def merge_max(a: dict, b: dict) -> dict:
    """Merge two `load_judge()` dicts with per-pair grade = max(judge A, judge B).

    Returns a dict with ``queries`` (the merged qrels entries, judge_qrels.py
    output shape plus the raw-per-judge block) and summary counts.
    """
    query_ids = sorted(set(a["text"]) | set(b["text"]))
    out_queries = []
    n_merged = n_failed = 0
    for qid in query_ids:
        text = a["text"].get(qid) or b["text"].get(qid) or ""
        a_failed = qid in a["failed"]
        b_failed = qid in b["failed"]
        if a_failed or b_failed:
            n_failed += 1
            failed_by = [j["name"] for j, is_f in ((a, a_failed), (b, b_failed)) if is_f]
            out_queries.append({
                "query_id": qid,
                "text": text,
                "relevant": {},
                "unverified": [],
                "judge_failed": f"upstream judge(s) failed this query: {failed_by}",
                "raw_grades": None,
            })
            continue

        n_merged += 1
        a_grades = a["labels"].get(qid, {})
        b_grades = b["labels"].get(qid, {})
        a_unverified = a["unverified"].get(qid, set())
        b_unverified = b["unverified"].get(qid, set())
        dataset_ids = sorted(set(a_grades) | set(b_grades))

        merged: dict[str, int] = {}
        raw: dict[str, dict] = {}
        merged_unverified: list[str] = []
        for did in dataset_ids:
            ga = int(a_grades.get(did, 0))
            gb = int(b_grades.get(did, 0))
            g = max(ga, gb)
            raw[did] = {a["name"]: ga, b["name"]: gb}
            if g > 0:
                merged[did] = g
            # Flagged if either judge flagged it -- record which one(s), so
            # "unverified because A said so" and "unverified because both
            # said so" stay distinguishable in the merged file.
            flagged_by = [j["name"] for j, flagged in
                         ((a, did in a_unverified), (b, did in b_unverified)) if flagged]
            if flagged_by:
                merged_unverified.append(did)
                raw[did]["unverified_by"] = flagged_by

        out_queries.append({
            "query_id": qid,
            "text": text,
            "relevant": merged,
            "unverified": sorted(merged_unverified),
            "raw_grades": raw,
        })

    return {
        "queries": out_queries,
        "queries_merged": n_merged,
        "queries_judge_failed": n_failed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qrels_a", help="first judge's qrels file")
    parser.add_argument("qrels_b", help="second judge's qrels file")
    parser.add_argument("--out", default="eval/benchmark/qrels_merged.json")
    args = parser.parse_args(argv)

    judge_a = load_judge(Path(args.qrels_a))
    judge_b = load_judge(Path(args.qrels_b))

    # Reuses judge_agreement.py's existing role-conflict refusal rather than
    # writing a second one (task 4.4): a judge that also authored material
    # under test is a contrast, never an input to a merged label set.
    conflicted = [j["name"] for j in (judge_a, judge_b) if j["role_conflict"]]
    if conflicted:
        raise SystemExit(
            f"refusing to merge labels from judge(s) with a role conflict: "
            f"{conflicted} — they authored material under test")

    merged = merge_max(judge_a, judge_b)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "code_version": code_version(),
        "_provenance": "PROVISIONAL merged GRADED (0/1/2) qrels. Two-judge "
                       "`max` aggregation: each pair's grade is the maximum "
                       "across the two source judges. NOT a `mean` merge -- "
                       "do not mistake this file for one. Both judges' raw "
                       "grades are retained per pair (raw_grades) so the "
                       "`mean` sensitivity row is computable without "
                       "re-judging.",
        "grade_scale": "0/1/2",
        "aggregation": "max",
        "judges": [
            {"model": judge_a["name"], "lab": judge_a["lab"], "qrels": judge_a["path"]},
            {"model": judge_b["name"], "lab": judge_b["lab"], "qrels": judge_b["path"]},
        ],
        "run": {
            "queries_merged": merged["queries_merged"],
            "queries_judge_failed": merged["queries_judge_failed"],
        },
        "queries": merged["queries"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Merged {merged['queries_merged']} queries "
          f"({merged['queries_judge_failed']} judge_failed, excluded from merge) -> {out}")
    print(f"  judges: {judge_a['name']} x {judge_b['name']}  aggregation=max")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
