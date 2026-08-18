"""Run the arm x query-class NDCG matrix + EDA diagnostics (report §2 P5).

Scores each description arm by isolating its field: a BM25 match on the arm field
ONLY (title excluded = the title-boost-0 control, so the title echo cannot make
all arms look equal), operator OR (paper BM25). Reuses ``metric_ndcg``.

The NDCG gain function ``(2**grade - 1)`` is scale-agnostic: it scores a 0/1
binary qrels file and a 0/1/2 graded one identically well, separating grade 2
from grade 1 wherever the input actually carries that distinction. The scale
consumed is read from the qrels artifact's own ``grade_scale`` declaration and
carried into the report (``"undeclared"`` for older files that predate it) --
graded-qrels-downstream-contract's re-read of this file (§6 item 12) confirmed
the arithmetic already handled 0/1/2 correctly; what was missing was the
report saying so instead of a hard-coded "binary" label.

Diagnostics the meeting needs:
  * the 6-arm x 4-class NDCG matrix (provisional),
  * per-facet arm separation (where AutoDDG's advantage lives),
  * k sensitivity (NDCG@5 vs @10).

Retrieval here is deterministic (BM25), so there is no retrieval-seed variance;
the stochastic element is the LLM judge (qrels) — re-judging is the stability
knob, noted but not run here. All numbers are PROVISIONAL (LLM-judged).

    python -m eval.run_matrix --queries eval/benchmark/queries.json \
        --qrels eval/benchmark/qrels.json --out eval/benchmark/matrix.json

Pass ``--per-query-out`` to additionally dump the individual (query, arm)
scores behind the aggregates, for arm x query-type interaction analysis.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from storage.opensearch_client import (
    AUCTUS_INDEX_NAME,
    DESCRIPTION_SOURCE_FIELDS,
    get_client,
)
from eval.corpus_frame import DEFAULT_CORPUS_FRAME, load_corpus_ids
from eval.provenance import code_version
from eval.retrieval_eval import metric_ndcg

ARMS = list(DESCRIPTION_SOURCE_FIELDS)
# Kept in lockstep with eval/generate_queries.py's QUERY_CLASSES (rewrite-query-generation
# tasks.md 1.2) — a duplicate constant, not re-exported, so both must move together.
QUERY_CLASSES = ("keyword", "describing")


def field_scores(
    os_client, field: str, text: str, size: int, operator: str = "or"
) -> dict[str, float]:
    """Single-field BM25 scores, id -> score, in the order OpenSearch returns
    them (score descending). The one place that builds this query shape --
    `search_arm` and `eval/three_channel_retrieval.py`'s BM25 channels both
    go through this rather than each constructing their own `match` body, so
    a change to how the field is queried can't drift out of step between the
    two paths silently (see three-channel-retrieval-scores tasks.md 5.1/5.2:
    that drift is exactly what a stale second copy produced before)."""
    resp = os_client.search(
        index=AUCTUS_INDEX_NAME,
        body={"query": {"match": {field: {"query": text, "operator": operator}}},
              "_source": False},
        size=size,
    )
    return {h["_id"]: h["_score"] for h in resp["hits"]["hits"]}


def search_arm(os_client, arm_field: str, text: str, size: int) -> list[str]:
    """Rank dataset ids by BM25 on the arm field ONLY (title excluded).

    ``size`` is the retrieval depth, NOT a reporting cutoff. Callers pass the
    corpus size so the full ranking comes back; which rank NDCG is cut at is
    then decided over the returned list (§2c: "not a retrieval-time
    commitment but a reporting parameter computed on the stored full
    ranking"). Passing a reporting k here is the bug this signature was
    renamed to make obvious.
    """
    return list(field_scores(os_client, arm_field, text, size, operator="or"))


def ndcg_for(ranked: list[str], grades: dict[str, int], k: int) -> float:
    retrieved_rel = [grades.get(i, 0) for i in ranked]
    ideal_rel = sorted(grades.values(), reverse=True)
    return metric_ndcg(retrieved_rel, ideal_rel, k)


def _mean(xs: list[float]) -> float:
    return round(statistics.mean(xs), 4) if xs else 0.0


def classify_query(qrels: dict[str, dict], query_id: str) -> str:
    """One of ``"scored"``, ``"judge_failed"``, ``"no_positive"``,
    ``"absent_from_qrels"`` (graded-qrels-downstream-contract D2). Shared
    between this module and ``three_channel_retrieval.py`` (D3) so the two
    readers' exclusion accounting cannot drift apart the way the BM25 query
    body once did before ``field_scores`` was extracted for the same reason."""
    entry = qrels.get(query_id)
    if entry is None:
        return "absent_from_qrels"
    if entry["judge_failed"]:
        return "judge_failed"
    if not any(entry["grades"].values()):
        return "no_positive"
    return "scored"


def parse_qrels(qrels_doc: dict) -> tuple[dict[str, dict], str]:
    """Parse a qrels artifact's ``queries`` list into the ``{query_id:
    {"grades": ..., "judge_failed": ...}}`` shape ``run()``/``classify_query``
    expect, plus the artifact's declared ``grade_scale`` (``"undeclared"`` if
    the artifact predates that field). Pure and side-effect free so it can be
    tested without an OpenSearch client."""
    grade_scale = qrels_doc.get("grade_scale", "undeclared")
    qrels = {
        q["query_id"]: {
            "grades": {i: int(v) for i, v in q.get("relevant", {}).items()},
            "judge_failed": bool(q.get("judge_failed")),
        }
        for q in qrels_doc["queries"]
    }
    return qrels, grade_scale


def run(os_client, queries: list[dict], qrels: dict[str, dict],
        cutoffs: list[int], retrieval_size: int | None = None):
    """Return ``{cutoff: aggregates}`` -- per-arm NDCG keyed overall / by
    class / by facet, for every requested reporting cutoff.

    **Retrieval happens once per (arm, query), at ``retrieval_size``**, and
    every cutoff is scored against that one stored ranking. This is §2c's
    rule in code: the rank NDCG is cut at is a reporting parameter, not a
    retrieval-time commitment, so asking for a second cutoff must not cost a
    second trip to the index. ``retrieval_size`` defaults to the round's
    corpus size read from the frame file -- never a live index document
    count, which is what let 144 indexed documents pass for a 100-dataset
    corpus unnoticed (plan-drift-audit finding 1).

    ``qrels`` maps query_id -> ``{"grades": {dataset_id: int}, "judge_failed":
    bool}``. Three distinct reasons a query does not contribute to the
    aggregates are counted separately rather than folded into one number
    (graded-qrels-downstream-contract D2): the query id is missing from the
    qrels artifact entirely (``skipped_absent_from_qrels`` -- the queries file
    and the qrels file disagree about what exists); the judge's batch for that
    query failed outright (``skipped_judge_failed`` -- a data hole, not a
    measurement); or the judge judged it and found nothing relevant
    (``skipped_no_positive`` -- a real result, cannot discriminate arms). A
    reconciliation assertion below is the check that no query silently falls
    through all four buckets.

    Also returns ``per_query``: the individual (query, arm) scores the
    aggregates are built from. These are the same floats that get averaged --
    collected here rather than recomputed downstream, so a dump and the
    aggregates cannot drift apart. ``facet`` is carried through verbatim --
    each query carries exactly one assigned facet (rewrite-query-generation),
    so no multi-label folding question arises here. Writing them out is the
    caller's decision; this function does no I/O.
    """
    if retrieval_size is None:
        retrieval_size = len(load_corpus_ids())
    overall = {k: {a: [] for a in ARMS} for k in cutoffs}
    by_class = {k: {a: {c: [] for c in QUERY_CLASSES} for a in ARMS} for k in cutoffs}
    by_facet: dict[int, dict[str, dict[str, list]]] = {
        k: {a: {} for a in ARMS} for k in cutoffs}
    per_query: dict[int, list[dict]] = {k: [] for k in cutoffs}
    scored = 0
    skipped_judge_failed = 0
    skipped_no_positive = 0
    skipped_absent_from_qrels = 0
    for q in queries:
        status = classify_query(qrels, q["query_id"])
        if status == "absent_from_qrels":
            skipped_absent_from_qrels += 1
            continue
        if status == "judge_failed":
            skipped_judge_failed += 1
            continue
        if status == "no_positive":
            skipped_no_positive += 1
            continue  # judged, no positive -> cannot discriminate arms
        grades = qrels[q["query_id"]]["grades"]
        scored += 1
        n_relevant = sum(1 for v in grades.values() if v > 0)
        n_relevant_by_grade: dict[int, int] = {}
        for v in grades.values():
            if v > 0:
                n_relevant_by_grade[v] = n_relevant_by_grade.get(v, 0) + 1
        for a in ARMS:
            # One retrieval, every cutoff scored from it.
            ranked = search_arm(os_client, DESCRIPTION_SOURCE_FIELDS[a],
                                q["text"], retrieval_size)
            for k in cutoffs:
                s = ndcg_for(ranked, grades, k)
                overall[k][a].append(s)
                if q["query_class"] in by_class[k][a]:
                    by_class[k][a][q["query_class"]].append(s)
                by_facet[k][a].setdefault(q["facet"], []).append(s)
                per_query[k].append({
                    "query_id": q["query_id"],
                    "arm": a,
                    "facet": q["facet"],
                    "query_class": q["query_class"],
                    "ndcg": s,
                    "n_relevant": n_relevant,
                    "n_relevant_by_grade": n_relevant_by_grade,
                })

    total_excluded = skipped_judge_failed + skipped_no_positive + skipped_absent_from_qrels
    assert scored + total_excluded == len(queries), (
        f"reconciliation failed: scored={scored} + skipped_judge_failed="
        f"{skipped_judge_failed} + skipped_no_positive={skipped_no_positive} + "
        f"skipped_absent_from_qrels={skipped_absent_from_qrels} = "
        f"{scored + total_excluded}, but {len(queries)} queries were supplied"
    )
    return {
        k: {
            "scored_queries": scored,
            "skipped_judge_failed": skipped_judge_failed,
            "skipped_no_positive": skipped_no_positive,
            "skipped_absent_from_qrels": skipped_absent_from_qrels,
            "retrieval_size": retrieval_size,
            "overall": {a: _mean(overall[k][a]) for a in ARMS},
            "by_class": {a: {c: _mean(by_class[k][a][c]) for c in QUERY_CLASSES}
                         for a in ARMS},
            "by_facet": {a: {f: _mean(v) for f, v in by_facet[k][a].items()}
                         for a in ARMS},
            "per_query": per_query[k],
        }
        for k in cutoffs
    }


def _print_matrix(title: str, agg: dict):
    print(f"\n{title}  (scored={agg['scored_queries']}, "
          f"skipped_judge_failed={agg['skipped_judge_failed']}, "
          f"skipped_no_positive={agg['skipped_no_positive']}, "
          f"skipped_absent_from_qrels={agg['skipped_absent_from_qrels']})")
    header = "arm".ljust(14) + "overall".rjust(9) + "".join(c[:9].rjust(11) for c in QUERY_CLASSES)
    print(header)
    for a in ARMS:
        row = a.ljust(14) + f"{agg['overall'][a]:.4f}".rjust(9)
        row += "".join(f"{agg['by_class'][a][c]:.4f}".rjust(11) for c in QUERY_CLASSES)
        print(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # No defaults on --queries/--qrels (graded-qrels-downstream-contract D5):
    # a default here would silently score whatever the last round left behind
    # against files that describe an entirely different round, with no error
    # -- the same footgun --per-query-out was already deliberately built
    # without a default to avoid. Every invocation must state its inputs.
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--out", default="eval/benchmark/matrix.json")
    parser.add_argument("--k", type=int, default=10,
                        help="primary NDCG reporting cutoff. This is applied "
                             "to a full ranking retrieved over the whole "
                             "corpus -- it does NOT limit retrieval, so "
                             "changing it is a recomputation, not a re-run "
                             "(§2c)")
    parser.add_argument("--corpus-frame", default=str(DEFAULT_CORPUS_FRAME),
                        help="frame file whose 'corpus_ids' sets the "
                             "retrieval depth (never a live index count)")
    parser.add_argument(
        "--per-query-out",
        help="Optional path for the per-(query, arm) NDCG dump at the primary k. "
             "No default on purpose: a default path could resolve onto an "
             "existing artifact, so a frozen run can only be overwritten by "
             "naming it explicitly. Omit to leave behavior unchanged.",
    )
    args = parser.parse_args(argv)

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    qrels_doc = json.loads(Path(args.qrels).read_text(encoding="utf-8"))
    qrels, grade_scale = parse_qrels(qrels_doc)

    os_client = get_client()
    # One retrieval pass, both cutoffs scored from it. Previously this was two
    # run() calls and therefore two full retrieval passes -- the sensitivity
    # row cost a second trip to the index for numbers the first pass's
    # rankings already contained (§2c).
    retrieval_size = len(load_corpus_ids(Path(args.corpus_frame)))
    cutoffs = [args.k] if args.k == 5 else [args.k, 5]
    scored = run(os_client, queries, qrels, cutoffs, retrieval_size)
    primary = scored[args.k]
    k_sens = scored[5]

    # graded-qrels-downstream-contract (2026-08-18) renamed "binary"/
    # "binary_at_k5" to "ndcg"/"ndcg_at_k5": those keys held graded numbers
    # under a name that promised binary ones, and a key encoding a scale is
    # guaranteed to misdescribe the artifact whenever the other scale is
    # scored. The scale now lives in the "grade_scale" field only. This is a
    # deliberate breaking change to the report schema -- the previous round's
    # matrix*.json files are shakedown artifacts this round owes no
    # compatibility to (author decision, 2026-08-18) -- so this file no
    # longer claims byte-identical output with what it produced before.
    per_query = primary.pop("per_query")
    k_sens.pop("per_query", None)

    report = {
        "code_version": code_version(),
        "index": AUCTUS_INDEX_NAME,
        "k": args.k,
        "controls": {"title_boost": 0, "operator": "or", "retrieval": "deterministic BM25"},
        "grade_scale": grade_scale,
        "provenance": f"PROVISIONAL: LLM-judged qrels (grade_scale={grade_scale}), "
                      "shakedown not benchmark-grade.",
        "ndcg": primary,
        "ndcg_at_k5": {"overall": k_sens["overall"]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.per_query_out:
        pq_report = {
            "code_version": report["code_version"],
            "index": AUCTUS_INDEX_NAME,
            "k": args.k,
            "queries_source": args.queries,
            "qrels_source": args.qrels,
            "controls": report["controls"],
            "grade_scale": grade_scale,
            "provenance": report["provenance"],
            "scored_queries": primary["scored_queries"],
            "records": per_query,
        }
        pq_out = Path(args.per_query_out)
        pq_out.parent.mkdir(parents=True, exist_ok=True)
        pq_out.write_text(json.dumps(pq_report, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    _print_matrix(f"NDCG@{args.k}  (grade_scale={grade_scale})", primary)
    print(f"\nOverall ordering (grade_scale={grade_scale}):",
          " > ".join(f"{a}={primary['overall'][a]:.3f}"
                     for a in sorted(ARMS, key=lambda a: -primary['overall'][a])))
    print(f"\nPer-facet NDCG (arm x facet, grade_scale={grade_scale}):")
    facets = sorted({f for a in ARMS for f in primary["by_facet"][a]})
    print("facet".ljust(20) + "".join(a[:9].rjust(11) for a in ARMS))
    for f in facets:
        print(f.ljust(20) + "".join(f"{primary['by_facet'][a].get(f,0):.3f}".rjust(11) for a in ARMS))
    print(f"\n-> {out}")
    if args.per_query_out:
        print(f"-> {args.per_query_out}  ({len(per_query)} per-query records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
