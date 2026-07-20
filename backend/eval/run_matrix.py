"""Run the arm x query-class NDCG matrix + EDA diagnostics (report §2 P5).

Scores each description arm by isolating its field: a BM25 match on the arm field
ONLY (title excluded = the title-boost-0 control, so the title echo cannot make
all arms look equal), operator OR (paper BM25). Reuses ``metric_ndcg``.

Qrels are BINARY (0/1) — aligned with the AutoDDG paper's own evaluation
protocol, and the only grade distinction our LLM judge + judgment pool can
actually support at this corpus size.

Diagnostics the meeting needs:
  * the 6-arm x 4-class NDCG matrix (provisional),
  * per-facet arm separation (where AutoDDG's advantage lives),
  * k sensitivity (NDCG@5 vs @10).

Retrieval here is deterministic (BM25), so there is no retrieval-seed variance;
the stochastic element is the LLM judge (qrels) — re-judging is the stability
knob, noted but not run here. All numbers are PROVISIONAL (LLM-judged).

    python -m eval.run_matrix --queries eval/benchmark/queries.json \
        --qrels eval/benchmark/qrels.json --out eval/benchmark/matrix.json
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
from eval.retrieval_eval import metric_ndcg

ARMS = list(DESCRIPTION_SOURCE_FIELDS)
QUERY_CLASSES = ("keyword", "nl_requesting", "nl_describing", "nl_implying")


def search_arm(os_client, arm_field: str, text: str, k: int) -> list[str]:
    """Rank dataset ids by BM25 on the arm field ONLY (title excluded)."""
    resp = os_client.search(
        index=AUCTUS_INDEX_NAME,
        body={"query": {"match": {arm_field: {"query": text, "operator": "or"}}},
              "_source": False},
        size=k,
    )
    return [h["_id"] for h in resp["hits"]["hits"]]


def ndcg_for(ranked: list[str], grades: dict[str, int], k: int) -> float:
    retrieved_rel = [grades.get(i, 0) for i in ranked]
    ideal_rel = sorted(grades.values(), reverse=True)
    return metric_ndcg(retrieved_rel, ideal_rel, k)


def _mean(xs: list[float]) -> float:
    return round(statistics.mean(xs), 4) if xs else 0.0


def run(os_client, queries: list[dict], qrels: dict[str, dict], k: int):
    """Return per-arm NDCG aggregates, keyed overall / by class / by facet."""
    def grades_of(qid: str) -> dict[str, int]:
        return qrels.get(qid, {})

    # cache ranked ids per (arm, query) — independent of grades/k within this call
    overall = {a: [] for a in ARMS}
    by_class = {a: {c: [] for c in QUERY_CLASSES} for a in ARMS}
    by_facet: dict[str, dict[str, list]] = {a: {} for a in ARMS}
    scored = skipped = 0
    for q in queries:
        grades = grades_of(q["query_id"])
        if not any(grades.values()):
            skipped += 1
            continue  # no positive judgment -> cannot discriminate arms
        scored += 1
        for a in ARMS:
            ranked = search_arm(os_client, DESCRIPTION_SOURCE_FIELDS[a], q["text"], k)
            s = ndcg_for(ranked, grades, k)
            overall[a].append(s)
            if q["query_class"] in by_class[a]:
                by_class[a][q["query_class"]].append(s)
            by_facet[a].setdefault(q["facet"], []).append(s)
    return {
        "scored_queries": scored,
        "skipped_no_positive": skipped,
        "overall": {a: _mean(overall[a]) for a in ARMS},
        "by_class": {a: {c: _mean(by_class[a][c]) for c in QUERY_CLASSES} for a in ARMS},
        "by_facet": {a: {f: _mean(v) for f, v in by_facet[a].items()} for a in ARMS},
    }


def _print_matrix(title: str, agg: dict):
    print(f"\n{title}  (scored={agg['scored_queries']}, "
          f"skipped_no_positive={agg['skipped_no_positive']})")
    header = "arm".ljust(14) + "overall".rjust(9) + "".join(c[:9].rjust(11) for c in QUERY_CLASSES)
    print(header)
    for a in ARMS:
        row = a.ljust(14) + f"{agg['overall'][a]:.4f}".rjust(9)
        row += "".join(f"{agg['by_class'][a][c]:.4f}".rjust(11) for c in QUERY_CLASSES)
        print(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="eval/benchmark/queries.json")
    parser.add_argument("--qrels", default="eval/benchmark/qrels.json")
    parser.add_argument("--out", default="eval/benchmark/matrix.json")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    qrels_raw = json.loads(Path(args.qrels).read_text(encoding="utf-8"))["queries"]
    qrels = {q["query_id"]: {i: int(v) for i, v in q["relevant"].items()} for q in qrels_raw}

    os_client = get_client()
    binary = run(os_client, queries, qrels, args.k)
    k_sens = run(os_client, queries, qrels, 5)

    report = {
        "index": AUCTUS_INDEX_NAME,
        "k": args.k,
        "controls": {"title_boost": 0, "operator": "or", "retrieval": "deterministic BM25"},
        "provenance": "PROVISIONAL: LLM-judged binary qrels, shakedown not benchmark-grade.",
        "binary": binary,
        "binary_at_k5": {"overall": k_sens["overall"]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_matrix(f"BINARY NDCG@{args.k}", binary)
    print("\nBINARY overall ordering:",
          " > ".join(f"{a}={binary['overall'][a]:.3f}"
                     for a in sorted(ARMS, key=lambda a: -binary['overall'][a])))
    print("\nPer-facet BINARY NDCG (arm x facet):")
    facets = sorted({f for a in ARMS for f in binary["by_facet"][a]})
    print("facet".ljust(20) + "".join(a[:9].rjust(11) for a in ARMS))
    for f in facets:
        print(f.ljust(20) + "".join(f"{binary['by_facet'][a].get(f,0):.3f}".rjust(11) for a in ARMS))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
