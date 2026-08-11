"""Inter-judge agreement over a shared judgment pool.

Reports pairwise Cohen's kappa plus each judge's positive rate. The positive
rate is not decoration: kappa collapses toward 0 when one class dominates, so a
low kappa on a heavily imbalanced pool must not be read as "the judges disagree
wildly" without seeing the marginals.

Agreement measures CONSISTENCY, not correctness. Human calibration is a separate,
gated phase — nothing here licenses calling any judge right.

Queries a judge failed on carry NO labels (a parse/API failure is not a judgment
of "nothing relevant"), so such queries are dropped from every pair they touch
and reported as excluded.

    python -m eval.judge_agreement --pool eval/benchmark/pool_n32.json \
        eval/benchmark/qrels_n32.json eval/benchmark/qrels_n32_gpt5mini.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def load_judge(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    judge = data.get("judge") or {}
    labels: dict[str, dict[str, int]] = {}
    failed: set[str] = set()
    for q in data["queries"]:
        if q.get("judge_failed"):
            failed.add(q["query_id"])
            continue
        labels[q["query_id"]] = {k: int(v) for k, v in (q.get("relevant") or {}).items()}
    return {
        "name": judge.get("model") or path.stem,
        "lab": judge.get("lab", "unknown"),
        "temperature_pinned": judge.get("temperature_pinned",
                                        judge.get("deterministic")),
        # Set on any judge that also authored material under test; such a judge
        # is a contrast, never an input to a merged label set.
        "role_conflict": judge.get("role_conflict"),
        "labels": labels,
        "failed": failed,
        "path": str(path),
    }


def vectorise(judge: dict, pool: dict, query_ids: list[str]) -> list[int]:
    """Flatten to one 0/1 label per pooled (query, dataset) pair."""
    out: list[int] = []
    for qid in query_ids:
        relevant = judge["labels"][qid]
        for dataset_id in pool[qid]:
            out.append(1 if relevant.get(dataset_id, 0) > 0 else 0)
    return out


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    pe = 0.0
    for label in (0, 1):
        pe += (a.count(label) / n) * (b.count(label) / n)
    if pe == 1.0:  # both judges constant and identical — kappa undefined
        return float("nan")
    return (po - pe) / (1 - pe)


def write_agreement_subset(judges: list[dict], pool: dict, query_ids: list[str],
                           out_path: Path) -> dict:
    """Emit the pairs the clean judges agree on, plus the ones they split on.

    This is a HANDOFF artifact for targeting human annotation, not a label set
    to score. Two structural guards keep it from being scored by accident:

    - agreed labels live under ``relevant_agreed``, not ``relevant``, so
      ``retrieval_eval`` finds no judgments and skips every query loudly
      instead of silently treating disputed pairs as non-relevant;
    - a judge carrying ``role_conflict`` (it authored material under test)
      is refused outright.
    """
    conflicted = [j["name"] for j in judges if j["role_conflict"]]
    if conflicted:
        raise SystemExit(
            f"refusing to merge labels from judge(s) with a role conflict: "
            f"{conflicted} — they authored material under test")
    if len(judges) != 2:
        raise SystemExit("the agreement subset is defined for exactly two clean "
                         f"judges; got {len(judges)}")

    a, b = judges
    out_queries, disagreements = [], []
    n_agreed = n_disputed = 0
    for qid in query_ids:
        agreed: dict[str, int] = {}
        for dataset_id in pool[qid]:
            la = 1 if a["labels"][qid].get(dataset_id, 0) > 0 else 0
            lb = 1 if b["labels"][qid].get(dataset_id, 0) > 0 else 0
            if la == lb:
                n_agreed += 1
                if la > 0:
                    agreed[dataset_id] = 1
            else:
                n_disputed += 1
                disagreements.append({"query_id": qid, "dataset_id": dataset_id,
                                      a["name"]: la, b["name"]: lb})
        out_queries.append({"query_id": qid, "relevant_agreed": agreed})

    report = {
        "_not_scorable": "DO NOT feed this file to retrieval_eval / run_matrix. "
                         "The harness treats an unlisted pair as non-relevant, "
                         "which would silently convert 'the judges disputed this' "
                         "into 'irrelevant'. Labels are therefore under "
                         "'relevant_agreed', not 'relevant'.",
        "_purpose": "Human-annotation targeting: the disputed pairs below are "
                    "where human labels buy the most.",
        "judges": [{"model": j["name"], "lab": j["lab"], "qrels": j["path"]}
                   for j in judges],
        "queries_covered": len(out_queries),
        "pairs_agreed": n_agreed,
        "pairs_disputed": n_disputed,
        "disagreements": disagreements,
        "queries": out_queries,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nAgreement subset -> {out_path}")
    print(f"  agreed {n_agreed} pairs, disputed {n_disputed} "
          f"({n_disputed / max(1, n_agreed + n_disputed):.1%}) — "
          f"NOT scorable, for human annotation only")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qrels", nargs="+", help="two or more qrels files")
    parser.add_argument("--pool", default="eval/benchmark/pool_n32.json")
    parser.add_argument("--out", default=None, help="optional JSON report path")
    parser.add_argument("--agreement-subset", default=None,
                        help="emit the clean-judge agreement subset (exactly two "
                             "conflict-free judges); a handoff artifact for human "
                             "annotation, NOT scorable")
    args = parser.parse_args(argv)

    pool_raw = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    pool = {q["query_id"]: q["pool"] for q in pool_raw["queries"]}

    judges = [load_judge(Path(p)) for p in args.qrels]
    if len(judges) < 2:
        raise SystemExit("need at least two qrels files to compare")

    # Only queries every judge actually labelled are comparable.
    shared = set.intersection(*(set(j["labels"]) for j in judges)) & set(pool)
    query_ids = sorted(shared)
    excluded = sorted(set(pool) - shared)

    print(f"Comparable queries: {len(query_ids)}  (excluded {len(excluded)})")
    for j in judges:
        if j["failed"]:
            print(f"  {j['name']}: {len(j['failed'])} failed queries -> {sorted(j['failed'])[:5]}")

    vectors = {j["name"]: vectorise(j, pool, query_ids) for j in judges}
    n_pairs = len(next(iter(vectors.values()))) if vectors else 0

    print(f"\nPositive rate over {n_pairs} pooled pairs:")
    marginals = {}
    for j in judges:
        v = vectors[j["name"]]
        rate = (sum(v) / len(v)) if v else float("nan")
        marginals[j["name"]] = rate
        det = "" if j["temperature_pinned"] in (None, True) else "  [temp unpinned]"
        print(f"  {j['name']:52s} {rate:6.1%}  ({j['lab']}){det}")

    print("\nPairwise Cohen's kappa:")
    pairs = []
    for a, b in itertools.combinations(judges, 2):
        k = cohens_kappa(vectors[a["name"]], vectors[b["name"]])
        same_lab = a["lab"] == b["lab"] and a["lab"] != "unknown"
        note = "  (same lab)" if same_lab else ""
        print(f"  {a['name']:40s} x {b['name']:40s}  kappa={k:.3f}{note}")
        pairs.append({"a": a["name"], "b": b["name"], "kappa": k, "same_lab": same_lab})

    print("\nNOTE: kappa is inter-judge consistency, NOT agreement with human "
          "ground truth. Human calibration remains a separate, gated phase.")

    if args.out:
        report = {
            "_note": "Inter-judge consistency only; not validated against humans.",
            "pool": args.pool,
            "comparable_queries": len(query_ids),
            "excluded_queries": excluded,
            "pooled_pairs": n_pairs,
            "judges": [{"model": j["name"], "lab": j["lab"],
                        "temperature_pinned": j["temperature_pinned"],
                        "positive_rate": marginals[j["name"]],
                        "failed_queries": sorted(j["failed"]),
                        "qrels": j["path"]} for j in judges],
            "pairwise_kappa": pairs,
        }
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nReport -> {args.out}")

    if args.agreement_subset:
        write_agreement_subset(judges, pool, query_ids, Path(args.agreement_subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
