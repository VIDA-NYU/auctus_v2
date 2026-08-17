"""Inter-judge agreement over a shared judgment pool.

Reports pairwise quadratic-weighted Cohen's kappa (over grades {0, 1, 2}) plus
each judge's grade distribution. The distribution is not decoration: kappa
collapses toward 0 when one class dominates, so a low kappa on a heavily
imbalanced pool must not be read as "the judges disagree wildly" without
seeing the marginals. Weighted kappa also means a 0-vs-2 split costs more than
a 0-vs-1 or 1-vs-2 split (quadratic weights), which plain agreement/disagreement
counting cannot express.

A graded round's dispute count is NOT comparable to a binary round's: three
grades admit more ways to differ than two, so a higher raw dispute count does
not by itself mean the judges agree less than they did under the old scale.

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
    # unverified/text captured for every query, succeeded or failed -- a
    # failed query still has a query_id/text worth carrying (e.g. for the
    # merge writer, which needs text regardless of whether either judge
    # actually produced grades for it).
    unverified: dict[str, set[str]] = {}
    text: dict[str, str] = {}
    failed: set[str] = set()
    for q in data["queries"]:
        text[q["query_id"]] = q.get("text", "")
        if q.get("judge_failed"):
            failed.add(q["query_id"])
            continue
        labels[q["query_id"]] = {k: int(v) for k, v in (q.get("relevant") or {}).items()}
        # Older (pre-judge-graded-qrels) qrels files carry no "unverified"
        # key at all -- .get() defaults to empty rather than erroring, so
        # this loader stays usable on binary-round files too.
        unverified[q["query_id"]] = set(q.get("unverified") or [])
    return {
        "name": judge.get("model") or path.stem,
        "lab": judge.get("lab", "unknown"),
        "temperature_pinned": judge.get("temperature_pinned",
                                        judge.get("deterministic")),
        "text": text,
        "unverified": unverified,
        # Set on any judge that also authored material under test; such a judge
        # is a contrast, never an input to a merged label set.
        "role_conflict": judge.get("role_conflict"),
        "labels": labels,
        "failed": failed,
        "path": str(path),
    }


GRADES = (0, 1, 2)


def vectorise(judge: dict, pool: dict, query_ids: list[str]) -> list[int]:
    """Flatten to one 0/1/2 grade per pooled (query, dataset) pair.

    Absent-as-0 is the stated qrels convention (D5, judge-graded-qrels
    design.md): a query that succeeded graded every pooled dataset, so a
    dataset missing from ``relevant`` was judged 0, not left unjudged.
    """
    out: list[int] = []
    for qid in query_ids:
        relevant = judge["labels"][qid]
        for dataset_id in pool[qid]:
            out.append(int(relevant.get(dataset_id, 0)))
    return out


def weighted_kappa(a: list[int], b: list[int], grades: tuple[int, ...] = GRADES) -> float:
    """Quadratic-weighted Cohen's kappa over an ordinal scale.

    Unlike plain kappa (exact-match agreement only), a weighted kappa charges
    a 0-vs-2 disagreement more than a 0-vs-1 or 1-vs-2 one — the right notion
    of "how much do two judges disagree" once the scale has an order, not just
    distinct categories. Quadratic weights (as opposed to linear) are the
    standard choice for ordinal agreement (Cohen 1968).

    Degenerate cases (n=0, or expected agreement pe=1 i.e. both judges
    constant and identical) return NaN, same guard as the binary version.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    k = len(grades)
    idx = {g: i for i, g in enumerate(grades)}
    # weight[i][j] = (i-j)^2 / (k-1)^2, so the max disagreement (0 vs k-1) is
    # weighted 1.0 and exact agreement is weighted 0.0.
    denom = (k - 1) ** 2
    weight = [[((i - j) ** 2) / denom for j in range(k)] for i in range(k)]

    observed = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1
    row_marg = [sum(observed[i]) for i in range(k)]
    col_marg = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    po = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(weight[i][j] * row_marg[i] * col_marg[j] for i in range(k) for j in range(k)) / (n * n)
    if pe == 0.0:
        # Both judges used one constant grade, and the same one (the only way
        # the weighted-expected-disagreement can be exactly 0) — 0/0,
        # undefined, same as the unweighted version's pe==1.0 guard.
        return float("nan")
    return 1.0 - (po / pe)


def grade_distribution(v: list[int], grades: tuple[int, ...] = GRADES) -> dict[int, int]:
    return {g: v.count(g) for g in grades}


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
            la = int(a["labels"][qid].get(dataset_id, 0))
            lb = int(b["labels"][qid].get(dataset_id, 0))
            # Disputed = grades differ by >=1 (the only kind of difference two
            # integers can have) — same effective test as the binary round's
            # exact-match check, now over {0,1,2} instead of {0,1}. Recorded
            # with its magnitude (0-vs-2 is a bigger split than 0-vs-1) since
            # that distinction didn't exist under binary.
            if la == lb:
                n_agreed += 1
                if la > 0:
                    agreed[dataset_id] = la
            else:
                n_disputed += 1
                disagreements.append({"query_id": qid, "dataset_id": dataset_id,
                                      a["name"]: la, b["name"]: lb,
                                      "diff": abs(la - lb)})
        out_queries.append({"query_id": qid, "relevant_agreed": agreed})

    report = {
        "_not_scorable": "DO NOT feed this file to retrieval_eval / run_matrix. "
                         "The harness treats an unlisted pair as non-relevant, "
                         "which would silently convert 'the judges disputed this' "
                         "into 'irrelevant'. Labels are therefore under "
                         "'relevant_agreed', not 'relevant'.",
        "_purpose": "Human-annotation targeting: the disputed pairs below are "
                    "where human labels buy the most.",
        "_dispute_count_note": "Not comparable to a binary round's dispute count "
                               "— three grades admit more ways to differ than "
                               "two, so a higher count here does not mean less "
                               "agreement than a prior binary run.",
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
    print("  NOTE: this dispute count is not comparable to a binary round's — "
          "grades {0,1,2} admit more ways to differ than {0,1}.")
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

    print(f"\nGrade distribution over {n_pairs} pooled pairs:")
    distributions = {}
    for j in judges:
        v = vectors[j["name"]]
        dist = grade_distribution(v)
        distributions[j["name"]] = dist
        shares = "  ".join(f"{g}={dist[g]/len(v):.1%}" if v else f"{g}=n/a" for g in GRADES)
        det = "" if j["temperature_pinned"] in (None, True) else "  [temp unpinned]"
        print(f"  {j['name']:52s} {shares}  ({j['lab']}){det}")

    print("\nPairwise quadratic-weighted kappa (over grades {0,1,2}):")
    pairs = []
    for a, b in itertools.combinations(judges, 2):
        k = weighted_kappa(vectors[a["name"]], vectors[b["name"]])
        same_lab = a["lab"] == b["lab"] and a["lab"] != "unknown"
        note = "  (same lab)" if same_lab else ""
        print(f"  {a['name']:40s} x {b['name']:40s}  kappa={k:.3f}{note}")
        pairs.append({"a": a["name"], "b": b["name"], "kappa": k, "same_lab": same_lab})

    print("\nNOTE: kappa is inter-judge consistency, NOT agreement with human "
          "ground truth. Human calibration remains a separate, gated phase.")
    print("NOTE: not comparable to a binary round's kappa or dispute count — "
          "three grades admit more ways to differ than two.")

    if args.out:
        report = {
            "_note": "Inter-judge consistency only; not validated against humans. "
                     "Grade distribution / weighted kappa are not comparable to "
                     "a binary round's positive rate / kappa.",
            "pool": args.pool,
            "comparable_queries": len(query_ids),
            "excluded_queries": excluded,
            "pooled_pairs": n_pairs,
            "judges": [{"model": j["name"], "lab": j["lab"],
                        "temperature_pinned": j["temperature_pinned"],
                        "grade_distribution": distributions[j["name"]],
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
