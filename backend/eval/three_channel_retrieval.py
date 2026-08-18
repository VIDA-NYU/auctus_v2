"""Three raw retrieval score channels per (query, dataset) — the main table.

Neither existing path can produce the main table. `retrieval_eval.py` posts to
`main.py`'s `/search` and gets one *combined* score back (title's share is
unrecoverable). `run_matrix.py`'s `search_arm` runs the arm field alone — no
title, no vector — a useful isolated control, but not the hybrid Auctus
actually serves.

This module retrieves three channels separately, straight against the
OpenSearch client (like `run_matrix.py`, NOT through `main.py`'s `/search` —
see design.md D4, and the note in `fix-stale-arm-literal`'s proposal.md that
this route has no dependency on that change landing first):

    title_BM25  -- `title` field alone
    arm_BM25    -- the arm's own field alone (channel `run_matrix.py:46`'s
                   `search_arm` already computes; reused via the same query
                   shape rather than re-derived, so §5's regression check is
                   comparing the same measurement, not a lookalike one)
    knn_score   -- `dataset_vector` alone, k = full corpus rather than
                   production's hardcoded k=10 (api/search.py:193, NOT edited
                   here -- that value is the departure being reported)

`title_BM25` and `knn_score` do not depend on which arm is being scored: the
title field is the same document field regardless of arm, and the production
vector is embedded once per document from `title + "\n\n" + description` at
ingest time (`storage/arq_worker.py:87-105`) -- arm-blind by construction, a
fixed constant per (query, dataset) as the proposal's "Why" section notes.
So each is retrieved ONCE per query and shared across all six arms, not
recomputed per arm; only `arm_BM25` actually varies per arm.

Both BM25 channels use `operator=or` (matches `run_matrix.py`'s `search_arm`
and `retrieval_eval.py`'s eval-harness default -- "plain BM25 as in the
AutoDDG paper" -- and is required for §5's regression check to compare the
same measurement; `main.py`'s different `and` default belongs to a different,
non-production endpoint). The per-field BM25 premise this whole module rests
on (a single-field query reproduces that field's own contribution inside a
`multi_match`) was verified under both operators before anything was built on
it -- see tasks.md §1.

"Exhaustive over the corpus" (design.md D2) means: no `size` cap is imposed
below the corpus's actual document count, so no BM25 match is truncated. It
does NOT mean padding in zero-score entries for documents that matched no
term on that field -- a document absent from a field's hit list has that
field's genuine BM25 relevance of 0, not a retrieval artifact. Corpus size is
read from the index at run time (`_count`), not hardcoded to the planned
100 -- item 14 (the resample) has not landed yet, so the live index may be a
different size, and hardcoding would silently truncate.

Usage:
    python -m eval.three_channel_retrieval \
        --queries eval/benchmark/queries.json --qrels eval/benchmark/qrels.json \
        --out eval/benchmark/three_channel.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from api.search import build_query_vector
from eval.provenance import code_version
from eval.retrieval_eval import metric_ndcg
from eval.run_matrix import ARMS, QUERY_CLASSES, classify_query, field_scores, parse_qrels
from storage.opensearch_client import (
    AUCTUS_INDEX_NAME,
    DEFAULT_TITLE_BOOST,
    DESCRIPTION_SOURCE_FIELDS,
    get_client,
)

VECTOR_FIELD = "dataset_vector"
BM25_OPERATOR = "or"  # see module docstring: matches run_matrix.py's search_arm


# --- Channels -------------------------------------------------------------


def corpus_size(os_client) -> int:
    return int(os_client.count(index=AUCTUS_INDEX_NAME).get("count", 0))


def bm25_channel(os_client, field: str, text: str, size: int) -> dict[str, float]:
    """Single-field BM25 scores for every matching document, no cap short of
    the corpus. A doc absent from the result matched no term on this field:
    genuine score 0, not a truncation (see module docstring).

    Thin wrapper over run_matrix.field_scores -- the query construction lives
    in exactly one place, shared with search_arm, so the two paths can't
    silently drift the way a second hand-written copy did before (see
    tasks.md 5.1/5.2)."""
    return field_scores(os_client, field, text, size, operator=BM25_OPERATOR)


def knn_channel(os_client, query_vector: list[float], k: int) -> dict[str, float]:
    resp = os_client.search(
        index=AUCTUS_INDEX_NAME,
        body={
            "query": {"knn": {VECTOR_FIELD: {"vector": query_vector, "k": k}}},
            "_source": False,
        },
        size=k,
    )
    return {h["_id"]: h["_score"] for h in resp["hits"]["hits"]}


# --- Combination rules, computed locally on stored channels ----------------


def combine_additive(
    title_ch: dict[str, float],
    arm_ch: dict[str, float],
    knn_ch: dict[str, float],
    title_weight: float,
) -> dict[str, float]:
    """`w*title + arm + knn` -- the main table's departure from production
    (design.md D1/D4): additive instead of best_fields, so no field is
    discarded and between-arm differences survive."""
    ids = set(title_ch) | set(arm_ch) | set(knn_ch)
    return {
        i: title_weight * title_ch.get(i, 0.0) + arm_ch.get(i, 0.0) + knn_ch.get(i, 0.0)
        for i in ids
    }


def combine_max(
    title_ch: dict[str, float],
    arm_ch: dict[str, float],
    knn_ch: dict[str, float],
    title_weight: float,
) -> dict[str, float]:
    """`max(w*title, arm) + knn` -- production's best_fields rule, reported
    alongside the additive table so the size of the departure is visible."""
    ids = set(title_ch) | set(arm_ch) | set(knn_ch)
    return {
        i: max(title_weight * title_ch.get(i, 0.0), arm_ch.get(i, 0.0)) + knn_ch.get(i, 0.0)
        for i in ids
    }


def rank(scores: dict[str, float]) -> list[str]:
    """Highest score first; ties broken by id so ranking is deterministic."""
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def title_share(
    ranked_ids: list[str],
    title_ch: dict[str, float],
    arm_ch: dict[str, float],
    knn_ch: dict[str, float],
    title_weight: float,
) -> list[float]:
    """`w*title / (w*title + arm + knn)` for docs actually in the reported
    ranking (task 4.4: "an additive combination is reported"). Deliberately
    scoped to `ranked_ids` (the top-k additive ranking) rather than the full
    exhaustive candidate pool: with k-NN retrieving up to the corpus size,
    most candidates are knn-only matches that never scored on title or arm at
    all (title_share == 0 by construction), and folding those in drags the
    median toward 0 regardless of what the actually-reported rows look like
    -- diluting the exact signal this check exists to catch."""
    shares = []
    for i in ranked_ids:
        t = title_weight * title_ch.get(i, 0.0)
        total = t + arm_ch.get(i, 0.0) + knn_ch.get(i, 0.0)
        if total > 0:
            shares.append(t / total)
    return shares


# --- Per-query retrieval ----------------------------------------------------


def retrieve_query(os_client, text: str, size: int) -> dict[str, Any]:
    """The channels shared by every arm for one query: title_BM25 (once) and
    knn_score (once, arm-blind -- see module docstring)."""
    query_vector = build_query_vector(text)
    return {
        "title_BM25": bm25_channel(os_client, "title", text, size),
        "knn_score": knn_channel(os_client, query_vector, size),
    }


def ndcg_for(ranked: list[str], grades: dict[str, int], k: int) -> float:
    retrieved_rel = [grades.get(i, 0) for i in ranked]
    ideal_rel = sorted(grades.values(), reverse=True)
    return metric_ndcg(retrieved_rel, ideal_rel, k)


def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 4) if xs else None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 4) if xs else None


# --- Full run ----------------------------------------------------------------


def run(
    os_client,
    queries: list[dict],
    qrels: dict[str, dict],
    k: int,
    title_weight: float,
    size: int | None = None,
) -> dict[str, Any]:
    """Retrieve all three channels per query, combine locally under both
    rules, and score. Returns aggregates for all three tables (additive,
    max_of_fields, isolated_bm25) plus the full per-query, per-arm record
    (raw channels + ranked lists) so the artifact is self-contained (task 3).
    """
    size = size or corpus_size(os_client)

    tables = {"additive", "max_of_fields", "isolated_bm25"}
    overall = {t: {a: [] for a in ARMS} for t in tables}
    by_class = {t: {a: {c: [] for c in QUERY_CLASSES} for a in ARMS} for t in tables}
    by_facet: dict[str, dict[str, dict[str, list]]] = {t: {a: {} for a in ARMS} for t in tables}
    title_shares: list[float] = []
    per_query: list[dict[str, Any]] = []
    scored = 0
    skipped_judge_failed = 0
    skipped_no_positive = 0
    skipped_absent_from_qrels = 0

    for q in queries:
        # classify_query is shared with run_matrix.run() (graded-qrels-
        # downstream-contract D2/D3) so the two readers' exclusion accounting
        # cannot drift apart the way the BM25 query body once did before
        # field_scores was extracted for the same reason.
        status = classify_query(qrels, q["query_id"])
        if status == "absent_from_qrels":
            skipped_absent_from_qrels += 1
            continue
        if status == "judge_failed":
            skipped_judge_failed += 1
            continue
        if status == "no_positive":
            skipped_no_positive += 1
            continue
        grades = qrels[q["query_id"]]["grades"]
        scored += 1

        shared = retrieve_query(os_client, q["text"], size)
        title_ch, knn_ch = shared["title_BM25"], shared["knn_score"]

        record: dict[str, Any] = {
            "query_id": q["query_id"],
            "facet": q["facet"],
            "query_class": q["query_class"],
            "title_BM25": title_ch,
            "knn_score": knn_ch,
            "arms": {},
        }

        for a in ARMS:
            arm_ch = bm25_channel(os_client, DESCRIPTION_SOURCE_FIELDS[a], q["text"], size)
            additive_scores = combine_additive(title_ch, arm_ch, knn_ch, title_weight)
            max_scores = combine_max(title_ch, arm_ch, knn_ch, title_weight)

            ranked_additive = rank(additive_scores)
            ranked_max = rank(max_scores)
            ranked_isolated = rank(arm_ch)

            per_table_ranked = {
                "additive": ranked_additive,
                "max_of_fields": ranked_max,
                "isolated_bm25": ranked_isolated,
            }
            record["arms"][a] = {
                "arm_BM25": arm_ch,
                "ranked": per_table_ranked,
            }

            title_shares.extend(
                title_share(ranked_additive[:k], title_ch, arm_ch, knn_ch, title_weight)
            )

            for t, ranked in per_table_ranked.items():
                s = ndcg_for(ranked, grades, k)
                overall[t][a].append(s)
                if q["query_class"] in by_class[t][a]:
                    by_class[t][a][q["query_class"]].append(s)
                by_facet[t][a].setdefault(q["facet"], []).append(s)

        per_query.append(record)

    total_excluded = skipped_judge_failed + skipped_no_positive + skipped_absent_from_qrels
    assert scored + total_excluded == len(queries), (
        f"reconciliation failed: scored={scored} + skipped_judge_failed="
        f"{skipped_judge_failed} + skipped_no_positive={skipped_no_positive} + "
        f"skipped_absent_from_qrels={skipped_absent_from_qrels} = "
        f"{scored + total_excluded}, but {len(queries)} queries were supplied"
    )
    return {
        "scored_queries": scored,
        "skipped_judge_failed": skipped_judge_failed,
        "skipped_no_positive": skipped_no_positive,
        "skipped_absent_from_qrels": skipped_absent_from_qrels,
        "corpus_size": size,
        "combination": {"title_weight": title_weight, "bm25_operator": BM25_OPERATOR},
        "tables": {
            t: {
                "overall": {a: _mean(overall[t][a]) for a in ARMS},
                "by_class": {a: {c: _mean(by_class[t][a][c]) for c in QUERY_CLASSES} for a in ARMS},
                "by_facet": {a: {f: _mean(v) for f, v in by_facet[t][a].items()} for a in ARMS},
            }
            for t in tables
        },
        "title_share": {
            "median": _median(title_shares),
            "n": len(title_shares),
        },
        "per_query": per_query,
    }


def _print_table(name: str, table: dict[str, Any]) -> None:
    print(f"\n{name}")
    header = "arm".ljust(14) + "overall".rjust(9) + "".join(c[:9].rjust(11) for c in QUERY_CLASSES)
    print(header)
    for a in ARMS:
        ov = table["overall"][a]
        row = a.ljust(14) + (f"{ov:.4f}" if ov is not None else "n/a").rjust(9)
        for c in QUERY_CLASSES:
            v = table["by_class"][a][c]
            row += (f"{v:.4f}" if v is not None else "n/a").rjust(11)
        print(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # No defaults on --queries/--qrels (graded-qrels-downstream-contract D5,
    # same reasoning as run_matrix.py): a stale default silently scores one
    # round's queries against a different round's qrels with no error.
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--out", default="eval/benchmark/three_channel.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--title-weight", type=float, default=DEFAULT_TITLE_BOOST,
        help="Title weight for the additive/max_of_fields tables (default: "
             "production's configured value)",
    )
    parser.add_argument(
        "--size", type=int, default=None,
        help="Result-count cap per channel; default is the live corpus size "
             "(exhaustive, task 2.3). Only for testing on a smaller slice.",
    )
    args = parser.parse_args(argv)

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    qrels_doc = json.loads(Path(args.qrels).read_text(encoding="utf-8"))
    qrels, grade_scale = parse_qrels(qrels_doc)

    os_client = get_client()
    result = run(os_client, queries, qrels, args.k, args.title_weight, args.size)

    per_query = result.pop("per_query")

    report = {
        "code_version": code_version(),
        "index": AUCTUS_INDEX_NAME,
        "k": args.k,
        "grade_scale": grade_scale,
        **result,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    per_query_out = out.with_name(out.stem + "_per_query.json")
    per_query_out.write_text(
        json.dumps(
            {
                "code_version": report["code_version"],
                "index": AUCTUS_INDEX_NAME,
                "k": args.k,
                "grade_scale": grade_scale,
                "combination": report["combination"],
                "records": per_query,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for t in ("additive", "max_of_fields", "isolated_bm25"):
        _print_table(f"{t.upper()} NDCG@{args.k}", report["tables"][t])
    print(f"\nMedian w*title share of additive total: {report['title_share']['median']} "
          f"(n={report['title_share']['n']})")
    print(f"\n-> {out}")
    print(f"-> {per_query_out}  ({len(per_query)} per-query records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
