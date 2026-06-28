"""Retrieval-performance evaluation for AutoDDG descriptions (NDCG@k).

Compares dataset retrieval quality of the live Auctus ``/search`` endpoint when
the full-text query targets different description fields:

    original -> the dataset's original portal description (the "before" baseline)
    ufd      -> AutoDDG User-Focused Description
    sfd      -> AutoDDG Search-Focused Description

For each (query, source) it issues the query to the running backend, takes the
ranked dataset ids, looks up their graded relevance from a qrels file, and
computes NDCG@k. It then averages NDCG@k across queries per source and prints a
comparison table. This is the extrinsic / retrieval evaluation from the AutoDDG
paper (§3.1, §4.2); OpenSearch's default scoring is BM25, matching the paper.

Prerequisites:
  * The backend is running and reachable (default http://localhost:8000).
  * The benchmark datasets referenced by the qrels are already ingested, AND
    re-ingested after the Phase 1-3 changes so UFD/SFD fields are populated
    (otherwise ufd/sfd will score ~0 because those fields are empty).

The NDCG implementation mirrors ``autoddg.ranking.metrics.compute_ndcg``; we
reuse that module when importable and fall back to an identical local
implementation so the script also runs in environments without ``autoddg``
installed (e.g. the local Intel-mac backend venv).

Usage:
    python -m eval.retrieval_eval --queries eval/sample_queries.json
    python -m eval.retrieval_eval --queries my.json --sources original ufd sfd \
        --ks 5 10 15 20 --endpoint http://localhost:8000/search

Queries file format (JSON):
    {
      "queries": [
        {
          "query_id": "q1",
          "text": "yellow taxi trips",
          "relevant": {"<dataset_id>": 2, "<dataset_id2>": 1}
        }
      ]
    }
``relevant`` maps dataset id -> graded relevance (e.g. 0-3). Datasets not listed
are treated as relevance 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from typing import Any, Iterable

# --- NDCG: reuse autoddg's implementation when available, else identical fallback ---
try:  # pragma: no cover - depends on environment
    from autoddg.ranking.metrics import compute_ndcg
except Exception:  # autoddg not installed here; use an identical local copy

    def _compute_dcg(relevances: Iterable[float], p: int) -> float:
        dcg = 0.0
        for index, relevance in enumerate(relevances):
            if index >= p:
                break
            dcg += (2**relevance - 1) / math.log2(index + 2)
        return dcg

    def compute_ndcg(
        retrieved_relevances: Iterable[float],
        ideal_relevances: Iterable[float],
        p: int,
    ) -> float:
        retrieved = list(retrieved_relevances)
        ideal = list(ideal_relevances)
        dcg = _compute_dcg(retrieved, p)
        idcg = _compute_dcg(ideal, p)
        return dcg / idcg if idcg > 0 else 0.0


DEFAULT_ENDPOINT = "http://localhost:8000/search"
DEFAULT_SOURCES = ["original", "ufd", "sfd"]
DEFAULT_KS = [5, 10, 15, 20]


def run_query(endpoint: str, text: str, description_source: str, size: int) -> list[str]:
    """POST one query to the /search endpoint and return ranked dataset ids."""
    payload = json.dumps(
        {"query": text, "description_source": description_source}
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    results = body.get("results", []) or []
    ids: list[str] = []
    for hit in results[:size]:
        dataset_id = hit.get("id") or hit.get("_id")
        if dataset_id is not None:
            ids.append(str(dataset_id))
    return ids


def ndcg_for_ranking(
    ranked_ids: list[str], relevant: dict[str, float], ks: list[int]
) -> dict[int, float]:
    """NDCG@k for one ranking given graded relevance judgments."""
    retrieved_rel = [float(relevant.get(did, 0.0)) for did in ranked_ids]
    ideal_rel = sorted((float(v) for v in relevant.values()), reverse=True)
    return {k: compute_ndcg(retrieved_rel, ideal_rel, k) for k in ks}


def evaluate(
    queries: list[dict[str, Any]],
    endpoint: str,
    sources: list[str],
    ks: list[int],
    size: int,
) -> dict[str, dict[int, float]]:
    """Average NDCG@k across all queries, per description source."""
    totals: dict[str, dict[int, float]] = {s: {k: 0.0 for k in ks} for s in sources}
    counts: dict[str, int] = {s: 0 for s in sources}

    for q in queries:
        text = q["text"]
        relevant = {str(k): float(v) for k, v in (q.get("relevant") or {}).items()}
        if not relevant:
            print(f"  [skip] query {q.get('query_id')!r} has no relevance judgments")
            continue
        for source in sources:
            try:
                ranked = run_query(endpoint, text, source, size)
            except urllib.error.URLError as exc:
                print(f"  [error] query {q.get('query_id')!r} source={source}: {exc}")
                continue
            scores = ndcg_for_ranking(ranked, relevant, ks)
            for k in ks:
                totals[source][k] += scores[k]
            counts[source] += 1

    averaged: dict[str, dict[int, float]] = {}
    for source in sources:
        n = counts[source] or 1
        averaged[source] = {k: totals[source][k] / n for k in ks}
    return averaged


def print_table(averaged: dict[str, dict[int, float]], ks: list[int]) -> None:
    header = "source".ljust(10) + "".join(f"NDCG@{k}".rjust(10) for k in ks)
    print("\n" + header)
    print("-" * len(header))
    for source, scores in averaged.items():
        row = source.ljust(10) + "".join(f"{scores[k]:.4f}".rjust(10) for k in ks)
        print(row)
    print("\n(Expected direction from the AutoDDG paper: sfd >= ufd >= original.)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="Path to queries+qrels JSON")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument(
        "--size", type=int, default=20, help="Top-N results to score per query"
    )
    args = parser.parse_args(argv)

    with open(args.queries, encoding="utf-8") as fh:
        data = json.load(fh)
    queries = data.get("queries", [])
    if not queries:
        print("No queries found in file.", file=sys.stderr)
        return 1

    print(f"Evaluating {len(queries)} queries against {args.endpoint}")
    print(f"Sources: {args.sources}  |  k: {args.ks}")
    averaged = evaluate(queries, args.endpoint, args.sources, args.ks, args.size)
    print_table(averaged, args.ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
