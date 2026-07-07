"""Retrieval-performance evaluation for AutoDDG descriptions (NDCG / MRR / Hit @k).

Compares dataset retrieval quality of the live Auctus ``/search`` endpoint when
the full-text query targets different description fields:

    original   -> the dataset's original portal description (the "before" baseline)
    llm_direct -> plain-LLM baseline written from the sample only (no AutoDDG grounding)
    ufd        -> AutoDDG User-Focused Description
    sfd        -> AutoDDG Search-Focused Description

For each query it issues the query once per source, takes the ranked dataset ids,
looks up their graded relevance from a qrels file, and computes the requested
metrics at each cut-off k. Metrics are averaged per source over the SAME set of
queries: if any source fails for a query (after one retry), the whole query is
discarded for every source, so the per-arm averages stay comparable.

Before scoring, a preflight against OpenSearch verifies that (a) every requested
arm's description field is actually populated in the index and (b) every dataset
id referenced by the qrels exists — otherwise an empty field or a typo'd id shows
up as a plausible-looking low score. Skip with --skip-preflight.

Results (run metadata, per-query rankings and scores, aggregates) are written to
a JSON file (--out) so runs are reproducible and per-query artifacts are available
for error analysis and significance testing; the printed table is just a view.

This is the extrinsic / retrieval evaluation from the AutoDDG paper (§3.1, §4.2).
The eval defaults to OR matching (--operator or) = plain BM25 as in the paper;
the backend's production default is AND. --title-boost 0 removes the title from
the match to eliminate the title-echo confound.

Prerequisites:
  * The backend is running and reachable (default http://localhost:8000), built
    from a commit whose /search supports size / match_operator / title_boost.
  * OpenSearch is reachable for the preflight (default http://localhost:9200).
  * The benchmark datasets referenced by the qrels are ingested WITH the AutoDDG
    arm fields populated (re-ingest, or run eval/backfill_descriptions.py).

Usage:
    python -m eval.retrieval_eval --queries eval/sample_queries.json
    python -m eval.retrieval_eval --queries my.json --sources original ufd sfd \
        --metrics ndcg mrr hit --ks 5 10 15 20 --title-boost 0 \
        --endpoint http://localhost:8000/search --out eval/results/run1.json

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
are treated as relevance 0. For known-item queries use a single id with grade 1
and the mrr/hit metrics.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
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
DEFAULT_OPENSEARCH = "http://localhost:9200"
DEFAULT_INDEX = "auctus_catalog_master"
DEFAULT_SOURCES = ["original", "llm_direct", "ufd", "sfd"]
DEFAULT_KS = [5, 10, 15, 20]
DEFAULT_METRICS = ["ndcg"]

SOURCE_FIELDS = {
    "original": "description",
    "llm_direct": "llm_direct_description",
    "ufd": "autoddg_description",
    "sfd": "autoddg_search_description",
}


# --- Metrics layer -----------------------------------------------------------
# Each metric maps (retrieved_rel, ideal_rel, k) -> float, where retrieved_rel is
# the graded relevance of the returned ranking (in rank order) and ideal_rel is
# the sorted-descending relevance of all judged datasets. A grade > 0 counts as
# relevant for the binary metrics (MRR / Hit).


def metric_ndcg(retrieved_rel: list[float], ideal_rel: list[float], k: int) -> float:
    return compute_ndcg(retrieved_rel, ideal_rel, k)


def metric_mrr(retrieved_rel: list[float], ideal_rel: list[float], k: int) -> float:
    for index, relevance in enumerate(retrieved_rel[:k]):
        if relevance > 0:
            return 1.0 / (index + 1)
    return 0.0


def metric_hit(retrieved_rel: list[float], ideal_rel: list[float], k: int) -> float:
    return 1.0 if any(rel > 0 for rel in retrieved_rel[:k]) else 0.0


METRICS = {"ndcg": metric_ndcg, "mrr": metric_mrr, "hit": metric_hit}


# --- HTTP helpers (urllib only, no extra dependencies) ------------------------


def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_query(
    endpoint: str,
    text: str,
    description_source: str,
    size: int,
    operator: str,
    title_boost: float,
) -> list[str]:
    """POST one query to the /search endpoint and return ranked dataset ids."""
    body = _post_json(
        endpoint,
        {
            "query": text,
            "description_source": description_source,
            "size": size,
            "match_operator": operator,
            "title_boost": title_boost,
        },
    )
    results = body.get("results", []) or []
    total = body.get("total_results", len(results))
    if len(results) < min(size, total):
        # An older backend that ignores `size` silently caps at 10 results,
        # which truncates every ranking beyond rank 10.
        raise RuntimeError(
            f"endpoint returned {len(results)} results but total={total} with "
            f"size={size}; is the backend built from a commit with /search size support?"
        )
    ids: list[str] = []
    for hit in results[:size]:
        dataset_id = hit.get("id") or hit.get("_id")
        if dataset_id is not None:
            ids.append(str(dataset_id))
    return ids


# --- Preflight ----------------------------------------------------------------


def preflight(
    opensearch: str,
    index: str,
    sources: list[str],
    qrels_ids: set[str],
) -> dict[str, Any]:
    """Verify arm-field coverage and qrels ids against the index.

    Returns a report dict (stored in the results metadata). Raises RuntimeError
    with an actionable message when the evaluation would be meaningless.
    """
    base = f"{opensearch.rstrip('/')}/{index}"

    def count(query: dict[str, Any] | None = None) -> int:
        payload = {"query": query} if query else {}
        return int(_post_json(f"{base}/_count", payload).get("count", 0))

    total_docs = count()
    coverage = {
        source: count({"exists": {"field": SOURCE_FIELDS[source]}})
        for source in sources
    }

    # Fetch the arm fields of the judged documents themselves: global coverage
    # can look fine while the specific qrels docs are unpopulated (e.g. after a
    # partial backfill), which would masquerade as plausible low scores.
    missing_ids: list[str] = []
    qrels_field_gaps: dict[str, list[str]] = {s: [] for s in sources}
    if qrels_ids:
        fields = ",".join(SOURCE_FIELDS[s] for s in sources)
        docs = _post_json(
            f"{base}/_mget?_source_includes={fields}", {"ids": sorted(qrels_ids)}
        ).get("docs", [])
        for d in docs:
            doc_id = d.get("_id", "?")
            if not d.get("found"):
                missing_ids.append(doc_id)
                continue
            source_doc = d.get("_source") or {}
            for s in sources:
                if not source_doc.get(SOURCE_FIELDS[s]):
                    qrels_field_gaps[s].append(doc_id)

    report = {
        "total_docs": total_docs,
        "field_coverage": coverage,
        "qrels_ids": len(qrels_ids),
        "missing_qrels_ids": missing_ids,
        "qrels_missing_field_ids": qrels_field_gaps,
    }

    problems = []
    if total_docs == 0:
        problems.append(f"index {index!r} is empty")
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        problems.append(
            f"{len(missing_ids)} qrels dataset id(s) not found in the index "
            f"(e.g. {preview}) — typo or not ingested"
        )
    for source in sources:
        # `original` is exempt: portal descriptions are legitimately absent for
        # some datasets, and that is exactly the "before" baseline being measured.
        if source == "original":
            continue
        gaps = qrels_field_gaps[source]
        if gaps and not missing_ids:
            preview = ", ".join(gaps[:5])
            problems.append(
                f"arm {source!r}: {len(gaps)} judged dataset(s) have an empty "
                f"{SOURCE_FIELDS[source]!r} field (e.g. {preview}) — "
                "re-ingest or run eval/backfill_descriptions.py first"
            )
    if problems:
        raise RuntimeError("preflight failed: " + "; ".join(problems))
    return report


# --- Evaluation ---------------------------------------------------------------


def score_ranking(
    ranked_ids: list[str],
    relevant: dict[str, float],
    metrics: list[str],
    ks: list[int],
) -> dict[str, dict[int, float]]:
    """All requested metrics @k for one ranking given graded judgments."""
    retrieved_rel = [float(relevant.get(did, 0.0)) for did in ranked_ids]
    ideal_rel = sorted((float(v) for v in relevant.values()), reverse=True)
    return {
        m: {k: METRICS[m](retrieved_rel, ideal_rel, k) for k in ks} for m in metrics
    }


def evaluate(
    queries: list[dict[str, Any]],
    endpoint: str,
    sources: list[str],
    metrics: list[str],
    ks: list[int],
    size: int,
    operator: str,
    title_boost: float,
) -> dict[str, Any]:
    """Run all queries against all sources; return per-query records + aggregates.

    A query is scored either for ALL sources or for none (failed queries are
    reported separately), so per-source averages are over the same denominator.
    """
    per_query: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    skipped: list[str] = []

    for q in queries:
        query_id = str(q.get("query_id"))
        text = q["text"]
        relevant = {str(k): float(v) for k, v in (q.get("relevant") or {}).items()}
        if not relevant:
            print(f"  [skip] query {query_id!r} has no relevance judgments")
            skipped.append(query_id)
            continue

        rankings: dict[str, list[str]] = {}
        error: str | None = None
        for source in sources:
            for attempt in (1, 2):
                try:
                    rankings[source] = run_query(
                        endpoint, text, source, size, operator, title_boost
                    )
                    break
                except (
                    urllib.error.URLError,
                    RuntimeError,
                    OSError,
                    json.JSONDecodeError,  # e.g. a proxy error page instead of JSON
                ) as exc:
                    if attempt == 2:
                        error = f"source={source}: {exc}"
            if error:
                break
        if error:
            # Atomic per query: one failing arm would silently shrink that arm's
            # denominator and make the averages incomparable.
            print(f"  [fail] query {query_id!r} dropped for all sources ({error})")
            failed.append({"query_id": query_id, "error": error})
            continue

        record: dict[str, Any] = {"query_id": query_id, "text": text}
        for source in sources:
            record[source] = {
                "ranked_ids": rankings[source],
                "scores": {
                    m: {str(k): v for k, v in per_k.items()}
                    for m, per_k in score_ranking(
                        rankings[source], relevant, metrics, ks
                    ).items()
                },
            }
        per_query.append(record)

    aggregates: dict[str, dict[str, dict[str, float | None]]] = {}
    n = len(per_query)
    for source in sources:
        aggregates[source] = {
            m: {
                str(k): (
                    sum(rec[source]["scores"][m][str(k)] for rec in per_query) / n
                    if n
                    else None  # not NaN: NaN is invalid JSON in the results artifact
                )
                for k in ks
            }
            for m in metrics
        }

    return {
        "per_query": per_query,
        "aggregates": aggregates,
        "evaluated_queries": n,
        "failed_queries": failed,
        "skipped_queries": skipped,
    }


# --- Reporting ----------------------------------------------------------------


def print_report(results: dict[str, Any], sources: list[str], metrics: list[str], ks: list[int]) -> None:
    n = results["evaluated_queries"]
    n_failed = len(results["failed_queries"])
    n_skipped = len(results["skipped_queries"])
    print(f"\nEvaluated {n} queries ({n_failed} failed, {n_skipped} skipped)")
    for metric in metrics:
        header = "source".ljust(10) + "".join(
            f"{metric.upper()}@{k}".rjust(10) for k in ks
        )
        print("\n" + header)
        print("-" * len(header))
        for source in sources:
            scores = results["aggregates"][source][metric]
            cells = "".join(
                (f"{scores[str(k)]:.4f}" if n else "n/a").rjust(10) for k in ks
            )
            print(source.ljust(10) + cells)
    print("\n(Expected direction from the AutoDDG paper: sfd >= ufd >= llm_direct ~ original.)")


def git_sha() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="Path to queries+qrels JSON")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, choices=sorted(SOURCE_FIELDS))
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=sorted(METRICS))
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument(
        "--size", type=int, default=20, help="Top-N results to request and score per query"
    )
    parser.add_argument(
        "--operator",
        choices=["or", "and"],
        default="or",
        help="Term matching: 'or' = plain BM25 as in the AutoDDG paper (default); "
        "'and' = the backend's stricter production behaviour",
    )
    parser.add_argument(
        "--title-boost",
        type=float,
        default=2.0,
        help="Weight of the title field; 0 removes the title from the match "
        "(eliminates the title-echo confound)",
    )
    parser.add_argument(
        "--opensearch",
        default=DEFAULT_OPENSEARCH,
        help="OpenSearch base URL for the preflight checks",
    )
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the field-coverage and qrels-id checks (not recommended)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Results JSON path (default eval/results/run-<timestamp>.json)",
    )
    args = parser.parse_args(argv)

    with open(args.queries, encoding="utf-8") as fh:
        data = json.load(fh)
    queries = data.get("queries", [])
    if not queries:
        print("No queries found in file.", file=sys.stderr)
        return 1
    if max(args.ks) > args.size:
        print(
            f"--size {args.size} is smaller than the largest k ({max(args.ks)}); "
            "metrics beyond --size would be computed on truncated rankings.",
            file=sys.stderr,
        )
        return 1

    qrels_ids = {
        str(did) for q in queries for did in (q.get("relevant") or {})
    }
    preflight_report: dict[str, Any] | None = None
    if args.skip_preflight:
        print("Preflight skipped (--skip-preflight).")
    else:
        try:
            preflight_report = preflight(args.opensearch, args.index, args.sources, qrels_ids)
        except (urllib.error.URLError, OSError) as exc:
            print(
                f"Preflight could not reach OpenSearch at {args.opensearch}: {exc}\n"
                "Fix the URL (--opensearch) or use --skip-preflight.",
                file=sys.stderr,
            )
            return 2
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            "Preflight OK: "
            + ", ".join(
                f"{s}={preflight_report['field_coverage'][s]}/{preflight_report['total_docs']}"
                for s in args.sources
            )
            + f" docs populated; {len(qrels_ids)} qrels ids present with populated arm fields"
        )

    print(f"Evaluating {len(queries)} queries against {args.endpoint}")
    print(
        f"Sources: {args.sources} | metrics: {args.metrics} | k: {args.ks} | "
        f"size={args.size} operator={args.operator} title_boost={args.title_boost:g}"
    )
    results = evaluate(
        queries,
        args.endpoint,
        args.sources,
        args.metrics,
        args.ks,
        args.size,
        args.operator,
        args.title_boost,
    )
    print_report(results, args.sources, args.metrics, args.ks)

    out_path = Path(
        args.out
        or Path(__file__).parent
        / "results"
        / f"run-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "endpoint": args.endpoint,
            "opensearch": args.opensearch,
            "index": args.index,
            "queries_file": args.queries,
            "sources": args.sources,
            "metrics": args.metrics,
            "ks": args.ks,
            "size": args.size,
            "operator": args.operator,
            "title_boost": args.title_boost,
            "git_sha": git_sha(),
            "preflight": preflight_report,
        },
        **results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    if results["evaluated_queries"] == 0:
        print("No query succeeded for all sources; aggregates are meaningless.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
