"""Known-item scoring for the wording-variation track (item 10,
variation-track-known-item).

Each variation is scored per arm by whether its *source* dataset — never a
same-agency sibling — is retrieved within the reporting cutoff. This is a
different measurement from the main table's graded NDCG (`run_matrix.py`):
binary hit, not a relevance grade, and zero judge calls (the ground truth is
`source_dataset_id`, not a qrels file).

**Headline ranking is the isolated arm-only BM25 ranking** — the same
`field_scores`/`search_arm` shape `run_matrix.py` uses for its own headline
(title excluded, `operator="or"`), reused rather than re-derived so a
hit-rate difference caused by a control difference is not mistaken for a
finding about arms (design.md Decision 3). **Title and k-NN channels are
additionally retrieved once per variation query, arm-blind, and persisted
alongside the arm channel** — reusing `three_channel_retrieval.py`'s
`bm25_channel`/`knn_channel` rather than a second copy — so the additive and
`max_of_fields` rankings (whichever headlines the main table, not decided by
this change) remain recomputable after the run without a second retrieval
pass (Decision 3). This costs no extra retrieval beyond what the main table's
scorer already pays per query.

**Every candidate is persisted, not a top-N** (Decision 4): the loose-reading
sensitivity check (separately gated, judge calls required) needs to see what
outranked a missed source dataset, and re-deriving that later requires every
arm index to still be byte-identical to this run's — retrieval is cheap,
guaranteeing the index has not drifted is not.

Headline metric is **mean hit@k per arm** (insensitive to `n`); "all-`n`-hit"
is reported as a labelled additional column only (its strictness moves with
`n`, so it is not comparable across settings). `k` is a reporting parameter
computed from the stored full ranking, never a retrieval-time commitment.

    python -m eval.variation_scoring \
        --variations eval/benchmark/variations.json \
        --out eval/benchmark/variation_scoring.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from api.search import build_query_vector
from eval.corpus_frame import DEFAULT_CORPUS_FRAME, load_corpus_ids
from eval.provenance import code_version
from eval.run_matrix import ARMS, QUERY_CLASSES, field_scores
from eval.three_channel_retrieval import bm25_channel, knn_channel
from storage.opensearch_client import AUCTUS_INDEX_NAME, DESCRIPTION_SOURCE_FIELDS, get_client

BM25_OPERATOR = "or"  # matches run_matrix.py's search_arm / three_channel_retrieval.py


def rank(scores: dict[str, float]) -> list[str]:
    """Highest score first; ties broken by id so ranking is deterministic."""
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def is_hit(ranked: list[str], source_dataset_id: str, k: int) -> bool:
    """Only the source dataset counts (spec: 'a sibling dataset from the same
    family SHALL NOT'). No sibling/family notion is applied here at all — the
    loose-reading check (separately gated) is where a judge, not a structural
    rule, decides whether an outranking dataset should also count."""
    return source_dataset_id in ranked[:k]


def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 4) if xs else None


def score_variation(
    os_client, variation: dict, size: int,
) -> dict[str, Any]:
    """Retrieve all three channels for one variation query, once (title_BM25
    and knn_score are arm-blind — same reasoning as three_channel_retrieval.py),
    then the arm-only BM25 channel per arm. Returns the full per-arm untruncated
    rankings plus raw channels, ready to persist."""
    text = variation["text"]
    title_ch = bm25_channel(os_client, "title", text, size)
    knn_ch = knn_channel(os_client, build_query_vector(text), size)

    arms: dict[str, dict] = {}
    for a in ARMS:
        arm_ch = field_scores(os_client, DESCRIPTION_SOURCE_FIELDS[a], text, size, operator=BM25_OPERATOR)
        arms[a] = {
            "arm_BM25": arm_ch,
            "ranked_isolated": rank(arm_ch),  # headline ranking (Decision 3)
        }
    return {"title_BM25": title_ch, "knn_score": knn_ch, "arms": arms}


def run(
    os_client, variations: list[dict], k: int, size: int | None = None,
    corpus_frame: Path = DEFAULT_CORPUS_FRAME,
) -> dict[str, Any]:
    """Score every variation, per arm, known-item. Returns aggregates (mean
    hit@k, all-n-hit) plus the full per-variation, per-arm record (channels +
    untruncated rankings) so the artifact is self-contained for the loose
    reading and for recomputing additive/max_of_fields rankings later."""
    size = size or len(load_corpus_ids(corpus_frame))

    hits: dict[str, list[bool]] = {a: [] for a in ARMS}
    by_class: dict[str, dict[str, list[bool]]] = {a: {c: [] for c in QUERY_CLASSES} for a in ARMS}
    by_facet: dict[str, dict[str, list[bool]]] = {a: {} for a in ARMS}
    # base_query_id -> arm -> list[bool], for the all-n-hit column (Decision 5:
    # variations carry their base_query_id, so grouping needs no extra lookup).
    by_base: dict[str, dict[str, list[bool]]] = {}
    per_variation: list[dict[str, Any]] = []

    for v in variations:
        scored = score_variation(os_client, v, size)
        base_id = v["base_query_id"]
        by_base.setdefault(base_id, {a: [] for a in ARMS})

        record: dict[str, Any] = {
            "variation_id": v["variation_id"],
            "base_query_id": base_id,
            "source_dataset_id": v["source_dataset_id"],
            "facet": v["facet"],
            "query_class": v["query_class"],
            "title_BM25": scored["title_BM25"],
            "knn_score": scored["knn_score"],
            "arms": {},
        }
        for a in ARMS:
            ranked = scored["arms"][a]["ranked_isolated"]
            hit = is_hit(ranked, v["source_dataset_id"], k)
            record["arms"][a] = {
                "arm_BM25": scored["arms"][a]["arm_BM25"],
                "ranked_isolated": ranked,
                "hit": hit,
            }
            hits[a].append(hit)
            if v["query_class"] in by_class[a]:
                by_class[a][v["query_class"]].append(hit)
            by_facet[a].setdefault(v["facet"], []).append(hit)
            by_base[base_id][a].append(hit)

        per_variation.append(record)

    # One entry per base query, per arm: True iff every one of its variations
    # hit under that arm (the all-n-hit column, spec: labelled, not headline).
    all_n_hit: dict[str, float | None] = {
        a: _mean([1.0 if all(groups[a]) else 0.0 for groups in by_base.values()])
        for a in ARMS
    }

    return {
        "n_variations_scored": len(variations),
        "n_base_queries": len(by_base),
        "corpus_size": size,
        "controls": {"title_boost": 0, "operator": BM25_OPERATOR, "ranking": "isolated_bm25 (headline, Decision 3)"},
        "mean_hit_at_k": {a: _mean([1.0 if h else 0.0 for h in hits[a]]) for a in ARMS},
        "mean_hit_at_k_by_class": {
            a: {c: _mean([1.0 if h else 0.0 for h in by_class[a][c]]) for c in QUERY_CLASSES} for a in ARMS
        },
        "mean_hit_at_k_by_facet": {
            a: {f: _mean([1.0 if h else 0.0 for h in v]) for f, v in by_facet[a].items()} for a in ARMS
        },
        "all_n_hit": all_n_hit,
        "per_variation": per_variation,
    }


def _print_summary(report: dict) -> None:
    print(f"\nmean hit@k  (n_variations={report['n_variations_scored']}, "
          f"n_base_queries={report['n_base_queries']})")
    header = "arm".ljust(14) + "mean_hit".rjust(10) + "all_n_hit".rjust(11)
    print(header)
    for a in ARMS:
        mh = report["mean_hit_at_k"][a]
        anh = report["all_n_hit"][a]
        row = a.ljust(14)
        row += (f"{mh:.4f}" if mh is not None else "n/a").rjust(10)
        row += (f"{anh:.4f}" if anh is not None else "n/a").rjust(11)
        print(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variations", required=True)
    parser.add_argument("--out", default="eval/benchmark/variation_scoring.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--size", type=int, default=None,
                        help="Result-count cap per channel; default is the frame "
                             "file's corpus size (exhaustive). Only for testing "
                             "on a smaller slice.")
    parser.add_argument("--corpus-frame", default=str(DEFAULT_CORPUS_FRAME))
    args = parser.parse_args(argv)

    variations = json.loads(Path(args.variations).read_text(encoding="utf-8"))["variations"]

    os_client = get_client()
    result = run(os_client, variations, args.k, args.size, Path(args.corpus_frame))

    per_variation = result.pop("per_variation")

    report = {
        "code_version": code_version(),
        "index": AUCTUS_INDEX_NAME,
        "k": args.k,
        **result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    per_variation_out = out.with_name(out.stem + "_per_variation.json")
    per_variation_out.write_text(
        json.dumps({
            "code_version": report["code_version"],
            "index": AUCTUS_INDEX_NAME,
            "k": args.k,
            "controls": report["controls"],
            "records": per_variation,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8")

    _print_summary(report)
    print(f"\n-> {out}")
    print(f"-> {per_variation_out}  ({len(per_variation)} per-variation records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
