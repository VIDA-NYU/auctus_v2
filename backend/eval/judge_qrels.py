"""LLM judge -> provisional BINARY qrels, under the anti-leakage code invariant.

For every pooled (query, dataset) pair the judge assigns a binary relevance
label 0/1 (aligned with the AutoDDG paper's evaluation protocol). The judge sees
ONLY the neutral bundle (title + profile + data sample) — the description arm
under test is PHYSICALLY ABSENT from the prompt (report F1b). If the judge saw
the arm, NDCG would systematically favour the description arms and the whole
evaluation would be void. The same ``assert_no_arm_leak`` invariant the query
generator uses is asserted on every judged dataset here.

Label: 1 = relevant (the dataset answers the query, fully or partially);
       0 = not relevant.

Datasets outside the pool are non-relevant by convention (not judged, grade 0).
These qrels are PROVISIONAL (LLM-only, un-calibrated on NYC) — a pipeline
shakedown, not benchmark ground truth.

    python -m eval.judge_qrels --pool eval/benchmark/pool.json \
        --out eval/benchmark/qrels.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client
from eval.backfill_descriptions import load_full_profile
from eval.generate_queries import build_neutral_bundle, assert_no_arm_leak
from eval.llm_client import complete, get_llm_client
from storage.minio_client import get_storage_client

LOGGER = logging.getLogger("judge_qrels")

JUDGE_PROMPT = """You are a relevance assessor for a dataset-search benchmark. \
Given a user QUERY and a numbered list of candidate datasets (described ONLY by \
neutral facts: title, an algorithmic column/coverage profile, and a data sample), \
label whether each dataset is relevant to the query.

Labels:
- 1 = relevant: the dataset answers the query, fully or partially (topically \
related, or its columns/coverage contain what the query asks for).
- 0 = not relevant.

Judge on the data itself (columns, coverage, sample) — a dataset whose columns \
contain what the query asks for is relevant even if the wording differs.

Return STRICT JSON mapping each dataset number to its label, e.g. \
{{"1": 1, "2": 0, "3": 1}}. No prose outside the JSON.

QUERY: {query}

DATASETS:
{datasets}
"""


def _render_bundle(idx: int, bundle: dict) -> str:
    return (f"[{idx}] Title: {bundle['title']}\n"
            f"    Profile: {bundle['profile']}\n"
            f"    Sample: {bundle['sample'][:600]}")


def judge_query(client, query_text: str, items: list[tuple[str, dict, str | None]]) -> dict:
    """Grade every (query, dataset) in ``items``. Returns {dataset_id: grade}."""
    bundles = []
    for _id, doc, sample in items:
        bundle = build_neutral_bundle(doc, sample)
        assert_no_arm_leak(bundle, doc)  # F1b invariant: no arm prose in the prompt
        bundles.append(bundle)
    rendered = "\n\n".join(_render_bundle(i + 1, b) for i, b in enumerate(bundles))
    prompt = JUDGE_PROMPT.format(query=query_text, datasets=rendered)

    raw = complete(client, prompt, temperature=0.0).strip()
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw).strip()
    grades = json.loads(raw)

    out: dict[str, int] = {}
    for i, (_id, _doc, _s) in enumerate(items, 1):
        g = grades.get(str(i), grades.get(i, 0))
        try:
            g = int(g)
        except (TypeError, ValueError):
            g = 0
        out[_id] = max(0, min(1, g))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="eval/benchmark/pool.json")
    parser.add_argument("--out", default="eval/benchmark/qrels.json")
    args = parser.parse_args(argv)

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    os_client = get_client()
    client = get_llm_client()
    if client is None:
        raise SystemExit("No LLM client (PORTKEY_API_KEY set? on NYU VPN?).")
    try:
        storage_client = get_storage_client()
    except Exception:
        storage_client = None

    # Cache doc + sample per dataset id (a dataset appears in many pools).
    doc_cache: dict[str, dict] = {}
    sample_cache: dict[str, str | None] = {}

    def load(dataset_id: str):
        if dataset_id not in doc_cache:
            doc_cache[dataset_id] = os_client.get(
                index=AUCTUS_INDEX_NAME, id=dataset_id,
                _source=["title", "profiler_metadata", "spatial_coverage"],
            ).get("_source") or {}
            rec = load_full_profile(storage_client, dataset_id) if storage_client else None
            sample_cache[dataset_id] = rec.get("sample") if isinstance(rec, dict) else None
        return doc_cache[dataset_id], sample_cache[dataset_id]

    out_queries = []
    total_pairs = 0
    for i, q in enumerate(pool["queries"], 1):
        items = []
        for dataset_id in q["pool"]:
            doc, sample = load(dataset_id)
            items.append((dataset_id, doc, sample))
        try:
            grades = judge_query(client, q["text"], items)
        except Exception as exc:
            LOGGER.warning("Judge failed for %s: %s", q["query_id"], exc)
            grades = {}
        # Keep only positives in the qrels file (retrieval_eval treats missing as 0).
        relevant = {k: v for k, v in grades.items() if v > 0}
        total_pairs += len(items)
        out_queries.append({
            "query_id": q["query_id"],
            "text": q["text"],
            "relevant": relevant,
        })
        print(f"  [{i}/{len(pool['queries'])}] {q['query_id']}: "
              f"{len(relevant)}/{len(items)} relevant")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_provenance": "PROVISIONAL LLM-judge BINARY qrels (un-calibrated on NYC); "
                       "anti-leakage guard: judge saw title+profile+sample only, "
                       "never any description arm.",
        "queries": out_queries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJudged {total_pairs} pooled pairs across {len(out_queries)} queries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
