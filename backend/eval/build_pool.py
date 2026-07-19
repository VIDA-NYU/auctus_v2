"""Build the judgment pool for each query (BM25 ∪ arm-neutral dense).

Pooling decides WHICH (query, dataset) pairs get judged; everything outside the
pool is non-relevant by convention. To avoid ACORDAR-2.0 lexical bias (a pool
from one lexical engine underrates semantic retrievers), the pool is the UNION of
two deliberately-different retrievers:

  * lexical  — BM25 over metadata (title + original description) via OpenSearch.
  * dense    — cosine over an ARM-NEUTRAL representation: the embedding of
    ``title + profile_only`` (the algorithmic profile facts, i.e. the same neutral
    grounding the query generator and judge use). NOT the original-description
    ``dataset_vector`` (that tilts the pool toward the original arm — report F6).

The portal's own ranker is never a pool contributor (circularity red line). The
output records, per query, the pooled ids with their contributing retriever(s)
and the pool depth, so pool-depth sensitivity can be computed downstream.

    python -m eval.build_pool --queries eval/benchmark/queries.json \
        --k 20 --out eval/benchmark/pool.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

LOGGER = logging.getLogger("build_pool")

# Arm-neutral dense representation: title + the algorithmic profile facts.
NEUTRAL_TEXT_FIELDS = ("title", "profile_only_description")
# Lexical metadata fields for BM25 pooling.
BM25_FIELDS = ("title", "description")


def _load_queries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data["queries"] if isinstance(data, dict) else data
    return [{"query_id": q["query_id"], "text": q["text"]} for q in queries]


def bm25_topk(os_client, text: str, k: int) -> list[str]:
    resp = os_client.search(
        index=AUCTUS_INDEX_NAME,
        body={
            "query": {"multi_match": {"query": text, "fields": list(BM25_FIELDS),
                                      "operator": "or"}},
            "_source": False,
        },
        size=k,
    )
    return [h["_id"] for h in resp["hits"]["hits"]]


def _neutral_corpus(os_client) -> tuple[list[str], list[str]]:
    """Return (ids, neutral_texts) for every doc, for arm-neutral dense pooling."""
    resp = os_client.search(
        index=AUCTUS_INDEX_NAME,
        body={"query": {"match_all": {}}, "_source": list(NEUTRAL_TEXT_FIELDS)},
        size=10000,
    )
    ids, texts = [], []
    for h in resp["hits"]["hits"]:
        s = h.get("_source") or {}
        ids.append(h["_id"])
        texts.append("\n".join(str(s.get(f, "")) for f in NEUTRAL_TEXT_FIELDS).strip())
    return ids, texts


def dense_retriever(os_client):
    """Build an in-memory arm-neutral dense index; returns a topk(text, k) fn."""
    from storage.arq_worker import get_embedding_model
    import numpy as np

    model = get_embedding_model()
    if model is None:
        raise RuntimeError("sentence-transformers unavailable; cannot pool dense arm")
    ids, texts = _neutral_corpus(os_client)
    mat = model.encode(texts, normalize_embeddings=True)
    mat = np.asarray(mat, dtype="float32")

    def topk(text: str, k: int) -> list[str]:
        q = np.asarray(model.encode([text], normalize_embeddings=True), dtype="float32")[0]
        scores = mat @ q  # cosine (both normalized)
        order = scores.argsort()[::-1][:k]
        return [ids[i] for i in order]

    return topk


def build_pool(os_client, queries: list[dict], k: int) -> dict:
    dense_topk = dense_retriever(os_client)
    per_query = []
    for q in queries:
        bm25 = bm25_topk(os_client, q["text"], k)
        dense = dense_topk(q["text"], k)
        bm25_rank = {i: r for r, i in enumerate(bm25)}
        dense_rank = {i: r for r, i in enumerate(dense)}
        pooled = {}
        for i in dict.fromkeys(bm25 + dense):  # preserve first-seen order, dedup
            contributors = []
            if i in bm25_rank:
                contributors.append("bm25")
            if i in dense_rank:
                contributors.append("dense")
            pooled[i] = {
                "contributors": contributors,
                "bm25_rank": bm25_rank.get(i),
                "dense_rank": dense_rank.get(i),
            }
        per_query.append({
            "query_id": q["query_id"],
            "text": q["text"],
            "pool_depth": len(pooled),
            "dense_only": [i for i, m in pooled.items() if m["contributors"] == ["dense"]],
            "pool": pooled,
        })
    return {
        "index": AUCTUS_INDEX_NAME,
        "k_per_retriever": k,
        "bm25_fields": list(BM25_FIELDS),
        "dense_neutral_fields": list(NEUTRAL_TEXT_FIELDS),
        "queries": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--k", type=int, default=20, help="top-k per retriever")
    parser.add_argument("--out", default="eval/benchmark/pool.json")
    args = parser.parse_args(argv)

    queries = _load_queries(Path(args.queries))
    manifest = build_pool(get_client(), queries, args.k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    depths = [q["pool_depth"] for q in manifest["queries"]]
    dense_only = sum(len(q["dense_only"]) for q in manifest["queries"])
    print(f"Pooled {len(queries)} queries at k={args.k}: "
          f"pool depth min/max {min(depths)}/{max(depths)}, "
          f"{dense_only} dense-only hit(s) BM25 missed (F6 signal).")
    print(f"Pool -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
