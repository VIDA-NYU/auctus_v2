# Retrieval evaluation (NDCG@k)

`retrieval_eval.py` measures how much AutoDDG descriptions improve dataset
retrieval, by querying the live Auctus `/search` endpoint against each
description field and scoring the rankings with **NDCG@k**.

It compares four description sources (the `description_source` parameter wired
into `/search`):

| source | field queried | role |
|---|---|---|
| `original` | original portal `description` | the **before** baseline |
| `llm_direct` | `llm_direct_description` | plain LLM from the sample only, no AutoDDG grounding (the paper's `LLM-GPT` baseline) |
| `ufd` | `autoddg_description` | AutoDDG User-Focused Description |
| `sfd` | `autoddg_search_description` | AutoDDG Search-Focused Description |

This is the extrinsic retrieval evaluation from the AutoDDG paper (§3.1, §4.2),
with the three arms from the eval plan (Original / LLM-direct / AutoDDG).
OpenSearch's default relevance is BM25, matching the paper's BM25 setup. Expected
direction: `sfd >= ufd >= llm_direct ~ original`.

## Prerequisites

1. Backend running and reachable (default `http://localhost:8000`).
2. Benchmark datasets **ingested**, and **re-ingested after the Phase 1-3 changes**
   so the `autoddg_description` / `autoddg_search_description` fields are populated.
   If those fields are empty, `ufd`/`sfd` will score ~0.
3. A queries + relevance-judgments (qrels) file — see `sample_queries.json`.
   Replace the placeholder dataset ids with ids that exist in your index (the
   ECIR-DDG / NTCIR-DDG ids once Phase 5 ingests the benchmarks).

## Run

```bash
cd backend
python -m eval.retrieval_eval --queries eval/sample_queries.json
# options:
#   --sources original ufd sfd      which fields to compare
#   --ks 5 10 15 20                 NDCG cut-offs
#   --endpoint http://localhost:8000/search
#   --size 20                       top-N results scored per query
```

Output is an averaged NDCG@k table, one row per source.

## Notes

- The NDCG implementation mirrors `autoddg.ranking.metrics.compute_ndcg`; the
  script reuses that module when `autoddg` is importable and otherwise falls back
  to an identical local copy, so it runs even where `autoddg` isn't installed.
- Use the pure-BM25 `/search` endpoint (the default) for a clean description-only
  comparison. `/api/v1/search` mixes in kNN over a vector that was embedded from
  the *original* description, which would confound the comparison.
