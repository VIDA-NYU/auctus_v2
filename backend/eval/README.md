# Retrieval evaluation (NDCG / MRR / Hit @k)

> **Branch scope.** This directory only exists on `autoddg-eval`, a test-only
> branch built on top of `integrate-autoddg` (the production line that merges
> into `main`). `autoddg-eval` is never merged; it's kept up to date by
> periodically merging `integrate-autoddg` into it. The production branch has
> no `llm_direct` arm and no `backend/eval/` — those are eval-only.

`retrieval_eval.py` measures how much AutoDDG descriptions improve dataset
retrieval, by querying the live Auctus `/search` endpoint against each
description field and scoring the rankings.

It compares four description sources (the `description_source` parameter wired
into `/search`):

| source | field queried | role |
|---|---|---|
| `original` | original portal `description` | the **before** baseline |
| `llm_direct` | `llm_direct_description` | plain LLM from the sample only, no AutoDDG grounding (the paper's `LLM-GPT` baseline) |
| `ufd` | `autoddg_description` | AutoDDG User-Focused Description |
| `sfd` | `autoddg_search_description` | AutoDDG Search-Focused Description |

This is the extrinsic retrieval evaluation from the AutoDDG paper (§3.1, §4.2).
Expected direction: `sfd >= ufd >= llm_direct ~ original`.

Key properties of the harness:

- **Paper-comparable BM25 by default.** The eval sends `match_operator: "or"`
  (plain BM25). The backend's production default is `"and"` (every term must
  match within one field), which collapses verbose queries and is not what the
  paper measured.
- **Title control.** `--title-boost 0` removes `title` from the match entirely,
  eliminating the title-echo confound (a title alone satisfying the query makes
  all arms look identical). Default is the production `title^2`.
- **Preflight.** Before scoring, the script checks against OpenSearch that each
  requested arm's field is actually populated and that every qrels dataset id
  exists in the index — otherwise empty fields or typo'd ids masquerade as
  plausible low scores. `--skip-preflight` bypasses (not recommended).
- **Same denominator per arm.** If any arm fails for a query (after one retry),
  the query is dropped for *all* arms, so per-arm averages stay comparable.
  Failed/skipped queries are reported explicitly.
- **Results file.** Every run writes JSON (run metadata incl. git sha and
  preflight coverage, per-query rankings and scores, aggregates) to
  `eval/results/run-<timestamp>.json` or `--out`. The printed table is a view;
  the JSON is the artifact for error analysis and significance tests.
- **Metrics.** `--metrics ndcg mrr hit` (default `ndcg`). MRR/Hit treat any
  grade > 0 as relevant — use them for known-item queries (single id, grade 1).

## Prerequisites

1. Backend running and reachable (default `http://localhost:8000`), built from a
   commit whose `/search` supports `size` / `match_operator` / `title_boost`
   (the script detects and refuses a backend that silently caps at 10 results).
2. OpenSearch reachable for the preflight (default `http://localhost:9200`).
3. Benchmark datasets ingested **with the AutoDDG arm fields populated** —
   either ingested on this branch, or refreshed with the backfill script (below).
   On an index created before this branch, restart the backend once so
   `init_db` adds the arm-field mappings **before** any document containing
   them is indexed; if the fields were already dynamic-mapped, recreate the
   index (analyzer mismatch skews cross-arm BM25).
4. A queries + relevance-judgments (qrels) file — see `sample_queries.json`.

## Run

```bash
cd backend
python -m eval.retrieval_eval --queries eval/sample_queries.json
# options:
#   --sources original ufd sfd      which arms to compare
#   --metrics ndcg mrr hit          metrics to compute (default ndcg)
#   --ks 5 10 15 20                 cut-offs
#   --size 20                       top-N requested and scored per query
#   --operator or|and               term matching (default: or = paper BM25)
#   --title-boost 0                 remove the title from the match
#   --endpoint http://localhost:8000/search
#   --opensearch http://localhost:9200   preflight target
#   --out eval/results/myrun.json   results artifact path
#   --skip-preflight                bypass coverage/qrels checks
```

## Backfilling descriptions without re-ingesting

`backfill_descriptions.py` regenerates the AutoDDG arms for already-ingested
datasets from the full profiles stored in MinIO (which include the CSV sample),
then updates only the description fields on the OpenSearch documents. Use it
after any prompt/arm change instead of a full re-ingest (no download, no
re-profiling). It must run where autoddg/portkey/minio are installed — i.e. the
worker container:

```bash
docker compose exec arq-worker python -m eval.backfill_descriptions --all
docker compose exec arq-worker python -m eval.backfill_descriptions --ids <id1> <id2>
docker compose exec arq-worker python -m eval.backfill_descriptions --all --dry-run
```

## Notes

- The NDCG implementation mirrors `autoddg.ranking.metrics.compute_ndcg`; the
  script reuses that module when `autoddg` is importable and otherwise falls back
  to an identical local copy.
- The eval payload is specific to the pure-BM25 `/search` endpoint (`query`
  field); don't point `--endpoint` at `/api/v1/search`, which expects a
  different schema (`keywords`) and mixes in kNN over a vector embedded from
  the *original* description — that would confound the comparison.
- The LLM-direct arm only exists on this branch, gated by `AUTODDG_EVAL_ARMS`
  (default on; set to `0` to skip it while still testing on this branch).
  `integrate-autoddg` doesn't carry this arm or the gate at all.
