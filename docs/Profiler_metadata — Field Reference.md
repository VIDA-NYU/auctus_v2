# Profiler_metadata — Field Reference & Profile Trimming

`profiler_metadata` is the JSON output of atlas-profiler for one dataset. It
describes the dataset's structure (columns, types, size, spatial coverage).
During ingestion Auctus passes a **trimmed** subset of it to AutoDDG as grounding
context for description generation (`describe_dataset(..., use_profile=True)`).

AutoDDG only needs the fields that describe **what the data is about**, so the
trimming is an **allowlist**: a deterministic Python step (no LLM call) copies
only the fields listed below into the profile string sent to AutoDDG — everything
else is simply left out, never explicitly deleted. So "fields we drop" below means
"fields present in the raw metadata that the allowlist does not select." It runs
before the description is generated; implementation: `build_profile_text()` in
[`backend/storage/arq_worker.py`](../backend/storage/arq_worker.py).

> The per-column detail is **adaptive to table width** (see "Adaptive trimming
> rule" below); it is no longer a single fixed allowlist for every dataset.

## Fields we feed to AutoDDG

Dataset-level:

| Field | Why kept |
|---|---|
| `nb_rows` | Number of rows (dataset size). |
| `nb_columns` | Number of columns. |
| `types` | Dataset-level types, e.g. `["categorical", "spatial"]`. |
| `spatial_coverage.label`, `spatial_coverage.bbox` | Geographic extent as a human label + bounding box, e.g. label "New York City", bbox `[-74.23, 40.52, -73.71, 40.90]`. Read from the **top-level record** (a sibling of `profiler_metadata`); only `label` and `bbox` are kept. |

Per-column (`columns[]`) — the core signal. For each column:

| Field | Kept when | Why |
|---|---|---|
| `name` | always (core) | The column name (e.g. Facility Name). |
| `structural_type` | always (core) | Storage type (e.g. http://schema.org/Text, integer, date). |
| `semantic_types` | always (core) | Meaning, when detected (e.g. latitude, city name). |
| `num_distinct_values` | narrow tables only | Cardinality (categorical columns). |
| `mean`, `std`, `min`, `max` | narrow tables only | Numeric stats; present only for numerical columns. Help the description state real ranges instead of guessing. |

## Fields we drop

- `attribute_keywords` — column names split into word tokens ("Hours of Operation"
  → "Hours", "of", "Operation"). Redundant with `columns[].name` and noisy
  (duplicates, stopwords like "of"). **Exception — fallback:** when `columns[]` is
  empty (profiling edge case), the allowlist instead emits
  `column_names: attribute_keywords` so the model still gets the column names.
- `spatial_coverage.geohashes4` / `spatial_coverage.ranges` — hundreds of
  fine-grained geohash grid cells describing point distribution. Unreadable for a
  description; the bounding box already summarizes the same coverage.
- `nb_profiled_rows`, `nb_spatial_columns`, `nb_temporal_columns`,
  `nb_numerical_columns`, `nb_categorical_columns` — operational counts; add little
  to a textual description.
- `sample` — the raw CSV sample. Already passed to AutoDDG separately as
  `dataset_sample`, so including it here would be duplication.
- `_sample_telemetry` — download mechanics (temp file path, bytes loaded,
  truncation flags). Unrelated to data content.
- `_profiling_times` — how long each profiling step took. Purely operational.
- `column_indexes` (inside spatial coverage) — internal positional indexes.

## Adaptive trimming rule (by table width)

The per-column detail adapts to how wide the table is, because wide tables would
otherwise produce an enormous, noisy profile:

| Condition | Per-column fields emitted | Column cap |
|---|---|---|
| `nb_columns < WIDE_TABLE_COLUMN_THRESHOLD` (narrow) | core **+** numeric stats | `MAX_COLUMNS_IN_PROFILE` |
| `nb_columns >= WIDE_TABLE_COLUMN_THRESHOLD` (wide) | core only | `MAX_COLUMNS_IN_PROFILE` |

Current thresholds (module-level constants in `arq_worker.py`, tune there):

- `WIDE_TABLE_COLUMN_THRESHOLD = 40`
- `MAX_COLUMNS_IN_PROFILE = 80`

When the table has more columns than `MAX_COLUMNS_IN_PROFILE`, only the first
`MAX_COLUMNS_IN_PROFILE` are emitted and a `columns_truncated` marker
(`{shown, total}`) is added so the LLM knows the schema is partial and does not
over-claim coverage.

### Rationale for the thresholds

- **40 columns** is roughly where the per-column numeric stats start to dominate the
  prompt without proportionally improving the description. Above it, the column
  names + types + semantic types alone already convey the schema; exact numeric
  ranges per column add bulk with diminishing returns.
- **80 columns** caps truly wide tables (the AutoDDG benchmarks contain tables with
  hundreds of columns) so a single dataset can't crowd out the rest of the prompt.

These are starting points; adjust as we evaluate description quality on real data.

## Notes on variability

The top-level schema is consistent across datasets, but field **contents** vary:

- `columns[]` differs per dataset; numeric columns add mean/std/min/max,
  categorical columns add num_distinct_values, spatial columns add coverage.
- Datasets with no spatial columns may have empty/absent
  `spatial_coverage` / `spatial_bbox`.
- If profiling hits an edge case, `columns` can be empty and an error field
  appears; in that case we fall back to the column names in `attribute_keywords`.
