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
| `num_distinct_values` | always (core) | Cardinality (categorical columns). Capped at `nb_rows` (atlas-profiler's estimate can exceed it). |
| `min`, `max` | always (core) | Numeric range. Without it the model reads ranges off the few sample rows. |
| `mean`, `stddev` | narrow tables only | Numeric moments; present only for numerical columns. |

Dataset-level, also sent: `temporal_coverage` (`start`, `end`), the file's date range from the record.

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

## Adaptive detail rule (by table width)

**Every column is sent**, for every table. The detail per column adapts to width:

| Condition | Per-column fields emitted | Grouping |
|---|---|---|
| `nb_columns < WIDE_TABLE_COLUMN_THRESHOLD` (narrow) | core **+** `mean`, `stddev` | none |
| `nb_columns >= WIDE_TABLE_COLUMN_THRESHOLD` (wide) | core (ranges and distinct counts included) | columns whose names differ only by numbers and share a type are one entry |

`WIDE_TABLE_COLUMN_THRESHOLD = 40` (module-level constant in `arq_worker.py`).

A group entry replaces its first member: `{"columns": [...], "count": n, "structural_type", "semantic_types" (union), "min", "max" (over the group), "num_distinct_values" (smallest and largest)}`. So `csa1_v2 ... csa14_v3` reads as one 44-item battery instead of 44 lines.

The semantic profile (AutoDDG's per-column meaning pass) also covers every column.

### Why every column (2026-11-27)

Wide tables used to stop at 80 columns (`MAX_COLUMNS_IN_PROFILE`, with a `columns_truncated` marker), and the semantic pass too. On the 38 UltraViolet tables with 40+ columns:

- the cut hid 52% of columns on average; a 230-column survey's description never mentioned its subject (child sexual abuse prevention), whose items start after column 80;
- with every column (and the groups): curator facts covered +12 points per record (8 of 10 records up, none down), unsupported claims 9.8% -> 8.2%;
- cost of the profile: same latency (~7.5 s per description), ~1k more input tokens per table;
- cost of the semantic pass on every column: 82-172 s and 23k-47k output tokens for 186-251 columns, against ~55 s and ~15k for the first 80. Wide tables only (38 of 606 UltraViolet datasets).

Coverage over speed for now: discovery needs what the data contains. Revisit with measurements if ingest time becomes the constraint.

## Notes on variability

The top-level schema is consistent across datasets, but field **contents** vary:

- `columns[]` differs per dataset; numeric columns add mean/std/min/max,
  categorical columns add num_distinct_values, spatial columns add coverage.
- Datasets with no spatial columns may have empty/absent
  `spatial_coverage` / `spatial_bbox`.
- If profiling hits an edge case, `columns` can be empty and an error field
  appears; in that case we fall back to the column names in `attribute_keywords`.
