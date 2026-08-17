# Profiler_metadata — Field Reference & Profile Trimming

`profiler_metadata` is the JSON output of atlas-profiler for one dataset. It
describes the dataset's structure (columns, types, size, spatial and temporal
coverage).
During ingestion Auctus passes a **trimmed** subset of it to AutoDDG as grounding
context for description generation (`describe_dataset(..., use_profile=True)`).

AutoDDG only needs the fields that describe **what the data is about**, so the
trimming is an **allowlist**: a deterministic Python step (no LLM call) copies
only the fields listed below into the profile string sent to AutoDDG — everything
else is simply left out, never explicitly deleted. So "fields we drop" below means
"fields present in the raw metadata that the allowlist does not select." It runs
before the description is generated; implementation: `build_profile_text()` in
[`backend/storage/arq_worker.py`](../backend/storage/arq_worker.py).

> The per-column field set is the same for every dataset regardless of table
> width (see "Column cap" below for the one remaining width-independent limit —
> how many columns are emitted at all).

## Fields we feed to AutoDDG

Dataset-level:

| Field | Why kept |
|---|---|
| `nb_rows` | Number of rows (dataset size). |
| `nb_columns` | Number of columns. |
| `types` | Dataset-level types, e.g. `["categorical", "spatial"]`. |
| `spatial_coverage.bbox` | Geographic extent as a bounding box, emitted as `{"type": "envelope", "coordinates": [[min_lon, max_lat], [max_lon, min_lat]]}` — a GeoJSON envelope object, not a flat four-number list. Read from the **top-level record** (a sibling of `profiler_metadata`). **Conditional, see below.** |
| `temporal_coverage` | The dataset's date range, flattened to `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`. **Conditional, see below.** |

### Coverage fields are conditional

Both coverage fields are emitted **only when the profiler derived them from this
dataset's own data**. An absent field therefore means "this dataset has no
geography / no dates", and is never a profiling failure.

- **Spatial** is gated on `profiler_metadata.spatial_coverage`, which exists only
  when the profiler found real coordinates. The gate is necessary because the
  top-level `spatial_coverage` block is written for *every* dataset:
  `transformer._safe_bbox_from_profiler_or_sample` falls back to a portal-level
  bbox when the data has none, so without the gate a purely categorical dataset
  would be told it covers New York City. That fallback is deliberate — the search
  API's geo filter (`api/search.py`) relies on it — so it stays in the record and
  in the index, and is excluded only here.
- **`spatial_coverage.label` is not kept.** It is portal-level, so it is identical
  for every dataset on the portal and says nothing about any one of them. As
  currently configured it renders as the bare domain string
  `data.cityofnewyork.us`.
- **Temporal** comes from `profiler_metadata.temporal_coverage` — the profiler's
  ranges over values in columns it typed as `DateTime`. It is **not** taken from
  the record's top-level `temporal_coverage`, which `transformer.infer_temporal_range`
  derives from a column-**name** regex over the sampled rows only, and which is
  therefore both wrongly selected and truncated to the sample. Where several
  datetime columns disagree, the profile reports the outer envelope of their
  ranges as a dataset-level summary, not a per-column statement.

Per-column (`columns[]`) — the core signal. For each column:

| Field | Kept when | Why |
|---|---|---|
| `name` | always | The column name (e.g. Facility Name). |
| `structural_type` | always | Storage type (e.g. http://schema.org/Text, integer, date). |
| `semantic_types` | always | Meaning, when detected (e.g. latitude, city name). |
| `num_distinct_values` | always, when the profiler computed it | Cardinality (categorical columns). |
| `mean`, `stddev` | always, when the profiler computed it | Numeric stats for numerical columns. |
| `coverage` | always, when the profiler computed it | The profiler's own numeric range(s) for the column — up to three disjoint `{"range": {"gte": ..., "lte": ...}}` entries, not flattened to a single min/max (a single span would cover gaps the data doesn't have). Grounds a value-span query ("salaries between $30k-$80k") in a range the data actually contains. |
| `min`, `max` | in the allowlist, but **never occurs in a stored profile** | Only computed on a profiling route the crawler does not use. Kept in the allowlist for forward-compatibility; do not rely on it being present. |

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
  to a textual description. All four `nb_*_columns` are the **profiler's own**
  values (`profiler/core.py` derives them from `determine_dataset_type`) and are
  carried through the crawler untouched. The profiler omits a key entirely when
  its count is zero, so an absent key means zero.
- `sample` — the raw CSV sample. Already passed to AutoDDG separately as
  `dataset_sample`, so including it here would be duplication.
- `_sample_telemetry` — download mechanics (temp file path, bytes loaded,
  truncation flags). Unrelated to data content.
- `_profiling_times` — how long each profiling step took. Purely operational.
- `column_indexes` (inside spatial coverage) — internal positional indexes.

## Column cap

Every column gets the same field set regardless of table width — there was
previously a width-based rule that dropped numeric stats (including `coverage`)
on tables at or above 40 columns; it was removed 2026-08-16
(`profile-enrichment-coverage-distinct`, design.md D3) because it made §1b's
value-span query sub-type structurally ungroundable on ~29% of the corpus. The
trade-off it existed to manage — wide tables producing a longer profile — is now
accepted rather than designed around; see that change's Risks section for the
current, unresolved status of that trade-off.

One cap remains, independent of width:

- `MAX_COLUMNS_IN_PROFILE = 80` (module-level constant in `arq_worker.py`) —
  when a table has more columns than this, only the first `MAX_COLUMNS_IN_PROFILE`
  are emitted and a `columns_truncated` marker (`{shown, total}`) is added so the
  LLM knows the schema is partial and does not over-claim coverage.

Measured rendered `profile_only` length on two real datasets (acceptance check,
2026-08-16): 2,478 characters for a 14-column table (2 columns with `coverage`),
7,130 characters for a 44-column table (6 columns with `coverage`). Both are
comfortably within normal prompt budgets; neither is close to
`MAX_COLUMNS_IN_PROFILE`. A table with many more numeric-measurement columns than
either of these has not yet been observed.

## Notes on variability

The top-level schema is consistent across datasets, but field **contents** vary:

- `columns[]` differs per dataset; numeric columns add `mean`/`stddev`/`coverage`,
  categorical columns add `num_distinct_values`.
- Datasets with no spatial columns may have empty/absent
  `spatial_coverage` / `spatial_bbox`.
- If profiling hits an edge case, `columns` can be empty and an error field
  appears; in that case we fall back to the column names in `attribute_keywords`.
