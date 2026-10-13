# Profiler and ingestion reliability benchmark

Does Auctus's ingestion path (InvenioRDM connector + atlas-profiler) produce correct, complete profiles for real repository files? And does a change to the connector make that better or worse?

Every check goes through the connector's own code (`crawlers.invenio.transformer`: `sample_local_file`, `profile_sample`, `assemble_record`). So the benchmark measures exactly what ingestion runs, and it can be re-run after any connector change to catch regressions.

## Data

- **Real files:** the tabular files downloaded by `invenio_fetching harvest` (any instance). The reference runs use NYU UltraViolet (66 files) and CaltechDATA (1,741 files from its 310 tabular records).
- **Synthetic tables** (`synthetic.py`): 26 columns with known types (numbers, formatted numbers, dates in four formats, lat/lon, places, booleans, categories, free text, identifiers), written in 7 file variants: comma, semicolon, tab, Windows-1252, UTF-8 BOM, XLSX, and title rows above the header.
- **README data dictionaries:** column types from UltraViolet's curator-written READMEs. They're extracted by `description_eval` (LLM-assisted) and must be human-verified before the numbers are reported (`verified` flag).

## Checks

| Check | Question | Metric |
|---|---|---|
| 1. Robustness | Does every file produce a usable profile (no error, more than one column)? | usable rate per instance × format; failures listed |
| 2. Type correctness | Are column types right? | accuracy on synthetic columns (exact answers) and on README-typed columns (acceptable-set scoring, strict score alongside) |
| 3. Sampling | Ingestion profiles only the first 500 rows / 2 MB. What does that lose? | share of columns whose type changes vs the whole file; temporal span recall |
| 4. Performance | Time and memory vs table shape | seconds and peak RSS on a rows × columns grid |

Robustness also reports time and space coverage two ways: what the profiler found, and what the pipeline put in the record. The gap between the two exposed the coverage bugs fixed in the connector.

Each input runs in its own process (`profile_one.py`), so a crash, an OOM kill or a hang is recorded as a status instead of stopping the run, and peak memory is measured per file.

## Running

Needs a backend checkout and its requirements, not the Docker stack. Run from `backend/`:

```bash
pip install -r eval/profiler_benchmark/requirements.txt
python -c "from datamart_geo import GeoData; GeoData.download(update=False)"   # once, ~1.5 GB

python -m eval.profiler_benchmark.run all \
  --harvest ../../data/ultraviolet ../../data/caltech --jobs 8 \
  --out eval/profiler_benchmark/results/<run-name>
```

`--per-instance N` samples N random files per instance (seeded) for a quick run. A full run of about 1,800 files takes about 30 min with 8 jobs on a 10-core laptop; each job peaks around 1.5 GB.

## Checking a connector change

```bash
python -m eval.profiler_benchmark.run all --harvest ... --out eval/profiler_benchmark/results/after-my-change
python -m eval.profiler_benchmark.run compare eval/profiler_benchmark/results/baseline eval/profiler_benchmark/results/after-my-change
```

`compare` exits with status 1 if:
- the usable rate drops by more than 0.5 points;
- any synthetic column that was correct becomes wrong;
- README type accuracy drops by more than 2 points;
- any file that profiled usably no longer does.

Refresh `results/baseline` when a change is accepted. For quick checks, the offline unit tests run in about 30 s: `python -m unittest tests.test_invenio_connector`.

## Results

- `results/2026-10-13-before-coverage-fixes/`: connector before the coverage fixes and the empty-first-sheet fix.
- `results/baseline/`: current reference.

Each has `report.md` (human-readable), `summary.json` (metrics + run provenance: commit, atlas-profiler and pandas versions) and per-item `*.jsonl`.
