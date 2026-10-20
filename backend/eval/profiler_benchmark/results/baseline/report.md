# Profiler benchmark report

Run: `{"harvest": ["../../data/ultraviolet"], "per_instance": 0, "seed": 0, "stage": "all", "commit": "e4ec05c", "atlas_profiler": "0.0.2b1", "pandas": "3.0.6", "at": "2026-10-19T15:43:18+00:00", "profiler_code": "stock atlas-profiler (upstream main b449447 = PyPI 0.0.2b1 code)", "gold": "2026-10-20: scanned docs (28 records with dictionaries), value labels -> categorical"}`

## 1. Robustness

Usable profiles: **66/66** (1.0). Median 3.97 s per file; max peak memory 1621.5 MB.

| instance/format | files | usable | errors | ≤1 column | profiler time cov. | pipeline time cov. | profiler spatial | pipeline bbox |
|---|---|---|---|---|---|---|---|---|
| ultraviolet/csv | 59 | 59 | 0 | 0 | 18 | 18 | 1 | 10 |
| ultraviolet/xlsx | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |


## 2. Type correctness

Synthetic (known answers): **120/182** (0.6593).

| synthetic file | accuracy |
|---|---|
| types.csv | 0.7692 |
| types.tsv | 0.7692 |
| types.xlsx | 0.7692 |
| types_bom.csv | 0.7692 |
| types_cp1252.csv | 0.7692 |
| types_preamble.csv | 0.0 |
| types_semicolon.csv | 0.7692 |

Wrong on the baseline file:
- bool_01 (got place, want boolean|categorical|numeric)
- currency (got text, want numeric)
- percent (got text, want numeric)
- record_code (got text, want identifier)
- thousands_sep (got text, want numeric)
- uuid (got text, want identifier)

README dictionaries: 498 of 3668 columns scorable; accuracy **0.504** (strict, README type only: 0.2711; gold verified: False). Most common errors: categorical->numeric (168), categorical->place (41), categorical->boolean (9), date->numeric (4), text->categorical (4), boolean->categorical (4), numeric->place (3), date->text (3), numeric->identifier (2), date->place (2)

## 3. Sampling

Nothing to compare: robustness profiles whole files, so no file was truncated. Use `run.py sampling-sweep` to measure how much a row sample would lose.

## 4. Performance (whole table, one run at a time)

| rows | columns | status | seconds | peak MB |
|---|---|---|---|---|
| 1000 | 10 | ok | 2.85 | 1482.5 |
| 1000 | 100 | ok | 5.9 | 1543.0 |
| 1000 | 1000 | ok | 13.09 | 1621.2 |
| 10000 | 10 | ok | 6.79 | 1509.6 |
| 10000 | 100 | ok | 14.26 | 1676.1 |
| 100000 | 10 | ok | 23.22 | 1650.1 |
| 100000 | 100 | ok | 57.02 | 1990.8 |
