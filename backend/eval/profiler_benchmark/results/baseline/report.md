# Profiler benchmark report

Run: `{"harvest": ["../../data/ultraviolet"], "per_instance": 0, "seed": 0, "stage": "all", "commit": "59f11c3", "atlas_profiler": "0.0.2b1", "pandas": "3.0.6", "at": "2026-10-19T03:12:11+00:00"}`

## 1. Robustness

Usable profiles: **66/66** (1.0). Median 4.34 s per file; max peak memory 1641.4 MB.

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

README dictionaries: 128 of 191 columns scorable; accuracy **0.8594** (strict, README type only: 0.5547; gold verified: False). Most common errors: categorical->numeric (12), categorical->place (3), date->numeric (2), boolean->place (1)

## 3. Sampling (first 500 rows / 2 MB vs whole file)

0 files compared. Column types change on 0 files (rate None of columns). Temporal span recall median None over 0 files; 0 files' samples cover less than half of the real time span.

## 4. Performance (whole table, one run at a time)

| rows | columns | status | seconds | peak MB |
|---|---|---|---|---|
| 1000 | 10 | ok | 3.42 | 1478.9 |
| 1000 | 100 | ok | 7.04 | 1533.3 |
| 1000 | 1000 | ok | 14.85 | 1613.5 |
| 10000 | 10 | ok | 8.23 | 1508.3 |
| 10000 | 100 | ok | 16.68 | 1659.0 |
| 100000 | 10 | ok | 27.69 | 1649.3 |
| 100000 | 100 | ok | 67.22 | 1976.7 |
