# Profiler benchmark report

Run: `{"harvest": ["../../data/ultraviolet", "../../data/caltech"], "per_instance": 0, "seed": 0, "stage": "all", "commit": "a54189f", "atlas_profiler": "0.0.2b1", "pandas": "3.0.6", "at": "2026-10-13T05:26:34+00:00"}`

## 1. Robustness

Usable profiles: **1713/1807** (0.948). Median 3.8 s per file; max peak memory 1510.1 MB.

| instance/format | files | usable | errors | ≤1 column | profiler time cov. | pipeline time cov. | profiler spatial | pipeline bbox |
|---|---|---|---|---|---|---|---|---|
| caltech/csv | 1343 | 1271 | 0 | 72 | 129 | 339 | 25 | 27 |
| caltech/tsv | 7 | 7 | 0 | 0 | 1 | 1 | 0 | 0 |
| caltech/xls | 16 | 14 | 2 | 0 | 0 | 0 | 0 | 0 |
| caltech/xlsx | 375 | 355 | 6 | 14 | 11 | 44 | 5 | 5 |
| ultraviolet/csv | 59 | 59 | 0 | 0 | 17 | 3 | 1 | 1 |
| ultraviolet/xlsx | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |

- `Chapter3_SupplementalTableS2.xlsx`: ok 
- `RawDataForFigures_v3.xlsx`: profiler_error profiler_error: No columns to parse from file
- `Alabama Education Data, 1885-1910, in Excel form.xlsx`: profiler_error profiler_error: No columns to parse from file
- `250730_ClCO3_RawDataForFigures.xlsx`: profiler_error profiler_error: No columns to parse from file
- `Clearance of Extracellular Human Amyloid-β Aggregates by Nutraceutical and Pharmaceutical Interventions.xlsx`: profiler_error profiler_error: No columns to parse from file
- `Data_Figure6_20250521.xlsx`: ok 
- `Data_Figure7_20250521.xlsx`: ok 
- `Table_Figure6_20240312.xlsx`: ok 
- `Table_Figure7_20240312.xlsx`: ok 
- `Dry Proteome Dataset.xlsx`: ok 
- `Amino Acid Dataset.xlsx`: ok 
- `teleseismic_earthquakes.csv`: ok 
- `Absorption_1600mTorr_100K_2mins.csv`: ok 
- `Absorption_1600mTorr_90K_10mins.csv`: ok 
- `Absorption_1600mTorr_140K_10mins.csv`: ok 
- `Absorption_1600mTorr_150K_2mins.csv`: ok 
- `Absorption_1600mTorr_140K_2mins.csv`: ok 
- `Absorption_400mTorr_130K_10mins.csv`: ok 
- `Absorption_400mTorr_150K_2mins.csv`: ok 
- `Absorption_1600mTorr_110K_2mins.csv`: ok 
- `Absorption_400mTorr_100K_10mins.csv`: ok 
- `Absorption_1600mTorr_120K_2mins.csv`: ok 
- `Absorption_400mTorr_110K_20mins.csv`: ok 
- `Absorption_400mTorr_100K_2mins.csv`: ok 
- `Absorption_800mTorr_120K_10mins.csv`: ok 

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

40 files compared. Column types change on 6 files (rate 0.1383 of columns). Temporal span recall median 0.25 over 8 files; 5 files' samples cover less than half of the real time span.

## 4. Performance (whole table, one run at a time)

| rows | columns | status | seconds | peak MB |
|---|---|---|---|---|
| 1000 | 10 | ok | 2.51 | 1430.5 |
| 1000 | 100 | ok | 4.04 | 1450.4 |
| 1000 | 1000 | ok | 9.03 | 1560.7 |
| 10000 | 10 | ok | 4.97 | 1448.7 |
| 10000 | 100 | ok | 9.1 | 1572.3 |
| 100000 | 10 | ok | 13.66 | 1592.2 |
| 100000 | 100 | ok | 23.76 | 2650.6 |
