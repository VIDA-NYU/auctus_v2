# Sampling sweep: first N rows vs random N rows, compared with the whole file

| rows | strategy | files | column types same as whole file | files with any type change | time-span recall (mean / worst) | map-box recall (mean) | median s |
|---|---|---|---|---|---|---|---|
| 500 | head | 23 | 83.3% | 7 | 0.405 / 0.125 (9 files) | 0.638 (9 files) | 3.99 |
| 500 | random | 23 | 85.0% | 9 | 0.991 / 0.921 (9 files) | 0.902 (9 files) | 4.58 |
| 2000 | head | 13 | 94.5% | 2 | 0.695 / 0.414 (8 files) | 0.877 (9 files) | 5.9 |
| 2000 | random | 13 | 88.0% | 6 | 1.0 / 1.0 (8 files) | 0.917 (9 files) | 6.24 |
| 5000 | head | 7 | 100.0% | 0 | 1.0 / 1.0 (6 files) | 0.985 (7 files) | 8.98 |
| 5000 | random | 7 | 99.1% | 1 | 1.0 / 1.0 (6 files) | 0.898 (7 files) | 8.75 |
