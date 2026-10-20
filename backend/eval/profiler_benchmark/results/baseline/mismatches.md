# README type mismatches by failure mode

19 of 128 scored README columns disagree with the profiler; 14 more README columns don't exist in their file (unscored).

| failure mode | columns | on | what it is |
|---|---|---|---|
| not_in_file | 14 | gold | README column name not found in the file (extraction error or README drift) |
| coded_categories | 11 | profiler | categories stored as numeric codes (1 = 'Not at all') typed numeric; the README says categorical |
| place_false_positive | 4 | profiler | geo-classifier labelled a non-place column (codes, flags, numbers) as a place |
| missing_codes | 2 | data | missing-value codes written as text (NA, '.', -99, 'NaN') mixed into a numeric or date column, so it gets a weaker type |
| other | 2 | unclear | doesn't fit a rule; read the values |

Scored mismatches by attribution: profiler 15, data 2, unclear 2

## not_in_file (14)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| 2ynm3-snr05 | global-connectedness-world-trends-2001-2 | index | continuous (2001 = 100) | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | improve_un derstanding_training | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | dev elop_skills_training | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | mor e_confident_training | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | s atisfaction_training | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | impor tant_topics_training | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | com fortability_training | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | improve_ understanding_videos | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | d evelop_skills_videos | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | m ore_confident_videos | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | imp ortant_topics_videos | Categorical: | None | - |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | c omfortability_videos | Categorical: | None | - |
| ... | 2 more | | | | |

## coded_categories (11)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | make_plan | Categorical: | numeric | 10, 9, 8, 7, 5 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | urge_parents | Categorical: | numeric | 10, 9, 8, 5, 6 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | listen_attentively | Categorical: | numeric | 10, 9, 8, 7, 6 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | nonverbal | Categorical: | numeric | 10, 9, 8, 6, 7 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | show_empathy | Categorical: | numeric | 10, 9, 8, 6, 7 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | clarify | Categorical: | numeric | 10, 9, 8, 5, 7 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | check_understanding | Categorical: | numeric | 10, 8, 9, 5, 4 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | shared_decision | Categorical: | numeric | 10, 9, 8, 7, 6 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | close_conversation | Categorical: | numeric | 10, 8, 9, 6, 1 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | cope_emotional | Categorical: | numeric | 10, 9, 8, 7, 5 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | cope_angry | Categorical: | numeric | 10, 9, 8, 1, 6 |

## place_false_positive (4)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | protect_stis | Categorical: | place | 2, 1 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | competency_videos | Categorical: | place | 4, 3 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | facilityid | Categorical: | place | 3, 1, 5, 4, 2 |
| vsszd-s6t21 | AnGaodhal_pageMetadata.csv | hasSong_Poem |  | place | No, Yes |

## missing_codes (2)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| 9wnjp-kez15 | Merged-Costs-1-4.csv | Start_year | YYYY | numeric | NaN, 2021, 2020, 2019, 2017 |
| 9wnjp-kez15 | Merged-Costs-1-4.csv | End_year | YYYY | numeric | NaN, 2020, 2022, 2021, 2019 |

## other (2)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | uniqueID | Continuous | identifier | 1101, 1102, 1103, 1104, 1105 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | identify_issues | Categorical: | identifier | 10, 9, 8, 5, 6 |

