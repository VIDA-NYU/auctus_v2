# README type mismatches by failure mode

247 of 498 scored README columns disagree with the profiler; 6 more README columns don't exist in their file (unscored).

| failure mode | columns | on | what it is |
|---|---|---|---|
| coded_categories | 141 | profiler | categories stored as numeric codes (1 = 'Not at all') typed numeric; the README says categorical |
| place_false_positive | 46 | profiler | geo-classifier labelled a non-place column (codes, flags, numbers) as a place |
| other | 35 | unclear | doesn't fit a rule; read the values |
| yes_no_flags | 15 | ambiguous | a yes/no or 0/1 column read as categorical/numeric vs boolean |
| text_vs_categorical | 6 | ambiguous | free text vs categorical: a granularity call, not an error |
| not_in_file | 6 | gold | README column name not found in the file (extraction error or README drift) |
| missing_codes | 2 | data | missing-value codes written as text (NA, '.', -99, 'NaN') mixed into a numeric or date column, so it gets a weaker type |
| formatted_numbers | 2 | profiler | numbers with $, % or thousands separators typed as text |

Scored mismatches by attribution: profiler 189, unclear 35, ambiguous 21, data 2

## coded_categories (141)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | make_plan | Categorical | numeric | 10, 9, 8, 7, 5 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | urge_parents | Categorical | numeric | 10, 9, 8, 5, 6 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | listen_attentively | Categorical | numeric | 10, 9, 8, 7, 6 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | nonverbal | Categorical | numeric | 10, 9, 8, 6, 7 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | show_empathy | Categorical | numeric | 10, 9, 8, 6, 7 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | clarify | Categorical | numeric | 10, 9, 8, 5, 7 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | check_understanding | Categorical | numeric | 10, 8, 9, 5, 4 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | shared_decision | Categorical | numeric | 10, 9, 8, 7, 6 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | close_conversation | Categorical | numeric | 10, 8, 9, 6, 1 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | cope_emotional | Categorical | numeric | 10, 9, 8, 7, 5 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | cope_angry | Categorical | numeric | 10, 9, 8, 1, 6 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | p_age_birth_cat | Numeric | numeric | 4, 3, 2, 5, 6 |
| ... | 129 more | | | | |

## place_false_positive (46)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| avp10-a8h86 | us_multicity_zipcode_covid_19_rates_data | Zip Code | integer | place | 33185, 33183, 33326, 33325, 33314 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | protect_stis | Categorical | place | 2, 1 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | competency_videos | Categorical | place | 4, 3 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | facilityid | Categorical | place | 3, 1, 5, 4, 2 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | hhaid___1 | Numeric | place | 1, 0 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | hhaid___2 | Numeric | place | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | hhaid___3 | Numeric | place | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | hhaid___4 | Numeric | place | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | hhaid___5 | Numeric | place | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | pbq1 | Numeric | place | 2, 1, 3 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | pbq2 | Numeric | place | 2, 1, 3 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | pbq4 | Numeric | place | 2, 1, 3 |
| ... | 34 more | | | | |

## other (35)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | uniqueID | Continuous | identifier | 1101, 1102, 1103, 1104, 1105 |
| bp0pk-te632 | PilotStrategiesDatasetForSharing.csv | identify_issues | Categorical | identifier | 10, 9, 8, 5, 6 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | record_id | Integer | identifier | 1, 2, 3, 4, 5 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | gender | Numeric | boolean | 1, 0 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | fies1 | Numeric | boolean | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | fies2 | Numeric | boolean | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | fies3 | Numeric | boolean | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | fies4 | Numeric | boolean | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | fies5 | Numeric | boolean | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | fies6 | Numeric | boolean | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | fies7 | Numeric | boolean | 0, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | fies8 | Numeric | boolean | 0, 1 |
| ... | 23 more | | | | |

## yes_no_flags (15)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | control | Numeric | numeric | 2, 1 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | survey0_pt1_demo_complete | Numeric | numeric | 2 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | survey0_pt2_base_complete | Numeric | numeric | 2 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | survey1_complete | Numeric | numeric | 2, 0 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | survey1_pt1_complete | Numeric | numeric | 2, 0 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | survey1_satisfaction_complete | Numeric | numeric | 2, 0 |
| d8z0p-brn64 | 2026-07-20_SPSKQuasi_UltraViolet.csv | survey2_complete | Numeric | numeric | 2, 0 |
| frv59-hzf81 | KS_data.csv | Gender |  | numeric | 1, 2 |
| frv59-hzf81 | KS_data.csv | Insur |  | numeric | 1, 2 |
| frv59-hzf81 | KS_data.xlsx | Gender |  | numeric | 1, 2 |
| frv59-hzf81 | KS_data.xlsx | Insur |  | numeric | 1, 2 |
| tn478-2kc30 | 2020_NYU_egress_covid19_raw_v3.csv | Nam_Frm | binary | categorical | ok |
| ... | 3 more | | | | |

## text_vs_categorical (6)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| dshqk-atn19 | ukraineinrussianpublishing.csv | Language | Text | categorical | Russian, Russian, translated from Ukrainian, Ukrainian, translated fro |
| dshqk-atn19 | ukraineinrussianpublishing.csv | Category | Text | categorical | History, Ukrainian literature, Russian literature, Political thought,  |
| dshqk-atn19 | ukraineinrussianpublishing.csv | Subcategory 1 | Text | categorical | Genre fiction, Post-Soviet, Poetry, World War II, Classics |
| dshqk-atn19 | ukraineinrussianpublishing.csv | Subcategory 2 | Text | categorical | Sci-fi/Fantasy, Historical fiction, Military advenure novels, Detectiv |
| fz2tx-v7x97 | TaggedTranscripts.csv | tag | Categorical | text | Inspiration.books, Timeconstraints, Research.online.artists, Inspirati |
| tn478-2kc30 | 2020_NYU_egress_covid19_raw_v3.csv | X3r_T_1 | Categorical | text | NA, Personal Object (non-cell phone), Vehicle, Building/Subway Part, S |

## not_in_file (6)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| 2ynm3-snr05 | global-connectedness-world-trends-2001-2 | index | continuous (2001 = 100) | None | - |
| frv59-hzf81 | KS_data.csv | BQ | number | None | - |
| frv59-hzf81 | KS_data.xlsx | BQ | number | None | - |
| x1xmj-5nq11 | global-connectedness-world-trends-2001-2 | index | continuous | None | - |
| ztrqs-yd018 | global-connectedness-world-trends-2001-2 | index | continuous | None | - |
| zw1dd-9xk53 | global-connectedness-world-trends-2001-2 | index | continuous (2001 = 100) | None | - |

## missing_codes (2)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| 9wnjp-kez15 | Merged-Costs-1-4.csv | Start_year | YYYY | numeric | NaN, 2021, 2020, 2019, 2017 |
| 9wnjp-kez15 | Merged-Costs-1-4.csv | End_year | YYYY | numeric | NaN, 2020, 2022, 2021, 2019 |

## formatted_numbers (2)

| dataset | file | column | README type | profiler | values seen |
|---|---|---|---|---|---|
| tn478-2kc30 | 2020_NYU_egress_covid19_raw_v3.csv | Humidity |  | categorical | 33%, 21%, 97%, 25%, 32% |
| tn478-2kc30 | 2020_NYU_egress_covid19_raw_v3.csv | Prc_2_p |  | categorical | 80%, 68%, 79%, 66%, 44% |

