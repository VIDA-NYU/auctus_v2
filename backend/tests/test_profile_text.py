"""build_profile_text: what AutoDDG is told about a file (no network, no LLM)."""

import json
import unittest

from storage.arq_worker import WIDE_TABLE_COLUMN_THRESHOLD, build_profile_text


def record(n_columns: int = 3, distinct: int = 418) -> dict:
    columns = [{"name": "participant_id", "structural_type": "Text", "num_distinct_values": distinct},
               {"name": "year", "structural_type": "Integer", "min": 2001, "max": 2021, "mean": 2011.0,
                "stddev": 6.1}]
    columns += [{"name": f"c{i}", "structural_type": "Integer", "min": 0, "max": 1} for i in range(n_columns - 2)]
    return {"profiler_metadata": {"nb_rows": 399, "nb_columns": n_columns, "columns": columns},
            "temporal_coverage": {"start": "2001-01-01", "end": "2021-12-31"},
            "spatial_coverage": {"label": "World", "bbox": [-180, -90, 180, 90]}}


class ProfileText(unittest.TestCase):
    def test_date_range_is_sent(self):
        out = json.loads(build_profile_text(record()))
        self.assertEqual(out["temporal_coverage"], {"start": "2001-01-01", "end": "2021-12-31"})

    def test_stddev_is_sent_for_narrow_tables(self):
        year = json.loads(build_profile_text(record()))["columns"][1]
        self.assertEqual((year["mean"], year["stddev"]), (2011.0, 6.1))

    def test_wide_tables_keep_ranges_not_moments(self):
        year = json.loads(build_profile_text(record(n_columns=WIDE_TABLE_COLUMN_THRESHOLD)))["columns"][1]
        self.assertEqual((year["min"], year["max"]), (2001, 2021))
        self.assertNotIn("mean", year)

    def test_distinct_count_capped_at_rows(self):
        pid = json.loads(build_profile_text(record()))["columns"][0]
        self.assertEqual(pid["num_distinct_values"], 399)


if __name__ == "__main__":
    unittest.main()
