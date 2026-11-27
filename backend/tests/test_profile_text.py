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

    def test_wide_tables_send_every_column(self):
        # used to stop at 80 columns: a 230-column survey lost its subject
        cols = json.loads(build_profile_text(record(n_columns=230)))["columns"]
        names = [c["name"] for c in cols if "name" in c] + [n for c in cols for n in c.get("columns", [])]
        self.assertEqual(len(names), 230)
        self.assertNotIn("columns_truncated", json.loads(build_profile_text(record(n_columns=230))))

    def test_wide_tables_group_numbered_columns(self):
        cols = json.loads(build_profile_text(record(n_columns=WIDE_TABLE_COLUMN_THRESHOLD)))["columns"]
        group = cols[2]
        self.assertEqual(group["count"], WIDE_TABLE_COLUMN_THRESHOLD - 2)  # c0 ... c37
        self.assertEqual(group["columns"][:2], ["c0", "c1"])
        self.assertEqual((group["min"], group["max"]), (0, 1))
        self.assertEqual([c.get("name") for c in cols[:2]], ["participant_id", "year"])  # others stay single

    def test_groups_need_the_same_type(self):
        r = record(n_columns=WIDE_TABLE_COLUMN_THRESHOLD)
        r["profiler_metadata"]["columns"][3]["structural_type"] = "Text"  # c1 differs
        cols = json.loads(build_profile_text(r))["columns"]
        self.assertIn("c1", [c.get("name") for c in cols])
        self.assertEqual(next(c for c in cols if "columns" in c)["count"], WIDE_TABLE_COLUMN_THRESHOLD - 3)

    def test_narrow_tables_are_not_grouped(self):
        cols = json.loads(build_profile_text(record(n_columns=5)))["columns"]
        self.assertEqual([c["name"] for c in cols], ["participant_id", "year", "c0", "c1", "c2"])

    def test_distinct_count_capped_at_rows(self):
        pid = json.loads(build_profile_text(record()))["columns"][0]
        self.assertEqual(pid["num_distinct_values"], 399)


if __name__ == "__main__":
    unittest.main()
