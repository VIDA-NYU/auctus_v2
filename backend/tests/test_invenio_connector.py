"""Offline reliability tests for the InvenioRDM connector.

Covers discovery parsing (crawler) and the full transformer on local fixture files,
including every failure mode found on UltraViolet, Zenodo and CaltechDATA so far.
No network, no Docker services. Run before merging any connector change:

    cd backend && python -m unittest tests.test_invenio_connector -v

Takes ~1-2 minutes: each transformer test runs atlas-profiler for real.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crawlers.invenio.crawler import file_entries, html_to_text, summarize_record
from crawlers.invenio.transformer import MAX_PROFILED_COLUMNS, profile_local_file

DOMAIN = "repo.example.edu"


def invenio_record(files: dict[str, int], access: str = "public") -> dict:
    """A minimal record in the standard InvenioRDM JSON shape."""
    return {
        "id": "abcde-12345",
        "updated": "2026-01-02T03:04:05+00:00",
        "pids": {"doi": {"identifier": "10.1234/abcde-12345"}},
        "access": {"files": access},
        "metadata": {
            "title": "Test record",
            "description": "<p>Line one &amp; <b>two</b></p>",
            "resource_type": {"id": "dataset"},
            "creators": [{"person_or_org": {"name": "Doe, Jane"}}],
            "subjects": [{"subject": "Testing"}],
            "rights": [{"id": "cc-by-4.0"}],
            "publication_date": "2026-01",
        },
        "files": {"entries": {key: {"id": f"id-{i}", "key": key, "size": size}
                              for i, (key, size) in enumerate(files.items())}},
    }


class DiscoveryTests(unittest.TestCase):
    def test_keeps_only_public_tabular_files_under_the_size_cap(self):
        record = invenio_record({"a.csv": 10, "b.XLSX": 10, "c.pdf": 10, "README.md": 10, "huge.csv": 10**9})
        keys = [e["file_key"] for e in file_entries(record, DOMAIN, max_file_bytes=10**6)]
        self.assertEqual(keys, ["a.csv", "b.XLSX"])

    def test_restricted_records_yield_nothing(self):
        record = invenio_record({"a.csv": 10}, access="restricted")
        self.assertEqual(file_entries(record, DOMAIN, max_file_bytes=10**6), [])

    def test_ids_are_domain_scoped_and_urls_are_quoted(self):
        entry = file_entries(invenio_record({"my data.csv": 10}), DOMAIN, max_file_bytes=10**6)[0]
        self.assertEqual(entry["dataset_id"], f"{DOMAIN}:abcde-12345:id-0")
        self.assertTrue(entry["file_url"].endswith("/files/my%20data.csv/content"))
        self.assertEqual(entry["source_updated_at"], "2026-01-02T03:04:05+00:00")

    def test_record_metadata_is_carried_and_html_stripped(self):
        summary = summarize_record(invenio_record({}), DOMAIN)
        self.assertEqual(summary["description"], "Line one & two")
        self.assertEqual(summary["doi"], "10.1234/abcde-12345")
        self.assertEqual(summary["creators"], ["Doe, Jane"])
        self.assertEqual(summary["landing_url"], f"https://{DOMAIN}/records/abcde-12345")
        self.assertEqual(html_to_text(None), "")


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        rng = np.random.default_rng(0)
        n = 300
        cls.frame = pd.DataFrame({
            "count": rng.integers(0, 100, n),
            "measure": rng.normal(10, 2, n).round(3),
            "date": pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
            "category": rng.choice(["low", "medium", "high"], n),
            "note": ["plain text value"] * n,
        })

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def profile(self, name: str, write) -> dict:
        path = self.dir / name
        write(path)
        return profile_local_file(path)

    def assert_profiled(self, record: dict) -> None:
        """No errors, the fixture's exact column names, and every data row (header not read as data)."""
        meta = record["profiler_metadata"]
        self.assertEqual(record["errors"], [])
        self.assertEqual([c["name"] for c in meta["columns"]], list(self.frame.columns))
        self.assertEqual(meta["nb_rows"], len(self.frame))

    def test_comma_csv(self):
        self.assert_profiled(self.profile("comma.csv", lambda p: self.frame.to_csv(p, index=False)))

    def test_semicolon_csv(self):  # found on CaltechDATA: read as one column before delimiter sniffing
        self.assert_profiled(self.profile("semi.csv", lambda p: self.frame.to_csv(p, index=False, sep=";")))

    def test_tsv(self):
        self.assert_profiled(self.profile("tabs.tsv", lambda p: self.frame.to_csv(p, index=False, sep="\t")))

    def test_windows_1252_csv(self):  # found on UltraViolet (xwvc0-mse16)
        frame = self.frame.assign(note="café – résumé")
        self.assert_profiled(self.profile("cp1252.csv", lambda p: frame.to_csv(p, index=False, encoding="cp1252")))

    def test_utf8_bom_csv(self):
        record = self.profile("bom.csv", lambda p: self.frame.to_csv(p, index=False, encoding="utf-8-sig"))
        self.assert_profiled(record)

    def test_xlsx(self):
        self.assert_profiled(self.profile("sheet.xlsx", lambda p: self.frame.to_excel(p, index=False)))

    def test_byte_cap_inside_quoted_multiline_field(self):  # found on UltraViolet: "EOF inside string"
        long_cell = '"' + ("line of text\n" * 40) + '"'
        body = "id,text\n" + "".join(f"{i},{long_cell}\n" for i in range(6000))  # > 2 MB
        record = self.profile("multiline.csv", lambda p: p.write_text(body))
        self.assertEqual(record["errors"], [])
        self.assertGreater(record["profiler_metadata"]["nb_rows"], 0)

    def test_very_wide_table_is_capped_but_all_names_kept(self):  # found on CaltechDATA (38k columns)
        wide = pd.DataFrame(np.ones((5, 1000)), columns=[f"gene_{i}" for i in range(1000)])
        record = self.profile("wide.csv", lambda p: wide.to_csv(p, index=False))
        meta = record["profiler_metadata"]
        self.assertEqual(meta["nb_columns"], 1000)
        self.assertEqual(len(meta["columns"]), MAX_PROFILED_COLUMNS)
        self.assertEqual(len(meta["attribute_keywords"]), 1000)
        self.assertEqual(meta["columns_truncated"], {"profiled": MAX_PROFILED_COLUMNS, "total": 1000})

    def test_no_fallback_bounding_box(self):  # repository data is global: no coordinates, no bbox
        record = self.profile("nocoords.csv", lambda p: self.frame.to_csv(p, index=False))
        self.assertNotIn("spatial_coverage", record)
        self.assertIsNone(record["profiler_metadata"]["spatial_bbox"])

    def test_lat_lon_gives_a_bounding_box(self):
        rng = np.random.default_rng(1)
        points = pd.DataFrame({"latitude": rng.uniform(40.5, 40.9, 50), "longitude": rng.uniform(-74.2, -73.7, 50)})
        record = self.profile("points.csv", lambda p: points.to_csv(p, index=False))
        self.assertIn("spatial_coverage", record)

    @unittest.expectedFailure
    def test_title_rows_above_header(self):  # known gap: header is assumed to be on row 1
        preamble = "Survey of permits 2020\nSource: NYC DOB\n\n"
        record = self.profile("preamble.csv", lambda p: p.write_text(preamble + self.frame.to_csv(index=False)))
        self.assert_profiled(record)


if __name__ == "__main__":
    unittest.main()
