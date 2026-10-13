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

from crawlers.coverage import admin_bbox, bbox_from_profiler, geo_data, temporal_from_profiler, temporal_from_year_columns
import zipfile

from crawlers.invenio.archives import member_dataset_id, select_archive_members
from crawlers.invenio.crawler import archive_member_entries, file_entries, html_to_text, summarize_record
from crawlers.socrata.transformer import _safe_datetime
from crawlers.invenio.transformer import MAX_PROFILED_COLUMNS, profile_archive, profile_local_file

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


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.zip_path = Path(cls.tmp.name) / "data.zip"
        frame = pd.DataFrame({"year": range(2000, 2020), "value": range(20), "label": ["a", "b"] * 10})
        sheet = Path(cls.tmp.name) / "sheet.xlsx"
        frame.to_excel(sheet, index=False)
        with zipfile.ZipFile(cls.zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data/b_table.csv", frame.to_csv(index=False))
            archive.writestr("data/a_table.tsv", frame.to_csv(index=False, sep="\t"))
            archive.write(sheet, "data/sheet.xlsx")
            archive.writestr("__MACOSX/data/._b_table.csv", "resource fork")
            archive.writestr("data/.hidden.csv", "x,y\n1,2\n")
            archive.writestr("data/~$lock.xlsx", "lock")
            archive.writestr("data/readme.txt", "not tabular")
            archive.writestr("data/empty_dir/", "")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_member_selection_skips_junk_and_caps(self):
        with zipfile.ZipFile(self.zip_path) as archive:
            members, total = select_archive_members(archive)
            self.assertEqual([m.filename for m in members], ["data/a_table.tsv", "data/b_table.csv", "data/sheet.xlsx"])
            self.assertEqual(total, 3)
            capped, total = select_archive_members(archive, limit=2)
            self.assertEqual((len(capped), total), (2, 3))

    def test_small_archive_profiles_each_member(self):
        meta = {"dataset_id": f"{DOMAIN}:abcde-12345:zip-id", "file_key": "data.zip"}
        records = profile_archive(self.zip_path, meta)
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertEqual(record["errors"], [])
            self.assertEqual([c["name"] for c in record["profiler_metadata"]["columns"]], ["year", "value", "label"])
            self.assertEqual(record["archive_dataset_id"], meta["dataset_id"])
            self.assertEqual(record["source_record"]["archive"]["tabular_members"], 3)
        self.assertEqual(records[1]["id"], member_dataset_id(meta["dataset_id"], "data/b_table.csv"))
        self.assertEqual(records[1]["temporal_coverage"], {"start": "2000-01-01", "end": "2019-12-31"})

    def test_large_archive_becomes_one_job_per_member(self):
        entry = {"dataset_id": f"{DOMAIN}:abcde-12345:zip-id", "file_key": "data.zip", "file_extension": "zip",
                 "file_size": 10**10, "file_url": "https://repo.example.edu/files/data.zip/content", "record": {}}
        with zipfile.ZipFile(self.zip_path) as archive:
            jobs = archive_member_entries(entry, archive)
        self.assertEqual([j["zip_member"] for j in jobs], ["data/a_table.tsv", "data/b_table.csv", "data/sheet.xlsx"])
        self.assertEqual([j["file_extension"] for j in jobs], ["tsv", "csv", "xlsx"])
        # Crawler and transformer must derive the same id for a member.
        self.assertEqual(jobs[1]["dataset_id"], member_dataset_id(entry["dataset_id"], "data/b_table.csv"))
        self.assertEqual(jobs[0]["archive"]["url"], entry["file_url"])

    def test_discovery_keeps_zips(self):
        keys = [e["file_key"] for e in file_entries(invenio_record({"a.zip": 10**11, "b.csv": 10}), DOMAIN, 10**6)]
        self.assertEqual(keys, ["a.zip", "b.csv"])  # zips have no size cap: they're read in place


class CoverageTests(unittest.TestCase):
    def test_year_columns_give_a_time_range(self):  # UltraViolet: profiler missed a plain `Year` column
        rows = [{"Year": str(y), "Title": "x"} for y in range(2000, 2006)] + [{"Year": "2003.0", "Title": "y"}]
        self.assertEqual(temporal_from_year_columns(["Year", "Title"], rows), ("2000-01-01", "2005-12-31"))
        self.assertEqual(temporal_from_year_columns(["Count"], [{"Count": "2001"}]), (None, None))
        not_years = [{"year_code": v} for v in ["A1", "B2", "C3"]]
        self.assertEqual(temporal_from_year_columns(["year_code"], not_years), (None, None))

    def test_profiler_ranges_to_dates_and_bbox(self):
        output = {
            "temporal_coverage": [{"ranges": [{"range": {"gte": 0, "lte": 86400 * 366}}]}],
            "spatial_coverage": [{"ranges": [{"range": {"type": "envelope", "coordinates": [[-74.2, 40.9], [-73.7, 40.5]]}}]}],
        }
        self.assertEqual(temporal_from_profiler(output), ("1970-01-01", "1971-01-02"))
        self.assertEqual(bbox_from_profiler(output), [-74.2, 40.5, -73.7, 40.9])
        self.assertEqual(temporal_from_profiler({}), (None, None))
        self.assertIsNone(bbox_from_profiler({}))


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

    def test_xlsx_with_empty_first_sheet(self):  # CaltechDATA: data on a later sheet
        def write(path):
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame().to_excel(writer, sheet_name="About", index=False)
                self.frame.to_excel(writer, sheet_name="Data", index=False)
        self.assert_profiled(self.profile("two_sheets.xlsx", write))

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

    def test_temporal_coverage_comes_from_the_profiler(self):
        record = self.profile("dates.csv", lambda p: self.frame.to_csv(p, index=False))
        self.assertEqual(record["temporal_coverage"]["start"], "2020-01-01")
        self.assertEqual(record["temporal_coverage"]["end"], str((pd.Timestamp("2020-01-01") + pd.Timedelta(days=len(self.frame) - 1)).date()))

    def test_bare_years_do_not_get_todays_month_and_day(self):  # dateutil filled "1975" with today's date
        years = pd.DataFrame({"start_year": range(1975, 2000), "value": range(25)})
        record = self.profile("years.csv", lambda p: years.to_csv(p, index=False))
        start, end = record["temporal_coverage"]["start"], record["temporal_coverage"]["end"]
        self.assertTrue(start.startswith("1975"), start)
        self.assertTrue(end.startswith("1999"), end)
        self.assertEqual(_safe_datetime("1975").date().isoformat(), "1975-01-01")
        self.assertIsNone(_safe_datetime("12"))

    def test_lat_lon_bbox_matches_the_points(self):
        points = pd.DataFrame({"latitude": [40.5, 40.9, 40.7], "longitude": [-74.2, -73.7, -74.0], "v": [1, 2, 3]})
        record = self.profile("three_points.csv", lambda p: points.to_csv(p, index=False))
        west, south, east, north = record["profiler_metadata"]["spatial_bbox"]
        self.assertLessEqual(west, -74.2)
        self.assertGreaterEqual(east, -73.7)
        self.assertLessEqual(south, 40.5)
        self.assertGreaterEqual(north, 40.9)

    @unittest.skipUnless(geo_data(), "datamart-geo data not downloaded")
    def test_place_names_give_a_bounding_box(self):  # needs datamart-geo
        places = pd.DataFrame({"country": ["Kenya", "Uganda", "Tanzania", "Rwanda"] * 10, "value": range(40)})
        record = self.profile("countries.csv", lambda p: places.to_csv(p, index=False))
        west, south, east, north = record["profiler_metadata"]["spatial_bbox"]
        self.assertTrue(28 <= west <= 30 and 40 <= east <= 42 and -12 <= south <= -11 and 4 <= north <= 6, (west, south, east, north))

    @unittest.skipUnless(geo_data(), "datamart-geo data not downloaded")
    def test_numeric_columns_are_never_geocoded(self):  # UltraViolet: rates resolved to a province of Iran
        output = {"columns": [{"name": "rate", "semantic_types": ["http://schema.org/AdministrativeArea"]}]}
        rows = [{"rate": v} for v in ["7", "10.5", "5.7", "6.2", "13", "21.1"]]
        self.assertIsNone(admin_bbox(output, rows))

    @unittest.expectedFailure
    def test_title_rows_above_header(self):  # known gap: header is assumed to be on row 1
        preamble = "Survey of permits 2020\nSource: NYC DOB\n\n"
        record = self.profile("preamble.csv", lambda p: p.write_text(preamble + self.frame.to_csv(index=False)))
        self.assert_profiled(record)


if __name__ == "__main__":
    unittest.main()
