"""Offline reliability tests for the InvenioRDM connector.

Covers discovery parsing (crawler) and the full transformer on local fixture files,
including every failure mode found on UltraViolet, Zenodo and CaltechDATA so far.
No network, no Docker services. Run before merging any connector change:

    cd backend && python -m unittest tests.test_invenio_connector -v

Takes ~1-2 minutes: each transformer test runs atlas-profiler for real.
"""

from __future__ import annotations

import io
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
        record = invenio_record({"a.csv": 10, "b.XLSX": 10, "c.pdf": 10, "README.md": 10, "huge.csv": 10**9,
                                 "d.csv.gz": 10, "e.parquet": 10, "f.fastq.gz": 10})
        entries = file_entries(record, DOMAIN, max_file_bytes=10**6)
        self.assertEqual([e["file_key"] for e in entries], ["a.csv", "b.XLSX", "d.csv.gz", "e.parquet"])
        self.assertEqual([e["file_extension"] for e in entries], ["csv", "xlsx", "csv.gz", "parquet"])

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

    def test_parquet_and_gz_members(self):
        frame = pd.DataFrame({"year": range(2000, 2020), "value": range(20)})
        path = Path(self.tmp.name) / "mixed.zip"
        import gzip, io
        parquet, gz = io.BytesIO(), io.BytesIO()
        frame.to_parquet(parquet, index=False)
        with gzip.GzipFile(fileobj=gz, mode="wb") as fh:
            fh.write(frame.to_csv(index=False).encode())
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("tiles/part-0.parquet", parquet.getvalue())
            archive.writestr("tables/table.csv.gz", gz.getvalue())
            archive.writestr("notes/other.txt.gz", b"not tabular")
        records = profile_archive(path, {"dataset_id": f"{DOMAIN}:r:z", "file_key": "mixed.zip"})
        self.assertEqual(sorted(r["source_record"]["archive"]["member"] for r in records),
                         ["tables/table.csv.gz", "tiles/part-0.parquet"])
        for record in records:
            self.assertEqual(record["profiler_metadata"]["nb_rows"], 20)
            self.assertEqual(record["temporal_coverage"], {"start": "2000-01-01", "end": "2019-12-31"})

    def run_discovery(self, record, fake_expand, **kwargs):
        import asyncio
        from unittest import mock
        import crawlers.invenio.crawler as crawler

        page = {"hits": {"hits": [record]}, "links": {}}

        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return page

        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **k): return FakeResponse()

        report = {}
        with mock.patch.object(crawler, "expand_archive", fake_expand), \
             mock.patch.object(crawler, "async_client", lambda *a, **k: FakeClient()):
            jobs = asyncio.run(crawler.discover_invenio_datasets(DOMAIN, report=report, **kwargs))
        return jobs, report

    def test_record_discovery_groups_tiles_and_caps_single_tables(self):
        letters = "abcdefghijklmnopqrstuvwxyz"
        record = invenio_record({f"city_{i}.zip": 10**9 for i in range(5)})

        def fake_expand(entry, unlisted=None):  # each zip: 30 tiles of one table + 20 distinct tables
            i = int(entry["file_key"].split("_")[1].split(".")[0])
            archive = {"dataset_id": entry["dataset_id"], "file_key": entry["file_key"], "url": entry["file_url"],
                       "size": entry["file_size"], "tabular_members": 50}
            names = [f"city{i}/part.{j}.parquet" for j in range(30)] + [f"{letters[i]}{letters[j]}.csv" for j in range(20)]
            return [{**entry, "dataset_id": f"{entry['dataset_id']}:{n}", "file_key": f"{entry['file_key']}/{n}",
                     "file_extension": n.rsplit(".", 1)[1], "file_size": 1000, "zip_member": n, "archive": archive}
                    for n in names]

        jobs, report = self.run_discovery(record, fake_expand, limit=1000)
        groups = [j for j in jobs if j.get("member_group")]
        self.assertEqual(len(groups), 1)  # 150 tiles over 5 zips -> one dataset
        group = groups[0]["member_group"]
        self.assertEqual((group["pattern"], group["total_members"], len(group["archives"])), ("part.#.parquet", 150, 5))
        self.assertEqual(len(group["samples"]), 40)
        self.assertEqual(len({s["archive"] for s in group["samples"]}), 5)  # every zip sampled
        self.assertEqual(len(jobs) - 1, 50)  # single tables: capped per record
        self.assertTrue(report["complete"])
        _, report = self.run_discovery(record, fake_expand, limit=10)
        self.assertFalse(report["complete"])  # cut by limit: never prune on this

    def test_small_zips_are_not_capped_per_record(self):  # UAE Open Data: 60 small zips of distinct tables
        record = invenio_record({f"table_{i}.zip": 1000 for i in range(60)})

        def fake_expand(entry, unlisted=None):
            return [{**entry, "archive_tabular_members": 3}]

        jobs, _ = self.run_discovery(record, fake_expand, limit=1000)
        self.assertEqual(len(jobs), 60)

    def test_same_table_in_two_formats_is_profiled_once(self):
        path = Path(self.tmp.name) / "formats.zip"
        frame = pd.DataFrame({"a": [1, 2]})
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("t/Social.csv", frame.to_csv(index=False))
            archive.writestr("t/Social.xls", b"old excel")
            archive.writestr("t/Other.xlsx", b"excel")
        with zipfile.ZipFile(path) as archive:
            members, total = select_archive_members(archive)
        self.assertEqual([m.filename for m in members], ["t/Other.xlsx", "t/Social.csv"])

    def make_tile_zips(self, cities):
        """One zip per city, each holding Parquet tiles with lat/lon and dates."""
        from crawlers.invenio.crawler import archive_member_entries
        members = []
        for i, (name, lon) in enumerate(cities):
            path = Path(self.tmp.name) / f"{name}.zip"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for j in range(6):
                    frame = pd.DataFrame({
                        "latitude": np.linspace(30 + j, 30.5 + j, 50), "longitude": np.linspace(lon, lon + 0.5, 50),
                        "observed": pd.date_range(f"{2000 + i}-01-01", periods=50, freq="D"),
                        "landuse": ["urban", "water"] * 25})
                    buf = io.BytesIO()
                    frame.to_parquet(buf, index=False)
                    archive.writestr(f"{name}/parts/part.{j}.parquet", buf.getvalue())
            entry = {"dataset_id": f"{DOMAIN}:tiles-rec:{name}", "file_key": f"{name}.zip", "file_extension": "zip",
                     "file_size": path.stat().st_size, "file_url": str(path),
                     "record": {"title": "Land use", "record_id": "tiles-rec"}}
            with zipfile.ZipFile(path) as archive:
                members += archive_member_entries(entry, archive)
        return members

    def test_group_profile_covers_every_zip(self):
        from crawlers.invenio.crawler import group_job
        from crawlers.invenio.transformer import profile_member_group
        members = self.make_tile_zips([("houston", -95.5), ("boston", -71.2), ("denver", -105.0)])
        job = group_job("part.#.parquet", members)
        records = profile_member_group(job, opener=lambda url, size: zipfile.ZipFile(url))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["id"], job["dataset_id"])
        self.assertEqual(record["archive_dataset_id"], job["dataset_id"])
        bbox = record["profiler_metadata"]["spatial_bbox"]
        self.assertAlmostEqual(bbox[0], -105.0)  # west edge from Denver
        self.assertAlmostEqual(bbox[2], -70.7)   # east edge from Boston
        self.assertEqual(record["temporal_coverage"]["start"], "2000-01-01")
        self.assertEqual(record["temporal_coverage"]["end"], "2002-02-19")
        self.assertEqual(record["profiler_metadata"]["nb_rows"], 18 * 50)  # all parts sampled: exact
        self.assertFalse(record["profiler_metadata"]["nb_rows_estimated"])
        self.assertEqual(record["source_record"]["member_group"]["total_members"], 18)

    def test_group_profile_from_a_sample_estimates_rows(self):
        from crawlers.invenio.crawler import group_job
        from crawlers.invenio.transformer import profile_member_group
        members = self.make_tile_zips([("austin", -97.7), ("miami", -80.2)])
        job = group_job("part.#.parquet", members, max_samples=4)
        record = profile_member_group(job, opener=lambda url, size: zipfile.ZipFile(url))[0]
        self.assertTrue(record["profiler_metadata"]["nb_rows_estimated"])
        self.assertAlmostEqual(record["profiler_metadata"]["nb_rows"], 12 * 50, delta=12 * 50 * 0.1)
        self.assertEqual(record["source_record"]["member_group"]["sampled_parts"], 4)

    def test_geoparquet_bbox_and_index_columns(self):  # uMORPH: WKB geometry + GeoParquet bbox
        import json as _json
        import pyarrow as pa
        import pyarrow.parquet as pq
        from crawlers.invenio.transformer import normalize_table
        table = pa.table({"height": [3.0, 9.5], "geometry": [b"\x01", b"\x02"], "__null_dask_index__": [0, 1]})
        geo = {"primary_column": "geometry", "columns": {"geometry": {"encoding": "WKB", "bbox": [142.0, -40.0, 144.0, -38.0]}}}
        path = Path(self.tmp.name) / "geo.parquet"
        pq.write_table(table.replace_schema_metadata({b"geo": _json.dumps(geo).encode()}), path)
        normalized = normalize_table(path, "parquet")
        self.assertEqual(normalized.headers, ["height", "geometry"])
        self.assertEqual(normalized.geo_bbox, [142.0, -40.0, 144.0, -38.0])
        geo["columns"]["geometry"]["crs"] = None  # unknown CRS: don't trust the numbers
        pq.write_table(table.replace_schema_metadata({b"geo": _json.dumps(geo).encode()}), path)
        self.assertIsNone(normalize_table(path, "parquet").geo_bbox)

    def test_group_with_two_schemas_gives_two_datasets(self):
        from crawlers.invenio.crawler import group_job
        from crawlers.invenio.transformer import profile_member_group
        members = self.make_tile_zips([("reno", -119.8)])
        path = Path(self.tmp.name) / "other.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for j in range(3):
                buf = io.BytesIO()
                pd.DataFrame({"height": range(30), "floors": range(30)}).to_parquet(buf, index=False)
                archive.writestr(f"other/part.{j}.parquet", buf.getvalue())
        from crawlers.invenio.crawler import archive_member_entries
        entry = {"dataset_id": f"{DOMAIN}:tiles-rec:other", "file_key": "other.zip", "file_extension": "zip",
                 "file_size": path.stat().st_size, "file_url": str(path), "record": {"title": "Land use"}}
        with zipfile.ZipFile(path) as archive:
            members += archive_member_entries(entry, archive)
        job = group_job("part.#.parquet", members)
        records = profile_member_group(job, opener=lambda url, size: zipfile.ZipFile(url))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], job["dataset_id"])  # the bigger schema keeps the group id
        self.assertTrue(records[1]["id"].startswith(job["dataset_id"] + ":"))
        self.assertEqual({r["archive_dataset_id"] for r in records}, {job["dataset_id"]})

    def test_stale_docs(self):
        from crawlers.invenio.crawler import stale_doc_ids
        discovered = [
            {"dataset_id": "d:r1:a.csv", "file_extension": "csv"},
            {"dataset_id": "d:r1:small.zip", "file_extension": "zip"},            # small archive job
            {"dataset_id": "d:r2:big.zip:m1", "file_extension": "parquet", "zip_member": "m1.parquet"},
        ]
        indexed = [
            {"id": "d:r1:a.csv"},                                                  # still there
            {"id": "d:r1:small.zip:x", "archive_dataset_id": "d:r1:small.zip"},    # member of a small archive
            {"id": "d:r2:big.zip:m1", "archive_dataset_id": "d:r2:big.zip"},       # selected member
            {"id": "d:r2:big.zip:m9", "archive_dataset_id": "d:r2:big.zip"},       # beyond the cap now
            {"id": "d:r3:gone.csv"},                                               # record deleted/restricted
            {"id": "d:r4:flaky.zip:y", "archive_dataset_id": "d:r4:flaky.zip"},    # zip unlisted this run
            {"id": "d:r5:group:abc:1f", "archive_dataset_id": "d:r5:group:abc"},   # second schema of a group
        ]
        discovered.append({"dataset_id": "d:r5:group:abc", "file_extension": "parquet", "member_group": {"pattern": "p.#"}})
        self.assertEqual(stale_doc_ids(indexed, discovered, {"d:r4:flaky.zip"}),
                         ["d:r2:big.zip:m9", "d:r3:gone.csv"])

    def test_superseded_archive_docs_are_removed(self):
        from unittest import mock
        from storage.arq_worker import remove_superseded_docs
        os_client, storage = mock.Mock(), mock.Mock()
        os_client.search.return_value = {"hits": {"hits": [{"_id": "g"}, {"_id": "g:old-schema"}]}}
        self.assertEqual(remove_superseded_docs(os_client, storage, "g", {"g"}), ["g:old-schema"])
        os_client.delete.assert_called_once()

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
        # The profiler's streaming CSV-path mode, not the in-memory fallback (which a
        # polars upgrade or a normalization bug would silently trigger).
        self.assertEqual(meta.get("profiling_mode"), "full-file")

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

    def test_gzipped_csv_and_tsv(self):  # CaltechDATA has ~170 .gz files
        import gzip
        for name, sep in (("data.csv.gz", ","), ("data.tsv.gz", "\t")):
            def write(path, sep=sep):
                with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
                    self.frame.to_csv(fh, index=False, sep=sep)
            self.assert_profiled(self.profile(name, write))

    def test_parquet(self):  # UltraViolet has ~34k Parquet files inside zips
        self.assert_profiled(self.profile("data.parquet", lambda p: self.frame.to_parquet(p, index=False)))

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

    def test_whole_file_time_range_when_sorted_by_date(self):  # the first 500 rows covered ~2% of this
        n = 20000
        frame = pd.DataFrame({"when": pd.date_range("2001-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
                              "value": range(n)})
        record = self.profile("sorted_by_time.csv", lambda p: frame.to_csv(p, index=False))
        self.assertEqual(record["profiler_metadata"]["nb_rows"], n)
        self.assertEqual(record["temporal_coverage"], {"start": "2001-01-01", "end": "2003-04-14"})

    def test_whole_file_bbox_when_sorted_by_place(self):
        lat = np.linspace(-30.0, 60.0, 8000)
        frame = pd.DataFrame({"latitude": lat, "longitude": np.linspace(-120.0, 100.0, 8000), "v": range(8000)})
        record = self.profile("sorted_by_lat.csv", lambda p: frame.to_csv(p, index=False))
        west, south, east, north = record["profiler_metadata"]["spatial_bbox"]
        self.assertTrue(south <= -29.9 and north >= 59.9 and west <= -119.9 and east >= 99.9, (west, south, east, north))

    @unittest.skipUnless(geo_data(), "datamart-geo data not downloaded")
    def test_place_names_after_the_first_rows_count(self):
        names = ["Kenya"] * 600 + ["Uganda", "Tanzania", "Rwanda", "Nigeria", "Ghana"] * 100
        frame = pd.DataFrame({"country": names, "value": range(len(names))})
        record = self.profile("late_places.csv", lambda p: frame.to_csv(p, index=False))
        west, south, east, north = record["profiler_metadata"]["spatial_bbox"]
        self.assertLess(west, 0)  # Ghana and Nigeria are west of the prime meridian / at 3°E: only in rows 600+

    def test_written_missing_values_keep_numbers_numeric(self):  # UltraViolet DHL files: literal "NULL"
        values = [str(i * 1.5) for i in range(400)] + ["NULL", "NA", "n/a", "NaN"] * 25
        frame = pd.DataFrame({"score": values, "id": range(len(values))})
        record = self.profile("nulls.csv", lambda p: frame.to_csv(p, index=False))
        column = next(c for c in record["profiler_metadata"]["columns"] if c["name"] == "score")
        self.assertTrue(column["structural_type"].endswith("Float"), column["structural_type"])

    def test_semicolon_file_counts_every_row(self):
        frame = pd.DataFrame({"a": range(3000), "b": ["x;y"] * 3000})  # a quoted separator inside values too
        record = self.profile("semi_big.csv", lambda p: frame.to_csv(p, index=False, sep=";"))
        self.assertEqual(record["profiler_metadata"]["nb_rows"], 3000)
        self.assertEqual([c["name"] for c in record["profiler_metadata"]["columns"]], ["a", "b"])

    @unittest.expectedFailure
    def test_title_rows_above_header(self):  # known gap: header is assumed to be on row 1
        preamble = "Survey of permits 2020\nSource: NYC DOB\n\n"
        record = self.profile("preamble.csv", lambda p: p.write_text(preamble + self.frame.to_csv(index=False)))
        self.assert_profiled(record)


if __name__ == "__main__":
    unittest.main()



class CodebookTests(unittest.TestCase):
    """Value labels from documentation mark coded columns categorical (codebook.py)."""

    def test_redcap_codebook(self):  # label before code, one variable per block
        from crawlers.invenio.codebook import value_labels
        doc = ("Variable Name,Variable Label,Answer Label,Answer Code,Variable Type\n"
               "marital,Marital status,,,Numeric\n,,Single,1,\n,,Married,2,\n,,Widowed,3,\n"
               "hhadults,Adults in household,Open ended,,Numeric\n").encode()
        found = value_labels([("Codebook.csv", doc)], ["marital", "hhadults"])
        self.assertEqual(found["marital"]["labels"], {"1": "Single", "2": "Married", "3": "Widowed"})
        self.assertNotIn("hhadults", found)  # a count: no labels, stays numeric

    def test_label_column_codebook(self):  # "Column Label | Values | Value Labels" layout, range labels
        from crawlers.invenio.codebook import value_labels
        doc = ("Column,Description,Column Label,Values,Value Labels\n"
               "B,Age group,AgeCat,1,18-29\n,,,2,30-39\n,,,3,40+\n"
               "C,Gender,Gender,1,Female\n,,,2,Male\n").encode()
        found = value_labels([("codebook.csv", doc)], ["AgeCat", "Gender"])
        self.assertEqual(found["AgeCat"]["labels"], {"1": "18-29", "2": "30-39", "3": "40+"})
        self.assertEqual(found["Gender"]["labels"], {"1": "Female", "2": "Male"})

    def test_readme_text(self):
        from crawlers.invenio.codebook import value_labels
        doc = b"make_plan: How certain are you?\n  1 (very uncertain)\n  5 (very certain)\nage: years\n"
        found = value_labels([("README.txt", doc)], ["make_plan", "age"])
        self.assertEqual(set(found["make_plan"]["labels"]), {"1", "5"})
        self.assertNotIn("age", found)

    def test_documented_codes_become_categorical(self):
        from crawlers.invenio.codebook import apply_value_labels
        output = {"columns": [
            {"name": "gender", "structural_type": "http://schema.org/Integer", "semantic_types": ["http://schema.org/Boolean"]},
            {"name": "note", "structural_type": "http://schema.org/Text", "semantic_types": []}]}
        documented = {"gender": {"labels": {"0": "Male", "1": "Female", "2": "Other"}, "source": "cb.csv"},
                      "note": {"labels": {"1": "a", "2": "b"}, "source": "cb.csv"}}
        output["columns"].append({"name": "depth_index", "structural_type": "http://schema.org/Float", "semantic_types": []})
        documented["depth_index"] = {"labels": {"0": "no flows", "1": "frictionless"}, "source": "README.txt"}
        self.assertEqual(apply_value_labels(output, documented), 1)  # text and float columns untouched
        gender = output["columns"][0]
        self.assertEqual(gender["semantic_types"], ["http://schema.org/Enumeration"])  # 3 options: not yes/no
        self.assertEqual(gender["value_labels_text"], "Male; Female; Other")

    def test_local_file_uses_its_codebook(self):
        tmp = Path(tempfile.mkdtemp())
        pd.DataFrame({"marital": [1, 2, 3, 1, 2] * 10, "visits": [0, 1, 2, 3, 4] * 10}).to_csv(tmp / "data.csv", index=False)
        (tmp / "codebook.csv").write_text("Variable,Label,Code\nmarital,,\n,Single,1\n,Married,2\n,Widowed,3\n")
        record = profile_local_file(tmp / "data.csv", {"local_documentation": [str(tmp / "codebook.csv")]})
        columns = {c["name"]: c for c in record["profiler_metadata"]["columns"]}
        self.assertIn("http://schema.org/Enumeration", columns["marital"]["semantic_types"])
        self.assertNotIn("value_labels", columns["visits"])

    def test_discovery_lists_documentation(self):
        record = invenio_record({"data.csv": 10, "README.md": 10, "codebook.xlsx": 10, "paper.pdf": 10})
        docs = file_entries(record, DOMAIN, max_file_bytes=10**6)[0]["record"]["documentation"]
        self.assertEqual(sorted(d["key"] for d in docs), ["README.md", "codebook.xlsx"])
