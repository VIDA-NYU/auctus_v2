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

        from unittest import mock
        import crawlers.invenio.crawler as crawler
        jobs, report = self.run_discovery(record, fake_expand, limit=1000)
        group = [j for j in jobs if j.get("member_group")][0]["member_group"]
        self.assertEqual(len(group["samples"]), 150)  # small group: every part is read
        self.assertTrue(group["full_scan"])
        self.assertEqual(group["partitions"], {f"city_{i}": 30 for i in range(5)})  # one zip per city
        with mock.patch.object(crawler, "FULL_SCAN_MEMBERS", 100):
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
        # Which zip holds which city, with each city's own box and time range.
        partitions = {p["key"]: p for p in record["source_record"]["member_group"]["partitions"]}
        self.assertEqual(sorted(partitions), ["boston", "denver", "houston"])
        self.assertEqual(partitions["denver"]["parts"], 6)
        self.assertEqual(partitions["denver"]["rows_sampled"], 300)
        self.assertAlmostEqual(partitions["denver"]["bbox"][0], -105.0)
        self.assertAlmostEqual(partitions["boston"]["bbox"][2], -70.7)
        self.assertEqual(partitions["houston"]["start"], "2000-01-01")

    def test_partitions_by_folder_in_one_zip(self):  # uMORPH: oceania/R066_C162.parquet
        from crawlers.invenio.crawler import partition_labels
        archive = {"file_key": "tiles.zip"}
        members = [{"zip_member": n, "archive": archive} for n in
                   ("data/oceania/r1.parquet", "data/oceania/r2.parquet", "data/europe/r1.parquet")]
        self.assertEqual(partition_labels(members), ["oceania", "oceania", "europe"])
        flat = [{"zip_member": f"parts/part.{i}.parquet", "archive": archive} for i in range(3)]
        self.assertEqual(partition_labels(flat), [None, None, None])

    def test_every_part_read_splits_rare_schemas(self):
        """A part with other columns is its own dataset even when sampling would miss it."""
        from crawlers.invenio.crawler import archive_member_entries, group_job
        from crawlers.invenio.transformer import profile_member_group
        path = Path(self.tmp.name) / "tables.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for j in range(12):  # Table1..Table12: one has different columns
                frame = pd.DataFrame({"region": ["a", "b"], "value": [1, 2]}) if j != 5 else \
                    pd.DataFrame({"school": ["x", "y"], "enrolled": [10, 20], "year": [2020, 2021]})
                archive.writestr(f"Table{j + 1}.csv", frame.to_csv(index=False))
        entry = {"dataset_id": f"{DOMAIN}:t-rec:tables", "file_key": "tables.zip", "file_extension": "zip",
                 "file_size": path.stat().st_size, "file_url": str(path), "record": {"title": "Tables"}}
        with zipfile.ZipFile(path) as archive:
            members = archive_member_entries(entry, archive)
        sampled = group_job("table#.csv", members, max_samples=4, full_scan_members=0)
        self.assertNotIn("Table6.csv", {s["member"] for s in sampled["member_group"]["samples"]})
        records = profile_member_group(group_job("table#.csv", members), opener=lambda url, size: zipfile.ZipFile(url))
        self.assertEqual(len(records), 2)
        self.assertEqual(sorted(r["source_record"]["member_group"]["sampled_parts"] for r in records), [1, 11])
        self.assertTrue(all(r["source_record"]["member_group"]["all_parts_read"] for r in records))

    def test_group_profile_from_a_sample_estimates_rows(self):
        from crawlers.invenio.crawler import group_job
        from crawlers.invenio.transformer import profile_member_group
        members = self.make_tile_zips([("austin", -97.7), ("miami", -80.2)])
        job = group_job("part.#.parquet", members, max_samples=4, full_scan_members=0)
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
        record = self.profile("cp1252.csv", lambda p: frame.to_csv(p, index=False, encoding="cp1252"))
        self.assert_profiled(record)
        self.assertIn("café – résumé", record["sample"])  # decoded, not replaced with U+FFFD

    def test_legacy_encodings_are_told_apart(self):  # UltraViolet yz98y-nkk86: one record, three encodings
        from crawlers.invenio.transformer import guess_encoding
        text = "Municipios de la Cartografía Geoestadística, Querétaro, Michoacán de Ocampo\n" * 20
        for encoding in ("cp1252", "mac_roman", "cp850"):
            self.assertEqual(guess_encoding(text.encode(encoding)), encoding)
        self.assertEqual(guess_encoding(text.encode("utf-8")), "utf-8-sig")

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
        self.assertIn("Single; Married; Widowed", record["profiler_metadata"]["attribute_keywords"])  # searchable

    def test_discovery_lists_documentation(self):
        record = invenio_record({"data.csv": 10, "README.md": 10, "codebook.xlsx": 10, "paper.pdf": 10})
        docs = file_entries(record, DOMAIN, max_file_bytes=10**6)[0]["record"]["documentation"]
        self.assertEqual(sorted(d["key"] for d in docs), ["README.md", "codebook.xlsx"])


class LookupTests(unittest.TestCase):
    """Exact lookups: US ZIP codes to a box, identifier schemes by check digit."""

    def test_zip_codes_give_a_bounding_box(self):  # UltraViolet avp10-a8h86: zips had no box
        from crawlers.coverage import zip_bbox
        output = {"columns": [{"name": "Zip Code"}, {"name": "cases"}]}
        rows = [{"Zip Code": z, "cases": "10003"} for z in ("10003", "10011", "2139", "02139-4307")]
        bbox = zip_bbox(output, rows)
        self.assertTrue(-74.1 < bbox[0] < -73.9 and -71.2 < bbox[2] < -71.0)  # Manhattan to Cambridge, MA
        self.assertTrue(40.6 < bbox[1] < 40.8 and 42.3 < bbox[3] < 42.5)

    def test_zip_box_needs_a_zip_name_and_known_zips(self):
        from crawlers.coverage import zip_bbox
        self.assertIsNone(zip_bbox({"columns": [{"name": "count"}]}, [{"count": "10003"}] * 3))
        self.assertIsNone(zip_bbox({"columns": [{"name": "zip"}]}, [{"zip": v} for v in ("00000", "99999", "x")]))

    def test_identifier_schemes(self):
        from crawlers.identifiers import identifier_scheme
        self.assertEqual(identifier_scheme(["No ISBN", "0-306-40615-2", "978-3-16-148410-0", "0-19-852663-6"]), "ISBN")
        self.assertEqual(identifier_scheme(["0317-8471", "2049-3630", "1050-124X"]), "ISSN")
        self.assertEqual(identifier_scheme(["0000-0002-1825-0097", "0000-0001-5109-3700", "0000-0002-1694-233X"]), "ORCID")
        self.assertEqual(identifier_scheme(["10.1000/xyz", "doi:10.5281/zenodo.1", "https://doi.org/10.1038/nphys1170"]), "DOI")
        self.assertIsNone(identifier_scheme(["1", "2", "3", "4"]))
        self.assertIsNone(identifier_scheme(["0-306-40615-3", "978-3-16-148410-1", "0-19-852663-7"]))  # bad check digits

    def test_identifier_columns_are_searchable(self):
        tmp = Path(tempfile.mkdtemp())
        pd.DataFrame({"title": ["a", "b", "c"], "ISBN": ["0-306-40615-2", "978-3-16-148410-0", "0-19-852663-6"]}
                     ).to_csv(tmp / "books.csv", index=False)
        record = profile_local_file(tmp / "books.csv")
        columns = {c["name"]: c for c in record["profiler_metadata"]["columns"]}
        self.assertEqual(columns["ISBN"]["identifier_scheme"], "ISBN")
        self.assertIn("ISBN", record["profiler_metadata"]["attribute_keywords"])


class FormatTests(unittest.TestCase):
    """Shapefiles (in zips), GeoJSON and Stata (crawlers/invenio/formats.py)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def write_shapefile(self, stem: Path, epsg: int | None):
        import shapefile
        from pyproj import CRS, Transformer
        with shapefile.Writer(str(stem), shapeType=shapefile.POINT) as w:
            w.field("name", "C")
            w.field("capacity", "N")
            to_crs = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True) if epsg else None
            for name, lon, lat, cap in (("PS 1", -74.0, 40.71, 300), ("PS 2", -73.9, 40.85, 450), ("PS 3", -73.95, 40.6, 120)):
                x, y = to_crs.transform(lon, lat) if to_crs else (lon, lat)
                w.point(x, y)
                w.record(name, cap)
        if epsg:
            Path(str(stem) + ".prj").write_text(CRS.from_epsg(epsg).to_wkt())

    def test_projected_shapefile_in_a_zip(self):  # NYU SDR layers are in NY State Plane (EPSG:2263, feet)
        self.write_shapefile(self.tmp / "schools", 2263)
        path = self.tmp / "layer.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for ext in ("shp", "shx", "dbf", "prj"):
                archive.write(self.tmp / f"schools.{ext}", f"nyu_2451_1/schools.{ext}")
            archive.writestr("nyu_2451_1/schools_iso.xml", "<xml/>")
        records = profile_archive(path, {"dataset_id": "t:layer", "file_key": "layer.zip", "record": {"title": "NYC"}})
        self.assertEqual(len(records), 1)
        pm = records[0]["profiler_metadata"]
        self.assertEqual((pm["nb_rows"], pm["geometry_type"]), (3, "POINT"))
        self.assertEqual([c["name"] for c in pm["columns"]], ["name", "capacity"])
        west, south, east, north = pm["spatial_bbox"]
        self.assertAlmostEqual(west, -74.0, places=3)
        self.assertAlmostEqual(north, 40.85, places=3)

    def test_shapefile_without_dbf_is_skipped(self):
        path = self.tmp / "geom_only.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.shp", b"x")
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(select_archive_members(archive)[1], 0)

    def test_geojson(self):  # UltraViolet zamja-26h98
        import json
        features = [{"type": "Feature", "properties": {"city": "Austin", "jobs": 120},
                     "geometry": {"type": "Polygon", "coordinates": [[[-97.8, 30.2], [-97.6, 30.2], [-97.6, 30.4], [-97.8, 30.2]]]}},
                    {"type": "Feature", "properties": {"city": "Boston", "jobs": 90, "note": "x"},
                     "geometry": {"type": "Point", "coordinates": [-71.06, 42.36]}}]
        path = self.tmp / "centers.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        record = profile_local_file(path)
        pm = record["profiler_metadata"]
        self.assertEqual(pm["nb_rows"], 2)
        self.assertEqual([c["name"] for c in pm["columns"]], ["city", "jobs", "note"])
        self.assertEqual(pm["spatial_bbox"], [-97.8, 30.2, -71.06, 42.36])

    def test_stata_value_labels(self):  # UltraViolet d8z0p-brn64 ships a .dta next to its CSV
        frame = pd.DataFrame({"marital": [1, 2, 3, 1] * 5, "visits": [0, 1, 2, 3] * 5})
        frame.to_stata(self.tmp / "survey.dta", write_index=False,
                       value_labels={"marital": {1: "Single", 2: "Married", 3: "Widowed"}})
        record = profile_local_file(self.tmp / "survey.dta")
        columns = {c["name"]: c for c in record["profiler_metadata"]["columns"]}
        self.assertEqual(record["profiler_metadata"]["nb_rows"], 20)
        self.assertEqual(columns["marital"]["value_labels_text"], "Single; Married; Widowed")
        self.assertNotIn("value_labels", columns["visits"])
        frame.to_csv(self.tmp / "survey.csv", index=False)  # the .dta as the CSV's codebook
        record = profile_local_file(self.tmp / "survey.csv", {"local_documentation": [str(self.tmp / "survey.dta")]})
        columns = {c["name"]: c for c in record["profiler_metadata"]["columns"]}
        self.assertEqual(columns["marital"]["value_labels_source"], "survey.dta")

    def test_one_table_in_several_formats_is_indexed_once(self):
        record = invenio_record({"KS_data.csv": 10, "KS_data.xlsx": 10, "study.csv": 10, "study.dta": 10,
                                 "bias.dta": 10, "nyu_2451_1_Roads.zip": 10, "nyu_2451_1_WGS84_Roads.zip": 10})
        entries = file_entries(record, DOMAIN, max_file_bytes=10**6)
        self.assertEqual(sorted(e["file_key"] for e in entries),
                         ["KS_data.csv", "bias.dta", "nyu_2451_1_WGS84_Roads.zip", "study.csv"])
        docs = {d["key"] for d in entries[0]["record"]["documentation"]}
        self.assertTrue({"study.dta", "bias.dta"} <= docs)  # Stata files double as codebooks


class SpectrumTests(unittest.TestCase):
    """JCAMP-DX spectra (NYU conservation FTIR): one dataset per record, a row per spectrum."""

    NICOLET = (
        "##TITLE=21-044 Item 4 Red Balloon (Fragment) Vulcanized Rubber?\r\n##JCAMP-DX=5.01  $$ Nicolet v. 970\r\n"
        "##DATATYPE=INFRARED SPECTRUM\r\n##LONGDATE=2021/06/04\r\n##TIME=15:07:41\r\n"
        "##SPECTROMETER/DATA SYSTEM=Nicolet iS5\r\n##DATA PROCESSING=\r\nRatio against background\r\n"
        "%Transmittance->Absorbance\r\n##XUNITS=1/CM\r\n##FIRSTX=400.156891\r\n##NPOINTS=7468\r\n"
        "##XYDATA=(X++(Y..Y))\r\n400.157 11383855 11386644\r\n##TITLE=not a header field\r\n##END=\r\n")

    def test_header(self):
        from crawlers.invenio.formats import read_jcamp_header, read_spectra
        fields = read_jcamp_header(self.NICOLET)
        self.assertEqual(fields["JCAMP-DX"], "5.01")  # $$ comment dropped
        self.assertEqual(fields["DATA PROCESSING"], "Ratio against background; %Transmittance->Absorbance")
        self.assertEqual(fields["TITLE"], "21-044 Item 4 Red Balloon (Fragment) Vulcanized Rubber?")  # data block not read
        row = dict(zip(*[(r := read_spectra([("a.JDX", fields)])).header, list(r.rows)[0]]))
        self.assertEqual((row["file"], row["measured"], row["x_units"], row["points"]),
                         ("a.JDX", "2021-06-04T15:07:41", "1/CM", "7468"))

    def test_spectra_are_one_entry_per_record(self):
        entries = file_entries(invenio_record({"a.JDX": 10, "b.jdx": 10, "README.md": 10}), DOMAIN, max_file_bytes=10**6)
        self.assertEqual([(e["file_key"], len(e["spectra"])) for e in entries], [("2 spectra (JCAMP-DX)", 2)])

    def test_profiled_with_sample_names_searchable(self):
        from crawlers.invenio.formats import read_jcamp_header
        from crawlers.invenio.transformer import profile_spectra
        second = self.NICOLET.replace("Red Balloon (Fragment) Vulcanized Rubber?", "yellow beads").replace("06/04", "06/16")
        record = profile_spectra([("a.JDX", read_jcamp_header(self.NICOLET)), ("b.JDX", read_jcamp_header(second))],
                                 {"dataset_id": "uv:r:spectra", "file_key": "2 spectra (JCAMP-DX)"})
        self.assertEqual(record["profiler_metadata"]["nb_rows"], 2)
        self.assertEqual(record["temporal_coverage"]["start"][:10], "2021-06-04")
        self.assertEqual(record["temporal_coverage"]["end"][:10], "2021-06-16")
        self.assertIn("21-044 Item 4 yellow beads", record["profiler_metadata"]["attribute_keywords"])
        reference = self.NICOLET.replace("2021/06/04", "2017/02/28")
        record = profile_spectra([("a.JDX", read_jcamp_header(self.NICOLET)),
                                  ("a_LibraryMatch.JDX", read_jcamp_header(reference))],
                                 {"dataset_id": "uv:r:spectra", "file_key": "2 spectra (JCAMP-DX)"})
        self.assertEqual(record["temporal_coverage"]["start"][:10], "2021-06-04")  # the library spectrum's 2017 date is left out


class DatabaseTests(unittest.TestCase):
    """SQLite / SpatiaLite / GeoPackage: one dataset per table (formats.py, profile_database)."""

    @staticmethod
    def spatialite_point(lon: float, lat: float, srid: int = 4326) -> bytes:
        import struct
        mbr = struct.pack("<4d", lon, lat, lon, lat)
        return bytes([0, 1]) + struct.pack("<i", srid) + mbr + bytes([0x7C]) + struct.pack("<i2d", 1, lon, lat) + bytes([0xFE])

    def make_db(self, path: Path):
        import sqlite3
        con = sqlite3.connect(path)
        con.execute("create table geometry_columns (f_table_name text, f_geometry_column text, srid int)")
        con.execute("insert into geometry_columns values ('stations', 'geometry', 4326)")
        con.execute("create table stations (stop_id text, name text, geometry blob)")
        con.executemany("insert into stations values (?, ?, ?)",
                        [(f"s{i}", f"Station {i}", self.spatialite_point(-74.0 + i / 10, 40.6 + i / 20)) for i in range(5)])
        con.execute("create table tract_census (geoid text, pop int, hshd int)")
        con.executemany("insert into tract_census values (?, ?, ?)", [(f"36061{i:06d}", 1000 + i, 400 + i) for i in range(30)])
        con.execute("create table spatial_ref_sys (srid int, auth_name text)")
        con.execute("create table idx_stations_geometry (pkid int)")
        con.execute("create view big_tracts as select geoid, pop from tract_census where pop > 1010")
        con.commit()
        con.close()

    def test_tables_views_and_extent(self):
        from crawlers.invenio.formats import database_tables, read_database_table
        path = Path(tempfile.mkdtemp()) / "gdb.sqlite"
        self.make_db(path)
        self.assertEqual(database_tables(path), ["big_tracts", "stations", "tract_census"])  # system tables skipped
        stations = read_database_table(path, "stations")
        self.assertEqual(stations.header, ["stop_id", "name"])  # geometry blob not profiled
        self.assertEqual(len(list(stations.rows)), 5)
        for got, want in zip(stations.bbox, [-74.0, 40.6, -73.6, 40.8]):
            self.assertAlmostEqual(got, want)
        self.assertIsNone(read_database_table(path, "tract_census").bbox)

    def test_database_in_a_zip_gives_one_dataset_per_table(self):
        tmp = Path(tempfile.mkdtemp())
        self.make_db(tmp / "gdb.sqlite")
        with zipfile.ZipFile(tmp / "gdb.zip", "w") as archive:
            archive.write(tmp / "gdb.sqlite", "nyc/gdb.sqlite")
        records = profile_archive(tmp / "gdb.zip", {"dataset_id": "t:gdb", "file_key": "gdb.zip", "record": {"title": "NYC"}})
        by_table = {r["source_record"]["database"]["table"]: r for r in records}
        self.assertEqual(sorted(by_table), ["big_tracts", "stations", "tract_census"])
        self.assertEqual(by_table["tract_census"]["profiler_metadata"]["nb_rows"], 30)
        self.assertEqual(by_table["stations"]["source_record"]["file_key"], "gdb.zip/nyc/gdb.sqlite#stations")
        self.assertEqual({r["archive_dataset_id"] for r in records}, {"t:gdb"})
        self.assertEqual(len({r["id"] for r in records}), 3)
        self.assertIsNotNone(by_table["stations"]["profiler_metadata"]["spatial_bbox"])
