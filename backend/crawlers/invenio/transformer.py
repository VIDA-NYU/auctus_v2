"""InvenioRDM transformation and profiling engine for Auctus v2.

Turns one discovered file (see crawlers.invenio.crawler) into a catalog record
with the same shape the Socrata transformer emits, plus a `source_record` block
carrying the parent InvenioRDM record's metadata.

Differences from the Socrata path:
- Metadata comes from discovery, so there is no per-dataset metadata request.
- Whole files are profiled: each file is downloaded, normalized once to UTF-8
  comma-separated CSV (any delimiter, encoding, BOM, spreadsheet or zip member),
  and passed to atlas-profiler as a path, which streams every row.
- No fallback bounding box. Repository data is global, so a dataset without
  coordinates gets no spatial coverage instead of a portal-wide default.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from atlas_profiler import process_dataset

from crawlers.coverage import admin_bbox, bbox_from_profiler, geo_data, temporal_from_profiler, temporal_from_year_columns
from crawlers.invenio.archives import (
    SMALL_ARCHIVE_BYTES,
    member_dataset_id,
    member_extension,
    open_remote_zip,
    select_archive_members,
)
from crawlers.invenio.http import async_client
from crawlers.socrata.transformer import (
    SampleStats,
    _merge_profiler_output,
    _valid_bbox,
    infer_spatial_bbox,
    infer_temporal_range,
)

LOGGER = logging.getLogger("crawlers.invenio.transformer")
csv.field_size_limit(2**31 - 1)  # research CSVs can hold very large text cells

# Whole files are profiled. The first HEAD_ROWS rows are also kept for the record's
# preview sample and for the column-name fallback heuristics.
HEAD_ROWS = 500
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(200 * 1024 * 1024)))
SPREADSHEET_EXTENSIONS = {"xlsx", "xls"}
# Spreadsheets are loaded into memory whole, and pandas inflates them several-fold.
MAX_SPREADSHEET_BYTES = 25 * 1024 * 1024


# Downloads run concurrently across worker jobs, but each profiler run holds the
# sample, the profiler's models and intermediate frames in memory. Unbounded, the
# worker's 10 concurrent jobs get it OOM-killed inside Docker's default memory.
PROFILER_SLOTS = asyncio.Semaphore(int(os.getenv("PROFILER_CONCURRENCY", "2")))

# Research tables can be extremely wide (CaltechDATA has CSVs with 38k columns, e.g.
# gene-by-sample matrices). Per-column profiling of those exhausts memory, so only the
# first columns are profiled; every column name stays searchable via attribute_keywords.
MAX_PROFILED_COLUMNS = int(os.getenv("MAX_PROFILED_COLUMNS", "300"))


@dataclass
class Table:
    """A file normalized for profiling: UTF-8, comma-separated, at most MAX_PROFILED_COLUMNS.

    atlas-profiler streams a CSV path over the whole file (row count, per-column
    stats, exact date and lat/long ranges), but only recognizes `,` or `.tsv` tab
    separators and plain UTF-8. Normalizing once lets every format (semicolons,
    Windows-1252, BOMs, spreadsheets, zip members) take that whole-file path.
    """

    path: Path
    headers: list[str]           # every original column name
    head_rows: list[dict[str, str]]  # first HEAD_ROWS rows, profiled columns only
    total_columns: int
    rows: int
    bytes: int

    def stats(self) -> SampleStats:
        return SampleStats(sample_path=self.path, bytes_written=self.bytes, newline_count=self.rows,
                           truncated_by_rows=False, truncated_by_bytes=False)


def _write_table(header: list[str], records, out: Path) -> Table:
    """Write rows to `out` as normalized CSV, keeping the first MAX_PROFILED_COLUMNS columns."""
    kept = [str(h) for h in header[:MAX_PROFILED_COLUMNS]]
    width, head_rows, count = len(kept), [], 0
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(kept)
        for row in records:
            values = [("" if v is None else str(v)) for v in row[:width]]
            if not any(v.strip() for v in values):
                continue  # blank lines
            values += [""] * (width - len(values))
            writer.writerow(values)
            if len(head_rows) < HEAD_ROWS:
                head_rows.append(dict(zip(kept, values)))
            count += 1
    return Table(path=out, headers=[str(h) for h in header], head_rows=head_rows, total_columns=len(header),
                 rows=count, bytes=out.stat().st_size)


def normalize_table(raw: Path, extension: str) -> Table:
    """Normalize a whole delimited file or the first non-empty sheet of a workbook."""
    out = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name)
    if extension in SPREADSHEET_EXTENSIONS:
        # Workbooks often start with a title or notes sheet and keep the data on a
        # later one (CaltechDATA), so empty sheets are skipped.
        engine = "openpyxl" if extension == "xlsx" else "xlrd"
        workbook = pd.ExcelFile(raw, engine=engine)
        frame = pd.DataFrame()
        for sheet in workbook.sheet_names:
            frame = workbook.parse(sheet, dtype=str).fillna("")
            if not frame.empty or len(frame.columns):
                break
        return _write_table(list(frame.columns), frame.itertuples(index=False, name=None), out)
    separator = detect_separator(raw, extension)
    # utf-8-sig drops a BOM; undecodable bytes (Windows-1252 uploads) are replaced.
    with open(raw, encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=separator)
        header = next(reader, [])
        return _write_table(header, reader, out)


def copy_capped(source, out: Path, limit: int) -> Path:
    """Copy a binary stream to `out`, refusing files larger than `limit` bytes."""
    written = 0
    with open(out, "wb") as fh:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                raise ValueError(f"file is larger than the {limit}-byte limit")
            fh.write(chunk)
    return out


def load_local_table(path: Path, extension: str) -> Table:
    """Normalize a local file for profiling (tests, benchmarks)."""
    return normalize_table(Path(path), extension)


def detect_separator(path: Path, extension: str) -> str:
    """Sniff the delimiter: repository CSVs are not always comma-separated (e.g. `;`
    from European locales). Falls back to the extension's default."""
    default = "\t" if extension == "tsv" else ","
    with open(path, mode="r", encoding="utf-8", errors="replace", newline="") as fh:
        head = fh.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return default


def build_sample_csv(headers: list[str], rows: list[dict[str, str]], limit: int = 20) -> str:
    """Header + first rows, for AutoDDG (stored in MinIO, stripped from the search index)."""
    if not headers:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows[:limit]:
        writer.writerow(row)
    return buffer.getvalue()


async def build_invenio_record(
    dataset_meta: dict[str, Any],
    max_sample_rows: int | None = None,
    max_sample_bytes: int | None = None,
    http_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Download one InvenioRDM file, profile all of it, and return a catalog record.

    max_sample_rows / max_sample_bytes are accepted for the worker's call signature
    but no longer limit profiling: whole files are profiled (2026-10-13 decision).
    """
    if dataset_meta.get("zip_member"):
        async with PROFILER_SLOTS:
            return await asyncio.to_thread(profile_remote_member, dataset_meta)

    extension = dataset_meta.get("file_extension") or ""
    limit = MAX_SPREADSHEET_BYTES if extension in SPREADSHEET_EXTENSIONS else MAX_FILE_BYTES
    raw = Path(tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}").name)
    try:
        async with async_client(max(http_timeout_seconds, 120.0)) as client:
            async with client.stream("GET", dataset_meta["file_url"], headers={"Accept": "*/*"}) as response:
                response.raise_for_status()
                written = 0
                with open(raw, "wb") as fh:
                    async for chunk in response.aiter_bytes():
                        written += len(chunk)
                        if written > limit:
                            raise ValueError(f"file is larger than the {limit}-byte limit")
                        fh.write(chunk)
        # Normalizing and profiling are CPU-bound and synchronous; run them off the event
        # loop so the worker's other jobs keep downloading instead of timing out.
        async with PROFILER_SLOTS:
            return await asyncio.to_thread(profile_raw_file, raw, extension, dataset_meta)
    finally:
        raw.unlink(missing_ok=True)


def profile_raw_file(raw: Path, extension: str, dataset_meta: dict[str, Any]) -> dict[str, Any]:
    table = normalize_table(raw, extension)
    try:
        headers, rows, output, errors = profile_table(table)
        return assemble_record(dataset_meta, table, headers, rows, output, errors)
    finally:
        table.path.unlink(missing_ok=True)


def profile_table(table: Table) -> tuple[list[str], list[dict[str, str]], dict[str, Any], list[str]]:
    """Run atlas-profiler over the whole normalized file. Never raises.

    The profiler's CSV-path mode streams every row; if it fails (e.g. a malformed
    quote polars can't parse), the file is profiled in memory as a DataFrame instead,
    and only if that fails too does the record get a minimal profile plus an error.
    """
    headers, rows, errors = table.headers, table.head_rows, []
    options = dict(datamart_geo_data=geo_data(), geo_classifier=True, geo_classifier_threshold=0.5,
                   coverage=True, plots=False, include_sample=True)
    mode = "full-file"
    try:
        try:
            profiler_output = process_dataset(str(table.path), **options)
        except Exception as exc:
            LOGGER.warning("Full-file profiling failed (%r); profiling in memory instead", exc)
            mode = "in-memory"
            frame = pd.read_csv(table.path, dtype=str, keep_default_na=False, encoding_errors="replace",
                                on_bad_lines="skip")
            profiler_output = process_dataset(frame, **options)
        if not isinstance(profiler_output, dict):
            profiler_output = dict(profiler_output)
        profiler_output["profiling_mode"] = mode
        if table.total_columns > MAX_PROFILED_COLUMNS:
            profiler_output["nb_columns"] = table.total_columns
            profiler_output["attribute_keywords"] = headers
            profiler_output["columns_truncated"] = {"profiled": MAX_PROFILED_COLUMNS, "total": table.total_columns}
    except Exception as exc:
        LOGGER.exception("atlas-profiler execution failed")
        profiler_output = {
            "nb_rows": table.rows,
            "types": ["categorical"],
            "columns": [],
            "attribute_keywords": headers,
            "error": str(exc),
        }
        errors.append(f"profiler_error: {exc}")
    return headers, rows, profiler_output, errors


def assemble_record(dataset_meta: dict[str, Any], table: Table, headers: list[str],
                    rows: list[dict[str, str]], profiler_output: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """Build the catalog record from a profiled table.

    Coverage comes from the whole file: the profiler's own ranges (exact over every
    row in full-file mode), then year columns and place names read over the whole
    column, and the column-name heuristics on the first rows only as a last resort.
    """
    source_record = dataset_meta.get("record") or {}
    full_file = profiler_output.get("profiling_mode") == "full-file"
    temporal_start, temporal_end = temporal_from_profiler(profiler_output, None if full_file else rows)
    if not temporal_start:
        temporal_start, temporal_end = temporal_from_year_columns(headers, rows, table.path)
    if not temporal_start:
        temporal_start, temporal_end = infer_temporal_range(headers, rows)
    inferred_bbox = infer_spatial_bbox(headers, rows)

    profiler_metadata = _merge_profiler_output(profiler_output, table.stats(), inferred_bbox, fallback_bbox=None)
    candidate_bbox = (bbox_from_profiler(profiler_output) or admin_bbox(profiler_output, rows, table.path)
                      or inferred_bbox)
    real_bbox = [float(v) for v in candidate_bbox] if _valid_bbox(candidate_bbox) else None
    # _merge_profiler_output always fills a bbox (worst case the whole world); keep only a real one.
    profiler_metadata["spatial_bbox"] = real_bbox

    file_key = dataset_meta.get("file_key", "")
    record = {
        "id": dataset_meta["dataset_id"],
        "title": f"{source_record.get('title', '')}: {file_key}".strip(": "),
        "description": source_record.get("description", ""),
        "source": "InvenioRDM",
        "publisher": source_record.get("publisher", ""),
        "download_url": dataset_meta.get("file_url"),
        "landing_url": source_record.get("landing_url"),
        "source_record": {**source_record, "file_key": file_key, "file_size": dataset_meta.get("file_size")},
        "sample": build_sample_csv(headers, rows),
        "types": profiler_metadata.get("types", []),
        "temporal_coverage": {"start": temporal_start, "end": temporal_end},
        "profiler_metadata": profiler_metadata,
        "errors": errors,
    }
    if real_bbox:
        record["spatial_coverage"] = {
            "label": "",
            "bbox": {
                "type": "envelope",
                "coordinates": [[real_bbox[0], real_bbox[3]], [real_bbox[2], real_bbox[1]]],
            },
        }
    return record


def member_table(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[Table, str]:
    """Extract one member (whole, within the size caps) and normalize it."""
    ext = member_extension(info.filename)
    limit = MAX_SPREADSHEET_BYTES if ext in SPREADSHEET_EXTENSIONS else MAX_FILE_BYTES
    if info.file_size > limit:
        raise ValueError(f"member is larger than the {limit}-byte limit")
    raw = Path(tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}").name)
    try:
        with archive.open(info) as source:
            copy_capped(source, raw, limit)
        return normalize_table(raw, ext), ext
    finally:
        raw.unlink(missing_ok=True)


def member_record(dataset_meta: dict[str, Any], archive_id: str, archive_key: str, info: zipfile.ZipInfo,
                  table: Table, ext: str, totals: dict[str, int]) -> dict[str, Any]:
    """Profile a member and build its catalog record, linked to its archive."""
    try:
        headers, rows, output, errors = profile_table(table)
        meta = {**dataset_meta, "dataset_id": member_dataset_id(archive_id, info.filename),
                "file_key": f"{archive_key}/{info.filename}", "file_extension": ext, "file_size": info.file_size}
        record = assemble_record(meta, table, headers, rows, output, errors)
    finally:
        table.path.unlink(missing_ok=True)
    record["archive_dataset_id"] = archive_id
    record["source_record"]["archive"] = {"file_key": archive_key, "member": info.filename, **totals}
    return record


def profile_archive(path: Path, dataset_meta: dict[str, Any], *_unused) -> list[dict[str, Any]]:
    """Profile the tabular members of a local zip; one catalog record per member."""
    archive_id = dataset_meta["dataset_id"]
    archive_key = dataset_meta.get("file_key", Path(path).name)
    records = []
    with zipfile.ZipFile(path) as archive:
        members, total = select_archive_members(archive)
        totals = {"tabular_members": total, "profiled_members": len(members)}
        for info in members:
            try:
                records.append(member_record(dataset_meta, archive_id, archive_key, info,
                                             *member_table(archive, info), totals))
            except Exception as exc:
                LOGGER.warning("Skipping archive member %s: %r", info.filename, exc)
    return records


def profile_remote_member(dataset_meta: dict[str, Any], *_unused) -> dict[str, Any]:
    """Profile one member of a remote zip, reading only that member's bytes."""
    archive = dataset_meta["archive"]
    with open_remote_zip(archive["url"], archive["size"]) as zf:
        info = zf.getinfo(dataset_meta["zip_member"])
        table, ext = member_table(zf, info)
    totals = {"tabular_members": archive["tabular_members"], "profiled_members": archive["profiled_members"]}
    return member_record(dataset_meta, archive["dataset_id"], archive["file_key"], info, table, ext, totals)


async def build_archive_records(dataset_meta: dict[str, Any], max_sample_rows: int | None = None,
                                max_sample_bytes: int | None = None,
                                http_timeout_seconds: float = 30.0) -> list[dict[str, Any]]:
    """Download a small zip (up to SMALL_ARCHIVE_BYTES) and profile its tabular members."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    path = Path(tmp.name)
    try:
        with tmp:
            async with async_client(max(http_timeout_seconds, 120.0)) as client:
                async with client.stream("GET", dataset_meta["file_url"], headers={"Accept": "*/*"}) as response:
                    response.raise_for_status()
                    downloaded = 0
                    async for chunk in response.aiter_bytes():
                        downloaded += len(chunk)
                        if downloaded > SMALL_ARCHIVE_BYTES:
                            raise ValueError(f"archive is larger than the {SMALL_ARCHIVE_BYTES}-byte download limit")
                        tmp.write(chunk)
        async with PROFILER_SLOTS:
            return await asyncio.to_thread(profile_archive, path, dataset_meta)
    finally:
        path.unlink(missing_ok=True)


def profile_local_file(path: Path, dataset_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the whole transformer on a local file, without network (tests, benchmarks)."""
    path = Path(path)
    extension = path.name.rsplit(".", 1)[-1].lower() if "." in path.name else ""
    meta = {"dataset_id": f"local:{path.name}", "file_key": path.name, **(dataset_meta or {})}
    table = load_local_table(path, extension)
    try:
        headers, rows, profiler_output, errors = profile_table(table)
        return assemble_record(meta, table, headers, rows, profiler_output, errors)
    finally:
        table.path.unlink(missing_ok=True)


async def main() -> None:
    """Usage: python -m crawlers.invenio.transformer <domain> [record_id]"""
    from crawlers.invenio.crawler import discover_invenio_datasets

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    domain = sys.argv[1] if len(sys.argv) > 1 else "ultraviolet.library.nyu.edu"
    query = f"id:{sys.argv[2]}" if len(sys.argv) > 2 else ""
    entries = await discover_invenio_datasets(domain, limit=1, query=query)
    if not entries:
        raise SystemExit("no tabular files found")
    result = await build_invenio_record(entries[0])
    result.pop("sample", None)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:4000])


if __name__ == "__main__":
    asyncio.run(main())
