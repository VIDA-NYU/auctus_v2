"""InvenioRDM transformation and profiling engine for Auctus v2.

Turns one discovered file (see crawlers.invenio.crawler) into a catalog record
with the same shape the Socrata transformer emits, plus a `source_record` block
carrying the parent InvenioRDM record's metadata.

Differences from the Socrata path:
- Metadata comes from discovery, so there is no per-dataset metadata request.
- XLSX/XLS files are downloaded whole (spreadsheets can't be read partially) and
  converted to a CSV sample before profiling.
- CSVs are decoded leniently: repository uploads are not always UTF-8.
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
from pathlib import Path
from typing import Any

import pandas as pd
from atlas_profiler import process_dataset

from crawlers.coverage import admin_bbox, bbox_from_profiler, geo_data, temporal_from_profiler
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
    stream_csv_sample,
)

LOGGER = logging.getLogger("crawlers.invenio.transformer")

DEFAULT_MAX_SAMPLE_ROWS = 500
DEFAULT_MAX_SAMPLE_BYTES = 2 * 1024 * 1024
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


def spreadsheet_to_sample(source, extension: str, max_rows: int) -> SampleStats:
    """Write the first non-empty sheet's first rows as a CSV sample (path or bytes in).

    Workbooks often start with a title or notes sheet and keep the data on a later
    one (CaltechDATA), so empty sheets are skipped.
    """
    engine = "openpyxl" if extension == "xlsx" else "xlrd"
    workbook = pd.ExcelFile(io.BytesIO(source) if isinstance(source, bytes) else source, engine=engine)
    frame = pd.DataFrame()
    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet, nrows=max_rows)
        if not frame.empty or len(frame.columns):
            break
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8", newline="")
    with tmp:
        frame.to_csv(tmp, index=False)
    path = Path(tmp.name)
    return SampleStats(
        sample_path=path,
        bytes_written=path.stat().st_size,
        newline_count=len(frame) + 1,
        truncated_by_rows=len(frame) >= max_rows,
        truncated_by_bytes=False,
    )


async def fetch_spreadsheet_sample(client, file_url: str, extension: str, max_rows: int) -> SampleStats:
    """Download a spreadsheet and turn it into a CSV sample."""
    head = await client.head(file_url, headers={"Accept": "*/*"})
    size = int(head.headers.get("content-length") or 0)
    if size > MAX_SPREADSHEET_BYTES:
        raise ValueError(f"spreadsheet is {size} bytes, above the {MAX_SPREADSHEET_BYTES}-byte limit")
    response = await client.get(file_url, headers={"Accept": "*/*"})
    response.raise_for_status()
    return spreadsheet_to_sample(response.content, extension, max_rows)


def _sample_stream(source, max_rows: int, max_bytes: int) -> SampleStats:
    """Copy a binary stream into a temp CSV until the byte cap or max_rows+1 newlines.

    Mirrors stream_csv_sample, checking the limits after each 64 KiB chunk.
    """
    tmp = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".csv")
    written = newlines = 0
    truncated_by_rows = truncated_by_bytes = False
    with tmp:
        while chunk := source.read(64 * 1024):
            remaining = max_bytes - written
            if remaining <= 0:
                truncated_by_bytes = True
                break
            if len(chunk) > remaining:
                chunk, truncated_by_bytes = chunk[:remaining], True
            tmp.write(chunk)
            written += len(chunk)
            newlines += chunk.count(b"\n")
            if newlines >= max_rows + 1:
                truncated_by_rows = True
                break
    return SampleStats(sample_path=Path(tmp.name), bytes_written=written, newline_count=newlines,
                       truncated_by_rows=truncated_by_rows, truncated_by_bytes=truncated_by_bytes)


def sample_local_file(path: Path, extension: str, max_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
                      max_bytes: int = DEFAULT_MAX_SAMPLE_BYTES) -> SampleStats:
    """Sample a local file with the same limits as the HTTP path (tests, benchmarks)."""
    if extension in SPREADSHEET_EXTENSIONS:
        return spreadsheet_to_sample(Path(path), extension, max_rows)
    with open(path, "rb") as source:
        return _sample_stream(source, max_rows, max_bytes)


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


def read_sample(path: Path, extension: str, truncated: bool) -> tuple[list[str], list[dict[str, str]], pd.DataFrame]:
    """Parse a CSV/TSV sample leniently (bad bytes replaced, a cut-off last line dropped)."""
    separator = detect_separator(path, extension)
    with open(path, mode="r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=separator)
        headers = reader.fieldnames or []
        rows = list(reader)
    if truncated and rows:
        rows = rows[:-1]
    try:
        frame = pd.read_csv(path, sep=separator, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    except pd.errors.ParserError:
        # The byte cap can cut inside a quoted multi-line field, which pandas rejects
        # ("EOF inside string"); the csv module's rows up to the cut are still usable.
        frame = pd.DataFrame(rows, columns=headers)
    return headers, rows, frame


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
    """Profile one InvenioRDM file and return a catalog-shaped record."""
    max_rows = int(max_sample_rows or DEFAULT_MAX_SAMPLE_ROWS)
    max_bytes = int(max_sample_bytes or DEFAULT_MAX_SAMPLE_BYTES)
    extension = dataset_meta.get("file_extension") or ""
    file_url = dataset_meta["file_url"]

    if dataset_meta.get("zip_member"):
        async with PROFILER_SLOTS:
            return await asyncio.to_thread(profile_remote_member, dataset_meta, max_rows, max_bytes)

    async with async_client(http_timeout_seconds) as client:
        if extension in SPREADSHEET_EXTENSIONS:
            sample_stats = await fetch_spreadsheet_sample(client, file_url, extension, max_rows)
        else:
            client.headers["Accept"] = "*/*"
            sample_stats = await stream_csv_sample(client, file_url, max_rows=max_rows, max_bytes=max_bytes)

    # atlas-profiler is CPU-bound and synchronous; run it off the event loop so the
    # worker's other concurrent jobs keep streaming their downloads instead of timing out.
    async with PROFILER_SLOTS:
        headers, rows, profiler_output, errors = await asyncio.to_thread(profile_sample, sample_stats, extension)
    return assemble_record(dataset_meta, sample_stats, headers, rows, profiler_output, errors)


def profile_sample(sample_stats: SampleStats, extension: str) -> tuple[list[str], list[dict[str, str]], dict[str, Any], list[str]]:
    """Parse a sample and run atlas-profiler on it. Never raises: failures become a
    minimal profile plus an entry in the returned errors list."""
    headers: list[str] = []
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    try:
        truncated = sample_stats.truncated_by_rows or sample_stats.truncated_by_bytes
        headers, rows, frame = read_sample(sample_stats.sample_path, extension, truncated)
        total_columns = frame.shape[1]
        if total_columns > MAX_PROFILED_COLUMNS:
            frame = frame.iloc[:, :MAX_PROFILED_COLUMNS]
        profiler_output = process_dataset(
            frame,
            datamart_geo_data=geo_data(),
            geo_classifier=True,
            geo_classifier_threshold=0.5,
            coverage=True,
            plots=False,
            include_sample=True,
        )
        if not isinstance(profiler_output, dict):
            profiler_output = dict(profiler_output)
        if total_columns > MAX_PROFILED_COLUMNS:
            profiler_output["nb_columns"] = total_columns
            profiler_output["attribute_keywords"] = [str(h) for h in headers]
            profiler_output["columns_truncated"] = {"profiled": MAX_PROFILED_COLUMNS, "total": total_columns}
    except Exception as exc:
        LOGGER.exception("atlas-profiler execution failed")
        profiler_output = {
            "nb_rows": len(rows),
            "types": ["categorical"],
            "columns": [],
            "attribute_keywords": headers,
            "error": str(exc),
        }
        errors.append(f"profiler_error: {exc}")
    return headers, rows, profiler_output, errors


def assemble_record(dataset_meta: dict[str, Any], sample_stats: SampleStats, headers: list[str],
                    rows: list[dict[str, str]], profiler_output: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """Build the catalog record from a profiled sample."""
    source_record = dataset_meta.get("record") or {}
    # The profiler's own coverage first; column-name heuristics only as a fallback.
    temporal_start, temporal_end = temporal_from_profiler(profiler_output, rows)
    if not temporal_start:
        temporal_start, temporal_end = infer_temporal_range(headers, rows)
    inferred_bbox = infer_spatial_bbox(headers, rows)

    profiler_metadata = _merge_profiler_output(profiler_output, sample_stats, inferred_bbox, fallback_bbox=None)
    candidate_bbox = bbox_from_profiler(profiler_output) or admin_bbox(profiler_output, rows) or inferred_bbox
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


def sample_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, max_rows: int,
                      max_bytes: int) -> tuple[SampleStats, str]:
    """Sample one member: CSV/TSV streamed up to the limits, spreadsheets read whole."""
    ext = member_extension(info.filename)
    if ext in SPREADSHEET_EXTENSIONS:
        return spreadsheet_to_sample(archive.read(info), ext, max_rows), ext
    with archive.open(info) as source:
        return _sample_stream(source, max_rows, max_bytes), ext


def member_record(dataset_meta: dict[str, Any], archive_id: str, archive_key: str, info: zipfile.ZipInfo,
                  sample_stats: SampleStats, ext: str, totals: dict[str, int]) -> dict[str, Any]:
    """Profile a sampled member and build its catalog record, linked to its archive."""
    headers, rows, output, errors = profile_sample(sample_stats, ext)
    meta = {**dataset_meta, "dataset_id": member_dataset_id(archive_id, info.filename),
            "file_key": f"{archive_key}/{info.filename}", "file_extension": ext, "file_size": info.file_size}
    record = assemble_record(meta, sample_stats, headers, rows, output, errors)
    record["archive_dataset_id"] = archive_id
    record["source_record"]["archive"] = {"file_key": archive_key, "member": info.filename, **totals}
    sample_stats.sample_path.unlink(missing_ok=True)
    return record


def profile_archive(path: Path, dataset_meta: dict[str, Any], max_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
                    max_bytes: int = DEFAULT_MAX_SAMPLE_BYTES) -> list[dict[str, Any]]:
    """Profile the tabular members of a local zip; one catalog record per member."""
    archive_id = dataset_meta["dataset_id"]
    archive_key = dataset_meta.get("file_key", Path(path).name)
    with zipfile.ZipFile(path) as archive:
        members, total = select_archive_members(archive)
        totals = {"tabular_members": total, "profiled_members": len(members)}
        return [member_record(dataset_meta, archive_id, archive_key, info,
                              *sample_zip_member(archive, info, max_rows, max_bytes), totals)
                for info in members]


def profile_remote_member(dataset_meta: dict[str, Any], max_rows: int, max_bytes: int) -> dict[str, Any]:
    """Profile one member of a remote zip, reading only that member's bytes."""
    archive = dataset_meta["archive"]
    with open_remote_zip(archive["url"], archive["size"]) as zf:
        info = zf.getinfo(dataset_meta["zip_member"])
        sample_stats, ext = sample_zip_member(zf, info, max_rows, max_bytes)
    totals = {"tabular_members": archive["tabular_members"], "profiled_members": archive["profiled_members"]}
    return member_record(dataset_meta, archive["dataset_id"], archive["file_key"], info, sample_stats, ext, totals)


async def build_archive_records(dataset_meta: dict[str, Any], max_sample_rows: int | None = None,
                                max_sample_bytes: int | None = None,
                                http_timeout_seconds: float = 30.0) -> list[dict[str, Any]]:
    """Download a small zip (up to SMALL_ARCHIVE_BYTES) and profile its tabular members."""
    max_rows = int(max_sample_rows or DEFAULT_MAX_SAMPLE_ROWS)
    max_bytes = int(max_sample_bytes or DEFAULT_MAX_SAMPLE_BYTES)
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
            return await asyncio.to_thread(profile_archive, path, dataset_meta, max_rows, max_bytes)
    finally:
        path.unlink(missing_ok=True)


def profile_local_file(path: Path, dataset_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the whole transformer on a local file, without network (tests, benchmarks)."""
    path = Path(path)
    extension = path.name.rsplit(".", 1)[-1].lower() if "." in path.name else ""
    meta = {"dataset_id": f"local:{path.name}", "file_key": path.name, **(dataset_meta or {})}
    sample_stats = sample_local_file(path, extension)
    headers, rows, profiler_output, errors = profile_sample(sample_stats, extension)
    return assemble_record(meta, sample_stats, headers, rows, profiler_output, errors)


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
