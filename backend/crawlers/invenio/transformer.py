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
from pathlib import Path
from typing import Any

import pandas as pd
from atlas_profiler import process_dataset

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


async def fetch_spreadsheet_sample(client, file_url: str, extension: str, max_rows: int) -> SampleStats:
    """Download a spreadsheet and write its first sheet's first rows as a CSV sample."""
    head = await client.head(file_url, headers={"Accept": "*/*"})
    size = int(head.headers.get("content-length") or 0)
    if size > MAX_SPREADSHEET_BYTES:
        raise ValueError(f"spreadsheet is {size} bytes, above the {MAX_SPREADSHEET_BYTES}-byte limit")
    response = await client.get(file_url, headers={"Accept": "*/*"})
    response.raise_for_status()
    engine = "openpyxl" if extension == "xlsx" else None
    frame = pd.read_excel(io.BytesIO(response.content), nrows=max_rows, engine=engine)
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


def read_sample(path: Path, extension: str, truncated: bool) -> tuple[list[str], list[dict[str, str]], pd.DataFrame]:
    """Parse a CSV/TSV sample leniently (bad bytes replaced, a cut-off last line dropped)."""
    separator = "\t" if extension == "tsv" else ","
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
    source_record = dataset_meta.get("record") or {}
    errors: list[str] = []

    async with async_client(http_timeout_seconds) as client:
        if extension in SPREADSHEET_EXTENSIONS:
            sample_stats = await fetch_spreadsheet_sample(client, file_url, extension, max_rows)
        else:
            client.headers["Accept"] = "*/*"
            sample_stats = await stream_csv_sample(client, file_url, max_rows=max_rows, max_bytes=max_bytes)

    headers: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        truncated = sample_stats.truncated_by_rows or sample_stats.truncated_by_bytes
        headers, rows, frame = read_sample(sample_stats.sample_path, extension, truncated)
        # atlas-profiler is CPU-bound and synchronous; run it off the event loop so the
        # worker's other concurrent jobs keep streaming their downloads instead of timing out.
        async with PROFILER_SLOTS:
            profiler_output = await asyncio.to_thread(
                process_dataset,
                frame,
                geo_classifier=True,
                geo_classifier_threshold=0.5,
                coverage=True,
                plots=False,
                include_sample=True,
            )
        if not isinstance(profiler_output, dict):
            profiler_output = dict(profiler_output)
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

    inferred_bbox = infer_spatial_bbox(headers, rows)
    temporal_start, temporal_end = infer_temporal_range(headers, rows)

    profiler_metadata = _merge_profiler_output(profiler_output, sample_stats, inferred_bbox, fallback_bbox=None)
    candidate_bbox = profiler_output.get("spatial_bbox") or profiler_output.get("bbox") or inferred_bbox
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
        "download_url": file_url,
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
