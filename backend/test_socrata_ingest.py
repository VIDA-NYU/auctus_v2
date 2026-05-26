#!/usr/bin/env python3
"""Phase 1 Socrata ingestion spike for Auctus v2.

This script validates the ingestion path against the NYC Public Restrooms
Socrata dataset (4x4 identifier: i7jb-7jku) using an async workflow:

1. Fetch Socrata metadata with httpx.
2. Stream a safe CSV sample into a temporary file (up to 500 rows or 2 MiB).
3. Run the project's atlas-profiler library against the sample using Pandas.
4. Harmonize spatial boundaries (handling State Plane vs decimal degrees fallback).
5. Print a unified JSON block that matches the synthetic_datasets.json schema.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from atlas_profiler import process_dataset
from dateutil.parser import parse as parse_datetime
import httpx

LOGGER = logging.getLogger("test_socrata_ingest")

DATASET_ID = "i7jb-7jku"
SOCRATA_BASE_URL = "https://data.cityofnewyork.us"
METADATA_URL = f"{SOCRATA_BASE_URL}/api/views/{DATASET_ID}.json"
RAW_CSV_URL = f"{SOCRATA_BASE_URL}/api/views/{DATASET_ID}/rows.csv?accessType=DOWNLOAD"
MAX_SAMPLE_ROWS = 500
MAX_SAMPLE_BYTES = 2 * 1024 * 1024
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=15.0)
NYC_FALLBACK_BBOX = [-74.259, 40.477, -73.7, 40.917]

LAT_NAME_RE = re.compile(r"(^|[^a-z])(lat|latitude)([^a-z]|$)", re.IGNORECASE)
LON_NAME_RE = re.compile(r"(^|[^a-z])(lon|lng|long|longitude)([^a-z]|$)", re.IGNORECASE)
DATE_NAME_RE = re.compile(
    r"(^|[^a-z])(date|time|timestamp|created|updated|opened|closed|start|end)([^a-z]|$)",
    re.IGNORECASE,
)
WKT_POINT_RE = re.compile(
    r"POINT\s*\(\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)


@dataclass
class SampleStats:
    sample_path: Path
    bytes_written: int
    newline_count: int
    truncated_by_rows: bool
    truncated_by_bytes: bool


# ---------------------------------------------------------------------------
# Socrata metadata helpers
# ---------------------------------------------------------------------------

def _normalize_iso_date(value: Any) -> str | None:
    """Normalize a Socrata timestamp or ISO string into YYYY-MM-DD."""
    if value in (None, ""):
        return None

    try:
        if isinstance(value, (int, float)):
            seconds = float(value)
            if seconds > 10_000_000_000:
                seconds /= 1000.0
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
            return dt.date().isoformat()

        text = str(value).strip()
        if not text:
            return None

        dt = parse_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date().isoformat()
    except Exception:
        return None


async def fetch_socrata_metadata(client: httpx.AsyncClient, dataset_id: str) -> dict[str, Any]:
    """Fetch the Socrata view metadata payload and extract core catalog fields."""
    response = await client.get(METADATA_URL)
    response.raise_for_status()
    payload = response.json()

    title = payload.get("name") or payload.get("title") or payload.get("displayName") or dataset_id
    description = payload.get("description") or payload.get("notes") or payload.get("blurb") or ""
    publisher = payload.get("ownerDisplayName") or payload.get("attribution") or payload.get("ownerName") or ""
    
    download_url = (
        payload.get("downloadUrl")
        or payload.get("csvDownloadUrl")
        or RAW_CSV_URL
    )

    return {
        "id": dataset_id,
        "title": title,
        "description": description,
        "publisher": publisher,
        "source": "Socrata",
        "last_update_date": _normalize_iso_date(payload.get("rowsUpdatedAt") or payload.get("viewLastModified")),
        "download_url": download_url,
        "raw_csv_url": download_url,
        "socrata_metadata": payload,
    }


# ---------------------------------------------------------------------------
# Sample streaming helpers
# ---------------------------------------------------------------------------

def _readable_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


async def stream_csv_sample(
    client: httpx.AsyncClient,
    csv_url: str,
    max_rows: int = MAX_SAMPLE_ROWS,
    max_bytes: int = MAX_SAMPLE_BYTES,
) -> SampleStats:
    """Stream a bounded sample from the CSV endpoint into a temp file."""
    tmp = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".csv")
    bytes_written = 0
    newline_count = 0
    truncated_by_rows = False
    truncated_by_bytes = False

    try:
        async with client.stream("GET", csv_url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue

                remaining = max_bytes - bytes_written
                if remaining <= 0:
                    truncated_by_bytes = True
                    break

                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                    truncated_by_bytes = True

                tmp.write(chunk)
                bytes_written += len(chunk)
                newline_count += chunk.count(b"\n")

                if newline_count >= max_rows + 1:
                    truncated_by_rows = True
                    break
    finally:
        tmp.flush()
        tmp.close()

    return SampleStats(
        sample_path=Path(tmp.name),
        bytes_written=bytes_written,
        newline_count=newline_count,
        truncated_by_rows=truncated_by_rows,
        truncated_by_bytes=truncated_by_bytes,
    )


# ---------------------------------------------------------------------------
# Fallback parsing / feature inference
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(str(value).strip())
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except Exception:
        return None


def _safe_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        dt = parse_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _candidate_wkt_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, str):
        return None
    match = WKT_POINT_RE.search(value)
    if not match:
        return None
    lon = _safe_float(match.group(1))
    lat = _safe_float(match.group(2))
    if lon is None or lat is None:
        return None
    return lon, lat


def infer_spatial_bbox(headers: list[str], rows: list[dict[str, str]]) -> list[float] | None:
    """Build [minLon, minLat, maxLon, maxLat] from raw sample fallback tracking."""
    lat_columns = [h for h in headers if LAT_NAME_RE.search(h)]
    lon_columns = [h for h in headers if LON_NAME_RE.search(h)]

    lat_values: list[float] = []
    lon_values: list[float] = []

    for row in rows:
        for lat_col in lat_columns:
            lat = _safe_float(row.get(lat_col))
            if lat is not None and -90.0 <= lat <= 90.0:
                lat_values.append(lat)
        for lon_col in lon_columns:
            lon = _safe_float(row.get(lon_col))
            if lon is not None and -180.0 <= lon <= 180.0:
                lon_values.append(lon)

        for value in row.values():
            point = _candidate_wkt_point(value)
            if point and -90.0 <= point[1] <= 90.0 and -180.0 <= point[0] <= 180.0:
                lon_values.append(point[0])
                lat_values.append(point[1])

    if not lat_values or not lon_values:
        return None

    return [min(lon_values), min(lat_values), max(lon_values), max(lat_values)]


def infer_temporal_range(headers: list[str], rows: list[dict[str, str]]) -> tuple[str | None, str | None]:
    candidate_columns = [h for h in headers if DATE_NAME_RE.search(h)]
    values: list[datetime] = []

    for row in rows:
        for col in candidate_columns:
            dt = _safe_datetime(row.get(col))
            if dt is not None:
                values.append(dt)

    if not values:
        return None, None

    return min(values).date().isoformat(), max(values).date().isoformat()


def _valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    except Exception:
        return False
    return (
        -180.0 <= min_lon <= 180.0 and -90.0 <= min_lat <= 90.0 and
        -180.0 <= max_lon <= 180.0 and -90.0 <= max_lat <= 90.0 and
        min_lon < max_lon and min_lat < max_lat
    )


def _safe_bbox_from_profiler_or_sample(profiler_output: dict[str, Any], inferred_bbox: list[float] | None) -> list[float]:
    """Return an explicit WGS84 bounding box or fallback cleanly for State Plane numbers."""
    candidate = profiler_output.get("spatial_bbox") or profiler_output.get("bbox") or inferred_bbox
    if _valid_bbox(candidate):
        return [float(v) for v in candidate]

    return NYC_FALLBACK_BBOX.copy()


# ---------------------------------------------------------------------------
# Unified catalog output
# ---------------------------------------------------------------------------

def _merge_profiler_output(
    profiler_output: dict[str, Any],
    sample_stats: SampleStats,
    spatial_bbox: list[float] | None,
) -> dict[str, Any]:
    """Flatten atlas-profiler fields directly into profiler_metadata layout."""
    
    # Initialize our profile metadata directly using the profiler's root dict output
    meta = dict(profiler_output)

    # Inject the additional environment sample telemetry stats at the root of profiler_metadata
    meta["_sample_telemetry"] = {
        "path": str(sample_stats.sample_path),
        "bytes_loaded": sample_stats.bytes_written,
        "bytes_human": _readable_bytes(sample_stats.bytes_written),
        "newline_count": sample_stats.newline_count,
        "truncated_by_rows": sample_stats.truncated_by_rows,
        "truncated_by_bytes": sample_stats.truncated_by_bytes,
    }

    # Ensure high-level summary count defaults exist if the actual library did not specify them
    raw_columns = meta.get("columns", [])
    meta.setdefault("nb_profiled_rows", len(raw_columns))
    
    meta["nb_spatial_columns"] = len([
        c for c in raw_columns if isinstance(c, dict) and 
        ("GeoCoordinates" in str(c.get("structural_type")) or "AdministrativeArea" in str(c.get("semantic_types")))
    ])
    meta["nb_temporal_columns"] = len([
        c for c in raw_columns if isinstance(c, dict) and "DateTime" in str(c.get("structural_type"))
    ])
    
    meta.setdefault("nb_numerical_columns", profiler_output.get("nb_numerical_columns", 0))
    meta.setdefault("nb_categorical_columns", profiler_output.get("nb_categorical_columns", 0))
    
    # Compute safe validated internal spatial coordinates
    meta["spatial_bbox"] = _safe_bbox_from_profiler_or_sample(profiler_output, spatial_bbox)

    return meta


async def build_validation_record() -> dict[str, Any]:
    """Run the full workflow and return a single catalog-shaped JSON record."""
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        try:
            socrata = await fetch_socrata_metadata(client, DATASET_ID)
        except Exception as exc:
            LOGGER.exception("Metadata fetch failed")
            socrata = {
                "id": DATASET_ID,
                "title": DATASET_ID,
                "description": "",
                "publisher": "",
                "source": "Socrata",
                "last_update_date": None,
                "download_url": RAW_CSV_URL,
                "raw_csv_url": RAW_CSV_URL,
                "socrata_metadata": {},
            }
            errors.append(f"metadata_fetch_error: {exc}")

        try:
            sample_stats = await stream_csv_sample(client, socrata["raw_csv_url"])
        except Exception as exc:
            LOGGER.exception("CSV sampling failed")
            errors.append(f"sample_stream_error: {exc}")
            raise RuntimeError("Unable to continue without a sampled CSV") from exc

    # Load complete lines from file to check sample shapes safely
    headers: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        with open(sample_stats.sample_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        errors.append(f"fallback_csv_parse_error: {exc}")

    spatial_bbox = infer_spatial_bbox(headers, rows)
    temporal_start, temporal_end = infer_temporal_range(headers, rows)

    try:
        df = pd.read_csv(sample_stats.sample_path)
        profiler_output = process_dataset(
            df,
            geo_classifier=True,
            geo_classifier_threshold=0.5,
            coverage=True,
            plots=True,
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

    profiler_metadata = _merge_profiler_output(
        profiler_output,
        sample_stats,
        spatial_bbox,
    )

    record = {
        "id": socrata["id"],
        "title": socrata["title"],
        "description": socrata["description"],
        "source": socrata["source"],
        "download_url": socrata["download_url"],
        
        # Mirror types array layout flat on the root level as established in synthetic sample
        "types": profiler_metadata.get("types", []),
        
        "temporal_coverage": {
            "start": temporal_start,
            "end": temporal_end,
        },
        "spatial_coverage": {
            "label": "New York City",
            "bbox": {
                "type": "envelope",
                "coordinates": [
                    [profiler_metadata["spatial_bbox"][0], profiler_metadata["spatial_bbox"][3]],
                    [profiler_metadata["spatial_bbox"][2], profiler_metadata["spatial_bbox"][1]],
                ]
            },
        },
        "profiler_metadata": profiler_metadata,
        "errors": errors,
    }

    return record


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        record = await build_validation_record()
    except Exception as exc:
        record = {
            "id": DATASET_ID,
            "title": "NYC Public Restrooms",
            "source": "Socrata",
            "download_url": RAW_CSV_URL,
            "errors": [f"fatal_error: {exc}"],
        }

    print(json.dumps(record, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())