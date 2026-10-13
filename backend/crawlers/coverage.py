"""Turn atlas-profiler's coverage output into catalog coverage fields.

atlas-profiler already computes temporal coverage (ranges of epoch seconds per
date column, at the column's resolution) and spatial coverage (envelopes from
lat/long columns, WKT points and, with datamart-geo, administrative-area names).
Transformers should use these first; their own column-name heuristics are only a
fallback for when the profiler finds nothing.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from dateutil.parser import parse as parse_datetime

LOGGER = logging.getLogger(__name__)

# Dates outside this window are treated as parsing noise (e.g. IDs read as epochs).
MIN_YEAR, MAX_YEAR = 1000, 2200
# Share of a place column's distinct values that must resolve to an admin area.
MIN_RESOLVED_SHARE = 0.5

_geo_data: Any = None
_geo_data_loaded = False


def geo_data():
    """The datamart-geo admin-area database, loaded once per process, or None.

    Lets the profiler resolve place-name columns (countries, states, cities) to
    bounding boxes offline. Needs the data downloaded once (see docs/installation.md);
    set DATAMART_GEO=0 to disable.
    """
    global _geo_data, _geo_data_loaded
    if _geo_data_loaded:
        return _geo_data
    _geo_data_loaded = True
    if os.getenv("DATAMART_GEO", "1") == "0":
        return None
    try:
        from datamart_geo import GeoData

        _geo_data = GeoData.from_local_cache()
        LOGGER.info("datamart-geo loaded: place names will be resolved to bounding boxes")
    except Exception as exc:
        LOGGER.warning("datamart-geo unavailable (%s); place-name columns get no spatial coverage", exc)
    return _geo_data


def _to_date(seconds: float) -> str | None:
    try:
        dt = datetime.fromtimestamp(float(seconds), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.date().isoformat() if MIN_YEAR <= dt.year <= MAX_YEAR else None


_YEAR_RE = re.compile(r"(1[0-9]|20|21)\d\d")
_BARE_NUMBER_RE = re.compile(r"[-+]?\d+(\.\d+)?")
_DATE_DEFAULT = datetime(2000, 1, 1)  # dateutil otherwise fills missing fields with today's date


def _parse_bound(value: Any, end: bool) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    if _YEAR_RE.fullmatch(text):  # a bare year spans the whole year
        return datetime(int(text), 12, 31) if end else datetime(int(text), 1, 1)
    if _BARE_NUMBER_RE.fullmatch(text):
        return None
    try:
        dt = parse_datetime(text, default=_DATE_DEFAULT)
    except (ValueError, OverflowError):
        return None
    return dt if MIN_YEAR <= dt.year <= MAX_YEAR else None


def temporal_from_profiler(profiler_output: dict[str, Any],
                           rows: list[dict[str, str]] | None = None) -> tuple[str | None, str | None]:
    """(start, end) as YYYY-MM-DD for the columns the profiler identified as temporal.

    The profiler is good at deciding which columns are dates, but its ranges are
    binned summaries that can miss the first and last days. With the sample rows
    available, the exact min/max of those columns is used instead.
    """
    items = profiler_output.get("temporal_coverage") or []
    if rows:
        starts, ends = [], []
        for item in items:
            for column in item.get("column_names") or []:
                for row in rows:
                    value = row.get(column)
                    if value not in (None, ""):
                        lo, hi = _parse_bound(value, end=False), _parse_bound(value, end=True)
                        if lo and hi:
                            starts.append(lo)
                            ends.append(hi)
        if starts:
            return min(starts).date().isoformat(), max(ends).date().isoformat()
    lows, highs = [], []
    for item in items:
        for entry in item.get("ranges") or []:
            bounds = entry.get("range") or {}
            if "gte" in bounds and "lte" in bounds:
                lows.append(bounds["gte"])
                highs.append(bounds["lte"])
    if not lows:
        return None, None
    return _to_date(min(lows)), _to_date(max(highs))


_YEAR_COLUMN_RE = re.compile(r"(^|[^a-z])(year|yr)s?([^a-z]|$)", re.IGNORECASE)


def temporal_from_year_columns(headers: list[str], rows: list[dict[str, str]]) -> tuple[str | None, str | None]:
    """(start, end) from columns named like "year" whose values are mostly 4-digit years.

    The profiler doesn't always type plain integer year columns as temporal
    (seen on UltraViolet: a `Year` column of 2000, 2001, ...), and these are common
    in research tables.
    """
    years = []
    for header in headers:
        if not _YEAR_COLUMN_RE.search(str(header)):
            continue
        values = [str(r.get(header, "")).strip() for r in rows]
        values = [v[:-2] if v.endswith(".0") else v for v in values if v]
        found = [int(v) for v in values if _YEAR_RE.fullmatch(v)]
        if values and len(found) >= 0.8 * len(values):
            years.extend(found)
    if not years:
        return None, None
    return f"{min(years)}-01-01", f"{max(years)}-12-31"


def admin_bbox(profiler_output: dict[str, Any], rows: list[dict[str, str]]) -> list[float] | None:
    """Bounding box of place-name columns, resolved with datamart-geo.

    Works around atlas-profiler 0.0.2b1: when its ML geo-classifier labels a column
    (the usual case for place names), it skips the datamart-geo resolution that
    produces admin-area coverage, so place names never get a bounding box. Uses the
    profiler's own disambiguation, so results match what it would have computed.
    """
    geo = geo_data()
    if geo is None or not rows:
        return None
    from profiler.spatial import disambiguate_admin_areas

    boxes = []
    for column in profiler_output.get("columns") or []:
        semantic = {s.rsplit("/", 1)[-1] for s in column.get("semantic_types") or []}
        if "AdministrativeArea" not in semantic:
            continue
        values = [str(r.get(column["name"], "")).strip() for r in rows]
        values = [v for v in values if v]
        distinct = sorted(set(values))
        if len(distinct) < 3:  # same threshold the profiler uses
            continue
        # The ML geo-classifier also labels non-place columns as places (seen on
        # UltraViolet: unemployment rates resolved to a province of Iran). Numbers are
        # never place names here, and most distinct names must actually resolve.
        if sum(bool(_BARE_NUMBER_RE.fullmatch(v)) for v in distinct) > len(distinct) / 2:
            continue
        # Resolve each distinct name once: the box is the same, and per-row lookups
        # made whole-file profiling of large tables very slow.
        resolved = [r for r in geo.resolve_names_all(distinct) if r]
        if len(resolved) < MIN_RESOLVED_SHARE * len(distinct):
            continue
        disambiguated = disambiguate_admin_areas(resolved) if resolved else None
        # Returns (level, areas), one area per name, all at the same admin level.
        areas = disambiguated[1] if isinstance(disambiguated, tuple) else disambiguated
        bounds = [a.bounds for a in areas or [] if a is not None and a.bounds]
        if bounds:  # datamart-geo bounds are (min_lon, max_lon, min_lat, max_lat)
            boxes.append((min(b[0] for b in bounds), min(b[2] for b in bounds),
                          max(b[1] for b in bounds), max(b[3] for b in bounds)))
    if not boxes:
        return None
    bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    if bbox[2] - bbox[0] <= 0.01 or bbox[3] - bbox[1] <= 0.01:
        return None
    return [float(v) for v in bbox]


def bbox_from_profiler(profiler_output: dict[str, Any]) -> list[float] | None:
    """[min_lon, min_lat, max_lon, max_lat] enclosing every envelope the profiler found."""
    boxes = []
    for item in profiler_output.get("spatial_coverage") or []:
        for entry in item.get("ranges") or []:
            shape = entry.get("range") or {}
            coords = shape.get("coordinates")
            if shape.get("type") == "envelope" and coords and len(coords) == 2:
                (west, north), (east, south) = coords
                boxes.append((west, south, east, north))
    if not boxes:
        return None
    bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    valid = -180 <= bbox[0] <= bbox[2] <= 180 and -90 <= bbox[1] <= bbox[3] <= 90
    return [float(v) for v in bbox] if valid else None
