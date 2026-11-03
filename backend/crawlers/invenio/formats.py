"""Readers for the non-delimited formats repositories hold: shapefiles, GeoJSON, Stata.

Each returns the table's header, an iterator over its rows (never the whole file in
memory where the format allows it), and what the format knows beyond the rows: a
lon/lat bounding box for the geographic formats, value labels for Stata. The
transformer writes the rows to the normalized CSV atlas-profiler reads, like any
other format. Geometries themselves are not profiled; their extent becomes the
dataset's spatial coverage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

LOGGER = logging.getLogger(__name__)

WGS84_NAMES = {"EPSG:4326", "OGC:CRS84", "URN:OGC:DEF:CRS:OGC:1.3:CRS84", "URN:OGC:DEF:CRS:EPSG::4326"}


@dataclass
class Read:
    header: list[str]
    rows: Iterable[tuple]
    bbox: list[float] | None = None              # [min_lon, min_lat, max_lon, max_lat]
    value_labels: dict[str, dict[str, str]] = field(default_factory=dict)
    geometry_type: str | None = None


def _plausible_lonlat(bbox) -> bool:
    return bool(bbox) and -180 <= bbox[0] <= bbox[2] <= 180 and -90 <= bbox[1] <= bbox[3] <= 90


def to_lonlat(bbox: list[float], crs_text: str | None) -> list[float] | None:
    """A bbox in lon/lat. No CRS: trusted only if it already looks like lon/lat.
    Projected (NY State Plane feet in the NYU Spatial Data Repository): reprojected."""
    if not crs_text:
        return list(bbox) if _plausible_lonlat(bbox) else None
    try:
        from pyproj import CRS, Transformer

        crs = CRS.from_user_input(crs_text)
        if crs.is_geographic:
            out = list(bbox)
        else:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            out = list(transformer.transform_bounds(*bbox))
    except Exception as exc:
        LOGGER.info("Can't use CRS %.60r: %r", crs_text, exc)
        return None
    return [float(v) for v in out] if _plausible_lonlat(out) else None


# --- shapefiles ---------------------------------------------------------------------


def read_shapefile(shp: Path) -> Read:
    """Attribute table (.dbf) rows, and the layer's extent from the .shp header,
    reprojected with the .prj. Sidecars must sit next to `shp` with the same stem."""
    import shapefile

    cpg = shp.with_suffix(".cpg")
    encoding = cpg.read_text(errors="replace").strip() if cpg.exists() else "utf-8"
    if encoding.isdigit():  # "1252" in some .cpg files
        encoding = f"cp{encoding}"
    reader = shapefile.Reader(str(shp), encoding=encoding or "utf-8", encodingErrors="replace")
    header = [f[0] for f in reader.fields[1:]]  # fields[0] is the deletion flag
    prj = shp.with_suffix(".prj")
    bbox = None
    if len(reader) and getattr(reader, "bbox", None):
        bbox = to_lonlat([float(v) for v in reader.bbox[:4]], prj.read_text(errors="replace") if prj.exists() else None)

    def rows() -> Iterator[tuple]:
        try:
            for record in reader.iterRecords():
                yield tuple(record)
        finally:
            reader.close()

    return Read(header=header, rows=rows(), bbox=bbox, geometry_type=reader.shapeTypeName)


# --- GeoJSON ------------------------------------------------------------------------


def _walk_coordinates(coords, box: list[float]) -> None:
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        x, y = float(coords[0]), float(coords[1])
        box[0], box[1], box[2], box[3] = min(box[0], x), min(box[1], y), max(box[2], x), max(box[3], y)
    elif isinstance(coords, (list, tuple)):
        for c in coords:
            _walk_coordinates(c, box)


def _geometry_bbox(geometry: dict | None, box: list[float]) -> None:
    if not geometry:
        return
    if geometry.get("type") == "GeometryCollection":
        for g in geometry.get("geometries") or []:
            _geometry_bbox(g, box)
    else:
        _walk_coordinates(geometry.get("coordinates"), box)


def read_geojson(path: Path) -> Read:
    """Feature properties as rows; the extent of every geometry as the bbox."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        data = json.load(fh)
    features = data.get("features") if data.get("type") == "FeatureCollection" else [data]
    header: dict[str, None] = {}
    box = [float("inf"), float("inf"), float("-inf"), float("-inf")]
    types: dict[str, int] = {}
    for feature in features or []:
        header.update(dict.fromkeys((feature.get("properties") or {}).keys()))
        geometry = feature.get("geometry") or {}
        if geometry.get("type"):
            types[geometry["type"]] = types.get(geometry["type"], 0) + 1
        _geometry_bbox(geometry, box)
    names = list(header)

    def cell(value):
        return json.dumps(value) if isinstance(value, (dict, list)) else value

    rows = (tuple(cell((f.get("properties") or {}).get(n)) for n in names) for f in features or [])
    bbox = box if box[0] != float("inf") else None
    # RFC 7946 GeoJSON is lon/lat; older files may name another CRS.
    crs = ((data.get("crs") or {}).get("properties") or {}).get("name")
    if bbox and crs and crs.upper() not in WGS84_NAMES:
        bbox = to_lonlat(bbox, crs)
    elif bbox and not _plausible_lonlat(bbox):
        bbox = None
    return Read(header=names, rows=rows, bbox=bbox,
                geometry_type=max(types, key=types.get) if types else None)


# --- Stata --------------------------------------------------------------------------

STATA_CHUNK_ROWS = 50_000


def read_stata(path: Path) -> Read:
    """Rows with the stored codes (not the labels), plus the file's value labels."""
    import pandas as pd

    from crawlers.invenio.codebook import labels_from_stata

    with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
        header = [str(v) for v in reader.variable_labels()]  # also reads the header (pandas opens lazily)
    labels = labels_from_stata(str(path), header)

    def rows() -> Iterator[tuple]:
        with pd.read_stata(path, iterator=True, convert_categoricals=False, chunksize=STATA_CHUNK_ROWS) as reader:
            for chunk in reader:
                # Integer codes with missing values come back as floats (1.0); keep them integers.
                for name in chunk.select_dtypes("float").columns:
                    values = chunk[name].dropna()
                    if len(values) and (values % 1 == 0).all():
                        chunk[name] = chunk[name].astype("Int64")
                yield from chunk.astype(object).where(chunk.notna(), None).itertuples(index=False, name=None)

    return Read(header=header, rows=rows(), value_labels=labels)


READERS: dict[str, Any] = {"shp": read_shapefile, "geojson": read_geojson, "dta": read_stata}
