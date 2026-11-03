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

    cpg, dbf = shp.with_suffix(".cpg"), shp.with_suffix(".dbf")
    if cpg.exists():
        encoding = cpg.read_text(errors="replace").strip()
    else:  # no declared encoding: UTF-8 if the attribute table decodes as such, else cp1252
        from crawlers.invenio.transformer import ENCODING_SAMPLE_BYTES, guess_encoding

        with open(dbf, "rb") as fh:
            encoding = guess_encoding(fh.read(ENCODING_SAMPLE_BYTES)).removesuffix("-sig")
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


# --- SQLite / SpatiaLite / GeoPackage -------------------------------------------------

DATABASE_EXTENSIONS = {"sqlite", "sqlite3", "gpkg"}
# Bookkeeping tables of SQLite, SpatiaLite and GeoPackage: never data.
_SYSTEM_TABLES = {
    "geometry_columns", "geometry_columns_auth", "geometry_columns_statistics", "geometry_columns_field_infos",
    "geometry_columns_time", "geom_cols_ref_sys", "spatial_ref_sys", "spatial_ref_sys_aux", "spatial_ref_sys_all",
    "spatialite_history", "sql_statements_log", "spatialindex", "elementarygeometries", "knn", "knn2",
    "views_geometry_columns", "views_geometry_columns_auth", "views_geometry_columns_statistics",
    "views_geometry_columns_field_infos", "virts_geometry_columns", "virts_geometry_columns_auth",
    "virts_geometry_columns_statistics", "virts_geometry_columns_field_infos", "vector_layers",
    "vector_layers_auth", "vector_layers_statistics", "vector_layers_field_infos", "data_licenses",
    "rl2map_configurations", "se_external_graphics", "se_fonts", "se_raster_styled_layers",
    "se_vector_styled_layers", "layer_statistics", "views_layer_statistics", "virts_layer_statistics",
}
_SYSTEM_PREFIXES = ("sqlite_", "idx_", "rtree_", "gpkg_", "gpkgext_")
MAX_DATABASE_TABLES = 200


def _blob_envelope(blob: bytes) -> tuple[int | None, tuple[float, float, float, float] | None]:
    """(srid, (minx, miny, maxx, maxy)) from a SpatiaLite or GeoPackage geometry blob header."""
    import struct

    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 8:
        return None, None
    if blob[:2] == b"GP":  # GeoPackage: magic, version, flags, srs_id, optional envelope
        flags = blob[3]
        order = "<" if flags & 1 else ">"
        srid = struct.unpack(order + "i", blob[4:8])[0]
        if (flags >> 1) & 7 == 0 or len(blob) < 40:
            return srid, None
        minx, maxx, miny, maxy = struct.unpack(order + "4d", blob[8:40])
        return srid, (minx, miny, maxx, maxy)
    if blob[0] == 0 and blob[1] in (0, 1) and len(blob) >= 38:  # SpatiaLite: start, endian, srid, MBR
        order = "<" if blob[1] == 1 else ">"
        srid = struct.unpack(order + "i", blob[2:6])[0]
        return srid, struct.unpack(order + "4d", blob[6:38])
    return None, None


def _geometry_columns(con) -> dict[str, set[str]]:
    """table -> its geometry column names, from SpatiaLite's or GeoPackage's registry."""
    found: dict[str, set[str]] = {}
    for query in ("select f_table_name, f_geometry_column from geometry_columns",
                  "select table_name, column_name from gpkg_geometry_columns"):
        try:
            for table, column in con.execute(query):
                found.setdefault(str(table).lower(), set()).add(str(column).lower())
        except Exception:
            continue
    return found


def database_tables(path: Path) -> list[str]:
    """User tables and views of a SQLite database, in name order."""
    import sqlite3

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [n for (n,) in con.execute("select name from sqlite_master where type in ('table', 'view') order by name")]
    finally:
        con.close()
    return [n for n in names if n.lower() not in _SYSTEM_TABLES and not n.lower().startswith(_SYSTEM_PREFIXES)
            ][:MAX_DATABASE_TABLES]


def read_database_table(path: Path, table: str) -> Read:
    """One table's rows; geometry blobs are left out and their extent becomes the bbox."""
    import sqlite3

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    quoted = '"' + table.replace('"', '""') + '"'
    columns = [r[1] for r in con.execute(f"pragma table_info({quoted})")]
    geometry = _geometry_columns(con).get(table.lower(), set())
    kept = [c for c in columns if c.lower() not in geometry]
    box, srid, types = None, None, None
    for column in [c for c in columns if c.lower() in geometry][:1]:
        for (blob,) in con.execute(f'select "{column}" from {quoted} where "{column}" is not null'):
            blob_srid, env = _blob_envelope(blob)
            if env is None:
                continue
            srid = srid or blob_srid
            box = list(env) if box is None else [min(box[0], env[0]), min(box[1], env[1]),
                                                 max(box[2], env[2]), max(box[3], env[3])]
        types = "geometry"
    bbox = to_lonlat(box, f"EPSG:{srid}" if srid and srid > 0 else None) if box else None

    def rows() -> Iterator[tuple]:
        try:
            select = ", ".join('"' + c.replace('"', '""') + '"' for c in kept) or "1"
            for row in con.execute(f"select {select} from {quoted}"):
                yield tuple(v.hex() if isinstance(v, (bytes, bytearray)) else v for v in row)
        finally:
            con.close()

    return Read(header=kept, rows=rows(), bbox=bbox, geometry_type=types)
