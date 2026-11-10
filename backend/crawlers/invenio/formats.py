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
import re
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


# --- JCAMP-DX spectra -----------------------------------------------------------------

SPECTRUM_EXTENSIONS = {"jdx", "dx"}
# The data block starts at one of these; everything before it is the header.
_JCAMP_DATA_LABELS = {"XYDATA", "XYPOINTS", "PEAK TABLE", "PEAKTABLE", "NTUPLES", "END"}
# (column, JCAMP label): the header fields kept, one row per spectrum.
SPECTRUM_COLUMNS = [
    ("title", "TITLE"), ("data_type", "DATATYPE"), ("measured", "LONGDATE"), ("instrument", "SPECTROMETER/DATA SYSTEM"),
    ("origin", "ORIGIN"), ("x_units", "XUNITS"), ("y_units", "YUNITS"), ("first_x", "FIRSTX"), ("last_x", "LASTX"),
    ("resolution", "RESOLUTION"), ("points", "NPOINTS"), ("data_processing", "DATA PROCESSING"),
    ("comments", "COMMENTS"), ("sample_description", "SAMPLE DESCRIPTION"),
    ("main_peaks", "_PEAKS"),  # computed from the data block, not a JCAMP label
]
MAX_PEAKS = 8
PEAK_SEPARATION = 20.0  # in x units (cm-1 for IR)
PROMINENCE_WINDOW = 60.0
IR_RANGE = (600.0, 3800.0)  # the ends of a mid-IR scan are detector noise
IR_CO2_BAND = (2280.0, 2400.0)  # atmospheric CO2 doublet: an instrument artifact, not the sample


def read_jcamp_header(text: str) -> dict[str, str]:
    """The labelled fields before the data block of a JCAMP-DX file.

    Lines are `##LABEL=value`; a value continues on the lines that follow it
    (Nicolet writes its processing steps one per line under DATA PROCESSING), and
    `$$` starts a comment.
    """
    fields: dict[str, str] = {}
    label = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("##"):
            label, _, value = line[2:].partition("=")
            label = label.strip().upper()
            if label in _JCAMP_DATA_LABELS:
                break
            fields[label] = value.split("$$")[0].strip()
        elif label and line and not line.startswith("$$"):
            fields[label] = f"{fields[label]}; {line}" if fields[label] else line
    return fields


def spectrum_peaks(text: str, fields: dict[str, str]) -> str:
    """The strongest peaks of an (X++(Y..Y)) spectrum, as x positions, strongest first.

    Conservators identify materials by where the absorption bands are (a carbonyl near
    1730 cm-1), so the peak positions describe the sample better than the curve does.
    Only plain-number (AFFN) data is read; compressed ASDF data gives "".
    """
    block = text.split("##XYDATA", 1)
    if len(block) < 2 or "(X++(Y..Y))" not in block[1][:40]:
        return ""
    try:
        first, delta = float(fields["FIRSTX"]), float(fields.get("DELTAX") or 0)
        xfactor = float(fields.get("XFACTOR") or 1)
        yfactor = float(fields.get("YFACTOR") or 1)
    except (KeyError, ValueError):
        return ""
    xs, ys = [], []
    for line in block[1].split("\n")[1:]:
        if line.startswith("##"):
            break
        parts = line.split()
        if not parts:
            continue
        try:
            x0, values = float(parts[0]) * xfactor, [float(v) * yfactor for v in parts[1:]]
        except ValueError:
            return ""  # ASDF-compressed or malformed
        for k, y in enumerate(values):
            xs.append(x0 + k * (delta if delta else 0))
            ys.append(y)
    if len(ys) < 10:
        return ""
    if "TRANSMITTANCE" in fields.get("YUNITS", "").upper():
        ys = [-y for y in ys]  # bands are dips in transmittance
    step = abs(xs[1] - xs[0]) or 1.0
    window = max(1, int(PROMINENCE_WINDOW / step))
    infrared = fields.get("XUNITS", "").upper() in ("1/CM", "CM-1", "1/CM-1")
    prominence = {}
    for k in range(window, len(ys) - window):
        if ys[k] < max(ys[k - 1], ys[k + 1]) or (infrared and not IR_RANGE[0] <= xs[k] <= IR_RANGE[1]) \
                or (infrared and IR_CO2_BAND[0] <= xs[k] <= IR_CO2_BAND[1]):
            continue
        # Height above the higher of the two lowest points around it: a band, not baseline drift.
        prominence[k] = ys[k] - max(min(ys[k - window:k]), min(ys[k + 1:k + window + 1]))
    picked: list[float] = []
    for k in sorted(prominence, key=lambda k: -prominence[k]):
        if all(abs(xs[k] - x) >= PEAK_SEPARATION for x in picked):
            picked.append(xs[k])
        if len(picked) == MAX_PEAKS:
            break
    return ", ".join(str(round(x)) for x in picked)


def _jcamp_date(fields: dict[str, str]) -> str:
    """ISO date and time from LONGDATE (yyyy/mm/dd, v5) or DATE (yy/mm/dd, v4) and TIME."""
    date = fields.get("LONGDATE") or ""
    if not date and fields.get("DATE"):
        yy, _, rest = fields["DATE"].partition("/")
        date = f"{'19' if int(yy or 0) > 50 else '20'}{yy.zfill(2)}/{rest}" if yy.isdigit() else ""
    date = date.split()[0].replace("/", "-") if date else ""
    time = fields.get("TIME", "").strip()
    return f"{date}T{time}" if date and time else date


# NYU conservation uploads the closest spectral-library entry next to each sample
# ("..._EarringsWrapper_LibraryMatch.JDX"). Its date is when the library spectrum was
# measured (years earlier), so it's kept out of `measured` and the time range.
_LIBRARY_MATCH = re.compile(r"library[ _-]?match", re.IGNORECASE)


def spectrum_row(file_key: str, fields: dict[str, str]) -> tuple:
    reference = bool(_LIBRARY_MATCH.search(file_key))
    values = []
    for name, label in SPECTRUM_COLUMNS:
        if name == "measured":
            values.append("" if reference else _jcamp_date(fields))
        else:
            values.append(fields.get(label, ""))
    return (file_key, *values, "library match" if reference else "sample")


def read_spectra(headers: list[tuple[str, dict[str, str]]]) -> Read:
    """One row per spectrum file, from (file key, header fields) pairs."""
    return Read(header=["file", *(name for name, _ in SPECTRUM_COLUMNS), "spectrum"],
                rows=[spectrum_row(key, fields) for key, fields in headers])


# --- TIFF / GeoTIFF -------------------------------------------------------------------

TIFF_EXTENSIONS = {"tif", "tiff"}
_TIFF_TYPES = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 6: ("b", 1), 7: ("B", 1),
               8: ("h", 2), 9: ("i", 4), 10: ("ii", 8), 11: ("f", 4), 12: ("d", 8), 16: ("Q", 8), 17: ("q", 8),
               18: ("Q", 8)}
_TIFF_TAGS = {256: "width", 257: "height", 258: "bits_per_sample", 259: "compression", 270: "description",
              271: "make", 272: "model", 277: "bands", 305: "software", 306: "datetime", 339: "sample_format",
              33550: "pixel_scale", 33922: "tiepoint", 34264: "transform", 34735: "geokeys", 34737: "geo_ascii",
              42113: "nodata"}
_MAX_TAG_BYTES = 1 << 20


def read_tiff_header(fh) -> dict[str, Any]:
    """The first image's tags of a TIFF or BigTIFF, read with a few seeks (works on a
    remote file opened with byte-range requests: a 4.9 GB GeoTIFF costs a few KB)."""
    import struct

    head = fh.read(16)
    order = {b"II": "<", b"MM": ">"}.get(head[:2])
    if order is None:
        raise ValueError("not a TIFF file")
    magic = struct.unpack(order + "H", head[2:4])[0]
    big = magic == 43
    if magic not in (42, 43):
        raise ValueError("not a TIFF file")
    offset = struct.unpack(order + ("Q" if big else "I"), head[8:16] if big else head[4:8])[0]
    fh.seek(offset)
    count_size, entry_size, field = (8, 20, 8) if big else (2, 12, 4)
    count = struct.unpack(order + ("Q" if big else "H"), fh.read(count_size))[0]
    entries = fh.read(count * entry_size)
    tags: dict[str, Any] = {}
    for k in range(count):
        entry = entries[k * entry_size:(k + 1) * entry_size]
        tag, typ = struct.unpack(order + "HH", entry[:4])
        if tag not in _TIFF_TAGS or typ not in _TIFF_TYPES:
            continue
        n = struct.unpack(order + ("Q" if big else "I"), entry[4:4 + field])[0]
        code, width = _TIFF_TYPES[typ]
        size = width * n
        if size > _MAX_TAG_BYTES:
            continue
        raw = entry[4 + field:4 + 2 * field]
        if size > field:
            fh.seek(struct.unpack(order + ("Q" if big else "I"), raw)[0])
            raw = fh.read(size)
        raw = raw[:size]
        if typ == 2:
            value: Any = raw.split(b"\0")[0].decode("latin-1").strip()
        else:
            value = struct.unpack(order + code * n, raw)
            if typ in (5, 10):  # rationals: numerator/denominator pairs
                value = tuple(value[i] / value[i + 1] if value[i + 1] else 0.0 for i in range(0, len(value), 2))
            value = value[0] if len(value) == 1 else list(value)
        tags[_TIFF_TAGS[tag]] = value
    return tags


def tiff_epsg(tags: dict[str, Any]) -> int | None:
    """EPSG code from the GeoKey directory (projected first, else geographic)."""
    keys = tags.get("geokeys")
    if not isinstance(keys, list) or len(keys) < 4:
        return None
    found = {}
    for k in range(keys[3]):
        key_id, location, _count, value = keys[4 + 4 * k:8 + 4 * k]
        if location == 0:
            found[key_id] = value
    for key_id in (3072, 2048):  # ProjectedCSTypeGeoKey, GeographicTypeGeoKey
        code = found.get(key_id)
        if code and code != 32767:  # 32767 = user-defined
            return code
    return None


def tiff_bbox(tags: dict[str, Any]) -> tuple[list[float] | None, list[float] | None]:
    """(native bbox, lon/lat bbox) from the tie point + pixel scale, or the transform."""
    width, height = tags.get("width"), tags.get("height")
    if not (width and height):
        return None, None
    tie, scale, matrix = tags.get("tiepoint"), tags.get("pixel_scale"), tags.get("transform")
    if isinstance(tie, list) and len(tie) >= 6 and isinstance(scale, list) and len(scale) >= 2:
        i, j, _k, x, y, _z = tie[:6]
        x0, y0 = x - i * scale[0], y + j * scale[1]
        corners = [(x0, y0), (x0 + width * scale[0], y0 - height * scale[1])]
    elif isinstance(matrix, list) and len(matrix) >= 8:
        a, b, _c, d, e, f, _g, h = matrix[:8]
        corners = [(a * i + b * j + d, e * i + f * j + h) for i, j in ((0, 0), (width, 0), (0, height), (width, height))]
    else:
        return None, None
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    native = [min(xs), min(ys), max(xs), max(ys)]
    epsg = tiff_epsg(tags)
    return native, to_lonlat(native, f"EPSG:{epsg}" if epsg else None)
