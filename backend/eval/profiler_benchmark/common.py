"""Shared logic for the profiler benchmark: corpus loading and type mapping.

Profiling itself always goes through the connector
(crawlers.invenio.transformer.load_local_table / profile_table / assemble_record),
so the benchmark measures exactly what ingestion runs. It needs a checkout of the
backend and its Python requirements, not the Docker stack.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TABULAR_EXTENSIONS = {"csv", "tsv", "xlsx", "xls"}

# Coarse classes used for scoring. Both the profiler's output and the gold labels
# are mapped onto these, because README dictionaries and users think in these terms,
# not in schema.org URIs.
COARSE_CLASSES = ("numeric", "date", "categorical", "boolean", "geo_coordinate", "place", "identifier", "text")


def extension(path: str | Path) -> str:
    name = str(path).lower()
    return name.rsplit(".", 1)[-1] if "." in name else ""


def coarse_from_profiler(column: dict) -> str:
    """Map one atlas-profiler column (structural + semantic types) to a coarse class."""
    structural = (column.get("structural_type") or "").rsplit("/", 1)[-1]
    semantic = {s.rsplit("/", 1)[-1] for s in column.get("semantic_types") or []}
    if semantic & {"latitude", "longitude"} or structural == "GeoCoordinates":
        return "geo_coordinate"
    if "DateTime" in semantic or structural == "DateTime":
        return "date"
    if "Boolean" in semantic:
        return "boolean"
    if semantic & {"AdministrativeArea", "address", "GeoShape"}:
        return "place"
    if "identifier" in semantic:  # schema.org/identifier (lowercase)
        return "identifier"
    if "Enumeration" in semantic or "Categorical" in semantic:
        return "categorical"
    if structural in ("Integer", "Float"):
        return "numeric"
    return "text"


# README dictionaries describe types in free text ("Continuous", "YYYY", "categorical:
# overall, trade, ..."). Only clear cases are mapped; anything else is left out of
# the type score rather than guessed.
_GOLD_RULES = [
    ("date", r"\b(yyyy|mm/dd|dd/mm|date|year|datetime|timestamp|time)\b"),
    ("geo_coordinate", r"\b(latitude|longitude|lat|lon|lng|coordinates?)\b"),
    ("boolean", r"\b(boolean|binary|yes/no|true/false|0\s*=\s*no|1\s*=\s*yes|dummy)\b"),
    ("identifier", r"\b(identifier|unique id|id number|uuid)\b"),
    ("categorical", r"\b(categorical|category|factor|nominal|ordinal|levels?)\b"),
    ("numeric", r"\b(continuous|numeric|number|integer|int|float|decimal|count|percent(age)?|ratio|index)\b"),
    ("text", r"\b(text|string|free[- ]text|character)\b"),
]


_PLACE_WORDS = r"\b(country|countries|state|city|cities|county|borough|region|province|municipality|neighbou?rhood|district|zip|postal)\b"
_CODED_VALUES = r"(\(\d+\)|\b\d+\s*=|\b[01]\s*,\s*[01]\b)"


def coarse_from_gold(type_text: str, units_text: str = "", column: str = "", definition: str = "") -> tuple[str | None, set[str]]:
    """(expected class, acceptable classes) for one README dictionary entry.

    The type text decides the expected class; the units text is used only when the
    type is empty. Acceptable answers widen where the README's type is right but
    coarser than the profiler's: a year stored as an integer is also a date, a
    categorical column of country names is also a place, and categories stored as
    numeric codes (1 = "Not at all") are also numeric. Unmappable entries return
    (None, set()) and are left out of the score.
    """
    expected = None
    for text in (type_text.lower(), "" if type_text.strip() else units_text.lower()):
        for label, pattern in _GOLD_RULES:
            if text and re.search(pattern, text):
                expected = label
                break
        if expected:
            break
    # A codebook listing value labels ("1: Single; 2: Married") documents categories,
    # whatever storage type it states (REDCap/SPSS exports say "Numeric" for these).
    labels = re.findall(r"(?:^|[;,\n])\s*-?\d+(?:\.\d+)?\s*[:=]\s*\S", units_text)
    if expected in (None, "numeric") and len(labels) >= 2:
        return "categorical", {"categorical", "boolean"} if len(labels) == 2 else {"categorical"}
    if expected is None:
        return None, set()
    acceptable = {expected}
    context = f"{column} {definition}".lower()
    if re.search(r"\byears?\b", context) and expected == "numeric":
        acceptable.add("date")
    if re.search(_PLACE_WORDS, context) and expected in ("categorical", "text"):
        acceptable.add("place")
    if expected in ("categorical", "boolean") and re.search(_CODED_VALUES, units_text):
        acceptable |= {"numeric", "categorical", "boolean"}
    return expected, acceptable


def load_corpus(harvest_dirs: list[Path]) -> list[dict]:
    """Local tabular files from invenio_fetching harvest directories.

    Each directory has file_manifest.json (rows with record, key, ext, size,
    local_path) and the downloaded files under files/. Any harvest made with
    `python -m invenio_fetching harvest <instance>` works.
    """
    corpus = []
    for harvest_dir in harvest_dirs:
        manifest = json.loads((harvest_dir / "file_manifest.json").read_text())
        for row in manifest:
            if row.get("local_path") and row["ext"] in TABULAR_EXTENSIONS:
                path = (harvest_dir / row["local_path"]).resolve()
                if path.exists():
                    corpus.append({"instance": harvest_dir.name, "record": row["record"], "key": row["key"],
                                   "ext": row["ext"], "size": row["size"], "path": str(path)})
    return corpus
