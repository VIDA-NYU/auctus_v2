"""Value labels from a record's documentation (codebooks, data dictionaries, READMEs).

From the data alone a survey code ("6 = Widowed") can't be told from a count
("6 adults"), so the profiler leaves small integers numeric. When the record documents
a column with value labels, that column is a category: this module finds those labels
and marks the profiled column categorical, keeping the labels.

Layouts handled, all by one rule (we know the table's column names):
- codebook tables (CSV/XLSX): a row naming a column, then that row and the following
  rows hold code/label pairs, the label on either side of the code (REDCap exports,
  "Column Label | Values | Value Labels" codebooks);
- text (README, txt, md): a line naming a column, then "1 = Yes", "2: No",
  "3 (very likely)" in the lines up to the next column named.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

DOC_NAME = re.compile(r"read[_ -]?me|codebook|code[_ -]book|dictionar|variables?|data[_ -]?desc|documentation"
                      r"|metadata|schema|legend|column", re.IGNORECASE)
TABLE_DOC_EXTENSIONS = {"csv", "tsv", "xlsx", "xls"}
TEXT_DOC_EXTENSIONS = {"txt", "md", "rst", ""}
MAX_DOC_BYTES = 5 * 1024 * 1024
MIN_LABELS = 2
CATEGORICAL = "http://schema.org/Enumeration"
BOOLEAN = "http://schema.org/Boolean"
INTEGER = "http://schema.org/Integer"
FLOAT = "http://schema.org/Float"

_CODE = re.compile(r"^-?\d{1,4}$")
_TEXT_PAIR = re.compile(r"(?<![\w.])(-?\d{1,4})\s*(?:=|:|\)|\(|-|–)\s*([A-Za-z][^|;,=\n]{0,60})")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def doc_extension(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


def is_documentation(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    ext = doc_extension(name)
    return bool(DOC_NAME.search(base)) and ext in TABLE_DOC_EXTENSIONS | TEXT_DOC_EXTENSIONS \
        and not base.startswith((".", "~$"))


def rows_from_bytes(data: bytes, name: str) -> list[list[str]] | None:
    """Rows of a tabular doc (every sheet of a workbook), or None for text docs."""
    ext = doc_extension(name)
    if ext in ("xlsx", "xls"):
        import pandas as pd

        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str, header=None,
                               engine="openpyxl" if ext == "xlsx" else "xlrd")
        return [[("" if v is None or str(v) == "nan" else str(v)) for v in row]
                for frame in sheets.values() for row in frame.itertuples(index=False, name=None)]
    if ext in ("csv", "tsv"):
        text = data.decode("utf-8-sig", errors="replace")
        return list(csv.reader(io.StringIO(text), delimiter="\t" if ext == "tsv" else ","))
    return None


def _pairs_in_row(row: list[str]) -> list[tuple[str, str]]:
    """(code, label) pairs in a codebook row: an integer cell with the nearest text cell
    to its right, else to its left; plus inline "1=Yes; 2=No" text."""
    cells = [c.strip() for c in row]
    pairs = []
    for i, cell in enumerate(cells):
        if not _CODE.match(cell):
            continue
        label = next((c for c in cells[i + 1:] if c and not _CODE.match(c)), None) if i + 1 < len(cells) else None
        if label is None or len(label) > 80:
            label = next((c for c in reversed(cells[:i]) if c and not _CODE.match(c)), None)
        if label and not re.fullmatch(r"-?\d+(\.\d+)?", label):  # "18-22" is a label, "3.5" isn't
            pairs.append((cell, label))
    for cell in cells:
        pairs += [(a, b.strip()) for a, b in _TEXT_PAIR.findall(cell) if (a, b.strip()) not in pairs]
    return pairs


def labels_from_rows(rows: list[list[str]], columns: Iterable[str]) -> dict[str, dict[str, str]]:
    wanted = {_norm(c): c for c in columns if _norm(c)}
    found: dict[str, dict[str, str]] = {}
    current = None
    for row in rows:
        named = [wanted[_norm(c)] for c in row if _norm(c) in wanted]
        if named:
            current = named[0]
            found.setdefault(current, {})
        if current is None:
            continue
        for code, label in _pairs_in_row(row):
            found[current].setdefault(code, label)
    return {c: v for c, v in found.items() if len(v) >= MIN_LABELS}


def labels_from_text(text: str, columns: Iterable[str]) -> dict[str, dict[str, str]]:
    wanted = {_norm(c): c for c in columns if len(_norm(c)) >= 3}
    found: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        tokens = {_norm(t) for t in re.split(r"[\s|,;:()`*\[\]]+", line) if t}
        named = [wanted[t] for t in tokens if t in wanted]
        if named:
            current = named[0]
            found.setdefault(current, {})
        if current is None:
            continue
        for code, label in _TEXT_PAIR.findall(line):
            found[current].setdefault(code, label.strip())
    return {c: v for c, v in found.items() if len(v) >= MIN_LABELS}


def value_labels(docs: Iterable[tuple[str, bytes]], columns: list[str]) -> dict[str, dict[str, Any]]:
    """column -> {"labels": {code: label}, "source": doc name}; tabular docs win over text."""
    result: dict[str, dict[str, Any]] = {}
    text_docs = []
    for name, data in docs:
        try:
            rows = rows_from_bytes(data, name)
        except Exception as exc:  # unreadable doc: skip it
            LOGGER.info("Can't read documentation %s: %r", name, exc)
            continue
        if rows is None:
            text_docs.append((name, data.decode("utf-8", errors="replace")))
            continue
        for column, labels in labels_from_rows(rows, columns).items():
            result.setdefault(column, {"labels": labels, "source": name})
    for name, text in text_docs:
        for column, labels in labels_from_text(text, columns).items():
            result.setdefault(column, {"labels": labels, "source": name})
    return result


def apply_value_labels(profiler_output: dict[str, Any], documented: dict[str, dict[str, Any]]) -> int:
    """Mark documented numeric columns categorical; returns how many were marked."""
    marked = 0
    for column in profiler_output.get("columns") or []:
        doc = documented.get(column.get("name"))
        # Integers only: a float column's documented "0 = none; 1 = perfect" describes the
        # ends of a continuous scale (DHL depth_index), not codes.
        if not doc or column.get("structural_type") != INTEGER:
            continue
        semantic = [s for s in column.get("semantic_types") or []
                    if not (s == BOOLEAN and len(doc["labels"]) > 2)]  # 0/1 data, 7 documented options
        if CATEGORICAL not in semantic:
            semantic.append(CATEGORICAL)
        column["semantic_types"] = semantic
        # A list, not a code-keyed dict: dict keys would each become an index field.
        column["value_labels"] = [{"code": code, "label": label} for code, label in doc["labels"].items()]
        column["value_labels_text"] = "; ".join(doc["labels"].values())  # searchable ("Widowed")
        column["value_labels_source"] = doc["source"]
        marked += 1
    return marked


_DOC_CACHE: dict[str, bytes | None] = {}
_DOC_CACHE_MAX = 256


def _fetch(url: str) -> bytes | None:
    """Download a documentation file once per worker process (a record's files share docs)."""
    if url in _DOC_CACHE:
        return _DOC_CACHE[url]
    import httpx

    from crawlers.invenio.http import USER_AGENT, ssl_context

    data = None
    try:
        with httpx.Client(timeout=60, verify=ssl_context(), follow_redirects=True,
                          headers={"User-Agent": USER_AGENT, "Accept": "*/*"}) as client:
            response = client.get(url)
            if response.status_code == 200 and len(response.content) <= MAX_DOC_BYTES:
                data = response.content
    except Exception as exc:
        LOGGER.info("Can't fetch documentation %s: %r", url, exc)
    if len(_DOC_CACHE) >= _DOC_CACHE_MAX:
        _DOC_CACHE.pop(next(iter(_DOC_CACHE)))
    _DOC_CACHE[url] = data
    return data


def documentation_for(dataset_meta: dict[str, Any]) -> list[tuple[str, bytes]]:
    """(name, bytes) of every documentation file known for a dataset: the record's
    docs (URLs from discovery), docs inside the same zip, and local paths (tests,
    benchmarks)."""
    from pathlib import Path

    docs = list(dataset_meta.get("_archive_documentation") or [])
    for path in dataset_meta.get("local_documentation") or []:
        try:
            docs.append((Path(path).name, Path(path).read_bytes()))
        except OSError:
            pass
    for doc in (dataset_meta.get("record") or {}).get("documentation") or []:
        data = _fetch(doc["url"])
        if data is not None:
            docs.append((doc["key"], data))
    return docs
