"""Find the documentation that could hold gold (no LLM): `run scan`.

For every record Auctus indexed on an instance, collect the files that may document
its tables: READMEs, codebooks, data dictionaries, variable lists, as loose files or
inside zips, in text, Markdown, CSV/TSV, spreadsheet or PDF form. Each candidate is
scored by how many of the record's indexed column names appear in it, so the LLM
extraction (`run gold`) only reads documents that actually describe columns.

Writes sources/<instance>/<record>.json: curator fields, indexed files and their
columns, and every candidate document with its text and column hits.
"""

from __future__ import annotations

import asyncio
import io
import re
from typing import Any
from urllib.parse import quote

from crawlers.invenio.archives import open_remote_zip
from crawlers.invenio.http import async_client
from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

from eval.description_eval.common import DATA_DIR, INSTANCES, normalize_name, write_json
from eval.description_eval.gold import curator_fields

SOURCES_DIR = DATA_DIR / "sources"
# File names that suggest documentation of the data (matched on the base name).
DOC_NAME = re.compile(r"read[_ -]?me|codebook|code[_ -]book|dictionar|variables?|data[_ -]?desc|documentation"
                      r"|metadata|schema|legend|column", re.IGNORECASE)
DOC_EXTENSIONS = {"md", "txt", "", "rst", "csv", "tsv", "xlsx", "xls", "pdf", "json", "html", "htm"}
MAX_DOC_BYTES = 5 * 1024 * 1024
MAX_DOC_CHARS = 120_000
# Column names shorter than this match too much prose ("id", "x") to count as hits.
MIN_NAME_CHARS = 4


def extension(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


def is_doc(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return bool(DOC_NAME.search(base)) and extension(name) in DOC_EXTENSIONS and not base.startswith((".", "~$"))


def to_text(data: bytes, ext: str) -> str:
    """Plain text of a documentation file (spreadsheets as CSV, PDFs via pypdf)."""
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages[:60])
    if ext in ("xlsx", "xls"):
        import pandas as pd

        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str, engine="openpyxl" if ext == "xlsx" else "xlrd")
        return "\n\n".join(f"## sheet {name}\n{frame.fillna('').to_csv(index=False)}" for name, frame in sheets.items())
    return data.decode("utf-8", errors="replace")


def column_hits(text: str, columns: list[str]) -> list[str]:
    """Indexed column names that occur in the document (case/punctuation-insensitive)."""
    flat = normalize_name(text)
    return sorted({c for c in columns if len(normalize_name(c)) >= MIN_NAME_CHARS and normalize_name(c) in flat})


def indexed_records(domain: str) -> dict[str, dict[str, Any]]:
    """record_id -> {files: [{dataset_id, file_key, columns}]} for an instance's indexed docs."""
    response = get_client().search(index=AUCTUS_INDEX_NAME, body={
        "size": 10_000, "_source": ["source_record.record_id", "source_record.file_key", "profiler_metadata.columns.name"],
        "query": {"term": {"domain": domain}}})
    records: dict[str, dict[str, Any]] = {}
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        columns = [c["name"] for c in (src.get("profiler_metadata") or {}).get("columns") or []]
        records.setdefault(src["source_record"]["record_id"], {"files": []})["files"].append(
            {"dataset_id": hit["_id"], "file_key": src["source_record"]["file_key"], "columns": columns})
    return records


def zip_docs(url: str, size: int, zip_key: str) -> list[dict[str, Any]]:
    """Documentation members of a remote zip, read in place."""
    docs = []
    with open_remote_zip(url, size) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/") or not is_doc(info.filename):
                continue
            if info.file_size > MAX_DOC_BYTES:
                continue
            docs.append({"name": info.filename, "where": f"zip:{zip_key}", "data": archive.read(info)})
    return docs


async def scan_record(client, domain: str, record_id: str, indexed: dict[str, Any]) -> dict[str, Any]:
    response = await client.get(f"https://{domain}/api/records/{record_id}")
    response.raise_for_status()
    record = response.json()
    entries = (record.get("files") or {}).get("entries") or {}
    raw_docs: list[dict[str, Any]] = []
    for key, entry in entries.items():
        url = f"https://{domain}/api/records/{record_id}/files/{quote(key)}/content"
        if is_doc(key) and (entry.get("size") or 0) <= MAX_DOC_BYTES:
            content = await client.get(url, headers={"Accept": "*/*"})
            if content.status_code == 200:
                raw_docs.append({"name": key, "where": "loose", "data": content.content})
        elif extension(key) == "zip":
            try:
                raw_docs += await asyncio.to_thread(zip_docs, url, entry.get("size") or 0, key)
            except Exception as exc:  # split or unreadable zips
                print(f"  {record_id}: can't list {key}: {exc!r}")

    columns = sorted({c for f in indexed["files"] for c in f["columns"]})
    docs = []
    for doc in raw_docs:
        try:
            text = to_text(doc["data"], extension(doc["name"]))
        except Exception as exc:
            print(f"  {record_id}: can't read {doc['name']}: {exc!r}")
            continue
        hits = column_hits(text, columns)
        eligible = [c for c in columns if len(normalize_name(c)) >= MIN_NAME_CHARS]
        docs.append({"name": doc["name"], "where": doc["where"], "chars": len(text),
                     "truncated": len(text) > MAX_DOC_CHARS, "column_hits": len(hits),
                     "coverage": round(len(hits) / len(eligible), 3) if eligible else None,
                     "text": text[:MAX_DOC_CHARS]})
    docs.sort(key=lambda d: -(d["column_hits"]))
    return {"record_id": record_id, "domain": domain, "curator": curator_fields(record),
            "files": indexed["files"], "docs": docs}


async def scan(instance: str) -> list[dict[str, Any]]:
    domain = INSTANCES[instance]
    records = indexed_records(domain)
    out = []
    async with async_client(120) as client:
        for record_id, indexed in sorted(records.items()):
            try:
                result = await scan_record(client, domain, record_id, indexed)
            except Exception as exc:
                print(f"  {record_id}: failed: {exc!r}")
                continue
            write_json(SOURCES_DIR / instance / f"{record_id}.json", result)
            out.append(result)
    return out


def summary(results: list[dict[str, Any]], instance: str, min_hits: int = 3) -> None:
    with_docs = [r for r in results if r["docs"]]
    with_dict = [r for r in results if any(d["column_hits"] >= min_hits for d in r["docs"])]
    columns = sum(len({c for f in r["files"] for c in f["columns"]}) for r in with_dict)
    print(f"{instance}: {len(results)} records scanned | {len(with_docs)} with documentation | "
          f"{len(with_dict)} with a doc naming >= {min_hits} of their columns ({columns} columns in those records)")
    for r in with_dict:
        best = r["docs"][0]
        print(f"  {r['record_id']}: {best['name'][:60]} ({best['where'][:30]}) hits={best['column_hits']} "
              f"coverage={best['coverage']}")


def main(instances: list[str]) -> None:
    for instance in instances:
        summary(asyncio.run(scan(instance)), instance)
