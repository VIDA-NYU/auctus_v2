"""Build the hand-metadata ("gold") side.

For every record Auctus indexed on an instance (UltraViolet, CaltechDATA):
- record-level gold straight from the repository API: curator description,
  subjects, dates, locations (no extraction, these are typed fields);
- column-level gold from the record's documentation: READMEs, codebooks, data
  dictionaries (loose or inside zips; found by `run scan`). They come in many formats
  (HTML tables, grid tables, indented lists, codebook CSVs, PDFs), so an LLM extracts
  them into JSON, and a person must check each file against the source and set
  "verified": true before it counts.

UltraViolet gold is gold/<record>.json; other instances are gold/<instance>/<record>.json.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from crawlers.invenio.crawler import html_to_text
from crawlers.invenio.http import async_client
from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

from eval.description_eval.common import DATA_DIR, GOLD_DIR, UV_DOMAIN, ask_json, read_json, write_json

README_TEXT_EXTENSIONS = {"md", "txt", ""}
MAX_README_CHARS = 120_000

EXTRACTION_PROMPT = """You are extracting a data dictionary from a dataset's documentation (README, codebook or data dictionary).

The record contains these tabular files:
{file_list}

Documentation:
<<<
{readme}
>>>

Return JSON only, in this shape:
{{"files": {{"<file name from the list, or * if the dictionary is not tied to one file>":
   [{{"column": "<column name exactly as written>", "definition": "<the README's explanation>",
      "type": "<type/format as written, or empty>", "units": "<units or value labels as written, or empty>"}}]}},
 "notes": "<anything ambiguous, e.g. a dictionary that seems to describe several files>"}}

Rules: copy only what the README states; do not infer or invent columns, definitions,
types or units. If the README has no variable-level documentation, return {{"files": {{}}, "notes": "no data dictionary"}}.
"""


def indexed_uv_files() -> dict[str, list[dict[str, str]]]:
    """record_id -> [{dataset_id, file_key}] for every UltraViolet file in the index."""
    response = get_client().search(
        index=AUCTUS_INDEX_NAME,
        body={"size": 1000, "_source": ["source_record.record_id", "source_record.file_key"],
              "query": {"term": {"domain": UV_DOMAIN}}},
    )
    records: dict[str, list[dict[str, str]]] = {}
    for hit in response["hits"]["hits"]:
        source_record = hit["_source"]["source_record"]
        records.setdefault(source_record["record_id"], []).append(
            {"dataset_id": hit["_id"], "file_key": source_record["file_key"]})
    return records


def curator_fields(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "title": metadata.get("title"),
        "description": html_to_text(metadata.get("description")),
        "additional_descriptions": [html_to_text(d.get("description")) for d in metadata.get("additional_descriptions") or []],
        "subjects": [s.get("subject") for s in metadata.get("subjects") or [] if s.get("subject")],
        "dates": [{"date": d.get("date"), "type": (d.get("type") or {}).get("id")} for d in metadata.get("dates") or []],
        "locations": [
            {"place": f.get("place"), "geometry": f.get("geometry"), "description": f.get("description")}
            for f in (metadata.get("locations") or {}).get("features") or []
        ],
    }


async def fetch_readme(client, record: dict[str, Any]) -> tuple[str | None, str]:
    """(README file name, text) for the record's first text README, if any."""
    for key in (record.get("files") or {}).get("entries") or {}:
        ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
        if key.lower().startswith("readme") and ext in README_TEXT_EXTENSIONS:
            response = await client.get(
                f"https://{UV_DOMAIN}/api/records/{record['id']}/files/{quote(key)}/content",
                headers={"Accept": "*/*"})
            response.raise_for_status()
            return key, response.content.decode("utf-8", errors="replace")
    return None, ""


def gold_path(instance: str, record_id: str):
    return GOLD_DIR / f"{record_id}.json" if instance == "ultraviolet" else GOLD_DIR / instance / f"{record_id}.json"


# Documents naming fewer of the record's columns than this aren't sent for extraction.
MIN_COLUMN_HITS = 3


def documentation_prompt(source: dict[str, Any]) -> tuple[list[str], str]:
    """(doc names, text) of the docs worth extracting, best first, within MAX_README_CHARS."""
    names, parts, used = [], [], 0
    for doc in source["docs"]:
        if doc["column_hits"] < MIN_COLUMN_HITS or used >= MAX_README_CHARS:
            continue
        chunk = f"### {doc['name']} ({doc['where']})\n{doc['text'][: MAX_README_CHARS - used]}"
        names.append(doc["name"])
        parts.append(chunk)
        used += len(chunk)
    return names, "\n\n".join(parts)


def build_gold_from_sources(instance: str, record_ids: list[str] | None = None, overwrite: bool = False) -> list[str]:
    """Gold from `run scan` output; never overwrites a hand-verified file."""
    written = []
    for source_path in sorted((DATA_DIR / "sources" / instance).glob("*.json")):
        source = read_json(source_path)
        record_id = source["record_id"]
        if record_ids and record_id not in record_ids:
            continue
        path = gold_path(instance, record_id)
        if path.exists() and (read_json(path).get("verified") or not overwrite):
            continue
        doc_names, text = documentation_prompt(source)
        dictionary: dict[str, Any] = {"files": {}, "notes": "no documentation naming the columns"}
        if text:
            file_list = "\n".join(f"- {f['file_key']}" for f in source["files"])
            dictionary = ask_json(EXTRACTION_PROMPT.format(file_list=file_list, readme=text), max_tokens=60000)
        write_json(path, {
            "record_id": record_id,
            "domain": source["domain"],
            "files": [{"dataset_id": f["dataset_id"], "file_key": f["file_key"]} for f in source["files"]],
            "curator": source["curator"],
            "readme_file": ", ".join(doc_names) or None,
            "readme_truncated": len(text) >= MAX_README_CHARS,
            "dictionary": dictionary.get("files") or {},
            "extraction_notes": dictionary.get("notes", ""),
            # Set to true by hand after checking `dictionary` against the documentation.
            "verified": False,
        })
        written.append(record_id)
        print(f"gold {instance}/{record_id}: docs={doc_names} "
              f"columns={sum(len(v) for v in (dictionary.get('files') or {}).values())}")
    return written


async def build_gold(record_ids: list[str] | None = None, overwrite: bool = False) -> list[str]:
    files_by_record = indexed_uv_files()
    written = []
    async with async_client(60) as client:
        for record_id, files in sorted(files_by_record.items()):
            if record_ids and record_id not in record_ids:
                continue
            path = GOLD_DIR / f"{record_id}.json"
            if path.exists() and not overwrite:
                continue  # never clobber a hand-verified file
            response = await client.get(f"https://{UV_DOMAIN}/api/records/{record_id}")
            response.raise_for_status()
            record = response.json()
            readme_key, readme = await fetch_readme(client, record)

            dictionary: dict[str, Any] = {"files": {}, "notes": "no README"}
            if readme:
                file_list = "\n".join(f"- {f['file_key']}" for f in files)
                dictionary = ask_json(EXTRACTION_PROMPT.format(file_list=file_list, readme=readme[:MAX_README_CHARS]))

            write_json(path, {
                "record_id": record_id,
                "files": files,
                "curator": curator_fields(record),
                "readme_file": readme_key,
                "readme_truncated": len(readme) > MAX_README_CHARS,
                "dictionary": dictionary.get("files") or {},
                "extraction_notes": dictionary.get("notes", ""),
                # Set to true by hand after checking `dictionary` against the README.
                "verified": False,
            })
            written.append(record_id)
            print(f"gold {record_id}: readme={readme_key} dictionary_files={len(dictionary.get('files') or {})} "
                  f"columns={sum(len(v) for v in (dictionary.get('files') or {}).values())}")
    return written


def main(record_ids: list[str] | None = None, overwrite: bool = False, instances: list[str] | None = None) -> None:
    if instances:  # from `run scan` output
        for instance in instances:
            build_gold_from_sources(instance, record_ids, overwrite)
    else:  # legacy: first README only, UltraViolet only
        asyncio.run(build_gold(record_ids, overwrite))
