"""Build the hand-metadata ("gold") side.

For every UltraViolet record that Auctus indexed:
- record-level gold straight from the UltraViolet API: curator description,
  subjects, dates, locations (no extraction, these are typed fields);
- column-level gold from the record's README data dictionary. Dictionaries come in
  several formats (HTML tables, grid tables, indented lists), so an LLM extracts them
  into JSON, and a person must check each file against the README and set
  "verified": true before it counts.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from crawlers.invenio.crawler import html_to_text
from crawlers.invenio.http import async_client
from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

from eval.description_eval.common import GOLD_DIR, UV_DOMAIN, ask_json, write_json

README_TEXT_EXTENSIONS = {"md", "txt", ""}
MAX_README_CHARS = 120_000

EXTRACTION_PROMPT = """You are extracting a data dictionary from a dataset README written by a librarian.

The record contains these tabular files:
{file_list}

README:
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


def main(record_ids: list[str] | None = None, overwrite: bool = False) -> None:
    asyncio.run(build_gold(record_ids, overwrite))
