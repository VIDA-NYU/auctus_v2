"""Async InvenioRDM crawler for discovering public tabular files.

Pages through an instance's /api/records search and returns one entry per public
tabular file. Each entry carries the parent record's metadata so the worker can
index the file without another round trip, and so file-level documents keep
their record context for later compendium work.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx

from crawlers.invenio.http import async_client

LOGGER = logging.getLogger(__name__)

TABULAR_EXTENSIONS = {"csv", "tsv", "xlsx", "xls"}
DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_FILE_BYTES = 200 * 1024 * 1024
TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(value: str | None) -> str:
    """InvenioRDM descriptions are HTML; index them as plain text."""
    if not value:
        return ""
    text = TAG_RE.sub(" ", value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def file_extension(key: str) -> str:
    return key.rsplit(".", 1)[-1].lower() if "." in key else ""


def summarize_record(record: dict[str, Any], domain: str) -> dict[str, Any]:
    """The record-level fields every file document inherits."""
    metadata = record.get("metadata") or {}
    record_id = record["id"]
    doi = (record.get("pids") or {}).get("doi", {}).get("identifier")
    return {
        "record_id": record_id,
        "title": metadata.get("title") or record_id,
        "description": html_to_text(metadata.get("description")),
        "doi": doi,
        "landing_url": f"https://{domain}/records/{record_id}",
        "publication_date": metadata.get("publication_date"),
        "resource_type": (metadata.get("resource_type") or {}).get("id"),
        "creators": [
            (c.get("person_or_org") or {}).get("name")
            for c in metadata.get("creators") or []
            if (c.get("person_or_org") or {}).get("name")
        ],
        "subjects": [s["subject"] for s in metadata.get("subjects") or [] if s.get("subject")],
        "license": [r.get("id") or (r.get("title") or {}).get("en") for r in metadata.get("rights") or []],
        "publisher": metadata.get("publisher") or domain,
        "access": (record.get("access") or {}).get("files"),
        "updated": record.get("updated"),
    }


def file_entries(record: dict[str, Any], domain: str, max_file_bytes: int) -> list[dict[str, Any]]:
    """One discovery entry per public tabular file in the record."""
    if (record.get("access") or {}).get("files") != "public":
        return []

    record_summary = summarize_record(record, domain)
    entries = []
    for key, entry in ((record.get("files") or {}).get("entries") or {}).items():
        extension = file_extension(key)
        if extension not in TABULAR_EXTENSIONS:
            continue
        if entry.get("size", 0) > max_file_bytes:
            LOGGER.info("Skipping %s/%s: %d bytes exceeds limit", record["id"], key, entry["size"])
            continue
        file_id = entry.get("id") or key
        entries.append({
            # Prefixed with the domain so records from different instances never collide.
            "dataset_id": f"{domain}:{record['id']}:{file_id}",
            "file_key": key,
            "file_extension": extension,
            "file_size": entry.get("size"),
            "file_url": f"https://{domain}/api/records/{record['id']}/files/{quote(key)}/content",
            "source_updated_at": record.get("updated"),
            "record": record_summary,
        })
    return entries


async def discover_invenio_datasets(
    domain: str,
    limit: int = 10,
    query: str = "",
    page_size: int = DEFAULT_PAGE_SIZE,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    http_timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """Return up to `limit` public tabular files from an InvenioRDM instance.

    `query` is a Lucene query string; it is ANDed with a tabular file-type filter so
    the crawl only pages through records worth profiling. Zenodo needs a narrow
    query: it has ~184k datasets with tabular files and allows 25 results per page
    and 30 requests per minute anonymously.
    """
    type_filter = "files.types:(" + " OR ".join(sorted(TABULAR_EXTENSIONS)) + ")"
    full_query = f"({query}) AND {type_filter}" if query else type_filter
    url: str | None = f"https://{domain}/api/records"
    params: dict[str, Any] | None = {"q": full_query, "size": page_size, "sort": "newest"}

    discovered: list[dict[str, Any]] = []
    async with async_client(http_timeout_seconds) as client:
        while url and len(discovered) < limit:
            try:
                response = await client.get(url, params=params)
                if response.status_code == 429:
                    wait = int(response.headers.get("retry-after", 60))
                    LOGGER.info("Rate limited by %s; waiting %d s", domain, wait)
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                page = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                LOGGER.error("Failed to list records from %s: %s", domain, exc)
                break

            for record in page.get("hits", {}).get("hits", []):
                discovered.extend(file_entries(record, domain, max_file_bytes))

            # links.next already carries the query string.
            url, params = (page.get("links") or {}).get("next"), None

    return discovered[:limit]


async def main() -> None:
    """Smoke test against NYU UltraViolet."""
    logging.basicConfig(level=logging.INFO)
    for entry in await discover_invenio_datasets("ultraviolet.library.nyu.edu", limit=5):
        print(entry["dataset_id"], entry["file_key"], entry["record"]["title"][:60])


if __name__ == "__main__":
    asyncio.run(main())
