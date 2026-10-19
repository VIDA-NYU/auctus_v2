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
import os
import re
import zipfile
from typing import Any
from urllib.parse import quote

import httpx

from crawlers.invenio.archives import (
    MAX_ARCHIVE_MEMBERS,
    SMALL_ARCHIVE_BYTES,
    member_dataset_id,
    member_extension,
    open_remote_zip,
    select_archive_members,
)
from crawlers.invenio.http import async_client

LOGGER = logging.getLogger(__name__)

TABULAR_EXTENSIONS = {"csv", "tsv", "xlsx", "xls", "parquet", "csv.gz", "tsv.gz"}
# Zips: small ones become one archive job (downloaded); large ones are listed in
# place and each selected tabular member becomes its own job (see archives.py).
ARCHIVE_EXTENSIONS = {"zip"}
MAX_ARCHIVE_MEMBERS_PER_RECORD = int(os.getenv("MAX_ARCHIVE_MEMBERS_PER_RECORD", "50"))
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
    return member_extension(key)


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
    """One discovery entry per public tabular file or zip archive in the record."""
    if (record.get("access") or {}).get("files") != "public":
        return []

    record_summary = summarize_record(record, domain)
    entries = []
    for key, entry in ((record.get("files") or {}).get("entries") or {}).items():
        extension = file_extension(key)
        if extension not in TABULAR_EXTENSIONS | ARCHIVE_EXTENSIONS:
            continue
        if extension not in ARCHIVE_EXTENSIONS and entry.get("size", 0) > max_file_bytes:
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


def archive_member_entries(entry: dict[str, Any], archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    """One job per selected tabular member of a (large) archive entry."""
    members, total = select_archive_members(archive)
    archive_info = {"dataset_id": entry["dataset_id"], "file_key": entry["file_key"], "url": entry["file_url"],
                    "size": entry["file_size"], "tabular_members": total, "profiled_members": len(members)}
    return [{
        **entry,
        "dataset_id": member_dataset_id(entry["dataset_id"], info.filename),
        "file_key": f"{entry['file_key']}/{info.filename}",
        "file_extension": member_extension(info.filename),
        "file_size": info.file_size,
        "zip_member": info.filename,
        "archive": archive_info,
    } for info in members]


def expand_archive(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """List an archive in place (HTTP Range) and turn it into jobs.

    No tabular members: no job. Small archive: one job that downloads it and
    profiles its members. Large archive: one job per selected member, each of
    which reads only that member's bytes.
    """
    try:
        with open_remote_zip(entry["file_url"], entry["file_size"]) as archive:
            members = archive_member_entries(entry, archive)
    except Exception as exc:  # split archives, unsupported compression, network errors
        LOGGER.warning("Could not list archive %s: %r", entry["file_key"], exc)
        return []
    if not members:
        return []
    if (entry.get("file_size") or 0) < SMALL_ARCHIVE_BYTES:
        return [{**entry, "archive_tabular_members": members[0]["archive"]["tabular_members"]}]
    return members


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
    # InvenioRDM indexes the last extension only (data.csv.gz -> "gz").
    searchable = {ext.rsplit(".", 1)[-1] for ext in TABULAR_EXTENSIONS | ARCHIVE_EXTENSIONS}
    type_filter = "files.types:(" + " OR ".join(sorted(searchable)) + ")"
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
                # Tile sets come split over many zips (an UltraViolet record has ~35 zips
                # of near-identical Parquet tiles), so the member cap is per record,
                # shared across its archives; once spent, its other zips aren't listed.
                # Trade-off: a record that is a collection of different datasets (UAE Open
                # Data, 67 tables) is capped too. Grouping same-schema members into one
                # logical dataset would remove the need for a cap.
                budget = MAX_ARCHIVE_MEMBERS_PER_RECORD
                for entry in file_entries(record, domain, max_file_bytes):
                    if entry["file_extension"] not in ARCHIVE_EXTENSIONS:
                        discovered.append(entry)
                        continue
                    if budget <= 0:
                        continue
                    jobs = await asyncio.to_thread(expand_archive, entry)
                    if jobs and not jobs[0].get("zip_member"):  # one small-archive job
                        budget -= min(jobs[0].get("archive_tabular_members", 0), MAX_ARCHIVE_MEMBERS)
                        discovered.extend(jobs)
                    else:
                        discovered.extend(jobs[:budget])
                        budget -= len(jobs[:budget])

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
