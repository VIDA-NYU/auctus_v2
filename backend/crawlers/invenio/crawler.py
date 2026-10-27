"""Async InvenioRDM crawler for discovering public tabular files.

Pages through an instance's /api/records search and returns one entry per public
tabular file. Each entry carries the parent record's metadata so the worker can
index the file without another round trip, and so file-level documents keep
their record context for later compendium work.
"""

from __future__ import annotations

import asyncio
import hashlib
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
    member_pattern,
    open_remote_zip,
    tabular_members,
)
from crawlers.invenio.codebook import MAX_DOC_BYTES, is_documentation
from crawlers.invenio.http import async_client

LOGGER = logging.getLogger(__name__)

TABULAR_EXTENSIONS = {"csv", "tsv", "xlsx", "xls", "parquet", "csv.gz", "tsv.gz"}
# Zips: small ones become one archive job (downloaded); large ones are listed in
# place and each selected tabular member becomes its own job (see archives.py).
ARCHIVE_EXTENSIONS = {"zip"}
# Members of large zips that share a name pattern (part.#.parquet) across a record
# are one table split into parts: they become one grouped dataset, profiled from a
# sample of parts (see transformer.profile_member_group). Other members of large
# zips are profiled one by one, at most MAX_ARCHIVE_MEMBERS per zip and
# MAX_ARCHIVE_MEMBERS_PER_RECORD per record. Small zips aren't capped per record.
MAX_ARCHIVE_MEMBERS_PER_RECORD = int(os.getenv("MAX_ARCHIVE_MEMBERS_PER_RECORD", "50"))
MIN_GROUP_MEMBERS = int(os.getenv("MIN_GROUP_MEMBERS", "5"))
MAX_GROUP_SAMPLES = int(os.getenv("MAX_GROUP_SAMPLES", "40"))
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
        # Codebooks / data dictionaries / READMEs: value labels mark coded columns
        # categorical when the record's tables are profiled (see codebook.py).
        "documentation": [
            {"key": key, "url": f"https://{domain}/api/records/{record_id}/files/{quote(key)}/content",
             "size": entry.get("size")}
            for key, entry in ((record.get("files") or {}).get("entries") or {}).items()
            if (record.get("access") or {}).get("files") == "public" and is_documentation(key)
            and (entry.get("size") or 0) <= MAX_DOC_BYTES
        ],
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
    """One entry per tabular member of a (large) archive entry; not yet capped."""
    members = tabular_members(archive)
    archive_info = {"dataset_id": entry["dataset_id"], "file_key": entry["file_key"], "url": entry["file_url"],
                    "size": entry["file_size"], "tabular_members": len(members)}
    return [{
        **entry,
        "dataset_id": member_dataset_id(entry["dataset_id"], info.filename),
        "file_key": f"{entry['file_key']}/{info.filename}",
        "file_extension": member_extension(info.filename),
        "file_size": info.file_size,
        "zip_member": info.filename,
        "archive": archive_info,
    } for info in members]


def expand_archive(entry: dict[str, Any], unlisted: set[str] | None = None) -> list[dict[str, Any]]:
    """List an archive in place (HTTP Range).

    No tabular members: []. Small archive: one job that downloads it and profiles its
    members. Large archive: one entry per tabular member, to be grouped or capped by
    plan_member_jobs. Archives that can't be listed are added to `unlisted`, so their
    already-indexed members aren't treated as stale.
    """
    try:
        with open_remote_zip(entry["file_url"], entry["file_size"]) as archive:
            members = archive_member_entries(entry, archive)
    except Exception as exc:  # split archives, unsupported compression, network errors
        LOGGER.warning("Could not list archive %s: %r", entry["file_key"], exc)
        if unlisted is not None:
            unlisted.add(entry["dataset_id"])
        return []
    if not members:
        return []
    if (entry.get("file_size") or 0) < SMALL_ARCHIVE_BYTES:
        return [{**entry, "archive_tabular_members": members[0]["archive"]["tabular_members"]}]
    return members


def spread_sample(items: list[Any], k: int) -> list[Any]:
    """k items evenly spaced over `items`, first and last included."""
    if len(items) <= k:
        return list(items)
    if k <= 1:
        return items[:1]
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def group_dataset_id(member: dict[str, Any], pattern: str) -> str:
    """Stable id of a grouped dataset: the record's id prefix plus the pattern."""
    prefix = member["archive"]["dataset_id"].rsplit(":", 1)[0]  # domain:record
    return f"{prefix}:group:{hashlib.sha1(pattern.encode()).hexdigest()[:12]}"


def group_job(pattern: str, members: list[dict[str, Any]], max_samples: int = MAX_GROUP_SAMPLES) -> dict[str, Any]:
    """One job for all parts of a split table, with the parts to sample.

    Parts are sampled across every zip they're spread over (one per zip first, then
    evenly within zips), so coverage reaches every zip, e.g. every city of a
    per-city tile set.
    """
    archives: dict[str, dict[str, Any]] = {}
    by_archive: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        archive = member["archive"]
        archives.setdefault(archive["dataset_id"], {k: archive[k] for k in ("dataset_id", "file_key", "url", "size")})
        by_archive.setdefault(archive["dataset_id"], []).append(member)
    per_archive = max(1, max_samples // max(1, len(by_archive)))
    samples = []
    for archive_id, parts in by_archive.items():
        samples += [{"archive": archive_id, "member": m["zip_member"], "file_size": m["file_size"]}
                    for m in spread_sample(parts, per_archive)]
    samples = spread_sample(samples, max_samples)
    first = members[0]
    extension = first["file_extension"]
    return {
        **{k: v for k, v in first.items() if k not in ("zip_member", "archive")},
        "dataset_id": group_dataset_id(first, pattern),
        "file_key": f"{pattern} ({len(members)} parts in {len(archives)} zip{'s' if len(archives) > 1 else ''})",
        "file_extension": extension,
        "file_size": sum(m["file_size"] or 0 for m in members),
        "file_url": first["archive"]["url"],
        "member_group": {
            "pattern": pattern,
            "total_members": len(members),
            "total_bytes": sum(m["file_size"] or 0 for m in members),
            "archives": list(archives.values()),
            "samples": samples,
        },
    }


def plan_member_jobs(members: list[dict[str, Any]], budget: int = MAX_ARCHIVE_MEMBERS_PER_RECORD,
                     per_archive: int = MAX_ARCHIVE_MEMBERS,
                     min_group: int = MIN_GROUP_MEMBERS) -> list[dict[str, Any]]:
    """Jobs for the members of a record's large zips.

    Members sharing a name pattern and extension (at least `min_group` of them)
    become one group job; the rest are profiled one by one, capped per zip and per
    record.
    """
    patterns: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for member in members:
        patterns.setdefault((member_pattern(member["zip_member"]), member["file_extension"]), []).append(member)
    jobs, singles = [], []
    for (pattern, _ext), group in patterns.items():
        if len(group) >= min_group:
            jobs.append(group_job(pattern, group))
        else:
            singles += group
    per_zip: dict[str, int] = {}
    taken = 0
    for member in sorted(singles, key=lambda m: m["file_key"]):
        archive_id = member["archive"]["dataset_id"]
        if taken >= budget:
            break
        if per_zip.get(archive_id, 0) >= per_archive:
            continue
        per_zip[archive_id] = per_zip.get(archive_id, 0) + 1
        taken += 1
        jobs.append(member)
    return jobs


async def discover_invenio_datasets(
    domain: str,
    limit: int = 10,
    query: str = "",
    page_size: int = DEFAULT_PAGE_SIZE,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    http_timeout_seconds: float = 30.0,
    report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return up to `limit` public tabular files from an InvenioRDM instance.

    If `report` is given it gets "complete" (every page listed, nothing cut by
    `limit`) and "unlisted_archives" (zip ids that couldn't be listed); pruning
    stale docs is only safe on a complete crawl.

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
    unlisted: set[str] = set()
    failed = False
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
                failed = True
                break

            for record in page.get("hits", {}).get("hits", []):
                # Tile sets come split over many zips (an UltraViolet record has 35
                # zips of Parquet tiles, one per city), so members of all of a
                # record's large zips are planned together.
                large_members = []
                for entry in file_entries(record, domain, max_file_bytes):
                    if entry["file_extension"] not in ARCHIVE_EXTENSIONS:
                        discovered.append(entry)
                        continue
                    jobs = await asyncio.to_thread(expand_archive, entry, unlisted)
                    if jobs and not jobs[0].get("zip_member"):  # one small-archive job
                        discovered.extend(jobs)
                    else:
                        large_members.extend(jobs)
                discovered.extend(plan_member_jobs(large_members))

            # links.next already carries the query string.
            url, params = (page.get("links") or {}).get("next"), None

    if report is not None:
        report["complete"] = not failed and not url and len(discovered) <= limit
        report["unlisted_archives"] = unlisted
    return discovered[:limit]


def stale_doc_ids(indexed: list[dict[str, Any]], discovered: list[dict[str, Any]],
                  unlisted_archives: set[str] = frozenset()) -> list[str]:
    """Ids of indexed docs that a complete crawl no longer produces.

    `indexed` holds {"id", "archive_dataset_id"} per doc of one domain. A doc is
    kept if discovery produced its id, or it is a member of a small archive that
    was discovered (its members are chosen when the archive is profiled), one of a
    discovered group's per-schema docs, or a member of an archive that couldn't be
    listed this time. Everything else (deleted or
    restricted records, files removed from a record, members beyond the caps) is
    stale.
    """
    keep_ids = {entry["dataset_id"] for entry in discovered}
    keep_archives = {entry["dataset_id"] for entry in discovered
                     if (entry.get("file_extension") in ARCHIVE_EXTENSIONS and not entry.get("zip_member"))
                     or entry.get("member_group")}  # a group's per-schema docs
    keep_archives |= set(unlisted_archives)
    return [doc["id"] for doc in indexed
            if doc["id"] not in keep_ids and doc.get("archive_dataset_id") not in keep_archives]


async def main() -> None:
    """Smoke test against NYU UltraViolet."""
    logging.basicConfig(level=logging.INFO)
    for entry in await discover_invenio_datasets("ultraviolet.library.nyu.edu", limit=5):
        print(entry["dataset_id"], entry["file_key"], entry["record"]["title"][:60])


if __name__ == "__main__":
    asyncio.run(main())
