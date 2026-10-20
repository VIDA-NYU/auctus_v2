"""Zip archives in repositories: which members to profile, and reading them in place.

Large archives are never downloaded. fsspec opens the remote file as a seekable
file over HTTP Range requests, so zipfile only fetches the table of contents at
the end of the archive, and later only the bytes of the member being sampled
(a 17 GB UltraViolet archive lists in well under a second). Archives below
SMALL_ARCHIVE_BYTES are simply downloaded, which needs fewer requests.
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from contextlib import contextmanager
from typing import Iterator

ARCHIVE_TABULAR_EXTENSIONS = {"csv", "tsv", "xlsx", "xls", "parquet", "csv.gz", "tsv.gz"}
# Two-part extensions recognized as one (data.csv.gz -> "csv.gz").
COMPOUND_EXTENSIONS = ("csv.gz", "tsv.gz")
SPREADSHEET_EXTENSIONS = {"xlsx", "xls"}
SMALL_ARCHIVE_BYTES = int(os.getenv("SMALL_ARCHIVE_BYTES", str(20 * 1024 * 1024)))
MAX_ARCHIVE_MEMBERS = int(os.getenv("MAX_ARCHIVE_MEMBERS", "25"))
# Spreadsheet members are read whole; pandas inflates them several-fold in memory.
MAX_SPREADSHEET_MEMBER_BYTES = 25 * 1024 * 1024
RANGE_BLOCK_BYTES = 256 * 1024
# When a table exists in several formats under one name, profile only the first of
# these (UAE Open Data ships SocialAssistance.xlsx next to SocialAssistance.xls).
FORMAT_PREFERENCE = ("csv", "tsv", "csv.gz", "tsv.gz", "parquet", "xlsx", "xls")


def member_extension(name: str) -> str:
    """Lowercase extension of a file or archive member; compound ones like csv.gz kept whole."""
    base = name.rsplit("/", 1)[-1].lower()
    for compound in COMPOUND_EXTENSIONS:
        if base.endswith("." + compound):
            return compound
    return base.rsplit(".", 1)[-1] if "." in base else ""


def member_dataset_id(archive_dataset_id: str, member: str) -> str:
    """Stable id for one member of an archive (crawler and transformer must agree)."""
    return f"{archive_dataset_id}:{hashlib.sha1(member.encode()).hexdigest()[:12]}"


def member_pattern(name: str) -> str:
    """A member's base name with digit runs collapsed: tiles of one table share it.

    "04_Houston_TX/parquet_parts_gz/part.4000.parquet" -> "part.#.parquet",
    "oceania/R066_C162.parquet" -> "r#_c#.parquet".
    """
    return re.sub(r"\d+", "#", name.rsplit("/", 1)[-1].lower())


def drop_duplicate_formats(names: list[str]) -> set[str]:
    """Names to skip because the same table (same path minus extension) exists in a
    preferred format."""
    by_stem: dict[str, list[str]] = {}
    for name in names:
        ext = member_extension(name)
        stem = name[: -(len(ext) + 1)] if ext else name
        by_stem.setdefault(stem.lower(), []).append(name)
    rank = {ext: i for i, ext in enumerate(FORMAT_PREFERENCE)}
    skip = set()
    for group in by_stem.values():
        if len(group) > 1:
            keep = min(group, key=lambda n: rank.get(member_extension(n), len(rank)))
            skip.update(n for n in group if n != keep)
    return skip


def tabular_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Every tabular member of a zip worth profiling, in name order.

    Skips macOS resource forks, hidden and Office lock files, spreadsheets too large
    to load, and copies of a table in a less preferred format.
    """
    tabular = []
    for info in archive.infolist():
        base = info.filename.rsplit("/", 1)[-1]
        if info.is_dir() or info.filename.startswith("__MACOSX/") or base.startswith((".", "~$")):
            continue
        ext = member_extension(info.filename)
        if ext not in ARCHIVE_TABULAR_EXTENSIONS:
            continue
        if ext in SPREADSHEET_EXTENSIONS and info.file_size > MAX_SPREADSHEET_MEMBER_BYTES:
            continue
        tabular.append(info)
    skip = drop_duplicate_formats([i.filename for i in tabular])
    return sorted((i for i in tabular if i.filename not in skip), key=lambda i: i.filename)


def select_archive_members(archive: zipfile.ZipFile,
                           limit: int = MAX_ARCHIVE_MEMBERS) -> tuple[list[zipfile.ZipInfo], int]:
    """(members to profile, number of tabular members) for a zip archive.

    Archives can hold thousands of tables (e.g. tile sets), so at most `limit` are
    profiled, in name order; the total is kept.
    """
    tabular = tabular_members(archive)
    return tabular[:limit], len(tabular)


@contextmanager
def open_remote_zip(url: str, size: int) -> Iterator[zipfile.ZipFile]:
    """A ZipFile over a remote archive, reading only the byte ranges it needs."""
    import aiohttp
    import fsspec

    from crawlers.invenio.http import USER_AGENT, ssl_context

    async def get_client(**kwargs):
        # UltraViolet omits its intermediate certificate; reuse the connector's bundle.
        return aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context()),
                                     headers={"User-Agent": USER_AGENT}, **kwargs)

    fs = fsspec.filesystem("http", get_client=get_client, skip_instance_cache=True)
    with fs.open(url, "rb", size=size, block_size=RANGE_BLOCK_BYTES) as remote:
        with zipfile.ZipFile(remote) as archive:
            yield archive
