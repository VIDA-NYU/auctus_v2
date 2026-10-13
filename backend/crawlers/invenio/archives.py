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
import zipfile
from contextlib import contextmanager
from typing import Iterator

ARCHIVE_TABULAR_EXTENSIONS = {"csv", "tsv", "xlsx", "xls"}
SPREADSHEET_EXTENSIONS = {"xlsx", "xls"}
SMALL_ARCHIVE_BYTES = int(os.getenv("SMALL_ARCHIVE_BYTES", str(20 * 1024 * 1024)))
MAX_ARCHIVE_MEMBERS = int(os.getenv("MAX_ARCHIVE_MEMBERS", "25"))
# Spreadsheet members are read whole; pandas inflates them several-fold in memory.
MAX_SPREADSHEET_MEMBER_BYTES = 25 * 1024 * 1024
RANGE_BLOCK_BYTES = 256 * 1024


def member_extension(name: str) -> str:
    base = name.rsplit("/", 1)[-1].lower()
    return base.rsplit(".", 1)[-1] if "." in base else ""


def member_dataset_id(archive_dataset_id: str, member: str) -> str:
    """Stable id for one member of an archive (crawler and transformer must agree)."""
    return f"{archive_dataset_id}:{hashlib.sha1(member.encode()).hexdigest()[:12]}"


def select_archive_members(archive: zipfile.ZipFile,
                           limit: int = MAX_ARCHIVE_MEMBERS) -> tuple[list[zipfile.ZipInfo], int]:
    """(members to profile, number of tabular members) for a zip archive.

    Skips macOS resource forks, hidden and Office lock files, and spreadsheets too
    large to load. Archives can hold thousands of tables (e.g. tile sets), so at most
    `limit` are profiled, in name order; the total is kept for later grouping.
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
    tabular.sort(key=lambda i: i.filename)
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
