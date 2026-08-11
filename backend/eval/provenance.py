"""Which code produced this artifact.

The pipeline spans two repositories. ``eval/`` is published as
VIDA-NYU/NYC-Open-Data-Benchmark; the profile code it depends on —
``storage.arq_worker.build_profile_text``, which grounds ``ufd`` / ``sfd`` /
``llm_direct``, *is* the ``profile_only`` arm verbatim, and forms the neutral
bundle read by both the query generator and the judge — lives in ``auctus_v2``.
The same scripts beside a different ``auctus_v2`` checkout produce different
profiles, queries and qrels, silently. So both versions are recorded, not one.

The ``auctus_v2`` version is resolved through the *imported* ``storage`` package
rather than through ``__file__``. Under the layout ``eval/README.md`` documents,
``eval/`` is symlinked in from a separate benchmark-repo checkout, and
``Path(__file__).resolve()`` lands in that checkout — recording the benchmark
repo's commit under the name of the dependency. Resolving through
``storage.__file__`` always names the checkout that actually supplied the code.

Nothing here may fail a run: a missing git, a shallow export or a non-repo
directory yields nulls, never an exception.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_TIMEOUT = 5


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT
        )
    except Exception:
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def _repo_version(cwd: Path) -> dict:
    """commit / branch / dirty for the checkout containing ``cwd``.

    ``dirty`` is not merely informational: a commit id read off a modified tree
    names code that was not the code that ran, so recording the id without the
    flag would assert a reproducibility the artifact cannot support. It stays
    ``None`` when the commit is unknown, rather than defaulting to False.
    """
    commit = _git(["rev-parse", "--short", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def _storage_dir() -> Path | None:
    try:
        import storage
    except Exception:
        return None
    location = getattr(storage, "__file__", None)
    return Path(location).resolve().parent if location else None


def code_version() -> dict:
    """``{"auctus": {...}, "eval": {...}}`` — both repositories, never raising.

    Under an in-tree layout the two entries resolve to the same checkout and hold
    identical values; under a symlinked layout they differ, and the artifact
    records which pair produced it.
    """
    storage_dir = _storage_dir()
    return {
        "auctus": (
            _repo_version(storage_dir)
            if storage_dir
            else {"commit": None, "branch": None, "dirty": None}
        ),
        "eval": _repo_version(Path(__file__).resolve().parent),
    }
