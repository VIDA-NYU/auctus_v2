"""Offline tests for run provenance (no network, no LLM).

The property under test is narrow but load-bearing: an artifact must name the
code that produced it, for *both* repositories the pipeline spans. The failure
this guards against is silent — the same ``eval/`` scripts beside a different
``auctus_v2`` checkout produce different profiles, queries and qrels with no
error, so a wrong or missing version is invisible in the results themselves.

Run: python -m eval.test_provenance   (or via pytest)
"""

from __future__ import annotations

from pathlib import Path

import storage

from eval import provenance
from eval.provenance import code_version

_KEYS = {"commit", "branch", "dirty"}


def test_shape_is_complete_for_both_repositories() -> None:
    v = code_version()
    assert set(v) == {"auctus", "eval"}, v
    for name, entry in v.items():
        assert set(entry) == _KEYS, (name, entry)


def test_auctus_entry_is_resolved_through_storage_not_this_file() -> None:
    """The symlink trap.

    Under the layout the benchmark repo's README documents, ``eval/`` is
    symlinked in from a separate checkout, so ``Path(__file__).resolve()`` lands
    in the *benchmark* repo. Resolving the dependency's version from there would
    record the wrong repository under the name ``auctus`` — a wrong answer that
    looks exactly like a right one. Assert the directory actually consulted.
    """
    assert provenance._storage_dir() == Path(storage.__file__).resolve().parent


def test_missing_repository_yields_nulls_not_an_exception() -> None:
    """Provenance recording must never be able to fail a scoring run."""
    entry = provenance._repo_version(Path("/"))
    assert entry == {"commit": None, "branch": None, "dirty": None}, entry


def test_absent_git_yields_nulls_not_an_exception(monkeypatch=None) -> None:
    """Same guarantee when ``git`` itself is unavailable."""
    original = provenance._git
    provenance._git = lambda args, cwd: None
    try:
        assert code_version()["eval"]["commit"] is None
    finally:
        provenance._git = original


def test_dirty_is_a_bool_whenever_a_commit_was_found() -> None:
    """A commit id off a modified tree names code that did not run.

    ``dirty`` is None only when the commit is unknown; whenever a commit is
    recorded the flag must be decisive, or the id reads as reproducible when it
    is not.
    """
    for name, entry in code_version().items():
        if entry["commit"] is None:
            assert entry["dirty"] is None, (name, entry)
        else:
            assert isinstance(entry["dirty"], bool), (name, entry)


def test_clean_and_unknown_are_different_values() -> None:
    """The flag must be able to reassure, not only to warn.

    ``dirty`` was previously routed through ``_git``, whose ``stdout.strip() or
    None`` convention made a clean tree (empty ``git status`` output) and a
    failed git call indistinguishable — both ``None``. A reader of an artifact
    then could not tell "this commit is the code that ran" from "no idea".

    Asserted structurally rather than by demanding ``False`` here, since the
    working tree this runs in may legitimately be modified: inside a real
    checkout the answer is decisive either way, outside one it is None.
    """
    inside = provenance._dirty(Path(__file__).resolve().parent)
    assert isinstance(inside, bool), inside

    outside = provenance._dirty(Path("/"))
    assert outside is None, outside


def test_every_offline_writer_imports_it() -> None:
    """The four artifact writers must all record it, not just some.

    A partially-covered pipeline is worse than an uncovered one: it invites the
    assumption that an artifact without the block was produced by the same code
    as one with it.
    """
    from eval import build_pool, generate_queries, judge_qrels, run_matrix

    for module in (generate_queries, build_pool, judge_qrels, run_matrix):
        assert module.code_version is code_version, module.__name__


def main() -> int:
    test_shape_is_complete_for_both_repositories()
    test_auctus_entry_is_resolved_through_storage_not_this_file()
    test_missing_repository_yields_nulls_not_an_exception()
    test_absent_git_yields_nulls_not_an_exception()
    test_dirty_is_a_bool_whenever_a_commit_was_found()
    test_clean_and_unknown_are_different_values()
    test_every_offline_writer_imports_it()
    print("OK: artifacts name both repositories' code versions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
