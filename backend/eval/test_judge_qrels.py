"""Offline tests for judge_qrels.py's exhaustive-chunked judging (no LLM, no
network) -- covers judge-exhaustive-chunked's design.md Decisions 1-4:
deterministic corpus chunking, reproducible-per-call order randomization
that never touches chunk membership, and whole-query failure on any one
chunk's failure with no partial grade leakage.

``corpus_chunks``, ``_shuffled``, and ``judge_query_chunked`` (with a fake
``judge_query_fn``) are pure/injectable, so this suite needs no OpenSearch,
MinIO, or LLM client at all.

Run: python -m eval.test_judge_qrels   (or via pytest)
"""

from __future__ import annotations

from pathlib import Path

from eval.judge_qrels import (
    DEFAULT_CORPUS_FRAME, _shuffled, assert_judge_seat_allowed, corpus_chunks,
    judge_query_chunked,
)
from eval.llm_client import CLAUDE_HAIKU, DEEPSEEK_V3, GEMINI_FLASH

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _fake_load(dataset_id: str) -> dict:
    return {"title": dataset_id, "profile": "p", "sample": "s"}


def test_corpus_chunks_covers_the_whole_corpus_disjointly() -> None:
    chunks = corpus_chunks(20)
    assert len(chunks) == 5
    assert all(len(c) == 20 for c in chunks)
    seen: set[str] = set()
    for c in chunks:
        assert seen.isdisjoint(c)  # no id appears in two chunks
        seen |= set(c)
    assert len(seen) == 100


def test_corpus_chunks_rejects_a_size_that_does_not_divide_evenly() -> None:
    try:
        corpus_chunks(30)
    except ValueError as exc:
        assert "does not evenly divide" in str(exc)
    else:
        raise AssertionError("expected ValueError for chunk_size=30 on a 100-id corpus")


def test_corpus_chunks_is_deterministic() -> None:
    """Membership is a pure function of the sorted id list -- two independent
    calls (as happen once per run, shared across every query and both judges)
    must agree exactly."""
    assert corpus_chunks(20) == corpus_chunks(20)


def test_shuffle_reproducible_with_same_seed_key() -> None:
    items = list(range(20))
    a = _shuffled(items, "seedA", "q1", 0)
    b = _shuffled(items, "seedA", "q1", 0)
    assert a == b


def test_shuffle_varies_by_chunk_index_and_query_id() -> None:
    items = list(range(20))
    base = _shuffled(items, "seedA", "q1", 0)
    diff_chunk = _shuffled(items, "seedA", "q1", 1)
    diff_query = _shuffled(items, "seedA", "q2", 0)
    assert base != diff_chunk
    assert base != diff_query


def test_shuffle_never_changes_membership() -> None:
    items = list(range(20))
    shuffled = _shuffled(items, "seedA", "q1", 3)
    assert sorted(shuffled) == sorted(items)


def test_judge_query_chunked_merges_all_five_chunks() -> None:
    chunks = corpus_chunks(20)

    def fake_judge_ok(client, query_text, items, model, seed=None, no_cache=False):
        return ({ds_id: 1 for ds_id, _ in items}, set(),
                {"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 0})

    grades, unverified, usage, failed = judge_query_chunked(
        None, "q1", "test query", chunks, _fake_load, model="m", order_seed="s1",
        judge_query_fn=fake_judge_ok,
    )
    assert failed is None
    assert len(grades) == 100  # every corpus dataset graded, none defaulted
    assert unverified == set()
    assert usage["prompt_tokens"] == 500  # 5 chunks x 100


def test_judge_query_chunked_one_failed_chunk_fails_the_whole_query() -> None:
    """design.md Decision 4: no partial-judgment state. A failure on chunk 3
    of 5 must not leave the first two chunks' grades in the qrels file."""
    chunks = corpus_chunks(20)
    calls = {"n": 0}

    def fake_judge_fail_on_3rd(client, query_text, items, model, seed=None, no_cache=False):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return ({ds_id: 1 for ds_id, _ in items}, set(),
                {"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 0})

    grades, unverified, usage, failed = judge_query_chunked(
        None, "q2", "test query", chunks, _fake_load, model="m", order_seed="s1",
        judge_query_fn=fake_judge_fail_on_3rd,
    )
    assert failed is not None
    assert "chunk 2/5" in failed  # 0-indexed, so the 3rd call is chunk_index 2
    assert grades == {}  # no partial grades from the 2 chunks that succeeded
    assert unverified == set()
    # usage from the 2 completed chunks is still counted -- cost accounting
    # must reflect money actually spent even on a failed query
    assert usage["prompt_tokens"] == 200


def test_judge_seat_allows_a_cross_lab_judge() -> None:
    """Both panel seats must pass against the gemini generator."""
    assert_judge_seat_allowed(CLAUDE_HAIKU, GEMINI_FLASH)
    assert_judge_seat_allowed(DEEPSEEK_V3, GEMINI_FLASH)


def test_judge_seat_refuses_the_generators_own_lab() -> None:
    """The finding-6 regression: §1g's constraint is unconditional, and the
    old `--model` default was the generator itself."""
    try:
        assert_judge_seat_allowed(GEMINI_FLASH, GEMINI_FLASH)
    except SystemExit as exc:
        assert "same lab" in str(exc)
    else:
        raise AssertionError("a same-lab judge was allowed")


def test_judge_seat_refuses_unknown_lineage() -> None:
    """Fails closed: an unrecorded lab cannot be shown to differ from the
    generator's, and this guard exists for the case nobody checked."""
    try:
        assert_judge_seat_allowed("@somewhere/never-seen-model", GEMINI_FLASH)
    except SystemExit as exc:
        assert "not recorded in MODEL_LAB" in str(exc)
    else:
        raise AssertionError("a model of unknown lineage was allowed")


def test_judge_seat_guard_follows_the_generator() -> None:
    """Derived, not hardcoded: re-seat the generator to an Anthropic model and
    the Anthropic judge becomes the refused one, with no edit to the guard."""
    assert_judge_seat_allowed(GEMINI_FLASH, CLAUDE_HAIKU)  # now allowed
    try:
        assert_judge_seat_allowed(CLAUDE_HAIKU, CLAUDE_HAIKU)
    except SystemExit:
        pass
    else:
        raise AssertionError("the guard did not follow the generator's lab")


def test_judge_query_chunked_default_corpus_frame_resolves_from_backend_cwd() -> None:
    """DEFAULT_CORPUS_FRAME is a relative path; confirm it resolves to the
    real, currently-tracked frame file rather than a stale/renamed one."""
    assert (_BACKEND_ROOT / DEFAULT_CORPUS_FRAME).exists()


def main() -> int:
    test_corpus_chunks_covers_the_whole_corpus_disjointly()
    test_corpus_chunks_rejects_a_size_that_does_not_divide_evenly()
    test_corpus_chunks_is_deterministic()
    test_shuffle_reproducible_with_same_seed_key()
    test_shuffle_varies_by_chunk_index_and_query_id()
    test_shuffle_never_changes_membership()
    test_judge_query_chunked_merges_all_five_chunks()
    test_judge_query_chunked_one_failed_chunk_fails_the_whole_query()
    test_judge_seat_allows_a_cross_lab_judge()
    test_judge_seat_refuses_the_generators_own_lab()
    test_judge_seat_refuses_unknown_lineage()
    test_judge_seat_guard_follows_the_generator()
    test_judge_query_chunked_default_corpus_frame_resolves_from_backend_cwd()
    print("OK: corpus chunking (coverage/disjointness/determinism/bad-size), "
          "order randomization (reproducible/varies/membership-preserving), "
          "chunked judging (full merge, whole-query failure on one bad chunk), "
          "judge-seat guard (cross-lab ok, same-lab/unknown refused, follows generator)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
