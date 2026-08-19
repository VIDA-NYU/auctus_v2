"""Offline tests for generate_variations.py (variation-track-known-item
tasks.md §3) — no LLM, no network. The module-level ``complete`` is
monkeypatched with a canned response queue, same pattern as
test_generate_queries_cells.py.

Run: python -m eval.test_generate_variations   (or via pytest)
"""

from __future__ import annotations

import json

import eval.generate_variations as gv

NEUTRAL_DOC = {"title": "unused", "profiler_metadata": {"columns": []}}

BASE_TOPIC = {
    "query_id": "d1_topic_keyword", "text": "restaurant inspections",
    "facet": "topic", "query_class": "keyword", "source_dataset_id": "d1",
}
BASE_TEMPORAL = {
    "query_id": "d1_temporal_keyword", "text": "taxi trips 2019",
    "facet": "temporal", "query_class": "keyword", "source_dataset_id": "d1",
}
BASE_COMPOSITE = {
    "query_id": "d1_composite_keyword", "text": "taxi trips brooklyn 2019",
    "facet": "composite", "query_class": "keyword", "source_dataset_id": "d1",
}
BASE_STATISTICAL_IA = {
    "query_id": "d1_statistical_keyword", "text": "individual airport records",
    "facet": "statistical", "query_class": "keyword", "source_dataset_id": "d1",
    "statistical_subtype": "i-a",
}
BASE_STATISTICAL_IB = {
    "query_id": "d1_statistical_keyword", "text": "counts in the range of 0 to 15",
    "facet": "statistical", "query_class": "keyword", "source_dataset_id": "d1",
    "statistical_subtype": "i-b",
}


class _FakeQueue:
    """Feeds canned response strings to eval.generate_variations.complete, in
    order, one per call. Raises if exhausted (a test bug, not a code bug)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, client, prompt, temperature=0.0):
        self.calls += 1
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("fake LLM queue exhausted — test provided too few responses")
        return self._responses.pop(0)


def _patch_complete(saved, responses):
    queue = _FakeQueue(responses)
    saved["complete"] = gv.complete
    gv.complete = queue
    return queue


def _unpatch_complete(saved):
    gv.complete = saved["complete"]


def test_ok_response_is_accepted_topic_no_constraint():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "variations": ["health inspection scores", "eatery violation records",
                                    "food safety grades"]}),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_TOPIC, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result == {
        "status": "ok",
        "variations": ["health inspection scores", "eatery violation records", "food safety grades"],
        "constraint_substrings": None,
    }


def test_facet_mismatch_retries_once_then_fails():
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",  # wrong facet both times
                    "variations": ["a", "b", "c"]}),
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "variations": ["a", "b", "c"]}),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_TEMPORAL, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "failed"
    assert "facet/class mismatch" in result["reason"]
    assert queue.calls == 2  # exactly one retry


def test_wrong_variation_count_fails():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword", "variations": ["only", "two"]}),
        json.dumps({"facet": "topic", "query_class": "keyword", "variations": ["only", "two"]}),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_TOPIC, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "failed"
    assert "expected 3 variations" in result["reason"]


def test_constraint_substring_preserved_across_all_rewrites_is_accepted():
    saved = {}
    _patch_complete(saved, [
        json.dumps({
            "facet": "temporal", "query_class": "keyword",
            "constraint_substrings": ["2019"],
            "variations": ["cab rides in 2019", "for-hire trips from 2019", "yellow cab logs, 2019"],
        }),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_TEMPORAL, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "ok"
    assert result["constraint_substrings"] == ["2019"]


def test_constraint_substring_missing_from_a_rewrite_retries_then_fails():
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({
            "facet": "temporal", "query_class": "keyword",
            "constraint_substrings": ["2019"],
            # third rewrite silently drops the year -- constraint not preserved
            "variations": ["cab rides in 2019", "for-hire trips from 2019", "yellow cab logs"],
        }),
        json.dumps({
            "facet": "temporal", "query_class": "keyword",
            "constraint_substrings": ["2019"],
            "variations": ["cab rides in 2019", "for-hire trips from 2019", "yellow cab logs"],
        }),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_TEMPORAL, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "failed"
    assert "constraint substring" in result["reason"]
    assert queue.calls == 2


def test_composite_requires_at_least_two_constraint_substrings():
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({
            "facet": "composite", "query_class": "keyword",
            "constraint_substrings": ["brooklyn"],  # only one -- composite needs >= 2
            "variations": ["brooklyn cab data 2019", "brooklyn taxi logs 2019", "brooklyn rides, 2019"],
        }),
        json.dumps({
            "facet": "composite", "query_class": "keyword",
            "constraint_substrings": ["brooklyn"],
            "variations": ["brooklyn cab data 2019", "brooklyn taxi logs 2019", "brooklyn rides, 2019"],
        }),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_COMPOSITE, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "failed"
    assert "constraint_substrings" in result["reason"]
    assert queue.calls == 2


def test_composite_with_two_constraint_substrings_is_accepted():
    saved = {}
    _patch_complete(saved, [
        json.dumps({
            "facet": "composite", "query_class": "keyword",
            "constraint_substrings": ["brooklyn", "2019"],
            "variations": ["brooklyn cab data 2019", "brooklyn taxi logs, 2019", "2019 rides in brooklyn"],
        }),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_COMPOSITE, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "ok"
    assert result["constraint_substrings"] == ["brooklyn", "2019"]


def test_constraint_check_is_case_insensitive():
    saved = {}
    _patch_complete(saved, [
        json.dumps({
            "facet": "temporal", "query_class": "keyword",
            "constraint_substrings": ["2019"],
            "variations": ["Cab rides in 2019", "For-hire trips, 2019", "Logs from 2019"],
        }),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_TEMPORAL, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "ok"


def test_statistical_i_a_requires_no_constraint_substring():
    """2026-08-18 refinement: i-a constrains record GRAIN, not a bare token --
    exempt from the constraint check, same as `topic`. A response with no
    `constraint_substrings` field at all must still be accepted."""
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "variations": ["single airport entries", "one row per airport",
                                    "airport-level rows"]}),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_STATISTICAL_IA, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "ok"
    assert result["constraint_substrings"] is None


def test_statistical_i_b_still_requires_constraint_substring():
    """i-b (a numeric value span) is unaffected by the i-a exemption -- the
    check still applies and still retries-then-fails when it's missing."""
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "variations": ["low counts", "few occurrences", "small tallies"]}),
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "variations": ["low counts", "few occurrences", "small tallies"]}),
    ])
    try:
        result = gv.generate_variations_for_query(None, BASE_STATISTICAL_IB, "profile", "sample")
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "failed"
    assert "constraint_substrings" in result["reason"]
    assert queue.calls == 2


# --- run()'s exclusion accounting (Decision 5) -------------------------------


class _FakeOSClient:
    def get(self, index, id, _source):
        return {"_source": dict(NEUTRAL_DOC)}


def test_run_excludes_a_query_with_no_source_dataset_id():
    saved = {}
    _patch_complete(saved, [])
    try:
        base_no_source = {**BASE_TOPIC, "source_dataset_id": None}
        result = gv.run(None, _FakeOSClient(), None, [base_no_source], corpus_ids={"d1"})
    finally:
        _unpatch_complete(saved)
    assert result["excluded_no_source"] == [{"query_id": "d1_topic_keyword"}]
    assert result["variations"] == []


def test_run_excludes_a_query_whose_dataset_is_not_in_the_corpus_frame():
    saved = {}
    _patch_complete(saved, [])
    try:
        result = gv.run(None, _FakeOSClient(), None, [BASE_TOPIC], corpus_ids={"some_other_dataset"})
    finally:
        _unpatch_complete(saved)
    assert result["excluded_not_in_frame"] == [{"query_id": "d1_topic_keyword", "source_dataset_id": "d1"}]
    assert result["variations"] == []


def test_run_produces_three_variation_records_per_successful_base_query():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "variations": ["a inspections", "b violations", "c grades"]}),
    ])
    try:
        result = gv.run(None, _FakeOSClient(), None, [BASE_TOPIC], corpus_ids={"d1"})
    finally:
        _unpatch_complete(saved)
    assert len(result["variations"]) == 3
    ids = {v["variation_id"] for v in result["variations"]}
    assert ids == {"d1_topic_keyword_var0", "d1_topic_keyword_var1", "d1_topic_keyword_var2"}
    for v in result["variations"]:
        assert v["base_query_id"] == "d1_topic_keyword"
        assert v["source_dataset_id"] == "d1"
        assert v["facet"] == "topic"
        assert v["query_class"] == "keyword"


def test_run_records_a_generation_failure_distinctly_from_exclusions():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "wrong_facet", "query_class": "keyword", "variations": ["a", "b", "c"]}),
        json.dumps({"facet": "wrong_facet", "query_class": "keyword", "variations": ["a", "b", "c"]}),
    ])
    try:
        result = gv.run(None, _FakeOSClient(), None, [BASE_TOPIC], corpus_ids={"d1"})
    finally:
        _unpatch_complete(saved)
    assert result["variations"] == []
    assert result["excluded_no_source"] == []
    assert result["excluded_not_in_frame"] == []
    assert len(result["generation_failures"]) == 1
    assert result["generation_failures"][0]["query_id"] == "d1_topic_keyword"


def run_tests() -> int:
    test_ok_response_is_accepted_topic_no_constraint()
    test_facet_mismatch_retries_once_then_fails()
    test_wrong_variation_count_fails()
    test_constraint_substring_preserved_across_all_rewrites_is_accepted()
    test_constraint_substring_missing_from_a_rewrite_retries_then_fails()
    test_composite_requires_at_least_two_constraint_substrings()
    test_composite_with_two_constraint_substrings_is_accepted()
    test_constraint_check_is_case_insensitive()
    test_statistical_i_a_requires_no_constraint_substring()
    test_statistical_i_b_still_requires_constraint_substring()
    test_run_excludes_a_query_with_no_source_dataset_id()
    test_run_excludes_a_query_whose_dataset_is_not_in_the_corpus_frame()
    test_run_produces_three_variation_records_per_successful_base_query()
    test_run_records_a_generation_failure_distinctly_from_exclusions()
    print("OK: variation generation accept/retry/fail contract, facet-constraint "
          "substring survival (single-facet and composite's >=2 rule), case-insensitive "
          "matching, and run()'s three-way exclusion accounting (no-source, not-in-frame, "
          "generation-failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
