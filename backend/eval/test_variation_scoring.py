"""Offline tests for variation_scoring.py (variation-track-known-item
tasks.md §3) — no LLM, no network. ``build_query_vector`` (which loads a real
local embedding model) is monkeypatched to a canned vector; the OpenSearch
client is a minimal in-memory stand-in, same style as test_run_matrix.py's
``_FakeOSClient``.

Run: python -m eval.test_variation_scoring   (or via pytest)
"""

from __future__ import annotations

import eval.variation_scoring as vs
from eval.run_matrix import ARMS
from storage.opensearch_client import DESCRIPTION_SOURCE_FIELDS

SFD_FIELD = DESCRIPTION_SOURCE_FIELDS["sfd"]
UFD_FIELD = DESCRIPTION_SOURCE_FIELDS["ufd"]


class _FakeOSClient:
    """Dispatches on body["query"] shape: `match` -> BM25 field channel
    (canned hits keyed by field name), `knn` -> vector channel (one canned
    list, arm-blind like the real channel)."""

    def __init__(self, hits_by_field: dict[str, list[tuple[str, float]]],
                 knn_hits: list[tuple[str, float]] | None = None):
        self.hits_by_field = hits_by_field
        self.knn_hits = knn_hits or []
        self.calls: list[dict] = []

    def search(self, index, body, size):
        query = body["query"]
        if "match" in query:
            field = next(iter(query["match"]))
            hits = self.hits_by_field.get(field, [])[:size]
        else:
            hits = self.knn_hits[:size]
        self.calls.append({"query": query, "size": size})
        return {"hits": {"hits": [{"_id": i, "_score": s} for i, s in hits]}}


def _patch_query_vector(saved):
    saved["build_query_vector"] = vs.build_query_vector
    vs.build_query_vector = lambda text: [0.0]


def _unpatch_query_vector(saved):
    vs.build_query_vector = saved["build_query_vector"]


def test_rank_orders_by_score_descending_ties_broken_by_id():
    scores = {"d3": 1.0, "d1": 2.0, "d2": 2.0}
    assert vs.rank(scores) == ["d1", "d2", "d3"]


def test_is_hit_true_when_source_in_top_k():
    assert vs.is_hit(["d2", "d1", "d3"], "d1", k=2) is True


def test_is_hit_false_when_source_outside_top_k():
    assert vs.is_hit(["d2", "d3", "d1"], "d1", k=2) is False


def test_is_hit_scores_a_miss_when_a_sibling_outranks_the_source_outside_k():
    """Spec scenario: a same-agency sibling ranks above the source, within the
    cutoff, and the source falls outside it -> a miss. No sibling/family
    notion is applied -- only the literal source_dataset_id counts."""
    ranked = ["sibling_dataset", "some_other", "d1"]  # d1 = source, rank 3
    assert vs.is_hit(ranked, "d1", k=2) is False


def test_score_variation_retrieves_title_and_knn_once_and_arm_per_arm():
    saved = {}
    _patch_query_vector(saved)
    try:
        client = _FakeOSClient(
            hits_by_field={
                "title": [("d1", 5.0)],
                SFD_FIELD: [("d1", 9.0), ("d2", 1.0)],
            },
            knn_hits=[("d1", 0.5)],
        )
        result = vs.score_variation(client, {"text": "restaurant inspections"}, size=10)
    finally:
        _unpatch_query_vector(saved)
    assert result["title_BM25"] == {"d1": 5.0}
    assert result["knn_score"] == {"d1": 0.5}
    assert set(result["arms"]) == set(ARMS)
    assert result["arms"]["sfd"]["ranked_isolated"] == ["d1", "d2"]
    assert result["arms"]["sfd"]["arm_BM25"] == {"d1": 9.0, "d2": 1.0}
    # an arm with no canned hits is an empty, not a crash
    assert result["arms"]["ufd"]["ranked_isolated"] == []


def test_run_mean_hit_at_k_across_two_variations():
    saved = {}
    _patch_query_vector(saved)
    try:
        # sfd hits d1 for variation 1 (d1 ranked first), misses for variation 2
        # (d1 absent entirely) -> mean hit@k for sfd = 0.5.
        client = _FakeOSClient(
            hits_by_field={SFD_FIELD: [("d1", 9.0)]},  # constant across both search calls
        )
        variations = [
            {"variation_id": "v0", "base_query_id": "b1", "source_dataset_id": "d1",
             "facet": "topic", "query_class": "keyword", "text": "x"},
            {"variation_id": "v1", "base_query_id": "b1", "source_dataset_id": "d2",  # different source -> miss under these hits
             "facet": "topic", "query_class": "keyword", "text": "y"},
        ]
        report = vs.run(client, variations, k=10, size=10)
    finally:
        _unpatch_query_vector(saved)
    assert report["mean_hit_at_k"]["sfd"] == 0.5
    assert report["n_variations_scored"] == 2
    assert report["n_base_queries"] == 1


def test_run_all_n_hit_requires_every_variation_in_the_group_to_hit():
    saved = {}
    _patch_query_vector(saved)
    try:
        client = _FakeOSClient(hits_by_field={SFD_FIELD: [("d1", 9.0)]})
        # Two base queries, each with 2 variations. b1's variations both hit
        # (source d1 both times); b2's variations: one hits, one misses.
        variations = [
            {"variation_id": "b1_v0", "base_query_id": "b1", "source_dataset_id": "d1",
             "facet": "topic", "query_class": "keyword", "text": "x"},
            {"variation_id": "b1_v1", "base_query_id": "b1", "source_dataset_id": "d1",
             "facet": "topic", "query_class": "keyword", "text": "x"},
            {"variation_id": "b2_v0", "base_query_id": "b2", "source_dataset_id": "d1",
             "facet": "topic", "query_class": "keyword", "text": "x"},
            {"variation_id": "b2_v1", "base_query_id": "b2", "source_dataset_id": "d9",  # miss
             "facet": "topic", "query_class": "keyword", "text": "x"},
        ]
        report = vs.run(client, variations, k=10, size=10)
    finally:
        _unpatch_query_vector(saved)
    # all_n_hit for sfd = mean over {b1: True, b2: False} = 0.5
    assert report["all_n_hit"]["sfd"] == 0.5
    # mean_hit_at_k for sfd = mean over 4 individual hits: True, True, True, False = 0.75
    assert report["mean_hit_at_k"]["sfd"] == 0.75


def test_run_by_class_and_by_facet_breakdown():
    saved = {}
    _patch_query_vector(saved)
    try:
        client = _FakeOSClient(hits_by_field={SFD_FIELD: [("d1", 9.0)]})
        variations = [
            {"variation_id": "v0", "base_query_id": "b1", "source_dataset_id": "d1",
             "facet": "topic", "query_class": "keyword", "text": "x"},
            {"variation_id": "v1", "base_query_id": "b2", "source_dataset_id": "d9",
             "facet": "temporal", "query_class": "describing", "text": "y"},
        ]
        report = vs.run(client, variations, k=10, size=10)
    finally:
        _unpatch_query_vector(saved)
    assert report["mean_hit_at_k_by_class"]["sfd"]["keyword"] == 1.0
    assert report["mean_hit_at_k_by_class"]["sfd"]["describing"] == 0.0
    assert report["mean_hit_at_k_by_facet"]["sfd"]["topic"] == 1.0
    assert report["mean_hit_at_k_by_facet"]["sfd"]["temporal"] == 0.0


def test_hit_at_different_k_is_recomputable_from_one_stored_ranking_without_retrieval():
    """Decision 4 / task 3.4: k is a reporting parameter over the persisted
    full ranking, not a retrieval-time commitment. is_hit is a pure function
    of an already-fetched ranking -- recomputing at a different k costs no
    retrieval call (no OS client involved in this test at all)."""
    ranked = ["d2", "d1", "d3"]  # a persisted, untruncated ranking; d1 is the source
    assert vs.is_hit(ranked, "d1", k=1) is False   # rank 2, outside k=1
    assert vs.is_hit(ranked, "d1", k=2) is True    # ...but within k=2
    assert vs.is_hit(ranked, "d1", k=10) is True   # and any k >= 2


def test_run_per_variation_records_are_untruncated_full_rankings():
    saved = {}
    _patch_query_vector(saved)
    try:
        client = _FakeOSClient(hits_by_field={SFD_FIELD: [("d1", 9.0), ("d2", 5.0), ("d3", 1.0)]})
        variations = [
            {"variation_id": "v0", "base_query_id": "b1", "source_dataset_id": "d1",
             "facet": "topic", "query_class": "keyword", "text": "x"},
        ]
        report = vs.run(client, variations, k=1, size=10)  # k=1 for scoring, but persistence is full
    finally:
        _unpatch_query_vector(saved)
    per_var = report["per_variation"][0]
    # every candidate persisted, not truncated to k=1 (Decision 4)
    assert per_var["arms"]["sfd"]["ranked_isolated"] == ["d1", "d2", "d3"]
    assert per_var["arms"]["sfd"]["hit"] is True  # d1 is rank 1, within k=1


def run_tests() -> int:
    test_rank_orders_by_score_descending_ties_broken_by_id()
    test_is_hit_true_when_source_in_top_k()
    test_is_hit_false_when_source_outside_top_k()
    test_is_hit_scores_a_miss_when_a_sibling_outranks_the_source_outside_k()
    test_score_variation_retrieves_title_and_knn_once_and_arm_per_arm()
    test_run_mean_hit_at_k_across_two_variations()
    test_run_all_n_hit_requires_every_variation_in_the_group_to_hit()
    test_run_by_class_and_by_facet_breakdown()
    test_hit_at_different_k_is_recomputable_from_one_stored_ranking_without_retrieval()
    test_run_per_variation_records_are_untruncated_full_rankings()
    print("OK: rank/is_hit primitives, sibling-does-not-count scoring, channel "
          "retrieval shape (title/knn once, arm per-arm), mean hit@k aggregation, "
          "all-n-hit grouping by base query, by-class/by-facet breakdown, "
          "untruncated per-variation persistence")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
