"""Offline tests for run_matrix.py's qrels-scale contract (no LLM, no network).

Covers graded-qrels-downstream-contract's re-read result: the 0/1/2 grade
scale does not degrade to binary at the metric level (test_ndcg_separates_grades
is the direct regression for the degradation §6 item 12 feared), and the three
distinct reasons a query is excluded from scoring -- judge failure, genuine
no-positive judgement, and absence from the qrels file entirely -- are counted
separately rather than folded into one number (D2).

``classify_query``/``parse_qrels`` are pure functions with no OpenSearch
dependency, so most of this suite needs no fake client at all; the tests that
do exercise ``run()`` use a minimal in-memory stand-in for the OpenSearch
client rather than a live index.

Run: python -m eval.test_run_matrix   (or via pytest)
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.run_matrix import ARMS, classify_query, ndcg_for, parse_qrels, run

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_QRELS_GRADED_LIVE = _BACKEND_ROOT / "eval/benchmark/qrels_n32_haiku_graded_live.json"
_QRELS_UNDECLARED = _BACKEND_ROOT / "eval/benchmark/qrels_n32_haiku.json"


class _FakeOSClient:
    """Enough of the OpenSearch client surface for run_matrix.search_arm:
    returns canned hits for a single-field `match` query, keyed by field.

    Records every `size` it was asked for and counts calls, so a test can
    assert what retrieval depth was requested and how many retrieval passes
    happened (retrieval-depth-and-judge-seat-guard)."""

    def __init__(self, hits_by_field: dict[str, list[tuple[str, float]]]):
        self.hits_by_field = hits_by_field
        self.requested_sizes: list[int] = []
        self.call_count = 0

    def search(self, index, body, size):
        field = next(iter(body["query"]["match"]))
        self.requested_sizes.append(size)
        self.call_count += 1
        hits = self.hits_by_field.get(field, [])[:size]
        return {"hits": {"hits": [{"_id": i, "_score": s} for i, s in hits]}}


def _synthetic_qrels() -> dict[str, dict]:
    """One query per exclusion reason, plus one scored query -- the fixture
    task 6.1 calls for. No real artifact contains a judge_failed entry (checked
    during design: qrels_n32_haiku.json and qrels_n32_haiku_graded_live.json
    both have zero), so this is the only input that exercises that path."""
    return {
        "q_scored": {"grades": {"d1": 2, "d2": 1}, "judge_failed": False},
        "q_judge_failed": {"grades": {}, "judge_failed": True},
        "q_no_positive": {"grades": {"d1": 0, "d2": 0}, "judge_failed": False},
        # "q_absent" deliberately has no entry here.
    }


def _synthetic_queries() -> list[dict]:
    return [
        {"query_id": "q_scored", "text": "x", "facet": "topic", "query_class": "keyword"},
        {"query_id": "q_judge_failed", "text": "x", "facet": "topic", "query_class": "keyword"},
        {"query_id": "q_no_positive", "text": "x", "facet": "topic", "query_class": "keyword"},
        {"query_id": "q_absent", "text": "x", "facet": "topic", "query_class": "keyword"},
    ]


def test_classify_query_four_states() -> None:
    qrels = _synthetic_qrels()
    assert classify_query(qrels, "q_scored") == "scored"
    assert classify_query(qrels, "q_judge_failed") == "judge_failed"
    assert classify_query(qrels, "q_no_positive") == "no_positive"
    assert classify_query(qrels, "q_absent") == "absent_from_qrels"


def test_parse_qrels_grade_scale_declared() -> None:
    doc = json.loads(_QRELS_GRADED_LIVE.read_text(encoding="utf-8"))
    qrels, grade_scale = parse_qrels(doc)
    assert grade_scale == "0/1/2"
    assert len(qrels) == len(doc["queries"])
    # judge_failed is a string (the error) when present, absent otherwise --
    # parse_qrels normalises it to a bool for classify_query.
    assert all(isinstance(e["judge_failed"], bool) for e in qrels.values())


def test_parse_qrels_grade_scale_undeclared() -> None:
    doc = json.loads(_QRELS_UNDECLARED.read_text(encoding="utf-8"))
    qrels, grade_scale = parse_qrels(doc)
    assert grade_scale == "undeclared"
    assert len(qrels) == len(doc["queries"])


def test_dpec_ucu7_4_classifies_as_no_positive_not_failure() -> None:
    """design.md open question 2: cross-judge-facet-tables.md quotes this
    query's skipped_no_positive count by name. Confirms the three-way split
    still lands it in the same bucket that document's claim depends on."""
    doc = json.loads(_QRELS_UNDECLARED.read_text(encoding="utf-8"))
    qrels, _ = parse_qrels(doc)
    assert "dpec-ucu7_4" in qrels
    assert classify_query(qrels, "dpec-ucu7_4") == "no_positive"
    assert qrels["dpec-ucu7_4"]["judge_failed"] is False
    assert qrels["dpec-ucu7_4"]["grades"] == {}


def test_ndcg_separates_grades() -> None:
    """The direct regression test for the degradation §6 item 12 feared:
    two rankings identical except which document is ranked first, one grade-2
    and one grade-1, must score differently."""
    grades = {"d_grade2": 2, "d_grade1": 1}
    ranked_best_first = ["d_grade2", "d_grade1"]
    ranked_worst_first = ["d_grade1", "d_grade2"]
    best = ndcg_for(ranked_best_first, grades, k=10)
    worst = ndcg_for(ranked_worst_first, grades, k=10)
    assert best == 1.0  # ideal ordering
    assert worst < best  # a binary-blind scorer would call these identical
    # And binary qrels genuinely tie under this ordering swap -- confirms the
    # difference above comes from the graded input, not from ndcg_for itself.
    binary_grades = {"d_grade2": 1, "d_grade1": 1}
    assert ndcg_for(ranked_best_first, binary_grades, k=10) == \
        ndcg_for(ranked_worst_first, binary_grades, k=10)


def test_run_reconciliation_and_exclusion_breakdown() -> None:
    qrels = _synthetic_qrels()
    queries = _synthetic_queries()
    fake_hits = {DESCRIPTION_FIELD: [("d1", 5.0), ("d2", 3.0)] for DESCRIPTION_FIELD in
                 _description_fields()}
    os_client = _FakeOSClient(fake_hits)

    result = run(os_client, queries, qrels, [10], retrieval_size=100)[10]

    assert result["scored_queries"] == 1
    assert result["skipped_judge_failed"] == 1
    assert result["skipped_no_positive"] == 1
    assert result["skipped_absent_from_qrels"] == 1
    # reconciliation: run() itself asserts this internally; re-check here too
    total = (result["scored_queries"] + result["skipped_judge_failed"]
              + result["skipped_no_positive"] + result["skipped_absent_from_qrels"])
    assert total == len(queries)


def test_per_query_grade_distribution() -> None:
    """D4: n_relevant is unchanged, n_relevant_by_grade is additive."""
    qrels = {"q1": {"grades": {"d1": 2, "d2": 2, "d3": 1}, "judge_failed": False}}
    queries = [{"query_id": "q1", "text": "x", "facet": "topic", "query_class": "keyword"}]
    fake_hits = {f: [("d1", 5.0), ("d2", 3.0), ("d3", 1.0)] for f in _description_fields()}
    os_client = _FakeOSClient(fake_hits)

    result = run(os_client, queries, qrels, [10], retrieval_size=100)[10]
    records = [r for r in result["per_query"] if r["arm"] == ARMS[0]]
    assert len(records) == 1
    rec = records[0]
    assert rec["n_relevant"] == 3  # unchanged definition: any grade > 0
    assert rec["n_relevant_by_grade"] == {2: 2, 1: 1}


def test_retrieval_size_is_the_corpus_not_the_reporting_cutoff() -> None:
    """The finding-5 regression: `--k 10` must not become an OpenSearch
    `size` of 10. Retrieval depth is the corpus; k only cuts the ranking."""
    qrels = {"q1": {"grades": {"d1": 2}, "judge_failed": False}}
    queries = [{"query_id": "q1", "text": "x", "facet": "topic", "query_class": "keyword"}]
    os_client = _FakeOSClient({f: [("d1", 5.0)] for f in _description_fields()})

    run(os_client, queries, qrels, [10], retrieval_size=100)

    assert os_client.requested_sizes, "no retrieval happened"
    assert set(os_client.requested_sizes) == {100}, (
        f"retrieval was capped at the reporting cutoff: {set(os_client.requested_sizes)}"
    )


def test_two_cutoffs_cost_one_retrieval_pass() -> None:
    """§2c: a second reporting cutoff is a recomputation over the stored
    ranking, never a second trip to the index."""
    qrels = {"q1": {"grades": {"d1": 2, "d2": 1}, "judge_failed": False}}
    queries = [{"query_id": "q1", "text": "x", "facet": "topic", "query_class": "keyword"}]
    hits = {f: [("d1", 5.0), ("d2", 3.0)] for f in _description_fields()}

    one_cutoff = _FakeOSClient(hits)
    run(one_cutoff, queries, qrels, [10], retrieval_size=100)

    two_cutoffs = _FakeOSClient(hits)
    run(two_cutoffs, queries, qrels, [10, 5], retrieval_size=100)

    assert two_cutoffs.call_count == one_cutoff.call_count, (
        "asking for a second cutoff triggered extra retrieval calls"
    )


def test_scores_match_what_separate_per_cutoff_runs_produced() -> None:
    """Scoring one retrieval at two cutoffs must equal scoring two separate
    runs at those cutoffs -- the equivalence the single-pass refactor rests
    on. Uses a ranking longer than the smaller cutoff so the two genuinely
    differ."""
    grades = {f"d{i}": (2 if i <= 2 else 1) for i in range(1, 9)}
    qrels = {"q1": {"grades": grades, "judge_failed": False}}
    queries = [{"query_id": "q1", "text": "x", "facet": "topic", "query_class": "keyword"}]
    # deliberately imperfect ranking so NDCG@3 and NDCG@8 are not both 1.0
    ranking = [("d8", 9.0), ("d1", 8.0), ("d7", 7.0), ("d2", 6.0),
               ("d3", 5.0), ("d4", 4.0), ("d5", 3.0), ("d6", 2.0)]
    hits = {f: ranking for f in _description_fields()}

    combined = run(_FakeOSClient(hits), queries, qrels, [8, 3], retrieval_size=100)
    separate_8 = run(_FakeOSClient(hits), queries, qrels, [8], retrieval_size=100)[8]
    separate_3 = run(_FakeOSClient(hits), queries, qrels, [3], retrieval_size=100)[3]

    assert combined[8]["overall"] == separate_8["overall"]
    assert combined[3]["overall"] == separate_3["overall"]
    # and the two cutoffs are genuinely different numbers, or this proves nothing
    assert combined[8]["overall"] != combined[3]["overall"]


def _description_fields() -> list[str]:
    from storage.opensearch_client import DESCRIPTION_SOURCE_FIELDS
    return list(DESCRIPTION_SOURCE_FIELDS.values())


def main() -> int:
    test_classify_query_four_states()
    test_parse_qrels_grade_scale_declared()
    test_parse_qrels_grade_scale_undeclared()
    test_dpec_ucu7_4_classifies_as_no_positive_not_failure()
    test_ndcg_separates_grades()
    test_run_reconciliation_and_exclusion_breakdown()
    test_per_query_grade_distribution()
    test_retrieval_size_is_the_corpus_not_the_reporting_cutoff()
    test_two_cutoffs_cost_one_retrieval_pass()
    test_scores_match_what_separate_per_cutoff_runs_produced()
    print("OK: classify_query's four states, grade_scale declared/undeclared parsing, "
          "graded NDCG separation, exclusion-reason accounting, per-grade distribution, "
          "retrieval depth decoupled from reporting cutoff (size, one pass, equivalence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
