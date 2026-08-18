"""Offline tests for merge_qrels.py's max/mean aggregation modes (no LLM, no
network) -- covers mean-aggregation-sensitivity-row's core claim: a `mean`
merge over a 1-vs-2 disagreement is stored as 1.5, never rounded, from the
same `raw_grades` a `max` merge would use.

``merge()`` is a pure function over ``load_judge()``-shaped dicts, so this
suite builds those dicts directly rather than writing qrels files to disk.

Run: python -m eval.test_merge_qrels   (or via pytest)
"""

from __future__ import annotations

from eval.merge_qrels import merge


def _judge(name: str, lab: str, labels: dict[str, dict[str, int]],
           unverified: dict[str, set] | None = None,
           failed: set[str] | None = None) -> dict:
    text = {qid: f"text for {qid}" for qid in set(labels) | (failed or set())}
    return {
        "name": name, "lab": lab, "text": text,
        "labels": labels, "unverified": unverified or {qid: set() for qid in labels},
        "failed": failed or set(), "role_conflict": None, "path": f"{name}.json",
    }


def test_max_merge_takes_the_higher_grade() -> None:
    a = _judge("a", "LabA", {"q1": {"d1": 0, "d2": 2}})
    b = _judge("b", "LabB", {"q1": {"d1": 1, "d2": 1}})
    result = merge(a, b, "max")
    q1 = next(q for q in result["queries"] if q["query_id"] == "q1")
    assert q1["relevant"] == {"d1": 1, "d2": 2}


def test_mean_merge_keeps_a_disagreement_fractional() -> None:
    """The core claim: a 1-vs-2 disagreement under `mean` is 1.5, not
    rounded to 1 or 2 -- `round(mean)` is a separately-rejected rule."""
    a = _judge("a", "LabA", {"q1": {"d1": 1}})
    b = _judge("b", "LabB", {"q1": {"d1": 2}})
    result = merge(a, b, "mean")
    q1 = next(q for q in result["queries"] if q["query_id"] == "q1")
    assert q1["relevant"]["d1"] == 1.5


def test_mean_merge_zero_vs_two_is_one_not_zero_or_two() -> None:
    a = _judge("a", "LabA", {"q1": {"d1": 0}})
    b = _judge("b", "LabB", {"q1": {"d1": 2}})
    result = merge(a, b, "mean")
    q1 = next(q for q in result["queries"] if q["query_id"] == "q1")
    assert q1["relevant"]["d1"] == 1.0


def test_max_and_mean_agree_when_judges_agree() -> None:
    a = _judge("a", "LabA", {"q1": {"d1": 2, "d2": 0}})
    b = _judge("b", "LabB", {"q1": {"d1": 2, "d2": 0}})
    max_result = merge(a, b, "max")
    mean_result = merge(a, b, "mean")
    max_q1 = next(q for q in max_result["queries"] if q["query_id"] == "q1")
    mean_q1 = next(q for q in mean_result["queries"] if q["query_id"] == "q1")
    assert max_q1["relevant"] == {"d1": 2}
    assert mean_q1["relevant"] == {"d1": 2.0}


def test_raw_grades_persisted_under_both_modes() -> None:
    """The whole reason a second aggregation is free later: raw_grades is
    written the same way regardless of which mode produced `relevant`."""
    a = _judge("a", "LabA", {"q1": {"d1": 1}})
    b = _judge("b", "LabB", {"q1": {"d1": 2}})
    for mode in ("max", "mean"):
        result = merge(a, b, mode)
        q1 = next(q for q in result["queries"] if q["query_id"] == "q1")
        assert q1["raw_grades"]["d1"] == {"a": 1, "b": 2}


def test_a_failed_query_is_excluded_under_either_mode() -> None:
    a = _judge("a", "LabA", {"q1": {"d1": 1}})
    b = _judge("b", "LabB", {}, failed={"q1"})
    for mode in ("max", "mean"):
        result = merge(a, b, mode)
        q1 = next(q for q in result["queries"] if q["query_id"] == "q1")
        assert q1["judge_failed"]
        assert q1["relevant"] == {}
        assert result["queries_judge_failed"] == 1
        assert result["queries_merged"] == 0


def test_unverified_is_a_union_under_either_mode() -> None:
    """design.md Open Questions: unverified is a union under both modes,
    stated explicitly rather than left as an unstated default."""
    a = _judge("a", "LabA", {"q1": {"d1": 1}}, unverified={"q1": {"d1"}})
    b = _judge("b", "LabB", {"q1": {"d1": 2}}, unverified={"q1": set()})
    for mode in ("max", "mean"):
        result = merge(a, b, mode)
        q1 = next(q for q in result["queries"] if q["query_id"] == "q1")
        assert q1["unverified"] == ["d1"]


def test_unknown_aggregation_raises() -> None:
    a = _judge("a", "LabA", {"q1": {"d1": 1}})
    b = _judge("b", "LabB", {"q1": {"d1": 2}})
    try:
        merge(a, b, "median")
    except ValueError as exc:
        assert "median" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown aggregation mode")


def run_tests() -> int:
    test_max_merge_takes_the_higher_grade()
    test_mean_merge_keeps_a_disagreement_fractional()
    test_mean_merge_zero_vs_two_is_one_not_zero_or_two()
    test_max_and_mean_agree_when_judges_agree()
    test_raw_grades_persisted_under_both_modes()
    test_a_failed_query_is_excluded_under_either_mode()
    test_unverified_is_a_union_under_either_mode()
    test_unknown_aggregation_raises()
    print("OK: max merge (higher grade), mean merge (fractional, never rounded), "
          "max/mean agreement when judges agree, raw_grades persisted under both modes, "
          "failed-query exclusion under both modes, unverified union under both modes, "
          "unknown aggregation rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
