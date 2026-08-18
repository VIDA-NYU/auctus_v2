"""Offline tests for judge_agreement.py's pool-free candidate set (no LLM, no
network) -- covers judge-agreement-pool-free's design.md decisions:
corpus-frame-derived candidate pairs with absent-means-zero, judge_failed
queries excluded rather than zeroed, and loud failure instead of a printed
NaN when the comparable-query count is zero or a corpus_frame mismatch is
found.

``vectorise``/``weighted_kappa``/``grade_distribution`` are pure functions
tested directly with in-memory judge dicts. The ``main()``-level exit paths
(zero comparable queries, mismatched corpus frames) need real files on disk,
since that is what ``main`` actually reads -- built into a temp directory
per test rather than depending on any committed fixture.

Run: python -m eval.test_judge_agreement   (or via pytest)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from eval.judge_agreement import grade_distribution, main, vectorise, weighted_kappa

_CORPUS = ["d1", "d2", "d3"]


def _judge(labels: dict[str, dict[str, int]], failed: set[str] | None = None) -> dict:
    return {"labels": labels, "failed": failed or set()}


def test_vectorise_absent_means_zero() -> None:
    """A dataset missing from a query's `relevant` map is a genuine 0, not a
    gap -- the convention this change's candidate set depends on."""
    judge = _judge({"q1": {"d1": 2}})  # d2, d3 absent
    v = vectorise(judge, _CORPUS, ["q1"])
    assert v == [2, 0, 0]


def test_vectorise_covers_every_corpus_id_per_query() -> None:
    judge = _judge({"q1": {"d1": 1, "d2": 2, "d3": 0}, "q2": {}})
    v = vectorise(judge, _CORPUS, ["q1", "q2"])
    assert v == [1, 2, 0, 0, 0, 0]  # q1's 3 corpus pairs, then q2's 3 (all absent-as-0)


def test_weighted_kappa_perfect_agreement_is_one() -> None:
    a = [0, 1, 2, 0, 1, 2]
    assert weighted_kappa(a, a) == 1.0


def test_weighted_kappa_charges_zero_vs_two_more_than_zero_vs_one() -> None:
    """A 0-vs-2 split is a bigger disagreement than a 0-vs-1 split under the
    quadratic weighting -- the whole reason weighted over plain kappa (§1c-3).
    A varied base vector (not a single constant) is required: two fully
    constant vectors always collapse to kappa=0 regardless of how far apart
    the constants are, since po/pe both concentrate on the one confusion cell."""
    a =      [0, 1, 2, 0, 1, 2, 0, 1, 2]
    b_near = [0, 1, 2, 1, 1, 2, 0, 1, 2]  # index 3 only: 0 -> 1 (one grade off)
    b_far  = [0, 1, 2, 2, 1, 2, 0, 1, 2]  # index 3 only: 0 -> 2 (two grades off)
    assert weighted_kappa(a, b_far) < weighted_kappa(a, b_near)


def test_weighted_kappa_nan_on_identical_constant_vectors() -> None:
    """Both judges grading everything the same constant value is the
    degenerate case main() must catch and refuse to print (task 2.3)."""
    import math
    a = [1, 1, 1, 1]
    assert math.isnan(weighted_kappa(a, a))


def test_grade_distribution_counts_each_grade() -> None:
    assert grade_distribution([0, 0, 1, 2, 2, 2]) == {0: 2, 1: 1, 2: 3}


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _qrels_doc(model: str, lab: str, corpus_frame: str, queries: list[dict]) -> dict:
    return {
        "judge": {"model": model, "lab": lab, "temperature_pinned": True,
                  "corpus_frame": corpus_frame},
        "queries": queries,
    }


def test_main_excludes_a_judge_failed_query_rather_than_zeroing_it() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        frame = d / "frame.json"
        _write(frame, {"corpus_ids": _CORPUS})

        a = d / "qrels_a.json"
        _write(a, _qrels_doc("judge-a", "LabA", str(frame), [
            {"query_id": "q1", "text": "t", "relevant": {"d1": 2}, "unverified": []},
            {"query_id": "q2", "text": "t", "relevant": {"d2": 1}, "unverified": []},
        ]))
        b = d / "qrels_b.json"
        _write(b, _qrels_doc("judge-b", "LabB", str(frame), [
            {"query_id": "q1", "text": "t", "relevant": {"d1": 1}, "unverified": []},
            {"query_id": "q2", "text": "t", "judge_failed": "chunk 3 raised"},
        ]))

        out = d / "report.json"
        rc = main(["--corpus-frame", str(frame), "--out", str(out), str(a), str(b)])
        assert rc == 0

        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["comparable_queries"] == 1  # only q1 -- q2 failed on judge-b
        assert report["excluded_queries"] == ["q2"]
        assert report["corpus_pairs"] == len(_CORPUS)  # 1 comparable query x 3 corpus ids


def test_main_zero_comparable_queries_exits_nonzero() -> None:
    """The exact defect being fixed: two qrels files with disjoint query ids
    used to silently report `Comparable queries: 0` and `kappa=nan` at exit
    0. This must now refuse instead."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        frame = d / "frame.json"
        _write(frame, {"corpus_ids": _CORPUS})

        a = d / "qrels_a.json"
        _write(a, _qrels_doc("judge-a", "LabA", str(frame), [
            {"query_id": "q1", "text": "t", "relevant": {"d1": 2}, "unverified": []},
        ]))
        b = d / "qrels_b.json"
        _write(b, _qrels_doc("judge-b", "LabB", str(frame), [
            {"query_id": "q2", "text": "t", "relevant": {"d2": 1}, "unverified": []},
        ]))

        try:
            main(["--corpus-frame", str(frame), str(a), str(b)])
        except SystemExit as exc:
            assert exc.code != 0
            assert "zero comparable queries" in str(exc.code).lower()
        else:
            raise AssertionError("expected SystemExit on zero comparable queries")


def test_main_mismatched_corpus_frame_exits_nonzero() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        frame = d / "frame.json"
        _write(frame, {"corpus_ids": _CORPUS})

        a = d / "qrels_a.json"
        _write(a, _qrels_doc("judge-a", "LabA", "eval/frame/some_other_frame.json", [
            {"query_id": "q1", "text": "t", "relevant": {"d1": 2}, "unverified": []},
        ]))
        b = d / "qrels_b.json"
        _write(b, _qrels_doc("judge-b", "LabB", str(frame), [
            {"query_id": "q1", "text": "t", "relevant": {"d1": 1}, "unverified": []},
        ]))

        try:
            main(["--corpus-frame", str(frame), str(a), str(b)])
        except SystemExit as exc:
            assert "disagree about their corpus frame" in str(exc.code)
        else:
            raise AssertionError("expected SystemExit on mismatched corpus_frame")


def main_tests() -> int:
    test_vectorise_absent_means_zero()
    test_vectorise_covers_every_corpus_id_per_query()
    test_weighted_kappa_perfect_agreement_is_one()
    test_weighted_kappa_charges_zero_vs_two_more_than_zero_vs_one()
    test_weighted_kappa_nan_on_identical_constant_vectors()
    test_grade_distribution_counts_each_grade()
    test_main_excludes_a_judge_failed_query_rather_than_zeroing_it()
    test_main_zero_comparable_queries_exits_nonzero()
    test_main_mismatched_corpus_frame_exits_nonzero()
    print("OK: absent-means-zero vectorisation, weighted kappa (perfect/graded-charge/nan), "
          "grade distribution, judge_failed exclusion (not zeroing), zero-comparable-queries "
          "refusal, mismatched-corpus-frame refusal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_tests())
