"""Offline tests for _bound_sample_for_llm (guard-sample-size-before-llm).

Guards the defect where attach_autoddg_description passed an unbounded CSV
sample to four LLM-bound consumers: a single oversized field (a WKT geometry)
could reach ~2M tokens against a 1,048,576-token model limit, failing all
three LLM-backed arms identically (c5vm-g2dk, gthc-hcne, qyh3-ddat).

Run: python -m storage.test_sample_size_guard   (or via pytest)
"""

from __future__ import annotations

import io

import pandas as pd

from storage.arq_worker import MAX_SAMPLE_CHARS_FOR_LLM, _bound_sample_for_llm


def _csv_with_row_count(n_rows: int, value_width: int = 10) -> str:
    header = "a,b\n"
    row = "x" * value_width + ",1\n"
    return header + row * n_rows


def test_under_limit_passed_through_unchanged() -> None:
    sample = _csv_with_row_count(50)
    assert len(sample) < MAX_SAMPLE_CHARS_FOR_LLM
    assert _bound_sample_for_llm(sample, "dataset-x") == sample


def test_over_limit_truncated_on_a_line_boundary() -> None:
    sample = _csv_with_row_count(50_000)  # comfortably over the limit
    assert len(sample) > MAX_SAMPLE_CHARS_FOR_LLM

    truncated = _bound_sample_for_llm(sample, "dataset-x")
    assert len(truncated) <= MAX_SAMPLE_CHARS_FOR_LLM
    # No partial trailing row: every remaining line matches the fixed row shape
    # (a severed row would not match "x"*10 + ",1" exactly).
    lines = truncated.splitlines()
    assert lines[0] == "a,b"
    assert all(line == "x" * 10 + ",1" for line in lines[1:])


def test_truncated_output_still_parses_as_csv() -> None:
    sample = _csv_with_row_count(50_000)
    truncated = _bound_sample_for_llm(sample, "dataset-x")
    df = pd.read_csv(io.StringIO(truncated))
    assert list(df.columns) == ["a", "b"]
    assert len(df) > 0


def test_single_pathological_row_still_returns_something_parseable() -> None:
    """c5vm-g2dk's exact shape: the header plus ONE row whose value alone
    exceeds the limit (a giant WKT MULTIPOLYGON field). No line boundary
    exists inside the limit, so the header-only fallback must still parse."""
    header = "a,b\n"
    huge_row = "P" * (MAX_SAMPLE_CHARS_FOR_LLM * 3) + ",1\n"
    sample = header + huge_row
    assert len(sample) > MAX_SAMPLE_CHARS_FOR_LLM

    truncated = _bound_sample_for_llm(sample, "c5vm-g2dk")
    assert len(truncated) <= MAX_SAMPLE_CHARS_FOR_LLM
    assert truncated  # not empty
    df = pd.read_csv(io.StringIO(truncated))
    assert list(df.columns) == ["a", "b"]


def test_limit_is_respected_exactly_at_the_boundary() -> None:
    sample = _csv_with_row_count(5)
    exact = sample[: len(sample)]
    truncated = _bound_sample_for_llm(exact, "dataset-x", limit=len(sample))
    assert truncated == exact  # no truncation when already exactly at the limit


def main() -> int:
    test_under_limit_passed_through_unchanged()
    test_over_limit_truncated_on_a_line_boundary()
    test_truncated_output_still_parses_as_csv()
    test_single_pathological_row_still_returns_something_parseable()
    test_limit_is_respected_exactly_at_the_boundary()
    print("OK: sample-size guard truncates on a line boundary, stays CSV-parseable, "
          "and is inert under the limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
