"""Offline tests for stratified_sample.py (no network, synthetic catalog).

Run: python -m eval.test_stratified_sample   (or via pytest)
"""

from __future__ import annotations

from eval.build_corpus_slice import SHORT_DESCRIPTION_CHARS
from eval.stratified_sample import _apportion_categories, _largest_remainder, draw_sample


def _make_record(i: int, category: str, description_len: int, agency: str | None = "Agency X") -> dict:
    return {
        "id": f"aaaa-{i:04d}",
        "title": f"Dataset {i}",
        "description": "x" * description_len,
        "description_len": description_len,
        "domain_category": category,
        "agency": agency,
        "asset_type": "dataset",
    }


def _synthetic_catalog() -> list[dict]:
    """400 records: a large category with a realistic length spread including
    a real short-description tail, a small category (would round to 0 seats
    at n=100 under strict proportionality), and one entirely short category."""
    records = []
    i = 0
    # Large category, 300 records: 20% under SHORT_DESCRIPTION_CHARS, rest long.
    for _ in range(60):
        records.append(_make_record(i, "Big", description_len=30))
        i += 1
    for _ in range(240):
        records.append(_make_record(i, "Big", description_len=300))
        i += 1
    # Small category, 3 records: strict proportionality (3/400*100 ~= 0.75) rounds to 0.
    for _ in range(3):
        records.append(_make_record(i, "Tiny", description_len=200))
        i += 1
    # Mid category, 97 records, entirely short.
    for _ in range(97):
        records.append(_make_record(i, "Short-heavy", description_len=10))
        i += 1
    return records


def test_largest_remainder_sums_to_n_and_zero_weight_gets_zero() -> None:
    shares = {"a": 60, "b": 3, "c": 0}
    targets = _largest_remainder(shares, 20)
    assert sum(targets.values()) == 20
    assert targets["c"] == 0


def test_small_category_is_floored_at_one_seat() -> None:
    shares = {"Big": 300, "Tiny": 3, "Short-heavy": 97}
    targets = _apportion_categories(shares, 100)
    assert sum(targets.values()) == 100
    assert targets["Tiny"] >= 1, "a non-empty category must not be floored to zero"


def test_draw_reaches_the_catalog_wide_short_description_share() -> None:
    """Regression: an earlier version of _draw_category apportioned decile
    seats by rounding-then-consuming deciles in size order, which silently
    starved small deciles — including the short-description one, since short
    descriptions cluster in the bottom deciles — whenever rounding overflow
    ate the remaining per-category budget before reaching them. On the
    synthetic catalog here (20% of the dominant category is short) that bug
    reproduced a realized share barely half the catalog's true share."""
    catalog = _synthetic_catalog()
    catalog_short_share = sum(
        1 for r in catalog if r["description_len"] < SHORT_DESCRIPTION_CHARS
    ) / len(catalog)

    result = draw_sample(catalog, n=100, seed=1)

    assert result["drawn_n"] == 100
    realized_share = result["realized"]["short_description_share"]
    assert realized_share >= catalog_short_share - 0.05, (
        f"realized short-description share {realized_share:.3f} fell short of the "
        f"catalog's {catalog_short_share:.3f} by more than the rounding tolerance"
    )


def test_draw_is_reproducible_from_snapshot_and_seed_alone() -> None:
    """tasks.md 3.6: re-running the draw against the same records and seed
    must reproduce the identical slice, with no hidden state or I/O."""
    catalog = _synthetic_catalog()
    first = draw_sample(catalog, n=100, seed=42)
    second = draw_sample(catalog, n=100, seed=42)
    assert [r["id"] for r in first["datasets"]] == [r["id"] for r in second["datasets"]]

    different_seed = draw_sample(catalog, n=100, seed=43)
    assert [r["id"] for r in first["datasets"]] != [r["id"] for r in different_seed["datasets"]]


def test_every_non_empty_category_survives_the_draw() -> None:
    catalog = _synthetic_catalog()
    result = draw_sample(catalog, n=100, seed=1)
    assert set(result["realized"]["category_counts"]) == {"Big", "Tiny", "Short-heavy"}


def main() -> int:
    test_largest_remainder_sums_to_n_and_zero_weight_gets_zero()
    test_small_category_is_floored_at_one_seat()
    test_draw_reaches_the_catalog_wide_short_description_share()
    test_draw_is_reproducible_from_snapshot_and_seed_alone()
    test_every_non_empty_category_survives_the_draw()
    print("OK: apportionment sums correctly, small categories floored, short-description "
          "share tracked, draw reproducible from snapshot+seed alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
