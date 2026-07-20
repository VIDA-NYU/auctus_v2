"""Offline tests for the benchmark leakage guards (no LLM, no network).

These encode the invariants that make the benchmark valid (report F1/F1b):
- the query-generation neutral bundle never contains a description arm's prose;
- the guard actually fires when an arm's text does leak in.

Run: python -m eval.test_leakage_guards   (or via pytest)
"""

from __future__ import annotations

from eval.generate_queries import build_neutral_bundle, assert_no_arm_leak

# A realistic-ish document: real neutral facts + every generated arm carrying a
# unique sentinel we can detect if it leaks.
_DOC = {
    "title": "Motor Vehicle Collisions - Crashes",
    "profiler_metadata": {
        "nb_rows": 1000,
        "nb_columns": 3,
        "types": ["temporal", "spatial", "categorical"],
        "columns": [
            {"name": "crash_date", "structural_type": "http://schema.org/DateTime"},
            {"name": "borough", "semantic_types": ["city name"]},
            {"name": "vehicle_type", "structural_type": "http://schema.org/Text"},
        ],
    },
    "spatial_coverage": {"label": "New York City", "bbox": [-74.2, 40.5, -73.7, 40.9]},
    # Arms — must NEVER appear in the neutral bundle:
    "autoddg_description": "SENTINEL_UFD unique ufd prose",
    "autoddg_search_description": "SENTINEL_SFD unique sfd prose",
    "llm_direct_description": "SENTINEL_LLMDIRECT unique prose",
    "tods_description": "SENTINEL_TODS unique prose",
    "profile_only_description": "SENTINEL_PROFILEONLY prose",
}


def test_bundle_excludes_arm_prose() -> None:
    bundle = build_neutral_bundle(_DOC, sample="crash_date,borough\n2020,Brooklyn\n")
    assert_no_arm_leak(bundle, _DOC)  # must not raise
    blob = str(bundle)
    for sentinel in ("SENTINEL_UFD", "SENTINEL_SFD", "SENTINEL_LLMDIRECT", "SENTINEL_TODS"):
        assert sentinel not in blob, f"{sentinel} leaked into neutral bundle"
    # The bundle DOES carry the neutral facts (title + profile) so it is useful.
    assert "Motor Vehicle Collisions" in blob
    assert "borough" in blob


def test_guard_fires_on_leak() -> None:
    # Simulate a leak: an arm's exact text ends up in the bundle (here via title).
    leaky_doc = dict(_DOC, title="SENTINEL_UFD unique ufd prose")
    bundle = build_neutral_bundle(leaky_doc, sample="")
    raised = False
    try:
        assert_no_arm_leak(bundle, leaky_doc)
    except AssertionError:
        raised = True
    assert raised, "guard failed to catch an arm's prose leaking into the bundle"


def main() -> int:
    test_bundle_excludes_arm_prose()
    test_guard_fires_on_leak()
    print("OK: leakage guards pass (arm prose excluded; guard fires on injected leak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
