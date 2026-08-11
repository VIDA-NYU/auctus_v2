"""Offline tests for the benchmark leakage guards (no LLM, no network).

These encode the invariant that makes the benchmark valid (report F1/F1b): no
description arm's prose may ground query generation or the judge prompt.

The enforcement is structural, in two layers, and these tests exercise both on the
document shape production actually produces:

- ``NEUTRAL_SOURCE_FIELDS`` — the one allowlist both callers fetch with, asserted
  disjoint from the arm fields;
- ``build_neutral_bundle`` — refuses any document that carries an arm field.

The predecessor of these tests passed against a hand-built document carrying all
six arm fields, a shape neither caller ever fetches; the guard it exercised could
therefore never fire in production. Hence ``test_call_sites_fetch_the_allowlist``:
the guarantee lives in what the call sites request, so that is what is asserted.

Run: python -m eval.test_leakage_guards   (or via pytest)
"""

from __future__ import annotations

from storage.opensearch_client import DESCRIPTION_SOURCE_FIELDS

from eval import generate_queries, judge_qrels
from eval.generate_queries import (
    FORBIDDEN_ARM_FIELDS,
    NEUTRAL_SOURCE_FIELDS,
    build_neutral_bundle,
)

# A neutral document, exactly as the two call sites fetch it: allowlist keys only.
_NEUTRAL_DOC = {
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
}

_SAMPLE = "crash_date,borough\n2020,Brooklyn\n"


def test_allowlist_and_arm_fields_are_disjoint() -> None:
    """The property the import-time assert in generate_queries protects."""
    assert not (set(NEUTRAL_SOURCE_FIELDS) & FORBIDDEN_ARM_FIELDS)
    # And the forbidden set really covers every arm, so a new arm is not missed.
    assert FORBIDDEN_ARM_FIELDS == frozenset(DESCRIPTION_SOURCE_FIELDS.values())
    assert len(FORBIDDEN_ARM_FIELDS) == 6, sorted(FORBIDDEN_ARM_FIELDS)


def test_neutral_document_builds_a_useful_bundle() -> None:
    """The guard must be satisfied by neutrality, not by an empty bundle."""
    bundle = build_neutral_bundle(_NEUTRAL_DOC, sample=_SAMPLE)
    blob = str(bundle)
    assert "Motor Vehicle Collisions" in blob   # title survives
    assert "borough" in blob                    # profile facts survive
    assert "Brooklyn" in blob                   # sample survives


def test_bundle_refuses_a_document_carrying_any_arm_field() -> None:
    """Parametrised over all six, so a newly added arm is covered automatically."""
    for field in sorted(FORBIDDEN_ARM_FIELDS):
        doc = dict(_NEUTRAL_DOC, **{field: "SENTINEL unique arm prose"})
        raised = False
        try:
            build_neutral_bundle(doc, sample=_SAMPLE)
        except AssertionError as exc:
            raised = True
            assert field in str(exc), (field, str(exc))
        assert raised, f"guard failed to refuse a document carrying {field!r}"


def test_guard_fires_on_a_present_but_empty_arm_field() -> None:
    """Key presence, not truthiness.

    An empty arm field means the caller fetched wrongly; the next document it
    fetches will not be empty. Keying on truthiness would let the wrong fetch pass
    on exactly the documents where it happens to do no damage.
    """
    doc = dict(_NEUTRAL_DOC, autoddg_search_description="")
    raised = False
    try:
        build_neutral_bundle(doc, sample=_SAMPLE)
    except AssertionError:
        raised = True
    assert raised, "guard ignored a present-but-empty arm field"


def test_call_sites_fetch_the_allowlist() -> None:
    """Both production fetches must request the shared constant.

    Asserted by identity against the constant rather than against a literal list,
    so this tracks the call sites instead of restating them.
    """
    for module in (generate_queries, judge_qrels):
        assert module.NEUTRAL_SOURCE_FIELDS is NEUTRAL_SOURCE_FIELDS, module.__name__


def test_leakage_audit_still_reads_every_arm() -> None:
    """The surviving content check must NOT be narrowed by the allowlist.

    ``leakage_audit`` needs exactly the fields the bundle is forbidden to see; it
    is the only place home advantage is detectable at all.
    """
    from eval import leakage_audit  # imported here: not part of the guard path

    assert FORBIDDEN_ARM_FIELDS <= set(
        leakage_audit.DESCRIPTION_SOURCE_FIELDS.values()
    )


def main() -> int:
    test_allowlist_and_arm_fields_are_disjoint()
    test_neutral_document_builds_a_useful_bundle()
    test_bundle_refuses_a_document_carrying_any_arm_field()
    test_guard_fires_on_a_present_but_empty_arm_field()
    test_call_sites_fetch_the_allowlist()
    test_leakage_audit_still_reads_every_arm()
    print("OK: neutral fetch allowlist checked; bundle refuses any arm-carrying doc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
