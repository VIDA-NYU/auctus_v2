"""Offline tests for the semantic-profile arms (no network, no LLM, no MinIO).

The two arms exist to answer "what do the profiles retrieve on their own,
before UFD/SFD synthesis?". That question is only answered if the indexed text
is the semantic profile UFD/SFD were actually generated from. So the properties
under test are about provenance of the text, not its quality: it is the stored
string verbatim, nothing stands in for it when it is absent, and the backfill
writes these two fields and no others (rewriting ``profile_only`` or ``t_od_s``
in passing would silently change two arms that are already scored).

Run: python -m eval.test_backfill_static_arms   (or via pytest)
"""

from __future__ import annotations

from eval import backfill_static_arms as bsa
from eval.backfill_static_arms import (
    SEMANTIC_ARM_FIELDS,
    SEMANTIC_PROFILE_FIELD,
    SEMANTIC_STRUCTURAL_FIELD,
    STATIC_ARM_FIELDS,
    backfill_semantic_one,
    build_semantic_arms,
    ensure_arm_field_mappings,
)
from storage.arq_worker import build_profile_text
from storage.opensearch_client import (
    DESCRIPTION_SOURCE_FIELDS,
    GENERATED_DESCRIPTION_FIELD_MAPPINGS,
)

OS_SOURCE = {
    "title": "t",
    "description": "d",
    "profiler_metadata": {
        "nb_rows": 3,
        "nb_columns": 1,
        "columns": [{"name": "boro", "structural_type": "http://schema.org/Text"}],
    },
}
# Leading/trailing whitespace and markdown are deliberate: "verbatim" must mean
# byte-identical, not "equal after tidying".
SEMANTIC = "  The key semantic information:\n**boro**: Represents borough. \n"


class _FakeOS:
    """Records writes; ``get`` serves one fixed document."""

    def __init__(self, source: dict | None = OS_SOURCE):
        self._source = source
        self.updates: list[dict] = []
        self.mappings: list[dict] = []
        self.indices = self

    def get(self, index, id, _source):
        if self._source is None:
            raise KeyError(id)
        return {"_source": self._source}

    def update(self, index, id, body, refresh):
        self.updates.append({"id": id, "doc": body["doc"]})

    def put_mapping(self, index, body):
        self.mappings.append(body)


def _with_full_profile(record: dict | None):
    """Point the module's MinIO read at a fixed record for one call."""
    original = bsa.load_full_profile
    bsa.load_full_profile = lambda _client, _id: record
    return original


def test_semantic_arm_is_the_stored_string_verbatim() -> None:
    arms = build_semantic_arms(OS_SOURCE, SEMANTIC)
    assert arms[SEMANTIC_PROFILE_FIELD] == SEMANTIC


def test_combined_arm_is_structural_blank_line_semantic() -> None:
    arms = build_semantic_arms(OS_SOURCE, SEMANTIC)
    assert arms[SEMANTIC_STRUCTURAL_FIELD] == build_profile_text(OS_SOURCE) + "\n\n" + SEMANTIC


def test_missing_semantic_profile_builds_nothing() -> None:
    """Absent, blank and non-string all mean "no semantic profile": neither
    field is built, so nothing is written and the structural text never stands
    in for it under the combined arm's name."""
    for absent in (None, "", "   \n", 0, {"x": 1}):
        assert build_semantic_arms(OS_SOURCE, absent) == {}, absent


def test_missing_structural_profile_omits_only_the_combined_arm() -> None:
    arms = build_semantic_arms({"title": "t"}, SEMANTIC)
    assert arms == {SEMANTIC_PROFILE_FIELD: SEMANTIC}


def test_backfill_writes_only_the_two_semantic_fields() -> None:
    os_client = _FakeOS()
    restore = _with_full_profile({"autoddg_semantic_profile": SEMANTIC, "sample": "a,b\n1,2"})
    try:
        status = backfill_semantic_one(os_client, object(), "abcd-1234", dry_run=False)
    finally:
        bsa.load_full_profile = restore
    assert status.startswith("updated "), status
    assert len(os_client.updates) == 1
    written = os_client.updates[0]["doc"]
    assert set(written) == set(SEMANTIC_ARM_FIELDS), written
    assert not set(written) & set(STATIC_ARM_FIELDS)
    assert written[SEMANTIC_PROFILE_FIELD] == SEMANTIC


def test_backfill_reports_a_missing_semantic_profile_and_writes_nothing() -> None:
    os_client = _FakeOS()
    restore = _with_full_profile({"sample": "a,b\n1,2"})
    try:
        status = backfill_semantic_one(os_client, object(), "abcd-1234", dry_run=False)
    finally:
        bsa.load_full_profile = restore
    assert status == "missing-semantic-profile"
    assert os_client.updates == []


def test_dry_run_writes_nothing_and_reports_lengths() -> None:
    os_client = _FakeOS()
    restore = _with_full_profile({"autoddg_semantic_profile": SEMANTIC})
    try:
        status = backfill_semantic_one(os_client, object(), "abcd-1234", dry_run=True)
    finally:
        bsa.load_full_profile = restore
    assert os_client.updates == []
    assert status.startswith("would set ")
    assert f"semantic={len(SEMANTIC)} chars" in status


def test_no_llm_client_is_reachable_from_the_backfill_module() -> None:
    """Building these arms must not be able to call a model: the module imports
    no LLM client and no AutoDDG entry point, so there is nothing to call."""
    forbidden = ("get_llm_client", "get_autoddg", "complete", "attach_autoddg_description")
    assert not [name for name in forbidden if hasattr(bsa, name)]


def test_semantic_fields_are_mapped_from_the_shared_definition() -> None:
    os_client = _FakeOS()
    ensure_arm_field_mappings(os_client, index="scratch", fields=SEMANTIC_ARM_FIELDS)
    assert os_client.mappings == [
        {"properties": {f: GENERATED_DESCRIPTION_FIELD_MAPPINGS[f] for f in SEMANTIC_ARM_FIELDS}}
    ]
    for field in SEMANTIC_ARM_FIELDS:
        assert GENERATED_DESCRIPTION_FIELD_MAPPINGS[field] == {
            "type": "text",
            "analyzer": "text_analyzer",
        }


def test_both_arms_are_registered_in_the_single_arm_map() -> None:
    assert DESCRIPTION_SOURCE_FIELDS["semantic_profile_only"] == SEMANTIC_PROFILE_FIELD
    assert DESCRIPTION_SOURCE_FIELDS["semantic_structural"] == SEMANTIC_STRUCTURAL_FIELD
    # Every arm field the map names has an explicit mapping, so none can fall
    # into dynamic mapping with a different analyzer.
    unmapped = [
        f for f in DESCRIPTION_SOURCE_FIELDS.values()
        if f != "description" and f not in GENERATED_DESCRIPTION_FIELD_MAPPINGS
    ]
    assert unmapped == []


def main() -> int:
    test_semantic_arm_is_the_stored_string_verbatim()
    test_combined_arm_is_structural_blank_line_semantic()
    test_missing_semantic_profile_builds_nothing()
    test_missing_structural_profile_omits_only_the_combined_arm()
    test_backfill_writes_only_the_two_semantic_fields()
    test_backfill_reports_a_missing_semantic_profile_and_writes_nothing()
    test_dry_run_writes_nothing_and_reports_lengths()
    test_no_llm_client_is_reachable_from_the_backfill_module()
    test_semantic_fields_are_mapped_from_the_shared_definition()
    test_both_arms_are_registered_in_the_single_arm_map()
    print("OK: semantic-profile arms are the stored text, verbatim, and nothing else is written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
