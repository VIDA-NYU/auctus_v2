"""Offline tests for the cell-targeted generation loop (rewrite-query-generation
tasks.md 5.4-5.8) — generate_for_cell's validate/retry/refuse contract and
top_up_cells' seeded, no-re-ask-a-refusal behaviour. No live LLM call: the
module-level ``complete`` is monkeypatched with a canned response queue.

Run: python -m eval.test_generate_queries_cells   (or via pytest)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import eval.generate_queries as gq

NEUTRAL_DOC = {"title": "unused", "profiler_metadata": {"columns": []}}


class _FakeQueue:
    """Feeds canned response strings to eval.generate_queries.complete, in
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


def _patch_complete(monkeypatch_dict, responses):
    queue = _FakeQueue(responses)
    monkeypatch_dict["complete"] = gq.complete
    gq.complete = queue
    return queue


def _unpatch_complete(monkeypatch_dict):
    gq.complete = monkeypatch_dict["complete"]


def test_ok_response_is_accepted():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "query": {"text": "restaurant inspections", "statistical_subtype": None}}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "topic", "keyword")
    finally:
        _unpatch_complete(saved)
    assert result == {
        "status": "ok", "text": "restaurant inspections", "statistical_subtype": None,
        "ib_column_exists": None,
    }


def test_refusal_is_parsed_not_treated_as_failure():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "spatial", "query_class": "describing",
                    "query": None, "refusal_reason": "no computed spatial_coverage"}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "spatial", "describing")
    finally:
        _unpatch_complete(saved)
    assert result == {
        "status": "refused", "reason": "no computed spatial_coverage",
        "ib_column_exists": None,
    }


def test_facet_mismatch_retries_once_then_fails():
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",  # wrong facet both times
                    "query": {"text": "x", "statistical_subtype": None}}),
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": None}}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "spatial", "keyword", max_retries=1)
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "failed"
    assert "facet/class mismatch" in result["reason"]
    assert queue.calls == 2  # exactly one retry, not relabelled to the wrong facet


def test_facet_mismatch_then_correct_on_retry_succeeds():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "query": {"text": "wrong facet", "statistical_subtype": None}}),
        json.dumps({"facet": "spatial", "query_class": "keyword",
                    "query": {"text": "right facet", "statistical_subtype": None}}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "spatial", "keyword", max_retries=1)
    finally:
        _unpatch_complete(saved)
    assert result == {
        "status": "ok", "text": "right facet", "statistical_subtype": None,
        "ib_column_exists": None,
    }


def test_unparseable_response_retries_then_fails():
    saved = {}
    _patch_complete(saved, ["not json", "still not json"])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "topic", "keyword", max_retries=1)
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "failed"
    assert "unparseable" in result["reason"]


def test_statistical_requires_valid_subtype():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": "bogus"},
                    "ib_column_exists": True}),
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": "i-b"},
                    "ib_column_exists": True}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "statistical", "keyword", max_retries=1)
    finally:
        _unpatch_complete(saved)
    assert result == {
        "status": "ok", "text": "x", "statistical_subtype": "i-b",
        "ib_column_exists": True,
    }


def test_statistical_requires_valid_ib_column_exists():
    """Malformed/missing ib_column_exists retries once then fails, same
    budget as facet mismatch (task 2.2) — never defaulted to a value the
    model didn't state."""
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": "i-a"},
                    "ib_column_exists": "yes"}),  # not a JSON boolean
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": "i-a"},
                    "ib_column_exists": "yes"}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "statistical", "keyword", max_retries=1)
    finally:
        _unpatch_complete(saved)
    assert result["status"] == "failed"
    assert "ib_column_exists" in result["reason"]
    assert queue.calls == 2


def test_ib_column_exists_independent_of_chosen_subtype():
    """task 5.1: a response with ib_column_exists: true and
    statistical_subtype: 'i-a' is valid — the model can judge a measurement
    column exists and still choose to write the grain query this call."""
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": "i-a"},
                    "ib_column_exists": True}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "statistical", "keyword")
    finally:
        _unpatch_complete(saved)
    assert result == {
        "status": "ok", "text": "x", "statistical_subtype": "i-a",
        "ib_column_exists": True,
    }


def test_ib_column_exists_recorded_on_a_statistical_refusal():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "statistical", "query_class": "describing",
                    "query": None, "refusal_reason": "no basis",
                    "ib_column_exists": False}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "statistical", "describing")
    finally:
        _unpatch_complete(saved)
    assert result == {
        "status": "refused", "reason": "no basis", "ib_column_exists": False,
    }


def test_ib_column_exists_is_inert_on_non_statistical_facets():
    """task 2.3: the field is never read for other facets, even if a model
    spuriously includes it."""
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": None},
                    "ib_column_exists": True}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "topic", "keyword")
    finally:
        _unpatch_complete(saved)
    assert result["ib_column_exists"] is None


def test_statistical_hint_has_no_bias_clause_and_states_exclusions():
    """task 5.4: inspect the actual rendered prompt, the pattern the
    avoid-section test already established — a canned response can't catch a
    prompt-wording regression."""
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({"facet": "statistical", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": "i-a"},
                    "ib_column_exists": False}),
    ])
    try:
        gq.generate_for_cell(None, NEUTRAL_DOC, None, "statistical", "keyword")
    finally:
        _unpatch_complete(saved)
    prompt = queue.prompts[0]
    assert "pick whichever fits the data better" not in prompt
    assert "record identifier" in prompt
    assert "geographic coordinate" in prompt


def test_resolve_ib_judgement_agreement_both_true_and_both_false():
    """task 5.2: agreement resolves supported to that value, consistent True
    — including the both-false case, which is consistent but unsupported."""
    resolved = gq.resolve_ib_judgement({
        "a": {"keyword": True, "describing": True},
        "b": {"keyword": False, "describing": False},
    })
    assert resolved["a"] == {
        "keyword_ib_exists": True, "describing_ib_exists": True,
        "supported": True, "consistent": True,
    }
    assert resolved["b"] == {
        "keyword_ib_exists": False, "describing_ib_exists": False,
        "supported": False, "consistent": True,
    }


def test_resolve_ib_judgement_disagreement_is_inconsistent_and_unsupported():
    """task 5.2: a keyword/describing disagreement — the exact shape observed
    on 5uac-w243 during investigation — resolves to unsupported, recorded as
    inconsistent, not resolved by majority."""
    resolved = gq.resolve_ib_judgement({
        "c": {"keyword": False, "describing": True},
    })
    assert resolved["c"]["supported"] is False
    assert resolved["c"]["consistent"] is False


def test_resolve_ib_judgement_missing_class_is_not_silently_true():
    """A refused/failed class call means no judgement for that class — must
    not be treated as agreement."""
    resolved = gq.resolve_ib_judgement({
        "d": {"keyword": True, "describing": None},
    })
    assert resolved["d"]["supported"] is False
    assert resolved["d"]["consistent"] is False


def test_composite_supported_reads_ib_judgement_not_a_lexical_predicate():
    """task 5.3: composite's third qualifier comes from the persisted
    judgement dict, and its exact >=2 count is otherwise unchanged."""
    doc_no_spatial_or_temporal = {"title": "t", "profiler_metadata": {"columns": []}}
    ib_judgement = {"x": {"supported": True, "consistent": True,
                          "keyword_ib_exists": True, "describing_ib_exists": True}}
    # only ib is true; spatial/temporal both false on this doc -> sum == 1 -> not supported
    assert gq.composite_supported(doc_no_spatial_or_temporal, "x", ib_judgement) is False

    ib_judgement_false = {"x": {"supported": False, "consistent": True,
                                "keyword_ib_exists": False, "describing_ib_exists": False}}
    assert gq.composite_supported(doc_no_spatial_or_temporal, "x", ib_judgement_false) is False


def test_composite_supported_fails_loudly_when_dataset_missing_from_judgement():
    doc = {"title": "t", "profiler_metadata": {"columns": []}}
    try:
        gq.composite_supported(doc, "not-in-artifact", {})
        raised = False
    except KeyError:
        raised = True
    assert raised


def test_allocate_cells_requires_ib_judgement_when_composite_is_allocated():
    """task 5.3: allocation fails loudly (ValueError, not a silent
    unsupported-everywhere default) when composite is requested with no
    ib_judgement."""
    docs = {"x": NEUTRAL_DOC}
    try:
        gq.allocate_cells(docs)  # default facets include composite
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_allocate_cells_statistical_only_needs_no_ib_judgement():
    docs = {"x": NEUTRAL_DOC}
    allocation = gq.allocate_cells(docs, facets=("statistical",))
    assert allocation["assignments"]["x"] == [
        {"facet": "statistical", "query_class": "keyword"},
        {"facet": "statistical", "query_class": "describing"},
    ]


def test_load_ib_judgement_fails_loudly_when_missing():
    missing = Path("eval/frame/__test_definitely_missing_ib_judgement__.json")
    assert not missing.exists()
    try:
        gq.load_ib_judgement(missing)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_save_and_load_ib_judgement_round_trip():
    judgement = {"a": {"supported": True, "consistent": True,
                       "keyword_ib_exists": True, "describing_ib_exists": True}}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ib.json"
        gq.save_ib_judgement(path, judgement)
        loaded = gq.load_ib_judgement(path)
    assert loaded == judgement


def test_ib_judgement_path_for_derives_from_slice_stem():
    path = gq.ib_judgement_path_for("eval/benchmark/corpus_slice_2026-08-17.json")
    assert path == Path("eval/frame/ib_support_corpus_slice_2026-08-17.json")


def test_non_statistical_subtype_forced_to_none_even_if_model_sets_one():
    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "query": {"text": "x", "statistical_subtype": "i-a"}}),
    ])
    try:
        result = gq.generate_for_cell(None, NEUTRAL_DOC, None, "topic", "keyword")
    finally:
        _unpatch_complete(saved)
    assert result["statistical_subtype"] is None


def test_avoid_section_is_empty_with_no_avoid_texts():
    assert gq._render_avoid_section([]) == ""
    assert gq._render_avoid_section(()) == ""


def test_avoid_section_changes_the_prompt_sent_to_the_model():
    """The whole point (advisor-caught defect): a top-up call must NOT send the
    byte-identical prompt the main pass already sent, or the gateway's
    identical-prompt cache (design.md D1) just returns the same text again."""
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "query": {"text": "first", "statistical_subtype": None}}),
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "query": {"text": "second", "statistical_subtype": None}}),
    ])
    try:
        gq.generate_for_cell(None, NEUTRAL_DOC, None, "topic", "keyword")
        gq.generate_for_cell(None, NEUTRAL_DOC, None, "topic", "keyword", avoid_texts=["first"])
    finally:
        _unpatch_complete(saved)
    prompt_without_avoid, prompt_with_avoid = queue.prompts
    assert prompt_without_avoid != prompt_with_avoid
    assert '"first"' not in prompt_without_avoid  # the quoted avoid-list form, not the substring "first rows"
    assert '"first"' in prompt_with_avoid
    assert "GENUINELY DIFFERENT" in prompt_with_avoid


def test_top_up_passes_the_main_pass_text_as_avoid_text():
    docs = {"a": NEUTRAL_DOC}
    samples = {"a": None}
    support = {"a": {"topic": True}}
    existing_queries = [
        {"facet": "topic", "query_class": "keyword", "source_dataset_id": "a", "text": "already produced"},
    ]
    saved = {}
    queue = _patch_complete(saved, [
        json.dumps({"facet": "topic", "query_class": "keyword",
                    "query": {"text": "a second one", "statistical_subtype": None}}),
    ])
    try:
        extra, _ = gq.top_up_cells(
            None, docs, samples, existing_queries, set(), support, target=2, seed=1,
        )
    finally:
        _unpatch_complete(saved)
    assert len(extra) == 1
    assert extra[0]["text"] == "a second one"
    assert '"already produced"' in queue.prompts[0]


def test_top_up_skips_a_dataset_that_refused_this_exact_cell():
    docs = {"a": NEUTRAL_DOC, "b": NEUTRAL_DOC}
    samples = {"a": None, "b": None}
    support = {"a": {"spatial": True}, "b": {"spatial": True}}
    refused_cells = {("a", "spatial", "keyword")}
    existing_queries = []  # cell starts at 0

    saved = {}
    _patch_complete(saved, [
        json.dumps({"facet": "spatial", "query_class": "keyword",
                    "query": {"text": "from b", "statistical_subtype": None}}),
    ])
    try:
        extra, summary = gq.top_up_cells(
            None, docs, samples, existing_queries, refused_cells, support,
            target=1, seed=1,
        )
    finally:
        _unpatch_complete(saved)
    assert len(extra) == 1
    assert extra[0]["source_dataset_id"] == "b"  # never asked "a" again
    assert summary["spatial/keyword"]["met"] is True


def test_top_up_is_a_noop_when_cell_already_meets_target():
    docs = {"a": NEUTRAL_DOC}
    samples = {"a": None}
    support = {"a": {"topic": True}}
    existing_queries = [
        {"facet": "topic", "query_class": "keyword", "source_dataset_id": "a", "text": "x"},
    ]
    saved = {}
    _patch_complete(saved, [])  # must not be called
    try:
        extra, summary = gq.top_up_cells(
            None, docs, samples, existing_queries, set(), support,
            target=1, seed=1,
        )
    finally:
        _unpatch_complete(saved)
    assert extra == []
    assert summary["topic/keyword"]["shortfall"] == 0


def test_top_up_is_reproducible_given_the_same_seed():
    docs = {f"d{i}": NEUTRAL_DOC for i in range(5)}
    samples = {k: None for k in docs}
    support = {k: {"spatial": True} for k in docs}

    def run():
        saved = {}
        _patch_complete(saved, [
            json.dumps({"facet": "spatial", "query_class": "keyword",
                        "query": {"text": "q", "statistical_subtype": None}}),
        ])
        try:
            extra, _ = gq.top_up_cells(
                None, docs, samples, [], set(), support, target=1, seed=42,
            )
        finally:
            _unpatch_complete(saved)
        return extra[0]["source_dataset_id"]

    first = run()
    second = run()
    assert first == second


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
