"""Facet-comprehensive query generation, grounded ONLY on neutral facts.

The AutoDDG-paper benchmark uses topic-keyword queries — the one facet where
every description arm ties. To exercise AutoDDG's full potential we generate a
facet x query-class grid over the 5-facet set (facet-definitions-v2.md,
approved 2026-08-16): topic, temporal, spatial, statistical, composite. Cell
targeted, not free-choice: one call per (dataset, facet, class), with the
facet assigned, not chosen (see the allocator below).

CRITICAL leakage guard (F1). Queries are generated from a NEUTRAL bundle only —
algorithmic profile facts + a data sample, and NEVER the title (§6a/§6f — the
generator is title-blind so it cannot echo the title into a keyword query and
inflate ``t_od_s``'s BM25 score for reasons unrelated to description quality;
the judge keeps the title via the same function's ``include_title=True``) —
and NEVER any description arm's prose. A query written from arm X's text would
trivially match X and let X "win" by construction. So ``build_neutral_bundle``
reads raw ``profiler_metadata`` (via the same ``build_profile_text`` used
elsewhere) and the sample — never a value in ``DESCRIPTION_SOURCE_FIELDS``. One
representation-agnostic query set is produced and later applied identically to
every arm. ``source_dataset_id`` is kept for the leakage audit only; it is NOT
a qrel (qrels come from the judge over the pool — F8).

How F1 is enforced, and where. Two layers, both structural:

  1. ``NEUTRAL_SOURCE_FIELDS`` is the single allowlist every caller fetches with —
     here and in ``judge_qrels`` — and an import-time assert keeps it disjoint from
     the arm fields. A document that never carries an arm field cannot produce a
     bundle containing one.
  2. ``build_neutral_bundle`` refuses any document that carries an arm field, so a
     caller who fetches too widely fails immediately instead of silently relying on
     this function reading only certain keys.

Neither layer looks at *content*. Whether a finished query happens to sit unusually
close to one arm's vocabulary is a property of the output, decidable only over the
final query set with every arm's text loaded — that is ``leakage_audit.py``'s job,
and it is the only content-level check in the pipeline. An earlier substring guard
(``assert_no_arm_leak``) tried to do content detection here and could not: both
call sites fetch neutrally, so it compared against absent fields and never fired.

    python -m eval.generate_queries --slice eval/benchmark/corpus_slice.json \
        --out eval/benchmark/queries.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from storage.arq_worker import build_profile_text
from storage.opensearch_client import (
    AUCTUS_INDEX_NAME,
    DESCRIPTION_SOURCE_FIELDS,
    get_client,
)
from eval.backfill_descriptions import load_full_profile
from eval.llm_client import complete, get_llm_client
from eval.provenance import code_version
from storage.minio_client import get_storage_client

LOGGER = logging.getLogger("generate_queries")

# The ONLY fields build_neutral_bundle may read off a document. Any description
# arm is forbidden grounding — see the module docstring (F1).
FORBIDDEN_ARM_FIELDS = frozenset(DESCRIPTION_SOURCE_FIELDS.values())

# The fields every neutral-bundle caller fetches, here and in judge_qrels. This
# allowlist is what actually enforces F1: a document that never carries an arm
# field cannot yield a bundle containing one. Shared rather than duplicated per
# call site, so widening it for one stage cannot silently widen only that stage.
NEUTRAL_SOURCE_FIELDS = ("title", "profiler_metadata", "spatial_coverage")

# Fail at import if the two ever overlap, rather than at the end of a run.
assert not (set(NEUTRAL_SOURCE_FIELDS) & FORBIDDEN_ARM_FIELDS), (
    "the neutral fetch allowlist must not contain a description-arm field: "
    f"{sorted(set(NEUTRAL_SOURCE_FIELDS) & FORBIDDEN_ARM_FIELDS)}"
)

SAMPLE_CHARS = 1500

QUERY_CLASSES = ("keyword", "describing")
# Query-side facets: "by which aspect does a user search for this dataset?"
# Each one must be (a) plausible to search by, (b) something retrieval can
# succeed or fail at, and (c) VERIFIABLE BY THE JUDGE from the neutral bundle.
# Criterion (c) is why provenance / quality / usage (from the 8 description
# features in "Less Is More?", 2606.02334) are deliberately NOT here: the
# profiler computes no publisher, update frequency, or known-limitations data,
# so the judge could not ground those grades. Those 8 features belong to a
# separate INTRINSIC description-quality axis, not to the query axis.
#
# 5-facet set (facet-definitions-v2.md, approved 2026-08-16). `schema` was
# dropped for an undefinable confound with `profile_only`; `entity`/`measure`
# was dropped as too abstract; `vocabulary_mismatch` became a separate
# wording-variation track (item 10) with its own metric, not a query facet.
# The facet is ASSIGNED per (dataset, facet, class) cell, not freely chosen —
# see the cell-targeted allocator below.
FACETS = ("topic", "temporal", "spatial", "statistical", "composite")

# Embedded verbatim from facet-definitions-v2.md §1 (approved 2026-08-16;
# composite row + Rule 3 corrected 2026-08-17 per rewrite-query-generation
# design.md D3/OQ5 — the removal self-check was retired, composite is now
# subject + at least TWO of {time, place, value-span}). This is prompt
# payload, written for the generator: no experiment-design context, no
# changelog. `repos/auctus_v2` does not read the vault at runtime, so this
# copy is the executable one; task 4.2 diffs it against the source doc, and
# drift between them is a real risk this change accepts (design.md D6).
FACET_DEFINITION_TABLE = """\
| Facet | Definition | Ground it in | Keyword example | Describing example | Negative example |
| --- | --- | --- | --- | --- | --- |
| `topic` | Names the topic/field/domain in plain topic words; no other constraint | `columns[].name`, `types`, data sample | `restaurant inspections` | "I'm looking for records of health inspections at restaurants." | `restaurant inspections in 2023` — a time constraint makes this `temporal`/`composite`, not `topic` |
| `temporal` | Needs a specific time range or recency on top of the subject; a dataset with the right subject but the wrong period fails the constraint | `temporal_coverage` (a list of blocks, each carrying `ranges: [{"range": {"gte": <epoch>, "lte": <epoch>}}, ...]` — not a bare `{start, end}`), date columns' `semantic_types` | `taxi trips 2019` | "I need collision reports from the last three years, not older archives." | `historical taxi trips` — "historical" names no period a dataset can satisfy or fail |
| `spatial` | Needs a specific geography on top of the subject; a dataset with the right subject but the wrong place fails the constraint | Place names in the data sample, address/geo columns' `name` and `semantic_types` | `air quality brooklyn` | "I want noise complaints filed in Staten Island." | `air quality in New York City` — every dataset here is NYC-wide, so this excludes nothing |
| `statistical` (i-a, record grain) | Constrains what one row is: an event / a person / a transaction, or an aggregate | `nb_rows`, `nb_columns`, column grain semantics | `trip-level taxi records` | "I need crash data where each row is a single collision, not yearly summaries." | "what is the mean/variance of X" — the user computes that themselves once they have the right column; retrieval can't succeed or fail on it |
| `statistical` (i-b, value span) | Constrains the magnitude/range the measured numbers fall in | `coverage` (at most 3 ranges), `min`/`max` | `restaurant inspection scores 0-100 scale` | "I need salary data in the $30k-$80k range." | "which borough has the highest average score" — a statistic the user computes from the data, not a property of the data |
| `composite` | States the subject **and at least two** of {time, place, value-span}. Grain (record-level vs. aggregate) and bare topic do not count toward the two — both are near-universal across the corpus, so pairing either alone with one other clause does not discriminate this dataset from most others | whichever fields the combined constraints need | `taxi trips brooklyn 2019` | "I need trip-level taxi records in Brooklyn from 2019." | `restaurant inspections in New York City` — the place clause filters nothing; this is `topic` wearing a spatial phrase |
"""

# Approved 2026-08-17, unmodified from the first draft ("这些种子我觉得没问题" —
# author, same turn). Content is FICTIONAL throughout — no real dataset id or
# title anywhere — so these carry no leakage risk (F1). They demonstrate
# REGISTER/SENTENCE SHAPE only, not facet boundaries (the facet table does
# that). Full record, provenance, and design rationale:
# useful/2026-08-17/generator-seed-pairs.md (design.md D9, tasks.md 3.1/3.2/3.6).
SEED_PAIRS = (
    ("keyword", "building violations bronx open"),
    ("describing", "I'm trying to find a dataset that lists city employee salaries by agency and job title."),
    ("keyword", "subway ridership 2023 weekday"),
    ("describing", "I need a dataset showing where street trees have been planted, broken out by borough."),
    ("keyword", "noise complaints by zip code"),
    ("describing", "Looking for a dataset with one row per parking ticket issued — not totals aggregated by day."),
    ("keyword", "rat sightings queens 2022"),
    ("describing", "I want food cart inspection scores, somewhere in the 0 to 100 range."),
)


def _render_seed_pairs() -> str:
    return "\n".join(f'- ({query_class}) "{text}"' for query_class, text in SEED_PAIRS)


# The exclusion criteria a numeric column must clear to ground i-b (value
# span) or the `ib_column_exists` judgement (model-judged-ib-support design.md
# D2). Stated positively, once, and shared by both uses — an earlier
# column-name blocklist (`_is_meaningful_ib_column`) tried to encode this
# lexically and was measured unreliable (80/100 -> 63/100 under a
# word-boundary-aware rewrite; profiler statistics don't separate a real
# measurement from an identifier either — `Consumption (therms)` and
# `CMPLNT_NUM` have near-identical distinct/rows ratios). Whether a column is
# a genuine measurement is a semantic judgement, so it is now the model's,
# made fresh per call rather than precomputed.
_IB_EXCLUSION_CRITERIA = (
    "a record identifier, an administrative or category code, a "
    "district/precinct/tract/borough number, a building or parcel "
    "identifier, or a geographic coordinate (latitude, longitude, or "
    "projected X/Y)"
)


def _statistical_hint() -> str:
    """Facet-neutral hint for `statistical` cells (model-judged-ib-support
    design.md D1). Dataset-independent: the old per-dataset branch read
    `statistical_ib_supported()`, a lexical predicate now removed (see the
    module-level comment above `_IB_EXCLUSION_CRITERIA`). Both sub-types are
    offered with NO preference clause — an earlier version ended with "pick
    whichever fits the data better", which rewarded i-a because record grain
    is groundable on every dataset while value span is not; a criterion that
    rewards fit structurally rewards the option that always qualifies. Do
    not reintroduce a preference clause in either direction (design.md D1)."""
    return (
        " You may choose either i-a (record grain — what one row is) or i-b "
        "(value span — the magnitude/range a genuine measurement column falls "
        f"in, grounded in the profile's real coverage/min/max). A numeric "
        f"column is NOT a genuine measurement — do not ground an i-b query in "
        f"it — if it is {_IB_EXCLUSION_CRITERIA}. Separately from which "
        "sub-type you write, report in `ib_column_exists` whether this "
        "dataset has at least one genuine measurement column by that same "
        "test."
    )


def _render_avoid_section(avoid_texts: tuple[str, ...] | list[str]) -> str:
    """Top-up asks a dataset that already contributed a query for this exact
    cell (tasks.md 5.8). Without this, the second call sends the byte-identical
    prompt at temperature=0 — which the gateway caches (design.md D1) — and the
    "extra" query is just a duplicate of the first, not a new one."""
    if not avoid_texts:
        return ""
    lines = "\n".join(f'- "{t}"' for t in avoid_texts)
    return (
        "\nThis dataset already contributed a query for this exact (facet, class) "
        "cell. Produce a GENUINELY DIFFERENT query — do not repeat or lightly "
        "reword any of these:\n" + lines + "\n"
    )


# Per-cell prompt (design.md D1/D6/D9, tasks.md 4.3/4.4/4.5). One call per
# (dataset, facet, query_class) — the facet is ASSIGNED, never freely chosen,
# and the model must echo its assignment back so the caller can validate it
# (tasks.md 5.5: a facet/class mismatch is a generation failure, retried once,
# never silently relabelled to match what the query happens to test). No
# title in NEUTRAL FACTS (§6f) and no dead vocabulary_mismatch/composite
# "MUST include" instructions — those were free-choice-era leftovers that
# make no sense once the facet is assigned per call.
PROMPT = """You are constructing exactly ONE search query a real user might issue to \
find a government open-data dataset, for a SPECIFIC facet and query class assigned to \
you below. You may not substitute a different facet or a different query class, even if \
another one would fit the data better.

You are given ONLY neutral facts about the dataset: an algorithmic profile of its \
columns/coverage, and a small data sample. Do NOT assume any prose description exists. \
Do NOT use the dataset's title, even if a title-like string happens to appear in the \
profile or sample.

FACET DEFINITIONS (your assigned facet is `{facet}` — getting the boundary right \
matters more than sounding natural; a fluent query that actually fits a different \
facet is wrong):
{facet_table}

QUERY CLASSES:
- `keyword`: short, like a real portal search — place/agency/topic terms, NOT a full \
sentence.
- `describing`: a natural-language, first-person explanation of what the user needs, \
not a command.

Register/shape examples (content is FICTIONAL — do not reuse any subject matter below; \
they demonstrate sentence shape only, not facet boundaries):
{seed_pairs}

YOUR ASSIGNMENT: produce one `{query_class}` query for the `{facet}` facet.{statistical_hint}
{avoid_section}
If the data genuinely does not support this facet — the profile carries nothing to \
ground it — do not invent one. Refuse instead.

Return STRICT JSON only, no prose outside it. Echo back the facet and query_class you \
were assigned, so your answer can be checked against the assignment:
- if you can produce the query: {{"facet": "{facet}", "query_class": "{query_class}", \
"query": {{"text": "...", "statistical_subtype": "i-a"|"i-b"|null}}{ib_field}}}
- if you cannot: {{"facet": "{facet}", "query_class": "{query_class}", "query": null, \
"refusal_reason": "<one line>"{ib_field}}}

`statistical_subtype` is required (exactly "i-a" or "i-b") only when the assigned facet \
is `statistical`; set it to null for every other facet.
{ib_instruction}
NEUTRAL FACTS:
Profile: {profile}
Data sample (first rows):
{sample}
"""


def build_neutral_bundle(doc: dict, sample: str | None, *, include_title: bool) -> dict:
    """Neutral grounding: profile facts + sample, and the title IFF ``include_title``.
    Never an arm's prose.

    ``build_profile_text`` recomputes the profile from raw ``profiler_metadata``,
    so even though the ``profile_only`` arm holds the same facts, this reads the
    metadata — not the arm field. No ``DESCRIPTION_SOURCE_FIELDS`` value is touched.

    Enforcing F1 here, at the door, is deliberately stricter than "no arm prose
    ends up in the bundle": the document must not carry an arm field *at all*. A
    caller holding a full OpenSearch document has to strip it rather than trust
    this function to read only certain keys — "reads only three keys" is a
    property of the current body, not of the contract, and a fourth key is a
    one-line change away.

    ``include_title`` is keyword-only with NO default (design.md D2, §6a): the
    judge keeps the title, the query generator does not, and neither call site
    may inherit the wrong intent by omission. The dangerous direction — the
    generator silently receiving the title — is the one that produces a
    finished, plausible-looking, wrong run, so it must be stated, not assumed.
    """
    present = sorted(FORBIDDEN_ARM_FIELDS & doc.keys())
    if present:
        raise AssertionError(
            f"non-neutral document: carries description-arm field(s) {present}. "
            f"Fetch with _source=NEUTRAL_SOURCE_FIELDS ({list(NEUTRAL_SOURCE_FIELDS)}) "
            "or strip the arm fields before building a neutral bundle."
        )
    bundle = {
        "profile": build_profile_text(doc),
        "sample": (sample or "")[:SAMPLE_CHARS],
    }
    if include_title:
        bundle["title"] = doc.get("title") or ""
    return bundle


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    return json.loads(text)


# --- Support predicates (design.md D3, tasks.md 5.1/5.1a) --------------------
#
# Keyed on what build_profile_text() RENDERS, not on raw metadata field
# presence — the generator only ever sees the rendered text, and raw presence
# over-predicts support (spatial: 54/100 have spatial-typed columns, but only
# 21/100 render a computed spatial_coverage; the 33-dataset gap would order
# cells the generator cannot ground, which come back as refusals or invented
# geography). Every predicate here is a pure function of a document's rendered
# profile — no LLM call, so the allocator can run before any generation call.

def spatial_supported(doc: dict) -> bool:
    """⇔ the rendered profile carries a computed spatial_coverage/bbox block."""
    text = build_profile_text(doc) or ""
    return "spatial_coverage" in text or '"bbox"' in text


def temporal_supported(doc: dict) -> bool:
    """⇔ the rendered profile carries a computed temporal_coverage block.

    NOTE the shape: a list of blocks, each carrying
    ``ranges: [{"range": {"gte": <epoch>, "lte": <epoch>}}, ...]`` — NOT a bare
    ``{start, end}``. A predicate (or a human) checking for `{start, end}`
    silently undercounts (the round notes' 32/100 vs. the real 45/100).
    """
    text = build_profile_text(doc) or ""
    return "temporal_coverage" in text


def composite_supported(doc: dict, dataset_id: str, ib_judgement: dict[str, dict]) -> bool:
    """⇔ at least TWO of {spatial, temporal, statistical (i-b)} are grounded.

    Grain (i-a) and bare topic are excluded from the combinable set — both are
    near-universal, so pairing either alone with one other clause would not
    discriminate this dataset from most others (design.md D3/OQ5). This is a
    count, not a judgement: it replaced a removal self-check that could not
    distinguish a genuine composite from a single-facet query (the test
    reports "not load-bearing" for essentially any clause, since relaxing a
    query always preserves the relevance of what already satisfied it).

    The third qualifier — a genuine measurement column — is no longer decided
    lexically (the removed ``_is_meaningful_ib_column`` blocklist). It is read
    from ``ib_judgement``, the persisted per-corpus-round measurement judgement
    (model-judged-ib-support design.md D3/D4). A dataset missing from
    ``ib_judgement`` fails loudly (KeyError) rather than silently reading as
    unsupported — the artifact should cover every dataset ``allocate_cells``
    was given; a gap means the artifact is stale for this corpus snapshot.
    """
    if dataset_id not in ib_judgement:
        raise KeyError(
            f"no persisted measurement judgement for dataset {dataset_id!r} — "
            "the ib_judgement artifact is missing or stale for this corpus "
            "snapshot (design.md D3/D4); re-run the statistical-cell phase"
        )
    qualifiers = (
        spatial_supported(doc),
        temporal_supported(doc),
        bool(ib_judgement[dataset_id]["supported"]),
    )
    return sum(qualifiers) >= 2


def support_predicate(
    facet: str, doc: dict, dataset_id: str, ib_judgement: dict[str, dict] | None,
) -> bool:
    """Single dispatch point tasks.md 5.2's allocator calls per (dataset, facet).

    ``dataset_id``/``ib_judgement`` are only consulted for ``composite`` — the
    other facets' predicates are pure functions of ``doc`` alone.
    """
    if facet == "spatial":
        return spatial_supported(doc)
    if facet == "temporal":
        return temporal_supported(doc)
    if facet == "composite":
        if ib_judgement is None:
            raise ValueError(
                "composite allocation requires ib_judgement (the persisted "
                "measurement judgement) — pass allocate_cells(..., "
                "ib_judgement=...) rather than defaulting it (design.md D3)"
            )
        return composite_supported(doc, dataset_id, ib_judgement)
    if facet in ("topic", "statistical"):
        return True  # universal: every dataset has columns and a row grain
    raise ValueError(f"unknown facet {facet!r}")


# --- Measurement-column judgement persistence (design.md D3/D4, tasks.md 3) --
#
# `statistical` cells' responses each carry an `ib_column_exists` judgement
# (per (dataset, query_class)). Resolved per-dataset and persisted as a
# corpus-round artifact, analogous to eval/frame/'s existing per-round
# snapshots, so composite allocation reads it instead of re-deriving it.

def resolve_ib_judgement(
    class_judgements: dict[str, dict[str, bool | None]],
) -> dict[str, dict]:
    """``class_judgements``: dataset_id -> {"keyword": bool|None, "describing":
    bool|None} (``None`` when that class's cell was refused/failed and no
    judgement was returned). Resolves each dataset conservatively (design.md
    D3): ``supported`` is True only if BOTH classes judged True; a
    disagreement — including either side missing — is recorded as
    inconsistent rather than resolved by majority or default.
    """
    resolved: dict[str, dict] = {}
    for dataset_id, classes in class_judgements.items():
        kw = classes.get("keyword")
        dc = classes.get("describing")
        resolved[dataset_id] = {
            "keyword_ib_exists": kw,
            "describing_ib_exists": dc,
            "supported": bool(kw) and bool(dc),
            "consistent": kw is not None and dc is not None and kw == dc,
        }
    return resolved


def ib_judgement_path_for(slice_path: str | Path) -> Path:
    """Corpus-round artifact path, tied to the specific corpus snapshot
    (design.md D4) — named after the slice file that defines that snapshot,
    matching eval/frame/'s existing per-round file-naming convention."""
    return Path("eval/frame") / f"ib_support_{Path(slice_path).stem}.json"


def save_ib_judgement(path: Path, judgement: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(judgement, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def load_ib_judgement(path: Path) -> dict[str, dict]:
    """Fail loudly (task 3.3) — a missing artifact must not silently read as
    'every dataset lacks a measurement column'."""
    if not path.exists():
        raise FileNotFoundError(
            f"no persisted ib judgement at {path} for this corpus snapshot — "
            "run the statistical-cell phase first (allocate_cells(facets="
            '("statistical",)) + resolve_ib_judgement + save_ib_judgement) '
            "before allocating composite cells (design.md D3/D4)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# --- Pre-flight checks (design.md D8, tasks.md 5.3) --------------------------
#
# Two DIFFERENT hazards, verified separately rather than assumed to be one.
# `c5vm-g2dk` was named in tasks.md 5.3 as "~2M tokens" from the round notes,
# but that figure is `attach_autoddg_description()`'s view of the UNTRIMMED
# profile at ingestion time — a different code path. What generate_queries.py
# actually reads is the POST-`isolate_search_payload()` INDEXED document, and
# measured directly (2026-08-17), `c5vm-g2dk`'s indexed profile renders to
# 302 chars with `columns: []` — nowhere near large enough for a size check to
# catch. A size-only implementation of 5.3 would silently never fire on the
# dataset it was written for.
#
# What actually distinguishes `c5vm-g2dk`: 9 of the 100 datasets carry a
# `profiler_metadata.error` field (a profiling failure, not a size problem —
# in `c5vm-g2dk`'s case "Error tokenizing data: EOF inside string starting at
# row 3", a malformed source CSV), but 8 of those 9 still generated all three
# LLM-backed arms successfully at ingestion. Only `c5vm-g2dk` failed all three
# (`llm_direct`/`ufd`/`sfd` all absent). So `error`-field presence over-triggers
# (would wrongly skip 8 healthy datasets); the arm-generation outcome is the
# precise, already-recorded signal for "this dataset already broke an LLM call".
MAX_PROFILE_CHARS = 4_000_000  # ~1M tokens at a conservative 4 chars/token — kept
# as a genuine defense against a differently-oversized future dataset, even
# though it is not what catches c5vm-g2dk.

_LLM_BACKED_ARM_FIELDS = (
    "llm_direct_description", "autoddg_description", "autoddg_search_description",
)


def profile_too_large(doc: dict) -> bool:
    profile = build_profile_text(doc) or ""
    return len(profile) > MAX_PROFILE_CHARS


def arm_generation_already_failed(doc_with_arm_fields: dict) -> bool:
    """True if none of the three LLM-backed arms generated at ingestion.

    MUST be evaluated on a document fetched *with* the arm fields, for this
    boolean decision only — that document must never reach ``build_neutral_bundle``
    or any prompt-construction path (F1). Callers fetch it separately from the
    neutral bundle fetch and discard it immediately after this check.
    """
    return not any(doc_with_arm_fields.get(f) for f in _LLM_BACKED_ARM_FIELDS)


def preflight_skip_reason(neutral_doc: dict, doc_with_arm_fields: dict) -> str | None:
    """Returns a one-line skip reason, or None if the dataset is generation-ready.

    Two independent checks, reported distinguishably — an allocation-time skip
    must never be confused with a model-issued refusal (D4).
    """
    if arm_generation_already_failed(doc_with_arm_fields):
        return "arm generation already failed at ingestion (llm_direct/ufd/sfd all absent)"
    if profile_too_large(neutral_doc):
        return f"rendered profile exceeds {MAX_PROFILE_CHARS} chars"
    return None


# --- Cell-targeted allocator (design.md D1/D5, tasks.md 5.2) -----------------

def allocate_cells(
    docs: dict[str, dict],
    preflight_skips: dict[str, str] | None = None,
    ib_judgement: dict[str, dict] | None = None,
    facets: tuple[str, ...] = FACETS,
) -> dict:
    """Build the per-dataset (facet, class) cell assignment for every facet
    this dataset can support, over every id in ``docs`` (id -> neutral document).

    ``preflight_skips`` (id -> reason) comes from the caller's separate
    arm-field fetch (``preflight_skip_reason``) — this function never touches
    arm fields itself, keeping it testable on neutral documents alone.

    ``facets`` restricts which facets are allocated. ``composite`` depends on
    the persisted measurement judgement (``ib_judgement``), which in turn
    depends on `statistical` cells having already been generated — so a
    caller allocates in two calls per corpus snapshot: ``facets=("statistical",)``
    first (no ``ib_judgement`` needed), then the remaining facets once
    ``ib_judgement`` has been derived and persisted from that first call's
    responses (model-judged-ib-support design.md D3). Once the persisted
    artifact exists, both calls happen every run — this is not a one-off
    special case, just an ordering constraint.

    ``ib_judgement`` is required — raises if ``None`` — whenever ``"composite"``
    is among ``facets`` (task 3.3/D3: allocation must fail loudly, not
    silently treat every dataset as lacking a measurement column).

    Returns a persisted-artifact-shaped dict:
      {"assignments": {dataset_id: [{"facet": ..., "query_class": ...}, ...]}},
       "skipped": {dataset_id: reason},
       "support": {dataset_id: {facet: bool, ...}}}

    Skipped datasets are excluded from assignment entirely rather than
    allocated and then failing every call. `support` is persisted alongside
    the assignment so an allocation imbalance (e.g. spatial's thin support) is
    reported rather than discovered later (D5).
    """
    if "composite" in facets and ib_judgement is None:
        raise ValueError(
            "allocate_cells(facets=...) includes 'composite' but no "
            "ib_judgement was given — persist the statistical-cell measurement "
            "judgement for this corpus snapshot first (design.md D3/D4)"
        )
    preflight_skips = preflight_skips or {}
    assignments: dict[str, list[dict]] = {}
    skipped: dict[str, str] = dict(preflight_skips)
    support: dict[str, dict[str, bool]] = {}

    for dataset_id, doc in docs.items():
        if dataset_id in skipped:
            continue
        if profile_too_large(doc):
            skipped[dataset_id] = f"rendered profile exceeds {MAX_PROFILE_CHARS} chars"
            continue
        facet_support = {
            facet: support_predicate(facet, doc, dataset_id, ib_judgement)
            for facet in facets
        }
        support[dataset_id] = facet_support
        cells = [
            {"facet": facet, "query_class": qc}
            for facet in facets if facet_support[facet]
            for qc in QUERY_CLASSES
        ]
        assignments[dataset_id] = cells

    return {
        "assignments": assignments,
        "skipped": skipped,
        "support": support,
    }


def generate_for_cell(
    client, doc: dict, sample: str | None, facet: str, query_class: str,
    *, max_retries: int = 1, avoid_texts: tuple[str, ...] | list[str] = (),
) -> dict:
    """Generate (or refuse, or fail) one query for one (dataset, facet, class)
    cell (design.md D1). Returns exactly one of:

      {"status": "ok", "text": ..., "statistical_subtype": "i-a"|"i-b"|None,
       "ib_column_exists": bool|None}
      {"status": "refused", "reason": ..., "ib_column_exists": bool|None}
        # model-issued (tasks.md 5.6)
      {"status": "failed", "reason": ...}    # unparseable, or facet/class
        mismatch surviving ``max_retries`` retries (tasks.md 5.5) — never
        relabelled to match what the query happens to test (OQ2: one retry,
        then record).

    ``ib_column_exists`` (model-judged-ib-support design.md D1) is populated
    only for ``facet == "statistical"`` — on both the "ok" and "refused"
    outcomes, since the judgement is independent of whether a query was
    produced this call. It is ``None`` for every other facet (task 2.3: the
    field is inert elsewhere, never read).

    ``avoid_texts``: queries this same dataset already produced for this exact
    cell (top-up only, tasks.md 5.8). Non-empty ``avoid_texts`` changes the
    prompt text, which both instructs the model not to repeat itself and
    breaks the gateway's identical-prompt cache (design.md D1) — without it,
    a top-up call at temperature=0 against an unchanged prompt would just
    return the same cached text again, not a new query.
    """
    bundle = build_neutral_bundle(doc, sample, include_title=False)
    assert "title" not in bundle, (
        "generator bundle carries a title — F1/§6f violation; the generator "
        "must be title-blind (6f pilot passed 2026-08-17)"
    )
    is_statistical = facet == "statistical"
    hint = _statistical_hint() if is_statistical else ""
    if is_statistical:
        ib_field = ', "ib_column_exists": true|false'
        ib_instruction = (
            "\n`ib_column_exists` (a JSON boolean) is required only when the "
            "assigned facet is `statistical`, on both a produced query and a "
            "refusal — your judgement of whether this dataset has at least "
            "one genuine measurement column (same exclusion test as above), "
            "independent of which sub-type you wrote or whether you refused.\n"
        )
    else:
        ib_field = ""
        ib_instruction = ""
    prompt = PROMPT.format(
        facet=facet, query_class=query_class,
        facet_table=FACET_DEFINITION_TABLE,
        seed_pairs=_render_seed_pairs(),
        statistical_hint=hint,
        avoid_section=_render_avoid_section(avoid_texts),
        ib_field=ib_field,
        ib_instruction=ib_instruction,
        profile=bundle["profile"],
        sample=bundle["sample"] or "(no sample available)",
    )

    last_reason = "no attempt made"
    for _attempt in range(max_retries + 1):
        try:
            parsed = _parse_json(complete(client, prompt, temperature=0.0))
        except Exception as exc:
            last_reason = f"unparseable response: {exc}"
            continue

        if parsed.get("facet") != facet or parsed.get("query_class") != query_class:
            last_reason = (
                f"facet/class mismatch: assigned ({facet}, {query_class}), "
                f"got ({parsed.get('facet')!r}, {parsed.get('query_class')!r})"
            )
            continue

        ib_column_exists = None
        if is_statistical:
            ib_column_exists = parsed.get("ib_column_exists")
            if not isinstance(ib_column_exists, bool):
                last_reason = f"missing/invalid ib_column_exists: {ib_column_exists!r}"
                continue

        query = parsed.get("query")
        if query is None:
            return {
                "status": "refused",
                "reason": parsed.get("refusal_reason") or "no reason given",
                "ib_column_exists": ib_column_exists,
            }

        text = (query.get("text") or "").strip()
        if not text:
            last_reason = "empty query text"
            continue

        subtype = query.get("statistical_subtype")
        if is_statistical:
            if subtype not in ("i-a", "i-b"):
                last_reason = f"missing/invalid statistical_subtype: {subtype!r}"
                continue
        else:
            subtype = None

        return {
            "status": "ok", "text": text, "statistical_subtype": subtype,
            "ib_column_exists": ib_column_exists,
        }

    return {"status": "failed", "reason": last_reason}


def run_cells(
    client, docs: dict[str, dict], samples: dict[str, str | None],
    assignments: dict[str, list[dict]],
) -> dict:
    """Generate every cell in ``assignments`` (design.md D1's per-cell loop,
    factored out so ``main`` can run it twice: once for `statistical` alone,
    once for the remaining facets, per the two-phase ordering `composite`
    allocation needs — design.md D3).

    Returns:
      {"queries": [...], "refusals": [...], "generation_failures": [...],
       "refused_cells": {(dataset_id, facet, query_class), ...},
       "class_judgements": {dataset_id: {"keyword": bool|None, "describing": bool|None}}}

    ``class_judgements`` only ever gets entries from `statistical` cells — the
    only facet whose response carries `ib_column_exists` (task 2.3) — and
    only for the "ok"/"refused" outcomes (a "failed" cell contributed no
    judgement to record).
    """
    queries: list[dict] = []
    refusals: list[dict] = []
    generation_failures: list[dict] = []
    refused_cells: set[tuple[str, str, str]] = set()
    class_judgements: dict[str, dict[str, bool | None]] = {}

    total_cells = sum(len(cells) for cells in assignments.values())
    done = 0
    for dataset_id, cells in assignments.items():
        for cell in cells:
            facet, query_class = cell["facet"], cell["query_class"]
            done += 1
            try:
                result = generate_for_cell(
                    client, docs[dataset_id], samples.get(dataset_id), facet, query_class,
                )
            except Exception as exc:
                result = {"status": "failed", "reason": f"unhandled exception: {exc}"}

            if facet == "statistical" and result["status"] in ("ok", "refused"):
                class_judgements.setdefault(dataset_id, {})[query_class] = (
                    result.get("ib_column_exists")
                )

            if result["status"] == "ok":
                queries.append({
                    "text": result["text"],
                    "facet": facet,
                    "query_class": query_class,
                    "statistical_subtype": result["statistical_subtype"],
                    "source_dataset_id": dataset_id,  # audit only; NOT a qrel (F8)
                    "query_id": f"{dataset_id}_{facet}_{query_class}",
                })
            elif result["status"] == "refused":
                refused_cells.add((dataset_id, facet, query_class))
                refusals.append({
                    "dataset_id": dataset_id, "facet": facet,
                    "query_class": query_class, "reason": result["reason"],
                })
            else:
                generation_failures.append({
                    "dataset_id": dataset_id, "facet": facet,
                    "query_class": query_class, "reason": result["reason"],
                })

            print(f"  [{done}/{total_cells}] {dataset_id} {facet}/{query_class}: {result['status']}")

    return {
        "queries": queries,
        "refusals": refusals,
        "generation_failures": generation_failures,
        "refused_cells": refused_cells,
        "class_judgements": class_judgements,
    }


# Target n per (facet, class) cell (design.md D3/§1d: "±0.08 needs ~50" per
# cell, stricter than the per-facet magnitude since each facet has two
# cells). Governs the top-up pass only — the corpus swap (D10) already fixed
# most of spatial's shortfall (21→39/100, corpus-swap-spatial-coverage
# round notes), so top-up now covers residual shortfall only (D5).
TARGET_QUERIES_PER_CELL = 50

# Seeded so top-up dataset selection is reproducible (design.md D5), matching
# this project's convention of dated seeds for anything drawn.
TOP_UP_SEED = 20260817


def top_up_cells(
    client, docs: dict[str, dict], samples: dict[str, str | None],
    queries: list[dict], refused_cells: set[tuple[str, str, str]],
    support: dict[str, dict[str, bool]],
    *, target: int = TARGET_QUERIES_PER_CELL, seed: int = TOP_UP_SEED,
) -> tuple[list[dict], dict[str, dict]]:
    """Ask under-filled (facet, class) cells for extra queries (design.md D5,
    tasks.md 5.8), from datasets that support the facet and did NOT refuse it
    in the main pass. Never re-asks a dataset that already refused this exact
    cell — a refusal is a statement about the data, not the phrasing.

    Returns ``(extra_queries, summary)`` where ``summary`` records, per cell,
    how many extra queries were requested and from how many distinct
    datasets — so an unmet target is reported, not silently accepted (D5).
    """
    by_cell: dict[tuple[str, str], int] = {}
    texts_by_dataset_cell: dict[tuple[str, str, str], list[str]] = {}
    for q in queries:
        by_cell[(q["facet"], q["query_class"])] = by_cell.get((q["facet"], q["query_class"]), 0) + 1
        key = (q["source_dataset_id"], q["facet"], q["query_class"])
        texts_by_dataset_cell.setdefault(key, []).append(q["text"])

    rng = random.Random(seed)
    extra_queries: list[dict] = []
    summary: dict[str, dict] = {}

    for facet in FACETS:
        for query_class in QUERY_CLASSES:
            cell_key = (facet, query_class)
            current = by_cell.get(cell_key, 0)
            shortfall = target - current
            cell_name = f"{facet}/{query_class}"
            if shortfall <= 0:
                summary[cell_name] = {"shortfall": 0, "requested": 0, "datasets_asked": 0}
                continue

            eligible = [
                dataset_id for dataset_id, doc in docs.items()
                if support.get(dataset_id, {}).get(facet)
                and (dataset_id, facet, query_class) not in refused_cells
            ]
            rng.shuffle(eligible)

            requested = 0
            datasets_asked = 0
            for dataset_id in eligible:
                if requested >= shortfall:
                    break
                result = generate_for_cell(
                    client, docs[dataset_id], samples.get(dataset_id), facet, query_class,
                    avoid_texts=texts_by_dataset_cell.get((dataset_id, facet, query_class), []),
                )
                datasets_asked += 1
                if result["status"] == "ok":
                    extra_queries.append({
                        "text": result["text"],
                        "facet": facet,
                        "query_class": query_class,
                        "statistical_subtype": result["statistical_subtype"],
                        "source_dataset_id": dataset_id,
                        "query_id": f"{dataset_id}_{facet}_{query_class}_topup{requested}",
                        "top_up": True,
                    })
                    requested += 1
                # refusals/failures during top-up are not retried further —
                # move to the next eligible dataset rather than looping

            summary[cell_name] = {
                "shortfall": shortfall,
                "requested": requested,
                "datasets_asked": datasets_asked,
                "met": requested >= shortfall,
            }

    return extra_queries, summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", default="eval/benchmark/corpus_slice.json")
    parser.add_argument("--out", default="eval/benchmark/queries.json")
    parser.add_argument("--ids", nargs="+", help="Limit to these dataset ids")
    parser.add_argument("--no-top-up", action="store_true", help="Skip the top-up pass (5.8)")
    parser.add_argument(
        "--ib-judgement-path",
        help="Persisted measurement-judgement artifact path (default: derived "
             "from --slice, eval/frame/ib_support_<slice-stem>.json). Written "
             "if absent (design.md D3/D4's one-time bootstrap), otherwise read.",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.slice).read_text(encoding="utf-8"))
    ids = args.ids or [d["id"] for d in manifest["datasets"] if d.get("included")]

    os_client = get_client()
    client = get_llm_client()
    if client is None:
        raise SystemExit("No LLM client (PORTKEY_API_KEY set? on NYU VPN?).")
    try:
        storage_client = get_storage_client()
    except Exception:
        storage_client = None

    # Two SEPARATE fetches per dataset (F1): the neutral document (never
    # carries an arm field) and, only for the pre-flight decision, a
    # narrow arm-fields-only fetch that never reaches build_neutral_bundle
    # (design.md D8).
    docs: dict[str, dict] = {}
    samples: dict[str, str | None] = {}
    preflight_skips: dict[str, str] = {}
    for dataset_id in ids:
        neutral_doc = os_client.get(
            index=AUCTUS_INDEX_NAME, id=dataset_id,
            _source=list(NEUTRAL_SOURCE_FIELDS),
        ).get("_source") or {}
        arm_doc = os_client.get(
            index=AUCTUS_INDEX_NAME, id=dataset_id,
            _source=list(_LLM_BACKED_ARM_FIELDS),
        ).get("_source") or {}
        reason = preflight_skip_reason(neutral_doc, arm_doc)
        if reason:
            preflight_skips[dataset_id] = reason
            continue
        docs[dataset_id] = neutral_doc
        record = load_full_profile(storage_client, dataset_id) if storage_client else None
        samples[dataset_id] = record.get("sample") if isinstance(record, dict) else None

    # `statistical` always generates first, in its own phase, over EVERY
    # non-skipped dataset (it is universal, so no allocation gating is
    # needed) — `composite`'s third qualifier depends on the ib_column_exists
    # judgement that phase's responses carry (design.md D3/D4). Once the
    # persisted artifact exists for this corpus snapshot, this phase still
    # runs (statistical query text is generated every run, same as any other
    # facet) but its judgement observations are simply not re-persisted.
    ib_path = Path(args.ib_judgement_path) if args.ib_judgement_path else (
        ib_judgement_path_for(args.slice)
    )
    bootstrapping = not ib_path.exists()

    statistical_allocation = allocate_cells(docs, preflight_skips, facets=("statistical",))
    print(f"-- statistical phase ({'bootstrap, will persist judgement' if bootstrapping else 'judgement already persisted'}) --")
    statistical_run = run_cells(client, docs, samples, statistical_allocation["assignments"])

    if bootstrapping:
        ib_judgement = resolve_ib_judgement(statistical_run["class_judgements"])
        save_ib_judgement(ib_path, ib_judgement)
        print(f"persisted ib judgement for {len(ib_judgement)} datasets -> {ib_path}")
    ib_judgement = load_ib_judgement(ib_path)  # fail loud if still missing somehow

    remaining_facets = tuple(f for f in FACETS if f != "statistical")
    remaining_allocation = allocate_cells(
        docs, preflight_skips, ib_judgement, facets=remaining_facets,
    )
    print("-- remaining facets --")
    remaining_run = run_cells(client, docs, samples, remaining_allocation["assignments"])

    all_queries = statistical_run["queries"] + remaining_run["queries"]
    refusals = statistical_run["refusals"] + remaining_run["refusals"]
    generation_failures = (
        statistical_run["generation_failures"] + remaining_run["generation_failures"]
    )
    refused_cells = statistical_run["refused_cells"] | remaining_run["refused_cells"]
    skipped = remaining_allocation["skipped"]  # same preflight/size skips both phases share

    support: dict[str, dict[str, bool]] = {}
    for dataset_id, facet_support in statistical_allocation["support"].items():
        support.setdefault(dataset_id, {}).update(facet_support)
    for dataset_id, facet_support in remaining_allocation["support"].items():
        support.setdefault(dataset_id, {}).update(facet_support)

    top_up_summary: dict = {}
    if not args.no_top_up:
        extra_queries, top_up_summary = top_up_cells(
            client, docs, samples, all_queries, refused_cells, support,
        )
        all_queries.extend(extra_queries)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "code_version": code_version(),
            "queries": all_queries,
            "refusals": refusals,
            "generation_failures": generation_failures,
            "skipped": skipped,
            "support": support,
            "top_up": top_up_summary,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8")

    by_class: dict[str, int] = {}
    by_facet: dict[str, int] = {}
    for q in all_queries:
        by_class[q["query_class"]] = by_class.get(q["query_class"], 0) + 1
        by_facet[q["facet"]] = by_facet.get(q["facet"], 0) + 1
    print(f"\n{len(all_queries)} queries, {len(refusals)} refusals, "
          f"{len(generation_failures)} generation failures, {len(skipped)} datasets skipped -> {out}")
    print(f"by class: {by_class}")
    print(f"by facet: {by_facet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
