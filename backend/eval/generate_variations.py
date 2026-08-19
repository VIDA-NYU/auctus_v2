"""Wording-variation generator for the known-item variation track (item 10,
variation-track-known-item).

Rewrites each base query from the facet x class grid (`generate_queries.py`)
into `n=3` differently-worded variations, source dataset kept as ground truth
for known-item scoring downstream (`variation_scoring.py`). Replaces the
retired `vocabulary_mismatch` facet with a different metric and a different
measurement direction (design.md Context; proposal.md).

Reused rather than rebuilt (design.md Decision 6, Context): `generate_queries.py`'s
LLM call path (`llm_client.complete`/`get_llm_client`), JSON parsing
(`_parse_json`), the neutral-bundle machinery (`build_neutral_bundle`,
`NEUTRAL_SOURCE_FIELDS`), and — new in this change, not just plumbing — its
`FACET_DEFINITION_TABLE` and `SEED_PAIRS` are embedded verbatim in this
module's own prompt, so a rewrite has the same facet/register grounding the
base generator used, rather than a prompt written independently that could
drift from it.

**Three prompt-level instructions, none of them programmatically verified
except one (design.md Decision 6, Risks):**

  1. Fidelity to the base query's information need — the rewrite must ask for
     the same thing, only in different words. NOT checked downstream: no
     reliable automated method exists for judging semantic drift on 3-5 word
     queries (embedding similarity tested and found unreliable; NLI models
     share the same short-text problem; filtering on retrieval success would
     bias known-item scoring's own validity). Known-item scoring's fixed
     ground truth means an undetected drifted variation is scored an honest
     miss, not silently corrupted.
  2. No content word shared across the four texts (base + 3 rewrites),
     excluding stopwords and this corpus's generic filler terms. NOT checked
     downstream either, though it is mechanically checkable (unlike #1) —
     left to the model on the strength of its demonstrated reliability on
     harder constraints in the base generation run (711/711, 0 failures).
  3. The facet's own constraint (period / place / value span) survives the
     rewrite verbatim (Decision 2, unrelated to Decision 1/6's drift
     question — this one predates and is independent of it). This one IS
     checked: the model echoes the constraint substring(s) it identifies in
     the original query, and every echoed substring must appear verbatim in
     every rewrite, or the attempt is retried like any other validation
     failure in `generate_for_cell`'s pattern.

**No check of a variation's wording against the dataset's title or `original`
description** (author decision, 2026-08-18 — design.md Decision 1, overriding
§3b (1)). The generator never reads either field regardless (F1/§6f, same
invariant `generate_queries.py` follows) — there is no check to disable here
because the check was never a downstream step, and Decision 1 removed it.

Decision 5's exclusion accounting: a base query missing `source_dataset_id`,
or whose dataset is not in the round's corpus frame, is excluded and counted
by reason, not folded into one skip number.

    python -m eval.generate_variations \
        --queries eval/benchmark/queries_production_2026-08-17.json \
        --out eval/benchmark/variations.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from eval.corpus_frame import DEFAULT_CORPUS_FRAME, load_corpus_ids
from eval.generate_queries import (
    FACET_DEFINITION_TABLE,
    NEUTRAL_SOURCE_FIELDS,
    SEED_PAIRS,  # noqa: F401 -- re-exported for tests that check prompt content
    _parse_json,
    _render_seed_pairs,
    build_neutral_bundle,
)
from eval.backfill_descriptions import load_full_profile
from eval.llm_client import complete, get_llm_client
from eval.provenance import code_version
from storage.minio_client import get_storage_client
from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

LOGGER = logging.getLogger("generate_variations")

# §3b (3): 3 rewrites, base excluded. Not raised to 5 (design.md Non-Goals,
# §3b (3)'s effective-sample-size argument; the under-coverage of human
# rewrite diversity is a limitations-section item, not a reason to revisit).
N_VARIATIONS = 3

# Facets whose base query carries a constraint that must survive the rewrite
# verbatim (Decision 2). `topic` has none. `statistical`'s i-a subtype
# constrains record GRAIN (e.g. "each row is a single forecast") -- a
# clause-shaped constraint, not a bare extractable token like a place name or
# date -- so it is exempt too, same as `topic`: grain fidelity is covered by
# the prompt's own "same information need" instruction, not verified here
# (2026-08-18 refinement, after the production run surfaced systematic
# false-fails on i-a queries and on prepositions the model folded into its
# echoed substring — see `_requires_constraint`). `statistical`'s i-b subtype
# (a numeric value span) and `composite` (which never combines i-a per
# `generate_queries.py`'s own combinability rule) both require the check.

# Corpus-generic filler terms every query in this corpus could plausibly use
# regardless of subject matter — exempt from the "no shared content word"
# instruction (design.md Decision 6), or that instruction would be
# unsatisfiable (every query is about NYC open data).
FILLER_TERMS = (
    "data", "dataset", "datasets", "record", "records", "information",
    "nyc", "city", "new", "york",
)

PROMPT = """You are producing {n} WORDING VARIATIONS of an existing search query for a \
government open-data dataset. This is a REWRITE task, not a fresh-generation task: the \
query already exists below and your job is to restate it in different words — never to \
invent a different query, even a plausible one.

You are given ONLY neutral facts about the dataset: an algorithmic profile of its \
columns/coverage, and a small data sample. Do NOT assume any prose description exists. \
Do NOT use the dataset's title, even if a title-like string happens to appear in the \
profile or sample.

ORIGINAL QUERY (facet `{facet}`, class `{query_class}`):
"{base_text}"

FACET DEFINITIONS (for reference — your rewrites stay in the `{facet}` facet; they do \
not change facet):
{facet_table}

QUERY CLASSES:
- `keyword`: short, like a real portal search — place/agency/topic terms, NOT a full \
sentence.
- `describing`: a natural-language, first-person explanation of what the user needs, \
not a command.

Register/shape examples (content is FICTIONAL — do not reuse any subject matter below; \
they demonstrate sentence shape only, not facet boundaries):
{seed_pairs}

YOUR TASK:
1. Produce exactly {n} rewrites of the original query, each a `{query_class}`-class \
query, each expressing PRECISELY THE SAME INFORMATION NEED as the original — only the \
wording changes, never what is being asked for. A rewrite that would plausibly be \
answered by a different dataset than the original is wrong, no matter how fluent it \
reads.
2. {constraint_instruction}
3. Across all {n_plus_one} texts together — the original query and your {n} rewrites — \
no content word (subject-matter word) may repeat, {constraint_exception}Ordinary \
function words (a, the, in, for, of, and the like) and this corpus's own generic terms \
({filler_terms}) are exempt and may repeat freely.

Return STRICT JSON only, no prose outside it:
{{"facet": "{facet}", "query_class": "{query_class}"{constraint_field}, \
"variations": ["...", "...", "..."]}}

{constraint_field_instruction}
NEUTRAL FACTS:
Profile: {profile}
Data sample (first rows):
{sample}
"""


def _requires_constraint(facet: str, statistical_subtype: str | None) -> bool:
    """Whether this query's facet carries a constraint value that must survive
    the rewrite verbatim (Decision 2). `topic` never does. `statistical`'s i-a
    subtype constrains record GRAIN, not a bare token, so it is exempt like
    `topic` (module-level comment above `N_VARIATIONS` has the full rationale).
    `temporal`/`spatial`/`statistical`(i-b)/`composite` all constrain a bare,
    extractable value and do require the check.
    """
    if facet == "statistical":
        return statistical_subtype == "i-b"
    return facet in ("temporal", "spatial", "composite")


def _constraint_instruction(facet: str, statistical_subtype: str | None) -> str:
    if not _requires_constraint(facet, statistical_subtype):
        return "This facet carries no time/place/value-span constraint to preserve."
    return (
        "Identify the MINIMAL substring of the ORIGINAL query that IS this facet's "
        "constraint value itself — the bare period for `temporal` (e.g. \"2019\", "
        "\"2021 and 2023\"), the bare place name for `spatial` (e.g. \"Manhattan\", "
        "NOT \"in Manhattan\"), the bare number or numeric range for `statistical` "
        "(e.g. \"100,000 lbs\", NOT \"total gross weight over 100,000 lbs\"), or — "
        "for `composite` — each such bare value present (at least two). Do NOT "
        "include the surrounding preposition, verb, or clause structure (\"in\", "
        "\"for\", \"located\", \"over\", \"issued between\", and the like) in the "
        "substring — only the value itself. Preserve every one of those bare "
        "substrings VERBATIM, unchanged, in every rewrite; everything else, "
        "including how the value is introduced grammatically, may change freely."
    )


def _constraint_field(facet: str, statistical_subtype: str | None) -> tuple[str, str]:
    if not _requires_constraint(facet, statistical_subtype):
        return "", ""
    return (
        ', "constraint_substrings": ["..."]',
        '`constraint_substrings` is a JSON list of the bare constraint value(s) — '
        "the place name, period, or number/range itself, with no surrounding "
        "preposition or clause — from the ORIGINAL query (one entry for `temporal`/"
        "`spatial`/`statistical`, two or more for `composite`) — required for this "
        "facet.\n",
    )


def build_variation_prompt(
    base_text: str, facet: str, query_class: str, profile: str, sample: str,
    statistical_subtype: str | None = None,
) -> str:
    constraint_field, constraint_field_instruction = _constraint_field(facet, statistical_subtype)
    constraint_exception = (
        "except for the facet-constraint substring(s) named above, which are "
        "*expected* to repeat verbatim — that is the point of preserving them. "
    ) if _requires_constraint(facet, statistical_subtype) else ""
    return PROMPT.format(
        n=N_VARIATIONS, n_plus_one=N_VARIATIONS + 1,
        facet=facet, query_class=query_class, base_text=base_text,
        facet_table=FACET_DEFINITION_TABLE, seed_pairs=_render_seed_pairs(),
        constraint_instruction=_constraint_instruction(facet, statistical_subtype),
        constraint_exception=constraint_exception,
        filler_terms=", ".join(FILLER_TERMS),
        constraint_field=constraint_field,
        constraint_field_instruction=constraint_field_instruction,
        profile=profile, sample=sample or "(no sample available)",
    )


def generate_variations_for_query(
    client, base_query: dict, profile: str, sample: str | None, *, max_retries: int = 1,
) -> dict:
    """Generate `n=3` wording variations for one base query. Returns exactly one of:

      {"status": "ok", "variations": [text, text, text],
       "constraint_substrings": [...] | None}
      {"status": "failed", "reason": ...}

    Constraint-substring survival (Decision 2, task 1.3) is verified here: for
    facets carrying a constraint, every echoed substring must appear verbatim
    (case-insensitive) in every one of the `n` variations, or the attempt is
    retried the same way a facet/class mismatch is in `generate_for_cell`.
    Semantic fidelity and inter-variation word-overlap are NOT verified here
    (module docstring, design.md Decision 6) — prompt-only.
    """
    facet = base_query["facet"]
    query_class = base_query["query_class"]
    statistical_subtype = base_query.get("statistical_subtype")
    prompt = build_variation_prompt(
        base_query["text"], facet, query_class, profile, sample or "", statistical_subtype,
    )

    requires_constraint = _requires_constraint(facet, statistical_subtype)
    min_constraints = 2 if facet == "composite" else 1
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

        variations = parsed.get("variations")
        if not isinstance(variations, list) or len(variations) != N_VARIATIONS:
            last_reason = f"expected {N_VARIATIONS} variations, got {variations!r}"
            continue
        variations = [(v or "").strip() for v in variations]
        if not all(variations):
            last_reason = "one or more variations was empty"
            continue

        constraint_substrings = None
        if requires_constraint:
            constraint_substrings = parsed.get("constraint_substrings")
            if not isinstance(constraint_substrings, list) or len(constraint_substrings) < min_constraints:
                last_reason = (
                    f"expected >= {min_constraints} constraint_substrings for "
                    f"facet={facet!r}, got {constraint_substrings!r}"
                )
                continue
            missing = [
                s for s in constraint_substrings
                if not all(str(s).lower() in v.lower() for v in variations)
            ]
            if missing:
                last_reason = f"constraint substring(s) not preserved in every rewrite: {missing!r}"
                continue

        return {"status": "ok", "variations": variations, "constraint_substrings": constraint_substrings}

    return {"status": "failed", "reason": last_reason}


def run(client, os_client, storage_client, base_queries: list[dict], corpus_ids: set[str]) -> dict:
    """Generate variations for every base query (Decision 5's exclusion accounting).

    Returns {"variations": [...], "excluded_no_source": [...],
    "excluded_not_in_frame": [...], "generation_failures": [...]}.
    """
    variations: list[dict] = []
    excluded_no_source: list[dict] = []
    excluded_not_in_frame: list[dict] = []
    generation_failures: list[dict] = []

    doc_cache: dict[str, dict] = {}
    sample_cache: dict[str, str | None] = {}

    total = len(base_queries)
    for done, bq in enumerate(base_queries, start=1):
        dataset_id = bq.get("source_dataset_id")
        if not dataset_id:
            excluded_no_source.append({"query_id": bq.get("query_id")})
            print(f"  [{done}/{total}] {bq.get('query_id')}: excluded (no source_dataset_id)")
            continue
        if dataset_id not in corpus_ids:
            excluded_not_in_frame.append({"query_id": bq["query_id"], "source_dataset_id": dataset_id})
            print(f"  [{done}/{total}] {bq['query_id']}: excluded (dataset not in corpus frame)")
            continue

        if dataset_id not in doc_cache:
            doc_cache[dataset_id] = os_client.get(
                index=AUCTUS_INDEX_NAME, id=dataset_id, _source=list(NEUTRAL_SOURCE_FIELDS),
            ).get("_source") or {}
            record = load_full_profile(storage_client, dataset_id) if storage_client else None
            sample_cache[dataset_id] = record.get("sample") if isinstance(record, dict) else None

        bundle = build_neutral_bundle(doc_cache[dataset_id], sample_cache[dataset_id], include_title=False)
        assert "title" not in bundle, (
            "variation generator bundle carries a title — F1/§6f violation; the "
            "generator must be title-blind (design.md Decision 6)"
        )

        try:
            result = generate_variations_for_query(client, bq, bundle["profile"], bundle["sample"])
        except Exception as exc:
            result = {"status": "failed", "reason": f"unhandled exception: {exc}"}

        if result["status"] == "ok":
            for i, text in enumerate(result["variations"]):
                variations.append({
                    "text": text,
                    "base_query_id": bq["query_id"],
                    "source_dataset_id": dataset_id,
                    "facet": bq["facet"],
                    "query_class": bq["query_class"],
                    "statistical_subtype": bq.get("statistical_subtype"),
                    "variation_id": f"{bq['query_id']}_var{i}",
                    "constraint_substrings": result["constraint_substrings"],
                })
        else:
            generation_failures.append({
                "query_id": bq["query_id"], "source_dataset_id": dataset_id,
                "reason": result["reason"],
            })

        print(f"  [{done}/{total}] {bq['query_id']}: {result['status']}")

    return {
        "variations": variations,
        "excluded_no_source": excluded_no_source,
        "excluded_not_in_frame": excluded_not_in_frame,
        "generation_failures": generation_failures,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True,
                        help="base queries artifact, e.g. queries_production_2026-08-17.json")
    parser.add_argument("--out", default="eval/benchmark/variations.json")
    parser.add_argument("--corpus-frame", default=str(DEFAULT_CORPUS_FRAME))
    args = parser.parse_args(argv)

    base_queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    corpus_ids = set(load_corpus_ids(Path(args.corpus_frame)))

    os_client = get_client()
    client = get_llm_client()
    if client is None:
        raise SystemExit("No LLM client (PORTKEY_API_KEY set? on NYU VPN?).")
    try:
        storage_client = get_storage_client()
    except Exception:
        storage_client = None

    result = run(client, os_client, storage_client, base_queries, corpus_ids)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "code_version": code_version(),
            "n_variations_per_base": N_VARIATIONS,
            "base_queries_read": len(base_queries),
            "variations": result["variations"],
            "excluded_no_source": result["excluded_no_source"],
            "excluded_not_in_frame": result["excluded_not_in_frame"],
            "generation_failures": result["generation_failures"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8")

    n_succeeded = len(result["variations"]) // N_VARIATIONS if N_VARIATIONS else 0
    print(f"\n{len(base_queries)} base queries read, {len(result['variations'])} variations "
          f"produced ({n_succeeded} base queries succeeded), "
          f"{len(result['excluded_no_source'])} excluded (no source_dataset_id), "
          f"{len(result['excluded_not_in_frame'])} excluded (dataset not in corpus frame), "
          f"{len(result['generation_failures'])} generation failures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
