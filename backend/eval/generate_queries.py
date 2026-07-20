"""Facet-comprehensive query generation, grounded ONLY on neutral facts.

The AutoDDG-paper benchmark uses topic-keyword queries — the one facet where
every description arm ties. To exercise AutoDDG's full potential we generate a
facet x query-class grid (report §3) that probes the facets where AutoDDG's
grounding and SFD expansion actually matter: temporal, spatial, schema, granularity,
entity, and vocabulary-mismatch.

CRITICAL leakage guard (F1). Queries are generated from a NEUTRAL bundle only —
title + algorithmic profile facts + a data sample — and NEVER from any description
arm's prose. A query written from arm X's text would trivially match X and let X
"win" by construction. So ``build_neutral_bundle`` reads raw ``profiler_metadata``
(via the same ``build_profile_text`` used elsewhere), the title, and the sample —
never a value in ``DESCRIPTION_SOURCE_FIELDS``. One representation-agnostic query
set is produced and later applied identically to every arm. ``source_dataset_id``
is kept for the leakage audit only; it is NOT a qrel (qrels come from the judge
over the pool — F8).

    python -m eval.generate_queries --slice eval/benchmark/corpus_slice.json \
        --out eval/benchmark/queries.json
"""

from __future__ import annotations

import argparse
import json
import logging
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
from storage.minio_client import get_storage_client

LOGGER = logging.getLogger("generate_queries")

# The ONLY fields build_neutral_bundle may read off a document. Any description
# arm is forbidden grounding — see the module docstring (F1).
FORBIDDEN_ARM_FIELDS = frozenset(DESCRIPTION_SOURCE_FIELDS.values())
SAMPLE_CHARS = 1500

QUERY_CLASSES = ("keyword", "nl_requesting", "nl_describing", "nl_implying")
FACETS = ("topic", "temporal", "spatial", "schema", "statistical", "entity",
          "vocabulary_mismatch", "composite")

PROMPT = """You are constructing search queries a real user might issue to find a \
government open-data dataset. You are given ONLY neutral facts about one dataset \
(its title, an algorithmic profile of its columns/coverage, and a small data sample). \
Do NOT assume any prose description exists.

Produce queries spanning these query classes:
- keyword: short, like a real portal search — place/agency/topic terms, NOT a sentence \
(e.g. "bike accidents brooklyn").
- nl_requesting: a natural-language question asking for the data.
- nl_describing: a natural-language description of the data need.
- nl_implying: a natural-language question whose data need is implied, not stated.

Across the whole set, cover as many of these FACETS as the data supports, and you \
MUST include at least one `vocabulary_mismatch` query (uses a synonym/related term that \
is NOT in the title) and at least one `composite` (combines 2+ facets): \
topic, temporal, spatial, schema (specific columns/attributes), statistical (record \
count/granularity), entity/measure, vocabulary_mismatch, composite.

Skip facets the data does not support (e.g. no spatial columns -> no spatial query).

Return STRICT JSON: {{"queries": [{{"text": "...", "query_class": "<one of the four>", \
"facet": "<one facet>"}}]}}. Aim for 8-10 queries. No prose outside the JSON.

NEUTRAL FACTS:
Title: {title}
Profile: {profile}
Data sample (first rows):
{sample}
"""


def build_neutral_bundle(doc: dict, sample: str | None) -> dict:
    """Neutral grounding: title + profile facts + sample. Never an arm's prose.

    ``build_profile_text`` recomputes the profile from raw ``profiler_metadata``,
    so even though the ``profile_only`` arm holds the same facts, this reads the
    metadata — not the arm field. No ``DESCRIPTION_SOURCE_FIELDS`` value is touched.
    """
    return {
        "title": doc.get("title") or "",
        "profile": build_profile_text(doc),
        "sample": (sample or "")[:SAMPLE_CHARS],
    }


def assert_no_arm_leak(bundle: dict, doc: dict) -> None:
    """Invariant (F1): no arm's stored prose may appear in the neutral bundle."""
    blob = json.dumps(bundle, ensure_ascii=False)
    for field in FORBIDDEN_ARM_FIELDS:
        if field == "description":
            continue  # 'description' is the ORIGINAL arm; never read here either,
            # but its text can coincide with facts — we assert on generated arms.
        val = doc.get(field)
        if val and isinstance(val, str) and val.strip() and val.strip() in blob:
            raise AssertionError(
                f"Arm field {field!r} leaked into the neutral query-generation bundle"
            )


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    return json.loads(text)


def generate_for_dataset(client, doc: dict, sample: str | None) -> list[dict]:
    bundle = build_neutral_bundle(doc, sample)
    assert_no_arm_leak(bundle, doc)
    prompt = PROMPT.format(title=bundle["title"], profile=bundle["profile"],
                           sample=bundle["sample"] or "(no sample available)")
    parsed = _parse_json(complete(client, prompt, temperature=0.0))
    out = []
    for q in parsed.get("queries", []):
        qc = q.get("query_class")
        if not q.get("text") or qc not in QUERY_CLASSES:
            continue
        out.append({
            "text": q["text"].strip(),
            "query_class": qc,
            "facet": q.get("facet", "unspecified"),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", default="eval/benchmark/corpus_slice.json")
    parser.add_argument("--out", default="eval/benchmark/queries.json")
    parser.add_argument("--ids", nargs="+", help="Limit to these dataset ids")
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

    all_queries = []
    for i, dataset_id in enumerate(ids, 1):
        doc = os_client.get(
            index=AUCTUS_INDEX_NAME, id=dataset_id,
            _source=["title", "profiler_metadata", "spatial_coverage"],
        ).get("_source") or {}
        record = load_full_profile(storage_client, dataset_id) if storage_client else None
        sample = record.get("sample") if isinstance(record, dict) else None
        try:
            qs = generate_for_dataset(client, doc, sample)
        except Exception as exc:
            LOGGER.warning("Query gen failed for %s: %s", dataset_id, exc)
            qs = []
        for j, q in enumerate(qs):
            q["query_id"] = f"{dataset_id}_{j}"
            q["source_dataset_id"] = dataset_id  # audit only; NOT a qrel (F8)
            all_queries.append(q)
        print(f"  [{i}/{len(ids)}] {dataset_id}: {len(qs)} queries")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"queries": all_queries}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    by_class: dict[str, int] = {}
    by_facet: dict[str, int] = {}
    for q in all_queries:
        by_class[q["query_class"]] = by_class.get(q["query_class"], 0) + 1
        by_facet[q["facet"]] = by_facet.get(q["facet"], 0) + 1
    print(f"\n{len(all_queries)} queries -> {out}")
    print(f"by class: {by_class}")
    print(f"by facet: {by_facet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
