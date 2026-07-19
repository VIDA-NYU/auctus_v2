"""Build a reproducible NYC-area EDA corpus slice manifest.

Reads the ingested datasets straight from OpenSearch and emits a manifest that
records, per dataset: provenance, arm coverage, and the metadata-completeness /
tabular / near-duplicate flags the benchmark needs. The manifest is the P0
artifact — the fixed list of ids the query generator, pool builder, and judge
all operate on.

Cleaning philosophy (report §1 / 07-15 a0): clean only by *removing non-dataset
assets and duplicates*; PRESERVE the natural distribution of description quality.
Empty/short descriptions and unnamed columns are FLAGGED, not dropped — "how bad
are wild descriptions, and how much does each arm help under them" is part of the
result. So this script never deletes; it tags. `included` is the recommended
evaluation set; excluded ids carry a reason and can be re-included by config.

    python -m eval.build_corpus_slice --out eval/benchmark/corpus_slice.json
    python -m eval.build_corpus_slice --domains data.cityofnewyork.us health.data.ny.gov
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

LOGGER = logging.getLogger("build_corpus_slice")

# Arms that must be present for a dataset to be scorable across the full matrix.
ARM_FIELDS = {
    "original": "description",
    "llm_direct": "llm_direct_description",
    "ufd": "autoddg_description",
    "sfd": "autoddg_search_description",
    "profile_only": "profile_only_description",
    "t_od_s": "tods_description",
}
SHORT_DESCRIPTION_CHARS = 80  # below this the in-the-wild description is "thin"
UNNAMED_COL_RE = re.compile(r"^(col|column|unnamed|field)[_ ]?\d+$", re.IGNORECASE)


def _norm_title(title: str) -> str:
    """Normalise a title for near-duplicate grouping: lowercase, strip years and
    snapshot qualifiers, collapse whitespace. Yearly snapshots of one dataset
    then collapse to the same key."""
    t = (title or "").lower()
    t = re.sub(r"\b(19|20)\d{2}\b", "", t)  # drop 4-digit years
    t = re.sub(r"\b(archived|current|ytd|q[1-4]|fy)\b", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _analyze(doc_id: str, src: dict) -> dict:
    pm = src.get("profiler_metadata") if isinstance(src.get("profiler_metadata"), dict) else {}
    columns = [c for c in (pm.get("columns") or []) if isinstance(c, dict)]
    col_names = [c.get("name", "") for c in columns]
    description = src.get("description") or ""

    # Tabular signal: a profiled table has columns and/or a row count. Non-dataset
    # assets (maps, story pages, API-only) profile to no schema.
    nb_rows = pm.get("nb_rows")
    is_tabular = bool(columns) or bool(nb_rows)

    arm_coverage = {arm: bool(src.get(field)) for arm, field in ARM_FIELDS.items()}

    flags = {
        "empty_description": not description.strip(),
        "short_description": 0 < len(description.strip()) < SHORT_DESCRIPTION_CHARS,
        "unnamed_columns": any(UNNAMED_COL_RE.match(n or "") for n in col_names),
        "no_profile": not pm,
        "not_tabular": not is_tabular,
        "missing_arms": [a for a, ok in arm_coverage.items() if not ok],
    }
    return {
        "id": doc_id,
        "domain": src.get("domain"),
        "provider": src.get("provider"),
        "title": src.get("title"),
        "nb_rows": nb_rows,
        "nb_columns": pm.get("nb_columns") or len(columns),
        "description_len": len(description.strip()),
        "norm_title": _norm_title(src.get("title") or ""),
        "arm_coverage": arm_coverage,
        "flags": flags,
    }


def build_slice(os_client, domains: list[str] | None) -> dict:
    query = {"match_all": {}}
    if domains:
        query = {"terms": {"domain": domains}}
    resp = os_client.search(
        index=AUCTUS_INDEX_NAME,
        body={"query": query, "_source": ["domain", "provider", "title", "description",
                                          "profiler_metadata", *ARM_FIELDS.values()]},
        size=1000,
    )
    entries = [_analyze(h["_id"], h.get("_source") or {}) for h in resp["hits"]["hits"]]

    # Near-duplicate grouping: same normalised title -> keep the first as the
    # representative, tag the rest as near_duplicate so they don't pollute qrels.
    by_title: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_title[e["norm_title"]].append(e["id"])
    representative = {ids[0] for ids in by_title.values()}

    for e in entries:
        reasons = []
        if e["flags"]["not_tabular"]:
            reasons.append("not_tabular")
        if e["id"] not in representative:
            reasons.append("near_duplicate")
        if e["flags"]["missing_arms"]:
            reasons.append("missing_arms:" + ",".join(e["flags"]["missing_arms"]))
        e["included"] = not reasons
        e["exclude_reasons"] = reasons

    included = [e for e in entries if e["included"]]
    return {
        "index": AUCTUS_INDEX_NAME,
        "total": len(entries),
        "included_count": len(included),
        "domains": sorted({e["domain"] for e in entries if e["domain"]}),
        "flag_summary": {
            k: sum(1 for e in entries if e["flags"].get(k))
            for k in ("empty_description", "short_description", "unnamed_columns",
                      "no_profile", "not_tabular")
        },
        "near_duplicate_groups": {t: ids for t, ids in by_title.items() if len(ids) > 1},
        "datasets": entries,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", help="Restrict to these portal domains")
    parser.add_argument("--out", default="eval/benchmark/corpus_slice.json")
    args = parser.parse_args(argv)

    manifest = build_slice(get_client(), args.domains)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Corpus slice: {manifest['included_count']}/{manifest['total']} included "
          f"across domains {manifest['domains']}")
    print(f"Flags: {manifest['flag_summary']}")
    if manifest["near_duplicate_groups"]:
        print(f"Near-duplicate groups: {manifest['near_duplicate_groups']}")
    print(f"Manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
