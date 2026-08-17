"""Pull a trimmed, dated snapshot of the NYC Socrata catalog and re-derive
the catalog-wide figures the stratified sampler (stratified_sample.py) draws
against.

`discover_socrata_datasets` (crawlers/socrata/crawler.py) cannot be reused
here: it returns bare 4x4 ids and reads none of `classification.domain_category`
or the attribution field the stratification needs (design.md, "Context"). This
module reads the same catalog API directly and keeps the fields sampling
consumes.

    python -m eval.catalog_frame --out-dir eval/frame

Writes two files per pull: a catalog snapshot and a figures report, both named
with the pull date so a re-sample adds a new pair rather than overwriting the
previous round's (design.md D1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from crawlers.socrata.crawler import (
    HTTP_TIMEOUT,
    _extract_dataset_id,
    _is_viable_tabular_asset,
    _load_active_portal_domain,
)
from eval.build_corpus_slice import SHORT_DESCRIPTION_CHARS

LOGGER = logging.getLogger("catalog_frame")

CATALOG_PAGE_SIZE = 500

# Last-known catalog-wide figures, from a single pull written to a session
# scratchpad that no longer exists (design.md D3). Retained only so the fresh
# pull can be diffed against them — never copied into the fresh figures, and
# never used to tune the draw. A large gap is a finding about catalog drift,
# not a bug in this module.
LAST_KNOWN_FIGURES = {
    "short_description_share": 0.173,
    "categories": {
        "Education": {"count": 582, "share": 0.243, "median_len": 139, "short_share": 0.357},
    },
}


def _trim_asset(asset: dict[str, Any]) -> dict[str, Any] | None:
    """Trim one catalog asset to the fields sampling consumes, or None if it
    is not a viable tabular dataset (same predicate the crawler applies)."""
    if not _is_viable_tabular_asset(asset):
        return None
    dataset_id = _extract_dataset_id(asset)
    if not dataset_id:
        return None

    resource = asset.get("resource") or {}
    classification = asset.get("classification") or {}
    description = str(resource.get("description") or "").strip()
    agency = str(resource.get("attribution") or "").strip() or None

    return {
        "id": dataset_id,
        "title": resource.get("name") or "",
        "description": description,
        "description_len": len(description),
        "domain_category": classification.get("domain_category") or None,
        "agency": agency,
        "asset_type": str(resource.get("type") or "").lower(),
    }


async def pull_catalog(domain: str, page_size: int = CATALOG_PAGE_SIZE) -> tuple[list[dict], int]:
    """Page through the Socrata catalog API for `domain` and return the
    trimmed, viable-tabular records plus the portal's reported result-set size
    (which includes non-tabular assets the trim drops, so it is not the same
    number as `len(records)`)."""
    records: list[dict] = []
    seen_ids: set[str] = set()
    offset = 0
    result_set_size = 0

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        while True:
            url = (
                "https://api.us.socrata.com/api/catalog/v1"
                f"?domains={domain}&search_context={domain}"
                f"&limit={page_size}&offset={offset}"
            )
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()

            result_set_size = int(payload.get("resultSetSize") or 0)
            results = payload.get("results") or []
            if not results:
                break

            for asset in results:
                if not isinstance(asset, dict):
                    continue
                trimmed = _trim_asset(asset)
                if trimmed and trimmed["id"] not in seen_ids:
                    seen_ids.add(trimmed["id"])
                    records.append(trimmed)

            offset += page_size
            if offset >= result_set_size:
                break

    return records, result_set_size


def compute_figures(records: list[dict]) -> dict[str, Any]:
    """Re-derive the description-length distribution and per-category shares
    from this pull (tasks.md 2.1/2.2). Every number here comes from `records`;
    nothing is copied from LAST_KNOWN_FIGURES."""
    total = len(records)
    lens = sorted(r["description_len"] for r in records)

    deciles = {}
    if len(lens) >= 2:
        cut_points = statistics.quantiles(lens, n=10)
        deciles = {f"p{(i + 1) * 10}": cut_points[i] for i in range(len(cut_points))}
    short_count = sum(1 for l in lens if l < SHORT_DESCRIPTION_CHARS)

    by_category: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_category[r["domain_category"] or "(uncategorized)"].append(r["description_len"])

    categories = {}
    for cat, cat_lens in by_category.items():
        cat_lens_sorted = sorted(cat_lens)
        cat_total = len(cat_lens_sorted)
        cat_short = sum(1 for l in cat_lens_sorted if l < SHORT_DESCRIPTION_CHARS)
        categories[cat] = {
            "count": cat_total,
            "share": cat_total / total if total else 0.0,
            "median_len": statistics.median(cat_lens_sorted) if cat_lens_sorted else None,
            "short_share": cat_short / cat_total if cat_total else 0.0,
        }

    return {
        "total": total,
        "description_len_deciles": deciles,
        "short_description_count": short_count,
        "short_description_share": short_count / total if total else 0.0,
        "categories": categories,
    }


def diff_against_last_known(figures: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side fresh vs. last-known, for the numbers LAST_KNOWN_FIGURES
    covers (tasks.md 2.3). Purely a report; never mutates `figures`."""
    fresh_short = figures["short_description_share"]
    last_short = LAST_KNOWN_FIGURES["short_description_share"]
    diff = {
        "short_description_share": {
            "fresh": fresh_short,
            "last_known": last_short,
            "delta": fresh_short - last_short,
        },
        "categories": {},
    }
    for cat, last_stats in LAST_KNOWN_FIGURES["categories"].items():
        fresh_stats = figures["categories"].get(cat)
        diff["categories"][cat] = {"fresh": fresh_stats, "last_known": last_stats}
    return diff


def write_snapshot(
    records: list[dict],
    domain: str,
    result_set_size: int,
    pulled_at: str,
    out_path: Path,
) -> None:
    payload = {
        "domain": domain,
        "pulled_at": pulled_at,
        "catalog_result_set_size": result_set_size,
        "kept_count": len(records),
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_figures(figures: dict[str, Any], diff: dict[str, Any], pulled_at: str, out_path: Path) -> None:
    payload = {"pulled_at": pulled_at, "figures": figures, "diff_vs_last_known": diff}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _run(domain: str, out_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pulled_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    round_date = pulled_at[:10]

    LOGGER.info("Pulling catalog for domain %s ...", domain)
    records, result_set_size = await pull_catalog(domain)
    LOGGER.info(
        "Pulled %d viable tabular datasets (portal reports %d total assets)",
        len(records),
        result_set_size,
    )

    figures = compute_figures(records)
    diff = diff_against_last_known(figures)

    snapshot_path = out_dir / f"catalog_snapshot_{domain}_{round_date}.json"
    figures_path = out_dir / f"catalog_figures_{domain}_{round_date}.json"
    write_snapshot(records, domain, result_set_size, pulled_at, snapshot_path)
    write_figures(figures, diff, pulled_at, figures_path)

    LOGGER.info("Snapshot -> %s", snapshot_path)
    LOGGER.info("Figures -> %s", figures_path)
    LOGGER.info(
        "Short-description share: fresh=%.3f last_known=%.3f",
        figures["short_description_share"],
        LAST_KNOWN_FIGURES["short_description_share"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=_load_active_portal_domain())
    parser.add_argument("--out-dir", default="eval/frame")
    args = parser.parse_args(argv)

    asyncio.run(_run(args.domain, Path(args.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
