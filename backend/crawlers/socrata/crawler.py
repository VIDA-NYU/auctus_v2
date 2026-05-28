"""Async Socrata portal crawler for discovering public tabular datasets.

This module queries Socrata's catalog API, filters to viable tabular assets,
and returns unique 4x4 dataset identifiers suitable for downstream ingestion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)

HTTP_TIMEOUT = httpx.Timeout(30.0, connect=15.0)
FOUR_BY_FOUR_RE = re.compile(r"^[a-z0-9]{4}-[a-z0-9]{4}$", re.IGNORECASE)


def _load_active_portal_domain() -> str:
    """Return the active portal domain from backend/config/config.json when available."""
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "config.json"
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        active_portal = cfg.get("active_portal")
        if isinstance(active_portal, str) and active_portal.strip():
            return active_portal.strip()
    except Exception:
        pass

    return "data.cityofnewyork.us"



def _configure_logging() -> None:
    """Configure a simple default logger if the application has not done so."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )



def _extract_dataset_id(asset: dict[str, Any]) -> str | None:
    """Extract a normalized Socrata 4x4 identifier from a catalog asset."""
    candidate_fields = (
        asset.get("id"),
        asset.get("resource") and asset["resource"].get("id"),
        asset.get("resource", {}).get("name"),
        asset.get("permalink"),
        asset.get("link"),
        asset.get("metadata", {}).get("id"),
    )

    for candidate in candidate_fields:
        if not candidate:
            continue
        text = str(candidate).strip()
        if FOUR_BY_FOUR_RE.match(text):
            return text.lower()

    return None



def _is_viable_tabular_asset(asset: dict[str, Any]) -> bool:
    """Return True when the catalog entry looks like a tabular dataset."""
    asset_type = str(asset.get("type") or asset.get("resource", {}).get("type") or "").lower()
    if asset_type not in {"dataset", "tabular"}:
        return False

    # Exclude obvious non-tabular or external assets.
    if asset_type in {"chart", "map", "story", "filter", "visualization", "external_link"}:
        return False

    download_count = asset.get("resource", {}).get("download_count")
    if download_count is not None and not isinstance(download_count, (int, float)):
        return False

    return True


async def discover_socrata_datasets(
    domain: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[str]:
    """Discover public tabular Socrata datasets and return unique 4x4 identifiers."""
    _configure_logging()
    domain = domain or _load_active_portal_domain()

    catalog_url = (
        "https://api.us.socrata.com/api/catalog/v1"
        f"?domains={domain}&search_context={domain}&limit={limit}&offset={offset}"
    )

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(catalog_url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        LOGGER.error("Failed to discover Socrata datasets from %s: %s", domain, exc)
        return []

    results = payload.get("results") or payload.get("data") or []
    dataset_ids: list[str] = []
    seen: set[str] = set()

    for asset in results:
        if not isinstance(asset, dict):
            continue
        if not _is_viable_tabular_asset(asset):
            continue

        dataset_id = _extract_dataset_id(asset)
        if dataset_id and dataset_id not in seen:
            seen.add(dataset_id)
            dataset_ids.append(dataset_id)

    return dataset_ids


async def main() -> None:
    """Test the crawler against the NYC Socrata portal and print the first 10 IDs."""
    _configure_logging()
    datasets = await discover_socrata_datasets(limit=10, offset=0)
    print(datasets)


if __name__ == "__main__":
    asyncio.run(main())
