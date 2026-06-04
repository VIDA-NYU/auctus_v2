"""Portals summary endpoint backed by the OpenSearch summary index."""

from __future__ import annotations

import logging

from fastapi import APIRouter

try:
    from opensearchpy import NotFoundError
except Exception:  # pragma: no cover - optional dependency fallback
    NotFoundError = Exception

from storage.opensearch_client import AUCTUS_PORTALS_INDEX_NAME, get_client

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_PORTALS = [
    {"domain": "data.cityofnewyork.us", "display_label": "Cityofnewyork.Us", "dataset_count": 0},
    {"domain": "data.cityofchicago.org", "display_label": "Cityofchicago", "dataset_count": 0},
]


@router.get("/api/v1/portals")
async def get_portals_summary():
    """Return portal summaries sorted by dataset count."""
    try:
        client = get_client()
        response = client.search(
            index=AUCTUS_PORTALS_INDEX_NAME,
            body={
                "query": {"match_all": {}},
                "sort": [{"dataset_count": {"order": "desc"}}],
                "size": 100,
            },
        )

        hits = response.get("hits", {}).get("hits", [])
        portals = []
        for hit in hits:
            source = hit.get("_source", {}) or {}
            portals.append(
                {
                    "domain": source.get("domain"),
                    "display_label": source.get("display_label") or source.get("domain"),
                    "dataset_count": int(source.get("dataset_count", 0) or 0),
                }
            )

        return {"portals": portals}
    except NotFoundError:
        logger.warning("Portal summary index %s not found; returning defaults", AUCTUS_PORTALS_INDEX_NAME)
        return {"portals": DEFAULT_PORTALS}
    except Exception as exc:
        logger.warning("Portal summary query failed: %s", exc)
        return {"portals": DEFAULT_PORTALS}