"""Research compendia: one entry per repository record, with its parts and relations
(crawlers/invenio/compendium.py builds them after ingestion)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from api.search import get_client
from crawlers.invenio.compendium import COMPENDIUM_INDEX

router = APIRouter()


@router.get("/api/v1/compendia")
async def list_compendia(q: Optional[str] = None, domain: Optional[str] = None, limit: int = 20, offset: int = 0):
    """Records as compendia: all of them, or those matching `q` (title, description, summary, subjects)."""
    client = get_client()
    filters = [{"term": {"domain": domain}}] if domain else []
    query = {"bool": {"filter": filters,
                      "must": [{"multi_match": {"query": q, "fields": ["title^3", "description^2", "subjects", "summary"]}}]
                      if q else [{"match_all": {}}]}}
    try:
        resp = client.search(index=COMPENDIUM_INDEX, body={"query": query, "_source": {"excludes": ["parts", "relations"]}},
                             size=limit, from_=offset)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Compendium index unavailable: {exc}")
    total = resp["hits"]["total"]
    return {"total": total.get("value", 0) if isinstance(total, dict) else total,
            "results": [h["_source"] for h in resp["hits"]["hits"]]}


@router.get("/api/v1/compendia/{record_id}")
async def get_compendium(record_id: str, domain: str = "ultraviolet.library.nyu.edu"):
    """One record's compendium: parts (with roles and dataset ids) and relations."""
    try:
        doc = get_client().get(index=COMPENDIUM_INDEX, id=f"{domain}:{record_id}")
    except Exception:
        raise HTTPException(status_code=404, detail=f"No compendium for {domain}:{record_id}")
    return doc["_source"]
