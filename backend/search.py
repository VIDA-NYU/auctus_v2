from typing import List, Optional
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from opensearchpy import OpenSearch, exceptions as os_exceptions
    from opensearchpy import helpers
except Exception:  # pragma: no cover - runtime
    OpenSearch = None
    helpers = None

router = APIRouter()


class SearchQueryRequest(BaseModel):
    keywords: Optional[str] = None
    source: Optional[List[str]] = None
    types: Optional[List[str]] = None
    temporal_start: Optional[str] = None
    temporal_end: Optional[str] = None
    bbox: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
    limit: int = 10
    offset: int = 0


def get_client():
    if OpenSearch is None:
        raise RuntimeError("opensearch-py not available")

    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    user = os.getenv("OPENSEARCH_USER") or os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASS") or os.getenv("OPENSEARCH_PASSWORD")

    conn_kwargs = {
        "hosts": [{"host": host, "port": port}],
        "use_ssl": False,
        "verify_certs": False,
        "ssl_show_warn": False,
    }
    if user and password:
        conn_kwargs["http_auth"] = (user, password)

    return OpenSearch(**conn_kwargs)


@router.post("/api/v1/search")
async def search(req: SearchQueryRequest):
    try:
        client = get_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    must_clauses = []

    if req.keywords:
        must_clauses.append(
            {
                "multi_match": {
                    "query": req.keywords,
                    "fields": ["title^2", "description"],
                    "operator": "and",
                }
            }
        )

    if req.source:
        # support both single string and list of strings
        if isinstance(req.source, list):
            must_clauses.append({"terms": {"source": req.source}})
        else:
            must_clauses.append({"term": {"source": {"value": req.source}}})

    if req.types:
        # support both single string and list of strings
        if isinstance(req.types, list):
            must_clauses.append({"terms": {"types": req.types}})
        else:
            must_clauses.append({"term": {"types": {"value": req.types}}})

    # Temporal overlap: ensure dataset window overlaps requested window
    if req.temporal_start:
        # dataset.end >= temporal_start
        must_clauses.append({"range": {"temporal_coverage.end": {"gte": req.temporal_start}}})
    if req.temporal_end:
        # dataset.start <= temporal_end
        must_clauses.append({"range": {"temporal_coverage.start": {"lte": req.temporal_end}}})

    if req.bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = req.bbox
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox format; expected [min_lon, min_lat, max_lon, max_lat]")

        envelope = [[min_lon, max_lat], [max_lon, min_lat]]
        must_clauses.append(
            {
                "geo_shape": {
                    "spatial_coverage.bbox": {
                        "shape": {"type": "envelope", "coordinates": envelope},
                        "relation": "intersects",
                    }
                }
            }
        )

    body = {
        "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
        "aggs": {
            "sources_count": {"terms": {"field": "source"}},
            "types_count": {"terms": {"field": "types"}},
        },
    }

    try:
        resp = client.search(index="datasets", body=body, size=req.limit, from_=req.offset)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search backend error: {exc}")

    hits = resp.get("hits", {}).get("hits", [])
    total = resp.get("hits", {}).get("total")
    if isinstance(total, dict):
        total_count = total.get("value", 0)
    else:
        total_count = int(total or 0)

    results = []
    for h in hits:
        src = h.get("_source", {})

        # Remove large plot objects from profiler_metadata.columns
        prof = src.get("profiler_metadata")
        if isinstance(prof, dict):
            cols = prof.get("columns")
            if isinstance(cols, list):
                for c in cols:
                    if isinstance(c, dict) and "plot" in c:
                        c.pop("plot", None)

        results.append(src)

    aggregations = resp.get("aggregations", {})

    return {"total": total_count, "results": results, "aggregations": aggregations}
