from typing import List, Optional
import json
import os
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from opensearchpy import OpenSearch, exceptions as os_exceptions
    from opensearchpy import helpers
except Exception:  # pragma: no cover - runtime
    OpenSearch = None
    helpers = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - runtime
    SentenceTransformer = None

router = APIRouter()
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


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


def build_query_vector(keywords: str) -> List[float]:
    model = get_embedding_model()
    vector = model.encode([keywords], convert_to_numpy=True)[0]
    try:
        return vector.tolist()
    except Exception:
        return [float(x) for x in vector]


@router.post("/api/v1/search")
async def search(req: SearchQueryRequest):
    try:
        client = get_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    filter_clauses = []
    should_clauses = []
    query_vector = None

    if req.keywords:
        query_vector = build_query_vector(req.keywords)

    if isinstance(req.source, list):
        source_values = [item for item in req.source if item]
        if source_values:
            filter_clauses.append({"terms": {"source": source_values}})
    elif req.source:
        filter_clauses.append({"term": {"source": {"value": req.source}}})

    if isinstance(req.types, list):
        type_values = [item for item in req.types if item]
        if type_values:
            filter_clauses.append({"terms": {"types": type_values}})
    elif req.types:
        filter_clauses.append({"term": {"types": {"value": req.types}}})

    # Temporal overlap: ensure dataset window overlaps requested window
    if req.temporal_start:
        # dataset.end >= temporal_start
        filter_clauses.append({"range": {"temporal_coverage.end": {"gte": req.temporal_start}}})
    if req.temporal_end:
        # dataset.start <= temporal_end
        filter_clauses.append({"range": {"temporal_coverage.start": {"lte": req.temporal_end}}})

    if req.bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = req.bbox
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox format; expected [min_lon, min_lat, max_lon, max_lat]")

        envelope = [[min_lon, max_lat], [max_lon, min_lat]]
        filter_clauses.append(
            {
                "geo_shape": {
                    "spatial_coverage.bbox": {
                        "shape": {"type": "envelope", "coordinates": envelope},
                        "relation": "intersects",
                    }
                }
            }
        )

    if req.keywords:
        should_clauses.append(
            {
                "multi_match": {
                    "query": req.keywords,
                    "fields": ["title^2", "description"],
                }
            }
        )
        should_clauses.append(
            {
                "knn": {
                    "dataset_vector": {
                        "vector": query_vector,
                        "k": 10,
                    }
                }
            }
        )

    query_bool = {"filter": filter_clauses}
    if should_clauses:
        query_bool["should"] = should_clauses
        query_bool["minimum_should_match"] = 1  # Require at least one text or vector match when keywords are present

    payload = {
        "query": {"bool": query_bool},
        "aggs": {
            "sources_count": {"terms": {"field": "source"}},
            "types_count": {"terms": {"field": "types"}},
        },
    }
    # print("RAW OPENSEARCH PAYLOAD:", json.dumps(payload, indent=2))

    try:
        resp = client.search(index="auctus_catalog_master", body=payload, size=req.limit, from_=req.offset)
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
