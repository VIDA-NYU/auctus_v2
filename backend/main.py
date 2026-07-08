import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from storage.opensearch_client import (
    AUCTUS_INDEX_NAME,
    DEFAULT_DESCRIPTION_SOURCE,
    DEFAULT_TITLE_BOOST,
    description_fields_for,
    get_client,
    init_db,
)
from api.search import router as search_router
from api.datasets import router as datasets_router
from app.api.endpoints.portals import router as portals_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app startup and shutdown."""
    # Startup
    logger.info("FastAPI startup: initializing OpenSearch...")
    try:
        init_db()
        logger.info("OpenSearch initialization complete")
    except Exception as e:
        logger.warning(
            f"OpenSearch initialization failed: {e}. Falling back to JSON-based search."
        )
    yield
    # Shutdown
    logger.info("FastAPI shutdown")


app = FastAPI(title="Auctus v2 API", lifespan=lifespan)

# Include API routers
app.include_router(search_router)
app.include_router(datasets_router)
app.include_router(portals_router)

# --- CORS Configuration ---
# This allows your React app to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class SearchFilters(BaseModel):
    date_range: Optional[dict] = None
    location: Optional[dict] = None
    source: Optional[List[str]] = None
    data_type: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    filters: Optional[SearchFilters] = None
    # Which description the full-text query targets: the original portal description,
    # the AutoDDG UFD, or the AutoDDG SFD.
    description_source: Literal["original", "ufd", "sfd"] = DEFAULT_DESCRIPTION_SOURCE
    # Number of hits to return. Without this OpenSearch silently caps at 10.
    size: int = Field(default=10, ge=1, le=200)
    # "and" (default) requires every query term to match within one field;
    # "or" is plain BM25, more forgiving for verbose natural-language queries.
    match_operator: Literal["and", "or"] = "and"
    # Weight of the title field in the match; 0 drops the title entirely.
    title_boost: float = Field(default=DEFAULT_TITLE_BOOST, ge=0)

# --- Routes ---
@app.get("/")
def read_root():
    return {"status": "Auctus v2 Backend Running"}

@app.post("/search")
async def search(request: SearchRequest):
    """Search auctus_catalog_master via OpenSearch multi_match against title and description."""
    try:
        client = get_client()

        must_clauses = [
            {
                "multi_match": {
                    "query": request.query,
                    "fields": description_fields_for(
                        request.description_source, title_boost=request.title_boost
                    ),
                    "type": "best_fields",
                    "operator": request.match_operator,
                }
            }
        ]

        filter_clauses = []
        if request.filters and request.filters.data_type:
            filter_clauses.append({"terms": {"types": request.filters.data_type}})

        search_body = {
            "size": request.size,
            "query": {
                "bool": {
                    "must": must_clauses,
                    **({"filter": filter_clauses} if filter_clauses else {}),
                }
            }
        }

        response = client.search(index=AUCTUS_INDEX_NAME, body=search_body)
        hits = response.get("hits", {}).get("hits", [])

        results = []
        for hit in hits:
            source = hit.get("_source", {}) or {}
            score = hit.get("_score")
            results.append(
                {
                    **source,
                    "_score": score,
                    "score": score,
                }
            )

        total = response.get("hits", {}).get("total", {})
        total_results = total.get("value", len(results)) if isinstance(total, dict) else len(results)

        return {
            "query": request.query,
            "total_results": total_results,
            "results": results,
        }
    except Exception as e:
        logger.error(f"OpenSearch search failed: {e}")
        raise HTTPException(status_code=503, detail="Search backend unavailable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)