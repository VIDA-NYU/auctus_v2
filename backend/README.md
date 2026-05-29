# Auctus v2 Backend

![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-blue?logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.9+-green?logo=python)
![OpenSearch](https://img.shields.io/badge/OpenSearch-2.13-orange?logo=opensearch)

## Overview

The **Auctus v2 Backend** is a FastAPI-powered hybrid semantic search engine that combines lexical full-text matching with k-NN vector similarity search over datasets profiled by the [atlas-profiler](https://github.com/uchicago-dsi-cdac/dfencoder) framework.

Each dataset is enriched with:
- A 384-dimensional embedding vector generated via `all-MiniLM-L6-v2` (SentenceTransformers)
- Metadata tracking (model name, version)
- Rich profiler metadata (column statistics, structural/semantic types, data quality metrics)

The hybrid search strategy flexibly combines:
- **Lexical scoring** via multi-field `multi_match` (title^2 + description)
- **Semantic scoring** via k-NN vector similarity on dataset embeddings
- **Hard filters** for source, data types, temporal ranges, and spatial bounding boxes

---

## Architecture Map

### File Structure

```
backend/
├── api/
│   └── search.py                    # FastAPI search router (/api/v1/search)
├── config/
│   └── config.json                  # Centralized portal + pipeline settings
├── crawlers/
│   └── socrata/
│       ├── crawler.py               # Socrata dataset discovery / ID harvesting
│       └── transformer.py           # CSV sampling, profiling, and metadata normalization
├── storage/
│   ├── minio_client.py              # MinIO client + profile upload helpers
│   ├── opensearch_client.py         # OpenSearch client, mapping, and helpers
│   └── arq_worker.py                # ARQ worker tasks for heavy ingestion jobs
│   └── initialize_os.py              # Schema-only OpenSearch index initializer
├── main.py                          # FastAPI app entrypoint; lifespan hooks for OpenSearch init
├── seed_synthetic.py                # Optional quick-start synthetic seeding utility
├── run_pipeline_ingest.py           # Dispatcher: discover → dedupe → enqueue ARQ jobs
├── data/
│   └── synthetic_datasets.json      # Local synthetic dataset seed file
└── README.md                        # This file
```

### Component Responsibilities

| File | Purpose |
|------|---------|
| **main.py** | FastAPI application factory; registers the `api.search` router; manages OpenSearch initialization via lifespan hook; responds to `/` health check |
| **api/search.py** | Core search business logic; defines `POST /api/v1/search`; constructs hybrid bool queries with optional k-NN; handles payload logging for debugging |
| **crawlers/socrata/crawler.py** | Discovers Socrata dataset IDs from the active portal using the centralized config |
| **crawlers/socrata/transformer.py** | Streams CSV samples, profiles datasets with atlas-profiler, and normalizes catalog metadata |
| **seed_synthetic.py** | Optional quick-start seeding script; generates embeddings and bulk-indexes synthetic datasets |
| **run_pipeline_ingest.py** | Dispatcher loop: discover datasets, compare Socrata/OpenSearch timestamps, and enqueue heavy work to ARQ |
| **storage/arq_worker.py** | ARQ worker implementation; downloads samples, profiles, uploads full records, generates embeddings, and indexes trimmed docs |
| **storage/minio_client.py** | MinIO connection + bucket/profile upload helpers |
| **storage/opensearch_client.py** | Low-level OpenSearch helpers; client connection; index mapping definition; data transform utilities |
| **storage/initialize_os.py** | Standalone schema initializer; deletes/recreates `auctus_catalog_master` and applies mappings only |

---

## Prerequisites & Setup

### 1. Install Python Dependencies

```bash
cd backend
pip3 install -r requirements.txt
```

**Key packages:**
- `fastapi` (0.104+) — Web framework
- `uvicorn` — ASGI server
- `opensearch-py` — OpenSearch client
- `sentence-transformers` — Embedding model
- `arq` — Async task queue worker/dispatcher
- `redis` — Queue backend used by ARQ
- `pydantic` — Request/response validation

### 2. Download the Embedding Model

The first time the backend runs, SentenceTransformers will download the `all-MiniLM-L6-v2` model (~90 MB) to `~/.cache/huggingface/hub/`. This is automatic when the initializer or search handler calls `SentenceTransformer('all-MiniLM-L6-v2')`.

**Manual download** (optional, to pre-cache):
```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 3. Ensure OpenSearch is Running

The backend expects OpenSearch to be available at `http://localhost:9200` (configurable via `OPENSEARCH_HOST` and `OPENSEARCH_PORT` environment variables).

**Start OpenSearch (Docker):**
```bash
docker-compose up -d opensearch opensearch-dashboards
```

Check health:
```bash
curl -s http://localhost:9200/ | jq .
```

### 4. Start the Backend Server

```bash
python3 main.py
```

The server will:
1. Initialize OpenSearch on startup (via lifespan)
2. Create/recreate the `auctus_catalog_master` index (if needed)
3. Listen on `http://localhost:8000`

**Alternative** (with auto-reload for development):
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Alternative import path check**:
```bash
python -c "from api.search import router; print(router)"
```

### 5. Start the ARQ Worker

The ingestion pipeline is split into a fast dispatcher and a background worker. Start the worker alongside the backend when you want dataset ingestion to run asynchronously.

```bash
arq storage.arq_worker.WorkerSettings
```

If you are using Docker Compose, the `arq-worker` service runs this command automatically.

---

## Data Ingestion Pipeline

### Step 1: Generate Synthetic Dataset (One-time)

```bash
# The seed dataset is maintained in `data/synthetic_datasets.json`
```

**Output:** `backend/data/synthetic_datasets.json` (10 test datasets with full profiler metadata)

### Step 2: Initialize OpenSearch Schema

```bash
python3 -m storage.initialize_os
```

**What this does:**
1. Connects to OpenSearch at `http://localhost:9200`
2. Deletes the `auctus_catalog_master` index (if it exists)
3. Creates a new `auctus_catalog_master` index with:
   - k-NN enabled (`index.knn: true`)
   - `dataset_vector` field (384-d knn_vector, HNSW/nmslib)
   - `embedding_metadata` object (model_name, version)
   - Standard text fields (title, description, analyzer)
   - Nested profiler metadata (columns with stats)
   - Geo-shape spatial coverage
   - Date-based temporal coverage
4. Leaves the index empty and ready for seeding

### Step 3: Seed Synthetic Data with Embeddings

```bash
python3 seed_synthetic.py
```

**What this does:**
1. Loads all datasets from `data/synthetic_datasets.json`
2. Generates embeddings by passing each `title + description` through `all-MiniLM-L6-v2`
3. Attaches `dataset_vector` and `embedding_metadata` to each document
4. Bulk-indexes enriched documents into `auctus_catalog_master`

This step is optional and intended for quick local trials.

**Performance note:** On Apple Silicon (MPS acceleration), embedding generation for 10 datasets typically completes in ~1–2 seconds.

### Step 4: Run the Dispatcher Pipeline

```bash
python3 run_pipeline_ingest.py
```

This will:
1. Read portal and pipeline settings from `config/config.json`
2. Discover the latest tabular Socrata datasets for the active portal
3. Compare Socrata timestamps against the existing OpenSearch document
4. Enqueue new or stale datasets to ARQ instead of processing them inline

The heavy work now runs in the ARQ worker:
1. Download and sample CSV data
2. Profile the dataset with `atlas-profiler`
3. Upload the full profile to MinIO
4. Generate the embedding vector
5. Index the trimmed search document into OpenSearch

**Tip:** If you want a synchronous end-to-end smoke test, use `python3 seed_synthetic.py`. For normal ingestion from live Socrata data, use `python3 run_pipeline_ingest.py` and keep the worker running.

### Step 5: Resetting the Environment (Optional)

When testing end-to-end pipelines, developers may need to reset services to start fresh. We provide two strategies depending on your needs.

#### Strategy 1: Fast Index Reset (Surgical)

**Use this if:** You want to clear OpenSearch search results but keep MinIO buckets and containers running. This is the fastest option for re-testing ingestion logic.

```bash
# Delete only the OpenSearch catalog index
curl -X DELETE "http://localhost:9200/auctus_catalog_master?pretty"

# Re-initialize the empty index schema
python3 -m storage.initialize_os
```

**Time to ready:** ~2 seconds (just index deletion + schema recreation)

**What's preserved:** MinIO bucket contents, container state, all other Docker services

**Next steps:** You can immediately re-run:
```bash
python3 seed_synthetic.py        # Quick-start seed
# or
python3 run_pipeline_ingest.py   # Full pipeline ingest
```

---

#### Strategy 2: Total Environment Purge (Nuclear)

**Use this if:** You are hitting MinIO WORM (Write-Once-Read-Many) or retention errors, or you want a completely blank slate for both OpenSearch and MinIO. This is the heaviest reset.

```bash
# Stop all containers and delete all volumes (complete data wipe)
docker-compose down -v

# Rebuild and restart containers from scratch
docker-compose up -d
docker stop auctus-backend
```

**Time to ready:** ~30 seconds (containers restart, then re-initialization)

**What's wiped:** All OpenSearch indices, all MinIO buckets, all persisted data

**What stays:** Your source code, config files, and Docker images

**Required next step:** You **must** re-initialize the index schema before running any pipeline:

```bash
python3 -m storage.initialize_os
```

Then proceed with:
```bash
python3 seed_synthetic.py        # Quick-start seed
# or
python3 run_pipeline_ingest.py   # Full pipeline ingest
```

**⚠️ Warning:** This command is **only safe for local Docker-based development**. Never run this on production or cloud-hosted MinIO instances.

---

## API Contract

### Endpoint: `POST /api/v1/search`

#### Request Payload

```json
{
  "keywords": "taxi yellow cabs",
  "source": ["Socrata", "NYC Open Data"],
  "types": ["spatial", "temporal"],
  "temporal_start": "2023-01-01",
  "temporal_end": "2023-12-31",
  "bbox": [-74.26, 40.47, -73.7, 40.92],
  "limit": 10,
  "offset": 0
}
```

**Field Descriptions:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `keywords` | string | No | Free-text query; triggers hybrid (lexical + k-NN) search. If empty/omitted, falls back to filter-only search. |
| `source` | [string] | No | Filter to datasets from specific sources (e.g., Socrata, Zenodo). |
| `types` | [string] | No | Filter to datasets with specific data types (spatial, numerical, temporal, categorical). |
| `temporal_start` | string (YYYY-MM-DD) | No | Filter to datasets with coverage ≥ this date. |
| `temporal_end` | string (YYYY-MM-DD) | No | Filter to datasets with coverage ≤ this date. |
| `bbox` | [float, float, float, float] | No | Spatial bounding box `[min_lon, min_lat, max_lon, max_lat]` for geo-intersect filtering. |
| `limit` | int (default: 10) | No | Number of results to return. |
| `offset` | int (default: 0) | No | Result offset for pagination. |

#### Response Payload

```json
{
  "total": 5,
  "results": [
    {
      "id": "ds-001",
      "title": "NYC Yellow Taxi Trips 2023",
      "description": "Daily trip records from yellow taxis...",
      "source": "Socrata",
      "types": ["spatial", "temporal", "numerical"],
      "temporal_coverage": {
        "start": "2023-01-01",
        "end": "2023-12-31"
      },
      "spatial_coverage": {
        "label": "New York City",
        "bbox": {
          "type": "envelope",
          "coordinates": [[-74.26, 40.92], [-73.7, 40.47]]
        }
      },
      "embedding_metadata": {
        "model_name": "all-MiniLM-L6-v2",
        "version": 1
      },
      "dataset_vector": [0.123, -0.456, ..., 0.789],
      "profiler_metadata": {
        "nb_rows": 1000,
        "nb_columns": 8,
        "columns": [ ... ]
      }
    },
    ...
  ],
  "aggregations": {
    "sources_count": {
      "buckets": [
        { "key": "Socrata", "doc_count": 3 },
        { "key": "NYC Open Data", "doc_count": 2 }
      ]
    },
    "types_count": {
      "buckets": [
        { "key": "spatial", "doc_count": 5 },
        { "key": "temporal", "doc_count": 4 }
      ]
    }
  }
}
```

#### Example cURL Requests

**Hybrid Search (Keywords + Filters):**
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": "taxi city",
    "source": ["Socrata"],
    "limit": 5
  }' | jq .
```

**Filter-Only Browse (No Keywords):**
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "source": ["NYC Open Data"],
    "types": ["spatial"],
    "limit": 10
  }' | jq .
```

**Spatial Query (Bounding Box):**
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": "census data",
    "bbox": [-74.3, 40.4, -73.7, 40.95],
    "limit": 20
  }' | jq .
```

---

## Query Logic

### When Keywords Are Provided

The backend constructs a **hybrid boolean query** in OpenSearch:

```python
{
  "query": {
    "bool": {
      "filter": [
        # Hard filters (if present)
        { "terms": { "source": ["Socrata", "..."] } },
        { "terms": { "types": ["spatial"] } },
        { "range": { "temporal_coverage.end": { "gte": "2023-01-01" } } },
        { "geo_shape": { "spatial_coverage.bbox": { ... } } }
      ],
      "should": [
        # Semantic clauses (when keywords exist)
        {
          "multi_match": {
            "query": "user keywords",
            "fields": ["title^2", "description"]
          }
        },
        {
          "knn": {
            "dataset_vector": {
              "vector": [0.123, -0.456, ..., 0.789],
              "k": 10
            }
          }
        }
      ]
    }
  },
  "size": 10,
  "from": 0,
  "aggs": { ... }
}
```

**Behavior:**
- All `filter` clauses must match (AND logic)
- At least one `should` clause must match (OR logic, with scoring)
- Results are ranked by relevance (title^2 boost + vector similarity)

### When No Keywords Are Provided

The `should` block is omitted entirely, resulting in a **clean filter-only query**:

```python
{
  "query": {
    "bool": {
      "filter": [/* active filters only */],
      # no "should" block → defaults to match_all
    }
  }
}
```

This allows safe **categorical browsing** without requiring keywords.

---

## Environment Variables

All configurable via environment:

```bash
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
export OPENSEARCH_USER=admin          # Optional
export OPENSEARCH_PASS=password       # Optional
```

---

## Debugging

### Enable Payload Logging

The search endpoint prints the raw OpenSearch payload before execution:

```bash
# Terminal output
RAW OPENSEARCH PAYLOAD: {
  "query": {
    "bool": { ... }
  },
  ...
}
```

This helps verify query structure and filter application.

### Check Index Mapping

```bash
curl -s http://localhost:9200/auctus_catalog_master/_mapping?pretty | jq .
```

### View Index Stats

```bash
curl -s http://localhost:9200/auctus_catalog_master/_stats | jq '.indices.auctus_catalog_master.primaries.docs'
```

---

## References

- **FastAPI Documentation:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **OpenSearch k-NN:** [opensearch.org/docs/latest/search-plugins/knn/](https://opensearch.org/docs/latest/search-plugins/knn/)
- **SentenceTransformers:** [www.sbert.net](https://www.sbert.net/)
- **Atlas Profiler:** [github.com/uchicago-dsi-cdac/dfencoder](https://github.com/uchicago-dsi-cdac/dfencoder)
- **MinIO:** [min.io/docs](https://min.io/docs/)

---

## License

Part of the Auctus dataset discovery platform. See root repository for license details.
