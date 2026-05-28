# Auctus v2 — Local Development (Overview)

Auctus v2 is a full-stack dataset discovery and search tool using OpenSearch for relevance, geospatial and temporal queries, a FastAPI backend, and a React + Vite frontend.

This top-level README is intentionally high-level. Detailed backend setup and developer commands are maintained in the backend README: [backend/README.md](backend/README.md).

Quick local workflow (summary)
-----------------------------

- Start the core infra with Docker Compose (OpenSearch + Dashboards):

```bash
docker compose up -d
```

- Stop the placeholder backend container if it conflicts with port 8000:

```bash
docker stop auctus-backend
```

- Run the backend locally for development (from the `backend/` directory):

```bash
cd backend
# (optional) create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# install Python dependencies
pip3 install -r requirements.txt

# Initialize services and schema (requires OpenSearch reachable at http://localhost:9200)
# Start OpenSearch/Dashboards with Docker Compose if not already running:
#   docker compose up -d
# Create index mappings for `auctus_catalog_master`:
python3 -m storage.initialize_os

# Optional: quick-start seed of synthetic data (seeds backend/data/synthetic_datasets.json)
# Use this for demo/testing. For real ingestion from source datasets run the pipeline instead.
python3 seed_synthetic.py

# To run the backend locally after initialization:
python3 main.py
```

The backend performs schema initialization on startup when OpenSearch is reachable. For more control you can run the schema initializer or the optional seeder manually (see the backend README).

- Run the frontend (from `frontend/`):

```bash
cd frontend
npm install
npm run dev
```

Useful service URLs
-------------------

- Frontend (Vite): http://localhost:5173
- Backend API (FastAPI): http://localhost:8000
- OpenSearch Engine: http://localhost:9200
- OpenSearch Dashboards: http://localhost:5601

If you need additional troubleshooting, MinIO setup details, or schema internals, see the backend README at [backend/README.md](backend/README.md). For real-data ingestion run the pipeline from the `backend/` directory:

```bash
# run ingestion pipeline against discovered sources (optional LIMIT arg):
python3 run_pipeline_ingest.py [LIMIT]
```

If you'd like, I can also add a short `docker-compose.dev.yml` that includes a ready-to-run OpenSearch + MinIO configuration for local testing.


