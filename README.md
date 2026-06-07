# Auctus v2 — Local Development (Overview)

Auctus v2 is a full-stack dataset discovery and search tool using OpenSearch for relevance, geospatial and temporal queries, a FastAPI backend, and a React + Vite frontend. During ingestion it also calls **AutoDDG** to automatically generate a plain-language description for every dataset (via the NYU Portkey gateway) and stores it alongside the profile.

This top-level README is intentionally high-level. Detailed backend setup and developer commands are maintained in the backend README: [backend/README.md](backend/README.md).

Prerequisites (AutoDDG)
-----------------------

AutoDDG runs inside the `arq-worker` container, which mounts the AutoDDG source from `../AutoDDG`. Two one-time steps:

- Clone AutoDDG as a **sibling directory** of `auctus_v2` (so the path `../AutoDDG` resolves), on the `portkey-integration` branch:

```bash
# from the directory that contains auctus_v2/
git clone https://github.com/VIDA-NYU/AutoDDG.git
cd AutoDDG && git checkout portkey-integration && cd -
```

- Provide a Portkey API key. Copy the template and fill in your key (`backend/.env` is git-ignored and must never be committed):

```bash
cp backend/.env_sample backend/.env
# then edit backend/.env and set PORTKEY_API_KEY=<your NYU Portkey key>
```

Without these, ingestion still runs — the description step is simply skipped.

Quick local workflow (summary)
-----------------------------

- Start the core infra with Docker Compose (OpenSearch + Dashboards + MinIO + Redis + the AutoDDG worker) in `/auctus_v2`:

```bash
open -a Docker
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

# install Python dependencies (one-time per venv; rerunning just skips what's already installed)
pip3 install -r requirements.txt

# Initialize services and schema (requires OpenSearch reachable at http://localhost:9200)
# Start OpenSearch/Dashboards with Docker Compose if not already running:
#   docker compose up -d
# Create the `auctus_catalog_master` index (the OpenSearch "table" that holds every
# dataset's search record) with the correct mappings. Run this once before ingesting —
# i.e. on first setup or after `docker compose down -v`.
# WARNING: this DROPS and recreates the index, so re-running it wipes all ingested data.
# (pip install above is safe to rerun; this one is not.)
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
# Run the ingestion pipeline. The optional argument is LIMIT = max datasets per domain;
# use `1` for a quick single-dataset smoke test (each dataset triggers one LLM call):
python3 run_pipeline_ingest.py 1
```

During ingestion the `arq-worker` profiles each dataset, calls AutoDDG to generate a description, and stores it in both MinIO (full profile) and OpenSearch (`autoddg_description` field). To confirm it worked, check a stored document:

```bash
curl -s "localhost:9200/auctus_catalog_master/_doc/<dataset_id>" | python3 -c \
  'import sys,json;print(json.load(sys.stdin)["_source"].get("autoddg_description"))'
```
