# Auctus v2 — Local Development

Project Overview
----------------

Auctus v2 is a full-stack dataset discovery and search tool powered by OpenSearch 2.13. The system replaces static JSON filtering with a real search engine that supports full-text relevance, geospatial (GeoJSON / `geo_shape`) and temporal queries. The backend (FastAPI) integrates with OpenSearch to provide production-like search behavior while the frontend (React + Vite) provides a modern UI for exploration.

The 'Hybrid' Development Workflow (recommended)
---------------------------------------------

For an efficient developer experience we recommend a hybrid workflow: run the OpenSearch infrastructure in Docker, run the backend locally for fast iteration and debugging, and run the frontend with Vite. Follow these steps in order.

Step 1 — The Infrastructure (body)

Start OpenSearch and OpenSearch Dashboards with Docker Compose:

```bash
cd auctus_v2
docker compose up -d
```

This brings up:
- OpenSearch Engine on http://localhost:9200
- OpenSearch Dashboards (the control room) on http://localhost:5601

Step 2 — Port Management

The compose file includes a placeholder backend container that may bind port 8000. To run the backend locally free the port by stopping the placeholder container:

```bash
docker stop auctus-backend
```

Step 3 — The Logic (nervous system)

Run the FastAPI backend locally for better debugging and instant reloads. From the `backend/` directory:

```bash
cd backend
# optional: create and activate a virtualenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the backend (lifespan hooks will initialize OpenSearch if available)
python main.py
```

Note: `python main.py` runs the FastAPI app with the built-in startup lifecycle that configures OpenSearch automatically. You may instead run `uvicorn main:app --reload` if you prefer `uvicorn` directly.

Step 4 — The UI

Run the React frontend with Vite:

```bash
cd frontend
npm install
npm run dev
```

Open the app at http://localhost:5173 and perform searches — the frontend calls the backend at http://localhost:8000 by default.

Automated Initialization (Infrastructure as Code)
------------------------------------------------

To minimize setup friction, the backend performs automated initialization when it starts:

- Configures OpenSearch mappings on `datasets_v2` (text analyzers, `geo_shape` for spatial coverage, and date types for temporal coverage).
- Seeds the index with synthetic datasets from `backend/data/synthetic_datasets.json` if the index is empty.
- Transforms ingestion bounding boxes into GeoJSON `envelope` shapes so spatial queries work out-of-the-box.
- The development `docker-compose.yml` disables OpenSearch security to provide a zero-config developer experience (do not use this in production).

These behaviors are implemented in `backend/opensearch_config.py` and invoked by a FastAPI lifespan context manager in `backend/main.py`.

Essential URLs
--------------

| Component | URL |
|---|---|
| Frontend (Vite) | http://localhost:5173 |
| Backend API (FastAPI) | http://localhost:8000 |
| OpenSearch Engine | http://localhost:9200 |
| OpenSearch Dashboards (Control Room) | http://localhost:5601 |

Quick Commands
--------------

```bash
# start OpenSearch & Dashboards
docker compose up -d

# stop placeholder backend to free port 8000
docker stop auctus-backend

# run backend locally (from backend/)
python main.py

# run frontend (from frontend/)
npm run dev

# view backend health
curl http://localhost:8000/

# simple POST search test
curl -sS http://localhost:8000/search -X POST -H 'Content-Type: application/json' -d '{"query":"test","filters":null}' | jq
```

Troubleshooting
---------------

- ModuleNotFoundError: `opensearch-py`

  If you see `ModuleNotFoundError: No module named 'opensearch'` or similar when running the backend, install the Python OpenSearch client into your virtualenv:

  ```bash
  cd backend
  source .venv/bin/activate   # if using virtualenv
  pip install -r requirements.txt
  # or at minimum
  pip install opensearch-py
  ```

- Verifying data in Dashboards (Discover)

  1. Open OpenSearch Dashboards at http://localhost:5601
  2. Go to the "Discover" app (the left nav) and create/select an index pattern for `datasets_v2`.
  3. Refresh the index pattern fields and search for sample documents. If the index is empty, restart the backend (it will seed synthetic data on first run) or check the backend logs for errors during initialization.

- If `POST /search` returns 404 in the browser console

  Ensure the backend is running at `http://localhost:8000` and that your browser is allowed to reach that host/port. Use `curl` from the host to confirm.

- If OpenSearch is unreachable

  Confirm Docker containers are running:

  ```bash
  docker compose ps
  docker compose logs opensearch
  curl http://localhost:9200/
  ```

Security and Production Notes
-----------------------------

This repository is configured for local developer convenience. The Docker Compose setup disables OpenSearch security plugins and is not safe for production use. For production deployments you should enable security, set proper resource limits, and secure network access.

Where to look in the code
-------------------------

- `backend/opensearch_config.py` — index mapping, ingestion, and transform logic
- `backend/main.py` — FastAPI app and lifespan initialization that triggers OpenSearch setup
- `frontend/` — React + Vite UI and the `Results` page that consumes the backend search API

If you'd like, I can also add a short section showing how to enable production-ready OpenSearch settings and index templates.

