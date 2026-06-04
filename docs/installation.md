# Local Development Setup

This guide walks through the sequence required to start the local infrastructure, initialize OpenSearch indexes, and run both the FastAPI backend and React frontend.

---

## 📋 System Prerequisites

Before starting, ensure your local development machine has the following dependencies installed:

* **Python 3.12+** (Backend runtime)
* **Node.js 18+** & **npm** (Frontend runtime)
* **Docker & Docker Compose** (For orchestrating OpenSearch, MinIO, and Redis)
* **Git**


---

## 🐳 Step 1: Start Core Infrastructure

Auctus v2 depends on OpenSearch, MinIO, and Redis.

From the repository root:

```bash
docker compose up -d
```

### Port Conflict Check

Stop the placeholder backend container if it conflicts with port`8000`:

```bash
docker stop auctus-backend
```

---

## 🐍 Step 2: Configure and Start the Backend

Navigate to the backend directory:

```bash
cd backend
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

### Initialize OpenSearch Indexes

With OpenSearch running on `http://localhost:9200`, create the required index mappings:

```bash
python3 -m storage.initialize_os
```

### Seed Synthetic Data (Optional)

Populate the catalog with sample data for local testing:

```bash
python3 seed_synthetic.py
```

This loads datasets defined in:

```text
backend/data/synthetic_datasets.json
```

### Start the Backend

```bash
python3 main.py
```

The application performs index and schema health checks during startup.

---

## ⚛️ Step 3: Configure and Start the Frontend

Open a second terminal window and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

---

## 🔄 Step 4: Run the Ingestion Pipeline (Optional)

To ingest real data instead of synthetic datasets, run the ingestion driver from the `backend/` directory:

```bash
python3 run_pipeline_ingest.py [LIMIT]
```

Where `LIMIT` is an optional argument to constrain the number of sources processed.

---

## 🧭 Service Endpoints

Once all services are running, the following endpoints are available:

| Service | URL | Purpose |
|----------|-----|----------|
| Frontend UI (Vite) | [http://localhost:5173](http://localhost:5173) | Interactive web application |
| Backend API (FastAPI) | [http://localhost:8000](http://localhost:8000) | REST API and OpenAPI documentation |
| OpenSearch Cluster | [http://localhost:9200](http://localhost:9200) | Search and indexing backend |
| OpenSearch Dashboards | [http://localhost:5601](http://localhost:5601) | Cluster administration and visualization |
| MinIO Console | [http://localhost:9001](http://localhost:9001) | Object storage administration |
| MinIO API | [http://localhost:9000](http://localhost:9000) | S3-compatible object storage endpoint |
---

## 🚀 Quick Start Summary

```bash
# Terminal 1 (repo root)
docker compose up -d
docker stop auctus-backend

# Terminal 2
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 -m storage.initialize_os
python3 seed_synthetic.py
python3 main.py

# Terminal 3
cd frontend
npm install
npm run dev
```
