# Auctus v2 Documentation Portal

Welcome to the internal engineering documentation for **Auctus v2**—our specialized dataset discovery engine and open-data harvesting pipeline. 

This platform is hosted and managed under our organization credentials to streamline catalog syncing, vector-accelerated dataset searches, and automated metadata profiling.

---

## 🧭 Document Navigation

Use the top navigation bar or the sidebar menus to explore the system layout:

* **[Architecture Overview](architecture.md):** A deep dive into our multi-tier infrastructure, data persistence layers (OpenSearch & MinIO), and background task worker lifecycles.
* **[Installation & Local Setup](installation.md):** Step-by-step developer environment assembly, database initializations (`storage.initialize_os`), testing seed rules, and live data ingestion runs.
* **[Search API Reference](api/search.md):** Automated code and docstring specifications for our vector and keyword search routing engine.
* **[Portals API Reference](api/portals.md):** Documentation covering dynamic multi-provider registry tracking and analytics aggregation.

---

## 🛠️ Quick Developer Stack Reference

| Service Component | Technology | Default Local Port | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend Web UI** | React / TailwindCSS | `5173` | Client dashboard & map interfaces |
| **API Gateway** | FastAPI (Python) | `8000` | Gateway & dynamic payload streaming |
| **Search Engine** | OpenSearch | `9200` | Core catalog index (Keyword & k-NN vector scoring) |
| **Search Dashboard** | OpenSearch Dashboards | `5601` | Cluster administration & index visualization |
| **Task Coordinator**| Redis / ARQ | `6379` | Asynchronous harvesting pipeline orchestration |
| **Object Data Lake** | MinIO Storage | `9000` / `9001` | High-volume structured JSON profile matrix storage |