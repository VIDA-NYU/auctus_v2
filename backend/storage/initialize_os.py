#!/usr/bin/env python3
"""Initialize a clean OpenSearch index and apply schema mappings.

Usage: run from repository root or the `backend/` directory:

    python -m storage.initialize_os

The script connects to http://localhost:9200 by default. It deletes the
`auctus_catalog_master` index if present and recreates it with the supplied
mapping. It also attempts to create an index pattern in OpenSearch Dashboards
at http://localhost:5601 for immediate UI access.
"""
import os
import time
import sys

try:
    from opensearchpy import OpenSearch
except Exception as exc:  # pragma: no cover - runtime dependency
    print("Required package 'opensearch-py' is not installed.")
    print("Install with: pip install opensearch-py")
    raise

try:
    import httpx
except Exception as exc:  # pragma: no cover - runtime dependency
    print("Required package 'httpx' is not installed.")
    print("Install with: pip install httpx")
    raise


AUCTUS_INDEX_NAME = "auctus_catalog_master"


def get_client():
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    user = os.getenv("OPENSEARCH_USER") or os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASS") or os.getenv("OPENSEARCH_PASSWORD")

    hosts = [{"host": host, "port": port}]

    kwargs = {
        "hosts": hosts,
        "use_ssl": False,
        "verify_certs": False,
        "ssl_show_warn": False,
    }

    # If credentials present, use them; otherwise rely on no-auth local instance.
    if user and password:
        kwargs["http_auth"] = (user, password)

    client = OpenSearch(**kwargs)
    return client


MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "analysis": {
            "analyzer": {
                "text_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "embedding_metadata": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "keyword"},
                    "version": {"type": "integer"},
                },
            },
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "description": {
                "type": "text",
                "analyzer": "text_analyzer",
            },
            "source": {"type": "keyword"},
            "download_url": {"type": "keyword", "index": False},
            "socrata_updated_at": {
                "type": "date",
                "format": "strict_date_optional_time||yyyy-MM-dd",
            },
            "source_updated_at": {
                "type": "date",
                "format": "strict_date_optional_time||yyyy-MM-dd",
            },
            "last_update_date": {
                "type": "date",
                "format": "yyyy-MM-dd",
            },
            "types": {"type": "keyword"},
            "temporal_coverage": {
                "type": "object",
                "properties": {
                    "start": {"type": "date", "format": "yyyy-MM-dd"},
                    "end": {"type": "date", "format": "yyyy-MM-dd"},
                },
            },
            "spatial_coverage": {
                "type": "object",
                "properties": {
                    "label": {"type": "text"},
                    "bbox": {"type": "geo_shape", "strategy": "recursive"},
                },
            },
            "dataset_vector": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "engine": "nmslib",
                },
            },
            "profiler_metadata": {
                "type": "object",
                "properties": {
                    "nb_rows": {"type": "long"},
                    "nb_profiled_rows": {"type": "long"},
                    "nb_columns": {"type": "long"},
                    "attribute_keywords": {"type": "text"},
                    "columns": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                            "structural_type": {"type": "keyword"},
                            "semantic_types": {"type": "keyword"},
                            "mean": {"type": "float"},
                            "stddev": {"type": "float"},
                            "plot": {"type": "object", "enabled": False},
                        },
                    },
                },
            },
        }
    },
}

def recreate_index(client):
    if client.indices.exists(index=AUCTUS_INDEX_NAME):
        print(f"Index '{AUCTUS_INDEX_NAME}' exists — deleting...")
        client.indices.delete(index=AUCTUS_INDEX_NAME)
        # small pause to let the cluster register deletion
        time.sleep(0.5)

    print(f"Creating index '{AUCTUS_INDEX_NAME}' with mappings...")
    client.indices.create(index=AUCTUS_INDEX_NAME, body=MAPPING)


def create_dashboard_index_pattern():
    """Automatically create the index pattern in OpenSearch Dashboards.
    
    This allows developers to immediately browse the `auctus_catalog_master`
    index without manual configuration via the Dashboards UI.
    
    Fails gracefully if Dashboards is not available or still bootstrapping.
    """
    dashboards_url = "http://localhost:5601"
    pattern_url = f"{dashboards_url}/api/saved_objects/index-pattern/{AUCTUS_INDEX_NAME}"
    
    payload = {
        "attributes": {
            "title": AUCTUS_INDEX_NAME
        }
    }
    
    headers = {
        "osd-xsrf": "true",
        "Content-Type": "application/json"
    }
    
    try:
        response = httpx.post(pattern_url, json=payload, headers=headers, timeout=5.0)
        if response.status_code in [200, 201]:
            print(f"✨ OpenSearch Dashboards index pattern '{AUCTUS_INDEX_NAME}' automatically created.")
        elif response.status_code == 409:
            # Pattern may already exist; this is not an error
            print(f"ℹ️  OpenSearch Dashboards index pattern '{AUCTUS_INDEX_NAME}' already exists.")
        else:
            print(f"⚠️  Failed to create Dashboards index pattern (HTTP {response.status_code}). Continuing without it.")
    except Exception as exc:
        # Dashboards may not be running; fail gracefully
        print(f"⚠️  OpenSearch Dashboards is not available ({exc}). Skipping index pattern creation.")
        print("    You can manually create the index pattern via the Dashboards UI at http://localhost:5601")



def main():
    print("Connecting to OpenSearch...")
    client = get_client()

    # basic cluster info
    try:
        info = client.info()
        print("Connected to OpenSearch:", info.get("version", {}).get("number", "unknown"))
    except Exception as exc:
        print("Failed to connect to OpenSearch at http://localhost:9200")
        print(exc)
        sys.exit(1)

    recreate_index(client)
    print("✅ OpenSearch index 'auctus_catalog_master' initialized cleanly with schema mappings.")
    
    # Attempt to create dashboard index pattern automatically
    create_dashboard_index_pattern()


if __name__ == "__main__":
    main()
