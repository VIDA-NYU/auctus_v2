#!/usr/bin/env python3
"""Initialize a clean OpenSearch index `datasets` and bulk-load v2 synthetic data.

Usage: run from repository root or the `backend/` directory:

    python initialize_opensearch.py

The script connects to http://localhost:9200 by default. It will delete
the `datasets` index if present, create it with the supplied mapping, and
bulk-index records from `backend/data/synthetic_datasets_v2.json`.
"""
import json
import os
import time
import sys
from pprint import pformat

try:
    from opensearchpy import OpenSearch, helpers
except Exception as exc:  # pragma: no cover - runtime dependency
    print("Required package 'opensearch-py' is not installed.")
    print("Install with: pip install opensearch-py")
    raise

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


INDEX_NAME = "datasets_v2"
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_datasets_v2.json")


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


def load_data(path):
    with open(path, "r") as fh:
        return json.load(fh)


def recreate_index(client):
    if client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' exists — deleting...")
        client.indices.delete(index=INDEX_NAME)
        # small pause to let the cluster register deletion
        time.sleep(0.5)

    print(f"Creating index '{INDEX_NAME}' with mappings...")
    client.indices.create(index=INDEX_NAME, body=MAPPING)

def bulk_index(client, docs):
    print(f"Preparing {len(docs)} documents for bulk indexing...")
    actions = []
    for doc in docs:
        action = {
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": doc.get("id"),
            "_source": doc,
        }
        actions.append(action)

    start = time.time()
    try:
        # Capture both the success count and the detailed response/failures list
        success, details = helpers.bulk(client, actions, raise_on_error=False)
        
        # If there are failures, print them out clearly
        if isinstance(details, list) and len(details) > 0:
            print("\n❌ OPENSEARCH INDEXING ERRORS DETECTD:")
            for item in details:
                # Look for items that don't have a 201 (Created) or 200 (OK) status
                for op, info in item.items():
                    if info.get('status', 200) not in [200, 201]:
                        print(f"Document ID {info.get('_id')} failed! Error: {info.get('error')}\n")
                        
    except Exception as exc:
        print("Bulk indexing raised an exception:", exc)
        raise
        
    duration = time.time() - start
    print(f"Bulk indexing completed: {success} docs indexed in {duration:.2f}s")
# def bulk_index(client, docs):
#     print(f"Preparing {len(docs)} documents for bulk indexing...")
#     actions = []
#     for doc in docs:
#         action = {
#             "_op_type": "index",
#             "_index": INDEX_NAME,
#             "_id": doc.get("id"),
#             "_source": doc,
#         }
#         actions.append(action)

#     start = time.time()
#     success, failures = 0, []
#     try:
#         resp = helpers.bulk(client, actions, raise_on_error=False)
#         # helpers.bulk returns (success_count, details) normally when raise_on_error=False
#         if isinstance(resp, tuple):
#             success = resp[0]
#         else:
#             success = resp
#     except Exception as exc:
#         print("Bulk indexing raised an exception:", exc)
#         raise
#     duration = time.time() - start
#     print(f"Bulk indexing completed: {success} docs indexed in {duration:.2f}s")


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

    print(f"Loading synthetic data from {DATA_PATH}...")
    docs = load_data(DATA_PATH)
    print(f"Loaded {len(docs)} documents; sample id: {docs[0].get('id') if docs else 'n/a'}")

    # Generate embeddings for each document using sentence-transformers
    if SentenceTransformer is None:
        print("Missing required package 'sentence-transformers'.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare texts by concatenating title + description
    texts = []
    for doc in docs:
        title = doc.get('title') or ''
        descr = doc.get('description') or ''
        texts.append(f"{title}\n\n{descr}")

    print(f"Generating embeddings for {len(texts)} documents...")
    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
    except Exception as e:
        print("Error generating embeddings:", e)
        raise

    # Attach embedding vector and metadata to each document
    for i, doc in enumerate(docs):
        vec = embeddings[i]
        # Convert numpy array to plain list of floats
        try:
            vec_list = vec.tolist()
        except Exception:
            # Fallback: iterate values
            vec_list = [float(x) for x in vec]

        doc['dataset_vector'] = vec_list
        doc['embedding_metadata'] = {
            'model_name': 'all-MiniLM-L6-v2',
            'version': 1,
        }

    bulk_index(client, docs)

    print("Indexing finished. Refreshing index...")
    client.indices.refresh(index=INDEX_NAME)
    print("Done.")


if __name__ == "__main__":
    main()
