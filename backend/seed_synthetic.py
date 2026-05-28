#!/usr/bin/env python3
"""Seed synthetic datasets into OpenSearch with generated embedding vectors."""

from __future__ import annotations

import json
import os
import sys
import time

try:
    from opensearchpy import helpers
except Exception:
    print("Required package 'opensearch-py' is not installed.")
    print("Install with: pip install opensearch-py")
    raise

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from storage.initialize_os import AUCTUS_INDEX_NAME, get_client

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_datasets.json")


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def bulk_index(client, docs):
    print(f"Preparing {len(docs)} documents for bulk indexing...")
    actions = []
    for doc in docs:
        action = {
            "_op_type": "index",
            "_index": AUCTUS_INDEX_NAME,
            "_id": doc.get("id"),
            "_source": doc,
        }
        actions.append(action)

    start = time.time()
    success = 0
    try:
        success, details = helpers.bulk(client, actions, raise_on_error=False)
        if isinstance(details, list) and len(details) > 0:
            print("\n❌ OPENSEARCH INDEXING ERRORS DETECTED:")
            for item in details:
                for op, info in item.items():
                    if info.get("status", 200) not in [200, 201]:
                        print(f"Document ID {info.get('_id')} failed! Error: {info.get('error')}\n")
    except Exception as exc:
        print("Bulk indexing raised an exception:", exc)
        raise

    duration = time.time() - start
    print(f"Bulk indexing completed: {success} docs indexed in {duration:.2f}s")
    return success


def main():
    print("Connecting to OpenSearch...")
    client = get_client()

    try:
        info = client.info()
        print("Connected to OpenSearch:", info.get("version", {}).get("number", "unknown"))
    except Exception as exc:
        print("Failed to connect to OpenSearch at http://localhost:9200")
        print(exc)
        sys.exit(1)

    print(f"Loading synthetic data from {DATA_PATH}...")
    docs = load_data(DATA_PATH)
    print(f"Loaded {len(docs)} documents; sample id: {docs[0].get('id') if docs else 'n/a'}")

    if SentenceTransformer is None:
        print("Missing required package 'sentence-transformers'.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = []
    for doc in docs:
        title = doc.get("title") or ""
        descr = doc.get("description") or ""
        texts.append(f"{title}\n\n{descr}")

    print(f"Generating embeddings for {len(texts)} documents...")
    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
    except Exception as exc:
        print("Error generating embeddings:", exc)
        raise

    for i, doc in enumerate(docs):
        vec = embeddings[i]
        try:
            vec_list = vec.tolist()
        except Exception:
            vec_list = [float(x) for x in vec]

        doc["dataset_vector"] = vec_list
        doc["embedding_metadata"] = {
            "model_name": "all-MiniLM-L6-v2",
            "version": 1,
        }

    success_count = bulk_index(client, docs)
    client.indices.refresh(index=AUCTUS_INDEX_NAME)
    print(f"✅ Successfully seeded {success_count} synthetic datasets into '{AUCTUS_INDEX_NAME}'.")


if __name__ == "__main__":
    main()
