"""Recompute dataset_vector for already-indexed documents, without re-ingesting.

Run after changing what the embedding is built from (storage/arq_worker.py::embedding_text):
    docker exec -w /app auctus-arq-worker python -m storage.reembed [--domain DOMAIN]
"""

from __future__ import annotations

import argparse

from opensearchpy import helpers

from storage.arq_worker import embedding_text, get_embedding_model
from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

FIELDS = ["title", "description", "source_record.subjects", "profiler_metadata.columns.name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="only documents from this domain")
    args = parser.parse_args()

    client, model = get_client(), get_embedding_model()
    query = {"term": {"domain": args.domain}} if args.domain else {"match_all": {}}
    docs = list(helpers.scan(client, index=AUCTUS_INDEX_NAME, query={"query": query}, _source=FIELDS))
    vectors = model.encode([embedding_text(d["_source"]) for d in docs], convert_to_numpy=True, batch_size=64)
    actions = ({"_op_type": "update", "_index": AUCTUS_INDEX_NAME, "_id": d["_id"], "doc": {"dataset_vector": v.tolist()}}
               for d, v in zip(docs, vectors))
    ok, errors = helpers.bulk(client, actions, raise_on_error=False, refresh=True)
    print(f"re-embedded {ok} documents, {len(errors)} errors")


if __name__ == "__main__":
    main()
