"""OpenSearch initialization and management for Auctus v2 backend."""

import json
import logging
import os
from pathlib import Path

try:
    from opensearchpy import OpenSearch, NotFoundError
except ImportError:
    from opensearchpy import OpenSearch
    NotFoundError = Exception

logger = logging.getLogger(__name__)

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
AUCTUS_INDEX_NAME = "auctus_catalog_master"
AUCTUS_PORTALS_INDEX_NAME = "auctus_portals_metadata"
# Define the mapping for auctus_catalog_master index
DATASETS_MAPPING = {
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
            "domain": {"type": "keyword"},
            "provider": {"type": "keyword"},
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
                    "start": {"type": "date"},
                    "end": {"type": "date"},
                },
            },
            "spatial_coverage": {
                "type": "geo_shape",
                "strategy": "recursive",
            },
            "profiler_metadata": {
                "type": "object",
                "properties": {
                    "nb_rows": {"type": "long"},
                    "nb_profiled_rows": {"type": "long"},
                    "nb_columns": {"type": "long"},
                    "nb_spatial_columns": {"type": "integer"},
                    "nb_temporal_columns": {"type": "integer"},
                    "nb_numerical_columns": {"type": "integer"},
                    "nb_categorical_columns": {"type": "integer"},
                    "attribute_keywords": {"type": "text", "analyzer": "standard"},
                    "columns": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "keyword"},
                            "structural_type": {"type": "keyword"},
                            "semantic_types": {"type": "keyword"},
                        },
                    },
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
            "relevance_score": {"type": "float"},
        }
    },
}

PORTALS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "domain": {"type": "keyword"},
            "provider": {"type": "keyword"},
            "display_label": {"type": "keyword"},
            "dataset_count": {"type": "integer"},
            "last_indexed_at": {
                "type": "date",
                "format": "strict_date_optional_time||yyyy-MM-dd'T'HH:mm:ssZ",
            },
        }
    },
}


def get_client():
    """Create and return an OpenSearch client."""
    try:
        return OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=None,  # Security disabled for development
            use_ssl=False,
            verify_certs=False,
        )
    except Exception as e:
        logger.error(f"Failed to connect to OpenSearch at {OPENSEARCH_HOST}:{OPENSEARCH_PORT}: {e}")
        raise


def index_exists(client, index_name):
    """Check if an index exists."""
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        logger.error(f"Error checking index existence: {e}")
        return False


def create_index(client, index_name, mapping):
    """Create an index with the given mapping."""
    try:
        client.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index: {index_name}")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        # Check if index already exists (various error message formats)
        if (
            "already exists" in error_msg
            or "already_exists" in error_msg
            or "resource_already_exists" in error_msg
        ):
            logger.info(f"Index {index_name} already exists (no action taken)")
            return True
        logger.error(f"Error creating index: {e}")
        return False


def index_count(client, index_name):
    """Get the document count in an index."""
    try:
        result = client.count(index=index_name)
        return result["count"]
    except Exception as e:
        logger.error(f"Error counting documents: {e}")
        return 0


def transform_spatial_coverage(spatial_coverage):
    """
    Transform spatial_coverage from bbox format to geo_shape envelope format.

    Input: {"bbox": [lat_min, lon_min, lat_max, lon_max], "label": "..."}
    Output: {"type": "envelope", "coordinates": [[lon_min, lat_min], [lon_max, lat_max]]}
    """
    if not spatial_coverage:
        return None

    bbox = spatial_coverage.get("bbox")
    label = spatial_coverage.get("label")

    if not bbox or len(bbox) != 4:
        logger.warning(f"Invalid bbox format: {bbox}")
        return None

    lat_min, lon_min, lat_max, lon_max = bbox

    return {
        "type": "envelope",
        "coordinates": [[lon_min, lat_max], [lon_max, lat_min]],
    }


def load_and_ingest_data(client, index_name, data_file):
    """Load datasets from a JSON file and ingest them into OpenSearch."""
    try:
        with open(data_file, "r") as f:
            datasets = json.load(f)

        if not isinstance(datasets, list):
            logger.error(f"Expected list of datasets, got {type(datasets)}")
            return 0

        ingested = 0
        for dataset in datasets:
            try:
                # Transform spatial_coverage to geo_shape format
                if "spatial_coverage" in dataset and dataset["spatial_coverage"]:
                    dataset["spatial_coverage"] = transform_spatial_coverage(
                        dataset["spatial_coverage"]
                    )

                client.index(index=index_name, id=dataset.get("id"), body=dataset)
                ingested += 1
            except Exception as e:
                logger.warning(f"Error ingesting dataset {dataset.get('id')}: {e}")

        logger.info(f"Ingested {ingested}/{len(datasets)} datasets into {index_name}")
        return ingested
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file}")
        return 0
    except Exception as e:
        logger.error(f"Error loading and ingesting data: {e}")
        return 0


def init_db():
    """
    Initialize the OpenSearch database.

    - Connects to OpenSearch
    - Creates auctus_catalog_master index if it doesn't exist
    - Loads synthetic data if the index is empty
    """
    try:
        client = get_client()

        # Test connection
        health = client.cluster.health()
        logger.info(f"OpenSearch cluster health: {health['status']}")

        # Check if index exists
        if not index_exists(client, AUCTUS_INDEX_NAME):
            logger.info(f"Index {AUCTUS_INDEX_NAME} does not exist. Creating...")
            if create_index(client, AUCTUS_INDEX_NAME, DATASETS_MAPPING):
                logger.info(f"Successfully created index {AUCTUS_INDEX_NAME}")
            else:
                logger.error(f"Failed to create index {AUCTUS_INDEX_NAME}")
                return
        else:
            try:
                client.indices.put_mapping(
                    index=AUCTUS_INDEX_NAME,
                    body={"properties": {"domain": {"type": "keyword"}, "provider": {"type": "keyword"}}},
                )
            except Exception as exc:
                logger.debug("Could not update dataset index mapping with portal fields: %s", exc)

        if not index_exists(client, AUCTUS_PORTALS_INDEX_NAME):
            logger.info(f"Index {AUCTUS_PORTALS_INDEX_NAME} does not exist. Creating...")
            if create_index(client, AUCTUS_PORTALS_INDEX_NAME, PORTALS_MAPPING):
                logger.info(f"Successfully created index {AUCTUS_PORTALS_INDEX_NAME}")
            else:
                logger.error(f"Failed to create index {AUCTUS_PORTALS_INDEX_NAME}")
                return

        # Check if index is empty
        count = index_count(client, AUCTUS_INDEX_NAME)
        if count == 0:
            logger.info(f"Index {AUCTUS_INDEX_NAME} is empty. Loading data...")
            data_file = Path(__file__).parent / "data" / "synthetic_datasets.json"
            load_and_ingest_data(client, AUCTUS_INDEX_NAME, str(data_file))
        else:
            logger.info(f"Index {AUCTUS_INDEX_NAME} already has {count} documents")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
