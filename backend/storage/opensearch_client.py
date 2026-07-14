"""OpenSearch initialization and management for Auctus v2 backend."""

import logging
import os

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

# Which description field the full-text (BM25) query targets. Selectable at query
# time so search can rank on the original portal description or on the AutoDDG
# User-Focused (UFD) / Search-Focused (SFD) descriptions
# (see "docs/Profiler_metadata — Field Reference.md").
DEFAULT_DESCRIPTION_SOURCE = os.getenv("DEFAULT_DESCRIPTION_SOURCE", "original")
DEFAULT_TITLE_BOOST = 2.0
DESCRIPTION_SOURCE_FIELDS = {
    "original": "description",
    "ufd": "autoddg_description",
    "sfd": "autoddg_search_description",
}

# The generated description fields, defined once so the index mapping, the
# post-hoc put_mapping in init_db, and initialize_os all agree. If these fields
# are ever created by dynamic mapping instead, they get the standard analyzer
# WITHOUT English stopwords while `description` uses text_analyzer WITH them,
# which skews BM25 scoring across description sources.
GENERATED_DESCRIPTION_FIELD_MAPPINGS = {
    "autoddg_description": {"type": "text", "analyzer": "text_analyzer"},
    "autoddg_search_description": {"type": "text", "analyzer": "text_analyzer"},
}


def description_fields_for(
    source: str | None, title_boost: float = DEFAULT_TITLE_BOOST
) -> list[str]:
    """Return the BM25 field list for a description source.

    ``title_boost`` weights the title field; 0 (or negative) drops the title
    entirely, which the retrieval eval uses to remove the title-echo confound.
    Unknown sources raise ValueError so a typo cannot silently evaluate the
    default arm — callers exposed over HTTP should turn that into a 400.
    """
    key = source or DEFAULT_DESCRIPTION_SOURCE
    if key not in DESCRIPTION_SOURCE_FIELDS:
        raise ValueError(
            f"Unknown description_source {source!r}; "
            f"expected one of {sorted(DESCRIPTION_SOURCE_FIELDS)}"
        )
    fields = []
    if title_boost > 0:
        fields.append(f"title^{title_boost:g}")
    fields.append(DESCRIPTION_SOURCE_FIELDS[key])
    return fields
# The single source of truth for the auctus_catalog_master index mapping.
# Every index-creation path (init_db here, storage/initialize_os.py) MUST use
# this dict rather than keeping its own copy: the codebase used to carry two
# hand-synced copies whose drift produced real bugs (e.g. a bare geo_shape
# spatial_coverage that rejected every document the pipeline actually emits).
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
            # AutoDDG-generated descriptions (UFD = readable, SFD = search-optimised).
            # Indexed so /search can rank on them as alternatives to the original.
            **GENERATED_DESCRIPTION_FIELD_MAPPINGS,
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
            # Must match what the ingestion pipeline actually emits
            # (crawlers/socrata/transformer.py: {"label": ..., "bbox": {"type": "envelope", ...}}).
            # Mapping spatial_coverage as a bare geo_shape made every document with
            # spatial coverage fail to index (mapper_parsing_exception).
            "spatial_coverage": {
                "type": "object",
                "properties": {
                    "label": {"type": "text"},
                    "bbox": {"type": "geo_shape", "strategy": "recursive"},
                },
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
                            # text for full-text matching, .raw for exact aggregations
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


def init_db():
    """
    Initialize the OpenSearch database.

    - Connects to OpenSearch
    - Creates auctus_catalog_master index if it doesn't exist
    - Does NOT seed data; use run_pipeline_ingest.py (real) or seed_synthetic.py (demo)
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
            # Pre-existing indices were created before the evaluation-arm description
            # fields existed. Without this, the first ingested document would create
            # them via dynamic mapping with the wrong analyzer (no English stopwords),
            # silently skewing cross-arm BM25 comparisons. put_mapping is a no-op when
            # the fields already exist with this definition, and fails loudly on a
            # conflicting dynamic mapping — in that case the index must be recreated.
            try:
                client.indices.put_mapping(
                    index=AUCTUS_INDEX_NAME,
                    body={"properties": GENERATED_DESCRIPTION_FIELD_MAPPINGS},
                )
            except Exception as exc:
                logger.warning(
                    "Could not add AutoDDG description fields to the %s mapping: %s. "
                    "If they were already created by dynamic mapping, recreate the index "
                    "before running the retrieval eval (analyzer mismatch skews BM25).",
                    AUCTUS_INDEX_NAME,
                    exc,
                )

        if not index_exists(client, AUCTUS_PORTALS_INDEX_NAME):
            logger.info(f"Index {AUCTUS_PORTALS_INDEX_NAME} does not exist. Creating...")
            if create_index(client, AUCTUS_PORTALS_INDEX_NAME, PORTALS_MAPPING):
                logger.info(f"Successfully created index {AUCTUS_PORTALS_INDEX_NAME}")
            else:
                logger.error(f"Failed to create index {AUCTUS_PORTALS_INDEX_NAME}")
                return

        # init_db only ensures the indexes and mappings exist — it does NOT seed
        # data. Real data is populated by the ingestion pipeline
        # (run_pipeline_ingest.py); synthetic demo data is loaded explicitly via
        # seed_synthetic.py. Just report the current document count here.
        count = index_count(client, AUCTUS_INDEX_NAME)
        logger.info(f"Index {AUCTUS_INDEX_NAME} has {count} documents")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
