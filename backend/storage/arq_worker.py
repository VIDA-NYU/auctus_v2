"""ARQ background worker for heavy Socrata ingestion jobs."""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arq.connections import RedisSettings

from crawlers.socrata.transformer import build_validation_record
from storage.minio_client import get_storage_client, upload_heavy_profile
from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client
from run_pipeline_ingest import apply_socrata_timestamp, isolate_search_payload, load_runtime_config, sync_portal_metadata

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# AutoDDG generates an LLM dataset description through the NYU Portkey gateway.
# Imports are optional so the worker still runs if the packages are absent.
try:
    from autoddg import AutoDDG
    from portkey_ai import Portkey
except Exception:
    AutoDDG = None
    Portkey = None

LOGGER = logging.getLogger(__name__)
_embedding_model = None
_autoddg = None

# Portkey gateway defaults (overridable via environment).
PORTKEY_BASE_URL = os.getenv("PORTKEY_BASE_URL", "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/")
AUTODDG_MODEL = os.getenv("AUTODDG_MODEL", "@vertexai/gemini-2.5-flash")


def get_embedding_model():
    """Load the sentence transformer model lazily on first use."""
    global _embedding_model
    if _embedding_model is None:
        if SentenceTransformer is None:
            LOGGER.warning("sentence-transformers not installed; embeddings will be skipped")
            return None
        LOGGER.info("Loading embedding model 'all-MiniLM-L6-v2'...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def attach_embedding(document: dict[str, Any], model: Any | None = None) -> dict[str, Any]:
    """Generate and attach a k-NN embedding vector to the document."""
    model = model or get_embedding_model()
    if model is None:
        return document

    try:
        title = document.get("title", "") or ""
        description = document.get("description", "") or ""
        text = f"{title}\n\n{description}"

        embedding_array = model.encode([text], convert_to_numpy=True)[0]
        try:
            vec_list = embedding_array.tolist()
        except Exception:
            vec_list = [float(x) for x in embedding_array]

        document["dataset_vector"] = vec_list
        document["embedding_metadata"] = {
            "model_name": "all-MiniLM-L6-v2",
            "version": 1,
        }
    except Exception as exc:
        LOGGER.warning("Failed to generate embedding for document %s: %s", document.get("id"), exc)

    return document


def get_autoddg():
    """Lazily build an AutoDDG instance backed by the Portkey gateway.

    Returns None (and logs a warning) if the packages are missing or no
    PORTKEY_API_KEY is configured, so ingestion can continue without it.
    """
    global _autoddg
    if _autoddg is None:
        if AutoDDG is None or Portkey is None:
            LOGGER.warning("autoddg/portkey not installed; AutoDDG description will be skipped")
            return None
        api_key = os.getenv("PORTKEY_API_KEY")
        if not api_key:
            LOGGER.warning("PORTKEY_API_KEY not set; AutoDDG description will be skipped")
            return None
        client = Portkey(base_url=PORTKEY_BASE_URL, api_key=api_key)
        _autoddg = AutoDDG(client=client, model_name=AUTODDG_MODEL, description_words=100)
        LOGGER.info("AutoDDG initialized (model=%s)", AUTODDG_MODEL)
    return _autoddg


def build_profile_text(record: dict[str, Any]) -> str:
    """Render a trimmed, content-focused profile as a JSON string for AutoDDG.

    AutoDDG injects this string into its prompt (use_profile=True). We keep only
    fields that describe *what the dataset contains* — size, column schema, and
    geographic coverage — and drop operational/rendering fields (telemetry,
    profiling times, raw geohash grids, redundant keyword splits). This is the
    profile-aware usage emphasized in the AutoDDG paper. Returns "" if there is
    nothing useful to report. The selection is deterministic (no LLM).
    """
    pm = record.get("profiler_metadata")
    if not isinstance(pm, dict):
        return ""

    profile: dict[str, Any] = {}
    for key in ("nb_rows", "nb_columns", "types"):
        if pm.get(key) is not None:
            profile[key] = pm[key]

    # Per-column schema: meaning-bearing fields, plus numeric stats when present.
    trimmed_columns: list[dict[str, Any]] = []
    for col in pm.get("columns") or []:
        if not isinstance(col, dict):
            continue
        entry = {
            key: col[key]
            for key in ("name", "structural_type", "semantic_types",
                        "num_distinct_values", "mean", "std", "min", "max")
            if col.get(key) not in (None, [], "")
        }
        if entry:
            trimmed_columns.append(entry)
    if trimmed_columns:
        profile["columns"] = trimmed_columns
    elif pm.get("attribute_keywords"):
        # Fallback when profiling produced no columns (edge cases): names only.
        profile["column_names"] = pm["attribute_keywords"]

    # Geographic coverage: clean label + bbox only (drop the raw geohash grids).
    spatial = record.get("spatial_coverage")
    if isinstance(spatial, dict) and (spatial.get("label") or spatial.get("bbox")):
        profile["spatial_coverage"] = {
            k: spatial[k] for k in ("label", "bbox") if spatial.get(k)
        }

    return json.dumps(profile, ensure_ascii=False) if profile else ""


def attach_autoddg_description(record: dict[str, Any]) -> dict[str, Any]:
    """Generate an AutoDDG description from the CSV sample and store it on the record.

    Runs before the MinIO upload so the text is persisted in the full profile and
    flows downstream into the trimmed search document as well. The atlas-profiler
    metadata is passed as a structural profile to improve quality. Failures are
    non-fatal: ingestion proceeds without the description.
    """
    autoddg = get_autoddg()
    if autoddg is None:
        return record

    sample = record.get("sample")
    if not sample:
        LOGGER.warning("No CSV sample for dataset %s; skipping AutoDDG", record.get("id"))
        return record

    profile_text = build_profile_text(record)

    try:
        _prompt, description = autoddg.describe_dataset(
            dataset_sample=sample,
            dataset_profile=profile_text or None,
            use_profile=bool(profile_text),
        )
        record["autoddg_description"] = description
        LOGGER.info(
            "AutoDDG description generated for %s (%d chars, use_profile=%s)",
            record.get("id"),
            len(description),
            bool(profile_text),
        )
    except Exception as exc:
        LOGGER.warning("AutoDDG description failed for %s: %s", record.get("id"), exc)

    return record


async def startup(ctx: dict[str, Any]) -> None:
    """Initialize long-lived worker resources."""
    ctx["os_client"] = get_client()
    ctx["storage_client"] = get_storage_client()
    ctx["embedding_model"] = get_embedding_model()
    ctx["autoddg"] = get_autoddg()


async def process_dataset_task(ctx: dict[str, Any], dataset_meta: dict[str, Any]) -> str:
    """Run the heavy ingestion workflow for one Socrata dataset."""
    dataset_id = dataset_meta.get("dataset_id") or dataset_meta.get("id")
    if not dataset_id:
        raise ValueError("dataset_meta missing required dataset_id")

    active_domain, portal_cfg, pipeline_settings = load_runtime_config()
    base_url = dataset_meta.get("base_url") or portal_cfg.get("base_url", f"https://{active_domain}")
    fallback_bbox = dataset_meta.get("fallback_bbox") or portal_cfg.get("fallback_bbox", [-74.259, 40.477, -73.7, 40.917])
    spatial_label = dataset_meta.get("spatial_label")
    if spatial_label is None:
        spatial_label = portal_cfg.get("label", "")
    max_sample_rows = int(dataset_meta.get("max_sample_rows") or pipeline_settings.get("max_sample_rows", 500))
    max_sample_bytes = int(dataset_meta.get("max_sample_bytes") or pipeline_settings.get("max_sample_bytes", 2_100_000))
    http_timeout_seconds = float(dataset_meta.get("http_timeout_seconds") or pipeline_settings.get("http_timeout_seconds", 30.0))
    socrata_updated_at = dataset_meta.get("socrata_updated_at")

    os_client = ctx.get("os_client") or get_client()
    storage_client = ctx.get("storage_client") or get_storage_client()
    embedding_model = ctx.get("embedding_model")
    if embedding_model is None:
        embedding_model = get_embedding_model()
        ctx["embedding_model"] = embedding_model

    try:
        try:
            full_metadata_record = await build_validation_record(
                dataset_id,
                base_url=base_url,
                max_sample_rows=max_sample_rows,
                max_sample_bytes=max_sample_bytes,
                http_timeout_seconds=http_timeout_seconds,
                fallback_bbox=fallback_bbox,
                spatial_label=spatial_label,
            )
        except Exception as exc:
            LOGGER.warning(
                "⚠️ Profiler mathematical edge-case failed for dataset %s. Skipping profiling metrics. Error: %s",
                dataset_id,
                exc,
            )
            # Create a minimal fallback profile so the rest of the pipeline can continue.
            full_metadata_record = {
                "id": dataset_id,
                "title": dataset_meta.get("title", "") or "",
                "description": dataset_meta.get("description", "") or "",
                "profiling": None,
                "metrics": {},
            }
        routing_key = full_metadata_record.get("id") or dataset_id
        apply_socrata_timestamp(full_metadata_record, socrata_updated_at)

        # AutoDDG: generate the LLM description BEFORE persisting/indexing so it
        # is stored in MinIO and propagates into the trimmed search document.
        full_metadata_record = attach_autoddg_description(full_metadata_record)

        LOGGER.info("Uploading full profile to MinIO for dataset %s", routing_key)
        upload_heavy_profile(storage_client, routing_key, full_metadata_record)

        # 1. Resolve provider and domain details from dataset_meta
        provider_type = str(dataset_meta.get("provider") or "socrata")
        domain_url = str(dataset_meta.get("domain") or base_url.replace("https://", "").replace("http://", ""))

        search_payload = isolate_search_payload(full_metadata_record)
        search_payload = attach_embedding(search_payload, model=embedding_model)
        apply_socrata_timestamp(search_payload, socrata_updated_at)

        # 2. Assign fields manually so they match the updated auctus_catalog_master mapping! 👈
        search_payload["domain"] = domain_url
        search_payload["provider"] = provider_type

        LOGGER.info("Indexing trimmed search document into OpenSearch for dataset %s", routing_key)
        os_client.index(
            index=AUCTUS_INDEX_NAME,
            id=routing_key,
            body=search_payload,
            refresh=True,
        )

        # provider_type = str(dataset_meta.get("provider") or "socrata")
        # domain_url = str(dataset_meta.get("domain") or base_url.replace("https://", "").replace("http://", ""))
        try:
            await sync_portal_metadata(domain_url=domain_url, provider_type=provider_type)
        except Exception as exc:
            LOGGER.warning("Portal metadata sync failed for domain %s: %s", domain_url, exc)
    except Exception as exc:
        LOGGER.exception("Dataset ingest failed for %s: %s", dataset_id, exc)
        raise

    return dataset_id


class WorkerSettings:
    """ARQ worker configuration."""

    functions = [process_dataset_task]
    redis_settings = RedisSettings(host="redis", port=6379)
    on_startup = startup
