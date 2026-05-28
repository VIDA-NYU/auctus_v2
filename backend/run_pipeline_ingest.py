"""End-to-end orchestration for profiling, MinIO archival, and OpenSearch indexing."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any

from crawlers.socrata.crawler import discover_socrata_datasets
from crawlers.socrata.transformer import build_validation_record
from storage.minio_client import get_storage_client, upload_heavy_profile
from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client, init_db

try:
	from sentence_transformers import SentenceTransformer
except ImportError:
	SentenceTransformer = None

LOGGER = logging.getLogger(__name__)
_embedding_model = None


def load_runtime_config() -> tuple[str, dict[str, Any], dict[str, Any]]:
	"""Load the active portal domain and runtime settings from backend/config/config.json."""
	cfg_path = Path(__file__).parent / "config" / "config.json"
	with open(cfg_path, "r", encoding="utf-8") as fh:
		cfg = json.load(fh)

	active_portal = cfg.get("active_portal", "data.cityofnewyork.us")
	portal_cfg = cfg.get("portals", {}).get(active_portal, {})
	pipeline_settings = cfg.get("pipeline_settings", {})
	return active_portal, portal_cfg, pipeline_settings


def isolate_search_payload(comprehensive_record: dict[str, Any]) -> dict[str, Any]:
	"""Trim a full catalog record down to fields that are efficient for search indexing."""
	payload = copy.deepcopy(comprehensive_record)
	payload.pop("sample", None)
	payload.pop("_sample_telemetry", None)

	profiler_metadata = payload.get("profiler_metadata")
	if isinstance(profiler_metadata, dict):
		columns = profiler_metadata.get("columns")
		if isinstance(columns, list):
			profiler_metadata["columns"] = [
				{
					key: column[key]
					for key in ("name", "structural_type", "semantic_types")
					if key in column
				}
				for column in columns
				if isinstance(column, dict)
			]

	return payload


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


def attach_embedding(document: dict[str, Any]) -> dict[str, Any]:
	"""Generate and attach a k-NN embedding vector to the document.
	
	Encodes title + description through 'all-MiniLM-L6-v2' and attaches
	the 384-dimensional vector along with metadata.
	
	Returns the modified document. If embedding fails, logs a warning and returns
	the unmodified document (allowing indexing to continue without k-NN search).
	"""
	model = get_embedding_model()
	if model is None:
		return document
	
	try:
		title = document.get("title", "") or ""
		description = document.get("description", "") or ""
		text = f"{title}\n\n{description}"
		
		# Encode and convert to list
		embedding_array = model.encode([text], convert_to_numpy=True)[0]
		try:
			vec_list = embedding_array.tolist()
		except Exception:
			vec_list = [float(x) for x in embedding_array]
		
		# Attach vector and metadata
		document["dataset_vector"] = vec_list
		document["embedding_metadata"] = {
			"model_name": "all-MiniLM-L6-v2",
			"version": 1,
		}
	except Exception as exc:
		LOGGER.warning("Failed to generate embedding for document %s: %s", document.get("id"), exc)
	
	return document


async def main() -> None:
	"""Run the profiling pipeline and route full and trimmed outputs to their stores."""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)

	LIMIT = 10
	if len(sys.argv) > 1:
		try:
			LIMIT = int(sys.argv[1])
		except (TypeError, ValueError):
			LIMIT = 10
	if LIMIT <= 0:
		LIMIT = 10

	LOGGER.info("Batch ingest target limit: %d datasets", LIMIT)

	LOGGER.info("Initializing OpenSearch index state")
	init_db()

	os_client = get_client()
	storage_client = get_storage_client()
	active_domain, portal_cfg, pipeline_settings = load_runtime_config()
	base_url = portal_cfg.get("base_url", f"https://{active_domain}")
	fallback_bbox = portal_cfg.get("fallback_bbox", [-74.259, 40.477, -73.7, 40.917])
	spatial_label = portal_cfg.get("label", "")
	max_sample_rows = int(pipeline_settings.get("max_sample_rows", 500))
	max_sample_bytes = int(pipeline_settings.get("max_sample_bytes", 2_100_000))
	http_timeout_seconds = float(pipeline_settings.get("http_timeout_seconds", 30.0))

	dataset_ids = await discover_socrata_datasets(domain=active_domain, limit=LIMIT)
	LOGGER.info("Starting batch ingest for %d datasets...", len(dataset_ids))

	for index, dataset_id in enumerate(dataset_ids, start=1):
		LOGGER.info("Processing dataset %d of %d (ID: %s)...", index, len(dataset_ids), dataset_id)
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
			routing_key = full_metadata_record.get("id") or dataset_id

			LOGGER.info("Uploading full profile to MinIO for dataset %s", routing_key)
			upload_heavy_profile(storage_client, routing_key, full_metadata_record)

			search_payload = isolate_search_payload(full_metadata_record)
			search_payload = attach_embedding(search_payload)
			LOGGER.info("Indexing trimmed search document into OpenSearch for dataset %s", routing_key)
			os_client.index(
				index=AUCTUS_INDEX_NAME,
				id=routing_key,
				body=search_payload,
				refresh=True,
			)
		except Exception as exc:
			LOGGER.exception("Dataset ingest failed for %s: %s", dataset_id, exc)
			continue

	LOGGER.info("Batch ingest completed for %d datasets", len(dataset_ids))


if __name__ == "__main__":
	asyncio.run(main())
