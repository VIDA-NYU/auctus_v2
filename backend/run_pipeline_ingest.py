"""Dispatcher for Socrata discovery and ARQ job enqueueing."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from arq.connections import RedisSettings, create_pool
from dateutil.parser import parse as parse_datetime

from crawlers.socrata.crawler import discover_socrata_datasets
from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client, init_db

LOGGER = logging.getLogger(__name__)


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


def normalize_timestamp_value(value: Any) -> str | None:
	"""Normalize Socrata/OpenSearch timestamp values into a comparable UTC ISO string."""
	if value in (None, ""):
		return None

	try:
		if isinstance(value, (int, float)):
			seconds = float(value)
			if seconds > 10_000_000_000:
				seconds /= 1000.0
			dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
		else:
			dt = parse_datetime(str(value).strip())
			if dt.tzinfo is None:
				dt = dt.replace(tzinfo=timezone.utc)
			dt = dt.astimezone(timezone.utc)
		return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
	except Exception:
		return None


def extract_socrata_update_timestamp(metadata_payload: dict[str, Any]) -> str | None:
	"""Extract the most recent Socrata update timestamp from a raw metadata payload."""
	if not isinstance(metadata_payload, dict):
		return None

	metadata_block = metadata_payload.get("metadata") if isinstance(metadata_payload.get("metadata"), dict) else {}
	resource_block = metadata_payload.get("resource") if isinstance(metadata_payload.get("resource"), dict) else {}

	candidate_values = (
		metadata_payload.get("updatedAt"),
		metadata_payload.get("dataUpdatedAt"),
		metadata_payload.get("rows_updated_at"),
		metadata_payload.get("rowsUpdatedAt"),
		metadata_payload.get("viewLastModified"),
		metadata_payload.get("lastModified"),
		metadata_block.get("updatedAt"),
		metadata_block.get("dataUpdatedAt"),
		resource_block.get("updatedAt"),
	)

	for candidate in candidate_values:
		normalized = normalize_timestamp_value(candidate)
		if normalized:
			return normalized

	return None


def extract_indexed_update_timestamp(existing_source: dict[str, Any] | None) -> str | None:
	"""Find the timestamp stored on an already indexed document."""
	if not isinstance(existing_source, dict):
		return None

	for key in (
		"socrata_updated_at",
		"source_updated_at",
		"updatedAt",
		"dataUpdatedAt",
		"rows_updated_at",
		"rowsUpdatedAt",
		"last_update_date",
	):
		normalized = normalize_timestamp_value(existing_source.get(key))
		if normalized:
			return normalized

	return None


def apply_socrata_timestamp(document: dict[str, Any], socrata_updated_at: str | None) -> dict[str, Any]:
	"""Persist the Socrata update timestamp in the document for future dedupe checks."""
	if not socrata_updated_at:
		return document

	document["socrata_updated_at"] = socrata_updated_at
	document["source_updated_at"] = socrata_updated_at
	document["last_update_date"] = socrata_updated_at.split("T", 1)[0]
	return document


async def fetch_socrata_update_timestamp(
	base_url: str,
	dataset_id: str,
	http_timeout_seconds: float,
) -> str | None:
	"""Fetch Socrata metadata and extract its latest update timestamp."""
	metadata_url = f"{base_url.rstrip('/')}/api/views/{dataset_id}.json"
	timeout = httpx.Timeout(http_timeout_seconds, connect=15.0)
	async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
		response = await client.get(metadata_url)
		response.raise_for_status()
		payload = response.json()
	return extract_socrata_update_timestamp(payload)


def build_dataset_meta(
	dataset_id: str,
	base_url: str,
	fallback_bbox: list[float],
	spatial_label: str,
	max_sample_rows: int,
	max_sample_bytes: int,
	http_timeout_seconds: float,
	socrata_updated_at: str | None,
) -> dict[str, Any]:
	"""Build the payload handed to the ARQ worker."""
	return {
		"dataset_id": dataset_id,
		"base_url": base_url,
		"fallback_bbox": fallback_bbox,
		"spatial_label": spatial_label,
		"max_sample_rows": max_sample_rows,
		"max_sample_bytes": max_sample_bytes,
		"http_timeout_seconds": http_timeout_seconds,
		"socrata_updated_at": socrata_updated_at,
	}


async def main() -> None:
	"""Discover datasets, dedupe by timestamp, and enqueue heavy jobs to ARQ."""
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
	active_domain, portal_cfg, pipeline_settings = load_runtime_config()
	base_url = portal_cfg.get("base_url", f"https://{active_domain}")
	fallback_bbox = portal_cfg.get("fallback_bbox", [-74.259, 40.477, -73.7, 40.917])
	spatial_label = portal_cfg.get("label", "")
	max_sample_rows = int(pipeline_settings.get("max_sample_rows", 500))
	max_sample_bytes = int(pipeline_settings.get("max_sample_bytes", 2_100_000))
	http_timeout_seconds = float(pipeline_settings.get("http_timeout_seconds", 30.0))

	redis_host = os.getenv("REDIS_HOST", "localhost")
	redis_port = int(os.getenv("REDIS_PORT", "6379"))
	redis_pool = await create_pool(RedisSettings(host=redis_host, port=redis_port))
	try:
		dataset_ids = await discover_socrata_datasets(domain=active_domain, limit=LIMIT)
		LOGGER.info("Starting batch ingest for %d datasets...", len(dataset_ids))

		for index, dataset_id in enumerate(dataset_ids, start=1):
			LOGGER.info("Processing dataset %d of %d (ID: %s)...", index, len(dataset_ids), dataset_id)
			socrata_updated_at = None
			try:
				socrata_updated_at = await fetch_socrata_update_timestamp(
					base_url=base_url,
					dataset_id=dataset_id,
					http_timeout_seconds=http_timeout_seconds,
				)

				existing_doc = os_client.get(index=AUCTUS_INDEX_NAME, id=dataset_id, ignore=[404])
				existing_found = not isinstance(existing_doc, dict) or existing_doc.get("found", True) is not False
				existing_source = existing_doc.get("_source") if existing_found and isinstance(existing_doc, dict) else None
				indexed_updated_at = extract_indexed_update_timestamp(existing_source)

				if existing_found and indexed_updated_at and socrata_updated_at and indexed_updated_at == socrata_updated_at:
					LOGGER.info("⏭️ Dataset %s is up-to-date in OpenSearch. Skipping processing.", dataset_id)
					continue

				if not existing_found:
					LOGGER.info("Dataset %s is new to OpenSearch. Proceeding with full ingestion.", dataset_id)

				if existing_found and indexed_updated_at and socrata_updated_at:
					LOGGER.info(
						"Dataset %s is stale in OpenSearch (indexed=%s, socrata=%s). Reprocessing.",
						dataset_id,
						indexed_updated_at,
						socrata_updated_at,
					)
				elif existing_found and indexed_updated_at and not socrata_updated_at:
					LOGGER.info(
						"Dataset %s has an indexed timestamp but Socrata did not return one. Proceeding with ingestion.",
						dataset_id,
					)
			except Exception as exc:
				LOGGER.warning("Deduplication check failed for %s: %s. Proceeding with ingestion.", dataset_id, exc)

			dataset_meta = build_dataset_meta(
				dataset_id=dataset_id,
				base_url=base_url,
				fallback_bbox=fallback_bbox,
				spatial_label=spatial_label,
				max_sample_rows=max_sample_rows,
				max_sample_bytes=max_sample_bytes,
				http_timeout_seconds=http_timeout_seconds,
				socrata_updated_at=socrata_updated_at,
			)
			await redis_pool.enqueue_job("process_dataset_task", dataset_meta)
			LOGGER.info("🚀 Enqueued dataset %s to ARQ background worker.", dataset_id)

		LOGGER.info("Batch ingest completed for %d datasets", len(dataset_ids))
	finally:
		await redis_pool.aclose()


if __name__ == "__main__":
	asyncio.run(main())
