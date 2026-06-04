"""Dispatcher for multi-provider discovery and ARQ job enqueueing."""

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

from crawlers.ckan.crawler import discover_ckan_datasets
from crawlers.socrata.crawler import discover_socrata_datasets
from storage.opensearch_client import AUCTUS_INDEX_NAME, AUCTUS_PORTALS_INDEX_NAME, get_client, init_db

LOGGER = logging.getLogger(__name__)
DEFAULT_FALLBACK_BBOX = [-74.259, 40.477, -73.7, 40.917]


def load_runtime_config() -> dict[str, Any]:
	"""Load runtime settings from backend/config/config.json."""
	cfg_path = Path(__file__).parent / "config" / "config.json"
	with open(cfg_path, "r", encoding="utf-8") as fh:
		return json.load(fh)


def resolve_registry_file_path(registry_file: str) -> Path:
	"""Resolve registry file paths from absolute, workspace-root, or backend/config locations."""
	candidate = Path(registry_file)
	if candidate.is_absolute():
		return candidate

	workspace_root = Path(__file__).resolve().parents[1]
	backend_config = Path(__file__).parent / "config"

	for path_candidate in (
		workspace_root / candidate,
		backend_config / candidate,
	):
		if path_candidate.exists():
			return path_candidate

	return workspace_root / candidate


def load_domains_from_registry_file(registry_file: str) -> list[str]:
	"""Load domain URLs from a dynamic provider registry file."""
	registry_path = resolve_registry_file_path(registry_file)
	if not registry_path.exists():
		LOGGER.warning("Registry file %s not found. Skipping dynamic provider domains.", registry_path)
		return []

	with open(registry_path, "r", encoding="utf-8") as fh:
		payload = json.load(fh)

	if not isinstance(payload, list):
		LOGGER.warning("Registry file %s is not a JSON array. Skipping.", registry_path)
		return []

	domains: list[str] = []
	seen: set[str] = set()
	for entry in payload:
		if not isinstance(entry, dict):
			continue
		url = entry.get("url")
		if not isinstance(url, str):
			continue
		domain = url.strip()
		if not domain or domain in seen:
			continue
		seen.add(domain)
		domains.append(domain)

	return domains


def resolve_provider_domains(provider_type: str, provider_cfg: dict[str, Any]) -> list[str]:
	"""Resolve provider domains based on strategy (DYNAMIC or EXPLICIT)."""
	strategy = str(provider_cfg.get("strategy", "EXPLICIT")).upper()

	if strategy == "DYNAMIC":
		registry_file = provider_cfg.get("registry_file")
		if not isinstance(registry_file, str) or not registry_file.strip():
			LOGGER.warning("Provider %s strategy DYNAMIC missing registry_file. Skipping.", provider_type)
			return []
		return load_domains_from_registry_file(registry_file)

	if strategy == "EXPLICIT":
		raw_domains = provider_cfg.get("domains", [])
		if not isinstance(raw_domains, list):
			LOGGER.warning("Provider %s EXPLICIT domains is not a list. Skipping.", provider_type)
			return []
		return [domain.strip() for domain in raw_domains if isinstance(domain, str) and domain.strip()]

	LOGGER.warning("Provider %s has unknown strategy %s. Skipping.", provider_type, strategy)
	return []


async def discover_datasets_for_provider(provider_type: str, domain_url: str, max_datasets: int) -> list[str]:
	"""Provider discovery factory: dispatch domain crawl to the matching engine."""
	provider = provider_type.lower()
	if provider == "socrata":
		return await discover_socrata_datasets(domain=domain_url, limit=max_datasets)
	if provider == "ckan":
		return await discover_ckan_datasets(domain=domain_url, limit=max_datasets)

	LOGGER.warning("Provider engine for %s is not implemented yet. Skipping.", provider_type)
	return []


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


async def sync_portal_metadata(domain_url: str, provider_type: str) -> None:
	"""Sync a small per-portal summary document into the portals metadata index."""
	if not domain_url:
		return

	client = get_client()
	current_utc_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
	count_response = client.count(
		index=AUCTUS_INDEX_NAME,
		body={"query": {"term": {"domain": domain_url}}},
	)
	total_counted = int(count_response.get("count", 0) or 0)
	summary_document = {
		"domain": domain_url,
		"provider": provider_type,
		"display_label": domain_url.replace("data.", "").replace(".gov", "").replace(".org", "").title(),
		"dataset_count": total_counted,
		"last_indexed_at": current_utc_timestamp,
	}
	client.index(index=AUCTUS_PORTALS_INDEX_NAME, id=domain_url, body=summary_document, refresh=True)


async def main() -> None:
	"""Discover datasets by provider/domain, dedupe by timestamp, and enqueue heavy jobs to ARQ."""
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

	LOGGER.info("Per-domain ingest target limit: %d datasets", LIMIT)
	LOGGER.info("Initializing OpenSearch index state")
	init_db()

	os_client = get_client()
	runtime_cfg = load_runtime_config()
	pipeline_settings = runtime_cfg.get("pipeline_settings", {})
	providers_cfg = runtime_cfg.get("providers", {})
	active_providers = runtime_cfg.get("active_providers", ["socrata"])

	if not isinstance(active_providers, list):
		LOGGER.warning("active_providers is not a list. Falling back to ['socrata']")
		active_providers = ["socrata"]

	max_sample_rows = int(pipeline_settings.get("max_sample_rows", 500))
	max_sample_bytes = int(pipeline_settings.get("max_sample_bytes", 2_100_000))
	http_timeout_seconds = float(pipeline_settings.get("http_timeout_seconds", 30.0))
	development_safeguards = pipeline_settings.get("development_safeguards", {})
	safeguards_enabled = bool(development_safeguards.get("enabled", False))
	max_domains_per_provider = int(development_safeguards.get("max_domains_per_provider", 0))
	max_datasets_per_domain = int(development_safeguards.get("max_datasets_per_domain", 0))

	if safeguards_enabled:
		LOGGER.info(
			"Development safeguards enabled (max_domains_per_provider=%d, max_datasets_per_domain=%d)",
			max_domains_per_provider,
			max_datasets_per_domain,
		)

	redis_host = os.getenv("REDIS_HOST", "localhost")
	redis_port = int(os.getenv("REDIS_PORT", "6379"))
	redis_pool = await create_pool(RedisSettings(host=redis_host, port=redis_port))
	try:
		for provider_type in active_providers:
			if not isinstance(provider_type, str) or not provider_type.strip():
				LOGGER.warning("Skipping invalid provider entry: %s", provider_type)
				continue

			provider_key = provider_type.strip()
			provider_cfg = providers_cfg.get(provider_key, {}) if isinstance(providers_cfg, dict) else {}
			if not isinstance(provider_cfg, dict):
				LOGGER.warning("Provider config for %s is invalid. Skipping.", provider_key)
				continue

			domain_pool = resolve_provider_domains(provider_key, provider_cfg)
			if safeguards_enabled and max_domains_per_provider > 0:
				domain_pool = domain_pool[:max_domains_per_provider]

			if not domain_pool:
				LOGGER.info("No domains resolved for provider %s. Skipping.", provider_key)
				continue

			LOGGER.info("Provider %s resolved %d domains", provider_key, len(domain_pool))

			for domain_index, domain_url in enumerate(domain_pool, start=1):
				domain_dataset_limit = LIMIT
				if safeguards_enabled and max_datasets_per_domain > 0:
					domain_dataset_limit = max_datasets_per_domain

				LOGGER.info(
					"Provider %s domain %d/%d: %s (dataset limit=%d)",
					provider_key,
					domain_index,
					len(domain_pool),
					domain_url,
					domain_dataset_limit,
				)

				dataset_ids = await discover_datasets_for_provider(
					provider_type=provider_key,
					domain_url=domain_url,
					max_datasets=domain_dataset_limit,
				)
				if not dataset_ids:
					continue

				base_url = str(provider_cfg.get("base_url", f"https://{domain_url}"))
				fallback_bbox = provider_cfg.get("fallback_bbox", DEFAULT_FALLBACK_BBOX)
				if not isinstance(fallback_bbox, list) or len(fallback_bbox) != 4:
					fallback_bbox = DEFAULT_FALLBACK_BBOX
				spatial_label = str(provider_cfg.get("label", domain_url))

				LOGGER.info(
					"Starting ingest for provider %s domain %s with %d datasets",
					provider_key,
					domain_url,
					len(dataset_ids),
				)

				for index, dataset_id in enumerate(dataset_ids, start=1):
					LOGGER.info(
						"Processing dataset %d of %d (provider=%s, domain=%s, id=%s)...",
						index,
						len(dataset_ids),
						provider_key,
						domain_url,
						dataset_id,
					)

					socrata_updated_at = None
					if provider_key.lower() == "socrata":
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

							if (
								existing_found
								and indexed_updated_at
								and socrata_updated_at
								and indexed_updated_at == socrata_updated_at
							):
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
					dataset_meta["provider"] = provider_key
					dataset_meta["domain"] = domain_url

					await redis_pool.enqueue_job("process_dataset_task", dataset_meta)
					LOGGER.info("🚀 Enqueued dataset %s to ARQ background worker.", dataset_id)

		LOGGER.info("Batch ingest completed for all active providers")
	finally:
		await redis_pool.aclose()


if __name__ == "__main__":
	asyncio.run(main())
