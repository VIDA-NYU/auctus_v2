"""End-to-end orchestration for profiling, MinIO archival, and OpenSearch indexing."""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any

from opensearch_config import AUCTUS_INDEX_NAME, get_client, init_db
from storage_config import get_storage_client, upload_heavy_profile
from test_socrata_ingest import build_validation_record

LOGGER = logging.getLogger(__name__)


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


async def main() -> None:
	"""Run the profiling pipeline and route full and trimmed outputs to their stores."""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)

	LOGGER.info("Initializing OpenSearch index state")
	init_db()

	os_client = get_client()
	storage_client = get_storage_client()

	LOGGER.info("Building comprehensive validation record")
	comprehensive_record = await build_validation_record()
	dataset_id = comprehensive_record.get("id")
	if not dataset_id:
		raise ValueError("Validation record did not include an 'id' field")

	LOGGER.info("Uploading full profile to MinIO for dataset %s", dataset_id)
	upload_heavy_profile(storage_client, dataset_id, comprehensive_record)

	search_payload = isolate_search_payload(comprehensive_record)
	LOGGER.info("Indexing trimmed search document into OpenSearch for dataset %s", dataset_id)
	os_client.index(
		index=AUCTUS_INDEX_NAME,
		id=dataset_id,
		body=search_payload,
		refresh=True,
	)

	LOGGER.info("Pipeline ingest completed successfully for dataset %s", dataset_id)


if __name__ == "__main__":
	asyncio.run(main())
